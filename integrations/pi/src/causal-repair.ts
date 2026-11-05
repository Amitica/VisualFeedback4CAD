import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import {
	assertSupportedViewSpec,
	imageFromWorkspace,
	inspectPngFile,
	requireDrawingSuccess,
} from "./drawing-reflection.js";
import { InfrastructureError } from "./errors.js";
import { isoNow, readJson, writeJsonAtomic } from "./files.js";
import type { WorkerClient } from "./local-worker.js";
import { cameraSha256 } from "./main-view-reflect.js";
import { artifactAccepted, inspectFormalCandidate, sha256File } from "./solution.js";
import {
	CAUSAL_REPAIR_PROFILE,
	PROTOCOL_VERSION,
	SOLUTION_FILE,
	type AgentResult,
	type ArtifactPayload,
	type CausalRepairProfile,
	type CausalRepairSummary,
	type DrawingSimilarityTransform,
	type DrawingViewManifest,
	type DrawingViewSpec,
	type ExecutionPayload,
	type WorkerFixedProjectionResponse,
	type WorkerInspectResponse,
	type WorkerRankDrawingViewsResponse,
	type WorkerRepairCandidateResponse,
	type WorkerRepairComparisonResponse,
} from "./types.js";
import type { RunController } from "./runner.js";

export const CAUSAL_REPAIR_AUDIT_PATH = "drawing_feedback/causal-repair/audit.json";

const IDENTIFIER_SCHEMA = Type.String({
	minLength: 1,
	maxLength: 128,
	pattern: "^[A-Za-z0-9][A-Za-z0-9._-]*$",
});

const STRUCTURED_EDIT_SCHEMA = Type.Object(
	{
		old_text: Type.String({ minLength: 1, maxLength: 20_000 }),
		new_text: Type.String({ maxLength: 20_000 }),
	},
	{ additionalProperties: false },
);

export const COMPARE_REPAIR_HYPOTHESES_PARAMETERS_SCHEMA = Type.Object(
	{
		region_id: IDENTIFIER_SCHEMA,
		hypotheses: Type.Array(
			Type.Object(
				{
					id: IDENTIFIER_SCHEMA,
					diagnosis: Type.String({ minLength: 1, maxLength: 4_000 }),
					expected_visual_effect: Type.String({ minLength: 1, maxLength: 4_000 }),
					edits: Type.Array(STRUCTURED_EDIT_SCHEMA, { minItems: 1, maxItems: 8 }),
				},
				{ additionalProperties: false },
			),
			{ minItems: 2, maxItems: 3 },
		),
	},
	{
		additionalProperties: false,
		description: "Compare two or three candidate patches in one 2D drawing region.",
	},
);

export interface StructuredRepairEdit {
	old_text: string;
	new_text: string;
}

interface RepairHypothesis {
	id: string;
	diagnosis: string;
	expected_visual_effect: string;
	edits: StructuredRepairEdit[];
}

interface RepairCandidateAudit {
	id: string;
	diagnosis: string;
	expected_visual_effect: string;
	edits: StructuredRepairEdit[];
	source_path: string;
	source_sha256: string;
	execution: ExecutionPayload;
	artifact: ArtifactPayload;
	status: "valid" | "invalid" | "missing" | "error" | "timeout";
	visual_effect?: "changed" | "none" | "unavailable";
	renders: Array<{
		region_id: string;
		path: string;
		sha256: string;
	}>;
}

interface RepairFrameAudit {
	region_id: string;
	camera_sha256: string;
	transform_sha256: string;
	view_spec: DrawingViewSpec;
	transform: DrawingSimilarityTransform;
	reference_path: string;
	current_render_path: string;
	current_render_sha256: string;
}

interface RepairEvaluationAudit {
	evaluation_id: string;
	sequence: number;
	created_at: string;
	formal_source_sha256: string;
	formal_step_sha256: string;
	region_id: string;
	frames: RepairFrameAudit[];
	candidates: RepairCandidateAudit[];
	feedback: "comparison" | "error_only";
	errors?: string[];
	sheet_path?: string;
	sheet_sha256?: string;
}

interface RepairFormalizationAudit {
	turn: number;
	created_at: string;
	source_sha256: string;
	step_sha256: string | null;
	evaluation_id: string;
	selection: "current" | "candidate" | "custom";
	candidate_id?: string;
}

interface PersistedCausalRepairAudit {
	version: 1;
	kind: "causal_repair_verification";
	profile: CausalRepairProfile;
	fixture_id: string;
	locked_view_manifest_sha256: string;
	started_at: string;
	updated_at: string;
	evaluations: RepairEvaluationAudit[];
	formalizations: RepairFormalizationAudit[];
}

export interface CausalRepairState {
	fixtureId: string;
	profile: CausalRepairProfile;
	lockedViewManifestSha256: string;
	auditPath: string;
	startedAt: string;
	evaluations: RepairEvaluationAudit[];
	formalizations: RepairFormalizationAudit[];
}

function sha256Text(value: string): string {
	return createHash("sha256").update(value).digest("hex");
}

function sha256Json(value: unknown): string {
	return sha256Text(JSON.stringify(value));
}

function countOccurrences(source: string, needle: string): { count: number; index: number } {
	let count = 0;
	let index = -1;
	let cursor = 0;
	while (cursor <= source.length - needle.length) {
		const found = source.indexOf(needle, cursor);
		if (found < 0) break;
		if (index < 0) index = found;
		count += 1;
		cursor = found + Math.max(1, needle.length);
	}
	return { count, index };
}

export function applyStructuredEdits(source: string, edits: readonly StructuredRepairEdit[]): string {
	const replacements = edits.map((edit, editIndex) => {
		if (edit.old_text.length === 0) throw new Error(`edits[${editIndex}].old_text must not be empty`);
		const occurrence = countOccurrences(source, edit.old_text);
		if (occurrence.count !== 1) {
			throw new Error(`edits[${editIndex}].old_text must occur exactly once in the last formally executed source; found ${occurrence.count}`);
		}
		return {
			start: occurrence.index,
			end: occurrence.index + edit.old_text.length,
			newText: edit.new_text,
			editIndex,
		};
	}).sort((left, right) => left.start - right.start);
	for (let index = 1; index < replacements.length; index += 1) {
		const previous = replacements[index - 1]!;
		const current = replacements[index]!;
		if (current.start < previous.end) {
			throw new Error(`edits[${previous.editIndex}] and edits[${current.editIndex}] overlap`);
		}
	}
	let result = source;
	for (const replacement of [...replacements].reverse()) {
		result = `${result.slice(0, replacement.start)}${replacement.newText}${result.slice(replacement.end)}`;
	}
	if (result === source) throw new Error("A repair hypothesis must change the formal source");
	return result;
}

function candidateStatus(response: WorkerRepairCandidateResponse): RepairCandidateAudit["status"] {
	if (response.execution.timed_out) return "timeout";
	if (artifactAccepted(response.artifact)) return "valid";
	if (!response.execution.success) return "error";
	if (response.artifact.status === "missing") return "missing";
	return "invalid";
}

function compactCandidateError(candidate: RepairCandidateAudit): string {
	if (candidate.status === "timeout") return "执行超时";
	const stderrLines = candidate.execution.stderr.trim().split(/\r?\n/).filter(Boolean);
	const stderr = stderrLines.at(-1);
	if (stderr) return stderr;
	if (candidate.artifact.validation_error) return candidate.artifact.validation_error;
	return `status=${candidate.status}, valid=${String(candidate.artifact.validation?.is_valid === true)}, watertight=${String(candidate.artifact.validation?.is_watertight === true)}`;
}

function candidateFeedbackErrors(
	candidates: readonly RepairCandidateAudit[],
	frames: readonly RepairFrameAudit[],
): string[] {
	return candidates.flatMap((candidate) => {
		if (candidate.status !== "valid") {
			return [`${candidate.id}: Patch 执行失败：${compactCandidateError(candidate)}`];
		}
		const hasVisibleChange = frames.some((frame) => {
			const render = candidate.renders.find(({ region_id }) => region_id === frame.region_id);
			return render !== undefined && render.sha256 !== frame.current_render_sha256;
		});
		if (!hasVisibleChange) {
			return [`${candidate.id}: Patch 执行成功，但在 \`${frames[0]?.region_id ?? "unknown"}\` 区域没有产生可见的二维投影变化。`];
		}
		return [];
	});
}

function persistedAudit(state: CausalRepairState): PersistedCausalRepairAudit {
	return {
		version: 1,
		kind: "causal_repair_verification",
		profile: state.profile,
		fixture_id: state.fixtureId,
		locked_view_manifest_sha256: state.lockedViewManifestSha256,
		started_at: state.startedAt,
		updated_at: isoNow(),
		evaluations: state.evaluations,
		formalizations: state.formalizations,
	};
}

async function persistState(workspace: string, state: CausalRepairState): Promise<void> {
	await writeJsonAtomic(join(workspace, state.auditPath), persistedAudit(state));
}

export async function restoreCausalRepairState(options: {
	workspace: string;
	fixtureId: string;
	profile: CausalRepairProfile;
	lockedViewManifestSha256: string;
}): Promise<CausalRepairState> {
	const stored = await readJson<PersistedCausalRepairAudit>(join(options.workspace, CAUSAL_REPAIR_AUDIT_PATH)).catch(() => null);
	if (!stored) {
		return {
			fixtureId: options.fixtureId,
			profile: options.profile,
			lockedViewManifestSha256: options.lockedViewManifestSha256,
			auditPath: CAUSAL_REPAIR_AUDIT_PATH,
			startedAt: isoNow(),
			evaluations: [],
			formalizations: [],
		};
	}
	if (
		stored.version !== 1 ||
		stored.kind !== "causal_repair_verification" ||
		stored.profile !== options.profile ||
		stored.fixture_id !== options.fixtureId ||
		stored.locked_view_manifest_sha256 !== options.lockedViewManifestSha256 ||
		!Array.isArray(stored.evaluations) ||
		!Array.isArray(stored.formalizations)
	) {
		throw new InfrastructureError("Persisted causal repair audit does not match this continuation");
	}
	return {
		fixtureId: options.fixtureId,
		profile: options.profile,
		lockedViewManifestSha256: options.lockedViewManifestSha256,
		auditPath: CAUSAL_REPAIR_AUDIT_PATH,
		startedAt: stored.started_at,
		evaluations: stored.evaluations,
		formalizations: stored.formalizations,
	};
}

export function causalRepairSummary(state: CausalRepairState): CausalRepairSummary {
	return {
		version: 1,
		profile: state.profile,
		audit_manifest_path: state.auditPath,
		evaluation_count: state.evaluations.length,
		candidate_count: state.evaluations.reduce((total, evaluation) => total + evaluation.candidates.length, 0),
		formalization_count: state.formalizations.length,
	};
}

export async function recordCausalRepairFormalExecution(options: {
	workspace: string;
	state: CausalRepairState;
	turn: number;
	sourceSha256: string;
	stepSha256: string | null;
}): Promise<void> {
	const latestEvaluation = options.state.evaluations.at(-1);
	if (!latestEvaluation || options.state.formalizations.some(({ turn }) => turn === options.turn)) return;
	const matchedEvaluation = [...options.state.evaluations].reverse().find(({ candidates }) =>
		candidates.some(({ source_sha256 }) => source_sha256 === options.sourceSha256));
	const evaluation = matchedEvaluation ?? latestEvaluation;
	const matchingCandidate = evaluation.candidates.find(({ source_sha256 }) => source_sha256 === options.sourceSha256);
	const selection = matchingCandidate
		? "candidate"
		: evaluation.formal_source_sha256 === options.sourceSha256
			? "current"
			: "custom";
	options.state.formalizations.push({
		turn: options.turn,
		created_at: isoNow(),
		source_sha256: options.sourceSha256,
		step_sha256: options.stepSha256,
		evaluation_id: evaluation.evaluation_id,
		selection,
		...(matchingCandidate ? { candidate_id: matchingCandidate.id } : {}),
	});
	await persistState(options.workspace, options.state);
}

async function currentFormalState(options: {
	workspace: string;
	worker: WorkerClient;
	controller: RunController;
	getResult: () => AgentResult;
}): Promise<{ sourceSha256: string; stepSha256: string; source: string }> {
	const latest = await options.worker.request<WorkerInspectResponse>(
		options.workspace,
		{ protocol_version: PROTOCOL_VERSION, operation: "inspect" },
		{ signal: options.controller.signal },
	);
	const inspected = await inspectFormalCandidate(options.workspace, latest.artifact, options.getResult());
	if (!inspected.formal_execution_current || !artifactAccepted(latest.artifact) || !inspected.source_sha256 || !latest.artifact.sha256) {
		throw new Error(`Repair hypotheses require a valid, watertight, formally executed current CAD. ${inspected.reason}`);
	}
	const source = await readFile(join(options.workspace, SOLUTION_FILE), "utf8");
	return {
		sourceSha256: inspected.source_sha256,
		stepSha256: latest.artifact.sha256,
		source,
	};
}

export function createCausalRepairTool(options: {
	workspace: string;
	state: CausalRepairState;
	lockedManifest: DrawingViewManifest;
	worker: WorkerClient;
	controller: RunController;
	getResult: () => AgentResult;
	onUpdate: (summary: CausalRepairSummary) => Promise<void>;
}) {
	const availableRegionIds = options.lockedManifest.regions.map(({ region_id }) => region_id);
	return defineTool({
		name: "compare_repair_hypotheses",
		label: "Compare repair hypotheses",
		description:
			`当你已经发现当前 CAD 存在视觉缺陷，但尚不确定应如何修改代码时，可以使用此工具在一个二维工程图区域中比较两到三个候选 Patch。工具会并列展示原始工程图、当前 CAD 的二维投影以及各候选 Patch 生成的二维投影，帮助你直接观察不同修改方案会使当前 CAD 的二维投影发生什么变化，并据此决定下一步如何修改 CAD 模型。可用的 \`region_id\`：\`${availableRegionIds.join(", ")}\`。`,
		parameters: COMPARE_REPAIR_HYPOTHESES_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			if (new Set(params.hypotheses.map(({ id }) => id)).size !== params.hypotheses.length) {
				throw new Error("hypothesis ids must be unique");
			}
			if (params.hypotheses.some(({ id }) => ["original", "current"].includes(id.toLowerCase()))) {
				throw new Error("hypothesis ids must not be Original or Current");
			}
			const region = options.lockedManifest.regions.find(({ region_id }) => region_id === params.region_id);
			if (!region?.selected) throw new Error(`Unknown or unavailable comparison region ${params.region_id}`);
			assertSupportedViewSpec(region.selected.view_spec);
			const regions = [region];
			const formal = await currentFormalState(options);
			if (Buffer.byteLength(formal.source, "utf8") > 512_000) {
				throw new Error("The formal source is too large for structured candidate verification");
			}
			const hypotheses: RepairHypothesis[] = params.hypotheses.map((hypothesis) => ({
				id: hypothesis.id,
				diagnosis: hypothesis.diagnosis,
				expected_visual_effect: hypothesis.expected_visual_effect,
				edits: hypothesis.edits,
			}));
			const candidateSources = hypotheses.map((hypothesis) => ({
				hypothesis,
				source: applyStructuredEdits(formal.source, hypothesis.edits),
			}));
			const candidateHashes = candidateSources.map(({ source }) => sha256Text(source));
			if (new Set(candidateHashes).size !== candidateHashes.length) {
				throw new Error("Repair hypotheses must produce distinct candidate sources");
			}
			const sequence = options.state.evaluations.length + 1;
			const evaluationDigest = sha256Json([
				formal.sourceSha256,
				formal.stepSha256,
				params.region_id,
				hypotheses,
			]).slice(0, 16);
			const evaluationId = `${String(sequence).padStart(3, "0")}-${evaluationDigest}`;
			const evaluationRoot = join("drawing_feedback", "causal-repair", "evaluations", evaluationId);

			const frames: RepairFrameAudit[] = [];
			for (const region of regions) {
				const candidateId = `current-${evaluationDigest}-${region.region_id}`;
				const viewSpec = { ...region.selected!.view_spec, name: candidateId };
				let transform: DrawingSimilarityTransform;
				let currentRenderPath: string;
				if (formal.stepSha256 === options.lockedManifest.step_sha256) {
					transform = region.selected!.transform;
					currentRenderPath = region.selected!.aligned_render_path;
				} else {
					const response = await options.worker.request<WorkerRankDrawingViewsResponse>(
						options.workspace,
						{
							protocol_version: PROTOCOL_VERSION,
							operation: "rank_drawing_views",
							region_id: region.region_id,
							crop_path: region.crop_path,
							step_path: "output.step",
							candidates: [{ candidate_id: candidateId, view_spec: viewSpec }],
							top_k: 1,
							allow_reflection: false,
							max_rotation_deg: 6,
							min_roi_fill: 0.70,
							seed: 0,
							max_iterations: 45,
							output_dir: join(evaluationRoot, "regions", region.region_id, "current-frame"),
						},
						{ timeoutMs: 240_000, signal: options.controller.signal },
					);
					requireDrawingSuccess(response, "rank_drawing_views");
					const ranked = response.rankings[0];
					if (!ranked || response.rankings.length !== 1 || response.step_sha256 !== formal.stepSha256 || ranked.transform.reflected) {
						throw new InfrastructureError("Current causal comparison frame is inconsistent with the formal STEP");
					}
					transform = ranked.transform;
					currentRenderPath = ranked.aligned_render_path;
				}
				if (transform.reflected) throw new InfrastructureError("Causal repair frames do not permit reflected alignment");
				const currentRender = await inspectPngFile(options.workspace, currentRenderPath);
				frames.push({
					region_id: region.region_id,
					camera_sha256: cameraSha256(viewSpec),
					transform_sha256: sha256Json(transform),
					view_spec: viewSpec,
					transform,
					reference_path: region.crop_path,
					current_render_path: currentRenderPath,
					current_render_sha256: currentRender.sha256,
				});
			}

			const candidates: RepairCandidateAudit[] = [];
			for (const [index, candidate] of candidateSources.entries()) {
				const sourcePath = join(evaluationRoot, "candidates", candidate.hypothesis.id, "submitted-solution.py");
				await mkdir(join(options.workspace, evaluationRoot, "candidates", candidate.hypothesis.id), { recursive: true });
				await writeFile(join(options.workspace, sourcePath), candidate.source, { encoding: "utf8", mode: 0o600 });
				const execution = await options.worker.request<WorkerRepairCandidateResponse>(
					options.workspace,
					{
						protocol_version: PROTOCOL_VERSION,
						operation: "execute_repair_candidate",
						candidate_id: candidate.hypothesis.id,
						source_path: sourcePath,
						formal_source_sha256: formal.sourceSha256,
						formal_step_sha256: formal.stepSha256,
						timeout_s: 120,
						output_dir: join(evaluationRoot, "candidates", candidate.hypothesis.id, "execution"),
					},
					{ timeoutMs: 135_000, signal: options.controller.signal },
				);
				if (
					execution.candidate_id !== candidate.hypothesis.id ||
					execution.source_sha256 !== candidateHashes[index] ||
					execution.formal_source_sha256 !== formal.sourceSha256 ||
					execution.formal_step_sha256 !== formal.stepSha256 ||
					execution.formal_unchanged !== true
				) throw new InfrastructureError(`Candidate ${candidate.hypothesis.id} execution violated its hash contract`);
				const audit: RepairCandidateAudit = {
					id: candidate.hypothesis.id,
					diagnosis: candidate.hypothesis.diagnosis,
					expected_visual_effect: candidate.hypothesis.expected_visual_effect,
					edits: candidate.hypothesis.edits,
					source_path: sourcePath,
					source_sha256: candidateHashes[index]!,
					execution: execution.execution,
					artifact: execution.artifact,
					status: candidateStatus(execution),
					...(artifactAccepted(execution.artifact) ? {} : { visual_effect: "unavailable" as const }),
					renders: [],
				};
				candidates.push(audit);
			}

			let feedbackErrors = candidates
				.filter(({ status }) => status !== "valid")
				.map((candidate) => `${candidate.id}: Patch 执行失败：${compactCandidateError(candidate)}`);
			if (feedbackErrors.length === 0) {
				for (const candidate of candidates) {
					if (!candidate.artifact.path) {
						throw new InfrastructureError(`Valid candidate ${candidate.id} has no STEP path`);
					}
					for (const frame of frames) {
						const render = await options.worker.request<WorkerFixedProjectionResponse>(
							options.workspace,
							{
								protocol_version: PROTOCOL_VERSION,
								operation: "render_fixed_projection",
								region_id: frame.region_id,
								crop_path: frame.reference_path,
								step_path: candidate.artifact.path,
								view_spec: frame.view_spec,
								transform: frame.transform,
								output_dir: join(evaluationRoot, "regions", frame.region_id, "candidates", candidate.id),
							},
							{ timeoutMs: 180_000, signal: options.controller.signal },
						);
						requireDrawingSuccess(render, "render_fixed_projection");
						if (
							render.step_sha256 !== candidate.artifact.sha256 ||
							sha256Json(render.transform) !== frame.transform_sha256 ||
							cameraSha256(render.view_spec) !== frame.camera_sha256
						) throw new InfrastructureError(`Candidate ${candidate.id} fixed-frame rendering drifted`);
						const inspectedRender = await inspectPngFile(options.workspace, render.render_path);
						if (inspectedRender.sha256 !== render.sha256) throw new InfrastructureError("Candidate render hash mismatch");
						candidate.renders.push({ region_id: frame.region_id, path: render.render_path, sha256: render.sha256 });
					}
					candidate.visual_effect = frames.some((frame) =>
						candidate.renders.some(({ region_id, sha256 }) =>
							region_id === frame.region_id && sha256 !== frame.current_render_sha256))
						? "changed"
						: "none";
				}
				feedbackErrors = candidateFeedbackErrors(candidates, frames);
			}

			const [finalSourceSha256, finalStepSha256] = await Promise.all([
				sha256File(join(options.workspace, SOLUTION_FILE)),
				sha256File(join(options.workspace, "output.step")),
			]);
			if (finalSourceSha256 !== formal.sourceSha256 || finalStepSha256 !== formal.stepSha256) {
				throw new InfrastructureError("Causal repair comparison changed the formal source or STEP");
			}
			if (feedbackErrors.length > 0) {
				const evaluation: RepairEvaluationAudit = {
					evaluation_id: evaluationId,
					sequence,
					created_at: isoNow(),
					formal_source_sha256: formal.sourceSha256,
					formal_step_sha256: formal.stepSha256,
					region_id: params.region_id,
					frames,
					candidates,
					feedback: "error_only",
					errors: feedbackErrors,
				};
				options.state.evaluations.push(evaluation);
				await persistState(options.workspace, options.state);
				await options.onUpdate(causalRepairSummary(options.state));
				return {
					content: [{
						type: "text" as const,
						text: `候选 Patch 无法形成有效的视觉比较，因此本次不返回对比图：\n${feedbackErrors.map((error) => `- ${error}`).join("\n")}\n此次比较没有修改正式 solution.py 和 output.step。`,
					}],
					details: evaluation,
				};
			}

			const sheetResponse = await options.worker.request<WorkerRepairComparisonResponse>(
				options.workspace,
				{
					protocol_version: PROTOCOL_VERSION,
					operation: "compose_repair_comparison",
					candidate_ids: candidates.map(({ id }) => id),
					rows: frames.map((frame) => ({
						region_id: frame.region_id,
						reference_path: frame.reference_path,
						current_render_path: frame.current_render_path,
						candidates: candidates.map((candidate) => ({
							candidate_id: candidate.id,
							status: candidate.status,
							render_path: candidate.renders.find(({ region_id }) => region_id === frame.region_id)?.path,
						})),
					})),
					output_dir: join(evaluationRoot, "sheet"),
				},
				{ timeoutMs: 180_000, signal: options.controller.signal },
			);
			requireDrawingSuccess(sheetResponse, "compose_repair_comparison");
			const sheet = await inspectPngFile(options.workspace, sheetResponse.sheet_path);
			if (sheet.sha256 !== sheetResponse.sha256) throw new InfrastructureError("Causal repair sheet hash mismatch");
			const evaluation: RepairEvaluationAudit = {
				evaluation_id: evaluationId,
				sequence,
				created_at: isoNow(),
				formal_source_sha256: formal.sourceSha256,
				formal_step_sha256: formal.stepSha256,
				region_id: params.region_id,
				frames,
				candidates,
				feedback: "comparison",
				sheet_path: sheetResponse.sheet_path,
				sheet_sha256: sheetResponse.sha256,
			};
			options.state.evaluations.push(evaluation);
			await persistState(options.workspace, options.state);
			await options.onUpdate(causalRepairSummary(options.state));
			const statusText = candidates.map(({ id, status, artifact }) =>
				`- ${id}: status=${status}, valid=${String(artifact.validation?.is_valid === true)}, watertight=${String(artifact.validation?.is_watertight === true)}`).join("\n");
			return {
				content: [
					{
						type: "text" as const,
						text: `以下对比图从左到右分别展示了 \`${params.region_id}\` 区域的原始工程图、当前 CAD 二维投影，以及各候选 Patch 生成的二维投影。请观察每个候选如何改变当前 CAD，并判断这些变化是否有助于修复已发现的视觉缺陷。你可以据此选择、组合、进一步改写候选。此次比较没有修改正式 solution.py 和 output.step。\n候选执行状态：\n${statusText}`,
					},
					await imageFromWorkspace(options.workspace, sheetResponse.sheet_path),
				],
				details: evaluation,
			};
		},
	});
}

export function assertCausalRepairProfile(value: unknown): asserts value is CausalRepairProfile {
	if (value !== CAUSAL_REPAIR_PROFILE) {
		throw new InfrastructureError(`Unsupported causal repair profile: ${String(value)}`);
	}
}
