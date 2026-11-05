import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";

import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import { defineTool, generateUnifiedPatch } from "@earendil-works/pi-coding-agent";
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
import type { RunController } from "./runner.js";
import { artifactAccepted, inspectFormalCandidate, sha256File } from "./solution.js";
import {
	PROTOCOL_VERSION,
	REPAIR_PREVIEW_PROFILE,
	SOLUTION_FILE,
	VISUAL_REVIEW_PROFILE,
	type AgentResult,
	type CodeHypothesisVisualEncoding,
	type DrawingSimilarityTransform,
	type DrawingViewManifest,
	type DrawingViewSpec,
	type ExecutionPayload,
	type NormalizedRoi,
	type VisualReviewGeometryEffect,
	type VisualReviewProfile,
	type WorkerCodeEditVisualizationResponse,
	type WorkerDrawingErrorResponse,
	type WorkerFixedProjectionResponse,
	type WorkerInspectResponse,
	type WorkerRepairCandidatePreviewResponse,
	type WorkerRankDrawingViewsResponse,
	type WorkerRepairCandidateResponse,
} from "./types.js";

export const VISUAL_REVIEW_STATE_PATH = "drawing_feedback/code-hypotheses/state.json";
export const VISUAL_REVIEW_CONTEXT_TYPE = "cadgenbench-persistent-code-hypotheses";

const MAX_STORED_HYPOTHESES = 5;

const editSchema = Type.Object({
	oldText: Type.String({
		minLength: 1,
		maxLength: 20_000,
		description: "Exact text that must match one unique, complete region of the current formal solution.py.",
	}),
	newText: Type.String({
		maxLength: 20_000,
		description: "Replacement CAD code executed only in an isolated edit preview.",
	}),
}, { additionalProperties: false });

const roiSchema = Type.Object({
	x_min: Type.Integer({ minimum: 0, maximum: 1000 }),
	y_min: Type.Integer({ minimum: 0, maximum: 1000 }),
	x_max: Type.Integer({ minimum: 0, maximum: 1000 }),
	y_max: Type.Integer({ minimum: 0, maximum: 1000 }),
}, {
	additionalProperties: false,
	description: "Target drawing ROI in normalized 0-1000 coordinates; width and height must each be at least 50.",
});

const repairPreviewCandidateSchema = Type.Object({
	id: Type.String({
		pattern: "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
		description: "A short identifier that distinguishes this candidate in the returned comparison.",
	}),
	hypothesis: Type.String({
		minLength: 1,
		maxLength: 2_000,
		description: "The concrete repair hypothesis tested by this candidate.",
	}),
	edits: Type.Array(editSchema, {
		minItems: 1,
		maxItems: 8,
		description: "Exact oldText/newText replacements, all matched against the same current formal solution.py.",
	}),
}, { additionalProperties: false });

export const PREVIEW_REPAIR_CANDIDATES_PARAMETERS_SCHEMA = Type.Object({
	region_id: Type.String({ pattern: "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$" }),
	candidates: Type.Array(repairPreviewCandidateSchema, {
		minItems: 2,
		maxItems: 3,
		description: "Two or three alternative code-edit hypotheses for the same observed visual defect.",
	}),
}, { additionalProperties: false });

export const VISUALIZE_CODE_HYPOTHESIS_PARAMETERS_SCHEMA = Type.Object({
	region_id: Type.String({ pattern: "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$" }),
	target_roi: roiSchema,
	edits: Type.Array(editSchema, {
		minItems: 1,
		maxItems: 8,
		description: "The same exact oldText/newText replacements used by Pi's native edit tool. All oldText values match the original formal source, not earlier edits in this array.",
	}),
}, { additionalProperties: false });

export interface PiStyleEdit {
	oldText: string;
	newText: string;
}

export interface CodeHypothesisRecord {
	hypothesis_id: string;
	epistemic_status: "unverified_hypothesis";
	region_id: string;
	target_roi: NormalizedRoi;
	edits: PiStyleEdit[];
	formal_source_sha256: string;
	formal_step_sha256: string;
	candidate_source_sha256: string;
	patch: string;
	execution: ExecutionPayload;
	visual_encoding: CodeHypothesisVisualEncoding;
	geometry: VisualReviewGeometryEffect[];
	card_path: string;
	card_sha256: string;
	created_at: string;
}

interface PersistedVisualReviewState {
	version: 4;
	profile: VisualReviewProfile;
	fixture_id: string;
	locked_view_manifest_sha256: string;
	hypotheses: CodeHypothesisRecord[];
	next_hypothesis_index: number;
	revision: number;
	started_at: string;
	updated_at: string;
}

export interface VisualReviewState {
	profile: VisualReviewProfile;
	fixtureId: string;
	lockedViewManifestSha256: string;
	hypotheses: CodeHypothesisRecord[];
	nextHypothesisIndex: number;
	revision: number;
	statePath: string;
	startedAt: string;
}

export interface FormalState {
	sourceSha256: string;
	stepSha256: string;
	source: string;
}

export interface CurrentFrame {
	viewSpec: DrawingViewSpec;
	transform: DrawingSimilarityTransform;
	referencePath: string;
	currentRenderPath: string;
	projectionOrigin: [number, number, number];
	projectionAnchorStepPath: string;
	projectionAnchorStepSha256: string;
	sectionOrigin?: [number, number, number];
	cameraSha256: string;
	transformSha256: string;
}

function sha256Text(value: string): string {
	return createHash("sha256").update(value).digest("hex");
}

function sha256Json(value: unknown): string {
	return sha256Text(JSON.stringify(value));
}

function finiteVector3(value: unknown): value is [number, number, number] {
	return Array.isArray(value) && value.length === 3 && value.every(component => typeof component === "number" && Number.isFinite(component));
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

/** Apply the exact replacement semantics of Pi's native edit tool to an in-memory source copy. */
export function applyPiStyleEdits(source: string, edits: readonly PiStyleEdit[]): string {
	if (edits.length === 0) throw new Error("edits must contain at least one replacement");
	const replacements = edits.map((edit, editIndex) => {
		if (edit.oldText.length === 0) throw new Error(`edits[${editIndex}].oldText must not be empty`);
		const occurrence = countOccurrences(source, edit.oldText);
		if (occurrence.count !== 1) {
			throw new Error(`edits[${editIndex}].oldText must occur exactly once in the current formal solution.py; found ${occurrence.count}`);
		}
		return {
			start: occurrence.index,
			end: occurrence.index + edit.oldText.length,
			newText: edit.newText,
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
	if (result === source) throw new Error("The executable hypothesis must change the isolated solution.py copy");
	return result;
}

function roiFromObject(value: { x_min: number; y_min: number; x_max: number; y_max: number }): NormalizedRoi {
	const roi: NormalizedRoi = [value.x_min, value.y_min, value.x_max, value.y_max];
	if (roi[0] >= roi[2] || roi[1] >= roi[3]) throw new Error("target_roi must satisfy x_min < x_max and y_min < y_max");
	if (roi[2] - roi[0] < 50 || roi[3] - roi[1] < 50) {
		throw new Error("target_roi width and height must each be at least 50");
	}
	return roi;
}

function cloneRecord(record: CodeHypothesisRecord): CodeHypothesisRecord {
	return {
		...record,
		target_roi: [...record.target_roi],
		edits: record.edits.map((edit) => ({ ...edit })),
		execution: {
			...record.execution,
			files_produced: { ...record.execution.files_produced },
		},
		visual_encoding: { ...record.visual_encoding },
		geometry: record.geometry.map((effect) => ({
			...effect,
			bbox_mm: { ...effect.bbox_mm },
			centroid_mm: [...effect.centroid_mm],
			surface_types: [...effect.surface_types],
		})),
	};
}

function formatMillimeters(value: number): string {
	const compact = value.toFixed(6).replace(/0+$/u, "").replace(/\.$/u, "");
	return compact.includes(".") ? compact : `${compact}.0`;
}

function geometryEffectLine(
	record: CodeHypothesisRecord,
	relation: VisualReviewGeometryEffect["relation"],
	label: "Added" | "Removed",
): string {
	const effect = record.geometry.find((candidate) => candidate.relation === relation);
	if (!effect) return `${label} 3D geometry: none.`;
	const size = [effect.bbox_mm.x, effect.bbox_mm.y, effect.bbox_mm.z]
		.map(formatMillimeters)
		.join(" × ");
	const subject = label.toLowerCase();
	const solids = `${effect.solid_count} solid${effect.solid_count === 1 ? "" : "s"}`;
	const faces = `${effect.face_count} face${effect.face_count === 1 ? "" : "s"}`;
	const edges = `${effect.edge_count} edge${effect.edge_count === 1 ? "" : "s"}`;
	return `${label} 3D geometry: XYZ bounding-box size = ${size} mm. This ${subject} geometry consists of ${solids} with ${faces} and ${edges}.`;
}

function editPreviewToolText(record: CodeHypothesisRecord): string {
	return [
		"Edit preview generated. The submitted code edits were used only to generate the preview; they were not written to solution.py, and output.step was not replaced.",
		"The left panel shows the selected region of the Target drawing. The right panel shows the code-edit effect overlaid on the CAD before editing.",
		"Colors: gray = visible CAD lines before editing; green = visible lines added by the code edits; red = visible lines removed by the code edits.",
		geometryEffectLine(record, "added_by_proposal", "Added"),
		geometryEffectLine(record, "removed_by_proposal", "Removed"),
	].join("\n");
}

function persistedState(state: VisualReviewState): PersistedVisualReviewState {
	return {
		version: 4,
		profile: state.profile,
		fixture_id: state.fixtureId,
		locked_view_manifest_sha256: state.lockedViewManifestSha256,
		hypotheses: state.hypotheses.map(cloneRecord),
		next_hypothesis_index: state.nextHypothesisIndex,
		revision: state.revision,
		started_at: state.startedAt,
		updated_at: isoNow(),
	};
}

async function persistState(workspace: string, state: VisualReviewState): Promise<void> {
	const root = resolve(workspace);
	const path = resolve(root, state.statePath);
	const rel = relative(root, path);
	if (rel === ".." || rel.startsWith("../") || isAbsolute(rel)) {
		throw new InfrastructureError("Code-hypothesis state escaped the workspace");
	}
	await mkdir(dirname(path), { recursive: true });
	await writeJsonAtomic(path, persistedState(state));
}

export async function restoreVisualReviewState(options: {
	workspace: string;
	fixtureId: string;
	profile: VisualReviewProfile;
	lockedViewManifestSha256: string;
	statePath?: string;
}): Promise<VisualReviewState> {
	const statePath = options.statePath ?? VISUAL_REVIEW_STATE_PATH;
	const stored = await readJson<PersistedVisualReviewState>(resolve(options.workspace, statePath)).catch(() => null);
	if (!stored) {
		return {
			profile: options.profile,
			fixtureId: options.fixtureId,
			lockedViewManifestSha256: options.lockedViewManifestSha256,
			hypotheses: [],
			nextHypothesisIndex: 1,
			revision: 0,
			statePath,
			startedAt: isoNow(),
		};
	}
	if (
		stored.version !== 4 || stored.profile !== VISUAL_REVIEW_PROFILE || stored.profile !== options.profile ||
		stored.fixture_id !== options.fixtureId ||
		stored.locked_view_manifest_sha256 !== options.lockedViewManifestSha256 ||
		!Array.isArray(stored.hypotheses) || stored.hypotheses.length > MAX_STORED_HYPOTHESES ||
		!Number.isInteger(stored.next_hypothesis_index) || stored.next_hypothesis_index < 1 ||
		!Number.isInteger(stored.revision) || stored.revision < 0
	) throw new InfrastructureError("Persisted code-hypothesis state does not match the continuation");
	return {
		profile: stored.profile,
		fixtureId: stored.fixture_id,
		lockedViewManifestSha256: stored.locked_view_manifest_sha256,
		hypotheses: stored.hypotheses.map(cloneRecord),
		nextHypothesisIndex: stored.next_hypothesis_index,
		revision: stored.revision,
		statePath,
		startedAt: stored.started_at,
	};
}

export async function currentFormalState(options: {
	workspace: string;
	worker: WorkerClient;
	controller: RunController;
	getResult: () => AgentResult;
}): Promise<FormalState> {
	const latest = await options.worker.request<WorkerInspectResponse>(
		options.workspace,
		{ protocol_version: PROTOCOL_VERSION, operation: "inspect" },
		{ signal: options.controller.signal },
	);
	const inspected = await inspectFormalCandidate(options.workspace, latest.artifact, options.getResult());
	if (!inspected.formal_execution_current || !artifactAccepted(latest.artifact) || !inspected.source_sha256 || !latest.artifact.sha256) {
		throw new Error(`Code hypothesis requires a valid, watertight, formally executed Current CAD. ${inspected.reason}`);
	}
	const source = await readFile(join(options.workspace, SOLUTION_FILE), "utf8");
	if (sha256Text(source) !== inspected.source_sha256) {
		throw new InfrastructureError("Formal solution.py changed while preparing a code hypothesis");
	}
	return { sourceSha256: inspected.source_sha256, stepSha256: latest.artifact.sha256, source };
}

export async function currentFrame(options: {
	workspace: string;
	worker: WorkerClient;
	controller: RunController;
	manifest: DrawingViewManifest;
	formal: FormalState;
	regionId: string;
	outputRoot: string;
}): Promise<CurrentFrame> {
	const region = options.manifest.regions.find(({ region_id }) => region_id === options.regionId);
	if (!region?.selected) throw new Error(`Unknown or unavailable comparison region ${options.regionId}`);
	assertSupportedViewSpec(region.selected.view_spec);
	const viewSpec = {
		...region.selected.view_spec,
		name: `code-hypothesis-${sha256Json([options.formal.stepSha256, options.regionId]).slice(0, 16)}`,
	};
	const fixedFrame = options.manifest.projection_frame;
	if (fixedFrame && (fixedFrame.policy !== "fixed" || fixedFrame.anchor_step_sha256 !== options.manifest.step_sha256)) {
		throw new InfrastructureError("Calibrated projection anchor does not match the locked View Search STEP");
	}
	const anchorPath = fixedFrame?.anchor_step_path ?? "output.step";
	const anchorSha256 = fixedFrame?.anchor_step_sha256 ?? options.formal.stepSha256;
	let transform: DrawingSimilarityTransform;
	if (fixedFrame || options.formal.stepSha256 === options.manifest.step_sha256) {
		transform = region.selected.transform;
	} else {
		const response = await options.worker.request<WorkerRankDrawingViewsResponse>(options.workspace, {
			protocol_version: PROTOCOL_VERSION,
			operation: "rank_drawing_views",
			region_id: region.region_id,
			crop_path: region.crop_path,
			step_path: "output.step",
			candidates: [{ candidate_id: viewSpec.name, view_spec: viewSpec }],
			top_k: 1,
			allow_reflection: false,
			max_rotation_deg: 6,
			min_roi_fill: 0.70,
			seed: 0,
			max_iterations: 45,
			output_dir: join(options.outputRoot, "current-frame"),
		}, { timeoutMs: 240_000, signal: options.controller.signal });
		requireDrawingSuccess(response, "rank_drawing_views");
		const ranked = response.rankings[0];
		if (!ranked || response.rankings.length !== 1 || response.step_sha256 !== options.formal.stepSha256 || ranked.transform.reflected) {
			throw new InfrastructureError("Current code-hypothesis frame drifted from the formal STEP");
		}
		transform = ranked.transform;
	}
	if (transform.reflected) throw new InfrastructureError("Code hypothesis does not permit reflected alignment");
	const current = await options.worker.request<WorkerFixedProjectionResponse>(options.workspace, {
		protocol_version: PROTOCOL_VERSION,
		operation: "render_fixed_projection",
		region_id: options.regionId,
		crop_path: region.crop_path,
		step_path: "output.step",
		projection_anchor_step_path: anchorPath,
		view_spec: viewSpec,
		transform,
		output_dir: join(options.outputRoot, "current-fixed-frame"),
	}, { timeoutMs: 180_000, signal: options.controller.signal });
	requireDrawingSuccess(current, "render_fixed_projection");
	const transformSha256 = sha256Json(transform);
	const cameraHash = cameraSha256(viewSpec);
	if (
		current.step_sha256 !== options.formal.stepSha256 ||
		current.projection_anchor_step_sha256 !== anchorSha256 ||
		!finiteVector3(current.projection_origin) ||
		(fixedFrame && (viewSpec.section_offset_fraction !== undefined
			? !finiteVector3(current.section_origin)
			: current.section_origin !== undefined)) ||
		sha256Json(current.transform) !== transformSha256 ||
		cameraSha256(current.view_spec) !== cameraHash
	) throw new InfrastructureError("Current CAD projection drifted from the frozen code-hypothesis frame");
	return {
		viewSpec,
		transform,
		referencePath: region.crop_path,
		currentRenderPath: current.render_path,
		projectionOrigin: current.projection_origin,
		projectionAnchorStepPath: anchorPath,
		projectionAnchorStepSha256: anchorSha256,
		...(current.section_origin ? { sectionOrigin: current.section_origin } : {}),
		cameraSha256: cameraHash,
		transformSha256,
	};
}

export async function assertFormalUnchanged(workspace: string, formal: FormalState): Promise<void> {
	const [sourceSha256, stepSha256] = await Promise.all([
		sha256File(join(workspace, SOLUTION_FILE)),
		sha256File(join(workspace, "output.step")),
	]);
	if (sourceSha256 !== formal.sourceSha256 || stepSha256 !== formal.stepSha256) {
		throw new InfrastructureError("Code hypothesis changed the formal source or STEP");
	}
}

function candidateFailure(response: WorkerRepairCandidateResponse): string {
	if (response.execution.timed_out) return "temporary CAD execution timed out";
	const stderr = response.execution.stderr.trim().split(/\r?\n/u).filter(Boolean).at(-1);
	if (stderr) return stderr;
	if (response.artifact.validation_error) return response.artifact.validation_error;
	return `candidate status=${response.artifact.status}`;
}

export async function injectPersistentVisualReviews<T>(options: {
	messages: T[];
	workspace: string;
	state: VisualReviewState;
	result: Pick<AgentResult, "last_executed_source_sha256" | "last_executed_step_sha256">;
}): Promise<T[]> {
	const filtered = options.messages.filter((message) => (
		typeof message !== "object" || message === null ||
		(message as { customType?: unknown }).customType !== VISUAL_REVIEW_CONTEXT_TYPE
	));
	const fresh = options.state.hypotheses.filter((hypothesis) => (
		hypothesis.formal_source_sha256 === options.result.last_executed_source_sha256 &&
		hypothesis.formal_step_sha256 === options.result.last_executed_step_sha256
	));
	if (fresh.length === 0) return filtered;
	const content: Array<TextContent | ImageContent> = [];
	for (const hypothesis of fresh) {
		const card = await inspectPngFile(options.workspace, hypothesis.card_path);
		if (card.sha256 !== hypothesis.card_sha256) {
			throw new InfrastructureError(`Code hypothesis ${hypothesis.hypothesis_id} no longer matches its hash`);
		}
		content.push({ type: "text", text: editPreviewToolText(hypothesis) });
		content.push(await imageFromWorkspace(options.workspace, hypothesis.card_path));
	}
	return [...filtered, {
		role: "custom",
		customType: VISUAL_REVIEW_CONTEXT_TYPE,
		content,
		display: false,
		timestamp: Date.now(),
	} as T];
}

export function createVisualReviewTool(options: {
	workspace: string;
	state: VisualReviewState;
	lockedManifest: DrawingViewManifest;
	worker: WorkerClient;
	controller: RunController;
	getResult: () => AgentResult;
}) {
	const availableRegionIds = options.lockedManifest.regions
		.filter(({ selected }) => selected !== undefined)
		.map(({ region_id }) => region_id);
	return defineTool({
		name: "visualize_code_hypothesis",
		label: "Preview CAD edit",
		description: "Preview how exact edits to solution.py change the current CAD beside a selected Target drawing ROI. Each edits[].oldText must match exactly once in the current file, and edits must not overlap. The edits run only in an isolated copy. Returns a two-panel Target ROI / Edit Preview image showing visible line changes in one fixed view, together with the bounding-box size and topology counts of 3D geometry added or removed by the edits. The formal solution.py and output.step remain unchanged.",
		parameters: VISUALIZE_CODE_HYPOTHESIS_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			if (!availableRegionIds.includes(params.region_id)) {
				throw new Error(`Unknown or unavailable comparison region ${params.region_id}`);
			}
			const formal = await currentFormalState(options);
			if (Buffer.byteLength(formal.source, "utf8") > 512_000) {
				throw new Error("The formal source is too large for an isolated code hypothesis");
			}
			const edits = params.edits.map((edit) => ({ oldText: edit.oldText, newText: edit.newText }));
			const candidateSource = applyPiStyleEdits(formal.source, edits);
			const candidateSourceSha256 = sha256Text(candidateSource);
			const patch = generateUnifiedPatch(SOLUTION_FILE, formal.source, candidateSource);
			const targetRoi = roiFromObject(params.target_roi);
			const hypothesisId = `H${String(options.state.nextHypothesisIndex).padStart(3, "0")}`;
			const digest = sha256Json([formal.sourceSha256, formal.stepSha256, params.region_id, targetRoi, edits]).slice(0, 16);
			const outputRoot = join("drawing_feedback", "code-hypotheses", `${hypothesisId}-${digest}`);
			const submittedSourcePath = join(outputRoot, "submitted-solution.py");
			await mkdir(join(options.workspace, outputRoot), { recursive: true });
			await writeFile(join(options.workspace, submittedSourcePath), candidateSource, { encoding: "utf8", mode: 0o600 });

			const execution = await options.worker.request<WorkerRepairCandidateResponse>(options.workspace, {
				protocol_version: PROTOCOL_VERSION,
				operation: "execute_repair_candidate",
				candidate_id: `code-hypothesis-${digest}`,
				source_path: submittedSourcePath,
				formal_source_sha256: formal.sourceSha256,
				formal_step_sha256: formal.stepSha256,
				timeout_s: 120,
				output_dir: join(outputRoot, "execution"),
			}, { timeoutMs: 135_000, signal: options.controller.signal });
			if (
				execution.source_sha256 !== candidateSourceSha256 ||
				execution.formal_source_sha256 !== formal.sourceSha256 ||
				execution.formal_step_sha256 !== formal.stepSha256 ||
				execution.formal_unchanged !== true
			) throw new InfrastructureError("Isolated code-hypothesis execution violated its formal hash contract");
			if (!artifactAccepted(execution.artifact) || !execution.artifact.path || !execution.artifact.sha256) {
				await assertFormalUnchanged(options.workspace, formal);
				throw new Error(`Executable hypothesis failed: ${candidateFailure(execution)}. Formal solution.py and output.step were not modified.`);
			}

			const frame = await currentFrame({
				workspace: options.workspace,
				worker: options.worker,
				controller: options.controller,
				manifest: options.lockedManifest,
				formal,
				regionId: params.region_id,
				outputRoot,
			});
			const editedRender = await options.worker.request<WorkerFixedProjectionResponse>(options.workspace, {
				protocol_version: PROTOCOL_VERSION,
				operation: "render_fixed_projection",
				region_id: params.region_id,
				crop_path: frame.referencePath,
				step_path: execution.artifact.path,
				projection_anchor_step_path: frame.projectionAnchorStepPath,
				view_spec: frame.viewSpec,
				transform: frame.transform,
				output_dir: join(outputRoot, "edited-frame"),
			}, { timeoutMs: 180_000, signal: options.controller.signal });
			requireDrawingSuccess(editedRender, "render_fixed_projection");
			if (
				editedRender.step_sha256 !== execution.artifact.sha256 ||
				editedRender.projection_anchor_step_sha256 !== frame.projectionAnchorStepSha256 ||
				!finiteVector3(editedRender.projection_origin) ||
				editedRender.projection_origin.some((value, index) => value !== frame.projectionOrigin[index]) ||
				sha256Json(editedRender.section_origin ?? null) !== sha256Json(frame.sectionOrigin ?? null) ||
				sha256Json(editedRender.transform) !== frame.transformSha256 ||
				cameraSha256(editedRender.view_spec) !== frame.cameraSha256
			) throw new InfrastructureError("Proposed CAD projection drifted from the frozen Current frame");

			const response = await options.worker.request<WorkerCodeEditVisualizationResponse>(options.workspace, {
				protocol_version: PROTOCOL_VERSION,
				operation: "compose_code_edit_visualization",
				region_id: params.region_id,
				hypothesis_id: hypothesisId,
				target_roi: targetRoi,
				reference_path: frame.referencePath,
				current_render_path: frame.currentRenderPath,
				edited_render_path: editedRender.render_path,
				current_step_path: "output.step",
				edited_step_path: execution.artifact.path,
				view_spec: frame.viewSpec,
				transform: frame.transform,
				projection_origin: frame.projectionOrigin,
				output_dir: join(outputRoot, "card"),
			}, { timeoutMs: 240_000, signal: options.controller.signal });
			requireDrawingSuccess(response, "compose_code_edit_visualization");
			if (
				response.hypothesis_id !== hypothesisId || response.region_id !== params.region_id ||
				response.target_roi.some((value, index) => value !== targetRoi[index])
			) throw new InfrastructureError("Code hypothesis returned mismatched structured evidence");
			const card = await inspectPngFile(options.workspace, response.card_path);
			if (card.sha256 !== response.sha256) throw new InfrastructureError("Code-hypothesis card hash mismatch");
			await assertFormalUnchanged(options.workspace, formal);

			const record: CodeHypothesisRecord = {
				hypothesis_id: hypothesisId,
				epistemic_status: "unverified_hypothesis",
				region_id: params.region_id,
				target_roi: targetRoi,
				edits,
				formal_source_sha256: formal.sourceSha256,
				formal_step_sha256: formal.stepSha256,
				candidate_source_sha256: candidateSourceSha256,
				patch,
				execution: execution.execution,
				visual_encoding: response.visual_encoding,
				geometry: response.geometry,
				card_path: response.card_path,
				card_sha256: response.sha256,
				created_at: isoNow(),
			};
			if (options.state.hypotheses.length >= MAX_STORED_HYPOTHESES) options.state.hypotheses.shift();
			options.state.hypotheses.push(record);
			options.state.nextHypothesisIndex += 1;
			options.state.revision += 1;
			await persistState(options.workspace, options.state);
			return {
				content: [
					{ type: "text" as const, text: editPreviewToolText(record) },
					await imageFromWorkspace(options.workspace, response.card_path),
				],
				details: { hypothesis: cloneRecord(record) },
			};
		},
	});
}

function repairPreviewToolText(response: WorkerRepairCandidatePreviewResponse): string {
	const current = response.current_topology;
	const lines = [
		`The image compares the Original drawing, Current CAD, and ${response.candidates.length} isolated repair candidates for ${response.region_id}.`,
		"In each candidate panel, gray shows the candidate CAD, green highlights exact B-rep geometry added relative to Current, and red highlights exact B-rep geometry removed relative to Current.",
		"Topology counts: b0 = connected solid components; b1 = independent through-handles or through-holes; b2 = enclosed internal voids. These are candidate-versus-Current diagnostics, not a score and not a comparison with hidden ground truth.",
		`Current topology: b0=${current.b0}, b1=${current.b1}, b2=${current.b2}.`,
	];
	for (const candidate of response.candidates) {
		const topology = candidate.topology;
		const delta = candidate.topology_delta;
		lines.push(
			`${candidate.candidate_id} (${candidate.hypothesis}): b0=${topology.b0}, b1=${topology.b1}, b2=${topology.b2}; ΔCurrent=(b0 ${delta.b0 >= 0 ? "+" : ""}${delta.b0}, b1 ${delta.b1 >= 0 ? "+" : ""}${delta.b1}, b2 ${delta.b2 >= 0 ? "+" : ""}${delta.b2}).`,
		);
		for (const relation of ["added_by_proposal", "removed_by_proposal"] as const) {
			const effect = candidate.geometry.find((item) => item.relation === relation);
			const label = relation === "added_by_proposal" ? "Added" : "Removed";
			if (!effect) {
				lines.push(`${candidate.candidate_id} ${label.toLowerCase()} geometry: none.`);
				continue;
			}
			const size = [effect.bbox_mm.x, effect.bbox_mm.y, effect.bbox_mm.z]
				.map(formatMillimeters)
				.join(" × ");
			lines.push(
				`${candidate.candidate_id} ${label.toLowerCase()} geometry: ${formatMillimeters(effect.volume_mm3)} mm³; XYZ bounding-box size ${size} mm; ${effect.face_count} faces and ${effect.edge_count} edges.`,
			);
		}
	}
	lines.push("The candidates were not scored, selected, or applied. The formal solution.py and output.step remain unchanged.");
	return lines.join("\n");
}

/** Create the multi-candidate, exact-B-rep repair preview used by the new profile. */
export function createRepairPreviewTool(options: {
	workspace: string;
	lockedManifest: DrawingViewManifest;
	worker: WorkerClient;
	controller: RunController;
	getResult: () => AgentResult;
}) {
	const availableRegionIds = options.lockedManifest.regions
		.filter(({ selected }) => selected !== undefined)
		.map(({ region_id }) => region_id);
	return defineTool({
		name: "preview_repair_candidates",
		label: "Preview repair candidates",
		description: `Preview two or three alternative solution.py edits for one observed visual defect before changing the formal model. Every candidate starts from the same currently executed source and is built in isolation. The returned full-view image compares the Original drawing, Current CAD, and each candidate; candidate panels highlight exact 3D B-rep geometry added in green and removed in red. The text also reports each candidate's b0 connected-component count, b1 through-handle/through-hole count, and b2 enclosed-void count, together with changes from Current. It does not use pixel residuals, score candidates, select a repair, or modify formal solution.py/output.step. Available region_id values: ${availableRegionIds.join(", ")}.`,
		parameters: PREVIEW_REPAIR_CANDIDATES_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			if (!availableRegionIds.includes(params.region_id)) {
				throw new Error(`Unknown or unavailable comparison region ${params.region_id}`);
			}
			const ids = params.candidates.map(({ id }) => id);
			if (new Set(ids).size !== ids.length) throw new Error("Candidate ids must be unique");
			const formal = await currentFormalState(options);
			if (Buffer.byteLength(formal.source, "utf8") > 512_000) {
				throw new Error("The formal source is too large for isolated repair previews");
			}
			const prepared = params.candidates.map((candidate) => {
				const edits = candidate.edits.map((edit) => ({ oldText: edit.oldText, newText: edit.newText }));
				const source = applyPiStyleEdits(formal.source, edits);
				return {
					id: candidate.id,
					hypothesis: candidate.hypothesis.trim(),
					edits,
					source,
					sourceSha256: sha256Text(source),
					patch: generateUnifiedPatch(SOLUTION_FILE, formal.source, source),
				};
			});
			if (new Set(prepared.map(({ sourceSha256 }) => sourceSha256)).size !== prepared.length) {
				throw new Error("Candidates must produce distinct source variants");
			}
			const digest = sha256Json([
				REPAIR_PREVIEW_PROFILE,
				formal.sourceSha256,
				formal.stepSha256,
				params.region_id,
				prepared.map(({ id, hypothesis, edits }) => ({ id, hypothesis, edits })),
			]).slice(0, 16);
			const outputRoot = join("drawing_feedback", "repair-candidate-previews", digest);
			await mkdir(join(options.workspace, outputRoot), { recursive: true });

			const completed: Array<{
				id: string;
				hypothesis: string;
				edits: PiStyleEdit[];
				patch: string;
				sourceSha256: string;
				stepSha256: string;
				stepPath: string;
				renderPath: string;
				execution: ExecutionPayload;
			}> = [];
			for (const [index, candidate] of prepared.entries()) {
				const candidateRoot = join(outputRoot, `${String(index + 1).padStart(2, "0")}-${candidate.id}`);
				const sourcePath = join(candidateRoot, "submitted-solution.py");
				await mkdir(join(options.workspace, candidateRoot), { recursive: true });
				await writeFile(join(options.workspace, sourcePath), candidate.source, { encoding: "utf8", mode: 0o600 });
				const execution = await options.worker.request<WorkerRepairCandidateResponse>(options.workspace, {
					protocol_version: PROTOCOL_VERSION,
					operation: "execute_repair_candidate",
					candidate_id: `repair-preview-${digest}-${candidate.id}`,
					source_path: sourcePath,
					formal_source_sha256: formal.sourceSha256,
					formal_step_sha256: formal.stepSha256,
					timeout_s: 120,
					output_dir: join(candidateRoot, "execution"),
				}, { timeoutMs: 135_000, signal: options.controller.signal });
				if (
					execution.source_sha256 !== candidate.sourceSha256 ||
					execution.formal_source_sha256 !== formal.sourceSha256 ||
					execution.formal_step_sha256 !== formal.stepSha256 ||
					execution.formal_unchanged !== true
				) throw new InfrastructureError(`Repair candidate ${candidate.id} violated the isolated execution contract`);
				if (!artifactAccepted(execution.artifact) || !execution.artifact.path || !execution.artifact.sha256) {
					await assertFormalUnchanged(options.workspace, formal);
					const message = `Candidate ${candidate.id} failed: ${candidateFailure(execution)}. No preview image was returned. The formal solution.py and output.step remain unchanged.`;
					await writeJsonAtomic(join(options.workspace, outputRoot, "audit.json"), {
						version: 1,
						profile: REPAIR_PREVIEW_PROFILE,
						region_id: params.region_id,
						formal_source_sha256: formal.sourceSha256,
						formal_step_sha256: formal.stepSha256,
						status: "candidate_failed",
						failed_candidate_id: candidate.id,
						message,
						created_at: isoNow(),
					});
					return {
						content: [{ type: "text" as const, text: message }],
						details: {
							status: "candidate_failed",
							candidate_id: candidate.id as string | null,
							preview: null as WorkerRepairCandidatePreviewResponse | null,
							audit_path: join(outputRoot, "audit.json"),
						},
					};
				}
				completed.push({
					id: candidate.id,
					hypothesis: candidate.hypothesis,
					edits: candidate.edits,
					patch: candidate.patch,
					sourceSha256: candidate.sourceSha256,
					stepSha256: execution.artifact.sha256,
					stepPath: execution.artifact.path,
					renderPath: "",
					execution: execution.execution,
				});
			}

			const frame = await currentFrame({
				workspace: options.workspace,
				worker: options.worker,
				controller: options.controller,
				manifest: options.lockedManifest,
				formal,
				regionId: params.region_id,
				outputRoot,
			});
			for (const [index, candidate] of completed.entries()) {
				const rendered = await options.worker.request<WorkerFixedProjectionResponse>(options.workspace, {
					protocol_version: PROTOCOL_VERSION,
					operation: "render_fixed_projection",
					region_id: params.region_id,
					crop_path: frame.referencePath,
					step_path: candidate.stepPath,
					projection_anchor_step_path: frame.projectionAnchorStepPath,
					view_spec: frame.viewSpec,
					transform: frame.transform,
					output_dir: join(outputRoot, `${String(index + 1).padStart(2, "0")}-${candidate.id}`, "fixed-frame"),
				}, { timeoutMs: 180_000, signal: options.controller.signal });
				requireDrawingSuccess(rendered, "render_fixed_projection");
				if (
					rendered.step_sha256 !== candidate.stepSha256 ||
					rendered.projection_anchor_step_sha256 !== frame.projectionAnchorStepSha256 ||
					!finiteVector3(rendered.projection_origin) ||
					rendered.projection_origin.some((value, originIndex) => value !== frame.projectionOrigin[originIndex]) ||
					sha256Json(rendered.section_origin ?? null) !== sha256Json(frame.sectionOrigin ?? null) ||
					sha256Json(rendered.transform) !== frame.transformSha256 ||
					cameraSha256(rendered.view_spec) !== frame.cameraSha256
				) throw new InfrastructureError(`Repair candidate ${candidate.id} drifted from the frozen Current frame`);
				candidate.renderPath = rendered.render_path;
			}

			const response = await options.worker.request<WorkerRepairCandidatePreviewResponse | WorkerDrawingErrorResponse>(options.workspace, {
				protocol_version: PROTOCOL_VERSION,
				operation: "compose_repair_candidate_preview",
				region_id: params.region_id,
				reference_path: frame.referencePath,
				current_render_path: frame.currentRenderPath,
				current_step_path: "output.step",
				view_spec: frame.viewSpec,
				transform: frame.transform,
				projection_origin: frame.projectionOrigin,
				candidates: completed.map((candidate) => ({
					candidate_id: candidate.id,
					hypothesis: candidate.hypothesis,
					render_path: candidate.renderPath,
					step_path: candidate.stepPath,
				})),
				output_dir: join(outputRoot, "card"),
			}, { timeoutMs: 300_000, signal: options.controller.signal });
			await assertFormalUnchanged(options.workspace, formal);
			if (response.status !== "ok" || response.operation !== "compose_repair_candidate_preview") {
				const message = `Repair preview failed: ${response.error?.message ?? "unknown drawing error"}. No preview image was returned. The formal solution.py and output.step remain unchanged.`;
				await writeJsonAtomic(join(options.workspace, outputRoot, "audit.json"), {
					version: 1,
					profile: REPAIR_PREVIEW_PROFILE,
					region_id: params.region_id,
					formal_source_sha256: formal.sourceSha256,
					formal_step_sha256: formal.stepSha256,
					status: "preview_failed",
					message,
					candidates: completed,
					created_at: isoNow(),
				});
				return {
					content: [{ type: "text" as const, text: message }],
					details: {
						status: "preview_failed",
						candidate_id: null as string | null,
						preview: null as WorkerRepairCandidatePreviewResponse | null,
						audit_path: join(outputRoot, "audit.json"),
					},
				};
			}
			if (response.region_id !== params.region_id || response.candidates.length !== completed.length) {
				throw new InfrastructureError("Repair preview returned mismatched structured evidence");
			}
			const card = await inspectPngFile(options.workspace, response.card_path);
			if (card.sha256 !== response.sha256) throw new InfrastructureError("Repair-preview card hash mismatch");
			const auditPath = join(outputRoot, "audit.json");
			await writeJsonAtomic(join(options.workspace, auditPath), {
				version: 1,
				profile: REPAIR_PREVIEW_PROFILE,
				region_id: params.region_id,
				formal_source_sha256: formal.sourceSha256,
				formal_step_sha256: formal.stepSha256,
				camera_sha256: frame.cameraSha256,
				transform_sha256: frame.transformSha256,
				candidates: completed,
				preview: response,
				created_at: isoNow(),
			});
			return {
				content: [
					{ type: "text" as const, text: repairPreviewToolText(response) },
					await imageFromWorkspace(options.workspace, response.card_path),
				],
				details: {
					status: "ok",
					candidate_id: null as string | null,
					preview: response as WorkerRepairCandidatePreviewResponse | null,
					audit_path: auditPath,
				},
			};
		},
	});
}
