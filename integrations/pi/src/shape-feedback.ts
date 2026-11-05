import { createHash, randomUUID } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import { defineTool, generateUnifiedPatch } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import { imageFromWorkspace, inspectPngFile, requireDrawingSuccess } from "./drawing-reflection.js";
import { InfrastructureError } from "./errors.js";
import { isoNow, writeJsonAtomic } from "./files.js";
import type { WorkerClient } from "./local-worker.js";
import { cameraSha256 } from "./main-view-reflect.js";
import type { RunController } from "./runner.js";
import { artifactAccepted, sha256File } from "./solution.js";
import {
	PROTOCOL_VERSION,
	SOLUTION_FILE,
	type AgentResult,
	type DrawingViewManifest,
	type NormalizedRoi,
	type RepairPreviewTopology,
	type WorkerCandidateTopologyErrorResponse,
	type WorkerCandidateTopologyResponse,
	type WorkerDrawingErrorResponse,
	type WorkerFixedProjectionResponse,
	type WorkerRepairCandidateResponse,
	type WorkerShapeFeedbackResponse,
} from "./types.js";
import {
	applyPiStyleEdits,
	assertFormalUnchanged,
	currentFormalState,
	currentFrame,
	type CurrentFrame,
} from "./visual-review.js";

const regionPattern = "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$";
const candidatePattern = "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$";
export const SHAPE_FEEDBACK_LATEST_PATH = "drawing_feedback/shape-feedback/latest-preview.json";

export const PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA = Type.Object({
	views: Type.Array(Type.Object({
		region_id: Type.String({ pattern: regionPattern }),
	}, { additionalProperties: false }), { minItems: 1, maxItems: 3 }),
	candidates: Type.Array(Type.Object({
		id: Type.String({ pattern: candidatePattern }),
		edits: Type.Array(Type.Object({
			oldText: Type.String({ minLength: 1, maxLength: 20_000 }),
			newText: Type.String({ maxLength: 20_000 }),
		}, { additionalProperties: false }), { minItems: 1, maxItems: 8 }),
	}, { additionalProperties: false }), { minItems: 1, maxItems: 1 }),
}, { additionalProperties: false });

function sha(value: string): string {
	return createHash("sha256").update(value).digest("hex");
}

function jsonSha(value: unknown): string {
	return sha(JSON.stringify(value));
}

type ShapeTopologyResponse = WorkerCandidateTopologyResponse | WorkerCandidateTopologyErrorResponse;
const bettiKeys = ["b0", "b1", "b2"] as const;
const panelLegend = "Left to right: Target | Current | Preview. Target is the original engineering-drawing view; Current and Preview are separate monochrome CAD 2D projections.";

/** Used in audit records and manual reports; not included in Preview's model-visible response. */
export function formatShapeTopologyFeedback(response: ShapeTopologyResponse): string {
	const meanings = "3D topology: b0 = disconnected solid components; b1 = independent loops; b2 = enclosed internal cavities. Delta = Preview Candidate minus Current.";
	if (response.status === "error") return `STEP topology unavailable: ${response.error.message}`;
	const absolute = (topology: RepairPreviewTopology) => bettiKeys.map((key) => `${key}=${topology[key]}`).join(", ");
	return [
		meanings,
		response.current_topology ? `Current: ${absolute(response.current_topology)}.` : `Current topology unavailable: ${response.current_error || "computation failed"}.`,
		...response.candidates.map((candidate) => {
			const label = `Preview Candidate ${candidate.candidate_id}`;
			if (candidate.status !== "ok" || !candidate.topology) return `${label}: topology and delta unavailable: ${candidate.error || "computation failed"}.`;
			const delta = candidate.topology_delta
				? `delta (${bettiKeys.map((key) => `${key}=${candidate.topology_delta![key] >= 0 ? "+" : ""}${candidate.topology_delta![key]}`).join(", ")})`
				: `delta unavailable: ${response.current_topology ? "comparison failed" : "Current topology unavailable"}`;
			return `${label}: ${absolute(candidate.topology)}; ${delta}.`;
		}),
		"Zero deltas mean unchanged topology counts; geometry may still change.",
	].join("\n");
}

function assertBettiCounts(topology: RepairPreviewTopology | null): void {
	if (topology && bettiKeys.some((key) => !Number.isSafeInteger(topology[key]) || topology[key] < 0)) {
		throw new InfrastructureError("Topology response returned invalid Betti counts");
	}
}

interface FrameAudit extends CurrentFrame {
	region_id: string;
	target_roi: NormalizedRoi;
	reference_sha256: string;
	current_render_sha256: string;
}

interface CandidateAudit {
	id: string;
	edits: Array<{ oldText: string; newText: string }>;
	patch: string;
	source_path: string;
	source_sha256: string;
	status: "pending" | "valid" | "invalid";
	execution?: WorkerRepairCandidateResponse;
	error?: string;
}

interface ProjectionAudit {
	region_id: string;
	candidate_id: string;
	step_sha256: string;
	render_path: string;
	render_sha256: string;
	camera_sha256: string;
	transform_sha256: string;
	projection_origin: [number, number, number];
	projection_anchor_step_sha256: string;
}

export interface ShapeFeedbackAudit {
	version: 1;
	kind: "shape_feedback_preview";
	evaluation_id: string;
	created_at: string;
	updated_at: string;
	status: "running" | "completed" | "partial" | "error";
	formal_source_sha256: string;
	formal_step_sha256: string;
	locked_view_manifest_sha256: string;
	requested_views: Array<{ region_id: string; target_roi: NormalizedRoi }>;
	candidates: CandidateAudit[];
	frames: FrameAudit[];
	projections: ProjectionAudit[];
	feedback: WorkerShapeFeedbackResponse[];
	topology: ShapeTopologyResponse | null;
	errors: string[];
}

/** Opt-in factory: not registered by any default run profile or prompt. */
export function createShapeFeedbackTool(options: {
	workspace: string;
	lockedManifest: DrawingViewManifest;
	worker: WorkerClient;
	controller: RunController;
	getResult: () => AgentResult;
	/** Reuse physical sections already fixed by the controller's calibration. */
	allowSections?: boolean;
}) {
	// Preserve the legacy ordinary-projection interface unless the caller enables
	// physical sections already fixed by the controller's calibration.
	const sectionRegions = new Set(options.lockedManifest.regions.filter((region) => !options.allowSections && (
		region.view_type === "section" ||
		region.selected?.view_spec.section_offset_fraction !== undefined ||
		region.selected?.view_spec.section_keep !== undefined
	)).map(({ region_id }) => region_id));
	const availableRegions = options.lockedManifest.regions
		.filter(({ selected, region_id }) => selected && !sectionRegions.has(region_id))
		.map(({ region_id }) => region_id);
	return defineTool({
		name: "preview_shape_feedback",
		label: "Preview shape feedback",
		description: `Use this tool to preview exactly one candidate code change in aligned
engineering-drawing views. Submit a candidates array containing one item with
an ID and one to eight code replacements; select one to three views that show
the relevant geometry. Each candidate starts from current formal solution.py,
never from a previous unadopted preview.
Select views by region_id; each card shows the complete aligned view.

The tool executes the candidate CAD in a temporary copy and projects
Current and Preview using the same camera and 2D transform.
It returns one three-panel card per view, arranged left to right:

- Target: the original engineering-drawing view.
- Current: the current CAD's 2D projection.
- Preview: the modified CAD's 2D projection.

${panelLegend}
All three panels use one shared display scale; Current and Preview use the
same camera, projection origin, and 2D transform. Images contain only the three
column labels and actual views, preserving text in the original drawing.

Compare the projections before and after the change against the target
drawing to assess whether geometric agreement has improved. Based on
this comparison, adopt the change, revise the code and preview again,
or discard the change.

This call does not modify solution.py or output.step in the workspace. To adopt
a candidate from the most recent preview, call adopt_shape_candidate with its
candidate_id. That tool applies the saved code replacements to solution.py.
Then call execute_build123d to rebuild the formal CAD and inspect_comparison
to review the final result. You may instead revise the edit and preview again,
or keep Current. A preview does not replace final formal CAD inspection.

Available region_id values: ${availableRegions.join(", ")}.`,
		parameters: PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			if ("hypothesis" in params || "edits" in params || "candidate" in params) {
				throw new Error("Submit one candidate with only id and edits in the candidates array; top-level hypothesis or edits are not supported");
			}
			if (!Array.isArray(params.views) || params.views.length < 1 || params.views.length > 3) {
				throw new Error("Provide one to three views");
			}
			if (params.views.some((view) => !view || typeof view.region_id !== "string" || Object.keys(view).some((key) => key !== "region_id"))) {
				throw new Error("Each view accepts only region_id; previews use the complete aligned view");
			}
			if (!Array.isArray(params.candidates) || params.candidates.length !== 1) {
				throw new Error("Provide exactly one candidate per call");
			}
			for (const candidate of params.candidates) {
				if (!candidate || typeof candidate.id !== "string" || !new RegExp(candidatePattern).test(candidate.id)) {
					throw new Error("Candidate ID must start with a letter or digit and contain only letters, digits, dots, underscores, or hyphens, at most 64 characters");
				}
				if (Object.keys(candidate).some((key) => key !== "id" && key !== "edits")) {
					throw new Error(`Candidate ${candidate.id}: only id and edits are supported`);
				}
				if (!Array.isArray(candidate.edits) || candidate.edits.length < 1 || candidate.edits.length > 8) {
					throw new Error(`Candidate ${candidate.id}: provide one to eight edits`);
				}
				if (candidate.edits.some((edit) => !edit || typeof edit.oldText !== "string" || typeof edit.newText !== "string" || !edit.oldText.length || edit.oldText.length > 20_000 || edit.newText.length > 20_000)) {
					throw new Error("Each edit needs non-empty oldText and string newText, each at most 20000 characters");
				}
			}
			if (new Set(params.views.map(({ region_id }) => region_id)).size !== params.views.length) throw new Error("View region_id values must be unique");
			const views = params.views.map(({ region_id }) => {
				if (sectionRegions.has(region_id)) throw new Error(`Section view ${region_id} is unsupported: shape feedback requires ordinary projections with a frozen frame`);
				if (!new RegExp(regionPattern).test(region_id) || !availableRegions.includes(region_id)) throw new Error(`Unknown or unavailable comparison region ${region_id}`);
				// Keep the worker/audit format while always comparing the complete view.
				return { region_id, target_roi: [0, 0, 1000, 1000] as NormalizedRoi };
			});
			const formal = await currentFormalState(options);
			if (Buffer.byteLength(formal.source, "utf8") > 512_000) throw new Error("The formal source is too large for isolated shape feedback");
			const prepared = params.candidates.map((candidate) => {
				const source = applyPiStyleEdits(formal.source, candidate.edits);
				return { ...candidate, source, sourceSha256: sha(source), patch: generateUnifiedPatch(SOLUTION_FILE, formal.source, source) };
			});
			const digest = jsonSha([formal.sourceSha256, formal.stepSha256, views, params.candidates]).slice(0, 16);
			const evaluationId = `${digest}-${randomUUID().slice(0, 8)}`;
			const outputRoot = join("drawing_feedback", "shape-feedback", evaluationId);
			const auditPath = join(outputRoot, "audit.json");
			const audit: ShapeFeedbackAudit = {
				version: 1, kind: "shape_feedback_preview", evaluation_id: evaluationId,
				created_at: isoNow(), updated_at: isoNow(), status: "running",
				formal_source_sha256: formal.sourceSha256, formal_step_sha256: formal.stepSha256,
				locked_view_manifest_sha256: jsonSha(options.lockedManifest), requested_views: views,
				candidates: prepared.map((candidate) => ({
					id: candidate.id, edits: candidate.edits, patch: candidate.patch,
					source_path: join(outputRoot, "candidates", candidate.id, "submitted-solution.py"),
					source_sha256: candidate.sourceSha256, status: "pending",
				})),
				frames: [], projections: [], feedback: [], topology: null, errors: [],
			};
			const topologyDiagnostics = new Set<string>();
			const recordTopologyDiagnostic = (message: string) => {
				audit.errors.push(message);
				topologyDiagnostics.add(message);
			};
			const persist = async () => {
				audit.updated_at = isoNow();
				await writeJsonAtomic(join(options.workspace, auditPath), audit);
			};
			await persist();
			// Candidate IDs refer to this preview even if it later fails or is
			// interrupted; adoption must never silently fall back to an older ID.
			await writeJsonAtomic(join(options.workspace, SHAPE_FEEDBACK_LATEST_PATH), {
				version: 1, evaluation_id: evaluationId,
			});
			try {
				for (const [index, candidate] of prepared.entries()) {
					const record = audit.candidates[index]!;
					const candidateRoot = join(outputRoot, "candidates", candidate.id);
					await mkdir(join(options.workspace, candidateRoot), { recursive: true });
					await writeFile(join(options.workspace, record.source_path), candidate.source, { encoding: "utf8", mode: 0o600 });
					const candidateId = `shape-${digest}-${candidate.id}`;
					const executed = await options.worker.request<WorkerRepairCandidateResponse>(options.workspace, {
						protocol_version: PROTOCOL_VERSION, operation: "execute_repair_candidate", candidate_id: candidateId,
						source_path: record.source_path, formal_source_sha256: formal.sourceSha256, formal_step_sha256: formal.stepSha256,
						timeout_s: 120, output_dir: join(candidateRoot, "execution"),
					}, { timeoutMs: 135_000, signal: options.controller.signal });
					record.execution = executed;
					if (executed.candidate_id !== candidateId || executed.source_sha256 !== candidate.sourceSha256 || executed.formal_source_sha256 !== formal.sourceSha256 || executed.formal_step_sha256 !== formal.stepSha256 || executed.formal_unchanged !== true) {
						throw new InfrastructureError(`Shape-feedback candidate ${candidate.id} violated the isolated execution contract`);
					}
					if (!executed.execution.success || executed.execution.timed_out || !artifactAccepted(executed.artifact) || !executed.artifact.path || !executed.artifact.sha256) {
						record.status = "invalid";
						record.error = executed.execution.timed_out ? "Candidate execution timed out" : executed.execution.stderr.trim().split(/\r?\n/u).at(-1) || executed.artifact.validation_error || `Candidate status=${executed.artifact.status}`;
						audit.errors.push(`${candidate.id}: ${record.error}`);
					} else {
						if (await sha256File(join(options.workspace, executed.artifact.path)) !== executed.artifact.sha256) throw new InfrastructureError(`Candidate ${candidate.id} STEP hash mismatch`);
						record.status = "valid";
					}
					await assertFormalUnchanged(options.workspace, formal);
					await persist();
				}

				const validCandidates = audit.candidates.filter(({ status }) => status === "valid");
				if (validCandidates.length) {
					let topology: ShapeTopologyResponse;
					try {
						topology = await options.worker.request<ShapeTopologyResponse>(options.workspace, {
							protocol_version: PROTOCOL_VERSION, operation: "compare_candidate_topology", current_step_path: "output.step",
							candidates: validCandidates.map((candidate) => ({ candidate_id: candidate.id, step_path: candidate.execution!.artifact.path! })),
							timeout_s: 150, output_dir: join(outputRoot, "topology"),
						}, { timeoutMs: 180_000, signal: options.controller.signal });
					} catch (error) {
						if (options.controller.signal.aborted) throw error;
						// Topology is optional evidence. A failed worker process must not
						// suppress independent 2D feedback; returned evidence is checked below.
						topology = {
							protocol_version: PROTOCOL_VERSION, operation: "compare_candidate_topology", status: "error",
							error: { kind: "drawing", message: `Topology worker request failed: ${error instanceof Error ? error.message : String(error)}` },
						};
					}
					audit.topology = topology;
					if (topology.protocol_version !== PROTOCOL_VERSION || topology.operation !== "compare_candidate_topology") throw new InfrastructureError("Topology response violated its protocol contract");
					if (topology.status === "error") {
						recordTopologyDiagnostic(`STEP topology unavailable: ${topology.error.message}`);
					} else {
						if (topology.current_step_path !== "output.step" || topology.current_step_sha256 !== formal.stepSha256 || !Array.isArray(topology.candidates) || topology.candidates.length !== validCandidates.length || new Set(topology.candidates.map(({ candidate_id }) => candidate_id)).size !== validCandidates.length) {
							throw new InfrastructureError("Topology returned mismatched Current STEP evidence or candidate identities/count");
						}
						assertBettiCounts(topology.current_topology);
						if (!topology.current_topology) recordTopologyDiagnostic(`Current topology unavailable: ${topology.current_error || "computation failed"}`);
						for (const candidate of topology.candidates) {
							const expected = validCandidates.find(({ id }) => id === candidate.candidate_id)?.execution!.artifact;
							if (!expected || candidate.step_path !== expected.path || candidate.step_sha256 !== expected.sha256 || await sha256File(join(options.workspace, candidate.step_path)) !== expected.sha256) throw new InfrastructureError("Topology returned mismatched candidate STEP evidence");
							assertBettiCounts(candidate.topology);
							if (candidate.status !== "ok" || !candidate.topology) {
								if (candidate.topology_delta !== null) throw new InfrastructureError("Unavailable topology must not report a delta");
								recordTopologyDiagnostic(`${candidate.candidate_id}: topology unavailable: ${candidate.error || "computation failed"}`);
							} else if (topology.current_topology) {
								if (!candidate.topology_delta || bettiKeys.some((key) => candidate.topology_delta![key] !== candidate.topology![key] - topology.current_topology![key])) throw new InfrastructureError("Topology delta disagrees with Current/candidate counts");
							} else if (candidate.topology_delta !== null) {
								throw new InfrastructureError("Unavailable Current topology must not report a delta");
							}
						}
					}
					await assertFormalUnchanged(options.workspace, formal);
					await persist();
					for (const view of views) {
						const frame = await currentFrame({ ...options, manifest: options.lockedManifest, formal, regionId: view.region_id, outputRoot: join(outputRoot, "views", view.region_id) });
						const [reference, current] = await Promise.all([
							inspectPngFile(options.workspace, frame.referencePath), inspectPngFile(options.workspace, frame.currentRenderPath),
						]);
						audit.frames.push({ ...frame, ...view, reference_sha256: reference.sha256, current_render_sha256: current.sha256 });
						await persist();
					}
				}
				for (const frame of audit.frames) {
					const renderedCandidates: Array<{ candidate_id: string; render_path: string; render_sha256: string; step_path: string; step_sha256: string }> = [];
					for (const candidate of validCandidates) {
						const artifact = candidate.execution!.artifact;
						const pairRoot = join(outputRoot, "views", frame.region_id, "candidates", candidate.id);
						const rendered = await options.worker.request<WorkerFixedProjectionResponse | WorkerDrawingErrorResponse>(options.workspace, {
							protocol_version: PROTOCOL_VERSION, operation: "render_fixed_projection", region_id: frame.region_id,
							crop_path: frame.referencePath, step_path: artifact.path, projection_anchor_step_path: frame.projectionAnchorStepPath,
							view_spec: frame.viewSpec, transform: frame.transform, output_dir: join(pairRoot, "projection"),
						}, { timeoutMs: 180_000, signal: options.controller.signal });
						if (rendered.status === "error") {
							audit.errors.push(`${candidate.id}/${frame.region_id}: ${rendered.error.message}`);
							await persist();
							continue;
						}
						requireDrawingSuccess(rendered, "render_fixed_projection");
						if (rendered.region_id !== frame.region_id || rendered.step_sha256 !== artifact.sha256 || rendered.projection_anchor_step_sha256 !== frame.projectionAnchorStepSha256 || !rendered.projection_origin || rendered.projection_origin.length !== 3 || rendered.projection_origin.some((value, index) => value !== frame.projectionOrigin[index]) || jsonSha(rendered.section_origin ?? null) !== jsonSha(frame.sectionOrigin ?? null) || jsonSha(rendered.transform) !== frame.transformSha256 || cameraSha256(rendered.view_spec) !== frame.cameraSha256) {
							throw new InfrastructureError(`Candidate ${candidate.id} drifted from frozen view ${frame.region_id}`);
						}
						const renderedImage = await inspectPngFile(options.workspace, rendered.render_path);
						if (renderedImage.sha256 !== rendered.sha256) throw new InfrastructureError("Candidate projection hash mismatch");
						audit.projections.push({ region_id: frame.region_id, candidate_id: candidate.id, step_sha256: artifact.sha256!, render_path: rendered.render_path, render_sha256: rendered.sha256, camera_sha256: frame.cameraSha256, transform_sha256: frame.transformSha256, projection_origin: rendered.projection_origin, projection_anchor_step_sha256: frame.projectionAnchorStepSha256 });
						renderedCandidates.push({ candidate_id: candidate.id, render_path: rendered.render_path, render_sha256: rendered.sha256, step_path: artifact.path!, step_sha256: artifact.sha256! });
						await persist();
					}
					if (!renderedCandidates.length) continue;
					const response = await options.worker.request<WorkerShapeFeedbackResponse | WorkerDrawingErrorResponse>(options.workspace, {
						protocol_version: PROTOCOL_VERSION, operation: "compose_shape_feedback", region_id: frame.region_id,
						reference_path: frame.referencePath, current_render_path: frame.currentRenderPath,
						// Retain worker protocol compatibility without requiring or inventing a label.
						candidates: renderedCandidates.map(({ candidate_id, render_path }) => ({ candidate_id, render_path, hypothesis: "" })),
						target_roi: frame.target_roi, output_dir: join(outputRoot, "views", frame.region_id, "feedback"),
					}, { timeoutMs: 180_000, signal: options.controller.signal });
					if (response.status === "error") {
						audit.errors.push(`${frame.region_id}: ${response.error.message}`);
						await persist();
						continue;
					}
					if (response.protocol_version !== PROTOCOL_VERSION || response.operation !== "compose_shape_feedback" || response.region_id !== frame.region_id || response.reference_sha256 !== frame.reference_sha256 || response.current_render_sha256 !== frame.current_render_sha256 || jsonSha(response.target_roi) !== jsonSha(frame.target_roi)
						|| !Array.isArray(response.candidates) || response.candidates.length !== renderedCandidates.length
						|| response.candidates.some((candidate, index) => !candidate || candidate.candidate_id !== renderedCandidates[index]!.candidate_id || candidate.render_sha256 !== renderedCandidates[index]!.render_sha256)) {
						throw new InfrastructureError("Shape feedback returned mismatched input evidence");
					}
					if (response.card_layout !== "three_panel_row" || !Array.isArray(response.cards) || response.cards.length !== renderedCandidates.length || response.card_count !== response.cards.length
						|| response.cards.some((card, index) => !card || card.candidate_id !== renderedCandidates[index]!.candidate_id || card.card_layout !== "three_panel_row" || card.panel_count !== 3
							|| jsonSha(card.panel_titles) !== jsonSha(["Target", "Current", "Preview"]))) {
						throw new InfrastructureError("Shape feedback returned mismatched candidate cards");
					}
					for (const item of response.cards) {
						const card = await inspectPngFile(options.workspace, item.card_path);
						if (card.sha256 !== item.sha256 || card.width !== item.width || card.height !== item.height) throw new InfrastructureError("Shape-feedback card hash or dimensions mismatch");
					}
					const firstCard = response.cards[0]!;
					if (response.card_path !== firstCard.card_path || response.sha256 !== firstCard.sha256 || response.width !== firstCard.width || response.height !== firstCard.height) throw new InfrastructureError("Shape-feedback first-card alias mismatch");
					audit.feedback.push(response);
					await persist();
				}
				await assertFormalUnchanged(options.workspace, formal);
				audit.status = audit.errors.length ? "partial" : "completed";
				await persist();
			} catch (error) {
				audit.status = "error";
				audit.errors.push(error instanceof Error ? error.message : String(error));
				await persist();
				throw error;
			} finally {
				await assertFormalUnchanged(options.workspace, formal);
			}
			const previewErrors = audit.errors.filter((error) => !topologyDiagnostics.has(error));
			const content: Array<TextContent | ImageContent> = [{ type: "text", text: [
				audit.feedback.length
					? `Preview ${previewErrors.length ? "incomplete" : "generated"}. Views returned: ${audit.feedback.length}/${views.length}.`
					: "Preview unavailable. No view images returned.",
				"Workspace: solution.py and output.step unchanged.",
				...(audit.feedback.length ? ["To adopt this candidate, call adopt_shape_candidate({ candidate_id: \"<id>\" })."] : []),
				...(audit.feedback.length ? [
					"",
					"One three-panel card per view: Target | Current | Preview.",
					panelLegend,
					"All panels show the complete aligned input frame, fitted with one shared display scale. Current and Preview use the same camera and 2D transform; no CAD is scaled independently.",
				] : []),
				...(previewErrors.length ? ["", "Errors:", ...previewErrors.map((error) => `- ${error}`)] : []),
			].join("\n") }];
			for (const feedback of audit.feedback) {
				const warnings = Array.isArray(feedback.warnings) ? feedback.warnings
					.filter((warning) => typeof warning === "string" && warning.trim())
					.map((warning) => warning.replace(/\bSelected ROI\b/giu, "Target view").replace(/\bROI\b/giu, "view")) : [];
				for (const card of feedback.cards) {
					content.push({ type: "text", text: [
						`View: ${feedback.region_id} | Candidate: ${card.candidate_id}`,
						"Left to right: Target | Current | Preview.",
						`Image: ${card.card_path}`,
						...(warnings.length ? ["Warnings:", ...warnings.map((warning) => `- ${warning}`)] : []),
					].join("\n") });
					content.push(await imageFromWorkspace(options.workspace, card.card_path));
				}
			}
			return { content, details: { audit_path: auditPath, ...audit } };
		},
	});
}
