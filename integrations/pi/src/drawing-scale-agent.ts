import { createHash } from "node:crypto";
import { mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";

import type { ImageContent } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import {
	imageFromWorkspace,
	inspectPngFile,
	requireDrawingSuccess,
} from "./drawing-reflection.js";
import { InfrastructureError } from "./errors.js";
import { isoNow } from "./files.js";
import type { WorkerClient } from "./local-worker.js";
import {
	PROTOCOL_VERSION,
	type ArtifactPayload,
	type DrawingSimilarityTransform,
	type DrawingViewManifest,
	type WorkerFixedProjectionResponse,
	type WorkerRankDrawingViewsResponse,
	type WorkerReflectionInspectionResponse,
} from "./types.js";
import type { RunController } from "./runner.js";

export const DRAWING_SCALE_AGENT_TOOLS = [
	"lock_target_dimension",
	"inspect_current_cad_dimensions",
	"preview_dimension_match",
	"submit_dimension_match",
] as const;

const TARGET_NAMES = ["overall_width", "overall_height", "overall_depth"] as const;
const CAD_AXES = ["x", "y", "z"] as const;
const TARGET_NAME_SCHEMA = Type.Union([
	Type.Literal("overall_width"),
	Type.Literal("overall_height"),
	Type.Literal("overall_depth"),
]);
const CAD_AXIS_SCHEMA = Type.Union([
	Type.Literal("x"),
	Type.Literal("y"),
	Type.Literal("z"),
]);

type TargetName = (typeof TARGET_NAMES)[number];
type CadAxis = (typeof CAD_AXES)[number];
type MatchQuality = "exact" | "proxy";

export interface DrawingScaleTarget {
	name: TargetName;
	value_mm: number;
	description: string;
	source_page: number;
	orientation: "horizontal" | "vertical";
	extent_scope: "complete_object" | "dominant_substructure";
	endpoint_a: string;
	endpoint_b: string;
	dimension_chain_mm: number[];
	derivation: string;
}

export interface DrawingScaleCurrentMeasurement {
	measurement: `bbox_${CadAxis}_extent`;
	value_mm: number;
	description: string;
	evidence: "output.step B-rep bounding box";
}

export interface DrawingScalePreview {
	preview_ref: string;
	target: DrawingScaleTarget;
	current: DrawingScaleCurrentMeasurement;
	match_quality: MatchQuality;
	correspondence_basis: string;
	limitations?: string;
	preview_paths: string[];
	created_at: string;
}

export type DrawingScaleAgentResult = {
	version: 2;
	status: "paired";
	justification: string;
	source_sha256: string;
	step_sha256: string;
} & DrawingScalePreview;

interface InternalPreview extends DrawingScalePreview {
	display_scale_multiplier: number;
}

export interface DrawingScaleAgentState {
	source: string;
	sourceSha256: string;
	stepSha256: string;
	artifact: ArtifactPayload;
	lockedTarget: DrawingScaleTarget | null;
	previews: InternalPreview[];
	submitted: DrawingScaleAgentResult | null;
}

export const DRAWING_SCALE_AGENT_SYSTEM_PROMPT = `You are the standalone Drawing Scale Correspondence Agent for CADGenBench.

Your only job is to pair one exact boundary dimension from the original engineering drawing with one current formal-CAD measurement and generate uniformly corrected comparison previews. You must always return one paired dimension; abstention is not allowed. You do not edit CAD, modify solution.py, produce STEP files, diagnose general geometry, or act as a subagent for another model.

The workflow has a strict information barrier. Initially you can see only the attached original drawing. First call lock_target_dimension. Until that call succeeds, you cannot inspect solution.py, STEP measurements, or any Current CAD information. Read the target only from explicit annotations and dimension chains in the original drawing; never infer a dimension from pixel proportions, pixel distances, apparent size, or later Current CAD evidence.

Prefer a complete-object overall width. Follow its extension lines all the way to the two measured boundary endpoints. A number is not an overall size merely because it is large or appears near an outer dimension line. Distinguish feature size, feature location from a datum, and complete boundary extent. When a complete extent is split across collinear dimension chains that meet at a shared datum, provide every segment and derive the total explicitly. For example, two opposite-side segments that end and begin at the same datum form the full extent only if their endpoint semantics prove this. If no complete-object overall width is supported, lock the clearest complete height or depth; use a dominant substructure only as the final fallback and say so. Once locked, the target cannot be changed after Current CAD is revealed.

After locking the target, call inspect_current_cad_dimensions. It reveals solution.py and the existing exact B-rep bounding-box measurements. Map the immutable target to the Current CAD using that evidence. Prefer a measurement with the same physical state, extent, and endpoints, and label it exact. Choose a CAD axis from the drawing orientation, source, and geometry, never merely because its numeric value is closest to the target. If no exact correspondence exists, choose the best semantically related proxy anchor and label it proxy. A developed or flat-pattern extent paired with a formed-part extent, or a partial feature paired with a whole-part extent, must be proxy and must explicitly state that mismatch. A proxy is useful only for establishing a repeatable visual scale; it is not evidence that the two dimensions describe identical geometry.

You must call preview_dimension_match for the selected pair. That tool applies one hidden dimension correction consistently to every locked comparison view and returns all previews. Inspect every returned preview. The preview is a visual sanity check, not proof of correspondence. If a view exposes a folded/unfolded mismatch, incompatible endpoints, or another semantic contradiction, keep the pair but label it proxy and record the limitation. If the first candidate is unusable or clipped, preview a better pair.

After inspecting every returned view, submit exactly one paired result. Never submit prose without the tool result and never refuse to pair. Do not output, estimate, discuss, or add a scale_ratio field; the renderer owns that internal calculation.`;

export function drawingScaleAgentPrompt(options: {
	fixtureId: string;
	pagePaths: string[];
}): string {
	return [
		`Evaluate physical drawing scale correspondence for CADGenBench fixture ${options.fixtureId}. Begin with target-drawing measurement only.`,
		`Attached original drawing pages, in order: ${options.pagePaths.join(", ")}.`,
		"No Current CAD source, dimensions, or render is available yet. Inspect the original drawing, trace the extension lines and any shared-datum dimension chain, then call lock_target_dimension. Prefer the complete developed-pattern overall width. After the target is locked, inspect the Current CAD, preview the pair in every locked view, and submit it. Abstention and prose-only answers are not results.",
	].join("\n\n");
}

const EMPTY_PARAMETERS_SCHEMA = Type.Object({}, { additionalProperties: false });

export const LOCK_TARGET_DIMENSION_PARAMETERS_SCHEMA = Type.Object({
	target_name: TARGET_NAME_SCHEMA,
	target_mm: Type.Number({ minimum: 0.001, maximum: 1_000_000 }),
	target_description: Type.String({ minLength: 30, maxLength: 1_500 }),
	source_page: Type.Integer({ minimum: 0, maximum: 999 }),
	orientation: Type.Union([Type.Literal("horizontal"), Type.Literal("vertical")]),
	extent_scope: Type.Union([Type.Literal("complete_object"), Type.Literal("dominant_substructure")]),
	endpoint_a: Type.String({ minLength: 20, maxLength: 1_000 }),
	endpoint_b: Type.String({ minLength: 20, maxLength: 1_000 }),
	dimension_chain_mm: Type.Array(
		Type.Number({ minimum: 0.001, maximum: 1_000_000 }),
		{ minItems: 1, maxItems: 12 },
	),
	derivation: Type.String({
		minLength: 30,
		maxLength: 2_000,
		description: "Trace the dimension extension lines and explain how the submitted chain reaches the two boundary endpoints.",
	}),
}, { additionalProperties: false });

export const PREVIEW_DIMENSION_MATCH_PARAMETERS_SCHEMA = Type.Object({
	current_axis: CAD_AXIS_SCHEMA,
	current_description: Type.String({ minLength: 20, maxLength: 1_000 }),
	match_quality: Type.Union([Type.Literal("exact"), Type.Literal("proxy")]),
	correspondence_basis: Type.String({
		minLength: 30,
		maxLength: 2_000,
		description: "Why this is the best available dimensional anchor, including whether physical state and endpoints are identical.",
	}),
	limitations: Type.Optional(Type.String({
		minLength: 20,
		maxLength: 2_000,
		description: "Required for a proxy: the exact state, endpoint, or extent mismatch that prevents exact correspondence.",
	})),
}, { additionalProperties: false });

export const SUBMIT_DIMENSION_MATCH_PARAMETERS_SCHEMA = Type.Object({
	preview_ref: Type.String({ pattern: "^P[0-9]{3,}$" }),
	justification: Type.String({ minLength: 30, maxLength: 2_000 }),
}, { additionalProperties: false });

function bboxMeasurement(state: DrawingScaleAgentState, axis: CadAxis): number {
	const value = state.artifact.validation?.bbox[axis];
	if (!Number.isFinite(value) || !value || value <= 0) {
		throw new InfrastructureError(`Current formal STEP has no positive bbox.${axis} measurement`);
	}
	return value;
}

function sha256Json(value: unknown): string {
	return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

async function renderUniformPreview(options: {
	workspace: string;
	worker: WorkerClient;
	controller: RunController;
	manifest: DrawingViewManifest;
	state: DrawingScaleAgentState;
	previewRef: string;
	displayScaleMultiplier: number;
}): Promise<string[]> {
	const paths: string[] = [];
	for (const region of options.manifest.regions) {
		if (!region.selected) continue;
		// The CAD worker intentionally confines generated evidence beneath
		// drawing_feedback/. Keep this standalone agent's previews in that
		// protocol-owned tree even though its run workspace is otherwise isolated.
		const outputDir = join("drawing_feedback", "drawing_scale_agent", options.previewRef, region.region_id);
		const candidateId = `scale-${sha256Json([options.previewRef, region.region_id]).slice(0, 16)}`;
		const viewSpec = { ...region.selected.view_spec, name: candidateId };
		const ranked = await options.worker.request<WorkerRankDrawingViewsResponse>(
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
				output_dir: join(outputDir, "free-fit"),
			},
			{ timeoutMs: 240_000, signal: options.controller.signal },
		);
		requireDrawingSuccess(ranked, "rank_drawing_views");
		const free = ranked.rankings[0];
		if (
			ranked.region_id !== region.region_id ||
			ranked.step_sha256 !== options.state.stepSha256 ||
			ranked.rankings.length !== 1 ||
			!free ||
			free.candidate_id !== candidateId ||
			free.transform.reflected
		) throw new InfrastructureError(`Scale preview free fit failed for ${region.region_id}`);
		const transform: DrawingSimilarityTransform = {
			...free.transform,
			scale_px_per_unit: free.transform.scale_px_per_unit * options.displayScaleMultiplier,
		};
		const fixed = await options.worker.request<WorkerFixedProjectionResponse>(
			options.workspace,
			{
				protocol_version: PROTOCOL_VERSION,
				operation: "render_fixed_projection",
				region_id: region.region_id,
				crop_path: region.crop_path,
				step_path: "output.step",
				view_spec: viewSpec,
				transform,
				output_dir: join(outputDir, "uniform-scale"),
			},
			{ timeoutMs: 180_000, signal: options.controller.signal },
		);
		requireDrawingSuccess(fixed, "render_fixed_projection");
		if (fixed.step_sha256 !== options.state.stepSha256) {
			throw new InfrastructureError(`Scale preview STEP changed for ${region.region_id}`);
		}
		const comparison = await options.worker.request<WorkerReflectionInspectionResponse>(
			options.workspace,
			{
				protocol_version: PROTOCOL_VERSION,
				operation: "compose_reflection_inspection",
				region_id: region.region_id,
				crop_path: region.crop_path,
				aligned_render_path: fixed.render_path,
				mode: "coordinate_overview",
				output_dir: join(outputDir, "comparison"),
			},
			{ timeoutMs: 180_000, signal: options.controller.signal },
		);
		requireDrawingSuccess(comparison, "compose_reflection_inspection");
		if (comparison.mode !== "coordinate_overview" || comparison.region_id !== region.region_id) {
			throw new InfrastructureError(`Scale preview comparison failed for ${region.region_id}`);
		}
		const image = await inspectPngFile(options.workspace, comparison.context_path);
		if (image.sha256 !== comparison.sha256.context) {
			throw new InfrastructureError(`Scale preview image hash mismatch for ${region.region_id}`);
		}
		paths.push(comparison.context_path);
	}
	if (paths.length === 0) throw new Error("The locked manifest contains no renderable comparison views");
	return paths;
}

export function createDrawingScaleAgentTools(options: {
	workspace: string;
	worker: WorkerClient;
	controller: RunController;
	manifest: DrawingViewManifest;
	pageCount: number;
	state: DrawingScaleAgentState;
}) {
	const lockTargetTool = defineTool({
		name: "lock_target_dimension",
		label: "Lock target drawing dimension",
		description: "Lock one target boundary dimension using only the original drawing, its extension-line endpoints, and an explicit dimension-chain derivation. This mandatory first step permanently fixes the target before any Current CAD evidence is revealed.",
		parameters: LOCK_TARGET_DIMENSION_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			if (options.state.lockedTarget) throw new Error("The target drawing dimension is already locked");
			if (params.source_page >= options.pageCount) {
				throw new Error(`Unknown source page ${String(params.source_page)}`);
			}
			const chainTotal = params.dimension_chain_mm.reduce((sum, value) => sum + value, 0);
			const tolerance = Math.max(0.01, params.target_mm * 1e-6);
			if (Math.abs(chainTotal - params.target_mm) > tolerance) {
				throw new Error(`Dimension-chain total ${chainTotal.toFixed(6)} mm does not equal target ${params.target_mm.toFixed(6)} mm`);
			}
			options.state.lockedTarget = {
				name: params.target_name,
				value_mm: params.target_mm,
				description: params.target_description,
				source_page: params.source_page,
				orientation: params.orientation,
				extent_scope: params.extent_scope,
				endpoint_a: params.endpoint_a,
				endpoint_b: params.endpoint_b,
				dimension_chain_mm: [...params.dimension_chain_mm],
				derivation: params.derivation,
			};
			return {
				content: [{
					type: "text" as const,
					text: `Target dimension locked permanently from the original drawing:\n${JSON.stringify(options.state.lockedTarget, null, 2)}\nNow call inspect_current_cad_dimensions to reveal Current CAD evidence.`,
				}],
				details: { target: options.state.lockedTarget },
			};
		},
	});

	const inspectTool = defineTool({
		name: "inspect_current_cad_dimensions",
		label: "Inspect exact current CAD dimensions",
		description: "After the target drawing dimension is locked, reveal solution.py and return the unchanged exact whole-model X, Y, and Z B-rep bounding-box extents from the current formal output.step. Takes no arguments.",
		parameters: EMPTY_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute() {
			if (!options.state.lockedTarget) {
				throw new Error("Lock the target drawing dimension before inspecting any Current CAD evidence");
			}
			const dimensions = {
				bbox_x_extent_mm: bboxMeasurement(options.state, "x"),
				bbox_y_extent_mm: bboxMeasurement(options.state, "y"),
				bbox_z_extent_mm: bboxMeasurement(options.state, "z"),
			};
			return {
				content: [{
					type: "text" as const,
					text: `The target is already immutable. Exact current output.step B-rep measurements:\n${JSON.stringify(dimensions, null, 2)}\nUse the following solution.py and endpoint semantics—not numeric closeness—to decide whether one axis corresponds to the locked target:\n\n${options.state.source}`,
				}],
				details: {
					...dimensions,
					source_sha256: options.state.sourceSha256,
					step_sha256: options.state.stepSha256,
				},
			};
		},
	});

	const previewTool = defineTool({
		name: "preview_dimension_match",
		label: "Preview one dimension match",
		description: "Pair one explicitly annotated target dimension with one current B-rep bounding-box axis as an exact or proxy anchor, then apply the hidden display correction consistently to every locked view. This call is mandatory and does not expose a ratio.",
		parameters: PREVIEW_DIMENSION_MATCH_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			const lockedTarget = options.state.lockedTarget;
			if (!lockedTarget) throw new Error("Lock the target drawing dimension before previewing a match");
			if (params.match_quality === "proxy" && !params.limitations) {
				throw new Error("A proxy dimension pair must state its endpoint, state, or extent limitation");
			}
			const currentValue = bboxMeasurement(options.state, params.current_axis);
			const displayScaleMultiplier = currentValue / lockedTarget.value_mm;
			if (!Number.isFinite(displayScaleMultiplier) || displayScaleMultiplier < 0.05 || displayScaleMultiplier > 20) {
				throw new Error("The proposed target/current dimensions produce an implausible display correction");
			}
			const previewRef = `P${String(options.state.previews.length + 1).padStart(3, "0")}`;
			const previewPaths = await renderUniformPreview({
				...options,
				previewRef,
				displayScaleMultiplier,
			});
			const publicPreview: DrawingScalePreview = {
				preview_ref: previewRef,
				target: lockedTarget,
				current: {
					measurement: `bbox_${params.current_axis}_extent`,
					value_mm: currentValue,
					description: params.current_description,
					evidence: "output.step B-rep bounding box",
				},
				match_quality: params.match_quality,
				correspondence_basis: params.correspondence_basis,
				...(params.limitations ? { limitations: params.limitations } : {}),
				preview_paths: previewPaths,
				created_at: isoNow(),
			};
			options.state.previews.push({ ...publicPreview, display_scale_multiplier: displayScaleMultiplier });
			const content: Array<{ type: "text"; text: string } | ImageContent> = [{
				type: "text",
				text: `${previewRef} records a ${params.match_quality} pair between locked target ${lockedTarget.name}=${lockedTarget.value_mm.toFixed(3)} mm and current bbox_${params.current_axis}_extent=${currentValue.toFixed(3)} mm. The same hidden display correction was applied to all ${String(previewPaths.length)} locked views. Inspect every image, then submit this pair or preview a better Current CAD axis. The target cannot change and abstention is not available.`,
			}];
			for (const path of previewPaths) content.push(await imageFromWorkspace(options.workspace, path));
			return { content, details: publicPreview };
		},
	});

	const submitTool = defineTool({
		name: "submit_dimension_match",
		label: "Submit scale correspondence",
		description: "Submit exactly one final exact or proxy dimension pair after inspecting all of its previews. Abstention is not available.",
		parameters: SUBMIT_DIMENSION_MATCH_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			if (options.state.submitted) throw new Error("A final drawing-scale result has already been submitted");
			const preview = options.state.previews.find(({ preview_ref }) => preview_ref === params.preview_ref);
			if (!preview) throw new Error(`Unknown preview_ref ${params.preview_ref}`);
			const { display_scale_multiplier: _internal, ...publicPreview } = preview;
			options.state.submitted = {
				version: 2,
				status: "paired",
				...publicPreview,
				justification: params.justification,
				source_sha256: options.state.sourceSha256,
				step_sha256: options.state.stepSha256,
			};
			return {
				content: [{ type: "text" as const, text: `Recorded ${preview.match_quality} dimension pair ${preview.preview_ref}.` }],
				details: { submitted: true as const },
			};
		},
	});

	const tools = [lockTargetTool, inspectTool, previewTool, submitTool] as const;
	if (tools.some((tool, index) => tool.name !== DRAWING_SCALE_AGENT_TOOLS[index])) {
		throw new InfrastructureError("Drawing Scale Agent tool order does not match its declared contract");
	}
	return tools;
}

export async function drawingScaleResultContent(
	workspace: string,
	result: DrawingScaleAgentResult,
): Promise<Array<{ type: "text"; text: string } | ImageContent>> {
	const content: Array<{ type: "text"; text: string } | ImageContent> = [{
		type: "text",
		text: `Drawing Scale Agent paired ${result.target.name}=${result.target.value_mm.toFixed(3)} mm against current ${result.current.measurement}=${result.current.value_mm.toFixed(3)} mm as ${result.match_quality}. ${result.justification}`,
	}];
	for (const path of result.preview_paths) content.push(await imageFromWorkspace(workspace, path));
	return content;
}

export async function hashFile(path: string): Promise<string> {
	return createHash("sha256").update(await readFile(path)).digest("hex");
}

export async function ensureScaleAgentOutputDirectory(path: string): Promise<void> {
	await mkdir(path, { recursive: true });
}
