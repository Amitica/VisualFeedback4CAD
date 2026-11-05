import { createHash } from "node:crypto";
import { copyFile, lstat, mkdir, readFile, realpath, rename } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { inflateSync } from "node:zlib";

import type { ImageContent } from "@earendil-works/pi-ai";
import {
	type AgentSession,
	type ToolDefinition,
	createAgentSession,
	DefaultResourceLoader,
	defineTool,
	getAgentDir,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import { addUsageTotals, emptyUsageTotals, sessionUsageTotals } from "./context.js";
import { InfrastructureError } from "./errors.js";
import { isoNow, readJson, writeJsonAtomic } from "./files.js";
import type { WorkerClient } from "./local-worker.js";
import { PausableDeadline } from "./pausable-deadline.js";
import {
	drawingSegmentationPrompt,
	REFLECTION_SYSTEM_PROMPT,
	reflectionPrompt,
	SEGMENTATION_SYSTEM_PROMPT,
	VIEW_SEARCH_SYSTEM_PROMPT,
	viewSearchPrompt,
} from "./prompts.js";
import { inspectFormalCandidate } from "./solution.js";
import {
	DRAWING_FEEDBACK_VERSION,
	PROTOCOL_VERSION,
	REFLECTION_AGENT_TOOLS,
	SEGMENTATION_AGENT_TOOLS,
	SOLUTION_FILE,
	VIEW_SEARCH_AGENT_TOOLS,
	type AgentJob,
	type AgentResult,
	type DrawingFeedbackSummary,
	type DrawingManifestRegion,
	type DrawingRegionManifest,
	type DrawingRegionManifestRegion,
	type DrawingReflectionReport,
	type DrawingRegionStatus,
	type DrawingViewCandidate,
	type DrawingViewManifest,
	type DrawingViewSpec,
	type NormalizedRoi,
	type SubagentRunUsage,
	type UsageTotals,
	type Vector3,
	type WorkerCompareDrawingSheetResponse,
	type WorkerComparisonSheetRegion,
	type WorkerCropDrawingRegionResponse,
	type WorkerInspectResponse,
	type WorkerPrepareResponse,
	type WorkerRankDrawingViewsResponse,
	type WorkerReflectionInspectionResponse,
	type WorkerRenderDrawingViewResponse,
} from "./types.js";
import type { ResolvedModel, RunController } from "./runner.js";

const REGION_ID_SCHEMA = Type.String({
	minLength: 1,
	maxLength: 128,
	pattern: "^[A-Za-z0-9][A-Za-z0-9._-]*$",
});
const FINDING_ID_SCHEMA = Type.String({
	minLength: 1,
	maxLength: 128,
	pattern: "^finding-[0-9]{3,}$",
	description: "Controller-generated finding_id returned by submit_initial_findings or update_findings.",
});
const GRID_CELL_SCHEMA = Type.Union([
	Type.Literal("A1"), Type.Literal("B1"), Type.Literal("C1"), Type.Literal("D1"),
	Type.Literal("A2"), Type.Literal("B2"), Type.Literal("C2"), Type.Literal("D2"),
	Type.Literal("A3"), Type.Literal("B3"), Type.Literal("C3"), Type.Literal("D3"),
	Type.Literal("A4"), Type.Literal("B4"), Type.Literal("C4"), Type.Literal("D4"),
], { description: "Exactly one fixed cell in the 4x4 grid shown by inspect_region." });
const NORMALIZED_COORDINATE_SCHEMA = Type.Integer({ minimum: 0, maximum: 1000 });
export const NORMALIZED_ROI_SCHEMA = Type.Unsafe<NormalizedRoi>(
	Type.Array(NORMALIZED_COORDINATE_SCHEMA, { minItems: 4, maxItems: 4 }),
);
const VECTOR_COMPONENT_SCHEMA = Type.Number({ description: "One finite world-space vector component." });
export const VECTOR3_SCHEMA = Type.Unsafe<Vector3>(
	Type.Array(VECTOR_COMPONENT_SCHEMA, {
		minItems: 3,
		maxItems: 3,
		description: "Exactly three finite numbers in [x, y, z] order.",
	}),
);
export const DRAWING_VIEW_SPEC_SCHEMA = Type.Object(
	{
		direction: Type.Unsafe<Vector3>(
			Type.Array(VECTOR_COMPONENT_SCHEMA, {
				minItems: 3,
				maxItems: 3,
				description:
					"Required camera direction [x,y,z], pointing from the CAD model toward the observer. Must be finite, non-zero, and perpendicular to up; magnitude is irrelevant.",
			}),
		),
		up: Type.Unsafe<Vector3>(
			Type.Array(VECTOR_COMPONENT_SCHEMA, {
				minItems: 3,
				maxItems: 3,
				description:
					"Required screen-up direction [x,y,z] in CAD world coordinates. Must be finite, non-zero, and perpendicular to direction; magnitude is irrelevant.",
			}),
		),
		section_offset_fraction: Type.Optional(
			Type.Number({
				exclusiveMinimum: -1,
				exclusiveMaximum: 1,
				description:
					"Optional straight-section plane position along direction through the model bounding box: 0 is the center and values approach -1 or 1 near the supporting faces. Supply together with section_keep; omit both for a normal exterior view.",
			}),
		),
		section_keep: Type.Optional(
			Type.Union([Type.Literal("front"), Type.Literal("back")], {
				description:
					"Which side of the straight section plane to retain. Required exactly when section_offset_fraction is supplied; omit both for a normal exterior view.",
			}),
		),
	},
	{
		additionalProperties: false,
		description:
			"One orthographic CAD camera. Do not provide a name or candidate_id; the controller assigns candidate IDs.",
	},
);
const DRAWING_VIEW_TYPE_SCHEMA = Type.Union([
	Type.Literal("orthographic"),
	Type.Literal("isometric"),
	Type.Literal("section"),
	Type.Literal("detail"),
	Type.Literal("unknown"),
], {
	description:
		"Semantic type of the target drawing region: ordinary projection, isometric, straight section, enlarged detail, or unknown.",
});
const CONFIDENCE_SCHEMA = Type.Union([
	Type.Literal("high"),
	Type.Literal("medium"),
	Type.Literal("low"),
], { description: "Your confidence in this submitted decision." });

export const CURRENT_CAD_REFERENCE_VIEW_SPEC: DrawingViewSpec = {
	name: "current-cad-isometric-reference",
	direction: [1, -1, 1],
	up: [1, 1, 0],
};

export const CROP_REGION_PARAMETERS_SCHEMA = Type.Object(
	{
		region_id: REGION_ID_SCHEMA,
		page: Type.Integer({ minimum: 0, maximum: 255 }),
		roi: NORMALIZED_ROI_SCHEMA,
	},
	{ additionalProperties: false },
);
const SUBMITTED_REGION_SCHEMA = Type.Object(
	{
		region_id: REGION_ID_SCHEMA,
		view_type_hint: DRAWING_VIEW_TYPE_SCHEMA,
		confidence: CONFIDENCE_SCHEMA,
		note: Type.String({ maxLength: 1000 }),
	},
	{ additionalProperties: false },
);
export const SUBMIT_REGION_MANIFEST_PARAMETERS_SCHEMA = Type.Object(
	{ regions: Type.Array(SUBMITTED_REGION_SCHEMA, { minItems: 1, maxItems: 12 }) },
	{ additionalProperties: false },
);
export const RENDER_CANDIDATE_PARAMETERS_SCHEMA = Type.Object(
	{ view_spec: DRAWING_VIEW_SPEC_SCHEMA },
	{
		additionalProperties: false,
		description:
			"Preview one new camera. Use rank_view_candidates afterwards with the returned candidate_id to score it.",
	},
);
export const RANK_CANDIDATES_PARAMETERS_SCHEMA = Type.Object(
	{
		candidate_ids: Type.Optional(
			Type.Array(
				Type.String({
					minLength: 1,
					maxLength: 128,
					pattern: "^[A-Za-z0-9][A-Za-z0-9._-]*$",
					description: "A candidate_id previously returned by render_candidate_view.",
				}),
				{
					minItems: 1,
					maxItems: 32,
					uniqueItems: true,
					description:
						"Score existing preview candidates. Supply this field OR candidates, never both.",
				},
			),
		),
		candidates: Type.Optional(
			Type.Array(DRAWING_VIEW_SPEC_SCHEMA, {
				minItems: 1,
				maxItems: 32,
				uniqueItems: true,
				description:
					"Render and score 1-32 new camera specs in one batch. Supply this field OR candidate_ids, never both.",
			}),
		),
	},
	{
		additionalProperties: false,
		description:
			"Choose exactly one mode: candidates for new cameras, or candidate_ids for previews already rendered. Returns the cumulative numeric ranking and labelled eligible Top-2 comparison images.",
		// This admits exactly one of candidate_ids/candidates while keeping the
		// top-level schema type="object" required by Moonshot's tool dialect.
		minProperties: 1,
		maxProperties: 1,
	},
);
export const SUBMIT_VIEW_RESULT_PARAMETERS_SCHEMA = Type.Object(
	{
		view_type: DRAWING_VIEW_TYPE_SCHEMA,
		status: Type.Union([
			Type.Literal("matched"),
			Type.Literal("ambiguous"),
		], {
			description:
				"Use matched when one candidate is semantically convincing; use ambiguous when uncertainty remains. Both require a selected candidate from the latest eligible Top-2.",
		}),
		selected_candidate_id: Type.String({
			minLength: 1,
			maxLength: 128,
			pattern: "^[A-Za-z0-9][A-Za-z0-9._-]*$",
			description:
				"Required candidate_id selected from the latest eligible Top-2 returned by rank_view_candidates. The other Top-2 candidate is stored automatically as the alternative.",
		}),
		confidence: CONFIDENCE_SCHEMA,
		note: Type.String({
			minLength: 1,
			maxLength: 1000,
			description:
				"Brief visual rationale for the choice, including the source-page projection or section evidence used to choose between Top-2 candidates.",
		}),
	},
	{
		additionalProperties: false,
		description:
			"Final submission for this one controller-bound region. Call only after rank_view_candidates has produced two eligible candidates.",
	},
);

export const INSPECT_REGION_PARAMETERS_SCHEMA = Type.Object(
	{ region_id: REGION_ID_SCHEMA },
	{ additionalProperties: false },
);
const INITIAL_FINDING_SCHEMA = Type.Object(
	{
		region_id: REGION_ID_SCHEMA,
		obs: Type.String({
			minLength: 1,
			maxLength: 2000,
			description:
				"Only the visible geometry difference: what the current CAD shows, what the input-drawing region shows instead, and where. Do not include a cause hypothesis or modification instruction.",
		}),
		suggested_change: Type.String({
			minLength: 1,
			maxLength: 2000,
			description:
				"The concrete CAD geometry change needed to remove this obs. It must correspond directly to obs and must not introduce an unobserved problem.",
		}),
	},
	{ additionalProperties: false },
);
export const SUBMIT_INITIAL_FINDINGS_PARAMETERS_SCHEMA = Type.Object(
	{
		findings: Type.Array(INITIAL_FINDING_SCHEMA, {
			maxItems: 288,
			description: "Complete global findings draft for every inspected comparable region; submit exactly once.",
		}),
	},
	{ additionalProperties: false },
);
export const COMPARE_REGION_GRID_CELLS_PARAMETERS_SCHEMA = Type.Object(
	{
		region_id: REGION_ID_SCHEMA,
		finding_ids: Type.Array(FINDING_ID_SCHEMA, {
			minItems: 1,
			maxItems: 24,
			uniqueItems: true,
			description: "Existing findings from this region that this fixed-cell comparison will verify.",
		}),
		reference_cell: GRID_CELL_SCHEMA,
		cad_cell: GRID_CELL_SCHEMA,
	},
	{
		additionalProperties: false,
		description:
			"Compare exactly one fixed A1-D4 cell from the original drawing with one fixed A1-D4 cell from the aligned CAD. Free ROI coordinates and paths are not accepted.",
	},
);
const UPDATE_FINDING_CHANGE_SCHEMA = Type.Union([
	Type.Object(
		{
			action: Type.Literal("add"),
			obs: Type.String({
				minLength: 1,
				maxLength: 2000,
				description:
					"Only the newly verified visible geometry difference: current CAD, input-drawing region, and location; no cause or modification instruction.",
			}),
			suggested_change: Type.String({
				minLength: 1,
				maxLength: 2000,
				description:
					"The concrete CAD geometry change that directly removes the newly verified obs; do not introduce an unobserved problem.",
			}),
		},
		{
			additionalProperties: false,
			description: "Append one new finding. The controller assigns its stable finding_id.",
		},
	),
	Type.Object(
		{
			action: Type.Literal("update"),
			finding_id: FINDING_ID_SCHEMA,
			field: Type.Union([Type.Literal("obs"), Type.Literal("suggested_change")], {
				description: "The one finding text field to edit.",
			}),
			old_text: Type.String({
				minLength: 1,
				maxLength: 2000,
				description: "Exact existing substring; it must occur exactly once in the selected field.",
			}),
			new_text: Type.String({
				maxLength: 2000,
				description: "Replacement substring. It may be empty if the complete field remains non-empty.",
			}),
		},
		{
			additionalProperties: false,
			description: "Replace exactly one local substring without overwriting the rest of the field.",
		},
	),
	Type.Object(
		{
			action: Type.Literal("delete"),
			finding_id: FINDING_ID_SCHEMA,
		},
		{
			additionalProperties: false,
			description: "Delete one finding while preserving all other finding IDs and their order.",
		},
	),
], {
	description: "One ordered, atomic change to the current region's persisted findings.",
});
export const UPDATE_FINDINGS_PARAMETERS_SCHEMA = Type.Object(
	{
		region_id: REGION_ID_SCHEMA,
		crop_id: Type.String({
			minLength: 20,
			maxLength: 20,
			pattern: "^[a-f0-9]{20}$",
			description: "Controller-generated crop_id from compare_region_grid_cells.",
		}),
		reviewed_finding_ids: Type.Array(FINDING_ID_SCHEMA, {
			minItems: 1,
			maxItems: 24,
			uniqueItems: true,
			description: "Existing findings verified by this crop, including findings retained unchanged.",
		}),
		changes: Type.Array(UPDATE_FINDING_CHANGE_SCHEMA, {
			maxItems: 24,
			description:
				"Ordered atomic additions, exact substring replacements, and deletions. [] retains the reviewed findings unchanged.",
		}),
	},
	{
		additionalProperties: false,
		description:
			"Update the persisted findings collection using one fixed-grid crop. Every successful call returns the complete current region state.",
	},
);
export const SUBMIT_REFLECTION_PARAMETERS_SCHEMA = Type.Object(
	{
		should_revise: Type.Boolean(),
		overall_summary: Type.String({ minLength: 1, maxLength: 4000 }),
		findings: Type.Array(
			Type.Object(
				{
					region_id: REGION_ID_SCHEMA,
					obs: Type.String({ minLength: 1, maxLength: 2000 }),
					suggested_change: Type.String({ minLength: 1, maxLength: 2000 }),
				},
				{ additionalProperties: false },
			),
			{ maxItems: 24 },
		),
	},
	{ additionalProperties: false },
);
export const SELF_REFLECT_PARAMETERS_SCHEMA = Type.Object({}, { additionalProperties: false });

export interface CropRecord {
	regionId: string;
	page: number;
	roi: NormalizedRoi;
	cropPath: string;
	width: number;
	height: number;
	cropSha256: string;
}

interface CandidateRecord {
	candidateId: string;
	regionId: string;
	viewSpec: DrawingViewSpec;
	renderPath: string;
	ranked?: DrawingViewCandidate;
}

interface RankingRecord {
	rankings: DrawingViewCandidate[];
	margin: number | null;
	comparisonPath: string;
}

export interface SegmentationState {
	crops: Map<string, CropRecord>;
	toolCalls: number;
	submitted: DrawingRegionManifest | null;
}

export interface SearchState {
	region: DrawingRegionManifestRegion;
	candidates: Map<string, CandidateRecord>;
	ranking: RankingRecord | null;
	nextCandidate: number;
	toolCalls: number;
	submitted: DrawingManifestRegion | null;
}

type GridCell = `${"A" | "B" | "C" | "D"}${1 | 2 | 3 | 4}`;

interface ReflectionGridOverviewRecord {
	region_id: string;
	context_path: string;
	pixel_width: number;
	pixel_height: number;
	sha256: string;
	created_at: string;
}

interface ReflectionCropRecord {
	crop_id: string;
	region_id: string;
	finding_ids: string[];
	reference_cell: GridCell;
	cad_cell: GridCell;
	reference_roi: NormalizedRoi;
	cad_roi: NormalizedRoi;
	reference_pixel_roi: [number, number, number, number];
	cad_pixel_roi: [number, number, number, number];
	reference_crop_path: string;
	cad_crop_path: string;
	context_path: string;
	comparison_path: string;
	sha256: {
		reference_crop: string;
		cad_crop: string;
		context: string;
		comparison: string;
	};
	created_at: string;
}

interface SavedRegionFinding {
	finding_id: string;
	obs: string;
	suggested_change: string;
	evidence_crop_ids: string[];
	created_from: "initial" | "grid_crop";
}

interface SavedRegionFindingsRecord {
	region_id: string;
	findings: SavedRegionFinding[];
	next_finding_sequence: number;
	saved_at: string;
}

export interface ReflectionState {
	manifest: DrawingViewManifest;
	report: DrawingReflectionReport | null;
	phase?: "global_inspection" | "grid_verification";
	initialFindingsSubmittedAt?: string;
	initialFindingKeys?: Set<string>;
	reviewedInitialFindingKeys?: Set<string>;
	currentRegionId?: string;
	inspectedRegions?: Set<string>;
	gridOverviews?: Map<string, ReflectionGridOverviewRecord>;
	crops?: Map<string, ReflectionCropRecord>;
	regionFindings?: Map<string, SavedRegionFindingsRecord>;
	inspectionManifestPath?: string;
}

export const REFLECTION_ACTIVE_TIMEOUT_MS = 30 * 60_000;
export const REFLECTION_WALL_TIMEOUT_MS = 120 * 60_000;

/** Scale internal Agent deadlines without changing their measured usage. */
export function restrictedSessionTimeouts(
	activeTimeoutMs: number,
	wallTimeoutMs: number,
	rawMultiplier: string | undefined = process.env.CADGENBENCH_SUBAGENT_TIMEOUT_MULTIPLIER,
): { activeTimeoutMs: number; wallTimeoutMs: number; multiplier: number } {
	const multiplier = rawMultiplier === undefined ? 1 : Number(rawMultiplier);
	if (!Number.isFinite(multiplier) || multiplier < 1 || multiplier > 12) {
		throw new InfrastructureError("CADGENBENCH_SUBAGENT_TIMEOUT_MULTIPLIER must be a finite number in [1, 12]");
	}
	const active = Math.round(activeTimeoutMs * multiplier);
	const wall = Math.round(wallTimeoutMs * multiplier);
	if (![active, wall].every(value => Number.isSafeInteger(value) && value > 0 && value <= 2_147_483_647)) {
		throw new InfrastructureError("Restricted session deadlines must fit positive Node.js timer limits");
	}
	return { activeTimeoutMs: active, wallTimeoutMs: wall, multiplier };
}

interface CacheEnvelope {
	version: typeof DRAWING_FEEDBACK_VERSION;
	summary: DrawingFeedbackSummary;
	manifest: DrawingViewManifest;
	report: DrawingReflectionReport;
}

export interface DrawingSegmentationResult {
	manifest: DrawingRegionManifest;
	usage?: SubagentRunUsage;
	cached: boolean;
	manifestPath: string;
	inputSha256: string;
}

export interface DrawingViewSearchOnlyResult {
	fixture_id: string;
	cache_key: string;
	input_sha256: string;
	source_sha256: string;
	step_sha256: string;
	segmentation_mode: "segmented" | "reused";
	search_mode: "searched" | "reused";
	region_manifest_path: string;
	manifest_path: string;
	manifest: DrawingViewManifest;
	usage: SubagentRunUsage[];
}

interface DrawingViewSearchStageResult {
	manifest: DrawingViewManifest;
	manifestPath: string;
	searchMode: "searched" | "reused";
	searchUsage: SubagentRunUsage[];
}

type LlmDrawingViewSpec = Omit<DrawingViewSpec, "name">;

interface RestrictedSessionResult<T> {
	value: T;
	usage: SubagentRunUsage;
}

function sameRoi(left: NormalizedRoi, right: NormalizedRoi): boolean {
	return left.every((value, index) => value === right[index]);
}

function roisOverlap(left: NormalizedRoi, right: NormalizedRoi): boolean {
	return Math.max(left[0], right[0]) < Math.min(left[2], right[2]) &&
		Math.max(left[1], right[1]) < Math.min(left[3], right[3]);
}

export function assertLockedTopTwo(
	rankedIds: string[],
	selectedId: string,
): string {
	const topTwo = rankedIds.slice(0, 2);
	if (topTwo.length !== 2 || !topTwo.includes(selectedId)) {
		throw new Error("Selected candidate must be one of the latest eligible Top-2 results");
	}
	return topTwo[0] === selectedId ? topTwo[1]! : topTwo[0]!;
}

export function assertRegionViewSemantics(
	viewType: "orthographic" | "isometric" | "section" | "detail" | "unknown",
	status: DrawingRegionStatus,
	viewSpecs: DrawingViewSpec[],
): void {
	if (status === "error") {
		if (viewSpecs.length !== 0) throw new Error("An error region cannot lock ranked view specs");
		return;
	}
	if (viewSpecs.length !== 2) throw new Error("A submitted region must lock exactly two ranked view specs");
	// Only the selected camera is used by Main and defines the region's view type.
	// The other eligible Top-2 item remains a competing hypothesis, which may
	// legitimately have a different section/non-section interpretation.
	const selected = viewSpecs[0]!;
	const selectedIsSection = selected.section_offset_fraction !== undefined && selected.section_keep !== undefined;
	if (viewType === "section" && !selectedIsSection) {
		throw new Error("The selected candidate for a section region must include section parameters");
	}
	if (viewType !== "section" && selectedIsSection) {
		throw new Error("Non-section regions cannot select a candidate with section parameters");
	}
}

export function assertNormalizedRoi(value: NormalizedRoi): void {
	if (!value.every((coordinate) => Number.isInteger(coordinate) && coordinate >= 0 && coordinate <= 1000)) {
		throw new Error("ROI coordinates must be integers in the normalized 0..1000 range");
	}
	if (value[0] >= value[2] || value[1] >= value[3]) {
		throw new Error("ROI must satisfy x0 < x1 and y0 < y1");
	}
}

export function assertSupportedViewSpec(spec: DrawingViewSpec | LlmDrawingViewSpec): void {
	const values = [...spec.direction, ...spec.up];
	if (!values.every(Number.isFinite)) throw new Error("View direction and up vectors must be finite");
	const directionLength = Math.hypot(...spec.direction);
	const upLength = Math.hypot(...spec.up);
	if (directionLength === 0) throw new Error("View direction must be non-zero");
	if (upLength === 0) throw new Error("View up must be non-zero");
	const dot = spec.direction.reduce((sum, value, index) => sum + value * (spec.up[index] ?? 0), 0);
	if (Math.abs(dot / (directionLength * upLength)) > 1e-8) {
		throw new Error("View direction and up vectors must be perpendicular");
	}
	const hasSectionOffset = spec.section_offset_fraction !== undefined;
	const hasSectionKeep = spec.section_keep !== undefined;
	if (hasSectionOffset !== hasSectionKeep) {
		throw new Error("Simple section candidates require both section_offset_fraction and section_keep");
	}
	if (
		spec.section_offset_fraction !== undefined &&
		(!Number.isFinite(spec.section_offset_fraction) ||
			spec.section_offset_fraction <= -1 ||
			spec.section_offset_fraction >= 1)
	) {
		throw new Error("Section offset fraction must be finite and strictly inside -1..1");
	}
}

function controllerViewSpec(candidateId: string, spec: LlmDrawingViewSpec): DrawingViewSpec {
	return { name: candidateId, ...spec };
}

function sameViewSpec(left: DrawingViewSpec, right: DrawingViewSpec): boolean {
	return (
		left.name === right.name &&
		left.direction.every((value, index) => value === right.direction[index]) &&
		left.up.every((value, index) => value === right.up[index]) &&
		left.section_offset_fraction === right.section_offset_fraction &&
		left.section_keep === right.section_keep
	);
}

function viewSpecIdentity(spec: DrawingViewSpec | LlmDrawingViewSpec): string {
	const normalise = (vector: Vector3): string => {
		const length = Math.hypot(...vector);
		return vector.map((value) => (value / length).toFixed(12)).join(",");
	};
	return [
		normalise(spec.direction),
		normalise(spec.up),
		String(spec.section_offset_fraction ?? "none"),
		spec.section_keep ?? "none",
	].join("|");
}

function safeWorkspacePath(workspace: string, relativePath: string): string {
	if (isAbsolute(relativePath)) throw new InfrastructureError("Drawing worker returned an absolute path");
	const root = resolve(workspace);
	const candidate = resolve(root, relativePath);
	const rel = relative(root, candidate);
	if (rel === ".." || rel.startsWith("../") || isAbsolute(rel)) {
		throw new InfrastructureError("Drawing worker returned a path outside the fixture workspace");
	}
	return candidate;
}

async function requireWorkspaceFile(workspace: string, relativePath: string): Promise<string> {
	const path = safeWorkspacePath(workspace, relativePath);
	const metadata = await lstat(path).catch(() => null);
	if (!metadata?.isFile() || metadata.isSymbolicLink()) {
		throw new InfrastructureError(`Drawing artifact is missing or unsafe: ${relativePath}`);
	}
	const [workspaceRealPath, fileRealPath] = await Promise.all([realpath(workspace), realpath(path)]);
	const realRelative = relative(workspaceRealPath, fileRealPath);
	if (realRelative === ".." || realRelative.startsWith("../") || isAbsolute(realRelative)) {
		throw new InfrastructureError(`Drawing artifact escaped the fixture workspace: ${relativePath}`);
	}
	return fileRealPath;
}

async function ensureWorkspaceDirectory(workspace: string, relativePath: string): Promise<string> {
	const root = resolve(workspace);
	const target = safeWorkspacePath(workspace, relativePath);
	const rel = relative(root, target);
	let current = root;
	for (const component of rel.split("/").filter((value) => value !== "" && value !== ".")) {
		current = join(current, component);
		const existing = await lstat(current).catch(() => null);
		if (!existing) {
			await mkdir(current);
		} else if (!existing.isDirectory() || existing.isSymbolicLink()) {
			throw new InfrastructureError(`Drawing cache directory is unsafe: ${relativePath}`);
		}
	}
	const [workspaceRealPath, directoryRealPath] = await Promise.all([realpath(root), realpath(target)]);
	const realRelative = relative(workspaceRealPath, directoryRealPath);
	if (realRelative === ".." || realRelative.startsWith("../") || isAbsolute(realRelative)) {
		throw new InfrastructureError(`Drawing cache directory escaped the fixture workspace: ${relativePath}`);
	}
	return directoryRealPath;
}

export async function inspectPngFile(
	workspace: string,
	relativePath: string,
): Promise<{ width: number; height: number; sha256: string }> {
	const bytes = await readFile(await requireWorkspaceFile(workspace, relativePath));
	return inspectPngBytes(bytes, relativePath);
}

function inspectPngBytes(
	bytes: Buffer,
	label: string,
): { width: number; height: number; sha256: string } {
	const maxFileBytes = 64 * 1024 * 1024;
	const maxEdge = 10_000;
	const maxPixels = 25_000_000;
	const pngSignature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
	if (
		bytes.length < 45 ||
		bytes.length > maxFileBytes ||
		pngSignature.some((value, index) => bytes[index] !== value) ||
		bytes.readUInt32BE(8) !== 13 ||
		bytes.toString("ascii", 12, 16) !== "IHDR"
	) {
		throw new InfrastructureError(`Drawing image is not a valid PNG: ${label}`);
	}

	let offset = 8;
	let width = 0;
	let height = 0;
	let channels = 0;
	let sawIhdr = false;
	let sawIdat = false;
	let idatEnded = false;
	let sawIend = false;
	const idatChunks: Buffer[] = [];
	let idatBytes = 0;
	while (offset < bytes.length) {
		if (bytes.length - offset < 12) {
			throw new InfrastructureError(`Drawing PNG has a truncated chunk: ${label}`);
		}
		const length = bytes.readUInt32BE(offset);
		const typeStart = offset + 4;
		const dataStart = offset + 8;
		const dataEnd = dataStart + length;
		const chunkEnd = dataEnd + 4;
		if (dataEnd < dataStart || chunkEnd > bytes.length) {
			throw new InfrastructureError(`Drawing PNG has an invalid chunk length: ${label}`);
		}
		const typeBytes = bytes.subarray(typeStart, dataStart);
		const type = typeBytes.toString("ascii");
		if (!/^[A-Za-z]{4}$/.test(type)) {
			throw new InfrastructureError(`Drawing PNG has an invalid chunk type: ${label}`);
		}
		const storedCrc = bytes.readUInt32BE(dataEnd);
		const computedCrc = pngCrc32(bytes.subarray(typeStart, dataEnd));
		if (storedCrc !== computedCrc) {
			throw new InfrastructureError(`Drawing PNG chunk CRC mismatch (${type}): ${label}`);
		}
		const data = bytes.subarray(dataStart, dataEnd);
		if (!sawIhdr && type !== "IHDR") {
			throw new InfrastructureError(`Drawing PNG does not start with IHDR: ${label}`);
		}
		if (type === "IHDR") {
			if (sawIhdr || offset !== 8 || length !== 13) {
				throw new InfrastructureError(`Drawing PNG has an invalid IHDR: ${label}`);
			}
			sawIhdr = true;
			width = data.readUInt32BE(0);
			height = data.readUInt32BE(4);
			const bitDepth = data[8];
			const colorType = data[9];
			const compression = data[10];
			const filter = data[11];
			const interlace = data[12];
			channels = colorType === 0 ? 1 : colorType === 2 ? 3 : colorType === 4 ? 2 : colorType === 6 ? 4 : 0;
			if (
				width < 1 ||
				height < 1 ||
				width > maxEdge ||
				height > maxEdge ||
				width * height > maxPixels ||
				bitDepth !== 8 ||
				channels === 0 ||
				compression !== 0 ||
				filter !== 0 ||
				interlace !== 0
			) {
				throw new InfrastructureError(`Drawing PNG uses unsupported image parameters: ${label}`);
			}
		} else if (type === "IDAT") {
			if (!sawIhdr || sawIend || idatEnded) {
				throw new InfrastructureError(`Drawing PNG has misplaced IDAT data: ${label}`);
			}
			sawIdat = true;
			idatBytes += length;
			if (idatBytes > maxFileBytes) {
				throw new InfrastructureError(`Drawing PNG has excessive compressed data: ${label}`);
			}
			idatChunks.push(data);
		} else if (type === "IEND") {
			if (!sawIhdr || !sawIdat || sawIend || length !== 0) {
				throw new InfrastructureError(`Drawing PNG has an invalid IEND: ${label}`);
			}
			sawIend = true;
			if (chunkEnd !== bytes.length) {
				throw new InfrastructureError(`Drawing PNG has trailing data after IEND: ${label}`);
			}
		} else {
			if (sawIdat) idatEnded = true;
			const isCritical = type.charCodeAt(0) >= 65 && type.charCodeAt(0) <= 90;
			if (isCritical && type !== "PLTE") {
				throw new InfrastructureError(`Drawing PNG has an unknown critical chunk (${type}): ${label}`);
			}
			if (type === "PLTE" && (sawIdat || length === 0 || length > 768 || length % 3 !== 0)) {
				throw new InfrastructureError(`Drawing PNG has an invalid PLTE: ${label}`);
			}
		}
		offset = chunkEnd;
		if (sawIend) break;
	}
	if (!sawIhdr || !sawIdat || !sawIend || offset !== bytes.length) {
		throw new InfrastructureError(`Drawing PNG is incomplete: ${label}`);
	}
	const rowBytes = width * channels;
	const expectedInflatedBytes = height * (rowBytes + 1);
	let inflated: Buffer;
	try {
		const compressed = Buffer.concat(idatChunks, idatBytes);
		const decoded = inflateSync(compressed, {
			maxOutputLength: expectedInflatedBytes + 1,
			info: true,
		}) as unknown as { buffer: Buffer; engine: { bytesWritten: number } };
		if (decoded.engine.bytesWritten !== compressed.length) {
			throw new Error("zlib stream has trailing or concatenated data");
		}
		inflated = decoded.buffer;
	} catch (error) {
		throw new InfrastructureError(`Drawing PNG zlib stream is invalid: ${label}`, { cause: error });
	}
	if (inflated.length !== expectedInflatedBytes) {
		throw new InfrastructureError(`Drawing PNG decompressed length is invalid: ${label}`);
	}
	for (let row = 0; row < height; row += 1) {
		const filterType = inflated[row * (rowBytes + 1)];
		if (filterType === undefined || filterType > 4) {
			throw new InfrastructureError(`Drawing PNG has an invalid row filter: ${label}`);
		}
	}
	return { width, height, sha256: createHash("sha256").update(bytes).digest("hex") };
}

const PNG_CRC32_TABLE = (() => {
	const table = new Uint32Array(256);
	for (let index = 0; index < table.length; index += 1) {
		let value = index;
		for (let bit = 0; bit < 8; bit += 1) {
			value = (value & 1) !== 0 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
		}
		table[index] = value >>> 0;
	}
	return table;
})();

function pngCrc32(bytes: Buffer): number {
	let value = 0xffffffff;
	for (const byte of bytes) {
		value = (PNG_CRC32_TABLE[(value ^ byte) & 0xff] ?? 0) ^ (value >>> 8);
	}
	return (value ^ 0xffffffff) >>> 0;
}

export async function imageFromWorkspace(workspace: string, path: string): Promise<ImageContent> {
	const bytes = await readFile(await requireWorkspaceFile(workspace, path));
	return { type: "image", data: bytes.toString("base64"), mimeType: "image/png" };
}

function effectiveSignal(primary: AbortSignal | undefined, controller: RunController): AbortSignal {
	return primary ? AbortSignal.any([primary, controller.signal]) : controller.signal;
}

function requireOperation(response: { operation?: string }, expected: string): void {
	if (response.operation !== expected) {
		throw new InfrastructureError(
			`Drawing worker returned operation=${String(response.operation)} for ${expected}`,
		);
	}
}

export function requireDrawingSuccess(
	response: { operation?: string; status?: "ok" | "error"; error?: { kind: string; message: string } },
	expected: string,
): void {
	requireOperation(response, expected);
	if (response.status === "error") {
		throw new Error(`Drawing ${response.error?.kind ?? "error"}: ${response.error?.message ?? "unknown failure"}`);
	}
	if (response.status !== "ok") {
		throw new InfrastructureError(`Drawing worker omitted status=ok for ${expected}`);
	}
}

export function makeSegmentationState(): SegmentationState {
	return { crops: new Map(), toolCalls: 0, submitted: null };
}

export function makeSearchState(region: DrawingRegionManifestRegion): SearchState {
	return {
		region,
		candidates: new Map(),
		ranking: null,
		nextCandidate: 1,
		toolCalls: 0,
		submitted: null,
	};
}

const SEGMENTATION_MAX_TOOL_CALLS = 32;
const SEGMENTATION_MAX_WORK_CALLS = SEGMENTATION_MAX_TOOL_CALLS - 1;

function consumeSegmentationWorkCall(state: SegmentationState): void {
	if (state.toolCalls >= SEGMENTATION_MAX_WORK_CALLS) {
		throw new Error(
			`Drawing segmentation has used ${state.toolCalls}/${SEGMENTATION_MAX_TOOL_CALLS} total calls; call ${SEGMENTATION_MAX_TOOL_CALLS} is reserved for submit_region_manifest`,
		);
	}
	state.toolCalls += 1;
}

function consumeSegmentationSubmitCall(state: SegmentationState): void {
	if (state.toolCalls >= SEGMENTATION_MAX_TOOL_CALLS) {
		throw new Error(
			`Drawing segmentation exhausted its ${SEGMENTATION_MAX_TOOL_CALLS}-call budget before submission`,
		);
	}
	state.toolCalls += 1;
}

function segmentationBudgetFeedback(state: SegmentationState): string {
	return `Segmentation budget: ${state.toolCalls}/${SEGMENTATION_MAX_TOOL_CALLS} total calls used; work calls stop at ${SEGMENTATION_MAX_WORK_CALLS} so submission remains available.`;
}

async function promoteCanonicalCrops(options: {
	workspace: string;
	canonicalRegionsDir: string;
	regions: Array<{
		crop: CropRecord;
		viewTypeHint: DrawingRegionManifestRegion["view_type_hint"];
		confidence: DrawingRegionManifestRegion["confidence"];
		note: string;
	}>;
}): Promise<DrawingRegionManifestRegion[]> {
	const canonicalRoot = safeWorkspacePath(options.workspace, options.canonicalRegionsDir);
	const nextRelative = `${options.canonicalRegionsDir}.next-${process.pid}-${Date.now()}`;
	const nextRoot = safeWorkspacePath(options.workspace, nextRelative);
	if (await lstat(nextRoot).catch(() => null)) {
		throw new InfrastructureError(`Temporary canonical crop directory already exists: ${nextRelative}`);
	}
	await ensureWorkspaceDirectory(options.workspace, nextRelative);
	const promoted: DrawingRegionManifestRegion[] = [];
	for (const item of options.regions) {
		const destinationRelative = join(nextRelative, item.crop.regionId, "crop.png");
		const destination = safeWorkspacePath(options.workspace, destinationRelative);
		await ensureWorkspaceDirectory(options.workspace, relative(options.workspace, dirname(destination)));
		await copyFile(await requireWorkspaceFile(options.workspace, item.crop.cropPath), destination);
		const inspected = await inspectPngFile(options.workspace, destinationRelative);
		if (
			inspected.width !== item.crop.width ||
			inspected.height !== item.crop.height ||
			inspected.sha256 !== item.crop.cropSha256
		) {
			throw new InfrastructureError(`Promoted crop changed unexpectedly for ${item.crop.regionId}`);
		}
		promoted.push({
			region_id: item.crop.regionId,
			page: item.crop.page,
			roi: item.crop.roi,
			crop_path: join(options.canonicalRegionsDir, item.crop.regionId, "crop.png"),
			view_type_hint: item.viewTypeHint,
			confidence: item.confidence,
			note: item.note,
			width: inspected.width,
			height: inspected.height,
			crop_sha256: inspected.sha256,
		});
	}

	const existing = await lstat(canonicalRoot).catch(() => null);
	let backupRoot: string | undefined;
	if (existing) {
		if (!existing.isDirectory() || existing.isSymbolicLink()) {
			throw new InfrastructureError(`Canonical crop path is not a safe directory: ${options.canonicalRegionsDir}`);
		}
		backupRoot = `${canonicalRoot}.previous-${process.pid}-${Date.now()}`;
		await rename(canonicalRoot, backupRoot);
	}
	try {
		await rename(nextRoot, canonicalRoot);
	} catch (error) {
		if (backupRoot) await rename(backupRoot, canonicalRoot).catch(() => undefined);
		throw error;
	}
	return promoted;
}

export function createDrawingSegmentationTools(options: {
	worker: WorkerClient;
	workspace: string;
	pagePaths: string[];
	workDir: string;
	canonicalRegionsDir: string;
	manifestPath: string;
	inputSha256: string;
	controller: RunController;
	state: SegmentationState;
}) {
	const cropTool = defineTool({
		name: "crop_region",
		label: "Crop drawing region",
		description:
			"Persist one complete geometry view and its associated dimension text, dimension lines, arrows, and leaders using whole-page normalized [x0,y0,x1,y1]. Same-page submitted crops must not overlap. Reusing a region_id replaces its current draft crop and returns the new image for inspection.",
		parameters: CROP_REGION_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params, signal) {
			consumeSegmentationWorkCall(options.state);
			assertNormalizedRoi(params.roi);
			const inputPath = options.pagePaths[params.page];
			if (!inputPath) throw new Error(`Drawing page ${params.page} does not exist`);
			const response = await options.worker.request<WorkerCropDrawingRegionResponse>(
				options.workspace,
				{
					protocol_version: PROTOCOL_VERSION,
					operation: "crop_drawing_region",
					input_path: inputPath,
					region_id: params.region_id,
					roi: params.roi,
					output_dir: join(options.workDir, "regions", params.region_id),
				},
				{ signal: effectiveSignal(signal, options.controller) },
			);
			requireDrawingSuccess(response, "crop_drawing_region");
			const expectedCropPath = join(options.workDir, "regions", params.region_id, "crop.png");
			if (
				response.region_id !== params.region_id ||
				response.input_path !== inputPath ||
				!sameRoi(response.roi, params.roi) ||
				response.crop_path !== expectedCropPath
			) {
				throw new InfrastructureError("Drawing worker crop identity did not match the request");
			}
			const inspected = await inspectPngFile(options.workspace, response.crop_path);
			if (response.width !== inspected.width || response.height !== inspected.height) {
				throw new InfrastructureError("Drawing worker crop dimensions did not match its PNG");
			}
			const replaced = options.state.crops.has(params.region_id);
			const record: CropRecord = {
				regionId: params.region_id,
				page: params.page,
				roi: params.roi,
				cropPath: response.crop_path,
				width: inspected.width,
				height: inspected.height,
				cropSha256: inspected.sha256,
			};
			options.state.crops.set(params.region_id, record);
			return {
				content: [
					{
						type: "text" as const,
						text: `${replaced ? "Replaced" : "Saved"} draft crop ${params.region_id} from page ${params.page}; ${inspected.width}x${inspected.height}, sha256=${inspected.sha256}. ${segmentationBudgetFeedback(options.state)}`,
					},
					await imageFromWorkspace(options.workspace, response.crop_path),
				],
				details: record,
			};
		},
	});

	const submitTool = defineTool({
		name: "submit_region_manifest",
		label: "Submit locked drawing regions",
		description:
			"Lock the final set of controller-saved crops. Accepts region identifiers and classification metadata only; paths, pixels, page numbers, and ROIs cannot be supplied.",
		parameters: SUBMIT_REGION_MANIFEST_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			consumeSegmentationSubmitCall(options.state);
			if (options.state.submitted) throw new Error("A drawing region manifest was already submitted");
			const ids = new Set<string>();
			const selected = params.regions.map((submitted) => {
				if (ids.has(submitted.region_id)) throw new Error(`Duplicate region ${submitted.region_id}`);
				ids.add(submitted.region_id);
				const crop = options.state.crops.get(submitted.region_id);
				if (!crop) throw new Error(`Region ${submitted.region_id} has no controller-saved crop`);
				return {
					crop,
					viewTypeHint: submitted.view_type_hint,
					confidence: submitted.confidence,
					note: submitted.note,
				};
			});
			for (const [index, item] of selected.entries()) {
				for (const other of selected.slice(index + 1)) {
					if (item.crop.page === other.crop.page && roisOverlap(item.crop.roi, other.crop.roi)) {
						throw new Error(
							`Drawing regions ${item.crop.regionId} and ${other.crop.regionId} overlap on page ${item.crop.page}; recrop them with disjoint ROIs`,
						);
					}
				}
			}
			const regions = await promoteCanonicalCrops({
				workspace: options.workspace,
				canonicalRegionsDir: options.canonicalRegionsDir,
				regions: selected,
			});
			const manifest: DrawingRegionManifest = {
				version: 1,
				input_sha256: options.inputSha256,
				regions,
			};
			await writeJsonAtomic(safeWorkspacePath(options.workspace, options.manifestPath), manifest);
			options.state.submitted = manifest;
			return {
				content: [
					{
						type: "text" as const,
						text: `Accepted and promoted ${regions.length} persistent drawing crops. ${segmentationBudgetFeedback(options.state)}`,
					},
				],
				details: manifest,
			};
		},
	});

	return [cropTool, submitTool] as const;
}

function consumeSearchWorkCall(state: SearchState): void {
	if (state.toolCalls >= 23) {
		throw new Error(
			`View search has used ${state.toolCalls}/24 total calls; call 24 is reserved for submit_view_result`,
		);
	}
	state.toolCalls += 1;
}

function consumeSearchSubmitCall(state: SearchState): void {
	if (state.toolCalls >= 24) throw new Error("View search exhausted its 24-call budget before submission");
	state.toolCalls += 1;
}

function searchBudgetFeedback(state: SearchState): string {
	return `Search budget: ${state.toolCalls}/24 total calls used; work calls stop at 23 so submission remains available.`;
}

export function createViewSearchTools(options: {
	worker: WorkerClient;
	workspace: string;
	region: DrawingRegionManifestRegion;
	outputDir: string;
	expectedStepSha256: string;
	controller: RunController;
	state: SearchState;
}) {
	const renderTool = defineTool({
		name: "render_candidate_view",
		label: "Render candidate view",
		description:
			"Optional preview step: render one new CAD camera without scoring it. Pass {view_spec:{direction:[x,y,z],up:[x,y,z]}} for an exterior view; add both section_offset_fraction and section_keep for a straight section. The result returns a controller-generated candidate_id; score that preview later with rank_view_candidates({candidate_ids:[candidate_id]}). direction/up must be finite, non-zero, and perpendicular.",
		parameters: RENDER_CANDIDATE_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params, signal) {
			consumeSearchWorkCall(options.state);
			if (options.state.candidates.size >= 32) throw new Error("This region already has the maximum 32 candidates");
			assertSupportedViewSpec(params.view_spec);
			const identity = viewSpecIdentity(params.view_spec);
			if ([...options.state.candidates.values()].some(({ viewSpec }) => viewSpecIdentity(viewSpec) === identity)) {
				throw new Error("This camera already exists in the region candidate pool");
			}
			const candidateId = `candidate-${String(options.state.nextCandidate).padStart(3, "0")}`;
			options.state.nextCandidate += 1;
			const requestedViewSpec = controllerViewSpec(candidateId, params.view_spec);
			const response = await options.worker.request<WorkerRenderDrawingViewResponse>(
				options.workspace,
				{
					protocol_version: PROTOCOL_VERSION,
					operation: "render_drawing_view",
					step_path: "output.step",
					region_id: options.region.region_id,
					view_spec: requestedViewSpec,
					output_dir: join(options.outputDir, "candidates", candidateId),
				},
				{ timeoutMs: 180_000, signal: effectiveSignal(signal, options.controller) },
			);
			requireDrawingSuccess(response, "render_drawing_view");
			if (response.region_id !== options.region.region_id || response.step_sha256 !== options.expectedStepSha256) {
				throw new InfrastructureError("Rendered candidate did not use the current formal STEP");
			}
			assertSupportedViewSpec(response.view_spec);
			if (!sameViewSpec(requestedViewSpec, response.view_spec)) {
				throw new InfrastructureError("2D renderer changed the controller-generated candidate name or view spec");
			}
			await requireWorkspaceFile(options.workspace, response.render_path);
			const record: CandidateRecord = {
				candidateId,
				regionId: options.region.region_id,
				viewSpec: response.view_spec,
				renderPath: response.render_path,
			};
			options.state.candidates.set(candidateId, record);
			return {
				content: [
					{
						type: "text" as const,
						text: `Rendered CAD image metadata: ${JSON.stringify({ candidate_id: candidateId, direction: response.view_spec.direction, up: response.view_spec.up, section_offset_fraction: response.view_spec.section_offset_fraction ?? null, section_keep: response.view_spec.section_keep ?? null })}. ${searchBudgetFeedback(options.state)}`,
					},
					await imageFromWorkspace(options.workspace, response.render_path),
				],
				details: record,
			};
		},
	});

	const rankTool = defineTool({
		name: "rank_view_candidates",
		label: "Rank candidate views",
		description:
			"Primary search step. Use {candidates:[view_spec,...]} to render and score 1-32 new cameras in one batch, OR {candidate_ids:[id,...]} to score previews returned by render_candidate_view; never provide both fields. At most 32 unique candidates may exist cumulatively. Returns the cumulative numeric ranking and labelled comparison images for the eligible Top-2 only.",
		parameters: RANK_CANDIDATES_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params, signal) {
			consumeSearchWorkCall(options.state);
			const batchMode = params.candidates !== undefined;
			if (batchMode === (params.candidate_ids !== undefined)) {
				throw new Error("Supply exactly one of candidates or candidate_ids");
			}
			if (batchMode && options.state.candidates.size + params.candidates!.length > 32) {
				throw new Error("A region may contain at most 32 cumulative candidates");
			}
			if (batchMode) {
				const identities = new Set(
					[...options.state.candidates.values()].map(({ viewSpec }) => viewSpecIdentity(viewSpec)),
				);
				for (const viewSpec of params.candidates!) {
					assertSupportedViewSpec(viewSpec);
					const identity = viewSpecIdentity(viewSpec);
					if (identities.has(identity)) throw new Error("Rank request contains an existing or duplicate camera");
					identities.add(identity);
				}
			}
			const requestedRecords: CandidateRecord[] = batchMode
				? params.candidates!.map((viewSpec) => {
						assertSupportedViewSpec(viewSpec);
						const candidateId = `candidate-${String(options.state.nextCandidate).padStart(3, "0")}`;
						options.state.nextCandidate += 1;
						return {
							candidateId,
							regionId: options.region.region_id,
							viewSpec: controllerViewSpec(candidateId, viewSpec),
							renderPath: "",
						};
					})
				: params.candidate_ids!.map((candidateId) => {
						const candidate = options.state.candidates.get(candidateId);
						if (!candidate) throw new Error(`Unknown candidate ${candidateId} for ${options.region.region_id}`);
						return candidate;
					});
			const candidates: Array<{
				candidate_id: string;
				view_spec: DrawingViewSpec;
				render_path?: string;
			}> = requestedRecords.map((candidate) => ({
				candidate_id: candidate.candidateId,
				view_spec: candidate.viewSpec,
				...(candidate.renderPath ? { render_path: candidate.renderPath } : {}),
			}));
			if (new Set(candidates.map(({ view_spec }) => view_spec.name)).size !== candidates.length) {
				throw new Error("Ranked candidates for one region must have unique view_spec.name values");
			}
			const response = await options.worker.request<WorkerRankDrawingViewsResponse>(
				options.workspace,
				{
					protocol_version: PROTOCOL_VERSION,
					operation: "rank_drawing_views",
					region_id: options.region.region_id,
					crop_path: options.region.crop_path,
					step_path: "output.step",
					candidates,
					top_k: Math.min(2, candidates.length),
					output_dir: join(options.outputDir, "rankings"),
				},
				{
					timeoutMs: 60_000 + 180_000 * candidates.length,
					signal: effectiveSignal(signal, options.controller),
				},
			);
			requireDrawingSuccess(response, "rank_drawing_views");
			if (
				response.region_id !== options.region.region_id ||
				response.step_sha256 !== options.expectedStepSha256 ||
				response.rankings.length === 0 ||
				response.scoring_mode !== "capacity_transport" ||
				response.scorer_profile !== "capacity_transport_v1"
			) {
				throw new InfrastructureError("Drawing ranker returned an invalid capacity-transport response");
			}
			const requestedById = new Map(candidates.map((candidate) => [candidate.candidate_id, candidate]));
			const requestedIds = new Set(requestedById.keys());
			const rankingIds = new Set<string>();
			const batchRankings = response.rankings.map((ranked): DrawingViewCandidate => {
				const requested = requestedById.get(ranked.candidate_id);
				if (
					!requested ||
					!requestedIds.has(ranked.candidate_id) ||
					rankingIds.has(ranked.candidate_id) ||
					!Number.isFinite(ranked.score) ||
					ranked.score < 0 ||
					ranked.score > 1 ||
					!Number.isFinite(ranked.raw_score) ||
					typeof ranked.eligible !== "boolean" ||
					!Array.isArray(ranked.failed_gates) ||
					ranked.scoring_mode !== "capacity_transport" ||
					!ranked.capacity_transport ||
					!sameViewSpec(requested.view_spec, ranked.view_spec) ||
					typeof ranked.render_path !== "string" ||
					ranked.render_path === "" ||
					typeof ranked.aligned_render_path !== "string" ||
					ranked.aligned_render_path === "" ||
					!ranked.transform ||
					![
						ranked.transform.scale_px_per_unit,
						ranked.transform.rotation_deg,
						ranked.transform.center_x_px,
						ranked.transform.center_y_px,
						ranked.transform.model_center_x,
						ranked.transform.model_center_y,
					].every(Number.isFinite) ||
					typeof ranked.transform.reflected !== "boolean" ||
					!ranked.alignment ||
					!Number.isFinite(ranked.alignment.objective) ||
					ranked.alignment.mode !== "capacity_transport" ||
					typeof ranked.alignment.scale_at_bound !== "boolean" ||
					typeof ranked.alignment.rotation_at_limit !== "boolean" ||
					typeof ranked.alignment.translation_at_limit !== "boolean" ||
					(!batchMode && ranked.render_path !== requested.render_path)
				) {
					throw new InfrastructureError("Drawing ranker returned an invalid candidate or score");
				}
				rankingIds.add(ranked.candidate_id);
				return {
					...ranked,
				};
			});
			if (rankingIds.size !== requestedIds.size || response.rankings.length !== candidates.length) {
				throw new InfrastructureError("Drawing ranker did not return every requested candidate exactly once");
			}
			if (response.margin !== null && (!Number.isFinite(response.margin) || response.margin < 0)) {
				throw new InfrastructureError("Drawing ranker returned an invalid best-vs-second margin");
			}
			await Promise.all([
				requireWorkspaceFile(options.workspace, response.comparison_path),
				...batchRankings.flatMap(({ render_path, comparison_path, aligned_render_path }) => [
					requireWorkspaceFile(options.workspace, render_path),
					requireWorkspaceFile(options.workspace, comparison_path),
					requireWorkspaceFile(options.workspace, aligned_render_path),
				]),
			]);
			for (const ranked of batchRankings) {
				options.state.candidates.set(ranked.candidate_id, {
					candidateId: ranked.candidate_id,
					regionId: options.region.region_id,
					viewSpec: ranked.view_spec,
					renderPath: ranked.render_path,
					ranked,
				});
			}
			const rankings = [...options.state.candidates.values()]
				.flatMap((candidate) => candidate.ranked ? [candidate.ranked] : [])
				.sort((left, right) =>
					Number(right.eligible) - Number(left.eligible) ||
					right.raw_score - left.raw_score ||
					left.candidate_id.localeCompare(right.candidate_id),
				);
			const eligibleTopTwo = rankings.filter(({ eligible }) => eligible).slice(0, 2);
			const margin = eligibleTopTwo.length === 2
				? eligibleTopTwo[0]!.score - eligibleTopTwo[1]!.score
				: null;
			const record: RankingRecord = {
				rankings,
				margin,
				comparisonPath: eligibleTopTwo[0]?.comparison_path ?? response.comparison_path,
			};
			options.state.ranking = record;
			const labelledTopTwo = await Promise.all(
				eligibleTopTwo.map(async (candidate, index) => [
					{
						type: "text" as const,
						text: `Top-2 CAD comparison image ${String(index + 1)} metadata: ${JSON.stringify({ numeric_rank: index + 1, candidate_id: candidate.candidate_id, direction: candidate.view_spec.direction, up: candidate.view_spec.up, section_offset_fraction: candidate.view_spec.section_offset_fraction ?? null, section_keep: candidate.view_spec.section_keep ?? null, score: candidate.score, raw_score: candidate.raw_score })}`,
					},
					await imageFromWorkspace(options.workspace, candidate.comparison_path),
				]),
			);
			return {
				content: [
					{
						type: "text" as const,
						text: `Cumulative ranking for ${options.region.region_id}: ${rankings.map(({ candidate_id, score, eligible }) => `${candidate_id}=${score.toFixed(3)}${eligible ? "" : " (ineligible)"}`).join(", ")}; eligible Top-2 margin=${String(margin)}. ${searchBudgetFeedback(options.state)}`,
					},
					...labelledTopTwo.flat(),
				],
				details: {
					...response,
					rankings,
					top: eligibleTopTwo,
					margin,
					comparison_path: record.comparisonPath,
				},
			};
		},
	});

	const submitTool = defineTool({
		name: "submit_view_result",
		label: "Submit region view result",
		description:
			"Final step. Submit {view_type,status,selected_candidate_id,confidence,note}. status is matched or ambiguous; both must select either candidate from the latest eligible Top-2. The controller automatically stores the other Top-2 candidate as the alternative.",
		parameters: SUBMIT_VIEW_RESULT_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			consumeSearchSubmitCall(options.state);
			if (options.state.submitted) throw new Error("A view result was already submitted");
			const eligibleTopTwo = options.state.ranking?.rankings.filter(({ eligible }) => eligible).slice(0, 2) ?? [];
			const alternativeId = assertLockedTopTwo(
				eligibleTopTwo.map(({ candidate_id }) => candidate_id),
				params.selected_candidate_id,
			);
			const selected = eligibleTopTwo.find(
				({ candidate_id }) => candidate_id === params.selected_candidate_id,
			)!;
			const alternatives = [eligibleTopTwo.find(({ candidate_id }) => candidate_id === alternativeId)!];
			assertRegionViewSemantics(
				params.view_type,
				params.status,
				[selected.view_spec, alternatives[0]!.view_spec],
			);
			const result: DrawingManifestRegion = {
				region_id: options.region.region_id,
				page: options.region.page,
				roi: options.region.roi,
				crop_path: options.region.crop_path,
				view_type: params.view_type,
				status: params.status,
				selected,
				alternatives,
				margin: options.state.ranking?.margin ?? null,
				confidence: params.confidence,
				note: params.note,
				width: options.region.width,
				height: options.region.height,
				crop_sha256: options.region.crop_sha256,
			};
			options.state.submitted = result;
			return {
				content: [
					{
						type: "text" as const,
						text: `Accepted the locked result for ${options.region.region_id}. ${searchBudgetFeedback(options.state)}`,
					},
				],
				details: result,
			};
		},
	});

	return [renderTool, rankTool, submitTool] as const;
}

export function alignmentNotice(region: DrawingManifestRegion): string {
	const alignment = region.selected?.alignment;
	const replayGates = region.selected?.locked_replay_original_failed_gates ?? [];
	const replayWarning = region.selected?.locked_replay_original_eligible === false || replayGates.length > 0
		? `WARNING: the historical locked camera failed current replay gate(s): ${replayGates.join(", ") || "ineligible"}. Treat this view and all local geometry claims from it as uncertain.`
		: null;
	if (!alignment) return replayWarning ?? "No locked alignment diagnostics are available.";
	const limits = [
		alignment.scale_at_bound ? "scale_at_bound" : null,
		alignment.rotation_at_limit ? "rotation_at_limit" : null,
		alignment.translation_at_limit ? "translation_at_limit" : null,
	].filter((value): value is string => value !== null);
	const alignmentMessage = limits.length > 0
		? `WARNING: global alignment may be unreliable because ${limits.join(", ")}. Treat local geometry claims as uncertain.`
		: "Global alignment did not hit scale, rotation, or translation limits.";
	return replayWarning ? `${replayWarning} ${alignmentMessage}` : alignmentMessage;
}

async function persistReflectionInspectionManifest(
	workspace: string,
	state: ReflectionState,
): Promise<void> {
	if (!state.inspectionManifestPath) return;
	await ensureWorkspaceDirectory(workspace, dirname(state.inspectionManifestPath));
	await writeJsonAtomic(safeWorkspacePath(workspace, state.inspectionManifestPath), {
		version: 7,
		view_manifest_version: state.manifest.version,
		input_sha256: state.manifest.input_sha256,
		source_sha256: state.manifest.source_sha256,
		step_sha256: state.manifest.step_sha256,
		phase: state.phase ?? "global_inspection",
		initial_findings_submitted_at: state.initialFindingsSubmittedAt ?? null,
		initial_finding_keys: [...(state.initialFindingKeys ?? [])].sort(),
		reviewed_initial_finding_keys: [...(state.reviewedInitialFindingKeys ?? [])].sort(),
		current_region_id: state.currentRegionId ?? null,
		inspected_regions: [...(state.inspectedRegions ?? [])].sort(),
		grid_overviews: [...(state.gridOverviews?.values() ?? [])].sort((left, right) =>
			left.region_id.localeCompare(right.region_id)),
		comparisons: [...(state.crops?.values() ?? [])].sort((left, right) =>
			left.crop_id.localeCompare(right.crop_id)),
		region_findings: state.manifest.regions.flatMap(({ region_id }) => {
			const saved = state.regionFindings?.get(region_id);
			return saved ? [saved] : [];
		}),
	});
}

export async function restoreReflectionInspectionManifest(
	workspace: string,
	state: ReflectionState,
): Promise<void> {
	if (!state.inspectionManifestPath) return;
	const manifestPath = await requireWorkspaceFile(workspace, state.inspectionManifestPath).catch(() => null);
	if (!manifestPath) return;
	const persisted = await readJson<{
		version: number;
		view_manifest_version: number;
		input_sha256: string;
		source_sha256: string;
		step_sha256: string;
		phase: "global_inspection" | "grid_verification";
		initial_findings_submitted_at: string | null;
		initial_finding_keys: string[];
		reviewed_initial_finding_keys: string[];
		current_region_id: string | null;
		inspected_regions: string[];
		grid_overviews: ReflectionGridOverviewRecord[];
		comparisons: ReflectionCropRecord[];
		region_findings: SavedRegionFindingsRecord[];
	}>(manifestPath).catch(() => null);
	if (
		!persisted ||
		persisted.version !== 7 ||
		persisted.view_manifest_version !== state.manifest.version ||
		persisted.input_sha256 !== state.manifest.input_sha256 ||
		persisted.source_sha256 !== state.manifest.source_sha256 ||
		persisted.step_sha256 !== state.manifest.step_sha256 ||
		!(["global_inspection", "grid_verification"] as const).includes(persisted.phase) ||
		!Array.isArray(persisted.initial_finding_keys) ||
		!Array.isArray(persisted.reviewed_initial_finding_keys) ||
		!Array.isArray(persisted.inspected_regions) ||
		!Array.isArray(persisted.grid_overviews) ||
		!Array.isArray(persisted.comparisons) ||
		!Array.isArray(persisted.region_findings)
	) return;
	const regionIds = new Set(state.manifest.regions.map(({ region_id }) => region_id));
	const comparableRegionIds = new Set(
		state.manifest.regions.filter((region) => region.selected).map(({ region_id }) => region_id),
	);
	try {
		if (
			persisted.current_region_id !== null &&
			(typeof persisted.current_region_id !== "string" ||
				!comparableRegionIds.has(persisted.current_region_id))
		) throw new Error("invalid current region");
		const inspectedRegions = new Set<string>();
		for (const regionId of persisted.inspected_regions) {
			if (!regionIds.has(regionId)) throw new Error("unknown inspected region");
			inspectedRegions.add(regionId);
		}
		const gridOverviews = new Map<string, ReflectionGridOverviewRecord>();
		for (const record of persisted.grid_overviews) {
			if (!record || !comparableRegionIds.has(record.region_id) || typeof record.context_path !== "string") {
				throw new Error("invalid grid overview record");
			}
			const inspected = await inspectPngFile(workspace, record.context_path);
			if (
				inspected.sha256 !== record.sha256 ||
				!Number.isInteger(record.pixel_width) ||
				!Number.isInteger(record.pixel_height) ||
				record.pixel_width < 1 ||
				record.pixel_height < 1
			) throw new Error("stale grid overview record");
			gridOverviews.set(record.region_id, record);
		}
		const comparisons = new Map<string, ReflectionCropRecord>();
		for (const record of persisted.comparisons) {
			if (
				!record ||
				!comparableRegionIds.has(record.region_id) ||
				typeof record.crop_id !== "string" ||
				!Array.isArray(record.finding_ids) ||
				record.finding_ids.length < 1 ||
				!record.finding_ids.every((findingId) => typeof findingId === "string") ||
				!/^([A-D][1-4])$/u.test(record.reference_cell) ||
				!/^([A-D][1-4])$/u.test(record.cad_cell)
			) {
				throw new Error("invalid local comparison record");
			}
			if (
				!sameRoi(record.reference_roi, gridCellRoi(record.reference_cell)) ||
				!sameRoi(record.cad_roi, gridCellRoi(record.cad_cell))
			) throw new Error("local comparison does not match its fixed cells");
			const inspected = await Promise.all([
				inspectPngFile(workspace, record.reference_crop_path),
				inspectPngFile(workspace, record.cad_crop_path),
				inspectPngFile(workspace, record.context_path),
				inspectPngFile(workspace, record.comparison_path),
			]);
			if (
				inspected[0].sha256 !== record.sha256.reference_crop ||
				inspected[1].sha256 !== record.sha256.cad_crop ||
				inspected[2].sha256 !== record.sha256.context ||
				inspected[3].sha256 !== record.sha256.comparison
			) throw new Error("stale local comparison record");
			comparisons.set(record.crop_id, record);
		}
		const regionFindings = new Map<string, SavedRegionFindingsRecord>();
		for (const record of persisted.region_findings) {
			if (
				!record ||
				!comparableRegionIds.has(record.region_id) ||
				!Array.isArray(record.findings) ||
					record.findings.length > 24 ||
				!Number.isInteger(record.next_finding_sequence) ||
				record.next_finding_sequence < 1 ||
				!record.findings.every((finding) =>
					finding &&
					typeof finding.finding_id === "string" &&
					/^finding-[0-9]{3,}$/u.test(finding.finding_id) &&
					typeof finding.obs === "string" &&
					finding.obs.length > 0 &&
					finding.obs.length <= 2000 &&
					typeof finding.suggested_change === "string" &&
					finding.suggested_change.length > 0 &&
					finding.suggested_change.length <= 2000 &&
					Array.isArray(finding.evidence_crop_ids) &&
					finding.evidence_crop_ids.every((cropId) => typeof cropId === "string") &&
					(finding.created_from === "initial" || finding.created_from === "grid_crop"))
			) throw new Error("invalid saved region findings");
			const findingIds = new Set<string>();
			for (const finding of record.findings) {
				if (findingIds.has(finding.finding_id)) throw new Error("duplicate saved finding id");
				findingIds.add(finding.finding_id);
			}
			const maximumFindingSequence = record.findings.reduce((maximum, finding) =>
				Math.max(maximum, Number.parseInt(finding.finding_id.slice("finding-".length), 10)), 0);
			if (record.next_finding_sequence <= maximumFindingSequence) {
				throw new Error("invalid next finding sequence");
			}
			for (const finding of record.findings) {
				for (const cropId of finding.evidence_crop_ids) {
					if (comparisons.get(cropId)?.region_id !== record.region_id) {
						throw new Error("saved finding has invalid grid evidence");
					}
				}
			}
			regionFindings.set(record.region_id, record);
		}
		const allKnownFindingKeys = new Set(
			[...regionFindings.values()].flatMap(({ region_id, findings }) =>
				findings.map(({ finding_id }) => findingKey(region_id, finding_id))),
		);
		const initialFindingKeys = new Set<string>();
		for (const key of persisted.initial_finding_keys) {
			if (typeof key !== "string") throw new Error("invalid initial finding key");
			initialFindingKeys.add(key);
		}
		const reviewedInitialFindingKeys = new Set<string>();
		for (const key of persisted.reviewed_initial_finding_keys) {
			if (typeof key !== "string" || !initialFindingKeys.has(key)) {
				throw new Error("invalid reviewed initial finding key");
			}
			reviewedInitialFindingKeys.add(key);
		}
		if (persisted.phase === "global_inspection" && initialFindingKeys.size > 0) {
			throw new Error("global phase cannot contain initial findings");
		}
		if (persisted.phase === "grid_verification") {
			if (typeof persisted.initial_findings_submitted_at !== "string") {
				throw new Error("grid verification lacks its initial submission timestamp");
			}
			for (const key of initialFindingKeys) {
				if (!allKnownFindingKeys.has(key) && !reviewedInitialFindingKeys.has(key)) {
					throw new Error("unreviewed initial finding is missing");
				}
			}
		}
		state.phase = persisted.phase;
		if (persisted.initial_findings_submitted_at === null) delete state.initialFindingsSubmittedAt;
		else state.initialFindingsSubmittedAt = persisted.initial_findings_submitted_at;
		state.initialFindingKeys = initialFindingKeys;
		state.reviewedInitialFindingKeys = reviewedInitialFindingKeys;
		state.inspectedRegions = inspectedRegions;
		state.gridOverviews = gridOverviews;
		state.crops = comparisons;
		state.regionFindings = regionFindings;
		if (persisted.current_region_id === null) delete state.currentRegionId;
		else state.currentRegionId = persisted.current_region_id;
	} catch {
		// A partial or stale manifest is not trusted. The Agent may rebuild its
		// inspection state without affecting the locked view manifest.
	}
}

function takeNextFindingId(record: SavedRegionFindingsRecord): string {
	const findingId = `finding-${String(record.next_finding_sequence).padStart(3, "0")}`;
	record.next_finding_sequence += 1;
	return findingId;
}

function findingKey(regionId: string, findingId: string): string {
	return `${regionId}:${findingId}`;
}

function gridCellRoi(cell: GridCell): NormalizedRoi {
	const column = cell.charCodeAt(0) - "A".charCodeAt(0);
	const row = Number.parseInt(cell.slice(1), 10) - 1;
	return [column * 250, row * 250, (column + 1) * 250, (row + 1) * 250];
}

function replaceFindingSubstring(
	value: string,
	oldText: string,
	newText: string,
	field: "obs" | "suggested_change",
): string {
	const firstIndex = value.indexOf(oldText);
	if (firstIndex < 0) {
		throw new Error(`old_text was not found in ${field}`);
	}
	if (value.indexOf(oldText, firstIndex + oldText.length) >= 0) {
		throw new Error(`old_text occurs more than once in ${field}; provide a more specific substring`);
	}
	const updated = `${value.slice(0, firstIndex)}${newText}${value.slice(firstIndex + oldText.length)}`;
	if (updated.length < 1 || updated.length > 2000) {
		throw new Error(`The updated ${field} must contain 1-2000 characters`);
	}
	return updated;
}

export function createReflectionTools(options: {
	workspace: string;
	state: ReflectionState;
	worker?: WorkerClient;
	controller?: RunController;
	outputDir?: string;
}) {
	const inspectTool = defineTool({
		name: "inspect_region",
		label: "Inspect locked comparison region",
		description:
			"Inspect the drawing crop and already-rendered locked CAD comparison for a region_id. No camera or view parameters are accepted.",
		parameters: INSPECT_REGION_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			const region = options.state.manifest.regions.find(({ region_id }) => region_id === params.region_id);
			if (!region) throw new Error(`Unknown comparison region ${params.region_id}`);
			const content: Array<{ type: "text"; text: string } | ImageContent> = [
				{
					type: "text",
					text: `${region.region_id}: status=${region.status}, view_type=${region.view_type}, score=${String(region.selected?.score ?? "none")}, margin=${String(region.margin)}. ${region.note}`,
				},
			];
			if (region.selected?.comparison_path) {
				content.push(await imageFromWorkspace(options.workspace, region.selected.comparison_path));
			} else {
				content.push(await imageFromWorkspace(options.workspace, region.crop_path));
				if (region.selected?.render_path) {
					content.push(await imageFromWorkspace(options.workspace, region.selected.render_path));
				}
			}
			return { content, details: region };
		},
	});
	const submitTool = defineTool({
		name: "submit_reflection",
		label: "Submit 2D self-reflection",
		description:
			"Submit a structured diagnosis and suggested geometry changes. Every finding must reference a locked region_id.",
		parameters: SUBMIT_REFLECTION_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			if (options.state.report) throw new Error("A reflection report was already submitted");
			const regionIds = new Set(options.state.manifest.regions.map(({ region_id }) => region_id));
			for (const finding of params.findings) {
				if (!regionIds.has(finding.region_id)) {
					throw new Error(`Reflection finding references unknown region ${finding.region_id}`);
				}
			}
			options.state.report = params;
			return {
				content: [{ type: "text" as const, text: "Accepted the structured 2D reflection report." }],
				details: params,
			};
		},
	});
	return [inspectTool, submitTool];
}

export function createFixedGridReflectionTools(options: {
	workspace: string;
	state: ReflectionState;
	worker?: WorkerClient;
	controller?: RunController;
	outputDir?: string;
}) {
	options.state.phase ??= "global_inspection";
	const initialFindingKeys = options.state.initialFindingKeys ??= new Set<string>();
	const reviewedInitialFindingKeys = options.state.reviewedInitialFindingKeys ??= new Set<string>();
	const inspectedRegions = options.state.inspectedRegions ??= new Set<string>();
	const gridOverviews = options.state.gridOverviews ??= new Map<string, ReflectionGridOverviewRecord>();
	const crops = options.state.crops ??= new Map<string, ReflectionCropRecord>();
	const regionFindings = options.state.regionFindings ??= new Map<string, SavedRegionFindingsRecord>();
	const reflectionRoot = join(options.outputDir ?? "drawing_feedback", "reflection");
	options.state.inspectionManifestPath ??= join(reflectionRoot, "inspection_manifest.json");
	const reflectionOutputDir = join(reflectionRoot, "inspections");
	const inspectTool = defineTool({
		name: "inspect_region",
		label: "Inspect locked 4x4 region comparison",
		description:
			"Inspect the complete locked drawing/CAD comparison with matching fixed A1-D4 grids. Call this for every comparable region before submitting the one global findings draft. No camera, path, ROI, or crop parameter is accepted.",
		parameters: INSPECT_REGION_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			const region = options.state.manifest.regions.find(({ region_id }) => region_id === params.region_id);
			if (!region) throw new Error(`Unknown comparison region ${params.region_id}`);
			const content: Array<{ type: "text"; text: string } | ImageContent> = [
				{
					type: "text",
					text: `${region.region_id}: status=${region.status}, view_type=${region.view_type}, score=${String(region.selected?.score ?? "none")}, margin=${String(region.margin)}. ${alignmentNotice(region)} ${region.note}`,
				},
			];
			if (region.selected) {
				if (!options.worker) throw new InfrastructureError("Grid inspection requires a drawing worker");
				options.state.currentRegionId = region.region_id;
				let overview = gridOverviews.get(region.region_id);
				if (!overview) {
					const response = await options.worker.request<WorkerReflectionInspectionResponse>(
						options.workspace,
						{
							protocol_version: PROTOCOL_VERSION,
							operation: "compose_reflection_inspection",
							region_id: region.region_id,
							crop_path: region.crop_path,
							aligned_render_path: region.selected.aligned_render_path,
							mode: "grid_overview",
							output_dir: join(reflectionOutputDir, region.region_id, "grid-overview"),
						},
						{ timeoutMs: 180_000, ...(options.controller ? { signal: options.controller.signal } : {}) },
					);
					requireDrawingSuccess(response, "compose_reflection_inspection");
					if (response.region_id !== region.region_id || response.mode !== "grid_overview") {
						throw new InfrastructureError("Reflection worker returned an invalid grid overview");
					}
					const inspected = await inspectPngFile(options.workspace, response.context_path);
					if (inspected.sha256 !== response.sha256.context) {
						throw new InfrastructureError("Reflection grid overview hash mismatch");
					}
					overview = {
						region_id: region.region_id,
						context_path: response.context_path,
						pixel_width: response.width,
						pixel_height: response.height,
						sha256: inspected.sha256,
						created_at: isoNow(),
					};
					gridOverviews.set(region.region_id, overview);
				}
				content.push(await imageFromWorkspace(options.workspace, overview.context_path));
			} else {
				content.push(await imageFromWorkspace(options.workspace, region.crop_path));
			}
			inspectedRegions.add(region.region_id);
			await persistReflectionInspectionManifest(options.workspace, options.state);
			return { content, details: region };
		},
	});

	const submitInitialFindingsTool = defineTool({
		name: "submit_initial_findings",
		label: "Submit complete global findings draft",
		description:
			"After inspecting every comparable region, submit the complete findings draft exactly once. The controller assigns stable finding IDs and then unlocks fixed-grid verification.",
		parameters: SUBMIT_INITIAL_FINDINGS_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			if (options.state.phase !== "global_inspection" || options.state.initialFindingsSubmittedAt) {
				throw new Error("The complete initial findings draft was already submitted");
			}
			const comparableRegions = options.state.manifest.regions.filter((region) => region.selected);
			const comparableRegionIds = new Set(comparableRegions.map(({ region_id }) => region_id));
			const missingInspections = comparableRegions
				.filter(({ region_id }) => !inspectedRegions.has(region_id))
				.map(({ region_id }) => region_id);
			if (missingInspections.length > 0) {
				throw new Error(`Inspect every comparable region before submitting initial findings: ${missingInspections.join(", ")}`);
			}
			const grouped = new Map<string, Array<{ obs: string; suggested_change: string }>>();
			for (const finding of params.findings) {
				if (!comparableRegionIds.has(finding.region_id)) {
					throw new Error(`Unknown or non-comparable finding region ${finding.region_id}`);
				}
				const findings = grouped.get(finding.region_id) ?? [];
				if (findings.length >= 24) throw new Error(`Region ${finding.region_id} has more than 24 initial findings`);
				findings.push({ obs: finding.obs, suggested_change: finding.suggested_change });
				grouped.set(finding.region_id, findings);
			}
			const nextRegionFindings = new Map<string, SavedRegionFindingsRecord>();
			const nextInitialKeys = new Set<string>();
			const submittedAt = isoNow();
			for (const { region_id } of comparableRegions) {
				const record: SavedRegionFindingsRecord = {
					region_id,
					findings: [],
					next_finding_sequence: 1,
					saved_at: submittedAt,
				};
				for (const finding of grouped.get(region_id) ?? []) {
					const findingId = takeNextFindingId(record);
					record.findings.push({
						finding_id: findingId,
						obs: finding.obs,
						suggested_change: finding.suggested_change,
						evidence_crop_ids: [],
						created_from: "initial",
					});
					nextInitialKeys.add(findingKey(region_id, findingId));
				}
				nextRegionFindings.set(region_id, record);
			}
			const previousRegionFindings = new Map(regionFindings);
			const previousInitialKeys = new Set(initialFindingKeys);
			regionFindings.clear();
			for (const [regionId, record] of nextRegionFindings) regionFindings.set(regionId, record);
			initialFindingKeys.clear();
			for (const key of nextInitialKeys) initialFindingKeys.add(key);
			options.state.phase = "grid_verification";
			options.state.initialFindingsSubmittedAt = submittedAt;
			try {
				await persistReflectionInspectionManifest(options.workspace, options.state);
			} catch (error) {
				regionFindings.clear();
				for (const [regionId, record] of previousRegionFindings) regionFindings.set(regionId, record);
				initialFindingKeys.clear();
				for (const key of previousInitialKeys) initialFindingKeys.add(key);
				options.state.phase = "global_inspection";
				delete options.state.initialFindingsSubmittedAt;
				throw error;
			}
			const findings = comparableRegions.flatMap(({ region_id }) =>
				(regionFindings.get(region_id)?.findings ?? []).map((finding) => ({ region_id, ...finding })));
			return {
				content: [{
					type: "text" as const,
					text: `Accepted the complete global draft with ${String(findings.length)} finding(s). Fixed-grid verification is now active.`,
				}],
				details: { phase: options.state.phase, findings },
			};
		},
	});

	const compareTool = defineTool({
		name: "compare_region_grid_cells",
		label: "Compare two fixed grid cells",
		description:
			"After the complete initial draft, verify existing findings by choosing exactly one A1-D4 drawing cell and one A1-D4 CAD cell. Free ROI coordinates, paths, and cameras are not accepted.",
		parameters: COMPARE_REGION_GRID_CELLS_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			const region = options.state.manifest.regions.find(({ region_id }) => region_id === params.region_id);
			if (!region) throw new Error(`Unknown comparison region ${params.region_id}`);
			if (!region.selected) throw new Error(`Region ${params.region_id} has no locked aligned CAD candidate`);
			if (options.state.phase !== "grid_verification" || !options.state.initialFindingsSubmittedAt) {
				throw new Error("Submit the complete initial findings draft before comparing fixed grid cells");
			}
			if (options.state.currentRegionId !== region.region_id) {
				throw new Error(`Call inspect_region for ${region.region_id} before comparing its fixed grid cells`);
			}
			if (!inspectedRegions.has(region.region_id)) {
				throw new Error(`Call inspect_region for ${region.region_id} before comparing fixed grid cells`);
			}
			if (!options.worker) throw new InfrastructureError("Fixed-grid comparison requires a drawing worker");
			const recordForRegion = regionFindings.get(region.region_id);
			if (!recordForRegion) throw new InfrastructureError("Initial findings state is missing for this region");
			const findingIds = [...params.finding_ids].sort();
			for (const findingId of findingIds) {
				if (!recordForRegion.findings.some(({ finding_id }) => finding_id === findingId)) {
					throw new Error(`Unknown finding ${findingId} for region ${region.region_id}`);
				}
			}
			const referenceCell = params.reference_cell as GridCell;
			const cadCell = params.cad_cell as GridCell;
			const referenceRoi = gridCellRoi(referenceCell);
			const cadRoi = gridCellRoi(cadCell);
			const cropId = createHash("sha256")
				.update(JSON.stringify([
					options.state.manifest.version,
					region.region_id,
					region.selected.candidate_id,
					region.crop_sha256 ?? region.crop_path,
					region.selected.aligned_render_path,
					findingIds,
					referenceCell,
					cadCell,
				]))
				.digest("hex")
				.slice(0, 20);
			let record = crops.get(cropId);
			if (!record) {
				const response = await options.worker.request<WorkerReflectionInspectionResponse>(
					options.workspace,
					{
						protocol_version: PROTOCOL_VERSION,
						operation: "compose_reflection_inspection",
						region_id: region.region_id,
						crop_path: region.crop_path,
						aligned_render_path: region.selected.aligned_render_path,
						mode: "grid_comparison",
						reference_cell: referenceCell,
						cad_cell: cadCell,
						output_dir: join(reflectionOutputDir, region.region_id, cropId),
					},
					{ timeoutMs: 180_000, ...(options.controller ? { signal: options.controller.signal } : {}) },
				);
				requireDrawingSuccess(response, "compose_reflection_inspection");
				if (
					response.region_id !== region.region_id ||
					response.mode !== "grid_comparison" ||
					response.reference_cell !== referenceCell ||
					response.cad_cell !== cadCell ||
					!sameRoi(response.reference_roi, referenceRoi) ||
					!sameRoi(response.cad_roi, cadRoi)
				) throw new InfrastructureError("Reflection worker returned an invalid fixed-grid comparison");
				const responseHashes = response.sha256;
				const [referenceImage, cadImage, contextImage, comparisonImage] = await Promise.all([
					inspectPngFile(options.workspace, response.reference_crop_path),
					inspectPngFile(options.workspace, response.cad_crop_path),
					inspectPngFile(options.workspace, response.context_path),
					inspectPngFile(options.workspace, response.comparison_path),
				]);
				const hashes = {
					reference_crop: referenceImage.sha256,
					cad_crop: cadImage.sha256,
					context: contextImage.sha256,
					comparison: comparisonImage.sha256,
				};
				if (Object.entries(hashes).some(([key, value]) => responseHashes[key as keyof typeof hashes] !== value)) {
					throw new InfrastructureError("Reflection dual-ROI artifact hash mismatch");
				}
				record = {
					crop_id: cropId,
					region_id: region.region_id,
					finding_ids: findingIds,
					reference_cell: referenceCell,
					cad_cell: cadCell,
					reference_roi: referenceRoi,
					cad_roi: cadRoi,
					reference_pixel_roi: response.reference_pixel_roi,
					cad_pixel_roi: response.cad_pixel_roi,
					reference_crop_path: response.reference_crop_path,
					cad_crop_path: response.cad_crop_path,
					context_path: response.context_path,
					comparison_path: response.comparison_path,
					sha256: hashes,
					created_at: isoNow(),
				};
				crops.set(cropId, record);
				await persistReflectionInspectionManifest(options.workspace, options.state);
			}
			return {
				content: [
					{
						type: "text" as const,
						text: `${record.region_id}/${record.crop_id}: reviewing ${record.finding_ids.join(", ")}; original cell=${record.reference_cell}, CAD cell=${record.cad_cell}. ${alignmentNotice(region)} Use this local evidence only to retain, revise, add, or delete findings; use the complete grid view to judge full-profile continuity.`,
					},
					await imageFromWorkspace(options.workspace, record.comparison_path),
				],
				details: record,
			};
		},
	});

	const submitTool = defineTool({
		name: "submit_reflection",
		label: "Submit 2D self-reflection",
		description:
			"Submit only should_revise and overall_summary after the complete initial draft and fixed-grid review of every initial finding. The controller assembles the final findings.",
		parameters: SUBMIT_REFLECTION_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			if (options.state.report) throw new Error("A reflection report was already submitted");
			if (options.state.phase !== "grid_verification" || !options.state.initialFindingsSubmittedAt) {
				throw new Error("Submit the complete initial findings draft before the final reflection report");
			}
			const comparableRegions = options.state.manifest.regions.filter((region) => region.selected);
			const unreviewedInitialFindings = [...initialFindingKeys]
				.filter((key) => !reviewedInitialFindingKeys.has(key));
			if (unreviewedInitialFindings.length > 0) {
				throw new Error(`Verify every initial finding with fixed grid cells before submitting: ${unreviewedInitialFindings.join(", ")}`);
			}
			const report: DrawingReflectionReport = {
				should_revise: params.should_revise,
				overall_summary: params.overall_summary,
				findings: comparableRegions.flatMap(({ region_id }) =>
					(regionFindings.get(region_id)?.findings ?? []).map(({ obs, suggested_change }) => ({
						region_id,
						obs,
						suggested_change,
					}))),
			};
			options.state.report = report;
			return {
				content: [{ type: "text" as const, text: "Accepted the structured 2D reflection report." }],
				details: report,
			};
		},
	});

	const updateFindingsTool = defineTool({
		name: "update_findings",
		label: "Update findings from fixed-grid evidence",
		description:
			"After compare_region_grid_cells, atomically retain, add, locally update, or delete findings for that crop's region. Every successful call persists and returns the complete current region findings.",
		parameters: UPDATE_FINDINGS_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			const region = options.state.manifest.regions.find(({ region_id }) => region_id === params.region_id);
			if (!region) throw new Error(`Unknown comparison region ${params.region_id}`);
			if (!region.selected) throw new Error(`Region ${params.region_id} has no locked aligned CAD candidate`);
			if (options.state.phase !== "grid_verification" || !options.state.initialFindingsSubmittedAt) {
				throw new Error("Submit the complete initial findings draft before updating findings");
			}
			if (options.state.currentRegionId !== region.region_id || !inspectedRegions.has(region.region_id)) {
				throw new Error(`Call inspect_region for ${region.region_id} before updating findings`);
			}
			const crop = crops.get(params.crop_id);
			if (!crop || crop.region_id !== region.region_id) {
				throw new Error(`Unknown fixed-grid crop ${params.crop_id} for region ${region.region_id}`);
			}
			const existing = regionFindings.get(region.region_id);
			if (!existing) throw new InfrastructureError("Initial findings state is missing for this region");
			const record: SavedRegionFindingsRecord = {
				...existing,
				findings: existing.findings.map((finding) => ({
					...finding,
					evidence_crop_ids: [...finding.evidence_crop_ids],
				})),
			};
			const reviewedFindingIds = new Set(params.reviewed_finding_ids);
			for (const findingId of reviewedFindingIds) {
				if (!crop.finding_ids.includes(findingId)) {
					throw new Error(`Finding ${findingId} was not selected when grid crop ${crop.crop_id} was created`);
				}
				if (!record.findings.some(({ finding_id }) => finding_id === findingId)) {
					throw new Error(`Unknown finding ${findingId} for region ${params.region_id}`);
				}
			}
			for (const change of params.changes) {
				if (change.action === "add") {
					if (record.findings.length >= 24) {
						throw new Error(`Region ${params.region_id} already has 24 findings`);
					}
					record.findings.push({
						finding_id: takeNextFindingId(record),
						obs: change.obs,
						suggested_change: change.suggested_change,
						evidence_crop_ids: [crop.crop_id],
						created_from: "grid_crop",
					});
					continue;
				}
				if (!reviewedFindingIds.has(change.finding_id)) {
					throw new Error(`Finding ${change.finding_id} must be listed in reviewed_finding_ids before it can be changed`);
				}
				const index = record.findings.findIndex(({ finding_id }) => finding_id === change.finding_id);
				if (index < 0) {
					throw new Error(`Unknown finding ${change.finding_id} for region ${params.region_id}`);
				}
				if (change.action === "delete") {
					record.findings.splice(index, 1);
					continue;
				}
				const finding = record.findings[index]!;
				finding[change.field] = replaceFindingSubstring(
					finding[change.field],
					change.old_text,
					change.new_text,
					change.field,
				);
			}
			for (const findingId of reviewedFindingIds) {
				const finding = record.findings.find(({ finding_id }) => finding_id === findingId);
				if (finding && !finding.evidence_crop_ids.includes(crop.crop_id)) {
					finding.evidence_crop_ids.push(crop.crop_id);
				}
			}
			record.saved_at = isoNow();
			const previousReviewedKeys = new Set(reviewedInitialFindingKeys);
			for (const findingId of reviewedFindingIds) {
				const key = findingKey(region.region_id, findingId);
				if (initialFindingKeys.has(key)) reviewedInitialFindingKeys.add(key);
			}
			regionFindings.set(region.region_id, record);
			try {
				await persistReflectionInspectionManifest(options.workspace, options.state);
			} catch (error) {
				regionFindings.set(region.region_id, existing);
				reviewedInitialFindingKeys.clear();
				for (const key of previousReviewedKeys) reviewedInitialFindingKeys.add(key);
				throw error;
			}
			return {
				content: [{
					type: "text" as const,
					text: `Reviewed ${String(reviewedFindingIds.size)} finding(s) with grid crop ${crop.crop_id}; persisted ${String(params.changes.length)} change(s); ${String(record.findings.length)} finding(s) remain for ${record.region_id}.`,
				}],
				details: record,
			};
		},
	});

	return [
		inspectTool,
		submitInitialFindingsTool,
		compareTool,
		updateFindingsTool,
		submitTool,
	] as const;
}

async function makeRestrictedResourceLoader(
	workspace: string,
	settingsManager: SettingsManager,
	systemPrompt: string,
): Promise<DefaultResourceLoader> {
	const loader = new DefaultResourceLoader({
		cwd: workspace,
		agentDir: getAgentDir(),
		settingsManager,
		noExtensions: true,
		noSkills: true,
		noPromptTemplates: true,
		noThemes: true,
		noContextFiles: true,
		systemPrompt,
		appendSystemPrompt: [],
		systemPromptOverride: () => systemPrompt,
		appendSystemPromptOverride: () => [],
	});
	await loader.reload();
	return loader;
}

async function relativeSessionPath(
	workspace: string,
	sessionsDir: string,
	session: AgentSession,
): Promise<string | undefined> {
	const path = session.getSessionStats().sessionFile;
	if (!path) return undefined;
	const absolutePath = isAbsolute(path) ? path : resolve(workspace, path);
	const [workspaceRealPath, sessionsRealPath, sessionRealPath] = await Promise.all([
		realpath(workspace),
		realpath(sessionsDir),
		realpath(absolutePath),
	]);
	const workspaceRelative = relative(workspaceRealPath, sessionRealPath);
	const sessionsRelative = relative(sessionsRealPath, sessionRealPath);
	if (
		workspaceRelative === ".." ||
		workspaceRelative.startsWith("../") ||
		isAbsolute(workspaceRelative) ||
		sessionsRelative === ".." ||
		sessionsRelative.startsWith("../") ||
		isAbsolute(sessionsRelative)
	) {
		throw new InfrastructureError("Drawing subagent session escaped the fixture workspace");
	}
	return workspaceRelative;
}

export async function runRestrictedSession<T>(options: {
	role: SubagentRunUsage["role"];
	regionId?: string;
	workspace: string;
	outputDir: string;
	resolvedModel: ResolvedModel;
	thinkingLevel: AgentJob["thinking_level"];
	controller: RunController;
	systemPrompt: string;
	prompt: string;
	images: ImageContent[];
	toolNames: readonly string[];
	tools: readonly ToolDefinition[];
	getSubmitted: () => T | null;
	activeTimeoutMs: number;
	wallTimeoutMs: number;
	pauseActiveTimeoutForTools?: readonly string[];
	resultPath: string;
	onUsage: (usage: SubagentRunUsage) => Promise<void>;
	signal?: AbortSignal;
}): Promise<RestrictedSessionResult<T>> {
	// Reject malformed configuration before allocating an SDK session or making requests.
	const timeouts = restrictedSessionTimeouts(options.activeTimeoutMs, options.wallTimeoutMs);
	const started = Date.now();
	const startedAt = isoNow();
	const settingsManager = SettingsManager.inMemory({}, { projectTrusted: false });
	const resourceLoader = await makeRestrictedResourceLoader(
		options.workspace,
		settingsManager,
		options.systemPrompt,
	);
	const sessionsDir = join(options.workspace, options.outputDir, "sessions", options.role);
	await mkdir(sessionsDir, { recursive: true });
	const created = await createAgentSession({
		cwd: options.workspace,
		agentDir: getAgentDir(),
		modelRuntime: options.resolvedModel.runtime,
		model: options.resolvedModel.model,
		thinkingLevel: options.thinkingLevel as "off" | "minimal" | "low" | "medium" | "high",
		tools: [...options.toolNames],
		customTools: [...options.tools],
		resourceLoader,
		sessionManager: SessionManager.create(options.workspace, sessionsDir),
		settingsManager,
	});
	const session = created.session;
	const cancelled = () => options.controller.interrupted || options.signal?.aborted === true;
	const activeNames = session.getActiveToolNames();
	const expectedNames = new Set(options.toolNames);
	if (activeNames.length !== expectedNames.size || activeNames.some((name) => !expectedNames.has(name))) {
		session.dispose();
		throw new InfrastructureError(
			`${options.role} subagent tool restriction failed: ${activeNames.join(", ")}`,
		);
	}
	options.controller.add(session);
	const onCancelled = () => { void session.abort(); };
	options.signal?.addEventListener("abort", onCancelled, { once: true });
	let timeoutCause: "active" | "wall" | undefined;
	let status: SubagentRunUsage["status"] = "error";
	let failureMessage: string | undefined;
	let completedUsage: SubagentRunUsage | undefined;
	const timeoutMessage = (): string => timeoutCause === "active"
		? `${options.role} subagent exhausted ${String(timeouts.activeTimeoutMs / 60_000)} minutes of active Agent time`
		: `${options.role} subagent exceeded ${String(timeouts.wallTimeoutMs / 60_000)} minutes total wall time`;
	const collectUsage = async (): Promise<SubagentRunUsage> => {
		const totals = sessionUsageTotals(session);
		const sessionFile = await relativeSessionPath(options.workspace, sessionsDir, session);
		return {
			role: options.role,
			...(options.regionId ? { region_id: options.regionId } : {}),
			accounting_id: session.sessionId,
			...totals,
			...(sessionFile ? { session_file: sessionFile } : {}),
			...(status === "success" ? { result_path: options.resultPath } : {}),
			started_at: startedAt,
			finished_at: isoNow(),
			duration_s: Math.max(0, (Date.now() - started) / 1000),
			status,
			...(failureMessage ? { error: failureMessage } : {}),
		};
	};
	const expire = (cause: "active" | "wall"): void => {
		if (timeoutCause !== undefined || cancelled()) return;
		timeoutCause = cause;
		void session.abort();
	};
	const activeDeadline = new PausableDeadline(timeouts.activeTimeoutMs, () => expire("active"));
	const wallTimer = setTimeout(() => expire("wall"), timeouts.wallTimeoutMs);
	const pausedToolNames = new Set(options.pauseActiveTimeoutForTools ?? []);
	const activePausedToolCalls = new Set<string>();
	const unsubscribe = session.subscribe((event) => {
		if (event.type === "tool_execution_start" && pausedToolNames.has(event.toolName)) {
			const wasEmpty = activePausedToolCalls.size === 0;
			activePausedToolCalls.add(event.toolCallId);
			if (wasEmpty) activeDeadline.pause();
			return;
		}
		if (
			event.type === "tool_execution_end" &&
			pausedToolNames.has(event.toolName) &&
			activePausedToolCalls.delete(event.toolCallId) &&
			activePausedToolCalls.size === 0
		) {
			activeDeadline.resume();
		}
	});
	activeDeadline.start();
	try {
		if (cancelled()) throw new Error("interrupted");
		await session.prompt(options.prompt, { images: options.images, expandPromptTemplates: false });
		for (let submitReminder = 0; submitReminder < 2; submitReminder += 1) {
			if (options.getSubmitted() || timeoutCause !== undefined || cancelled()) break;
			await session.prompt(
				submitReminder === 0
					? "You have not submitted the required structured result. Use the submit tool now. Correct any rejected fields once; do not add prose-only output."
					: "Your required submit tool call was rejected or still did not produce a structured result. Read the latest tool error, correct the selected candidate and submitted fields so they are mutually consistent, and call the submit tool once more now. Do not search further and do not return prose-only output.",
				{ expandPromptTemplates: false },
			);
		}
		const value = options.getSubmitted();
		if (cancelled()) throw new Error("interrupted");
		// A submit tool may finish just before the active deadline while the model is
		// still producing a trailing response. The controller-owned structured value
		// is the completion boundary; a later abort must not discard it.
		if (value) {
			status = "success";
			completedUsage = await collectUsage();
			return {
				value,
				usage: completedUsage,
			};
		}
		if (timeoutCause !== undefined) throw new InfrastructureError(timeoutMessage());
		throw new InfrastructureError(`${options.role} subagent did not submit a valid structured result`);
	} catch (error) {
		const submitted = options.getSubmitted();
		if (submitted && !cancelled()) {
			status = "success";
			completedUsage = await collectUsage();
			return {
				value: submitted,
				usage: completedUsage,
			};
		}
		status = cancelled() ? "interrupted" : timeoutCause !== undefined ? "timeout" : "error";
		if (timeoutCause !== undefined && !cancelled()) {
			failureMessage = timeoutMessage();
			throw new InfrastructureError(failureMessage, { cause: error });
		}
		failureMessage = error instanceof Error ? error.message : String(error);
		throw error;
	} finally {
		activeDeadline.dispose();
		options.signal?.removeEventListener("abort", onCancelled);
		clearTimeout(wallTimer);
		unsubscribe();
		try {
			await options.onUsage(completedUsage ?? await collectUsage());
		} finally {
			options.controller.delete(session);
			session.dispose();
		}
	}
}

async function inputDigest(workspace: string, paths: string[]): Promise<string> {
	const hash = createHash("sha256");
	for (const path of paths) {
		hash.update(path);
		hash.update("\0");
		hash.update(await readFile(await requireWorkspaceFile(workspace, path)));
		hash.update("\0");
	}
	return hash.digest("hex");
}

export function regionManifestLooksCurrent(
	value: DrawingRegionManifest,
	inputSha256: string,
	pageCount?: number,
): boolean {
	if (
		!value ||
		value.version !== 1 ||
		value.input_sha256 !== inputSha256 ||
		!Array.isArray(value.regions) ||
		value.regions.length < 1 ||
		value.regions.length > 12
	) return false;
	try {
		const ids = new Set<string>();
		for (const region of value.regions) {
			if (
				!region ||
				typeof region.region_id !== "string" ||
				!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(region.region_id) ||
				ids.has(region.region_id) ||
				!Number.isInteger(region.page) ||
				region.page < 0 ||
				(pageCount !== undefined && region.page >= pageCount) ||
				typeof region.crop_path !== "string" ||
				region.crop_path.length === 0 ||
				!(["orthographic", "isometric", "section", "detail", "unknown"] as const).includes(region.view_type_hint) ||
				!(["high", "medium", "low"] as const).includes(region.confidence) ||
				typeof region.note !== "string" ||
				region.note.length > 1000 ||
				!Number.isInteger(region.width) ||
				region.width < 1 ||
				!Number.isInteger(region.height) ||
				region.height < 1 ||
				typeof region.crop_sha256 !== "string" ||
				!/^[a-f0-9]{64}$/.test(region.crop_sha256)
			) return false;
			ids.add(region.region_id);
			assertNormalizedRoi(region.roi);
		}
		return true;
	} catch {
		return false;
	}
}

async function loadRegionManifest(options: {
	workspace: string;
	manifestPath: string;
	cacheDir: string;
	inputSha256: string;
	pageCount: number;
}): Promise<DrawingRegionManifest | null> {
	const manifest = await readJson<DrawingRegionManifest>(
		safeWorkspacePath(options.workspace, options.manifestPath),
	).catch(() => null);
	if (!manifest || !regionManifestLooksCurrent(manifest, options.inputSha256, options.pageCount)) {
		return null;
	}
	try {
		for (const region of manifest.regions) {
			const expectedPath = join(options.cacheDir, "regions", region.region_id, "crop.png");
			if (region.crop_path !== expectedPath) return null;
			const inspected = await inspectPngFile(options.workspace, region.crop_path);
			if (
				inspected.width !== region.width ||
				inspected.height !== region.height ||
				inspected.sha256 !== region.crop_sha256
			) return null;
		}
		return manifest;
	} catch {
		return null;
	}
}

async function prepareSegmentationImages(options: {
	worker: WorkerClient;
	workspace: string;
	stagedPaths: string[];
	controller: RunController;
}): Promise<ImageContent[]> {
	await ensureWorkspaceDirectory(options.workspace, "prepared_inputs");
	for (const index of options.stagedPaths.keys()) {
		const expectedPath = safeWorkspacePath(
			options.workspace,
			join("prepared_inputs", `${String(index).padStart(2, "0")}.png`),
		);
		const existing = await lstat(expectedPath).catch(() => null);
		if (existing && (!existing.isFile() || existing.isSymbolicLink() || existing.nlink !== 1)) {
			throw new InfrastructureError(`Prepared drawing output path is unsafe at index ${index}`);
		}
	}
	const response = await options.worker.request<WorkerPrepareResponse>(
		options.workspace,
		{
			protocol_version: PROTOCOL_VERSION,
			operation: "prepare_images",
			paths: options.stagedPaths,
			max_edge: 2000,
		},
		{ signal: options.controller.signal },
	);
	requireOperation(response, "prepare_images");
	if (!Array.isArray(response.images) || response.images.length !== options.stagedPaths.length) {
		throw new InfrastructureError("Drawing image preparation returned the wrong number of images");
	}
	const seenPaths = new Set<string>();
	const prepared: ImageContent[] = [];
	for (const [index, image] of response.images.entries()) {
		const expectedSource = options.stagedPaths[index];
		const expectedPath = join("prepared_inputs", `${String(index).padStart(2, "0")}.png`);
		if (
			image.source_path !== expectedSource ||
			image.path !== expectedPath ||
			image.mime_type !== "image/png" ||
			seenPaths.has(image.path) ||
			!Number.isInteger(image.original_width) ||
			image.original_width < 1 ||
			!Number.isInteger(image.original_height) ||
			image.original_height < 1 ||
			!Number.isInteger(image.width) ||
			image.width < 1 ||
			!Number.isInteger(image.height) ||
			image.height < 1 ||
			Math.max(image.width, image.height) > 2000
		) {
			throw new InfrastructureError(`Drawing image preparation returned invalid metadata at index ${index}`);
		}
		seenPaths.add(image.path);
		const bytes = await readFile(await requireWorkspaceFile(options.workspace, image.path));
		const inspected = inspectPngBytes(bytes, image.path);
		if (inspected.width !== image.width || inspected.height !== image.height) {
			throw new InfrastructureError(`Prepared drawing PNG dimensions changed at index ${index}`);
		}
		prepared.push({ type: "image", data: bytes.toString("base64"), mimeType: "image/png" });
	}
	return prepared;
}

interface CurrentCadReference {
	renderPath: string;
	viewSpec: DrawingViewSpec;
	image: ImageContent;
}

async function prepareCurrentCadReference(options: {
	worker: WorkerClient;
	workspace: string;
	outputDir: string;
	stepSha256: string;
	controller: RunController;
}): Promise<CurrentCadReference> {
	const response = await options.worker.request<WorkerRenderDrawingViewResponse>(
		options.workspace,
		{
			protocol_version: PROTOCOL_VERSION,
			operation: "render_drawing_view",
			step_path: "output.step",
			region_id: "current-cad-reference",
			view_spec: CURRENT_CAD_REFERENCE_VIEW_SPEC,
			show_hidden_lines: true,
			output_dir: join(options.outputDir, "current-cad-reference"),
		},
		{ timeoutMs: 180_000, signal: options.controller.signal },
	);
	requireDrawingSuccess(response, "render_drawing_view");
	if (
		response.region_id !== "current-cad-reference" ||
		response.step_sha256 !== options.stepSha256 ||
		!sameViewSpec(response.view_spec, CURRENT_CAD_REFERENCE_VIEW_SPEC)
	) {
		throw new InfrastructureError("Current CAD reference render did not use the locked STEP and camera");
	}
	await requireWorkspaceFile(options.workspace, response.render_path);
	await writeJsonAtomic(
		safeWorkspacePath(options.workspace, join(options.outputDir, "current-cad-reference", "metadata.json")),
		{
			version: 1,
			purpose: "shape_reference_only",
			not_a_ranked_candidate: true,
			step_sha256: options.stepSha256,
			render_path: response.render_path,
			view_spec: response.view_spec,
		},
	);
	return {
		renderPath: response.render_path,
		viewSpec: response.view_spec,
		image: await imageFromWorkspace(options.workspace, response.render_path),
	};
}

export async function runDrawingSegmentation(options: {
	job: AgentJob;
	worker: WorkerClient;
	workspace: string;
	resolvedModel: ResolvedModel;
	controller: RunController;
	onUsage: (usage: SubagentRunUsage) => Promise<void>;
}): Promise<DrawingSegmentationResult> {
	const inputSha256 = await inputDigest(options.workspace, options.job.staged_paths);
	const cacheDir = join("drawing_feedback", "input_cache", inputSha256);
	const manifestPath = join(cacheDir, "region_manifest.json");
	const cached = await loadRegionManifest({
		workspace: options.workspace,
		manifestPath,
		cacheDir,
		inputSha256,
		pageCount: options.job.staged_paths.length,
	});
	if (cached) {
		const finalInputSha256 = await inputDigest(options.workspace, options.job.staged_paths);
		if (finalInputSha256 !== inputSha256) {
			throw new InfrastructureError("Drawing inputs changed while loading the segmentation cache");
		}
		return { manifest: cached, cached: true, manifestPath, inputSha256 };
	}

	const state = makeSegmentationState();
	const preparedImages = await prepareSegmentationImages({
		worker: options.worker,
		workspace: options.workspace,
		stagedPaths: options.job.staged_paths,
		controller: options.controller,
	});
	const workDir = join(cacheDir, "work");
	const canonicalRegionsDir = join(cacheDir, "regions");
	await ensureWorkspaceDirectory(options.workspace, workDir);
	const segmentation = await runRestrictedSession({
		role: "segmentation",
		workspace: options.workspace,
		outputDir: cacheDir,
		resolvedModel: options.resolvedModel,
		thinkingLevel: options.job.thinking_level,
		controller: options.controller,
		systemPrompt: SEGMENTATION_SYSTEM_PROMPT,
		prompt: drawingSegmentationPrompt({
			fixture: options.job.fixture,
			pagePaths: options.job.staged_paths,
		}),
		images: preparedImages,
		toolNames: SEGMENTATION_AGENT_TOOLS,
		tools: createDrawingSegmentationTools({
			worker: options.worker,
			workspace: options.workspace,
			pagePaths: options.job.staged_paths,
			workDir,
			canonicalRegionsDir,
			manifestPath,
			inputSha256,
			controller: options.controller,
			state,
		}),
		getSubmitted: () => state.submitted,
		activeTimeoutMs: 600_000,
		wallTimeoutMs: 1_200_000,
		pauseActiveTimeoutForTools: ["crop_region"],
		resultPath: manifestPath,
		onUsage: options.onUsage,
	});
	await writeJsonAtomic(safeWorkspacePath(options.workspace, manifestPath), segmentation.value);
	const finalInputSha256 = await inputDigest(options.workspace, options.job.staged_paths);
	if (finalInputSha256 !== inputSha256) {
		throw new InfrastructureError("Drawing inputs changed during segmentation");
	}
	const published = await loadRegionManifest({
		workspace: options.workspace,
		manifestPath,
		cacheDir,
		inputSha256,
		pageCount: options.job.staged_paths.length,
	});
	if (!published) throw new InfrastructureError("Published drawing region manifest failed verification");
	return {
		manifest: published,
		usage: segmentation.usage,
		cached: false,
		manifestPath,
		inputSha256,
	};
}

function feedbackCacheKey(options: {
	inputSha256: string;
	stepSha256: string;
	provider: string;
	model: string;
}): string {
	return createHash("sha256")
		.update(
			[
				DRAWING_FEEDBACK_VERSION,
				options.inputSha256,
				options.stepSha256,
				options.provider,
				options.model,
			].join("\0"),
		)
		.digest("hex");
}

export function manifestLooksCurrent(
	value: DrawingViewManifest,
	inputSha256: string,
	sourceSha256?: string,
	stepSha256?: string,
	pageCount?: number,
	allowErrorRegions = false,
): boolean {
	if (
		!value ||
		value.version !== 3 ||
		value.scorer_profile !== "capacity_transport_v1" ||
		value.input_sha256 !== inputSha256 ||
		(sourceSha256 !== undefined && value.source_sha256 !== sourceSha256) ||
		(stepSha256 !== undefined && value.step_sha256 !== stepSha256) ||
		!Array.isArray(value.regions) ||
		value.regions.length === 0 ||
		value.regions.length > 12
	) return false;
	if (!allowErrorRegions && value.regions.some(({ status }) => status === "error")) return false;
	try {
		const ids = new Set<string>();
		for (const region of value.regions) {
			if (
				!region ||
				typeof region.region_id !== "string" ||
				!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(region.region_id) ||
				ids.has(region.region_id)
			) return false;
			ids.add(region.region_id);
			assertNormalizedRoi(region.roi);
			if (
				!Number.isInteger(region.page) ||
				region.page < 0 ||
				(pageCount !== undefined && region.page >= pageCount) ||
				typeof region.crop_path !== "string" ||
				region.crop_path.length === 0 ||
				!(["orthographic", "isometric", "section", "detail", "unknown"] as const).includes(region.view_type) ||
				!(["matched", "ambiguous", "error"] as const).includes(region.status) ||
				!(["high", "medium", "low"] as const).includes(region.confidence) ||
				typeof region.note !== "string" ||
				region.note.length > 1000 ||
				!Array.isArray(region.alternatives) ||
				(region.margin !== null && (!Number.isFinite(region.margin) || region.margin < 0 || region.margin > 1))
			) return false;
			const supported = region.status === "matched" || region.status === "ambiguous";
			if (supported !== (region.selected !== undefined)) return false;
			if ((supported && region.alternatives.length !== 1) || (!supported && region.alternatives.length !== 0)) {
				return false;
			}
			const candidates = region.selected ? [region.selected, ...region.alternatives] : [];
			const candidateIds = new Set<string>();
			const candidateNames = new Set<string>();
			for (const candidate of candidates) {
				if (
					!candidate ||
					typeof candidate.candidate_id !== "string" ||
					!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(candidate.candidate_id) ||
					candidateIds.has(candidate.candidate_id) ||
					!Number.isFinite(candidate.score) ||
					candidate.score < 0 ||
					candidate.score > 1 ||
					!Number.isFinite(candidate.raw_score) ||
					typeof candidate.eligible !== "boolean" ||
					!candidate.eligible ||
					!Array.isArray(candidate.failed_gates) ||
					candidate.scoring_mode !== "capacity_transport" ||
					!candidate.capacity_transport ||
					typeof candidate.render_path !== "string" ||
					candidate.render_path.length === 0 ||
					typeof candidate.comparison_path !== "string" ||
					candidate.comparison_path.length === 0 ||
					typeof candidate.aligned_render_path !== "string" ||
					candidate.aligned_render_path.length === 0 ||
					!candidate.transform ||
					![
						candidate.transform.scale_px_per_unit,
						candidate.transform.rotation_deg,
						candidate.transform.center_x_px,
						candidate.transform.center_y_px,
						candidate.transform.model_center_x,
						candidate.transform.model_center_y,
					].every(Number.isFinite) ||
					typeof candidate.transform.reflected !== "boolean" ||
					!candidate.alignment ||
					!Number.isFinite(candidate.alignment.objective) ||
					candidate.alignment.mode !== "capacity_transport" ||
					typeof candidate.alignment.scale_at_bound !== "boolean" ||
					typeof candidate.alignment.rotation_at_limit !== "boolean" ||
					typeof candidate.alignment.translation_at_limit !== "boolean"
				) return false;
				assertSupportedViewSpec(candidate.view_spec);
				if (candidateNames.has(candidate.view_spec.name)) return false;
				candidateIds.add(candidate.candidate_id);
				candidateNames.add(candidate.view_spec.name);
			}
			assertRegionViewSemantics(
				region.view_type,
				region.status,
				candidates.map(({ view_spec }) => view_spec),
			);
		}
		return true;
	} catch {
		return false;
	}
}

/** A searched manifest is not a completed search while any region is an error. */
export function assertCompleteDrawingViewSearch(
	manifest: DrawingViewManifest,
	inputSha256: string,
	sourceSha256: string,
	stepSha256: string,
	pageCount?: number,
): void {
	if (!manifestLooksCurrent(manifest, inputSha256, sourceSha256, stepSha256, pageCount, true)) {
		throw new InfrastructureError("Automatic View Search manifest is invalid or does not match its source CAD");
	}
	const failed = manifest.regions.filter(region => region.status === "error");
	if (failed.length) {
		const detail = failed.map(region => `${region.region_id}: ${region.note}`).join("; ");
		throw new InfrastructureError(
			`Automatic View Search is incomplete for ${failed.length} region(s): ${detail}. Resume search to retry failed regions; successful regions remain cached.`,
		);
	}
}

async function readableManifestArtifacts(workspace: string, manifest: DrawingViewManifest): Promise<boolean> {
	try {
		for (const region of manifest.regions) await requireWorkspaceFile(workspace, region.crop_path);
		return true;
	} catch {
		return false;
	}
}

async function readableLockedManifestArtifacts(
	workspace: string,
	manifest: DrawingViewManifest,
): Promise<boolean> {
	try {
		for (const region of manifest.regions) {
			await requireWorkspaceFile(workspace, region.crop_path);
			for (const candidate of region.selected ? [region.selected, ...region.alternatives] : []) {
				await requireWorkspaceFile(workspace, candidate.render_path);
				await requireWorkspaceFile(workspace, candidate.comparison_path);
				await requireWorkspaceFile(workspace, candidate.aligned_render_path);
			}
		}
		return true;
	} catch {
		return false;
	}
}

async function loadPartialManifest(
	workspace: string,
	path: string,
	inputSha256: string,
	sourceSha256: string,
	stepSha256: string,
	pageCount: number,
): Promise<DrawingViewManifest | null> {
	const manifest = await readJson<DrawingViewManifest>(safeWorkspacePath(workspace, path)).catch(() => null);
	if (!manifest || !manifestLooksCurrent(manifest, inputSha256, sourceSha256, stepSha256, pageCount)) {
		return null;
	}
	return (await readableLockedManifestArtifacts(workspace, manifest)) ? manifest : null;
}

function completedSearchUsages(result: AgentResult, resultPaths: Set<string>): SubagentRunUsage[] {
	const latestByPath = new Map<string, SubagentRunUsage>();
	for (const run of result.subagent_runs ?? []) {
		if (
			run.role === "view_search" &&
			run.status === "success" &&
			run.result_path &&
			resultPaths.has(run.result_path)
		) latestByPath.set(run.result_path, run);
	}
	return [...latestByPath.values()];
}

function regionResultPath(outputDir: string, regionId: string): string {
	return join(outputDir, "view-search", regionId, "result.json");
}

async function loadRegionViewResult(options: {
	workspace: string;
	path: string;
	region: DrawingRegionManifestRegion;
	inputSha256: string;
	sourceSha256: string;
	stepSha256: string;
	pageCount: number;
}): Promise<DrawingManifestRegion | null> {
	const region = await readJson<DrawingManifestRegion>(safeWorkspacePath(options.workspace, options.path)).catch(() => null);
	if (!region || region.status === "error") return null;
	const manifest: DrawingViewManifest = {
		version: 3,
		scorer_profile: "capacity_transport_v1",
		input_sha256: options.inputSha256,
		source_sha256: options.sourceSha256,
		step_sha256: options.stepSha256,
		regions: [region],
	};
	if (
		!manifestLooksCurrent(
			manifest,
			options.inputSha256,
			options.sourceSha256,
			options.stepSha256,
			options.pageCount,
		) ||
		region.region_id !== options.region.region_id ||
		region.page !== options.region.page ||
		!sameRoi(region.roi, options.region.roi) ||
		region.crop_path !== options.region.crop_path ||
		region.width !== options.region.width ||
		region.height !== options.region.height ||
		region.crop_sha256 !== options.region.crop_sha256
	) return null;
	return (await readableLockedManifestArtifacts(options.workspace, manifest)) ? region : null;
}

function failedRegionResult(region: DrawingRegionManifestRegion, error: unknown): DrawingManifestRegion {
	const message = error instanceof Error ? error.message : String(error);
	return {
		region_id: region.region_id,
		page: region.page,
		roi: region.roi,
		crop_path: region.crop_path,
		view_type: region.view_type_hint,
		status: "error",
		alternatives: [],
		margin: null,
		confidence: "low",
		note: `Independent view search failed: ${message}`.slice(0, 1000),
		width: region.width,
		height: region.height,
		crop_sha256: region.crop_sha256,
	};
}

async function runDrawingViewSearchStage(options: {
	job: AgentJob;
	worker: WorkerClient;
	workspace: string;
	resolvedModel: ResolvedModel;
	controller: RunController;
	getResult: () => AgentResult;
	onUsage: (usage: SubagentRunUsage) => Promise<void>;
	segmentation: DrawingSegmentationResult;
	sourceSha256: string;
	stepSha256: string;
	outputDir: string;
}): Promise<DrawingViewSearchStageResult> {
	const regionManifest = options.segmentation.manifest;
	const inputSha256 = options.segmentation.inputSha256;
	const manifestPath = join(options.outputDir, "view_manifest.json");
	const resultPaths = new Set(
		regionManifest.regions.map(({ region_id }) => regionResultPath(options.outputDir, region_id)),
	);
	const cachedRegions = await Promise.all(
		regionManifest.regions.map(async (region) => await loadRegionViewResult({
			workspace: options.workspace,
			path: regionResultPath(options.outputDir, region.region_id),
			region,
			inputSha256,
			sourceSha256: options.sourceSha256,
			stepSha256: options.stepSha256,
			pageCount: options.job.staged_paths.length,
		})),
	);
	const needsSearch = cachedRegions.some((region) => region === null);
	const preparedPages = needsSearch
		? await prepareSegmentationImages({
				worker: options.worker,
				workspace: options.workspace,
				stagedPaths: options.job.staged_paths,
				controller: options.controller,
			})
		: [];
	const currentCadReference = needsSearch
		? await prepareCurrentCadReference({
				worker: options.worker,
				workspace: options.workspace,
				outputDir: options.outputDir,
				stepSha256: options.stepSha256,
				controller: options.controller,
			})
		: null;
	const usageByRegion = new Map<string, SubagentRunUsage>();
	for (const usage of completedSearchUsages(options.getResult(), resultPaths)) {
		if (usage.region_id) usageByRegion.set(usage.region_id, usage);
	}
	let usageWrite = Promise.resolve();
	const persistUsage = async (usage: SubagentRunUsage): Promise<void> => {
		if (usage.region_id) usageByRegion.set(usage.region_id, usage);
		usageWrite = usageWrite.catch(() => undefined).then(async () => await options.onUsage(usage));
		await usageWrite;
	};
	const settled = await Promise.allSettled(
		regionManifest.regions.map(async (region, index): Promise<DrawingManifestRegion> => {
			const cached = cachedRegions[index];
			if (cached) return cached;
			const sourcePage = preparedPages[region.page];
			const sourcePagePath = options.job.staged_paths[region.page];
			if (!sourcePage || !sourcePagePath || !currentCadReference) {
				throw new InfrastructureError(`Missing locked page or CAD reference for ${region.region_id}`);
			}
			const agentOutputDir = join(options.outputDir, "view-search", region.region_id);
			const resultPath = regionResultPath(options.outputDir, region.region_id);
			const searchState = makeSearchState(region);
			const search = await runRestrictedSession({
				role: "view_search",
				regionId: region.region_id,
				workspace: options.workspace,
				outputDir: agentOutputDir,
				resolvedModel: options.resolvedModel,
				thinkingLevel: options.job.thinking_level,
				controller: options.controller,
				systemPrompt: VIEW_SEARCH_SYSTEM_PROMPT,
				prompt: viewSearchPrompt({
					fixture: options.job.fixture,
					region,
					pagePath: sourcePagePath,
					cadReference: currentCadReference,
				}),
				images: [
					await imageFromWorkspace(options.workspace, region.crop_path),
					sourcePage,
					currentCadReference.image,
				],
				toolNames: VIEW_SEARCH_AGENT_TOOLS,
				tools: createViewSearchTools({
					worker: options.worker,
					workspace: options.workspace,
					region,
					outputDir: agentOutputDir,
					expectedStepSha256: options.stepSha256,
					controller: options.controller,
					state: searchState,
				}),
				getSubmitted: () => searchState.submitted,
				activeTimeoutMs: 600_000,
				wallTimeoutMs: 7_200_000,
				pauseActiveTimeoutForTools: ["render_candidate_view", "rank_view_candidates"],
				resultPath,
				onUsage: persistUsage,
			});
			await writeJsonAtomic(safeWorkspacePath(options.workspace, resultPath), search.value);
			return search.value;
		}),
	);
	await usageWrite;
	if (options.controller.interrupted) throw new Error("interrupted");
	const regions = settled.map((outcome, index) =>
		outcome.status === "fulfilled"
			? outcome.value
			: failedRegionResult(regionManifest.regions[index]!, outcome.reason),
	);
	const manifest: DrawingViewManifest = {
		version: 3,
		scorer_profile: "capacity_transport_v1",
		input_sha256: inputSha256,
		source_sha256: options.sourceSha256,
		step_sha256: options.stepSha256,
		regions,
	};
	const searchMode: DrawingViewSearchStageResult["searchMode"] = needsSearch ? "searched" : "reused";
	const searchUsage = regionManifest.regions.flatMap(({ region_id }) => {
		const usage = usageByRegion.get(region_id);
		return usage ? [usage] : [];
	});
	await writeJsonAtomic(join(options.workspace, manifestPath), manifest);
	return { manifest, manifestPath, searchMode, searchUsage };
}

export async function runDrawingViewSearchOnly(options: {
	job: AgentJob;
	worker: WorkerClient;
	workspace: string;
	resolvedModel: ResolvedModel;
	controller: RunController;
	getResult: () => AgentResult;
	onUsage: (usage: SubagentRunUsage) => Promise<void>;
}): Promise<DrawingViewSearchOnlyResult> {
	const latest = await options.worker.request<WorkerInspectResponse>(
		options.workspace,
		{ protocol_version: PROTOCOL_VERSION, operation: "inspect" },
		{ signal: options.controller.signal },
	);
	const inspection = await inspectFormalCandidate(options.workspace, latest.artifact, options.getResult());
	if (
		!inspection.formal_execution_current ||
		!inspection.source_sha256 ||
		!latest.artifact.sha256 ||
		!latest.artifact.path
	) {
		throw new Error(
			`Drawing view search requires a current formal execute_build123d result. ${inspection.reason}`,
		);
	}
	const sourceSha256 = inspection.source_sha256;
	const stepSha256 = latest.artifact.sha256;
	const segmentation = await runDrawingSegmentation({
		job: options.job,
		worker: options.worker,
		workspace: options.workspace,
		resolvedModel: options.resolvedModel,
		controller: options.controller,
		onUsage: options.onUsage,
	});
	const cacheKey = feedbackCacheKey({
		inputSha256: segmentation.inputSha256,
		stepSha256,
		provider: options.job.provider,
		model: options.job.model,
	});
	const outputDir = join("drawing_feedback", "runs", cacheKey);
	await mkdir(join(options.workspace, outputDir), { recursive: true });
	const stage = await runDrawingViewSearchStage({
		...options,
		segmentation,
		sourceSha256,
		stepSha256,
		outputDir,
	});
	const after = await options.worker.request<WorkerInspectResponse>(
		options.workspace,
		{ protocol_version: PROTOCOL_VERSION, operation: "inspect" },
		{ signal: options.controller.signal },
	);
	const verified = await inspectFormalCandidate(options.workspace, after.artifact, options.getResult());
	if (
		!verified.formal_execution_current ||
		verified.source_sha256 !== sourceSha256 ||
		after.artifact.sha256 !== stepSha256
	) {
		throw new InfrastructureError("Formal source or STEP changed during drawing view search");
	}
	assertCompleteDrawingViewSearch(
		stage.manifest,
		segmentation.inputSha256,
		sourceSha256,
		stepSha256,
		options.job.staged_paths.length,
	);
	return {
		fixture_id: options.job.fixture.id,
		cache_key: cacheKey,
		input_sha256: segmentation.inputSha256,
		source_sha256: sourceSha256,
		step_sha256: stepSha256,
		segmentation_mode: segmentation.cached ? "reused" : "segmented",
		search_mode: stage.searchMode,
		region_manifest_path: segmentation.manifestPath,
		manifest_path: stage.manifestPath,
		manifest: stage.manifest,
		usage: [
			...(segmentation.usage ? [segmentation.usage] : []),
			...stage.searchUsage,
		],
	};
}

async function loadExactCache(
	workspace: string,
	path: string,
	inputSha256: string,
	sourceSha256: string,
	stepSha256: string,
	pageCount: number,
): Promise<CacheEnvelope | null> {
	const cached = await readJson<CacheEnvelope>(safeWorkspacePath(workspace, path)).catch(() => null);
	if (
		!cached ||
		cached.version !== DRAWING_FEEDBACK_VERSION ||
		!manifestLooksCurrent(cached.manifest, inputSha256, sourceSha256, stepSha256, pageCount) ||
		cached.summary.source_sha256 !== sourceSha256 ||
		cached.summary.step_sha256 !== stepSha256 ||
		cached.summary.input_sha256 !== inputSha256
	) return null;
	try {
		await Promise.all([
			requireWorkspaceFile(workspace, cached.summary.manifest_path),
			requireWorkspaceFile(workspace, cached.summary.report_path),
			requireWorkspaceFile(workspace, cached.summary.sheet_path),
			...(cached.summary.inspection_manifest_path
				? [requireWorkspaceFile(workspace, cached.summary.inspection_manifest_path)]
				: []),
		]);
	} catch {
		return null;
	}
	return cached;
}

function viewManifestMatchesRegionManifest(
	viewManifest: DrawingViewManifest,
	regionManifest: DrawingRegionManifest,
): boolean {
	const lockedById = new Map(regionManifest.regions.map((region) => [region.region_id, region]));
	if (viewManifest.regions.length !== lockedById.size) return false;
	return viewManifest.regions.every((region) => {
		const locked = lockedById.get(region.region_id);
		return Boolean(
			locked &&
			region.page === locked.page &&
			sameRoi(region.roi, locked.roi) &&
			region.crop_path === locked.crop_path &&
			region.width === locked.width &&
			region.height === locked.height &&
			region.crop_sha256 === locked.crop_sha256,
		);
	});
}

async function makeComparisonSheet(options: {
	worker: WorkerClient;
	workspace: string;
	outputDir: string;
	manifest: DrawingViewManifest;
	controller: RunController;
}): Promise<string> {
	const regions: WorkerComparisonSheetRegion[] = options.manifest.regions.map((region) => ({
		region_id: region.region_id,
		crop_path: region.crop_path,
		...(region.selected
			? {
					render_path: region.selected.render_path,
					...(region.selected.comparison_path
						? { comparison_path: region.selected.comparison_path }
						: {}),
					view_spec: region.selected.view_spec,
					score: region.selected.score,
					margin: region.margin,
				}
			: {}),
		status: region.status,
	}));
	const response = await options.worker.request<WorkerCompareDrawingSheetResponse>(
		options.workspace,
		{
			protocol_version: PROTOCOL_VERSION,
			operation: "compare_drawing_sheet",
			regions,
			output_dir: join(options.outputDir, "sheet"),
		},
		{ timeoutMs: 180_000, signal: options.controller.signal },
	);
	requireDrawingSuccess(response, "compare_drawing_sheet");
	await requireWorkspaceFile(options.workspace, response.sheet_path);
	return response.sheet_path;
}

function formatReflectionFeedback(
	summary: DrawingFeedbackSummary,
	report: DrawingReflectionReport,
	cached: boolean,
): string {
	return [
		`2D drawing self-reflection ${cached ? "loaded from cache" : "completed"}.`,
		`Regions: total=${summary.region_count}, matched=${summary.matched_count}, ambiguous=${summary.ambiguous_count}, error=${summary.error_count}. Segmentation=${summary.segmentation_mode ?? "reused"}; view search=${summary.search_mode}.`,
		`should_revise=${report.should_revise}.`,
		`Summary: ${report.overall_summary}`,
		report.findings.length > 0
			? `Findings:\n${report.findings
					.map((finding) => `- ${finding.region_id}: ${finding.obs} Suggested change: ${finding.suggested_change}`)
					.join("\n")}`
			: "Findings: none.",
	].join("\n\n");
}

function summaryCounts(regions: DrawingManifestRegion[]): Record<DrawingRegionStatus, number> {
	const counts: Record<DrawingRegionStatus, number> = {
		matched: 0,
		ambiguous: 0,
		error: 0,
	};
	for (const region of regions) counts[region.status] += 1;
	return counts;
}

export function createDrawingReflectionTool(options: {
	job: AgentJob;
	worker: WorkerClient;
	workspace: string;
	resolvedModel: ResolvedModel;
	controller: RunController;
	lockedViewManifestPath?: string;
	getResult: () => AgentResult;
	onUsage: (usage: SubagentRunUsage) => Promise<void>;
	onComplete: (summary: DrawingFeedbackSummary) => Promise<void>;
}) {
	return defineTool({
		name: "self_reflect_2d_drawing",
		label: "Self-reflect against 2D drawing",
		description:
			"Compare the current formally executed STEP with all merged 2D drawings. Internally persists drawing-view crops with a segmentation subagent, searches and locks cameras with a separate subagent, then returns source-level diagnostic feedback. Takes no arguments.",
		promptSnippet: "Search drawing views and self-reflect on the current formal STEP without camera arguments",
		promptGuidelines: [
			"Call this no-argument tool only after execute_build123d has produced the current output.step.",
			"If its report suggests source changes, edit solution.py and formally execute it again.",
		],
		parameters: SELF_REFLECT_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, _params, signal) {
			const executionSignal = effectiveSignal(signal, options.controller);
			if (executionSignal.aborted) throw new Error("interrupted");
			const currentResult = options.getResult();
			const latest = await options.worker.request<WorkerInspectResponse>(
				options.workspace,
				{ protocol_version: PROTOCOL_VERSION, operation: "inspect" },
				{ signal: executionSignal },
			);
			const inspection = await inspectFormalCandidate(options.workspace, latest.artifact, currentResult);
			if (
				!inspection.formal_execution_current ||
				!inspection.source_sha256 ||
				!latest.artifact.sha256 ||
				!latest.artifact.path
			) {
				throw new Error(
					`2D self-reflection requires a current formal execute_build123d result. ${inspection.reason}`,
				);
			}
			const sourceSha256 = inspection.source_sha256;
			const stepSha256 = latest.artifact.sha256;
			const segmentation = await runDrawingSegmentation({
				job: options.job,
				worker: options.worker,
				workspace: options.workspace,
				resolvedModel: options.resolvedModel,
				controller: options.controller,
				onUsage: options.onUsage,
			});
			const regionManifest = segmentation.manifest;
			const inputSha256 = segmentation.inputSha256;
			const cacheKey = feedbackCacheKey({
				inputSha256,
				stepSha256,
				provider: options.job.provider,
				model: options.job.model,
			});
			const rootDir = "drawing_feedback";
			const outputDir = join(rootDir, "runs", cacheKey);
			const exactCachePath = join(outputDir, "run.json");
			const manifestPath = join(outputDir, "view_manifest.json");
			const reportPath = join(outputDir, "reflection.json");
			await mkdir(join(options.workspace, outputDir), { recursive: true });

			const exactCandidate = await loadExactCache(
				options.workspace,
				exactCachePath,
				inputSha256,
				sourceSha256,
				stepSha256,
				options.job.staged_paths.length,
			);
			const exact = exactCandidate && viewManifestMatchesRegionManifest(exactCandidate.manifest, regionManifest)
				? exactCandidate
				: null;
			if (exact) {
				const cacheLatest = await options.worker.request<WorkerInspectResponse>(
					options.workspace,
					{ protocol_version: PROTOCOL_VERSION, operation: "inspect" },
					{ signal: executionSignal },
				);
				const cacheInspection = await inspectFormalCandidate(
					options.workspace,
					cacheLatest.artifact,
					options.getResult(),
				);
				if (
					!cacheInspection.formal_execution_current ||
					cacheInspection.source_sha256 !== sourceSha256 ||
					cacheLatest.artifact.sha256 !== stepSha256
				) {
					throw new InfrastructureError("Formal source or STEP changed while loading cached 2D feedback");
				}
				await options.onComplete(exact.summary);
				return {
					content: [
						{ type: "text" as const, text: formatReflectionFeedback(exact.summary, exact.report, true) },
						await imageFromWorkspace(options.workspace, exact.summary.sheet_path),
					],
					details: { ...exact, cached: true },
				};
			}

			let viewSearch: DrawingViewSearchStageResult;
			if (options.lockedViewManifestPath) {
				const lockedManifest = await loadPartialManifest(
					options.workspace,
					options.lockedViewManifestPath,
					inputSha256,
					sourceSha256,
					stepSha256,
					options.job.staged_paths.length,
				);
				if (!lockedManifest || !viewManifestMatchesRegionManifest(lockedManifest, regionManifest)) {
					throw new InfrastructureError(
						"The controller-bound View Search manifest is stale, incomplete, or does not match the locked regions",
					);
				}
				await writeJsonAtomic(join(options.workspace, manifestPath), lockedManifest);
				viewSearch = {
					manifest: lockedManifest,
					manifestPath,
					searchMode: "reused",
					searchUsage: [],
				};
			} else {
				viewSearch = await runDrawingViewSearchStage({
					job: options.job,
					worker: options.worker,
					workspace: options.workspace,
					resolvedModel: options.resolvedModel,
					controller: options.controller,
					getResult: options.getResult,
					onUsage: options.onUsage,
					segmentation,
					sourceSha256,
					stepSha256,
					outputDir,
				});
			}
			const manifest = viewSearch.manifest;
			const searchMode = viewSearch.searchMode;
			const searchUsage = viewSearch.searchUsage;
			const sheetPath = await makeComparisonSheet({
				worker: options.worker,
				workspace: options.workspace,
				outputDir,
				manifest,
				controller: options.controller,
			});
			const source = await readFile(join(options.workspace, SOLUTION_FILE), "utf8");
			const reflectionState: ReflectionState = { manifest, report: null };
			const reflection = await runRestrictedSession({
				role: "reflection",
				workspace: options.workspace,
				outputDir,
				resolvedModel: options.resolvedModel,
				thinkingLevel: options.job.thinking_level,
				controller: options.controller,
				systemPrompt: REFLECTION_SYSTEM_PROMPT,
				prompt: reflectionPrompt({ fixture: options.job.fixture, manifest, artifact: latest.artifact, source }),
				images: [await imageFromWorkspace(options.workspace, sheetPath)],
				toolNames: REFLECTION_AGENT_TOOLS,
				tools: createReflectionTools({ workspace: options.workspace, state: reflectionState }),
				getSubmitted: () => reflectionState.report,
				activeTimeoutMs: REFLECTION_ACTIVE_TIMEOUT_MS,
				wallTimeoutMs: REFLECTION_WALL_TIMEOUT_MS,
				resultPath: reportPath,
				onUsage: options.onUsage,
			});
			const report = reflection.value;
			await writeJsonAtomic(join(options.workspace, reportPath), report);

			const after = await options.worker.request<WorkerInspectResponse>(
				options.workspace,
				{ protocol_version: PROTOCOL_VERSION, operation: "inspect" },
				{ signal: executionSignal },
			);
			const verified = await inspectFormalCandidate(options.workspace, after.artifact, options.getResult());
			if (
				!verified.formal_execution_current ||
				verified.source_sha256 !== sourceSha256 ||
				after.artifact.sha256 !== stepSha256
			) {
				throw new InfrastructureError("Formal source or STEP changed during 2D self-reflection");
			}

			const usageTotal = addUsageTotals(segmentation.usage, ...searchUsage, reflection.usage);
			const counts = summaryCounts(manifest.regions);
			const summary: DrawingFeedbackSummary = {
				cache_key: cacheKey,
				created_at: isoNow(),
				input_sha256: inputSha256,
				source_sha256: sourceSha256,
				step_sha256: stepSha256,
				segmentation_mode: segmentation.cached ? "reused" : "segmented",
				search_mode: searchMode,
				region_manifest_path: segmentation.manifestPath,
				manifest_path: manifestPath,
				report_path: reportPath,
				sheet_path: sheetPath,
				region_count: manifest.regions.length,
				matched_count: counts.matched,
				ambiguous_count: counts.ambiguous,
				error_count: counts.error,
				should_revise: report.should_revise,
				usage: {
					...(segmentation.usage ? { segmentation: segmentation.usage } : {}),
					...(searchUsage.length > 0 ? { view_search: searchUsage } : {}),
					reflection: reflection.usage,
					total: usageTotal,
				},
			};
			const envelope: CacheEnvelope = { version: DRAWING_FEEDBACK_VERSION, summary, manifest, report };
			await writeJsonAtomic(join(options.workspace, exactCachePath), envelope);
			await options.onComplete(summary);
			return {
				content: [
					{ type: "text" as const, text: formatReflectionFeedback(summary, report, false) },
					await imageFromWorkspace(options.workspace, sheetPath),
				],
				details: { ...envelope, cached: false },
			};
		},
	});
}

export function mergeResultUsage(result: AgentResult, mainUsage: UsageTotals): void {
	result.main_agent_usage = mainUsage;
	const total = addUsageTotals(mainUsage, result.subagent_usage ?? emptyUsageTotals());
	result.tokens = total.tokens;
	result.cost = total.cost;
	result.tool_calls = total.tool_calls;
}

export function recordSubagentUsage(result: AgentResult, usage: SubagentRunUsage): boolean {
	const recorded = result.recorded_subagent_usage_ids ?? [];
	const runs = result.subagent_runs ?? [];
	const existingIndex = runs.findIndex(({ accounting_id }) => accounting_id === usage.accounting_id);
	if (existingIndex >= 0) {
		result.subagent_runs = runs.map((record, index) => index === existingIndex ? usage : record);
		return false;
	}
	if (recorded.includes(usage.accounting_id)) {
		result.subagent_runs = [...runs, usage];
		return false;
	}
	result.subagent_usage = addUsageTotals(result.subagent_usage, usage);
	result.recorded_subagent_usage_ids = [...recorded, usage.accounting_id];
	result.subagent_runs = [...runs, usage];
	return true;
}
