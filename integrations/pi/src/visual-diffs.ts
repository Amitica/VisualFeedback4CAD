import { mkdir } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve } from "node:path";

import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import type { CodeConstructionState } from "./code-construction.js";
import { imageFromWorkspace, inspectPngFile, requireDrawingSuccess } from "./drawing-reflection.js";
import { InfrastructureError } from "./errors.js";
import { isoNow, readJson, writeJsonAtomic } from "./files.js";
import type { WorkerClient } from "./local-worker.js";
import type { RunController } from "./runner.js";
import {
	PROTOCOL_VERSION,
	VISUAL_DIFF_PROFILE,
	type AgentResult,
	type NormalizedRoi,
	type VisualDiffProfile,
	type WorkerVisualDiffResponse,
} from "./types.js";

export const VISUAL_DIFF_STATE_PATH = "drawing_feedback/visual-diffs/state.json";
export const VISUAL_DIFF_CONTEXT_TYPE = "cadgenbench-persistent-visual-diffs";

const DIFF_ID_PATTERN = /^D[0-9]{3,}$/u;
const CONSTRUCTION_REF_PATTERN = /^C[0-9]{3,}$/u;
const REGION_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u;
const SHA256_PATTERN = /^[a-f0-9]{64}$/u;
const MAX_STORED_DIFFS = 5;

export interface VisualDiffMetrics {
	overlap_pixel_count: number;
	target_only_pixel_count: number;
	current_only_pixel_count: number;
	iou: number;
	centroid_delta_px: [number, number];
}

export interface VisualDiffRecord {
	diff_id: string;
	region_id: string;
	target_roi: NormalizedRoi;
	current_construction_ref: string;
	source_sha256: string;
	step_sha256: string;
	camera_sha256: string;
	code_ablation_sequence: number;
	comparison_path: string;
	comparison_sha256: string;
	target_mask_path: string;
	target_mask_sha256: string;
	current_mask_path: string;
	current_mask_sha256: string;
	target_pixel_roi: [number, number, number, number];
	target_geometry_pixel_bbox: [number, number, number, number];
	current_geometry_pixel_bbox: [number, number, number, number];
	metrics: VisualDiffMetrics;
	created_at: string;
	updated_at: string;
}

interface VisualDiffHistoryEvent {
	revision: number;
	diff_id: string;
	created_at: string;
}

interface PersistedVisualDiffState {
	version: 1;
	profile: VisualDiffProfile;
	fixture_id: string;
	locked_view_manifest_sha256: string;
	diffs: VisualDiffRecord[];
	next_diff_index: number;
	revision: number;
	started_at: string;
	updated_at: string;
	history: VisualDiffHistoryEvent[];
}

export interface VisualDiffState {
	profile: VisualDiffProfile;
	fixtureId: string;
	lockedViewManifestSha256: string;
	diffs: VisualDiffRecord[];
	nextDiffIndex: number;
	revision: number;
	statePath: string;
	startedAt: string;
	history: VisualDiffHistoryEvent[];
}

const roiSchema = Type.Object({
	x_min: Type.Integer({ minimum: 0, maximum: 1000 }),
	y_min: Type.Integer({ minimum: 0, maximum: 1000 }),
	x_max: Type.Integer({ minimum: 0, maximum: 1000 }),
	y_max: Type.Integer({ minimum: 0, maximum: 1000 }),
}, {
	additionalProperties: false,
	description: "Target geometry ROI in normalized 0-1000 coordinates; width and height must each be at least 50.",
});

export const VISUAL_DIFF_PARAMETERS_SCHEMA = Type.Object({
	region_id: Type.String({ pattern: "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$" }),
	target_roi: roiSchema,
	current_construction_ref: Type.String({ pattern: "^C[0-9]{3,}$" }),
}, { additionalProperties: false });

function roiFromObject(value: { x_min: number; y_min: number; x_max: number; y_max: number }): NormalizedRoi {
	const roi: NormalizedRoi = [value.x_min, value.y_min, value.x_max, value.y_max];
	if (roi[0] >= roi[2] || roi[1] >= roi[3]) throw new Error("target_roi must satisfy x_min < x_max and y_min < y_max");
	if (roi[2] - roi[0] < 50 || roi[3] - roi[1] < 50) {
		throw new Error("target_roi width and height must each be at least 50");
	}
	return roi;
}

function cloneDiff(diff: VisualDiffRecord): VisualDiffRecord {
	return {
		...diff,
		target_roi: [...diff.target_roi],
		target_pixel_roi: [...diff.target_pixel_roi],
		target_geometry_pixel_bbox: [...diff.target_geometry_pixel_bbox],
		current_geometry_pixel_bbox: [...diff.current_geometry_pixel_bbox],
		metrics: {
			...diff.metrics,
			centroid_delta_px: [...diff.metrics.centroid_delta_px],
		},
	};
}

function persistedState(state: VisualDiffState): PersistedVisualDiffState {
	return {
		version: 1,
		profile: state.profile,
		fixture_id: state.fixtureId,
		locked_view_manifest_sha256: state.lockedViewManifestSha256,
		diffs: state.diffs.map(cloneDiff),
		next_diff_index: state.nextDiffIndex,
		revision: state.revision,
		started_at: state.startedAt,
		updated_at: isoNow(),
		history: state.history,
	};
}

async function persistVisualDiffState(workspace: string, state: VisualDiffState): Promise<void> {
	const root = resolve(workspace);
	const path = resolve(root, state.statePath);
	const rel = relative(root, path);
	if (rel === ".." || rel.startsWith("../") || isAbsolute(rel)) {
		throw new InfrastructureError("Visual-diff state path escaped the fixture workspace");
	}
	await mkdir(dirname(path), { recursive: true });
	await writeJsonAtomic(path, persistedState(state));
}

function isOrderedIntegerQuad(value: unknown, maximum?: number): value is NormalizedRoi {
	if (!Array.isArray(value) || value.length !== 4 || value.some((item) => !Number.isInteger(item))) return false;
	const [xMin, yMin, xMax, yMax] = value as [number, number, number, number];
	return xMin >= 0 && yMin >= 0 && xMin < xMax && yMin < yMax &&
		(maximum === undefined || value.every((item) => item <= maximum));
}

function isSafeRelativePngPath(value: unknown): value is string {
	if (typeof value !== "string" || value.length === 0 || isAbsolute(value) || !value.endsWith(".png")) return false;
	const normalized = relative(".", value);
	return normalized !== ".." && !normalized.startsWith("../") && !isAbsolute(normalized);
}

function isNonNegativeInteger(value: unknown): value is number {
	return Number.isInteger(value) && Number(value) >= 0;
}

function assertStoredDiff(diff: VisualDiffRecord): void {
	if (
		typeof diff !== "object" || diff === null ||
		!DIFF_ID_PATTERN.test(diff.diff_id) ||
		!REGION_ID_PATTERN.test(diff.region_id) ||
		!CONSTRUCTION_REF_PATTERN.test(diff.current_construction_ref) ||
		!isOrderedIntegerQuad(diff.target_roi, 1000) ||
		diff.target_roi[2] - diff.target_roi[0] < 50 ||
		diff.target_roi[3] - diff.target_roi[1] < 50 ||
		!SHA256_PATTERN.test(diff.source_sha256) ||
		!SHA256_PATTERN.test(diff.step_sha256) ||
		!SHA256_PATTERN.test(diff.camera_sha256) ||
		!Number.isInteger(diff.code_ablation_sequence) || diff.code_ablation_sequence < 1 ||
		!isSafeRelativePngPath(diff.comparison_path) ||
		!isSafeRelativePngPath(diff.target_mask_path) ||
		!isSafeRelativePngPath(diff.current_mask_path) ||
		!SHA256_PATTERN.test(diff.comparison_sha256) ||
		!SHA256_PATTERN.test(diff.target_mask_sha256) ||
		!SHA256_PATTERN.test(diff.current_mask_sha256) ||
		!isOrderedIntegerQuad(diff.target_pixel_roi) ||
		!isOrderedIntegerQuad(diff.target_geometry_pixel_bbox) ||
		!isOrderedIntegerQuad(diff.current_geometry_pixel_bbox) ||
		!isNonNegativeInteger(diff.metrics?.overlap_pixel_count) ||
		!isNonNegativeInteger(diff.metrics?.target_only_pixel_count) ||
		!isNonNegativeInteger(diff.metrics?.current_only_pixel_count) ||
		typeof diff.metrics?.iou !== "number" || !Number.isFinite(diff.metrics.iou) ||
		diff.metrics.iou < 0 || diff.metrics.iou > 1 ||
		!Array.isArray(diff.metrics?.centroid_delta_px) ||
		diff.metrics.centroid_delta_px.length !== 2 ||
		diff.metrics.centroid_delta_px.some((item) => typeof item !== "number" || !Number.isFinite(item)) ||
		Number.isNaN(Date.parse(diff.created_at)) ||
		Number.isNaN(Date.parse(diff.updated_at))
	) throw new InfrastructureError("Persisted visual diff is invalid");
}

export async function restoreVisualDiffState(options: {
	workspace: string;
	fixtureId: string;
	profile: VisualDiffProfile;
	lockedViewManifestSha256: string;
	statePath?: string;
}): Promise<VisualDiffState> {
	const statePath = options.statePath ?? VISUAL_DIFF_STATE_PATH;
	const stored = await readJson<PersistedVisualDiffState>(resolve(options.workspace, statePath)).catch(() => null);
	if (!stored) {
		return {
			profile: options.profile,
			fixtureId: options.fixtureId,
			lockedViewManifestSha256: options.lockedViewManifestSha256,
			diffs: [],
			nextDiffIndex: 1,
			revision: 0,
			statePath,
			startedAt: isoNow(),
			history: [],
		};
	}
	if (
		stored.version !== 1 || stored.profile !== VISUAL_DIFF_PROFILE || stored.profile !== options.profile ||
		stored.fixture_id !== options.fixtureId ||
		stored.locked_view_manifest_sha256 !== options.lockedViewManifestSha256 ||
		!Array.isArray(stored.diffs) || stored.diffs.length > MAX_STORED_DIFFS ||
		!Number.isInteger(stored.next_diff_index) || stored.next_diff_index < 1 ||
		!Number.isInteger(stored.revision) || stored.revision < 0 || !Array.isArray(stored.history)
	) throw new InfrastructureError("Persisted visual-diff state does not match the continuation");
	for (const diff of stored.diffs) assertStoredDiff(diff);
	if (new Set(stored.diffs.map(({ diff_id }) => diff_id)).size !== stored.diffs.length) {
		throw new InfrastructureError("Persisted visual diffs contain duplicate IDs");
	}
	const maximumIndex = stored.diffs.reduce((maximum, { diff_id }) => (
		Math.max(maximum, Number.parseInt(diff_id.slice(1), 10))
	), 0);
	if (stored.next_diff_index <= maximumIndex) throw new InfrastructureError("Persisted next visual-diff index is invalid");
	return {
		profile: stored.profile,
		fixtureId: stored.fixture_id,
		lockedViewManifestSha256: stored.locked_view_manifest_sha256,
		diffs: stored.diffs.map(cloneDiff),
		nextDiffIndex: stored.next_diff_index,
		revision: stored.revision,
		statePath,
		startedAt: stored.started_at,
		history: stored.history,
	};
}

function currentFormalHashes(result: AgentResult): { sourceSha256: string; stepSha256: string } {
	if (!result.last_executed_source_sha256 || !result.last_executed_step_sha256) {
		throw new Error("Visual diff requires a formally executed Current CAD");
	}
	return { sourceSha256: result.last_executed_source_sha256, stepSha256: result.last_executed_step_sha256 };
}

export function visualDiffFreshness(
	diff: VisualDiffRecord,
	result: Pick<AgentResult, "last_executed_source_sha256" | "last_executed_step_sha256">,
	codeState: CodeConstructionState,
): "fresh" | "stale" {
	if (diff.source_sha256 !== result.last_executed_source_sha256 || diff.step_sha256 !== result.last_executed_step_sha256) {
		return "stale";
	}
	const construction = codeState.constructions.find(({ construction_ref }) => (
		construction_ref === diff.current_construction_ref
	));
	return construction !== undefined &&
		codeState.activeConstructionRefs.includes(diff.current_construction_ref) &&
		(construction.binding_status === undefined || construction.binding_status === "current")
		? "fresh"
		: "stale";
}

async function assertVisualDiffIntegrity(workspace: string, diff: VisualDiffRecord): Promise<void> {
	const files = await Promise.all([
		inspectPngFile(workspace, diff.comparison_path),
		inspectPngFile(workspace, diff.target_mask_path),
		inspectPngFile(workspace, diff.current_mask_path),
	]);
	if (
		files[0].sha256 !== diff.comparison_sha256 ||
		files[1].sha256 !== diff.target_mask_sha256 ||
		files[2].sha256 !== diff.current_mask_sha256
	) throw new InfrastructureError(`Visual diff ${diff.diff_id} no longer matches its stored hashes`);
}

export async function injectPersistentVisualDiffs<T>(options: {
	messages: T[];
	workspace: string;
	state: VisualDiffState;
	codeState: CodeConstructionState;
	result: Pick<AgentResult, "last_executed_source_sha256" | "last_executed_step_sha256">;
}): Promise<T[]> {
	const filtered = options.messages.filter((message) => (
		typeof message !== "object" || message === null ||
		(message as { customType?: unknown }).customType !== VISUAL_DIFF_CONTEXT_TYPE
	));
	if (options.state.diffs.length === 0) return filtered;
	const ordered = [...options.state.diffs].sort((left, right) => right.updated_at.localeCompare(left.updated_at));
	const content: Array<TextContent | ImageContent> = [{
		type: "text",
		text: `[VisualDiff revision ${options.state.revision}]\n${ordered.map((diff) => [
			diff.diff_id,
			`freshness=${visualDiffFreshness(diff, options.result, options.codeState)}`,
			`target_roi=${diff.target_roi.join(",")}`,
			`current=${diff.current_construction_ref}`,
			`iou=${diff.metrics.iou.toFixed(4)}`,
		].join(" | ")).join("\n")}`,
	}];
	for (const diff of ordered) {
		if (visualDiffFreshness(diff, options.result, options.codeState) !== "fresh") continue;
		await assertVisualDiffIntegrity(options.workspace, diff);
		content.push(await imageFromWorkspace(options.workspace, diff.comparison_path));
	}
	return [...filtered, {
		role: "custom",
		customType: VISUAL_DIFF_CONTEXT_TYPE,
		content,
		display: false,
		timestamp: Date.now(),
	} as T];
}

export function createVisualDiffTool(options: {
	workspace: string;
	state: VisualDiffState;
	codeState: CodeConstructionState;
	worker: WorkerClient;
	controller: RunController;
	getResult: () => AgentResult;
}) {
	return defineTool({
		name: "visual_diff",
		label: "Visual diff",
		description: "Select one Target ROI and one Current construction already highlighted by inspect_code_ablation. Returns one same-coordinate overlay: gray is overlap, orange exists only in Target, and blue exists only in Current. The tool automatically extracts masks, computes residuals, persists the latest comparison, and replaces an earlier diff for the same region and construction. No prose description, relation, tags, proposal, or confirmation is required.",
		parameters: VISUAL_DIFF_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			const formal = currentFormalHashes(options.getResult());
			const construction = options.codeState.constructions.find(({ construction_ref }) => (
				construction_ref === params.current_construction_ref
			));
			if (
				construction === undefined ||
				!options.codeState.activeConstructionRefs.includes(params.current_construction_ref) ||
				(construction.binding_status !== undefined && construction.binding_status !== "current")
			) throw new Error(`Current construction ${params.current_construction_ref} is inactive, modified, or unresolved`);
			const ablation = options.codeState.inspections.findLast((inspection) => (
				inspection.status === "highlighted" &&
				inspection.construction_ref === params.current_construction_ref &&
				inspection.region_id === params.region_id &&
				inspection.formal_source_sha256 === formal.sourceSha256 &&
				inspection.formal_step_sha256 === formal.stepSha256 &&
				inspection.camera_sha256 !== undefined &&
				inspection.reference_path !== undefined &&
				inspection.current_render_path !== undefined &&
				inspection.ablated_render_path !== undefined
			));
			if (!ablation) {
				throw new Error(`Run inspect_code_ablation for ${params.current_construction_ref} in ${params.region_id} on the current formal CAD first`);
			}
			const targetRoi = roiFromObject(params.target_roi);
			const existingIndex = options.state.diffs.findIndex((diff) => (
				diff.region_id === params.region_id && diff.current_construction_ref === params.current_construction_ref
			));
			const diffId = existingIndex >= 0
				? options.state.diffs[existingIndex]!.diff_id
				: `D${String(options.state.nextDiffIndex).padStart(3, "0")}`;
			const outputDir = resolve(
				options.workspace,
				"drawing_feedback",
				"visual-diffs",
				diffId,
				`revision-${String(options.state.revision + 1).padStart(3, "0")}`,
			);
			const response = await options.worker.request<WorkerVisualDiffResponse>(options.workspace, {
				protocol_version: PROTOCOL_VERSION,
				operation: "compose_visual_diff",
				region_id: params.region_id,
				diff_id: diffId,
				construction_ref: params.current_construction_ref,
				target_roi: targetRoi,
				reference_path: ablation.reference_path!,
				current_render_path: ablation.current_render_path!,
				ablated_render_path: ablation.ablated_render_path!,
				output_dir: relative(resolve(options.workspace), outputDir),
			}, { timeoutMs: 180_000, signal: options.controller.signal });
			requireDrawingSuccess(response, "compose_visual_diff");
			if (
				response.diff_id !== diffId || response.region_id !== params.region_id ||
				response.construction_ref !== params.current_construction_ref ||
				response.target_roi.some((value, index) => value !== targetRoi[index])
			) throw new InfrastructureError("Visual-diff renderer returned mismatched structured evidence");
			const inspected = await Promise.all([
				inspectPngFile(options.workspace, response.comparison_path),
				inspectPngFile(options.workspace, response.target_mask_path),
				inspectPngFile(options.workspace, response.current_mask_path),
			]);
			if (
				inspected[0].sha256 !== response.sha256.comparison ||
				inspected[1].sha256 !== response.sha256.target_mask ||
				inspected[2].sha256 !== response.sha256.current_mask
			) throw new InfrastructureError("Visual-diff artifact hash mismatch");
			const now = isoNow();
			const record: VisualDiffRecord = {
				diff_id: diffId,
				region_id: params.region_id,
				target_roi: targetRoi,
				current_construction_ref: params.current_construction_ref,
				source_sha256: formal.sourceSha256,
				step_sha256: formal.stepSha256,
				camera_sha256: ablation.camera_sha256!,
				code_ablation_sequence: ablation.sequence,
				comparison_path: response.comparison_path,
				comparison_sha256: response.sha256.comparison,
				target_mask_path: response.target_mask_path,
				target_mask_sha256: response.sha256.target_mask,
				current_mask_path: response.current_mask_path,
				current_mask_sha256: response.sha256.current_mask,
				target_pixel_roi: response.target_pixel_roi,
				target_geometry_pixel_bbox: response.target_geometry_pixel_bbox,
				current_geometry_pixel_bbox: response.current_geometry_pixel_bbox,
				metrics: response.metrics,
				created_at: existingIndex >= 0 ? options.state.diffs[existingIndex]!.created_at : now,
				updated_at: now,
			};
			if (existingIndex >= 0) options.state.diffs[existingIndex] = record;
			else {
				if (options.state.diffs.length >= MAX_STORED_DIFFS) {
					options.state.diffs.sort((left, right) => left.updated_at.localeCompare(right.updated_at)).shift();
				}
				options.state.diffs.push(record);
				options.state.nextDiffIndex += 1;
			}
			options.state.revision += 1;
			options.state.history.push({ revision: options.state.revision, diff_id: diffId, created_at: now });
			await persistVisualDiffState(options.workspace, options.state);
			return {
				content: [await imageFromWorkspace(options.workspace, response.comparison_path)],
				details: { revision: options.state.revision, diff: cloneDiff(record) },
			};
		},
	});
}
