import { createHash } from "node:crypto";
import { lstat, mkdir, readFile, realpath } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";

import type { ImageContent } from "@earendil-works/pi-ai";
import { defineTool, type SessionEntry } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import {
	assertSupportedViewSpec,
	imageFromWorkspace,
	inspectPngFile,
	manifestLooksCurrent,
	requireDrawingSuccess,
} from "./drawing-reflection.js";
import { InfrastructureError } from "./errors.js";
import { isoNow, readJson, writeJsonAtomic } from "./files.js";
import type { WorkerClient } from "./local-worker.js";
import { inspectFormalCandidate } from "./solution.js";
import {
	createTargetAnnotationTool,
	targetAnnotationsForRegion,
	type TargetAnnotation,
	type TargetAnnotationState,
} from "./target-annotations.js";
import {
	MAIN_VIEW_REFLECT_AGENT_TOOLS,
	PROTOCOL_VERSION,
	TARGET_ANNOTATION_AGENT_TOOLS,
	type AgentResult,
	type DrawingManifestRegion,
	type DrawingSimilarityTransform,
	type DrawingViewManifest,
	type DrawingViewSpec,
	type MainViewReflectSummary,
	type NormalizedRoi,
	type Vector3,
	type WorkerInspectResponse,
	type WorkerFixedProjectionResponse,
	type WorkerRankDrawingViewsResponse,
	type WorkerReflectionInspectionResponse,
} from "./types.js";
import type { RunController } from "./runner.js";

export const MAIN_VIEW_REFLECT_KIND = "main_view_reflect" as const;
export const MAIN_VIEW_REFLECT_AUDIT_PATH = "drawing_feedback/main-view-reflect/audit.json";

const REGION_ID_SCHEMA = Type.String({
	minLength: 1,
	maxLength: 128,
	pattern: "^[A-Za-z0-9][A-Za-z0-9._-]*$",
});

const ROI_SCHEMA = Type.Object(
	{
		x_min: Type.Integer({ minimum: 0, maximum: 1000 }),
		y_min: Type.Integer({ minimum: 0, maximum: 1000 }),
		x_max: Type.Integer({ minimum: 0, maximum: 1000 }),
		y_max: Type.Integer({ minimum: 0, maximum: 1000 }),
	},
	{
		additionalProperties: false,
		description: "One ROI on its own image canvas in normalized 0-1000 coordinates; width and height must each be at least 50.",
	},
);

export const INSPECT_COMPARISON_PARAMETERS_SCHEMA = Type.Object(
	{ region_id: REGION_ID_SCHEMA },
	{
		additionalProperties: false,
		description: "Inspect one drawing/CAD comparison using the controller-locked View Search camera.",
	},
);

export const DUAL_ROI_CROP_PARAMETERS_SCHEMA = Type.Object(
	{
		region_id: REGION_ID_SCHEMA,
		reference_roi: ROI_SCHEMA,
		cad_roi: ROI_SCHEMA,
	},
	{
		additionalProperties: false,
		description: "Crop corresponding details from the Original drawing and current fixed-camera CAD comparison. Paths and cameras are controller-owned.",
	},
);

const DRAWING_SCALE_BOUNDARIES = ["overall_width", "overall_height"] as const;

export const CALIBRATE_DRAWING_SCALE_PARAMETERS_SCHEMA = Type.Union([
	Type.Object({
		operation: Type.Literal("propose"),
		region_id: REGION_ID_SCHEMA,
		boundary: Type.Union(DRAWING_SCALE_BOUNDARIES.map((value) => Type.Literal(value))),
		target_dimension_mm: Type.Number({
			minimum: 0.001,
			maximum: 1_000_000,
			description: "Exact overall boundary dimension read from the original engineering drawing, in millimetres.",
		}),
	}, { additionalProperties: false }),
	Type.Object({
		operation: Type.Literal("confirm"),
		calibration_ref: Type.String({ pattern: "^S[0-9]{3,}$" }),
	}, { additionalProperties: false }),
	Type.Object({
		operation: Type.Literal("delete"),
		calibration_ref: Type.String({ pattern: "^S[0-9]{3,}$" }),
	}, { additionalProperties: false }),
]);

export interface MainViewReflectSnapshotRecord {
	snapshot_id: string;
	region_id: string;
	source_sha256: string;
	step_sha256: string;
	camera_sha256: string;
	locked_candidate_id: string;
	view_spec: DrawingViewSpec;
	crop_path: string;
	render_path: string;
	aligned_render_path: string;
	comparison_path: string;
	sha256: {
		crop: string;
		render: string;
		aligned_render: string;
		comparison: string;
	};
	score: number;
	eligible: boolean;
	failed_gates: string[];
	transform?: DrawingSimilarityTransform;
	fixed_frame_sha256?: string;
	projection_origin?: Vector3;
	projection_anchor_step_sha256?: string;
	section_origin?: Vector3;
	scale_calibration_ref?: string;
	free_fit_scale_px_per_unit?: number;
	reused_locked_artifact: boolean;
	created_at: string;
}

export interface DrawingScaleCalibrationRecord {
	calibration_ref: string;
	region_id: string;
	boundary: (typeof DRAWING_SCALE_BOUNDARIES)[number];
	status: "proposed" | "confirmed" | "superseded" | "deleted";
	target_dimension_mm: number;
	current_dimension_units: number;
	free_fit_scale_px_per_unit: number;
	locked_scale_px_per_unit: number;
	scale_ratio: number;
	source_sha256: string;
	step_sha256: string;
	camera_sha256: string;
	free_transform: DrawingSimilarityTransform;
	locked_transform: DrawingSimilarityTransform;
	preview_path: string;
	preview_sha256: string;
	created_at: string;
	updated_at: string;
}

export interface MainViewReflectInspectionRecord {
	inspection_id: string;
	tool_call_id?: string;
	snapshot_id: string;
	region_id: string;
	source_sha256: string;
	step_sha256: string;
	camera_sha256: string;
	context_path: string;
	context_sha256: string;
	created_at: string;
}

export interface MainViewReflectInspectionAttempt {
	sequence: number;
	inspection_id: string;
	tool_call_id?: string;
	region_id: string;
	source_sha256: string;
	step_sha256: string;
	cache_hit: boolean;
	created_at: string;
}

export interface MainViewReflectCropRecord {
	crop_id: string;
	snapshot_id: string;
	region_id: string;
	source_sha256: string;
	step_sha256: string;
	camera_sha256: string;
	reference_roi: NormalizedRoi;
	cad_roi: NormalizedRoi;
	reference_pixel_roi: [number, number, number, number];
	cad_pixel_roi: [number, number, number, number];
	reference_crop_path: string;
	cad_crop_path: string;
	context_path: string;
	comparison_path: string;
	sha256: {
		reference_source: string;
		cad_source: string;
		reference_crop: string;
		cad_crop: string;
		context: string;
		comparison: string;
	};
	created_at: string;
}

export interface MainViewReflectCropAttempt {
	sequence: number;
	crop_id: string;
	region_id: string;
	source_sha256: string;
	step_sha256: string;
	cache_hit: boolean;
	created_at: string;
}

interface PersistedMainViewReflectAudit {
	version: 1;
	kind: typeof MAIN_VIEW_REFLECT_KIND;
	fixture_id: string;
	locked_view_manifest_path: string;
	locked_view_manifest_sha256: string;
	baseline_source_sha256: string;
	baseline_step_sha256: string;
	started_at: string;
	updated_at: string;
	snapshots: MainViewReflectSnapshotRecord[];
	inspections: MainViewReflectInspectionRecord[];
	inspection_attempts: MainViewReflectInspectionAttempt[];
	crops: MainViewReflectCropRecord[];
	crop_attempts: MainViewReflectCropAttempt[];
	scale_calibrations?: DrawingScaleCalibrationRecord[];
}

export interface MainViewReflectState {
	fixtureId: string;
	lockedManifest: DrawingViewManifest;
	lockedManifestPath: string;
	lockedManifestSha256: string;
	baselineSourceSha256: string;
	baselineStepSha256: string;
	auditPath: string;
	startedAt: string;
	snapshots: Map<string, MainViewReflectSnapshotRecord>;
	inspections: Map<string, MainViewReflectInspectionRecord>;
	inspectionAttempts: MainViewReflectInspectionAttempt[];
	crops: Map<string, MainViewReflectCropRecord>;
	cropAttempts: MainViewReflectCropAttempt[];
	scaleCalibrations: DrawingScaleCalibrationRecord[];
}

interface CurrentFormalState {
	sourceSha256: string;
	stepSha256: string;
}

function sha256Json(value: unknown): string {
	return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

async function sha256File(path: string): Promise<string> {
	return createHash("sha256").update(await readFile(path)).digest("hex");
}

export async function loadLockedMainViewManifest(options: {
	workspace: string;
	manifestPath: string;
	manifestSha256: string;
	baselineSourceSha256: string;
	baselineStepSha256: string;
}): Promise<DrawingViewManifest> {
	if (!/^[a-f0-9]{64}$/u.test(options.manifestSha256)) {
		throw new InfrastructureError("Locked View Search manifest hash is invalid");
	}
	const manifestFile = await requireSafeFile(options.workspace, options.manifestPath);
	if (await sha256File(manifestFile) !== options.manifestSha256) {
		throw new InfrastructureError("Locked View Search manifest SHA-256 mismatch");
	}
	const manifest = await readJson<DrawingViewManifest>(manifestFile).catch(() => null);
	if (
		!manifest ||
		!manifestLooksCurrent(
			manifest,
			manifest.input_sha256,
			options.baselineSourceSha256,
			options.baselineStepSha256,
		)
	) {
		const failedRegions = Array.isArray(manifest?.regions)
			? manifest.regions.filter(region => region?.status === "error")
			: [];
		const detail = failedRegions.length
			? `; failed regions: ${failedRegions.map(region => `${region.region_id}: ${region.note}`).join("; ")}`
			: "";
		throw new InfrastructureError(`Locked View Search manifest is invalid or does not match the Direct baseline${detail}`);
	}
	await validateFixedProjectionAnchor(options.workspace, manifest);
	for (const region of manifest.regions) {
		if (!region.selected) {
			throw new InfrastructureError(`Locked View Search region ${region.region_id} has no selected camera`);
		}
		await Promise.all([
			requireSafeFile(options.workspace, region.crop_path),
			requireSafeFile(options.workspace, region.selected.render_path),
			requireSafeFile(options.workspace, region.selected.aligned_render_path),
			requireSafeFile(options.workspace, region.selected.comparison_path),
		]);
	}
	return manifest;
}

export function filterLockedMainViewManifest(
	manifest: DrawingViewManifest,
	regionId: string,
): DrawingViewManifest {
	const region = manifest.regions.find((candidate) => candidate.region_id === regionId);
	if (!region) {
		throw new InfrastructureError(`Locked View Search manifest has no region ${regionId}`);
	}
	if (!region.selected) {
		throw new InfrastructureError(`Locked View Search region ${regionId} has no selected camera`);
	}
	return { ...manifest, regions: [region] };
}

function roiFromObject(value: { x_min: number; y_min: number; x_max: number; y_max: number }): NormalizedRoi {
	const roi: NormalizedRoi = [value.x_min, value.y_min, value.x_max, value.y_max];
	if (roi[0] >= roi[2] || roi[1] >= roi[3]) throw new Error("ROI must satisfy x_min < x_max and y_min < y_max");
	if (roi[2] - roi[0] < 50 || roi[3] - roi[1] < 50) {
		throw new Error("ROI width and height must each be at least 50 in normalized coordinates");
	}
	return roi;
}

function sameRoi(left: NormalizedRoi, right: NormalizedRoi): boolean {
	return left.every((value, index) => value === right[index]);
}

function cameraIdentity(spec: DrawingViewSpec): object {
	return {
		direction: spec.direction,
		up: spec.up,
		section_offset_fraction: spec.section_offset_fraction ?? null,
		section_keep: spec.section_keep ?? null,
	};
}

export function cameraSha256(spec: DrawingViewSpec): string {
	return sha256Json(cameraIdentity(spec));
}

function sameCamera(left: DrawingViewSpec, right: DrawingViewSpec): boolean {
	return cameraSha256(left) === cameraSha256(right);
}

function fixedFrameIdentity(state: MainViewReflectState, region: DrawingManifestRegion): string | undefined {
	if (!state.lockedManifest.projection_frame) return undefined;
	return sha256Json({
		version: 1,
		manifest_sha256: state.lockedManifestSha256,
		projection_frame: state.lockedManifest.projection_frame,
		camera_sha256: cameraSha256(region.selected!.view_spec),
		transform: confirmedScaleCalibration(state, region.region_id)?.locked_transform ?? region.selected!.transform,
		crop_path: region.crop_path,
		crop_sha256: region.crop_sha256,
	});
}

function snapshotId(region: DrawingManifestRegion, formal: CurrentFormalState, state?: MainViewReflectState): string {
	if (!region.selected) throw new Error(`Region ${region.region_id} has no selected View Search camera`);
	return sha256Json([
		MAIN_VIEW_REFLECT_KIND,
		region.region_id,
		formal.stepSha256,
		cameraSha256(region.selected.view_spec),
		...(state?.lockedManifest.projection_frame ? [fixedFrameIdentity(state, region)] : []),
	]).slice(0, 20);
}

async function requireSafeFile(workspace: string, path: string): Promise<string> {
	if (isAbsolute(path)) throw new InfrastructureError("Main View Reflect artifact path must be relative");
	const root = resolve(workspace);
	const candidate = resolve(root, path);
	const rel = relative(root, candidate);
	if (rel === ".." || rel.startsWith("../") || isAbsolute(rel)) {
		throw new InfrastructureError("Main View Reflect artifact escaped the fixture workspace");
	}
	const metadata = await lstat(candidate).catch(() => null);
	if (!metadata?.isFile() || metadata.isSymbolicLink()) {
		throw new InfrastructureError(`Main View Reflect artifact is missing or unsafe: ${path}`);
	}
	const [rootReal, candidateReal] = await Promise.all([realpath(root), realpath(candidate)]);
	const realRel = relative(rootReal, candidateReal);
	if (realRel === ".." || realRel.startsWith("../") || isAbsolute(realRel)) {
		throw new InfrastructureError("Main View Reflect artifact resolved outside the fixture workspace");
	}
	return candidateReal;
}

async function validateFixedProjectionAnchor(workspace: string, manifest: DrawingViewManifest): Promise<void> {
	const frame = manifest.projection_frame;
	if (frame === undefined) return;
	if (
		!frame || frame.policy !== "fixed" || typeof frame.anchor_step_path !== "string" ||
		!/^[a-f0-9]{64}$/u.test(frame.anchor_step_sha256) || frame.anchor_step_sha256 !== manifest.step_sha256
	) throw new InfrastructureError("Locked projection frame is invalid");
	const anchor = await requireSafeFile(workspace, frame.anchor_step_path);
	if (await sha256File(anchor) !== frame.anchor_step_sha256) {
		throw new InfrastructureError("Locked projection anchor STEP SHA-256 mismatch");
	}
}

function finiteVector3(value: unknown): value is Vector3 {
	return Array.isArray(value) && value.length === 3 && value.every((component) => (
		typeof component === "number" && Number.isFinite(component)
	));
}

function regionForInspection(state: MainViewReflectState, regionId: string): DrawingManifestRegion {
	const region = state.lockedManifest.regions.find(({ region_id }) => region_id === regionId);
	if (!region) throw new Error(`Unknown locked comparison region ${regionId}`);
	if (!region.selected?.aligned_render_path) {
		throw new Error(`Region ${regionId} has no selected aligned View Search camera`);
	}
	assertSupportedViewSpec(region.selected.view_spec);
	return region;
}

function confirmedScaleCalibration(
	state: MainViewReflectState,
	regionId: string,
): DrawingScaleCalibrationRecord | undefined {
	return state.scaleCalibrations.findLast((record) => (
		record.region_id === regionId && record.status === "confirmed"
	));
}

function clearRegionVisualCache(state: MainViewReflectState, regionId: string): void {
	for (const [id, record] of state.snapshots) {
		if (record.region_id === regionId) state.snapshots.delete(id);
	}
	for (const [id, record] of state.inspections) {
		if (record.region_id === regionId) state.inspections.delete(id);
	}
	for (const [id, record] of state.crops) {
		if (record.region_id === regionId) state.crops.delete(id);
	}
}

async function currentFormalState(options: {
	worker: WorkerClient;
	workspace: string;
	controller: RunController;
	getResult: () => AgentResult;
}): Promise<CurrentFormalState> {
	const latest = await options.worker.request<WorkerInspectResponse>(
		options.workspace,
		{ protocol_version: PROTOCOL_VERSION, operation: "inspect" },
		{ signal: options.controller.signal },
	);
	const inspected = await inspectFormalCandidate(options.workspace, latest.artifact, options.getResult());
	const result = options.getResult();
	if (
		!result.last_executed_source_sha256 ||
		!result.last_executed_step_sha256 ||
		!latest.artifact.sha256 ||
		latest.artifact.sha256 !== result.last_executed_step_sha256
	) {
		throw new Error(`2D comparison requires an output.step produced by execute_build123d. ${inspected.reason}`);
	}
	if (latest.artifact.validation?.is_valid !== true || latest.artifact.validation.is_watertight !== true) {
		throw new Error("2D comparison requires a valid and watertight current formal STEP");
	}
	return {
		sourceSha256: result.last_executed_source_sha256,
		stepSha256: latest.artifact.sha256,
	};
}

async function baselineSnapshot(options: {
	workspace: string;
	state: MainViewReflectState;
	region: DrawingManifestRegion;
	formal: CurrentFormalState;
}): Promise<MainViewReflectSnapshotRecord> {
	const selected = options.region.selected!;
	const [crop, render, alignedRender, comparison] = await Promise.all([
		inspectPngFile(options.workspace, options.region.crop_path),
		inspectPngFile(options.workspace, selected.render_path),
		inspectPngFile(options.workspace, selected.aligned_render_path),
		inspectPngFile(options.workspace, selected.comparison_path),
	]);
	return {
		snapshot_id: snapshotId(options.region, options.formal, options.state),
		region_id: options.region.region_id,
		source_sha256: options.formal.sourceSha256,
		step_sha256: options.formal.stepSha256,
		camera_sha256: cameraSha256(selected.view_spec),
		locked_candidate_id: selected.candidate_id,
		view_spec: selected.view_spec,
		crop_path: options.region.crop_path,
		render_path: selected.render_path,
		aligned_render_path: selected.aligned_render_path,
		comparison_path: selected.comparison_path,
		sha256: {
			crop: crop.sha256,
			render: render.sha256,
			aligned_render: alignedRender.sha256,
			comparison: comparison.sha256,
		},
		score: selected.score,
		eligible: selected.eligible,
		failed_gates: [...selected.failed_gates],
		transform: selected.transform,
		reused_locked_artifact: true,
		created_at: isoNow(),
	};
}

export function candidateScopedArtifactMatches(path: string, candidateId: string): boolean {
	const parts = path.replaceAll("\\", "/").split("/");
	const candidateDirectoryIndex = parts.lastIndexOf("candidates");
	return candidateDirectoryIndex < 0 || parts[candidateDirectoryIndex + 1] === candidateId;
}

function lockedSelectedArtifactsAreConsistent(region: DrawingManifestRegion): boolean {
	const selected = region.selected!;
	return candidateScopedArtifactMatches(selected.comparison_path, selected.candidate_id)
		&& candidateScopedArtifactMatches(selected.aligned_render_path, selected.candidate_id);
}

async function fixedFrameSnapshot(options: {
	worker: WorkerClient;
	workspace: string;
	controller: RunController;
	state: MainViewReflectState;
	region: DrawingManifestRegion;
	formal: CurrentFormalState;
}): Promise<MainViewReflectSnapshotRecord> {
	const selected = options.region.selected!;
	const frame = options.state.lockedManifest.projection_frame!;
	await validateFixedProjectionAnchor(options.workspace, options.state.lockedManifest);
	const calibration = confirmedScaleCalibration(options.state, options.region.region_id);
	const transform = calibration?.locked_transform ?? selected.transform;
	const id = snapshotId(options.region, options.formal, options.state);
	const frameSha256 = fixedFrameIdentity(options.state, options.region)!;
	const outputDir = join("drawing_feedback", "main-view-reflect", "snapshots", id);
	const fixed = await options.worker.request<WorkerFixedProjectionResponse>(
		options.workspace,
		{
			protocol_version: PROTOCOL_VERSION,
			operation: "render_fixed_projection",
			region_id: options.region.region_id,
			crop_path: options.region.crop_path,
			step_path: "output.step",
			projection_anchor_step_path: frame.anchor_step_path,
			view_spec: selected.view_spec,
			transform,
			output_dir: join(outputDir, "locked-frame"),
		},
		{ timeoutMs: 180_000, signal: options.controller.signal },
	);
	requireDrawingSuccess(fixed, "render_fixed_projection");
	const hasSection = selected.view_spec.section_offset_fraction !== undefined;
	if (
		fixed.region_id !== options.region.region_id || fixed.step_path !== "output.step" ||
		fixed.step_sha256 !== options.formal.stepSha256 ||
		fixed.projection_anchor_step_sha256 !== frame.anchor_step_sha256 ||
		!finiteVector3(fixed.projection_origin) ||
		(hasSection ? !finiteVector3(fixed.section_origin) : fixed.section_origin !== undefined) ||
		!sameCamera(fixed.view_spec, selected.view_spec) ||
		sha256Json(fixed.transform) !== sha256Json(transform)
	) throw new InfrastructureError("Locked-frame projection returned a mismatched STEP, anchor, camera, transform, or physical origin");
	for (const previous of options.state.snapshots.values()) {
		if (previous.fixed_frame_sha256 === frameSha256 && (
			sha256Json(previous.projection_origin) !== sha256Json(fixed.projection_origin) ||
			sha256Json(previous.section_origin ?? null) !== sha256Json(fixed.section_origin ?? null)
		)) throw new InfrastructureError("Locked-frame projection changed its physical origin or section plane");
	}
	const overview = await options.worker.request<WorkerReflectionInspectionResponse>(
		options.workspace,
		{
			protocol_version: PROTOCOL_VERSION,
			operation: "compose_reflection_inspection",
			region_id: options.region.region_id,
			crop_path: options.region.crop_path,
			aligned_render_path: fixed.render_path,
			mode: "coordinate_overview",
			output_dir: join(outputDir, "locked-frame-comparison"),
		},
		{ timeoutMs: 180_000, signal: options.controller.signal },
	);
	requireDrawingSuccess(overview, "compose_reflection_inspection");
	if (overview.mode !== "coordinate_overview" || overview.region_id !== options.region.region_id) {
		throw new InfrastructureError("Locked-frame comparison worker returned a mismatched overview");
	}
	const [crop, render, comparison] = await Promise.all([
		inspectPngFile(options.workspace, options.region.crop_path),
		inspectPngFile(options.workspace, fixed.render_path),
		inspectPngFile(options.workspace, overview.context_path),
	]);
	if (
		render.sha256 !== fixed.sha256 || comparison.sha256 !== overview.sha256.context ||
		render.width !== crop.width || render.height !== crop.height
	) throw new InfrastructureError("Locked-frame projection image hash or complete-crop size mismatch");
	return {
		snapshot_id: id,
		region_id: options.region.region_id,
		source_sha256: options.formal.sourceSha256,
		step_sha256: options.formal.stepSha256,
		camera_sha256: cameraSha256(selected.view_spec),
		locked_candidate_id: selected.candidate_id,
		view_spec: selected.view_spec,
		crop_path: options.region.crop_path,
		render_path: fixed.render_path,
		aligned_render_path: fixed.render_path,
		comparison_path: overview.context_path,
		sha256: { crop: crop.sha256, render: render.sha256, aligned_render: render.sha256, comparison: comparison.sha256 },
		// These remain the locked View Search diagnostics; no alignment is scored again.
		score: selected.score,
		eligible: selected.eligible,
		failed_gates: [...selected.failed_gates],
		transform,
		fixed_frame_sha256: frameSha256,
		projection_origin: fixed.projection_origin,
		projection_anchor_step_sha256: fixed.projection_anchor_step_sha256,
		...(fixed.section_origin ? { section_origin: fixed.section_origin } : {}),
		...(calibration ? { scale_calibration_ref: calibration.calibration_ref } : {}),
		reused_locked_artifact: false,
		created_at: isoNow(),
	};
}

async function refreshedSnapshot(options: {
	worker: WorkerClient;
	workspace: string;
	controller: RunController;
	state: MainViewReflectState;
	region: DrawingManifestRegion;
	formal: CurrentFormalState;
	lockComparisonScale?: boolean;
}): Promise<MainViewReflectSnapshotRecord> {
	if (options.state.lockedManifest.projection_frame) return await fixedFrameSnapshot(options);
	const selected = options.region.selected!;
	const id = snapshotId(options.region, options.formal, options.state);
	const candidateId = `fixed-${id}`;
	const lockedViewSpec: DrawingViewSpec = { ...selected.view_spec, name: candidateId };
	const outputDir = join("drawing_feedback", "main-view-reflect", "snapshots", id);
	const response = await options.worker.request<WorkerRankDrawingViewsResponse>(
		options.workspace,
		{
			protocol_version: PROTOCOL_VERSION,
			operation: "rank_drawing_views",
			region_id: options.region.region_id,
			crop_path: options.region.crop_path,
			step_path: "output.step",
			candidates: [{ candidate_id: candidateId, view_spec: lockedViewSpec }],
			top_k: 1,
			allow_reflection: false,
			max_rotation_deg: 6,
			min_roi_fill: 0.70,
			seed: 0,
			max_iterations: 45,
			output_dir: outputDir,
		},
		{ timeoutMs: 240_000, signal: options.controller.signal },
	);
	requireDrawingSuccess(response, "rank_drawing_views");
	const ranked = response.rankings[0];
	if (
		response.region_id !== options.region.region_id ||
		response.step_sha256 !== options.formal.stepSha256 ||
		response.rankings.length !== 1 ||
		response.scoring_mode !== "capacity_transport" ||
		response.scorer_profile !== "capacity_transport_v1" ||
		!ranked ||
		ranked.candidate_id !== candidateId ||
		!Number.isFinite(ranked.score) ||
		!sameCamera(ranked.view_spec, selected.view_spec) ||
		ranked.transform.reflected
	) {
		throw new InfrastructureError("Fixed-camera refresh returned a mismatched view or formal STEP");
	}
	const calibration = confirmedScaleCalibration(options.state, options.region.region_id);
	const lockedScalePxPerUnit = calibration?.locked_scale_px_per_unit
		?? (options.lockComparisonScale === false
			? undefined
			: selected.transform.scale_px_per_unit);
	let alignedRenderPath = ranked.aligned_render_path;
	let comparisonPath = ranked.comparison_path;
	let transform = ranked.transform;
	if (lockedScalePxPerUnit !== undefined) {
		transform = {
			...ranked.transform,
			scale_px_per_unit: lockedScalePxPerUnit,
		};
		const scaleLabel = calibration ? `scale-${calibration.calibration_ref}` : "locked-scale";
		const fixed = await options.worker.request<WorkerFixedProjectionResponse>(
			options.workspace,
			{
				protocol_version: PROTOCOL_VERSION,
				operation: "render_fixed_projection",
				region_id: options.region.region_id,
				crop_path: options.region.crop_path,
					step_path: "output.step",
					view_spec: lockedViewSpec,
					transform,
					output_dir: join(outputDir, scaleLabel),
				},
			{ timeoutMs: 180_000, signal: options.controller.signal },
		);
		requireDrawingSuccess(fixed, "render_fixed_projection");
		if (
			fixed.step_sha256 !== options.formal.stepSha256 ||
			cameraSha256(fixed.view_spec) !== cameraSha256(selected.view_spec) ||
			fixed.transform.scale_px_per_unit !== lockedScalePxPerUnit
		) throw new InfrastructureError("Scale-locked projection returned a mismatched view, STEP, or scale");
		const overview = await options.worker.request<WorkerReflectionInspectionResponse>(
			options.workspace,
			{
				protocol_version: PROTOCOL_VERSION,
				operation: "compose_reflection_inspection",
				region_id: options.region.region_id,
					crop_path: options.region.crop_path,
					aligned_render_path: fixed.render_path,
					mode: "coordinate_overview",
					output_dir: join(outputDir, `${scaleLabel}-comparison`),
			},
			{ timeoutMs: 180_000, signal: options.controller.signal },
		);
		requireDrawingSuccess(overview, "compose_reflection_inspection");
		if (overview.mode !== "coordinate_overview" || overview.region_id !== options.region.region_id) {
			throw new InfrastructureError("Scale-locked comparison worker returned a mismatched overview");
		}
		alignedRenderPath = fixed.render_path;
		comparisonPath = overview.context_path;
	}
	const [crop, render, alignedRender, comparison] = await Promise.all([
		inspectPngFile(options.workspace, options.region.crop_path),
		inspectPngFile(options.workspace, ranked.render_path),
		inspectPngFile(options.workspace, alignedRenderPath),
		inspectPngFile(options.workspace, comparisonPath),
	]);
	return {
		snapshot_id: id,
		region_id: options.region.region_id,
		source_sha256: options.formal.sourceSha256,
		step_sha256: options.formal.stepSha256,
		camera_sha256: cameraSha256(selected.view_spec),
		locked_candidate_id: selected.candidate_id,
		view_spec: lockedViewSpec,
		crop_path: options.region.crop_path,
		render_path: ranked.render_path,
		aligned_render_path: alignedRenderPath,
		comparison_path: comparisonPath,
		sha256: {
			crop: crop.sha256,
			render: render.sha256,
			aligned_render: alignedRender.sha256,
			comparison: comparison.sha256,
		},
		score: ranked.score,
		eligible: ranked.eligible,
		failed_gates: [...ranked.failed_gates],
		transform,
		...(lockedScalePxPerUnit !== undefined ? {
			free_fit_scale_px_per_unit: ranked.transform.scale_px_per_unit,
		} : {}),
		...(calibration ? { scale_calibration_ref: calibration.calibration_ref } : {}),
		reused_locked_artifact: false,
		created_at: isoNow(),
	};
}

async function snapshotForFormal(options: {
	worker: WorkerClient;
	workspace: string;
	controller: RunController;
	state: MainViewReflectState;
	region: DrawingManifestRegion;
	formal: CurrentFormalState;
}): Promise<MainViewReflectSnapshotRecord> {
	await validateFixedProjectionAnchor(options.workspace, options.state.lockedManifest);
	const id = snapshotId(options.region, options.formal, options.state);
	const frameSha256 = fixedFrameIdentity(options.state, options.region);
	const calibration = confirmedScaleCalibration(options.state, options.region.region_id);
	const lockedArtifactsAreConsistent = lockedSelectedArtifactsAreConsistent(options.region);
	const cached = options.state.snapshots.get(id);
	const fixedCacheMatches = !frameSha256 || (cached !== undefined &&
		cached.fixed_frame_sha256 === frameSha256 &&
		cached.camera_sha256 === cameraSha256(options.region.selected!.view_spec) &&
		sha256Json(cached.transform ?? null) === sha256Json(calibration?.locked_transform ?? options.region.selected!.transform) &&
		cached.projection_anchor_step_sha256 === options.state.lockedManifest.projection_frame!.anchor_step_sha256 &&
		finiteVector3(cached.projection_origin) &&
		(options.region.selected!.view_spec.section_offset_fraction !== undefined
			? finiteVector3(cached.section_origin)
			: cached.section_origin === undefined)
	);
	if (cached && frameSha256 && !fixedCacheMatches) clearRegionVisualCache(options.state, options.region.region_id);
	if (
		cached &&
		cached.fixed_frame_sha256 === frameSha256 &&
		fixedCacheMatches &&
		cached.scale_calibration_ref === calibration?.calibration_ref &&
		(!cached.reused_locked_artifact || lockedArtifactsAreConsistent)
	) {
		const [crop, render, alignedRender, comparison] = await Promise.all([
			inspectPngFile(options.workspace, cached.crop_path),
			inspectPngFile(options.workspace, cached.render_path),
			inspectPngFile(options.workspace, cached.aligned_render_path),
			inspectPngFile(options.workspace, cached.comparison_path),
		]);
		if (
			crop.sha256 !== cached.sha256.crop ||
			render.sha256 !== cached.sha256.render ||
			alignedRender.sha256 !== cached.sha256.aligned_render ||
			comparison.sha256 !== cached.sha256.comparison
		) throw new InfrastructureError("Cached fixed-view snapshot image hash mismatch");
		return cached;
	}
	const isBaseline = options.formal.stepSha256 === options.state.baselineStepSha256;
	const snapshot = !frameSha256 && isBaseline && lockedArtifactsAreConsistent && !calibration
		? await baselineSnapshot(options)
		: await refreshedSnapshot(options);
	options.state.snapshots.set(snapshot.snapshot_id, snapshot);
	return snapshot;
}

function persistedAudit(state: MainViewReflectState): PersistedMainViewReflectAudit {
	return {
		version: 1,
		kind: MAIN_VIEW_REFLECT_KIND,
		fixture_id: state.fixtureId,
		locked_view_manifest_path: state.lockedManifestPath,
		locked_view_manifest_sha256: state.lockedManifestSha256,
		baseline_source_sha256: state.baselineSourceSha256,
		baseline_step_sha256: state.baselineStepSha256,
		started_at: state.startedAt,
		updated_at: isoNow(),
		snapshots: [...state.snapshots.values()],
		inspections: [...state.inspections.values()],
		inspection_attempts: state.inspectionAttempts,
		crops: [...state.crops.values()],
		crop_attempts: state.cropAttempts,
		scale_calibrations: state.scaleCalibrations,
	};
}

export async function persistMainViewReflectState(workspace: string, state: MainViewReflectState): Promise<void> {
	const path = resolve(workspace, state.auditPath);
	const root = resolve(workspace);
	const rel = relative(root, path);
	if (rel === ".." || rel.startsWith("../") || isAbsolute(rel)) {
		throw new InfrastructureError("Main View Reflect audit path escaped the fixture workspace");
	}
	await mkdir(dirname(path), { recursive: true });
	await writeJsonAtomic(path, persistedAudit(state));
}

// Optional Reflect callers pass the full session journal, including entries before compaction.
// An audit written before a failed/interrupted tool return is not completion evidence.
export async function reconcileMainViewInspectionEvidence(
	workspace: string,
	state: MainViewReflectState,
	entries: readonly SessionEntry[],
): Promise<void> {
	const successfulReturns = new Set<string>();
	const evidenceKey = (toolCallId: string, inspectionId: string, stepSha256: string) => (
		JSON.stringify([toolCallId, inspectionId, stepSha256])
	);
	for (const entry of entries) {
		if (entry.type !== "message") continue;
		const message = entry.message;
		if (
			message.role !== "toolResult" ||
			message.toolName !== "inspect_comparison" ||
			message.isError !== false ||
			!message.content.some((block) => block.type === "image" && block.data.length > 0)
		) continue;
		const details = message.details as { inspection_id?: unknown; step_sha256?: unknown } | undefined;
		if (typeof details?.inspection_id !== "string" || typeof details.step_sha256 !== "string") continue;
		successfulReturns.add(evidenceKey(message.toolCallId, details.inspection_id, details.step_sha256));
	}
	const hasSuccessfulReturn = (record: MainViewReflectInspectionRecord | MainViewReflectInspectionAttempt) => (
		typeof record.tool_call_id === "string" &&
		record.tool_call_id.length > 0 &&
		successfulReturns.has(evidenceKey(record.tool_call_id, record.inspection_id, record.step_sha256))
	);
	let changed = false;
	for (const [id, record] of state.inspections) {
		if (hasSuccessfulReturn(record)) continue;
		state.inspections.delete(id);
		changed = true;
	}
	const attempts = state.inspectionAttempts.filter((attempt) => (
		state.inspections.has(attempt.inspection_id) && hasSuccessfulReturn(attempt)
	));
	if (attempts.length !== state.inspectionAttempts.length) {
		state.inspectionAttempts = attempts.map((attempt, index) => ({ ...attempt, sequence: index + 1 }));
		changed = true;
	}
	if (changed) await persistMainViewReflectState(workspace, state);
}

export function mainViewReflectSummary(state: MainViewReflectState): MainViewReflectSummary {
	const seen = new Set<string>();
	const inspectedFormalStates = [...state.inspections.values()].flatMap(({ source_sha256, step_sha256 }) => {
		if (seen.has(step_sha256)) return [];
		seen.add(step_sha256);
		return [{ source_sha256, step_sha256 }];
	});
	const croppedSeen = new Set<string>();
	const croppedFormalStates = [...state.crops.values()].flatMap(({ source_sha256, step_sha256 }) => {
		if (croppedSeen.has(step_sha256)) return [];
		croppedSeen.add(step_sha256);
		return [{ source_sha256, step_sha256 }];
	});
	return {
		version: 1,
		kind: MAIN_VIEW_REFLECT_KIND,
		locked_view_manifest_sha256: state.lockedManifestSha256,
		baseline_source_sha256: state.baselineSourceSha256,
		baseline_step_sha256: state.baselineStepSha256,
		audit_manifest_path: state.auditPath,
		inspection_count: state.inspectionAttempts.length,
		crop_count: state.cropAttempts.length,
		inspected_formal_states: inspectedFormalStates,
		cropped_formal_states: croppedFormalStates,
	};
}

export function summaryHasInspectedStep(
	summary: MainViewReflectSummary | undefined,
	stepSha256: string | null,
): boolean {
	if (!summary || !stepSha256) return false;
	return summary.inspected_formal_states.some((state) => state.step_sha256 === stepSha256);
}

export function summaryHasInspectedFormalState(
	summary: MainViewReflectSummary | undefined,
	_sourceSha256: string | null,
	stepSha256: string | null,
): boolean {
	return summaryHasInspectedStep(summary, stepSha256);
}

export function summaryHasCroppedStep(
	summary: MainViewReflectSummary | undefined,
	stepSha256: string | null,
): boolean {
	if (!summary || !stepSha256) return false;
	return (summary.cropped_formal_states ?? []).some((state) => state.step_sha256 === stepSha256);
}

export function summaryHasCroppedFormalState(
	summary: MainViewReflectSummary | undefined,
	_sourceSha256: string | null,
	stepSha256: string | null,
): boolean {
	return summaryHasCroppedStep(summary, stepSha256);
}

export async function restoreMainViewReflectState(options: {
	workspace: string;
	fixtureId: string;
	lockedManifest: DrawingViewManifest;
	lockedManifestPath: string;
	lockedManifestSha256: string;
	baselineSourceSha256: string;
	baselineStepSha256: string;
	auditPath?: string;
}): Promise<MainViewReflectState> {
	const auditPath = options.auditPath ?? MAIN_VIEW_REFLECT_AUDIT_PATH;
	const path = resolve(options.workspace, auditPath);
	const stored = await readJson<PersistedMainViewReflectAudit>(path).catch(() => null);
	if (!stored) {
		return {
			fixtureId: options.fixtureId,
			lockedManifest: options.lockedManifest,
			lockedManifestPath: options.lockedManifestPath,
			lockedManifestSha256: options.lockedManifestSha256,
			baselineSourceSha256: options.baselineSourceSha256,
			baselineStepSha256: options.baselineStepSha256,
			auditPath,
			startedAt: isoNow(),
			snapshots: new Map(),
			inspections: new Map(),
			inspectionAttempts: [],
			crops: new Map(),
			cropAttempts: [],
			scaleCalibrations: [],
		};
	}
	if (
		stored.version !== 1 ||
		stored.kind !== MAIN_VIEW_REFLECT_KIND ||
		stored.fixture_id !== options.fixtureId ||
		stored.locked_view_manifest_path !== options.lockedManifestPath ||
		stored.locked_view_manifest_sha256 !== options.lockedManifestSha256 ||
		stored.baseline_source_sha256 !== options.baselineSourceSha256 ||
		stored.baseline_step_sha256 !== options.baselineStepSha256 ||
		!Array.isArray(stored.snapshots) ||
		!Array.isArray(stored.inspections) ||
		!Array.isArray(stored.inspection_attempts) ||
		!Array.isArray(stored.crops) ||
		!Array.isArray(stored.crop_attempts) ||
		(stored.scale_calibrations !== undefined && !Array.isArray(stored.scale_calibrations))
	) throw new InfrastructureError("Persisted Main View Reflect audit does not match the locked continuation");
	const scaleCalibrations = stored.scale_calibrations ?? [];
	if (scaleCalibrations.some((record) => (
			!/^S[0-9]{3,}$/u.test(record.calibration_ref) ||
			!options.lockedManifest.regions.some(({ region_id }) => region_id === record.region_id) ||
			!DRAWING_SCALE_BOUNDARIES.includes(record.boundary) ||
			!["proposed", "confirmed", "superseded", "deleted"].includes(record.status) ||
			!Number.isFinite(record.target_dimension_mm) ||
			!Number.isFinite(record.current_dimension_units) ||
			!Number.isFinite(record.free_fit_scale_px_per_unit) ||
			!Number.isFinite(record.locked_scale_px_per_unit) ||
			!Number.isFinite(record.scale_ratio) ||
			record.target_dimension_mm <= 0 ||
			record.current_dimension_units <= 0 ||
			record.free_fit_scale_px_per_unit <= 0 ||
			record.locked_scale_px_per_unit <= 0 ||
			record.scale_ratio <= 0 ||
			!/^[a-f0-9]{64}$/u.test(record.source_sha256) ||
			!/^[a-f0-9]{64}$/u.test(record.step_sha256) ||
			!/^[a-f0-9]{64}$/u.test(record.camera_sha256) ||
			!/^[a-f0-9]{64}$/u.test(record.preview_sha256)
	))) throw new InfrastructureError("Persisted drawing-scale calibration is invalid");
	const confirmedRegions = scaleCalibrations
		.filter(({ status }) => status === "confirmed")
		.map(({ region_id }) => region_id);
	if (new Set(confirmedRegions).size !== confirmedRegions.length) {
		throw new InfrastructureError("Persisted drawing-scale calibrations contain multiple confirmed scales for one region");
	}
	return {
		fixtureId: options.fixtureId,
		lockedManifest: options.lockedManifest,
		lockedManifestPath: options.lockedManifestPath,
		lockedManifestSha256: options.lockedManifestSha256,
		baselineSourceSha256: options.baselineSourceSha256,
		baselineStepSha256: options.baselineStepSha256,
		auditPath,
		startedAt: stored.started_at,
		snapshots: new Map(stored.snapshots.map((record) => [record.snapshot_id, record])),
		inspections: new Map(stored.inspections.map((record) => [record.inspection_id, record])),
		inspectionAttempts: stored.inspection_attempts,
		crops: new Map(stored.crops.map((record) => [record.crop_id, record])),
		cropAttempts: stored.crop_attempts,
		scaleCalibrations,
	};
}

export function mainViewReflectContinuationPrompt(
	fixtureId: string,
	manifest: DrawingViewManifest,
): string {
	const regions = manifest.regions.map((region) => ({
		region_id: region.region_id,
		view_type: region.view_type,
		status: region.status,
		confidence: region.confidence,
		note: region.note,
		camera_sha256: region.selected ? cameraSha256(region.selected.view_spec) : null,
	}));
	return [
		`Re-evaluate the current CAD for CADGenBench fixture ${fixtureId} against the input engineering drawings and the available visual comparisons.`,
		"The controller preserved the Direct baseline under /workspace/baseline/ and imported a hash-locked View Search manifest. Never edit baseline/ or drawing_feedback/.",
		"Your task is to build reliable global and local visual evidence about the differences between the current CAD and the input engineering drawings, then decide whether solution.py should change. Use only images returned by inspect_comparison and dual_roi_crop as CAD visual evidence; standalone 3D or isometric renders are outside this experiment.",
		`Locked comparison regions:\n${JSON.stringify(regions, null, 2)}`,
		"Before using pixel proportions to judge an exact overall size, call calibrate_drawing_scale with an exact overall width or height read from the original engineering drawing. Inspect its preview and confirm the calibration_ref only when that dimension spans the complete projected boundary. A confirmed calibration fixes the displayed CAD scale for that region across later executions; it does not establish local feature correspondence.",
		"First establish the global comparison with inspect_comparison({ region_id }) across the regions needed to understand the whole part. Compare silhouette, proportions, feature presence and count, placement, connectivity, symmetry, and section structure. Treat ambiguous regions as uncertain evidence rather than proof of an error.",
		"Next use dual_roi_crop on the corresponding Original and CAD features that are important to the decision. Verify holes, slots, bosses, ribs, steps, thicknesses, edge profiles, radii, and intersections at local scale. Match features by their complete-view position and connected geometry, and use global context together with local detail before drawing a conclusion.",
		"Then decide whether each observed difference is a reliable geometric discrepancy. Before editing, identify which parameter or construction in solution.py causes it, how that code should change, and how the change should affect every relevant view. Preserve geometry that is already correct; if no reliable actionable discrepancy remains, retain the Direct output.",
		"When a change is supported, edit the live solution.py and call execute_build123d({}) when you are ready to formalize the edited geometry. Editing source alone does not invalidate visual evidence for the existing STEP. If formal execution changes the STEP, inspect the updated CAD globally and locally again, verify the intended improvement and check for regressions, then revise and repeat for as many rounds as needed. If execution produces the same STEP, its existing visual evidence remains current. Finish only when the current formal STEP is valid, watertight, and that exact STEP has both a complete comparison inspection and a corresponding local comparison.",
	].join("\n\n");
}

export function createMainViewReflectTools(options: {
	workspace: string;
	state: MainViewReflectState;
	worker: WorkerClient;
	controller: RunController;
	getResult: () => AgentResult;
	onUpdate: (summary: MainViewReflectSummary) => Promise<void>;
	targetAnnotationState?: TargetAnnotationState;
	dimensionComparison?: (regionId: string) => Promise<{ content: Array<{ type: "text"; text: string } | ImageContent> }>;
}) {
	const availableRegionIds = options.state.lockedManifest.regions
		.filter((region) => region.selected !== undefined)
		.map((region) => region.region_id);
	const renderAnnotatedComparison = async (
		region: DrawingManifestRegion,
		snapshot: MainViewReflectSnapshotRecord,
		annotations: readonly TargetAnnotation[],
	): Promise<ImageContent> => {
		const regionAnnotations = annotations.filter(({ region_id }) => region_id === region.region_id);
		const renderedAnnotations = regionAnnotations.map(({ mark, name, geometry }) => ({
			mark,
			name,
			...(geometry.kind === "point" ? { point: geometry.point } : { box: geometry.box }),
		}));
		const renderId = sha256Json([
			"target-annotations",
			snapshot.snapshot_id,
			renderedAnnotations,
		]).slice(0, 20);
		const response = await options.worker.request<WorkerReflectionInspectionResponse>(
			options.workspace,
			{
				protocol_version: PROTOCOL_VERSION,
				operation: "compose_reflection_inspection",
				region_id: region.region_id,
				crop_path: region.crop_path,
				aligned_render_path: snapshot.aligned_render_path,
				mode: "coordinate_overview",
				target_annotations: renderedAnnotations,
				output_dir: join("drawing_feedback", "target-annotations", "renders", renderId),
			},
			{ timeoutMs: 180_000, signal: options.controller.signal },
		);
		requireDrawingSuccess(response, "compose_reflection_inspection");
		if (response.mode !== "coordinate_overview" || response.region_id !== region.region_id) {
			throw new InfrastructureError("Target annotation worker returned a mismatched comparison overview");
		}
		const context = await inspectPngFile(options.workspace, response.context_path);
		if (context.sha256 !== response.sha256.context) {
			throw new InfrastructureError("Target annotation overview hash mismatch");
		}
		return await imageFromWorkspace(options.workspace, response.context_path);
	};
	const inspectTool = defineTool({
		name: "inspect_comparison",
		label: "Inspect fixed-view comparison",
		description:
			`Inspect the 2D comparison for one engineering-drawing region. The left side shows the input Original drawing, and the right side shows the aligned 2D projection of the CAD generated by your execution. Use this image to identify similarities and differences between the current CAD and the input drawing. Available region_id values: ${availableRegionIds.join(", ")}.`,
		parameters: INSPECT_COMPARISON_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(toolCallId, params) {
			const region = regionForInspection(options.state, params.region_id);
			const formal = await currentFormalState(options);
			const beforeDimensions = options.dimensionComparison
				? await Promise.all(["solution.py", "output.step"].map(file => sha256File(join(options.workspace, file))))
				: undefined;
			const dimensionEvidence = options.dimensionComparison ? await options.dimensionComparison(region.region_id) : undefined;
			const refreshedFormal = dimensionEvidence ? await currentFormalState(options) : formal;
			if (beforeDimensions) {
				const afterDimensions = await Promise.all(["solution.py", "output.step"].map(file => sha256File(join(options.workspace, file))));
				if (beforeDimensions.some((hash, index) => !hash || hash !== afterDimensions[index])) throw new InfrastructureError("CAD changed during dimension annotation");
			}
			if (refreshedFormal.sourceSha256 !== formal.sourceSha256 || refreshedFormal.stepSha256 !== formal.stepSha256) throw new InfrastructureError("CAD changed during dimension annotation");
			const snapshot = await snapshotForFormal({ ...options, region, formal });
			const inspectionId = sha256Json([
				MAIN_VIEW_REFLECT_KIND,
				"inspection",
				snapshot.snapshot_id,
			]).slice(0, 20);
			let record = options.state.inspections.get(inspectionId);
			const cacheHit = record !== undefined;
			if (!record) {
				const response = await options.worker.request<WorkerReflectionInspectionResponse>(
					options.workspace,
					{
						protocol_version: PROTOCOL_VERSION,
						operation: "compose_reflection_inspection",
						region_id: region.region_id,
						crop_path: region.crop_path,
						aligned_render_path: snapshot.aligned_render_path,
						mode: "coordinate_overview",
						output_dir: join("drawing_feedback", "main-view-reflect", "inspections", inspectionId),
					},
					{ timeoutMs: 180_000, signal: options.controller.signal },
				);
				requireDrawingSuccess(response, "compose_reflection_inspection");
				if (response.mode !== "coordinate_overview" || response.region_id !== region.region_id) {
					throw new InfrastructureError("Fixed-view comparison worker returned a mismatched coordinate overview");
				}
				const context = await inspectPngFile(options.workspace, response.context_path);
				if (context.sha256 !== response.sha256.context) {
					throw new InfrastructureError("Fixed-view comparison overview hash mismatch");
				}
				record = {
					inspection_id: inspectionId,
					snapshot_id: snapshot.snapshot_id,
					region_id: region.region_id,
					source_sha256: formal.sourceSha256,
					step_sha256: formal.stepSha256,
					camera_sha256: snapshot.camera_sha256,
					context_path: response.context_path,
					context_sha256: context.sha256,
					created_at: isoNow(),
				};
			} else {
				await requireSafeFile(options.workspace, record.context_path);
			}
			const annotations = targetAnnotationsForRegion(options.targetAnnotationState, region.region_id);
			const comparisonImage = annotations.length === 0
				? await imageFromWorkspace(options.workspace, record.context_path)
				: await renderAnnotatedComparison(region, snapshot, annotations);
			// Only a comparison with a prepared image can count as completion evidence.
			const previousRecord = options.state.inspections.get(inspectionId);
			const previousAttempts = [...options.state.inspectionAttempts];
			record = { ...record, tool_call_id: toolCallId };
			options.state.inspections.set(inspectionId, record);
			options.state.inspectionAttempts.push({
				sequence: options.state.inspectionAttempts.length + 1,
				inspection_id: inspectionId,
				tool_call_id: toolCallId,
				region_id: region.region_id,
				source_sha256: formal.sourceSha256,
				step_sha256: formal.stepSha256,
				cache_hit: cacheHit,
				created_at: isoNow(),
			});
			try {
				await persistMainViewReflectState(options.workspace, options.state);
				await options.onUpdate(mainViewReflectSummary(options.state));
			} catch (error) {
				if (previousRecord) options.state.inspections.set(inspectionId, previousRecord);
				else options.state.inspections.delete(inspectionId);
				options.state.inspectionAttempts = previousAttempts;
				// Preserve the original error even if the same storage failure affects rollback.
				await persistMainViewReflectState(options.workspace, options.state).catch(() => undefined);
				await options.onUpdate(mainViewReflectSummary(options.state)).catch(() => undefined);
				throw error;
			}
			const ambiguity = region.status === "ambiguous"
				? "This View Search result is ambiguous; treat differences as uncertain evidence."
				: "";
			const content: Array<{ type: "text"; text: string } | ImageContent> = [
				{
					type: "text",
					text: `${region.region_id}: LEFT is the Original drawing and RIGHT is the current CAD. Each panel has its own 0-1000 coordinates.${annotations.length > 0 ? " Persistent numeric Target marks are shown only on the LEFT panel." : ""} ${ambiguity}`.trim(),
				},
				comparisonImage,
				...(dimensionEvidence?.content ?? []),
			];
			return { content, details: { ...record, snapshot } };
		},
	});

	const cropTool = defineTool({
		name: "dual_roi_crop",
		label: "Dual ROI Crop",
		description:
			`查看原始工程图与当前 CAD 二维投影中相互对应的局部细节。分别指定两侧需要放大的区域，以检查局部结构的形状、位置、方向和连接关系。两个 ROI 使用各自面板独立的 0–1000 坐标。可用的 \`region_id\`：\`${availableRegionIds.join(", ")}\``,
		parameters: DUAL_ROI_CROP_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			const region = regionForInspection(options.state, params.region_id);
			const formal = await currentFormalState(options);
			const currentInspection = [...options.state.inspections.values()].find(
				(record) =>
					record.region_id === region.region_id &&
					record.step_sha256 === formal.stepSha256,
			);
			if (!currentInspection) {
				throw new Error(`Establish the current complete-view context for ${region.region_id} with inspect_comparison, then request its corresponding local evidence`);
			}
			const snapshot = await snapshotForFormal({ ...options, region, formal });
			const inspectionId = sha256Json([
				MAIN_VIEW_REFLECT_KIND,
				"inspection",
				snapshot.snapshot_id,
			]).slice(0, 20);
			if (!options.state.inspections.has(inspectionId)) throw new InfrastructureError("Current inspection cache is inconsistent");
			const referenceRoi = roiFromObject(params.reference_roi);
			const cadRoi = roiFromObject(params.cad_roi);
			const cropId = sha256Json([
				MAIN_VIEW_REFLECT_KIND,
				"dual-roi",
				snapshot.snapshot_id,
				referenceRoi,
				cadRoi,
			]).slice(0, 20);
			let record = options.state.crops.get(cropId);
			const cacheHit = record !== undefined;
			if (!record) {
				const [referenceSource, cadSource] = await Promise.all([
					inspectPngFile(options.workspace, region.crop_path),
					inspectPngFile(options.workspace, snapshot.aligned_render_path),
				]);
				const response = await options.worker.request<WorkerReflectionInspectionResponse>(
					options.workspace,
					{
						protocol_version: PROTOCOL_VERSION,
						operation: "compose_reflection_inspection",
						region_id: region.region_id,
						crop_path: region.crop_path,
						aligned_render_path: snapshot.aligned_render_path,
						mode: "dual_roi",
						reference_roi: referenceRoi,
						cad_roi: cadRoi,
						output_dir: join("drawing_feedback", "main-view-reflect", "crops", cropId),
					},
					{ timeoutMs: 180_000, signal: options.controller.signal },
				);
				requireDrawingSuccess(response, "compose_reflection_inspection");
				if (
					response.mode !== "dual_roi" ||
					response.region_id !== region.region_id ||
					!sameRoi(response.reference_roi, referenceRoi) ||
					!sameRoi(response.cad_roi, cadRoi)
				) throw new InfrastructureError("Dual ROI worker returned a mismatched crop");
				const [referenceCrop, cadCrop, context, comparison] = await Promise.all([
					inspectPngFile(options.workspace, response.reference_crop_path),
					inspectPngFile(options.workspace, response.cad_crop_path),
					inspectPngFile(options.workspace, response.context_path),
					inspectPngFile(options.workspace, response.comparison_path),
				]);
				if (
					referenceCrop.sha256 !== response.sha256.reference_crop ||
					cadCrop.sha256 !== response.sha256.cad_crop ||
					context.sha256 !== response.sha256.context ||
					comparison.sha256 !== response.sha256.comparison
				) throw new InfrastructureError("Dual ROI artifact hash mismatch");
				record = {
					crop_id: cropId,
					snapshot_id: snapshot.snapshot_id,
					region_id: region.region_id,
					source_sha256: formal.sourceSha256,
					step_sha256: formal.stepSha256,
					camera_sha256: snapshot.camera_sha256,
					reference_roi: referenceRoi,
					cad_roi: cadRoi,
					reference_pixel_roi: response.reference_pixel_roi,
					cad_pixel_roi: response.cad_pixel_roi,
					reference_crop_path: response.reference_crop_path,
					cad_crop_path: response.cad_crop_path,
					context_path: response.context_path,
					comparison_path: response.comparison_path,
					sha256: {
						reference_source: referenceSource.sha256,
						cad_source: cadSource.sha256,
						reference_crop: referenceCrop.sha256,
						cad_crop: cadCrop.sha256,
						context: context.sha256,
						comparison: comparison.sha256,
					},
					created_at: isoNow(),
				};
				options.state.crops.set(cropId, record);
			} else {
				await requireSafeFile(options.workspace, record.comparison_path);
			}
			options.state.cropAttempts.push({
				sequence: options.state.cropAttempts.length + 1,
				crop_id: cropId,
				region_id: region.region_id,
				source_sha256: formal.sourceSha256,
				step_sha256: formal.stepSha256,
				cache_hit: cacheHit,
				created_at: isoNow(),
			});
			await persistMainViewReflectState(options.workspace, options.state);
			await options.onUpdate(mainViewReflectSummary(options.state));
			return {
				content: [
					{
						type: "text" as const,
						text: `以下图像对比了 \`${region.region_id}\` 中原始工程图与当前 CAD 的对应局部。上排红框标出了两个 ROI 的位置，下排展示放大结果。在使用这些图像判断缺陷之前，请先确认：（1）两侧 ROI 是否对应同一几何特征；（2）当前视觉证据是否足以确认当前 CAD 相比原始工程图存在明确的形状错误、位置偏差、方向错误或连接关系不一致。若两侧不是同一几何特征，请调整 ROI 后重新调用 \`dual_roi_crop\`；若两侧对应同一特征但视觉证据仍不充分，也请调整或扩大 ROI 后重新检查。不要依据不充分的局部证据推断缺陷。涉及具体尺寸修改时，仍需仔细查看原始输入工程图中的尺寸标注。`,
					},
					await imageFromWorkspace(options.workspace, record.comparison_path),
				],
				details: record,
			};
		},
	});

	const scaleTool = defineTool({
		name: "calibrate_drawing_scale",
		label: "Calibrate drawing scale",
		description:
			`Calibrate and persist a fixed pixel-per-millimetre scale for one orthographic comparison region without selecting target pixels. For a proposal, read an exact overall width or height from the original engineering drawing and provide only region_id, boundary, and target_dimension_mm. The tool measures the current CAD projection in model units, uses the existing free-fit transform as the image-scale bridge, and returns a preview. Inspect that preview, then confirm its calibration_ref. Confirmed scale remains fixed across later CAD executions; only translation and small rotation are refreshed. Delete a calibration to return that region to free-scale alignment. Use only a dimension that spans the complete projected boundary. Available region_id values: ${availableRegionIds.join(", ")}.`,
		parameters: CALIBRATE_DRAWING_SCALE_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			if (params.operation === "propose") {
				const region = regionForInspection(options.state, params.region_id);
				if (region.view_type !== "orthographic") {
					throw new Error("Drawing-scale calibration requires an orthographic region");
				}
				if (confirmedScaleCalibration(options.state, region.region_id)) {
					throw new Error(`Region ${region.region_id} already has a confirmed scale; delete it before recalibrating`);
				}
				const formal = await currentFormalState(options);
				const freeSnapshot = await refreshedSnapshot({
					...options,
					region,
					formal,
					lockComparisonScale: false,
				});
				if (!freeSnapshot.transform) throw new InfrastructureError("Free-fit snapshot has no transform");
				const projectionFrame = options.state.lockedManifest.projection_frame;
				const anchorRequest = projectionFrame
					? { projection_anchor_step_path: projectionFrame.anchor_step_path }
					: {};
				const validateScaleProjection = (response: WorkerFixedProjectionResponse, transform: DrawingSimilarityTransform): void => {
					if (projectionFrame && (
						response.region_id !== region.region_id || response.step_sha256 !== formal.stepSha256 ||
						response.projection_anchor_step_sha256 !== projectionFrame.anchor_step_sha256 ||
						!sameCamera(response.view_spec, freeSnapshot.view_spec) ||
						sha256Json(response.transform) !== sha256Json(transform) ||
						sha256Json(response.projection_origin ?? null) !== sha256Json(freeSnapshot.projection_origin ?? null) ||
						sha256Json(response.section_origin ?? null) !== sha256Json(freeSnapshot.section_origin ?? null)
					)) throw new InfrastructureError("Drawing-scale projection drifted from the locked physical frame");
				};
				const nextIndex = options.state.scaleCalibrations.reduce((maximum, record) => {
					const parsed = Number.parseInt(record.calibration_ref.slice(1), 10);
					return Number.isFinite(parsed) ? Math.max(maximum, parsed) : maximum;
				}, 0) + 1;
				const calibrationRef = `S${String(nextIndex).padStart(3, "0")}`;
				const outputDir = join(
					"drawing_feedback",
					"drawing-scale",
					calibrationRef,
				);
				const freeMeasurement = await options.worker.request<WorkerFixedProjectionResponse>(
					options.workspace,
					{
						protocol_version: PROTOCOL_VERSION,
						operation: "render_fixed_projection",
						region_id: region.region_id,
						crop_path: region.crop_path,
						step_path: "output.step",
						...anchorRequest,
						view_spec: freeSnapshot.view_spec,
						transform: freeSnapshot.transform,
						output_dir: join(outputDir, "free-measurement"),
					},
					{ timeoutMs: 180_000, signal: options.controller.signal },
				);
				requireDrawingSuccess(freeMeasurement, "render_fixed_projection");
				validateScaleProjection(freeMeasurement, freeSnapshot.transform);
				const currentDimension = params.boundary === "overall_width"
					? freeMeasurement.projection_width_units
					: freeMeasurement.projection_height_units;
				if (!currentDimension || !Number.isFinite(currentDimension) || currentDimension <= 0) {
					throw new InfrastructureError("Fixed projection did not return a valid CAD boundary measurement");
				}
				const scaleRatio = currentDimension / params.target_dimension_mm;
				const lockedScale = freeSnapshot.transform.scale_px_per_unit * scaleRatio;
				if (!Number.isFinite(lockedScale) || lockedScale <= 0 || scaleRatio < 0.05 || scaleRatio > 20) {
					throw new Error("The proposed target dimension produces an implausible scale ratio");
				}
				const lockedTransform: DrawingSimilarityTransform = {
					...freeSnapshot.transform,
					scale_px_per_unit: lockedScale,
				};
				const lockedRender = await options.worker.request<WorkerFixedProjectionResponse>(
					options.workspace,
					{
						protocol_version: PROTOCOL_VERSION,
						operation: "render_fixed_projection",
						region_id: region.region_id,
						crop_path: region.crop_path,
						step_path: "output.step",
						...anchorRequest,
						view_spec: freeSnapshot.view_spec,
						transform: lockedTransform,
						output_dir: join(outputDir, "locked-render"),
					},
					{ timeoutMs: 180_000, signal: options.controller.signal },
				);
				requireDrawingSuccess(lockedRender, "render_fixed_projection");
				validateScaleProjection(lockedRender, lockedTransform);
				if (
					freeMeasurement.step_sha256 !== formal.stepSha256 ||
					lockedRender.step_sha256 !== formal.stepSha256 ||
					lockedRender.transform.scale_px_per_unit !== lockedScale
				) throw new InfrastructureError("Drawing-scale calibration drifted from the current formal STEP or requested scale");
				const preview = await options.worker.request<WorkerReflectionInspectionResponse>(
					options.workspace,
					{
						protocol_version: PROTOCOL_VERSION,
						operation: "compose_reflection_inspection",
						region_id: region.region_id,
						crop_path: region.crop_path,
						aligned_render_path: lockedRender.render_path,
						mode: "coordinate_overview",
						output_dir: join(outputDir, "preview"),
					},
					{ timeoutMs: 180_000, signal: options.controller.signal },
				);
				requireDrawingSuccess(preview, "compose_reflection_inspection");
				if (preview.mode !== "coordinate_overview" || preview.region_id !== region.region_id) {
					throw new InfrastructureError("Drawing-scale preview returned a mismatched overview");
				}
				const previewImage = await inspectPngFile(options.workspace, preview.context_path);
				if (previewImage.sha256 !== preview.sha256.context) {
					throw new InfrastructureError("Drawing-scale preview hash mismatch");
				}
				const now = isoNow();
				const record: DrawingScaleCalibrationRecord = {
					calibration_ref: calibrationRef,
					region_id: region.region_id,
					boundary: params.boundary,
					status: "proposed",
					target_dimension_mm: params.target_dimension_mm,
					current_dimension_units: currentDimension,
					free_fit_scale_px_per_unit: freeSnapshot.transform.scale_px_per_unit,
					locked_scale_px_per_unit: lockedScale,
					scale_ratio: scaleRatio,
					source_sha256: formal.sourceSha256,
					step_sha256: formal.stepSha256,
					camera_sha256: freeSnapshot.camera_sha256,
					free_transform: freeSnapshot.transform,
					locked_transform: lockedTransform,
					preview_path: preview.context_path,
					preview_sha256: previewImage.sha256,
					created_at: now,
					updated_at: now,
				};
				options.state.scaleCalibrations.push(record);
				await persistMainViewReflectState(options.workspace, options.state);
				return {
					content: [
						{
							type: "text" as const,
							text: `Proposed ${calibrationRef} for ${region.region_id}: the current CAD ${params.boundary} is ${currentDimension.toFixed(3)} model units; the exact target boundary supplied from the original drawing is ${params.target_dimension_mm.toFixed(3)} mm. Free-fit scale ${freeSnapshot.transform.scale_px_per_unit.toFixed(6)} px/unit becomes fixed scale ${lockedScale.toFixed(6)} px/mm (ratio ${scaleRatio.toFixed(6)}). Inspect the preview, especially whether the supplied dimension really spans the complete projected boundary. Confirm ${calibrationRef} to lock this scale; this proposal alone does not change later comparisons.`,
						},
						await imageFromWorkspace(options.workspace, preview.context_path),
					],
					details: record,
				};
			}

			const record = options.state.scaleCalibrations.find(({ calibration_ref }) => (
				calibration_ref === params.calibration_ref
			));
			if (!record || record.status === "deleted") {
				throw new Error(`Unknown drawing-scale calibration ${params.calibration_ref}`);
			}
			if (params.operation === "confirm") {
				if (record.status !== "proposed") {
					throw new Error(`Drawing-scale calibration ${params.calibration_ref} is not awaiting confirmation`);
				}
				const formal = await currentFormalState(options);
				if (formal.sourceSha256 !== record.source_sha256 || formal.stepSha256 !== record.step_sha256) {
					throw new Error("The formal CAD changed after this scale proposal; propose a new calibration from the current STEP");
				}
				for (const existing of options.state.scaleCalibrations) {
					if (existing.region_id === record.region_id && existing.status === "confirmed") {
						existing.status = "superseded";
						existing.updated_at = isoNow();
					}
				}
				record.status = "confirmed";
				record.updated_at = isoNow();
				clearRegionVisualCache(options.state, record.region_id);
				await persistMainViewReflectState(options.workspace, options.state);
				await options.onUpdate(mainViewReflectSummary(options.state));
				return {
					content: [{
						type: "text" as const,
						text: `Confirmed ${record.calibration_ref}. Future ${record.region_id} comparisons use fixed scale ${record.locked_scale_px_per_unit.toFixed(6)} px/mm; free-fit scale will no longer be used for the displayed CAD size. Run inspect_comparison again to inspect the scale-locked view.`,
					}],
					details: record,
				};
			}

			const wasConfirmed = record.status === "confirmed";
			record.status = "deleted";
			record.updated_at = isoNow();
			if (wasConfirmed) clearRegionVisualCache(options.state, record.region_id);
			await persistMainViewReflectState(options.workspace, options.state);
			await options.onUpdate(mainViewReflectSummary(options.state));
			return {
				content: [{
					type: "text" as const,
					text: `Deleted ${record.calibration_ref}.${wasConfirmed ? ` Region ${record.region_id} has returned to free-scale alignment.` : ""}`,
				}],
				details: record,
			};
		},
	});

	const targetAnnotationTool = options.targetAnnotationState
		? createTargetAnnotationTool({
				workspace: options.workspace,
				state: options.targetAnnotationState,
				getLatestInspectedRegionId: () => options.state.inspectionAttempts.at(-1)?.region_id,
				renderComparison: async (regionId, annotations) => {
					const latestAttempt = options.state.inspectionAttempts.at(-1);
					if (!latestAttempt || latestAttempt.region_id !== regionId) {
						throw new Error(`Inspect comparison region ${regionId} before annotating its Target drawing`);
					}
					const inspection = options.state.inspections.get(latestAttempt.inspection_id);
					const snapshot = inspection
						? options.state.snapshots.get(inspection.snapshot_id)
						: undefined;
					if (!inspection || !snapshot) {
						throw new InfrastructureError("The latest inspected comparison is unavailable for Target annotation");
					}
					return await renderAnnotatedComparison(
						regionForInspection(options.state, regionId),
						snapshot,
						annotations,
					);
				},
			})
		: undefined;
	const tools = [
		inspectTool,
		cropTool,
		scaleTool,
		...(targetAnnotationTool ? [targetAnnotationTool] : []),
	];
	const expectedToolNames = [
		...MAIN_VIEW_REFLECT_AGENT_TOOLS,
		...(targetAnnotationTool ? TARGET_ANNOTATION_AGENT_TOOLS : []),
	];
	if (tools.some((tool, index) => tool.name !== expectedToolNames[index])) {
		throw new InfrastructureError("Main View Reflect tool order does not match the declared contract");
	}
	return tools;
}
