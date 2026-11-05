import { createHash } from "node:crypto";
import { lstat, readFile, realpath } from "node:fs/promises";
import { isAbsolute, join, relative, resolve } from "node:path";

import type { ImageContent } from "@earendil-works/pi-ai";
import {
	type AgentSession,
	createAgentSession,
	DefaultResourceLoader,
	defineTool,
	getAgentDir,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import {
	alignmentNotice,
	imageFromWorkspace,
	inspectPngFile,
	REFLECTION_ACTIVE_TIMEOUT_MS,
	REFLECTION_WALL_TIMEOUT_MS,
	requireDrawingSuccess,
} from "./drawing-reflection.js";
import { InfrastructureError } from "./errors.js";
import { isoNow, readJson, writeJsonAtomic } from "./files.js";
import type { WorkerClient } from "./local-worker.js";
import { PausableDeadline } from "./pausable-deadline.js";
import {
	REFLECTION_ACCEPTANCE_SYSTEM_PROMPT,
	reflectionAcceptancePrompt,
	REFLECTION_REFINEMENT_SYSTEM_PROMPT,
	reflectionRefinementPrompt,
} from "./prompts.js";
import { sessionUsageTotals } from "./context.js";
import {
	PROTOCOL_VERSION,
	REFLECTION_ACCEPTANCE_PROFILE,
	REFLECTION_REFINEMENT_AGENT_TOOLS,
	REFLECTION_REFINEMENT_PROFILE,
	type DrawingReflectionReport,
	type DrawingViewManifest,
	type NormalizedRoi,
	type UsageTotals,
	type WorkerReflectionInspectionResponse,
} from "./types.js";
import type { ResolvedModel, RunController } from "./runner.js";

const REGION_ID_SCHEMA = Type.String({
	minLength: 1,
	maxLength: 128,
	pattern: "^[A-Za-z0-9][A-Za-z0-9._-]*$",
});
const FINDING_ID_SCHEMA = Type.String({
	minLength: 11,
	maxLength: 128,
	pattern: "^finding-[0-9]{3,}$",
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
		description:
			"One independent ROI on its own image canvas in normalized 0-1000 coordinates. Width and height must each be at least 50.",
	},
);

export const COMPARE_REGION_CROPS_PARAMETERS_SCHEMA = Type.Object(
	{
		region_id: REGION_ID_SCHEMA,
		reference_roi: ROI_SCHEMA,
		cad_roi: ROI_SCHEMA,
	},
	{
		additionalProperties: false,
		description:
			"Compare one semantic feature using independent ROIs on the locked drawing-region image and aligned CAD image. No paths, cameras, candidates, or output locations are accepted.",
	},
);

const REFINEMENT_CHANGE_SCHEMA = Type.Union([
	Type.Object(
		{
			action: Type.Literal("add"),
			obs: Type.String({
				minLength: 1,
				maxLength: 2000,
				description: "A concrete difference stated as what the Original drawing shows versus what the CAD shows, including location.",
			}),
			suggested_change: Type.String({
				minLength: 1,
				maxLength: 2000,
				description: "One concrete CAD geometry change directly supported by the accurate crop and complete view.",
			}),
		},
		{
			additionalProperties: false,
			description: "Add only a genuinely new geometry problem after operation=read confirms no current finding already covers it.",
		},
	),
	Type.Object(
		{
			action: Type.Literal("update"),
			finding_id: FINDING_ID_SCHEMA,
			field: Type.Union([Type.Literal("obs"), Type.Literal("suggested_change")]),
			old_text: Type.String({ minLength: 1, maxLength: 2000 }),
			new_text: Type.String({ maxLength: 2000 }),
		},
		{
			additionalProperties: false,
			description: "Replace one exact substring occurring once, preserving the rest of the finding.",
		},
	),
	Type.Object(
		{
			action: Type.Literal("delete"),
			finding_id: FINDING_ID_SCHEMA,
		},
		{ additionalProperties: false },
	),
]);

const CROP_ID_SCHEMA = Type.String({
	minLength: 20,
	maxLength: 20,
	pattern: "^[a-f0-9]{20}$",
	description: "Controller-generated crop_id returned by compare_region_crops.",
});

export const MANAGE_REFLECTION_FINDINGS_PARAMETERS_SCHEMA = Type.Union(
	[
		Type.Object(
			{ operation: Type.Literal("read") },
			{
				additionalProperties: false,
				description: "Return the current complete persisted findings without changing them.",
			},
		),
		Type.Object(
			{
				operation: Type.Literal("apply_changes"),
				crop_id: CROP_ID_SCHEMA,
				changes: Type.Array(REFINEMENT_CHANGE_SCHEMA, {
					minItems: 1,
					maxItems: 24,
					description: "One or more ordered additions, exact substring updates, or deletions supported by this crop.",
				}),
			},
			{ additionalProperties: false },
		),
		Type.Object(
			{
				operation: Type.Literal("confirm_no_change"),
				crop_id: CROP_ID_SCHEMA,
				reviewed_finding_ids: Type.Optional(Type.Array(FINDING_ID_SCHEMA, {
					minItems: 1,
					maxItems: 24,
					description: "Acceptance phase only: carried findings confirmed by this same-region current-CAD crop.",
				})),
			},
			{
				additionalProperties: false,
				description: "Record that this valid crop does not require any finding change.",
			},
		),
	],
	{
		description: "Read or edit the existing first-stage findings. Empty change arrays are never used as an operation.",
	},
);

export const SUBMIT_REFINED_REFLECTION_PARAMETERS_SCHEMA = Type.Object(
	{
		should_revise: Type.Boolean(),
		overall_summary: Type.String({ minLength: 1, maxLength: 4000 }),
	},
	{ additionalProperties: false },
);

export interface RefinementFinding {
	finding_id: string;
	region_id: string;
	obs: string;
	suggested_change: string;
	evidence_crop_ids: string[];
	created_from: "initial_report" | "dual_roi_crop";
}

export interface RefinementCropRecord {
	crop_id: string;
	region_id: string;
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

export interface RefinementCoordinateInspectionRecord {
	inspection_id: string;
	region_id: string;
	candidate_id: string;
	context_path: string;
	width: number;
	height: number;
	sha256: {
		reference_source: string;
		cad_source: string;
		context: string;
	};
	created_at: string;
}

export interface RefinementCropAttemptRecord {
	sequence: number;
	crop_id: string;
	region_id: string;
	cache_hit: boolean;
	created_at: string;
}

export interface RefinementChangeRecord {
	operation: "apply_changes" | "confirm_no_change";
	crop_id: string;
	region_id: string;
	changes: unknown[];
	reviewed_finding_ids?: string[];
	created_at: string;
}

export type ReflectionRefinementPhase = "refinement" | "acceptance";

export interface ReflectionRefinementState {
	phase: ReflectionRefinementPhase;
	fixtureId: string;
	viewManifest: DrawingViewManifest;
	initialReport: DrawingReflectionReport;
	initialReportSha256: string;
	findings: RefinementFinding[];
	nextFindingSequence: number;
	findingsRevision: number;
	lastReadRevision: number | null;
	coordinateInspections: Map<string, RefinementCoordinateInspectionRecord>;
	crops: Map<string, RefinementCropRecord>;
	cropAttempts: RefinementCropAttemptRecord[];
	changeHistory: RefinementChangeRecord[];
	requiredFindingIds: Set<string>;
	reviewedFindingIds: Set<string>;
	refinedReport: DrawingReflectionReport | null;
	manifestPath: string;
	startedAt: string;
}

interface PersistedReflectionRefinementManifest {
	version: 3;
	profile: typeof REFLECTION_REFINEMENT_PROFILE | typeof REFLECTION_ACCEPTANCE_PROFILE;
	phase?: ReflectionRefinementPhase;
	fixture_id: string;
	initial_report_sha256: string;
	started_at: string;
	updated_at: string;
	next_finding_sequence: number;
	findings_revision: number;
	findings: RefinementFinding[];
	coordinate_inspections: RefinementCoordinateInspectionRecord[];
	crops: RefinementCropRecord[];
	crop_attempts: RefinementCropAttemptRecord[];
	change_history: RefinementChangeRecord[];
	required_finding_ids?: string[];
	reviewed_finding_ids?: string[];
	refined_report: DrawingReflectionReport | null;
}

function profileForPhase(phase: ReflectionRefinementPhase) {
	return phase === "acceptance" ? REFLECTION_ACCEPTANCE_PROFILE : REFLECTION_REFINEMENT_PROFILE;
}

function findingId(sequence: number): string {
	return `finding-${String(sequence).padStart(3, "0")}`;
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

function replaceExactSubstring(value: string, oldText: string, newText: string, field: string): string {
	const index = value.indexOf(oldText);
	if (index < 0) throw new Error(`old_text was not found in ${field}`);
	if (value.indexOf(oldText, index + oldText.length) >= 0) {
		throw new Error(`old_text occurs more than once in ${field}; use a more specific substring`);
	}
	const updated = `${value.slice(0, index)}${newText}${value.slice(index + oldText.length)}`;
	if (updated.length < 1 || updated.length > 2000) {
		throw new Error(`Updated ${field} must contain 1-2000 characters`);
	}
	return updated;
}

function normalizedFindingText(value: string): string {
	return value.trim().replaceAll(/\s+/gu, " ").toLocaleLowerCase("en-US");
}

async function requireSafeFile(workspace: string, path: string): Promise<string> {
	if (isAbsolute(path)) throw new InfrastructureError("Reflection artifact path must be relative");
	const root = resolve(workspace);
	const candidate = resolve(root, path);
	const rel = relative(root, candidate);
	if (rel === ".." || rel.startsWith("../") || isAbsolute(rel)) {
		throw new InfrastructureError("Reflection artifact escaped the fixture workspace");
	}
	const metadata = await lstat(candidate).catch(() => null);
	if (!metadata?.isFile() || metadata.isSymbolicLink()) {
		throw new InfrastructureError(`Reflection artifact is missing or unsafe: ${path}`);
	}
	const [rootReal, candidateReal] = await Promise.all([realpath(root), realpath(candidate)]);
	const realRel = relative(rootReal, candidateReal);
	if (realRel === ".." || realRel.startsWith("../") || isAbsolute(realRel)) {
		throw new InfrastructureError("Reflection artifact resolved outside the fixture workspace");
	}
	return candidateReal;
}

export function createReflectionRefinementState(options: {
	fixtureId: string;
	viewManifest: DrawingViewManifest;
	initialReport: DrawingReflectionReport;
	initialReportSha256: string;
	manifestPath?: string;
	phase?: ReflectionRefinementPhase;
}): ReflectionRefinementState {
	const phase = options.phase ?? "refinement";
	const findings = options.initialReport.findings.map((finding, index) => ({
		finding_id: findingId(index + 1),
		region_id: finding.region_id,
		obs: finding.obs,
		suggested_change: finding.suggested_change,
		evidence_crop_ids: [],
		created_from: "initial_report" as const,
	}));
	return {
		phase,
		fixtureId: options.fixtureId,
		viewManifest: options.viewManifest,
		initialReport: options.initialReport,
		initialReportSha256: options.initialReportSha256,
		findings,
		nextFindingSequence: findings.length + 1,
		findingsRevision: 0,
		lastReadRevision: null,
		coordinateInspections: new Map(),
		crops: new Map(),
		cropAttempts: [],
		changeHistory: [],
		requiredFindingIds: new Set(phase === "acceptance" ? findings.map(({ finding_id }) => finding_id) : []),
		reviewedFindingIds: new Set(),
		refinedReport: null,
		manifestPath: options.manifestPath ?? (phase === "acceptance"
			? "reflection_acceptance/inspection_manifest.json"
			: "reflection_refinement/inspection_manifest.json"),
		startedAt: isoNow(),
	};
}

export async function restoreReflectionRefinementState(options: {
	workspace: string;
	fixtureId: string;
	viewManifest: DrawingViewManifest;
	initialReport: DrawingReflectionReport;
	initialReportSha256: string;
	manifestPath?: string;
	phase?: ReflectionRefinementPhase;
}): Promise<ReflectionRefinementState> {
	const fresh = createReflectionRefinementState(options);
	const absolutePath = resolve(options.workspace, fresh.manifestPath);
	const persisted = await readJson<PersistedReflectionRefinementManifest>(absolutePath).catch(() => null);
	if (!persisted) return fresh;
	if (
		persisted.version !== 3 ||
		persisted.profile !== profileForPhase(fresh.phase) ||
		(persisted.phase ?? "refinement") !== fresh.phase ||
		persisted.fixture_id !== options.fixtureId ||
		persisted.initial_report_sha256 !== options.initialReportSha256
	) throw new InfrastructureError("Reflection refinement manifest does not match the locked initial report");
	const persistedRequired = new Set(persisted.required_finding_ids ?? []);
	const persistedReviewed = new Set(persisted.reviewed_finding_ids ?? []);
	if (fresh.phase === "acceptance") {
		const expectedRequired = [...fresh.requiredFindingIds].sort();
		if (JSON.stringify([...persistedRequired].sort()) !== JSON.stringify(expectedRequired)) {
			throw new InfrastructureError("Persisted acceptance finding coverage does not match the carried report");
		}
		if ([...persistedReviewed].some((id) => !persistedRequired.has(id))) {
			throw new InfrastructureError("Persisted acceptance review references an unknown carried finding");
		}
	}
	const regionById = new Map(options.viewManifest.regions.map((region) => [region.region_id, region]));
	const regionIds = new Set(regionById.keys());
	for (const finding of persisted.findings) {
		if (!regionIds.has(finding.region_id)) throw new InfrastructureError("Persisted refinement finding has an unknown region");
	}
	for (const inspection of persisted.coordinate_inspections) {
		if (!regionIds.has(inspection.region_id)) throw new InfrastructureError("Persisted coordinate inspection has an unknown region");
		const region = regionById.get(inspection.region_id)!;
		if (!region.selected?.aligned_render_path || region.selected.candidate_id !== inspection.candidate_id) {
			throw new InfrastructureError("Persisted coordinate inspection does not match the locked candidate");
		}
		await requireSafeFile(options.workspace, inspection.context_path);
		const [reference, cad, context] = await Promise.all([
			inspectPngFile(options.workspace, region.crop_path),
			inspectPngFile(options.workspace, region.selected.aligned_render_path),
			inspectPngFile(options.workspace, inspection.context_path),
		]);
		if (
			reference.sha256 !== inspection.sha256.reference_source ||
			cad.sha256 !== inspection.sha256.cad_source ||
			context.sha256 !== inspection.sha256.context
		) {
			throw new InfrastructureError("Persisted coordinate inspection hash mismatch");
		}
	}
	const inspectedRegionIds = new Set(persisted.coordinate_inspections.map(({ region_id }) => region_id));
	for (const crop of persisted.crops) {
		if (!regionIds.has(crop.region_id)) throw new InfrastructureError("Persisted refinement crop has an unknown region");
		if (!inspectedRegionIds.has(crop.region_id)) {
			throw new InfrastructureError("Persisted refinement crop lacks its coordinate inspection");
		}
		await Promise.all([
			requireSafeFile(options.workspace, crop.reference_crop_path),
			requireSafeFile(options.workspace, crop.cad_crop_path),
			requireSafeFile(options.workspace, crop.context_path),
			requireSafeFile(options.workspace, crop.comparison_path),
		]);
		const [referenceCrop, cadCrop, context, comparison] = await Promise.all([
			inspectPngFile(options.workspace, crop.reference_crop_path),
			inspectPngFile(options.workspace, crop.cad_crop_path),
			inspectPngFile(options.workspace, crop.context_path),
			inspectPngFile(options.workspace, crop.comparison_path),
		]);
		if (
			referenceCrop.sha256 !== crop.sha256.reference_crop ||
			cadCrop.sha256 !== crop.sha256.cad_crop ||
			context.sha256 !== crop.sha256.context ||
			comparison.sha256 !== crop.sha256.comparison
		) throw new InfrastructureError("Persisted refinement crop hash mismatch");
	}
	const cropById = new Map(persisted.crops.map((crop) => [crop.crop_id, crop]));
	for (const [index, attempt] of persisted.crop_attempts.entries()) {
		const crop = cropById.get(attempt.crop_id);
		if (attempt.sequence !== index + 1 || !crop || crop.region_id !== attempt.region_id) {
			throw new InfrastructureError("Persisted crop attempt history is invalid");
		}
	}
	return {
		...fresh,
		findings: persisted.findings,
		nextFindingSequence: persisted.next_finding_sequence,
		findingsRevision: persisted.findings_revision,
		lastReadRevision: null,
		coordinateInspections: new Map(persisted.coordinate_inspections.map((inspection) => [inspection.region_id, inspection])),
		crops: new Map(persisted.crops.map((crop) => [crop.crop_id, crop])),
		cropAttempts: persisted.crop_attempts,
		changeHistory: persisted.change_history,
		requiredFindingIds: fresh.phase === "acceptance" ? persistedRequired : new Set(),
		reviewedFindingIds: fresh.phase === "acceptance" ? persistedReviewed : new Set(),
		refinedReport: persisted.refined_report,
		startedAt: persisted.started_at,
	};
}

export async function persistReflectionRefinementState(
	workspace: string,
	state: ReflectionRefinementState,
): Promise<void> {
	await writeJsonAtomic(resolve(workspace, state.manifestPath), {
		version: 3,
		profile: profileForPhase(state.phase),
		phase: state.phase,
		fixture_id: state.fixtureId,
		initial_report_sha256: state.initialReportSha256,
		started_at: state.startedAt,
		updated_at: isoNow(),
		next_finding_sequence: state.nextFindingSequence,
		findings_revision: state.findingsRevision,
		findings: state.findings,
		coordinate_inspections: [...state.coordinateInspections.values()],
		crops: [...state.crops.values()],
		crop_attempts: state.cropAttempts,
		change_history: state.changeHistory,
		required_finding_ids: [...state.requiredFindingIds],
		reviewed_finding_ids: [...state.reviewedFindingIds],
		refined_report: state.refinedReport,
	} satisfies PersistedReflectionRefinementManifest);
}

export function promptFindings(state: ReflectionRefinementState) {
	return state.findings.map(({ finding_id, region_id, obs, suggested_change }) => ({
		finding_id,
		region_id,
		obs,
		suggested_change,
	}));
}

export function createReflectionRefinementTools(options: {
	workspace: string;
	state: ReflectionRefinementState;
	worker: WorkerClient;
	controller?: RunController;
	outputDir?: string;
}) {
	const inspectionRoot = options.outputDir ?? "drawing_feedback/reflection-refinement-post-submit/inspections";
	const inspectTool = defineTool({
		name: "inspect_region_for_crop",
		label: "Inspect independent ROI coordinates",
		description:
			"Show one locked drawing/CAD comparison with a separate normalized 0-1000 coordinate grid on each panel. Call once before the first crop for that region.",
		parameters: Type.Object({ region_id: REGION_ID_SCHEMA }, { additionalProperties: false }),
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			const region = options.state.viewManifest.regions.find(({ region_id }) => region_id === params.region_id);
			if (!region) throw new Error(`Unknown comparison region ${params.region_id}`);
			if (!region.selected?.aligned_render_path) {
				throw new Error(`Region ${params.region_id} has no locked aligned CAD image`);
			}
			const [referenceSource, cadSource] = await Promise.all([
				inspectPngFile(options.workspace, region.crop_path),
				inspectPngFile(options.workspace, region.selected.aligned_render_path),
			]);
			if (referenceSource.width !== cadSource.width || referenceSource.height !== cadSource.height) {
				throw new InfrastructureError("Locked drawing crop and aligned CAD image dimensions differ");
			}
			let record = options.state.coordinateInspections.get(region.region_id);
			if (
				record &&
				(record.candidate_id !== region.selected.candidate_id ||
					record.sha256.reference_source !== referenceSource.sha256 ||
					record.sha256.cad_source !== cadSource.sha256)
			) throw new InfrastructureError(`Cached coordinate inspection for ${region.region_id} is stale`);
			if (!record) {
				const inspectionId = createHash("sha256")
					.update(JSON.stringify([
						profileForPhase(options.state.phase),
						options.state.viewManifest.version,
						region.region_id,
						region.selected.candidate_id,
						referenceSource.sha256,
						cadSource.sha256,
					]))
					.digest("hex")
					.slice(0, 20);
				const response = await options.worker.request<WorkerReflectionInspectionResponse>(
					options.workspace,
					{
						protocol_version: PROTOCOL_VERSION,
						operation: "compose_reflection_inspection",
						region_id: region.region_id,
						crop_path: region.crop_path,
						aligned_render_path: region.selected.aligned_render_path,
						mode: "coordinate_overview",
						output_dir: join(inspectionRoot, region.region_id, inspectionId),
					},
					{ timeoutMs: 180_000, ...(options.controller ? { signal: options.controller.signal } : {}) },
				);
				requireDrawingSuccess(response, "compose_reflection_inspection");
				if (response.mode !== "coordinate_overview" || response.region_id !== region.region_id) {
					throw new InfrastructureError("Reflection worker returned a mismatched coordinate overview");
				}
				const context = await inspectPngFile(options.workspace, response.context_path);
				if (context.sha256 !== response.sha256.context) {
					throw new InfrastructureError("Reflection coordinate overview artifact hash mismatch");
				}
				record = {
					inspection_id: inspectionId,
					region_id: region.region_id,
					candidate_id: region.selected.candidate_id,
					context_path: response.context_path,
					width: response.width,
					height: response.height,
					sha256: {
						reference_source: referenceSource.sha256,
						cad_source: cadSource.sha256,
						context: context.sha256,
					},
					created_at: isoNow(),
				};
				options.state.coordinateInspections.set(region.region_id, record);
				await persistReflectionRefinementState(options.workspace, options.state);
			}
			const content: Array<{ type: "text"; text: string } | ImageContent> = [
				{
					type: "text",
					text: `${region.region_id}: the LEFT Original drawing has its own reference_roi 0-1000 grid; the RIGHT Aligned CAD has its own cad_roi 0-1000 grid. Coordinates are never measured over the combined image. One inspection permits any number of crop attempts for this region. ${alignmentNotice(region)}`,
				},
				await imageFromWorkspace(options.workspace, record.context_path),
			];
			return { content, details: record };
		},
	});

	const compareTool = defineTool({
		name: "compare_region_crops",
		label: "Compare independent drawing and CAD ROIs",
		description:
			"Compare one semantic feature using independent normalized 0-1000 ROIs on the locked drawing-region image and aligned CAD image. Returns a controller crop_id and a 2x2 comparison.",
		parameters: COMPARE_REGION_CROPS_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			const region = options.state.viewManifest.regions.find(({ region_id }) => region_id === params.region_id);
			if (!region) throw new Error(`Unknown comparison region ${params.region_id}`);
			if (!region.selected?.aligned_render_path) {
				throw new Error(`Region ${params.region_id} has no locked aligned CAD image`);
			}
			const coordinateInspection = options.state.coordinateInspections.get(region.region_id);
			if (!coordinateInspection) {
				throw new Error(`Call inspect_region_for_crop for ${region.region_id} before its first crop`);
			}
			const referenceRoi = roiFromObject(params.reference_roi);
			const cadRoi = roiFromObject(params.cad_roi);
			const [referenceSource, cadSource] = await Promise.all([
				inspectPngFile(options.workspace, region.crop_path),
				inspectPngFile(options.workspace, region.selected.aligned_render_path),
			]);
			if (referenceSource.width !== cadSource.width || referenceSource.height !== cadSource.height) {
				throw new InfrastructureError("Locked drawing crop and aligned CAD image dimensions differ");
			}
			if (
				coordinateInspection.candidate_id !== region.selected.candidate_id ||
				coordinateInspection.sha256.reference_source !== referenceSource.sha256 ||
				coordinateInspection.sha256.cad_source !== cadSource.sha256
			) throw new InfrastructureError(`Coordinate inspection for ${region.region_id} no longer matches the locked images`);
			const cropId = createHash("sha256")
				.update(JSON.stringify([
					profileForPhase(options.state.phase),
					options.state.viewManifest.version,
					region.region_id,
					region.selected.candidate_id,
					referenceSource.sha256,
					cadSource.sha256,
					referenceRoi,
					cadRoi,
				]))
				.digest("hex")
				.slice(0, 20);
			let record = options.state.crops.get(cropId);
			const cacheHit = record !== undefined;
			if (!record) {
				const response = await options.worker.request<WorkerReflectionInspectionResponse>(
					options.workspace,
					{
						protocol_version: PROTOCOL_VERSION,
						operation: "compose_reflection_inspection",
						region_id: region.region_id,
						crop_path: region.crop_path,
						aligned_render_path: region.selected.aligned_render_path,
						mode: "dual_roi",
						reference_roi: referenceRoi,
						cad_roi: cadRoi,
						output_dir: join(inspectionRoot, region.region_id, cropId),
					},
					{ timeoutMs: 180_000, ...(options.controller ? { signal: options.controller.signal } : {}) },
				);
				requireDrawingSuccess(response, "compose_reflection_inspection");
				if (
					response.mode !== "dual_roi" ||
					response.region_id !== region.region_id ||
					!sameRoi(response.reference_roi, referenceRoi) ||
					!sameRoi(response.cad_roi, cadRoi)
				) throw new InfrastructureError("Reflection worker returned a mismatched dual-ROI comparison");
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
				) throw new InfrastructureError("Reflection dual-ROI artifact hash mismatch");
				record = {
					crop_id: cropId,
					region_id: region.region_id,
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
			}
			options.state.cropAttempts.push({
				sequence: options.state.cropAttempts.length + 1,
				crop_id: cropId,
				region_id: region.region_id,
				cache_hit: cacheHit,
				created_at: isoNow(),
			});
			await persistReflectionRefinementState(options.workspace, options.state);
			return {
				content: [
					{
						type: "text" as const,
						text: `${region.region_id}/${record.crop_id}: original ROI=${JSON.stringify(record.reference_roi)}, CAD ROI=${JSON.stringify(record.cad_roi)}. ${alignmentNotice(region)} Use manage_reflection_findings with this crop_id if the local evidence changes or confirms the report.`,
					},
					await imageFromWorkspace(options.workspace, record.comparison_path),
				],
				details: record,
			};
		},
	});

	const manageTool = defineTool({
		name: "manage_reflection_findings",
		label: "Read or edit persisted Reflection findings",
		description:
			"Use operation=read to retrieve the current report, apply_changes with a crop_id to add/update/delete, or confirm_no_change with a crop_id. Empty changes are not an operation.",
		parameters: MANAGE_REFLECTION_FINDINGS_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			if (params.operation === "read") {
				options.state.lastReadRevision = options.state.findingsRevision;
				return {
					content: [{
						type: "text" as const,
						text: `Read findings revision ${String(options.state.findingsRevision)} containing ${String(options.state.findings.length)} finding(s). No changes were made.`,
					}],
					details: {
						operation: "read" as "read" | "apply_changes" | "confirm_no_change",
						crop_id: null as string | null,
						region_id: null as string | null,
						findings_revision: options.state.findingsRevision,
						findings: promptFindings(options.state),
						required_finding_ids: [...options.state.requiredFindingIds],
						reviewed_finding_ids: [...options.state.reviewedFindingIds],
					},
				};
			}
			if (!("crop_id" in params) || !params.crop_id) {
				throw new Error("crop_id is required for apply_changes and confirm_no_change");
			}
			const crop = options.state.crops.get(params.crop_id);
			if (!crop) throw new Error(`Unknown dual-ROI crop ${params.crop_id}`);
			const changes = params.operation === "apply_changes" ? params.changes : [];
			const reviewedFindingIds = params.operation === "confirm_no_change"
				? (params.reviewed_finding_ids ?? [])
				: [];
			if (options.state.phase === "acceptance" && params.operation === "confirm_no_change") {
				if (reviewedFindingIds.length < 1) {
					throw new Error("Acceptance confirm_no_change requires reviewed_finding_ids");
				}
				if (new Set(reviewedFindingIds).size !== reviewedFindingIds.length) {
					throw new Error("reviewed_finding_ids must not contain duplicates");
				}
				for (const id of reviewedFindingIds) {
					const finding = options.state.findings.find(({ finding_id }) => finding_id === id);
					if (!finding) throw new Error(`Unknown finding ${id}`);
					if (!options.state.requiredFindingIds.has(id)) {
						throw new Error(`Finding ${id} was not carried into the acceptance phase`);
					}
					if (finding.region_id !== crop.region_id) {
						throw new Error(`Finding ${id} belongs to ${finding.region_id}, not crop region ${crop.region_id}`);
					}
				}
			}
			if (
				changes.some(({ action }) => action === "add") &&
				options.state.lastReadRevision !== options.state.findingsRevision
			) {
				throw new Error("Read the current findings with operation=read immediately before adding a finding");
			}
			const previousFindings = options.state.findings.map((finding) => ({
				...finding,
				evidence_crop_ids: [...finding.evidence_crop_ids],
			}));
			const previousSequence = options.state.nextFindingSequence;
			const previousRevision = options.state.findingsRevision;
			const previousReadRevision = options.state.lastReadRevision;
			const previousReviewedFindingIds = new Set(options.state.reviewedFindingIds);
			let nextSequence = previousSequence;
			const next = previousFindings.map((finding) => ({
				...finding,
				evidence_crop_ids: [...finding.evidence_crop_ids],
			}));
			for (const change of changes) {
				if (change.action === "add") {
					if (next.length >= 48) throw new Error("The refined report cannot exceed 48 findings");
					const duplicate = next.find((finding) => (
						normalizedFindingText(finding.obs) === normalizedFindingText(change.obs) &&
						normalizedFindingText(finding.suggested_change) === normalizedFindingText(change.suggested_change)
					));
					if (duplicate) {
						throw new Error(`New finding duplicates existing ${duplicate.finding_id}; update the existing finding instead`);
					}
					next.push({
						finding_id: findingId(nextSequence),
						region_id: crop.region_id,
						obs: change.obs,
						suggested_change: change.suggested_change,
						evidence_crop_ids: [crop.crop_id],
						created_from: "dual_roi_crop",
					});
					nextSequence += 1;
					continue;
				}
				const index = next.findIndex(({ finding_id: id }) => id === change.finding_id);
				if (index < 0) throw new Error(`Unknown finding ${change.finding_id}`);
				const existing = next[index]!;
				if (existing.region_id !== crop.region_id) {
					throw new Error(`Finding ${change.finding_id} belongs to ${existing.region_id}, not crop region ${crop.region_id}`);
				}
				if (change.action === "delete") {
					if (options.state.phase === "acceptance" && options.state.requiredFindingIds.has(change.finding_id)) {
						options.state.reviewedFindingIds.add(change.finding_id);
					}
					next.splice(index, 1);
					continue;
				}
				existing[change.field] = replaceExactSubstring(
					existing[change.field],
					change.old_text,
					change.new_text,
					change.field,
				);
				if (!existing.evidence_crop_ids.includes(crop.crop_id)) existing.evidence_crop_ids.push(crop.crop_id);
				if (options.state.phase === "acceptance" && options.state.requiredFindingIds.has(change.finding_id)) {
					options.state.reviewedFindingIds.add(change.finding_id);
				}
			}
			if (options.state.phase === "acceptance" && params.operation === "confirm_no_change") {
				for (const id of reviewedFindingIds) options.state.reviewedFindingIds.add(id);
			}
			options.state.findings = next;
			options.state.nextFindingSequence = nextSequence;
			if (params.operation === "apply_changes") {
				options.state.findingsRevision += 1;
				options.state.lastReadRevision = null;
			}
			const history: RefinementChangeRecord = {
				operation: params.operation,
				crop_id: crop.crop_id,
				region_id: crop.region_id,
				changes,
				...(reviewedFindingIds.length > 0 ? { reviewed_finding_ids: reviewedFindingIds } : {}),
				created_at: isoNow(),
			};
			options.state.changeHistory.push(history);
			try {
				await persistReflectionRefinementState(options.workspace, options.state);
			} catch (error) {
				options.state.findings = previousFindings;
				options.state.nextFindingSequence = previousSequence;
				options.state.findingsRevision = previousRevision;
				options.state.lastReadRevision = previousReadRevision;
				options.state.reviewedFindingIds = previousReviewedFindingIds;
				options.state.changeHistory.pop();
				throw error;
			}
			return {
				content: [{
					type: "text" as const,
					text: params.operation === "confirm_no_change"
						? `Recorded no finding changes for crop ${crop.crop_id}.`
						: `Persisted ${String(changes.length)} change(s) from crop ${crop.crop_id}. The report now contains ${String(next.length)} finding(s).`,
				}],
				details: {
					operation: params.operation as "read" | "apply_changes" | "confirm_no_change",
					crop_id: crop.crop_id as string | null,
					region_id: crop.region_id as string | null,
					findings_revision: options.state.findingsRevision,
					findings: promptFindings(options.state),
					required_finding_ids: [...options.state.requiredFindingIds],
					reviewed_finding_ids: [...options.state.reviewedFindingIds],
				},
			};
		},
	});

	const submitTool = defineTool({
		name: "submit_refined_reflection",
		label: "Finish editing the existing Reflection report",
		description:
			"Submit only the final should_revise decision and summary after at least one dual-ROI comparison. The controller exports the edited first-stage findings without regenerating them.",
		parameters: SUBMIT_REFINED_REFLECTION_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			if (options.state.refinedReport) throw new Error("The edited Reflection report was already submitted");
			if (options.state.crops.size < 1) throw new Error("Call compare_region_crops at least once before submitting");
			if (options.state.lastReadRevision !== options.state.findingsRevision) {
				throw new Error("Read the current findings with manage_reflection_findings operation=read after the final edit and before submitting");
			}
			if (options.state.phase === "acceptance") {
				const comparableRegionIds = options.state.viewManifest.regions
					.filter(({ selected }) => Boolean(selected?.aligned_render_path))
					.map(({ region_id }) => region_id);
				const missingRegions = comparableRegionIds.filter((id) => !options.state.coordinateInspections.has(id));
				if (missingRegions.length > 0) {
					throw new Error(`Inspect every current comparable region before acceptance submission; missing: ${missingRegions.join(", ")}`);
				}
				const missingFindings = [...options.state.requiredFindingIds]
					.filter((id) => !options.state.reviewedFindingIds.has(id));
				if (missingFindings.length > 0) {
					throw new Error(`Review every carried finding before acceptance submission; missing: ${missingFindings.join(", ")}`);
				}
				if (params.should_revise !== (options.state.findings.length > 0)) {
					throw new Error("In acceptance, should_revise must be true exactly when one or more findings remain");
				}
			}
			const report: DrawingReflectionReport = {
				should_revise: params.should_revise,
				overall_summary: params.overall_summary,
				findings: options.state.findings.map(({ region_id, obs, suggested_change }) => ({
					region_id,
					obs,
					suggested_change,
				})),
			};
			options.state.refinedReport = report;
			await persistReflectionRefinementState(options.workspace, options.state);
			return {
				content: [{ type: "text" as const, text: "Accepted the final summary and exported the edited first-stage findings." }],
				details: report,
			};
		},
	});

	return [inspectTool, compareTool, manageTool, submitTool] as const;
}

export async function sha256File(path: string): Promise<string> {
	return createHash("sha256").update(await readFile(path)).digest("hex");
}

function subtractUsage(after: UsageTotals, before: UsageTotals): UsageTotals {
	return {
		tokens: {
			input: Math.max(0, after.tokens.input - before.tokens.input),
			output: Math.max(0, after.tokens.output - before.tokens.output),
			cache_read: Math.max(0, after.tokens.cache_read - before.tokens.cache_read),
			cache_write: Math.max(0, after.tokens.cache_write - before.tokens.cache_write),
			total: Math.max(0, after.tokens.total - before.tokens.total),
		},
		cost: Math.max(0, after.cost - before.cost),
		tool_calls: Math.max(0, after.tool_calls - before.tool_calls),
	};
}

async function makeRefinementResourceLoader(
	workspace: string,
	settingsManager: SettingsManager,
	phase: ReflectionRefinementPhase,
): Promise<DefaultResourceLoader> {
	const systemPrompt = phase === "acceptance"
		? REFLECTION_ACCEPTANCE_SYSTEM_PROMPT
		: REFLECTION_REFINEMENT_SYSTEM_PROMPT;
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

async function relativeRefinementSessionPath(
	workspace: string,
	sessionsDir: string,
	session: AgentSession,
): Promise<string> {
	const path = session.getSessionStats().sessionFile;
	if (!path) throw new InfrastructureError("Continued Reflection session has no persisted session file");
	const absolutePath = isAbsolute(path) ? path : resolve(workspace, path);
	const [workspaceReal, sessionsReal, sessionReal] = await Promise.all([
		realpath(workspace),
		realpath(sessionsDir),
		realpath(absolutePath),
	]);
	for (const [root, label] of [[workspaceReal, "workspace"], [sessionsReal, "session directory"]] as const) {
		const rel = relative(root, sessionReal);
		if (rel === ".." || rel.startsWith("../") || isAbsolute(rel)) {
			throw new InfrastructureError(`Continued Reflection session escaped its ${label}`);
		}
	}
	return relative(workspaceReal, sessionReal);
}

export interface ReflectionRefinementRunResult {
	fixture_id: string;
	phase: ReflectionRefinementPhase;
	status: "completed";
	started_at: string;
	finished_at: string;
	duration_s: number;
	session_id: string;
	session_file: string;
	initial_report_sha256: string;
	refined_report_path: string;
	inspection_manifest_path: string;
	coordinate_inspection_count: number;
	crop_count: number;
	crop_attempt_count: number;
	change_count: number;
	usage_delta: UsageTotals;
	report: DrawingReflectionReport;
}

export async function runReflectionRefinementSession(options: {
	workspace: string;
	sessionFile: string;
	sessionsDir: string;
	state: ReflectionRefinementState;
	worker: WorkerClient;
	resolvedModel: ResolvedModel;
	thinkingLevel: "off" | "minimal" | "low" | "medium" | "high";
	controller: RunController;
	refinedReportPath?: string;
	activeTimeoutMs?: number;
	wallTimeoutMs?: number;
}): Promise<ReflectionRefinementRunResult> {
	const started = Date.now();
	const startedAt = isoNow();
	if (options.state.refinedReport) {
		const reportPath = options.refinedReportPath ?? (options.state.phase === "acceptance"
			? "reflection_acceptance/reflection.json"
			: "reflection_refinement/reflection.json");
		await writeJsonAtomic(resolve(options.workspace, reportPath), options.state.refinedReport);
		return {
			fixture_id: options.state.fixtureId,
			phase: options.state.phase,
			status: "completed",
			started_at: startedAt,
			finished_at: isoNow(),
			duration_s: 0,
			session_id: "cached",
			session_file: relative(resolve(options.workspace), resolve(options.sessionFile)),
			initial_report_sha256: options.state.initialReportSha256,
			refined_report_path: reportPath,
			inspection_manifest_path: options.state.manifestPath,
			coordinate_inspection_count: options.state.coordinateInspections.size,
			crop_count: options.state.crops.size,
			crop_attempt_count: options.state.cropAttempts.length,
			change_count: options.state.changeHistory.length,
			usage_delta: {
				tokens: { input: 0, output: 0, cache_read: 0, cache_write: 0, total: 0 },
				cost: 0,
				tool_calls: 0,
			},
			report: options.state.refinedReport,
		};
	}
	const settingsManager = SettingsManager.inMemory({}, { projectTrusted: false });
	const resourceLoader = await makeRefinementResourceLoader(options.workspace, settingsManager, options.state.phase);
	const sessionManager = SessionManager.open(options.sessionFile, options.sessionsDir, options.workspace);
	const tools = createReflectionRefinementTools({
		workspace: options.workspace,
		state: options.state,
		worker: options.worker,
		controller: options.controller,
		...(options.state.phase === "acceptance"
			? { outputDir: "drawing_feedback/reflection-acceptance/inspections" }
			: {}),
	});
	const created = await createAgentSession({
		cwd: options.workspace,
		agentDir: getAgentDir(),
		modelRuntime: options.resolvedModel.runtime,
		model: options.resolvedModel.model,
		thinkingLevel: options.thinkingLevel,
		tools: [...REFLECTION_REFINEMENT_AGENT_TOOLS],
		customTools: [...tools],
		resourceLoader,
		sessionManager,
		settingsManager,
	});
	const session = created.session;
	const activeNames = session.getActiveToolNames();
	const expectedNames = new Set<string>(REFLECTION_REFINEMENT_AGENT_TOOLS);
	if (activeNames.length !== expectedNames.size || activeNames.some((name) => !expectedNames.has(name))) {
		session.dispose();
		throw new InfrastructureError(`Reflection refinement tool restriction failed: ${activeNames.join(", ")}`);
	}
	const beforeUsage = sessionUsageTotals(session);
	options.controller.add(session);
	let timeoutCause: "active" | "wall" | undefined;
	const expire = (cause: "active" | "wall"): void => {
		if (timeoutCause !== undefined || options.controller.interrupted) return;
		timeoutCause = cause;
		void session.abort();
	};
	const activeTimeoutMs = options.activeTimeoutMs ?? REFLECTION_ACTIVE_TIMEOUT_MS;
	const wallTimeoutMs = options.wallTimeoutMs ?? REFLECTION_WALL_TIMEOUT_MS;
	const activeDeadline = new PausableDeadline(activeTimeoutMs, () => expire("active"));
	const wallTimer = setTimeout(() => expire("wall"), wallTimeoutMs);
	const pausedCalls = new Set<string>();
	const unsubscribe = session.subscribe((event) => {
		if (event.type === "tool_execution_start" && expectedNames.has(event.toolName)) {
			const wasEmpty = pausedCalls.size === 0;
			pausedCalls.add(event.toolCallId);
			if (wasEmpty) activeDeadline.pause();
			return;
		}
		if (
			event.type === "tool_execution_end" &&
			expectedNames.has(event.toolName) &&
			pausedCalls.delete(event.toolCallId) &&
			pausedCalls.size === 0
		) activeDeadline.resume();
	});
	activeDeadline.start();
	try {
		const prompt = options.state.phase === "acceptance"
			? reflectionAcceptancePrompt({
				fixtureId: options.state.fixtureId,
				findings: promptFindings(options.state),
			})
			: reflectionRefinementPrompt({
				fixtureId: options.state.fixtureId,
				findings: promptFindings(options.state),
			});
		await session.prompt(prompt, { expandPromptTemplates: false });
		if (!options.state.refinedReport && timeoutCause === undefined && !options.controller.interrupted) {
			await session.prompt(
				options.state.phase === "acceptance"
					? "The acceptance report has not been submitted. Inspect every current comparable region, review every carried finding with a valid same-feature crop and an explicit delete, update, or confirm_no_change action, then read the current findings and call submit_refined_reflection."
					: "The edited first-stage report has not been submitted. If no crop succeeded, call inspect_region_for_crop for one chosen region and then compare_region_crops. Apply only evidence-supported edits through manage_reflection_findings, then read the current findings with operation=read and call submit_refined_reflection without rewriting the findings.",
				{ expandPromptTemplates: false },
			);
		}
		if (options.controller.interrupted) throw new Error("interrupted");
		if (!options.state.refinedReport) {
			if (timeoutCause === "active") throw new InfrastructureError("Reflection refinement exhausted 30 minutes of active Agent time");
			if (timeoutCause === "wall") throw new InfrastructureError("Reflection refinement exceeded 120 minutes total wall time");
			throw new InfrastructureError("Reflection refinement did not submit a valid structured result");
		}
		const reportPath = options.refinedReportPath ?? (options.state.phase === "acceptance"
			? "reflection_acceptance/reflection.json"
			: "reflection_refinement/reflection.json");
		await writeJsonAtomic(resolve(options.workspace, reportPath), options.state.refinedReport);
		const afterUsage = sessionUsageTotals(session);
		return {
			fixture_id: options.state.fixtureId,
			phase: options.state.phase,
			status: "completed",
			started_at: startedAt,
			finished_at: isoNow(),
			duration_s: Math.max(0, (Date.now() - started) / 1000),
			session_id: session.sessionId,
			session_file: await relativeRefinementSessionPath(options.workspace, options.sessionsDir, session),
			initial_report_sha256: options.state.initialReportSha256,
			refined_report_path: reportPath,
			inspection_manifest_path: options.state.manifestPath,
			coordinate_inspection_count: options.state.coordinateInspections.size,
			crop_count: options.state.crops.size,
			crop_attempt_count: options.state.cropAttempts.length,
			change_count: options.state.changeHistory.length,
			usage_delta: subtractUsage(afterUsage, beforeUsage),
			report: options.state.refinedReport,
		};
	} finally {
		activeDeadline.dispose();
		clearTimeout(wallTimer);
		unsubscribe();
		options.controller.delete(session);
		session.dispose();
	}
}
