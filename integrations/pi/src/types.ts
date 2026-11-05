export const PROTOCOL_VERSION = 5 as const;
export const PI_VERSION = "0.84.1";
export const DEFAULT_IMAGE = "cadgenbench-pi-agent:0.5.0";
export const SMOKE_FIXTURES = ["127", "133", "145"] as const;
export const SOLUTION_FILE = "solution.py";
export const BASE_AGENT_TOOLS = [
	"read",
	"write",
	"edit",
	"bash",
	"grep",
	"find",
	"ls",
	"execute_build123d",
] as const;
export const ACTIVE_AGENT_TOOLS = [
	...BASE_AGENT_TOOLS,
	"self_reflect_2d_drawing",
] as const;

export type PiAgentMode = "base-only" | "drawing-subagents";
export type PiContinuationKind =
	| "reflection_report"
	| "main_self_reflect"
	| "main_self_reflect_inspect"
	| "main_view_reflect";

export function agentToolsForMode(mode: PiAgentMode | undefined): readonly string[] {
	return mode === "base-only" ? BASE_AGENT_TOOLS : ACTIVE_AGENT_TOOLS;
}

export const SEGMENTATION_AGENT_TOOLS = ["crop_region", "submit_region_manifest"] as const;

export const VIEW_SEARCH_AGENT_TOOLS = [
	"render_candidate_view",
	"rank_view_candidates",
	"submit_view_result",
] as const;

export const REFLECTION_AGENT_TOOLS = ["inspect_region", "submit_reflection"] as const;

export const REFLECTION_REFINEMENT_AGENT_TOOLS = [
	"inspect_region_for_crop",
	"compare_region_crops",
	"manage_reflection_findings",
	"submit_refined_reflection",
] as const;

export const MAIN_VIEW_REFLECT_AGENT_TOOLS = [
	"inspect_comparison",
	"dual_roi_crop",
	"calibrate_drawing_scale",
] as const;

export const MAIN_SELF_REFLECT_INSPECT_AGENT_TOOLS = [
	"inspect_comparison",
] as const;

export const DIMENSION_ANNOTATION_PROFILE = "aligned-dimension-subagent-v1" as const;
export type DimensionAnnotationProfile = typeof DIMENSION_ANNOTATION_PROFILE;

export interface DimensionAnnotationRegionSummary {
	region_id: string;
	status: "annotated" | "not_needed" | "unresolved" | "unsupported";
	reason?: string;
	comparison_path?: string;
	comparison_sha256?: string;
}

export interface DimensionAnnotationSummary {
	profile: DimensionAnnotationProfile;
	state_path: string;
	revision_id: string;
	source_sha256: string;
	step_sha256: string;
	locked_manifest_sha256: string;
	status: "completed" | "running" | "failed";
	regions: DimensionAnnotationRegionSummary[];
}

export const TARGET_ANNOTATION_PROFILE = "persistent-target-annotations-v1" as const;
export type TargetAnnotationProfile = typeof TARGET_ANNOTATION_PROFILE;
export const TARGET_ANNOTATION_AGENT_TOOLS = ["annotate_target"] as const;

export const DUAL_ROI_CROP_PROFILE = "optional-dual-roi-v1" as const;
export type DualRoiCropProfile = typeof DUAL_ROI_CROP_PROFILE;

export const VISUAL_FINDING_PROFILE = "persistent-visual-evidence-v2" as const;
export type VisualFindingProfile = typeof VISUAL_FINDING_PROFILE;
export const VISUAL_FINDING_AGENT_TOOLS = ["update_visual_findings"] as const;

export const VISUAL_DIFF_PROFILE = "target-current-visual-diff-v1" as const;
export type VisualDiffProfile = typeof VISUAL_DIFF_PROFILE;
export const VISUAL_DIFF_AGENT_TOOLS = ["visual_diff"] as const;

export const VISUAL_REVIEW_PROFILE = "executable-geometry-hypothesis-v1" as const;
export const REPAIR_PREVIEW_PROFILE = "multi-candidate-brep-preview-v1" as const;
export const SHAPE_FEEDBACK_PROFILE = "aligned-shape-feedback-v1" as const;
export const GEOMETRY_REPAIR_PROFILE = "atomic-geometry-repair-v1" as const;
export type VisualReviewProfile = typeof VISUAL_REVIEW_PROFILE | typeof REPAIR_PREVIEW_PROFILE | typeof SHAPE_FEEDBACK_PROFILE | typeof GEOMETRY_REPAIR_PROFILE;
export const VISUAL_REVIEW_AGENT_TOOLS = ["visualize_code_hypothesis"] as const;
export const REPAIR_PREVIEW_AGENT_TOOLS = ["preview_repair_candidates"] as const;
export const SHAPE_FEEDBACK_TOOLS = ["preview_shape_feedback", "adopt_shape_candidate"] as const;
export const GEOMETRY_REPAIR_TOOLS = ["repair_geometry", "adopt_shape_candidate"] as const;

export function visualReviewAgentTools(profile?: VisualReviewProfile): readonly string[] {
	if (profile === VISUAL_REVIEW_PROFILE) return VISUAL_REVIEW_AGENT_TOOLS;
	if (profile === REPAIR_PREVIEW_PROFILE) return REPAIR_PREVIEW_AGENT_TOOLS;
	if (profile === SHAPE_FEEDBACK_PROFILE) return SHAPE_FEEDBACK_TOOLS;
	if (profile === GEOMETRY_REPAIR_PROFILE) return GEOMETRY_REPAIR_TOOLS;
	return [];
}

export const CODE_CONSTRUCTION_PROFILE = "optional-code-ablation-v1" as const;
export type CodeConstructionProfile = typeof CODE_CONSTRUCTION_PROFILE;
export const CODE_CONSTRUCTION_AGENT_TOOLS = ["inspect_code_ablation"] as const;

export function mainSelfReflectInspectAgentTools(
	dualRoiCropProfile?: DualRoiCropProfile,
	codeConstructionProfile?: CodeConstructionProfile,
	targetAnnotationProfile?: TargetAnnotationProfile,
): readonly string[] {
	return [
		...MAIN_SELF_REFLECT_INSPECT_AGENT_TOOLS,
		...(dualRoiCropProfile === DUAL_ROI_CROP_PROFILE ? ["dual_roi_crop"] : []),
		...(codeConstructionProfile === CODE_CONSTRUCTION_PROFILE ? CODE_CONSTRUCTION_AGENT_TOOLS : []),
		...(targetAnnotationProfile === TARGET_ANNOTATION_PROFILE ? TARGET_ANNOTATION_AGENT_TOOLS : []),
	];
}

export const CAUSAL_REPAIR_AGENT_TOOLS = ["compare_repair_hypotheses"] as const;

export const CAUSAL_REPAIR_PROFILE = "counterfactual-patch-v1" as const;
export type CausalRepairProfile = typeof CAUSAL_REPAIR_PROFILE;

export const REFLECTION_REFINEMENT_PROFILE = "post-submit-coordinate-dual-roi-v3" as const;
export const REFLECTION_ACCEPTANCE_PROFILE = "post-main-edit-reflection-acceptance-v1" as const;

export const DRAWING_FEEDBACK_VERSION = "2d-reflection-v5-explicit-view-tools-matched-ambiguous-v1" as const;

export const JOB_FILE = ".cadgenbench-pi-job.json";
export const RESPONSE_FILE = ".cadgenbench-pi-response.json";

export type RunStatus =
	| "running"
	| "valid"
	| "invalid"
	| "missing"
	| "interrupted"
	| "infrastructure_error";

export interface ValidationPayload {
	is_valid: boolean;
	is_watertight: boolean;
	topology_errors: string[];
	solid_count: number;
	shell_count: number;
	face_count: number;
	volume: number;
	bbox: Record<string, number>;
}

export interface ArtifactPayload {
	status: "valid" | "invalid" | "missing" | "stale";
	path: string | null;
	sha256: string | null;
	validation: ValidationPayload | null;
	validation_error: string | null;
	render_path: string | null;
	render_error: string | null;
}

export interface ExecutionPayload {
	success: boolean;
	exit_code: number | null;
	timed_out: boolean;
	stdout: string;
	stderr: string;
	files_produced: Record<string, number>;
	duration_s: number;
}

export interface WorkerBaseResponse {
	protocol_version: number;
	operation?: string;
	status?: "ok" | "error";
	error?: { kind: "protocol" | "worker" | "drawing" | "timeout"; message: string };
	worker_diagnostics?: string;
}

export interface WorkerDrawingErrorResponse extends WorkerBaseResponse {
	operation:
		| "crop_drawing_region"
		| "render_drawing_view"
		| "rank_drawing_views"
		| "compare_drawing_sheet"
		| "compose_reflection_inspection"
		| "render_fixed_projection"
		| "compose_repair_comparison"
		| "compose_code_construction_highlight"
		| "compose_code_ablation_highlight"
		| "compose_visual_diff"
		| "compose_code_edit_visualization"
		| "compose_repair_candidate_preview"
		| "compose_shape_feedback"
		| "compare_candidate_topology";
	status: "error";
	error: { kind: "drawing" | "timeout"; message: string };
}

export interface WorkerExecuteResponse extends WorkerBaseResponse {
	operation: "execute";
	turn: number;
	source_sha256: string;
	execution: ExecutionPayload;
	artifact: ArtifactPayload;
}

export interface WorkerInspectResponse extends WorkerBaseResponse {
	operation: "inspect";
	artifact: ArtifactPayload;
}

export interface PreparedImage {
	source_path: string;
	path: string;
	mime_type: "image/png";
	original_width: number;
	original_height: number;
	width: number;
	height: number;
}

export interface WorkerPrepareResponse extends WorkerBaseResponse {
	operation: "prepare_images";
	images: PreparedImage[];
}

export interface WorkerDoctorDrawing2DSuccess {
	ok: true;
	input_path: string;
	crop_path: string;
	render_path: string;
	comparison_path: string;
	sheet_path: string;
	best_view: DrawingViewSpec;
	score: number;
	margin: number;
}

export interface WorkerDoctorDrawing2DFailure {
	ok: false;
	error: { kind: string; error_type?: string; message: string };
}

export interface WorkerDoctorResponse extends WorkerBaseResponse {
	operation: "doctor";
	ok: boolean;
	python: string;
	build123d: string;
	execution: ExecutionPayload;
	artifact: ArtifactPayload;
	drawing_2d: WorkerDoctorDrawing2DSuccess | WorkerDoctorDrawing2DFailure;
}

export interface WorkerFinalizeResponse extends WorkerBaseResponse {
	operation: "finalize";
	artifact: ArtifactPayload;
	finalized: boolean;
	result_path?: string;
	renders_path?: string;
}

export type NormalizedRoi = [number, number, number, number];
export type NormalizedPoint = [number, number];
export type Vector3 = [number, number, number];

export interface DrawingViewSpec {
	name: string;
	direction: Vector3;
	up: Vector3;
	section_offset_fraction?: number;
	section_keep?: "front" | "back";
}

export interface DrawingSimilarityTransform {
	scale_px_per_unit: number;
	rotation_deg: number;
	center_x_px: number;
	center_y_px: number;
	model_center_x: number;
	model_center_y: number;
	reflected: boolean;
}

export interface WorkerCropDrawingRegionResponse extends WorkerBaseResponse {
	operation: "crop_drawing_region";
	status: "ok";
	region_id: string;
	input_path: string;
	roi: NormalizedRoi;
	crop_path: string;
	width: number;
	height: number;
}

export interface WorkerRenderDrawingViewResponse extends WorkerBaseResponse {
	operation: "render_drawing_view";
	status: "ok";
	region_id: string;
	view_spec: DrawingViewSpec;
	render_path: string;
	width: number;
	height: number;
	step_sha256: string;
}

export interface DrawingRankedCandidate {
	candidate_id: string;
	score: number;
	raw_score: number;
	eligible: boolean;
	failed_gates: string[];
	scoring_mode: "capacity_transport";
	capacity_transport: {
		version: number;
		score: number;
		normalized_cost: number;
		distance_scale_px: number;
		unmatched_cad_cost: number;
		unmatched_reference_cost: number;
		reference_edge_pixels: number;
		cad_edge_pixels: number;
		reference_tokens: number;
		cad_tokens: number;
		matched_pairs: number;
		unmatched_cad_tokens: number;
		unmatched_reference_tokens: number;
		total_cost: number;
		matched_cad_fraction: number;
		matched_reference_fraction: number;
		mean_matched_distance_px: number;
		p90_matched_distance_px: number;
	};
	view_spec: DrawingViewSpec;
	render_path: string;
	comparison_path: string;
	aligned_render_path: string;
	transform: DrawingSimilarityTransform;
	alignment: {
		objective: number;
		mode: "capacity_transport";
		scale_at_bound: boolean;
		rotation_at_limit: boolean;
		translation_at_limit: boolean;
	};
}

export interface WorkerRankDrawingViewsResponse extends WorkerBaseResponse {
	operation: "rank_drawing_views";
	status: "ok";
	region_id: string;
	step_path: string;
	step_sha256: string;
	rankings: DrawingRankedCandidate[];
	margin: number | null;
	comparison_path: string;
	scoring_mode: "capacity_transport";
	scorer_profile: "capacity_transport_v1";
}

export interface WorkerComparisonSheetRegion {
	region_id: string;
	crop_path: string;
	render_path?: string;
	comparison_path?: string;
	view_spec?: DrawingViewSpec;
	score?: number;
	margin?: number | null;
	status: DrawingRegionStatus;
}

export interface WorkerCompareDrawingSheetResponse extends WorkerBaseResponse {
	operation: "compare_drawing_sheet";
	status: "ok";
	sheet_path: string;
	regions: WorkerComparisonSheetRegion[];
}

export interface WorkerReflectionGridOverviewResponse extends WorkerBaseResponse {
	operation: "compose_reflection_inspection";
	status: "ok";
	region_id: string;
	mode: "grid_overview";
	context_path: string;
	width: number;
	height: number;
	sha256: {
		context: string;
	};
}

export interface WorkerReflectionCoordinateOverviewResponse extends WorkerBaseResponse {
	operation: "compose_reflection_inspection";
	status: "ok";
	region_id: string;
	mode: "coordinate_overview";
	context_path: string;
	width: number;
	height: number;
	sha256: {
		context: string;
	};
}

export interface WorkerReflectionGridComparisonResponse extends WorkerBaseResponse {
	operation: "compose_reflection_inspection";
	status: "ok";
	region_id: string;
	mode: "grid_comparison";
	comparison_path: string;
	width: number;
	height: number;
	reference_cell: string;
	cad_cell: string;
	reference_roi: NormalizedRoi;
	cad_roi: NormalizedRoi;
	reference_pixel_roi: [number, number, number, number];
	cad_pixel_roi: [number, number, number, number];
	reference_crop_path: string;
	cad_crop_path: string;
	context_path: string;
	sha256: {
		reference_crop: string;
		cad_crop: string;
		context: string;
		comparison: string;
	};
}

export interface WorkerReflectionDualRoiResponse extends WorkerBaseResponse {
	operation: "compose_reflection_inspection";
	status: "ok";
	region_id: string;
	mode: "dual_roi";
	comparison_path: string;
	width: number;
	height: number;
	reference_roi: NormalizedRoi;
	cad_roi: NormalizedRoi;
	reference_pixel_roi: [number, number, number, number];
	cad_pixel_roi: [number, number, number, number];
	reference_crop_path: string;
	cad_crop_path: string;
	context_path: string;
	sha256: {
		reference_crop: string;
		cad_crop: string;
		context: string;
		comparison: string;
	};
}

export type WorkerReflectionInspectionResponse =
	| WorkerReflectionGridOverviewResponse
	| WorkerReflectionCoordinateOverviewResponse
	| WorkerReflectionGridComparisonResponse
	| WorkerReflectionDualRoiResponse;

export interface WorkerRepairCandidateResponse extends WorkerBaseResponse {
	operation: "execute_repair_candidate";
	candidate_id: string;
	source_path: string;
	source_sha256: string;
	formal_source_sha256: string;
	formal_step_sha256: string;
	formal_unchanged: true;
	execution: ExecutionPayload;
	artifact: ArtifactPayload;
}

export interface WorkerFixedProjectionResponse extends WorkerBaseResponse {
	operation: "render_fixed_projection";
	status: "ok";
	region_id: string;
	step_path: string;
	step_sha256: string;
	view_spec: DrawingViewSpec;
	transform: DrawingSimilarityTransform;
	render_path: string;
	width: number;
	height: number;
	mime_type: "image/png";
	sha256: string;
	projection_bounds?: [number, number, number, number];
	projection_width_units?: number;
	projection_height_units?: number;
	projection_origin?: Vector3;
	projection_anchor_step_sha256?: string;
	/** Physical cutting-plane origin resolved from the immutable projection anchor. */
	section_origin?: Vector3;
}

export interface WorkerRepairComparisonResponse extends WorkerBaseResponse {
	operation: "compose_repair_comparison";
	status: "ok";
	sheet_path: string;
	width: number;
	height: number;
	mime_type: "image/png";
	sha256: string;
}

export interface WorkerCodeConstructionProbeResponse extends WorkerBaseResponse {
	operation: "execute_code_construction_probe";
	group_name: string;
	construction_names: string[];
	probe_status: "renderable" | "not_found" | "not_renderable" | "execution_error";
	reason: string;
	available_names: string[];
	instance_count: number;
	instance_counts: Record<string, number>;
	source_path: string;
	source_sha256: string;
	formal_source_sha256: string;
	formal_step_sha256: string;
	formal_unchanged: true;
	execution: ExecutionPayload;
	artifact: ArtifactPayload;
}

export interface WorkerCodeConstructionHighlightResponse extends WorkerBaseResponse {
	operation: "compose_code_construction_highlight";
	status: "ok";
	region_id: string;
	visual_status: "highlighted" | "no_visible_highlights";
	groups: Array<{
		group_name: string;
		color: "red" | "blue" | "orange" | "magenta";
		visual_status: "highlighted" | "invisible" | "full_model";
		visible_pixel_count: number;
		difference_pixel_count: number;
	}>;
	comparison_path?: string;
	width?: number;
	height?: number;
	mime_type?: "image/png";
	sha256?: string;
}

export interface WorkerCodeAblationHighlightResponse extends WorkerBaseResponse {
	operation: "compose_code_ablation_highlight";
	status: "ok";
	region_id: string;
	construction_ref: string;
	visual_status: "highlighted" | "no_visible_change";
	current_pixel_count: number;
	removed_pixel_count: number;
	revealed_pixel_count: number;
	comparison_path?: string;
	width?: number;
	height?: number;
	mime_type?: "image/png";
	sha256?: string;
}

export interface WorkerVisualDiffResponse extends WorkerBaseResponse {
	operation: "compose_visual_diff";
	status: "ok";
	region_id: string;
	diff_id: string;
	construction_ref: string;
	target_roi: NormalizedRoi;
	target_pixel_roi: [number, number, number, number];
	target_geometry_pixel_bbox: [number, number, number, number];
	current_geometry_pixel_bbox: [number, number, number, number];
	comparison_path: string;
	target_mask_path: string;
	current_mask_path: string;
	metrics: {
		overlap_pixel_count: number;
		target_only_pixel_count: number;
		current_only_pixel_count: number;
		iou: number;
		centroid_delta_px: [number, number];
	};
	width: number;
	height: number;
	mime_type: "image/png";
	sha256: {
		comparison: string;
		target_mask: string;
		current_mask: string;
	};
}

export interface VisualReviewGeometryEffect {
	relation: "removed_by_proposal" | "added_by_proposal";
	solid_count: number;
	face_count: number;
	edge_count: number;
	volume_mm3: number;
	surface_area_mm2: number;
	bbox_mm: {
		x: number;
		y: number;
		z: number;
		x_min: number;
		x_max: number;
		y_min: number;
		y_max: number;
		z_min: number;
		z_max: number;
	};
	centroid_mm: Vector3;
	surface_types: string[];
}

export interface CodeHypothesisVisualEncoding {
	current_cad: "gray";
	removed_by_proposal: "red";
	added_by_proposal: "green";
}

export interface WorkerCodeEditVisualizationResponse extends WorkerBaseResponse {
	operation: "compose_code_edit_visualization";
	status: "ok";
	region_id: string;
	hypothesis_id: string;
	target_roi: NormalizedRoi;
	target_pixel_roi: [number, number, number, number];
	visual_encoding: CodeHypothesisVisualEncoding;
	geometry: VisualReviewGeometryEffect[];
	card_path: string;
	width: number;
	height: number;
	mime_type: "image/png";
	sha256: string;
}

export interface RepairPreviewTopology {
	b0: number;
	b1: number;
	b2: number;
	chi_surface: number;
	n_components: number;
	n_triangles: number;
	n_vertices: number;
	linear_deflection_mm: number;
}

export interface WorkerCandidateTopologyResponse extends WorkerBaseResponse {
	operation: "compare_candidate_topology";
	status: "ok";
	current_step_path: string;
	current_step_sha256: string;
	current_topology: RepairPreviewTopology | null;
	current_error: string | null;
	candidates: Array<{
		candidate_id: string;
		step_path: string;
		step_sha256: string;
		status: "ok" | "error";
		topology: RepairPreviewTopology | null;
		topology_delta: { b0: number; b1: number; b2: number } | null;
		error: string | null;
	}>;
}

export interface WorkerCandidateTopologyErrorResponse extends WorkerDrawingErrorResponse {
	operation: "compare_candidate_topology";
}

export interface WorkerRepairPreviewCandidate {
	candidate_id: string;
	hypothesis: string;
	topology: RepairPreviewTopology;
	topology_delta: { b0: number; b1: number; b2: number };
	geometry: VisualReviewGeometryEffect[];
}

export interface WorkerRepairCandidatePreviewResponse extends WorkerBaseResponse {
	operation: "compose_repair_candidate_preview";
	status: "ok";
	region_id: string;
	current_topology: RepairPreviewTopology;
	candidates: WorkerRepairPreviewCandidate[];
	visual_encoding: CodeHypothesisVisualEncoding;
	card_path: string;
	width: number;
	height: number;
	mime_type: "image/png";
	sha256: string;
}

/** Separate Target, Current and Preview panels from verified aligned projections. */
export interface WorkerShapeFeedbackCard {
	candidate_id: string;
	card_path: string;
	sha256: string;
	width: number;
	height: number;
	mime_type: "image/png";
	card_layout: "three_panel_row";
	panel_count: 3;
	panel_titles: string[];
}

export interface WorkerShapeFeedbackResponse extends WorkerBaseResponse {
	operation: "compose_shape_feedback";
	status: "ok";
	region_id: string;
	card_path: string;
	sha256: string;
	width: number;
	height: number;
	card_layout: "three_panel_row";
	card_count: number;
	cards: WorkerShapeFeedbackCard[];
	reference_sha256: string;
	current_render_sha256: string;
	candidates: Array<{ candidate_id: string; render_sha256: string }>;
	target_roi: NormalizedRoi;
	target_pixel_roi?: [number, number, number, number];
	files: Record<string, unknown>;
	diagnostics: Record<string, unknown>;
	warnings?: string[];
}

export type WorkerResponse =
	| WorkerExecuteResponse
	| WorkerInspectResponse
	| WorkerPrepareResponse
	| WorkerDoctorResponse
	| WorkerFinalizeResponse
	| WorkerCropDrawingRegionResponse
	| WorkerRenderDrawingViewResponse
	| WorkerRankDrawingViewsResponse
	| WorkerCompareDrawingSheetResponse
	| WorkerReflectionInspectionResponse
	| WorkerRepairCandidateResponse
	| WorkerFixedProjectionResponse
	| WorkerRepairComparisonResponse
	| WorkerCodeConstructionProbeResponse
	| WorkerCodeConstructionHighlightResponse
	| WorkerCodeAblationHighlightResponse
	| WorkerVisualDiffResponse
	| WorkerCodeEditVisualizationResponse
	| WorkerRepairCandidatePreviewResponse
	| WorkerCandidateTopologyResponse
	| WorkerShapeFeedbackResponse
	| WorkerDrawingErrorResponse
	| WorkerBaseResponse;

export type DrawingRegionStatus = "matched" | "ambiguous" | "error";
export type DrawingViewType = "orthographic" | "isometric" | "section" | "detail" | "unknown";
export type DrawingConfidence = "high" | "medium" | "low";

export interface DrawingRegionManifestRegion {
	region_id: string;
	page: number;
	roi: NormalizedRoi;
	crop_path: string;
	view_type_hint: DrawingViewType;
	confidence: DrawingConfidence;
	note: string;
	width: number;
	height: number;
	crop_sha256: string;
}

/**
 * A persistent, CAD-independent split of the merged input drawings.
 * Its identity and contents depend only on input_sha256.
 */
export interface DrawingRegionManifest {
	version: 1;
	input_sha256: string;
	regions: DrawingRegionManifestRegion[];
}

export interface DrawingViewCandidate extends DrawingRankedCandidate {
	comparison_path: string;
	/** Diagnostics preserved when an experiment replays a historical locked camera on edited geometry. */
	locked_replay_original_eligible?: boolean;
	locked_replay_original_failed_gates?: string[];
}

export interface DrawingManifestRegion {
	region_id: string;
	page: number;
	roi: NormalizedRoi;
	crop_path: string;
	view_type: DrawingViewType;
	status: DrawingRegionStatus;
	selected?: DrawingViewCandidate;
	alternatives: DrawingViewCandidate[];
	margin: number | null;
	confidence: DrawingConfidence;
	note: string;
	/** Present on manifests produced by the v2 segmentation pipeline. */
	width?: number;
	/** Present on manifests produced by the v2 segmentation pipeline. */
	height?: number;
	/** Present on manifests produced by the v2 segmentation pipeline. */
	crop_sha256?: string;
}

export interface DrawingViewManifest {
	version: 3;
	scorer_profile: "capacity_transport_v1";
	input_sha256: string;
	source_sha256: string;
	step_sha256: string;
	/** Preserve calibrated image coordinates while formal CAD geometry changes. */
	projection_frame?: {
		policy: "fixed";
		anchor_step_path: string;
		anchor_step_sha256: string;
	};
	regions: DrawingManifestRegion[];
}

export interface ReflectionFinding {
	region_id: string;
	obs: string;
	suggested_change: string;
}

export interface DrawingReflectionReport {
	should_revise: boolean;
	overall_summary: string;
	findings: ReflectionFinding[];
}

export interface SubagentRunUsage extends UsageTotals {
	role: "segmentation" | "view_search" | "reflection" | "geometry_repair" | "dimension_annotation";
	region_id?: string;
	accounting_id: string;
	session_file?: string;
	result_path?: string;
	started_at: string;
	finished_at: string;
	duration_s: number;
	status: "success" | "error" | "interrupted" | "timeout";
	error?: string;
}

export interface DrawingFeedbackSummary {
	cache_key: string;
	created_at: string;
	input_sha256: string;
	source_sha256: string;
	step_sha256: string;
	segmentation_mode?: "segmented" | "reused";
	search_mode: "searched" | "reused";
	region_manifest_path?: string;
	manifest_path: string;
	report_path: string;
	sheet_path: string;
	inspection_manifest_path?: string;
	region_count: number;
	matched_count: number;
	ambiguous_count: number;
	error_count: number;
	should_revise: boolean;
	usage: {
		segmentation?: SubagentRunUsage;
		view_search?: SubagentRunUsage[];
		reflection: SubagentRunUsage;
		total: UsageTotals;
	};
}

export interface MainViewReflectInspectedFormalState {
	source_sha256: string;
	step_sha256: string;
}

export interface MainViewReflectSummary {
	version: 1;
	kind: "main_view_reflect";
	locked_view_manifest_sha256: string;
	baseline_source_sha256: string;
	baseline_step_sha256: string;
	audit_manifest_path: string;
	inspection_count: number;
	crop_count: number;
	inspected_formal_states: MainViewReflectInspectedFormalState[];
	cropped_formal_states: MainViewReflectInspectedFormalState[];
}

export interface CausalRepairSummary {
	version: 1;
	profile: CausalRepairProfile;
	audit_manifest_path: string;
	evaluation_count: number;
	candidate_count: number;
	formalization_count: number;
}

export interface FixtureDefinition {
	id: string;
	description: string;
	inputFiles: string[];
	inputType: string;
	taskType: string;
	sourceDir: string;
}

export interface UsageTotals {
	tokens: {
		input: number;
		output: number;
		cache_read: number;
		cache_write: number;
		total: number;
	};
	cost: number;
	tool_calls: number;
}

export interface AgentResult extends UsageTotals {
	fixture_id: string;
	status: RunStatus;
	agent_mode?: PiAgentMode;
	provider: string;
	model: string;
	thinking_level: string;
	pi_version: string;
	container_image: string;
	container_image_digest: string;
	started_at: string;
	updated_at: string;
	duration_s: number;
	session_file?: string;
	last_turn: number;
	artifact: ArtifactPayload | null;
	last_executed_source_sha256?: string;
	last_executed_step_sha256?: string;
	stop_reason?: string;
	resume_reason?: string;
	main_agent_usage?: UsageTotals;
	subagent_usage?: UsageTotals;
	subagent_runs?: SubagentRunUsage[];
	drawing_feedback?: DrawingFeedbackSummary;
	main_view_reflection?: MainViewReflectSummary;
	dimension_annotations?: DimensionAnnotationSummary;
	causal_repair?: CausalRepairSummary;
	recorded_subagent_usage_ids?: string[];
	applied_continuation?: {
		id: string;
		prompt_sha256: string;
		applied_at: string;
	};
	optional_reflect_turn?: {
		continuation_id: string;
		prompt_sha256: string;
		status: "running" | "completed";
		started_at: string;
		completed_at?: string;
	};
}

export interface RunParams {
	protocol_version: 5;
	runtime_mode: "persistent-agent-container";
	initialization_status?: "pending" | "complete";
	run_id: string;
	created_at: string;
	agent_mode?: PiAgentMode;
	provider: string;
	model: string;
	thinking_level: string;
	pi_version: string;
	container_image: string;
	container_image_digest: string;
	data_inputs_dir: string;
	fixtures: string[];
	parallel: number;
	mode: "fixtures" | "smoke" | "all-generation";
}

export interface LegacyRunParams extends Omit<RunParams, "protocol_version" | "runtime_mode"> {
	protocol_version: 1 | 3 | 4;
}

export interface AgentJob {
	protocol_version: 5;
	operation: "run_fixture";
	run_id: string;
	agent_mode?: PiAgentMode;
	fixture: FixtureDefinition;
	staged_paths: string[];
	provider: string;
	model: string;
	thinking_level: string;
	pi_version: string;
	container_image: string;
	container_image_digest: string;
	continuation?: {
		kind?: PiContinuationKind;
		id: string;
		prompt: string;
		image_paths?: string[];
		/** Direct reflection from the existing Initial session, with no new image feedback. */
		no_new_images?: boolean;
		baseline_source_sha256: string;
		baseline_step_sha256: string;
		locked_view_manifest_path?: string;
		locked_view_manifest_sha256?: string;
		causal_repair_profile?: CausalRepairProfile;
		dual_roi_crop_profile?: DualRoiCropProfile;
		visual_finding_profile?: VisualFindingProfile;
		visual_diff_profile?: VisualDiffProfile;
		visual_review_profile?: VisualReviewProfile;
		code_construction_profile?: CodeConstructionProfile;
		target_annotation_profile?: TargetAnnotationProfile;
		dimension_annotation_profile?: DimensionAnnotationProfile;
	};
}

export interface DoctorJob {
	protocol_version: 5;
	operation: "doctor";
	provider: string;
	model: string;
}

export type ContainerJob = AgentJob | DoctorJob;

export interface ContainerResponse {
	protocol_version: 5;
	operation: ContainerJob["operation"];
	ok: boolean;
	model_inputs?: string[];
	node?: string;
	pi_version?: string;
	doctor?: WorkerDoctorResponse;
	error?: string;
}

export interface ContainerState {
	protocol_version: 5;
	container_name: string;
	container_id: string;
	image: string;
	image_id: string;
	workspace: string;
	pi_agent_dir: string;
	status: "created" | "running" | "stopped";
	created_at: string;
	started_at?: string;
	updated_at: string;
	exit_code?: number;
}

export interface CandidateInspection {
	artifact: ArtifactPayload;
	formal_execution_current: boolean;
	reason: string;
	source_sha256: string | null;
}

export interface PiRunSummary extends UsageTotals {
	run_id: string;
	status: RunStatus;
	agent_mode: PiAgentMode;
	provider: string;
	model: string;
	pi_version: string;
	container_image_digest: string;
	created_at: string;
	updated_at: string;
	duration_s: number;
	fixture_count: number;
	completed_count: number;
	valid_count: number;
	completion_rate: number;
	validity_rate: number;
	results: Array<{
		fixture_id: string;
		status: RunStatus;
		is_valid: boolean | null;
		is_watertight: boolean | null;
		duration_s: number;
		cost: number;
		tool_calls: number;
	}>;
	stop_reason?: string;
}
