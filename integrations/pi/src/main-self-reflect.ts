import { createHash } from "node:crypto";

import { InfrastructureError } from "./errors.js";
import { GEOMETRY_DIFFERENCE_INSTRUCTIONS, GEOMETRY_REPAIR_COMPLETION_INSTRUCTION } from "./geometry-repair-instructions.js";

import {
	CAUSAL_REPAIR_PROFILE,
	CODE_CONSTRUCTION_PROFILE,
	DIMENSION_ANNOTATION_PROFILE,
	DUAL_ROI_CROP_PROFILE,
	GEOMETRY_REPAIR_PROFILE,
	REPAIR_PREVIEW_PROFILE,
	SHAPE_FEEDBACK_PROFILE,
	TARGET_ANNOTATION_PROFILE,
	VISUAL_FINDING_PROFILE,
	VISUAL_DIFF_PROFILE,
	VISUAL_REVIEW_PROFILE,
	type AgentJob,
	type AgentResult,
	type CausalRepairProfile,
	type CodeConstructionProfile,
	type DimensionAnnotationProfile,
	type DualRoiCropProfile,
	type TargetAnnotationProfile,
	type VisualFindingProfile,
	type VisualDiffProfile,
	type VisualReviewProfile,
} from "./types.js";

export const MAIN_SELF_REFLECT_KIND = "main_self_reflect" as const;
export const MAIN_SELF_REFLECT_INSPECT_KIND = "main_self_reflect_inspect" as const;

export type SelfReflectSourceKind = "original" | "main_self_reflect" | "incompatible";

export function classifySelfReflectSource(job: AgentJob, result: AgentResult): SelfReflectSourceKind {
	const continuation = job.continuation;
	const applied = result.applied_continuation;
	if (continuation === undefined && applied === undefined) return "original";
	if (
		(continuation?.kind === MAIN_SELF_REFLECT_KIND || continuation?.kind === MAIN_SELF_REFLECT_INSPECT_KIND) &&
		applied?.id === continuation.id &&
		applied.prompt_sha256 === createHash("sha256").update(continuation.prompt).digest("hex")
	) return "main_self_reflect";
	return "incompatible";
}

export function mainSelfReflectPrompt(fixtureId: string): string {
	return [
		`Continue CADGenBench fixture ${fixtureId} in this exact existing Pi session.`,
		"The controller preserved the previous formal solution.py, output.step, render, result, and session under /workspace/baseline/. Never edit anything under baseline/.",
		"The attached images are only the original drawing pages from the original task. No additional images or external analysis are supplied.",
		"Review the original drawings and the live /workspace/solution.py yourself. Before deciding what to change, first call execute_build123d({}) on the current live source and inspect its fresh formal render.",
		"Identify any missing, extra, incorrectly shaped, incorrectly dimensioned, or misaligned geometry yourself. Then directly improve the live solution.py; do not stop at an explanation, plan, or separate analysis report. Preserve geometry that is already correct.",
		"Iterate with execute_build123d({}) and inspect every formal render. Continue until both solution.py and output.step differ from the preserved baseline, and the final STEP is valid and watertight.",
	].join("\n\n");
}

export function mainSelfReflectNoNewImagesPrompt(fixtureId: string, isChained = false): string {
	return [
		isChained
			? `Continue CADGenBench fixture ${fixtureId} in this exact existing Pi session, retaining the full history of the previous completed reflection round.`
			: `Continue CADGenBench fixture ${fixtureId} in this exact existing Initial Pi session.`,
		"Use only the original task, drawing and CAD evidence already present in this session, together with the live solution.py. No new images are attached or provided by tools. Do not request, open or generate additional CAD images, aligned views, dimension annotations or external analysis.",
		isChained
			? "The controller preserved the source, STEP, render and full session from the previous completed reflection round under /workspace/baseline/. Never edit baseline/."
			: "The controller preserved the Initial source, STEP, render and session under /workspace/baseline/. Never edit baseline/.",
		isChained
			? "Reflect on missing, extra, incorrectly shaped, incorrectly dimensioned or misaligned geometry using the existing evidence. Make focused edits to the live solution.py only when they address a supported discrepancy. Preserve correct geometry; keeping the previous round's output unchanged is allowed when no justified improvement remains."
			: "Reflect on missing, extra, incorrectly shaped, incorrectly dimensioned or misaligned geometry using the existing evidence. Make focused edits to the live solution.py only when they address a supported discrepancy. Preserve correct geometry; keeping Initial unchanged is allowed when no justified improvement remains.",
		"After any source edit call execute_build123d({}) to produce and validate a fresh output.step. The tool returns execution and validity diagnostics without render images. Continue until the current source and STEP are formally bound, valid and watertight, then finish your reflection.",
	].join("\n\n");
}

function visualFindingInstruction(): string {
	return "Use `update_visual_findings` to maintain the important unresolved differences supported by the 2D evidence. The persistent collection may initially be empty and has no fixed finding count; never recreate or clear it at the start of a new iteration. For each finding, describe what the target drawing depicts in `target_geometry`, what the Current CAD depicts in `current_geometry`, and their specific unresolved geometric mismatch in `difference`. Reconcile existing findings before adding new ones. Update the same finding when its evidence changes, rewriting it to contain only the remaining unresolved mismatch, and delete it only when its entire difference is resolved or unsupported.";
}

function codeAblationInstruction(visualFindingsEnabled: boolean, visualDiffsEnabled: boolean): string {
	const findingBoundary = visualDiffsEnabled
		? "before comparing the related geometry or editing its code"
		: visualFindingsEnabled
			? "before recording that finding or editing the related code"
			: "before deciding on or making the related code edit";
	const referenceGuidance = visualFindingsEnabled
		? "A `construction_ref` precisely identifies only the highlighted Current CAD geometry and its code. Put it in the finding's `current_construction_refs` only after confirming the target correspondence. A recorded reference is not cleared when its code changes: the controller rebinds it to the modified source block. Reinspect that construction by passing its `construction_ref` to `inspect_code_ablation`; do not resubmit its code. You may add or remove references when updating a finding."
		: visualDiffsEnabled
			? "A `construction_ref` identifies only the highlighted Current CAD geometry and its code. Pass it with the Target ROI to `visual_diff`. If the code changes, re-highlight the rebound construction before comparing it again."
		: "A `construction_ref` precisely identifies only the highlighted Current CAD geometry and its code.";
	const missingGuidance = visualFindingsEnabled
		? "If the target structure is completely missing from the Current CAD, still inspect plausible related existing constructions so that another Current structure is not mistaken for its counterpart. After confirming that no Current counterpart exists, `current_construction_refs` may be empty."
		: visualDiffsEnabled
			? "Visual diff compares an existing Current construction. If the Target structure is completely missing from Current, use the complete comparison instead of inventing a Current selection."
		: "If the target structure is completely missing from the Current CAD, still inspect plausible related existing constructions so that another Current structure is not mistaken for its counterpart.";
	const reinspectionGuidance = visualFindingsEnabled
		? "After formally executing a modified `solution.py`, inspect the 2D comparison again and reconcile the existing findings before making another edit. For an affected finding whose registered construction code changed, call `inspect_code_ablation` with its existing `construction_ref` to highlight the rebound modified block before updating the finding or editing the related code again."
		: visualDiffsEnabled
			? "After formally executing modified source, re-highlight affected constructions and call `visual_diff` again before relying on the old comparison."
		: "After formally executing a modified `solution.py`, inspect the 2D comparison again before deciding whether an affected difference remains.";
	return [
		`After finding any important unresolved visual difference in a 2D comparison, ${findingBoundary}, you must use \`inspect_code_ablation\` to inspect a code block that may produce the relevant Current CAD structure.`,
		visualDiffsEnabled
			? "Inspect the returned highlight, then make one `visual_diff` call with that construction_ref and the corresponding Target ROI. Read the returned same-coordinate overlay directly; do not replace it with a prose geometry description."
			: "Inspect the returned highlight and use the highlighted geometry as the visual anchor for the Current CAD. Then identify the corresponding structure in the target engineering drawing, describe the highlighted Current CAD structure, and state their concrete differences in count, shape, extent, orientation, relative position, connectivity, and spatial relationship to adjacent structures.",
		"If the first highlight does not identify the relevant structure, you must inspect another, more appropriate code block. Do not determine the difference from source-code interpretation or unverified visual judgment alone.",
		referenceGuidance,
		"The red overlay on the target side is only a projection of the Current CAD geometry, not a segmentation of the target structure and not proof that the two correspond. Use the complete view, nearby structures, and connectivity to identify the target counterpart.",
		"Before describing a bead, edge, contour, or path as open or interrupted, trace it to a visible endpoint in the target drawing. A junction, overlap, change of direction, occlusion, or mismatch with the projected Current CAD outline is not evidence of a break.",
		missingGuidance,
		reinspectionGuidance,
	].join(" ");
}

function visualDiffInstruction(): string {
	return "Use `visual_diff` instead of natural-language visual findings. Select the Current geometry once with `inspect_code_ablation`, then pass only its construction_ref and the corresponding Target ROI to `visual_diff`. The returned image uses one coordinate system: gray is overlap, orange exists only in Target, and blue exists only in Current. No relation, difference tags, proposal, confirmation, or resolution step is required. If the selection is wrong, call `visual_diff` again with a corrected ROI or construction.";
}

function visualCodeEditInstruction(): string {
	return "When you have prepared a set of exact edits for `solution.py` and want to determine whether they make a local structure in the Current CAD closer to the Target drawing, you may first call `visualize_code_hypothesis`. Select the corresponding local structure in the Target drawing as the ROI and submit the matching `oldText/newText`. Compare the two returned panels directly: apply the same edits to `solution.py` only when the Edit Preview on the right is closer to the Target structure on the left than the CAD before editing; otherwise revise the code and preview it again.";
}

function repairCandidatePreviewInstruction(): string {
	return "When you have identified a visual defect and want to compare alternative ways to repair it, you may call `preview_repair_candidates` before editing the formal `solution.py`. Provide two or three distinct code-edit hypotheses for the same defect. Compare the full Original drawing, Current CAD, and highlighted candidate results, and use the reported b0/b1/b2 changes when the defect may involve connected components, through-holes, or enclosed voids. The candidates may all be wrong: adopt, revise, combine, or reject them according to whether their geometry is closer to the Original drawing. After choosing a repair, edit the formal `solution.py` yourself and validate it with `execute_build123d({})`.";
}

function shapeFeedbackInstruction(): string {
	return "When a discrepancy in the aligned 2D comparisons suggests possible code changes, you may use `preview_shape_feedback` before modifying the formal CAD. Submit `candidates` as an array containing exactly one candidate per call, with only an `id` and one to eight `edits`. Select one to three views that show the relevant geometry. The edits are applied to a temporary copy of the Current source, never a previous unadopted preview. Inspect one three-panel comparison per view, arranged horizontally as Target | Current | Preview. Target shows the engineering drawing; Current and Preview show separate 2D CAD projections using the same camera, 2D transform, and display scale. The image contains only the three column labels and actual views, preserving original drawing annotations. Compare both CAD projections against Target, checking whether the edit improves the intended geometry and preserves geometry that is already correct. If you judge that the candidate reduces the geometric discrepancy between the current CAD and the target engineering drawing, call `adopt_shape_candidate` with that candidate's `id` from the most recent preview as `candidate_id`, then call `execute_build123d({})` and inspect the final formal CAD with `inspect_comparison`. Otherwise, revise the edit and preview again, or keep Current. A preview does not replace the final formal CAD inspection.";
}

function geometryRepairInstruction(): string {
	return [
		"Inspect the target engineering drawing and the current CAD in the 2D comparison views for geometric discrepancies. When you notice a possible issue, call `repair_geometry` so a SubAgent can independently verify it against the drawing. If the discrepancy is confirmed, the SubAgent will modify the code, inspect previews, and propose a repair candidate.",
		"To help the SubAgent locate the same geometry you mean, identify one part or group of related features in `part` and provide the corresponding `view_ids`. When needed, describe its position in that view and its neighboring features or connections, for example, 'the support plate to the left of the central oblong hole, connected to the base plate'.",
		"To keep each repair focused and its result easy to check, submit only one atomic problem per call: one specific geometric requirement for the identified part or feature group. In `suspected_difference`, describe what the drawing shows at that location and how the current CAD differs. State any uncertainty for the SubAgent to verify. You do not need to provide code or decide how to implement the repair.",
		"Choose one `difference_type` according to the main problem to resolve:",
		...Object.values(GEOMETRY_DIFFERENCE_INSTRUCTIONS).map((instruction, index) => `${index + 1}. ${instruction}`),
		"Classify by the repair goal, and allow associated changes necessary to achieve it. For example, adding a missing mounting foot is a `count` repair and may include creating its required connecting segment and mounting hole. Connecting two existing supports that are disconnected is a `connectivity` repair, even if it also requires adding a solid. Submit other independent problems separately, even when they share the same `difference_type`.",
		"After receiving a candidate, compare Current and Preview against the target drawing. Check whether the assigned problem has improved and whether other correct geometry has been damaged. If you judge that the candidate better matches the drawing without introducing other errors, call `adopt_shape_candidate` to adopt it. Otherwise, describe the remaining problem and call `repair_geometry` again, or discard the candidate.",
	].join("\n\n");
}

function targetAnnotationInstruction(): string {
	return "To ground visual differences and code edits in precisely identified Target structures, use `annotate_target` to add persistent numbered marks to geometric structures in the Target engineering drawing. Each number provides a visual reference, while its name and description convey its semantics. Inspect the returned image, and refer to each marked structure as `[number] name` in subsequent reasoning. Handle only one atomic Target-drawing discrepancy per iteration. After making the corresponding CAD change, you must inspect the updated comparison and verify that the result is closer to the Target engineering drawing.";
}

export function mainSelfReflectInspectPrompt(
	fixtureId: string,
	causalRepairProfile?: CausalRepairProfile,
	dualRoiCropProfile?: DualRoiCropProfile,
	visualFindingProfile?: VisualFindingProfile,
	codeConstructionProfile?: CodeConstructionProfile,
	visualDiffProfile?: VisualDiffProfile,
	visualReviewProfile?: VisualReviewProfile,
	targetAnnotationProfile?: TargetAnnotationProfile,
	dimensionAnnotationProfile?: DimensionAnnotationProfile,
): string {
	const paragraphs = [
		`Continue CADGenBench fixture ${fixtureId} in this exact existing Pi session.`,
		"The controller preserved the previous formal solution.py, output.step, render, result, and session under /workspace/baseline/. Never edit anything under baseline/.",
		dimensionAnnotationProfile === DIMENSION_ANNOTATION_PROFILE
			? "Review the original drawings and the live /workspace/solution.py yourself. The controller checks the current formally executed CAD and prepares dimensions before your reflection begins. First call inspect_comparison({ region_id }) to inspect the aligned CAD and its actual dimensions before deciding what to change. After editing solution.py, call execute_build123d({}) and inspect the refreshed comparison."
			: "Review the original drawings and the live /workspace/solution.py yourself. The controller checks the current formally executed CAD before your reflection begins. First call inspect_comparison({ region_id }) to inspect the aligned CAD before deciding what to change. After editing solution.py, call execute_build123d({}) and inspect the refreshed comparison.",
		visualFindingProfile === VISUAL_FINDING_PROFILE
			? dualRoiCropProfile === DUAL_ROI_CROP_PROFILE
				? "Use inspect_comparison({ region_id }) to examine the aligned 2D comparison between the input drawing and the CAD generated by your execution. Inspect one region_id image at a time and examine it before requesting another Inspect or Crop image. The tool description lists the available region_id values."
				: "Use inspect_comparison({ region_id }) to examine the aligned 2D comparison between the input drawing and the CAD generated by your execution. Inspect one region_id image at a time and examine it before requesting another Inspect image. The tool description lists the available region_id values."
			: "Use inspect_comparison({ region_id }) to examine the aligned 2D comparison between the input drawing and the CAD generated by your execution, as needed during reflection and for the final review described below. The tool description lists the available region_id values.",
	];
	if (dimensionAnnotationProfile === DIMENSION_ANNOTATION_PROFILE) {
		paragraphs.push("A dedicated dimension-annotation SubAgent measures the current CAD and adds dimensions to the locked aligned views before you begin reflection and after every successful formal execution. Use `inspect_comparison({ region_id })` to read the engineering drawing beside the dimension-annotated Current CAD. The measurements come from the current executed STEP; compare them with the drawing's dimension labels and verify that they refer to the same features. The SubAgent only selects and measures geometry and refreshes annotations; you remain responsible for identifying discrepancies, editing solution.py, and deciding whether to adopt a repair. After changing source, call `execute_build123d({})` and inspect the refreshed comparison before relying on dimensions. Entity IDs are local to one STEP revision: never reuse an old face or edge ID after the CAD changes. An unresolved or unsupported measurement is not evidence that the CAD matches the drawing.");
	}
	if (targetAnnotationProfile === TARGET_ANNOTATION_PROFILE) {
		paragraphs.push(targetAnnotationInstruction());
	}
	if (dualRoiCropProfile === DUAL_ROI_CROP_PROFILE) {
		paragraphs.push("Use `dual_roi_crop` to enlarge corresponding local structures on the two sides and inspect the Current CAD's shape, size, position, orientation, connectivity, and feature correctness. You may inspect new corresponding ROIs for different fine-grained differences. After local inspection, continue examining the complete view before deciding whether and how to modify the code.");
	}
	if (visualFindingProfile === VISUAL_FINDING_PROFILE) {
		paragraphs.push(visualFindingInstruction());
	}
	if (visualDiffProfile === VISUAL_DIFF_PROFILE) {
		paragraphs.push(visualDiffInstruction());
	}
	if (visualReviewProfile === VISUAL_REVIEW_PROFILE) {
		paragraphs.push(visualCodeEditInstruction());
	} else if (visualReviewProfile === REPAIR_PREVIEW_PROFILE) {
		paragraphs.push(repairCandidatePreviewInstruction());
	} else if (visualReviewProfile === SHAPE_FEEDBACK_PROFILE) {
		paragraphs.push(shapeFeedbackInstruction());
	}
	if (causalRepairProfile === CAUSAL_REPAIR_PROFILE) {
		paragraphs.push("When you are ready to modify the formal `solution.py` from visual evidence but need to verify which repair is more appropriate, you may first use `compare_repair_hypotheses`. Its two or three candidates should represent materially different repair hypotheses rather than different parameters for the same change. The candidates are only hypotheses and may all be wrong. Judge whether each candidate makes the Current CAD closer to the original engineering drawing and whether all candidates preserve the same error. If none genuinely improves the model, reject them all and form a new hypothesis. You may use the tool again for different problems or hypotheses. After selecting a repair, modify the formal `solution.py` and call `execute_build123d({})`.");
	}
	if (codeConstructionProfile === CODE_CONSTRUCTION_PROFILE) {
		paragraphs.push(codeAblationInstruction(
			visualFindingProfile === VISUAL_FINDING_PROFILE,
			visualDiffProfile === VISUAL_DIFF_PROFILE,
		));
	}
	paragraphs.push(
		visualReviewProfile === GEOMETRY_REPAIR_PROFILE
			? geometryRepairInstruction()
			: "Identify any missing, extra, incorrectly shaped, incorrectly dimensioned, or misaligned geometry yourself. When the drawing supports a clear, actionable discrepancy, directly improve the live solution.py; do not stop at an explanation, plan, or separate analysis report. Preserve geometry that is already correct.",
		...(visualReviewProfile === GEOMETRY_REPAIR_PROFILE ? [GEOMETRY_REPAIR_COMPLETION_INSTRUCTION] : [
			"Iterate with execute_build123d({}) and inspect every formal render.",
			"Before finishing, inspect the aligned 2D comparisons of the final formal CAD in the views relevant to your edits. If you made no edits, inspect the views relevant to the suspected discrepancy. Continue if these comparisons reveal a clear, actionable mismatch. Otherwise, finish with the current model, ensuring that the final STEP is valid and watertight. The final model may remain unchanged from the baseline.",
		]),
	);
	return paragraphs.join("\n\n");
}

export function auxiliaryStateFields(result: AgentResult): string[] {
	const fields: string[] = [];
	if (result.drawing_feedback !== undefined) fields.push("drawing_feedback");
	if (result.subagent_usage !== undefined) fields.push("subagent_usage");
	if (result.subagent_runs !== undefined) fields.push("subagent_runs");
	if (result.recorded_subagent_usage_ids !== undefined) fields.push("recorded_subagent_usage_ids");
	if (result.dimension_annotations !== undefined) fields.push("dimension_annotations");
	return fields;
}

/** Preserve baseline isolation while permitting only explicitly selected auxiliary roles. */
export function assertSelfReflectAuxiliaryState(
	result: AgentResult,
	profile?: VisualReviewProfile,
	dimensionProfile?: DimensionAnnotationProfile,
): void {
	const roles = [
		...(profile === GEOMETRY_REPAIR_PROFILE ? ["geometry_repair"] : []),
		...(dimensionProfile === DIMENSION_ANNOTATION_PROFILE ? ["dimension_annotation"] : []),
	];
	if (roles.length === 0) {
		if (auxiliaryStateFields(result).length > 0) throw new InfrastructureError("Reflection/subagent state is forbidden in Self Reflect plus Inspect");
		return;
	}
	if (result.drawing_feedback !== undefined) throw new InfrastructureError("Drawing reflection state is forbidden in Self Reflect plus Inspect auxiliary profiles");
	if (result.dimension_annotations !== undefined && dimensionProfile !== DIMENSION_ANNOTATION_PROFILE) {
		throw new InfrastructureError("Dimension annotation state requires the dimension annotation profile");
	}
	const runs = result.subagent_runs;
	if (runs === undefined) {
		if (result.subagent_usage !== undefined || result.recorded_subagent_usage_ids !== undefined) {
			throw new InfrastructureError("Auxiliary usage must have corresponding subagent run records");
		}
		return;
	}
	if (!Array.isArray(runs) || runs.some((run) => !run || !roles.includes(run.role) || typeof run.accounting_id !== "string" || !run.accounting_id)) {
		throw new InfrastructureError(`Only ${roles.join(" and ")} subagent records are allowed in the selected Self Reflect profiles`);
	}
	const ids = runs.map(({ accounting_id }) => accounting_id);
	const recorded = result.recorded_subagent_usage_ids;
	if (!Array.isArray(recorded) || new Set(ids).size !== ids.length || new Set(recorded).size !== recorded.length || recorded.length !== ids.length || recorded.some((id) => !ids.includes(id)) || (runs.length > 0 && result.subagent_usage === undefined)) {
		throw new InfrastructureError("Auxiliary usage records are incomplete or duplicated");
	}
}

export class CaseTimeoutError extends Error {
	constructor(readonly timeoutMs: number) {
		super(`Case exceeded the isolated ${timeoutMs} ms wall-clock limit`);
		this.name = "CaseTimeoutError";
	}
}

export async function withIsolatedTimeout<T>(options: {
	operation: Promise<T>;
	timeoutMs: number;
	onTimeout: () => Promise<void>;
}): Promise<T> {
	let timer: ReturnType<typeof setTimeout> | undefined;
	const timeout = new Promise<never>((_resolve, reject) => {
		timer = setTimeout(() => {
			void options.onTimeout().then(
				() => reject(new CaseTimeoutError(options.timeoutMs)),
				(error: unknown) => reject(error),
			);
		}, options.timeoutMs);
	});
	try {
		return await Promise.race([options.operation, timeout]);
	} finally {
		if (timer !== undefined) clearTimeout(timer);
	}
}
