import { createHash } from "node:crypto";
import { lstat, mkdir, readdir, readFile } from "node:fs/promises";
import { isAbsolute, join, relative, resolve } from "node:path";

import type { ImageContent, Model } from "@earendil-works/pi-ai";
import {
	type AgentSession,
	createAgentSession,
	DefaultResourceLoader,
	getAgentDir,
	ModelRuntime,
	SessionManager,
	SettingsManager,
	VERSION,
} from "@earendil-works/pi-coding-agent";

import { createAdoptShapeCandidateTool } from "./adopt-shape-candidate.js";
import { createCadTool, type CadToolState } from "./cad-tool.js";
import { createDimensionAnnotationService, dimensionsCurrent } from "./dimension-annotations.js";
import {
	createCodeConstructionTool,
	injectPersistentCodeConstructionRegistry,
	rebindCodeConstructionsAfterEdit,
	reconcileCodeConstructionBindings,
	resolveActiveCodeConstructionGroundings,
	restoreCodeConstructionState,
} from "./code-construction.js";
import {
	assertCausalRepairProfile,
	causalRepairSummary,
	createCausalRepairTool,
	recordCausalRepairFormalExecution,
	restoreCausalRepairState,
} from "./causal-repair.js";
import { historicalImageKeys, restrictToHistoricalImages, pruneContextImages, restrictMainViewReflectImages, sessionUsageTotals } from "./context.js";
import {
	createDrawingReflectionTool,
	mergeResultUsage,
	recordSubagentUsage,
} from "./drawing-reflection.js";
import { errorMessage, InfrastructureError } from "./errors.js";
import { elapsedSeconds, isoNow, readJson, writeJsonAtomic } from "./files.js";
import { GEOMETRY_REPAIR_COMPLETION_INSTRUCTION } from "./geometry-repair-instructions.js";
import type { WorkerClient } from "./local-worker.js";
import {
	createMainViewReflectTools,
	loadLockedMainViewManifest,
	mainViewReflectSummary,
	reconcileMainViewInspectionEvidence,
	restoreMainViewReflectState,
	summaryHasCroppedStep,
	summaryHasInspectedStep,
} from "./main-view-reflect.js";
import { hasCompletedOptionalReflectTurn, runOptionalReflectTurn } from "./optional-reflect-turn.js";
import { assertSelfReflectAuxiliaryState } from "./main-self-reflect.js";
import { continuationPrompt, initialPrompt, loadSystemPrompt } from "./prompts.js";
import { createRepairGeometryTool } from "./repair-geometry.js";
import { createShapeFeedbackTool } from "./shape-feedback.js";
import { artifactAccepted, ensureSolution, inspectFormalCandidate } from "./solution.js";
import {
	injectPersistentTargetAnnotations,
	restoreTargetAnnotationState,
} from "./target-annotations.js";
import {
	agentToolsForMode,
	CAUSAL_REPAIR_AGENT_TOOLS,
	CODE_CONSTRUCTION_PROFILE,
	DUAL_ROI_CROP_PROFILE,
	DIMENSION_ANNOTATION_PROFILE,
	GEOMETRY_REPAIR_PROFILE,
	mainSelfReflectInspectAgentTools,
	MAIN_VIEW_REFLECT_AGENT_TOOLS,
	PI_VERSION,
	PROTOCOL_VERSION,
	REPAIR_PREVIEW_PROFILE,
	SHAPE_FEEDBACK_PROFILE,
	TARGET_ANNOTATION_PROFILE,
	VISUAL_FINDING_AGENT_TOOLS,
	VISUAL_FINDING_PROFILE,
	VISUAL_DIFF_AGENT_TOOLS,
	VISUAL_DIFF_PROFILE,
	VISUAL_REVIEW_PROFILE,
	visualReviewAgentTools,
	type AgentJob,
	type AgentResult,
	type CandidateInspection,
	type WorkerFinalizeResponse,
	type WorkerInspectResponse,
	type WorkerPrepareResponse,
} from "./types.js";
import {
	createUpdateVisualFindingsTool,
	injectPersistentVisualFindings,
	restoreVisualFindingState,
} from "./visual-findings.js";
import {
	createVisualDiffTool,
	injectPersistentVisualDiffs,
	restoreVisualDiffState,
} from "./visual-diffs.js";
import {
	createRepairPreviewTool,
	createVisualReviewTool,
	injectPersistentVisualReviews,
	restoreVisualReviewState,
} from "./visual-review.js";

const THINKING_LEVELS = new Set(["off", "minimal", "low", "medium", "high"]);

export interface ResolvedModel {
	runtime: ModelRuntime;
	model: Model<any>;
}

export function assertImageCapableModel(model: Pick<Model<any>, "provider" | "id" | "input">): void {
	if (!model.input.includes("image")) {
		throw new InfrastructureError(`Model ${model.provider}/${model.id} does not declare image input support`);
	}
}

export function applyProviderCredentialCompatibility(
	provider: string,
	environment: NodeJS.ProcessEnv = process.env,
): void {
	if (
		provider === "moonshotai-cn" &&
		!environment.MOONSHOT_API_KEY?.trim() &&
		environment.KIMI_API_KEY?.trim()
	) {
		environment.MOONSHOT_API_KEY = environment.KIMI_API_KEY.trim();
	}
}

export async function resolveVisionModel(provider: string, modelId: string): Promise<ResolvedModel> {
	if (VERSION !== PI_VERSION) {
		throw new InfrastructureError(`Loaded Pi ${VERSION}, but this integration requires ${PI_VERSION}`);
	}
	applyProviderCredentialCompatibility(provider);
	const runtime = await ModelRuntime.create();
	const model = runtime.getModel(provider, modelId);
	if (!model) throw new InfrastructureError(`Unknown Pi model: ${provider}/${modelId}`);
	assertImageCapableModel(model);
	if (!runtime.hasConfiguredAuth(provider) || !(await runtime.getAuth(model))) {
		throw new InfrastructureError(
			`No Pi credential is configured for provider ${provider}. Use Pi's standard auth.json, OAuth, or provider environment variable.`,
		);
	}
	return { runtime, model };
}

function applySessionStats(result: AgentResult, session: AgentSession, workspace: string): void {
	mergeResultUsage(result, sessionUsageTotals(session));
	const sessionFile = session.getSessionStats().sessionFile;
	if (sessionFile) {
		const normalized = isAbsolute(sessionFile) ? relative(workspace, sessionFile) : sessionFile;
		if (
			normalized === ".." ||
			normalized.startsWith("../") ||
			!/^sessions\/[^/]+\.jsonl$/u.test(normalized)
		) {
			throw new InfrastructureError("Pi session file escaped the fixture workspace");
		}
		result.session_file = normalized;
	} else delete result.session_file;
}

async function safeSessionFile(workspace: string, sessionFile: string): Promise<string> {
	if (isAbsolute(sessionFile) || !/^sessions\/[^/]+\.jsonl$/u.test(sessionFile)) {
		throw new InfrastructureError("Recorded Pi session path must be sessions/<file>.jsonl inside the workspace");
	}
	const path = resolve(workspace, sessionFile);
	const metadata = await lstat(path).catch(() => null);
	if (metadata === null || !metadata.isFile() || metadata.isSymbolicLink()) {
		throw new InfrastructureError("Recorded Pi session file is missing, non-regular, or a symbolic link");
	}
	return path;
}

export async function driveUntilValid(options: {
	initialPromptPending: boolean;
	promptInitial: () => Promise<void>;
	promptContinue: (inspection: CandidateInspection | null) => Promise<void>;
	inspect: () => Promise<CandidateInspection>;
	onInspection: (inspection: CandidateInspection) => Promise<void>;
	isInterrupted: () => boolean;
	isAccepted?: (inspection: CandidateInspection) => boolean;
}): Promise<CandidateInspection> {
	const isAccepted = options.isAccepted ?? ((inspection) =>
		inspection.formal_execution_current && artifactAccepted(inspection.artifact));
	let initialPromptPending = options.initialPromptPending;
	let current: CandidateInspection | null = null;
	if (!initialPromptPending) {
		current = await options.inspect();
		await options.onInspection(current);
		if (options.isInterrupted()) throw new Error("interrupted");
		if (isAccepted(current)) return current;
	}
	while (true) {
		if (options.isInterrupted()) throw new Error("interrupted");
		if (initialPromptPending) {
			await options.promptInitial();
			initialPromptPending = false;
		} else await options.promptContinue(current);
		if (options.isInterrupted()) throw new Error("interrupted");
		current = await options.inspect();
		await options.onInspection(current);
		if (options.isInterrupted()) throw new Error("interrupted");
		if (isAccepted(current)) return current;
	}
}

/** Keep a completed CAD execution visible when only its dimension refresh fails. */
export async function dimensionRefreshFeedback(options: {
	refresh: () => Promise<unknown>;
	result: AgentResult;
	controller: Pick<RunController, "interrupted" | "signal">;
	persistResult: () => Promise<void>;
	phase?: "formal_execution" | "before_reflection";
}): Promise<string | undefined> {
	try {
		await options.refresh();
		return undefined;
	} catch (error) {
		if (options.controller.interrupted || options.controller.signal.aborted ||
			(error instanceof Error && error.name === "AbortError")) throw error;
		// The service normally saves failed state itself. Also invalidate an older
		// summary if the refresh failed before the service could start a new revision.
		if (options.result.dimension_annotations) {
			options.result.dimension_annotations = { ...options.result.dimension_annotations, status: "failed" };
		}
		await options.persistResult();
		const context = options.phase === "before_reflection"
			? "Initial dimension preparation failed before Main reflection."
			: "Formal CAD execution succeeded; its diagnostics and render are preserved. Dimension annotation refresh failed.";
		return `${context} ${errorMessage(error)}. Retry inspect_comparison({ region_id }) to refresh and inspect dimensions on the current formally executed CAD. Dimension verification and final completion remain pending; do not rely on previous dimension values, entity IDs, or images.`;
	}
}

export function continuationAccepted(job: AgentJob, result: AgentResult, inspection: CandidateInspection): boolean {
	const continuation = job.continuation;
	if (!inspection.formal_execution_current || !artifactAccepted(inspection.artifact)) return false;
	if (!continuation) return true;
	if (continuation.kind === "main_self_reflect_inspect") {
		return (
			result.applied_continuation?.id === continuation.id &&
			hasCompletedOptionalReflectTurn(job, result) &&
			summaryHasInspectedStep(result.main_view_reflection, inspection.artifact.sha256) &&
			(!continuation.dimension_annotation_profile || dimensionsCurrent(result.dimension_annotations, inspection.source_sha256, inspection.artifact.sha256, continuation.locked_view_manifest_sha256))
		);
	}
	if (continuation.kind === "main_view_reflect") {
		return (
			result.applied_continuation?.id === continuation.id &&
			summaryHasInspectedStep(
				result.main_view_reflection,
				inspection.artifact.sha256,
			) &&
			summaryHasCroppedStep(
				result.main_view_reflection,
				inspection.artifact.sha256,
			)
		);
	}
	if (continuation.no_new_images) {
		return hasCompletedOptionalReflectTurn(job, result) && result.applied_continuation?.id === continuation.id &&
			result.applied_continuation.prompt_sha256 === createHash("sha256").update(continuation.prompt).digest("hex");
	}
	return (
		result.applied_continuation?.id === continuation.id &&
		inspection.source_sha256 !== continuation.baseline_source_sha256 &&
		inspection.artifact.sha256 !== continuation.baseline_step_sha256
	);
}

export function unchangedContinuationPrompt(job: AgentJob): string {
	if (job.continuation?.kind === "main_self_reflect_inspect") {
		if (job.continuation.dimension_annotation_profile) return "Inspect the current formal CAD with inspect_comparison. The dimension SubAgent must finish its current-STEP dimension review, then you must review the aligned comparison and actual dimensions before completing. Following each edit, execute_build123d updates the STEP and automatically checks dimension annotations again. Resolve any failed dimension review by retrying inspection; never rely on old-STEP entity IDs or annotation images.";
		if (job.continuation.visual_review_profile === GEOMETRY_REPAIR_PROFILE) return GEOMETRY_REPAIR_COMPLETION_INSTRUCTION;
		return "The current formal CAD is valid and watertight, but this final STEP has not yet been inspected in an aligned 2D comparison. Use inspect_comparison({ region_id }) in the views relevant to your edits, or to the suspected discrepancy if you made no edits. Continue if these comparisons reveal a clear, actionable mismatch. Otherwise, finish with the current model. The final model may remain unchanged from the baseline.";
	}
	if (job.continuation?.kind === "main_view_reflect") {
		return "The current formal CAD is valid and watertight, but this exact STEP has not yet completed the required comparison evidence loop. Use inspect_comparison({ region_id }) for the complete Original/current-CAD comparison, then use dual_roi_crop for corresponding decisive local details. Base the decision on those comparison images rather than standalone 3D/isometric renders. If you find a reliable discrepancy, trace it to the responsible construction in solution.py, make focused edits, and call execute_build123d({}) when ready to formalize them. Repeat both comparison scales only if that execution changes the STEP; visual evidence remains current when the STEP is unchanged. If no reliable actionable discrepancy remains, preserve the Direct output and finish after the current formal STEP has both global and local comparison evidence.";
	}
	if (job.continuation?.kind === "main_self_reflect") {
		return "The formal CAD is valid, but this self-reflection continuation has not changed both solution.py and output.step from the preserved baseline. Re-examine the original drawings and the current formal render yourself, implement concrete improvements in the live solution.py, call execute_build123d({}), and keep working until the revised source and STEP are both changed, valid, and watertight. Do not stop at an analysis or report.";
	}
	return "The formal CAD is valid, but the Reflection-guided continuation has not changed both solution.py and output.step from the preserved baseline. Implement the supplied Reflection findings in the live solution.py, call execute_build123d({}), and keep working until the revised source and STEP are both changed, valid, and watertight.";
}

export function assertContinuation(job: AgentJob): void {
	const continuation = job.continuation;
	if (!continuation) return;
	if (
		continuation.kind !== undefined &&
		!["reflection_report", "main_self_reflect", "main_self_reflect_inspect", "main_view_reflect"].includes(continuation.kind)
	) {
		throw new InfrastructureError(`Unsupported continuation kind: ${String(continuation.kind)}`);
	}
	if (!/^[A-Za-z0-9._-]{1,128}$/u.test(continuation.id)) {
		throw new InfrastructureError("Continuation id must be a safe non-empty identifier");
	}
	if (continuation.no_new_images !== undefined && typeof continuation.no_new_images !== "boolean") throw new InfrastructureError("no_new_images must be boolean");
	if (continuation.no_new_images && (continuation.kind !== "main_self_reflect" || (continuation.image_paths?.length ?? 0) > 0)) {
		throw new InfrastructureError("no_new_images requires direct main_self_reflect with no attached images");
	}
	if (continuation.prompt.trim() === "") throw new InfrastructureError("Continuation prompt is empty");
	if (!/^[a-f0-9]{64}$/u.test(continuation.baseline_source_sha256)) {
		throw new InfrastructureError("Continuation baseline source hash is invalid");
	}
	if (!/^[a-f0-9]{64}$/u.test(continuation.baseline_step_sha256)) {
		throw new InfrastructureError("Continuation baseline STEP hash is invalid");
	}
	for (const path of continuation.image_paths ?? []) {
		if (isAbsolute(path) || path.split(/[\\/]/u).includes("..")) {
			throw new InfrastructureError("Continuation image path escaped the fixture workspace");
		}
	}
	if (continuation.kind === "main_view_reflect" || continuation.kind === "main_self_reflect_inspect") {
		if (job.agent_mode !== "base-only") {
			throw new InfrastructureError("Locked comparison continuation requires base-only agent mode");
		}
		if (!continuation.locked_view_manifest_path || !continuation.locked_view_manifest_sha256) {
			throw new InfrastructureError("Locked comparison continuation requires a View Search manifest and SHA-256");
		}
		if (
			isAbsolute(continuation.locked_view_manifest_path) ||
			continuation.locked_view_manifest_path.split(/[\\/]/u).includes("..")
		) {
			throw new InfrastructureError("Locked View Search manifest path escaped the fixture workspace");
		}
		if (!/^[a-f0-9]{64}$/u.test(continuation.locked_view_manifest_sha256)) {
			throw new InfrastructureError("Locked View Search manifest hash is invalid");
		}
	} else if (continuation.locked_view_manifest_path || continuation.locked_view_manifest_sha256) {
		throw new InfrastructureError("Locked View Search manifest fields require a locked-comparison continuation");
	}
	if (continuation.dimension_annotation_profile !== undefined) {
		if (continuation.dimension_annotation_profile !== DIMENSION_ANNOTATION_PROFILE || continuation.kind !== "main_self_reflect_inspect") throw new InfrastructureError("Dimension annotation profile requires Main Self Reflect plus Inspect");
		if ((continuation.visual_review_profile && continuation.visual_review_profile !== SHAPE_FEEDBACK_PROFILE) || continuation.causal_repair_profile || continuation.dual_roi_crop_profile || continuation.visual_finding_profile || continuation.visual_diff_profile || continuation.code_construction_profile || continuation.target_annotation_profile) throw new InfrastructureError("Dimension annotation supports Inspect alone or aligned shape feedback only");
	}
	if (continuation.causal_repair_profile !== undefined) {
		assertCausalRepairProfile(continuation.causal_repair_profile);
		if (continuation.kind !== "main_self_reflect_inspect") {
			throw new InfrastructureError("Causal repair verification requires main_self_reflect_inspect");
		}
	}
	if (continuation.dual_roi_crop_profile !== undefined) {
		if (continuation.dual_roi_crop_profile !== DUAL_ROI_CROP_PROFILE) {
			throw new InfrastructureError(
				`Unsupported Dual ROI Crop profile: ${String(continuation.dual_roi_crop_profile)}`,
			);
		}
		if (continuation.kind !== "main_self_reflect_inspect") {
			throw new InfrastructureError("Optional Dual ROI Crop requires main_self_reflect_inspect");
		}
	}
	if (continuation.visual_finding_profile !== undefined) {
		if (continuation.visual_finding_profile !== VISUAL_FINDING_PROFILE) {
			throw new InfrastructureError(
				`Unsupported visual finding profile: ${String(continuation.visual_finding_profile)}`,
			);
		}
		if (continuation.kind !== "main_self_reflect_inspect") {
			throw new InfrastructureError("Persistent visual findings require main_self_reflect_inspect");
		}
		if (continuation.causal_repair_profile !== undefined) {
			throw new InfrastructureError("Persistent visual findings cannot be combined with causal repair verification");
		}
	}
	if (continuation.visual_diff_profile !== undefined) {
		if (continuation.visual_diff_profile !== VISUAL_DIFF_PROFILE) {
			throw new InfrastructureError(
				`Unsupported visual-diff profile: ${String(continuation.visual_diff_profile)}`,
			);
		}
		if (continuation.kind !== "main_self_reflect_inspect") {
			throw new InfrastructureError("Persistent visual diffs require main_self_reflect_inspect");
		}
		if (continuation.code_construction_profile !== CODE_CONSTRUCTION_PROFILE) {
			throw new InfrastructureError("Persistent visual diffs require code-ablation localization");
		}
		if (continuation.visual_finding_profile !== undefined) {
			throw new InfrastructureError("Persistent visual diffs cannot be combined with natural-language visual findings");
		}
	}
	if (continuation.visual_review_profile !== undefined) {
		if (
			continuation.visual_review_profile !== VISUAL_REVIEW_PROFILE &&
			continuation.visual_review_profile !== REPAIR_PREVIEW_PROFILE &&
			continuation.visual_review_profile !== SHAPE_FEEDBACK_PROFILE &&
			continuation.visual_review_profile !== GEOMETRY_REPAIR_PROFILE
		) {
			throw new InfrastructureError(
				`Unsupported visual-review profile: ${String(continuation.visual_review_profile)}`,
			);
		}
		if (continuation.kind !== "main_self_reflect_inspect") {
			throw new InfrastructureError("Visual review requires main_self_reflect_inspect");
		}
		if (
			continuation.visual_finding_profile !== undefined ||
			continuation.visual_diff_profile !== undefined ||
			continuation.code_construction_profile !== undefined
		) {
			throw new InfrastructureError("Visual review replaces natural-language findings, pixel visual diffs, and code-ablation localization");
		}
		if (continuation.visual_review_profile === SHAPE_FEEDBACK_PROFILE && (
			continuation.causal_repair_profile !== undefined ||
			continuation.dual_roi_crop_profile !== undefined ||
			continuation.target_annotation_profile !== undefined
		)) {
			throw new InfrastructureError("Aligned shape feedback cannot be combined with causal repair, Dual ROI Crop, or target annotations");
		}
		if (continuation.visual_review_profile === GEOMETRY_REPAIR_PROFILE && (
			continuation.causal_repair_profile !== undefined ||
			continuation.dual_roi_crop_profile !== undefined ||
			continuation.target_annotation_profile !== undefined
		)) {
			throw new InfrastructureError("Atomic geometry repair cannot be combined with causal repair, Dual ROI Crop, or target annotations");
		}
	}
	if (continuation.code_construction_profile !== undefined) {
		if (continuation.code_construction_profile !== CODE_CONSTRUCTION_PROFILE) {
			throw new InfrastructureError(
				`Unsupported code-construction profile: ${String(continuation.code_construction_profile)}`,
			);
		}
		if (continuation.kind !== "main_self_reflect_inspect") {
			throw new InfrastructureError("Code-ablation localization requires main_self_reflect_inspect");
		}
		if (
			continuation.causal_repair_profile !== undefined ||
			continuation.dual_roi_crop_profile !== undefined
		) {
			throw new InfrastructureError("Code-ablation localization cannot be combined with causal repair or Dual ROI Crop");
		}
	}
	if (continuation.target_annotation_profile !== undefined) {
		if (continuation.target_annotation_profile !== TARGET_ANNOTATION_PROFILE) {
			throw new InfrastructureError(
				`Unsupported target annotation profile: ${String(continuation.target_annotation_profile)}`,
			);
		}
		if (continuation.kind !== "main_self_reflect_inspect") {
			throw new InfrastructureError("Persistent target annotations require main_self_reflect_inspect");
		}
	}
}

async function nextTurnNumber(workspace: string): Promise<number> {
	const entries = await readdir(join(workspace, "turns"), { withFileTypes: true }).catch(() => []);
	const numbers = entries
		.filter((entry) => entry.isDirectory() && /^\d+$/u.test(entry.name))
		.map((entry) => Number.parseInt(entry.name, 10));
	return numbers.length === 0 ? 1 : Math.max(...numbers) + 1;
}

async function prepareImages(
	worker: WorkerClient,
	workspace: string,
	paths: string[],
	signal?: AbortSignal,
): Promise<ImageContent[]> {
	const response = await worker.request<WorkerPrepareResponse>(workspace, {
		protocol_version: PROTOCOL_VERSION,
		operation: "prepare_images",
		paths,
		max_edge: 2000,
	}, signal ? { signal } : {});
	return await Promise.all(
		response.images.map(async (image) => ({
			type: "image" as const,
			data: (await readFile(join(workspace, image.path))).toString("base64"),
			mimeType: image.mime_type,
		})),
	);
}

export class RunController {
	interrupted = false;
	interruptReason: string | undefined;
	private readonly sessions = new Set<AgentSession>();
	private readonly abortController = new AbortController();

	get signal(): AbortSignal {
		return this.abortController.signal;
	}

	add(session: AgentSession): void {
		this.sessions.add(session);
	}

	delete(session: AgentSession): void {
		this.sessions.delete(session);
	}

	async interrupt(reason = "SIGINT"): Promise<void> {
		if (this.interrupted) return;
		this.interrupted = true;
		this.interruptReason = reason;
		this.abortController.abort(reason);
		await Promise.allSettled([...this.sessions].map(async (session) => await session.abort()));
	}
}

function newAgentResult(job: AgentJob): AgentResult {
	const now = isoNow();
	return {
		fixture_id: job.fixture.id,
		status: "running",
		agent_mode: job.agent_mode ?? "drawing-subagents",
		provider: job.provider,
		model: job.model,
		thinking_level: job.thinking_level,
		pi_version: job.pi_version,
		container_image: job.container_image,
		container_image_digest: job.container_image_digest,
		started_at: now,
		updated_at: now,
		duration_s: 0,
		last_turn: 0,
		artifact: null,
		tokens: { input: 0, output: 0, cache_read: 0, cache_write: 0, total: 0 },
		cost: 0,
		tool_calls: 0,
	};
}

async function saveResult(workspace: string, result: AgentResult): Promise<void> {
	result.updated_at = isoNow();
	result.duration_s = elapsedSeconds(result.started_at);
	await writeJsonAtomic(join(workspace, "agent_result.json"), result);
}

async function finalizeValid(
	worker: WorkerClient,
	workspace: string,
	result: AgentResult,
	inspection: CandidateInspection,
	controller: RunController,
): Promise<void> {
	const assertRunning = (): void => {
		if (controller.interrupted) throw new Error("interrupted");
	};
	assertRunning();
	if (!inspection.formal_execution_current) throw new InfrastructureError(inspection.reason);
	const beforeFinalize = await worker.request<WorkerInspectResponse>(workspace, {
		protocol_version: PROTOCOL_VERSION,
		operation: "inspect",
	}, { signal: controller.signal });
	assertRunning();
	const verifiedBeforeFinalize = await inspectFormalCandidate(workspace, beforeFinalize.artifact, result);
	if (!verifiedBeforeFinalize.formal_execution_current || !artifactAccepted(verifiedBeforeFinalize.artifact)) {
		throw new InfrastructureError(`Formal candidate changed before finalization: ${verifiedBeforeFinalize.reason}`);
	}
	const response = await worker.request<WorkerFinalizeResponse>(
		workspace,
		{ protocol_version: PROTOCOL_VERSION, operation: "finalize" },
		{ timeoutMs: 600_000, signal: controller.signal },
	);
	assertRunning();
	const afterFinalize = await worker.request<WorkerInspectResponse>(workspace, {
		protocol_version: PROTOCOL_VERSION,
		operation: "inspect",
	}, { signal: controller.signal });
	assertRunning();
	const verifiedAfterFinalize = await inspectFormalCandidate(workspace, afterFinalize.artifact, result);
	if (
		!response.finalized ||
		!verifiedAfterFinalize.formal_execution_current ||
		!artifactAccepted(verifiedAfterFinalize.artifact)
	) {
		throw new InfrastructureError("Final validity gate changed or formal candidate evaluation failed");
	}
	result.artifact = afterFinalize.artifact;
	result.status = "valid";
	result.stop_reason =
		"formal source/STEP hashes match; is_valid=true and is_watertight=true; candidate-only evaluation completed";
	await saveResult(workspace, result);
}

export async function runFixtureInContainer(options: {
	job: AgentJob;
	workspace: string;
	repositoryRoot: string;
	integrationRoot: string;
	worker: WorkerClient;
	resolvedModel: ResolvedModel;
	controller: RunController;
}): Promise<void> {
	const { job, workspace, worker, controller } = options;
	assertContinuation(job);
	if (!THINKING_LEVELS.has(job.thinking_level)) {
		throw new InfrastructureError(`Invalid thinking level: ${job.thinking_level}`);
	}
	await mkdir(join(workspace, "sessions"), { recursive: true });
	await ensureSolution(workspace);
	let result =
		(await readJson<AgentResult>(join(workspace, "agent_result.json")).catch(() => null)) ?? newAgentResult(job);
	const isMainViewReflect = job.continuation?.kind === "main_view_reflect";
	const isMainSelfReflectInspect = job.continuation?.kind === "main_self_reflect_inspect";
	const noNewImages = job.continuation?.no_new_images === true;
	if (isMainSelfReflectInspect) assertSelfReflectAuxiliaryState(result, job.continuation?.visual_review_profile, job.continuation?.dimension_annotation_profile);
	const isTrackedReflect = isMainSelfReflectInspect || noNewImages;
	const optionalReflectStarted = isTrackedReflect &&
		result.optional_reflect_turn?.continuation_id === job.continuation!.id &&
		result.optional_reflect_turn.prompt_sha256 === createHash("sha256").update(job.continuation!.prompt).digest("hex");
	const restoredSessionManager = isTrackedReflect && result.session_file
		? SessionManager.open(await safeSessionFile(workspace, result.session_file), join(workspace, "sessions"), workspace)
		: undefined;
	const usesLockedViewManifest = isMainViewReflect || isMainSelfReflectInspect;
	const lockedViewManifest = usesLockedViewManifest
		? await loadLockedMainViewManifest({
				workspace,
				manifestPath: job.continuation!.locked_view_manifest_path!,
				manifestSha256: job.continuation!.locked_view_manifest_sha256!,
				baselineSourceSha256: job.continuation!.baseline_source_sha256,
				baselineStepSha256: job.continuation!.baseline_step_sha256,
			})
		: undefined;
	const mainViewReflectState = lockedViewManifest
		? await restoreMainViewReflectState({
				workspace,
				fixtureId: job.fixture.id,
				lockedManifest: lockedViewManifest,
				lockedManifestPath: job.continuation!.locked_view_manifest_path!,
				lockedManifestSha256: job.continuation!.locked_view_manifest_sha256!,
				baselineSourceSha256: job.continuation!.baseline_source_sha256,
				baselineStepSha256: job.continuation!.baseline_step_sha256,
				// A new Optional Reflect task must inspect its final CAD even if
				// it shares the same baseline and views with a previous task.
				...(isMainSelfReflectInspect ? {
					auditPath: join("drawing_feedback", "main-self-reflect", `${job.continuation!.id}.json`),
				} : {}),
			})
		: undefined;
	if (mainViewReflectState) {
		if (isMainSelfReflectInspect) {
			await reconcileMainViewInspectionEvidence(workspace, mainViewReflectState, restoredSessionManager?.getEntries() ?? []);
		}
		result.main_view_reflection = mainViewReflectSummary(mainViewReflectState);
	}
	const causalRepairState = job.continuation?.causal_repair_profile && lockedViewManifest
		? await restoreCausalRepairState({
				workspace,
				fixtureId: job.fixture.id,
				profile: job.continuation.causal_repair_profile,
				lockedViewManifestSha256: job.continuation.locked_view_manifest_sha256!,
			})
		: undefined;
	if (causalRepairState) result.causal_repair = causalRepairSummary(causalRepairState);
	const visualFindingState = job.continuation?.visual_finding_profile && lockedViewManifest
		? await restoreVisualFindingState({
				workspace,
				fixtureId: job.fixture.id,
				profile: job.continuation.visual_finding_profile,
				lockedViewManifestSha256: job.continuation.locked_view_manifest_sha256!,
			})
		: undefined;
	const codeConstructionState = job.continuation?.code_construction_profile && lockedViewManifest
		? await restoreCodeConstructionState({
				workspace,
				fixtureId: job.fixture.id,
				profile: job.continuation.code_construction_profile,
				lockedViewManifestSha256: job.continuation.locked_view_manifest_sha256!,
			})
		: undefined;
	const visualDiffState = job.continuation?.visual_diff_profile && lockedViewManifest && codeConstructionState
		? await restoreVisualDiffState({
				workspace,
				fixtureId: job.fixture.id,
				profile: job.continuation.visual_diff_profile,
				lockedViewManifestSha256: job.continuation.locked_view_manifest_sha256!,
			})
		: undefined;
	const visualReviewState = job.continuation?.visual_review_profile === VISUAL_REVIEW_PROFILE && lockedViewManifest
		? await restoreVisualReviewState({
				workspace,
				fixtureId: job.fixture.id,
				profile: job.continuation.visual_review_profile,
				lockedViewManifestSha256: job.continuation.locked_view_manifest_sha256!,
			})
		: undefined;
	const repairPreviewEnabled =
		job.continuation?.visual_review_profile === REPAIR_PREVIEW_PROFILE && lockedViewManifest !== undefined;
	const shapeFeedbackEnabled =
		job.continuation?.visual_review_profile === SHAPE_FEEDBACK_PROFILE && lockedViewManifest !== undefined;
	const geometryRepairEnabled =
		job.continuation?.visual_review_profile === GEOMETRY_REPAIR_PROFILE && lockedViewManifest !== undefined;
	const targetAnnotationState = job.continuation?.target_annotation_profile && lockedViewManifest
		? await restoreTargetAnnotationState({
				workspace,
				fixtureId: job.fixture.id,
				profile: job.continuation.target_annotation_profile,
				lockedViewManifestSha256: job.continuation.locked_view_manifest_sha256!,
			})
		: undefined;
	const reconcileCodeConstructionContext = async (): Promise<void> => {
		if (!codeConstructionState) return;
		await reconcileCodeConstructionBindings({
			workspace,
			state: codeConstructionState,
		});
	};
	const interruptedMainViewReflectPrompt =
		isMainViewReflect &&
		job.continuation !== undefined &&
		result.applied_continuation?.id !== job.continuation.id &&
		(result.main_view_reflection?.inspection_count ?? 0) > 0;
	if (interruptedMainViewReflectPrompt && job.continuation) {
		result.applied_continuation = {
			id: job.continuation.id,
			prompt_sha256: createHash("sha256").update(job.continuation.prompt).digest("hex"),
			applied_at: isoNow(),
		};
	}
	const continuationPending =
		job.continuation !== undefined && !optionalReflectStarted && result.applied_continuation?.id !== job.continuation.id;
	if (continuationPending && !result.session_file) {
		throw new InfrastructureError("A continuation requires an existing Pi session");
	}
	if (usesLockedViewManifest && continuationPending) {
		const baselineResponse = await worker.request<WorkerInspectResponse>(workspace, {
			protocol_version: PROTOCOL_VERSION,
			operation: "inspect",
		}, { signal: controller.signal });
		const baselineInspection = await inspectFormalCandidate(workspace, baselineResponse.artifact, result);
		if (
			!baselineInspection.formal_execution_current ||
			!artifactAccepted(baselineInspection.artifact) ||
			baselineInspection.source_sha256 !== job.continuation!.baseline_source_sha256 ||
			baselineInspection.artifact.sha256 !== job.continuation!.baseline_step_sha256
		) {
			throw new InfrastructureError(
				"Locked comparison workspace no longer matches the hash-locked Direct source/STEP baseline",
			);
		}
	}
	const agentMode = job.agent_mode ?? "drawing-subagents";
	if (result.agent_mode !== undefined && result.agent_mode !== agentMode) {
		throw new InfrastructureError(
			`Agent mode changed across resume: prior=${result.agent_mode}, requested=${agentMode}`,
		);
	}
	result.agent_mode = agentMode;
	await reconcileCodeConstructionContext();
	if (result.status === "valid") {
		const latest = await worker.request<WorkerInspectResponse>(workspace, {
			protocol_version: PROTOCOL_VERSION,
			operation: "inspect",
		}, { signal: controller.signal });
		const inspection = await inspectFormalCandidate(workspace, latest.artifact, result);
		if (continuationAccepted(job, result, inspection)) return;
	}
	result.status = "running";
	result.resume_reason = continuationPending
		? `resumed ${result.session_file} for continuation ${job.continuation?.id}`
		: result.session_file
			? `resumed ${result.session_file}`
			: "new session";
	delete result.stop_reason;
	await saveResult(workspace, result);

	let session: AgentSession | undefined;
	try {
		if (controller.interrupted) throw new Error("interrupted");
		const images = noNewImages ? [] : await prepareImages(worker, workspace, job.staged_paths, controller.signal);
		const continuationImagePaths = job.continuation?.image_paths ?? [];
		const continuationImages = continuationPending && continuationImagePaths.length > 0
			? await prepareImages(worker, workspace, continuationImagePaths, controller.signal)
			: [];
		const isNewSession = !result.session_file;
		const sessionsDir = join(workspace, "sessions");
		const sessionManager = restoredSessionManager ?? (result.session_file
			? SessionManager.open(await safeSessionFile(workspace, result.session_file), sessionsDir, workspace)
			: SessionManager.create(workspace, sessionsDir));
		const initialImageKeys = noNewImages
			? historicalImageKeys((await readFile(join(workspace, "baseline/session.jsonl"), "utf8")).split("\n").filter(Boolean)
				.map(line => JSON.parse(line) as { message?: { content?: unknown } }).flatMap(entry => entry.message ? [entry.message] : []))
			: new Set<string>();
		const settingsManager = SettingsManager.inMemory({}, { projectTrusted: false });
		const systemPrompt = await loadSystemPrompt(
			options.repositoryRoot,
			options.integrationRoot,
			geometryRepairEnabled ? "geometry-repair" : isMainViewReflect ? "main-view-reflect" : agentMode,
		);
		const resourceLoader = new DefaultResourceLoader({
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
			extensionFactories: [
				(pi) => {
					pi.on("tool_result", async (event) => {
						if (noNewImages && event.content.some(block => block.type === "image")) {
							return { content: event.content.filter(block => block.type !== "image") };
						}
						if (!codeConstructionState || event.toolName !== "edit" || event.isError) return;
						await rebindCodeConstructionsAfterEdit({
							workspace,
							state: codeConstructionState,
							path: event.input.path,
							edits: event.input.edits,
						});
					});
					pi.on("context", async (event) => {
						await reconcileCodeConstructionContext();
						let messages = noNewImages
							? restrictToHistoricalImages(event.messages, initialImageKeys)
							: isMainViewReflect
							? restrictMainViewReflectImages(event.messages)
							: pruneContextImages(event.messages);
						if (codeConstructionState && !visualReviewState) {
							messages = injectPersistentCodeConstructionRegistry(messages, codeConstructionState);
						}
						if (visualFindingState) {
							const activeConstructionRefs = new Set(
								codeConstructionState?.activeConstructionRefs ?? [],
							);
							const resolvableConstructionRefs = new Set(
								codeConstructionState?.constructions
									.filter(({ construction_ref, binding_status }) => (
										activeConstructionRefs.has(construction_ref) && binding_status !== "unresolved"
									))
									.map(({ construction_ref }) => construction_ref) ?? [],
							);
							const constructionGroundings = codeConstructionState
								? await resolveActiveCodeConstructionGroundings({
									workspace,
									state: codeConstructionState,
									constructionRefs: visualFindingState.findings.flatMap((finding) => (
										finding.current_construction_refs.filter((reference) => (
											resolvableConstructionRefs.has(reference)
										))
									)),
								})
								: undefined;
							messages = injectPersistentVisualFindings(
								messages,
								visualFindingState,
								constructionGroundings,
							);
						}
						if (visualDiffState && codeConstructionState) {
							messages = await injectPersistentVisualDiffs({
								messages,
								workspace,
								state: visualDiffState,
								codeState: codeConstructionState,
								result,
							});
						}
						if (visualReviewState) {
							messages = await injectPersistentVisualReviews({
								messages,
								workspace,
								state: visualReviewState,
								result,
							});
						}
						if (targetAnnotationState) {
							messages = injectPersistentTargetAnnotations(messages, targetAnnotationState);
						}
						return {
							messages,
						};
					});
				},
			],
		});
		await resourceLoader.reload();
		const toolState: CadToolState = { nextTurn: await nextTurnNumber(workspace), lastResponse: null };
		const dimensionService = job.continuation?.dimension_annotation_profile && lockedViewManifest
			? createDimensionAnnotationService({
				workspace, job, lockedManifest: lockedViewManifest, worker, controller,
				resolvedModel: options.resolvedModel, getResult: () => result,
				onUpdate: async (summary) => { result.dimension_annotations = summary; await saveResult(workspace, result); },
				onUsage: async (usage) => {
					recordSubagentUsage(result, usage);
					if (session) applySessionStats(result, session, workspace);
					await saveResult(workspace, result);
				},
			}) : undefined;
		const cadTool = createCadTool({
			worker,
			workspace,
			state: toolState,
			includeRender: !isMainViewReflect && !noNewImages,
			noVisualFeedback: noNewImages,
			onResult: async (response) => {
				result.last_turn = response.turn;
				result.artifact = response.artifact;
				result.last_executed_source_sha256 = response.source_sha256;
				if (response.artifact.sha256) result.last_executed_step_sha256 = response.artifact.sha256;
				else delete result.last_executed_step_sha256;
				await reconcileCodeConstructionContext();
				if (causalRepairState) {
					await recordCausalRepairFormalExecution({
						workspace,
						state: causalRepairState,
						turn: response.turn,
						sourceSha256: response.source_sha256,
						stepSha256: response.artifact.sha256,
					});
					result.causal_repair = causalRepairSummary(causalRepairState);
				}
				if (session) applySessionStats(result, session, workspace);
				await saveResult(workspace, result);
				if (dimensionService && response.execution.success && artifactAccepted(response.artifact)) {
					return dimensionRefreshFeedback({
						refresh: () => dimensionService.ensureCurrent(`execute_build123d turn ${response.turn}`),
						result, controller, persistResult: () => saveResult(workspace, result),
					});
				}
				return undefined;
			},
		});
		const drawingReflectionTool =
			agentMode === "drawing-subagents"
				? createDrawingReflectionTool({
						job,
						worker,
						workspace,
						resolvedModel: options.resolvedModel,
						controller,
						getResult: () => result,
						onUsage: async (usage) => {
							recordSubagentUsage(result, usage);
							if (session) applySessionStats(result, session, workspace);
							await saveResult(workspace, result);
						},
						onComplete: async (summary) => {
							result.drawing_feedback = summary;
							if (session) applySessionStats(result, session, workspace);
							await saveResult(workspace, result);
						},
					})
				: undefined;
		const allMainViewReflectTools = mainViewReflectState
			? createMainViewReflectTools({
					workspace,
					state: mainViewReflectState,
					worker,
					controller,
					getResult: () => result,
					onUpdate: async (summary) => {
						result.main_view_reflection = summary;
						if (session) applySessionStats(result, session, workspace);
						await saveResult(workspace, result);
					},
					...(targetAnnotationState ? { targetAnnotationState } : {}),
					...(dimensionService ? { dimensionComparison: (regionId: string) => dimensionService.comparison(regionId) } : {}),
				})
			: [];
		const mainSelfReflectInspectToolNames = mainSelfReflectInspectAgentTools(
			job.continuation?.dual_roi_crop_profile,
			job.continuation?.visual_review_profile ? undefined : job.continuation?.code_construction_profile,
			job.continuation?.target_annotation_profile,
		);
		const mainViewReflectTools = isMainSelfReflectInspect
			? allMainViewReflectTools.filter(({ name }) => mainSelfReflectInspectToolNames.includes(name))
			: allMainViewReflectTools;
		const causalRepairTool = causalRepairState && lockedViewManifest
			? createCausalRepairTool({
					workspace,
					state: causalRepairState,
					lockedManifest: lockedViewManifest,
					worker,
					controller,
					getResult: () => result,
					onUpdate: async (summary) => {
						result.causal_repair = summary;
						if (session) applySessionStats(result, session, workspace);
						await saveResult(workspace, result);
					},
				})
			: undefined;
		const visualFindingsTool = visualFindingState
			? createUpdateVisualFindingsTool({
					workspace,
					state: visualFindingState,
					getActiveConstructionRefs: () => codeConstructionState?.activeConstructionRefs ?? [],
				})
			: undefined;
		const codeConstructionTool = codeConstructionState && lockedViewManifest && !visualReviewState
			? createCodeConstructionTool({
					workspace,
					state: codeConstructionState,
					lockedManifest: lockedViewManifest,
					worker,
					controller,
					getResult: () => result,
				})
			: undefined;
		const visualDiffTool = visualDiffState && codeConstructionState
			? createVisualDiffTool({
					workspace,
					state: visualDiffState,
					codeState: codeConstructionState,
					worker,
					controller,
					getResult: () => result,
				})
			: undefined;
		const visualReviewTool = visualReviewState && lockedViewManifest
			? createVisualReviewTool({
					workspace,
					state: visualReviewState,
					lockedManifest: lockedViewManifest,
					worker,
					controller,
					getResult: () => result,
				})
			: undefined;
		const repairPreviewTool = repairPreviewEnabled && lockedViewManifest
			? createRepairPreviewTool({
					workspace,
					lockedManifest: lockedViewManifest,
					worker,
					controller,
					getResult: () => result,
				})
			: undefined;
		const shapeFeedbackTool = shapeFeedbackEnabled && lockedViewManifest
			? createShapeFeedbackTool({
					workspace,
					lockedManifest: lockedViewManifest,
					allowSections: lockedViewManifest.projection_frame?.policy === "fixed",
					worker,
					controller,
					getResult: () => result,
				})
			: undefined;
		const repairGeometryTool = geometryRepairEnabled && lockedViewManifest
			? createRepairGeometryTool({
					workspace,
					fixture: job.fixture,
					stagedPaths: job.staged_paths,
					worker,
					controller,
					lockedManifest: lockedViewManifest,
					resolvedModel: options.resolvedModel,
					thinkingLevel: job.thinking_level,
					getResult: () => result,
					onUsage: async (usage) => {
						recordSubagentUsage(result, usage);
						if (session) applySessionStats(result, session, workspace);
						await saveResult(workspace, result);
					},
				})
			: undefined;
		const adoptShapeCandidateTool = (shapeFeedbackEnabled || geometryRepairEnabled) && lockedViewManifest
			? createAdoptShapeCandidateTool({ workspace, lockedManifest: lockedViewManifest, controller })
			: undefined;
		const customTools = drawingReflectionTool
			? [cadTool, drawingReflectionTool]
			: [
					cadTool,
					...mainViewReflectTools,
					...(causalRepairTool ? [causalRepairTool] : []),
					...(visualFindingsTool ? [visualFindingsTool] : []),
					...(codeConstructionTool ? [codeConstructionTool] : []),
					...(visualDiffTool ? [visualDiffTool] : []),
					...(visualReviewTool ? [visualReviewTool] : []),
					...(repairPreviewTool ? [repairPreviewTool] : []),
					...(shapeFeedbackTool ? [shapeFeedbackTool] : []),
					...(repairGeometryTool ? [repairGeometryTool] : []),
					...(adoptShapeCandidateTool ? [adoptShapeCandidateTool] : []),
				];
		const enabledTools = isMainViewReflect
			? [...agentToolsForMode(agentMode), ...MAIN_VIEW_REFLECT_AGENT_TOOLS]
			: isMainSelfReflectInspect
				? [
						...agentToolsForMode(agentMode),
						...mainSelfReflectInspectToolNames,
						...(causalRepairState ? CAUSAL_REPAIR_AGENT_TOOLS : []),
						...(visualFindingState ? VISUAL_FINDING_AGENT_TOOLS : []),
						...(visualDiffState ? VISUAL_DIFF_AGENT_TOOLS : []),
						...visualReviewAgentTools(job.continuation?.visual_review_profile),
					]
				: [...agentToolsForMode(agentMode)];
		const created = await createAgentSession({
			cwd: workspace,
			agentDir: getAgentDir(),
			modelRuntime: options.resolvedModel.runtime,
			model: options.resolvedModel.model,
			thinkingLevel: job.thinking_level as "off" | "minimal" | "low" | "medium" | "high",
			tools: enabledTools,
			customTools,
			resourceLoader,
			sessionManager,
			settingsManager,
		});
		session = created.session;
		controller.add(session);
		applySessionStats(result, session, workspace);
		await saveResult(workspace, result);
		// Prepare dimensions before Main can reason about or edit the aligned CAD.
		// Resume uses the service's verified cache; an explicit failure is delivered
		// in Main's first prompt so Inspect can retry without losing the CAD state.
		let initialDimensionFeedback = dimensionService
			? await dimensionRefreshFeedback({
				refresh: () => dimensionService.ensureCurrent("before Main reflection"),
				result, controller, persistResult: () => saveResult(workspace, result), phase: "before_reflection",
			}) : undefined;
		const promptOptionalReflect = async (prompt: string, promptImages?: ImageContent[]): Promise<void> => {
			const effectivePrompt = initialDimensionFeedback ? `${prompt}\n\n${initialDimensionFeedback}` : prompt;
			initialDimensionFeedback = undefined;
			await runOptionalReflectTurn({
				job, result, session: created.session, prompt: effectivePrompt,
				...(promptImages ? { images: promptImages } : {}),
				signal: controller.signal,
				persist: async () => { await saveResult(workspace, result); },
				onCompleted: async () => {
					if (mainViewReflectState) {
						await reconcileMainViewInspectionEvidence(workspace, mainViewReflectState, sessionManager.getEntries());
						result.main_view_reflection = mainViewReflectSummary(mainViewReflectState);
					}
				},
			});
		};

		const inspect = async (): Promise<CandidateInspection> => {
				const latest = await worker.request<WorkerInspectResponse>(workspace, {
					protocol_version: PROTOCOL_VERSION,
					operation: "inspect",
				}, { signal: controller.signal });
			return await inspectFormalCandidate(workspace, latest.artifact, result);
		};
		const accepted = await driveUntilValid({
			initialPromptPending: isNewSession || continuationPending,
			promptInitial: async () => {
				if (continuationPending && job.continuation) {
					if (isTrackedReflect) {
						await promptOptionalReflect(job.continuation.prompt, [...images, ...continuationImages]);
					} else {
						await session?.prompt(job.continuation.prompt, {
							images: [...images, ...continuationImages],
							expandPromptTemplates: false,
						});
					}
					result.applied_continuation = {
						id: job.continuation.id,
						prompt_sha256: createHash("sha256").update(job.continuation.prompt).digest("hex"),
						applied_at: isoNow(),
					};
					if (session) applySessionStats(result, session, workspace);
					await saveResult(workspace, result);
					return;
				}
				await session?.prompt(initialPrompt(job.fixture), { images, expandPromptTemplates: false });
			},
			promptContinue: async (inspection) => {
				if (isTrackedReflect && !hasCompletedOptionalReflectTurn(job, result)) {
					await promptOptionalReflect(`${job.continuation!.prompt}\n\nResume from the current workspace and the available comparison evidence. The previous reflection turn did not finish. Complete your review and decide whether to make further edits or finish with the current model.`, images);
					return;
				}
				const pendingContinuationCheck =
					job.continuation !== undefined &&
					inspection?.formal_execution_current === true &&
					artifactAccepted(inspection.artifact) &&
					(job.continuation.kind === "main_view_reflect" ||
						job.continuation.kind === "main_self_reflect_inspect" ||
						inspection.source_sha256 === job.continuation.baseline_source_sha256 ||
						inspection.artifact.sha256 === job.continuation.baseline_step_sha256);
				const prompt = pendingContinuationCheck
					? unchangedContinuationPrompt(job)
					: continuationPrompt(inspection);
				if (isTrackedReflect) await promptOptionalReflect(prompt);
				else await session?.prompt(prompt, { expandPromptTemplates: false });
			},
			inspect,
			onInspection: async (inspection) => {
				if (session) applySessionStats(result, session, workspace);
				result.artifact = inspection.artifact;
				result.last_turn = toolState.nextTurn - 1;
				await saveResult(workspace, result);
			},
			isInterrupted: () => controller.interrupted,
			isAccepted: (inspection) => continuationAccepted(job, result, inspection),
		});
		if (controller.interrupted) throw new Error("interrupted");
		await finalizeValid(worker, workspace, result, accepted, controller);
	} catch (error) {
		if (session) applySessionStats(result, session, workspace);
		if (controller.interrupted) {
			result.status = "interrupted";
			result.stop_reason = controller.interruptReason ?? "interrupted";
		} else {
			result.status = "infrastructure_error";
			result.stop_reason = errorMessage(error);
		}
		await saveResult(workspace, result);
		if (!controller.interrupted) {
			throw error instanceof InfrastructureError
				? error
				: new InfrastructureError(`Fixture ${job.fixture.id} failed: ${errorMessage(error)}`, { cause: error });
		}
	} finally {
		if (session) {
			controller.delete(session);
			session.dispose();
		}
	}
}
