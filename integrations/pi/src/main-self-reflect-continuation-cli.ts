#!/usr/bin/env node
import { createHash } from "node:crypto";
import { cp, lstat, mkdir, readFile, readdir, realpath, stat } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { DockerAgentRuntime } from "./docker-worker.js";
import { CODE_CONSTRUCTION_AUDIT_PATH } from "./code-construction.js";
import { errorMessage, InfrastructureError, UsageError } from "./errors.js";
import {
	type ExperimentModelSelection,
	resolveExperimentModelSelection,
} from "./experiment-model-profile.js";
import { elapsedSeconds, isoNow, readJson, writeJsonAtomic } from "./files.js";
import {
	assertSelfReflectAuxiliaryState,
	auxiliaryStateFields,
	CaseTimeoutError,
	classifySelfReflectSource,
	MAIN_SELF_REFLECT_INSPECT_KIND,
	MAIN_SELF_REFLECT_KIND,
	mainSelfReflectInspectPrompt,
	mainSelfReflectPrompt,
	mainSelfReflectNoNewImagesPrompt,
	withIsolatedTimeout,
} from "./main-self-reflect.js";
import { filterLockedMainViewManifest, loadLockedMainViewManifest, summaryHasInspectedStep } from "./main-view-reflect.js";
import { artifactAccepted } from "./solution.js";
import { TARGET_ANNOTATION_STATE_PATH } from "./target-annotations.js";
import { VISUAL_FINDING_STATE_PATH } from "./visual-findings.js";
import { VISUAL_DIFF_STATE_PATH } from "./visual-diffs.js";
import { VISUAL_REVIEW_STATE_PATH } from "./visual-review.js";
import {
	CAUSAL_REPAIR_PROFILE,
	CODE_CONSTRUCTION_PROFILE,
	DIMENSION_ANNOTATION_PROFILE,
	DUAL_ROI_CROP_PROFILE,
	GEOMETRY_REPAIR_PROFILE,
	JOB_FILE,
	PI_VERSION,
	PROTOCOL_VERSION,
	REPAIR_PREVIEW_PROFILE,
	SHAPE_FEEDBACK_PROFILE,
	SHAPE_FEEDBACK_TOOLS,
	TARGET_ANNOTATION_PROFILE,
	VISUAL_FINDING_PROFILE,
	VISUAL_DIFF_PROFILE,
	VISUAL_REVIEW_PROFILE,
	visualReviewAgentTools,
	type AgentJob,
	type AgentResult,
	type CausalRepairProfile,
	type CodeConstructionProfile,
	type DimensionAnnotationProfile,
	type DualRoiCropProfile,
	type DrawingViewManifest,
	type RunStatus,
	type TargetAnnotationProfile,
	type VisualFindingProfile,
	type VisualDiffProfile,
	type VisualReviewProfile,
} from "./types.js";

interface ParsedArguments {
	positionals: string[];
	flags: Map<string, string>;
}

interface PreparedCase {
	fixtureId: string;
	workspace: string;
	sourceWorkspace: string;
	continuationId: string;
	promptSha256: string;
	baselineSourceSha256: string;
	baselineStepSha256: string;
	baselineTreeSha256: string;
	viewWorkspace?: string;
	viewTreeSha256?: string;
	inspectRegionId?: string;
	manifestPath?: string;
	manifestSha256?: string;
	lockedAssets: LockedAsset[];
	baselineLastTurn: number;
	baselineUsage: Pick<AgentResult, "tokens" | "cost" | "tool_calls">;
	job: AgentJob;
	runtime: DockerAgentRuntime;
	sourceContinuationKind: "original" | "main_self_reflect";
}

interface ViewSearchOnlyResult {
	status?: string;
	fixture_id?: string;
	source_workspace?: string;
	reflection_agent_started?: boolean;
	agent_roles?: string[];
	manifest_path?: string;
	source_sha256?: string;
	step_sha256?: string;
}

interface LockedAsset {
	path: string;
	sha256: string;
}

interface SelfReflectAfter {
	source_sha256: string;
	step_sha256: string;
	render_path: string;
	session_file: string;
	last_turn: number;
	formal_executions_added: number;
	is_valid: boolean;
	is_watertight: boolean;
	inspection_count: number;
	audit_manifest_path?: string;
	causal_repair_evaluation_count?: number;
	causal_repair_audit_manifest_path?: string;
	visual_findings_state_path?: string;
	visual_diffs_state_path?: string;
	geometry_hypotheses_state_path?: string;
	target_annotations_state_path?: string;
	dimension_annotations_state_path?: string;
	code_construction_inspection_count?: number;
	code_construction_reference_count?: number;
	code_construction_active_reference_count?: number;
	code_construction_audit_path?: string;
}

type SelfReflectCaseStatus = "completed" | "timeout" | "error" | "interrupted";

interface SelfReflectCaseResult {
	fixture_id: string;
	status: SelfReflectCaseStatus;
	started_at: string;
	finished_at: string;
	duration_s: number;
	timeout_minutes: number;
	source_workspace: string;
	workspace: string;
	continuation_id: string;
	prompt_sha256: string;
	baseline: {
		source_sha256: string;
		step_sha256: string;
		workspace_tree_sha256_before: string;
		workspace_tree_sha256_after: string;
		render_path: string;
		session_file: string;
		last_turn: number;
		view_search_workspace?: string;
		view_search_tree_sha256_before?: string;
		view_search_tree_sha256_after?: string;
		locked_view_manifest_path?: string;
		locked_view_manifest_sha256?: string;
		view_origin_source_sha256?: string;
		view_origin_step_sha256?: string;
		previous_continuation_id?: string;
	};
	after?: SelfReflectAfter;
	agent_status?: RunStatus;
	usage_delta?: {
		tokens: number;
		cost: number;
		tool_calls: number;
	};
	no_auxiliary_agent_state: boolean;
	no_reflection_agent_state: boolean;
	allowed_auxiliary_agent_state?: boolean;
	error?: string;
}

const DEFAULT_TIMEOUT_MINUTES = 120;
const MAX_TIMEOUT_MINUTES = 1440;
const THINKING_LEVELS = ["off", "minimal", "low", "medium", "high"] as const;
type ThinkingLevel = typeof THINKING_LEVELS[number];
const DEFAULT_THINKING: ThinkingLevel = "high";

const HELP = `cadgenbench-pi-main-self-reflect-continue - resume original base-only Pi sessions

Usage:
  cadgenbench-pi-main-self-reflect-continue <source-workspace>... \\
    --output-dir <new-run-root> \\
    --image <container-tag> \\
    [--timeout-minutes <1..1440; default 120>] \\
    [--thinking-level <off|minimal|low|medium|high>] \\
    [--inspect-view-search-root <completed-view-search-root>] \\
	[--inspect-region-id <locked-region-id>] \\
    [--causal-repair-profile counterfactual-patch-v1] \\
    [--dual-roi-crop-profile optional-dual-roi-v1] \\
    [--target-annotation-profile persistent-target-annotations-v1] \\
    [--visual-finding-profile persistent-visual-evidence-v2] \\
    [--visual-diff-profile target-current-visual-diff-v1] \\
    [--visual-review-profile <executable-geometry-hypothesis-v1|multi-candidate-brep-preview-v1|aligned-shape-feedback-v1|atomic-geometry-repair-v1>] \\
    [--dimension-annotation-profile aligned-dimension-subagent-v1] \\
    [--code-construction-profile optional-code-ablation-v1] \\
    [--no-new-images] \\
    [--allow-chained-self-reflect] \\
    [--reuse-fixed-view-from-source] \\
    [--model-profile <gpt-5.6-sol|kimi> | --provider <provider> --model <model>]

The output directory must not exist or must be empty. Each source workspace must
be a valid base-only run that matches the selected continuation model and
thinking level (--thinking-level defaults to high). The source job and result
must already use that thinking level; their history is never relabeled. Only
original drawing pages are reattached. Each case gets an independent runtime and
wall-clock timeout. The default model profile is gpt-5.6-sol through UCloud; use
--model-profile kimi for the historical model.

--no-new-images selects direct reflection using only the Initial session evidence.
It attaches no new images, withholds all newly returned tool images, and allows
unchanged valid Initial output after a completed reflection turn. It cannot be
combined with --inspect-view-search-root.

By default, sources must be original Main Agent sessions. Use
--allow-chained-self-reflect only to continue a source whose currently applied
continuation is a verified main_self_reflect turn.

--reuse-fixed-view-from-source requires --allow-chained-self-reflect and the
dimension annotation profile. It continues a completed matching dimension
reflection round using that source's exact original fixed manifest, anchor,
camera and transforms. No new alignment is run. The previous-round CAD is the
new formal baseline; the original alignment origin remains separately bound.
Each new round starts a fresh dimension cache even if its STEP is unchanged.

When --inspect-view-search-root is provided, the Self Reflect prompt omits the
obsolete no-additional-images/external-analysis sentence and adds
inspect_comparison, backed by hash-locked View Search assets under
<root>/<fixture-id>. Before finishing, the Agent must inspect the final formal
CAD in relevant aligned 2D views. Completion requires source and STEP matching
the latest formal execution, a valid, watertight STEP, and a comparison
inspection of that exact STEP.
The model may remain unchanged from the baseline. Without this option, the
original changed-source/STEP completion gate is preserved.

--inspect-region-id restricts every 2D comparison tool to one region from the
locked View Search manifest. The controller writes a derived locked manifest in
the new run workspace; the original View Search workspace remains unchanged.

--causal-repair-profile counterfactual-patch-v1 requires
--inspect-view-search-root and adds optional isolated counterfactual Patch
verification without changing the formal solution.py/output.step.

--dual-roi-crop-profile optional-dual-roi-v1 requires
--inspect-view-search-root and adds optional local Original/current-CAD crops
without adding completion requirements beyond the final aligned 2D review.

--target-annotation-profile persistent-target-annotations-v1 requires the
locked View Search root. It lets the Agent persist either a compact Point or
an extended Box with a concise name and Target-only description on the
Original panel of an inspected comparison. Stable numeric marks reappear in
later comparisons and their semantic mappings remain in subsequent context.
It does not create findings or code references.

--visual-finding-profile persistent-visual-evidence-v2 requires the locked View
Search root. It adds one persistent structured collection of target-versus-CAD
geometric differences and can be used with Inspect alone or together with
optional Dual ROI Crop.

--code-construction-profile optional-code-ablation-v1 requires the locked View
Search root and adds one optional counterfactual code-ablation localization
tool. It can be combined with --visual-finding-profile so registered source
construction references can be retained in the visual findings.

--visual-diff-profile target-current-visual-diff-v1 requires
--code-construction-profile optional-code-ablation-v1. It adds the
visual_diff tool for one-step, same-coordinate Target-versus-Current residual
overlays without natural-language geometry descriptions.

--visual-review-profile accepts executable-geometry-hypothesis-v1,
multi-candidate-brep-preview-v1, aligned-shape-feedback-v1, or
atomic-geometry-repair-v1 and requires the locked View Search root. The
first retains the legacy single-edit Target-ROI preview. The second adds
preview_repair_candidates for two or three isolated full-view code-edit
candidates with exact B-rep highlights and b0/b1/b2 changes.
The third adds only preview_shape_feedback and adopt_shape_candidate:
exactly one candidate per call in one to three aligned views. Keep the candidates array
with exactly one entry containing only id and one to eight edits.
The edits are applied to a temporary copy of the Current source.
The tool returns one three-panel card per view: Target | Current | Preview.
Images contain only the three column labels and actual views, preserving
original drawing annotations.
Target is the original engineering-drawing view.
Current and Preview are separate monochrome CAD 2D projections.
Each views entry contains only region_id, selecting a complete aligned view.
The panels share one display scale; Current and Preview use the same camera and 2D transform.
To adopt a candidate from the most recent preview, call adopt_shape_candidate
with its candidate_id. It applies the verified edits to solution.py only;
it rejects stale previews after the formal source changes.
It cannot be combined with other auxiliary tool profiles. A preview does not
replace formal execution or the final inspect_comparison requirement.

The fourth adds repair_geometry and adopt_shape_candidate to the Main Agent.
The Main Agent delegates one part and one type of suspected difference at a time.
A restricted SubAgent independently checks the premise and holds Preview;
preview_shape_feedback is not exposed to the Main Agent in this profile.
The SubAgent uses the same model and thinking level as Main. It iterates as needed,
with exactly one candidate per Preview call, and ends the repair stage with
submit_geometry_repair. There is no preview-call count limit; existing active/wall
timeouts and cancellation still apply. Main receives at most one selected candidate
with its final comparison images, reviews it, and adopts it using both
evaluation_id and candidate_id. It may instead receive an unsupported premise,
insufficient evidence, or a failed repair. Intermediate images remain in the
SubAgent session. Only geometry_repair auxiliary Agent usage is permitted.
This profile cannot be combined with other auxiliary tool profiles.

--dimension-annotation-profile aligned-dimension-subagent-v1 requires
--inspect-view-search-root with a calibrated fixed projection frame
(projection_frame.policy=fixed). A dedicated SubAgent measures the current STEP and
adds dimensions to the locked aligned views before Main reflection and
after every successful formal execution. Main still repairs the CAD and uses
inspect_comparison to review the refreshed dimensions. Entity IDs are local to
one STEP revision and cannot be reused after a model change. This profile may
be combined with aligned-shape-feedback-v1, but no other auxiliary profile.
Completion additionally requires dimensions checked for the final source,
STEP, and locked view manifest.
`;

function parseArguments(args: string[]): ParsedArguments {
	const positionals: string[] = [];
	const flags = new Map<string, string>();
	for (let index = 0; index < args.length; index += 1) {
		const token = args[index];
		if (token === undefined) continue;
		if (!token.startsWith("--")) {
			positionals.push(token);
			continue;
		}
		const name = token.slice(2);
		if (!["output-dir", "image", "timeout-minutes", "thinking-level", "inspect-view-search-root", "inspect-region-id", "causal-repair-profile", "dual-roi-crop-profile", "target-annotation-profile", "visual-finding-profile", "visual-diff-profile", "visual-review-profile", "dimension-annotation-profile", "code-construction-profile", "model-profile", "provider", "model", "allow-chained-self-reflect", "no-new-images", "reuse-fixed-view-from-source"].includes(name)) {
			throw new UsageError(`Unknown option --${name}`);
		}
		if (flags.has(name)) throw new UsageError(`Option --${name} was provided more than once`);
		if (name === "allow-chained-self-reflect" || name === "no-new-images" || name === "reuse-fixed-view-from-source") {
			flags.set(name, "true");
			continue;
		}
		const value = args[index + 1];
		if (value === undefined || value.startsWith("--")) throw new UsageError(`Option --${name} requires a value`);
		flags.set(name, value);
		index += 1;
	}
	return { positionals, flags };
}

function requiredFlag(parsed: ParsedArguments, name: string): string {
	const value = parsed.flags.get(name);
	if (!value?.trim()) throw new UsageError(`--${name} is required`);
	return value;
}

function timeoutMinutes(parsed: ParsedArguments): number {
	const raw = parsed.flags.get("timeout-minutes");
	if (raw === undefined) return DEFAULT_TIMEOUT_MINUTES;
	const value = Number(raw);
	if (!Number.isInteger(value) || value < 1 || value > MAX_TIMEOUT_MINUTES) {
		throw new UsageError(`--timeout-minutes must be an integer from 1 to ${MAX_TIMEOUT_MINUTES}`);
	}
	return value;
}

function thinkingLevel(parsed: ParsedArguments): ThinkingLevel {
	const value = parsed.flags.get("thinking-level") ?? DEFAULT_THINKING;
	if (!THINKING_LEVELS.includes(value as ThinkingLevel)) {
		throw new UsageError(`--thinking-level must be one of ${THINKING_LEVELS.join(", ")}`);
	}
	return value as ThinkingLevel;
}

function inspectRegionId(parsed: ParsedArguments): string | undefined {
	const value = parsed.flags.get("inspect-region-id");
	if (value === undefined) return undefined;
	if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u.test(value)) {
		throw new UsageError("--inspect-region-id must be a valid locked region_id");
	}
	return value;
}

function causalRepairProfile(parsed: ParsedArguments): CausalRepairProfile | undefined {
	const value = parsed.flags.get("causal-repair-profile");
	if (value === undefined) return undefined;
	if (value !== CAUSAL_REPAIR_PROFILE) {
		throw new UsageError(`--causal-repair-profile must be ${CAUSAL_REPAIR_PROFILE}`);
	}
	return value;
}

function dualRoiCropProfile(parsed: ParsedArguments): DualRoiCropProfile | undefined {
	const value = parsed.flags.get("dual-roi-crop-profile");
	if (value === undefined) return undefined;
	if (value !== DUAL_ROI_CROP_PROFILE) {
		throw new UsageError(`--dual-roi-crop-profile must be ${DUAL_ROI_CROP_PROFILE}`);
	}
	return value;
}

function visualFindingProfile(parsed: ParsedArguments): VisualFindingProfile | undefined {
	const value = parsed.flags.get("visual-finding-profile");
	if (value === undefined) return undefined;
	if (value !== VISUAL_FINDING_PROFILE) {
		throw new UsageError(`--visual-finding-profile must be ${VISUAL_FINDING_PROFILE}`);
	}
	return value;
}

function targetAnnotationProfile(parsed: ParsedArguments): TargetAnnotationProfile | undefined {
	const value = parsed.flags.get("target-annotation-profile");
	if (value === undefined) return undefined;
	if (value !== TARGET_ANNOTATION_PROFILE) {
		throw new UsageError(`--target-annotation-profile must be ${TARGET_ANNOTATION_PROFILE}`);
	}
	return value;
}

function codeConstructionProfile(parsed: ParsedArguments): CodeConstructionProfile | undefined {
	const value = parsed.flags.get("code-construction-profile");
	if (value === undefined) return undefined;
	if (value !== CODE_CONSTRUCTION_PROFILE) {
		throw new UsageError(`--code-construction-profile must be ${CODE_CONSTRUCTION_PROFILE}`);
	}
	return value;
}

function visualDiffProfile(parsed: ParsedArguments): VisualDiffProfile | undefined {
	const value = parsed.flags.get("visual-diff-profile");
	if (value === undefined) return undefined;
	if (value !== VISUAL_DIFF_PROFILE) {
		throw new UsageError(`--visual-diff-profile must be ${VISUAL_DIFF_PROFILE}`);
	}
	return value;
}

function dimensionAnnotationProfile(parsed: ParsedArguments): DimensionAnnotationProfile | undefined {
	const value = parsed.flags.get("dimension-annotation-profile");
	if (value === undefined) return undefined;
	if (value !== DIMENSION_ANNOTATION_PROFILE) {
		throw new UsageError(`--dimension-annotation-profile must be ${DIMENSION_ANNOTATION_PROFILE}`);
	}
	return value;
}

function visualReviewProfile(parsed: ParsedArguments): VisualReviewProfile | undefined {
	const value = parsed.flags.get("visual-review-profile");
	if (value === undefined) return undefined;
	if (value !== VISUAL_REVIEW_PROFILE && value !== REPAIR_PREVIEW_PROFILE && value !== SHAPE_FEEDBACK_PROFILE && value !== GEOMETRY_REPAIR_PROFILE) {
		throw new UsageError(`--visual-review-profile must be ${VISUAL_REVIEW_PROFILE}, ${REPAIR_PREVIEW_PROFILE}, ${SHAPE_FEEDBACK_PROFILE}, or ${GEOMETRY_REPAIR_PROFILE}`);
	}
	return value;
}

async function sha256(path: string): Promise<string> {
	return createHash("sha256").update(await readFile(path)).digest("hex");
}

async function treeSha256(root: string): Promise<string> {
	const hash = createHash("sha256");
	const visit = async (directory: string): Promise<void> => {
		const entries = await readdir(directory, { withFileTypes: true });
		entries.sort((left, right) => left.name.localeCompare(right.name));
		for (const entry of entries) {
			const path = join(directory, entry.name);
			const name = relative(root, path);
			const metadata = await lstat(path);
			if (metadata.isSymbolicLink()) throw new InfrastructureError(`Source workspace contains a symlink: ${path}`);
			hash.update(`${entry.isDirectory() ? "d" : "f"}\0${name}\0`);
			if (entry.isDirectory()) await visit(path);
			else if (entry.isFile()) hash.update(await readFile(path));
			else throw new InfrastructureError(`Source workspace contains an unsupported entry: ${path}`);
		}
	};
	await visit(root);
	return hash.digest("hex");
}

async function requireFile(path: string, label: string): Promise<void> {
	const metadata = await stat(path).catch(() => null);
	if (!metadata?.isFile()) throw new InfrastructureError(`${label} is missing: ${path}`);
}

async function fileExists(path: string): Promise<boolean> {
	return (await stat(path).catch(() => null))?.isFile() === true;
}

async function directoryExists(path: string): Promise<boolean> {
	return (await stat(path).catch(() => null))?.isDirectory() === true;
}

async function requireEmptyOutput(path: string): Promise<void> {
	const metadata = await stat(path).catch(() => null);
	if (metadata === null) {
		await mkdir(path, { recursive: true });
		return;
	}
	if (!metadata.isDirectory()) throw new UsageError(`--output-dir is not a directory: ${path}`);
	if ((await readdir(path)).length > 0) throw new UsageError(`--output-dir must be empty: ${path}`);
}

async function safeSourceFile(root: string, path: string, label: string): Promise<string> {
	if (isAbsolute(path) || path.split(/[\\/]/u).includes("..")) {
		throw new InfrastructureError(`${label} path escaped its workspace: ${path}`);
	}
	const rootReal = await realpath(root);
	const candidate = resolve(root, path);
	await requireFile(candidate, label);
	const candidateReal = await realpath(candidate);
	const rel = relative(rootReal, candidateReal);
	if (rel === ".." || rel.startsWith("../") || isAbsolute(rel)) {
		throw new InfrastructureError(`${label} resolved outside its workspace: ${path}`);
	}
	return candidateReal;
}

async function copyLockedFile(sourceRoot: string, destinationRoot: string, path: string): Promise<LockedAsset> {
	const source = await safeSourceFile(sourceRoot, path, "Locked View Search artifact");
	const destination = resolve(destinationRoot, path);
	const rel = relative(resolve(destinationRoot), destination);
	if (rel === ".." || rel.startsWith("../") || isAbsolute(rel)) {
		throw new InfrastructureError(`Locked View Search destination escaped the Case workspace: ${path}`);
	}
	await mkdir(dirname(destination), { recursive: true });
	await cp(source, destination);
	return { path, sha256: await sha256(destination) };
}

async function verifyLockedAssets(workspace: string, assets: LockedAsset[]): Promise<void> {
	for (const asset of assets) {
		const path = await safeSourceFile(workspace, asset.path, "Copied locked View Search artifact");
		if (await sha256(path) !== asset.sha256) {
			throw new InfrastructureError(`Locked View Search artifact changed during the run: ${asset.path}`);
		}
	}
}

async function assertNoAuxiliaryArtifacts(workspace: string, result: AgentResult): Promise<void> {
	const stateFields = auxiliaryStateFields(result);
	if (stateFields.length > 0) {
		throw new InfrastructureError(`Auxiliary Agent state is forbidden: ${stateFields.join(", ")}`);
	}
	if (await directoryExists(join(workspace, "drawing_feedback"))) {
		throw new InfrastructureError(`Auxiliary Agent output is forbidden: ${join(workspace, "drawing_feedback")}`);
	}
}

async function assertNoReflectionAgentState(result: AgentResult): Promise<void> {
	assertSelfReflectAuxiliaryState(result);
}

/** Bind each round to its immediate predecessor, including unchanged CAD rounds. */
export function chainedContinuationSuffix(previousId: string, sourceSha256: string, stepSha256: string): string {
	return `chain-${createHash("sha256").update(JSON.stringify([previousId, sourceSha256, stepSha256])).digest("hex").slice(0, 20)}`;
}

async function copySourceWorkspace(source: string, destination: string): Promise<void> {
	await cp(source, destination, {
		recursive: true,
		filter: (path) => {
			if (path === source) return true;
			return ![
				"container.json",
				JOB_FILE,
				".cadgenbench-pi-response.json",
				"baseline",
				"after",
				"partial",
				"continuation_result.json",
				"drawing_feedback",
			].includes(basename(path));
		},
	});
}

async function copyBaselineRenders(sourceWorkspace: string, baselineDir: string, lastTurn: number): Promise<string> {
	const rootRenderDir = join(baselineDir, "root-renders");
	await mkdir(rootRenderDir, { recursive: true });
	for (const entry of await readdir(sourceWorkspace, { withFileTypes: true })) {
		if (entry.isFile() && entry.name.toLowerCase().endsWith(".png")) {
			await cp(join(sourceWorkspace, entry.name), join(rootRenderDir, entry.name));
		}
	}
	const lastIso = join(sourceWorkspace, "turns", String(lastTurn).padStart(4, "0"), "iso.png");
	await requireFile(lastIso, "Last baseline turn render");
	await cp(lastIso, join(baselineDir, "iso.png"));
	return "baseline/iso.png";
}

async function prepareCase(options: {
	sourceWorkspace: string;
	viewSearchRoot?: string;
	runDir: string;
	runId: string;
	image: string;
	imageId: string;
	runtime: DockerAgentRuntime;
	modelSelection: ExperimentModelSelection;
	thinkingLevel: ThinkingLevel;
	allowChainedSelfReflect: boolean;
	noNewImages: boolean;
	reuseFixedViewFromSource: boolean;
	inspectRegionId?: string;
	causalRepairProfile?: CausalRepairProfile;
	dualRoiCropProfile?: DualRoiCropProfile;
	targetAnnotationProfile?: TargetAnnotationProfile;
	visualFindingProfile?: VisualFindingProfile;
	codeConstructionProfile?: CodeConstructionProfile;
	visualDiffProfile?: VisualDiffProfile;
	visualReviewProfile?: VisualReviewProfile;
	dimensionAnnotationProfile?: DimensionAnnotationProfile;
}): Promise<PreparedCase> {
	const sourceWorkspace = resolve(options.sourceWorkspace);
	const sourceResultPath = join(sourceWorkspace, "agent_result.json");
	const sourceJobPath = join(sourceWorkspace, JOB_FILE);
	await Promise.all([
		requireFile(sourceResultPath, "Source Agent result"),
		requireFile(sourceJobPath, "Source Agent job"),
		requireFile(join(sourceWorkspace, "solution.py"), "Source solution.py"),
		requireFile(join(sourceWorkspace, "output.step"), "Source output.step"),
	]);
	const [sourceResult, sourceJob] = await Promise.all([
		readJson<AgentResult>(sourceResultPath),
		readJson<AgentJob>(sourceJobPath),
	]);
	const fixtureId = sourceResult.fixture_id;
	if (sourceResult.status !== "valid" || sourceJob.fixture.id !== fixtureId) {
		throw new InfrastructureError(`Source workspace ${sourceWorkspace} is not a valid matching Pi run`);
	}
	if (sourceResult.agent_mode !== "base-only" || sourceJob.agent_mode !== "base-only") {
		throw new InfrastructureError(`Case ${fixtureId} is not an exact base-only Pi run`);
	}
	if (
		sourceJob.provider !== options.modelSelection.provider ||
		sourceJob.model !== options.modelSelection.model ||
		sourceJob.thinking_level !== options.thinkingLevel ||
		sourceResult.provider !== options.modelSelection.provider ||
		sourceResult.model !== options.modelSelection.model ||
		sourceResult.thinking_level !== options.thinkingLevel
	) {
		throw new InfrastructureError(
			`Case ${fixtureId} is not the required ${options.modelSelection.provider}/${options.modelSelection.model} ${options.thinkingLevel} run`,
		);
	}
	const sourceContinuationKind = classifySelfReflectSource(sourceJob, sourceResult);
	if (sourceContinuationKind === "incompatible") {
		throw new InfrastructureError(`Case ${fixtureId} has an incompatible or unverifiable prior continuation`);
	}
	if (sourceContinuationKind === "main_self_reflect" && !options.allowChainedSelfReflect) {
		throw new InfrastructureError(
			`Case ${fixtureId} is already a self-reflection continuation; pass --allow-chained-self-reflect to continue it`,
		);
	}
	if (options.reuseFixedViewFromSource) {
		const prior = sourceJob.continuation;
		if (sourceContinuationKind !== "main_self_reflect" || prior?.kind !== MAIN_SELF_REFLECT_INSPECT_KIND || prior.dimension_annotation_profile !== DIMENSION_ANNOTATION_PROFILE || prior.dimension_annotation_profile !== options.dimensionAnnotationProfile || prior.visual_review_profile !== options.visualReviewProfile) {
			throw new InfrastructureError(`Case ${fixtureId} fixed-view reuse requires a prior matching completed dimension reflection profile`);
		}
		if (prior.no_new_images || [prior.causal_repair_profile, prior.dual_roi_crop_profile, prior.target_annotation_profile, prior.visual_finding_profile, prior.visual_diff_profile, prior.code_construction_profile].some(profile => profile !== undefined)) throw new InfrastructureError("Fixed-view dimension chaining cannot import other reflection profiles");
		assertSelfReflectAuxiliaryState(sourceResult, options.visualReviewProfile, options.dimensionAnnotationProfile);
	} else await assertNoAuxiliaryArtifacts(sourceWorkspace, sourceResult);
	if (!sourceResult.session_file) throw new InfrastructureError(`Case ${fixtureId} has no resumable Pi session`);
	await requireFile(join(sourceWorkspace, sourceResult.session_file), `Case ${fixtureId} Pi session`);

	const baselineSourceSha256 = await sha256(join(sourceWorkspace, "solution.py"));
	const baselineStepSha256 = await sha256(join(sourceWorkspace, "output.step"));
	if (
		baselineSourceSha256 !== sourceResult.last_executed_source_sha256 ||
		baselineStepSha256 !== sourceResult.last_executed_step_sha256
	) {
		throw new InfrastructureError(`Case ${fixtureId} source/STEP no longer match its recorded formal result`);
	}
	if (sourceContinuationKind === "main_self_reflect" && !selfReflectCompletionAccepted(sourceJob, sourceResult, {
		source_sha256: baselineSourceSha256,
		step_sha256: baselineStepSha256,
		formal_executions_added: sourceResult.last_turn,
		is_valid: sourceResult.artifact?.validation?.is_valid === true,
		is_watertight: sourceResult.artifact?.validation?.is_watertight === true,
	})) throw new InfrastructureError(`Case ${fixtureId} prior self-reflection round is incomplete or has stale final evidence`);

	let viewWorkspace: string | undefined;
	let viewTreeSha256: string | undefined;
	let manifestPath: string | undefined;
	let manifestSha256: string | undefined;
	let viewResultPath: string | undefined;
	let lockedManifest: DrawingViewManifest | undefined;
	let viewOriginSourceSha256: string | undefined;
	let viewOriginStepSha256: string | undefined;
	if (options.viewSearchRoot || options.reuseFixedViewFromSource) {
		let sourceManifestPath: string;
		if (options.reuseFixedViewFromSource) {
			viewWorkspace = sourceWorkspace;
			const prior = sourceJob.continuation!;
			if (!prior.locked_view_manifest_path || !prior.locked_view_manifest_sha256) throw new InfrastructureError("Prior dimension round lacks a locked view manifest binding");
			sourceManifestPath = prior.locked_view_manifest_path;
			manifestSha256 = prior.locked_view_manifest_sha256;
			viewOriginSourceSha256 = prior.view_origin_source_sha256 ?? prior.baseline_source_sha256;
			viewOriginStepSha256 = prior.view_origin_step_sha256 ?? prior.baseline_step_sha256;
			if (Boolean(prior.view_origin_source_sha256) !== Boolean(prior.view_origin_step_sha256)) throw new InfrastructureError("Prior alignment origin hash pair is incomplete");
		} else {
			viewWorkspace = resolve(options.viewSearchRoot!, fixtureId);
			viewResultPath = join(viewWorkspace, "view_search_only_result.json");
			await requireFile(viewResultPath, `Case ${fixtureId} View Search result`);
			const viewResult = await readJson<ViewSearchOnlyResult>(viewResultPath);
			if (
				viewResult.status !== "completed" ||
				viewResult.fixture_id !== fixtureId ||
				viewResult.reflection_agent_started !== false ||
				viewResult.agent_roles?.includes("reflection") ||
				!viewResult.manifest_path ||
				viewResult.source_sha256 !== baselineSourceSha256 ||
				viewResult.step_sha256 !== baselineStepSha256 ||
				resolve(viewResult.source_workspace ?? "") !== sourceWorkspace
			) throw new InfrastructureError(`Case ${fixtureId} View Search result does not exactly match the Self Reflect baseline`);
			if (
				await sha256(join(viewWorkspace, "solution.py")) !== baselineSourceSha256 ||
				await sha256(join(viewWorkspace, "output.step")) !== baselineStepSha256
			) throw new InfrastructureError(`Case ${fixtureId} View Search workspace source/STEP mismatch`);
			sourceManifestPath = viewResult.manifest_path;
		}
		const manifestFile = await safeSourceFile(viewWorkspace, sourceManifestPath, `Case ${fixtureId} View Search manifest`);
		manifestSha256 ??= await sha256(manifestFile);
		lockedManifest = await loadLockedMainViewManifest({
			workspace: viewWorkspace,
			manifestPath: sourceManifestPath,
			manifestSha256,
			baselineSourceSha256: viewOriginSourceSha256 ?? baselineSourceSha256,
			baselineStepSha256: viewOriginStepSha256 ?? baselineStepSha256,
		});
		if (options.visualReviewProfile === GEOMETRY_REPAIR_PROFILE && lockedManifest.projection_frame?.policy !== "fixed") {
			throw new InfrastructureError("Atomic geometry repair requires a calibrated fixed View Search manifest (projection_frame.policy=fixed)");
		}
		if (options.dimensionAnnotationProfile && lockedManifest.projection_frame?.policy !== "fixed") {
			throw new InfrastructureError("Dimension annotations require a calibrated fixed View Search manifest (projection_frame.policy=fixed)");
		}
		if (options.inspectRegionId) {
			lockedManifest = filterLockedMainViewManifest(lockedManifest, options.inspectRegionId);
			manifestPath = "drawing_feedback/locked-view-manifest/view_manifest.json";
			manifestSha256 = undefined;
		} else {
			manifestPath = sourceManifestPath;
		}
		viewTreeSha256 = await treeSha256(viewWorkspace);
	}

	const baselineTreeSha256 = await treeSha256(sourceWorkspace);
	const workspace = join(options.runDir, fixtureId);
	await copySourceWorkspace(sourceWorkspace, workspace);
	const lockedAssets: LockedAsset[] = [];
	if (viewWorkspace && manifestPath && lockedManifest) {
		const lockedPaths = new Set<string>();
		if (lockedManifest.projection_frame) lockedPaths.add(lockedManifest.projection_frame.anchor_step_path);
		if (!options.inspectRegionId) lockedPaths.add(manifestPath);
		for (const region of lockedManifest.regions) {
			lockedPaths.add(region.crop_path);
			if (region.selected) {
				lockedPaths.add(region.selected.render_path);
				lockedPaths.add(region.selected.aligned_render_path);
				lockedPaths.add(region.selected.comparison_path);
			}
		}
		for (const path of [...lockedPaths].sort()) {
			lockedAssets.push(await copyLockedFile(viewWorkspace, workspace, path));
		}
		if (options.inspectRegionId) {
			await writeJsonAtomic(join(workspace, manifestPath), lockedManifest);
			manifestSha256 = await sha256(join(workspace, manifestPath));
			lockedAssets.push({ path: manifestPath, sha256: manifestSha256 });
		}
		if (!manifestSha256) throw new InfrastructureError("Locked View Search manifest hash was not established");
		await loadLockedMainViewManifest({
			workspace,
			manifestPath,
			manifestSha256,
			baselineSourceSha256: viewOriginSourceSha256 ?? baselineSourceSha256,
			baselineStepSha256: viewOriginStepSha256 ?? baselineStepSha256,
		});
	}
	const baselineDir = join(workspace, "baseline");
	await mkdir(baselineDir, { recursive: true });
	await Promise.all([
		cp(join(sourceWorkspace, "solution.py"), join(baselineDir, "solution.py")),
		cp(join(sourceWorkspace, "output.step"), join(baselineDir, "output.step")),
		cp(sourceResultPath, join(baselineDir, "agent_result.json")),
		cp(join(sourceWorkspace, sourceResult.session_file), join(baselineDir, "session.jsonl")),
		...(viewResultPath ? [cp(viewResultPath, join(baselineDir, "view_search_only_result.json"))] : []),
	]);
	const baselineRenderPath = await copyBaselineRenders(sourceWorkspace, baselineDir, sourceResult.last_turn);
	const prompt = viewWorkspace
		? mainSelfReflectInspectPrompt(
				fixtureId,
				options.causalRepairProfile,
				options.dualRoiCropProfile,
				options.visualFindingProfile,
				options.codeConstructionProfile,
				options.visualDiffProfile,
				options.visualReviewProfile,
				options.targetAnnotationProfile,
				options.dimensionAnnotationProfile,
			)
		: options.noNewImages ? mainSelfReflectNoNewImagesPrompt(fixtureId, sourceContinuationKind === "main_self_reflect") : mainSelfReflectPrompt(fixtureId);
	const promptSha256 = createHash("sha256").update(prompt).digest("hex");
	const continuationKind = viewWorkspace ? MAIN_SELF_REFLECT_INSPECT_KIND : MAIN_SELF_REFLECT_KIND;
	const continuationId = [
		`${continuationKind.replaceAll("_", "-")}-${fixtureId}-${promptSha256.slice(0, 16)}`,
		...(options.inspectRegionId && manifestSha256
			? [`region-${options.inspectRegionId}-${manifestSha256.slice(0, 12)}`]
			: []),
		...(sourceContinuationKind === "main_self_reflect" ? [chainedContinuationSuffix(sourceJob.continuation!.id, baselineSourceSha256, baselineStepSha256)] : []),
	].join("-");
	await writeJsonAtomic(join(baselineDir, "manifest.json"), {
		fixture_id: fixtureId,
		created_at: isoNow(),
		source_workspace: sourceWorkspace,
		source_workspace_tree_sha256: baselineTreeSha256,
		session_file: sourceResult.session_file,
		session_id: basename(sourceResult.session_file, ".jsonl").split("_").at(-1),
		source_sha256: baselineSourceSha256,
		step_sha256: baselineStepSha256,
		render_path: baselineRenderPath,
		last_turn: sourceResult.last_turn,
		continuation_kind: continuationKind,
		continuation_id: continuationId,
		prompt_sha256: promptSha256,
		baseline_provider: sourceJob.provider,
		baseline_model: sourceJob.model,
		baseline_thinking_level: sourceJob.thinking_level,
		continuation_model_profile: options.modelSelection.profile,
		continuation_provider: options.modelSelection.provider,
		continuation_model: options.modelSelection.model,
		continuation_thinking_level: options.thinkingLevel,
		attached_continuation_images: [],
		no_new_images: options.noNewImages,
		reuse_fixed_view_from_source: options.reuseFixedViewFromSource,
		...(viewOriginSourceSha256 && viewOriginStepSha256 ? { view_origin_source_sha256: viewOriginSourceSha256, view_origin_step_sha256: viewOriginStepSha256 } : {}),
		causal_repair_profile: options.causalRepairProfile ?? null,
		dual_roi_crop_profile: options.dualRoiCropProfile ?? null,
		target_annotation_profile: options.targetAnnotationProfile ?? null,
		visual_finding_profile: options.visualFindingProfile ?? null,
		visual_diff_profile: options.visualDiffProfile ?? null,
		visual_review_profile: options.visualReviewProfile ?? null,
		dimension_annotation_profile: options.dimensionAnnotationProfile ?? null,
		code_construction_profile: options.codeConstructionProfile ?? null,
		inspect_region_id: options.inspectRegionId ?? null,
		source_continuation_kind: sourceContinuationKind,
		source_continuation_id: sourceResult.applied_continuation?.id ?? null,
		...(viewWorkspace && viewTreeSha256 && manifestPath && manifestSha256 ? {
			view_search_workspace: viewWorkspace,
			view_search_workspace_tree_sha256: viewTreeSha256,
			locked_view_manifest_path: manifestPath,
			locked_view_manifest_sha256: manifestSha256,
			locked_assets: lockedAssets,
		} : {}),
	});

	const now = isoNow();
	const copiedResult: AgentResult = {
		...sourceResult,
		status: "valid",
		provider: options.modelSelection.provider,
		model: options.modelSelection.model,
		thinking_level: options.thinkingLevel,
		container_image: options.image,
		container_image_digest: options.imageId,
		started_at: now,
		updated_at: now,
		duration_s: 0,
	};
	delete copiedResult.stop_reason;
	delete copiedResult.resume_reason;
	delete copiedResult.applied_continuation;
	delete copiedResult.optional_reflect_turn;
	delete copiedResult.main_view_reflection;
	delete copiedResult.causal_repair;
	delete copiedResult.dimension_annotations;
	await writeJsonAtomic(join(workspace, "agent_result.json"), copiedResult);

	const job: AgentJob = {
		...sourceJob,
		protocol_version: PROTOCOL_VERSION,
		run_id: options.runId,
		agent_mode: "base-only",
		provider: options.modelSelection.provider,
		model: options.modelSelection.model,
		thinking_level: options.thinkingLevel,
		pi_version: PI_VERSION,
		container_image: options.image,
		container_image_digest: options.imageId,
		continuation: {
			kind: continuationKind,
			id: continuationId,
			prompt,
			image_paths: [],
			...(options.noNewImages ? { no_new_images: true } : {}),
			baseline_source_sha256: baselineSourceSha256,
			baseline_step_sha256: baselineStepSha256,
			...(sourceContinuationKind === "main_self_reflect" ? { previous_continuation_id: sourceJob.continuation!.id } : {}),
			...(options.reuseFixedViewFromSource ? { reuse_fixed_view_from_source: true, view_origin_source_sha256: viewOriginSourceSha256!, view_origin_step_sha256: viewOriginStepSha256! } : {}),
			...(manifestPath && manifestSha256 ? {
				locked_view_manifest_path: manifestPath,
				locked_view_manifest_sha256: manifestSha256,
			} : {}),
			...(options.causalRepairProfile ? { causal_repair_profile: options.causalRepairProfile } : {}),
			...(options.dualRoiCropProfile ? { dual_roi_crop_profile: options.dualRoiCropProfile } : {}),
			...(options.targetAnnotationProfile ? { target_annotation_profile: options.targetAnnotationProfile } : {}),
			...(options.visualFindingProfile ? { visual_finding_profile: options.visualFindingProfile } : {}),
			...(options.visualDiffProfile ? { visual_diff_profile: options.visualDiffProfile } : {}),
			...(options.visualReviewProfile ? { visual_review_profile: options.visualReviewProfile } : {}),
			...(options.dimensionAnnotationProfile ? { dimension_annotation_profile: options.dimensionAnnotationProfile } : {}),
			...(options.codeConstructionProfile ? { code_construction_profile: options.codeConstructionProfile } : {}),
		},
	};
	await writeJsonAtomic(join(workspace, JOB_FILE), job);
	return {
		fixtureId,
		workspace,
		sourceWorkspace,
		continuationId,
		promptSha256,
		baselineSourceSha256,
		baselineStepSha256,
		baselineTreeSha256,
		...(viewWorkspace && viewTreeSha256 && manifestPath && manifestSha256 ? {
			viewWorkspace,
			viewTreeSha256,
			...(options.inspectRegionId ? { inspectRegionId: options.inspectRegionId } : {}),
			manifestPath,
			manifestSha256,
		} : {}),
		lockedAssets,
		baselineLastTurn: sourceResult.last_turn,
		baselineUsage: {
			tokens: sourceResult.tokens,
			cost: sourceResult.cost,
			tool_calls: sourceResult.tool_calls,
		},
		job,
		runtime: options.runtime,
		sourceContinuationKind,
	};
}

async function captureAfter(prepared: PreparedCase): Promise<SelfReflectAfter> {
	const result = await readJson<AgentResult>(join(prepared.workspace, "agent_result.json"));
	if (prepared.viewWorkspace) {
		assertSelfReflectAuxiliaryState(result, prepared.job.continuation?.visual_review_profile, prepared.job.continuation?.dimension_annotation_profile);
		await verifyLockedAssets(prepared.workspace, prepared.lockedAssets);
	} else {
		await assertNoAuxiliaryArtifacts(prepared.workspace, result);
	}
	if (!result.session_file) throw new InfrastructureError(`Case ${prepared.fixtureId} lost its Pi session path`);
	const afterDir = join(prepared.workspace, "after");
	await mkdir(afterDir, { recursive: true });
	const sourceSha256 = await sha256(join(prepared.workspace, "solution.py"));
	const stepSha256 = await sha256(join(prepared.workspace, "output.step"));
	const isoPath = join(prepared.workspace, "turns", String(result.last_turn).padStart(4, "0"), "iso.png");
	await requireFile(isoPath, `Case ${prepared.fixtureId} final render`);
	const copies = [
		cp(join(prepared.workspace, "solution.py"), join(afterDir, "solution.py")),
		cp(join(prepared.workspace, "output.step"), join(afterDir, "output.step")),
		cp(join(prepared.workspace, "agent_result.json"), join(afterDir, "agent_result.json")),
		cp(join(prepared.workspace, result.session_file), join(afterDir, "session.jsonl")),
		cp(isoPath, join(afterDir, "iso.png")),
	];
	const inspectionCount = result.main_view_reflection?.inspection_count ?? 0;
	const auditSource = result.main_view_reflection?.audit_manifest_path;
	if (inspectionCount > 0 && auditSource && await fileExists(join(prepared.workspace, auditSource))) {
		copies.push(cp(join(prepared.workspace, auditSource), join(afterDir, "inspect_comparison_audit.json")));
	}
	const causalRepairEvaluationCount = result.causal_repair?.evaluation_count ?? 0;
	const causalAuditSource = result.causal_repair?.audit_manifest_path;
	if (causalRepairEvaluationCount > 0 && causalAuditSource && await fileExists(join(prepared.workspace, causalAuditSource))) {
		copies.push(cp(join(prepared.workspace, causalAuditSource), join(afterDir, "causal_repair_audit.json")));
	}
	const visualFindingsStateExists =
		prepared.job.continuation?.visual_finding_profile === VISUAL_FINDING_PROFILE &&
		await fileExists(join(prepared.workspace, VISUAL_FINDING_STATE_PATH));
	if (visualFindingsStateExists) {
		copies.push(cp(join(prepared.workspace, VISUAL_FINDING_STATE_PATH), join(afterDir, "visual_findings_state.json")));
	}
	const visualDiffsStateExists =
		prepared.job.continuation?.visual_diff_profile === VISUAL_DIFF_PROFILE &&
		await fileExists(join(prepared.workspace, VISUAL_DIFF_STATE_PATH));
	if (visualDiffsStateExists) {
		copies.push(cp(join(prepared.workspace, VISUAL_DIFF_STATE_PATH), join(afterDir, "visual_diffs_state.json")));
	}
	const visualReviewsStateExists =
		prepared.job.continuation?.visual_review_profile === VISUAL_REVIEW_PROFILE &&
		await fileExists(join(prepared.workspace, VISUAL_REVIEW_STATE_PATH));
	if (visualReviewsStateExists) {
		copies.push(cp(join(prepared.workspace, VISUAL_REVIEW_STATE_PATH), join(afterDir, "geometry_hypotheses_state.json")));
	}
	const targetAnnotationsStateExists =
		prepared.job.continuation?.target_annotation_profile === TARGET_ANNOTATION_PROFILE &&
		await fileExists(join(prepared.workspace, TARGET_ANNOTATION_STATE_PATH));
	if (targetAnnotationsStateExists) {
		copies.push(cp(join(prepared.workspace, TARGET_ANNOTATION_STATE_PATH), join(afterDir, "target_annotations_state.json")));
	}
	const dimensionStatePath = prepared.job.continuation?.dimension_annotation_profile === DIMENSION_ANNOTATION_PROFILE
		? result.dimension_annotations?.state_path : undefined;
	if (dimensionStatePath) {
		const stateFile = await safeSourceFile(prepared.workspace, dimensionStatePath, "Dimension annotation state");
		copies.push(cp(stateFile, join(afterDir, "dimension_annotations_state.json")));
	}
	const codeConstructionAuditExists =
		prepared.job.continuation?.code_construction_profile === CODE_CONSTRUCTION_PROFILE &&
		await fileExists(join(prepared.workspace, CODE_CONSTRUCTION_AUDIT_PATH));
	const codeConstructionAudit = codeConstructionAuditExists
		? await readJson<{
				constructions?: unknown[];
				active_construction_refs?: unknown[];
				inspections?: unknown[];
			}>(join(prepared.workspace, CODE_CONSTRUCTION_AUDIT_PATH))
		: null;
	if (codeConstructionAuditExists) {
		copies.push(cp(join(prepared.workspace, CODE_CONSTRUCTION_AUDIT_PATH), join(afterDir, "code_construction_audit.json")));
	}
	await Promise.all(copies);
	return {
		source_sha256: sourceSha256,
		step_sha256: stepSha256,
		render_path: "after/iso.png",
		session_file: result.session_file,
		last_turn: result.last_turn,
		formal_executions_added: result.last_turn - prepared.baselineLastTurn,
		is_valid: result.artifact?.validation?.is_valid === true,
		is_watertight: result.artifact?.validation?.is_watertight === true,
		inspection_count: inspectionCount,
		...(inspectionCount > 0 ? { audit_manifest_path: "after/inspect_comparison_audit.json" } : {}),
		...(causalRepairEvaluationCount > 0 ? {
			causal_repair_evaluation_count: causalRepairEvaluationCount,
			causal_repair_audit_manifest_path: "after/causal_repair_audit.json",
		} : {}),
		...(visualFindingsStateExists ? { visual_findings_state_path: "after/visual_findings_state.json" } : {}),
		...(visualDiffsStateExists ? { visual_diffs_state_path: "after/visual_diffs_state.json" } : {}),
		...(visualReviewsStateExists ? { geometry_hypotheses_state_path: "after/geometry_hypotheses_state.json" } : {}),
		...(targetAnnotationsStateExists ? { target_annotations_state_path: "after/target_annotations_state.json" } : {}),
		...(dimensionStatePath ? { dimension_annotations_state_path: "after/dimension_annotations_state.json" } : {}),
		...(codeConstructionAuditExists ? {
			code_construction_inspection_count: codeConstructionAudit?.inspections?.length ?? 0,
			code_construction_reference_count: codeConstructionAudit?.constructions?.length ?? 0,
			code_construction_active_reference_count:
				codeConstructionAudit?.active_construction_refs?.length ?? codeConstructionAudit?.constructions?.length ?? 0,
			code_construction_audit_path: "after/code_construction_audit.json",
		} : {}),
	};
}

async function capturePartial(prepared: PreparedCase): Promise<void> {
	const partialDir = join(prepared.workspace, "partial");
	const candidates = [
		["solution.py", "solution.py"],
		["output.step", "output.step"],
		["agent_result.json", "agent_result.json"],
	] as const;
	let copied = false;
	for (const [sourceName, targetName] of candidates) {
		const source = join(prepared.workspace, sourceName);
		if (await fileExists(source)) {
			await mkdir(partialDir, { recursive: true });
			await cp(source, join(partialDir, targetName));
			copied = true;
		}
	}
	const result = await readJson<AgentResult>(join(prepared.workspace, "agent_result.json")).catch(() => null);
	if (result?.session_file && await fileExists(join(prepared.workspace, result.session_file))) {
		if (!copied) await mkdir(partialDir, { recursive: true });
		await cp(join(prepared.workspace, result.session_file), join(partialDir, "session.jsonl"));
	}
}

export function selfReflectCompletionAccepted(
	job: AgentJob,
	result: AgentResult,
	after: Pick<SelfReflectAfter, "source_sha256" | "step_sha256" | "formal_executions_added" | "is_valid" | "is_watertight">,
): boolean {
	const continuation = job.continuation;
	if (
		!continuation ||
		result.status !== "valid" ||
		result.agent_mode !== "base-only" ||
		result.applied_continuation?.id !== continuation.id ||
		result.applied_continuation.prompt_sha256 !== createHash("sha256").update(continuation.prompt).digest("hex") ||
		!after.is_valid ||
		!after.is_watertight
	) return false;
	if (continuation.kind === MAIN_SELF_REFLECT_INSPECT_KIND) {
		const turn = result.optional_reflect_turn;
		const dimensions = result.dimension_annotations;
		if (continuation.dimension_annotation_profile === DIMENSION_ANNOTATION_PROFILE && (
			dimensions?.profile !== DIMENSION_ANNOTATION_PROFILE ||
			dimensions.status !== "completed" ||
			dimensions.source_sha256 !== after.source_sha256 ||
			dimensions.step_sha256 !== after.step_sha256 ||
			!continuation.locked_view_manifest_sha256 ||
			dimensions.locked_manifest_sha256 !== continuation.locked_view_manifest_sha256
		)) return false;
		return turn?.status === "completed" &&
			turn.continuation_id === continuation.id &&
			turn.prompt_sha256 === result.applied_continuation.prompt_sha256 &&
			artifactAccepted(result.artifact) &&
			after.source_sha256 === result.last_executed_source_sha256 &&
			after.step_sha256 === result.last_executed_step_sha256 &&
			after.step_sha256 === result.artifact?.sha256 &&
			summaryHasInspectedStep(result.main_view_reflection, after.step_sha256);
	}
	if (continuation.no_new_images) {
		return result.optional_reflect_turn?.status === "completed" &&
			result.optional_reflect_turn.continuation_id === continuation.id &&
			result.optional_reflect_turn.prompt_sha256 === result.applied_continuation.prompt_sha256 &&
			artifactAccepted(result.artifact) &&
			after.source_sha256 === result.last_executed_source_sha256 &&
			after.step_sha256 === result.last_executed_step_sha256 &&
			after.step_sha256 === result.artifact?.sha256;
	}
	return after.formal_executions_added >= 1 &&
		after.source_sha256 !== continuation.baseline_source_sha256 &&
		after.step_sha256 !== continuation.baseline_step_sha256;
}

async function runCase(
	prepared: PreparedCase,
	imageId: string,
	timeoutMinutesValue: number,
	isInterrupted: () => boolean,
): Promise<SelfReflectCaseResult> {
	const startedAt = isoNow();
	let status: SelfReflectCaseStatus = "error";
	let after: SelfReflectAfter | undefined;
	let result: AgentResult | null = null;
	let failure: string | undefined;
	try {
		await withIsolatedTimeout({
			operation: prepared.runtime.runFixture(prepared.workspace, prepared.job, imageId),
			timeoutMs: timeoutMinutesValue * 60_000,
			onTimeout: async () => await prepared.runtime.interruptAll(),
		});
		result = await readJson<AgentResult>(join(prepared.workspace, "agent_result.json"));
		after = await captureAfter(prepared);
		if (!selfReflectCompletionAccepted(prepared.job, result, after)) {
			const gate = prepared.job.continuation?.kind === MAIN_SELF_REFLECT_INSPECT_KIND
				? "current-formal-source/STEP, final-STEP comparison inspection, valid/watertight"
				: "fresh-execution, changed-source/STEP, valid/watertight";
			throw new InfrastructureError(`Case ${prepared.fixtureId} failed the ${gate} gate`);
		}
		status = "completed";
	} catch (error) {
		status = error instanceof CaseTimeoutError ? "timeout" : isInterrupted() ? "interrupted" : "error";
		failure = errorMessage(error);
		await capturePartial(prepared).catch(() => undefined);
		result = await readJson<AgentResult>(join(prepared.workspace, "agent_result.json")).catch(() => null);
	}
	const sourceTreeAfter = await treeSha256(prepared.sourceWorkspace);
	const viewTreeAfter = prepared.viewWorkspace ? await treeSha256(prepared.viewWorkspace) : undefined;
	if (
		sourceTreeAfter !== prepared.baselineTreeSha256 ||
		(prepared.viewTreeSha256 !== undefined && viewTreeAfter !== prepared.viewTreeSha256)
	) {
		status = "error";
		failure = "InfrastructureError: Original source or View Search workspace changed during the run";
	}
	const noAuxiliaryState = result !== null && await assertNoReflectionAgentState(result)
		.then(() => true, () => false);
	const allowedAuxiliaryAgentState = result !== null && (() => {
		try { assertSelfReflectAuxiliaryState(result, prepared.job.continuation?.visual_review_profile, prepared.job.continuation?.dimension_annotation_profile); return true; }
		catch { return false; }
	})();
	const noReflectionAgentState = prepared.job.continuation?.visual_review_profile === GEOMETRY_REPAIR_PROFILE ||
		prepared.job.continuation?.dimension_annotation_profile === DIMENSION_ANNOTATION_PROFILE
		? allowedAuxiliaryAgentState
		: noAuxiliaryState;
	const noAuxiliaryAgentState = prepared.viewWorkspace
		? noAuxiliaryState
		: result !== null && auxiliaryStateFields(result).length === 0 && !(await directoryExists(join(prepared.workspace, "drawing_feedback")));
	const finishedAt = isoNow();
	return {
		fixture_id: prepared.fixtureId,
		status,
		started_at: startedAt,
		finished_at: finishedAt,
		duration_s: elapsedSeconds(startedAt),
		timeout_minutes: timeoutMinutesValue,
		source_workspace: prepared.sourceWorkspace,
		workspace: prepared.workspace,
		continuation_id: prepared.continuationId,
		prompt_sha256: prepared.promptSha256,
		baseline: {
			source_sha256: prepared.baselineSourceSha256,
			step_sha256: prepared.baselineStepSha256,
			workspace_tree_sha256_before: prepared.baselineTreeSha256,
			workspace_tree_sha256_after: sourceTreeAfter,
			render_path: "baseline/iso.png",
			session_file: "baseline/session.jsonl",
			last_turn: prepared.baselineLastTurn,
			...(prepared.job.continuation?.reuse_fixed_view_from_source ? {
				view_origin_source_sha256: prepared.job.continuation.view_origin_source_sha256,
				view_origin_step_sha256: prepared.job.continuation.view_origin_step_sha256,
				previous_continuation_id: prepared.job.continuation.previous_continuation_id,
			} : {}),
			...(prepared.viewWorkspace && prepared.viewTreeSha256 && viewTreeAfter && prepared.manifestPath && prepared.manifestSha256 ? {
				view_search_workspace: prepared.viewWorkspace,
				view_search_tree_sha256_before: prepared.viewTreeSha256,
				view_search_tree_sha256_after: viewTreeAfter,
				locked_view_manifest_path: prepared.manifestPath,
				locked_view_manifest_sha256: prepared.manifestSha256,
			} : {}),
		},
		...(after ? { after } : {}),
		...(result ? {
			agent_status: result.status,
			usage_delta: {
				tokens: Math.max(0, result.tokens.total - prepared.baselineUsage.tokens.total),
				cost: Math.max(0, result.cost - prepared.baselineUsage.cost),
				tool_calls: Math.max(0, result.tool_calls - prepared.baselineUsage.tool_calls),
			},
		} : {}),
		no_auxiliary_agent_state: noAuxiliaryAgentState,
		no_reflection_agent_state: noReflectionAgentState,
		allowed_auxiliary_agent_state: allowedAuxiliaryAgentState,
		...(failure ? { error: failure } : {}),
	};
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
	if (argv.length === 0 || argv[0] === "--help" || argv[0] === "-h") {
		process.stdout.write(HELP);
		return 0;
	}
	const parsed = parseArguments(argv);
	if (parsed.positionals.length === 0) throw new UsageError("At least one source workspace is required");
	const outputDir = resolve(requiredFlag(parsed, "output-dir"));
	const image = requiredFlag(parsed, "image");
	const viewSearchRoot = parsed.flags.get("inspect-view-search-root");
	const resolvedViewSearchRoot = viewSearchRoot ? resolve(viewSearchRoot) : undefined;
	const selectedInspectRegionId = inspectRegionId(parsed);
	const timeoutMinutesValue = timeoutMinutes(parsed);
	const selectedThinkingLevel = thinkingLevel(parsed);
	const allowChainedSelfReflect = parsed.flags.get("allow-chained-self-reflect") === "true";
	const reuseFixedViewFromSource = parsed.flags.get("reuse-fixed-view-from-source") === "true";
	const usesLockedViews = resolvedViewSearchRoot !== undefined || reuseFixedViewFromSource;
	if (reuseFixedViewFromSource && !allowChainedSelfReflect) throw new UsageError("--reuse-fixed-view-from-source requires --allow-chained-self-reflect");
	if (reuseFixedViewFromSource && resolvedViewSearchRoot) throw new UsageError("--reuse-fixed-view-from-source cannot be combined with --inspect-view-search-root");
	if (reuseFixedViewFromSource && selectedInspectRegionId) throw new UsageError("--reuse-fixed-view-from-source preserves all original views and cannot be combined with --inspect-region-id");
	const noNewImages = parsed.flags.get("no-new-images") === "true";
	if (noNewImages && usesLockedViews) throw new UsageError("--no-new-images cannot be combined with --inspect-view-search-root");
	const selectedCausalRepairProfile = causalRepairProfile(parsed);
	const selectedDualRoiCropProfile = dualRoiCropProfile(parsed);
	const selectedTargetAnnotationProfile = targetAnnotationProfile(parsed);
	const selectedVisualFindingProfile = visualFindingProfile(parsed);
	const selectedCodeConstructionProfile = codeConstructionProfile(parsed);
	const selectedVisualDiffProfile = visualDiffProfile(parsed);
	const selectedVisualReviewProfile = visualReviewProfile(parsed);
	const selectedDimensionAnnotationProfile = dimensionAnnotationProfile(parsed);
	if (reuseFixedViewFromSource && selectedDimensionAnnotationProfile !== DIMENSION_ANNOTATION_PROFILE) throw new UsageError("--reuse-fixed-view-from-source requires --dimension-annotation-profile aligned-dimension-subagent-v1");
	if (selectedDimensionAnnotationProfile && !usesLockedViews) {
		throw new UsageError("--dimension-annotation-profile requires --inspect-view-search-root");
	}
	if (selectedDimensionAnnotationProfile && (
		selectedCausalRepairProfile || selectedDualRoiCropProfile || selectedTargetAnnotationProfile ||
		selectedVisualFindingProfile || selectedCodeConstructionProfile || selectedVisualDiffProfile ||
		(selectedVisualReviewProfile && selectedVisualReviewProfile !== SHAPE_FEEDBACK_PROFILE)
	)) {
		throw new UsageError(`${DIMENSION_ANNOTATION_PROFILE} may only be combined with ${SHAPE_FEEDBACK_PROFILE}`);
	}
	if (selectedInspectRegionId && !resolvedViewSearchRoot) {
		throw new UsageError("--inspect-region-id requires --inspect-view-search-root");
	}
	if (selectedCausalRepairProfile && !resolvedViewSearchRoot) {
		throw new UsageError("--causal-repair-profile requires --inspect-view-search-root");
	}
	if (selectedDualRoiCropProfile && !resolvedViewSearchRoot) {
		throw new UsageError("--dual-roi-crop-profile requires --inspect-view-search-root");
	}
	if (selectedTargetAnnotationProfile && !resolvedViewSearchRoot) {
		throw new UsageError("--target-annotation-profile requires --inspect-view-search-root");
	}
	if (selectedVisualFindingProfile && !resolvedViewSearchRoot) {
		throw new UsageError("--visual-finding-profile requires --inspect-view-search-root");
	}
	if (selectedVisualFindingProfile && selectedCausalRepairProfile) {
		throw new UsageError("--visual-finding-profile cannot be combined with --causal-repair-profile");
	}
	if (selectedCodeConstructionProfile && !resolvedViewSearchRoot) {
		throw new UsageError("--code-construction-profile requires --inspect-view-search-root");
	}
	if (selectedVisualDiffProfile && !selectedCodeConstructionProfile) {
		throw new UsageError("--visual-diff-profile requires --code-construction-profile");
	}
	if (selectedVisualDiffProfile && selectedVisualFindingProfile) {
		throw new UsageError("--visual-diff-profile cannot be combined with --visual-finding-profile");
	}
	if (selectedVisualReviewProfile && !usesLockedViews) {
		throw new UsageError("--visual-review-profile requires --inspect-view-search-root");
	}
	if (selectedVisualReviewProfile && (selectedVisualFindingProfile || selectedVisualDiffProfile)) {
		throw new UsageError("--visual-review-profile replaces --visual-finding-profile and --visual-diff-profile");
	}
	if (selectedVisualReviewProfile && selectedCodeConstructionProfile) {
		throw new UsageError("--visual-review-profile replaces --code-construction-profile");
	}
	if (selectedVisualReviewProfile === SHAPE_FEEDBACK_PROFILE &&
		(selectedCausalRepairProfile || selectedDualRoiCropProfile || selectedTargetAnnotationProfile)) {
		throw new UsageError(`${SHAPE_FEEDBACK_PROFILE} cannot be combined with --causal-repair-profile, --dual-roi-crop-profile, or --target-annotation-profile`);
	}
	if (selectedVisualReviewProfile === GEOMETRY_REPAIR_PROFILE &&
		(selectedCausalRepairProfile || selectedDualRoiCropProfile || selectedTargetAnnotationProfile)) {
		throw new UsageError(`${GEOMETRY_REPAIR_PROFILE} cannot be combined with --causal-repair-profile, --dual-roi-crop-profile, or --target-annotation-profile`);
	}
	if (
		selectedCodeConstructionProfile &&
		(selectedCausalRepairProfile || selectedDualRoiCropProfile)
	) {
		throw new UsageError("--code-construction-profile cannot be combined with causal repair or Dual ROI Crop");
	}
	const modelSelection = resolveExperimentModelSelection({
		profile: parsed.flags.get("model-profile") ?? process.env.CADGENBENCH_PI_MODEL_PROFILE,
		provider: parsed.flags.get("provider"),
		model: parsed.flags.get("model"),
	});
	await requireEmptyOutput(outputDir);
	const runDir = join(outputDir, "run");
	await mkdir(runDir, { recursive: true });
	const runId = `${selectedDimensionAnnotationProfile ? (selectedVisualReviewProfile === SHAPE_FEEDBACK_PROFILE ? "main-self-reflect-inspect-shape-feedback-dimensions" : "main-self-reflect-inspect-dimensions") : selectedVisualReviewProfile === GEOMETRY_REPAIR_PROFILE ? "main-self-reflect-inspect-geometry-repair" : selectedVisualReviewProfile === SHAPE_FEEDBACK_PROFILE ? "main-self-reflect-inspect-shape-feedback" : selectedVisualReviewProfile ? "main-self-reflect-inspect-geometry-hypothesis" : selectedVisualDiffProfile ? "main-self-reflect-inspect-visual-diffs" : selectedCodeConstructionProfile ? "main-self-reflect-inspect-code-ablation" : selectedVisualFindingProfile ? "main-self-reflect-inspect-visual-findings" : selectedTargetAnnotationProfile ? "main-self-reflect-inspect-target-annotations" : selectedCausalRepairProfile ? "main-self-reflect-inspect-causal" : selectedDualRoiCropProfile ? "main-self-reflect-inspect-dual-roi" : usesLockedViews ? "main-self-reflect-inspect" : "main-self-reflect"}-${new Date().toISOString().replaceAll(/[-:.]/gu, "")}-${process.pid}`;
	const probeRuntime = new DockerAgentRuntime({ image, provider: modelSelection.provider });
	await probeRuntime.daemonVersion();
	const imageId = await probeRuntime.imageDigest();
	const runtimes = parsed.positionals.map(() => new DockerAgentRuntime({ image, provider: modelSelection.provider }));
	const startedAt = isoNow();
	let interrupted = false;
	const onSigint = (): void => {
		interrupted = true;
		void Promise.allSettled(runtimes.map(async (runtime) => await runtime.interruptAll()));
	};
	process.once("SIGINT", onSigint);
	try {
		const prepared: PreparedCase[] = [];
		for (const [index, sourceWorkspace] of parsed.positionals.entries()) {
			const runtime = runtimes[index];
			if (!runtime) throw new InfrastructureError(`Missing isolated runtime for ${sourceWorkspace}`);
			prepared.push(await prepareCase({
				sourceWorkspace,
				...(resolvedViewSearchRoot ? { viewSearchRoot: resolvedViewSearchRoot } : {}),
				runDir,
				runId,
				image,
				imageId,
				runtime,
				modelSelection,
				thinkingLevel: selectedThinkingLevel,
				allowChainedSelfReflect,
				noNewImages,
				reuseFixedViewFromSource,
				...(selectedInspectRegionId ? { inspectRegionId: selectedInspectRegionId } : {}),
				...(selectedCausalRepairProfile ? { causalRepairProfile: selectedCausalRepairProfile } : {}),
				...(selectedDualRoiCropProfile ? { dualRoiCropProfile: selectedDualRoiCropProfile } : {}),
				...(selectedTargetAnnotationProfile ? { targetAnnotationProfile: selectedTargetAnnotationProfile } : {}),
				...(selectedVisualFindingProfile ? { visualFindingProfile: selectedVisualFindingProfile } : {}),
				...(selectedCodeConstructionProfile ? { codeConstructionProfile: selectedCodeConstructionProfile } : {}),
				...(selectedVisualDiffProfile ? { visualDiffProfile: selectedVisualDiffProfile } : {}),
				...(selectedVisualReviewProfile ? { visualReviewProfile: selectedVisualReviewProfile } : {}),
				...(selectedDimensionAnnotationProfile ? { dimensionAnnotationProfile: selectedDimensionAnnotationProfile } : {}),
			}));
		}
		if (new Set(prepared.map(({ fixtureId }) => fixtureId)).size !== prepared.length) {
			throw new UsageError("Source fixture IDs must be unique");
		}
		await writeJsonAtomic(join(outputDir, "experiment_contract.json"), {
			experiment: usesLockedViews
				? selectedVisualReviewProfile === GEOMETRY_REPAIR_PROFILE
					? "Main Self Reflect with atomic geometry repair delegation, SubAgent-only Preview, and Main-reviewed candidate adoption"
					: selectedVisualReviewProfile === SHAPE_FEEDBACK_PROFILE
					? "Main Self Reflect with final aligned 2D review, optional single-candidate CAD Edit Preview with Target/CAD source overlays, and adoption by candidate ID"
					: selectedVisualReviewProfile
					? "Main Self Reflect with final aligned 2D review and optional isolated executable geometry hypotheses"
					: selectedVisualDiffProfile
					? "Main Self Reflect with final aligned 2D review and persistent image-grounded Visual Links"
					: selectedCodeConstructionProfile
					? selectedVisualFindingProfile
						? "Main Self Reflect with final aligned 2D review, persistent visual findings, and source-bound code-ablation localization"
						: "Main Self Reflect with final aligned 2D review and code-ablation localization"
					: selectedVisualFindingProfile
					? "Main Self Reflect with final aligned 2D review, optional Dual ROI Crop, and persistent visual findings"
					: selectedTargetAnnotationProfile
					? "Main Self Reflect with final aligned 2D review and persistent numeric Target marks"
					: selectedCausalRepairProfile
					? "Main Self Reflect with final aligned 2D review and counterfactual causal repair verification"
					: selectedDualRoiCropProfile
						? "Main Self Reflect with final aligned 2D review and optional Dual ROI Crop"
						: "Main Self Reflect with final aligned 2D review"
				: "Main-Agent-only Self Reflect",
			continuation_kind: usesLockedViews ? MAIN_SELF_REFLECT_INSPECT_KIND : MAIN_SELF_REFLECT_KIND,
			protocol_version: PROTOCOL_VERSION,
			run_id: runId,
			created_at: startedAt,
			model_profile: modelSelection.profile,
			provider: modelSelection.provider,
			model: modelSelection.model,
			thinking_level: selectedThinkingLevel,
			pi_version: PI_VERSION,
			container_image: image,
			container_image_digest: imageId,
			exact_command: ["cadgenbench-pi-main-self-reflect-continue", ...argv],
			fixtures: prepared.map(({ fixtureId }) => fixtureId),
			case_concurrency: prepared.length,
			independent_runtime_per_case: true,
			timeout_minutes_per_case: timeoutMinutesValue,
			turn_limit: null,
			agent_mode: "base-only",
			continuation_image_paths: [],
			historical_self_reflect_prompt_delta: usesLockedViews
				? selectedVisualReviewProfile === GEOMETRY_REPAIR_PROFILE
					? "required final aligned 2D review, formal execute_build123d validity/watertightness checks, and continued repair until clear comparison discrepancies are eliminated; the formal tool's 3D render may also be inspected; delegate one part and one suspected difference through repair_geometry; SubAgent verifies the premise before isolated previews; Main reviews only the selected candidate's final images and adopts using evaluation_id and candidate_id; no Main Preview tool"
					: selectedVisualReviewProfile === SHAPE_FEEDBACK_PROFILE
					? "deleted the no-additional-images/external-analysis sentence; required final aligned 2D review and allowed unchanged output; added inspect_comparison and exactly one candidate per preview_shape_feedback call in a one-entry candidates array with only id and one to eight edits, shown in one to three complete aligned views selected by region_id with one three-panel card per view (Target | Current | Preview, with separate monochrome CAD 2D projections and a shared display scale); added adopt_shape_candidate for the ID from the most recent preview"
					: selectedVisualReviewProfile
					? "deleted the no-additional-images/external-analysis sentence; required final aligned 2D review and allowed unchanged output; added inspect_comparison and isolated Pi-compatible executable geometry hypotheses"
					: selectedCodeConstructionProfile
					? `deleted the no-additional-images/external-analysis sentence; required final aligned 2D review and allowed unchanged output; added inspect_comparison and source-bound code-ablation localization${selectedVisualFindingProfile ? "; added persistent visual findings that reference registered constructions" : ""}${selectedVisualDiffProfile ? "; added one-step same-coordinate visual diffs" : ""}`
					: `deleted the no-additional-images/external-analysis sentence; required final aligned 2D review and allowed unchanged output; added inspect_comparison${selectedCausalRepairProfile ? "; added optional counterfactual Patch verification before formal editing" : ""}${selectedDualRoiCropProfile ? "; added optional dual_roi_crop for local inspection" : ""}${selectedTargetAnnotationProfile ? "; added persistent numeric Point/Box Target marks" : ""}${selectedVisualFindingProfile ? "; added persistent controller-managed visual findings" : ""}`
				: null,
			formal_3d_render_enabled: !noNewImages,
			no_new_images: noNewImages,
			inspect_comparison_optional: false,
			causal_repair_profile: selectedCausalRepairProfile ?? null,
			compare_repair_hypotheses_optional: selectedCausalRepairProfile !== undefined,
			dual_roi_crop_profile: selectedDualRoiCropProfile ?? null,
			dual_roi_crop_optional: selectedDualRoiCropProfile !== undefined,
			target_annotation_profile: selectedTargetAnnotationProfile ?? null,
			persistent_target_annotations_enabled: selectedTargetAnnotationProfile !== undefined,
			visual_finding_profile: selectedVisualFindingProfile ?? null,
			persistent_visual_findings_enabled: selectedVisualFindingProfile !== undefined,
			visual_diff_profile: selectedVisualDiffProfile ?? null,
			persistent_visual_diffs_enabled: selectedVisualDiffProfile !== undefined,
			visual_review_profile: selectedVisualReviewProfile ?? null,
			dimension_annotation_profile: selectedDimensionAnnotationProfile ?? null,
			dimension_annotations_enabled: selectedDimensionAnnotationProfile !== undefined,
			allowed_subagent_roles: [
				...(selectedVisualReviewProfile === GEOMETRY_REPAIR_PROFILE ? ["geometry_repair"] : []),
				...(selectedDimensionAnnotationProfile ? ["dimension_annotation"] : []),
			],
			...(selectedDimensionAnnotationProfile ? {
				dimension_annotation: {
					model: "same_as_main",
					thinking_level: "same_as_main",
					refresh: "before_main_reflection_and_after_successful_formal_execution",
					formal_source_editing: false,
					entity_ids: "scoped_to_current_step_sha256",
				},
			} : {}),
			executable_geometry_hypotheses_enabled: selectedVisualReviewProfile === VISUAL_REVIEW_PROFILE || selectedVisualReviewProfile === REPAIR_PREVIEW_PROFILE,
			shape_feedback_enabled: selectedVisualReviewProfile === SHAPE_FEEDBACK_PROFILE,
			geometry_repair_enabled: selectedVisualReviewProfile === GEOMETRY_REPAIR_PROFILE,
			...(selectedVisualReviewProfile === GEOMETRY_REPAIR_PROFILE ? {
				geometry_repair: {
					max_preview_calls_per_task: null,
					model: "same_as_main",
					thinking_level: "same_as_main",
					intermediate_images_returned_to_main: false,
					allowed_subagent_roles: ["geometry_repair"],
				},
			} : {}),
			code_construction_profile: selectedCodeConstructionProfile ?? null,
			inspect_code_ablation_optional: selectedCodeConstructionProfile !== undefined,
			same_region_before_after_prompt_required: false,
			final_key_view_reinspection_prompt_required: usesLockedViews,
			view_search_root: resolvedViewSearchRoot ?? null,
			...(reuseFixedViewFromSource ? { fixed_view_origins: Object.fromEntries(prepared.map(item => [item.fixtureId, {
				source_workspace: item.sourceWorkspace,
				manifest_sha256: item.job.continuation!.locked_view_manifest_sha256,
				source_sha256: item.job.continuation!.view_origin_source_sha256,
				step_sha256: item.job.continuation!.view_origin_step_sha256,
				previous_continuation_id: item.job.continuation!.previous_continuation_id,
			}])) } : {}),
			inspect_region_id: selectedInspectRegionId ?? null,
			allow_chained_self_reflect: allowChainedSelfReflect,
			reuse_fixed_view_from_source: reuseFixedViewFromSource,
			source_continuation_kinds: Object.fromEntries(
				prepared.map(({ fixtureId, sourceContinuationKind }) => [fixtureId, sourceContinuationKind]),
			),
			allowed_tools: [
				"read", "write", "edit", "bash", "grep", "find", "ls", "execute_build123d",
				...(usesLockedViews ? ["inspect_comparison"] : []),
				...(selectedDualRoiCropProfile ? ["dual_roi_crop"] : []),
				...(selectedTargetAnnotationProfile ? ["annotate_target"] : []),
				...(selectedCausalRepairProfile ? ["compare_repair_hypotheses"] : []),
				...(selectedVisualFindingProfile ? ["update_visual_findings"] : []),
				...(selectedCodeConstructionProfile ? ["inspect_code_ablation"] : []),
				...(selectedVisualDiffProfile ? ["visual_diff"] : []),
				...visualReviewAgentTools(selectedVisualReviewProfile),
			],
			forbidden_tools: usesLockedViews
				? [
						...(selectedDualRoiCropProfile ? [] : ["dual_roi_crop"]),
						...(selectedTargetAnnotationProfile ? [] : ["annotate_target"]),
						...(selectedCausalRepairProfile ? [] : ["compare_repair_hypotheses"]),
						...(selectedVisualFindingProfile ? [] : ["update_visual_findings"]),
						...(selectedCodeConstructionProfile ? [] : ["inspect_code_ablation"]),
						...(selectedVisualDiffProfile ? [] : ["visual_diff"]),
						...(selectedVisualReviewProfile === VISUAL_REVIEW_PROFILE ? [] : ["visualize_code_hypothesis"]),
						...(selectedVisualReviewProfile === REPAIR_PREVIEW_PROFILE ? [] : ["preview_repair_candidates"]),
						...(selectedVisualReviewProfile === SHAPE_FEEDBACK_PROFILE ? [] : ["preview_shape_feedback"]),
						...(selectedVisualReviewProfile === SHAPE_FEEDBACK_PROFILE || selectedVisualReviewProfile === GEOMETRY_REPAIR_PROFILE ? [] : ["adopt_shape_candidate"]),
						...(selectedVisualReviewProfile === GEOMETRY_REPAIR_PROFILE ? [] : ["repair_geometry"]),
						"self_reflect_2d_drawing", "view_search", "submit_reflection",
					]
				: ["inspect_comparison", "dual_roi_crop", "annotate_target", "inspect_code_ablation", "visual_diff", "visualize_code_hypothesis", "preview_repair_candidates", ...SHAPE_FEEDBACK_TOOLS, "repair_geometry", "self_reflect_2d_drawing", "view_search", "submit_reflection"],
			acceptance: {
				fresh_formal_execution: !usesLockedViews && !noNewImages,
				current_formal_execution: true,
				source_changed: !usesLockedViews && !noNewImages,
				step_changed: !usesLockedViews && !noNewImages,
				is_valid: true,
				is_watertight: true,
				no_auxiliary_agents: selectedVisualReviewProfile !== GEOMETRY_REPAIR_PROFILE && !selectedDimensionAnnotationProfile,
				allowed_auxiliary_agent_roles: [
					...(selectedVisualReviewProfile === GEOMETRY_REPAIR_PROFILE ? ["geometry_repair"] : []),
					...(selectedDimensionAnnotationProfile ? ["dimension_annotation"] : []),
				],
				final_dimension_annotation_required: selectedDimensionAnnotationProfile !== undefined,
				inspect_comparison_required: usesLockedViews,
				final_step_inspection_required: usesLockedViews,
			},
		});
		await Promise.all(prepared.map(async (item) => {
			await item.runtime.prepareFixture(item.workspace, runId, item.fixtureId, imageId);
		}));
		const settled = await Promise.allSettled(prepared.map(async (item) =>
			await runCase(item, imageId, timeoutMinutesValue, () => interrupted)));
		const cases = settled.map((outcome, index): SelfReflectCaseResult => {
			if (outcome.status === "fulfilled") return outcome.value;
			const item = prepared[index];
			if (!item) throw outcome.reason;
			const now = isoNow();
			return {
				fixture_id: item.fixtureId,
				status: interrupted ? "interrupted" : "error",
				started_at: startedAt,
				finished_at: now,
				duration_s: elapsedSeconds(startedAt),
				timeout_minutes: timeoutMinutesValue,
				source_workspace: item.sourceWorkspace,
				workspace: item.workspace,
				continuation_id: item.continuationId,
				prompt_sha256: item.promptSha256,
				baseline: {
					source_sha256: item.baselineSourceSha256,
					step_sha256: item.baselineStepSha256,
					workspace_tree_sha256_before: item.baselineTreeSha256,
					workspace_tree_sha256_after: item.baselineTreeSha256,
					render_path: "baseline/iso.png",
					session_file: "baseline/session.jsonl",
					last_turn: item.baselineLastTurn,
					...(item.viewWorkspace && item.viewTreeSha256 && item.manifestPath && item.manifestSha256 ? {
						view_search_workspace: item.viewWorkspace,
						view_search_tree_sha256_before: item.viewTreeSha256,
						view_search_tree_sha256_after: item.viewTreeSha256,
						locked_view_manifest_path: item.manifestPath,
						locked_view_manifest_sha256: item.manifestSha256,
					} : {}),
				},
				no_auxiliary_agent_state: false,
				no_reflection_agent_state: false,
				error: errorMessage(outcome.reason),
			};
		});
		const finishedAt = isoNow();
		const summary = {
			run_id: runId,
			model_profile: modelSelection.profile,
			provider: modelSelection.provider,
			model: modelSelection.model,
			thinking_level: selectedThinkingLevel,
			status: interrupted ? "interrupted" : cases.every(({ status }) => status === "completed") ? "completed" : "partial",
			started_at: startedAt,
			finished_at: finishedAt,
			duration_s: elapsedSeconds(startedAt),
			case_concurrency: prepared.length,
			independent_runtime_per_case: true,
			timeout_minutes_per_case: timeoutMinutesValue,
			cases,
		};
		await writeJsonAtomic(join(outputDir, "continuation_summary.json"), summary);
		for (const result of cases) {
			await writeJsonAtomic(join(runDir, result.fixture_id, "continuation_result.json"), result);
		}
		process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
		return summary.status === "completed" ? 0 : interrupted ? 130 : 2;
	} finally {
		process.removeListener("SIGINT", onSigint);
	}
}

if (
	process.argv[1] &&
	await realpath(process.argv[1]).catch(() => null) === await realpath(fileURLToPath(import.meta.url))
) {
	main()
		.then((code) => {
			process.exitCode = code;
		})
		.catch((error: unknown) => {
			const prefix = error instanceof UsageError ? "usage error" : "infrastructure error";
			process.stderr.write(`${prefix}: ${errorMessage(error)}\n`);
			process.exitCode = error instanceof UsageError ? 64 : 2;
		});
}
