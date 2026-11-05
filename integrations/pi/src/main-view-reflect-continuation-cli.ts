#!/usr/bin/env node
import { createHash } from "node:crypto";
import { cp, lstat, mkdir, readFile, readdir, realpath, stat } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { DockerAgentRuntime } from "./docker-worker.js";
import { errorMessage, InfrastructureError, UsageError } from "./errors.js";
import {
	type ExperimentModelSelection,
	resolveExperimentModelSelection,
} from "./experiment-model-profile.js";
import { elapsedSeconds, isoNow, readJson, writeJsonAtomic } from "./files.js";
import { CaseTimeoutError, withIsolatedTimeout } from "./main-self-reflect.js";
import {
	cameraSha256,
	loadLockedMainViewManifest,
	MAIN_VIEW_REFLECT_KIND,
	mainViewReflectContinuationPrompt,
	summaryHasCroppedStep,
	summaryHasInspectedStep,
} from "./main-view-reflect.js";
import {
	BASE_AGENT_TOOLS,
	JOB_FILE,
	MAIN_VIEW_REFLECT_AGENT_TOOLS,
	PI_VERSION,
	PROTOCOL_VERSION,
	type AgentJob,
	type AgentResult,
	type DrawingViewManifest,
	type RunStatus,
} from "./types.js";

interface ParsedArguments {
	positionals: string[];
	flags: Map<string, string>;
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

interface PreparedCase {
	fixtureId: string;
	workspace: string;
	sourceWorkspace: string;
	viewWorkspace: string;
	continuationId: string;
	promptSha256: string;
	baselineSourceSha256: string;
	baselineStepSha256: string;
	sourceTreeSha256: string;
	viewTreeSha256: string;
	manifestPath: string;
	manifestSha256: string;
	lockedAssets: LockedAsset[];
	baselineLastTurn: number;
	baselineUsage: Pick<AgentResult, "tokens" | "cost" | "tool_calls">;
	job: AgentJob;
	runtime: DockerAgentRuntime;
}

interface MainViewReflectAfter {
	source_sha256: string;
	step_sha256: string;
	source_changed: boolean;
	step_changed: boolean;
	render_path: string;
	session_file: string;
	last_turn: number;
	formal_executions_added: number;
	is_valid: boolean;
	is_watertight: boolean;
	audit_manifest_path: string;
	inspection_count: number;
	crop_count: number;
}

type CaseStatus = "completed" | "timeout" | "error" | "interrupted";

interface CaseResult {
	fixture_id: string;
	status: CaseStatus;
	started_at: string;
	finished_at: string;
	duration_s: number;
	timeout_minutes: number;
	source_workspace: string;
	view_search_workspace: string;
	workspace: string;
	continuation_id: string;
	prompt_sha256: string;
	baseline: {
		source_sha256: string;
		step_sha256: string;
		source_tree_sha256_before: string;
		source_tree_sha256_after: string;
		view_search_tree_sha256_before: string;
		view_search_tree_sha256_after: string;
		locked_view_manifest_path: string;
		locked_view_manifest_sha256: string;
		render_path: string;
		session_file: string;
		last_turn: number;
	};
	after?: MainViewReflectAfter;
	agent_status?: RunStatus;
	usage_delta?: { tokens: number; cost: number; tool_calls: number };
	no_reflection_agent_state: boolean;
	error?: string;
}

const DEFAULT_TIMEOUT_MINUTES = 120;
const EXPECTED_THINKING = "high";

const HELP = `cadgenbench-pi-main-view-reflect-continue - resume Direct Main Agent sessions with locked 2D views

Usage:
  cadgenbench-pi-main-view-reflect-continue <direct-workspace>... \\
    --view-search-root <completed-view-search-root> \\
    --output-dir <new-run-root> \\
    --image <container-tag> \\
    [--timeout-minutes 120] \\
    [--model-profile <gpt-5.6-sol|kimi> | --provider <provider> --model <model>]

Each Direct workspace must be an original base-only run. Its fixture ID selects
<view-search-root>/<fixture-id>. The View Search manifest and referenced assets
are hash-locked and copied without Reflection Agent reports. Cases run in
independent containers. A no-op is accepted only after the final formal STEP is
inspected under at least one locked camera.
`;

function parseArguments(args: string[]): ParsedArguments {
	const positionals: string[] = [];
	const flags = new Map<string, string>();
	const allowed = new Set([
		"view-search-root",
		"output-dir",
		"image",
		"timeout-minutes",
		"model-profile",
		"provider",
		"model",
	]);
	for (let index = 0; index < args.length; index += 1) {
		const token = args[index];
		if (token === undefined) continue;
		if (!token.startsWith("--")) {
			positionals.push(token);
			continue;
		}
		const name = token.slice(2);
		if (!allowed.has(name)) throw new UsageError(`Unknown option --${name}`);
		if (flags.has(name)) throw new UsageError(`Option --${name} was provided more than once`);
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
	if (!Number.isInteger(value) || value < 1 || value > DEFAULT_TIMEOUT_MINUTES) {
		throw new UsageError(`--timeout-minutes must be an integer from 1 to ${DEFAULT_TIMEOUT_MINUTES}`);
	}
	return value;
}

async function sha256(path: string): Promise<string> {
	return createHash("sha256").update(await readFile(path)).digest("hex");
}

export async function treeSha256(root: string): Promise<string> {
	const hash = createHash("sha256");
	const visit = async (directory: string): Promise<void> => {
		const entries = await readdir(directory, { withFileTypes: true });
		entries.sort((left, right) => left.name.localeCompare(right.name));
		for (const entry of entries) {
			const path = join(directory, entry.name);
			const name = relative(root, path);
			const metadata = await lstat(path);
			if (metadata.isSymbolicLink()) throw new InfrastructureError(`Baseline tree contains a symlink: ${path}`);
			hash.update(`${entry.isDirectory() ? "d" : "f"}\0${name}\0`);
			if (entry.isDirectory()) await visit(path);
			else if (entry.isFile()) hash.update(await readFile(path));
			else throw new InfrastructureError(`Baseline tree contains an unsupported entry: ${path}`);
		}
	};
	await visit(root);
	return hash.digest("hex");
}

async function requireFile(path: string, label: string): Promise<void> {
	const metadata = await lstat(path).catch(() => null);
	if (!metadata?.isFile() || metadata.isSymbolicLink()) {
		throw new InfrastructureError(`${label} is missing, non-regular, or a symbolic link: ${path}`);
	}
}

async function fileExists(path: string): Promise<boolean> {
	return (await stat(path).catch(() => null))?.isFile() === true;
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

async function copyLockedFile(
	sourceRoot: string,
	destinationRoot: string,
	path: string,
): Promise<LockedAsset> {
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

async function copyBaselineRender(sourceWorkspace: string, baselineDir: string, lastTurn: number): Promise<string> {
	const lastIso = join(sourceWorkspace, "turns", String(lastTurn).padStart(4, "0"), "iso.png");
	await requireFile(lastIso, "Last Direct turn render");
	await cp(lastIso, join(baselineDir, "iso.png"));
	return "baseline/iso.png";
}

function assertDirectResult(result: AgentResult, fixtureId: string): void {
	const forbidden = [
		result.drawing_feedback,
		result.main_view_reflection,
		result.subagent_usage,
		result.subagent_runs,
		result.recorded_subagent_usage_ids,
	].some((value) => value !== undefined);
	if (forbidden) throw new InfrastructureError(`Case ${fixtureId} Direct baseline contains auxiliary Agent state`);
}

export async function prepareMainViewReflectCase(options: {
	sourceWorkspace: string;
	viewSearchRoot: string;
	runDir: string;
	runId: string;
	image: string;
	imageId: string;
	runtime: DockerAgentRuntime;
	modelSelection: ExperimentModelSelection;
}): Promise<PreparedCase> {
	const sourceWorkspace = resolve(options.sourceWorkspace);
	const sourceResultPath = join(sourceWorkspace, "agent_result.json");
	const sourceJobPath = join(sourceWorkspace, JOB_FILE);
	await Promise.all([
		requireFile(sourceResultPath, "Direct Agent result"),
		requireFile(sourceJobPath, "Direct Agent job"),
		requireFile(join(sourceWorkspace, "solution.py"), "Direct solution.py"),
		requireFile(join(sourceWorkspace, "output.step"), "Direct output.step"),
	]);
	const [sourceResult, sourceJob] = await Promise.all([
		readJson<AgentResult>(sourceResultPath),
		readJson<AgentJob>(sourceJobPath),
	]);
	const fixtureId = sourceResult.fixture_id;
	if (
		sourceResult.status !== "valid" ||
		sourceJob.operation !== "run_fixture" ||
		sourceJob.fixture.id !== fixtureId ||
		sourceJob.continuation !== undefined ||
		sourceResult.applied_continuation !== undefined ||
		sourceResult.agent_mode !== "base-only" ||
		sourceJob.agent_mode !== "base-only"
	) throw new InfrastructureError(`Case ${fixtureId} is not an original valid base-only Direct run`);
	assertDirectResult(sourceResult, fixtureId);
	if (
		sourceJob.provider !== options.modelSelection.provider ||
		sourceJob.model !== options.modelSelection.model ||
		sourceJob.thinking_level !== EXPECTED_THINKING ||
		sourceResult.provider !== options.modelSelection.provider ||
		sourceResult.model !== options.modelSelection.model ||
		sourceResult.thinking_level !== EXPECTED_THINKING ||
		sourceJob.pi_version !== PI_VERSION ||
		sourceResult.pi_version !== PI_VERSION
	) throw new InfrastructureError(`Case ${fixtureId} does not match the selected model, thinking level, and Pi version`);
	if (!sourceResult.session_file) throw new InfrastructureError(`Case ${fixtureId} has no resumable Main Agent session`);
	await requireFile(join(sourceWorkspace, sourceResult.session_file), `Case ${fixtureId} Main Agent session`);

	const baselineSourceSha256 = await sha256(join(sourceWorkspace, "solution.py"));
	const baselineStepSha256 = await sha256(join(sourceWorkspace, "output.step"));
	if (
		baselineSourceSha256 !== sourceResult.last_executed_source_sha256 ||
		baselineStepSha256 !== sourceResult.last_executed_step_sha256
	) throw new InfrastructureError(`Case ${fixtureId} Direct source/STEP do not match the recorded formal result`);

	const viewWorkspace = resolve(options.viewSearchRoot, fixtureId);
	const viewResultPath = join(viewWorkspace, "view_search_only_result.json");
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
	) throw new InfrastructureError(`Case ${fixtureId} View Search result does not exactly match the Direct baseline`);
	if (
		await sha256(join(viewWorkspace, "solution.py")) !== baselineSourceSha256 ||
		await sha256(join(viewWorkspace, "output.step")) !== baselineStepSha256
	) throw new InfrastructureError(`Case ${fixtureId} View Search workspace source/STEP mismatch`);
	const manifestPath = viewResult.manifest_path;
	const manifestFile = await safeSourceFile(viewWorkspace, manifestPath, `Case ${fixtureId} View Search manifest`);
	const manifestSha256 = await sha256(manifestFile);
	const manifest = await loadLockedMainViewManifest({
		workspace: viewWorkspace,
		manifestPath,
		manifestSha256,
		baselineSourceSha256,
		baselineStepSha256,
	});

	const [sourceTreeSha256, viewTreeSha256] = await Promise.all([
		treeSha256(sourceWorkspace),
		treeSha256(viewWorkspace),
	]);
	const workspace = join(options.runDir, fixtureId);
	await copySourceWorkspace(sourceWorkspace, workspace);
	const lockedPaths = new Set<string>([manifestPath]);
	for (const region of manifest.regions) {
		lockedPaths.add(region.crop_path);
		if (region.selected) {
			lockedPaths.add(region.selected.render_path);
			lockedPaths.add(region.selected.aligned_render_path);
			lockedPaths.add(region.selected.comparison_path);
		}
	}
	const lockedAssets: LockedAsset[] = [];
	for (const path of [...lockedPaths].sort()) {
		lockedAssets.push(await copyLockedFile(viewWorkspace, workspace, path));
	}
	await loadLockedMainViewManifest({
		workspace,
		manifestPath,
		manifestSha256,
		baselineSourceSha256,
		baselineStepSha256,
	});

	const baselineDir = join(workspace, "baseline");
	await mkdir(baselineDir, { recursive: true });
	await Promise.all([
		cp(join(sourceWorkspace, "solution.py"), join(baselineDir, "solution.py")),
		cp(join(sourceWorkspace, "output.step"), join(baselineDir, "output.step")),
		cp(sourceResultPath, join(baselineDir, "agent_result.json")),
		cp(join(sourceWorkspace, sourceResult.session_file), join(baselineDir, "session.jsonl")),
		cp(viewResultPath, join(baselineDir, "view_search_only_result.json")),
	]);
	const baselineRenderPath = await copyBaselineRender(sourceWorkspace, baselineDir, sourceResult.last_turn);
	const prompt = mainViewReflectContinuationPrompt(fixtureId, manifest);
	const promptSha256 = createHash("sha256").update(prompt).digest("hex");
	const continuationId = `main-view-reflect-${fixtureId}-${promptSha256.slice(0, 16)}`;
	await writeJsonAtomic(join(baselineDir, "manifest.json"), {
		fixture_id: fixtureId,
		created_at: isoNow(),
		direct_workspace: sourceWorkspace,
		view_search_workspace: viewWorkspace,
		direct_workspace_tree_sha256: sourceTreeSha256,
		view_search_workspace_tree_sha256: viewTreeSha256,
		session_file: sourceResult.session_file,
		source_sha256: baselineSourceSha256,
		step_sha256: baselineStepSha256,
		render_path: baselineRenderPath,
		last_turn: sourceResult.last_turn,
		locked_view_manifest_path: manifestPath,
		locked_view_manifest_sha256: manifestSha256,
		locked_assets: lockedAssets,
		continuation_kind: MAIN_VIEW_REFLECT_KIND,
		continuation_id: continuationId,
		prompt_sha256: promptSha256,
		provider: options.modelSelection.provider,
		model: options.modelSelection.model,
		thinking_level: EXPECTED_THINKING,
	});

	const now = isoNow();
	const copiedResult: AgentResult = {
		...sourceResult,
		status: "valid",
		provider: options.modelSelection.provider,
		model: options.modelSelection.model,
		thinking_level: EXPECTED_THINKING,
		container_image: options.image,
		container_image_digest: options.imageId,
		started_at: now,
		updated_at: now,
		duration_s: 0,
	};
	delete copiedResult.stop_reason;
	delete copiedResult.resume_reason;
	delete copiedResult.applied_continuation;
	delete copiedResult.main_view_reflection;
	await writeJsonAtomic(join(workspace, "agent_result.json"), copiedResult);

	const job: AgentJob = {
		...sourceJob,
		protocol_version: PROTOCOL_VERSION,
		run_id: options.runId,
		agent_mode: "base-only",
		provider: options.modelSelection.provider,
		model: options.modelSelection.model,
		thinking_level: EXPECTED_THINKING,
		pi_version: PI_VERSION,
		container_image: options.image,
		container_image_digest: options.imageId,
		continuation: {
			kind: MAIN_VIEW_REFLECT_KIND,
			id: continuationId,
			prompt,
			image_paths: [],
			baseline_source_sha256: baselineSourceSha256,
			baseline_step_sha256: baselineStepSha256,
			locked_view_manifest_path: manifestPath,
			locked_view_manifest_sha256: manifestSha256,
		},
	};
	await writeJsonAtomic(join(workspace, JOB_FILE), job);
	return {
		fixtureId,
		workspace,
		sourceWorkspace,
		viewWorkspace,
		continuationId,
		promptSha256,
		baselineSourceSha256,
		baselineStepSha256,
		sourceTreeSha256,
		viewTreeSha256,
		manifestPath,
		manifestSha256,
		lockedAssets,
		baselineLastTurn: sourceResult.last_turn,
		baselineUsage: { tokens: sourceResult.tokens, cost: sourceResult.cost, tool_calls: sourceResult.tool_calls },
		job,
		runtime: options.runtime,
	};
}

async function assertNoReflectionAgentState(workspace: string, result: AgentResult): Promise<void> {
	if (
		result.drawing_feedback !== undefined ||
		result.subagent_usage !== undefined ||
		result.subagent_runs !== undefined ||
		result.recorded_subagent_usage_ids !== undefined
	) throw new InfrastructureError("Reflection/subagent state is forbidden in Main View Reflect");
	const visit = async (directory: string): Promise<void> => {
		for (const entry of await readdir(directory, { withFileTypes: true })) {
			const path = join(directory, entry.name);
			const rel = relative(workspace, path);
			if (entry.isDirectory()) {
				if (/(^|\/)sessions\/reflection$/u.test(rel)) {
					throw new InfrastructureError(`Reflection Agent session is forbidden: ${rel}`);
				}
				await visit(path);
			} else if (/(^|\/)(reflection|reflection_report|report)\.json$/u.test(rel)) {
				throw new InfrastructureError(`Reflection report is forbidden: ${rel}`);
			}
		}
	};
	await visit(workspace);
}

async function validateCameraAudit(
	workspace: string,
	manifest: DrawingViewManifest,
	result: AgentResult,
): Promise<void> {
	const auditPath = result.main_view_reflection?.audit_manifest_path;
	if (!auditPath) throw new InfrastructureError("Main View Reflect audit manifest is missing");
	const audit = await readJson<{
		locked_view_manifest_sha256?: string;
		snapshots?: Array<{ region_id?: string; camera_sha256?: string }>;
	}>(await safeSourceFile(workspace, auditPath, "Main View Reflect audit manifest"));
	const cameras = new Map(
		manifest.regions.flatMap((region) => region.selected
			? [[region.region_id, cameraSha256(region.selected.view_spec)] as const]
			: []),
	);
	if (audit.locked_view_manifest_sha256 !== result.main_view_reflection?.locked_view_manifest_sha256) {
		throw new InfrastructureError("Audit manifest does not preserve the locked View Search hash");
	}
	for (const snapshot of audit.snapshots ?? []) {
		if (!snapshot.region_id || snapshot.camera_sha256 !== cameras.get(snapshot.region_id)) {
			throw new InfrastructureError("Final audit contains a camera outside the locked View Search manifest");
		}
	}
}

async function captureAfter(prepared: PreparedCase): Promise<MainViewReflectAfter> {
	const result = await readJson<AgentResult>(join(prepared.workspace, "agent_result.json"));
	await assertNoReflectionAgentState(prepared.workspace, result);
	await verifyLockedAssets(prepared.workspace, prepared.lockedAssets);
	if (!result.session_file) throw new InfrastructureError(`Case ${prepared.fixtureId} lost its Main Agent session`);
	const sourceSha256 = await sha256(join(prepared.workspace, "solution.py"));
	const stepSha256 = await sha256(join(prepared.workspace, "output.step"));
	const sourceChanged = sourceSha256 !== prepared.baselineSourceSha256;
	const stepChanged = stepSha256 !== prepared.baselineStepSha256;
	if (stepChanged && !sourceChanged) {
		throw new InfrastructureError(`Case ${prepared.fixtureId} changed output.step without changing and formally executing solution.py`);
	}
	if (!summaryHasInspectedStep(result.main_view_reflection, stepSha256)) {
		throw new InfrastructureError(`Case ${prepared.fixtureId} final STEP was not inspected`);
	}
	if (!summaryHasCroppedStep(result.main_view_reflection, stepSha256)) {
		throw new InfrastructureError(`Case ${prepared.fixtureId} final STEP has no verified local comparison`);
	}
	if (
		result.main_view_reflection?.locked_view_manifest_sha256 !== prepared.manifestSha256 ||
		result.main_view_reflection.baseline_source_sha256 !== prepared.baselineSourceSha256 ||
		result.main_view_reflection.baseline_step_sha256 !== prepared.baselineStepSha256
	) throw new InfrastructureError(`Case ${prepared.fixtureId} Main View Reflect summary does not match its locked baseline`);
	const manifest = await loadLockedMainViewManifest({
		workspace: prepared.workspace,
		manifestPath: prepared.manifestPath,
		manifestSha256: prepared.manifestSha256,
		baselineSourceSha256: prepared.baselineSourceSha256,
		baselineStepSha256: prepared.baselineStepSha256,
	});
	await validateCameraAudit(prepared.workspace, manifest, result);
	const afterDir = join(prepared.workspace, "after");
	await mkdir(afterDir, { recursive: true });
	const isoPath = join(prepared.workspace, "turns", String(result.last_turn).padStart(4, "0"), "iso.png");
	await requireFile(isoPath, `Case ${prepared.fixtureId} final formal render`);
	await Promise.all([
		cp(join(prepared.workspace, "solution.py"), join(afterDir, "solution.py")),
		cp(join(prepared.workspace, "output.step"), join(afterDir, "output.step")),
		cp(join(prepared.workspace, "agent_result.json"), join(afterDir, "agent_result.json")),
		cp(join(prepared.workspace, result.session_file), join(afterDir, "session.jsonl")),
		cp(isoPath, join(afterDir, "iso.png")),
		cp(join(prepared.workspace, result.main_view_reflection!.audit_manifest_path), join(afterDir, "main_view_reflect_audit.json")),
	]);
	return {
		source_sha256: sourceSha256,
		step_sha256: stepSha256,
		source_changed: sourceChanged,
		step_changed: stepChanged,
		render_path: "after/iso.png",
		session_file: result.session_file,
		last_turn: result.last_turn,
		formal_executions_added: result.last_turn - prepared.baselineLastTurn,
		is_valid: result.artifact?.validation?.is_valid === true,
		is_watertight: result.artifact?.validation?.is_watertight === true,
		audit_manifest_path: "after/main_view_reflect_audit.json",
		inspection_count: result.main_view_reflection!.inspection_count,
		crop_count: result.main_view_reflection!.crop_count,
	};
}

async function capturePartial(prepared: PreparedCase): Promise<void> {
	const partialDir = join(prepared.workspace, "partial");
	for (const name of ["solution.py", "output.step", "agent_result.json"] as const) {
		const source = join(prepared.workspace, name);
		if (await fileExists(source)) {
			await mkdir(partialDir, { recursive: true });
			await cp(source, join(partialDir, name));
		}
	}
	const result = await readJson<AgentResult>(join(prepared.workspace, "agent_result.json")).catch(() => null);
	if (result?.session_file && await fileExists(join(prepared.workspace, result.session_file))) {
		await mkdir(partialDir, { recursive: true });
		await cp(join(prepared.workspace, result.session_file), join(partialDir, "session.jsonl"));
	}
}

async function runCase(
	prepared: PreparedCase,
	imageId: string,
	timeoutMinutesValue: number,
	isInterrupted: () => boolean,
): Promise<CaseResult> {
	const startedAt = isoNow();
	let status: CaseStatus = "error";
	let after: MainViewReflectAfter | undefined;
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
		if (
			result.status !== "valid" ||
			result.agent_mode !== "base-only" ||
			result.applied_continuation?.id !== prepared.continuationId ||
			result.applied_continuation.prompt_sha256 !== prepared.promptSha256 ||
			!after.is_valid ||
			!after.is_watertight ||
			after.inspection_count < 1 ||
			after.crop_count < 1 ||
			(after.source_changed && after.formal_executions_added < 1)
		) throw new InfrastructureError(`Case ${prepared.fixtureId} failed the final validity/inspection gate`);
		status = "completed";
	} catch (error) {
		status = error instanceof CaseTimeoutError ? "timeout" : isInterrupted() ? "interrupted" : "error";
		failure = errorMessage(error);
		await capturePartial(prepared).catch(() => undefined);
		result = await readJson<AgentResult>(join(prepared.workspace, "agent_result.json")).catch(() => null);
	}
	const [sourceTreeAfter, viewTreeAfter] = await Promise.all([
		treeSha256(prepared.sourceWorkspace),
		treeSha256(prepared.viewWorkspace),
	]);
	if (sourceTreeAfter !== prepared.sourceTreeSha256 || viewTreeAfter !== prepared.viewTreeSha256) {
		status = "error";
		failure = "Original Direct or View Search baseline tree changed during the run";
	}
	const noReflectionAgentState = result !== null && await assertNoReflectionAgentState(prepared.workspace, result)
		.then(() => true, () => false);
	if (!noReflectionAgentState && status === "completed") {
		status = "error";
		failure = "Reflection Agent state/report appeared in the Main Agent experiment";
	}
	const finishedAt = isoNow();
	return {
		fixture_id: prepared.fixtureId,
		status,
		started_at: startedAt,
		finished_at: finishedAt,
		duration_s: elapsedSeconds(startedAt),
		timeout_minutes: timeoutMinutesValue,
		source_workspace: prepared.sourceWorkspace,
		view_search_workspace: prepared.viewWorkspace,
		workspace: prepared.workspace,
		continuation_id: prepared.continuationId,
		prompt_sha256: prepared.promptSha256,
		baseline: {
			source_sha256: prepared.baselineSourceSha256,
			step_sha256: prepared.baselineStepSha256,
			source_tree_sha256_before: prepared.sourceTreeSha256,
			source_tree_sha256_after: sourceTreeAfter,
			view_search_tree_sha256_before: prepared.viewTreeSha256,
			view_search_tree_sha256_after: viewTreeAfter,
			locked_view_manifest_path: prepared.manifestPath,
			locked_view_manifest_sha256: prepared.manifestSha256,
			render_path: "baseline/iso.png",
			session_file: "baseline/session.jsonl",
			last_turn: prepared.baselineLastTurn,
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
		no_reflection_agent_state: noReflectionAgentState,
		...(failure ? { error: failure } : {}),
	};
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
	if (argv.length === 0 || argv[0] === "--help" || argv[0] === "-h") {
		process.stdout.write(HELP);
		return 0;
	}
	const parsed = parseArguments(argv);
	if (parsed.positionals.length === 0) throw new UsageError("At least one Direct workspace is required");
	const viewSearchRoot = resolve(requiredFlag(parsed, "view-search-root"));
	const outputDir = resolve(requiredFlag(parsed, "output-dir"));
	const image = requiredFlag(parsed, "image");
	const timeoutMinutesValue = timeoutMinutes(parsed);
	const modelSelection = resolveExperimentModelSelection({
		profile: parsed.flags.get("model-profile") ?? process.env.CADGENBENCH_PI_MODEL_PROFILE,
		provider: parsed.flags.get("provider"),
		model: parsed.flags.get("model"),
	});
	await requireEmptyOutput(outputDir);
	const runDir = join(outputDir, "run");
	await mkdir(runDir, { recursive: true });
	const runId = `main-view-reflect-${new Date().toISOString().replaceAll(/[-:.]/gu, "")}-${process.pid}`;
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
			prepared.push(await prepareMainViewReflectCase({
				sourceWorkspace,
				viewSearchRoot,
				runDir,
				runId,
				image,
				imageId,
				runtime,
				modelSelection,
			}));
		}
		if (new Set(prepared.map(({ fixtureId }) => fixtureId)).size !== prepared.length) {
			throw new UsageError("Direct fixture IDs must be unique");
		}
		await writeJsonAtomic(join(outputDir, "experiment_contract.json"), {
			experiment: "E-main-view-reflect",
			continuation_kind: MAIN_VIEW_REFLECT_KIND,
			protocol_version: PROTOCOL_VERSION,
			run_id: runId,
			created_at: startedAt,
			exact_argv: process.argv,
			model_profile: modelSelection.profile,
			provider: modelSelection.provider,
			model: modelSelection.model,
			thinking_level: EXPECTED_THINKING,
			pi_version: PI_VERSION,
			container_image: image,
			container_image_digest: imageId,
			fixtures: prepared.map(({ fixtureId }) => fixtureId),
			case_concurrency: prepared.length,
			independent_runtime_per_case: true,
			timeout_minutes_per_case: timeoutMinutesValue,
			turn_limit: null,
			agent_mode: "base-only",
			allowed_tools: [...BASE_AGENT_TOOLS, ...MAIN_VIEW_REFLECT_AGENT_TOOLS],
			forbidden_tools: ["self_reflect_2d_drawing", "view_search", "submit_reflection", "reflection_report"],
			locked_manifests: Object.fromEntries(prepared.map((item) => [item.fixtureId, {
				path: item.manifestPath,
				sha256: item.manifestSha256,
				direct_source_sha256: item.baselineSourceSha256,
				direct_step_sha256: item.baselineStepSha256,
			}])),
			acceptance: {
				final_formal_step_complete_comparison_inspected: true,
				final_formal_step_local_comparison_inspected: true,
				visual_evidence_invalidated_only_when_step_changes: true,
				standalone_3d_render_visible_to_agent: false,
				is_valid: true,
				is_watertight: true,
				no_op_allowed: true,
				if_source_changed_fresh_formal_execution_required: true,
				camera_fingerprint_unchanged: true,
				no_reflection_agent_or_report: true,
			},
		});
		await Promise.all(prepared.map(async (item) => {
			await item.runtime.prepareFixture(item.workspace, runId, item.fixtureId, imageId);
		}));
		const settled = await Promise.allSettled(prepared.map(async (item) =>
			await runCase(item, imageId, timeoutMinutesValue, () => interrupted)));
		const cases = settled.map((outcome, index): CaseResult => {
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
				view_search_workspace: item.viewWorkspace,
				workspace: item.workspace,
				continuation_id: item.continuationId,
				prompt_sha256: item.promptSha256,
				baseline: {
					source_sha256: item.baselineSourceSha256,
					step_sha256: item.baselineStepSha256,
					source_tree_sha256_before: item.sourceTreeSha256,
					source_tree_sha256_after: item.sourceTreeSha256,
					view_search_tree_sha256_before: item.viewTreeSha256,
					view_search_tree_sha256_after: item.viewTreeSha256,
					locked_view_manifest_path: item.manifestPath,
					locked_view_manifest_sha256: item.manifestSha256,
					render_path: "baseline/iso.png",
					session_file: "baseline/session.jsonl",
					last_turn: item.baselineLastTurn,
				},
				no_reflection_agent_state: false,
				error: errorMessage(outcome.reason),
			};
		});
		const finishedAt = isoNow();
		const summary = {
			run_id: runId,
			experiment: "E-main-view-reflect",
			model_profile: modelSelection.profile,
			provider: modelSelection.provider,
			model: modelSelection.model,
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

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
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
