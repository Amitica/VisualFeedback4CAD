#!/usr/bin/env node
import { createHash } from "node:crypto";
import { cp, lstat, mkdir, readFile, readdir } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";

import {
	persistReflectionRefinementState,
	type ReflectionRefinementRunResult,
	restoreReflectionRefinementState,
	runReflectionRefinementSession,
	sha256File,
} from "./drawing-reflection-refinement.js";
import { requireDrawingSuccess } from "./drawing-reflection.js";
import { errorMessage, InfrastructureError, UsageError } from "./errors.js";
import { resolveExperimentModelSelection } from "./experiment-model-profile.js";
import { elapsedSeconds, isoNow, readJson, writeJsonAtomic } from "./files.js";
import { LocalWorker } from "./local-worker.js";
import {
	REFLECTION_ACCEPTANCE_SYSTEM_PROMPT,
	reflectionAcceptancePrompt,
} from "./prompts.js";
import { resolveVisionModel, RunController } from "./runner.js";
import {
	JOB_FILE,
	PROTOCOL_VERSION,
	REFLECTION_ACCEPTANCE_PROFILE,
	type AgentJob,
	type AgentResult,
	type DrawingReflectionReport,
	type DrawingViewManifest,
	type WorkerCompareDrawingSheetResponse,
	type WorkerComparisonSheetRegion,
} from "./types.js";

interface ParsedArguments {
	reflectionWorkspaces: string[];
	currentCadRoot: string;
	alignedRoot: string;
	outputDir: string;
	pythonBinary: string;
	modelProfile?: string;
	provider?: string;
	model?: string;
}

interface RefinementResult {
	status: string;
	fixture_id: string;
	session_id: string;
	session_file: string;
	refined_report_path: string;
}

interface AlignedSearchResult {
	status: string;
	fixture_id: string;
	input_sha256: string;
	source_sha256: string;
	step_sha256: string;
	manifest_path: string;
	manifest?: DrawingViewManifest;
}

interface PreparedCase {
	fixtureId: string;
	reflectionWorkspace: string;
	currentCadWorkspace: string;
	alignedWorkspace: string;
	workspace: string;
	job: AgentJob;
	viewManifest: DrawingViewManifest;
	initialReport: DrawingReflectionReport;
	initialReportSha256: string;
	originalSessionPath: string;
	originalSessionSha256: string;
	copiedSessionPath: string;
	copiedSessionSha256: string;
	sessionsDir: string;
	promptSha256: string;
	currentSourceSha256: string;
	currentStepSha256: string;
	currentSheetPath: string;
}

const HELP = `cadgenbench-pi-reflection-accept - validate two-stage findings after Main-Agent CAD edits

Usage:
  cadgenbench-pi-reflection-accept <two-stage-reflection-workspace>... \\
    --current-cad-root <post-continuation-run-root> \\
    --aligned-root <updated-locked-view-replay-root> \\
    --output-dir <new-acceptance-root> \\
    [--python <python-binary>] \\
    [--model-profile <gpt-5.6-sol|kimi> | --provider <provider> --model <model>]

Each copied Reflection session is continued in place. The historical session and
report remain read-only, while every carried finding is reviewed against locked
views regenerated from the current post-continuation STEP.
`;

function parseArguments(arguments_: string[]): ParsedArguments {
	const reflectionWorkspaces: string[] = [];
	let currentCadRoot: string | undefined;
	let alignedRoot: string | undefined;
	let outputDir: string | undefined;
	let pythonBinary = "python";
	let modelProfile: string | undefined;
	let provider: string | undefined;
	let model: string | undefined;
	for (let index = 0; index < arguments_.length; index += 1) {
		const value = arguments_[index];
		if (!value) continue;
		if (value === "--help" || value === "-h") {
			process.stdout.write(HELP);
			process.exit(0);
		}
		if (["--current-cad-root", "--aligned-root", "--output-dir", "--python", "--model-profile", "--provider", "--model"].includes(value)) {
			const next = arguments_[index + 1];
			if (!next || next.startsWith("--")) throw new UsageError(`${value} requires a value`);
			if (value === "--current-cad-root") currentCadRoot = next;
			else if (value === "--aligned-root") alignedRoot = next;
			else if (value === "--output-dir") outputDir = next;
			else if (value === "--python") pythonBinary = next;
			else if (value === "--model-profile") modelProfile = next;
			else if (value === "--provider") provider = next;
			else model = next;
			index += 1;
			continue;
		}
		if (value.startsWith("--")) throw new UsageError(`Unknown option ${value}`);
		reflectionWorkspaces.push(resolve(value));
	}
	if (reflectionWorkspaces.length === 0) throw new UsageError("At least one two-stage Reflection workspace is required");
	if (!currentCadRoot) throw new UsageError("--current-cad-root is required");
	if (!alignedRoot) throw new UsageError("--aligned-root is required");
	if (!outputDir) throw new UsageError("--output-dir is required");
	return {
		reflectionWorkspaces,
		currentCadRoot: resolve(currentCadRoot),
		alignedRoot: resolve(alignedRoot),
		outputDir: resolve(outputDir),
		pythonBinary,
		...(modelProfile ? { modelProfile } : {}),
		...(provider ? { provider } : {}),
		...(model ? { model } : {}),
	};
}

async function requireEmptyDirectory(path: string): Promise<void> {
	const metadata = await lstat(path).catch(() => null);
	if (!metadata) {
		await mkdir(path, { recursive: true });
		return;
	}
	if (!metadata.isDirectory() || metadata.isSymbolicLink()) throw new UsageError(`Output must be a real directory: ${path}`);
	if ((await readdir(path)).length > 0) throw new UsageError(`Output directory must be empty: ${path}`);
}

async function requireRegularFile(path: string, label: string): Promise<void> {
	const metadata = await lstat(path).catch(() => null);
	if (!metadata?.isFile() || metadata.isSymbolicLink()) throw new InfrastructureError(`${label} is missing or unsafe: ${path}`);
}

function safeRelativePath(root: string, path: string, label: string): string {
	if (!path || isAbsolute(path)) throw new InfrastructureError(`${label} must be a relative workspace path`);
	const candidate = resolve(root, path);
	const rel = relative(resolve(root), candidate);
	if (rel === ".." || rel.startsWith("../") || isAbsolute(rel)) throw new InfrastructureError(`${label} escaped its workspace`);
	return candidate;
}

function stableJson(value: unknown): string {
	if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
	if (value && typeof value === "object") {
		return `{${Object.entries(value as Record<string, unknown>)
			.sort(([left], [right]) => left.localeCompare(right))
			.map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`)
			.join(",")}}`;
	}
	return JSON.stringify(value);
}

function sha256Text(value: string): string {
	return createHash("sha256").update(value).digest("hex");
}

function lockedCameraIdentity(manifest: DrawingViewManifest): unknown {
	return manifest.regions.map((region) => ({
		region_id: region.region_id,
		page: region.page,
		roi: region.roi,
		crop_sha256: region.crop_sha256,
		status: region.status,
		view_type: region.view_type,
		selected: region.selected ? {
			candidate_id: region.selected.candidate_id,
			view_spec: region.selected.view_spec,
		} : null,
		alternatives: region.alternatives.map(({ candidate_id, view_spec }) => ({ candidate_id, view_spec })),
	}));
}

async function readSessionId(path: string): Promise<string> {
	const firstLine = (await readFile(path, "utf8")).split("\n", 1)[0];
	if (!firstLine) throw new InfrastructureError("Reflection session is empty");
	const header = JSON.parse(firstLine) as { type?: string; id?: string };
	if (header.type !== "session" || !header.id) throw new InfrastructureError("Reflection session has an invalid header");
	return header.id;
}

async function generateCurrentSheet(options: {
	workspace: string;
	manifest: DrawingViewManifest;
	worker: LocalWorker;
	controller: RunController;
}): Promise<string> {
	const regions: WorkerComparisonSheetRegion[] = options.manifest.regions.map((region) => ({
		region_id: region.region_id,
		crop_path: region.crop_path,
		...(region.selected ? {
			render_path: region.selected.render_path,
			...(region.selected.comparison_path ? { comparison_path: region.selected.comparison_path } : {}),
			view_spec: region.selected.view_spec,
			score: region.selected.score,
			margin: region.margin,
		} : {}),
		status: region.status,
	}));
	const response = await options.worker.request<WorkerCompareDrawingSheetResponse>(
		options.workspace,
		{
			protocol_version: PROTOCOL_VERSION,
			operation: "compare_drawing_sheet",
			regions,
			output_dir: "drawing_feedback/reflection-acceptance/current-sheet",
		},
		{ timeoutMs: 180_000, signal: options.controller.signal },
	);
	requireDrawingSuccess(response, "compare_drawing_sheet");
	await requireRegularFile(safeRelativePath(options.workspace, response.sheet_path, "Current comparison sheet"), "Current comparison sheet");
	return response.sheet_path;
}

async function prepareCase(options: {
	reflectionWorkspace: string;
	currentCadRoot: string;
	alignedRoot: string;
	runDir: string;
	worker: LocalWorker;
	controller: RunController;
}): Promise<PreparedCase> {
	const reflectionWorkspace = resolve(options.reflectionWorkspace);
	const [job, refinement] = await Promise.all([
		readJson<AgentJob>(join(reflectionWorkspace, JOB_FILE)),
		readJson<RefinementResult>(join(reflectionWorkspace, "reflection_refinement_result.json")),
	]);
	const fixtureId = refinement.fixture_id;
	if (refinement.status !== "completed" || job.fixture.id !== fixtureId) {
		throw new InfrastructureError(`Case ${fixtureId} is not a completed matching two-stage Reflection workspace`);
	}
	const originalSessionPath = safeRelativePath(reflectionWorkspace, refinement.session_file, "Two-stage Reflection session");
	const initialReportPath = safeRelativePath(reflectionWorkspace, refinement.refined_report_path, "Two-stage Reflection report");
	const oldLockedManifestPath = join(reflectionWorkspace, "reflection_refinement", "locked_view_manifest.json");
	await Promise.all([
		requireRegularFile(originalSessionPath, "Two-stage Reflection session"),
		requireRegularFile(initialReportPath, "Two-stage Reflection report"),
		requireRegularFile(oldLockedManifestPath, "Two-stage locked view manifest"),
	]);

	const currentCadWorkspace = join(options.currentCadRoot, fixtureId);
	const currentResult = await readJson<AgentResult>(join(currentCadWorkspace, "agent_result.json"));
	if (
		currentResult.fixture_id !== fixtureId ||
		currentResult.status !== "valid" ||
		currentResult.artifact?.validation?.is_valid !== true ||
		currentResult.artifact.validation.is_watertight !== true
	) throw new InfrastructureError(`Case ${fixtureId} current Main-Agent CAD is not valid and watertight`);
	const [currentSourceSha256, currentStepSha256] = await Promise.all([
		sha256File(join(currentCadWorkspace, "solution.py")),
		sha256File(join(currentCadWorkspace, "output.step")),
	]);
	if (
		currentSourceSha256 !== currentResult.last_executed_source_sha256 ||
		currentStepSha256 !== currentResult.last_executed_step_sha256
	) throw new InfrastructureError(`Case ${fixtureId} current source/STEP hashes are stale`);

	const alignedWorkspace = join(options.alignedRoot, fixtureId);
	const aligned = await readJson<AlignedSearchResult>(join(alignedWorkspace, "view_search_only_result.json"));
	if (
		aligned.status !== "completed" ||
		aligned.fixture_id !== fixtureId ||
		aligned.source_sha256 !== currentSourceSha256 ||
		aligned.step_sha256 !== currentStepSha256
	) throw new InfrastructureError(`Case ${fixtureId} current aligned workspace does not match the post-continuation CAD`);
	const [oldManifest, currentManifest] = await Promise.all([
		readJson<DrawingViewManifest>(oldLockedManifestPath),
		aligned.manifest
			? Promise.resolve(aligned.manifest)
			: readJson<DrawingViewManifest>(safeRelativePath(alignedWorkspace, aligned.manifest_path, "Current aligned view manifest")),
	]);
	if (
		oldManifest.input_sha256 !== aligned.input_sha256 ||
		stableJson(lockedCameraIdentity(oldManifest)) !== stableJson(lockedCameraIdentity(currentManifest))
	) throw new InfrastructureError(`Case ${fixtureId} current regions or locked camera parameters differ from the two-stage Reflection`);
	for (const region of currentManifest.regions) {
		if (region.selected && !region.selected.aligned_render_path) {
			throw new InfrastructureError(`Case ${fixtureId}/${region.region_id} lacks a current aligned CAD render`);
		}
	}

	const workspace = join(options.runDir, fixtureId);
	await cp(alignedWorkspace, workspace, { recursive: true });
	const baselineDir = join(workspace, "reflection_acceptance", "baseline");
	const sessionsDir = join(workspace, "reflection_acceptance", "sessions", "reflection");
	await Promise.all([mkdir(baselineDir, { recursive: true }), mkdir(sessionsDir, { recursive: true })]);
	const copiedSessionPath = join(sessionsDir, basename(originalSessionPath));
	await Promise.all([
		cp(initialReportPath, join(baselineDir, "reflection.json")),
		cp(originalSessionPath, join(baselineDir, basename(originalSessionPath))),
		cp(originalSessionPath, copiedSessionPath),
		cp(join(currentCadWorkspace, "agent_result.json"), join(baselineDir, "current_agent_result.json")),
		writeJsonAtomic(join(workspace, "reflection_acceptance", "locked_view_manifest.json"), currentManifest),
	]);
	const [initialReport, initialReportSha256, originalSessionSha256, copiedSessionSha256] = await Promise.all([
		readJson<DrawingReflectionReport>(initialReportPath),
		sha256File(initialReportPath),
		sha256File(originalSessionPath),
		sha256File(copiedSessionPath),
	]);
	if (originalSessionSha256 !== copiedSessionSha256) throw new InfrastructureError(`Case ${fixtureId} copied session hash mismatch`);
	const [originalSessionId, copiedSessionId] = await Promise.all([
		readSessionId(originalSessionPath),
		readSessionId(copiedSessionPath),
	]);
	if (originalSessionId !== copiedSessionId || originalSessionId !== refinement.session_id) {
		throw new InfrastructureError(`Case ${fixtureId} two-stage Reflection session identity mismatch`);
	}
	const promptFindings = initialReport.findings.map((finding, index) => ({
		finding_id: `finding-${String(index + 1).padStart(3, "0")}`,
		...finding,
	}));
	const promptSha256 = sha256Text(reflectionAcceptancePrompt({ fixtureId, findings: promptFindings }));
	const currentSheetPath = await generateCurrentSheet({
		workspace,
		manifest: currentManifest,
		worker: options.worker,
		controller: options.controller,
	});
	await writeJsonAtomic(join(baselineDir, "manifest.json"), {
		profile: REFLECTION_ACCEPTANCE_PROFILE,
		fixture_id: fixtureId,
		reflection_workspace: reflectionWorkspace,
		current_cad_workspace: currentCadWorkspace,
		aligned_workspace: alignedWorkspace,
		input_sha256: aligned.input_sha256,
		current_source_sha256: currentSourceSha256,
		current_step_sha256: currentStepSha256,
		initial_report_sha256: initialReportSha256,
		session_id: originalSessionId,
		original_session_sha256: originalSessionSha256,
		copied_session_sha256: copiedSessionSha256,
		acceptance_system_prompt_sha256: sha256Text(REFLECTION_ACCEPTANCE_SYSTEM_PROMPT),
		acceptance_prompt_sha256: promptSha256,
		current_sheet_path: currentSheetPath,
	});
	return {
		fixtureId,
		reflectionWorkspace,
		currentCadWorkspace,
		alignedWorkspace,
		workspace,
		job,
		viewManifest: currentManifest,
		initialReport,
		initialReportSha256,
		originalSessionPath,
		originalSessionSha256,
		copiedSessionPath,
		copiedSessionSha256,
		sessionsDir,
		promptSha256,
		currentSourceSha256,
		currentStepSha256,
		currentSheetPath,
	};
}

async function runCase(options: {
	prepared: PreparedCase;
	worker: LocalWorker;
	resolvedModel: Awaited<ReturnType<typeof resolveVisionModel>>;
	controller: RunController;
}): Promise<Record<string, unknown>> {
	const startedAt = isoNow();
	try {
		const state = await restoreReflectionRefinementState({
			workspace: options.prepared.workspace,
			fixtureId: options.prepared.fixtureId,
			viewManifest: options.prepared.viewManifest,
			initialReport: options.prepared.initialReport,
			initialReportSha256: options.prepared.initialReportSha256,
			phase: "acceptance",
		});
		await persistReflectionRefinementState(options.prepared.workspace, state);
		const result: ReflectionRefinementRunResult = await runReflectionRefinementSession({
			workspace: options.prepared.workspace,
			sessionFile: options.prepared.copiedSessionPath,
			sessionsDir: options.prepared.sessionsDir,
			state,
			worker: options.worker,
			resolvedModel: options.resolvedModel,
			thinkingLevel: "high",
			controller: options.controller,
		});
		const [originalSessionAfter, copiedSessionAfter, originalReportAfter] = await Promise.all([
			sha256File(options.prepared.originalSessionPath),
			sha256File(options.prepared.copiedSessionPath),
			sha256File(join(options.prepared.workspace, "reflection_acceptance", "baseline", "reflection.json")),
		]);
		if (originalSessionAfter !== options.prepared.originalSessionSha256) {
			throw new InfrastructureError("Original two-stage Reflection session was modified");
		}
		if (originalReportAfter !== options.prepared.initialReportSha256) {
			throw new InfrastructureError("Two-stage Reflection report snapshot was modified");
		}
		if (copiedSessionAfter === options.prepared.copiedSessionSha256) {
			throw new InfrastructureError("Copied Reflection session did not receive the acceptance turn");
		}
		const output = {
			...result,
			profile: REFLECTION_ACCEPTANCE_PROFILE,
			reflection_workspace: options.prepared.reflectionWorkspace,
			current_cad_workspace: options.prepared.currentCadWorkspace,
			aligned_workspace: options.prepared.alignedWorkspace,
			workspace: options.prepared.workspace,
			current_source_sha256: options.prepared.currentSourceSha256,
			current_step_sha256: options.prepared.currentStepSha256,
			current_sheet_path: options.prepared.currentSheetPath,
			accepted_report_path: result.refined_report_path,
			original_session_sha256: options.prepared.originalSessionSha256,
			continued_session_sha256: copiedSessionAfter,
			acceptance_prompt_sha256: options.prepared.promptSha256,
		};
		await writeJsonAtomic(join(options.prepared.workspace, "reflection_acceptance_result.json"), output);
		return output;
	} catch (error) {
		const output = {
			fixture_id: options.prepared.fixtureId,
			status: options.controller.interrupted ? "interrupted" : "error",
			started_at: startedAt,
			finished_at: isoNow(),
			duration_s: elapsedSeconds(startedAt),
			reflection_workspace: options.prepared.reflectionWorkspace,
			current_cad_workspace: options.prepared.currentCadWorkspace,
			workspace: options.prepared.workspace,
			error: errorMessage(error),
		};
		await writeJsonAtomic(join(options.prepared.workspace, "reflection_acceptance_result.json"), output);
		throw Object.assign(new Error(errorMessage(error)), { result: output });
	}
}

export async function main(arguments_ = process.argv.slice(2)): Promise<number> {
	const parsed = parseArguments(arguments_);
	const modelSelection = resolveExperimentModelSelection({
		profile: parsed.modelProfile ?? process.env.CADGENBENCH_PI_MODEL_PROFILE,
		provider: parsed.provider,
		model: parsed.model,
	});
	await requireEmptyDirectory(parsed.outputDir);
	const runDir = join(parsed.outputDir, "run");
	await mkdir(runDir, { recursive: true });
	const worker = new LocalWorker(parsed.pythonBinary);
	const controllers = parsed.reflectionWorkspaces.map(() => new RunController());
	const preparedCases = await Promise.all(parsed.reflectionWorkspaces.map(async (reflectionWorkspace, index) => await prepareCase({
		reflectionWorkspace,
		currentCadRoot: parsed.currentCadRoot,
		alignedRoot: parsed.alignedRoot,
		runDir,
		worker,
		controller: controllers[index]!,
	})));
	if (new Set(preparedCases.map(({ fixtureId }) => fixtureId)).size !== preparedCases.length) {
		throw new UsageError("Case fixture IDs must be unique");
	}
	const modelKeys = new Set(preparedCases.map(({ job }) => `${job.provider}\0${job.model}`));
	if (modelKeys.size !== 1) throw new UsageError("All two-stage Reflection cases must use the same historical provider/model");
	const resolvedModel = await resolveVisionModel(modelSelection.provider, modelSelection.model);
	const startedAt = isoNow();
	const runId = `reflection-acceptance-${startedAt.replaceAll(/[-:.]/gu, "")}-${process.pid}`;
	const onSigint = (): void => {
		void Promise.allSettled(controllers.map(async (controller) => await controller.interrupt("SIGINT")));
	};
	process.once("SIGINT", onSigint);
	try {
		await writeJsonAtomic(join(parsed.outputDir, "params.json"), {
			run_id: runId,
			profile: REFLECTION_ACCEPTANCE_PROFILE,
			created_at: startedAt,
			model_profile: modelSelection.profile,
			provider: modelSelection.provider,
			model: modelSelection.model,
			thinking_level: "high",
			active_timeout_minutes: 30,
			wall_timeout_minutes: 120,
			case_concurrency: preparedCases.length,
			fixtures: preparedCases.map(({ fixtureId }) => fixtureId),
			current_cad_root: parsed.currentCadRoot,
			aligned_root: parsed.alignedRoot,
			python: parsed.pythonBinary,
		});
		const settled = await Promise.allSettled(preparedCases.map(async (prepared, index) => await runCase({
			prepared,
			worker,
			resolvedModel,
			controller: controllers[index]!,
		})));
		const cases = settled.map((outcome, index) => outcome.status === "fulfilled"
			? outcome.value
			: ((outcome.reason as { result?: Record<string, unknown> }).result ?? {
				fixture_id: preparedCases[index]!.fixtureId,
				status: "error",
				error: errorMessage(outcome.reason),
			}));
		const completed = cases.filter(({ status }) => status === "completed").length;
		const summary = {
			run_id: runId,
			profile: REFLECTION_ACCEPTANCE_PROFILE,
			model_profile: modelSelection.profile,
			provider: modelSelection.provider,
			model: modelSelection.model,
			status: completed === cases.length ? "completed" : completed > 0 ? "partial" : "error",
			started_at: startedAt,
			finished_at: isoNow(),
			duration_s: elapsedSeconds(startedAt),
			case_concurrency: preparedCases.length,
			cases,
		};
		await writeJsonAtomic(join(parsed.outputDir, "reflection_acceptance_summary.json"), summary);
		process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
		return completed === cases.length ? 0 : controllers.some(({ interrupted }) => interrupted) ? 130 : 2;
	} finally {
		process.removeListener("SIGINT", onSigint);
	}
}

main().then(
	(code) => {
		process.exitCode = code;
	},
	(error: unknown) => {
		process.stderr.write(`${errorMessage(error)}\n`);
		process.exitCode = error instanceof UsageError ? 64 : 2;
	},
);
