#!/usr/bin/env node
import { createHash } from "node:crypto";
import { cp, lstat, mkdir, readFile, readdir } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";

import {
	persistReflectionRefinementState,
	ReflectionRefinementRunResult,
	restoreReflectionRefinementState,
	runReflectionRefinementSession,
	sha256File,
} from "./drawing-reflection-refinement.js";
import { errorMessage, InfrastructureError, UsageError } from "./errors.js";
import { resolveExperimentModelSelection } from "./experiment-model-profile.js";
import { elapsedSeconds, isoNow, readJson, writeJsonAtomic } from "./files.js";
import { LocalWorker } from "./local-worker.js";
import {
	REFLECTION_REFINEMENT_SYSTEM_PROMPT,
	reflectionContinuationPrompt,
	reflectionRefinementPrompt,
} from "./prompts.js";
import { resolveVisionModel, RunController } from "./runner.js";
import {
	JOB_FILE,
	REFLECTION_REFINEMENT_PROFILE,
	type AgentJob,
	type DrawingReflectionReport,
	type DrawingViewManifest,
	type SubagentRunUsage,
} from "./types.js";

interface ParsedArguments {
	workspaces: string[];
	alignedRoot: string;
	outputDir: string;
	pythonBinary: string;
	modelProfile?: string;
	provider?: string;
	model?: string;
}

interface ReflectionOnlyResult {
	status: string;
	fixture_id: string;
	summary: {
		input_sha256: string;
		source_sha256: string;
		step_sha256: string;
		manifest_path: string;
		report_path: string;
		sheet_path: string;
	};
	usage: SubagentRunUsage[];
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
	sourceWorkspace: string;
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
	continuationPromptSha256: string;
}

const HELP = `cadgenbench-pi-reflection-refine-session - continue submitted 0.0165 Reflection sessions with selective coordinate-guided dual-ROI checks

Usage:
  cadgenbench-pi-reflection-refine-session <0.0165-case-workspace>... \\
    --aligned-root <verified-aligned-run-root> \\
    --output-dir <new-experiment-root> \\
    [--python <python-binary>] \\
    [--model-profile <gpt-5.6-sol|kimi> | --provider <provider> --model <model>]

The original 0.0165 workspaces and sessions remain read-only. Each case is copied
to an isolated workspace, the exact historical Reflection JSONL is reopened, and
the second-stage turn is appended to that copied session. The default model profile
is gpt-5.6-sol through UCloud; use --model-profile kimi for the historical model.
`;

function parseArguments(arguments_: string[]): ParsedArguments {
	const workspaces: string[] = [];
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
		if (["--aligned-root", "--output-dir", "--python", "--model-profile", "--provider", "--model"].includes(value)) {
			const next = arguments_[index + 1];
			if (!next || next.startsWith("--")) throw new UsageError(`${value} requires a value`);
			if (value === "--aligned-root") alignedRoot = next;
			else if (value === "--output-dir") outputDir = next;
			else if (value === "--python") pythonBinary = next;
			else if (value === "--model-profile") modelProfile = next;
			else if (value === "--provider") provider = next;
			else model = next;
			index += 1;
			continue;
		}
		if (value.startsWith("--")) throw new UsageError(`Unknown option ${value}`);
		workspaces.push(resolve(value));
	}
	if (workspaces.length === 0) throw new UsageError("At least one 0.0165 Case workspace is required");
	if (!alignedRoot) throw new UsageError("--aligned-root is required");
	if (!outputDir) throw new UsageError("--output-dir is required");
	return {
		workspaces,
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

async function copyRelativeFile(sourceRoot: string, destinationRoot: string, relativePath: string): Promise<void> {
	const source = safeRelativePath(sourceRoot, relativePath, "Aligned artifact");
	const destination = safeRelativePath(destinationRoot, relativePath, "Aligned artifact destination");
	await requireRegularFile(source, "Aligned artifact");
	await mkdir(dirname(destination), { recursive: true });
	await cp(source, destination);
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

function lockedRegionIdentity(manifest: DrawingViewManifest): unknown {
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
			transform: region.selected.transform,
			alignment: region.selected.alignment,
		} : null,
	}));
}

async function readSessionId(path: string): Promise<string> {
	const firstLine = (await readFile(path, "utf8")).split("\n", 1)[0];
	if (!firstLine) throw new InfrastructureError("Reflection session is empty");
	const header = JSON.parse(firstLine) as { type?: string; id?: string };
	if (header.type !== "session" || !header.id) throw new InfrastructureError("Reflection session has an invalid header");
	return header.id;
}

async function prepareCase(options: {
	sourceWorkspace: string;
	alignedRoot: string;
	runDir: string;
}): Promise<PreparedCase> {
	const sourceWorkspace = resolve(options.sourceWorkspace);
	const [job, reflectionOnly] = await Promise.all([
		readJson<AgentJob>(join(sourceWorkspace, JOB_FILE)),
		readJson<ReflectionOnlyResult>(join(sourceWorkspace, "reflection_only_result.json")),
	]);
	const fixtureId = reflectionOnly.fixture_id;
	if (reflectionOnly.status !== "completed" || job.fixture.id !== fixtureId) {
		throw new InfrastructureError(`Case ${fixtureId} is not a completed matching 0.0165 Reflection workspace`);
	}
	const reflectionUsage = reflectionOnly.usage.find(({ role, status }) => role === "reflection" && status === "success");
	if (!reflectionUsage?.session_file) throw new InfrastructureError(`Case ${fixtureId} has no successful historical Reflection session`);
	const originalSessionPath = safeRelativePath(sourceWorkspace, reflectionUsage.session_file, "Historical Reflection session");
	const initialReportPath = safeRelativePath(sourceWorkspace, reflectionOnly.summary.report_path, "Initial Reflection report");
	const baselineManifestPath = safeRelativePath(sourceWorkspace, reflectionOnly.summary.manifest_path, "0.0165 view manifest");
	await Promise.all([
		requireRegularFile(originalSessionPath, "Historical Reflection session"),
		requireRegularFile(initialReportPath, "Initial Reflection report"),
		requireRegularFile(baselineManifestPath, "0.0165 view manifest"),
		requireRegularFile(join(sourceWorkspace, "solution.py"), "0.0165 solution.py"),
		requireRegularFile(join(sourceWorkspace, "output.step"), "0.0165 STEP"),
	]);
	const [sourceSha256, stepSha256] = await Promise.all([
		sha256File(join(sourceWorkspace, "solution.py")),
		sha256File(join(sourceWorkspace, "output.step")),
	]);
	if (sourceSha256 !== reflectionOnly.summary.source_sha256 || stepSha256 !== reflectionOnly.summary.step_sha256) {
		throw new InfrastructureError(`Case ${fixtureId} source/STEP hashes no longer match the 0.0165 Reflection`);
	}
	const alignedWorkspace = join(options.alignedRoot, fixtureId);
	const aligned = await readJson<AlignedSearchResult>(join(alignedWorkspace, "view_search_only_result.json"));
	if (
		aligned.status !== "completed" ||
		aligned.fixture_id !== fixtureId ||
		aligned.input_sha256 !== reflectionOnly.summary.input_sha256 ||
		aligned.source_sha256 !== reflectionOnly.summary.source_sha256 ||
		aligned.step_sha256 !== reflectionOnly.summary.step_sha256
	) throw new InfrastructureError(`Case ${fixtureId} aligned workspace hashes do not match 0.0165`);
	const [baselineManifest, viewManifest] = await Promise.all([
		readJson<DrawingViewManifest>(baselineManifestPath),
		aligned.manifest
			? Promise.resolve(aligned.manifest)
			: readJson<DrawingViewManifest>(safeRelativePath(alignedWorkspace, aligned.manifest_path, "Aligned view manifest")),
	]);
	if (stableJson(lockedRegionIdentity(baselineManifest)) !== stableJson(lockedRegionIdentity(viewManifest))) {
		throw new InfrastructureError(`Case ${fixtureId} aligned candidate/ROI/camera identity differs from 0.0165`);
	}
	for (const region of viewManifest.regions) {
		if (region.selected && !region.selected.aligned_render_path) {
			throw new InfrastructureError(`Case ${fixtureId}/${region.region_id} lacks a selected aligned render`);
		}
	}

	const workspace = join(options.runDir, fixtureId);
	await cp(sourceWorkspace, workspace, { recursive: true });
	const artifactPaths = new Set<string>();
	for (const region of viewManifest.regions) {
		artifactPaths.add(region.crop_path);
		if (region.selected?.comparison_path) artifactPaths.add(region.selected.comparison_path);
		if (region.selected?.aligned_render_path) artifactPaths.add(region.selected.aligned_render_path);
	}
	await Promise.all([...artifactPaths].map(async (path) => await copyRelativeFile(alignedWorkspace, workspace, path)));
	const lockedManifestPath = "reflection_refinement/locked_view_manifest.json";
	const initialReportCopyPath = "reflection_refinement/baseline/reflection.json";
	const initialSessionCopyPath = `reflection_refinement/baseline/${basename(originalSessionPath)}`;
	const sessionsDir = join(workspace, "reflection_refinement", "sessions", "reflection");
	const copiedSessionPath = join(sessionsDir, basename(originalSessionPath));
	await Promise.all([
		mkdir(dirname(join(workspace, lockedManifestPath)), { recursive: true }),
		mkdir(dirname(join(workspace, initialReportCopyPath)), { recursive: true }),
		mkdir(dirname(join(workspace, initialSessionCopyPath)), { recursive: true }),
		mkdir(sessionsDir, { recursive: true }),
	]);
	await Promise.all([
		writeJsonAtomic(join(workspace, lockedManifestPath), viewManifest),
		cp(initialReportPath, join(workspace, initialReportCopyPath)),
		cp(originalSessionPath, join(workspace, initialSessionCopyPath)),
		cp(originalSessionPath, copiedSessionPath),
		cp(safeRelativePath(sourceWorkspace, reflectionOnly.summary.sheet_path, "0.0165 comparison sheet"), join(workspace, "reflection_refinement", "baseline", "comparison.png")),
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
	if (originalSessionId !== copiedSessionId || originalSessionId !== reflectionUsage.accounting_id) {
		throw new InfrastructureError(`Case ${fixtureId} historical session identity mismatch`);
	}
	const initialPromptFindings = initialReport.findings.map((finding, index) => ({
		finding_id: `finding-${String(index + 1).padStart(3, "0")}`,
		...finding,
	}));
	const promptSha256 = sha256Text(reflectionRefinementPrompt({ fixtureId, findings: initialPromptFindings }));
	const continuationPromptSha256 = sha256Text(reflectionContinuationPrompt(fixtureId, initialReport));
	await writeJsonAtomic(join(workspace, "reflection_refinement", "baseline", "manifest.json"), {
		profile: REFLECTION_REFINEMENT_PROFILE,
		fixture_id: fixtureId,
		source_workspace: sourceWorkspace,
		aligned_workspace: alignedWorkspace,
		input_sha256: reflectionOnly.summary.input_sha256,
		source_sha256: sourceSha256,
		step_sha256: stepSha256,
		initial_report_sha256: initialReportSha256,
		session_id: originalSessionId,
		original_session_sha256: originalSessionSha256,
		copied_session_sha256: copiedSessionSha256,
		refinement_system_prompt_sha256: sha256Text(REFLECTION_REFINEMENT_SYSTEM_PROMPT),
		refinement_prompt_sha256: promptSha256,
		continuation_prompt_sha256: continuationPromptSha256,
	});
	return {
		fixtureId,
		sourceWorkspace,
		alignedWorkspace,
		workspace,
		job,
		viewManifest,
		initialReport,
		initialReportSha256,
		originalSessionPath,
		originalSessionSha256,
		copiedSessionPath,
		copiedSessionSha256,
		sessionsDir,
		promptSha256,
		continuationPromptSha256,
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
		if (result.crop_count < 1) throw new InfrastructureError("Refined Reflection completed without a dual-ROI crop");
		const [originalAfter, copiedAfter] = await Promise.all([
			sha256File(options.prepared.originalSessionPath),
			sha256File(options.prepared.copiedSessionPath),
		]);
		const baselineReportAfter = await sha256File(join(
			options.prepared.workspace,
			"reflection_refinement",
			"baseline",
			"reflection.json",
		));
		if (originalAfter !== options.prepared.originalSessionSha256) {
			throw new InfrastructureError("Original 0.0165 Reflection session was modified");
		}
		if (baselineReportAfter !== options.prepared.initialReportSha256) {
			throw new InfrastructureError("Original 0.0165 Reflection report snapshot was modified");
		}
		if (copiedAfter === options.prepared.copiedSessionSha256) {
			throw new InfrastructureError("Copied Reflection session did not receive the second-stage turn");
		}
		const output = {
			...result,
			source_workspace: options.prepared.sourceWorkspace,
			aligned_workspace: options.prepared.alignedWorkspace,
			workspace: options.prepared.workspace,
			original_session_sha256: options.prepared.originalSessionSha256,
			continued_session_sha256: copiedAfter,
			refinement_prompt_sha256: options.prepared.promptSha256,
			continuation_prompt_sha256: options.prepared.continuationPromptSha256,
		};
		await writeJsonAtomic(join(options.prepared.workspace, "reflection_refinement_result.json"), output);
		return output;
	} catch (error) {
		const output = {
			fixture_id: options.prepared.fixtureId,
			status: options.controller.interrupted ? "interrupted" : "error",
			started_at: startedAt,
			finished_at: isoNow(),
			duration_s: elapsedSeconds(startedAt),
			source_workspace: options.prepared.sourceWorkspace,
			workspace: options.prepared.workspace,
			error: errorMessage(error),
		};
		await writeJsonAtomic(join(options.prepared.workspace, "reflection_refinement_result.json"), output);
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
	const preparedCases: PreparedCase[] = [];
	for (const sourceWorkspace of parsed.workspaces) {
		preparedCases.push(await prepareCase({ sourceWorkspace, alignedRoot: parsed.alignedRoot, runDir }));
	}
	if (new Set(preparedCases.map(({ fixtureId }) => fixtureId)).size !== preparedCases.length) {
		throw new UsageError("Case fixture IDs must be unique");
	}
	const modelKeys = new Set(preparedCases.map(({ job }) => `${job.provider}\0${job.model}`));
	if (modelKeys.size !== 1) throw new UsageError("All historical cases must use the same provider/model");
	const firstJob = preparedCases[0]!.job;
	const resolvedModel = await resolveVisionModel(modelSelection.provider, modelSelection.model);
	const worker = new LocalWorker(parsed.pythonBinary);
	const controllers = preparedCases.map(() => new RunController());
	const startedAt = isoNow();
	const runId = `reflection-refinement-${startedAt.replaceAll(/[-:.]/gu, "")}-${process.pid}`;
	const onSigint = (): void => {
		void Promise.allSettled(controllers.map(async (controller) => await controller.interrupt("SIGINT")));
	};
	process.once("SIGINT", onSigint);
	try {
		await writeJsonAtomic(join(parsed.outputDir, "params.json"), {
			run_id: runId,
			profile: REFLECTION_REFINEMENT_PROFILE,
			created_at: startedAt,
			model_profile: modelSelection.profile,
			provider: modelSelection.provider,
			model: modelSelection.model,
			historical_provider: firstJob.provider,
			historical_model: firstJob.model,
			thinking_level: "high",
			active_timeout_minutes: 30,
			wall_timeout_minutes: 120,
			case_concurrency: preparedCases.length,
			fixtures: preparedCases.map(({ fixtureId }) => fixtureId),
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
			profile: REFLECTION_REFINEMENT_PROFILE,
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
		await writeJsonAtomic(join(parsed.outputDir, "reflection_refinement_summary.json"), summary);
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
