#!/usr/bin/env node
import { copyFile, lstat, mkdir, readdir } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";

import { runDrawingViewSearchOnly } from "./drawing-reflection.js";
import { errorMessage, UsageError } from "./errors.js";
import { elapsedSeconds, isoNow, readJson, writeJsonAtomic } from "./files.js";
import { LocalWorker } from "./local-worker.js";
import { resolveVisionModel, RunController } from "./runner.js";
import {
	JOB_FILE,
	PROTOCOL_VERSION,
	SOLUTION_FILE,
	type AgentJob,
	type AgentResult,
	type SubagentRunUsage,
} from "./types.js";

interface ParsedArguments {
	sourceWorkspaces: string[];
	outputDir: string;
	pythonBinary: string;
	resume: boolean;
}

interface PreparedCase {
	sourceWorkspace: string;
	workspace: string;
	job: AgentJob;
	result: AgentResult;
}

const HELP = `cadgenbench-pi-drawing-search - segmentation + concurrent View Search only

Usage:
  cadgenbench-pi-drawing-search <main-agent-workspace>... --output-dir <dir> [--python <path>] [--resume]

The source workspaces are read-only. The command copies solution.py, output.step,
and staged drawing inputs into one isolated workspace per fixture, runs the three
selected fixture pipelines concurrently, and stops before Reflection Agent.
With --resume, verified prepared workspaces and valid segmentation/search caches
are reused; the preceding summary and per-Case result files are archived first.
`;

function parseArguments(arguments_: string[]): ParsedArguments {
	const sourceWorkspaces: string[] = [];
	let outputDir: string | undefined;
	let pythonBinary = "python";
	let resume = false;
	for (let index = 0; index < arguments_.length; index += 1) {
		const value = arguments_[index];
		if (!value) continue;
		if (value === "--help" || value === "-h") {
			process.stdout.write(HELP);
			process.exit(0);
		}
		if (value === "--resume") {
			resume = true;
			continue;
		}
		if (value === "--output-dir" || value === "--python") {
			const next = arguments_[index + 1];
			if (!next || next.startsWith("--")) throw new UsageError(`${value} requires a value`);
			if (value === "--output-dir") outputDir = next;
			else pythonBinary = next;
			index += 1;
			continue;
		}
		if (value.startsWith("--")) throw new UsageError(`Unknown option ${value}`);
		sourceWorkspaces.push(value);
	}
	if (!outputDir) throw new UsageError("--output-dir is required");
	if (sourceWorkspaces.length === 0) throw new UsageError("At least one main-Agent workspace is required");
	return {
		sourceWorkspaces: sourceWorkspaces.map((path) => resolve(path)),
		outputDir: resolve(outputDir),
		pythonBinary,
		resume,
	};
}

async function requireDirectory(path: string, label: string): Promise<void> {
	const metadata = await lstat(path).catch(() => null);
	if (!metadata?.isDirectory() || metadata.isSymbolicLink()) {
		throw new UsageError(`${label} must be a real directory: ${path}`);
	}
}

function safeRelativePath(path: string, label: string): string {
	if (isAbsolute(path)) throw new UsageError(`${label} must be relative: ${path}`);
	const normalized = relative(".", path);
	if (normalized === ".." || normalized.startsWith("../") || isAbsolute(normalized)) {
		throw new UsageError(`${label} escapes its workspace: ${path}`);
	}
	return normalized;
}

async function copyRegularFile(sourceRoot: string, destinationRoot: string, relativePath: string): Promise<void> {
	const safePath = safeRelativePath(relativePath, "Copied file path");
	const source = join(sourceRoot, safePath);
	const metadata = await lstat(source).catch(() => null);
	if (!metadata?.isFile() || metadata.isSymbolicLink()) {
		throw new UsageError(`Required source file is missing or unsafe: ${source}`);
	}
	const destination = join(destinationRoot, safePath);
	await mkdir(dirname(destination), { recursive: true });
	await copyFile(source, destination);
}

async function prepareOutputDirectory(outputDir: string, resume: boolean): Promise<void> {
	const metadata = await lstat(outputDir).catch(() => null);
	if (!metadata) {
		await mkdir(outputDir, { recursive: true });
		return;
	}
	if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
		throw new UsageError(`--output-dir must be a real directory: ${outputDir}`);
	}
	if (!resume && (await readdir(outputDir)).length > 0) {
		throw new UsageError(`Refusing to use non-empty --output-dir: ${outputDir}`);
	}
}

async function archivePreviousRun(outputDir: string): Promise<void> {
	const summaryPath = join(outputDir, "drawing_search_only_summary.json");
	const summaryMetadata = await lstat(summaryPath).catch(() => null);
	if (!summaryMetadata) return;
	if (!summaryMetadata.isFile() || summaryMetadata.isSymbolicLink()) {
		throw new UsageError(`Previous summary is unsafe: ${summaryPath}`);
	}
	const summary = await readJson<{ run_id?: string }>(summaryPath);
	if (!summary.run_id || !/^[A-Za-z0-9._-]+$/u.test(summary.run_id)) {
		throw new UsageError(`Previous summary has an unsafe run_id: ${summaryPath}`);
	}
	const archiveDir = join(outputDir, "previous_runs", summary.run_id);
	await mkdir(archiveDir, { recursive: true });
	await copyFile(summaryPath, join(archiveDir, "drawing_search_only_summary.json"));
	for (const entry of await readdir(outputDir, { withFileTypes: true })) {
		if (!entry.isDirectory() || entry.isSymbolicLink() || entry.name === "previous_runs") continue;
		for (const name of ["view_search_only_progress.json", "view_search_only_result.json"]) {
			const source = join(outputDir, entry.name, name);
			const metadata = await lstat(source).catch(() => null);
			if (!metadata) continue;
			if (!metadata.isFile() || metadata.isSymbolicLink()) {
				throw new UsageError(`Previous per-Case result is unsafe: ${source}`);
			}
			const destinationDir = join(archiveDir, entry.name);
			await mkdir(destinationDir, { recursive: true });
			await copyFile(source, join(destinationDir, name));
		}
	}
}

async function prepareCase(
	sourceWorkspace: string,
	outputDir: string,
	runId: string,
	resume: boolean,
): Promise<PreparedCase> {
	await requireDirectory(sourceWorkspace, "Source workspace");
	const [rawJob, result] = await Promise.all([
		readJson<AgentJob>(join(sourceWorkspace, JOB_FILE)),
		readJson<AgentResult>(join(sourceWorkspace, "agent_result.json")),
	]);
	if (rawJob.operation !== "run_fixture" || rawJob.fixture.id !== result.fixture_id) {
		throw new UsageError(`Source job/result fixture mismatch in ${sourceWorkspace}`);
	}
	if (
		result.status !== "valid" ||
		!result.last_executed_source_sha256 ||
		!result.last_executed_step_sha256 ||
		result.artifact?.sha256 !== result.last_executed_step_sha256
	) {
		throw new UsageError(`Source fixture ${result.fixture_id} has no current valid formal STEP`);
	}
	const workspace = join(outputDir, result.fixture_id);
	const workspaceMetadata = await lstat(workspace).catch(() => null);
	if (workspaceMetadata) {
		if (!resume) throw new UsageError(`Prepared Case already exists: ${workspace}`);
		if (!workspaceMetadata.isDirectory() || workspaceMetadata.isSymbolicLink()) {
			throw new UsageError(`Prepared Case workspace is unsafe: ${workspace}`);
		}
		const [sourceRecord, copiedResult] = await Promise.all([
			readJson<{ source_workspace?: string }>(join(workspace, "source_workspace.json")),
			readJson<AgentResult>(join(workspace, "source_agent_result.json")),
		]);
		if (sourceRecord.source_workspace !== sourceWorkspace) {
			throw new UsageError(`Prepared Case ${result.fixture_id} belongs to another source workspace`);
		}
		if (
			copiedResult.fixture_id !== result.fixture_id ||
			copiedResult.last_executed_source_sha256 !== result.last_executed_source_sha256 ||
			copiedResult.last_executed_step_sha256 !== result.last_executed_step_sha256
		) {
			throw new UsageError(`Prepared Case ${result.fixture_id} no longer matches its source result`);
		}
		for (const path of [SOLUTION_FILE, "output.step", ...rawJob.staged_paths]) {
			const safePath = safeRelativePath(path, "Prepared file path");
			const metadata = await lstat(join(workspace, safePath)).catch(() => null);
			if (!metadata?.isFile() || metadata.isSymbolicLink()) {
				throw new UsageError(`Prepared Case file is missing or unsafe: ${join(workspace, safePath)}`);
			}
		}
	} else {
		await mkdir(workspace, { recursive: false });
		for (const path of [SOLUTION_FILE, "output.step", ...rawJob.staged_paths]) {
			await copyRegularFile(sourceWorkspace, workspace, path);
		}
	}
	const job: AgentJob = {
		...rawJob,
		protocol_version: PROTOCOL_VERSION,
		run_id: runId,
		agent_mode: "drawing-subagents",
	};
	await Promise.all([
		writeJsonAtomic(join(workspace, JOB_FILE), job),
		writeJsonAtomic(join(workspace, "source_agent_result.json"), result),
		writeJsonAtomic(join(workspace, "source_workspace.json"), { source_workspace: sourceWorkspace }),
	]);
	return { sourceWorkspace, workspace, job, result };
}

async function runCase(options: {
	prepared: PreparedCase;
	worker: LocalWorker;
	resolvedModel: Awaited<ReturnType<typeof resolveVisionModel>>;
	controller: RunController;
}): Promise<Record<string, unknown>> {
	const startedAt = isoNow();
	const usages: SubagentRunUsage[] = [];
	const progressPath = join(options.prepared.workspace, "view_search_only_progress.json");
	try {
		const search = await runDrawingViewSearchOnly({
			job: options.prepared.job,
			worker: options.worker,
			workspace: options.prepared.workspace,
			resolvedModel: options.resolvedModel,
			controller: options.controller,
			getResult: () => ({ ...options.prepared.result, subagent_runs: usages }),
			onUsage: async (usage) => {
				const existing = usages.findIndex(({ accounting_id }) => accounting_id === usage.accounting_id);
				if (existing >= 0) usages[existing] = usage;
				else usages.push(usage);
				await writeJsonAtomic(progressPath, {
					fixture_id: options.prepared.job.fixture.id,
					status: "running",
					started_at: startedAt,
					updated_at: isoNow(),
					usage: usages,
				});
			},
		});
		const output = {
			status: "completed",
			started_at: startedAt,
			finished_at: isoNow(),
			duration_s: elapsedSeconds(startedAt),
			source_workspace: options.prepared.sourceWorkspace,
			agent_roles: ["segmentation", "view_search"],
			reflection_agent_started: false,
			...search,
		};
		await writeJsonAtomic(join(options.prepared.workspace, "view_search_only_result.json"), output);
		return output;
	} catch (error) {
		const output = {
			fixture_id: options.prepared.job.fixture.id,
			status: options.controller.interrupted ? "interrupted" : "error",
			started_at: startedAt,
			finished_at: isoNow(),
			duration_s: elapsedSeconds(startedAt),
			source_workspace: options.prepared.sourceWorkspace,
			agent_roles: ["segmentation", "view_search"],
			reflection_agent_started: false,
			error: errorMessage(error),
			usage: usages,
		};
		await writeJsonAtomic(join(options.prepared.workspace, "view_search_only_result.json"), output);
		throw Object.assign(new Error(errorMessage(error)), { result: output });
	}
}

export async function main(arguments_ = process.argv.slice(2)): Promise<number> {
	const parsed = parseArguments(arguments_);
	await prepareOutputDirectory(parsed.outputDir, parsed.resume);
	if (parsed.resume) await archivePreviousRun(parsed.outputDir);
	const startedAt = isoNow();
	const runId = `drawing-search-only-${startedAt.replaceAll(/[-:.]/gu, "")}-${process.pid}`;
	const preparedCases = await Promise.all(
		parsed.sourceWorkspaces.map(
			async (workspace) => await prepareCase(workspace, parsed.outputDir, runId, parsed.resume),
		),
	);
	if (new Set(preparedCases.map(({ job }) => job.fixture.id)).size !== preparedCases.length) {
		throw new UsageError("Source workspaces must have unique fixture IDs");
	}
	const modelKeys = new Set(preparedCases.map(({ job }) => `${job.provider}\0${job.model}`));
	if (modelKeys.size !== 1) throw new UsageError("All source workspaces must use the same provider/model");
	const firstJob = preparedCases[0]!.job;
	const resolvedModel = await resolveVisionModel(firstJob.provider, firstJob.model);
	const worker = new LocalWorker(parsed.pythonBinary);
	const controllers = preparedCases.map(() => new RunController());
	const onSigint = (): void => {
		void Promise.allSettled(controllers.map(async (controller) => await controller.interrupt("SIGINT")));
	};
	process.once("SIGINT", onSigint);
	try {
		const settled = await Promise.allSettled(
			preparedCases.map(async (prepared, index) => await runCase({
				prepared,
				worker,
				resolvedModel,
				controller: controllers[index]!,
			})),
		);
		const cases = settled.map((outcome, index) =>
			outcome.status === "fulfilled"
				? outcome.value
				: ((outcome.reason as { result?: Record<string, unknown> }).result ?? {
						fixture_id: preparedCases[index]!.job.fixture.id,
						status: "error",
						error: errorMessage(outcome.reason),
					}),
		);
		const completed = cases.filter(({ status }) => status === "completed").length;
		const summary = {
			run_id: runId,
			status: completed === cases.length ? "completed" : completed > 0 ? "partial" : "error",
			started_at: startedAt,
			finished_at: isoNow(),
			duration_s: elapsedSeconds(startedAt),
			provider: firstJob.provider,
			model: firstJob.model,
			python: parsed.pythonBinary,
			case_concurrency: preparedCases.length,
			reflection_agent_started: false,
			cases,
		};
		await writeJsonAtomic(join(parsed.outputDir, "drawing_search_only_summary.json"), summary);
		process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
		return completed === cases.length ? 0 : 2;
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
		process.exitCode = 2;
	},
);
