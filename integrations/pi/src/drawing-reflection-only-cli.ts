#!/usr/bin/env node
import { lstat, mkdir } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

import { createDrawingReflectionTool } from "./drawing-reflection.js";
import { errorMessage, UsageError } from "./errors.js";
import { elapsedSeconds, isoNow, readJson, writeJsonAtomic } from "./files.js";
import { LocalWorker } from "./local-worker.js";
import { resolveVisionModel, RunController } from "./runner.js";
import {
	JOB_FILE,
	type AgentJob,
	type AgentResult,
	type DrawingFeedbackSummary,
	type SubagentRunUsage,
} from "./types.js";

interface ParsedArguments {
	workspaces: string[];
	summaryDir: string;
	pythonBinary: string;
}

interface PreparedCase {
	workspace: string;
	job: AgentJob;
	result: AgentResult;
	lockedViewManifestPath: string;
}

interface ExecutableTool {
	execute: (...arguments_: unknown[]) => Promise<{ details?: Record<string, unknown> }>;
}

const HELP = `cadgenbench-pi-reflection-only - run Reflection Agent from locked View Search results

Usage:
  cadgenbench-pi-reflection-only <prepared-case-workspace>... --summary-dir <dir> [--python <path>]

Every workspace must already contain a completed view_search_only_result.json.
Cases run concurrently. No segmentation or View Search Agent is started when the
locked caches are current.
`;

function parseArguments(arguments_: string[]): ParsedArguments {
	const workspaces: string[] = [];
	let summaryDir: string | undefined;
	let pythonBinary = "python";
	for (let index = 0; index < arguments_.length; index += 1) {
		const value = arguments_[index];
		if (!value) continue;
		if (value === "--help" || value === "-h") {
			process.stdout.write(HELP);
			process.exit(0);
		}
		if (value === "--summary-dir" || value === "--python") {
			const next = arguments_[index + 1];
			if (!next || next.startsWith("--")) throw new UsageError(`${value} requires a value`);
			if (value === "--summary-dir") summaryDir = next;
			else pythonBinary = next;
			index += 1;
			continue;
		}
		if (value.startsWith("--")) throw new UsageError(`Unknown option ${value}`);
		workspaces.push(resolve(value));
	}
	if (workspaces.length === 0) throw new UsageError("At least one prepared Case workspace is required");
	if (!summaryDir) throw new UsageError("--summary-dir is required");
	return { workspaces, summaryDir: resolve(summaryDir), pythonBinary };
}

async function prepareCase(workspace: string): Promise<PreparedCase> {
	const metadata = await lstat(workspace).catch(() => null);
	if (!metadata?.isDirectory() || metadata.isSymbolicLink()) {
		throw new UsageError(`Prepared Case workspace must be a real directory: ${workspace}`);
	}
	const [job, result, search] = await Promise.all([
		readJson<AgentJob>(join(workspace, JOB_FILE)),
		readJson<AgentResult>(join(workspace, "source_agent_result.json")),
		readJson<{ status?: string; fixture_id?: string; manifest_path?: string }>(
			join(workspace, "view_search_only_result.json"),
		),
	]);
	if (
		job.operation !== "run_fixture" ||
		job.fixture.id !== result.fixture_id ||
		search.status !== "completed" ||
		search.fixture_id !== job.fixture.id ||
		!search.manifest_path
	) {
		throw new UsageError(`Case ${basename(workspace)} does not have a completed locked View Search result`);
	}
	const manifestMetadata = await lstat(join(workspace, search.manifest_path)).catch(() => null);
	if (!manifestMetadata?.isFile() || manifestMetadata.isSymbolicLink()) {
		throw new UsageError(`Case ${job.fixture.id} View Search manifest is missing or unsafe`);
	}
	return { workspace, job, result, lockedViewManifestPath: search.manifest_path };
}

async function runCase(options: {
	prepared: PreparedCase;
	worker: LocalWorker;
	resolvedModel: Awaited<ReturnType<typeof resolveVisionModel>>;
	controller: RunController;
}): Promise<Record<string, unknown>> {
	const startedAt = isoNow();
	const usage: SubagentRunUsage[] = [];
	let summary: DrawingFeedbackSummary | undefined;
	const progressPath = join(options.prepared.workspace, "reflection_only_progress.json");
	try {
		const tool = createDrawingReflectionTool({
			job: options.prepared.job,
			worker: options.worker,
			workspace: options.prepared.workspace,
			resolvedModel: options.resolvedModel,
			controller: options.controller,
			lockedViewManifestPath: options.prepared.lockedViewManifestPath,
			getResult: () => ({ ...options.prepared.result, subagent_runs: usage }),
			onUsage: async (record) => {
				const existing = usage.findIndex(({ accounting_id }) => accounting_id === record.accounting_id);
				if (existing >= 0) usage[existing] = record;
				else usage.push(record);
				await writeJsonAtomic(progressPath, {
					fixture_id: options.prepared.job.fixture.id,
					status: "running",
					started_at: startedAt,
					updated_at: isoNow(),
					usage,
				});
			},
			onComplete: async (completed) => {
				summary = completed;
			},
		});
		const response = await (tool as unknown as ExecutableTool).execute(
			`reflection-only-${options.prepared.job.fixture.id}`,
			{},
			undefined,
			undefined,
			{},
		);
		if (!summary) throw new Error("Reflection controller completed without a summary");
		const output = {
			fixture_id: options.prepared.job.fixture.id,
			status: "completed",
			started_at: startedAt,
			finished_at: isoNow(),
			duration_s: elapsedSeconds(startedAt),
			agent_roles: ["reflection"],
			reflection_agent_started: usage.some(({ role }) => role === "reflection"),
			summary,
			usage,
			result: response.details,
		};
		await writeJsonAtomic(join(options.prepared.workspace, "reflection_only_result.json"), output);
		return output;
	} catch (error) {
		const output = {
			fixture_id: options.prepared.job.fixture.id,
			status: options.controller.interrupted ? "interrupted" : "error",
			started_at: startedAt,
			finished_at: isoNow(),
			duration_s: elapsedSeconds(startedAt),
			agent_roles: ["reflection"],
			reflection_agent_started: usage.some(({ role }) => role === "reflection"),
			error: errorMessage(error),
			usage,
		};
		await writeJsonAtomic(join(options.prepared.workspace, "reflection_only_result.json"), output);
		throw Object.assign(new Error(errorMessage(error)), { result: output });
	}
}

export async function main(arguments_ = process.argv.slice(2)): Promise<number> {
	const parsed = parseArguments(arguments_);
	await mkdir(parsed.summaryDir, { recursive: true });
	const preparedCases = await Promise.all(parsed.workspaces.map(async (workspace) => await prepareCase(workspace)));
	if (new Set(preparedCases.map(({ job }) => job.fixture.id)).size !== preparedCases.length) {
		throw new UsageError("Prepared workspaces must have unique fixture IDs");
	}
	const modelKeys = new Set(preparedCases.map(({ job }) => `${job.provider}\0${job.model}`));
	if (modelKeys.size !== 1) throw new UsageError("All Cases must use the same provider/model");
	const firstJob = preparedCases[0]!.job;
	const resolvedModel = await resolveVisionModel(firstJob.provider, firstJob.model);
	const worker = new LocalWorker(parsed.pythonBinary);
	const controllers = preparedCases.map(() => new RunController());
	const startedAt = isoNow();
	const runId = `reflection-only-${startedAt.replaceAll(/[-:.]/gu, "")}-${process.pid}`;
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
		const aggregate = {
			run_id: runId,
			status: completed === cases.length ? "completed" : completed > 0 ? "partial" : "error",
			started_at: startedAt,
			finished_at: isoNow(),
			duration_s: elapsedSeconds(startedAt),
			provider: firstJob.provider,
			model: firstJob.model,
			case_concurrency: preparedCases.length,
			cases,
		};
		await writeJsonAtomic(join(parsed.summaryDir, `${runId}.json`), aggregate);
		await writeJsonAtomic(join(parsed.summaryDir, "reflection_only_summary.json"), aggregate);
		process.stdout.write(`${JSON.stringify(aggregate, null, 2)}\n`);
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
