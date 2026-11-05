#!/usr/bin/env node
import { createHash } from "node:crypto";
import { cp, mkdir, readFile, readdir, stat } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";

import { DockerAgentRuntime } from "./docker-worker.js";
import { errorMessage, InfrastructureError, UsageError } from "./errors.js";
import {
	type ExperimentModelSelection,
	resolveExperimentModelSelection,
} from "./experiment-model-profile.js";
import { elapsedSeconds, isoNow, readJson, writeJsonAtomic } from "./files.js";
import {
	reflectionContinuationPrompt,
	reflectionContinuationWithout3dRenderReflectionPrompt,
} from "./prompts.js";
import {
	JOB_FILE,
	PI_VERSION,
	PROTOCOL_VERSION,
	type AgentJob,
	type AgentResult,
	type DrawingReflectionReport,
	type RunStatus,
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
	baselineSourceSha256: string;
	baselineStepSha256: string;
	baselineUsage: Pick<AgentResult, "tokens" | "cost" | "tool_calls">;
	job: AgentJob;
}

interface ContinuationAfter {
	source_sha256: string;
	step_sha256: string;
	render_path: string;
	session_file: string;
	is_valid: boolean;
	is_watertight: boolean;
}

interface ContinuationCaseResult {
	fixture_id: string;
	status: "completed" | "error" | "interrupted";
	started_at: string;
	finished_at: string;
	duration_s: number;
	source_workspace: string;
	workspace: string;
	continuation_id: string;
	baseline: {
		source_sha256: string;
		step_sha256: string;
		render_path: string;
		session_file: string;
	};
	after?: ContinuationAfter;
	agent_status?: RunStatus;
	usage_delta?: {
		tokens: number;
		cost: number;
		tool_calls: number;
	};
	error?: string;
}

const HELP = `cadgenbench-pi-reflection-continue - resume copied Pi sessions with final Reflection reports

Usage:
  cadgenbench-pi-reflection-continue <source-workspace>... \\
    --reflection-root <drawing-search-run> \\
    --output-dir <new-run-root> \\
    --image <container-tag> \\
    [--no-3d-render-reflection] \\
    [--model-profile <gpt-5.6-sol|kimi> | --provider <provider> --model <model>]

The output directory must not exist or must be empty. Original source workspaces
are read-only baselines and are never modified. The default model profile is
gpt-5.6-sol through UCloud; use --model-profile kimi for the historical model.
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
		if (!["reflection-root", "output-dir", "image", "model-profile", "provider", "model", "no-3d-render-reflection"].includes(name)) {
			throw new UsageError(`Unknown option --${name}`);
		}
		if (flags.has(name)) throw new UsageError(`Option --${name} was provided more than once`);
		if (name === "no-3d-render-reflection") {
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

async function sha256(path: string): Promise<string> {
	return createHash("sha256").update(await readFile(path)).digest("hex");
}

async function requireFile(path: string, label: string): Promise<void> {
	const metadata = await stat(path).catch(() => null);
	if (!metadata?.isFile()) throw new InfrastructureError(`${label} is missing: ${path}`);
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

async function copySourceWorkspace(source: string, destination: string): Promise<void> {
	await cp(source, destination, {
		recursive: true,
		filter: (path) => {
			if (path === source) return true;
			const name = basename(path);
			return ![
				"container.json",
				JOB_FILE,
				".cadgenbench-pi-response.json",
				"baseline",
				"after",
				"reflection_input",
				"continuation_result.json",
			].includes(name);
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
	const target = join(baselineDir, "iso.png");
	await cp(lastIso, target);
	return "baseline/iso.png";
}

async function prepareCase(options: {
	sourceWorkspace: string;
	reflectionRoot: string;
	runDir: string;
	runId: string;
	image: string;
	imageId: string;
	modelSelection: ExperimentModelSelection;
	without3dRenderReflection: boolean;
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
	const sourceResult = await readJson<AgentResult>(sourceResultPath);
	const sourceJob = await readJson<AgentJob>(sourceJobPath);
	const fixtureId = sourceResult.fixture_id;
	if (sourceResult.status !== "valid" || sourceJob.fixture.id !== fixtureId) {
		throw new InfrastructureError(`Source workspace ${sourceWorkspace} is not a valid matching Pi run`);
	}
	if (!sourceResult.session_file) throw new InfrastructureError(`Case ${fixtureId} has no resumable Pi session`);
	await requireFile(join(sourceWorkspace, sourceResult.session_file), `Case ${fixtureId} Pi session`);

	const reflectionWorkspace = join(options.reflectionRoot, fixtureId);
	const acceptance = await readJson<{
		status: string;
		accepted_report_path: string;
		current_sheet_path: string;
	}>(join(reflectionWorkspace, "reflection_acceptance_result.json")).catch(() => null);
	const refined = await readJson<{
		status: string;
		refined_report_path: string;
	}>(join(reflectionWorkspace, "reflection_refinement_result.json")).catch(() => null);
	const reflectionOnly = acceptance?.status === "completed" || refined?.status === "completed" ? null : await readJson<{
		status: string;
		summary: { report_path: string; sheet_path: string };
	}>(join(reflectionWorkspace, "reflection_only_result.json"));
	if (acceptance?.status !== "completed" && refined?.status !== "completed" && reflectionOnly?.status !== "completed") {
		throw new InfrastructureError(`Case ${fixtureId} Reflection is not completed`);
	}
	const reflectionPath = join(
		reflectionWorkspace,
		acceptance?.status === "completed"
			? acceptance.accepted_report_path
			: refined?.status === "completed"
				? refined.refined_report_path
				: reflectionOnly!.summary.report_path,
	);
	const sheetPath = join(
		reflectionWorkspace,
		acceptance?.status === "completed"
			? acceptance.current_sheet_path
			: refined?.status === "completed"
				? "reflection_refinement/baseline/comparison.png"
				: reflectionOnly!.summary.sheet_path,
	);
	await Promise.all([
		requireFile(reflectionPath, `Case ${fixtureId} Reflection report`),
		requireFile(sheetPath, `Case ${fixtureId} Reflection sheet`),
	]);
	const report = await readJson<DrawingReflectionReport>(reflectionPath);
	if (!report.should_revise) throw new InfrastructureError(`Case ${fixtureId} Reflection does not request revision`);

	const workspace = join(options.runDir, fixtureId);
	await copySourceWorkspace(sourceWorkspace, workspace);
	const baselineDir = join(workspace, "baseline");
	const reflectionInputDir = join(workspace, "reflection_input");
	await Promise.all([mkdir(baselineDir, { recursive: true }), mkdir(reflectionInputDir, { recursive: true })]);
	const baselineSourceSha256 = await sha256(join(sourceWorkspace, "solution.py"));
	const baselineStepSha256 = await sha256(join(sourceWorkspace, "output.step"));
	if (
		baselineSourceSha256 !== sourceResult.last_executed_source_sha256 ||
		baselineStepSha256 !== sourceResult.last_executed_step_sha256
	) {
		throw new InfrastructureError(`Case ${fixtureId} source/STEP no longer match its recorded formal result`);
	}
	await Promise.all([
		cp(join(sourceWorkspace, "solution.py"), join(baselineDir, "solution.py")),
		cp(join(sourceWorkspace, "output.step"), join(baselineDir, "output.step")),
		cp(sourceResultPath, join(baselineDir, "agent_result.json")),
		cp(join(sourceWorkspace, sourceResult.session_file), join(baselineDir, "session.jsonl")),
		cp(reflectionPath, join(reflectionInputDir, "reflection.json")),
		cp(sheetPath, join(reflectionInputDir, "comparison.png")),
	]);
	const baselineRenderPath = await copyBaselineRenders(sourceWorkspace, baselineDir, sourceResult.last_turn);
	const prompt = options.without3dRenderReflection
		? reflectionContinuationWithout3dRenderReflectionPrompt(fixtureId, report)
		: reflectionContinuationPrompt(fixtureId, report);
	const reflectionSha256 = await sha256(join(reflectionInputDir, "reflection.json"));
	const continuationId = `reflection-${fixtureId}-${reflectionSha256.slice(0, 16)}`;
	await writeJsonAtomic(join(baselineDir, "manifest.json"), {
		fixture_id: fixtureId,
		created_at: isoNow(),
		source_workspace: sourceWorkspace,
		session_file: sourceResult.session_file,
		session_id: basename(sourceResult.session_file, ".jsonl").split("_").at(-1),
		source_sha256: baselineSourceSha256,
		step_sha256: baselineStepSha256,
		render_path: baselineRenderPath,
		reflection_sha256: reflectionSha256,
		continuation_id: continuationId,
		baseline_provider: sourceJob.provider,
		baseline_model: sourceJob.model,
		continuation_model_profile: options.modelSelection.profile,
		continuation_provider: options.modelSelection.provider,
		continuation_model: options.modelSelection.model,
		continuation_prompt_profile: options.without3dRenderReflection
			? "reflection-report-no-3d-render-reflection-v1"
			: "reflection-report-default-v1",
	});

	const now = isoNow();
	const copiedResult: AgentResult = {
		...sourceResult,
		status: "valid",
		provider: options.modelSelection.provider,
		model: options.modelSelection.model,
		thinking_level: "high",
		container_image: options.image,
		container_image_digest: options.imageId,
		started_at: now,
		updated_at: now,
		duration_s: 0,
	};
	delete copiedResult.stop_reason;
	delete copiedResult.resume_reason;
	delete copiedResult.applied_continuation;
	await writeJsonAtomic(join(workspace, "agent_result.json"), copiedResult);

	const job: AgentJob = {
		...sourceJob,
		protocol_version: PROTOCOL_VERSION,
		run_id: options.runId,
		agent_mode: "base-only",
		provider: options.modelSelection.provider,
		model: options.modelSelection.model,
		thinking_level: "high",
		pi_version: PI_VERSION,
		container_image: options.image,
		container_image_digest: options.imageId,
		continuation: {
			kind: "reflection_report",
			id: continuationId,
			prompt,
			image_paths: ["reflection_input/comparison.png"],
			baseline_source_sha256: baselineSourceSha256,
			baseline_step_sha256: baselineStepSha256,
		},
	};
	await writeJsonAtomic(join(workspace, JOB_FILE), job);
	return {
		fixtureId,
		workspace,
		sourceWorkspace,
		continuationId,
		baselineSourceSha256,
		baselineStepSha256,
		baselineUsage: {
			tokens: sourceResult.tokens,
			cost: sourceResult.cost,
			tool_calls: sourceResult.tool_calls,
		},
		job,
	};
}

async function captureAfter(prepared: PreparedCase): Promise<ContinuationAfter> {
	const result = await readJson<AgentResult>(join(prepared.workspace, "agent_result.json"));
	if (!result.session_file) throw new InfrastructureError(`Case ${prepared.fixtureId} lost its Pi session path`);
	const afterDir = join(prepared.workspace, "after");
	await mkdir(afterDir, { recursive: true });
	const sourceSha256 = await sha256(join(prepared.workspace, "solution.py"));
	const stepSha256 = await sha256(join(prepared.workspace, "output.step"));
	const isoPath = join(prepared.workspace, "turns", String(result.last_turn).padStart(4, "0"), "iso.png");
	await requireFile(isoPath, `Case ${prepared.fixtureId} final render`);
	await Promise.all([
		cp(join(prepared.workspace, "solution.py"), join(afterDir, "solution.py")),
		cp(join(prepared.workspace, "output.step"), join(afterDir, "output.step")),
		cp(join(prepared.workspace, "agent_result.json"), join(afterDir, "agent_result.json")),
		cp(join(prepared.workspace, result.session_file), join(afterDir, "session.jsonl")),
		cp(isoPath, join(afterDir, "iso.png")),
	]);
	return {
		source_sha256: sourceSha256,
		step_sha256: stepSha256,
		render_path: "after/iso.png",
		session_file: result.session_file,
		is_valid: result.artifact?.validation?.is_valid === true,
		is_watertight: result.artifact?.validation?.is_watertight === true,
	};
}

async function runCase(runtime: DockerAgentRuntime, imageId: string, prepared: PreparedCase): Promise<ContinuationCaseResult> {
	const startedAt = isoNow();
	try {
		await runtime.runFixture(prepared.workspace, prepared.job, imageId);
		const result = await readJson<AgentResult>(join(prepared.workspace, "agent_result.json"));
		const after = await captureAfter(prepared);
		if (
			result.status !== "valid" ||
			result.applied_continuation?.id !== prepared.continuationId ||
			after.source_sha256 === prepared.baselineSourceSha256 ||
			after.step_sha256 === prepared.baselineStepSha256 ||
			!after.is_valid ||
			!after.is_watertight
		) {
			throw new InfrastructureError(`Case ${prepared.fixtureId} failed the changed valid/watertight acceptance gate`);
		}
		const finishedAt = isoNow();
		return {
			fixture_id: prepared.fixtureId,
			status: "completed",
			started_at: startedAt,
			finished_at: finishedAt,
			duration_s: elapsedSeconds(startedAt),
			source_workspace: prepared.sourceWorkspace,
			workspace: prepared.workspace,
			continuation_id: prepared.continuationId,
			baseline: {
				source_sha256: prepared.baselineSourceSha256,
				step_sha256: prepared.baselineStepSha256,
				render_path: "baseline/iso.png",
				session_file: "baseline/session.jsonl",
			},
			after,
			agent_status: result.status,
			usage_delta: {
				tokens: Math.max(0, result.tokens.total - prepared.baselineUsage.tokens.total),
				cost: Math.max(0, result.cost - prepared.baselineUsage.cost),
				tool_calls: Math.max(0, result.tool_calls - prepared.baselineUsage.tool_calls),
			},
		};
	} catch (error) {
		const finishedAt = isoNow();
		return {
			fixture_id: prepared.fixtureId,
			status: "error",
			started_at: startedAt,
			finished_at: finishedAt,
			duration_s: elapsedSeconds(startedAt),
			source_workspace: prepared.sourceWorkspace,
			workspace: prepared.workspace,
			continuation_id: prepared.continuationId,
			baseline: {
				source_sha256: prepared.baselineSourceSha256,
				step_sha256: prepared.baselineStepSha256,
				render_path: "baseline/iso.png",
				session_file: "baseline/session.jsonl",
			},
			error: errorMessage(error),
		};
	}
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
	if (argv.length === 0 || argv[0] === "--help" || argv[0] === "-h") {
		process.stdout.write(HELP);
		return 0;
	}
	const parsed = parseArguments(argv);
	if (parsed.positionals.length === 0) throw new UsageError("At least one source workspace is required");
	const reflectionRoot = resolve(requiredFlag(parsed, "reflection-root"));
	const outputDir = resolve(requiredFlag(parsed, "output-dir"));
	const image = requiredFlag(parsed, "image");
	const modelSelection = resolveExperimentModelSelection({
		profile: parsed.flags.get("model-profile") ?? process.env.CADGENBENCH_PI_MODEL_PROFILE,
		provider: parsed.flags.get("provider"),
		model: parsed.flags.get("model"),
	});
	const without3dRenderReflection = parsed.flags.get("no-3d-render-reflection") === "true";
	await requireEmptyOutput(outputDir);
	const runDir = join(outputDir, "run");
	await mkdir(runDir, { recursive: true });
	const runId = `reflection-continuation-${new Date().toISOString().replaceAll(/[-:.]/gu, "")}-${process.pid}`;
	const runtime = new DockerAgentRuntime({ image, provider: modelSelection.provider });
	await runtime.daemonVersion();
	const imageId = await runtime.imageDigest();
	const startedAt = isoNow();
	let interrupted = false;
	const onSigint = (): void => {
		interrupted = true;
		void runtime.interruptAll();
	};
	process.once("SIGINT", onSigint);
	try {
		const prepared: PreparedCase[] = [];
		for (const sourceWorkspace of parsed.positionals) {
			prepared.push(await prepareCase({
				sourceWorkspace,
				reflectionRoot,
				runDir,
				runId,
				image,
				imageId,
				modelSelection,
				without3dRenderReflection,
			}));
		}
		if (new Set(prepared.map(({ fixtureId }) => fixtureId)).size !== prepared.length) {
			throw new UsageError("Source fixture IDs must be unique");
		}
		await writeJsonAtomic(join(outputDir, "params.json"), {
			protocol_version: PROTOCOL_VERSION,
			run_id: runId,
			created_at: startedAt,
			model_profile: modelSelection.profile,
			provider: modelSelection.provider,
			model: modelSelection.model,
			thinking_level: "high",
			pi_version: PI_VERSION,
			container_image: image,
			container_image_digest: imageId,
			reflection_root: reflectionRoot,
			fixtures: prepared.map(({ fixtureId }) => fixtureId),
			case_concurrency: prepared.length,
			continuation_prompt_profile: without3dRenderReflection
				? "reflection-report-no-3d-render-reflection-v1"
				: "reflection-report-default-v1",
		});
		await Promise.all(prepared.map(async ({ workspace, fixtureId }) => {
			await runtime.prepareFixture(workspace, runId, fixtureId, imageId);
		}));
		const cases = await Promise.all(prepared.map(async (item) => await runCase(runtime, imageId, item)));
		const finishedAt = isoNow();
		const summary = {
			run_id: runId,
			model_profile: modelSelection.profile,
			provider: modelSelection.provider,
			model: modelSelection.model,
			status: interrupted ? "interrupted" : cases.every(({ status }) => status === "completed") ? "completed" : "partial",
			started_at: startedAt,
			finished_at: finishedAt,
			duration_s: elapsedSeconds(startedAt),
			case_concurrency: prepared.length,
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

main()
	.then((code) => {
		process.exitCode = code;
	})
	.catch((error: unknown) => {
		const prefix = error instanceof UsageError ? "usage error" : "infrastructure error";
		process.stderr.write(`${prefix}: ${errorMessage(error)}\n`);
		process.exitCode = error instanceof UsageError ? 64 : 2;
	});
