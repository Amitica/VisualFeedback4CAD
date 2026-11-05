#!/usr/bin/env node
import { mkdir, mkdtemp, readdir, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { DockerAgentRuntime } from "./docker-worker.js";
import { errorMessage, InfrastructureError, UsageError } from "./errors.js";
import { discoverGenerationFixtures, defaultDataInputsDir, loadFixture, stageFixtureInputs } from "./fixtures.js";
import { elapsedSeconds, isoNow, readJson, writeJsonAtomic } from "./files.js";
import { migrateRunToV2 } from "./migration.js";
import { loadAgentResult, saveAgentResult, updateRunSummary } from "./results.js";
import { ensureSolution, recordedCandidateAccepted } from "./solution.js";
import {
	DEFAULT_IMAGE,
	PI_VERSION,
	PROTOCOL_VERSION,
	SMOKE_FIXTURES,
	type AgentJob,
	type AgentResult,
	type LegacyRunParams,
	type RunParams,
	type WorkerDoctorResponse,
} from "./types.js";

type FlagValue = string | boolean;

interface ParsedArguments {
	positionals: string[];
	flags: Map<string, FlagValue>;
}

const moduleDirectory = dirname(fileURLToPath(import.meta.url));
const integrationRoot = resolve(moduleDirectory, "..", "..");
const repositoryRoot = resolve(integrationRoot, "..", "..");
const THINKING_LEVELS = new Set(["off", "minimal", "low", "medium", "high"]);

const HELP = `cadgenbench-pi - Pi × CADGenBench persistent Agent containers

Usage:
  cadgenbench-pi doctor --provider <provider> --model <model> [--image <tag>]
  cadgenbench-pi run --smoke --provider <provider> --model <model>
  cadgenbench-pi run 127 133 145 --provider <provider> --model <model>
  cadgenbench-pi run --all-generation --parallel 1 --provider <provider> --model <model>
  cadgenbench-pi resume results/pi/<run-id>

Options:
  --provider <id>             Required Pi provider
  --model <id>                Required Pi model; must declare image input
  --thinking-level <level>    off|minimal|low|medium|high (default: medium)
  --parallel <N>              Concurrent fixture containers (default: 1; smoke requires 1)
  --base-agent-only           Do not expose the project drawing SubAgent tool
  --image <tag>               Agent image (default: ${DEFAULT_IMAGE})
  --data-inputs-dir <path>    CADGenBench data/inputs directory
  --output-dir <path>         Exact output run directory

Containers are retained after completion or interruption. Manage them explicitly with Docker.
`;

function parseArguments(args: string[], booleanFlags: Set<string>): ParsedArguments {
	const positionals: string[] = [];
	const flags = new Map<string, FlagValue>();
	for (let index = 0; index < args.length; index += 1) {
		const token = args[index];
		if (token === undefined) continue;
		if (!token.startsWith("--")) {
			positionals.push(token);
			continue;
		}
		const equalIndex = token.indexOf("=");
		const name = equalIndex === -1 ? token.slice(2) : token.slice(2, equalIndex);
		if (flags.has(name)) throw new UsageError(`Option --${name} was provided more than once`);
		if (booleanFlags.has(name)) {
			if (equalIndex !== -1) throw new UsageError(`Boolean option --${name} does not take a value`);
			flags.set(name, true);
			continue;
		}
		const value = equalIndex === -1 ? args[index + 1] : token.slice(equalIndex + 1);
		if (value === undefined || value.startsWith("--")) throw new UsageError(`Option --${name} requires a value`);
		if (equalIndex === -1) index += 1;
		flags.set(name, value);
	}
	return { positionals, flags };
}

function rejectUnknown(flags: Map<string, FlagValue>, allowed: Set<string>): void {
	for (const name of flags.keys()) if (!allowed.has(name)) throw new UsageError(`Unknown option --${name}`);
}

function requiredFlag(parsed: ParsedArguments, name: string): string {
	const value = parsed.flags.get(name);
	if (typeof value !== "string" || value.trim() === "") throw new UsageError(`--${name} is required`);
	return value;
}

function stringFlag(parsed: ParsedArguments, name: string, fallback: string): string {
	const value = parsed.flags.get(name);
	return typeof value === "string" ? value : fallback;
}

function positiveIntegerFlag(parsed: ParsedArguments, name: string, fallback: number): number {
	const raw = parsed.flags.get(name);
	if (raw === undefined) return fallback;
	if (typeof raw !== "string" || !/^\d+$/u.test(raw) || Number(raw) < 1) {
		throw new UsageError(`--${name} must be an integer >= 1`);
	}
	return Number(raw);
}

function makeRunId(model: string): string {
	const timestamp = new Date().toISOString().replaceAll(/[-:.]/gu, "").replace("Z", "Z");
	const modelSlug = model.replaceAll(/[^A-Za-z0-9._-]+/gu, "-").slice(0, 48);
	return `${timestamp}-${modelSlug}-${process.pid}`;
}

async function prepareNewRunDirectory(runDir: string): Promise<void> {
	const existing = await stat(runDir).catch(() => null);
	if (existing === null) {
		await mkdir(runDir, { recursive: true });
		return;
	}
	if (!existing.isDirectory()) throw new UsageError(`--output-dir is not a directory: ${runDir}`);
	if ((await readdir(runDir)).length > 0) {
		throw new UsageError(`Refusing to start a new run in non-empty directory: ${runDir}`);
	}
}

async function hardDockerPreflight(
	runtime: DockerAgentRuntime,
	provider: string,
	model: string,
): Promise<{
	daemonVersion: string;
	digest: string;
	doctor: WorkerDoctorResponse;
	modelInputs: string[];
	node: string;
}> {
	const daemonVersion = await runtime.daemonVersion();
	const digest = await runtime.imageDigest();
	const workspace = await mkdtemp(join(tmpdir(), "cadgenbench-pi-doctor-"));
	try {
		const response = await runtime.doctor(workspace, {
			protocol_version: PROTOCOL_VERSION,
			operation: "doctor",
			provider,
			model,
		});
		const doctor = response.doctor;
		if (
			!doctor?.ok ||
			doctor.artifact.status !== "valid" ||
			!doctor.artifact.render_path ||
			!doctor.drawing_2d.ok ||
			!doctor.drawing_2d.sheet_path
		) {
			throw new InfrastructureError(
				"Agent image doctor failed build123d validation, VTK rendering, or deterministic 2D drawing comparison",
			);
		}
		return {
			daemonVersion,
			digest,
			doctor,
			modelInputs: response.model_inputs ?? [],
			node: response.node ?? "unknown",
		};
	} finally {
		await rm(workspace, { recursive: true, force: true });
	}
}

async function doctorCommand(args: string[]): Promise<number> {
	const parsed = parseArguments(args, new Set());
	rejectUnknown(parsed.flags, new Set(["provider", "model", "image"]));
	if (parsed.positionals.length > 0) throw new UsageError("doctor does not accept positional arguments");
	const provider = requiredFlag(parsed, "provider");
	const model = requiredFlag(parsed, "model");
	const image = stringFlag(parsed, "image", DEFAULT_IMAGE);
	const runtime = new DockerAgentRuntime({ image, provider });
	let interrupted = false;
	let interruptPromise: Promise<void> = Promise.resolve();
	const onSigint = (): void => {
		interrupted = true;
		interruptPromise = runtime.interruptAll().catch(() => undefined);
	};
	process.once("SIGINT", onSigint);
	try {
		const preflight = await hardDockerPreflight(runtime, provider, model);
		process.stdout.write(
			`${JSON.stringify({
			ok: true,
			provider,
			model,
			model_inputs: preflight.modelInputs,
			pi_version: PI_VERSION,
			container_image: image,
			container_image_digest: preflight.digest,
			docker_server_version: preflight.daemonVersion,
			node: preflight.node,
			python: preflight.doctor.python,
			build123d: preflight.doctor.build123d,
			box_validation: preflight.doctor.artifact.validation,
			vtk_render: preflight.doctor.artifact.render_path !== null,
			drawing_2d: preflight.doctor.drawing_2d,
			}, null, 2)}\n`,
		);
		return 0;
	} catch (error) {
		if (interrupted) {
			await interruptPromise.catch(() => undefined);
			return 130;
		}
		throw error;
	} finally {
		process.removeListener("SIGINT", onSigint);
	}
}

async function selectRun(parsed: ParsedArguments): Promise<{
	fixtureIds: string[];
	mode: RunParams["mode"];
	dataInputsDir: string;
}> {
	const dataInputsDir = resolve(stringFlag(parsed, "data-inputs-dir", defaultDataInputsDir(repositoryRoot)));
	const smoke = parsed.flags.get("smoke") === true;
	const all = parsed.flags.get("all-generation") === true;
	const modes = Number(smoke) + Number(all) + Number(parsed.positionals.length > 0);
	if (modes !== 1) throw new UsageError("Choose exactly one of positional fixtures, --smoke, or --all-generation");
	const fixtureIds = smoke
		? [...SMOKE_FIXTURES]
		: all
			? await discoverGenerationFixtures(dataInputsDir)
			: parsed.positionals;
	if (fixtureIds.length === 0) throw new UsageError("No generation fixtures were selected");
	if (new Set(fixtureIds).size !== fixtureIds.length) throw new UsageError("Fixture IDs must be unique");
	for (const fixtureId of fixtureIds) await loadFixture(dataInputsDir, fixtureId);
	return { fixtureIds, mode: smoke ? "smoke" : all ? "all-generation" : "fixtures", dataInputsDir };
}

async function stageNewWorkspace(runDir: string, params: RunParams, fixtureId: string): Promise<void> {
	const workspace = join(runDir, fixtureId);
	await mkdir(workspace, { recursive: true });
	await ensureSolution(workspace);
	const fixture = await loadFixture(params.data_inputs_dir, fixtureId);
	await stageFixtureInputs(fixture, workspace);
}

function emptyResult(params: RunParams, fixtureId: string): AgentResult {
	const now = isoNow();
	return {
		fixture_id: fixtureId,
		status: "running",
		agent_mode: params.agent_mode ?? "drawing-subagents",
		provider: params.provider,
		model: params.model,
		thinking_level: params.thinking_level,
		pi_version: params.pi_version,
		container_image: params.container_image,
		container_image_digest: params.container_image_digest,
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

async function recordInfrastructureError(
	runDir: string,
	params: RunParams,
	fixtureId: string,
	error: unknown,
): Promise<void> {
	const result = (await loadAgentResult(runDir, fixtureId)) ?? emptyResult(params, fixtureId);
	result.status = "infrastructure_error";
	result.stop_reason = errorMessage(error);
	result.updated_at = isoNow();
	result.duration_s = elapsedSeconds(result.started_at);
	await saveAgentResult(runDir, result);
}

async function recordInterrupted(runDir: string, params: RunParams, fixtureId: string): Promise<void> {
	const result = (await loadAgentResult(runDir, fixtureId)) ?? emptyResult(params, fixtureId);
	if (result.status !== "valid") {
		result.status = "interrupted";
		result.stop_reason = "SIGINT";
		await saveAgentResult(runDir, result);
	}
}

async function makeJob(runDir: string, params: RunParams, fixtureId: string): Promise<AgentJob> {
	const fixture = await loadFixture(params.data_inputs_dir, fixtureId);
	return {
		protocol_version: PROTOCOL_VERSION,
		operation: "run_fixture",
		run_id: params.run_id,
		agent_mode: params.agent_mode ?? "drawing-subagents",
		fixture: { ...fixture, sourceDir: "" },
		staged_paths: fixture.inputFiles.map((inputFile, index) =>
			join("inputs", `${String(index).padStart(2, "0")}-${inputFile.split(/[\\/]/u).at(-1)}`),
		),
		provider: params.provider,
		model: params.model,
		thinking_level: params.thinking_level,
		pi_version: params.pi_version,
		container_image: params.container_image,
		container_image_digest: params.container_image_digest,
	};
}

async function executeRun(runDir: string, params: RunParams, runtime: DockerAgentRuntime): Promise<number> {
	let interrupted = false;
	let fatal: unknown;
	let interruptFailure: unknown;
	let interruptPromise: Promise<void> = Promise.resolve();
	const onSigint = (): void => {
		if (interrupted) return;
		interrupted = true;
		interruptPromise = runtime.interruptAll().catch((error: unknown) => {
			interruptFailure = error;
		});
	};
	process.once("SIGINT", onSigint);
	let cursor = 0;
	try {
		const workers = Array.from(
			{ length: Math.min(params.parallel, params.fixtures.length) },
			async () => {
				while (!interrupted && fatal === undefined) {
					const fixtureId = params.fixtures[cursor];
					cursor += 1;
					if (fixtureId === undefined) return;
					const existing = await loadAgentResult(runDir, fixtureId);
					if (
						existing?.status === "valid" &&
						(await recordedCandidateAccepted(join(runDir, fixtureId), existing))
					) continue;
					try {
						const job = await makeJob(runDir, params, fixtureId);
						await runtime.runFixture(join(runDir, fixtureId), job, params.container_image_digest);
						await updateRunSummary(runDir, params);
					} catch (error) {
						if (interrupted) await recordInterrupted(runDir, params, fixtureId);
						else await recordInfrastructureError(runDir, params, fixtureId, error);
						await updateRunSummary(runDir, params, errorMessage(error));
						if (!interrupted) {
							fatal = error;
							await runtime.interruptAll();
						}
					}
				}
			},
		);
		await Promise.all(workers);
		await interruptPromise;
		if (interruptFailure !== undefined) throw interruptFailure;
		if (fatal !== undefined) throw fatal;
		const summary = await updateRunSummary(runDir, params, interrupted ? "SIGINT" : undefined);
		process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
		return summary.valid_count === params.fixtures.length ? 0 : interrupted ? 130 : 2;
	} finally {
		process.removeListener("SIGINT", onSigint);
	}
}

async function runCommand(args: string[]): Promise<number> {
	const parsed = parseArguments(args, new Set(["smoke", "all-generation", "base-agent-only"]));
	rejectUnknown(
		parsed.flags,
		new Set([
			"provider",
			"model",
			"thinking-level",
			"parallel",
			"image",
			"data-inputs-dir",
			"output-dir",
			"smoke",
			"all-generation",
			"base-agent-only",
		]),
	);
	const provider = requiredFlag(parsed, "provider");
	const model = requiredFlag(parsed, "model");
	const thinkingLevel = stringFlag(parsed, "thinking-level", "medium");
	if (!THINKING_LEVELS.has(thinkingLevel)) {
		throw new UsageError("--thinking-level must be off|minimal|low|medium|high");
	}
	const parallel = positiveIntegerFlag(parsed, "parallel", 1);
	const agentMode = parsed.flags.get("base-agent-only") === true ? "base-only" : "drawing-subagents";
	const selected = await selectRun(parsed);
	if (selected.mode === "smoke" && parallel !== 1) throw new UsageError("--smoke requires --parallel 1");
	const image = stringFlag(parsed, "image", DEFAULT_IMAGE);
	const runId = makeRunId(model);
	const runDir = resolve(stringFlag(parsed, "output-dir", join(repositoryRoot, "results", "pi", runId)));
	await prepareNewRunDirectory(runDir);
	const runtime = new DockerAgentRuntime({ image, provider });
	const preflight = await hardDockerPreflight(runtime, provider, model);
	const params: RunParams = {
		protocol_version: PROTOCOL_VERSION,
		runtime_mode: "persistent-agent-container",
		initialization_status: "pending",
		run_id: runId,
		created_at: isoNow(),
		agent_mode: agentMode,
		provider,
		model,
		thinking_level: thinkingLevel,
		pi_version: PI_VERSION,
		container_image: image,
		container_image_digest: preflight.digest,
		data_inputs_dir: selected.dataInputsDir,
		fixtures: selected.fixtureIds,
		parallel,
		mode: selected.mode,
	};
	await writeJsonAtomic(join(runDir, "params.json"), params);
	for (const fixtureId of params.fixtures) {
		await stageNewWorkspace(runDir, params, fixtureId);
		await runtime.prepareFixture(join(runDir, fixtureId), params.run_id, fixtureId, params.container_image_digest);
	}
	params.initialization_status = "complete";
	await writeJsonAtomic(join(runDir, "params.json"), params);
	await updateRunSummary(runDir, params);
	return await executeRun(runDir, params, runtime);
}

async function resumeCommand(args: string[]): Promise<number> {
	const parsed = parseArguments(args, new Set());
	rejectUnknown(parsed.flags, new Set());
	if (parsed.positionals.length !== 1) throw new UsageError("resume requires exactly one run directory");
	const runDir = resolve(parsed.positionals[0] as string);
	const raw = await readJson<RunParams | LegacyRunParams>(join(runDir, "params.json"));
	const wasLegacy = raw.protocol_version === 1;
	const image = raw.protocol_version === 1 ? DEFAULT_IMAGE : raw.container_image;
	const runtime = new DockerAgentRuntime({ image, provider: raw.provider });
	await runtime.daemonVersion();
	const digest = await runtime.imageDigest();
	const params = await migrateRunToV2(runDir, raw, digest);
	const initializationPending = wasLegacy || params.initialization_status === "pending";
	if (params.pi_version !== PI_VERSION || params.runtime_mode !== "persistent-agent-container") {
		throw new InfrastructureError("Run Pi version or runtime mode does not match this integration");
	}
	if (digest !== params.container_image_digest) {
		throw new InfrastructureError(
			`Container image digest changed: run=${params.container_image_digest}, current=${digest}`,
		);
	}
	for (const fixtureId of params.fixtures) {
		await loadFixture(params.data_inputs_dir, fixtureId);
		const containerExists = (await stat(join(runDir, fixtureId, "container.json")).catch(() => null)) !== null;
		if (!containerExists && initializationPending) {
			await stageNewWorkspace(runDir, params, fixtureId);
			await runtime.prepareFixture(join(runDir, fixtureId), params.run_id, fixtureId, digest);
		}
		try {
			await runtime.validateFixture(join(runDir, fixtureId), params.run_id, fixtureId, digest);
		} catch (error) {
			await recordInfrastructureError(runDir, params, fixtureId, error);
			await updateRunSummary(runDir, params, errorMessage(error));
			throw error;
		}
	}
	if (initializationPending) {
		params.initialization_status = "complete";
		await writeJsonAtomic(join(runDir, "params.json"), params);
	}
	const prior = await Promise.all(params.fixtures.map(async (id) => await loadAgentResult(runDir, id)));
	const allValid = await Promise.all(
		prior.map(async (result, index) =>
			result?.status === "valid" &&
			await recordedCandidateAccepted(join(runDir, params.fixtures[index] as string), result),
		),
	);
	if (allValid.every(Boolean)) {
		const summary = await updateRunSummary(runDir, params);
		process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
		return 0;
	}
	return await executeRun(runDir, params, runtime);
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
	const command = argv[0];
	if (command === undefined || command === "--help" || command === "-h" || command === "help") {
		process.stdout.write(HELP);
		return 0;
	}
	if (command === "doctor") return await doctorCommand(argv.slice(1));
	if (command === "run") return await runCommand(argv.slice(1));
	if (command === "resume") return await resumeCommand(argv.slice(1));
	throw new UsageError(`Unknown command: ${command}`);
}

main()
	.then((code) => {
		process.exitCode = code;
	})
	.catch((error: unknown) => {
		const prefix = error instanceof UsageError ? "usage error" : "infrastructure error";
		process.stderr.write(`${prefix}: ${errorMessage(error)}\n`);
		if (error instanceof UsageError) process.stderr.write(`\n${HELP}`);
		process.exitCode = error instanceof UsageError ? 64 : 2;
	});
