#!/usr/bin/env node
import { lstat } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { errorMessage } from "./errors.js";
import { readJson, writeJsonAtomic } from "./files.js";
import { LocalWorker } from "./local-worker.js";
import { resolveVisionModel, RunController, runFixtureInContainer } from "./runner.js";
import {
	JOB_FILE,
	PI_VERSION,
	PROTOCOL_VERSION,
	RESPONSE_FILE,
	type ContainerJob,
	type ContainerResponse,
	type WorkerDoctorResponse,
} from "./types.js";

const WORKSPACE = resolve(process.env.CADGENBENCH_PI_WORKSPACE ?? "/workspace");
const moduleDirectory = dirname(fileURLToPath(import.meta.url));
const integrationRoot = resolve(moduleDirectory, "..", "..");
const repositoryRoot = resolve(
	process.env.CADGENBENCH_REPOSITORY_ROOT ?? resolve(integrationRoot, "..", ".."),
);

async function writeResponse(response: ContainerResponse): Promise<void> {
	await writeJsonAtomic(join(WORKSPACE, RESPONSE_FILE), response);
}

async function doctor(job: Extract<ContainerJob, { operation: "doctor" }>): Promise<void> {
	const [nodeMajor = 0, nodeMinor = 0] = process.versions.node.split(".").map((part) => Number.parseInt(part, 10));
	if (nodeMajor < 22 || (nodeMajor === 22 && nodeMinor < 19)) {
		throw new Error(`Node ${process.versions.node} is too old; expected >=22.19`);
	}
	const resolved = await resolveVisionModel(job.provider, job.model);
	const networkProbe = await resolved.runtime.completeSimple(
		resolved.model,
		{
			messages: [
				{ role: "user", content: "Reply with exactly: OK", timestamp: Date.now() },
			],
		},
		{ reasoning: "low" },
	);
	if (networkProbe.stopReason === "error") {
		throw new Error(`Pi provider network/authentication probe failed: ${networkProbe.errorMessage ?? "unknown error"}`);
	}
	const worker = new LocalWorker();
	const result = await worker.request<WorkerDoctorResponse>(
		WORKSPACE,
		{ protocol_version: PROTOCOL_VERSION, operation: "doctor" },
		{ timeoutMs: 300_000 },
	);
	if (
		!result.ok ||
		result.artifact.status !== "valid" ||
		!result.artifact.render_path ||
		!result.drawing_2d.ok ||
		!result.drawing_2d.sheet_path
	) {
		throw new Error("build123d validation, VTK rendering, or deterministic 2D drawing pipeline failed during doctor");
	}
	if (result.drawing_2d.ok) {
		const sheetPath = resolve(WORKSPACE, result.drawing_2d.sheet_path);
		const relativeSheet = relative(WORKSPACE, sheetPath);
		const metadata =
			relativeSheet !== ".." && !relativeSheet.startsWith("../") && !isAbsolute(relativeSheet)
				? await lstat(sheetPath).catch(() => null)
				: null;
		if (!metadata?.isFile() || metadata.isSymbolicLink()) {
			throw new Error("2D doctor reported a missing or unsafe comparison sheet artifact");
		}
	}
	if (!result.python.startsWith("3.12.")) {
		throw new Error(`Python ${result.python} does not match the required 3.12 runtime`);
	}
	if (result.build123d !== "0.10.0") {
		throw new Error(`build123d ${result.build123d} does not match the required 0.10.0`);
	}
	await writeResponse({
		protocol_version: PROTOCOL_VERSION,
		operation: "doctor",
		ok: true,
		model_inputs: [...resolved.model.input],
		node: process.versions.node,
		pi_version: PI_VERSION,
		doctor: result,
	});
}

async function runFixture(job: Extract<ContainerJob, { operation: "run_fixture" }>): Promise<number> {
	const controller = new RunController();
	let interrupted = false;
	const onSigint = (): void => {
		interrupted = true;
		void controller.interrupt("SIGINT");
	};
	process.once("SIGINT", onSigint);
	try {
		const resolvedModel = await resolveVisionModel(job.provider, job.model);
		await runFixtureInContainer({
			job,
			workspace: WORKSPACE,
			repositoryRoot,
			integrationRoot,
			worker: new LocalWorker(),
			resolvedModel,
			controller,
		});
		await writeResponse({
			protocol_version: PROTOCOL_VERSION,
			operation: "run_fixture",
			ok: true,
		});
		return interrupted ? 130 : 0;
	} finally {
		process.removeListener("SIGINT", onSigint);
	}
}

export async function main(): Promise<number> {
	const job = await readJson<ContainerJob>(join(WORKSPACE, JOB_FILE));
	if (job.protocol_version !== PROTOCOL_VERSION) {
		throw new Error(`Unsupported job protocol ${String(job.protocol_version)}; expected ${PROTOCOL_VERSION}`);
	}
	if (job.operation === "doctor") {
		await doctor(job);
		return 0;
	}
	if (job.operation === "run_fixture") return await runFixture(job);
	throw new Error(`Unsupported container operation: ${String((job as { operation?: unknown }).operation)}`);
}

main()
	.then((code) => {
		process.exitCode = code;
	})
	.catch(async (error: unknown) => {
		const message = errorMessage(error);
		const job = await readJson<ContainerJob>(join(WORKSPACE, JOB_FILE)).catch(() => null);
		if (job) {
			await writeResponse({
				protocol_version: PROTOCOL_VERSION,
				operation: job.operation,
				ok: false,
				error: message,
			}).catch(() => undefined);
		}
		process.stderr.write(`${message}\n`);
		process.exitCode = 2;
	});
