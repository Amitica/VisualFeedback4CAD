import { copyFile, mkdir, readdir, stat } from "node:fs/promises";
import { basename, join } from "node:path";

import { readJson, writeJsonAtomic } from "./files.js";
import { ensureSolution } from "./solution.js";
import {
	DEFAULT_IMAGE,
	PI_VERSION,
	PROTOCOL_VERSION,
	type AgentResult,
	type LegacyRunParams,
	type RunParams,
} from "./types.js";

async function copyIfMissing(source: string, destination: string): Promise<void> {
	if ((await stat(destination).catch(() => null)) === null) await copyFile(source, destination);
}

async function latestTurnCode(workspace: string): Promise<string | null> {
	const turnsDir = join(workspace, "turns");
	const entries = await readdir(turnsDir, { withFileTypes: true }).catch(() => []);
	const turns = entries
		.filter((entry) => entry.isDirectory() && /^\d+$/u.test(entry.name))
		.sort((left, right) => Number.parseInt(right.name, 10) - Number.parseInt(left.name, 10));
	for (const turn of turns) {
		const path = join(turnsDir, turn.name, "code.py");
		if ((await stat(path).catch(() => null))?.isFile()) return path;
	}
	return null;
}

async function migrateFixture(runDir: string, fixtureId: string, imageId: string): Promise<void> {
	const workspace = join(runDir, fixtureId);
	await mkdir(workspace, { recursive: true });
	const solution = join(workspace, "solution.py");
	if ((await stat(solution).catch(() => null)) === null) {
		const latest = await latestTurnCode(workspace);
		if (latest) await copyFile(latest, solution);
		else await ensureSolution(workspace);
	}
	const resultPath = join(workspace, "agent_result.json");
	const result = await readJson<AgentResult>(resultPath).catch(() => null);
	if (result === null) return;
	await copyIfMissing(resultPath, join(workspace, "agent_result.v1.json"));
	if (result.session_file) result.session_file = join("sessions", basename(result.session_file));
	delete result.last_executed_source_sha256;
	delete result.last_executed_step_sha256;
	result.pi_version = PI_VERSION;
	result.container_image = DEFAULT_IMAGE;
	result.container_image_digest = imageId;
	await writeJsonAtomic(resultPath, result);
}

export async function migrateRunToV2(
	runDir: string,
	raw: RunParams | LegacyRunParams,
	imageId: string,
): Promise<RunParams> {
	const version = Number((raw as { protocol_version?: unknown }).protocol_version);
	if (version === PROTOCOL_VERSION) return raw as RunParams;
	if (version !== 1 && version !== 3 && version !== 4) {
		throw new Error(`Unsupported run protocol: ${String(version)}`);
	}
	const paramsPath = join(runDir, "params.json");
	await copyIfMissing(paramsPath, join(runDir, `params.v${String(version)}.json`));
	for (const fixtureId of raw.fixtures) await migrateFixture(runDir, fixtureId, imageId);
	const migrated: RunParams = {
		...raw,
		protocol_version: PROTOCOL_VERSION,
		runtime_mode: "persistent-agent-container",
		initialization_status: "pending",
		pi_version: PI_VERSION,
		container_image: DEFAULT_IMAGE,
		container_image_digest: imageId,
	};
	await writeJsonAtomic(paramsPath, migrated);
	return migrated;
}
