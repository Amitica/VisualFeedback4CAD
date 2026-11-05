import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { readJson, writeJsonAtomic } from "../src/files.js";
import { migrateRunToV2 } from "../src/migration.js";
import { INITIAL_SOLUTION } from "../src/solution.js";
import type { AgentResult, LegacyRunParams, RunParams } from "../src/types.js";

const temporaryDirectories: string[] = [];

async function runDirectory(): Promise<string> {
	const directory = await mkdtemp(join(tmpdir(), "pi-migration-"));
	temporaryDirectories.push(directory);
	return directory;
}

function legacyParams(runId: string, fixtures: string[]): LegacyRunParams {
	return {
		protocol_version: 1,
		run_id: runId,
		created_at: "2026-01-02T03:04:05.000Z",
		provider: "provider",
		model: "model",
		thinking_level: "medium",
		pi_version: "0.84.1",
		container_image: "cadgenbench-pi-build123d:0.1.0",
		container_image_digest: "sha256:old",
		data_inputs_dir: "/data/inputs",
		fixtures,
		parallel: 1,
		mode: "fixtures",
	};
}

function legacyResult(fixtureId: string, sessionFile: string): AgentResult {
	return {
		fixture_id: fixtureId,
		status: "interrupted",
		provider: "provider",
		model: "model",
		thinking_level: "medium",
		pi_version: "0.84.1",
		container_image: "cadgenbench-pi-build123d:0.1.0",
		container_image_digest: "sha256:old",
		started_at: "2026-01-02T03:04:05.000Z",
		updated_at: "2026-01-02T03:05:05.000Z",
		duration_s: 60,
		session_file: sessionFile,
		last_turn: 2,
		artifact: null,
		tokens: { input: 1, output: 2, cache_read: 3, cache_write: 4, total: 10 },
		cost: 0.1,
		tool_calls: 2,
	};
}

afterEach(async () => {
	await Promise.all(temporaryDirectories.splice(0).map(async (directory) => await rm(directory, { recursive: true })));
});

describe("v1 persistent-container migration", () => {
	it("uses the newest turn code, normalizes an absolute session path, backs up state, and is idempotent", async () => {
		const runDir = await runDirectory();
		const raw = legacyParams("legacy-run", ["145"]);
		const workspace = join(runDir, "145");
		await mkdir(join(workspace, "turns", "2"), { recursive: true });
		await mkdir(join(workspace, "turns", "10"), { recursive: true });
		await writeFile(join(workspace, "turns", "2", "code.py"), "# old turn\n");
		await writeFile(join(workspace, "turns", "10", "code.py"), "# newest turn\n");
		await writeJsonAtomic(join(runDir, "params.json"), raw);
		await writeJsonAtomic(
			join(workspace, "agent_result.json"),
			legacyResult("145", "/old/host/results/145/sessions/session-145.jsonl"),
		);

		const migrated = await migrateRunToV2(runDir, raw, "sha256:new-image");

		expect(migrated.protocol_version).toBe(5);
		expect(migrated.runtime_mode).toBe("persistent-agent-container");
		expect(migrated.initialization_status).toBe("pending");
		expect(migrated.container_image_digest).toBe("sha256:new-image");
		expect(await readFile(join(workspace, "solution.py"), "utf8")).toBe("# newest turn\n");
		const result = await readJson<AgentResult>(join(workspace, "agent_result.json"));
		expect(result.session_file).toBe(join("sessions", "session-145.jsonl"));
		expect(result.last_executed_source_sha256).toBeUndefined();
		expect(result.last_executed_step_sha256).toBeUndefined();
		expect((await readJson<LegacyRunParams>(join(runDir, "params.v1.json"))).protocol_version).toBe(1);
		expect(
			(await readJson<AgentResult>(join(workspace, "agent_result.v1.json"))).session_file,
		).toBe("/old/host/results/145/sessions/session-145.jsonl");

		await writeFile(join(workspace, "solution.py"), "# user edit after migration\n");
		const second = await migrateRunToV2(runDir, migrated, "sha256:new-image");
		expect(second).toEqual(migrated);
		expect(await readFile(join(workspace, "solution.py"), "utf8")).toBe("# user edit after migration\n");
		expect((await readJson<LegacyRunParams>(join(runDir, "params.v1.json"))).protocol_version).toBe(1);
	});

	it("creates the initial solution template when the v1 run has no historical code", async () => {
		const runDir = await runDirectory();
		const raw = legacyParams("empty-legacy-run", ["127"]);
		await writeJsonAtomic(join(runDir, "params.json"), raw);

		const migrated = await migrateRunToV2(runDir, raw, "sha256:new-image");

		expect(migrated).toMatchObject<Partial<RunParams>>({
			protocol_version: 5,
			runtime_mode: "persistent-agent-container",
		});
		expect(await readFile(join(runDir, "127", "solution.py"), "utf8")).toBe(INITIAL_SOLUTION);
	});
});
