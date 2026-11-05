import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { inspectFormalCandidate, recordedCandidateAccepted, sha256File } from "../src/solution.js";
import type { AgentResult, ArtifactPayload } from "../src/types.js";

const temporaryDirectories: string[] = [];

async function workspace(): Promise<string> {
	const directory = await mkdtemp(join(tmpdir(), "pi-solution-"));
	temporaryDirectories.push(directory);
	return directory;
}

function validArtifact(stepSha: string): ArtifactPayload {
	return {
		status: "valid",
		path: "output.step",
		sha256: stepSha,
		validation: {
			is_valid: true,
			is_watertight: true,
			topology_errors: [],
			solid_count: 1,
			shell_count: 1,
			face_count: 6,
			volume: 1,
			bbox: {},
		},
		validation_error: null,
		render_path: null,
		render_error: null,
	};
}

function validResult(sourceSha: string, stepSha: string): AgentResult {
	return {
		fixture_id: "145",
		status: "valid",
		provider: "provider",
		model: "model",
		thinking_level: "medium",
		pi_version: "0.84.1",
		container_image: "agent:image",
		container_image_digest: "sha256:image",
		started_at: "2026-01-01T00:00:00.000Z",
		updated_at: "2026-01-01T00:01:00.000Z",
		duration_s: 60,
		last_turn: 1,
		artifact: validArtifact(stepSha),
		last_executed_source_sha256: sourceSha,
		last_executed_step_sha256: stepSha,
		tokens: { input: 0, output: 0, cache_read: 0, cache_write: 0, total: 0 },
		cost: 0,
		tool_calls: 1,
	};
}

afterEach(async () => {
	await Promise.all(temporaryDirectories.splice(0).map(async (directory) => await rm(directory, { recursive: true })));
});

describe("formal candidate hash gate", () => {
	it("accepts only the source and STEP recorded by the latest execute_build123d call", async () => {
		const directory = await workspace();
		await writeFile(join(directory, "solution.py"), "print('formal')\n");
		await writeFile(join(directory, "output.step"), "formal step\n");
		const sourceSha = await sha256File(join(directory, "solution.py"));
		const stepSha = await sha256File(join(directory, "output.step"));
		expect(sourceSha).not.toBeNull();
		expect(stepSha).not.toBeNull();

		const inspected = await inspectFormalCandidate(directory, validArtifact(stepSha!), {
			last_executed_source_sha256: sourceSha!,
			last_executed_step_sha256: stepSha!,
		});

		expect(inspected.formal_execution_current).toBe(true);
		expect(inspected.reason).toContain("match");
	});

	it("rejects a valid STEP made only through bash because no formal execution is recorded", async () => {
		const directory = await workspace();
		await writeFile(join(directory, "solution.py"), "print('bash-only')\n");
		await writeFile(join(directory, "output.step"), "bash step\n");
		const stepSha = await sha256File(join(directory, "output.step"));

		const inspected = await inspectFormalCandidate(directory, validArtifact(stepSha!), {});

		expect(inspected.formal_execution_current).toBe(false);
		expect(inspected.reason).toContain("No formal execute_build123d");
	});

	it("rejects source edits and STEP replacement after the formal execution", async () => {
		const directory = await workspace();
		await writeFile(join(directory, "solution.py"), "print('original')\n");
		await writeFile(join(directory, "output.step"), "original step\n");
		const sourceSha = await sha256File(join(directory, "solution.py"));
		const stepSha = await sha256File(join(directory, "output.step"));

		await writeFile(join(directory, "solution.py"), "print('edited')\n");
		const editedSource = await inspectFormalCandidate(directory, validArtifact(stepSha!), {
			last_executed_source_sha256: sourceSha!,
			last_executed_step_sha256: stepSha!,
		});
		expect(editedSource.formal_execution_current).toBe(false);
		expect(editedSource.reason).toContain("solution.py changed");

		await writeFile(join(directory, "solution.py"), "print('original')\n");
		await writeFile(join(directory, "output.step"), "replaced step\n");
		const replacementSha = await sha256File(join(directory, "output.step"));
		const replacedStep = await inspectFormalCandidate(directory, validArtifact(replacementSha!), {
			last_executed_source_sha256: sourceSha!,
			last_executed_step_sha256: stepSha!,
		});
		expect(replacedStep.formal_execution_current).toBe(false);
		expect(replacedStep.reason).toContain("output.step");
	});

	it("rechecks both files before treating a persisted valid result as complete on resume", async () => {
		const directory = await workspace();
		await writeFile(join(directory, "solution.py"), "print('persisted')\n");
		await writeFile(join(directory, "output.step"), "persisted step\n");
		const sourceSha = (await sha256File(join(directory, "solution.py")))!;
		const stepSha = (await sha256File(join(directory, "output.step")))!;
		const result = validResult(sourceSha, stepSha);

		expect(await recordedCandidateAccepted(directory, result)).toBe(true);
		await writeFile(join(directory, "output.step"), "bash replacement\n");
		expect(await recordedCandidateAccepted(directory, result)).toBe(false);
		await writeFile(join(directory, "output.step"), "persisted step\n");
		await writeFile(join(directory, "solution.py"), "print('edited after execute')\n");
		expect(await recordedCandidateAccepted(directory, result)).toBe(false);
		const missingFormalHash = { ...result };
		delete missingFormalHash.last_executed_step_sha256;
		expect(await recordedCandidateAccepted(directory, missingFormalHash)).toBe(false);
	});
});
