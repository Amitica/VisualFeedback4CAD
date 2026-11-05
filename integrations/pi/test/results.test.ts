import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { writeJsonAtomic } from "../src/files.js";
import { updateRunSummary } from "../src/results.js";
import type { AgentResult, RunParams } from "../src/types.js";

describe("result aggregation", () => {
	it("reports completion and validity without inventing CAD Score", async () => {
		const runDir = await mkdtemp(join(tmpdir(), "pi-results-"));
		try {
			const params: RunParams = {
				protocol_version: 5,
				runtime_mode: "persistent-agent-container",
				initialization_status: "complete",
				run_id: "test",
				created_at: new Date().toISOString(),
				provider: "p",
				model: "m",
				thinking_level: "medium",
				pi_version: "0.84.1",
				container_image: "image",
				container_image_digest: "sha256:image",
				data_inputs_dir: "/data",
				fixtures: ["127", "133"],
				parallel: 1,
				mode: "fixtures",
			};
			const base: AgentResult = {
				fixture_id: "127",
				status: "valid",
				provider: "p",
				model: "m",
				thinking_level: "medium",
				pi_version: "0.84.1",
				container_image: "image",
				container_image_digest: "sha256:image",
				started_at: params.created_at,
				updated_at: params.created_at,
				duration_s: 1,
				last_turn: 1,
				artifact: null,
				tokens: { input: 2, output: 3, cache_read: 4, cache_write: 5, total: 14 },
				cost: 0.5,
				tool_calls: 1,
			};
			await mkdir(join(runDir, "127"));
			await writeJsonAtomic(join(runDir, "127", "agent_result.json"), base);
			const summary = await updateRunSummary(runDir, params);
		expect(params.protocol_version).toBe(5);
			expect(params.runtime_mode).toBe("persistent-agent-container");
			expect(summary.validity_rate).toBe(0.5);
			expect(summary.completion_rate).toBe(0.5);
			expect(summary.tokens.total).toBe(14);
			expect(summary).not.toHaveProperty("cad_score");
		} finally {
			await rm(runDir, { recursive: true });
		}
	});
});
