import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it, vi } from "vitest";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";

import { createCadTool, type CadToolState } from "../src/cad-tool.js";
import type { WorkerClient } from "../src/local-worker.js";
import type { WorkerExecuteResponse } from "../src/types.js";
import { makePng } from "./png-fixture.js";

function successfulExecution(): WorkerExecuteResponse {
	return {
		protocol_version: 5, operation: "execute", turn: 1, source_sha256: "source-sha",
		execution: { success: true, exit_code: 0, timed_out: false, duration_s: 1, stdout: "CAD exported", stderr: "", files_produced: {} },
		artifact: {
			status: "valid", path: "output.step", sha256: "step-sha", render_path: "iso.png", render_error: null, validation_error: null,
			validation: { is_valid: true, is_watertight: true, topology_errors: [], solid_count: 1, shell_count: 1, face_count: 6, volume: 1, bbox: {} },
		},
	};
}

describe("formal CAD tool auxiliary feedback", () => {
	it("returns formal execution diagnostics without loading or supplying a render in no-new-images mode", async () => {
		const response = successfulExecution();
		const tool = createCadTool({
			workspace: "/tmp/unused-no-render-file", state: { nextTurn: 1, lastResponse: null },
			includeRender: false, noVisualFeedback: true,
			onResult: async () => undefined,
			worker: { request: vi.fn().mockResolvedValue(response) } as unknown as WorkerClient,
		});
		const outcome = await tool.execute("execute-1", {}, undefined, undefined, {} as ExtensionContext);
		expect(outcome.content.every(block => block.type === "text")).toBe(true);
		expect(outcome.content[0]).toMatchObject({ text: expect.stringContaining("is_watertight") });
		expect(outcome.details).toBe(response);
		expect(tool.description).not.toContain("comparison tools");
	});

	it("preserves successful diagnostics, STEP identity and the render alongside an annotation warning", async () => {
		const workspace = await mkdtemp(join(tmpdir(), "cad-tool-feedback-"));
		try {
			const png = makePng(200, 160);
			await writeFile(join(workspace, "iso.png"), png);
			const response = successfulExecution();
			const state: CadToolState = { nextTurn: 1, lastResponse: null };
			const onResult = vi.fn(async () => "Dimension annotation failed. Retry inspect_comparison.");
			const tool = createCadTool({
				workspace, state, onResult,
				worker: { request: vi.fn().mockResolvedValue(response) } as unknown as WorkerClient,
			});
			const outcome = await tool.execute("execute-1", {}, undefined, undefined, {} as ExtensionContext);
			expect(outcome.content[0]).toMatchObject({ type: "text", text: expect.stringContaining("execution: success=true") });
			expect(outcome.content).toContainEqual({ type: "image", data: png.toString("base64"), mimeType: "image/png" });
			expect(outcome.content.at(-1)).toEqual({ type: "text", text: "Dimension annotation failed. Retry inspect_comparison." });
			expect(outcome.details).toBe(response);
			expect(state.lastResponse).toBe(response);
			expect(state.nextTurn).toBe(2);
			expect(onResult).toHaveBeenCalledWith(response);
		} finally { await rm(workspace, { recursive: true, force: true }); }
	});
	it("propagates a controller abort instead of reporting normal completion", async () => {
		const response = successfulExecution();
		const abort = new Error("cancelled");
		abort.name = "AbortError";
		const tool = createCadTool({
			workspace: "/tmp/unused-cad-tool", state: { nextTurn: 1, lastResponse: null },
			onResult: async () => { throw abort; },
			worker: { request: vi.fn().mockResolvedValue(response) } as unknown as WorkerClient,
		});
		await expect(tool.execute("execute-1", {}, undefined, undefined, {} as ExtensionContext)).rejects.toBe(abort);
	});
});
