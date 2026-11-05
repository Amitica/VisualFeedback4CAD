import { readFile } from "node:fs/promises";
import { isAbsolute, relative, resolve } from "node:path";

import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import { InfrastructureError } from "./errors.js";
import { PROTOCOL_VERSION, type WorkerExecuteResponse } from "./types.js";
import type { WorkerClient } from "./local-worker.js";

export interface CadToolState {
	nextTurn: number;
	lastResponse: WorkerExecuteResponse | null;
}

export function normalizeCadToolArguments(args: unknown): unknown {
	if (typeof args !== "object" || args === null || Array.isArray(args)) return args;
	const entries = Object.entries(args);
	if (entries.length === 0) return {};
	if (entries.length === 1 && entries[0]?.[0] === "code" && typeof entries[0][1] === "string") return {};
	return args;
}

function safeArtifactPath(workspace: string, relativePath: string): string {
	if (isAbsolute(relativePath)) throw new InfrastructureError("Worker returned an absolute artifact path");
	const root = resolve(workspace);
	const candidate = resolve(root, relativePath);
	const rel = relative(root, candidate);
	if (rel === ".." || rel.startsWith("../") || isAbsolute(rel)) {
		throw new InfrastructureError("Worker returned an artifact outside the fixture workspace");
	}
	return candidate;
}

export function formatToolFeedback(response: WorkerExecuteResponse): string {
	const { execution, artifact } = response;
	const lines = [
		`Turn ${response.turn} execution: success=${execution.success}, exit_code=${String(execution.exit_code)}, timed_out=${execution.timed_out}, duration_s=${execution.duration_s}.`,
		`Artifact: status=${artifact.status}, sha256=${artifact.sha256 ?? "none"}.`,
	];
	if (execution.stdout.trim()) lines.push(`stdout:\n${execution.stdout.trim()}`);
	if (execution.stderr.trim()) lines.push(`stderr:\n${execution.stderr.trim()}`);
	if (artifact.validation) lines.push(`validation:\n${JSON.stringify(artifact.validation, null, 2)}`);
	if (artifact.validation_error) lines.push(`validation_error: ${artifact.validation_error}`);
	if (artifact.render_error) lines.push(`render_error: ${artifact.render_error}`);
	return lines.join("\n\n");
}

export function createCadTool(options: {
	worker: WorkerClient;
	workspace: string;
	state: CadToolState;
	includeRender?: boolean;
	noVisualFeedback?: boolean;
	/** Optional controller feedback is appended without replacing the formal execution result. */
	onResult: (response: WorkerExecuteResponse) => Promise<void | string>;
}) {
	const includeRender = options.includeRender ?? true;
	return defineTool({
		name: "execute_build123d",
		label: "Execute build123d",
		description: includeRender
			? "Formally execute the current /workspace/solution.py. It must export a fresh output.step. Returns execution diagnostics, source and STEP hashes, CAD validity, measurements, and an isometric render."
			: options.noVisualFeedback
				? "Formally execute the current /workspace/solution.py. It must export a fresh output.step. Returns execution diagnostics, source and STEP hashes, CAD validity, and measurements. No image feedback is provided in this experiment."
				: "Formally execute the current /workspace/solution.py. It must export a fresh output.step. Returns execution diagnostics, source and STEP hashes, CAD validity, and measurements. Use the comparison tools for all visual evaluation.",
		promptSnippet: includeRender
			? "Formally execute the current solution.py and inspect its STEP validation and render"
			: options.noVisualFeedback
				? "Formally execute the current solution.py and inspect execution and STEP validity diagnostics, without images"
				: "Formally execute the current solution.py and inspect its STEP validation; use comparison tools for visual evidence",
		promptGuidelines: [
			"Call execute_build123d repeatedly until is_valid and is_watertight are both true.",
			"Edit solution.py before calling this no-argument tool; only its current contents are executed.",
		],
		parameters: Type.Object({}, { additionalProperties: false }),
		prepareArguments: (args) => normalizeCadToolArguments(args) as never,
		executionMode: "sequential",
		async execute(_toolCallId, _params, signal) {
			const turn = options.state.nextTurn;
			const response = await options.worker.request<WorkerExecuteResponse>(
				options.workspace,
				{
					protocol_version: PROTOCOL_VERSION,
					operation: "execute",
					turn,
					timeout_s: 120,
				},
				{ timeoutMs: 135_000, ...(signal ? { signal } : {}) },
			);
			options.state.nextTurn += 1;
			options.state.lastResponse = response;
			const additionalFeedback = await options.onResult(response);
			const content: Array<{ type: "text"; text: string } | { type: "image"; data: string; mimeType: string }> = [
				{ type: "text", text: formatToolFeedback(response) },
			];
			if (includeRender && response.artifact.render_path) {
				const render = await readFile(safeArtifactPath(options.workspace, response.artifact.render_path));
				content.push({ type: "image", data: render.toString("base64"), mimeType: "image/png" });
			}
			if (additionalFeedback) content.push({ type: "text", text: additionalFeedback });
			return { content, details: response };
		},
	});
}
