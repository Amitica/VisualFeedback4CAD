import { spawn } from "node:child_process";

import { InfrastructureError } from "./errors.js";
import { PROTOCOL_VERSION, type WorkerResponse } from "./types.js";

export interface WorkerRequest {
	protocol_version: 5;
	operation:
		| "doctor"
		| "prepare_images"
		| "execute"
		| "inspect"
		| "finalize"
		| "crop_drawing_region"
		| "render_drawing_view"
		| "rank_drawing_views"
		| "compare_drawing_sheet"
		| "compose_reflection_inspection"
		| "execute_repair_candidate"
		| "execute_code_construction_probe"
		| "render_fixed_projection"
		| "compose_repair_comparison"
		| "compose_code_construction_highlight"
		| "compose_code_ablation_highlight"
		| "compose_visual_diff"
		| "compose_code_edit_visualization"
		| "compose_repair_candidate_preview"
		| "compose_shape_feedback"
		| "compare_candidate_topology"
		| "prepare_dimension_view"
		| "measure_dimension_view";
	[key: string]: unknown;
}

export interface WorkerClient {
	request<T extends WorkerResponse>(
		workspace: string,
		request: WorkerRequest,
		options?: { timeoutMs?: number; signal?: AbortSignal },
	): Promise<T>;
}

export function isInfrastructureWorkerFailure(response: WorkerResponse, exitCode: number | null): boolean {
	return (
		response.protocol_version !== PROTOCOL_VERSION ||
		response.error?.kind === "protocol" ||
		response.error?.kind === "worker" ||
		exitCode !== 0
	);
}

export class LocalWorker implements WorkerClient {
	constructor(private readonly pythonBinary = "python") {}

	async request<T extends WorkerResponse>(
		workspace: string,
		request: WorkerRequest,
		options: { timeoutMs?: number; signal?: AbortSignal } = {},
	): Promise<T> {
		return await new Promise<T>((resolve, reject) => {
			const child = spawn(
				this.pythonBinary,
				["-m", "cadgenbench.integrations.pi_worker", "--workspace", workspace],
				{ stdio: ["pipe", "pipe", "pipe"] },
			);
			let stdout = "";
			let stderr = "";
			let settled = false;
			let timedOut = false;
			const finish = (error?: Error, response?: T): void => {
				if (settled) return;
				settled = true;
				clearTimeout(timer);
				options.signal?.removeEventListener("abort", onAbort);
				if (error) reject(error);
				else if (response) resolve(response);
			};
			const stop = (): void => {
				child.kill("SIGTERM");
			};
			const onAbort = (): void => stop();
			const timer = setTimeout(() => {
				timedOut = true;
				stop();
			}, options.timeoutMs ?? 180_000);
			options.signal?.addEventListener("abort", onAbort, { once: true });
			if (options.signal?.aborted) stop();
			child.stdout.setEncoding("utf8");
			child.stderr.setEncoding("utf8");
			child.stdout.on("data", (chunk: string) => {
				stdout += chunk;
			});
			child.stderr.on("data", (chunk: string) => {
				stderr += chunk;
			});
			child.once("error", (error) => finish(new InfrastructureError(`Cannot start CAD worker: ${error.message}`)));
			child.once("close", (code) => {
				if (options.signal?.aborted) {
					finish(new InfrastructureError("CAD worker was aborted"));
					return;
				}
				if (timedOut) {
					finish(new InfrastructureError("CAD worker exceeded its controller timeout"));
					return;
				}
				let response: WorkerResponse;
				try {
					response = JSON.parse(stdout) as WorkerResponse;
				} catch (error) {
					finish(
						new InfrastructureError(
							`CAD worker returned malformed JSON (exit ${String(code)}): ${stderr.trim() || stdout.slice(0, 500)}`,
							{ cause: error },
						),
					);
					return;
				}
				if (isInfrastructureWorkerFailure(response, code)) {
					const detail = response.error?.message ?? stderr.trim() ?? `exit ${String(code)}`;
					finish(new InfrastructureError(`CAD worker protocol/infrastructure failure: ${detail}`));
					return;
				}
				finish(undefined, response as T);
			});
			child.stdin.end(`${JSON.stringify(request)}\n`);
		});
	}
}
