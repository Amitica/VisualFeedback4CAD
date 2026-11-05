import { join } from "node:path";

import { elapsedSeconds, isoNow, readJson, writeJsonAtomic } from "./files.js";
import type { AgentResult, PiRunSummary, RunParams, RunStatus, UsageTotals } from "./types.js";

export const EMPTY_USAGE: UsageTotals = {
	tokens: { input: 0, output: 0, cache_read: 0, cache_write: 0, total: 0 },
	cost: 0,
	tool_calls: 0,
};

export async function loadAgentResult(runDir: string, fixtureId: string): Promise<AgentResult | null> {
	return await readJson<AgentResult>(join(runDir, fixtureId, "agent_result.json")).catch(() => null);
}

export async function saveAgentResult(runDir: string, result: AgentResult): Promise<void> {
	result.updated_at = isoNow();
	result.duration_s = elapsedSeconds(result.started_at);
	await writeJsonAtomic(join(runDir, result.fixture_id, "agent_result.json"), result);
}

function aggregateStatus(results: AgentResult[], fixtureCount: number): RunStatus {
	if (results.some((result) => result.status === "infrastructure_error")) return "infrastructure_error";
	if (results.some((result) => result.status === "interrupted")) return "interrupted";
	if (results.length < fixtureCount || results.some((result) => result.status === "running")) return "running";
	if (results.every((result) => result.status === "valid")) return "valid";
	if (results.some((result) => result.status === "invalid")) return "invalid";
	return "missing";
}

export async function updateRunSummary(runDir: string, params: RunParams, stopReason?: string): Promise<PiRunSummary> {
	const results = (
		await Promise.all(params.fixtures.map(async (id) => await loadAgentResult(runDir, id)))
	).filter((result): result is AgentResult => result !== null);
	const validCount = results.filter((result) => result.status === "valid").length;
	const completedCount = results.filter((result) => result.status === "valid").length;
	const tokenTotals = results.reduce(
		(accumulator, result) => ({
			input: accumulator.input + result.tokens.input,
			output: accumulator.output + result.tokens.output,
			cache_read: accumulator.cache_read + result.tokens.cache_read,
			cache_write: accumulator.cache_write + result.tokens.cache_write,
			total: accumulator.total + result.tokens.total,
		}),
		{ input: 0, output: 0, cache_read: 0, cache_write: 0, total: 0 },
	);
	const summary: PiRunSummary = {
		run_id: params.run_id,
		status: aggregateStatus(results, params.fixtures.length),
		agent_mode: params.agent_mode ?? "drawing-subagents",
		provider: params.provider,
		model: params.model,
		pi_version: params.pi_version,
		container_image_digest: params.container_image_digest,
		created_at: params.created_at,
		updated_at: isoNow(),
		duration_s: elapsedSeconds(params.created_at),
		fixture_count: params.fixtures.length,
		completed_count: completedCount,
		valid_count: validCount,
		completion_rate: params.fixtures.length === 0 ? 0 : completedCount / params.fixtures.length,
		validity_rate: params.fixtures.length === 0 ? 0 : validCount / params.fixtures.length,
		tokens: tokenTotals,
		cost: results.reduce((sum, result) => sum + result.cost, 0),
		tool_calls: results.reduce((sum, result) => sum + result.tool_calls, 0),
		results: results.map((result) => ({
			fixture_id: result.fixture_id,
			status: result.status,
			is_valid: result.artifact?.validation?.is_valid ?? null,
			is_watertight: result.artifact?.validation?.is_watertight ?? null,
			duration_s: result.duration_s,
			cost: result.cost,
			tool_calls: result.tool_calls,
		})),
		...(stopReason ? { stop_reason: stopReason } : {}),
	};
	await writeJsonAtomic(join(runDir, "pi_run_summary.json"), summary);
	return summary;
}
