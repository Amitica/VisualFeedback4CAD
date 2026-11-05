import { createHash } from "node:crypto";
import { readFile, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";

import type { AgentResult, ArtifactPayload, CandidateInspection } from "./types.js";
import { SOLUTION_FILE } from "./types.js";

export const INITIAL_SOLUTION = `"""CADGenBench solution.

Edit this file in place, then call execute_build123d with no arguments.
The formal candidate must be exported to output.step.
"""

from build123d import Box, export_step

# Replace this placeholder with the part described by the fixture drawings.
part = Box(10, 10, 10)
export_step(part, "output.step")
`;

export async function ensureSolution(workspace: string): Promise<string> {
	const path = join(workspace, SOLUTION_FILE);
	const existing = await stat(path).catch(() => null);
	if (existing === null) await writeFile(path, INITIAL_SOLUTION, { encoding: "utf8", mode: 0o600 });
	else if (!existing.isFile()) throw new Error(`${SOLUTION_FILE} is not a regular file`);
	return path;
}

export async function sha256File(path: string): Promise<string | null> {
	const bytes = await readFile(path).catch(() => null);
	return bytes === null ? null : createHash("sha256").update(bytes).digest("hex");
}

export function artifactAccepted(artifact: ArtifactPayload | null): boolean {
	return Boolean(
		artifact?.status === "valid" && artifact.validation?.is_valid && artifact.validation.is_watertight,
	);
}

export async function inspectFormalCandidate(
	workspace: string,
	artifact: ArtifactPayload,
	result: Pick<AgentResult, "last_executed_source_sha256" | "last_executed_step_sha256">,
): Promise<CandidateInspection> {
	const sourceSha = await sha256File(join(workspace, SOLUTION_FILE));
	if (!result.last_executed_source_sha256 || !result.last_executed_step_sha256) {
		return {
			artifact,
			formal_execution_current: false,
			reason: "No formal execute_build123d result is recorded for the current run.",
			source_sha256: sourceSha,
		};
	}
	if (sourceSha !== result.last_executed_source_sha256) {
		return {
			artifact,
			formal_execution_current: false,
			reason: "solution.py changed after the latest execute_build123d call.",
			source_sha256: sourceSha,
		};
	}
	if (artifact.sha256 !== result.last_executed_step_sha256) {
		return {
			artifact,
			formal_execution_current: false,
			reason: "output.step is missing or changed after the latest execute_build123d call.",
			source_sha256: sourceSha,
		};
	}
	return {
		artifact,
		formal_execution_current: true,
		reason: "Current solution.py and output.step match the latest formal execution.",
		source_sha256: sourceSha,
	};
}

export async function recordedCandidateAccepted(workspace: string, result: AgentResult): Promise<boolean> {
	if (!artifactAccepted(result.artifact)) return false;
	const [sourceSha, stepSha] = await Promise.all([
		sha256File(join(workspace, SOLUTION_FILE)),
		sha256File(join(workspace, "output.step")),
	]);
	return Boolean(
		result.last_executed_source_sha256 &&
		result.last_executed_step_sha256 &&
		sourceSha === result.last_executed_source_sha256 &&
		stepSha === result.last_executed_step_sha256 &&
		result.artifact?.sha256 === stepSha,
	);
}
