import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, realpath, rename, unlink } from "node:fs/promises";
import { isAbsolute, join } from "node:path";

import { defineTool, generateUnifiedPatch } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import { inspectPngFile } from "./drawing-reflection.js";
import type { RunController } from "./runner.js";
import { SHAPE_FEEDBACK_LATEST_PATH, type ShapeFeedbackAudit } from "./shape-feedback.js";
import { artifactAccepted } from "./solution.js";
import { PROTOCOL_VERSION, SOLUTION_FILE, type DrawingViewManifest } from "./types.js";
import { applyPiStyleEdits } from "./visual-review.js";

const candidatePattern = "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$";
const evaluationPattern = /^[0-9a-f]{16}-[0-9a-f]{8}$/u;
const hashPattern = /^[0-9a-f]{64}$/u;

export const ADOPT_SHAPE_CANDIDATE_PARAMETERS_SCHEMA = Type.Object({
	candidate_id: Type.String({ pattern: candidatePattern }),
	evaluation_id: Type.Optional(Type.String({ pattern: evaluationPattern.source })),
}, { additionalProperties: false });

function sha(bytes: string | Buffer): string {
	return createHash("sha256").update(bytes).digest("hex");
}

/** Reject symlinks in every component, including links to other workspace files. */
async function safeFile(workspace: string, relativePath: string): Promise<string> {
	if (typeof relativePath !== "string" || !relativePath || isAbsolute(relativePath) || relativePath.includes("\\") || relativePath.includes("\0")) {
		throw new Error("Preview evidence contains an unsafe file path; run preview_shape_feedback again");
	}
	const parts = relativePath.split("/");
	if (parts.some((part) => !part || part === "." || part === "..")) throw new Error("Preview evidence contains an unsafe file path");
	let path = workspace;
	for (const [index, part] of parts.entries()) {
		path = join(path, part);
		const info = await lstat(path).catch(() => null);
		if (!info || info.isSymbolicLink() || (index === parts.length - 1 ? !info.isFile() : !info.isDirectory())) {
			throw new Error(`Missing or unsafe file ${relativePath}; run preview_shape_feedback again`);
		}
	}
	return path;
}

async function readSafeFile(workspace: string, relativePath: string, maxBytes = 128 * 1024 * 1024): Promise<{ bytes: Buffer; mode: number }> {
	const path = await safeFile(workspace, relativePath);
	const handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
	try {
		const info = await handle.stat();
		if (!info.isFile() || info.size > maxBytes) throw new Error(`File ${relativePath} is not a supported regular file`);
		return { bytes: await handle.readFile(), mode: info.mode & 0o777 };
	} finally {
		await handle.close();
	}
}

async function readLatest(workspace: string): Promise<string> {
	let pointer: { version?: unknown; evaluation_id?: unknown };
	try {
		pointer = JSON.parse((await readSafeFile(workspace, SHAPE_FEEDBACK_LATEST_PATH, 4_096)).bytes.toString("utf8"));
	} catch {
		throw new Error("The latest preview pointer is missing or invalid; run preview_shape_feedback first");
	}
	if (!pointer || pointer.version !== 1 || typeof pointer.evaluation_id !== "string" || !evaluationPattern.test(pointer.evaluation_id)) {
		throw new Error("The latest preview pointer is invalid; run preview_shape_feedback again");
	}
	return pointer.evaluation_id;
}

export interface ShapeCandidateReference {
	workspace: string;
	lockedManifest: DrawingViewManifest;
	evaluationId: string;
	candidateId: string;
}

export interface VerifiedShapeCandidateEvidence {
	audit: ShapeFeedbackAudit;
	candidate: ShapeFeedbackAudit["candidates"][number];
	auditPath: string;
	cards: Array<{ region_id: string; card_path: string; sha256: string; width: number; height: number }>;
}

async function loadShapeCandidateEvidence({ workspace, lockedManifest, evaluationId, candidateId }: ShapeCandidateReference) {
	if (!evaluationPattern.test(evaluationId) || !new RegExp(candidatePattern).test(candidateId)) {
		throw new Error("Provide a valid evaluation_id and candidate_id from the shape preview");
	}
	const outputRoot = join("drawing_feedback", "shape-feedback", evaluationId);
	const auditPath = join(outputRoot, "audit.json");
	const auditBytes = (await readSafeFile(workspace, auditPath, 8 * 1024 * 1024)).bytes;
	let audit: ShapeFeedbackAudit;
	try { audit = JSON.parse(auditBytes.toString("utf8")) as ShapeFeedbackAudit; }
	catch { throw new Error("The selected preview audit is invalid; run preview_shape_feedback again"); }
	if (!audit || audit.version !== 1 || audit.kind !== "shape_feedback_preview" || audit.evaluation_id !== evaluationId) {
		throw new Error("The selected preview audit does not match its evaluation ID; run preview_shape_feedback again");
	}
	if (audit.status !== "completed" && audit.status !== "partial") throw new Error(`The selected preview is ${audit.status}; finish a successful preview before adopting`);
	if (audit.locked_view_manifest_sha256 !== sha(JSON.stringify(lockedManifest))) throw new Error("The locked drawing views changed; run preview_shape_feedback again");
	if (!hashPattern.test(audit.formal_source_sha256) || !hashPattern.test(audit.formal_step_sha256)
		|| !Array.isArray(audit.candidates) || !Array.isArray(audit.feedback) || !Array.isArray(audit.projections)) throw new Error("The selected preview audit is incomplete");
	const matches = audit.candidates.filter((candidate) => candidate?.id === candidateId);
	if (matches.length !== 1) throw new Error(`Candidate ${candidateId} is not uniquely available in the selected preview`);
	const candidate = matches[0]!;
	const sourcePath = join(outputRoot, "candidates", candidate.id, "submitted-solution.py");
	const stepPath = join(outputRoot, "candidates", candidate.id, "execution", "output.step");
	const execution = candidate.execution;
	if (candidate.status !== "valid" || !hashPattern.test(candidate.source_sha256) || candidate.source_path !== sourcePath
		|| !execution || execution.protocol_version !== PROTOCOL_VERSION || execution.operation !== "execute_repair_candidate"
		|| execution.candidate_id !== `shape-${evaluationId.slice(0, 16)}-${candidate.id}` || execution.source_path !== sourcePath
		|| execution.source_sha256 !== candidate.source_sha256 || execution.formal_source_sha256 !== audit.formal_source_sha256
		|| execution.formal_step_sha256 !== audit.formal_step_sha256 || execution.formal_unchanged !== true
		|| execution.execution?.success !== true || execution.execution.timed_out !== false || execution.execution.exit_code !== 0
		|| !artifactAccepted(execution.artifact) || execution.artifact.path !== stepPath || !execution.artifact.sha256 || !hashPattern.test(execution.artifact.sha256)) {
		throw new Error(`Candidate ${candidate.id} has no valid successful preview execution; preview it again`);
	}
	if (!Array.isArray(candidate.edits) || candidate.edits.length < 1 || candidate.edits.length > 8
		|| candidate.edits.some((edit) => !edit || typeof edit.oldText !== "string" || !edit.oldText.length || edit.oldText.length > 20_000 || typeof edit.newText !== "string" || edit.newText.length > 20_000)
		|| typeof candidate.patch !== "string") throw new Error("The saved candidate edits are invalid; preview it again");
	const [savedSource, savedStep, current] = await Promise.all([
		readSafeFile(workspace, sourcePath, 2 * 1024 * 1024),
		readSafeFile(workspace, stepPath),
		readSafeFile(workspace, SOLUTION_FILE, 2 * 1024 * 1024),
	]);
	if (sha(savedSource.bytes) !== candidate.source_sha256 || sha(savedStep.bytes) !== execution.artifact.sha256) throw new Error("The saved candidate source or STEP changed; run preview_shape_feedback again");
	const cards: VerifiedShapeCandidateEvidence["cards"] = [];
	for (const feedback of audit.feedback) {
		if (!feedback || feedback.protocol_version !== PROTOCOL_VERSION || feedback.operation !== "compose_shape_feedback" || feedback.status !== "ok" || !Array.isArray(feedback.candidates)) continue;
		const shown = feedback.candidates.filter((item) => item?.candidate_id === candidate.id);
		if (!shown.length) continue;
		if (shown.length !== 1) throw new Error("The preview image has duplicate candidate references; preview it again");
		const expectedAnchor = lockedManifest.projection_frame?.anchor_step_sha256 ?? audit.formal_step_sha256;
		const projections = audit.projections.filter((item) => item?.candidate_id === candidate.id && item.region_id === feedback.region_id && item.step_sha256 === execution.artifact.sha256 && item.render_sha256 === shown[0]!.render_sha256 && item.projection_anchor_step_sha256 === expectedAnchor);
		if (projections.length !== 1) throw new Error("The preview image has no unique matching projection; preview it again");
		const projection = projections[0]!;
		const matchingCards = Array.isArray(feedback.cards) ? feedback.cards.filter(item => item.candidate_id === candidate.id) : [feedback];
		if (matchingCards.length !== 1) throw new Error("The preview image has no unique candidate card; preview it again");
		const shownCard = matchingCards[0]!;
		await safeFile(workspace, shownCard.card_path);
		await safeFile(workspace, projection.render_path);
		const [card, render] = await Promise.all([inspectPngFile(workspace, shownCard.card_path), inspectPngFile(workspace, projection.render_path)]);
		if (card.sha256 !== shownCard.sha256 || card.width !== shownCard.width || card.height !== shownCard.height || render.sha256 !== projection.render_sha256) throw new Error("The saved preview image changed; run preview_shape_feedback again");
		if (cards.some(item => item.region_id === feedback.region_id)) throw new Error("The preview contains duplicate view evidence; preview it again");
		cards.push({ region_id: feedback.region_id, card_path: shownCard.card_path, ...card });
	}
	if (!cards.length) throw new Error(`Candidate ${candidate.id} was not shown in a successful preview image; preview it again`);
	return { audit, candidate, auditPath, cards, auditBytes, savedSource, current };
}

function verifySavedEdits(
	current: Buffer,
	savedSource: Buffer,
	candidate: ShapeFeedbackAudit["candidates"][number],
): Buffer {
	const source = current.toString("utf8");
	const proposed = applyPiStyleEdits(source, candidate.edits);
	const proposedBytes = Buffer.from(proposed, "utf8");
	if (!proposedBytes.equals(savedSource) || sha(proposedBytes) !== candidate.source_sha256 || generateUnifiedPatch(SOLUTION_FILE, source, proposed) !== candidate.patch) {
		throw new Error("The saved candidate does not match its edits; run preview_shape_feedback again");
	}
	return proposedBytes;
}

/** Read-only verification for a submitted repair; never resolves the latest pointer or writes source. */
export async function verifyShapeCandidateEvidence(options: ShapeCandidateReference): Promise<VerifiedShapeCandidateEvidence> {
	const workspace = await realpath(options.workspace);
	const evidence = await loadShapeCandidateEvidence({ ...options, workspace });
	if (sha(evidence.current.bytes) !== evidence.audit.formal_source_sha256) throw new Error("solution.py changed after the preview; run preview_shape_feedback again");
	const currentStep = await readSafeFile(workspace, "output.step");
	if (sha(currentStep.bytes) !== evidence.audit.formal_step_sha256) throw new Error("output.step changed after the preview; run preview_shape_feedback again");
	verifySavedEdits(evidence.current.bytes, evidence.savedSource.bytes, evidence.candidate);
	const [latestSource, latestStep, latestAudit] = await Promise.all([
		readSafeFile(workspace, SOLUTION_FILE, 2 * 1024 * 1024), readSafeFile(workspace, "output.step"),
		readSafeFile(workspace, evidence.auditPath, 8 * 1024 * 1024),
	]);
	if (!latestSource.bytes.equals(evidence.current.bytes) || !latestStep.bytes.equals(currentStep.bytes) || !latestAudit.bytes.equals(evidence.auditBytes)) {
		throw new Error("The preview or Current changed during verification; run preview_shape_feedback again");
	}
	return { audit: evidence.audit, candidate: evidence.candidate, auditPath: evidence.auditPath, cards: evidence.cards };
}

export function createAdoptShapeCandidateTool(options: {
	workspace: string;
	lockedManifest: DrawingViewManifest;
	controller: RunController;
}) {
	return defineTool({
		name: "adopt_shape_candidate",
		label: "Adopt shape candidate",
		description: "Apply a saved preview candidate to solution.py after reviewing its returned views. Provide candidate_id and, to select a specific preview including an earlier repair attempt, evaluation_id. Without evaluation_id, the most recent preview_shape_feedback call is used. The tool verifies and applies exactly that candidate's saved code replacements, and rejects stale previews when the formal source or STEP has changed. It does not rebuild CAD or change output.step.",
		parameters: ADOPT_SHAPE_CANDIDATE_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			if (!params || Object.keys(params).some((key) => key !== "candidate_id" && key !== "evaluation_id")
				|| typeof params.candidate_id !== "string" || !new RegExp(candidatePattern).test(params.candidate_id)
				|| ("evaluation_id" in params && (typeof params.evaluation_id !== "string" || !evaluationPattern.test(params.evaluation_id)))) {
				throw new Error("Provide candidate_id and optionally evaluation_id, using the IDs from a shape preview");
			}
			const checkCancelled = () => {
				if (options.controller.signal.aborted) throw new Error("Candidate adoption was cancelled");
			};
			checkCancelled();
			const workspace = await realpath(options.workspace);
			const explicitEvaluation = params.evaluation_id !== undefined;
			const evaluationId = params.evaluation_id ?? await readLatest(workspace);
			const { audit, candidate, auditPath, auditBytes, savedSource, current } = await loadShapeCandidateEvidence({
				workspace, lockedManifest: options.lockedManifest, evaluationId, candidateId: params.candidate_id,
			});
			const details = {
				candidate_id: candidate.id, evaluation_id: evaluationId, audit_path: auditPath,
				path: SOLUTION_FILE, source_sha256: candidate.source_sha256,
				previous_source_sha256: audit.formal_source_sha256, patch: candidate.patch,
			};
			if (current.bytes.equals(savedSource.bytes)) {
				if (!explicitEvaluation && await readLatest(workspace) !== evaluationId) throw new Error("A newer preview started; select a candidate from that preview");
				if (explicitEvaluation && sha((await readSafeFile(workspace, "output.step")).bytes) !== audit.formal_step_sha256) {
					throw new Error("output.step changed after the preview; run preview_shape_feedback again");
				}
				checkCancelled();
				return { content: [{ type: "text", text: `Candidate ${candidate.id} is already applied to solution.py. This call did not modify solution.py or output.step.` }], details: { ...details, status: "already_applied" } };
			}
			if (sha(current.bytes) !== audit.formal_source_sha256) throw new Error("solution.py changed after the preview; run preview_shape_feedback again");
			const currentStep = await readSafeFile(workspace, "output.step");
			if (sha(currentStep.bytes) !== audit.formal_step_sha256) throw new Error("output.step changed after the preview; run preview_shape_feedback again");
			const proposedBytes = verifySavedEdits(current.bytes, savedSource.bytes, candidate);
			checkCancelled();
			const temporary = join(workspace, `.solution.py.adopt-${randomUUID()}.tmp`);
			try {
				const handle = await open(temporary, "wx", current.mode);
				try { await handle.writeFile(proposedBytes); await handle.sync(); }
				finally { await handle.close(); }
				const [latest, latestSource, latestStep, latestAudit] = await Promise.all([
					explicitEvaluation ? Promise.resolve(evaluationId) : readLatest(workspace), readSafeFile(workspace, SOLUTION_FILE, 2 * 1024 * 1024),
					readSafeFile(workspace, "output.step"), readSafeFile(workspace, auditPath, 8 * 1024 * 1024),
				]);
				if (latest !== evaluationId || !latestAudit.bytes.equals(auditBytes)) throw new Error("The preview changed during adoption; run preview_shape_feedback again");
				if (!latestSource.bytes.equals(current.bytes) || !latestStep.bytes.equals(currentStep.bytes)) throw new Error("solution.py or output.step changed during adoption; run preview_shape_feedback again");
				checkCancelled();
				await rename(temporary, join(workspace, SOLUTION_FILE));
			} finally {
				await unlink(temporary).catch((error: NodeJS.ErrnoException) => { if (error.code !== "ENOENT") throw error; });
			}
			return {
				content: [{ type: "text", text: `Applied candidate ${candidate.id} to solution.py. output.step unchanged; CAD has not been rebuilt.` }],
				details: { ...details, status: "applied" },
			};
		},
	});
}
