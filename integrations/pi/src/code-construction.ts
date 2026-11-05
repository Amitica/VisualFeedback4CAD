import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";

import type { ImageContent } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import {
	assertSupportedViewSpec,
	imageFromWorkspace,
	inspectPngFile,
	requireDrawingSuccess,
} from "./drawing-reflection.js";
import { InfrastructureError } from "./errors.js";
import { isoNow, readJson, writeJsonAtomic } from "./files.js";
import type { WorkerClient } from "./local-worker.js";
import { cameraSha256, candidateScopedArtifactMatches } from "./main-view-reflect.js";
import { artifactAccepted, inspectFormalCandidate, sha256File } from "./solution.js";
import {
	CODE_CONSTRUCTION_PROFILE,
	PROTOCOL_VERSION,
	SOLUTION_FILE,
	type AgentResult,
	type ArtifactPayload,
	type CodeConstructionProfile,
	type DrawingSimilarityTransform,
	type DrawingViewManifest,
	type DrawingViewSpec,
	type ExecutionPayload,
	type Vector3,
	type WorkerCodeAblationHighlightResponse,
	type WorkerFixedProjectionResponse,
	type WorkerInspectResponse,
	type WorkerRankDrawingViewsResponse,
	type WorkerRepairCandidateResponse,
} from "./types.js";
import type { RunController } from "./runner.js";

export const CODE_CONSTRUCTION_AUDIT_PATH = "drawing_feedback/code-ablation/audit.json";
export const CODE_CONSTRUCTION_CONTEXT_TYPE = "cadgenbench-code-construction-registry";

const regionIdSchema = Type.String({
	minLength: 1,
	maxLength: 128,
	pattern: "^[A-Za-z0-9][A-Za-z0-9._-]*$",
});

const PARAMETERS_SCHEMA = Type.Union([
	Type.Object({
		region_id: regionIdSchema,
		construction_ref: Type.String({
			pattern: "^C[0-9]{3,}$",
			description: "An existing construction reference to re-highlight using its code block rebound to the current solution.py.",
		}),
	}, { additionalProperties: false }),
	Type.Object({
		region_id: regionIdSchema,
		code_block: Type.String({
			minLength: 1,
			maxLength: 20_000,
			description: "For first-time registration only: an exact, uniquely matching, whole-line code block copied from the current solution.py.",
		}),
	}, { additionalProperties: false }),
]);

type InspectionStatus = "highlighted" | "no_visible_change" | "execution_error" | "invalid";

export interface CodeAblationInspectionRecord {
	sequence: number;
	created_at: string;
	construction_ref?: string;
	region_id: string;
	status: InspectionStatus;
	formal_source_sha256: string;
	formal_step_sha256: string;
	code_block: string;
	code_block_sha256: string;
	line_start: number;
	line_end: number;
	ablated_source_path: string;
	ablated_source_sha256: string;
	ablation_replacement: string;
	execution: ExecutionPayload;
	artifact: ArtifactPayload;
	camera_sha256?: string;
	transform_sha256?: string;
	view_spec?: DrawingViewSpec;
	transform?: DrawingSimilarityTransform;
	reference_path?: string;
	reference_sha256?: string;
	current_render_path?: string;
	current_render_sha256?: string;
	reused_locked_current_render?: boolean;
	ablated_render_path?: string;
	ablated_render_sha256?: string;
	projection_origin?: Vector3;
	projection_origin_sha256?: string;
	current_pixel_count?: number;
	removed_pixel_count?: number;
	revealed_pixel_count?: number;
	comparison_path?: string;
	comparison_sha256?: string;
}

export interface CodeConstructionReferenceRecord {
	construction_ref: string;
	created_at: string;
	formal_source_sha256: string;
	formal_step_sha256: string;
	code_block: string;
	code_block_sha256: string;
	line_start: number;
	line_end: number;
	binding_status?: "current" | "modified" | "unresolved";
}

interface PersistedCodeConstructionAudit {
	version: 1;
	kind: "code_ablation_localization";
	profile: CodeConstructionProfile;
	fixture_id: string;
	locked_view_manifest_sha256: string;
	started_at: string;
	updated_at: string;
	constructions?: CodeConstructionReferenceRecord[];
	active_construction_refs?: string[];
	inspections: CodeAblationInspectionRecord[];
}

export interface CodeConstructionState {
	fixtureId: string;
	profile: CodeConstructionProfile;
	lockedViewManifestSha256: string;
	auditPath: string;
	startedAt: string;
	constructions: CodeConstructionReferenceRecord[];
	activeConstructionRefs: string[];
	inspections: CodeAblationInspectionRecord[];
}

export interface ActiveCodeConstructionGrounding {
	construction_ref: string;
	current_source_sha256: string;
	current_line_start: number;
	current_line_end: number;
	code_block: string;
	code_block_sha256: string;
	binding_status: "current" | "modified";
}

interface FormalState {
	sourceSha256: string;
	stepSha256: string;
	source: string;
}

interface CurrentFrame {
	viewSpec: DrawingViewSpec;
	transform: DrawingSimilarityTransform;
	referencePath: string;
	currentRenderPath: string;
	cameraSha256: string;
	transformSha256: string;
	reusedLockedCurrentRender: boolean;
}

export interface AppliedCodeAblation {
	source: string;
	replacement: string;
	lineStart: number;
	lineEnd: number;
}

function sha256Text(value: string): string {
	return createHash("sha256").update(value).digest("hex");
}

function sha256Json(value: unknown): string {
	return sha256Text(JSON.stringify(value));
}

export function applyCodeAblation(source: string, codeBlock: string): AppliedCodeAblation {
	if (codeBlock.includes("\0")) throw new Error("code_block must not contain NUL characters");
	const lines = codeBlock.split(/\r?\n/u);
	if (lines[0]?.trim() === "" || lines.at(-1)?.trim() === "") {
		throw new Error("code_block must begin and end with a non-empty source line");
	}
	const matches: number[] = [];
	let cursor = 0;
	while (cursor <= source.length - codeBlock.length) {
		const found = source.indexOf(codeBlock, cursor);
		if (found < 0) break;
		matches.push(found);
		cursor = found + Math.max(1, codeBlock.length);
	}
	if (matches.length !== 1) {
		throw new Error(`code_block must match the current solution.py exactly once; found ${matches.length} matches`);
	}
	const start = matches[0]!;
	const end = start + codeBlock.length;
	if ((start > 0 && source[start - 1] !== "\n") || (end < source.length && source[end] !== "\n")) {
		throw new Error("code_block must cover complete source lines, not a partial expression");
	}
	const indentation = lines[0]!.match(/^[\t ]*/u)?.[0] ?? "";
	const lineEnding = codeBlock.includes("\r\n") ? "\r\n" : "\n";
	const replacement = `${indentation}pass  # ablated by inspect_code_ablation`;
	const replacementWithEnding = codeBlock.endsWith(lineEnding) ? `${replacement}${lineEnding}` : replacement;
	return {
		source: `${source.slice(0, start)}${replacementWithEnding}${source.slice(end)}`,
		replacement,
		lineStart: source.slice(0, start).split("\n").length,
		lineEnd: source.slice(0, end).split("\n").length,
	};
}

function persistedAudit(state: CodeConstructionState): PersistedCodeConstructionAudit {
	return {
		version: 1,
		kind: "code_ablation_localization",
		profile: state.profile,
		fixture_id: state.fixtureId,
		locked_view_manifest_sha256: state.lockedViewManifestSha256,
		started_at: state.startedAt,
		updated_at: isoNow(),
		constructions: state.constructions,
		active_construction_refs: state.activeConstructionRefs,
		inspections: state.inspections,
	};
}

async function persistState(workspace: string, state: CodeConstructionState): Promise<void> {
	const path = join(workspace, state.auditPath);
	await mkdir(dirname(path), { recursive: true });
	await writeJsonAtomic(path, persistedAudit(state));
}

export async function restoreCodeConstructionState(options: {
	workspace: string;
	fixtureId: string;
	profile: CodeConstructionProfile;
	lockedViewManifestSha256: string;
	auditPath?: string;
}): Promise<CodeConstructionState> {
	const auditPath = options.auditPath ?? CODE_CONSTRUCTION_AUDIT_PATH;
	const stored = await readJson<PersistedCodeConstructionAudit>(join(options.workspace, auditPath)).catch(() => null);
	if (!stored) {
		return {
			fixtureId: options.fixtureId,
			profile: options.profile,
			lockedViewManifestSha256: options.lockedViewManifestSha256,
			auditPath,
			startedAt: isoNow(),
			constructions: [],
			activeConstructionRefs: [],
			inspections: [],
		};
	}
	if (
		stored.version !== 1 ||
		stored.kind !== "code_ablation_localization" ||
		stored.profile !== CODE_CONSTRUCTION_PROFILE ||
		stored.profile !== options.profile ||
		stored.fixture_id !== options.fixtureId ||
		stored.locked_view_manifest_sha256 !== options.lockedViewManifestSha256 ||
		(stored.constructions !== undefined && !Array.isArray(stored.constructions)) ||
		(stored.active_construction_refs !== undefined && !Array.isArray(stored.active_construction_refs)) ||
		!Array.isArray(stored.inspections)
	) throw new InfrastructureError("Persisted code-ablation audit does not match the locked continuation");
	const constructions = stored.constructions ?? [];
	const activeConstructionRefs = stored.active_construction_refs ?? constructions.map(({ construction_ref }) => construction_ref);
	if (constructions.some((construction) => (
		!/^C[0-9]{3,}$/u.test(construction.construction_ref) ||
		!/^[a-f0-9]{64}$/u.test(construction.formal_source_sha256) ||
		!/^[a-f0-9]{64}$/u.test(construction.formal_step_sha256) ||
		!/^[a-f0-9]{64}$/u.test(construction.code_block_sha256) ||
		construction.code_block.length === 0 ||
		(construction.binding_status !== undefined &&
			!["current", "modified", "unresolved"].includes(construction.binding_status)) ||
		!Number.isInteger(construction.line_start) ||
		!Number.isInteger(construction.line_end) ||
		construction.line_start < 1 ||
		construction.line_end < construction.line_start
	))) throw new InfrastructureError("Persisted code-construction registry is invalid");
	if (
		new Set(constructions.map(({ construction_ref }) => construction_ref)).size !== constructions.length ||
		new Set(constructions.map(({ formal_source_sha256, code_block_sha256 }) => (
			`${formal_source_sha256}:${code_block_sha256}`
		))).size !== constructions.length
	) throw new InfrastructureError("Persisted code-construction registry contains duplicate references");
	if (
		new Set(activeConstructionRefs).size !== activeConstructionRefs.length ||
		activeConstructionRefs.some((reference) => !constructions.some(({ construction_ref }) => construction_ref === reference))
	) throw new InfrastructureError("Persisted active code-construction registry is invalid");
	return {
		fixtureId: options.fixtureId,
		profile: options.profile,
		lockedViewManifestSha256: options.lockedViewManifestSha256,
		auditPath,
		startedAt: stored.started_at,
		constructions,
		activeConstructionRefs,
		inspections: stored.inspections,
	};
}

function activeReferenceSet(state: CodeConstructionState): Set<string> {
	return new Set(state.activeConstructionRefs);
}

function referencePattern(references: string[]): RegExp | undefined {
	if (references.length === 0) return undefined;
	return new RegExp(`\\b(?:${references.join("|")})\\b`, "u");
}

function containsReference(value: unknown, pattern: RegExp): boolean {
	if (typeof value === "string") return pattern.test(value);
	if (Array.isArray(value)) return value.some((item) => containsReference(item, pattern));
	if (typeof value !== "object" || value === null) return false;
	return Object.values(value as Record<string, unknown>).some((item) => containsReference(item, pattern));
}

function exactCompleteBlockRemains(source: string, codeBlock: string): boolean {
	try {
		applyCodeAblation(source, codeBlock);
		return true;
	} catch {
		return false;
	}
}

function normalizedNonEmptyBlock(value: string): string | undefined {
	const lines = value.replaceAll("\r\n", "\n").split("\n");
	while (lines[0]?.trim() === "") lines.shift();
	while (lines.at(-1)?.trim() === "") lines.pop();
	if (lines.length === 0) return undefined;
	return lines.join("\n");
}

function reboundBlockFromEdits(
	codeBlock: string,
	edits: ReadonlyArray<{ oldText: string; newText: string }>,
): string | undefined {
	let rebound = codeBlock;
	let touched = false;
	for (const edit of edits) {
		const occurrences = rebound.split(edit.oldText).length - 1;
		if (edit.oldText.length > 0 && occurrences === 1) {
			rebound = rebound.replace(edit.oldText, edit.newText);
			touched = true;
			continue;
		}
		if (edit.oldText.includes(rebound)) {
			const blockStart = edit.oldText.indexOf(rebound);
			const prefix = edit.oldText.slice(0, blockStart);
			const suffix = edit.oldText.slice(blockStart + rebound.length);
			rebound = edit.newText.startsWith(prefix) && edit.newText.endsWith(suffix)
				? edit.newText.slice(prefix.length, edit.newText.length - suffix.length)
				: edit.newText;
			touched = true;
		}
	}
	return touched ? normalizedNonEmptyBlock(rebound) : undefined;
}

/** Rebind stable construction references after a successful exact edit of solution.py. */
export async function rebindCodeConstructionsAfterEdit(options: {
	workspace: string;
	state: CodeConstructionState;
	path: unknown;
	edits: unknown;
}): Promise<string[]> {
	if (typeof options.path !== "string" || !Array.isArray(options.edits)) return [];
	const editedPath = isAbsolute(options.path)
		? resolve(options.path)
		: resolve(options.workspace, options.path);
	if (editedPath !== resolve(options.workspace, SOLUTION_FILE)) return [];
	const edits = options.edits.filter((edit): edit is { oldText: string; newText: string } => (
		typeof edit === "object" &&
		edit !== null &&
		typeof (edit as { oldText?: unknown }).oldText === "string" &&
		typeof (edit as { newText?: unknown }).newText === "string"
	));
	if (edits.length === 0) return [];
	const source = await readFile(editedPath, "utf8");
	const reboundReferences: string[] = [];
	const active = activeReferenceSet(options.state);
	for (const construction of options.state.constructions) {
		if (!active.has(construction.construction_ref)) continue;
		const codeBlock = reboundBlockFromEdits(construction.code_block, edits);
		if (codeBlock === undefined) continue;
		try {
			const applied = applyCodeAblation(source, codeBlock);
			construction.code_block = codeBlock;
			construction.code_block_sha256 = sha256Text(codeBlock);
			construction.line_start = applied.lineStart;
			construction.line_end = applied.lineEnd;
			construction.binding_status = "modified";
		} catch {
			construction.binding_status = "unresolved";
		}
		reboundReferences.push(construction.construction_ref);
	}
	if (reboundReferences.length > 0) await persistState(options.workspace, options.state);
	return reboundReferences;
}

/**
 * Resolve active references against the live source so downstream context gets
 * the exact block and its current line range rather than registration-time
 * line numbers that may have shifted after unrelated edits.
 */
export async function resolveActiveCodeConstructionGroundings(options: {
	workspace: string;
	state: CodeConstructionState;
	constructionRefs: readonly string[];
}): Promise<ActiveCodeConstructionGrounding[]> {
	const requested = [...new Set(options.constructionRefs)];
	if (requested.length === 0) return [];
	const active = activeReferenceSet(options.state);
	const source = await readFile(join(options.workspace, SOLUTION_FILE), "utf8");
	const currentSourceSha256 = sha256Text(source);
	return requested.map((reference) => {
		if (!active.has(reference)) {
			throw new InfrastructureError(`Visual finding references inactive construction ${reference}`);
		}
		const construction = options.state.constructions.find(({ construction_ref }) => (
			construction_ref === reference
		));
		if (!construction) {
			throw new InfrastructureError(`Active construction ${reference} has no registry record`);
		}
		let applied: AppliedCodeAblation;
		try {
			applied = applyCodeAblation(source, construction.code_block);
		} catch (error) {
			throw new InfrastructureError(
				`Active construction ${reference} no longer resolves uniquely in the current source`,
				{ cause: error },
			);
		}
		return {
			construction_ref: reference,
			current_source_sha256: currentSourceSha256,
			current_line_start: applied.lineStart,
			current_line_end: applied.lineEnd,
			code_block: construction.code_block,
			code_block_sha256: construction.code_block_sha256,
			binding_status: construction.binding_status === "modified" ? "modified" : "current",
		};
	});
}

/**
 * Keep stable construction references bound to the live source. Exact edit
 * hooks perform the primary rebinding; this fallback reuses the recorded line
 * range when another tool changed solution.py and marks unresolved bindings
 * without deleting their reference identity.
 */
export async function reconcileCodeConstructionBindings(options: {
	workspace: string;
	state: CodeConstructionState;
}): Promise<string[]> {
	if (options.state.activeConstructionRefs.length === 0) return [];
	const source = await readFile(join(options.workspace, SOLUTION_FILE), "utf8");
	const active = activeReferenceSet(options.state);
	const activeRecords = options.state.constructions.filter(({ construction_ref }) => active.has(construction_ref));
	const sourceLines = source.replaceAll("\r\n", "\n").split("\n");
	const changed: string[] = [];
	for (const construction of activeRecords) {
		if (exactCompleteBlockRemains(source, construction.code_block)) continue;
		const candidate = normalizedNonEmptyBlock(
			sourceLines.slice(construction.line_start - 1, construction.line_end).join("\n"),
		);
		if (candidate !== undefined && exactCompleteBlockRemains(source, candidate)) {
			const applied = applyCodeAblation(source, candidate);
			construction.code_block = candidate;
			construction.code_block_sha256 = sha256Text(candidate);
			construction.line_start = applied.lineStart;
			construction.line_end = applied.lineEnd;
			construction.binding_status = "modified";
		} else {
			construction.binding_status = "unresolved";
		}
		changed.push(construction.construction_ref);
	}
	if (changed.length > 0) await persistState(options.workspace, options.state);
	return changed;
}

/**
 * Remove retired construction registrations from provider context, including
 * their ablation tool calls/results and any assistant-owned text that still
 * names a retired reference. The append-only session and audit files are not
 * modified.
 */
export function pruneRetiredCodeConstructionContext<T>(
	messages: T[],
	state: CodeConstructionState,
): T[] {
	const active = activeReferenceSet(state);
	const retired = state.constructions
		.map(({ construction_ref }) => construction_ref)
		.filter((reference) => !active.has(reference));
	const pattern = referencePattern(retired);
	if (!pattern) return messages;

	const removedToolCallIds = new Set<string>();
	for (const message of messages) {
		if (typeof message !== "object" || message === null) continue;
		const typed = message as { role?: unknown; toolCallId?: unknown; content?: unknown };
		if (
			typed.role === "toolResult" &&
			typeof typed.toolCallId === "string" &&
			containsReference(typed, pattern)
		) removedToolCallIds.add(typed.toolCallId);
		if (!Array.isArray(typed.content)) continue;
		for (const block of typed.content) {
			if (
				typeof block === "object" &&
				block !== null &&
				(block as { type?: unknown }).type === "toolCall" &&
				typeof (block as { id?: unknown }).id === "string" &&
				containsReference(block, pattern)
			) removedToolCallIds.add((block as { id: string }).id);
		}
	}

	return messages.flatMap((message) => {
		if (typeof message !== "object" || message === null) return [message];
		const typed = message as {
			role?: unknown;
			toolCallId?: unknown;
			content?: unknown;
		};
		if (
			typed.role === "toolResult" &&
			typeof typed.toolCallId === "string" &&
			removedToolCallIds.has(typed.toolCallId)
		) return [];
		if (typed.role !== "user" && typeof typed.content === "string" && containsReference(typed.content, pattern)) {
			return [];
		}
		if (!Array.isArray(typed.content)) return [message];
		const content = typed.content.filter((block) => {
			if (typeof block !== "object" || block === null) return !containsReference(block, pattern);
			if (
				(block as { type?: unknown }).type === "toolCall" &&
				typeof (block as { id?: unknown }).id === "string" &&
				removedToolCallIds.has((block as { id: string }).id)
			) return false;
			return typed.role === "user" || !containsReference(block, pattern);
		});
		if (content.length === 0 && typed.role !== "user") return [];
		return [{ ...message, content }];
	}) as T[];
}

function pendingConstructionReference(options: {
	state: CodeConstructionState;
	formal: FormalState;
	applied: AppliedCodeAblation;
	codeBlock: string;
}): { record: CodeConstructionReferenceRecord; isNew: boolean } {
	const codeBlockSha256 = sha256Text(options.codeBlock);
	const active = activeReferenceSet(options.state);
	const existing = options.state.constructions.find((construction) => (
		active.has(construction.construction_ref) &&
		construction.code_block_sha256 === codeBlockSha256 &&
		construction.code_block === options.codeBlock
	));
	if (existing) {
		return { record: existing, isNew: false };
	}
	const nextIndex = options.state.constructions.reduce((maximum, construction) => {
		const parsed = Number.parseInt(construction.construction_ref.slice(1), 10);
		return Number.isFinite(parsed) ? Math.max(maximum, parsed) : maximum;
	}, 0) + 1;
	return {
		record: {
			construction_ref: `C${String(nextIndex).padStart(3, "0")}`,
			created_at: isoNow(),
			formal_source_sha256: options.formal.sourceSha256,
			formal_step_sha256: options.formal.stepSha256,
			code_block: options.codeBlock,
			code_block_sha256: codeBlockSha256,
			line_start: options.applied.lineStart,
			line_end: options.applied.lineEnd,
			binding_status: "current",
		},
		isNew: true,
	};
}

export function injectPersistentCodeConstructionRegistry<T>(
	messages: T[],
	state: CodeConstructionState,
): T[] {
	const active = activeReferenceSet(state);
	const cleaned = pruneRetiredCodeConstructionContext(messages, state);
	const filtered = cleaned.filter((message) => {
		if (typeof message !== "object" || message === null) return true;
		return (message as { customType?: unknown }).customType !== CODE_CONSTRUCTION_CONTEXT_TYPE;
	});
	const activeConstructions = state.constructions.filter(({ construction_ref }) => active.has(construction_ref));
	if (activeConstructions.length === 0) return filtered;
	const entries = activeConstructions.map((construction) => {
		const firstLine = construction.code_block
			.split(/\r?\n/u)
			.find((line) => line.trim().length > 0)
			?.trim()
			.slice(0, 180) ?? "";
		return `${construction.construction_ref} | status=${construction.binding_status ?? "current"} | source=${construction.formal_source_sha256.slice(0, 12)} | lines=${construction.line_start}-${construction.line_end} | block=${construction.code_block_sha256.slice(0, 12)} | ${firstLine}`;
	});
	return [...filtered, {
		role: "custom",
		customType: CODE_CONSTRUCTION_CONTEXT_TYPE,
		content: `[Persistent code-construction registry]\n${entries.join("\n")}\nA modified reference is rebound to edited Current code and can be passed directly to inspect_code_ablation for a fresh highlight. These references do not prove correspondence with target features.`,
		display: false,
		timestamp: Date.now(),
	} as T];
}

async function currentFormalState(options: {
	workspace: string;
	worker: WorkerClient;
	controller: RunController;
	getResult: () => AgentResult;
}): Promise<FormalState> {
	const latest = await options.worker.request<WorkerInspectResponse>(
		options.workspace,
		{ protocol_version: PROTOCOL_VERSION, operation: "inspect" },
		{ signal: options.controller.signal },
	);
	const inspected = await inspectFormalCandidate(options.workspace, latest.artifact, options.getResult());
	if (!inspected.formal_execution_current || !artifactAccepted(latest.artifact) || !inspected.source_sha256 || !latest.artifact.sha256) {
		throw new Error(`Code ablation requires a valid, watertight, formally executed current CAD. ${inspected.reason}`);
	}
	const source = await readFile(join(options.workspace, SOLUTION_FILE), "utf8");
	if (sha256Text(source) !== inspected.source_sha256) {
		throw new InfrastructureError("Formal solution.py changed while preparing code ablation");
	}
	return { sourceSha256: inspected.source_sha256, stepSha256: latest.artifact.sha256, source };
}

async function currentFrame(options: {
	workspace: string;
	worker: WorkerClient;
	controller: RunController;
	manifest: DrawingViewManifest;
	formal: FormalState;
	regionId: string;
	outputRoot: string;
}): Promise<CurrentFrame> {
	const region = options.manifest.regions.find(({ region_id }) => region_id === options.regionId);
	if (!region?.selected) throw new Error(`Unknown or unavailable comparison region ${options.regionId}`);
	assertSupportedViewSpec(region.selected.view_spec);
	if (region.selected.view_spec.section_offset_fraction !== undefined) {
		throw new Error("Code ablation currently requires a non-section comparison region");
	}
	const viewSpec = {
		...region.selected.view_spec,
		name: `ablation-${sha256Json([options.formal.stepSha256, options.regionId]).slice(0, 16)}`,
	};
	let transform: DrawingSimilarityTransform;
	let currentRenderPath: string;
	const lockedArtifactsAreConsistent = candidateScopedArtifactMatches(
		region.selected.comparison_path,
		region.selected.candidate_id,
	) && candidateScopedArtifactMatches(
		region.selected.aligned_render_path,
		region.selected.candidate_id,
	);
	const reusedLockedCurrentRender = options.formal.stepSha256 === options.manifest.step_sha256
		&& lockedArtifactsAreConsistent;
	if (reusedLockedCurrentRender) {
		transform = region.selected.transform;
		currentRenderPath = region.selected.aligned_render_path;
	} else {
		const response = await options.worker.request<WorkerRankDrawingViewsResponse>(
			options.workspace,
			{
				protocol_version: PROTOCOL_VERSION,
				operation: "rank_drawing_views",
				region_id: region.region_id,
				crop_path: region.crop_path,
				step_path: "output.step",
				candidates: [{ candidate_id: viewSpec.name, view_spec: viewSpec }],
				top_k: 1,
				allow_reflection: false,
				max_rotation_deg: 6,
				min_roi_fill: 0.70,
				seed: 0,
				max_iterations: 45,
				output_dir: join(options.outputRoot, "current-frame"),
			},
			{ timeoutMs: 240_000, signal: options.controller.signal },
		);
		requireDrawingSuccess(response, "rank_drawing_views");
		const ranked = response.rankings[0];
		if (
			response.region_id !== region.region_id ||
			!ranked ||
			response.rankings.length !== 1 ||
			response.step_sha256 !== options.formal.stepSha256 ||
			ranked.candidate_id !== viewSpec.name ||
			cameraSha256(ranked.view_spec) !== cameraSha256(region.selected.view_spec) ||
			ranked.transform.reflected
		) throw new InfrastructureError("Code-ablation Current frame drifted from the formal STEP");
		transform = ranked.transform;
		currentRenderPath = ranked.aligned_render_path;
	}
	if (transform.reflected) throw new InfrastructureError("Code-ablation frames do not permit reflected alignment");
	return {
		viewSpec,
		transform,
		referencePath: region.crop_path,
		currentRenderPath,
		cameraSha256: cameraSha256(viewSpec),
		transformSha256: sha256Json(transform),
		reusedLockedCurrentRender,
	};
}

async function assertFormalUnchanged(workspace: string, formal: FormalState): Promise<void> {
	const [sourceSha256, stepSha256] = await Promise.all([
		sha256File(join(workspace, SOLUTION_FILE)),
		sha256File(join(workspace, "output.step")),
	]);
	if (sourceSha256 !== formal.sourceSha256 || stepSha256 !== formal.stepSha256) {
		throw new InfrastructureError("Code-ablation inspection changed the formal source or STEP");
	}
}

function executionFailureReason(response: WorkerRepairCandidateResponse): string {
	if (response.execution.timed_out) return "the ablated source timed out";
	if (!response.execution.success) {
		return response.execution.stderr.trim().split("\n").at(-1) || "the ablated source failed to execute";
	}
	if (response.artifact.validation?.is_valid !== true) return "the ablated STEP is invalid";
	if (response.artifact.validation.is_watertight !== true) return "the ablated STEP is not watertight";
	return "the ablated source did not produce an accepted STEP";
}

export function createCodeConstructionTool(options: {
	workspace: string;
	state: CodeConstructionState;
	lockedManifest: DrawingViewManifest;
	worker: WorkerClient;
	controller: RunController;
	getResult: () => AgentResult;
}) {
	const availableRegionIds = options.lockedManifest.regions
		.filter(({ selected }) => selected !== undefined)
		.map(({ region_id }) => region_id);
	return defineTool({
		name: "inspect_code_ablation",
		label: "Inspect code ablation",
		description:
			`Use this tool for every important unresolved 2D difference before recording a finding or editing related code. To register a construction for the first time, submit one exact, self-contained whole-line \`code_block\` from the current \`solution.py\`. To inspect a previously registered construction, submit its \`construction_ref\` instead; if its code was edited, the controller rebinds that reference to the modified block, so do not copy the code again. Provide exactly one of \`code_block\` or \`construction_ref\`. Prefer a block whose removal should still allow the script to export a valid STEP. Red marks the Current CAD geometry removed with that block. On the Target panel, red is only the Current geometry overlaid for comparison; it is not target segmentation or proof of correspondence. Use surrounding geometry and connectivity to identify the Target counterpart and describe the difference. If the highlight is wrong, incomplete, or the ablated model fails, try another code block or reference. Read exact dimensions from the original drawing; never infer them from pixel ratios or distances. Available \`region_id\` values: \`${availableRegionIds.join(", ")}\`.`,
		parameters: PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			if (!availableRegionIds.includes(params.region_id)) {
				throw new Error(`Unknown or unavailable comparison region ${params.region_id}`);
			}
			const formal = await currentFormalState(options);
			if (Buffer.byteLength(formal.source, "utf8") > 512_000) {
				throw new Error("The formal source is too large for controlled code ablation");
			}
			const requestedReference = "construction_ref" in params ? params.construction_ref : undefined;
			const registeredConstruction = requestedReference === undefined
				? undefined
				: options.state.constructions.find(({ construction_ref }) => construction_ref === requestedReference);
			if (
				requestedReference !== undefined &&
				(!options.state.activeConstructionRefs.includes(requestedReference) || !registeredConstruction)
			) throw new Error(`Unknown or inactive construction reference ${requestedReference}`);
			if (registeredConstruction?.binding_status === "unresolved") {
				throw new Error(
					`Construction reference ${requestedReference} could not be rebound to a unique current source block`,
				);
			}
			const codeBlock = registeredConstruction?.code_block ?? ("code_block" in params ? params.code_block : "");
			const applied = applyCodeAblation(formal.source, codeBlock);
			const construction = registeredConstruction
				? { record: registeredConstruction, isNew: false }
				: pendingConstructionReference({
						state: options.state,
						formal,
						applied,
						codeBlock,
					});
			const sequence = options.state.inspections.length + 1;
			const digest = sha256Json([
				formal.sourceSha256,
				formal.stepSha256,
				params.region_id,
				codeBlock,
				sequence,
			]).slice(0, 16);
			const outputRoot = join(
				"drawing_feedback",
				"code-ablation",
				"inspections",
				`${String(sequence).padStart(3, "0")}-${digest}`,
			);
			const submittedSourcePath = join(outputRoot, "submitted-solution.py");
			await mkdir(join(options.workspace, outputRoot), { recursive: true });
			await writeFile(join(options.workspace, submittedSourcePath), applied.source, { encoding: "utf8", mode: 0o600 });
			const ablatedSourceSha256 = sha256Text(applied.source);
			const candidateId = `ablation-${digest}`;
			const execution = await options.worker.request<WorkerRepairCandidateResponse>(
				options.workspace,
				{
					protocol_version: PROTOCOL_VERSION,
					operation: "execute_repair_candidate",
					candidate_id: candidateId,
					source_path: submittedSourcePath,
					formal_source_sha256: formal.sourceSha256,
					formal_step_sha256: formal.stepSha256,
					timeout_s: 120,
					output_dir: join(outputRoot, "execution"),
				},
				{ timeoutMs: 135_000, signal: options.controller.signal },
			);
			if (
				execution.candidate_id !== candidateId ||
				execution.source_sha256 !== ablatedSourceSha256 ||
				execution.formal_source_sha256 !== formal.sourceSha256 ||
				execution.formal_step_sha256 !== formal.stepSha256 ||
				execution.formal_unchanged !== true
			) throw new InfrastructureError("Code-ablation execution violated its formal hash contract");

			const baseRecord = {
				sequence,
				created_at: isoNow(),
				region_id: params.region_id,
				formal_source_sha256: formal.sourceSha256,
				formal_step_sha256: formal.stepSha256,
				code_block: codeBlock,
				code_block_sha256: sha256Text(codeBlock),
				line_start: applied.lineStart,
				line_end: applied.lineEnd,
				ablated_source_path: submittedSourcePath,
				ablated_source_sha256: ablatedSourceSha256,
				ablation_replacement: applied.replacement,
				execution: execution.execution,
				artifact: execution.artifact,
			};
			if (!artifactAccepted(execution.artifact)) {
				const record: CodeAblationInspectionRecord = {
					...baseRecord,
					status: execution.execution.success ? "invalid" : "execution_error",
				};
				options.state.inspections.push(record);
				await assertFormalUnchanged(options.workspace, formal);
				await persistState(options.workspace, options.state);
				return {
					content: [{
						type: "text" as const,
						text: `The isolated run without \`solution.py\` lines ${applied.lineStart}-${applied.lineEnd} could not produce a valid CAD comparison because ${executionFailureReason(execution)}. No image is returned. Check whether the selected code block is complete and can be omitted independently, or try a different complete code block. The current \`solution.py\` and \`output.step\` were not modified.`,
					}],
					details: record,
				};
			}
			if (!execution.artifact.path || !execution.artifact.sha256) {
				throw new InfrastructureError("Accepted code ablation has no STEP artifact");
			}

			const frame = await currentFrame({
				workspace: options.workspace,
				worker: options.worker,
				controller: options.controller,
				manifest: options.lockedManifest,
				formal,
				regionId: params.region_id,
				outputRoot,
			});
			const [reference, currentRender] = await Promise.all([
				inspectPngFile(options.workspace, frame.referencePath),
				inspectPngFile(options.workspace, frame.currentRenderPath),
			]);
			const ablatedRender = await options.worker.request<WorkerFixedProjectionResponse>(
				options.workspace,
				{
					protocol_version: PROTOCOL_VERSION,
					operation: "render_fixed_projection",
					region_id: params.region_id,
					crop_path: frame.referencePath,
					step_path: execution.artifact.path,
					projection_anchor_step_path: "output.step",
					view_spec: frame.viewSpec,
					transform: frame.transform,
					output_dir: join(outputRoot, "ablated-frame"),
				},
				{ timeoutMs: 180_000, signal: options.controller.signal },
			);
			requireDrawingSuccess(ablatedRender, "render_fixed_projection");
			if (
				ablatedRender.step_sha256 !== execution.artifact.sha256 ||
				ablatedRender.projection_anchor_step_sha256 !== formal.stepSha256 ||
				!ablatedRender.projection_origin ||
				sha256Json(ablatedRender.transform) !== frame.transformSha256 ||
				cameraSha256(ablatedRender.view_spec) !== frame.cameraSha256
			) throw new InfrastructureError("Ablated projection drifted from the frozen formal frame");
			const inspectedAblatedRender = await inspectPngFile(options.workspace, ablatedRender.render_path);
			if (inspectedAblatedRender.sha256 !== ablatedRender.sha256) {
				throw new InfrastructureError("Ablated projection hash mismatch");
			}

			const highlight = await options.worker.request<WorkerCodeAblationHighlightResponse>(
				options.workspace,
				{
					protocol_version: PROTOCOL_VERSION,
					operation: "compose_code_ablation_highlight",
					region_id: params.region_id,
					construction_ref: construction.record.construction_ref,
					reference_path: frame.referencePath,
					current_render_path: frame.currentRenderPath,
					ablated_render_path: ablatedRender.render_path,
					output_dir: join(outputRoot, "comparison"),
				},
				{ timeoutMs: 180_000, signal: options.controller.signal },
			);
			requireDrawingSuccess(highlight, "compose_code_ablation_highlight");
			if (highlight.construction_ref !== construction.record.construction_ref) {
				throw new InfrastructureError("Code-ablation highlight returned the wrong construction reference");
			}
			let comparisonPath: string | undefined;
			let comparisonSha256: string | undefined;
			if (highlight.visual_status === "highlighted" && highlight.comparison_path && highlight.sha256) {
				const comparison = await inspectPngFile(options.workspace, highlight.comparison_path);
				if (comparison.sha256 !== highlight.sha256) throw new InfrastructureError("Code-ablation comparison hash mismatch");
				comparisonPath = highlight.comparison_path;
				comparisonSha256 = highlight.sha256;
			}
			const record: CodeAblationInspectionRecord = {
				...baseRecord,
				...(comparisonPath ? { construction_ref: construction.record.construction_ref } : {}),
				status: comparisonPath ? "highlighted" : "no_visible_change",
				camera_sha256: frame.cameraSha256,
				transform_sha256: frame.transformSha256,
				view_spec: frame.viewSpec,
				transform: frame.transform,
				reference_path: frame.referencePath,
				reference_sha256: reference.sha256,
				current_render_path: frame.currentRenderPath,
				current_render_sha256: currentRender.sha256,
				reused_locked_current_render: frame.reusedLockedCurrentRender,
				ablated_render_path: ablatedRender.render_path,
				ablated_render_sha256: ablatedRender.sha256,
				projection_origin: ablatedRender.projection_origin,
				projection_origin_sha256: sha256Json(ablatedRender.projection_origin),
				current_pixel_count: highlight.current_pixel_count,
				removed_pixel_count: highlight.removed_pixel_count,
				revealed_pixel_count: highlight.revealed_pixel_count,
				...(comparisonPath && comparisonSha256 ? {
					comparison_path: comparisonPath,
					comparison_sha256: comparisonSha256,
				} : {}),
			};
			if (comparisonPath) {
				if (construction.isNew) {
					options.state.constructions.push(construction.record);
					options.state.activeConstructionRefs.push(construction.record.construction_ref);
				} else {
					construction.record.formal_source_sha256 = formal.sourceSha256;
					construction.record.formal_step_sha256 = formal.stepSha256;
					construction.record.code_block = codeBlock;
					construction.record.code_block_sha256 = sha256Text(codeBlock);
					construction.record.line_start = applied.lineStart;
					construction.record.line_end = applied.lineEnd;
					construction.record.binding_status = "current";
				}
			}
			options.state.inspections.push(record);
			await assertFormalUnchanged(options.workspace, formal);
			await persistState(options.workspace, options.state);
			if (!comparisonPath) {
				return {
					content: [{
						type: "text" as const,
						text: `Omitting \`solution.py\` lines ${applied.lineStart}-${applied.lineEnd} produced a valid CAD, but no reliable visible difference was detected in \`${params.region_id}\`, so no image is returned. This result only applies to the selected 2D view; the code block may affect geometry that is hidden, occluded, or not visible in this region. You may inspect another \`region_id\` or select a different complete code block. The current \`solution.py\` and \`output.step\` were not modified.`,
					}],
					details: record,
				};
			}
			const content: Array<{ type: "text"; text: string } | ImageContent> = [
				{
					type: "text",
					text: `Construction reference \`${construction.record.construction_ref}\` ${construction.isNew ? "has been registered" : requestedReference ? "has been re-highlighted from its current rebound code" : "was reused"} for \`solution.py\` lines ${applied.lineStart}-${applied.lineEnd} at source revision \`${formal.sourceSha256.slice(0, 12)}\`. Use the red geometry as the Current CAD anchor. Identify its Target counterpart from surrounding topology, then describe their difference. Red on the Target panel is not target segmentation. If the grounding is incorrect, inspect another block or reference. Read exact dimensions from the original drawing, not from pixel ratios or distances. The current \`solution.py\` and \`output.step\` were not modified.`,
				},
				await imageFromWorkspace(options.workspace, comparisonPath),
			];
			return { content, details: record };
		},
	});
}
