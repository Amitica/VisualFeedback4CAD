import { mkdir } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve } from "node:path";

import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import type { ActiveCodeConstructionGrounding } from "./code-construction.js";
import { InfrastructureError } from "./errors.js";
import { isoNow, readJson, writeJsonAtomic } from "./files.js";
import { VISUAL_FINDING_PROFILE, type VisualFindingProfile } from "./types.js";

export const VISUAL_FINDING_STATE_PATH = "drawing_feedback/visual-findings/state.json";
export const VISUAL_FINDINGS_CONTEXT_TYPE = "cadgenbench-persistent-visual-findings";

export interface VisualFinding {
	finding_id: string;
	target_geometry: string;
	current_geometry: string;
	difference: string;
	current_construction_refs: string[];
}

interface AddVisualFindingOperation {
	operation: "add";
	target_geometry: string;
	current_geometry: string;
	difference: string;
	current_construction_refs?: string[];
}

interface UpdateVisualFindingOperation {
	operation: "update";
	finding_id: string;
	target_geometry?: string;
	current_geometry?: string;
	difference?: string;
	current_construction_refs?: string[];
}

interface DeleteVisualFindingOperation {
	operation: "delete";
	finding_id: string;
}

type VisualFindingOperation =
	| AddVisualFindingOperation
	| UpdateVisualFindingOperation
	| DeleteVisualFindingOperation;

interface AppliedVisualFindingOperation {
	operation: VisualFindingOperation["operation"];
	finding_id: string;
	finding?: VisualFinding;
}

interface VisualFindingRevision {
	revision: number;
	findings: VisualFinding[];
	operations: AppliedVisualFindingOperation[];
	created_at: string;
}

interface PersistedVisualFindingState {
	version: 2;
	profile: VisualFindingProfile;
	fixture_id: string;
	locked_view_manifest_sha256: string;
	findings: VisualFinding[];
	next_finding_index: number;
	revision: number;
	started_at: string;
	updated_at: string;
	history: VisualFindingRevision[];
}

export interface VisualFindingState {
	profile: VisualFindingProfile;
	fixtureId: string;
	lockedViewManifestSha256: string;
	findings: VisualFinding[];
	nextFindingIndex: number;
	revision: number;
	statePath: string;
	startedAt: string;
	history: VisualFindingRevision[];
}

const FINDING_ID_PATTERN = "^F[0-9]{3,}$";
const CONSTRUCTION_REF_PATTERN = "^C[0-9]{3,}$";
const INLINE_CONSTRUCTION_REF_PATTERN = /\bC[0-9]{3,}\b/u;
const MAX_FINDING_TEXT_LENGTH = 8_000;
const MAX_FINDINGS_JSON_LENGTH = 64_000;

const findingTextSchema = (description: string) => Type.String({
	minLength: 1,
	maxLength: MAX_FINDING_TEXT_LENGTH,
	description,
});

const constructionRefsSchema = Type.Array(Type.String({ pattern: CONSTRUCTION_REF_PATTERN }), {
	maxItems: 50,
	uniqueItems: true,
	description: "Construction references currently associated with the Current CAD geometry in this finding. The Agent may add or remove registered references when updating the finding.",
});

export const UPDATE_VISUAL_FINDINGS_PARAMETERS_SCHEMA = Type.Object({
	operations: Type.Array(Type.Union([
		Type.Object({
			operation: Type.Literal("add"),
			target_geometry: findingTextSchema("What the target engineering drawing depicts for this unresolved comparison."),
			current_geometry: findingTextSchema("What the corresponding Current CAD geometry looks like now."),
			difference: findingTextSchema("Only the unresolved geometric difference between Current CAD and the target."),
			current_construction_refs: Type.Optional(constructionRefsSchema),
		}, { additionalProperties: false }),
		Type.Object({
			operation: Type.Literal("update"),
			finding_id: Type.String({ pattern: FINDING_ID_PATTERN }),
			target_geometry: Type.Optional(findingTextSchema("Revised target-geometry description.")),
			current_geometry: Type.Optional(findingTextSchema("Revised Current-CAD description.")),
			difference: Type.Optional(findingTextSchema("Revised unresolved geometric difference.")),
			current_construction_refs: Type.Optional(constructionRefsSchema),
		}, { additionalProperties: false }),
		Type.Object({
			operation: Type.Literal("delete"),
			finding_id: Type.String({ pattern: FINDING_ID_PATTERN }),
		}, { additionalProperties: false }),
	]), {
		minItems: 1,
		maxItems: 50,
		description: "Ordered add, update, and delete operations over the persistent visual-finding collection.",
	}),
}, {
	additionalProperties: false,
	description: "Add, update, or delete persistent target-versus-Current geometric differences.",
});

function cloneFinding(finding: VisualFinding): VisualFinding {
	return {
		...finding,
		current_construction_refs: [...finding.current_construction_refs],
	};
}

function cloneFindings(findings: VisualFinding[]): VisualFinding[] {
	return findings.map(cloneFinding);
}

function assertFindingText(field: string, value: string): string {
	const trimmed = value.trim();
	if (trimmed.length === 0) throw new Error(`${field} must not be empty`);
	if (trimmed.length > MAX_FINDING_TEXT_LENGTH) {
		throw new Error(`${field} must not exceed ${MAX_FINDING_TEXT_LENGTH} characters`);
	}
	if (INLINE_CONSTRUCTION_REF_PATTERN.test(trimmed)) {
		throw new Error(`${field} must not embed construction references; use current_construction_refs`);
	}
	return trimmed;
}

function assertConstructionRefs(
	references: string[],
	activeConstructionRefs: ReadonlySet<string>,
): string[] {
	if (new Set(references).size !== references.length) {
		throw new Error("current_construction_refs must not contain duplicates");
	}
	for (const reference of references) {
		if (!new RegExp(CONSTRUCTION_REF_PATTERN, "u").test(reference)) {
			throw new Error(`Invalid construction reference: ${reference}`);
		}
		if (!activeConstructionRefs.has(reference)) {
			throw new Error(`Construction reference ${reference} is not active`);
		}
	}
	return [...references];
}

function assertStoredFinding(value: VisualFinding): void {
	if (
		typeof value !== "object" ||
		value === null ||
		!new RegExp(FINDING_ID_PATTERN, "u").test(value.finding_id) ||
		typeof value.target_geometry !== "string" ||
		typeof value.current_geometry !== "string" ||
		typeof value.difference !== "string" ||
		!Array.isArray(value.current_construction_refs) ||
		value.current_construction_refs.some((reference) => (
			typeof reference !== "string" || !new RegExp(CONSTRUCTION_REF_PATTERN, "u").test(reference)
		))
	) throw new InfrastructureError("Persisted visual finding is invalid");
	try {
		assertFindingText("target_geometry", value.target_geometry);
		assertFindingText("current_geometry", value.current_geometry);
		assertFindingText("difference", value.difference);
	} catch (error) {
		throw new InfrastructureError(error instanceof Error ? error.message : "Persisted visual finding is invalid");
	}
	if (new Set(value.current_construction_refs).size !== value.current_construction_refs.length) {
		throw new InfrastructureError("Persisted visual finding contains duplicate construction references");
	}
}

function assertFindingsSize(findings: VisualFinding[]): void {
	if (JSON.stringify(findings).length > MAX_FINDINGS_JSON_LENGTH) {
		throw new Error(`Visual findings must not exceed ${MAX_FINDINGS_JSON_LENGTH} serialized characters`);
	}
}

export function renderVisualFindings(
	findings: VisualFinding[],
	constructionGroundings?: readonly ActiveCodeConstructionGrounding[],
): string {
	const groundingsByReference = constructionGroundings === undefined
		? undefined
		: new Map(constructionGroundings.map((grounding) => [grounding.construction_ref, grounding]));
	const renderReferences = (references: string[]): string => references
		.map((reference) => {
			if (groundingsByReference === undefined) return reference;
			const grounding = groundingsByReference.get(reference);
			return grounding ? `${reference} ${grounding.binding_status}` : `${reference} unavailable`;
		})
		.join(", ");
	const findingsText = findings.map((finding) => [
		`[${finding.finding_id}]`,
		`Target: ${finding.target_geometry}`,
		`Current${finding.current_construction_refs.length > 0 ? ` [${renderReferences(finding.current_construction_refs)}]` : ""}: ${finding.current_geometry}`,
		`Difference: ${finding.difference}`,
	].join("\n")).join("\n\n");
	if (constructionGroundings === undefined) return findingsText;
	const referenced = new Set(findings.flatMap(({ current_construction_refs }) => current_construction_refs));
	const hasUnavailableReferences = [...referenced].some((reference) => !groundingsByReference?.has(reference));
	const hasModifiedReferences = constructionGroundings.some(({ binding_status }) => binding_status === "modified");
	const groundingText = constructionGroundings
		.filter(({ construction_ref }) => referenced.has(construction_ref))
		.map((grounding) => [
			`[${grounding.construction_ref} code grounding]`,
			`Current source: solution.py:${grounding.current_line_start}-${grounding.current_line_end}`,
			`Current source SHA-256: ${grounding.current_source_sha256}`,
			`Exact block SHA-256: ${grounding.code_block_sha256}`,
			"Exact current code block:",
			"```python",
			grounding.code_block,
			"```",
		].join("\n"))
		.join("\n\n");
	const sections = [findingsText];
	if (hasUnavailableReferences) {
		sections.push(
			"[Unavailable Current-code grounding]",
			"An unavailable construction reference is retained in the finding but cannot currently be mapped to a unique source block. Update or remove it after checking the Current construction.",
		);
	}
	if (hasModifiedReferences) {
		sections.push(
			"[Modified Current-code grounding]",
			"A modified construction reference has been rebound to edited Current code. Pass its construction_ref directly to inspect_code_ablation for a fresh highlight; do not resubmit the code block.",
		);
	}
	if (groundingText.length > 0) {
		sections.push(
			"[Automatically resolved Current-code grounding]",
			"The exact blocks below identify Current CAD constructions referenced by the findings; they do not establish target correspondence.",
			groundingText,
		);
	}
	return sections.join("\n\n");
}

function persistedState(state: VisualFindingState): PersistedVisualFindingState {
	return {
		version: 2,
		profile: state.profile,
		fixture_id: state.fixtureId,
		locked_view_manifest_sha256: state.lockedViewManifestSha256,
		findings: cloneFindings(state.findings),
		next_finding_index: state.nextFindingIndex,
		revision: state.revision,
		started_at: state.startedAt,
		updated_at: isoNow(),
		history: state.history,
	};
}

async function persistVisualFindingState(workspace: string, state: VisualFindingState): Promise<void> {
	const root = resolve(workspace);
	const path = resolve(root, state.statePath);
	const rel = relative(root, path);
	if (rel === ".." || rel.startsWith("../") || isAbsolute(rel)) {
		throw new InfrastructureError("Visual findings state path escaped the fixture workspace");
	}
	await mkdir(dirname(path), { recursive: true });
	await writeJsonAtomic(path, persistedState(state));
}

export async function restoreVisualFindingState(options: {
	workspace: string;
	fixtureId: string;
	profile: VisualFindingProfile;
	lockedViewManifestSha256: string;
	statePath?: string;
}): Promise<VisualFindingState> {
	const statePath = options.statePath ?? VISUAL_FINDING_STATE_PATH;
	const stored = await readJson<PersistedVisualFindingState>(resolve(options.workspace, statePath)).catch(() => null);
	if (!stored) {
		return {
			profile: options.profile,
			fixtureId: options.fixtureId,
			lockedViewManifestSha256: options.lockedViewManifestSha256,
			findings: [],
			nextFindingIndex: 1,
			revision: 0,
			statePath,
			startedAt: isoNow(),
			history: [],
		};
	}
	if (
		stored.version !== 2 ||
		stored.profile !== VISUAL_FINDING_PROFILE ||
		stored.profile !== options.profile ||
		stored.fixture_id !== options.fixtureId ||
		stored.locked_view_manifest_sha256 !== options.lockedViewManifestSha256 ||
		!Array.isArray(stored.findings) ||
		!Number.isInteger(stored.next_finding_index) ||
		stored.next_finding_index < 1 ||
		!Number.isInteger(stored.revision) ||
		stored.revision < 0 ||
		!Array.isArray(stored.history)
	) throw new InfrastructureError("Persisted visual findings state does not match the continuation");
	for (const finding of stored.findings) assertStoredFinding(finding);
	if (new Set(stored.findings.map(({ finding_id }) => finding_id)).size !== stored.findings.length) {
		throw new InfrastructureError("Persisted visual findings contain duplicate IDs");
	}
	const maximumFindingIndex = stored.findings.reduce((maximum, { finding_id }) => (
		Math.max(maximum, Number.parseInt(finding_id.slice(1), 10))
	), 0);
	if (stored.next_finding_index <= maximumFindingIndex) {
		throw new InfrastructureError("Persisted next visual-finding index is invalid");
	}
	assertFindingsSize(stored.findings);
	return {
		profile: stored.profile,
		fixtureId: stored.fixture_id,
		lockedViewManifestSha256: stored.locked_view_manifest_sha256,
		findings: cloneFindings(stored.findings),
		nextFindingIndex: stored.next_finding_index,
		revision: stored.revision,
		statePath,
		startedAt: stored.started_at,
		history: stored.history,
	};
}

export function injectPersistentVisualFindings<T>(
	messages: T[],
	state: VisualFindingState,
	constructionGroundings?: readonly ActiveCodeConstructionGrounding[],
): T[] {
	const filtered = messages.filter((message) => {
		if (typeof message !== "object" || message === null) return true;
		return (message as { customType?: unknown }).customType !== VISUAL_FINDINGS_CONTEXT_TYPE;
	});
	if (state.findings.length === 0) return filtered;
	return [...filtered, {
		role: "custom",
		customType: VISUAL_FINDINGS_CONTEXT_TYPE,
		content: `[Persistent visual findings — revision ${state.revision}]\n${renderVisualFindings(state.findings, constructionGroundings)}`,
		display: false,
		timestamp: Date.now(),
	} as T];
}

export function createUpdateVisualFindingsTool(options: {
	workspace: string;
	state: VisualFindingState;
	getActiveConstructionRefs?: () => readonly string[];
}) {
	return defineTool({
		name: "update_visual_findings",
		label: "Manage visual findings",
		description: "Maintain a persistent collection of unresolved geometric differences between the target engineering drawing and the Current CAD. The collection may initially be empty and has no fixed finding count; never recreate or clear it at the start of a new iteration. Use explicit add, update, and delete operations. Every finding must describe, in order: `target_geometry` (what the engineering drawing depicts), `current_geometry` (what the corresponding Current CAD geometry looks like), and `difference` (only the unresolved geometric mismatch in count, position, orientation, shape, extent, or connectivity). Do not record similarities, correctly implemented or preserved geometry, resolved issues, repair progress, prior achievements, repair instructions, or expected improvements. Delete a finding when its entire difference is resolved, duplicated, unsupported, or no longer relevant; never turn it into a completion note. Keep construction registrations only in `current_construction_refs`, never inside prose. References are not cleared automatically when their code changes: the controller rebinds them to the modified source block, and the Agent can pass the same construction_ref to inspect_code_ablation for a fresh highlight without resubmitting code. The Agent may add or remove references when updating a finding. An empty reference list means the Current geometry is not code-localized. A construction reference names Current CAD geometry only and does not prove target correspondence or incorrectness. The collection is persisted and included in subsequent model context only after at least one finding exists.",
		parameters: UPDATE_VISUAL_FINDINGS_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			const findings = cloneFindings(options.state.findings);
			let nextFindingIndex = options.state.nextFindingIndex;
			const activeConstructionRefs = new Set(options.getActiveConstructionRefs?.() ?? []);
			const applied: AppliedVisualFindingOperation[] = [];
			for (const [operationIndex, operation] of params.operations.entries()) {
				if (operation.operation === "add") {
					const finding: VisualFinding = {
						finding_id: `F${String(nextFindingIndex).padStart(3, "0")}`,
						target_geometry: assertFindingText("target_geometry", operation.target_geometry),
						current_geometry: assertFindingText("current_geometry", operation.current_geometry),
						difference: assertFindingText("difference", operation.difference),
						current_construction_refs: assertConstructionRefs(
							operation.current_construction_refs ?? [],
							activeConstructionRefs,
						),
					};
					nextFindingIndex += 1;
					findings.push(finding);
					applied.push({ operation: "add", finding_id: finding.finding_id, finding: cloneFinding(finding) });
					continue;
				}
				const findingIndex = findings.findIndex(({ finding_id }) => finding_id === operation.finding_id);
				if (findingIndex < 0) {
					throw new Error(`operations[${operationIndex}] references unknown finding_id ${operation.finding_id}`);
				}
				if (operation.operation === "delete") {
					findings.splice(findingIndex, 1);
					applied.push({ operation: "delete", finding_id: operation.finding_id });
					continue;
				}
				if (
					operation.target_geometry === undefined &&
					operation.current_geometry === undefined &&
					operation.difference === undefined &&
					operation.current_construction_refs === undefined
				) throw new Error(`operations[${operationIndex}] update must change at least one field`);
				const existing = findings[findingIndex]!;
				const updated: VisualFinding = {
					finding_id: existing.finding_id,
					target_geometry: operation.target_geometry === undefined
						? existing.target_geometry
						: assertFindingText("target_geometry", operation.target_geometry),
					current_geometry: operation.current_geometry === undefined
						? existing.current_geometry
						: assertFindingText("current_geometry", operation.current_geometry),
					difference: operation.difference === undefined
						? existing.difference
						: assertFindingText("difference", operation.difference),
					current_construction_refs: operation.current_construction_refs === undefined
						? [...existing.current_construction_refs]
						: assertConstructionRefs(operation.current_construction_refs, activeConstructionRefs),
				};
				if (JSON.stringify(updated) === JSON.stringify(existing)) {
					throw new Error(`operations[${operationIndex}] does not change finding ${operation.finding_id}`);
				}
				findings[findingIndex] = updated;
				applied.push({ operation: "update", finding_id: updated.finding_id, finding: cloneFinding(updated) });
			}
			assertFindingsSize(findings);
			options.state.findings = findings;
			options.state.nextFindingIndex = nextFindingIndex;
			options.state.revision += 1;
			options.state.history.push({
				revision: options.state.revision,
				findings: cloneFindings(findings),
				operations: applied,
				created_at: isoNow(),
			});
			await persistVisualFindingState(options.workspace, options.state);
			const added = applied.filter(({ operation }) => operation === "add").map(({ finding_id }) => finding_id);
			const updated = applied.filter(({ operation }) => operation === "update").map(({ finding_id }) => finding_id);
			const deleted = applied.filter(({ operation }) => operation === "delete").map(({ finding_id }) => finding_id);
			return {
				content: [{
					type: "text" as const,
					text: [
						`Visual findings revision ${options.state.revision} has been saved.`,
						`Added: ${added.join(", ") || "none"}`,
						`Updated: ${updated.join(", ") || "none"}`,
						`Deleted: ${deleted.join(", ") || "none"}`,
						`Active finding count: ${findings.length}`,
						"This tool records the supplied visual judgment; it does not independently verify the geometric observations or target correspondence.",
					].join("\n"),
				}],
				details: {
					revision: options.state.revision,
					operation_count: applied.length,
					added_finding_ids: added,
					updated_finding_ids: updated,
					deleted_finding_ids: deleted,
					active_finding_count: findings.length,
					state_path: options.state.statePath,
				},
			};
		},
	});
}
