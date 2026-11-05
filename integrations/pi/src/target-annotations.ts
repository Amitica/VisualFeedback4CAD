import { mkdir } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve } from "node:path";

import type { ImageContent } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import { InfrastructureError } from "./errors.js";
import { isoNow, readJson, writeJsonAtomic } from "./files.js";
import { TARGET_ANNOTATION_PROFILE, type TargetAnnotationProfile } from "./types.js";

export const TARGET_ANNOTATION_STATE_PATH = "drawing_feedback/target-annotations/state.json";
export const TARGET_ANNOTATIONS_CONTEXT_TYPE = "cadgenbench-persistent-target-annotations";

export type TargetAnnotationGeometry =
	| { kind: "point"; point: [number, number] }
	| { kind: "box"; box: [number, number, number, number] };

export interface TargetAnnotation {
	mark: number;
	name: string;
	description: string;
	region_id: string;
	geometry: TargetAnnotationGeometry;
	created_at: string;
	updated_at: string;
}

interface LegacyTargetAnnotation {
	name: string;
	description: string;
	region_id: string;
	box: [number, number, number, number];
	created_at: string;
	updated_at: string;
}

interface PersistedTargetAnnotationStateV1 {
	version: 1;
	profile: TargetAnnotationProfile;
	fixture_id: string;
	locked_view_manifest_sha256: string;
	annotations: LegacyTargetAnnotation[];
	revision: number;
	started_at: string;
	updated_at: string;
}

interface PersistedTargetAnnotationStateV2 {
	version: 2;
	profile: TargetAnnotationProfile;
	fixture_id: string;
	locked_view_manifest_sha256: string;
	annotations: TargetAnnotation[];
	next_mark: number;
	revision: number;
	started_at: string;
	updated_at: string;
}

type PersistedTargetAnnotationState = PersistedTargetAnnotationStateV1 | PersistedTargetAnnotationStateV2;

export interface TargetAnnotationState {
	profile: TargetAnnotationProfile;
	fixtureId: string;
	lockedViewManifestSha256: string;
	annotations: TargetAnnotation[];
	nextMark: number;
	revision: number;
	statePath: string;
	startedAt: string;
}

export interface TargetAnnotationRegistryEntry {
	mark: number;
	name: string;
	description: string;
	region_id: string;
	mode: "Point" | "Box";
	geometry: TargetAnnotationGeometry;
}

const MAX_ANNOTATIONS = 50;
const MAX_NAME_LENGTH = 80;
const MAX_DESCRIPTION_LENGTH = 2_000;

const addNameSchema = Type.String({
	minLength: 1,
	maxLength: MAX_NAME_LENGTH,
	description: "A short, unique, specific semantic name for this Target structure. This name becomes its update/delete handle.",
});

const existingNameSchema = Type.String({
	minLength: 1,
	maxLength: MAX_NAME_LENGTH,
	description: "The exact semantic name of an existing Target mark. Use its name, never its numeric visual mark.",
});

const newNameSchema = Type.String({
	minLength: 1,
	maxLength: MAX_NAME_LENGTH,
	description: "A new short, unique, specific semantic name for this existing Target mark.",
});

const descriptionSchema = Type.String({
	minLength: 1,
	maxLength: MAX_DESCRIPTION_LENGTH,
	description: "One Target-only natural-language sentence identifying this structure by its geometry and, where needed, its position, adjacency, or connectivity. Do not describe the Current CAD, a difference, repair instructions, code, progress, or completion.",
});

const pointSchema = Type.Tuple([
	Type.Integer({ minimum: 0, maximum: 1000 }),
	Type.Integer({ minimum: 0, maximum: 1000 }),
], {
	description: "The actual [x, y] anchor of one compact Target feature in the most recently inspected Target panel's independent 0-1000 coordinates. Do not move the anchor into blank space to place the number; badge placement is automatic.",
});

const boxSchema = Type.Tuple([
	Type.Integer({ minimum: 0, maximum: 1000 }),
	Type.Integer({ minimum: 0, maximum: 1000 }),
	Type.Integer({ minimum: 0, maximum: 1000 }),
	Type.Integer({ minimum: 0, maximum: 1000 }),
], {
	description: "A complete but tight [left, top, right, bottom] extent around one extended or compound Target structure in the most recently inspected Target panel's independent 0-1000 coordinates. Avoid unrelated adjacent structures where possible.",
});

const updateFields = {
	operation: Type.Literal("update"),
	name: existingNameSchema,
	new_name: Type.Optional(newNameSchema),
	description: Type.Optional(descriptionSchema),
};

export const ANNOTATE_TARGET_PARAMETERS_SCHEMA = Type.Union([
	Type.Object({ operation: Type.Literal("add"), name: addNameSchema, description: descriptionSchema, point: pointSchema }, { additionalProperties: false }),
	Type.Object({ operation: Type.Literal("add"), name: addNameSchema, description: descriptionSchema, box: boxSchema }, { additionalProperties: false }),
	Type.Object(updateFields, { additionalProperties: false }),
	Type.Object({ ...updateFields, point: pointSchema }, { additionalProperties: false }),
	Type.Object({ ...updateFields, box: boxSchema }, { additionalProperties: false }),
	Type.Object({ operation: Type.Literal("delete"), name: existingNameSchema }, { additionalProperties: false }),
], {
	description: "Add, correct, rename, or delete one persistent numeric Target mark. For add, provide point for one compact local feature or box for one extended or compound structure; never provide both.",
});

function cloneGeometry(geometry: TargetAnnotationGeometry): TargetAnnotationGeometry {
	return geometry.kind === "point"
		? { kind: "point", point: [...geometry.point] }
		: { kind: "box", box: [...geometry.box] };
}

function cloneAnnotation(annotation: TargetAnnotation): TargetAnnotation {
	return { ...annotation, geometry: cloneGeometry(annotation.geometry) };
}

function cloneAnnotations(annotations: readonly TargetAnnotation[]): TargetAnnotation[] {
	return annotations.map(cloneAnnotation);
}

function annotationName(value: string): string {
	const trimmed = value.trim();
	if (trimmed.length === 0 || trimmed.length > MAX_NAME_LENGTH || /[\r\n\u0000-\u001f\u007f]/u.test(trimmed)) {
		throw new Error(`Annotation name must be one line with 1-${MAX_NAME_LENGTH} visible characters`);
	}
	return trimmed;
}

function annotationNameKey(value: string): string {
	return value.normalize("NFKC").toLocaleLowerCase("en-US");
}

function annotationDescription(value: string): string {
	const trimmed = value.trim();
	if (trimmed.length === 0 || trimmed.length > MAX_DESCRIPTION_LENGTH) {
		throw new Error(`Annotation description must contain 1-${MAX_DESCRIPTION_LENGTH} characters`);
	}
	return trimmed;
}

function annotationPoint(value: readonly number[]): [number, number] {
	if (value.length !== 2 || value.some((coordinate) => !Number.isInteger(coordinate) || coordinate < 0 || coordinate > 1000)) {
		throw new Error("Annotation point must contain two integer Target-panel coordinates from 0 to 1000");
	}
	const [x, y] = value;
	if (x === undefined || y === undefined) throw new Error("Annotation point is incomplete");
	return [x, y];
}

function annotationBox(value: readonly number[]): [number, number, number, number] {
	if (value.length !== 4 || value.some((coordinate) => !Number.isInteger(coordinate) || coordinate < 0 || coordinate > 1000)) {
		throw new Error("Annotation box must contain four integer Target-panel coordinates from 0 to 1000");
	}
	const [left, top, right, bottom] = value;
	if (left === undefined || top === undefined || right === undefined || bottom === undefined) throw new Error("Annotation box is incomplete");
	if (right - left < 10 || bottom - top < 10) {
		throw new Error("Annotation box width and height must each be at least 10 Target-panel units");
	}
	return [left, top, right, bottom];
}

function annotationGeometry(value: object): TargetAnnotationGeometry | undefined {
	const candidate = value as { point?: readonly number[]; box?: readonly number[] };
	if (candidate.point !== undefined && candidate.box !== undefined) throw new Error("Target annotation must use point or box, never both");
	if (candidate.point !== undefined) return { kind: "point", point: annotationPoint(candidate.point) };
	if (candidate.box !== undefined) return { kind: "box", box: annotationBox(candidate.box) };
	return undefined;
}

function assertLegacyAnnotation(value: LegacyTargetAnnotation): void {
	if (
		typeof value !== "object" || value === null || typeof value.name !== "string" ||
		typeof value.description !== "string" || typeof value.region_id !== "string" ||
		!Array.isArray(value.box) || typeof value.created_at !== "string" || typeof value.updated_at !== "string"
	) throw new InfrastructureError("Persisted legacy target annotation is invalid");
	try {
		annotationName(value.name);
		annotationDescription(value.description);
		annotationBox(value.box);
	} catch (error) {
		throw new InfrastructureError(error instanceof Error ? error.message : "Persisted legacy target annotation is invalid");
	}
}

function assertAnnotation(value: TargetAnnotation): void {
	if (
		typeof value !== "object" || value === null || !Number.isSafeInteger(value.mark) || value.mark < 1 ||
		typeof value.name !== "string" || typeof value.description !== "string" || typeof value.region_id !== "string" ||
		typeof value.geometry !== "object" || value.geometry === null ||
		typeof value.created_at !== "string" || typeof value.updated_at !== "string"
	) throw new InfrastructureError("Persisted target annotation is invalid");
	try {
		annotationName(value.name);
		annotationDescription(value.description);
		if (value.geometry.kind === "point") annotationPoint(value.geometry.point);
		else if (value.geometry.kind === "box") annotationBox(value.geometry.box);
		else throw new Error("Target annotation geometry must be point or box");
	} catch (error) {
		throw new InfrastructureError(error instanceof Error ? error.message : "Persisted target annotation is invalid");
	}
}

function assertUniqueAnnotations(annotations: readonly TargetAnnotation[]): void {
	const nameKeys = annotations.map(({ name }) => annotationNameKey(name));
	if (new Set(nameKeys).size !== nameKeys.length) throw new InfrastructureError("Persistent target annotation names must be unique");
	const marks = annotations.map(({ mark }) => mark);
	if (new Set(marks).size !== marks.length) throw new InfrastructureError("Persistent target annotation numbers must be unique");
	if (annotations.length > MAX_ANNOTATIONS) throw new InfrastructureError(`Persistent target annotations must not exceed ${MAX_ANNOTATIONS}`);
}

function persistedState(state: TargetAnnotationState): PersistedTargetAnnotationStateV2 {
	return {
		version: 2,
		profile: state.profile,
		fixture_id: state.fixtureId,
		locked_view_manifest_sha256: state.lockedViewManifestSha256,
		annotations: cloneAnnotations(state.annotations),
		next_mark: state.nextMark,
		revision: state.revision,
		started_at: state.startedAt,
		updated_at: isoNow(),
	};
}

async function persistTargetAnnotationState(workspace: string, state: TargetAnnotationState): Promise<void> {
	const root = resolve(workspace);
	const path = resolve(root, state.statePath);
	const rel = relative(root, path);
	if (rel === ".." || rel.startsWith("../") || isAbsolute(rel)) throw new InfrastructureError("Target annotation state path escaped the fixture workspace");
	await mkdir(dirname(path), { recursive: true });
	await writeJsonAtomic(path, persistedState(state));
}

function validateStoredState(
	stored: PersistedTargetAnnotationState,
	options: { fixtureId: string; profile: TargetAnnotationProfile; lockedViewManifestSha256: string },
): void {
	if (
		(stored.version !== 1 && stored.version !== 2) || stored.profile !== TARGET_ANNOTATION_PROFILE ||
		stored.profile !== options.profile || stored.fixture_id !== options.fixtureId ||
		stored.locked_view_manifest_sha256 !== options.lockedViewManifestSha256 || !Array.isArray(stored.annotations) ||
		!Number.isInteger(stored.revision) || stored.revision < 0 || typeof stored.started_at !== "string"
	) throw new InfrastructureError("Persisted target annotation state does not match the continuation");
}

export async function restoreTargetAnnotationState(options: {
	workspace: string;
	fixtureId: string;
	profile: TargetAnnotationProfile;
	lockedViewManifestSha256: string;
	statePath?: string;
}): Promise<TargetAnnotationState> {
	const statePath = options.statePath ?? TARGET_ANNOTATION_STATE_PATH;
	const stored = await readJson<PersistedTargetAnnotationState>(resolve(options.workspace, statePath)).catch(() => null);
	if (!stored) {
		return {
			profile: options.profile,
			fixtureId: options.fixtureId,
			lockedViewManifestSha256: options.lockedViewManifestSha256,
			annotations: [],
			nextMark: 1,
			revision: 0,
			statePath,
			startedAt: isoNow(),
		};
	}
	validateStoredState(stored, options);
	if (stored.version === 1) {
		for (const annotation of stored.annotations) assertLegacyAnnotation(annotation);
		const annotations = stored.annotations.map((annotation, index): TargetAnnotation => ({
			mark: index + 1,
			name: annotation.name,
			description: annotation.description,
			region_id: annotation.region_id,
			geometry: { kind: "box", box: annotationBox(annotation.box) },
			created_at: annotation.created_at,
			updated_at: annotation.updated_at,
		}));
		assertUniqueAnnotations(annotations);
		const migrated: TargetAnnotationState = {
			profile: stored.profile,
			fixtureId: stored.fixture_id,
			lockedViewManifestSha256: stored.locked_view_manifest_sha256,
			annotations,
			nextMark: annotations.length + 1,
			revision: stored.revision,
			statePath,
			startedAt: stored.started_at,
		};
		await persistTargetAnnotationState(options.workspace, migrated);
		return migrated;
	}
	for (const annotation of stored.annotations) assertAnnotation(annotation);
	assertUniqueAnnotations(stored.annotations);
	if (!Number.isSafeInteger(stored.next_mark) || stored.next_mark < 1) throw new InfrastructureError("Persisted target annotation next_mark is invalid");
	if (stored.annotations.some(({ mark }) => mark >= stored.next_mark)) {
		throw new InfrastructureError("Persisted target annotation next_mark must exceed every assigned number");
	}
	return {
		profile: stored.profile,
		fixtureId: stored.fixture_id,
		lockedViewManifestSha256: stored.locked_view_manifest_sha256,
		annotations: cloneAnnotations(stored.annotations),
		nextMark: stored.next_mark,
		revision: stored.revision,
		statePath,
		startedAt: stored.started_at,
	};
}

export function targetAnnotationsForRegion(state: TargetAnnotationState | undefined, regionId: string): TargetAnnotation[] {
	return cloneAnnotations(state?.annotations.filter(({ region_id }) => region_id === regionId) ?? []);
}

function annotationMode(annotation: TargetAnnotation): "Point" | "Box" {
	return annotation.geometry.kind === "point" ? "Point" : "Box";
}

function registryEntry(annotation: TargetAnnotation): TargetAnnotationRegistryEntry {
	return {
		mark: annotation.mark,
		name: annotation.name,
		description: annotation.description,
		region_id: annotation.region_id,
		mode: annotationMode(annotation),
		geometry: cloneGeometry(annotation.geometry),
	};
}

export function targetAnnotationRegistry(annotations: readonly TargetAnnotation[]): TargetAnnotationRegistryEntry[] {
	return [...annotations]
		.sort((left, right) => left.mark - right.mark)
		.map(registryEntry);
}

export function renderTargetAnnotationRegistry(
	annotations: readonly TargetAnnotation[],
	revision: number,
	shownRegionId?: string,
): string {
	const header = `Current Target mark registry — revision ${revision}`;
	if (annotations.length === 0) return `${header}\n\n(empty)`;
	const ordered = [...annotations].sort((left, right) => left.mark - right.mark);
	const grouped = new Map<string, TargetAnnotation[]>();
	if (shownRegionId !== undefined) grouped.set(shownRegionId, []);
	for (const annotation of ordered) {
		const entries = grouped.get(annotation.region_id) ?? [];
		entries.push(annotation);
		grouped.set(annotation.region_id, entries);
	}
	const regions = [...grouped.entries()].map(([regionId, entries]) => {
		const regionHeader = `${regionId}${regionId === shownRegionId ? " — shown in the returned image" : ""}`;
		if (entries.length === 0) return `${regionHeader}\n(no marks)`;
		return [
			regionHeader,
			...entries.map((annotation) => [
				`[${annotation.mark}] ${annotation.name} · ${annotationMode(annotation)}`,
				annotation.description,
			].join("\n")),
		].join("\n\n");
	});
	return [header, ...regions].join("\n\n");
}

export function renderTargetAnnotationContext(state: TargetAnnotationState): string {
	return renderTargetAnnotationRegistry(state.annotations, state.revision);
}

function renderVisibleTargetMarks(annotations: readonly TargetAnnotation[], regionId: string): string {
	const visible = annotations
		.filter(({ region_id }) => region_id === regionId)
		.sort((left, right) => left.mark - right.mark);
	if (visible.length === 0) return "No Target marks remain in this image.";
	return [
		"Marks shown in this image:",
		...visible.map((annotation) => `[${annotation.mark}] ${annotation.name} · ${annotationMode(annotation)}`),
	].join("\n\n");
}

export function injectPersistentTargetAnnotations<T>(messages: T[], state: TargetAnnotationState): T[] {
	const filtered = messages.filter((message) => {
		if (typeof message !== "object" || message === null) return true;
		return (message as { customType?: unknown }).customType !== TARGET_ANNOTATIONS_CONTEXT_TYPE;
	});
	if (state.annotations.length === 0) return filtered;
	return [...filtered, {
		role: "custom",
		customType: TARGET_ANNOTATIONS_CONTEXT_TYPE,
		content: renderTargetAnnotationContext(state),
		display: false,
		timestamp: Date.now(),
	} as T];
}

const ANNOTATE_TARGET_TOOL_DESCRIPTION = "Add numbered marks to the Target engineering drawing on the left side of the comparison you just inspected, or adjust or delete existing marks. Coordinates use the Target panel's own 0-1000 coordinate system. Use point for a hole, endpoint, local bend, or another feature that can be identified by one position; the point must land on the actual feature. The tool places the number in nearby blank space and draws a short leader back to that position. Use box for a grille, bracket, frame, or another structure with a clear extent, and enclose its complete range. Give every mark a unique short name and one description of only the Target structure. The tool returns a newly marked comparison, the affected mark's description, and the number and name of every mark shown in that image.";

export function createTargetAnnotationTool(options: {
	workspace: string;
	state: TargetAnnotationState;
	getLatestInspectedRegionId: () => string | undefined;
	renderComparison: (regionId: string, annotations: readonly TargetAnnotation[]) => Promise<ImageContent>;
}) {
	return defineTool({
		name: "annotate_target",
		label: "Annotate Target drawing",
		description: ANNOTATE_TARGET_TOOL_DESCRIPTION,
		parameters: ANNOTATE_TARGET_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_toolCallId, params) {
			const latestRegionId = options.getLatestInspectedRegionId();
			if (!latestRegionId) throw new Error("Call inspect_comparison before annotating the Target drawing");
			const annotations = cloneAnnotations(options.state.annotations);
			const existingIndex = annotations.findIndex(({ name }) => annotationNameKey(name) === annotationNameKey(params.name));
			const now = isoNow();
			let affected: TargetAnnotation;
			let nextMark = options.state.nextMark;
			if (params.operation === "add") {
				if (existingIndex >= 0) throw new Error(`Target annotation name already exists: ${params.name}`);
				if (annotations.length >= MAX_ANNOTATIONS) throw new Error(`Persistent target annotations must not exceed ${MAX_ANNOTATIONS}`);
				const geometry = annotationGeometry(params);
				if (!geometry) throw new Error("Target annotation add requires point or box");
				const annotation: TargetAnnotation = {
					mark: nextMark,
					name: annotationName(params.name),
					description: annotationDescription(params.description),
					region_id: latestRegionId,
					geometry,
					created_at: now,
					updated_at: now,
				};
				annotations.push(annotation);
				nextMark += 1;
				affected = cloneAnnotation(annotation);
			} else {
				if (existingIndex < 0) throw new Error(`Unknown target annotation: ${params.name}`);
				const existing = annotations[existingIndex]!;
				if (existing.region_id !== latestRegionId) throw new Error(`Inspect comparison region ${existing.region_id} before changing target annotation ${existing.name}`);
				if (params.operation === "delete") {
					annotations.splice(existingIndex, 1);
					affected = cloneAnnotation(existing);
				} else {
					const requestedGeometry = annotationGeometry(params);
					if (params.new_name === undefined && params.description === undefined && requestedGeometry === undefined) {
						throw new Error("Target annotation update must change the name, description, point, or box");
					}
					const nextName = params.new_name === undefined ? existing.name : annotationName(params.new_name);
					const duplicate = annotations.some((annotation, index) => index !== existingIndex && annotationNameKey(annotation.name) === annotationNameKey(nextName));
					if (duplicate) throw new Error(`Target annotation name already exists: ${nextName}`);
					const nextDescription = params.description === undefined ? existing.description : annotationDescription(params.description);
					const nextGeometry = requestedGeometry ?? cloneGeometry(existing.geometry);
					if (nextName === existing.name && nextDescription === existing.description && JSON.stringify(nextGeometry) === JSON.stringify(existing.geometry)) {
						throw new Error(`Target annotation update does not change ${existing.name}`);
					}
					const updated: TargetAnnotation = { ...existing, name: nextName, description: nextDescription, geometry: nextGeometry, updated_at: now };
					annotations[existingIndex] = updated;
					affected = cloneAnnotation(updated);
				}
			}
			assertUniqueAnnotations(annotations);
			const image = await options.renderComparison(latestRegionId, annotations);
			const nextRevision = options.state.revision + 1;
			await persistTargetAnnotationState(options.workspace, { ...options.state, annotations, nextMark, revision: nextRevision });
			options.state.annotations = annotations;
			options.state.nextMark = nextMark;
			options.state.revision = nextRevision;
			const status = params.operation === "add"
				? `Added Target mark [${affected.mark}] ${affected.name} · ${annotationMode(affected)}.`
				: params.operation === "update"
					? `Updated Target mark [${affected.mark}] ${affected.name} · ${annotationMode(affected)}. Its number is unchanged.`
					: `Deleted Target mark [${affected.mark}] ${affected.name}.`;
			const visibleMarks = renderVisibleTargetMarks(annotations, latestRegionId);
			const hasVisibleMarks = annotations.some(({ region_id }) => region_id === latestRegionId);
			const affectedDescription = params.operation === "delete" ? "" : `\n${affected.description}`;
			return {
				content: [
					{
						type: "text" as const,
						text: `${status}${affectedDescription}\n\n${visibleMarks}${hasVisibleMarks ? "\n\nInspect the returned image. Update an inaccurate mark by its semantic name." : ""}`,
					},
					image,
				],
				details: {
					operation: params.operation,
					mark: affected.mark,
					name: affected.name,
					region_id: affected.region_id,
					affected: registryEntry(affected),
					revision: options.state.revision,
					annotation_count: annotations.length,
					registry: targetAnnotationRegistry(annotations),
				},
			};
		},
	});
}
