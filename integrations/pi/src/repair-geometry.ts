import { createHash, randomUUID } from "node:crypto";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import { defineTool, type ToolDefinition } from "@earendil-works/pi-coding-agent";
import { Type, type Static } from "typebox";
import { Value } from "typebox/value";

import { verifyShapeCandidateEvidence } from "./adopt-shape-candidate.js";
import { imageFromWorkspace, inspectPngFile, requireDrawingSuccess, runRestrictedSession } from "./drawing-reflection.js";
import { InfrastructureError } from "./errors.js";
import { isoNow, writeJsonAtomic } from "./files.js";
import { GEOMETRY_DIFFERENCE_INSTRUCTIONS } from "./geometry-repair-instructions.js";
import type { WorkerClient } from "./local-worker.js";
import type { ResolvedModel, RunController } from "./runner.js";
import { createShapeFeedbackTool, formatShapeTopologyFeedback, PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA } from "./shape-feedback.js";
import { sha256File } from "./solution.js";
import {
	PROTOCOL_VERSION, type AgentJob, type AgentResult, type DrawingViewManifest,
	type FixtureDefinition, type SubagentRunUsage, type WorkerPrepareResponse,
	type WorkerReflectionInspectionResponse,
} from "./types.js";
import { assertFormalUnchanged, currentFormalState, currentFrame } from "./visual-review.js";

const regionPattern = "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$";
const shortText = () => Type.String({ minLength: 1, maxLength: 2_000 });

export const REPAIR_GEOMETRY_PARAMETERS_SCHEMA = Type.Object({
	part: Type.String({ minLength: 1, maxLength: 500, description: "One feature or related feature group to examine." }),
	difference_type: Type.Union([Type.Literal("size"), Type.Literal("count"), Type.Literal("position"), Type.Literal("orientation"), Type.Literal("profile"), Type.Literal("connectivity")]),
	suspected_difference: shortText(),
	view_ids: Type.Array(Type.String({ pattern: regionPattern }), { minItems: 1, maxItems: 2, uniqueItems: true }),
}, { additionalProperties: false });
export type GeometryRepairTask = Static<typeof REPAIR_GEOMETRY_PARAMETERS_SCHEMA>;

export const SUBMIT_GEOMETRY_REPAIR_PARAMETERS_SCHEMA = Type.Object({
	status: Type.Union([Type.Literal("proposed"), Type.Literal("premise_rejected"), Type.Literal("insufficient_evidence"), Type.Literal("repair_failed"), Type.Literal("needs_split")]),
	summary: shortText(),
	evaluation_id: Type.Optional(Type.Union([Type.String(), Type.Null()], { description: "For proposed, copy evaluation_id from the selected Preview. Otherwise omit or use null; unused references are ignored." })),
	candidate_id: Type.Optional(Type.Union([Type.String(), Type.Null()], { description: "For proposed, copy candidate_id from the same Preview. Otherwise omit or use null; unused references are ignored." })),
}, { additionalProperties: false });
type RepairSubmission = Omit<Static<typeof SUBMIT_GEOMETRY_REPAIR_PARAMETERS_SCHEMA>, "evaluation_id" | "candidate_id"> & {
	evaluation_id?: string;
	candidate_id?: string;
};
type EvidenceImage = { region_id: string; path: string; sha256: string };
export interface GeometryRepairResult extends RepairSubmission {
	version: 1;
	repair_id: string;
	task: GeometryRepairTask;
	formal_source_sha256: string;
	formal_step_sha256: string;
	locked_view_manifest_sha256: string;
	images: EvidenceImage[];
	audit_path: string;
	patch_path?: string;
	topology_text?: string;
	error?: string;
}

export const GEOMETRY_REPAIR_SUBAGENT_TOOLS = [
	"read_repair_input", "inspect_comparison", "preview_shape_feedback", "submit_geometry_repair",
] as const;

export const GEOMETRY_REPAIR_SYSTEM_PROMPT = `Your task is to verify and repair one atomic geometric problem assigned by the Main Agent. The task specifies the part, difference type, suspected discrepancy, and relevant views. Follow the definition and repair boundary of the assigned difference type.

First verify the discrepancy. Compare the target engineering drawing, Current's 2D projections, and source code to establish what the drawing requires, what Current actually contains, and whether they refer to the same part. When needed, inspect the complete original drawing or other available views, using neighboring features and connections to locate the part. The Main Agent's judgment may be incorrect; preserve uncertainty when the evidence is unclear, and do not guess precise dimensions. If the discrepancy is absent, evidence is insufficient, or the task contains multiple independent problems, submit the corresponding outcome.

If the discrepancy is confirmed, establish the specific geometric requirement to satisfy and implement it through code changes. Address only that requirement and its necessary associated geometry. For example, adding a missing mounting foot requires its position, connection, and mounting hole to be correct; independent problems with other existing mounting feet are outside this repair's scope.

Use preview_shape_feedback to check your changes. Preview exactly one candidate per call. All code replacements are relative to the Current source at the start of this task. Submit cumulative changes on subsequent attempts; do not write an incremental patch against a previous unadopted Preview.

First check whether the required geometry actually appears at the specified location in Preview. Then compare it against Target to judge whether it satisfies the assigned requirement, and check affected geometry and relevant views for errors introduced by this change. Base your judgment on the actual projections; code intent and successful execution cannot replace visual evidence. If a relevant view contradicts your conclusion about the assigned requirement or reveals an error introduced by this change, investigate the reason. Do not claim completion of the assigned requirement while such a contradiction remains unexplained.

Revise an incorrect implementation. A failed candidate does not mean the drawing requirement is absent; changing your target judgment requires drawing evidence.

You are responsible only for the assigned atomic geometric requirement. Once Preview shows that this requirement is satisfied, call submit_geometry_repair with proposed and the inspected candidate's references. You may select an earlier candidate. Do not wait for other existing independent discrepancies to be fixed, and do not report them to Main. Necessary associated geometry and errors introduced by this change remain within this task's review scope.

In summary, briefly state only the outcome for the assigned requirement and its drawing or preview evidence. Do not list unrelated geometry problems, judge whether the entire CAD is repaired, explain the code, or paste the patch. If the assigned discrepancy is absent, cannot be reliably judged, has not been repaired, or contains independent problems, submit the corresponding outcome and its task-specific reason without candidate references. Correct submission-parameter errors without changing your geometric judgment merely to make the call succeed. A successful submission ends this task. You do not modify formal files; the Main Agent decides whether to adopt.`;

function sha(value: string | Buffer): string { return createHash("sha256").update(value).digest("hex"); }
function assertNonblank(value: unknown): void {
	if (typeof value === "string" && !value.trim()) throw new Error("Text fields must not be blank");
	if (Array.isArray(value)) value.forEach(assertNonblank);
	else if (value && typeof value === "object") Object.values(value).forEach(assertNonblank);
}
function submissionError(message: string): Error {
	return new Error(`Not submitted: ${message} This repair task is still open.`);
}
function formatPatch(result: GeometryRepairResult, patch: string): string {
	// Unchanged diff context can contain Markdown fences inside Python strings.
	let fenceLength = 3;
	for (const match of patch.matchAll(/`+/gu)) fenceLength = Math.max(fenceLength, match[0].length + 1);
	const fence = "`".repeat(fenceLength);
	return `Patch (evaluation_id=${result.evaluation_id}, candidate_id=${result.candidate_id}):\n${fence}diff\n${patch}${patch.endsWith("\n") ? "" : "\n"}${fence}`;
}

export function createRepairGeometryTool(options: {
	workspace: string;
	fixture: FixtureDefinition;
	stagedPaths: string[];
	worker: WorkerClient;
	controller: RunController;
	lockedManifest: DrawingViewManifest;
	resolvedModel: ResolvedModel;
	thinkingLevel: AgentJob["thinking_level"];
	getResult: () => AgentResult;
	onUsage: (usage: SubagentRunUsage) => Promise<void>;
	/** Dependency seam for deterministic integration checks; production uses the restricted SDK session. */
	runSession?: typeof runRestrictedSession;
}) {
	return defineTool({
		name: "repair_geometry",
		label: "Verify and repair one geometric discrepancy",
		description: [
			"Use this tool to have a SubAgent verify and repair one suspected geometric discrepancy between the target engineering drawing and the current CAD.",
			"In `part`, identify the specific part or related feature group, using neighboring features to clarify its location when needed. Choose one `difference_type`, describe what the drawing shows at that location and how the current CAD differs in `suspected_difference`, and provide one or two corresponding `view_ids`. You can leave uncertain points for the SubAgent to verify; no code is required.",
			"Assign only one specific geometric requirement per call. The SubAgent first verifies the discrepancy, then attempts to repair it through code changes and previews. Associated changes necessary to achieve that goal are allowed; other independent problems are left for later tasks.",
			"Returns one brief summary and evidence images. When a repair candidate is proposed, it also returns the complete code patch, with `evaluation_id` and `candidate_id` in the patch heading, and a Target | Current | Preview three-panel image for each view. The summary covers only the assigned requirement's outcome and drawing or preview evidence, without code explanations or reports of other existing independent discrepancies. When no candidate can be proposed, the summary gives the reason for this task, accompanied by Target/Current evidence images. Intermediate attempts stay in the child session.",
			"This tool does not modify the formal `solution.py` or `output.step`. After reviewing a candidate, call `adopt_shape_candidate` with the returned `evaluation_id` and `candidate_id` to apply its code changes to `solution.py`.",
			`Available view_ids: ${options.lockedManifest.regions.filter(r => r.selected).map(r => r.region_id).join(", ")}.`,
		].join("\n\n"),
		parameters: REPAIR_GEOMETRY_PARAMETERS_SCHEMA,
		executionMode: "sequential",
		async execute(_callId, params, signal) {
			if (!Value.Check(REPAIR_GEOMETRY_PARAMETERS_SCHEMA, params)) throw new Error("Provide one part, one difference_type, one suspected_difference, and one or two unique view_ids");
			assertNonblank(params);
			if (options.lockedManifest.projection_frame?.policy !== "fixed") throw new Error("Geometry repair requires a calibrated fixed-frame view manifest");
			const task = structuredClone(params);
			const manifest = structuredClone(options.lockedManifest);
			const manifestHash = sha(JSON.stringify(manifest));
			if (task.view_ids.some(id => !manifest.regions.some(r => r.region_id === id && r.selected))) throw new Error("Unknown or unavailable repair view_id");
			let activeToolSignal: AbortSignal | undefined;
			const invocationSignal = signal ? AbortSignal.any([signal, options.controller.signal]) : options.controller.signal;
			const worker: WorkerClient = {
				request: (workspace, request, requestOptions = {}) => options.worker.request(workspace, request, {
					...requestOptions,
					signal: AbortSignal.any([invocationSignal, ...(activeToolSignal ? [activeToolSignal] : []), ...(requestOptions.signal ? [requestOptions.signal] : [])]),
				}),
			};
			const formal = await currentFormalState({ ...options, worker });
			if (Buffer.byteLength(formal.source) > 512_000) throw new Error("Current source is too large for a repair session");
			const repairId = `repair-${randomUUID()}`;
			const outputRoot = join("drawing_feedback", "geometry-repair", repairId);
			const auditPath = join(outputRoot, "audit.json");
			const resultPath = join(outputRoot, "result.json");
			const baselineRoot = join(outputRoot, "baseline");
			await mkdir(join(options.workspace, baselineRoot), { recursive: true });
			await writeFile(join(options.workspace, baselineRoot, "solution.py"), formal.source, { mode: 0o600 });
			await copyFile(join(options.workspace, "output.step"), join(options.workspace, baselineRoot, "output.step"));
			await writeJsonAtomic(join(options.workspace, baselineRoot, "view_manifest.json"), manifest);
			const inputHashes = new Map<string, string>();
			for (const path of new Set([...options.stagedPaths, ...manifest.regions.filter(r => r.selected).map(r => r.crop_path), manifest.projection_frame!.anchor_step_path])) {
				const hash = await sha256File(join(options.workspace, path));
				if (!hash) throw new InfrastructureError(`Missing frozen repair input: ${path}`);
				inputHashes.set(path, hash);
			}
			if (inputHashes.get(manifest.projection_frame!.anchor_step_path) !== manifest.projection_frame!.anchor_step_sha256) throw new InfrastructureError("Repair projection anchor changed");
			const attempts: Array<{ index: number; evaluation_id?: string; candidate_id?: string; status: string; error?: string }> = [];
			const originalDrawings: Array<{ source_path: string; path: string; sha256: string }> = [];
			const evidenceViews = new Map<string, { target: EvidenceImage; comparison: EvidenceImage }>();
			let submitted: GeometryRepairResult | null = null;
			let sessionError: string | undefined;
			const persist = async () => writeJsonAtomic(join(options.workspace, auditPath), {
				version: 1, kind: "atomic_geometry_repair", repair_id: repairId, task,
				formal_source_sha256: formal.sourceSha256, formal_step_sha256: formal.stepSha256,
				locked_view_manifest_sha256: manifestHash, baseline_path: baselineRoot,
				input_hashes: Object.fromEntries(inputHashes), max_previews: null,
				preview_attempts: attempts,
				original_drawings: originalDrawings,
				evidence_views: Object.fromEntries(evidenceViews), result: submitted, error: sessionError,
				updated_at: isoNow(),
			});
			const checkFrozen = async () => {
				if (invocationSignal.aborted) throw new InfrastructureError("Geometry repair was cancelled");
				await assertFormalUnchanged(options.workspace, formal);
				if (sha(JSON.stringify(options.lockedManifest)) !== manifestHash) throw new InfrastructureError("Repair calibration changed during delegation");
				for (const [path, hash] of inputHashes) if (await sha256File(join(options.workspace, path)) !== hash) throw new InfrastructureError(`Frozen repair input changed: ${path}`);
				for (const item of originalDrawings) if (await sha256File(join(options.workspace, item.path)) !== item.sha256) throw new InfrastructureError("Original drawing snapshot changed during delegation");
				if (await sha256File(join(options.workspace, baselineRoot, "output.step")) !== formal.stepSha256) throw new InfrastructureError("Repair STEP snapshot does not match Current");
			};
			const inspectView = async (regionId: string) => {
				await checkFrozen();
				const previous = evidenceViews.get(regionId);
				if (previous) return previous;
				const frame = await currentFrame({ ...options, worker, manifest, formal, regionId, outputRoot: join(outputRoot, "views", regionId) });
				const response = await worker.request<WorkerReflectionInspectionResponse>(options.workspace, {
					protocol_version: PROTOCOL_VERSION, operation: "compose_reflection_inspection", region_id: regionId,
					crop_path: frame.referencePath, aligned_render_path: frame.currentRenderPath,
					mode: "coordinate_overview", output_dir: join(outputRoot, "views", regionId, "comparison"),
				}, { timeoutMs: 180_000 });
				requireDrawingSuccess(response, "compose_reflection_inspection");
				if (response.region_id !== regionId || response.mode !== "coordinate_overview") throw new InfrastructureError("Repair inspection returned the wrong view");
				const target = await inspectPngFile(options.workspace, frame.referencePath);
				const comparison = await inspectPngFile(options.workspace, response.context_path);
				if (comparison.sha256 !== response.sha256.context) throw new InfrastructureError("Repair comparison image hash mismatch");
				const item = {
					target: { region_id: regionId, path: frame.referencePath, sha256: target.sha256 },
					comparison: { region_id: regionId, path: response.context_path, sha256: comparison.sha256 },
				};
				evidenceViews.set(regionId, item);
				await checkFrozen();
				await persist();
				return item;
			};
			const loadImage = async (item: EvidenceImage) => {
				if ((await inspectPngFile(options.workspace, item.path)).sha256 !== item.sha256) throw new InfrastructureError("Repair evidence image changed");
				return imageFromWorkspace(options.workspace, item.path);
			};
			await checkFrozen();
			await persist();
			const initialImages: ImageContent[] = [];
			if (!options.stagedPaths.length) throw new InfrastructureError("Geometry repair requires the original engineering drawing inputs");
			const originals = await worker.request<WorkerPrepareResponse>(options.workspace, {
				protocol_version: PROTOCOL_VERSION, operation: "prepare_images", paths: options.stagedPaths, max_edge: 2000,
			});
			if (!Array.isArray(originals.images) ||
				options.stagedPaths.some(path => !originals.images.some(item => item.source_path === path)) ||
				originals.images.some(item => !options.stagedPaths.includes(item.source_path))) {
				throw new InfrastructureError("Original drawing preparation did not return all requested inputs");
			}
			await mkdir(join(options.workspace, baselineRoot, "original-drawings"), { recursive: true });
			for (const sourcePath of options.stagedPaths) {
				for (const item of originals.images.filter(image => image.source_path === sourcePath)) {
					await inspectPngFile(options.workspace, item.path);
					const snapshotPath = join(baselineRoot, "original-drawings", `${originalDrawings.length}.png`);
					await copyFile(join(options.workspace, item.path), join(options.workspace, snapshotPath));
					const prepared = await inspectPngFile(options.workspace, snapshotPath);
					originalDrawings.push({ source_path: sourcePath, path: snapshotPath, sha256: prepared.sha256 });
					initialImages.push(await imageFromWorkspace(options.workspace, snapshotPath));
				}
			}
			await checkFrozen();
			await persist();
			for (const id of task.view_ids) {
				const view = await inspectView(id);
				initialImages.push(await loadImage(view.target), await loadImage(view.comparison));
			}
			const guard = <T extends ToolDefinition>(tool: T): T => ({
				...tool,
				async execute(callId, arguments_, toolSignal, onUpdate, context) {
					if (submitted) throw new Error("This repair was already submitted");
					activeToolSignal = toolSignal;
					try {
						await checkFrozen();
						return await tool.execute(callId, arguments_, toolSignal, onUpdate, context);
					} finally {
						activeToolSignal = undefined;
						await checkFrozen();
					}
				},
			});
			const readTool = guard(defineTool({
				name: "read_repair_input", label: "Read repair source or original drawing",
				description: `Read frozen Current source or an original drawing input (zero-based page index, 0-${Math.max(0, options.stagedPaths.length - 1)}). No filesystem writes or arbitrary paths are available.`,
				parameters: Type.Object({ kind: Type.Union([Type.Literal("source"), Type.Literal("drawing")]), page: Type.Optional(Type.Integer({ minimum: 0 })) }, { additionalProperties: false }),
				executionMode: "sequential",
				async execute(_id, args): Promise<{ content: Array<TextContent | ImageContent>; details: { source_sha256?: string; path?: string } }> {
					if (args.kind === "source") return { content: [{ type: "text" as const, text: formal.source }], details: { source_sha256: formal.sourceSha256 } };
					const path = options.stagedPaths[args.page ?? 0];
					if (args.kind !== "drawing" || !path) throw new Error("Unknown original drawing page");
					const prepared = await worker.request<WorkerPrepareResponse>(options.workspace, { protocol_version: PROTOCOL_VERSION, operation: "prepare_images", paths: [path], max_edge: 2000 });
					const content: Array<TextContent | ImageContent> = [{ type: "text", text: `Original drawing input ${args.page ?? 0}: ${path}. Read dimensions, feature counts and relationships in their full drawing context.` }];
					for (const item of prepared.images) content.push(await imageFromWorkspace(options.workspace, item.path));
					return { content, details: { path } };
				},
			}));
			const inspectTool = guard(defineTool({
				name: "inspect_comparison", label: "Inspect frozen Target and Current",
				description: `Read one complete Target and its fixed-frame Current comparison. Available region_id values: ${manifest.regions.filter(r => r.selected).map(r => r.region_id).join(", ")}. Camera and transform cannot be changed.`,
				parameters: Type.Object({ region_id: Type.String({ pattern: regionPattern }) }, { additionalProperties: false }), executionMode: "sequential",
				async execute(_id, args) {
					const view = await inspectView(args.region_id);
					return { content: [{ type: "text" as const, text: `View ${args.region_id}: original Target, followed by the frozen Target/Current comparison.` }, await loadImage(view.target), await loadImage(view.comparison)], details: view };
				},
			}));
			const rawPreview = createShapeFeedbackTool({ ...options, worker, lockedManifest: manifest, allowSections: true });
			const previewTool = guard(defineTool({
				...rawPreview,
				parameters: PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA,
				description: rawPreview.description.replace(/Compare the projections[\s\S]*?Available region_id values:/u, "Compare Current and Preview against Target to check the assigned repair and any damage to other geometry. Revise the code and preview again as needed. To submit an inspected repair candidate, call submit_geometry_repair with evaluation_id and candidate_id; the Main Agent decides whether to adopt it.\n\nThis call does not modify formal files. Every patch remains relative to Current at the start of this task. Inspect any additional view before previewing it.\n\nAvailable region_id values:"),
				async execute(callId, args, toolSignal, onUpdate, context) {
					if (args.views.some(view => !evidenceViews.has(view.region_id))) throw new Error("Inspect each requested view before using it to evaluate a repair");
					const attempt: (typeof attempts)[number] = { index: attempts.length + 1, status: "running" };
					attempts.push(attempt);
					await persist();
					try {
						const preview = await rawPreview.execute(callId, args, toolSignal, onUpdate, context);
						attempt.evaluation_id = preview.details.evaluation_id;
						if (preview.details.candidates[0]) attempt.candidate_id = preview.details.candidates[0].id;
						attempt.status = preview.details.status;
						const content = preview.content.map(item => item.type === "text" ? { ...item, text: item.text.replace(/To adopt this candidate[^\n]*/gu, `To submit this preview, use evaluation_id=${attempt.evaluation_id} and candidate_id=${attempt.candidate_id}. Main decides adoption.`) } : item);
						return { content, details: preview.details };
					} catch (error) {
						attempt.status = "error";
						attempt.error = error instanceof Error ? error.message : String(error);
						throw error;
					} finally { await persist(); }
				},
			}));
			const submitTool = guard(defineTool({
				name: "submit_geometry_repair", label: "Submit one repair outcome",
				description: [
					"Finish this task by submitting the outcome for the assigned atomic geometric requirement to the Main Agent.",
					"Use proposed when you have inspected Preview and judge that the assigned requirement is satisfied without evident errors introduced by the change. Other existing independent discrepancies do not prevent submission. Use premise_rejected when drawing evidence contradicts the assigned discrepancy, insufficient_evidence when it cannot be reliably judged, repair_failed when the confirmed assigned discrepancy has not been repaired, or needs_split when the request contains independent problems.",
					"In summary, briefly state only this requirement's outcome and its drawing or preview evidence, or why this task could not be completed. Do not list other geometry problems, evaluate completion of the entire CAD, explain the code changes, or paste the patch.",
					"For proposed, copy evaluation_id and candidate_id from the same successful Preview in this task. You may select an earlier candidate that satisfies the requirement. Without a candidate, provide only status and summary; references may be omitted or null, and any unused references are ignored.",
					"The runtime attaches the selected candidate's complete patch and final preview images for Main. Without a candidate, it attaches existing Target/Current evidence images. A successful submission ends this task and returns a brief confirmation. Invalid submissions explain how to correct the call and leave the task open. This tool does not change formal files; Main decides adoption.",
				].join("\n\n"),
				parameters: SUBMIT_GEOMETRY_REPAIR_PARAMETERS_SCHEMA, executionMode: "sequential",
				async execute(_id, args) {
					if (!Value.Check(SUBMIT_GEOMETRY_REPAIR_PARAMETERS_SCHEMA, args)) throw submissionError("Provide a supported status, a nonblank summary of 1-2000 characters, and optional evaluation_id/candidate_id strings or null. Remove unsupported fields and retry.");
					if (!args.summary.trim()) throw submissionError("summary must state the assigned requirement's outcome and evidence; replace the blank summary and retry.");
					// Non-proposals never use candidate references. Normalize them away
					// before checking IDs so placeholders cannot block a negative verdict.
					const submission: RepairSubmission = { status: args.status, summary: args.summary };
					let images: EvidenceImage[];
					let patchPath: string | undefined;
					let topologyText: string | undefined;
					if (args.status === "proposed") {
						if (typeof args.evaluation_id !== "string" || !/^[0-9a-f]{16}-[0-9a-f]{8}$/u.test(args.evaluation_id)) throw submissionError("proposed requires a valid evaluation_id. Copy it from the selected Preview in this repair and retry.");
						if (typeof args.candidate_id !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/u.test(args.candidate_id)) throw submissionError("proposed requires a valid candidate_id. Copy it from the same Preview as evaluation_id and retry.");
						const selectedAttempt = attempts.find(item => item.evaluation_id === args.evaluation_id && item.candidate_id === args.candidate_id && (item.status === "completed" || item.status === "partial"));
						if (!selectedAttempt) throw submissionError("Candidate was not returned by a successful Preview in this repair. Copy both references from the same successful Preview in this task and retry.");
						const verified = await verifyShapeCandidateEvidence({ workspace: options.workspace, lockedManifest: manifest, evaluationId: args.evaluation_id, candidateId: args.candidate_id });
						if ([...task.view_ids, ...verified.audit.requested_views.map(view => view.region_id)].some(id => !verified.cards.some(card => card.region_id === id))) throw submissionError("Candidate lacks complete preview evidence for a requested task view. Preview the candidate in every requested task view, then submit the returned references.");
						submission.evaluation_id = args.evaluation_id;
						submission.candidate_id = args.candidate_id;
						images = verified.cards.map(card => ({ region_id: card.region_id, path: card.card_path, sha256: card.sha256 }));
						patchPath = join(outputRoot, "selected.patch");
						await writeFile(join(options.workspace, patchPath), verified.candidate.patch, { mode: 0o600 });
						topologyText = verified.audit.topology ? formatShapeTopologyFeedback(verified.audit.topology) : "Topology unavailable.";
					} else {
						images = [...evidenceViews.values()].slice(-3).map(view => view.comparison);
					}
					submitted = { ...submission, version: 1, repair_id: repairId, task, formal_source_sha256: formal.sourceSha256, formal_step_sha256: formal.stepSha256, locked_view_manifest_sha256: manifestHash, images, audit_path: auditPath, ...(patchPath ? { patch_path: patchPath } : {}), ...(topologyText ? { topology_text: topologyText } : {}) };
					await writeJsonAtomic(join(options.workspace, resultPath), submitted);
					await persist();
					return { content: [{ type: "text" as const, text: "Result submitted. This repair task is finished." }], details: { status: submission.status, repair_id: repairId } };
				},
			}));
			try {
				await (options.runSession ?? runRestrictedSession)({
					role: "geometry_repair", workspace: options.workspace, outputDir: outputRoot,
					resolvedModel: options.resolvedModel, thinkingLevel: options.thinkingLevel, controller: options.controller,
					systemPrompt: GEOMETRY_REPAIR_SYSTEM_PROMPT,
						prompt: `Fixture: ${options.fixture.id}\nTask (one suspected difference): ${JSON.stringify(task)}\n\nDifference type definition and repair boundary:\n${GEOMETRY_DIFFERENCE_INSTRUCTIONS[task.difference_type]}\n\nThe first ${originalDrawings.length} images are complete original drawing pages, in this order: ${originalDrawings.map((item, index) => `${index + 1}: ${item.source_path}`).join(", ")}. These preserve the full drawing context, including dimensions and other views. They are followed by the complete Target crop then Target/Current comparison for each selected view: ${task.view_ids.join(", ")}.\nOriginal drawing inputs available to reread: ${options.stagedPaths.map((path, index) => `${index}: ${path}`).join(", ")}.\nYour final candidate must include preview evidence for every task view.\nCurrent STEP snapshot: ${join(baselineRoot, "output.step")} (SHA-256 ${formal.stepSha256}).\n\nFrozen Current solution.py:\n${formal.source}`,
					images: initialImages, toolNames: GEOMETRY_REPAIR_SUBAGENT_TOOLS,
					tools: [readTool, inspectTool, previewTool, submitTool], getSubmitted: () => submitted,
					activeTimeoutMs: 10 * 60_000, wallTimeoutMs: 20 * 60_000,
					pauseActiveTimeoutForTools: ["preview_shape_feedback", "inspect_comparison", "read_repair_input"],
					resultPath, onUsage: options.onUsage, signal: invocationSignal,
				});
			} catch (error) {
				sessionError = error instanceof Error ? error.message : String(error);
				await persist();
				await checkFrozen();
				// Infrastructure failures are explicit tool failures, never a fabricated semantic verdict.
				throw error;
			}
			await checkFrozen();
			// Submission is controller-owned; a session's final prose is never the deliverable.
			const final: GeometryRepairResult | null = submitted;
			if (!final) throw new InfrastructureError("Repair SubAgent did not submit a structured outcome");
			const result = final as GeometryRepairResult;
			const content: Array<TextContent | ImageContent> = [{ type: "text", text: result.summary }];
			if (result.status === "proposed") {
				const verified = await verifyShapeCandidateEvidence({ workspace: options.workspace, lockedManifest: manifest, evaluationId: result.evaluation_id!, candidateId: result.candidate_id! });
				content.push({ type: "text", text: formatPatch(result, verified.candidate.patch) });
			}
			for (const item of result.images) {
				content.push(await loadImage(item));
			}
			await checkFrozen();
			return { content, details: result };
		},
	});
}
