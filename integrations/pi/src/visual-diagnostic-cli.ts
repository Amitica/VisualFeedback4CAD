#!/usr/bin/env node

import { createHash } from "node:crypto";
import { mkdir, readFile } from "node:fs/promises";
import { extname, isAbsolute, join, relative, resolve } from "node:path";

import type { ImageContent } from "@earendil-works/pi-ai";
import {
	createAgentSession,
	DefaultResourceLoader,
	getAgentDir,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";

import { sessionUsageTotals } from "./context.js";
import { isoNow, writeJsonAtomic } from "./files.js";
import { resolveVisionModel } from "./runner.js";
import { VISUAL_FINDING_PROFILE } from "./types.js";
import {
	createUpdateVisualFindingsTool,
	injectPersistentVisualFindings,
	renderVisualFindings,
	restoreVisualFindingState,
} from "./visual-findings.js";

type DiagnosticMode = "target-only" | "drawing-vs-cad";

interface Options {
	mode: DiagnosticMode;
	fixtureId: string;
	outputDir: string;
	images: string[];
	provider: string;
	model: string;
	thinkingLevel: "off" | "minimal" | "low" | "medium" | "high";
}

const SYSTEM_PROMPT = [
	"You are a visual CAD analysis agent in a controlled diagnostic experiment.",
	"Use only the attached images as visual evidence. You have no source code, CAD-editing tools, file tools, external analysis, or hidden reference answer.",
	"Report only observations supported by the images, and state uncertainty when the evidence is ambiguous.",
].join("\n");

export function targetOnlyPrompt(fixtureId: string): string {
	return [
		`Conduct a controlled visual analysis of the original engineering drawing for CADGenBench Case ${fixtureId}.`,
		"The attached image is the complete original drawing and is the only task evidence available in this experiment. No current CAD model is provided.",
		"Analyze the depicted target geometry as completely and precisely as the image permits. Record reliable geometric observations as well as any important uncertainty. Do not discuss source code or propose repairs.",
		"You must finish by calling `update_visual_findings` with one or more `add` operations. The collection starts empty and has no fixed finding count. For each record, describe the depicted target geometry in `target_geometry`; because this mode supplies no Current CAD, state that `current_geometry` and `difference` cannot be assessed from the available evidence. Do not return a prose-only analysis without recording the findings.",
	].join("\n\n");
}

export function drawingVsCadPrompt(fixtureId: string): string {
	return [
		`Conduct a controlled visual comparison for CADGenBench Case ${fixtureId}.`,
		"The attached image is the standardized aligned 2D comparison for one engineering-drawing region. Its left panel shows the original input and its right panel shows the current CAD projection at the corresponding view. This image is the only task evidence available in this experiment.",
		"Compare the depicted target and current CAD as completely and precisely as the image permits. Record only reliable unresolved geometric differences and any important uncertainty. Do not record similarities, resolved issues, repair progress, source code, or proposed repairs.",
		"You must finish by calling `update_visual_findings` with one or more `add` operations. The collection starts empty and has no fixed finding count. Each finding must describe `target_geometry`, then `current_geometry`, then their unresolved `difference`. Do not return a prose-only analysis without recording the findings.",
	].join("\n\n");
}

function usage(): string {
	return [
		"Usage:",
		"  visual-diagnostic-cli --mode target-only --fixture-id 103 --output-dir <dir> --image <drawing>",
		"  visual-diagnostic-cli --mode drawing-vs-cad --fixture-id 103 --output-dir <dir> --image <comparison-sheet>",
		"Options:",
		"  --provider <provider>       Default: ucloud",
		"  --model <model>             Default: gpt-5.6-sol",
		"  --thinking <level>          Default: high",
	].join("\n");
}

function parseArgs(argv: string[]): Options {
	let mode: DiagnosticMode | undefined;
	let fixtureId: string | undefined;
	let outputDir: string | undefined;
	const images: string[] = [];
	let provider = "ucloud";
	let model = "gpt-5.6-sol";
	let thinkingLevel: Options["thinkingLevel"] = "high";
	for (let index = 0; index < argv.length; index += 1) {
		const argument = argv[index]!;
		const value = argv[index + 1];
		if (argument === "--help" || argument === "-h") {
			console.log(usage());
			process.exit(0);
		}
		if (value === undefined) throw new Error(`Missing value for ${argument}`);
		if (argument === "--mode") {
			if (value !== "target-only" && value !== "drawing-vs-cad") {
				throw new Error(`Unsupported diagnostic mode: ${value}`);
			}
			mode = value;
		} else if (argument === "--fixture-id") fixtureId = value;
		else if (argument === "--output-dir") outputDir = value;
		else if (argument === "--image") images.push(value);
		else if (argument === "--provider") provider = value;
		else if (argument === "--model") model = value;
		else if (argument === "--thinking") {
			if (!["off", "minimal", "low", "medium", "high"].includes(value)) {
				throw new Error(`Unsupported thinking level: ${value}`);
			}
			thinkingLevel = value as Options["thinkingLevel"];
		} else throw new Error(`Unknown argument: ${argument}`);
		index += 1;
	}
	if (!mode || !fixtureId || !outputDir) throw new Error(`Missing required arguments\n${usage()}`);
	const expectedImages = 1;
	if (images.length !== expectedImages) {
		throw new Error(`${mode} requires exactly ${expectedImages} image${expectedImages === 1 ? "" : "s"}`);
	}
	return { mode, fixtureId, outputDir, images, provider, model, thinkingLevel };
}

function mimeType(path: string): "image/png" | "image/jpeg" | "image/webp" {
	const extension = extname(path).toLowerCase();
	if (extension === ".png") return "image/png";
	if (extension === ".jpg" || extension === ".jpeg") return "image/jpeg";
	if (extension === ".webp") return "image/webp";
	throw new Error(`Unsupported image extension: ${extension}`);
}

async function loadImages(paths: string[]): Promise<{ content: ImageContent[]; hashes: string[] }> {
	const content: ImageContent[] = [];
	const hashes: string[] = [];
	for (const path of paths) {
		const bytes = await readFile(path);
		hashes.push(createHash("sha256").update(bytes).digest("hex"));
		content.push({ type: "image", data: bytes.toString("base64"), mimeType: mimeType(path) });
	}
	return { content, hashes };
}

async function main(): Promise<void> {
	const options = parseArgs(process.argv.slice(2));
	const outputDir = resolve(options.outputDir);
	await mkdir(join(outputDir, "sessions"), { recursive: true });
	const resolvedImages = options.images.map((path) => resolve(path));
	const images = await loadImages(resolvedImages);
	const evidenceHash = createHash("sha256")
		.update(options.mode)
		.update("\0")
		.update(images.hashes.join("\0"))
		.digest("hex");
	const prompt = options.mode === "target-only"
		? targetOnlyPrompt(options.fixtureId)
		: drawingVsCadPrompt(options.fixtureId);
	const state = await restoreVisualFindingState({
		workspace: outputDir,
		fixtureId: options.fixtureId,
		profile: VISUAL_FINDING_PROFILE,
		lockedViewManifestSha256: evidenceHash,
	});
	if (state.revision !== 0) throw new Error("Diagnostic output directory already contains visual findings");
	const startedAt = isoNow();
	await writeJsonAtomic(join(outputDir, "experiment.json"), {
		version: 1,
		fixture_id: options.fixtureId,
		mode: options.mode,
		provider: options.provider,
		model: options.model,
		thinking_level: options.thinkingLevel,
		prompt,
		system_prompt: SYSTEM_PROMPT,
		image_paths: resolvedImages.map((path) => isAbsolute(path) ? path : resolve(path)),
		image_sha256: images.hashes,
		evidence_sha256: evidenceHash,
		started_at: startedAt,
	});
	const resolvedModel = await resolveVisionModel(options.provider, options.model);
	const settingsManager = SettingsManager.inMemory({}, { projectTrusted: false });
	const resourceLoader = new DefaultResourceLoader({
		cwd: outputDir,
		agentDir: getAgentDir(),
		settingsManager,
		noExtensions: true,
		noSkills: true,
		noPromptTemplates: true,
		noThemes: true,
		noContextFiles: true,
		systemPrompt: SYSTEM_PROMPT,
		appendSystemPrompt: [],
		systemPromptOverride: () => SYSTEM_PROMPT,
		appendSystemPromptOverride: () => [],
		extensionFactories: [
			(pi) => {
				pi.on("context", (event) => ({
					messages: injectPersistentVisualFindings(event.messages, state),
				}));
			},
		],
	});
	await resourceLoader.reload();
	const visualFindingsTool = createUpdateVisualFindingsTool({ workspace: outputDir, state });
	const created = await createAgentSession({
		cwd: outputDir,
		agentDir: getAgentDir(),
		modelRuntime: resolvedModel.runtime,
		model: resolvedModel.model,
		thinkingLevel: options.thinkingLevel,
		tools: ["update_visual_findings"],
		customTools: [visualFindingsTool],
		resourceLoader,
		sessionManager: SessionManager.create(outputDir, join(outputDir, "sessions")),
		settingsManager,
	});
	const session = created.session;
	let reminderUsed = false;
	try {
		const activeTools = session.getActiveToolNames();
		if (activeTools.length !== 1 || activeTools[0] !== "update_visual_findings") {
			throw new Error(`Restricted diagnostic tool set is invalid: ${activeTools.join(", ")}`);
		}
		await session.prompt(prompt, { images: images.content, expandPromptTemplates: false });
		if (state.revision === 0) {
			reminderUsed = true;
			await session.prompt(
				"You have not recorded the required visual analysis. Call `update_visual_findings` now with one or more evidence-based `add` operations. Do not add a prose-only answer.",
				{ expandPromptTemplates: false },
			);
		}
		if (state.revision === 0) throw new Error("Agent did not record visual findings");
		const sessionFile = session.getSessionStats().sessionFile;
		await writeJsonAtomic(join(outputDir, "result.json"), {
			version: 1,
			fixture_id: options.fixtureId,
			mode: options.mode,
			status: "completed",
			visual_findings: state.findings,
			revision: state.revision,
			reminder_used: reminderUsed,
			usage: sessionUsageTotals(session),
			...(sessionFile ? {
				session_file: isAbsolute(sessionFile) ? relative(outputDir, sessionFile) : sessionFile,
			} : {}),
			started_at: startedAt,
			finished_at: isoNow(),
		});
		console.log(renderVisualFindings(state.findings));
	} finally {
		session.dispose();
	}
}

await main();
