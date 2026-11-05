#!/usr/bin/env node

import { copyFile, mkdir, readFile, readdir } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";

import {
	createAgentSession,
	DefaultResourceLoader,
	getAgentDir,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";

import { sessionUsageTotals } from "./context.js";
import {
	createDrawingScaleAgentTools,
	DRAWING_SCALE_AGENT_SYSTEM_PROMPT,
	DRAWING_SCALE_AGENT_TOOLS,
	drawingScaleAgentPrompt,
	hashFile,
	type DrawingScaleAgentState,
} from "./drawing-scale-agent.js";
import { imageFromWorkspace } from "./drawing-reflection.js";
import { InfrastructureError } from "./errors.js";
import { isoNow, writeJsonAtomic } from "./files.js";
import { LocalWorker } from "./local-worker.js";
import { resolveVisionModel, RunController } from "./runner.js";
import {
	PROTOCOL_VERSION,
	type DrawingViewManifest,
	type WorkerInspectResponse,
} from "./types.js";

interface Options {
	fixtureId: string;
	sourceWorkspace: string;
	manifestPath: string;
	drawings: string[];
	outputDir: string;
	provider: string;
	model: string;
	thinkingLevel: "off" | "minimal" | "low" | "medium" | "high";
	pythonBinary: string;
}

function usage(): string {
	return [
		"Usage:",
		"  drawing-scale-agent-cli --fixture-id <id> --source-workspace <dir> --manifest <json> --drawing <image> --output-dir <dir>",
		"Options:",
		"  --drawing <path>       Repeat once per original page, in page order.",
		"  --provider <name>      Default: ucloud",
		"  --model <id>           Default: gpt-5.6-sol",
		"  --thinking <level>     Default: high",
		"  --python <path>        Default: python",
	].join("\n");
}

function parseArgs(argv: string[]): Options {
	let fixtureId: string | undefined;
	let sourceWorkspace: string | undefined;
	let manifestPath: string | undefined;
	let outputDir: string | undefined;
	const drawings: string[] = [];
	let provider = "ucloud";
	let model = "gpt-5.6-sol";
	let thinkingLevel: Options["thinkingLevel"] = "high";
	let pythonBinary = "python";
	for (let index = 0; index < argv.length; index += 1) {
		const argument = argv[index]!;
		if (argument === "--help" || argument === "-h") {
			console.log(usage());
			process.exit(0);
		}
		const value = argv[index + 1];
		if (value === undefined) throw new Error(`Missing value for ${argument}`);
		if (argument === "--fixture-id") fixtureId = value;
		else if (argument === "--source-workspace") sourceWorkspace = value;
		else if (argument === "--manifest") manifestPath = value;
		else if (argument === "--drawing") drawings.push(value);
		else if (argument === "--output-dir") outputDir = value;
		else if (argument === "--provider") provider = value;
		else if (argument === "--model") model = value;
		else if (argument === "--thinking") {
			if (!["off", "minimal", "low", "medium", "high"].includes(value)) {
				throw new Error(`Unsupported thinking level ${value}`);
			}
			thinkingLevel = value as Options["thinkingLevel"];
		} else if (argument === "--python") pythonBinary = value;
		else throw new Error(`Unknown argument ${argument}`);
		index += 1;
	}
	if (!fixtureId || !sourceWorkspace || !manifestPath || !outputDir || drawings.length === 0) {
		throw new Error(`Missing required argument\n${usage()}`);
	}
	return {
		fixtureId,
		sourceWorkspace: resolve(sourceWorkspace),
		manifestPath: resolve(manifestPath),
		drawings: drawings.map((path) => resolve(path)),
		outputDir: resolve(outputDir),
		provider,
		model,
		thinkingLevel,
		pythonBinary,
	};
}

function assertManifest(value: unknown, pageCount: number): asserts value is DrawingViewManifest {
	if (!value || typeof value !== "object") throw new Error("Drawing view manifest is not an object");
	const manifest = value as Partial<DrawingViewManifest>;
	if (manifest.version !== 3 || !Array.isArray(manifest.regions) || manifest.regions.length === 0) {
		throw new Error("Drawing view manifest is invalid or empty");
	}
	for (const region of manifest.regions) {
		if (!region.selected || region.page < 0 || region.page >= pageCount) {
			throw new Error(`Manifest region ${region.region_id} has no usable page or selected camera`);
		}
	}
}

async function prepareStandaloneWorkspace(options: Options): Promise<{
	manifest: DrawingViewManifest;
	pagePaths: string[];
}> {
	await mkdir(options.outputDir, { recursive: true });
	const entries = await readdir(options.outputDir);
	if (entries.length > 0) throw new Error(`Output directory must be empty: ${options.outputDir}`);
	await Promise.all([
		copyFile(join(options.sourceWorkspace, "solution.py"), join(options.outputDir, "solution.py")),
		copyFile(join(options.sourceWorkspace, "output.step"), join(options.outputDir, "output.step")),
		mkdir(join(options.outputDir, "inputs"), { recursive: true }),
		mkdir(join(options.outputDir, "evidence", "regions"), { recursive: true }),
		mkdir(join(options.outputDir, "sessions"), { recursive: true }),
	]);
	const pagePaths: string[] = [];
	for (const [index, source] of options.drawings.entries()) {
		const extension = basename(source).toLowerCase().endsWith(".png") ? ".png" : ".jpg";
		const destination = join("inputs", `page-${String(index).padStart(3, "0")}${extension}`);
		await copyFile(source, join(options.outputDir, destination));
		pagePaths.push(destination);
	}
	const parsed = JSON.parse(await readFile(options.manifestPath, "utf8")) as unknown;
	assertManifest(parsed, pagePaths.length);
	const manifest: DrawingViewManifest = {
		...parsed,
		regions: await Promise.all(parsed.regions.map(async (region) => {
			const sourceCrop = resolve(options.sourceWorkspace, region.crop_path);
			const cropPath = join("evidence", "regions", region.region_id, "crop.png");
			await mkdir(dirname(join(options.outputDir, cropPath)), { recursive: true });
			await copyFile(sourceCrop, join(options.outputDir, cropPath));
			return { ...region, crop_path: cropPath };
		})),
	};
	await writeJsonAtomic(join(options.outputDir, "locked-view-manifest.json"), manifest);
	return { manifest, pagePaths };
}

export async function runDrawingScaleAgentCli(argv: string[]): Promise<void> {
	const options = parseArgs(argv);
	const prepared = await prepareStandaloneWorkspace(options);
	const worker = new LocalWorker(options.pythonBinary);
	const controller = new RunController();
	const inspected = await worker.request<WorkerInspectResponse>(options.outputDir, {
		protocol_version: PROTOCOL_VERSION,
		operation: "inspect",
	});
	if (
		inspected.artifact.status !== "valid" ||
		inspected.artifact.validation?.is_valid !== true ||
		inspected.artifact.validation.is_watertight !== true ||
		!inspected.artifact.sha256
	) throw new InfrastructureError("Drawing Scale Agent requires a valid and watertight output.step");
	const sourceSha256 = await hashFile(join(options.outputDir, "solution.py"));
	const source = await readFile(join(options.outputDir, "solution.py"), "utf8");
	const state: DrawingScaleAgentState = {
		source,
		sourceSha256,
		stepSha256: inspected.artifact.sha256,
		artifact: inspected.artifact,
		lockedTarget: null,
		previews: [],
		submitted: null,
	};
	const prompt = drawingScaleAgentPrompt({
		fixtureId: options.fixtureId,
		pagePaths: prepared.pagePaths,
	});
	const startedAt = isoNow();
	await writeJsonAtomic(join(options.outputDir, "experiment.json"), {
		version: 1,
		fixture_id: options.fixtureId,
		provider: options.provider,
		model: options.model,
		thinking_level: options.thinkingLevel,
		system_prompt: DRAWING_SCALE_AGENT_SYSTEM_PROMPT,
		prompt,
		page_paths: prepared.pagePaths,
		source_sha256: sourceSha256,
		step_sha256: inspected.artifact.sha256,
		started_at: startedAt,
	});
	const resolvedModel = await resolveVisionModel(options.provider, options.model);
	const settingsManager = SettingsManager.inMemory({}, { projectTrusted: false });
	const resourceLoader = new DefaultResourceLoader({
		cwd: options.outputDir,
		agentDir: getAgentDir(),
		settingsManager,
		noExtensions: true,
		noSkills: true,
		noPromptTemplates: true,
		noThemes: true,
		noContextFiles: true,
		systemPrompt: DRAWING_SCALE_AGENT_SYSTEM_PROMPT,
		appendSystemPrompt: [],
		systemPromptOverride: () => DRAWING_SCALE_AGENT_SYSTEM_PROMPT,
		appendSystemPromptOverride: () => [],
	});
	await resourceLoader.reload();
	const tools = createDrawingScaleAgentTools({
		workspace: options.outputDir,
		worker,
		controller,
		manifest: prepared.manifest,
		pageCount: prepared.pagePaths.length,
		state,
	});
	const created = await createAgentSession({
		cwd: options.outputDir,
		agentDir: getAgentDir(),
		modelRuntime: resolvedModel.runtime,
		model: resolvedModel.model,
		thinkingLevel: options.thinkingLevel,
		tools: [...DRAWING_SCALE_AGENT_TOOLS],
		customTools: [...tools],
		resourceLoader,
		sessionManager: SessionManager.create(options.outputDir, join(options.outputDir, "sessions")),
		settingsManager,
	});
	const session = created.session;
	try {
		const activeNames = session.getActiveToolNames();
		if (
			activeNames.length !== DRAWING_SCALE_AGENT_TOOLS.length ||
			activeNames.some((name) => !DRAWING_SCALE_AGENT_TOOLS.includes(name as (typeof DRAWING_SCALE_AGENT_TOOLS)[number]))
		) throw new InfrastructureError(`Drawing Scale Agent tool restriction failed: ${activeNames.join(", ")}`);
		const images = await Promise.all(prepared.pagePaths.map(async (path) => (
			await imageFromWorkspace(options.outputDir, path)
		)));
		await session.prompt(prompt, { images, expandPromptTemplates: false });
		for (let attempt = 0; attempt < 2 && !state.submitted; attempt += 1) {
			await session.prompt(
				attempt === 0
					? "You have not submitted the required dimension pair. If the target is not locked, lock it from original-drawing evidence first. Then inspect Current CAD, preview the best available exact or proxy anchor in every locked view, and call submit_dimension_match. Abstention is not available."
					: "Call submit_dimension_match now with the best previewed pair. If exact correspondence is unavailable, submit a proxy with its concrete limitation instead of returning prose.",
				{ expandPromptTemplates: false },
			);
		}
		if (!state.submitted) throw new Error("Drawing Scale Agent did not submit a structured result");
		const [finalSourceSha256, finalStepSha256] = await Promise.all([
			hashFile(join(options.outputDir, "solution.py")),
			hashFile(join(options.outputDir, "output.step")),
		]);
		if (finalSourceSha256 !== sourceSha256 || finalStepSha256 !== state.stepSha256) {
			throw new InfrastructureError("Drawing Scale Agent workspace source or STEP changed during analysis");
		}
		await writeJsonAtomic(join(options.outputDir, "result.json"), {
			...state.submitted,
			fixture_id: options.fixtureId,
			usage: sessionUsageTotals(session),
			started_at: startedAt,
			finished_at: isoNow(),
		});
		console.log(JSON.stringify(state.submitted, null, 2));
	} finally {
		session.dispose();
	}
}

if (isAbsolute(process.argv[1] ?? "") && resolve(process.argv[1]!) === resolve(new URL(import.meta.url).pathname)) {
	await runDrawingScaleAgentCli(process.argv.slice(2));
}
