import { copyFile, mkdir, readdir, readFile, realpath, stat } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { parse } from "yaml";

import { UsageError } from "./errors.js";
import type { FixtureDefinition } from "./types.js";

interface DescriptionYaml {
	description?: unknown;
	input_files?: unknown;
	input_type?: unknown;
	task_type?: unknown;
}

function requireString(value: unknown, label: string): string {
	if (typeof value !== "string" || value.trim() === "") {
		throw new UsageError(`${label} must be a non-empty string`);
	}
	return value.trim();
}

function pathInside(root: string, candidate: string): boolean {
	const rel = relative(root, candidate);
	return rel === "" || (!rel.startsWith(`..${sep}`) && rel !== ".." && !isAbsolute(rel));
}

export async function loadFixture(dataInputsDir: string, id: string): Promise<FixtureDefinition> {
	if (!/^[A-Za-z0-9._-]+$/.test(id)) {
		throw new UsageError(`Invalid fixture id: ${id}`);
	}
	const root = await realpath(dataInputsDir);
	const sourceDir = await realpath(join(root, id)).catch(() => {
		throw new UsageError(`Fixture ${id} does not exist under ${dataInputsDir}`);
	});
	if (!pathInside(root, sourceDir)) {
		throw new UsageError(`Fixture ${id} resolves outside the inputs directory`);
	}
	const yamlPath = join(sourceDir, "description.yaml");
	const parsed = parse(await readFile(yamlPath, "utf8")) as DescriptionYaml;
	const description = requireString(parsed.description, `${id}.description`);
	if (!Array.isArray(parsed.input_files) || parsed.input_files.length === 0) {
		throw new UsageError(`${id}.input_files must be a non-empty array`);
	}
	const inputFiles = parsed.input_files.map((value, index) =>
		requireString(value, `${id}.input_files[${index}]`),
	);
	const inputType = requireString(parsed.input_type, `${id}.input_type`);
	const taskType = parsed.task_type === undefined ? "generation" : requireString(parsed.task_type, `${id}.task_type`);
	if (taskType !== "generation") {
		throw new UsageError(`Fixture ${id} is ${taskType}; the Pi integration supports generation only`);
	}
	if (inputType !== "text+image") {
		throw new UsageError(`Fixture ${id} has input_type=${inputType}; the Pi integration requires text+image`);
	}
	for (const inputFile of inputFiles) {
		if (isAbsolute(inputFile) || inputFile.split(/[\\/]/u).includes("..")) {
			throw new UsageError(`Fixture ${id} has an unsafe input path: ${inputFile}`);
		}
		const source = await realpath(join(sourceDir, inputFile)).catch(() => {
			throw new UsageError(`Fixture ${id} input is missing: ${inputFile}`);
		});
		if (!pathInside(sourceDir, source) || !(await stat(source)).isFile()) {
			throw new UsageError(`Fixture ${id} input is not a regular file inside the fixture: ${inputFile}`);
		}
	}
	return { id, description, inputFiles, inputType, taskType, sourceDir };
}

export async function discoverGenerationFixtures(dataInputsDir: string): Promise<string[]> {
	const entries = await readdir(dataInputsDir, { withFileTypes: true });
	const ids: string[] = [];
	for (const entry of entries) {
		if (!entry.isDirectory() || entry.name.startsWith(".")) continue;
		try {
			await loadFixture(dataInputsDir, entry.name);
			ids.push(entry.name);
		} catch (error) {
			if (!(error instanceof UsageError) || !/supports generation only|requires text\+image/u.test(error.message)) {
				throw error;
			}
		}
	}
	return ids.sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));
}

export async function stageFixtureInputs(
	fixture: FixtureDefinition,
	workspace: string,
): Promise<string[]> {
	const inputDir = join(workspace, "inputs");
	await mkdir(inputDir, { recursive: true });
	const paths: string[] = [];
	for (const [index, inputFile] of fixture.inputFiles.entries()) {
		const stagedName = `${String(index).padStart(2, "0")}-${basename(inputFile)}`;
		await copyFile(join(fixture.sourceDir, inputFile), join(inputDir, stagedName));
		paths.push(join("inputs", stagedName));
	}
	return paths;
}

export function defaultDataInputsDir(repositoryRoot: string): string {
	return resolve(process.env.CADGENBENCH_DATA_DIR ?? join(repositoryRoot, "data"), "inputs");
}

export function repositoryRootFromModule(moduleDirectory: string): string {
	return resolve(moduleDirectory, "..", "..", "..");
}

export function fixtureIdFromWorkspace(workspace: string): string {
	return basename(dirname(join(workspace, "."))) || basename(workspace);
}
