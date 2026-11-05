import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { discoverGenerationFixtures, loadFixture, stageFixtureInputs } from "../src/fixtures.js";

const temporaryDirectories: string[] = [];

async function temporaryInputs(): Promise<string> {
	const root = await mkdtemp(join(tmpdir(), "pi-fixtures-"));
	temporaryDirectories.push(root);
	return root;
}

async function fixture(root: string, id: string, yaml: string, images: string[]): Promise<void> {
	const directory = join(root, id);
	await mkdir(directory, { recursive: true });
	await writeFile(join(directory, "description.yaml"), yaml);
	for (const image of images) await writeFile(join(directory, image), image);
}

afterEach(async () => {
	await Promise.all(temporaryDirectories.splice(0).map(async (directory) => await rm(directory, { recursive: true })));
});

describe("fixture selection", () => {
	it("preserves description.yaml image order for multi-image inputs", async () => {
		const root = await temporaryInputs();
		await fixture(
			root,
			"127",
			"description: make it\ninput_files: [back.png, front.png]\ninput_type: text+image\n",
			["front.png", "back.png"],
		);
		const loaded = await loadFixture(root, "127");
		expect(loaded.inputFiles).toEqual(["back.png", "front.png"]);
		const workspace = join(root, "workspace");
		expect(await stageFixtureInputs(loaded, workspace)).toEqual([
			join("inputs", "00-back.png"),
			join("inputs", "01-front.png"),
		]);
	});

	it("rejects editing and non-image fixtures and discovers only compatible generation fixtures", async () => {
		const root = await temporaryInputs();
		await fixture(root, "1", "description: ok\ninput_files: [a.png]\ninput_type: text+image\n", ["a.png"]);
		await fixture(
			root,
			"2",
			"description: edit\ninput_files: [a.png]\ninput_type: text+image\ntask_type: editing\n",
			["a.png"],
		);
		await fixture(root, "3", "description: text\ninput_files: [a.png]\ninput_type: text\n", ["a.png"]);
		expect(await discoverGenerationFixtures(root)).toEqual(["1"]);
		await expect(loadFixture(root, "2")).rejects.toThrow("generation only");
		await expect(loadFixture(root, "3")).rejects.toThrow("requires text+image");
	});

	it("ignores hidden dataset cache directories during discovery", async () => {
		const root = await temporaryInputs();
		await fixture(root, "145", "description: ok\ninput_files: [a.png]\ninput_type: text+image\n", ["a.png"]);
		await mkdir(join(root, ".cache", "huggingface"), { recursive: true });
		expect(await discoverGenerationFixtures(root)).toEqual(["145"]);
	});
});
