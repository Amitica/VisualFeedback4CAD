import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

export async function readJson<T>(path: string): Promise<T> {
	return JSON.parse(await readFile(path, "utf8")) as T;
}

export async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
	await mkdir(dirname(path), { recursive: true });
	const temporary = `${path}.tmp-${process.pid}-${Date.now()}`;
	await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
	await rename(temporary, path);
}

export function isoNow(): string {
	return new Date().toISOString();
}

export function elapsedSeconds(startedAt: string): number {
	return Math.max(0, (Date.now() - Date.parse(startedAt)) / 1000);
}

