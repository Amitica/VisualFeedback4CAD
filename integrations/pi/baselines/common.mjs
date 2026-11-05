import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { createWriteStream } from "node:fs";
import { copyFile, lstat, mkdir, readFile, readdir, rename, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { safeFile, validateFormalWorkspace } from "../portable-full-method/run.mjs";

export const integrationRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const repoRoot = resolve(integrationRoot, "../..");
export const readJson = async path => JSON.parse(await readFile(path, "utf8"));
export async function writeJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}-${randomUUID()}`;
  await writeFile(temporary, JSON.stringify(value, null, 2) + "\n");
  await rename(temporary, path);
}
export const sha256 = async path => createHash("sha256").update(await readFile(path)).digest("hex");

export async function findInitial(root, id) {
  for (const path of [join(root, id), join(root, "cases", id, "01-generate", id), join(root, "generation", id), join(root, "01-generate", id), join(root, "00-main-agent", id)]) {
    if ((await lstat(join(path, ".cadgenbench-pi-job.json")).catch(() => null))?.isFile()) return resolve(path);
  }
  throw Error(`No original Main workspace for case ${id} under ${root}; an evaluation-only STEP export cannot resume its original Agent session`);
}

export async function discoverInitials(root) {
  const ids = new Set();
  for (const parent of [root, join(root, "cases"), join(root, "generation"), join(root, "00-main-agent"), join(root, "01-generate")]) {
    for (const entry of await readdir(parent, { withFileTypes: true }).catch(() => [])) {
      if (entry.isDirectory() && /^[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(entry.name)) {
        try { await findInitial(root, entry.name); ids.add(entry.name); } catch {}
      }
    }
  }
  return [...ids].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

export async function exportEvaluation(source, destination, metadata = {}) {
  await mkdir(destination, { recursive: true });
  for (const file of ["solution.py", "output.step", "result.json", "agent_result.json", "baseline_result.json"]) {
    const path = join(source, file);
    if ((await lstat(path).catch(() => null))?.isFile()) await copyFile(await safeFile(source, file), join(destination, file));
  }
  if (!(await lstat(join(destination, "output.step")).catch(() => null))?.isFile()) throw Error(`Missing model to evaluate: ${source}`);
  const binding = { ...metadata, source_workspace: resolve(source), source_sha256: await sha256(join(destination, "solution.py")), step_sha256: await sha256(join(destination, "output.step")) };
  await writeJson(join(destination, "baseline_provenance.json"), binding);
  return binding;
}

export async function stagePaperCase(source, destination, method, config) {
  const formal = await validateFormalWorkspace(source, config);
  await mkdir(destination, { recursive: true });
  await mkdir(join(destination, "initial"), { recursive: true });
  for (const file of ["solution.py", "output.step", "agent_result.json", ".cadgenbench-pi-job.json", ...formal.job.staged_paths]) {
    const from = await safeFile(source, file);
    const target = join(destination, file === "agent_result.json" ? "initial/agent_result.json" : file);
    await mkdir(dirname(target), { recursive: true });
    await copyFile(from, target);
  }
  await copyFile(join(destination, "solution.py"), join(destination, "initial/solution.py"));
  await copyFile(join(destination, "output.step"), join(destination, "initial/output.step"));
  await writeJson(join(destination, "baseline-job.json"), { ...config, method, fixture_id: formal.result.fixture_id, description: formal.job.fixture.description, staged_paths: formal.job.staged_paths, initial_source_sha256: formal.source, initial_step_sha256: formal.step, initial_workspace: resolve(source) });
  return formal;
}

export async function command(argv, { log, cwd = repoRoot, timeoutMs } = {}) {
  if (timeoutMs !== undefined && (!Number.isFinite(timeoutMs) || timeoutMs <= 0)) throw Error("Command timeout must be positive");
  if (log) await mkdir(dirname(log), { recursive: true });
  const stream = log ? createWriteStream(log, { flags: "a" }) : undefined;
  await new Promise((yes, no) => {
    const child = spawn(argv[0], argv.slice(1), { cwd, env: process.env, stdio: ["ignore", "pipe", "pipe"] });
    let timedOut = false;
    let killTimer;
    const timer = timeoutMs === undefined ? undefined : setTimeout(() => {
      timedOut = true;
      child.kill("SIGINT");
      killTimer = setTimeout(() => child.kill("SIGKILL"), 30_000);
    }, timeoutMs);
    const interrupt = () => child.kill("SIGINT");
    process.once("SIGINT", interrupt);
    child.stdout.on("data", d => { process.stdout.write(d); stream?.write(d); });
    child.stderr.on("data", d => { process.stderr.write(d); stream?.write(d); });
    let settled = false;
    const finish = error => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      clearTimeout(killTimer);
      process.removeListener("SIGINT", interrupt);
      stream?.end();
      if (error) no(error); else yes();
    };
    child.once("error", finish);
    child.once("close", code => finish(timedOut ? Error(`Command timed out after ${timeoutMs} ms: ${argv[0]}`) : code === 0 ? undefined : Error(`Command exited ${code}: ${argv[0]}`)));
  });
}
