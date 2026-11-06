#!/usr/bin/env node
// Host orchestration only. CAD workers, including View Search, run in Docker.
import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { createWriteStream } from "node:fs";
import { copyFile, lstat, mkdir, readFile, realpath, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
export const repositoryRoot = resolve(here, "../../..");
const jobFile = ".cadgenbench-pi-job.json";
const dimensionProfile = "aligned-dimension-subagent-v1";
const help = `Full CAD method: Main generation -> automatic View Search -> dimensions -> Main repair

  node integrations/pi/portable-full-method/run.mjs [--cases 102 103 145 146]
  node integrations/pi/portable-full-method/run.mjs --all-generation
  node integrations/pi/portable-full-method/run.mjs --dry-run
  node integrations/pi/portable-full-method/run.mjs --doctor

Options: --config <json>, --output-dir <new-dir>, --data-inputs-dir <dir>, --image <tag>
All stages use the configured provider/model/thinking level. Case concurrency defaults to 1.
--dry-run prints the complete command plan without Docker, credentials, or model calls.
--doctor checks the image, model registration, credentials and geometry without model inference.
An actual run always requires a NEW output directory; existing experiments are never overwritten.
`;

export async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

async function writeJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}-${randomUUID()}`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`);
  await rename(temporary, path);
}

export async function digestFile(path) {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

export async function safeFile(root, path) {
  if (typeof path !== "string" || !path || isAbsolute(path)) throw Error(`Unsafe artifact path: ${path}`);
  const absolute = resolve(root, path);
  const rel = relative(resolve(root), absolute);
  if (!rel || rel === ".." || rel.startsWith(`..${sep}`) || isAbsolute(rel)) throw Error(`Artifact escapes workspace: ${path}`);
  const info = await lstat(absolute);
  const canonical = await realpath(absolute);
  const canonicalRoot = await realpath(root);
  const canonicalRelative = relative(canonicalRoot, canonical);
  if (!info.isFile() || info.isSymbolicLink() || canonicalRelative.startsWith(`..${sep}`) || canonicalRelative === ".." || isAbsolute(canonicalRelative)) {
    throw Error(`Artifact is not a regular workspace file: ${path}`);
  }
  return absolute;
}

export function validateConfig(config) {
  if (!config || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(config.experiment_id ?? "")) throw Error("Invalid experiment_id");
  for (const key of ["provider", "model", "image", "data_inputs_dir"]) {
    if (typeof config[key] !== "string" || !config[key].trim()) throw Error(`Invalid ${key}`);
  }
  if (!["off", "minimal", "low", "medium", "high"].includes(config.thinking_level)) throw Error("Invalid thinking_level");
  if (!Array.isArray(config.cases) || config.cases.length === 0 || new Set(config.cases).size !== config.cases.length || config.cases.some(id => !/^[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(id))) throw Error("Invalid/duplicate cases");
  for (const key of ["generation_timeout_minutes", "view_search_timeout_minutes", "repair_timeout_minutes", "search_cpus"]) {
    if (!Number.isFinite(config[key]) || config[key] <= 0) throw Error(`Invalid ${key}`);
  }
  if (!/^[1-9][0-9]*[mg]$/iu.test(config.search_memory)) throw Error("Invalid search_memory");
  if (!Number.isInteger(config.case_parallelism ?? 1) || (config.case_parallelism ?? 1) < 1 || (config.case_parallelism ?? 1) > 64) throw Error("case_parallelism must be an integer in [1, 64]");
  if (!Number.isFinite(config.subagent_timeout_multiplier ?? 1) || (config.subagent_timeout_multiplier ?? 1) < 1 || (config.subagent_timeout_multiplier ?? 1) > 12) throw Error("subagent_timeout_multiplier must be in [1, 12]");
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/u.test(config.docker_network ?? "bridge")) throw Error("Invalid docker_network");
  return config;
}

export async function forEachCaseBounded(plans, concurrency, runCase) {
  let cursor = 0;
  let stopped = false;
  await Promise.all(Array.from({ length: Math.min(concurrency, plans.length) }, async () => {
    while (!stopped) {
      const plan = plans[cursor++];
      if (!plan) return;
      if (await runCase(plan) === false) stopped = true;
    }
  }));
  return !stopped;
}

export function casePaths(outputDir, id) {
  const root = join(outputDir, "cases", id);
  const generation = join(root, "01-generate");
  const search = join(root, "02-view-search");
  const repair = join(root, "03-repair");
  return { root, generation, search, repair, source: join(generation, id), view: join(search, id), final: join(repair, "run", id) };
}

export function buildPlan(config, outputDir, piAgentDir, environmentNames = []) {
  for (const path of [outputDir, piAgentDir]) {
    if (/[\n\r,]/u.test(path)) throw Error("Docker mount paths cannot contain commas or newlines");
  }
  const integration = join(repositoryRoot, "integrations", "pi");
  return config.cases.map(id => {
    const paths = casePaths(outputDir, id);
    const searchContainer = `cad-full-search-${createHash("sha256").update(`${outputDir}\0${id}`).digest("hex").slice(0, 16)}`;
    const identity = typeof process.getuid === "function" ? ["--user", `${process.getuid()}:${process.getgid()}`] : [];
    return {
      fixture_id: id, paths, search_container: searchContainer,
      generate: [process.execPath, join(integration, "dist/src/cli.js"), "run", id, "--base-agent-only", "--provider", config.provider, "--model", config.model, "--thinking-level", config.thinking_level, "--parallel", "1", "--image", config.image, "--data-inputs-dir", config.data_inputs_dir, "--output-dir", paths.generation],
      search: ["docker", "run", "--rm", "--init", "--name", searchContainer, "--network", config.docker_network ?? "bridge", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--cpus", String(config.search_cpus), "--memory", config.search_memory, "--pids-limit", "512", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g,mode=1777", ...identity, "--env", "HOME=/tmp/home", "--env", "PI_CODING_AGENT_DIR=/pi-agent", ...environmentNames.flatMap(name => ["--env", name]), "--mount", `type=bind,src=${outputDir},dst=${outputDir}`, "--mount", `type=bind,src=${piAgentDir},dst=/pi-agent`, "--workdir", outputDir, config.image, "node", "/opt/cadgenbench-pi/dist/src/drawing-search-cli.js", paths.source, "--output-dir", paths.search, "--python", "python"],
      repair: [process.execPath, join(integration, "dist/src/main-self-reflect-continuation-cli.js"), paths.source, "--output-dir", paths.repair, "--image", config.image, "--provider", config.provider, "--model", config.model, "--thinking-level", config.thinking_level, "--timeout-minutes", String(config.repair_timeout_minutes), "--inspect-view-search-root", paths.search, "--dimension-annotation-profile", dimensionProfile],
    };
  });
}

export async function validateFormalWorkspace(workspace, config, { requireBaseOnly = true } = {}) {
  const [job, result] = await Promise.all([readJson(await safeFile(workspace, jobFile)), readJson(await safeFile(workspace, "agent_result.json"))]);
  for (const record of [job, result]) {
    if (record.provider !== config.provider || record.model !== config.model || record.thinking_level !== config.thinking_level) throw Error(`Model/thinking mismatch: ${workspace}`);
    if (requireBaseOnly && record.agent_mode !== "base-only") throw Error(`Not a base-only Main generation: ${workspace}`);
  }
  if (result.status !== "valid" || job.fixture?.id !== result.fixture_id || basename(workspace) !== result.fixture_id) throw Error(`Invalid formal CAD result: ${workspace}`);
  const source = await digestFile(await safeFile(workspace, "solution.py"));
  const step = await digestFile(await safeFile(workspace, "output.step"));
  if (result.last_executed_source_sha256 !== source || result.last_executed_step_sha256 !== step || result.artifact?.sha256 !== step) throw Error(`Formal CAD hashes are stale: ${workspace}`);
  if (!/^sessions\/[^/]+\.jsonl$/u.test(result.session_file ?? "")) throw Error(`Missing resumable Main session: ${workspace}`);
  await safeFile(workspace, result.session_file);
  return { job, result, source, step };
}

// This freezes AUTO-search output; it does not calibrate or manually correct a view.
export async function freezeAutomaticViews(sourceWorkspace, viewWorkspace, config) {
  const formal = await validateFormalWorkspace(sourceWorkspace, config);
  const resultPath = await safeFile(viewWorkspace, "view_search_only_result.json");
  const result = await readJson(resultPath);
  if (result.status !== "completed" || result.fixture_id !== formal.result.fixture_id || result.reflection_agent_started !== false || result.agent_roles?.includes("reflection") || !result.agent_roles?.includes("view_search") || resolve(result.source_workspace ?? "") !== resolve(sourceWorkspace) || result.source_sha256 !== formal.source || result.step_sha256 !== formal.step) throw Error("Automatic View Search does not match its generated CAD");
  if (await digestFile(await safeFile(viewWorkspace, "solution.py")) !== formal.source || await digestFile(await safeFile(viewWorkspace, "output.step")) !== formal.step) throw Error("View Search changed the source CAD");
  const originalPath = result.manifest_path;
  const originalFile = await safeFile(viewWorkspace, originalPath);
  const original = await readJson(originalFile);
  const { loadLockedMainViewManifest } = await import("../dist/src/main-view-reflect.js");
  await loadLockedMainViewManifest({ workspace: viewWorkspace, manifestPath: originalPath, manifestSha256: await digestFile(originalFile), baselineSourceSha256: formal.source, baselineStepSha256: formal.step });
  if (original.projection_frame) throw Error("Expected fresh automatic search, not an already calibrated manifest");
  for (const region of original.regions) {
    const transform = region.selected?.transform;
    if (!transform || !Number.isFinite(transform.scale_px_per_unit) || transform.scale_px_per_unit <= 0) throw Error(`Missing fixed 2D transform: ${region.region_id}`);
    if (region.crop_sha256 && await digestFile(await safeFile(viewWorkspace, region.crop_path)) !== region.crop_sha256) throw Error(`Changed reference crop: ${region.region_id}`);
  }
  const fixedPath = "drawing_feedback/automatic-fixed-frame/view_manifest.json";
  const anchorPath = "drawing_feedback/automatic-fixed-frame/anchor.step";
  await mkdir(dirname(join(viewWorkspace, fixedPath)), { recursive: true });
  await copyFile(join(viewWorkspace, "output.step"), join(viewWorkspace, anchorPath));
  const fixed = { ...original, projection_frame: { policy: "fixed", anchor_step_path: anchorPath, anchor_step_sha256: formal.step } };
  await writeJson(join(viewWorkspace, fixedPath), fixed);
  await loadLockedMainViewManifest({ workspace: viewWorkspace, manifestPath: fixedPath, manifestSha256: await digestFile(join(viewWorkspace, fixedPath)), baselineSourceSha256: formal.source, baselineStepSha256: formal.step });
  // Retain the original search result and manifest for audit.
  await copyFile(resultPath, join(viewWorkspace, "view_search_only_result.before-frame-lock.json"));
  await writeJson(resultPath, { ...result, manifest_path: fixedPath, automatic_search_manifest_path: originalPath, frame_lock_origin: "automatic_view_search", manually_aligned: false });
  const record = { fixture_id: formal.result.fixture_id, origin: "automatic_view_search", manually_aligned: false, original_manifest_path: originalPath, original_manifest_sha256: await digestFile(originalFile), fixed_manifest_path: fixedPath, fixed_manifest_sha256: await digestFile(join(viewWorkspace, fixedPath)), anchor_step_sha256: formal.step, selected_regions: fixed.regions.map(region => ({ region_id: region.region_id, status: region.status, candidate_id: region.selected.candidate_id, score: region.selected.score })), created_at: new Date().toISOString() };
  await writeJson(join(viewWorkspace, "automatic_frame_lock.json"), record);
  return record;
}

export async function runCommand(command, { logPath, timeoutMinutes, env = process.env } = {}) {
  if (logPath) await mkdir(dirname(logPath), { recursive: true });
  const log = logPath ? createWriteStream(logPath, { flags: "a" }) : undefined;
  return await new Promise((resolvePromise, reject) => {
    const child = spawn(command[0], command.slice(1), { cwd: repositoryRoot, env, stdio: ["ignore", "pipe", "pipe"] });
    let timedOut = false;
    let interrupted = false;
    let settled = false;
    let forceTimer;
    const stop = () => {
      if (interrupted) return;
      interrupted = true;
      child.kill("SIGINT");
      forceTimer = setTimeout(() => child.kill("SIGKILL"), 60_000);
      forceTimer.unref();
    };
    const timer = timeoutMinutes ? setTimeout(() => { timedOut = true; stop(); }, timeoutMinutes * 60_000) : undefined;
    process.once("SIGINT", stop);
    process.once("SIGTERM", stop);
    child.stdout.on("data", data => { process.stdout.write(data); log?.write(data); });
    child.stderr.on("data", data => { process.stderr.write(data); log?.write(data); });
    const finish = error => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      clearTimeout(forceTimer);
      process.removeListener("SIGINT", stop);
      process.removeListener("SIGTERM", stop);
      if (log) log.end();
      if (error) reject(error); else resolvePromise();
    };
    child.once("error", finish);
    child.once("close", (code, signal) => {
      if (timedOut || interrupted || code !== 0) {
        const error = Error(`${timedOut ? "Timed out" : interrupted ? "Interrupted" : "Command failed"}: ${command[0]} (exit=${code}, signal=${signal})`);
        error.interrupted = interrupted && !timedOut;
        finish(error);
      } else finish();
    });
  });
}

function parseArgs(args) {
  const parsed = { cases: [] };
  for (let i = 0; i < args.length; i++) {
    const token = args[i];
    if (["--help", "-h", "--dry-run", "--doctor", "--all-generation"].includes(token)) { parsed[token] = true; continue; }
    if (token === "--cases") {
      while (args[i + 1] && !args[i + 1].startsWith("--")) parsed.cases.push(...args[++i].split(",").filter(Boolean));
      if (!parsed.cases.length) throw Error("--cases requires IDs");
      continue;
    }
    if (!["--config", "--output-dir", "--data-inputs-dir", "--image"].includes(token) || !args[i + 1] || args[i + 1].startsWith("--")) throw Error(`Unknown/incomplete option: ${token}`);
    if (parsed[token]) throw Error(`Duplicate option: ${token}`);
    parsed[token] = args[++i];
  }
  if (parsed["--all-generation"] && parsed.cases.length) throw Error("Choose --cases or --all-generation");
  return parsed;
}

export async function main(args = process.argv.slice(2)) {
  const parsed = parseArgs(args);
  if (parsed["--help"] || parsed["-h"]) { console.log(help); return 0; }
  const config = await readJson(resolve(parsed["--config"] ?? join(here, "config.json")));
  config.data_inputs_dir = resolve(repositoryRoot, parsed["--data-inputs-dir"] ?? config.data_inputs_dir);
  if (parsed["--image"] || process.env.CAD_FULL_METHOD_IMAGE) config.image = parsed["--image"] ?? process.env.CAD_FULL_METHOD_IMAGE;
  if (parsed.cases.length) config.cases = parsed.cases;
  const { loadFixture, discoverGenerationFixtures } = await import("../dist/src/fixtures.js");
  if (parsed["--all-generation"]) config.cases = await discoverGenerationFixtures(config.data_inputs_dir);
  validateConfig(config);
  process.env.CADGENBENCH_SUBAGENT_TIMEOUT_MULTIPLIER = String(config.subagent_timeout_multiplier ?? 1);
  process.env.CADGENBENCH_DOCKER_NETWORK = config.docker_network ?? "bridge";
  for (const id of config.cases) await loadFixture(config.data_inputs_dir, id);
  const piAgentDir = resolve(process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi/agent"));
  process.env.PI_CODING_AGENT_DIR = piAgentDir;
  const outputDir = resolve(parsed["--output-dir"] ?? join(repositoryRoot, "results/pi", `${config.experiment_id}-${new Date().toISOString().replaceAll(/[-:.]/gu, "")}`));
  const { providerEnvironment } = await import("../dist/src/docker-worker.js");
  const forwarded = providerEnvironment(config.provider);
  // Compatibility aliases (e.g. KIMI_API_KEY) are resolved exactly as for Main.
  for (const name of forwarded.names) process.env[name] = forwarded.environment[name];
  const plans = buildPlan(config, outputDir, piAgentDir, forwarded.names);
  const doctor = [process.execPath, join(repositoryRoot, "integrations/pi/dist/src/cli.js"), "doctor", "--provider", config.provider, "--model", config.model, "--image", config.image];
  if (parsed["--dry-run"]) {
    console.log(JSON.stringify({ dry_run: true, config, output_dir: outputDir, pi_agent_dir: piAgentDir, doctor, cases: plans }, null, 2));
    return 0;
  }
  if (parsed["--doctor"]) { await runCommand(doctor); return 0; }
  if (await lstat(outputDir).catch(() => null)) throw Error(`Output already exists; use a new --output-dir: ${outputDir}`);
  await mkdir(outputDir, { recursive: true });
  const contract = { schema_version: 1, config, started_at: new Date().toISOString(), output_dir: outputDir, pi_agent_dir: piAgentDir, pipeline: ["main_generation", "automatic_segmentation_and_view_search", "lock_automatic_projection_frame", "dimension_annotation_before_main_and_after_edits", "same_main_session_repair"], manually_aligned_views: false, extra_shape_or_repair_subagents: false, cases: plans };
  await writeJson(join(outputDir, "experiment_contract.json"), contract);
  const summary = { ...contract, status: "running", cases: [] };
  await writeJson(join(outputDir, "full_method_summary.json"), summary);
  let summaryWrites = Promise.resolve();
  const persistSummary = () => {
    const snapshot = structuredClone(summary);
    summaryWrites = summaryWrites.then(() => writeJson(join(outputDir, "full_method_summary.json"), snapshot));
    return summaryWrites;
  };
  const uninterrupted = await forEachCaseBounded(plans, config.case_parallelism ?? 1, async plan => {
    const record = { fixture_id: plan.fixture_id, status: "running", stage: "generation", started_at: new Date().toISOString(), paths: plan.paths };
    summary.cases.push(record);
    await persistSummary();
    try {
      await runCommand(plan.generate, { logPath: join(plan.paths.root, "generation.log"), timeoutMinutes: config.generation_timeout_minutes });
      await validateFormalWorkspace(plan.paths.source, config);
      record.stage = "view_search";
      await persistSummary();
      await runCommand(plan.search, { logPath: join(plan.paths.root, "view_search.log"), timeoutMinutes: config.view_search_timeout_minutes });
      record.stage = "lock_automatic_frame";
      await persistSummary();
      record.frame_lock = await freezeAutomaticViews(plan.paths.source, plan.paths.view, config);
      record.stage = "dimensions_and_main_repair";
      await persistSummary();
      await runCommand(plan.repair, { logPath: join(plan.paths.root, "repair.log"), timeoutMinutes: config.repair_timeout_minutes + 5 });
      const final = await validateFormalWorkspace(plan.paths.final, config);
      const continuation = await readJson(join(plan.paths.repair, "continuation_summary.json"));
      const dimensions = final.result.dimension_annotations;
      const lockedHash = final.job.continuation?.locked_view_manifest_sha256;
      if (continuation.status !== "completed" || dimensions?.status !== "completed" || dimensions.source_sha256 !== final.source || dimensions.step_sha256 !== final.step || dimensions.locked_manifest_sha256 !== lockedHash) throw Error("Final dimensions/inspection do not match final CAD");
      record.status = "completed";
      record.stage = "finished";
      record.final_source_sha256 = final.source;
      record.final_step_sha256 = final.step;
      record.dimensions = dimensions;
      record.final_agent_result_path = join(plan.paths.final, "agent_result.json");
      record.final_result_path = join(plan.paths.final, "result.json");
    } catch (error) {
      record.status = error.interrupted ? "interrupted" : "error";
      record.error = error.message;
      console.error(`[${plan.fixture_id}] ${record.stage}: ${error.message}`);
      if (error.interrupted) {
        record.finished_at = new Date().toISOString();
        summary.status = "interrupted";
        summary.finished_at = record.finished_at;
        await persistSummary();
        return false;
      }
    }
    record.finished_at = new Date().toISOString();
    await persistSummary();
    return true;
  });
  await summaryWrites;
  const completed = summary.cases.filter(item => item.status === "completed").length;
  summary.status = !uninterrupted ? "interrupted" : completed === plans.length ? "completed" : completed ? "partial" : "error";
  summary.finished_at = new Date().toISOString();
  await writeJson(join(outputDir, "full_method_summary.json"), summary);
  console.log(JSON.stringify({ status: summary.status, completed, total: plans.length, output_dir: outputDir }, null, 2));
  return summary.status === "completed" ? 0 : summary.status === "interrupted" ? 130 : 2;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().then(code => { process.exitCode = code; }).catch(error => { console.error(error.message); process.exitCode = 2; });
}
