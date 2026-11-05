#!/usr/bin/env node
// One fresh Main Initial is shared by Vanilla/CCV/Seek. IterCAD retains its
// official CadQuery generation loop and therefore has a separate Initial.
import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { lstat, mkdir, readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { availableFixtures, validate } from "./run.mjs";
import { repoRoot, readJson, writeJson } from "./common.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const help = `Run every engineering-drawing CAD baseline with one command.

bash integrations/pi/baselines/run-all.sh --cases 131 --output-dir <new-dir>
bash integrations/pi/baselines/run-all.sh --all-generation --parallel all --output-dir <new-dir>

Options: --config <json>, --cases <ids...> OR --all-generation,
 --output-dir <new-dir>, --data-inputs-dir <dir>, --parallel <N|all>,
 --methods <initial,vanilla,cadcodeverify,seek-cad,itercad>, --initial-root <root>,
 --network <docker-network>, --itercad-network <network>, --image <common-image>, --itercad-image <image>,
 --setup (build all images first), --skip-setup (default), --dry-run.

Uses Pi credentials for all methods, including the IterCAD Responses bridge.
IterCAD may instead use GENERATOR_API + OPENAI_API_KEY; GEN_MODEL defaults to
the common model. It keeps the official 5 turns / 2 attempts, not a shared
build123d Initial. No model calls occur under --dry-run.
`;

export function parse(argv) {
  const args = { cases: [] };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (["--help", "-h", "--all-generation", "--setup", "--skip-setup", "--dry-run"].includes(arg)) { args[arg] = true; continue; }
    if (arg === "--cases") {
      while (argv[i + 1] && !argv[i + 1].startsWith("--")) args.cases.push(...argv[++i].split(",").filter(Boolean));
      if (!args.cases.length) throw Error("--cases requires at least one ID");
      continue;
    }
    if (!["--config", "--output-dir", "--data-inputs-dir", "--parallel", "--network", "--itercad-network", "--image", "--itercad-image", "--methods", "--initial-root"].includes(arg) || !argv[i + 1] || argv[i + 1].startsWith("--")) throw Error(`Unknown/incomplete option ${arg}`);
    if (args[arg]) throw Error(`Duplicate option ${arg}`);
    args[arg] = argv[++i];
  }
  if (args.cases.length && args["--all-generation"]) throw Error("Choose --cases or --all-generation");
  if (args["--setup"] && args["--skip-setup"]) throw Error("Choose --setup or --skip-setup");
  return args;
}

export function child(argv, { env = process.env, log, capture = false } = {}) {
  return new Promise((yes, no) => {
    const stream = log ? createWriteStream(log, { flags: "a" }) : undefined;
    const proc = spawn(argv[0], argv.slice(1), { cwd: repoRoot, env, stdio: ["ignore", "pipe", "pipe"] });
    let settled = false;
    let stdout = "";
    const interrupt = () => proc.kill("SIGINT");
    process.once("SIGINT", interrupt);
    const finish = error => {
      if (settled) return;
      settled = true;
      process.removeListener("SIGINT", interrupt);
      stream?.end();
      if (error) no(error); else yes(stdout);
    };
    proc.stdout.on("data", data => { if (capture) stdout += data; else process.stdout.write(data); stream?.write(data); });
    proc.stderr.on("data", data => { process.stderr.write(data); stream?.write(data); });
    proc.once("error", finish);
    proc.once("close", code => finish(code === 0 ? undefined : Error(`Command exited ${code}: ${argv[0]}`)));
  });
}

export function nativeNetwork(config, environment = process.env) {
  if (config.itercad_network) return { network: config.itercad_network, reason: "explicit_itercad_network" };
  if (["host", "none"].includes(config.docker_network)) return { network: config.docker_network, reason: "common_network" };
  const hostLoopback = value => {
    try {
      const host = new URL(value).hostname;
      return host === "localhost" || host === "::1" || host === "[::1]" || /^127\./u.test(host);
    } catch { return false; }
  };
  const localEndpoint = hostLoopback(environment.GENERATOR_API);
  const localProxy = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"].some(key => hostLoopback(environment[key]));
  return localEndpoint || localProxy ? { network: "host", reason: localEndpoint ? "host_loopback_api" : "host_loopback_proxy" } : { network: config.docker_network, reason: "common_network" };
}

export function nativeCommand(config, output, image, environment = process.env) {
  return ["docker", "run", "--rm", "--read-only", "--network", nativeNetwork(config, environment).network,
    "--tmpfs", "/tmp:rw,size=2g", "--user", `${process.getuid()}:${process.getgid()}`,
    "--cpus", "4", "--memory", "12g", "--pids-limit", "512",
    "-e", "GENERATOR_API", "-e", "GEN_MODEL", "-e", "OPENAI_API_KEY",
    ...["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY"].filter(name => environment[name]).flatMap(name => ["-e", name]),
    "--mount", `type=bind,src=${output},dst=/out`, image, "run",
    "--data-jsonl", "/out/itercad-inputs/samples.jsonl", "--output-dir", "/out/itercad",
    "--workers", String(Math.min(config.parallel, 4)), "--max-turns", "5", "--pass-k", "2",
    "--api-protocol", "responses", "--reasoning-effort", config.thinking_level,
    "--request-timeout-seconds", "600"];
}

export function nativeAuthEnvironment(config, model, resolution) {
  if (!resolution?.auth.apiKey) throw Error("IterCAD automatic credential bridge requires an API key; set GENERATOR_API and OPENAI_API_KEY explicitly for another endpoint");
  const headers = { ...(model.headers ?? {}), ...(resolution.auth.headers ?? {}) };
  // Pi may supply both apiKey and the corresponding standard Bearer header.
  // The OpenAI client regenerates precisely this header from api_key.
  const custom = Object.entries(headers).filter(([key, value]) => !(key.toLowerCase() === "authorization" && value === `Bearer ${resolution.auth.apiKey}`));
  if (custom.length) throw Error("IterCAD does not silently discard custom authorization headers; use a compatible GENERATOR_API endpoint");
  return { ...process.env, GENERATOR_API: resolution.auth.baseUrl ?? model.baseUrl,
    GEN_MODEL: model.id, OPENAI_API_KEY: resolution.auth.apiKey };
}

export async function iterEnvironment(config) {
  if (process.env.GENERATOR_API) {
    if (!process.env.OPENAI_API_KEY) throw Error("GENERATOR_API requires OPENAI_API_KEY in the environment");
    if (process.env.GEN_MODEL && process.env.GEN_MODEL !== config.model) throw Error("GEN_MODEL differs from the common suite model; use the standalone native runner for a different checkpoint");
    return { ...process.env, GEN_MODEL: process.env.GEN_MODEL ?? config.model };
  }
  const { resolveVisionModel } = await import("../dist/src/runner.js");
  const { runtime, model } = await resolveVisionModel(config.provider, config.model);
  const resolution = await runtime.getAuth(model);
  return nativeAuthEnvironment(config, model, resolution);
}

export async function main(argv = process.argv.slice(2)) {
  const args = parse(argv);
  if (args["--help"] || args["-h"]) { console.log(help); return 0; }
  if (!args["--output-dir"]) throw Error("--output-dir is required");
  const config = await readJson(resolve(args["--config"] ?? join(here, "config.json")));
  config.data_inputs_dir = resolve(repoRoot, args["--data-inputs-dir"] ?? config.data_inputs_dir);
  config.docker_network = args["--network"] ?? process.env.CADGENBENCH_DOCKER_NETWORK ?? config.docker_network;
  config.itercad_network = args["--itercad-network"] ?? config.itercad_network;
  if (config.itercad_network && !/^[A-Za-z0-9][A-Za-z0-9_.-]*$/u.test(config.itercad_network)) throw Error("Invalid IterCAD Docker network");
  config.image = args["--image"] ?? process.env.CAD_BASELINE_IMAGE ?? config.image;
  const output = resolve(args["--output-dir"]);
  if (!args["--dry-run"] && await lstat(output).catch(() => null)) throw Error(`Output exists; choose a new directory: ${output}`);
  if (args["--setup"] && !args["--dry-run"]) {
    const setupEnv = { ...process.env, CAD_BASELINE_IMAGE: config.image,
      ITERCAD_IMAGE: args["--itercad-image"] ?? process.env.ITERCAD_IMAGE ?? "cadgenbench-itercad-native:55998c4" };
    await child(["bash", join(here, "setup.sh")], { env: setupEnv });
    await child(["bash", join(here, "itercad/setup.sh")], { env: setupEnv });
  }
  if (args.cases.length) config.cases = args.cases;
  else if (args["--all-generation"]) config.cases = (await availableFixtures(config.data_inputs_dir, false)).ids;
  config.parallel = args["--parallel"] === "all" ? config.cases.length : Number(args["--parallel"] ?? config.parallel);
  validate(config);
  const { loadFixture } = await import("../dist/src/fixtures.js");
  for (const id of config.cases) await loadFixture(config.data_inputs_dir, id);
  const methods = args["--methods"] ? args["--methods"].split(",") : ["initial", "vanilla", "cadcodeverify", "seek-cad", "itercad"];
  if (!methods.length || new Set(methods).size !== methods.length || methods.some(method => !["initial", "vanilla", "cadcodeverify", "seek-cad", "itercad"].includes(method))) throw Error("Invalid or duplicate --methods");
  if (methods.includes("itercad") && !["low", "medium", "high"].includes(config.thinking_level)) throw Error("The IterCAD Responses bridge supports low/medium/high; choose a supported common thinking level");
  const initialRoot = args["--initial-root"] ? resolve(args["--initial-root"]) : join(output, "initial/generation");
  if (methods.some(method => ["vanilla", "cadcodeverify", "seek-cad"].includes(method)) && !methods.includes("initial") && !args["--initial-root"]) throw Error("Reflection methods need --initial-root when --methods omits initial");
  const image = args["--itercad-image"] ?? process.env.ITERCAD_IMAGE ?? "cadgenbench-itercad-native:55998c4";
  const configPath = join(output, "resolved-config.json");
  const common = ["--config", configPath, "--cases", ...config.cases, "--parallel", String(config.parallel)];
  const run = method => [process.execPath, join(here, "run.mjs"), "--method", method, ...common,
    ...(method !== "initial" || args["--initial-root"] ? ["--initial-root", initialRoot] : []),
    "--output-dir", join(output, method)];
  const prepare = ["docker", "run", "--rm", "--read-only", "--network", "none", "--tmpfs", "/tmp:rw,size=1g",
    "--user", `${process.getuid()}:${process.getgid()}`, "--mount", `type=bind,src=${config.data_inputs_dir},dst=/inputs,readonly`,
    "--mount", `type=bind,src=${output},dst=/out`, image, "prepare", "--data-inputs-dir", "/inputs",
    "--output-dir", "/out/itercad-inputs", "--cases", ...config.cases];
  const commands = Object.fromEntries(methods.filter(method => method !== "itercad").map(method => [method, run(method)]));
  if (methods.includes("itercad")) { commands.itercad_prepare = prepare; commands.itercad = nativeCommand(config, output, image); }
  const plan = { schema_version: 1, config, commands, methods, initial_root: initialRoot,
    initial_policy: `${args["--initial-root"] ? "Existing verified" : "Fresh"} Main Initial shared by Vanilla/CADCodeVerify/Seek-CAD; IterCAD uses its official independent CadQuery generation loop`,
    model_calls_under_dry_run: 0, geometry_accuracy_scored: false };
  if (args["--dry-run"]) { console.log(JSON.stringify({ dry_run: true, ...plan }, null, 2)); return 0; }
  const imageIds = {};
  for (const tag of [...new Set([...(methods.some(method => method !== "itercad") ? [config.image] : []), ...(methods.includes("itercad") ? [image] : [])])]) imageIds[tag] = (await child(["docker", "image", "inspect", tag, "--format", "{{.Id}}"], { capture: true })).trim();
  const env = methods.includes("itercad") ? await iterEnvironment(config) : process.env; // Resolve before any paid model work.
  if (methods.includes("itercad")) {
    commands.itercad = nativeCommand(config, output, image, env);
    plan.itercad_transport = nativeNetwork(config, env);
  }
  await mkdir(dirname(output), { recursive: true });
  await mkdir(output, { recursive: false });
  await writeJson(configPath, config);
  await writeJson(join(output, "run-plan.json"), { ...plan, image_ids: imageIds });
  const statuses = {};
  const launch = async (method, command = commands[method], environment = process.env) => {
    statuses[method] = { status: "running", started_at: new Date().toISOString() };
    await writeJson(join(output, "suite-status.json"), statuses);
    try {
      await child(command, { env: environment, log: join(output, `${method}.log`) });
      const summary = await readJson(join(output, method, "summary.json"));
      if (summary.status !== "completed") throw Error(`${method} finished with summary status ${summary.status}`);
      statuses[method].status = "completed";
    } catch (error) {
      statuses[method].status = "error"; statuses[method].error = error.message;
    }
    statuses[method].finished_at = new Date().toISOString();
    await writeJson(join(output, "suite-status.json"), statuses);
  };
  // Native preparation has no model calls and fails before starting Initial.
  if (methods.includes("itercad")) await child(prepare, { log: join(output, "itercad-prepare.log") });
  const native = methods.includes("itercad") ? launch("itercad", commands.itercad, env) : Promise.resolve();
  if (methods.includes("initial")) await launch("initial");
  const reflectMethods = methods.filter(method => ["vanilla", "cadcodeverify", "seek-cad"].includes(method));
  // Each method checks Initial per case and keeps missing/error cases in its
  // output denominator. A single failed Initial must not block good cases.
  await Promise.all(reflectMethods.map(method => launch(method)));
  await native;
  const summary = { status: Object.values(statuses).every(record => record.status === "completed") ? "completed" : "partial",
    fixture_ids: config.cases, methods: statuses, geometry_accuracy_scored: false, initial_policy: plan.initial_policy };
  await writeJson(join(output, "suite-status.json"), statuses);
  await writeJson(join(output, "summary.json"), summary);
  console.log(JSON.stringify(summary, null, 2));
  return summary.status === "completed" ? 0 : 2;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().then(code => { process.exitCode = code; }).catch(error => { console.error(error.message); process.exitCode = 2; });
