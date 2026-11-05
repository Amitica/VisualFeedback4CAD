import assert from "node:assert/strict";
import { mkdtemp, lstat, rm, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { main, nativeAuthEnvironment, nativeCommand, nativeNetwork, parse } from "./run-all.mjs";

test("suite rejects ambiguous selection and unsafe method omissions", async () => {
  assert.throws(() => parse(["--cases", "131", "--all-generation"]), /Choose/u);
  assert.throws(() => parse(["--setup", "--skip-setup"]), /Choose/u);
  await assert.rejects(() => main(["--methods", "vanilla", "--cases", "131", "--output-dir", "/tmp/not-created", "--dry-run"]), /initial-root/u);
});

test("dry-run creates no output and shares one Initial across the three repair protocols", async t => {
  const folder = await mkdtemp(join(tmpdir(), "baseline-suite-"));
  t.after(() => rm(folder, { recursive: true, force: true }));
  const output = join(folder, "unused");
  const old = console.log, logs = [];
  console.log = data => logs.push(data);
  try { assert.equal(await main(["--cases", "131", "--parallel", "1", "--output-dir", output, "--dry-run"]), 0); }
  finally { console.log = old; }
  assert.equal(await lstat(output).catch(() => null), null);
  const plan = JSON.parse(logs[0]);
  assert.deepEqual(plan.methods, ["initial", "vanilla", "cadcodeverify", "seek-cad", "itercad"]);
  assert.equal(plan.model_calls_under_dry_run, 0);
  for (const method of ["vanilla", "cadcodeverify", "seek-cad"]) {
    const command = plan.commands[method];
    assert.equal(command[command.indexOf("--initial-root") + 1], join(output, "initial/generation"));
  }
  assert.equal(plan.commands.initial.includes("--initial-root"), false);
  assert.equal(plan.commands.itercad[plan.commands.itercad.indexOf("--max-turns") + 1], "5");
  assert.equal(plan.commands.itercad[plan.commands.itercad.indexOf("--pass-k") + 1], "2");
});

test("subset continues the supplied Main Initial and excludes independent native model calls", async () => {
  const old = console.log, logs = [];
  console.log = data => logs.push(data);
  try { assert.equal(await main(["--methods", "vanilla,cadcodeverify,seek-cad", "--initial-root", "/tmp/shared-main", "--cases", "131", "--output-dir", "/tmp/subset-dry-run", "--dry-run"]), 0); }
  finally { console.log = old; }
  const plan = JSON.parse(logs[0]);
  assert.deepEqual(Object.keys(plan.commands).sort(), ["cadcodeverify", "seek-cad", "vanilla"]);
  assert.equal(plan.commands.vanilla[plan.commands.vanilla.indexOf("--initial-root") + 1], "/tmp/shared-main");
});

test("native Docker arguments never contain secret environment values", () => {
  const previous = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = "synthetic-secret-must-never-appear";
  try {
    const command = nativeCommand({ docker_network: "bridge", parallel: 1, thinking_level: "low" }, "/tmp/suite", "pinned-image");
    assert.equal(command.includes("OPENAI_API_KEY"), true);
    assert.equal(command.join(" ").includes(process.env.OPENAI_API_KEY), false);
  } finally {
    if (previous === undefined) delete process.env.OPENAI_API_KEY; else process.env.OPENAI_API_KEY = previous;
  }
});

test("Pi's duplicate Bearer header is reproduced exactly; custom authorization is never dropped", () => {
  const model = { id: "test-model", baseUrl: "https://gateway.invalid/v1" };
  const resolution = { auth: { apiKey: "synthetic-key", headers: { Authorization: "Bearer synthetic-key" } } };
  const env = nativeAuthEnvironment({}, model, resolution);
  assert.equal(env.OPENAI_API_KEY, "synthetic-key");
  assert.equal(env.GENERATOR_API, model.baseUrl);
  assert.equal(env.GEN_MODEL, model.id);
  assert.throws(() => nativeAuthEnvironment({}, model, { auth: { ...resolution.auth, headers: { Authorization: "Bearer a-different-key" } } }), /custom authorization/u);
  assert.throws(() => nativeAuthEnvironment({}, { ...model, headers: { "X-Required": "synthetic" } }, resolution), /custom authorization/u);
});

test("a suite without IterCAD accepts the Main off reasoning setting", async t => {
  const folder = await mkdtemp(join(tmpdir(), "baseline-no-native-"));
  t.after(() => rm(folder, { recursive: true, force: true }));
  const config = JSON.parse(await readFile(new URL("./config.json", import.meta.url), "utf8"));
  config.thinking_level = "off";
  await writeFile(join(folder, "config.json"), JSON.stringify(config));
  const old = console.log;
  console.log = () => {};
  try { assert.equal(await main(["--config", join(folder, "config.json"), "--methods", "initial", "--cases", "140", "--output-dir", join(folder, "unused"), "--dry-run"]), 0); }
  finally { console.log = old; }
});

test("native networking makes host loopback endpoints and proxies reachable without dropping TLS or proxy settings", () => {
  const c = { docker_network: "private-bridge" };
  assert.deepEqual(nativeNetwork(c, { HTTPS_PROXY: "http://127.0.0.1:7890" }), { network: "host", reason: "host_loopback_proxy" });
  assert.deepEqual(nativeNetwork(c, { GENERATOR_API: "http://localhost:8000/v1" }), { network: "host", reason: "host_loopback_api" });
  assert.equal(nativeNetwork(c, { GENERATOR_API: "https://gateway.example/v1", HTTPS_PROXY: "http://proxy.example:8080" }).network, "private-bridge");
  assert.equal(nativeNetwork({ ...c, itercad_network: "explicit-bridge" }, { HTTPS_PROXY: "http://127.0.0.1:7890" }).network, "explicit-bridge");
  assert.equal(nativeNetwork({ docker_network: "none" }, { HTTPS_PROXY: "http://127.0.0.1:7890" }).network, "none");
  const cmd = nativeCommand({ ...c, parallel: 1, thinking_level: "low" }, "/tmp/out", "native", { HTTPS_PROXY: "http://127.0.0.1:7890" });
  assert.equal(cmd[cmd.indexOf("--network") + 1], "host");
  assert.ok(cmd.includes("HTTPS_PROXY"));
  assert.equal(cmd.join(" ").includes("127.0.0.1"), false);
});
