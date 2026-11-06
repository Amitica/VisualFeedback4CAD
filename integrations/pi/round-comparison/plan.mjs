import { isAbsolute, join, relative, sep } from "node:path";
import {
  buildPlan as buildFullMethodPlan,
  validateConfig as validateFullMethodConfig,
} from "../portable-full-method/run.mjs";

const METHODS = ["initial", "vanilla", "ours"];
const DIMENSION_PROFILE = "aligned-dimension-subagent-v1";

function validateRounds(value) {
  if (!Number.isInteger(value) || value < 1 || value > 20) {
    throw Error("rounds must be an integer in [1, 20]");
  }
  return value;
}

/** Normalize defaults without changing the caller's shared configuration. */
export function validateConfig(config) {
  validateFullMethodConfig(config);
  if (Object.hasOwn(config, "vanilla_rounds")) {
    throw Error("Vanilla runs exactly one reflection round; vanilla_rounds is not supported");
  }
  const rounds = validateRounds(config.rounds === undefined ? 3 : config.rounds);
  const methods = config.methods === undefined ? [...METHODS] : config.methods;
  if (!Array.isArray(methods) || !methods.length || new Set(methods).size !== methods.length || methods.some(method => !METHODS.includes(method))) {
    throw Error("methods must contain unique initial, vanilla, or ours entries");
  }
  return { ...config, rounds, methods: [...methods] };
}

export function roundName(number) {
  return `round-${String(validateRounds(number)).padStart(2, "0")}`;
}

function within(root, candidate) {
  const path = relative(root, candidate);
  return path === "" || (path !== ".." && !path.startsWith(`..${sep}`) && !isAbsolute(path));
}

/**
 * Initial and the fixed views are shared evidence, not additional rounds.
 * Vanilla has one historical Direct Reflect turn. Each Ours round starts from
 * the preceding completed CAD/session and refreshes dimensions in fixed views.
 */
export function buildPlan(config, outputDir, piAgentDir, environmentNames = [], initialSources = {}) {
  const checked = validateConfig(config);
  return buildFullMethodPlan(checked, outputDir, piAgentDir, environmentNames).map(base => {
    const reused = initialSources[base.fixture_id] !== undefined;
    const initialSource = reused ? initialSources[base.fixture_id] : base.paths.source;
    if (typeof initialSource !== "string" || !initialSource || !isAbsolute(initialSource)) {
      throw Error(`Initial workspace must be an absolute path: ${base.fixture_id}`);
    }
    if (/[\n\r,]/u.test(initialSource)) throw Error("Docker mount paths cannot contain commas or newlines");

    const search = [...base.search];
    const searchCli = search.indexOf("/opt/cadgenbench-pi/dist/src/drawing-search-cli.js");
    if (searchCli < 0) throw Error("Automatic View Search command is missing its CLI");
    search[searchCli + 1] = initialSource;
    if (reused && !within(outputDir, initialSource)) {
      // The standard command mounts only outputDir. An existing Initial may
      // live elsewhere and is read without permitting the search to alter it.
      search.splice(search.indexOf("--workdir"), 0, "--mount", `type=bind,src=${initialSource},dst=${initialSource},readonly`);
    }

    const continuationCli = base.repair[1];
    const makeRounds = method => {
      if (!checked.methods.includes(method)) return [];
      const count = method === "vanilla" ? 1 : checked.rounds;
      const rounds = [];
      let source = initialSource;
      for (let round = 1; round <= count; round++) {
        const name = roundName(round);
        const outputRoot = join(base.paths.root, method, name);
        const workspace = join(outputRoot, "run", base.fixture_id);
        const command = [
          process.execPath, continuationCli, source,
          "--output-dir", outputRoot,
          "--image", checked.image,
          "--provider", checked.provider,
          "--model", checked.model,
          "--thinking-level", checked.thinking_level,
          "--timeout-minutes", String(checked.repair_timeout_minutes),
        ];
        if (method === "ours") {
          command.push("--dimension-annotation-profile", DIMENSION_PROFILE);
          if (round === 1) command.push("--inspect-view-search-root", base.paths.search);
          else command.push("--allow-chained-self-reflect", "--reuse-fixed-view-from-source");
        }
        rounds.push({
          method, round, source, output_root: outputRoot, workspace,
          evaluation_dir: join(outputDir, "evaluation", method, name, base.fixture_id),
          command,
        });
        source = workspace;
      }
      return rounds;
    };

    return {
      fixture_id: base.fixture_id,
      paths: base.paths,
      initial_source: initialSource,
      initial_evaluation_dir: join(outputDir, "evaluation", "initial", base.fixture_id),
      generate: reused ? null : base.generate,
      search,
      search_container: base.search_container,
      search_root: base.paths.search,
      view_workspace: base.paths.view,
      vanilla: makeRounds("vanilla"),
      ours: makeRounds("ours"),
    };
  });
}
