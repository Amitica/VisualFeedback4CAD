import { describe, expect, it } from "vitest";

import {
	DEFAULT_EXPERIMENT_MODEL_PROFILE,
	resolveExperimentModelSelection,
} from "../src/experiment-model-profile.js";

describe("experiment model profiles", () => {
	it("uses GPT-5.6 Sol through UCloud by default", () => {
		expect(DEFAULT_EXPERIMENT_MODEL_PROFILE).toBe("gpt-5.6-sol");
		expect(resolveExperimentModelSelection({})).toEqual({
			profile: "gpt-5.6-sol",
			provider: "ucloud",
			model: "gpt-5.6-sol",
		});
	});

	it("switches back to Kimi with one profile value", () => {
		expect(resolveExperimentModelSelection({ profile: "kimi" })).toEqual({
			profile: "kimi",
			provider: "moonshotai-cn",
			model: "kimi-k3",
		});
	});

	it("accepts an explicit provider and model pair", () => {
		expect(resolveExperimentModelSelection({ provider: "test-provider", model: "test-model" })).toEqual({
			profile: "custom",
			provider: "test-provider",
			model: "test-model",
		});
	});

	it("rejects incomplete or conflicting selections", () => {
		expect(() => resolveExperimentModelSelection({ provider: "ucloud" })).toThrow(
			"--provider and --model must be provided together",
		);
		expect(() => resolveExperimentModelSelection({ model: "gpt-5.6-sol" })).toThrow(
			"--provider and --model must be provided together",
		);
		expect(() => resolveExperimentModelSelection({
			profile: "kimi",
			provider: "ucloud",
			model: "gpt-5.6-sol",
		})).toThrow("--model-profile cannot be combined with --provider/--model");
		expect(() => resolveExperimentModelSelection({ profile: "unknown" })).toThrow("Unknown model profile unknown");
	});
});
