import { UsageError } from "./errors.js";

export const DEFAULT_EXPERIMENT_MODEL_PROFILE = "gpt-5.6-sol";

export interface ExperimentModelSelection {
	profile: string;
	provider: string;
	model: string;
}

const MODEL_PROFILES: Readonly<Record<string, Omit<ExperimentModelSelection, "profile">>> = {
	"gpt-5.6-sol": {
		provider: "ucloud",
		model: "gpt-5.6-sol",
	},
	kimi: {
		provider: "moonshotai-cn",
		model: "kimi-k3",
	},
};

export const EXPERIMENT_MODEL_PROFILE_NAMES = Object.freeze(Object.keys(MODEL_PROFILES));

export function resolveExperimentModelSelection(options: {
	profile?: string | undefined;
	provider?: string | undefined;
	model?: string | undefined;
}): ExperimentModelSelection {
	const profile = options.profile?.trim();
	const provider = options.provider?.trim();
	const model = options.model?.trim();
	if ((provider && !model) || (!provider && model)) {
		throw new UsageError("--provider and --model must be provided together");
	}
	if (profile && (provider || model)) {
		throw new UsageError("--model-profile cannot be combined with --provider/--model");
	}
	if (provider && model) return { profile: "custom", provider, model };
	const selectedProfile = profile || DEFAULT_EXPERIMENT_MODEL_PROFILE;
	const selected = MODEL_PROFILES[selectedProfile];
	if (!selected) {
		throw new UsageError(
			`Unknown model profile ${selectedProfile}; expected one of ${EXPERIMENT_MODEL_PROFILE_NAMES.join(", ")}`,
		);
	}
	return { profile: selectedProfile, ...selected };
}
