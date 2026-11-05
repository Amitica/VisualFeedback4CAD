import { describe, expect, it } from "vitest";

import { restrictedSessionTimeouts } from "../src/drawing-reflection.js";

describe("restricted session timeout configuration", () => {
	it("keeps historical deadlines when no multiplier is supplied", () => {
		expect(restrictedSessionTimeouts(600_000, 7_200_000, undefined)).toEqual({
			activeTimeoutMs: 600_000, wallTimeoutMs: 7_200_000, multiplier: 1,
		});
	});

	it("extends both active and wall deadlines for a rerun", () => {
		expect(restrictedSessionTimeouts(600_000, 7_200_000, "3")).toEqual({
			activeTimeoutMs: 1_800_000, wallTimeoutMs: 21_600_000, multiplier: 3,
		});
	});

	it.each(["", "0", "-1", "NaN", "Infinity", "13", "long"])("rejects invalid multiplier %j", multiplier => {
		expect(() => restrictedSessionTimeouts(600_000, 7_200_000, multiplier)).toThrow("finite number in [1, 12]");
	});

	it("rejects deadlines that would overflow Node timers instead of firing immediately", () => {
		expect(() => restrictedSessionTimeouts(600_000, 2_147_483_647, "3")).toThrow("timer limits");
	});
});
