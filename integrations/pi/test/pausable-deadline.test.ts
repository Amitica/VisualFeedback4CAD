import { afterEach, describe, expect, it, vi } from "vitest";

import { PausableDeadline } from "../src/pausable-deadline.js";

afterEach(() => {
	vi.useRealTimers();
});

describe("PausableDeadline", () => {
	it("does not consume remaining time while paused", async () => {
		vi.useFakeTimers();
		const onExpire = vi.fn();
		const deadline = new PausableDeadline(1_000, onExpire);

		deadline.start();
		await vi.advanceTimersByTimeAsync(400);
		expect(deadline.remainingMs).toBeCloseTo(600, 5);

		deadline.pause();
		await vi.advanceTimersByTimeAsync(5_000);
		expect(deadline.remainingMs).toBeCloseTo(600, 5);
		expect(onExpire).not.toHaveBeenCalled();

		deadline.resume();
		await vi.advanceTimersByTimeAsync(599);
		expect(deadline.remainingMs).toBeCloseTo(1, 5);
		expect(onExpire).not.toHaveBeenCalled();

		await vi.advanceTimersByTimeAsync(1);
		expect(deadline.remainingMs).toBe(0);
		expect(onExpire).toHaveBeenCalledOnce();
	});

	it("makes start, pause, and resume idempotent", async () => {
		vi.useFakeTimers();
		const onExpire = vi.fn();
		const deadline = new PausableDeadline(100, onExpire);

		deadline.pause();
		deadline.resume();
		expect(vi.getTimerCount()).toBe(0);

		deadline.start();
		deadline.start();
		expect(vi.getTimerCount()).toBe(1);

		await vi.advanceTimersByTimeAsync(40);
		deadline.pause();
		deadline.pause();
		expect(vi.getTimerCount()).toBe(0);

		deadline.resume();
		deadline.resume();
		expect(vi.getTimerCount()).toBe(1);

		await vi.advanceTimersByTimeAsync(60);
		expect(onExpire).toHaveBeenCalledOnce();

		deadline.start();
		deadline.pause();
		deadline.resume();
		await vi.advanceTimersByTimeAsync(1_000);
		expect(onExpire).toHaveBeenCalledOnce();
	});

	it("dispose freezes the remaining time and clears the pending callback", async () => {
		vi.useFakeTimers();
		const onExpire = vi.fn();
		const deadline = new PausableDeadline(1_000, onExpire);

		deadline.start();
		await vi.advanceTimersByTimeAsync(250);
		deadline.dispose();
		deadline.dispose();

		expect(deadline.remainingMs).toBeCloseTo(750, 5);
		expect(vi.getTimerCount()).toBe(0);

		deadline.start();
		deadline.resume();
		await vi.advanceTimersByTimeAsync(10_000);
		expect(onExpire).not.toHaveBeenCalled();
		expect(vi.getTimerCount()).toBe(0);
	});

	it("rejects invalid timeout values", () => {
		expect(() => new PausableDeadline(-1, () => undefined)).toThrow(RangeError);
		expect(() => new PausableDeadline(Number.NaN, () => undefined)).toThrow(RangeError);
		expect(() => new PausableDeadline(Number.POSITIVE_INFINITY, () => undefined)).toThrow(RangeError);
	});
});
