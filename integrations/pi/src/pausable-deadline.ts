const MAX_TIMER_DELAY_MS = 2_147_483_647;

/**
 * A one-shot deadline whose remaining time is consumed only while it is running.
 *
 * `pause()` freezes the remaining budget, `resume()` continues it, and
 * `dispose()` permanently cancels the callback. Lifecycle methods are
 * idempotent so event-driven callers can safely handle duplicate notifications.
 */
export class PausableDeadline {
	private remaining: number;
	private timer: ReturnType<typeof setTimeout> | undefined;
	private runningSince: number | undefined;
	private started = false;
	private expired = false;
	private disposed = false;

	constructor(
		timeoutMs: number,
		private readonly onExpire: () => void,
	) {
		if (!Number.isFinite(timeoutMs) || timeoutMs < 0) {
			throw new RangeError("PausableDeadline timeout must be a finite non-negative number");
		}
		this.remaining = timeoutMs;
	}

	get remainingMs(): number {
		if (this.runningSince === undefined) return this.remaining;
		const elapsed = Math.max(0, globalThis.performance.now() - this.runningSince);
		return Math.max(0, this.remaining - elapsed);
	}

	start(): void {
		if (this.started || this.expired || this.disposed) return;
		this.started = true;
		this.runningSince = globalThis.performance.now();
		this.schedule();
	}

	pause(): void {
		if (!this.started || this.runningSince === undefined || this.expired || this.disposed) return;
		this.consumeElapsed();
		this.runningSince = undefined;
		this.clearTimer();
		if (this.remaining === 0) this.expire();
	}

	resume(): void {
		if (!this.started || this.runningSince !== undefined || this.expired || this.disposed) return;
		this.runningSince = globalThis.performance.now();
		this.schedule();
	}

	dispose(): void {
		if (this.disposed) return;
		if (this.runningSince !== undefined) {
			this.consumeElapsed();
			this.runningSince = undefined;
		}
		this.clearTimer();
		this.disposed = true;
	}

	private consumeElapsed(): void {
		if (this.runningSince === undefined) return;
		const now = globalThis.performance.now();
		const elapsed = Math.max(0, now - this.runningSince);
		this.remaining = Math.max(0, this.remaining - elapsed);
		this.runningSince = now;
	}

	private schedule(): void {
		this.clearTimer();
		const delay = Math.min(this.remaining, MAX_TIMER_DELAY_MS);
		this.timer = setTimeout(() => this.handleTimer(), delay);
	}

	private handleTimer(): void {
		this.timer = undefined;
		if (this.runningSince === undefined || this.expired || this.disposed) return;
		this.consumeElapsed();
		if (this.remaining === 0) this.expire();
		else this.schedule();
	}

	private expire(): void {
		if (this.expired || this.disposed) return;
		this.expired = true;
		this.remaining = 0;
		this.runningSince = undefined;
		this.clearTimer();
		this.onExpire();
	}

	private clearTimer(): void {
		if (this.timer === undefined) return;
		clearTimeout(this.timer);
		this.timer = undefined;
	}
}
