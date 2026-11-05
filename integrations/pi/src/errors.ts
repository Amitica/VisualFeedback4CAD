export class UsageError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "UsageError";
	}
}

export class InfrastructureError extends Error {
	constructor(message: string, options?: ErrorOptions) {
		super(message, options);
		this.name = "InfrastructureError";
	}
}

export function errorMessage(error: unknown): string {
	return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

