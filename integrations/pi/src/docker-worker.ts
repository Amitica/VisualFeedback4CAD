import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { mkdir, unlink } from "node:fs/promises";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

import { findEnvKeys } from "@earendil-works/pi-ai/compat";

import { InfrastructureError } from "./errors.js";
import { isoNow, readJson, writeJsonAtomic } from "./files.js";
import {
	DEFAULT_IMAGE,
	JOB_FILE,
	PROTOCOL_VERSION,
	RESPONSE_FILE,
	type AgentJob,
	type ContainerResponse,
	type ContainerState,
	type DoctorJob,
} from "./types.js";

interface CommandResult {
	code: number | null;
	stdout: string;
	stderr: string;
	timedOut: boolean;
}

export interface DockerInspect {
	Id: string;
	Image: string;
	Name: string;
	State: {
		Running: boolean;
		ExitCode: number;
		StartedAt: string;
		FinishedAt: string;
	};
	Mounts: Array<{ Source: string; Destination: string; RW: boolean }>;
}

export type ContainerLifecycleAction = "create" | "monitor" | "restart";

export function decideContainerLifecycle(options: {
	statePresent: boolean;
	containerPresent: boolean;
	allowCreate: boolean;
	running: boolean;
}): ContainerLifecycleAction {
	if (!options.statePresent) {
		if (options.allowCreate) return "create";
		throw new InfrastructureError("Persistent container state is missing; a replacement will not be created");
	}
	if (!options.containerPresent) {
		throw new InfrastructureError("The recorded persistent container is missing; it will not be replaced");
	}
	return options.running ? "monitor" : "restart";
}

export interface DockerAgentRuntimeOptions {
	image?: string;
	dockerBinary?: string;
	piAgentDir?: string;
	provider: string;
	environment?: NodeJS.ProcessEnv;
}

async function runCommand(
	command: string,
	args: string[],
	options: { input?: string; timeoutMs?: number; environment?: NodeJS.ProcessEnv } = {},
): Promise<CommandResult> {
	return await new Promise<CommandResult>((resolvePromise, reject) => {
		const child = spawn(command, args, {
			stdio: ["pipe", "pipe", "pipe"],
			...(options.environment ? { env: options.environment } : {}),
		});
		let stdout = "";
		let stderr = "";
		let settled = false;
		let timedOut = false;
		const timer = setTimeout(() => {
			timedOut = true;
			child.kill("SIGTERM");
		}, options.timeoutMs ?? 30_000);
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout += chunk;
		});
		child.stderr.on("data", (chunk: string) => {
			stderr += chunk;
		});
		child.once("error", (error) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			reject(error);
		});
		child.once("close", (code) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			resolvePromise({ code, stdout, stderr, timedOut });
		});
		child.stdin.end(options.input);
	});
}

function baseConfinementArgs(network = "bridge"): string[] {
	return [
		"--network",
		network,
		"--read-only",
		"--cap-drop",
		"ALL",
		"--security-opt",
		"no-new-privileges",
		"--cpus",
		"2",
		"--memory",
		"6g",
		"--pids-limit",
		"256",
		"--tmpfs",
		"/tmp:rw,nosuid,nodev,size=1g,mode=1777",
		"--env",
		"HOME=/tmp/home",
		"--env",
		"PI_CODING_AGENT_DIR=/pi-agent",
	];
}

export function buildDockerCreateArgs(
	containerName: string,
	workspace: string,
	piAgentDir: string,
	image: string,
	forwardedEnvironmentNames: string[] = [],
	network = "bridge",
): string[] {
	const args = ["create", "--name", containerName, ...baseConfinementArgs(network)];
	if (typeof process.getuid === "function" && typeof process.getgid === "function") {
		args.push("--user", `${process.getuid()}:${process.getgid()}`);
	}
	for (const name of forwardedEnvironmentNames) args.push("--env", name);
	args.push(
		"--mount",
		`type=bind,src=${resolve(workspace)},dst=/workspace`,
		"--mount",
		`type=bind,src=${resolve(piAgentDir)},dst=/pi-agent`,
		image,
	);
	return args;
}

export function deterministicContainerName(runId: string, fixtureId: string): string {
	const readable = `${runId}-${fixtureId}`.toLowerCase().replaceAll(/[^a-z0-9_.-]+/gu, "-").slice(0, 48);
	const digest = createHash("sha256").update(`${runId}\0${fixtureId}`).digest("hex").slice(0, 12);
	return `cadgenbench-pi-${readable}-${digest}`;
}

export function providerEnvironment(
	provider: string,
	source: NodeJS.ProcessEnv = process.env,
): { environment: NodeJS.ProcessEnv; names: string[] } {
	const environment: NodeJS.ProcessEnv = { ...source };
	if (provider === "moonshotai-cn" && !environment.MOONSHOT_API_KEY?.trim() && environment.KIMI_API_KEY?.trim()) {
		environment.MOONSHOT_API_KEY = environment.KIMI_API_KEY.trim();
	}
	const definedEnvironment = Object.fromEntries(
		Object.entries(environment).filter((entry): entry is [string, string] => typeof entry[1] === "string"),
	);
	const discovered = findEnvKeys(provider, definedEnvironment) ?? [];
	const compatibility = ["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY", "CADGENBENCH_SUBAGENT_TIMEOUT_MULTIPLIER"];
	// UCloud is configured through Pi's models.json, outside its built-in env map.
	// Pass only the variable name to Docker, as for built-in provider credentials.
	if (provider === "ucloud") compatibility.push("UCLOUD_API_KEY");
	if (provider.includes("bedrock")) {
		compatibility.push(
			"AWS_ACCESS_KEY_ID",
			"AWS_SECRET_ACCESS_KEY",
			"AWS_SESSION_TOKEN",
			"AWS_REGION",
			"AWS_DEFAULT_REGION",
		);
	}
	if (provider.includes("vertex")) compatibility.push("GOOGLE_APPLICATION_CREDENTIALS");
	const names = [...new Set([...discovered, ...compatibility])]
		.filter((name) => environment[name]?.trim())
		.sort();
	return { environment, names };
}

function statePath(workspace: string): string {
	return join(workspace, "container.json");
}

function responsePath(workspace: string): string {
	return join(workspace, RESPONSE_FILE);
}

function jobPath(workspace: string): string {
	return join(workspace, JOB_FILE);
}

function findMount(inspect: DockerInspect, destination: string): DockerInspect["Mounts"][number] | undefined {
	return inspect.Mounts.find((mount) => mount.Destination === destination);
}

export function assertPersistentContainerMatches(options: {
	inspect: DockerInspect;
	state: ContainerState;
	workspace: string;
	piAgentDir: string;
	image: string;
	imageId: string;
	expectedName: string;
}): void {
	const { inspect, state, imageId, expectedName } = options;
	const expectedWorkspace = resolve(options.workspace);
	const expectedPiDir = resolve(options.piAgentDir);
	const workspaceMount = findMount(inspect, "/workspace");
	const piMount = findMount(inspect, "/pi-agent");
	const actualName = inspect.Name.replace(/^\//u, "");
	if (
		inspect.Id !== state.container_id ||
		actualName !== state.container_name ||
		state.container_name !== expectedName
	) {
		throw new InfrastructureError("Persistent container ID or name does not match container.json");
	}
	if (inspect.Image !== imageId || state.image_id !== imageId || state.image !== options.image) {
		throw new InfrastructureError("Persistent container image does not match the recorded run image");
	}
	if (
		state.workspace !== expectedWorkspace ||
		workspaceMount?.Source !== expectedWorkspace ||
		workspaceMount.RW !== true
	) {
		throw new InfrastructureError("Persistent container /workspace mount is missing or mismatched");
	}
	if (state.pi_agent_dir !== expectedPiDir || piMount?.Source !== expectedPiDir || piMount.RW !== true) {
		throw new InfrastructureError("Persistent container /pi-agent mount is missing or mismatched");
	}
}

export class DockerAgentRuntime {
	readonly image: string;
	readonly dockerBinary: string;
	readonly piAgentDir: string;
	private readonly dockerEnvironment: NodeJS.ProcessEnv;
	private readonly forwardedEnvironmentNames: string[];
	private readonly activeContainers = new Set<string>();
	private readonly activeDoctorContainers = new Set<string>();
	private interrupted = false;

	constructor(options: DockerAgentRuntimeOptions) {
		this.image = options.image ?? DEFAULT_IMAGE;
		this.dockerBinary = options.dockerBinary ?? "docker";
		const sourceEnvironment = options.environment ?? process.env;
		this.piAgentDir = resolve(
			options.piAgentDir ?? sourceEnvironment.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent"),
		);
		const forwarded = providerEnvironment(options.provider, sourceEnvironment);
		this.dockerEnvironment = forwarded.environment;
		this.forwardedEnvironmentNames = forwarded.names;
	}

	private async command(args: string[], timeoutMs = 30_000): Promise<CommandResult> {
		return await runCommand(this.dockerBinary, args, {
			timeoutMs,
			environment: this.dockerEnvironment,
		}).catch((error: unknown) => {
			throw new InfrastructureError(`Cannot run Docker CLI: ${String(error)}`, { cause: error });
		});
	}

	async imageDigest(): Promise<string> {
		const result = await this.command(["image", "inspect", "--format", "{{.Id}}", this.image]);
		if (result.code !== 0 || result.stdout.trim() === "") {
			throw new InfrastructureError(
				`Container image ${this.image} is unavailable: ${result.stderr.trim() || "docker image inspect failed"}`,
			);
		}
		return result.stdout.trim();
	}

	async daemonVersion(): Promise<string> {
		const versionFormat = "{{.Server." + "Version}}";
		const result = await this.command(["version", "--format", versionFormat]);
		if (result.code !== 0 || result.stdout.trim() === "") {
			throw new InfrastructureError(`Docker daemon is unavailable: ${result.stderr.trim()}`);
		}
		return result.stdout.trim();
	}

	private async inspect(identifier: string): Promise<DockerInspect | null> {
		const result = await this.command(["container", "inspect", identifier]);
		if (result.code !== 0) return null;
		try {
			const parsed = JSON.parse(result.stdout) as DockerInspect[];
			return parsed[0] ?? null;
		} catch (error) {
			throw new InfrastructureError(`Docker returned malformed inspect data for ${identifier}`, { cause: error });
		}
	}

	private async persistState(workspace: string, state: ContainerState): Promise<void> {
		state.updated_at = isoNow();
		await writeJsonAtomic(statePath(workspace), state);
	}

	private validateContainer(
		inspect: DockerInspect,
		state: ContainerState,
		workspace: string,
		imageId: string,
		expectedName: string,
	): void {
		assertPersistentContainerMatches({
			inspect,
			state,
			workspace,
			piAgentDir: this.piAgentDir,
			image: this.image,
			imageId,
			expectedName,
		});
	}

	private async createContainer(
		workspace: string,
		runId: string,
		fixtureId: string,
		imageId: string,
	): Promise<ContainerState> {
		await mkdir(workspace, { recursive: true });
		await mkdir(this.piAgentDir, { recursive: true });
		const name = deterministicContainerName(runId, fixtureId);
		if ((await this.inspect(name)) !== null) {
			throw new InfrastructureError(`Container ${name} already exists but this fixture has no container.json`);
		}
		const args = buildDockerCreateArgs(
			name,
			workspace,
			this.piAgentDir,
			this.image,
			this.forwardedEnvironmentNames,
			this.dockerEnvironment.CADGENBENCH_DOCKER_NETWORK || "bridge",
		);
		const created = await this.command(args, 120_000);
		if (created.code !== 0 || created.stdout.trim() === "") {
			throw new InfrastructureError(`Failed to create persistent Agent container: ${created.stderr.trim()}`);
		}
		const inspect = await this.inspect(created.stdout.trim());
		if (inspect === null) throw new InfrastructureError("Created Agent container disappeared before inspection");
		const now = isoNow();
		const state: ContainerState = {
			protocol_version: PROTOCOL_VERSION,
			container_name: name,
			container_id: inspect.Id,
			image: this.image,
			image_id: imageId,
			workspace: resolve(workspace),
			pi_agent_dir: resolve(this.piAgentDir),
			status: "created",
			created_at: now,
			updated_at: now,
		};
		this.validateContainer(inspect, state, workspace, imageId, name);
		await this.persistState(workspace, state);
		return state;
	}

	private async loadAndValidateState(
		workspace: string,
		runId: string,
		fixtureId: string,
		imageId: string,
		allowCreate = false,
	): Promise<{ state: ContainerState; inspect: DockerInspect; action: ContainerLifecycleAction }> {
		const state = await readJson<ContainerState>(statePath(workspace)).catch(() => null);
		const expectedName = deterministicContainerName(runId, fixtureId);
		if (state === null) {
			decideContainerLifecycle({
				statePresent: false,
				containerPresent: false,
				allowCreate,
				running: false,
			});
			const created = await this.createContainer(workspace, runId, fixtureId, imageId);
			const inspect = await this.inspect(created.container_id);
			if (inspect === null) throw new InfrastructureError("Created Agent container is missing");
			return { state: created, inspect, action: "create" };
		}
		if (state.protocol_version !== PROTOCOL_VERSION) {
			throw new InfrastructureError(`Unsupported container state protocol: ${String(state.protocol_version)}`);
		}
		const inspect = await this.inspect(state.container_id);
		if (inspect === null) {
			decideContainerLifecycle({
				statePresent: true,
				containerPresent: false,
				allowCreate,
				running: false,
			});
			throw new InfrastructureError(`Persistent container ${state.container_name} is missing`);
		}
		this.validateContainer(inspect, state, workspace, imageId, expectedName);
		const action = decideContainerLifecycle({
			statePresent: true,
			containerPresent: true,
			allowCreate,
			running: inspect.State.Running,
		});
		return { state, inspect, action };
	}

	async prepareFixture(workspace: string, runId: string, fixtureId: string, imageId: string): Promise<void> {
		await this.loadAndValidateState(workspace, runId, fixtureId, imageId, true);
	}

	async validateFixture(workspace: string, runId: string, fixtureId: string, imageId: string): Promise<void> {
		await this.loadAndValidateState(workspace, runId, fixtureId, imageId);
	}

	private async waitForContainer(name: string): Promise<number> {
		const waited = await this.command(["wait", name], 7 * 24 * 60 * 60 * 1000);
		if (waited.timedOut || waited.code !== 0 || !/^\d+$/u.test(waited.stdout.trim())) {
			throw new InfrastructureError(`Failed while monitoring persistent container ${name}: ${waited.stderr.trim()}`);
		}
		return Number.parseInt(waited.stdout.trim(), 10);
	}

	private async stopRetainedContainer(name: string): Promise<void> {
		const current = await this.inspect(name);
		if (current === null || !current.State.Running) return;
		await this.command(["kill", "--signal", "SIGINT", name], 30_000);
		const waited = await this.command(["wait", name], 30_000);
		if (waited.timedOut || waited.code !== 0) {
			await this.command(["stop", "--time", "10", name], 30_000);
		}
	}

	async runFixture(workspace: string, job: AgentJob, imageId: string): Promise<ContainerResponse> {
		const loaded = await this.loadAndValidateState(workspace, job.run_id, job.fixture.id, imageId);
		const { state, action } = loaded;
		if (this.interrupted) throw new InfrastructureError("Agent container start was interrupted");
		this.activeContainers.add(state.container_name);
		try {
			if (action !== "monitor") {
				await writeJsonAtomic(jobPath(workspace), job);
				await unlink(responsePath(workspace)).catch(() => undefined);
				if (this.interrupted) throw new InfrastructureError("Agent container start was interrupted");
				const started = await this.command(["start", state.container_name], 120_000);
				if (started.code !== 0) {
					throw new InfrastructureError(`Failed to start persistent container: ${started.stderr.trim()}`);
				}
				state.status = "running";
				state.started_at = isoNow();
				delete state.exit_code;
				await this.persistState(workspace, state);
				if (this.interrupted) await this.command(["kill", "--signal", "SIGINT", state.container_name]);
			}
			const exitCode = await this.waitForContainer(state.container_name);
			state.status = "stopped";
			state.exit_code = exitCode;
			await this.persistState(workspace, state);
			const response = await readJson<ContainerResponse>(responsePath(workspace)).catch(() => null);
			if (response === null) {
				throw new InfrastructureError(
					`Agent container exited ${exitCode} without writing ${RESPONSE_FILE}`,
				);
			}
			if (response.protocol_version !== PROTOCOL_VERSION || response.operation !== "run_fixture") {
				throw new InfrastructureError("Agent container wrote a mismatched response protocol");
			}
			if (!response.ok) throw new InfrastructureError(response.error ?? `Agent container exited ${exitCode}`);
			return response;
		} catch (error) {
			await this.stopRetainedContainer(state.container_name).catch(() => undefined);
			const latest = await this.inspect(state.container_name).catch(() => null);
			if (latest) {
				state.status = latest.State.Running ? "running" : "stopped";
				if (!latest.State.Running) state.exit_code = latest.State.ExitCode;
				await this.persistState(workspace, state).catch(() => undefined);
			}
			throw error;
		} finally {
			this.activeContainers.delete(state.container_name);
		}
	}

	async doctor(workspace: string, job: DoctorJob): Promise<ContainerResponse> {
		await mkdir(workspace, { recursive: true });
		await mkdir(this.piAgentDir, { recursive: true });
		await writeJsonAtomic(jobPath(workspace), job);
		await unlink(responsePath(workspace)).catch(() => undefined);
		const name = `cadgenbench-pi-doctor-${process.pid}-${randomUUID().slice(0, 10)}`;
		const createArgs = buildDockerCreateArgs(
			name,
			workspace,
			this.piAgentDir,
			this.image,
			this.forwardedEnvironmentNames,
			this.dockerEnvironment.CADGENBENCH_DOCKER_NETWORK || "bridge",
		);
		const args = ["run", "--rm", ...createArgs.slice(1)];
		this.activeDoctorContainers.add(name);
		try {
			const result = await this.command(args, 600_000);
			if (result.timedOut || result.code !== 0) {
				throw new InfrastructureError(
					`Agent image doctor failed: ${result.stderr.trim() || result.stdout.trim() || `exit ${String(result.code)}`}`,
				);
			}
			const response = await readJson<ContainerResponse>(responsePath(workspace)).catch(() => null);
			if (response === null || response.protocol_version !== PROTOCOL_VERSION || response.operation !== "doctor") {
				throw new InfrastructureError("Agent image doctor did not write a valid response");
			}
			if (!response.ok) throw new InfrastructureError(response.error ?? "Agent image doctor failed");
			return response;
		} catch (error) {
			await this.stopRetainedContainer(name).catch(() => undefined);
			await this.command(["container", "rm", "--force", name], 30_000).catch(() => undefined);
			throw error;
		} finally {
			this.activeDoctorContainers.delete(name);
		}
	}

	async interruptAll(): Promise<void> {
		this.interrupted = true;
		await Promise.all(
			[...this.activeContainers, ...this.activeDoctorContainers].map(
				async (name) => await this.stopRetainedContainer(name),
			),
		);
	}
}
