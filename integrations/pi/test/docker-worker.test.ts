import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
	assertPersistentContainerMatches,
	buildDockerCreateArgs,
	decideContainerLifecycle,
	deterministicContainerName,
	DockerAgentRuntime,
	providerEnvironment,
	type DockerInspect,
} from "../src/docker-worker.js";
import type { ContainerState } from "../src/types.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
	await Promise.all(temporaryDirectories.splice(0).map(async (directory) => await rm(directory, { recursive: true })));
});

describe("persistent Agent container configuration", () => {
	it("uses bridge networking, two read-write bind mounts, and never enables automatic removal", () => {
		const args = buildDockerCreateArgs(
			"cadgenbench-pi-run-145",
			"/tmp/fixture-145",
			"/tmp/pi-agent",
			"worker:image",
			["MOONSHOT_API_KEY", "HTTPS_PROXY"],
		);

		expect(args.slice(0, 3)).toEqual(["create", "--name", "cadgenbench-pi-run-145"]);
		expect(args).toContain("bridge");
		expect(args).toContain("--read-only");
		expect(args).toContain("ALL");
		expect(args).toContain("no-new-privileges");
		expect(args).not.toContain("--rm");
		expect(args.filter((value) => value.startsWith("type=bind,"))).toEqual([
			"type=bind,src=/tmp/fixture-145,dst=/workspace",
			"type=bind,src=/tmp/pi-agent,dst=/pi-agent",
		]);
		expect(args).toContain("PI_CODING_AGENT_DIR=/pi-agent");
		expect(args).toContain("MOONSHOT_API_KEY");
		expect(args).toContain("HTTPS_PROXY");
		expect(args.at(-1)).toBe("worker:image");
	});

	it("derives a stable, Docker-safe container name from the run and fixture", () => {
		const first = deterministicContainerName("Run With Spaces/2026", "145");
		const second = deterministicContainerName("Run With Spaces/2026", "145");
		expect(first).toBe(second);
		expect(first).toMatch(/^cadgenbench-pi-[a-z0-9_.-]+-[a-f0-9]{12}$/u);
		expect(deterministicContainerName("Run With Spaces/2026", "146")).not.toBe(first);
	});

	it("allows a dedicated experiment network while preserving confinement", () => {
		const args = buildDockerCreateArgs("network-test", "/tmp/workspace", "/tmp/pi-agent", "worker:image", [], "cadgenbench-retry12");
		expect(args[args.indexOf("--network") + 1]).toBe("cadgenbench-retry12");
		expect(args).toContain("--read-only");
		expect(args).toContain("no-new-privileges");
	});

	it("forwards discovered credentials and maps KIMI_API_KEY for Moonshot China", () => {
		const forwarded = providerEnvironment("moonshotai-cn", {
			KIMI_API_KEY: "kimi-secret",
			HTTPS_PROXY: "http://proxy.invalid:8080",
			EMPTY_VALUE: "",
		});

		expect(forwarded.environment.MOONSHOT_API_KEY).toBe("kimi-secret");
		expect(forwarded.names).toContain("MOONSHOT_API_KEY");
		expect(forwarded.names).toContain("HTTPS_PROXY");
		expect(forwarded.names).not.toContain("EMPTY_VALUE");
	});

	it("forwards the custom UCloud credential by name without putting its value in Docker arguments", () => {
		const forwarded = providerEnvironment("ucloud", {
			UCLOUD_API_KEY: "test-ucloud-credential",
			OPENAI_API_KEY: "unrelated-provider-credential",
		});
		expect(forwarded.names).toEqual(["UCLOUD_API_KEY"]);
		const args = buildDockerCreateArgs("ucloud-test", "/tmp/workspace", "/tmp/pi-agent", "worker:image", forwarded.names);
		expect(args).toContain("UCLOUD_API_KEY");
		expect(args.join(" ")).not.toContain("test-ucloud-credential");
		expect(args).not.toContain("OPENAI_API_KEY");
		expect(providerEnvironment("ucloud", { UCLOUD_API_KEY: "  " }).names).toEqual([]);
		expect(providerEnvironment("moonshotai-cn", { UCLOUD_API_KEY: "test-ucloud-credential" }).names).toEqual([]);
	});

	it("forwards the configured subagent timeout multiplier to every provider container", () => {
		const forwarded = providerEnvironment("ucloud", { CADGENBENCH_SUBAGENT_TIMEOUT_MULTIPLIER: "3" });
		expect(forwarded.names).toEqual(["CADGENBENCH_SUBAGENT_TIMEOUT_MULTIPLIER"]);
		expect(forwarded.environment.CADGENBENCH_SUBAGENT_TIMEOUT_MULTIPLIER).toBe("3");
	});

	it("monitors running containers, restarts stopped containers, and refuses missing replacements", () => {
		expect(
			decideContainerLifecycle({ statePresent: true, containerPresent: true, allowCreate: false, running: true }),
		).toBe("monitor");
		expect(
			decideContainerLifecycle({ statePresent: true, containerPresent: true, allowCreate: false, running: false }),
		).toBe("restart");
		expect(
			decideContainerLifecycle({ statePresent: false, containerPresent: false, allowCreate: true, running: false }),
		).toBe("create");
		expect(() =>
			decideContainerLifecycle({ statePresent: false, containerPresent: false, allowCreate: false, running: false }),
		).toThrow("replacement will not be created");
		expect(() =>
			decideContainerLifecycle({ statePresent: true, containerPresent: false, allowCreate: false, running: false }),
		).toThrow("will not be replaced");
	});

	it("rejects image, ID, name, and mount drift against container.json", () => {
		const state: ContainerState = {
			protocol_version: 5,
			container_name: "cadgenbench-pi-run-145",
			container_id: "container-id",
			image: "worker:image",
			image_id: "sha256:image",
			workspace: "/tmp/fixture-145",
			pi_agent_dir: "/tmp/pi-agent",
			status: "stopped",
			created_at: "2026-01-01T00:00:00.000Z",
			updated_at: "2026-01-01T00:00:00.000Z",
		};
		const inspect: DockerInspect = {
			Id: "container-id",
			Image: "sha256:image",
			Name: "/cadgenbench-pi-run-145",
			State: { Running: false, ExitCode: 0, StartedAt: "", FinishedAt: "" },
			Mounts: [
				{ Source: "/tmp/fixture-145", Destination: "/workspace", RW: true },
				{ Source: "/tmp/pi-agent", Destination: "/pi-agent", RW: true },
			],
		};
		const validate = (candidate: DockerInspect, candidateState = state) =>
			assertPersistentContainerMatches({
				inspect: candidate,
				state: candidateState,
				workspace: "/tmp/fixture-145",
				piAgentDir: "/tmp/pi-agent",
				image: "worker:image",
				imageId: "sha256:image",
				expectedName: "cadgenbench-pi-run-145",
			});
		expect(() => validate(inspect)).not.toThrow();
		expect(() => validate({ ...inspect, Image: "sha256:other" })).toThrow("image");
		expect(() => validate({ ...inspect, Id: "other" })).toThrow("ID or name");
		expect(() => validate({ ...inspect, Name: "/other" })).toThrow("ID or name");
		expect(() => validate({ ...inspect, Mounts: [] })).toThrow("workspace mount");
		expect(() => validate(inspect, { ...state, image: "other:image" })).toThrow("image");
		expect(() => validate(inspect, { ...state, image_id: "sha256:other" })).toThrow("image");
		expect(() => validate(inspect, { ...state, workspace: "/tmp/moved-fixture" })).toThrow("workspace mount");
		expect(() => validate(inspect, { ...state, pi_agent_dir: "/tmp/other-pi" })).toThrow("pi-agent mount");
		expect(() =>
			validate({
				...inspect,
				Mounts: [
					{ Source: "/tmp/fixture-145", Destination: "/workspace", RW: false },
					{ Source: "/tmp/pi-agent", Destination: "/pi-agent", RW: true },
				],
			}),
		).toThrow("workspace mount");
		expect(() =>
			validate({
				...inspect,
				Mounts: [
					{ Source: "/tmp/fixture-145", Destination: "/workspace", RW: true },
					{ Source: "/tmp/pi-agent", Destination: "/pi-agent", RW: false },
				],
			}),
		).toThrow("pi-agent mount");
	});

	it("sends SIGINT and waits for a fixture container without removing it", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-fake-docker-"));
		temporaryDirectories.push(directory);
		const fakeDocker = join(directory, "docker");
		const commandLog = join(directory, "commands.log");
		await writeFile(
			fakeDocker,
			`#!/bin/sh
printf '%s\\n' "$*" >> "$DOCKER_LOG"
if [ "$1" = "container" ] && [ "$2" = "inspect" ]; then
  printf '%s\\n' '[{"Id":"fixture-id","Image":"sha256:image","Name":"/retained-fixture","State":{"Running":true,"ExitCode":0,"StartedAt":"","FinishedAt":""},"Mounts":[]}]'
elif [ "$1" = "wait" ]; then
  printf '%s\\n' '130'
fi
exit 0
`,
		);
		await chmod(fakeDocker, 0o700);
		const runtime = new DockerAgentRuntime({
			dockerBinary: fakeDocker,
			provider: "test",
			piAgentDir: join(directory, "pi-agent"),
			environment: { DOCKER_LOG: commandLog },
		});
		const internals = runtime as unknown as { activeContainers: Set<string> };
		internals.activeContainers.add("retained-fixture");

		await runtime.interruptAll();

		const commands = await readFile(commandLog, "utf8");
		expect(commands).toContain("container inspect retained-fixture");
		expect(commands).toContain("kill --signal SIGINT retained-fixture");
		expect(commands).toContain("wait retained-fixture");
		expect(commands).not.toContain("container rm");
		expect(commands).not.toContain("--rm");
	});
});
