import { describe, expect, it } from "vitest";

import { historicalImageKeys, restrictToHistoricalImages, pruneContextImages } from "../src/context.js";

const image = (name: string) => ({ type: "image", data: name, mimeType: "image/png" });

describe("pruneContextImages", () => {
	it("retains the original drawings and only the two newest rendered tool results", () => {
		const messages = [
			{ role: "user", content: [{ type: "text", text: "drawing" }, image("original-a"), image("original-b")] },
			{ role: "assistant", content: "call one" },
			{ role: "toolResult", content: [{ type: "text", text: "turn 1" }, image("render-1")] },
			{ role: "toolResult", content: [{ type: "text", text: "turn 2" }, image("render-2")] },
			{ role: "toolResult", content: [{ type: "text", text: "turn 3" }, image("render-3")] },
		];

		const result = pruneContextImages(messages);

		expect(result[0]).toEqual(messages[0]);
		expect(JSON.stringify(result[2])).not.toContain("render-1");
		expect(JSON.stringify(result[2])).toContain("Older CAD render omitted");
		expect(JSON.stringify(result[3])).toContain("render-2");
		expect(JSON.stringify(result[4])).toContain("render-3");
		expect(messages[2]).toEqual({
			role: "toolResult",
			content: [{ type: "text", text: "turn 1" }, image("render-1")],
		});
	});

	it("leaves short image histories untouched", () => {
		const messages = [{ role: "user", content: [image("drawing")] }, { role: "toolResult", content: [image("one")] }];
		expect(pruneContextImages(messages)).toBe(messages);
	});
});



describe("direct reflection historical image boundary", () => {
	it("preserves Initial images while stripping new images from every later tool and message", () => {
		const original = [{ role: "user", content: [image("drawing")] }, { role: "toolResult", toolName: "execute_build123d", content: [image("initial-render")] }];
		const allowed = historicalImageKeys(original);
		const extra = [
			{ role: "toolResult", toolName: "read", content: [{ type: "text", text: "diagnostic" }, image("new-render")] },
			{ role: "toolResult", toolName: "execute_build123d", content: [image("new-execute-render")] },
			{ role: "user", content: [image("new-attachment")] },
		];
		const visible = restrictToHistoricalImages([...original, ...extra], allowed);
		expect(visible.slice(0, 2)).toEqual(original);
		expect(JSON.stringify(visible)).not.toContain("new-render");
		expect(JSON.stringify(visible)).not.toContain("new-execute-render");
		expect(JSON.stringify(visible)).not.toContain("new-attachment");
		expect(JSON.stringify(visible)).toContain("diagnostic");
		expect(extra[0]!.content).toHaveLength(2);
	});
});
