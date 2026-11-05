import { createHash } from "node:crypto";
import type { AgentSession } from "@earendil-works/pi-coding-agent";

import type { UsageTotals } from "./types.js";

type ContentBlock = { type?: unknown; [key: string]: unknown };
export type ContextMessage = { role?: unknown; toolName?: unknown; content?: unknown };

function hasImage(message: ContextMessage): boolean {
	return (
		Array.isArray(message.content) &&
		(message.content as ContentBlock[]).some((block) => block.type === "image")
	);
}

/**
 * Keep the original drawing attachment and the newest render feedback only.
 * The returned messages are provider context copies; Pi's append-only JSONL is
 * untouched and therefore remains a complete audit record.
 */
export function pruneContextImages<T>(messages: T[], recentRenderCount = 2): T[] {
	const typedMessages = messages as ContextMessage[];
	const imageIndices = typedMessages.flatMap((message, index) => (hasImage(message) ? [index] : []));
	if (imageIndices.length <= recentRenderCount + 1) return messages;
	const originalIndex = imageIndices[0];
	const recentIndices = imageIndices.filter((index) => index !== originalIndex).slice(-recentRenderCount);
	const retained = new Set([originalIndex, ...recentIndices]);
	return typedMessages.map((message, index) => {
		if (retained.has(index) || !hasImage(message) || !Array.isArray(message.content)) return message;
		const content = (message.content as ContentBlock[]).filter((block) => block.type !== "image");
		content.push({ type: "text", text: "[Older CAD render omitted from provider context; retained in session JSONL.]" });
		return { ...message, content };
	}) as T[];
}

/** Capture image identities only from the immutable Initial session snapshot. */
export function historicalImageKeys(messages: ContextMessage[]): Set<string> {
	const keys = new Set<string>();
	for (const message of messages) {
		if (!Array.isArray(message.content)) continue;
		for (const block of message.content as ContentBlock[]) {
			if (block.type === "image") keys.add(createHash("sha256").update(JSON.stringify(block)).digest("hex"));
		}
	}
	return keys;
}

/** Keep all Initial evidence; withhold new images from any later message/tool. */
export function restrictToHistoricalImages<T>(messages: T[], allowed: ReadonlySet<string>): T[] {
	return (messages as ContextMessage[]).map(message => {
		if (!Array.isArray(message.content)) return message;
		const content = (message.content as ContentBlock[]).filter(block => block.type !== "image" ||
			allowed.has(createHash("sha256").update(JSON.stringify(block)).digest("hex")));
		return content.length === message.content.length ? message : { ...message, content };
	}) as T[];
}

const MAIN_VIEW_REFLECT_IMAGE_TOOLS = new Set([
	"read",
	"inspect_comparison",
	"annotate_target",
	"dual_roi_crop",
	"calibrate_drawing_scale",
]);

/**
 * The locked-view continuation may see engineering drawings attached by the
 * user, images opened through the normal read tool, and images returned by its
 * comparison tools. Other tool images, including execute_build123d CAD
 * renders, are removed from provider context while remaining in the append-only
 * session audit.
 */
export function restrictMainViewReflectImages<T>(messages: T[]): T[] {
	return (messages as ContextMessage[]).map((message) => {
		if (!hasImage(message) || !Array.isArray(message.content)) return message;
		const allowed = message.role === "user" || (
			message.role === "toolResult" &&
			typeof message.toolName === "string" &&
			MAIN_VIEW_REFLECT_IMAGE_TOOLS.has(message.toolName)
		);
		if (allowed) return message;
		const content = (message.content as ContentBlock[]).filter((block) => block.type !== "image");
		content.push({
			type: "text",
			text: "[Image withheld from this experiment. Use inspect_comparison, dual_roi_crop, and calibrate_drawing_scale for CAD visual evidence.]",
		});
		return { ...message, content };
	}) as T[];
}

export function emptyUsageTotals(): UsageTotals {
	return {
		tokens: { input: 0, output: 0, cache_read: 0, cache_write: 0, total: 0 },
		cost: 0,
		tool_calls: 0,
	};
}

export function sessionUsageTotals(session: AgentSession): UsageTotals {
	const stats = session.getSessionStats();
	return {
		tokens: {
			input: stats.tokens.input,
			output: stats.tokens.output,
			cache_read: stats.tokens.cacheRead,
			cache_write: stats.tokens.cacheWrite,
			total: stats.tokens.total,
		},
		cost: stats.cost,
		tool_calls: stats.toolCalls,
	};
}

export function addUsageTotals(...items: Array<UsageTotals | undefined>): UsageTotals {
	const total = emptyUsageTotals();
	for (const item of items) {
		if (!item) continue;
		total.tokens.input += item.tokens.input;
		total.tokens.output += item.tokens.output;
		total.tokens.cache_read += item.tokens.cache_read;
		total.tokens.cache_write += item.tokens.cache_write;
		total.tokens.total += item.tokens.total;
		total.cost += item.cost;
		total.tool_calls += item.tool_calls;
	}
	return total;
}
