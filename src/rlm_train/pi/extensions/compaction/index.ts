// Context compaction during the tool loop (only when PI_COMPACT_AT_TOKENS is set). pi's own compaction only runs
// on agent_end, which never comes mid-task under `pi -p`. Older messages are summarized with pi's generateSummary,
// the last PI_COMPACT_KEEP_RECENT_TOKENS are kept, and the output-format footer is re-attached.
// Prints nothing: stdout is the graded answer.

import { existsSync, readFileSync } from "node:fs";
import { calculateContextTokens, estimateTokens, generateSummary } from "@mariozechner/pi-coding-agent";

const TRIGGER = Number(process.env.PI_COMPACT_AT_TOKENS || 0);
const KEEP_RECENT = Number(process.env.PI_COMPACT_KEEP_RECENT_TOKENS || 8192);
const RESERVE = Number(process.env.PI_COMPACT_RESERVE_TOKENS || 16384);
const FOOTER_FILE = process.env.PI_OUTPUT_FORMAT_FILE || "pi/output_format.txt";
const GUARD_OFF = ["1", "true", "yes"].includes(
  String(process.env.PI_COMPACTGUARD_DISABLE || "").toLowerCase(),
);

const FOOTER_MARKER = "\n\n--- STILL REQUIRED (restored after context compaction) ---\n";
const FOOTER_PREFACE =
  "The conversation above was compacted, which dropped the verbatim wording of " +
  "the original request. Your task is unchanged, and so is how it is scored. " +
  "Finish the task and put your final answer on its own line at the very end, " +
  "in exactly this format:\n\n";

let footerCache: string | null | undefined;

function footer(): string | null {
  if (footerCache !== undefined) return footerCache;
  try {
    footerCache =
      existsSync(FOOTER_FILE) && readFileSync(FOOTER_FILE, "utf8").trim().length > 0
        ? readFileSync(FOOTER_FILE, "utf8").trim()
        : null;
  } catch {
    footerCache = null;
  }
  return footerCache;
}

// one pi process per rollout or sub-agent
let summary: string | null = null; // covers messages[0 .. summarizedCount)
let summarizedCount = 0;

function summaryMessage(): any {
  const f = GUARD_OFF ? null : footer();
  const text = f ? `${summary}${FOOTER_MARKER}${FOOTER_PREFACE}${f}` : String(summary);
  return { role: "compactionSummary", summary: text, tokensBefore: 0, timestamp: Date.now() };
}

function view(messages: any[]): any[] {
  if (summary === null) return messages;
  return [summaryMessage(), ...messages.slice(summarizedCount)];
}

/** Context size in tokens: the max of pi's usage-based figure and a chars/4 estimate over all messages (the
 *  interception endpoint reports no usage, so the usage-based figure alone under-counts). */
function contextTokens(messages: any[]): number {
  let lastUsage = -1;
  let hybrid = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (
      m?.role === "assistant" &&
      m.usage &&
      m.stopReason !== "aborted" &&
      m.stopReason !== "error"
    ) {
      hybrid = calculateContextTokens(m.usage);
      lastUsage = i;
      break;
    }
  }
  for (let i = lastUsage + 1; i < messages.length; i++) hybrid += estimateTokens(messages[i]);
  let estimate = 0;
  for (const m of messages) estimate += estimateTokens(m);
  return Math.max(hybrid, estimate);
}

/** Newest index that may START the kept tail: walks back until KEEP_RECENT is
 *  covered, then backs off any tool results so none is split from its call. */
function findCut(messages: any[], start: number): number {
  let acc = 0;
  let cut = messages.length;
  for (let i = messages.length - 1; i > start; i--) {
    acc += estimateTokens(messages[i]);
    cut = i;
    if (acc >= KEEP_RECENT) break;
  }
  while (cut > start && messages[cut]?.role === "toolResult") cut--;
  return cut;
}

async function compactNow(messages: any[], ctx: any): Promise<boolean> {
  const cut = findCut(messages, summarizedCount);
  if (cut <= summarizedCount) return false;
  const model = ctx?.model;
  if (!model) return false;
  const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
  if (!auth?.ok || !auth.apiKey) return false;
  const head = messages.slice(summarizedCount, cut);
  const next = await generateSummary(
    head,
    model,
    RESERVE,
    auth.apiKey,
    auth.headers,
    ctx.signal,
    undefined,
    summary ?? undefined,
    undefined,
  );
  if (typeof next !== "string" || next.trim().length === 0) return false;
  summary = next;
  summarizedCount = cut;
  return true;
}

export default function (pi: any) {
  if (!Number.isFinite(TRIGGER) || TRIGGER <= 0) return;
  pi.on("context", async (event: any, ctx: any) => {
    const messages = event?.messages;
    if (!Array.isArray(messages) || messages.length === 0) return;

    let out = view(messages);
    if (contextTokens(out) > TRIGGER) {
      try {
        await compactNow(messages, ctx);
      } catch {
        // on a failed summarization, send the largest view available
      }
      out = view(messages);
    }
    if (out === messages) return;
    return { messages: out };
  });
}
