// Truncates tool results to PI_TOOL_OUTPUT_CAP_BYTES (default 6000; 70% head, 30% tail).
// Also a tool-call budget, PI_MAX_TURNS: soft mode replaces later tool results with a stop message, hard mode
// prints the sentinel and exits (scored 0).

import * as fs from "node:fs";

const MAX_BYTES = Number(process.env.PI_TOOL_OUTPUT_CAP_BYTES || 6000);
const MAX_TURNS = Number(process.env.PI_MAX_TURNS || 30);

let toolTurns = 0; // tool results seen by this process

const STOP_DIRECTIVE =
  `⛔ TURN LIMIT REACHED (${MAX_TURNS} tool-using turns). Do NOT call any more tools. ` +
  `Using only what you have already gathered, output your FINAL ANSWER now as plain text, ` +
  `in the exact required OUTPUT FORMAT.`;

function capText(text: string): string {
  const bytes = Buffer.byteLength(text, "utf8");
  if (bytes <= MAX_BYTES) return text;
  const headBudget = Math.floor(MAX_BYTES * 0.7);
  const tailBudget = MAX_BYTES - headBudget;
  const head = text.slice(0, headBudget);
  const tail = text.slice(-tailBudget);
  const omitted = bytes - Buffer.byteLength(head, "utf8") - Buffer.byteLength(tail, "utf8");
  return (
    head +
    `\n\n...[read-cap: ${omitted} bytes omitted. Your context window is small — do NOT read large chunks. ` +
    `Narrow the search: grep a more specific pattern, or read a small line range with \`sed -n 'A,Bp' context.txt\`.]...\n\n` +
    tail
  );
}

// replacement in the same shape as the original content
function asContent(sample: unknown, text: string): unknown {
  if (typeof sample === "string") return text;
  return [{ type: "text", text }];
}

function capContent(content: unknown): { changed: boolean; value: unknown } {
  if (typeof content === "string") {
    const t = capText(content);
    return { changed: t !== content, value: t };
  }
  if (Array.isArray(content)) {
    let changed = false;
    const mapped = content.map((block: any) => {
      if (block && block.type === "text" && typeof block.text === "string") {
        const t = capText(block.text);
        if (t !== block.text) changed = true;
        return { ...block, text: t };
      }
      return block;
    });
    return { changed, value: mapped };
  }
  return { changed: false, value: content };
}

const HARD_CAP = (process.env.PI_TURN_CAP_MODE || "soft").toLowerCase() === "hard";
const TURN_CAP_SENTINEL = "__PI_TURN_CAP_HIT__"; // keep in sync with rlm_train/pi/v1_env.py

let toolCalls = 0;

export default function (pi: any) {
  pi.on("tool_call", async () => {
    if (!HARD_CAP) return;
    toolCalls += 1;
    if (toolCalls > MAX_TURNS) {
      fs.writeSync(1, `\n${TURN_CAP_SENTINEL} (${MAX_TURNS} tool calls)\n`);
      process.exit(0);
    }
  });

  pi.on("tool_result", async (event: any) => {
    toolTurns += 1;
    if (!HARD_CAP && toolTurns > MAX_TURNS) {
      return { content: asContent(event.content, STOP_DIRECTIVE) };
    }
    const { changed, value } = capContent(event.content);
    if (changed) return { content: value };
  });
}
