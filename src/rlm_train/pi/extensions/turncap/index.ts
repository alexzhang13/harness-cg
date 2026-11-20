// Turn cap and wall-clock deadline for one pi process (root or sub-agent). A turn is one model call.
// When a budget runs out the process prints SENTINEL and exits 0, so the rollout is scored as no answer (a non-zero
// exit would be rescheduled instead).
//
//   PI_MAX_TURNS         model calls allowed (default 30)
//   PI_WALL_TIMEOUT_S    deadline in seconds (default 0 = none)
//   PI_TURNCAP_DISABLE   1 disables the extension
//   PI_TURN_WARN_BEFORE  append a turns-left notice to tool results in the last N turns (at most a third of the
//                        budget; default 0 = off)

import * as fs from "node:fs";

const SENTINEL = "__PI_TURN_CAP_HIT__"; // keep in sync with rlm_train/pi/v1_env.py
const MAX_TURNS = Number(process.env.PI_MAX_TURNS || 30);
const WALL_S = Number(process.env.PI_WALL_TIMEOUT_S || 0);
const WARN_BEFORE = Number(process.env.PI_TURN_WARN_BEFORE || 0);
const DISABLED =["1", "true", "yes"].includes((process.env.PI_TURNCAP_DISABLE || "").toLowerCase());

function stop(reason: string): never {
	fs.writeSync(1, `\n${SENTINEL} (${reason})\n`);
	process.exit(0);
}

export default function (pi: any) {
	if (DISABLED) return;

	if (WALL_S > 0) {
		// unref so the timer does not keep the process alive
		const timer: any = setTimeout(() => stop(`wall-clock ${WALL_S}s`), WALL_S * 1000);
		if (timer && typeof timer.unref === "function") timer.unref();
	}

	let turns = 0;
	pi.on("turn_start", async () => {
		turns += 1;
		if (turns > MAX_TURNS) stop(`${MAX_TURNS} turns`);
	});

	const warnBefore = Math.min(WARN_BEFORE, Math.floor(MAX_TURNS / 3));
	if (warnBefore > 0) {
		pi.on("tool_result", async (event: any) => {
			const remaining = MAX_TURNS - turns;
			if (remaining < 1 || remaining > warnBefore) return;
			const notice =
				remaining === 1
					? `[TURN BUDGET: ${turns} of ${MAX_TURNS} model calls used. Your NEXT response is your last: write the final answer now, with NO tool calls, or you score zero.]`
					: `[Turn budget: ${turns} of ${MAX_TURNS} model calls used, ${remaining} left. Start writing your final answer; use a tool only if it is essential.]`;
			const c = event.content;
			if (typeof c === "string") return { content: `${c}\n\n${notice}` };
			if (Array.isArray(c)) return { content: [...c, { type: "text", text: notice }] };
			return { content: [{ type: "text", text: notice }] };
		});
	}
}
