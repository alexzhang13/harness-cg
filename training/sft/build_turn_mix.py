"""Per-turn SFT dataset: each sample is one assistant turn with its full history; the loss is on that turn only.

    SFT_TOK_MODEL=Qwen/Qwen3.8-27B python build_turn_mix.py --input source.jsonl --out DIR \
        --turns-per-traj 100 --reverify --reasoning keep

Input: trajectories from build_source_rows.py. Samples whose history is not a token prefix of the full render, or
that exceed 32768 tokens, are dropped.
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = Path(os.environ.get("SFT_TEMPLATE", HERE / "qwen3.8_nothink_chat_template.jinja"))
MODEL = os.environ.get("SFT_TOK_MODEL", "Qwen/Qwen3.8-27B")
SEQ = 32768
_tok = None
_mask = None


def _init():
    global _tok, _mask
    from transformers import AutoTokenizer
    sys.path.insert(0, str(HERE / "patches"))
    import sft_lastturn
    from prime_rl.trainer.sft import data as D
    sft_lastturn.install()
    _mask = D.build_incremental_token_mask
    _tok = AutoTokenizer.from_pretrained(MODEL)
    _tok.chat_template = TEMPLATE.read_text()


def _measure(arg):
    msgs, kw = arg if isinstance(arg, tuple) else (arg, None)
    try:
        ids, mask = _mask(_tok, msgs, role_to_mask=None, chat_template_kwargs=kw or None)
        return len(ids), sum(mask)
    except Exception:  # noqa: BLE001
        return -1, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns-per-traj", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=18)
    ap.add_argument("--input", type=Path, required=True, help="whole trajectories (build_source_rows.py output)")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--reverify", action="store_true",
                    help="rows with meta.reverify: no target before live_from or on an error turn; replan turns and "
                         "the submit turn are always targets, plus --turns-per-traj sampled others")
    ap.add_argument("--reasoning", choices=["drop", "keep"], default="drop",
                    help="keep = the target turn keeps its reasoning_content")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    samples = []
    rows = [json.loads(x) for x in open(a.input)]
    for r in rows:
        meta = r.get("meta") or {}
        msgs = r["messages"]
        if a.reasoning == "drop":
            msgs = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in msgs]
        idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        rv = meta.get("reverify") if a.reverify else None
        if rv:
            # turn numbers in meta.reverify are 1-based over assistant turns
            live = int(rv.get("live_from") or 1)
            err = {int(t) for t in rv.get("error_turns") or []}
            ok = [t for t in range(live, len(idx) + 1) if t not in err]
            must = {int(c["replan_turn"]) for c in rv.get("cycles") or [] if c.get("result") == "fail" and c.get("replan_turn")}
            must |= {len(idx)}
            must &= set(ok)
            rest = [t for t in ok if t not in must]
            pick = sorted(must | set(rng.sample(rest, min(a.turns_per_traj, len(rest)))))
            targets = [idx[t - 1] for t in pick]
        else:
            targets = sorted(rng.sample(idx, min(a.turns_per_traj, len(idx))))
        for k in targets:
            # only the target turn keeps its reasoning
            m = [dict(x) for x in msgs[: k + 1]]
            for x in m[:-1]:
                x.pop("reasoning_content", None)
            samples.append({"messages": m, "env": r.get("env") or meta.get("env"), "uid": r.get("uid") or meta.get("uid"),
                            "task": r.get("task") or meta.get("task"), "turn": idx.index(k), "n_turns": len(idx),
                            **({"k": rv.get("k")} if rv else {}),
                            **({"chat_template_kwargs": r["chat_template_kwargs"]} if r.get("chat_template_kwargs") else {})})
    with mp.Pool(96, initializer=_init) as pool:
        lens = pool.map(_measure, [(s["messages"], s.get("chat_template_kwargs")) for s in samples], chunksize=32)
    bad = sum(1 for n, _ in lens if n < 0)
    long_ = sum(1 for n, _ in lens if n > SEQ)
    keep = [(s, n, nl) for s, (n, nl) in zip(samples, lens) if 0 < n <= SEQ]
    rng.shuffle(keep)
    out = meta = a.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as f:
        for s, _, _ in keep:
            f.write(json.dumps(s) + "\n")
    # packs the trainer will produce (patches/sitecustomize.py)
    packs, cur = 0, 0
    for _, n, _ in keep:
        x = n - 1
        if cur and cur + x > SEQ:
            packs, cur = packs + 1, 0
        cur += x
        if cur >= SEQ:
            packs, cur = packs + 1, 0
    ns = [n for _, n, _ in keep]
    stats = {"samples": len(keep), "dropped_not_prefix": bad, "dropped_over_seq": long_,
             "tokens": sum(ns), "loss_tokens": sum(nl for _, _, nl in keep), "len_mean": round(statistics.mean(ns)),
             "len_max": max(ns), "packs": packs, "fill": sum(ns) / packs / SEQ, f"steps_at_batch_{a.batch}": packs / a.batch,
             "turn_position": dict(sorted(collections.Counter(min(s["turn"], 9) for s, _, _ in keep).items()))}
    stats["model"], stats["template"] = MODEL, str(TEMPLATE)
    (meta / "stats.json").write_text(json.dumps(stats, indent=1))
    _init()
    ids, mask = _mask(_tok, keep[0][0]["messages"], role_to_mask=None)
    j = mask.index(True)
    (meta / "example_render.txt").write_text(
        _tok.decode(ids[:j])[-600:] + "\n=====LOSS STARTS=====\n" + _tok.decode(ids[j:]))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
