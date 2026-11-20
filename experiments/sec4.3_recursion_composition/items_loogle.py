"""LooGLE-v2 pool tasks (>= 500K-token documents): the env's root prompt, the document as context, the env's scorer
without the judge fallback."""
import asyncio

from datasets import load_dataset

import loogle_v2_real.env as E

_SCORE = E._make_score(1e-3)


def load_items(keys):
    want = set(keys)
    ds = load_dataset(E.HF_DATASET, split="test")
    out = {}
    for r in ds.filter(lambda x: (x.get("id") or "") in want):
        ctx = r.get("context") or ""
        question = (r.get("question") or "").strip()
        gold = (r.get("answer") or "").strip()
        instruction = (r.get("instruction") or "").strip()
        options = E._parse_options(r.get("options"))
        opt_block = "" if "{options}" in instruction else E._format_options(r.get("options"))
        if instruction:
            prompt = (instruction.replace("{context}", E._CONTEXT_POINTER).replace("{question}", question)
                      .replace("{options}", "\n".join(options)))
            if "{question}" not in instruction and question not in prompt:
                prompt = f"{prompt}\n\nQuestion: {question}"
            prompt += opt_block
        else:
            prompt = f"{question}{opt_block}"
        out[r["id"]] = dict(env="loogle", key=r["id"], kind=(r.get("type") or "").upper(), gold=gold, question=prompt,
                            context=ctx, options=options, source=r.get("source"), task=r.get("task"))
    return out


def score(item, answer):
    state = {"rlm_final_answer": answer}
    info = {"gold_answer": item["gold"], "answer_type": item["kind"], "question": item["question"], "options": item["options"]}
    return float(asyncio.run(_SCORE(info, state)))
