"""CorpusQA (1M, real_estate_en) pool tasks: the env's root prompt, the document list as context, the env's scorer."""
import asyncio
import json

from huggingface_hub import hf_hub_download

import corpusqa_real.env as E

_SCORE = E._make_score(1e-4, "containment")
JOIN = "\n\n=== DOCUMENT {i}/{n} ===\n\n"


def _rows(keys):
    want = set(keys)
    path = hf_hub_download(E.HF_DATASET, E.SET_FILES["1m"], repo_type="dataset")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("id") in want:
                yield r


def load_items(keys):
    out = {}
    for r in _rows(keys):
        question = (r.get("question") or "").strip()
        gold = r.get("answer")
        user = system = ""
        for m in r.get("prompt") or []:
            if m.get("role") == "user":
                user = m.get("content") or ""
            elif m.get("role") == "system":
                system = m.get("content") or ""
        docs = E._split_documents(user)
        docs[-1], output_req = E._split_task_block(docs[-1])
        kind = E._classify_answer(gold)
        shape = (f"{output_req}\n\nSubmit the final answer through `answer[\"content\"]`, in exactly the format required "
                 "above (the \"The answer is:\" prefix is optional).") if output_req else {
            "list": 'Answer with a JSON array of the matching entity names, e.g. ["Name A", "Name B"]. Include every match and nothing else.',
            "number": "Answer with the single number only — no units, no commas, no prose.",
            "str": "Answer with the value only — no explanation."}[kind]
        sep = "\n\n" if output_req else " "
        prompt = (f"{system.strip() or 'You are a professional data analyst.'}\n\nQuestion: {question}\n\n"
                  f"`context` is a list of {len(docs)} documents (the source corpus); the answer requires aggregating across them.{sep}{shape}")
        n = len(docs)
        out[r["id"]] = dict(env="corpusqa", key=r["id"], kind=kind, gold=gold, question=prompt, n_docs=n,
                            context="".join(JOIN.format(i=i + 1, n=n) + d for i, d in enumerate(docs)), context_docs=docs)
    return out


def score(item, answer):
    state = {"rlm_final_answer": answer}
    info = {"gold_answer": item["gold"], "answer_kind": item["kind"]}
    return float(asyncio.run(_SCORE(info, state)))
