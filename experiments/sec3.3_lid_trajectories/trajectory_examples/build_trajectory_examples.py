"""Build the six appendix trajectory examples: trajectory_examples.tex and trajectory_examples.json.

Each pairs an eval trajectory (Fig. jaccard checkpoint) with one of its 3 nearest training trajectories
(neighbor_rewards.py). The pairs are hand-picked from the paper's runs; edit `main` for other runs.

    pip install -r requirements.txt
    python build_trajectory_examples.py [--outputs $PRIME_RL_DIR/outputs] [--neighbors ../results/neighbor_rewards]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from trajectory_render import build, decode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from simlib import runs as R  # noqa: E402


def load(outputs, slug, harness, split, step, row_index, context, task, gold):
    run = R.run_id(slug, harness)
    path = R.run_dir(outputs, slug, harness) / "run_default" / "rollouts" / f"step_{step}" / f"{split}_rollouts.jsonl"
    digest, selected = hashlib.sha256(), None
    with path.open("rb") as f:
        for index, line in enumerate(f):
            digest.update(line)
            if index == row_index:
                selected = line
    row = json.loads(selected)
    assert not row.get("error") and not row.get("is_truncated")
    messages = row["completion"]
    turns = sum(m["role"] == "assistant" for m in messages)
    calls = [decode(c) for m in messages for c in (m.get("tool_calls") or [])]
    if calls:  # Pi
        agents = sum(c.get("function", c)["name"] == "subagent" for c in calls)
        shape = f"{turns} turns, {len(calls)} tool calls, {agents} sub-agent call{'s' * (agents != 1)}"
    else:  # RLM
        subcalls = int(row["metrics"]["rlm_sub_llm_calls"])
        shape = f"{turns} root turns, {subcalls} sub-call{'s' * (subcalls != 1)}"
    return dict(uid=f"{run}/{split}@{step}/{row_index}", run=run, split=split, archive_step=step,
                row_index=row_index, example_id=row["example_id"], reward=row["reward"],
                # missing for offline evals
                sampled_policy_step=row.get("sampling_args", {}).get("extra_body", {}).get("cache_salt"),
                metrics=row.get("metrics"), root_turns=turns, shape=shape,
                context_summary=context, task_summary=task, gold_display=gold,
                source=str(path.relative_to(outputs)), source_sha256=digest.hexdigest(),
                row_sha256=hashlib.sha256(selected).hexdigest(), task_info=decode(row["info"]),
                gold_answer_archived=row.get("answer"), initial_prompt=row.get("prompt"),
                root_completion=messages)


neighbor_rows, cells = [], {}
ORDINAL = {1: "nearest", 2: "second-nearest", 3: "third-nearest"}
cases = []
PAIRS_TASK = ("List all pairs of users where one has at least one 'abbreviation' and exactly one 'human being' "
              "question, and the other has at least one 'entity' and at least one 'numeric value' question.")
PI_OOLONG_TASK = "Is label 'numeric value' more or less common than 'description and abstract concept'?"


def case(key, title, harness, environment, metric, train, evaluation, observe, note=""):
    assert train["archive_step"] <= evaluation["archive_step"]
    (hit,) = [r for r in neighbor_rows if r["eval_uid"] == evaluation["uid"] and r["uid"] == train["uid"]]
    cell = cells[train["run"]]
    assert int(cell["eval_step"]) == evaluation["archive_step"]
    assert abs(float(hit["reward"]) - train["reward"]) < 1e-9
    assert abs(float(hit["eval_reward"]) - evaluation["reward"]) < 1e-9
    rank, similarity = int(hit["rank"]), float(hit["similarity"])
    text = (f"Weighted Jaccard {similarity:.2f}, the {ORDINAL[rank]} of {int(cell['n_train']):,} training "
            f"trajectories in this run (checkpoint at step {evaluation['archive_step']})."
            + (" " + note if note else ""))
    cases.append(dict(key=key, title=title, harness=harness, environment=environment, metric=metric,
                      train=train, eval=evaluation, observe=observe, similarity_note=text,
                      weighted_jaccard=similarity, neighbor_rank=rank, neighbor_pool=int(cell["n_train"]),
                      panel=hit["panel"], method=hit["method"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=R.DEFAULT_OUTPUTS, help="prime-rl outputs directory")
    ap.add_argument("--neighbors", type=Path, default=HERE.parent / "results" / "neighbor_rewards",
                    help="neighbor_rewards.py output directory (neighbors_k3.csv, paired_summary_k3.csv)")
    ap.add_argument("--tex", type=Path, default=HERE / "trajectory_examples.tex")
    ap.add_argument("--json", type=Path, default=HERE / "trajectory_examples.json")
    args = ap.parse_args()
    outputs = args.outputs
    neighbor_rows.extend(csv.DictReader((args.neighbors / "neighbors_k3.csv").open()))
    cells.update({r["run_id"]: r for r in csv.DictReader((args.neighbors / "paired_summary_k3.csv").open())})

    case("rlm-graphwalks", "RLM on GraphWalks: a programmatic task solved directly", "RLM",
         r"GraphWalks (110K $\to$ 437K chars)", "F1",
         load(outputs, "graphwalks", "rlm", "train", 84, 36, "Edge list, 110,213 characters.",
              "Find the parents of node 25b2822c2f.", "2 parent nodes."),
         load(outputs, "graphwalks", "rlm", "eval", 131, 23, "Edge list, 437,413 characters.",
              "Find the parents of node a96b65a721.", "10 parent nodes."),
         "The task is programmatic, so the root solves it directly: it prints the start and end of the "
         "context, extracts the answer with one line of string processing, and submits. Neither trajectory "
         "makes a sub-call. The evaluation context is 4x longer, but nothing in the root's program depends "
         "on its length.")

    case("rlm-oolong", "RLM on OOLONG: the same root program, with more sub-calls", "RLM",
         r"OOLONG (787 $\to$ 6{,}374 questions)", "Score",
         load(outputs, "oolong", "rlm", "train", 60, 46, "787 questions, 78,138 characters.",
              "Is label 'location' more or less common than 'human being'?", "less common than."),
         load(outputs, "oolong", "rlm", "eval", 101, 16, "6,374 questions, 618,252 characters.",
              "Is label 'human being' more or less common than 'entity'?", "more common than."),
         "The root writes the same program in both trajectories: split the context into questions, send "
         "them to sub-calls in batches of 20 for labeling, count the returned labels, and compare. The 8x "
         "longer evaluation context changes how many sub-calls run (41 vs. 321), not the program. The "
         "per-question work happens inside the sub-calls. The root sees only batch counts and label "
         "totals, so its own trajectory stays close to the training one.",
         "The nearest (0.60) runs the same program with batches of 30 and one failed cell.")

    case("pi-graphwalks", "Pi on GraphWalks: a programmatic task solved with shell commands", "Pi",
         r"GraphWalks (110K $\to$ 1.7M chars)", "F1",
         load(outputs, "graphwalks", "pi", "train", 140, 22, "context.txt, 4,392 lines, 110,363 bytes.",
              "Return the nodes at exactly depth 4 in a BFS from node 898aef0932.", "1 node (a385d7d1e5)."),
         load(outputs, "graphwalks", "pi", "eval", 150, 13, "context.txt, 69,912 lines, 1,748,363 bytes.",
              "Return the nodes at exactly depth 4 in a BFS from node e57c68e8f2.", "1 node (a0604c9d9e)."),
         "Pi also solves the programmatic task directly, here with shell commands: find the operation "
         "line, then run one grep per BFS hop. The sequence of commands is the same on a 16x longer file. "
         "Only the node IDs and line numbers differ. The whole-file read in turn 3 is cut off by the tool "
         "and is not needed for the answer. No sub-agent is called.")

    case("pi-oolong-pairs", "Pi on OOLONG-Pairs: failure in training and in evaluation", "Pi",
         r"OOLONG-Pairs (188 $\to$ 787 questions)", "F1",
         load(outputs, "oolong-pairs", "pi", "train", 122, 24, "context.txt, 188 questions.",
              PAIRS_TASK,
              "75 pairs (4 predicted)."),
         load(outputs, "oolong-pairs", "pi", "eval", 150, 10, "context.txt, 787 questions.",
              PAIRS_TASK, "980 pairs (2 predicted)."),
         "Both trajectories fail in the same way. Pi reads the file into its own context, asks a single "
         "reader sub-agent to label every question, receives a partial list (67 of 188 labels in training, "
         "103 of 787 in evaluation), and computes a few pairs from it. The evaluation trajectory is close "
         "to a training trajectory that itself scored 0.03: there is no successful behavior to reuse.")

    case("pi-oolong", "Pi on OOLONG: a copied training trajectory that does not generalize", "Pi",
         r"OOLONG (787 $\to$ 6{,}374 questions)", "Score",
         load(outputs, "oolong", "pi", "train", 98, 17, "context.txt, 795 lines, 76,803 bytes.",
              PI_OOLONG_TASK, "more common than."),
         load(outputs, "oolong", "pi", "eval", 102, 24, "context.txt, 6,382 lines, 618,252 bytes.",
              PI_OOLONG_TASK, "less common than."),
         "The evaluation trajectory is nearly a copy of a rewarded training trajectory. Pi sends the same "
         "two requests to a reader sub-agent and gets back the same two counts (12 and 1), although the "
         "evaluation file is 8x longer. It then gives the same answer, which was correct in training and "
         "is wrong in evaluation. Here high similarity to training goes with failure.")

    case("rlm-obliq", "RLM on OBLIQ: the same strategy on a different domain", "RLM",
         r"OBLIQ (Twitter $\to$ WildChat)", "nDCG@50",
         load(outputs, "obliq_twitter_to_wildchat", "rlm", "train", 123, 52, "300 tweets, 77,145 characters.",
              "Rank tweets that implicitly accuse a Gulf monarchy of duplicitous hedging.",
              "3 relevant tweets."),
         load(outputs, "obliq_twitter_to_wildchat", "rlm", "eval", 150, 5,
              "300 Human-AI conversations, 572,017 characters.",
              "Rank conversations where the AI asserts a role-played identity as its real one.",
              "10 relevant conversations."),
         "Training documents are tweets and the query concerns a geopolitical stance. Evaluation documents "
         "are Human-AI conversations and the query concerns an AI failure mode. The root program is the "
         "same: extract the document IDs, ask one sub-call per document for a 1-10 relevance score, sort, "
         "and return the ranking. The domain enters only through the query and documents passed to the "
         "sub-calls, where the judgments are made.")

    build(cases, args.tex, args.json, dict(
        rule="Evaluation trajectory at the similarity-figure checkpoint, paired with one of its three nearest "
             "training trajectories of the same run (archive train step <= eval step, failures included) "
             "under weighted Jaccard on the full root trajectory.",
        neighbors="neighbors_k3.csv",
        neighbors_sha256=hashlib.sha256((args.neighbors / "neighbors_k3.csv").read_bytes()).hexdigest(),
        gold_note="Gold sizes for the Pi OOLONG-Pairs example are read from the archived answers of the same "
                  "questions in run length-gen/rlm_orchestrator_prompt/oolong-pairs (train@100/56, eval@111/10); gold for "
                  "the Pi GraphWalks example is the submitted answer, which has F1 = 1."))


if __name__ == "__main__":
    main()
