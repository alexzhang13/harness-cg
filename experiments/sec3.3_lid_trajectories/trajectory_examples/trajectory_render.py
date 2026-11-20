"""Side-by-side LaTeX rendering of train/eval root trajectories (appendix examples).

Similarities use the original trajectories; alignment and shortening only affect the display.
"""
import ast
from collections import Counter
import difflib
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import sys
import unicodedata

import black
from pygments import lex
from pygments.lexers import BashLexer, PythonLexer, TextLexer
from pygments.token import Token
from rapidfuzz.distance import Levenshtein

WIDTH = 59  # monospace characters per column
# longer blocks keep head, tail and sub-call lines
CAPS = {"prose": (3, 1), "python": (18, 4), "call": (3, 1), "output": (3, 1)}
MATCH = 0.3  # minimum action similarity for two turns to be placed side by side
REPL = re.compile(r"```repl[ \t]*\n(.*?)```", re.S)
SUBCALL = re.compile(r"\b(?:llm|rlm)_query(?:_batched)?\s*\(")
NUDGE = re.compile(r"Turn \d+/\d+:")
PROGRESS = re.compile(r"\d+%\|.*\|\s*\d+/\d+\s*\[")
ASCII = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": "--", "–": "-", "‑": "-",
         "…": "...", "→": "->", "≈": "~", "≥": ">=", "≤": "<=", "×": "x", "\u00a0": " "}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def decode(value):
    if not isinstance(value, str):
        return value
    for candidate in (value, value.replace('\\"', '"')):
        try:
            return json.loads(candidate)
        except ValueError:
            pass
    raise ValueError("Cannot decode archived field")


def tex(text, mono=False):
    escapes = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "#": r"\#", "$": r"\$",
               "%": r"\%", "&": r"\&", "_": r"\_", "~": r"\textasciitilde{}",
               "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}"}
    if mono:
        escapes.update({" ": r"\ ", "`": r"\char96{}", '"': r"\char34{}", "'": r"\char39{}",
                        "-": "-{}", ",": ",{}"})
    return "".join(escapes.get(c, c) for c in str(text))


def plain(text, replaced):
    """ASCII text; untransliterable characters become '?' and are counted."""
    text = "".join(ASCII.get(c, c) for c in text.replace("\r\n", "\n").replace("\r", "\n"))
    out = []
    for char in text:
        if 32 <= ord(char) < 127 or char in "\n\t":
            out.append(char)
        elif unicodedata.category(char) not in ("Cf", "Mn"):
            base = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode()
            out.append(base or "?")
            replaced[char] += not base
    return "".join(out)


def tidy_python(raw):
    """Black-format a REPL cell if that leaves its AST unchanged."""
    try:
        tree = ast.parse(raw)
        text = black.format_str(raw, mode=black.Mode(line_length=WIDTH, string_normalization=False))
        strings = [n for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.Constant, ast.JoinedStr))]
        inside = {i for n in strings for i in range(n.lineno, n.end_lineno + 1)}
        lines = [line for i, line in enumerate(text.splitlines(), 1) if line.strip() or i in inside]
        for candidate in ("\n".join(lines), text):
            if ast.dump(ast.parse(candidate)) == ast.dump(tree):
                return candidate
    except (SyntaxError, black.InvalidInput):
        pass
    return raw.strip("\n")


def call_block(name, args):
    if name == "bash" and list(args) == ["command"]:
        return dict(kind="call", text=args["command"], language="bash", prefix="$ ", subcall=False)
    shown = {k: v for k, v in args.items() if k != "cwd"}
    text = name + "(" + ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in shown.items()) + ")"
    return dict(kind="call", text=text, language="python", prefix="", subcall=name == "subagent")


def quoted(text):
    """Header text with typographic quotes."""
    return re.sub(r"'([^']*)'", r"`\1'", tex(text))


def split_record(record):
    """Messages grouped into turns of display blocks."""
    messages = record["root_completion"]
    results = {m["tool_call_id"]: m["content"] or "" for m in messages if m["role"] == "tool"}
    turns, used, nudges = [], set(), 0
    for message in messages:
        role, content = message["role"], message.get("content") or ""
        if role == "assistant":
            blocks, cursor = [], 0
            for match in REPL.finditer(content):
                blocks.append(dict(kind="prose", text=content[cursor:match.start()]))
                blocks.append(dict(kind="python", text=tidy_python(match.group(1)), language="python"))
                cursor = match.end()
            blocks.append(dict(kind="prose", text=content[cursor:]))
            blocks = [b for b in blocks if b["text"].strip()]
            for raw in message.get("tool_calls") or []:
                call = decode(raw)
                fn = call.get("function", call)
                blocks.extend((call_block(fn["name"], decode(fn["arguments"])),
                               dict(kind="output", text=results[call["id"]])))
                used.add(call["id"])
            turns.append(blocks)
        elif role == "user" and NUDGE.fullmatch(content.strip()):
            nudges += 1
        elif role == "user":
            turns[-1].append(dict(kind="output", text=content))
        else:
            assert role == "tool" and message["tool_call_id"] in used
    assert used == set(results) and len(turns) == record["root_turns"]
    for turn in turns:
        for b in turn:
            b.update(language=b.get("language", "text"), prefix=b.get("prefix", ""),
                     subcall=b.get("subcall", False))
    record["hidden_turn_nudges"] = nudges
    return turns


def token_color(token):
    for parent, color in ((Token.Comment, "trcomment"), (Token.Keyword, "trkeyword"),
                          (Token.Literal.String, "trstring"), (Token.Literal.Number, "trnumber"),
                          (Token.Name.Builtin, "trbuiltin"), (Token.Name.Function, "trbuiltin")):
        if token in parent:
            return color
    return "trink"


def highlighted_lines(text, language):
    lexer = {"python": PythonLexer, "bash": BashLexer}.get(language, TextLexer)(stripnl=False, ensurenl=False)
    chars = [(char, token_color(token)) for token, value in lex(text, lexer) for char in value]
    assert "".join(c for c, _ in chars) == text
    lines = [[]]
    for char, color in chars:
        if char == "\n":
            lines.append([])
        else:
            lines[-1].extend([(" ", color)] * 4 if char == "\t" else [(char, color)])
    return lines


def wrap(line):
    """Wrap a line to the column width."""
    rows, rest = [], line[:]
    while len(rest) > WIDTH:
        cut = max((i + 1 for i, (c, _) in enumerate(rest[:WIDTH]) if c == " "), default=WIDTH)
        if cut < WIDTH // 2:
            cut = WIDTH
        rows.append(rest[:cut])
        rest = rest[cut:]
    rows.append(rest)
    assert [ch for row in rows for ch in row] == line
    return rows


def clean_output(text):
    """Strip REPL framing, blank lines and progress bars."""
    text = re.split(r"\n?REPL variables:", re.sub(r"^REPL output:\s*", "", text), maxsplit=1)[0]
    lines = []
    for line in text.splitlines():
        if PROGRESS.search(line):
            line = "[progress bar omitted]"
        if line.strip() and not (lines and line == lines[-1] == "[progress bar omitted]"):
            lines.append(line.rstrip())
    return "\n".join(lines) or "[no output]"


def rows_of(b, replaced):
    """(text, is_continuation, background) rows of a block."""
    if b is None:
        return []
    kind, text = b["kind"], plain(b["text"], replaced)
    if kind == "prose":
        text = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    elif kind == "output":
        text = clean_output(text)
    base = {"prose": "white", "output": "trout"}.get(kind, "trcode")
    rows = []
    for i, line in enumerate(highlighted_lines(text, b["language"])):
        source = "".join(c for c, _ in line)
        background = "trsub" if b["subcall"] or (kind == "python" and SUBCALL.search(source)) else base
        if kind == "output":
            line = [(c, "trmuted") for c, _ in line]
        if i == 0:
            line = [(c, "trmuted") for c in b["prefix"]] + line
        rows.extend((part, j > 0 and kind != "prose", background) for j, part in enumerate(wrap(line)))
    head, tail = CAPS[kind]
    b["display_lines"], b["omitted_lines"] = len(rows), 0
    if len(rows) > head + tail + 1:
        keep = [i < head or i >= len(rows) - tail or row[2] == "trsub" for i, row in enumerate(rows)]
        for i in range(1, len(rows) - 1):
            keep[i] = keep[i] or (keep[i - 1] and keep[i + 1])
        shown, i = [], 0
        while i < len(rows):
            if keep[i]:
                shown.append(rows[i])
                i += 1
                continue
            j = next(k for k in range(i, len(rows)) if keep[k])
            shown.append(([(c, "trmuted") for c in f"[... {j - i} lines omitted ...]"], False, base))
            b["omitted_lines"] += j - i
            i = j
        rows = shown
    return rows


def align_turns(left, right, tokenize):
    """Monotone turn pairing maximizing summed similarity above MATCH."""
    def counts(turn):
        return Counter(tokenize("\n".join(b["text"] for b in turn if b["kind"] != "output")))
    a, b = [counts(t) for t in left], [counts(t) for t in right]
    gain = [[sum((x & y).values()) / max(1, sum((x | y).values())) - MATCH for y in b] for x in a]
    best = [[0.0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            best[i][j] = max(best[i + 1][j + 1] + gain[i][j], best[i + 1][j], best[i][j + 1])
    pairs, i, j = [], 0, 0
    while i < len(a) or j < len(b):
        if i < len(a) and j < len(b) and best[i][j] == best[i + 1][j + 1] + gain[i][j]:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif j == len(b) or (i < len(a) and best[i][j] == best[i + 1][j]):
            pairs.append((i, None))
            i += 1
        else:
            pairs.append((None, j))
            j += 1
    merged = []
    for pair in pairs:
        if merged and merged[-1][1] is None and pair[0] is None:
            merged[-1] = (merged[-1][0], pair[1])
        elif merged and merged[-1][0] is None and pair[1] is None:
            merged[-1] = (pair[0], merged[-1][1])
        else:
            merged.append(pair)
    assert [i for i, _ in merged if i is not None] == list(range(len(a)))
    assert [j for _, j in merged if j is not None] == list(range(len(b)))
    return merged


def cell(row):
    if row is None:
        return "white", ""
    chars, continuation, background = row
    groups = []
    for char, color in chars:
        if groups and groups[-1][0] == color:
            groups[-1][1] += char
        else:
            groups.append([color, char])
    return background, (r"\trmore " if continuation else "") + "".join(
        r"\textcolor{" + color + "}{" + tex(value, mono=True) + "}" for color, value in groups)


def render_turns(case, tokenize):
    train, evaluation = (split_record(case[s]) for s in ("train", "eval"))
    pairs = align_turns(train, evaluation, tokenize)
    case["turn_alignment"] = [[None if i is None else i + 1, None if j is None else j + 1] for i, j in pairs]
    replaced = case["characters_shown_as_question_mark"] = Counter()
    lines = [r"\begingroup\offinterlineskip\vbadness=10000"]
    for i, j in pairs:
        left, right = ([] if i is None else train[i]), ([] if j is None else evaluation[j])
        heads = [r"\trnoturn" if k is None else rf"\trturn{{{color}}}{{{name} \textperiodcentered{{}} turn {k + 1}}}"
                 for k, color, name in ((i, "trtrain", "TRAIN"), (j, "treval", "EVAL"))]
        lines.append(r"\trhead{" + heads[0] + "}{" + heads[1] + "}")
        kinds = [[b["kind"] for b in side] for side in (left, right)]
        rows = []
        for _, i1, i2, j1, j2 in difflib.SequenceMatcher(None, *kinds, autojunk=False).get_opcodes():
            for x, y in zip_longest(left[i1:i2], right[j1:j2]):
                rows.extend(zip_longest(rows_of(x, replaced), rows_of(y, replaced)))
        for n, (x, y) in enumerate(rows):
            (lc, lt), (rc, rt) = cell(x), cell(y)
            lines.append(r"\trrow{" + lc + "}{" + lt + "}{" + rc + "}{" + rt + "}" 
                         + (r"\nobreak" if n == 0 or len(rows) - 3 <= n < len(rows) - 1 else ""))
    lines.append(r"\endgroup")
    for split, turns in (("train", train), ("eval", evaluation)):
        case[split]["display_blocks"] = [[{k: b[k] for k in ("kind", "subcall", "display_lines", "omitted_lines")}
                                          for b in turn] for turn in turns]
    return "\n".join(lines)


STYLES = r"""% Generated by appendix/build_trajectory_examples.py from archived root completions.
% Edit the generator, not this file.
\begingroup
\definecolor{trtrain}{RGB}{24,101,87}
\definecolor{treval}{RGB}{43,86,139}
\definecolor{trfail}{RGB}{159,54,48}
\definecolor{trink}{RGB}{38,46,54}
\definecolor{trkeyword}{RGB}{115,60,155}
\definecolor{trstring}{RGB}{30,105,79}
\definecolor{trnumber}{RGB}{160,76,33}
\definecolor{trbuiltin}{RGB}{38,83,145}
\definecolor{trcomment}{RGB}{112,121,130}
\definecolor{trmuted}{RGB}{96,102,108}
\definecolor{trcode}{RGB}{240,244,249}
\definecolor{trout}{RGB}{226,228,231}
\definecolor{trsub}{RGB}{255,238,190}
\newcommand{\trcodefont}{\fontencoding{T1}\fontfamily{lmtt}\fontsize{6}{7}\selectfont}
\newcommand{\trsans}{\fontencoding{T1}\fontfamily{lmss}\fontsize{7}{8.5}\selectfont}
\newcommand{\trcell}[2]{\colorbox{#1}{\makebox[.492\linewidth][l]{\hspace{7pt}\rule[-1.6pt]{0pt}{7pt}#2}}}
\newcommand{\trrow}[4]{\hbox to\linewidth{\trcodefont\trcell{#1}{#2}\hfill\trcell{#3}{#4}}}
\newcommand{\trturn}[2]{\colorbox{#1!14}{\makebox[.492\linewidth][l]{\hspace{7pt}\rule[-2.4pt]{0pt}{10pt}\trsans\bfseries\color{#1}#2}}}
\newcommand{\trnoturn}{\makebox[.492\linewidth]{}}
\newcommand{\trhead}[2]{\vskip 3pt\hbox to\linewidth{#1\hfill#2}\nobreak}
\newcommand{\trmore}{\makebox[0pt][r]{\color{black!35}\tiny$\hookrightarrow$\hspace{1pt}}}
\newcommand{\trkey}[2]{{\setlength{\fboxsep}{0pt}\setlength{\fboxrule}{.3pt}\fcolorbox{black!45}{#1}{\rule[-1.5pt]{0pt}{7pt}\hspace{9pt}}}~#2}
\ifcsname c@trexample\endcsname\else\newcounter{trexample}\fi
\setcounter{trexample}{0}
\setlength{\fboxsep}{0pt}
\setlength{\parindent}{0pt}
"""


def score(value):
    return f"{value:.3f}" if 0 < value < 0.01 else f"{value:.2f}"


def overview(cases):
    lines = [r"\begin{center}\small\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.15}",
             r"\begin{tabular}{@{}cllccc@{}}\toprule",
             r"\textbf{Ex.} & \textbf{Harness} & \textbf{Environment (train $\to$ eval)} & "
             r"\textbf{Train} & \textbf{Eval} & \textbf{Similarity} \\\midrule"]
    for n, case in enumerate(cases, 1):
        lines.append(" & ".join([rf"\ref{{ex:{case['key']}}}", tex(case["harness"]), case["environment"],
                                 score(case["train"]["reward"]), score(case["eval"]["reward"]),
                                 f"{case['weighted_jaccard']:.2f}"]) + r" \\")
    lines.extend([r"\bottomrule\end{tabular}\end{center}",
                  r"\begin{center}\small\trkey{white}{model text}\quad\trkey{trcode}{code or tool call}\quad"
                  r"\trkey{trsub}{sub-call request}\quad\trkey{trout}{returned output}\end{center}"])
    return "\n".join(lines)


def heading(case):
    lines = [r"\clearpage", rf"\refstepcounter{{trexample}}\label{{ex:{case['key']}}}",
             r"{\normalsize\bfseries Example \thetrexample. " + tex(case["title"]) + r"\par}\vspace{5pt}"]
    shared = case["train"]["task_summary"] == case["eval"]["task_summary"]
    for split, name, color in (("train", "TRAIN", "trtrain"), ("eval", "EVAL", "treval")):
        record = case[split]
        fail = record["reward"] < 0.1
        lines.extend([
            r"\begin{minipage}[t]{.492\linewidth}\footnotesize\raggedright\setlength{\parskip}{1.5pt}",
            rf"{{\trsans\bfseries\color{{{color}}}{name}}}\hfill{{\trsans\bfseries\color{{{'trfail' if fail else color}}}"
            + tex(case["metric"]) + " " + score(record["reward"]) + r"}\par",
            r"\textbf{Context.} " + quoted(record["context_summary"]) + r"\par",
            "" if shared else r"\textbf{Task.} " + quoted(record["task_summary"]) + r"\par",
            r"\textbf{Gold.} " + quoted(record["gold_display"]) + r"\par",
            r"\textbf{Trajectory.} " + tex(record["shape"]) + r"\par",
            r"\end{minipage}" + (r"\hfill" if split == "train" else r"\par")])
    if shared:
        lines.append(r"\vspace{1.5pt}{\footnotesize\textbf{Task (both).} " + quoted(case["train"]["task_summary"]) + r"\par}")
    lines.extend([r"\vspace{5pt}{\footnotesize\textbf{Similarity.} " + quoted(case["similarity_note"]) + r"\par}",
                  r"\vspace{4pt}{\small\color{red}\textbf{What we observe.} " + quoted(case["observe"]) + r"\par}",
                  r"\vspace{3pt}\nobreak"])
    return "\n".join(lines)


def add_scores(case, tokenize, word_ngrams, join):
    """Appendix metrics on the original text, full trajectory and actions only."""
    case["similarity"] = {}
    for view, roles in (("full_root", ("assistant", "user", "tool")), ("actions_only", ("assistant",))):
        texts = [join(case[s]["root_completion"], include_roles=roles) for s in ("train", "eval")]
        a, b = map(tokenize, texts)
        ga, gb = (set(word_ngrams(t, 3)) for t in (a, b))
        ca, cb = Counter(a), Counter(b)
        case["similarity"][view] = dict(
            edit=Levenshtein.normalized_similarity(a, b), contain=len(ga & gb) / len(gb),
            jaccard=len(ga & gb) / len(ga | gb),
            weighted_jaccard=sum((ca & cb).values()) / sum((ca | cb).values()),
            length=min(len(a), len(b)) / max(len(a), len(b)), train_tokens=len(a), eval_tokens=len(b),
            train_text_sha256=sha(texts[0]), eval_text_sha256=sha(texts[1]))


def build(cases, tex_path, json_path, provenance):
    simlib_parent = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(simlib_parent))
    from simlib.metrics import tokenize, word_ngrams
    from simlib.trajectories import _join_completion
    body = []
    for case in cases:
        add_scores(case, tokenize, word_ngrams, _join_completion)
        # must match the nearest-neighbour analysis
        assert abs(case["similarity"]["full_root"]["weighted_jaccard"] - case["weighted_jaccard"]) < 1e-9
        body.extend((heading(case), render_turns(case, tokenize)))
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join([STYLES, overview(cases), *body, r"\endgroup"]) + "\n")
    payload = dict(
        selection=provenance,
        display="Left: training; right: evaluation. Turns are paired by a monotone alignment on action-token "
                f"weighted Jaccard (pairs below {MATCH} are not matched); blocks within a turn are paired by kind. "
                "Each tool result is shown under its call. Not shown: initial prompts, the long context, "
                "'Turn k/N:' harness messages, REPL variable lists, blank lines, progress bars and the sandbox "
                "cwd argument. Python cells are Black-formatted only if the AST is unchanged. Non-ASCII "
                "punctuation is transliterated. Blocks longer than head+tail+1 display lines keep "
                f"{CAPS} (head, tail) lines; omission counts are printed and stored per block. "
                "Amber marks llm_query/rlm_query lines and subagent calls.",
        scores="Computed on the original root completion (full_root) and on assistant messages only "
               "(actions_only) with the analysis tokenizer and serializer; word/code tokens, not BPE.",
        tokenizer_pattern=r"[A-Za-z_]\w+|\d+|[^\sA-Za-z0-9_]",
        metric_source_sha256={name: hashlib.sha256((simlib_parent / "simlib" / name).read_bytes()).hexdigest()
                              for name in ("metrics.py", "trajectories.py")},
        cases=cases)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    for case in cases:
        shown = {s: sum(b["display_lines"] - b["omitted_lines"] + bool(b["omitted_lines"])
                        for t in case[s]["display_blocks"] for b in t) for s in ("train", "eval")}
        print(case["key"], "turns", case["turn_alignment"], "rows", shown,
              {k: round(v, 3) for k, v in case["similarity"]["full_root"].items() if isinstance(v, float)})
