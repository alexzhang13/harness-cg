"""Generated tasks for the gated SFT corpus, with a scripted sub-call reader. No eval-suite templates or data are used.
"""
import json, re

SYL = ["ka", "lo", "mi", "re", "su", "ta", "vo", "ne", "pi", "do", "fu", "ga", "he", "jo", "ru", "zi", "ba", "xe", "yo", "wa"]
def word(rng, k=None): return "".join(rng.choice(SYL) for _ in range(k or rng.choice([2, 3]))).capitalize()
def code(rng, p="ID"): return f"{p}-{rng.randint(100, 9999)}"
COLORS = ["teal", "amber", "slate", "ivory", "coral", "olive", "indigo", "rust"]
UNITS = ["kg", "mm", "ppm", "volts", "units", "liters", "minutes"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_FULL = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

FACTS = {}

FORMATS = {
    "number": ["Reply with only the number.", "Give the final value as a plain number, no units.", "Final answer format: ANSWER=<number>", "Answer with the number alone (two decimals if it is not a whole number)."],
    "name": ["Reply with just the name.", "Final answer format: ANSWER=<name>", "Answer with the name exactly as it appears in the data, nothing else."],
    "list": ["Answer with a JSON array of strings in alphabetical order.", "Give a comma-separated list in alphabetical order and nothing else.", "Final answer format: ANSWER=[...] (a JSON list, alphabetical)."],
    "letter": ["Respond with the option letter only.", "Final answer format: ANSWER=<letter>", "Answer with a single letter."],
}
INTROS = ["You are given some data in the variable `context`.", "The material for this task is stored in `context`.", "Work from the data provided in `context`.", "Task data is available as `context`.", "Use `context` as your only source."]

def frame(rng, body, kind):
    return f"{rng.choice(INTROS)} {body}\n\n{rng.choice(FORMATS[kind])}"

def f_ledger(rng):
    item, loc = word(rng) + " " + rng.choice(["wire", "plate", "resin", "bolt", "tape"]), word(rng)
    fmt = rng.choice(["{id} | site: {loc} | item: {item} | qty: {q} | unit_cost: {c:.2f}", "{id}; {item}; {loc}; {q} pcs @ {c:.2f}", "record {id}: {q} x {item} at {loc}, cost each {c:.2f}"])
    trap = rng.random() < .6
    recs, total = [], 0.0
    for _ in range(rng.randint(40, 260)):
        it = item if rng.random() < .3 else word(rng) + " " + rng.choice(["wire", "plate", "resin", "bolt", "tape"])
        lc = loc if rng.random() < .5 else word(rng)
        q, c = rng.randint(1, 90), round(rng.uniform(.5, 40), 2)
        line = fmt.format(id=code(rng, "TX"), loc=lc, item=it, q=q, c=c); sign = 1
        if trap:
            r = rng.random()
            if r < .12: line += "  [VOID]"; sign = 0
            elif r < .22: line = "RETURN " + line; sign = -1
        if it == item and lc == loc: total += sign * q * c
        recs.append(line)
    if abs(total) < 1: return f_ledger(rng)
    rule = " Records marked [VOID] are cancelled and must be ignored; records starting with RETURN subtract their value." if trap and rng.random() < .4 else ""
    qs = [f"What is the total value (quantity times unit cost) of all records for '{item}' at site {loc}?{rule}", f"Add up quantity x cost for every record of {item} held at {loc}.{rule} What is the sum?"]
    return dict(family="ledger", kind="number", question=frame(rng, rng.choice(qs), "number"), context=recs if rng.random() < .5 else "\n".join(recs), gold=f"{total:.2f}", answer_kind="number")

def f_chain(rng):
    n = rng.randint(30, 120); people = list({word(rng, 3) for _ in range(n * 2)})[:n]
    boss = {}; lines = []
    for i in range(1, n):
        if rng.random() < .9: boss[people[i]] = people[rng.randrange(i)]
    tmpl = ["{a} reports to {b}.", "{a}'s manager is {b}.", "In the org chart, {a} sits directly under {b}."]
    trap = rng.random() < .6
    for a, b in list(boss.items()):
        if trap and rng.random() < .25:
            old_b = rng.choice([p for p in people if p != a and p != b]); lines.append(rng.choice(tmpl).format(a=a, b=old_b))
        lines.append(rng.choice(tmpl).format(a=a, b=b))
    noise = [f"{p} joined in {rng.randint(1995, 2024)}." for p in people if rng.random() < .6]
    for x in noise: lines.insert(rng.randrange(len(lines) + 1), x)
    def root(p):
        seen = 0
        while p in boss: p = boss[p]; seen += 1
        return p
    cand = [p for p in people if p in boss]; start = rng.choice(cand)
    rule = " Statements appear in chronological order; if a person has more than one manager listed, only the latest statement counts." if trap else ""
    q = rng.choice([f"Follow the chain of managers upward starting from {start}. Who is at the very top?{rule}", f"Who is the topmost manager above {start}?{rule}"])
    return dict(family="chain", kind="name", question=frame(rng, q, "name"), context="\n".join(lines), gold=root(start), answer_kind="name")

def f_readings(rng):
    """Prose measurements; with the trap some are in grams while the question asks for kilograms."""
    sensors = [word(rng, 2) + "-" + str(rng.randint(1, 9)) for _ in range(rng.randint(6, 14))]
    trap = rng.random() < .6
    tmpl = ["On day {d}, the {s} unit logged {v} {u}.", "{s} measured {v} {u} during shift {d}.", "Sensor {s} reported a reading of {v} {u} on day {d}."]
    best, bestv, paras = None, -1.0, []
    for _ in range(rng.randint(30, 180)):
        s_, kg, d = rng.choice(sensors), round(rng.uniform(1, 500), 1), rng.randint(1, 90)
        as_g = trap and rng.random() < .3
        v, u = (round(kg * 1000, 1), "g") if as_g else (kg, "kg")
        sent = rng.choice(tmpl).format(s=s_, v=v, u=u, d=d); FACTS[sent] = (s_, v, u)
        if kg > bestv: best, bestv = s_, kg
        paras.append(sent + " " + rng.choice(["The area was otherwise quiet.", "Weather held steady.", "Maintenance was not required.", "Operators noted nothing unusual."]))
    ties = sorted((rng.random(), 0) for _ in range(0))
    vals = []
    for p in paras:
        m = re.search(r"([\d.]+) (kg|g)\b", p); x = float(m.group(1)) / (1000.0 if m.group(2) == "g" else 1.0); vals.append(x)
    if sorted(vals)[-1] - sorted(vals)[-2] < 0.05: return f_readings(rng)
    rule = " Some readings are logged in grams (g) and others in kilograms (kg); compare them in the same unit." if trap and rng.random() < .4 else ""
    q = rng.choice([f"Which sensor recorded the single highest reading?{rule}", f"Name the instrument with the largest value logged anywhere in the log.{rule}"])
    ctx = "\n\n".join(paras)
    return dict(family="readings", kind="name", question=frame(rng, q, "name"), context=ctx if rng.random() < .6 else paras, gold=best, answer_kind="name")

def f_timeline(rng):
    ev = [(word(rng, 3) + " " + rng.choice(["launch", "audit", "outage", "review", "merger"]), rng.randint(0, 3000)) for _ in range(rng.randint(25, 120))]
    base = 18000
    def date(dn):
        y, r = 2000 + dn // 360, dn % 360; m, d = r // 30, r % 30 + 1
        return y, m, d
    def show(dn):
        y, m, d = date(dn)
        return rng.choice([f"{y}-{m + 1:02d}-{d:02d}", f"{d} {MONTHS[m]} {y}", f"{MONTHS_FULL[m]} {d}, {y}"])
    lines = [f"{show(dn)}: {name}." for name, dn in ev]; rng.shuffle(lines)
    lo, hi = sorted(rng.sample(range(0, 3000), 2))
    cnt = sum(1 for _, dn in ev if lo < dn < hi)
    def mk(dn): y, m, d = date(dn); return f"{y}-{m + 1:02d}-{d:02d}"
    q = f"How many events happened strictly after {mk(lo)} and strictly before {mk(hi)}? (All dates in the list are valid; months have 30 days in this calendar.)"
    return dict(family="timeline", kind="number", question=frame(rng, q, "number"), context=lines, gold=str(cnt), answer_kind="number")

def f_count(rng):
    ks = rng.randint(60, 400); col, thr = rng.choice(COLORS), rng.randint(20, 80); lines, cnt = [], 0
    trap = rng.random() < .6
    for _ in range(ks):
        c, sz, st = rng.choice(COLORS), rng.randint(1, 100), rng.choice(["ok", "flagged", "retired"])
        shown = st
        if trap: shown = rng.choice([st, st.upper(), st.capitalize(), " " + st])
        if c == col and sz > thr and st != "retired": cnt += 1
        lines.append(f"item={code(rng, 'I')}; color={c}; size={sz}; status={shown}")
    rule = " Status values are not spelled consistently (case, spaces); treat them case-insensitively and ignore surrounding spaces." if trap and rng.random() < .4 else ""
    q = rng.choice([f"Count the items whose color is {col}, whose size is greater than {thr}, and whose status is not 'retired'.{rule}", f"How many records have color={col}, size>{thr} and are not retired?{rule}"])
    return dict(family="count", kind="number", question=frame(rng, q, "number"), context="\n".join(lines), gold=str(cnt), answer_kind="number")

def f_diff(rng):
    pool = list({code(rng, "R") for _ in range(rng.randint(80, 300))}); k = len(pool)
    a = rng.sample(pool, int(k * .7)); b = rng.sample(pool, int(k * .6))
    trap = rng.random() < .6
    b_shown = [x.lower() if trap and rng.random() < .4 else x for x in b]
    only = sorted(set(a) - set(b))
    if not only: return f_diff(rng)
    show = lambda L: "\n".join(f"{x} {rng.choice(['active', 'pending', 'seen'])}" for x in L)
    rule = " Ids should be compared ignoring upper/lower case; report them as written in the first list." if trap and rng.random() < .4 else ""
    q = f"The context has two lists (first, second). Which ids appear in the first list but not in the second?{rule}"
    return dict(family="diff", kind="list", question=frame(rng, q, "list"), context=[show(a), show(b_shown)], gold=json.dumps(only), answer_kind="list")

def f_pick(rng):
    ppl = rng.sample(["Ana", "Bo", "Cy", "Dee", "Eli", "Fay", "Gus", "Hal"], 4); order = ppl[:]; rng.shuffle(order)
    facts = [f"{order[i]} owns more books than {order[i + 1]}." for i in range(3)]; rng.shuffle(facts)
    fill = " ".join(f"{word(rng)} visited the {rng.choice(['market', 'harbor', 'library'])} on day {rng.randint(1, 30)}." for _ in range(rng.randint(60, 600)))
    pos = rng.randint(0, len(fill)); text = fill[:pos] + " " + " ".join(facts) + " " + fill[pos:]
    opts = ppl[:]; rng.shuffle(opts); gold = "ABCD"[opts.index(order[3])]
    q = "Who owns the fewest books?\n" + "\n".join(f"{'ABCD'[i]}) {o}" for i, o in enumerate(opts))
    return dict(family="pick", kind="letter", question=frame(rng, q, "letter"), context=text, gold=gold, answer_kind="letter")

FAMILIES = {"ledger": f_ledger, "chain": f_chain, "readings": f_readings, "timeline": f_timeline, "count": f_count, "diff": f_diff, "pick": f_pick}

FAMILY_WEIGHTS = {"ledger": 3.0, "readings": 3.0, "timeline": 1.5, "chain": 1.0, "count": 1.0, "diff": 1.0, "pick": 1.0}

def make_item(rng, family=None):
    if family is None: family = rng.choices(list(FAMILIES), weights=[FAMILY_WEIGHTS[f] for f in FAMILIES])[0]
    return FAMILIES[family](rng)

def _clean(a):
    a = (a or "").strip()
    a = re.sub(r"(?is)^.*?\bANSWER\s*=\s*", "", a) if re.search(r"ANSWER\s*=", a) else re.sub(r"(?is)^.*?\bthe (?:final )?answer is[:\s]*", "", a)
    return a.strip().strip("`'\" .")

def score(item, answer):
    a, g, k = _clean(answer), item["gold"], item["answer_kind"]
    try:
        if k == "number": return float(abs(float(a.replace(",", "")) - float(g)) <= 0.011)
        if k == "name": return float(a.lower() == g.lower())
        if k == "letter":
            m = re.findall(r"\b([A-D])\b", a.upper()); return float(bool(m) and m[-1] == g)
        if k == "list":
            try: v = json.loads(a)
            except Exception: v = [x.strip().strip("'\"") for x in a.strip("[]").split(",") if x.strip()]
            return float(sorted(str(x) for x in v) == sorted(json.loads(g)))
    except Exception: return 0.0
    return 0.0

def fake_reader_answer(prompt):
    found = [(s, FACTS[s]) for s in FACTS if s in prompt]
    if not found: return "No relevant facts found in the passage."
    found.sort(key=lambda t: prompt.index(t[0]))
    return "Facts found:\n" + "\n".join(f"- {e}: {v} {u}" for _, (e, v, u) in found)

def install_fake_reader():
    from rlm.environments.local_repl import LocalREPL
    if getattr(LocalREPL, "_fake_reader", False): return
    LocalREPL._llm_query = lambda self, prompt, model=None: fake_reader_answer(prompt)
    LocalREPL._llm_query_batched = lambda self, prompts, model=None: [fake_reader_answer(p) for p in prompts]
    LocalREPL._fake_reader = True
