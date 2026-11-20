"""Fake sub-LM for trick_synth, calibrated to Qwen3.8-27B on OBLIQ math.

Direct "same technique?" judgements are weak, while describing each problem's core trick is mostly right. So the
best root program describes every candidate and then flags in description space.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from synth_common.oracle import FakeSubLM, ReadModel, rng_for
from . import data as D

_DESC = re.compile(r"sketch|fingerprint|meta-?program|core (trick|idea|technique)|describe|how (would|to|you)\b.*solv|solve it|key (idea|step|insight)|technique", re.I)
_FLAG = re.compile(r"\bflag\b|which (of (these|them)|candidates|ones|descriptions|problems)\b.*\b(share|same|match|could)|list (the )?(numbers|ids)", re.I | re.S)
_YESNO = re.compile(r"\byes\b.*\bno\b|\byes/no\b", re.I)
_JUDGE = re.compile(r"\bscore|\brate\b|\brating|\brank|relevan|\byes\b|\bsimilar(ity)? to the query|same (core )?(technique|trick|idea) as the query", re.I)
_SCALE = re.compile(r"(\d+)\s*[-–to]+\s*(\d+)")
_LABEL = re.compile(r"(?:doc_id:\s*|\[|\b)([A-Za-z]?[0-9a-f]{6,12}|\d{1,4})\]?\s*[.:)\]|]?\s*$")

@dataclass
class TrickReadModel(ReadModel):
    budget_chars: int = 90_000
    reliable_items: int = 40
    miss_rate: float = 0.3
    desc_acc_focus: float = 0.85
    desc_acc_scan: float = 0.62
    yes_pos: float = 0.33; yes_fam: float = 0.28; yes_neg: float = 0.12
    flag_same: float = 0.80; flag_fam: float = 0.25; flag_neg: float = 0.03
    rawflag_pos: float = 0.50; rawflag_fam: float = 0.20; rawflag_neg: float = 0.06
    fault_q: float = 0.0
    fault_desc: float = 0.0
    fault_flag: float = 0.0
    fp_generic: float = 0.0
    fp_surface: float = 0.0
    sk_surface: float = 0.0
    flag_weak_same: float = 0.45; flag_weak_fam: float = 0.15
    q_acc_brief: float | None = None
    batch_focus: int = 20; batch_scan: int = 100

_STRICT = re.compile(r"EXACTLY|no other text|do not add|nothing else|ONLY (these|this|one)|strict")
_STRICT_I = re.compile(r"no other text|do not add|nothing else|exactly this format|strictly", re.I)
PRESETS = {"real_v1": dict(fp_generic=0.55, fp_surface=0.3, sk_surface=0.2, flag_weak_same=0.65, q_acc_brief=0.65,
                           batch_focus=8, batch_scan=60, desc_acc_scan=0.55)}

_GENERIC = ["Apply a standard {f} argument that reduces the problem to a simpler, well-understood structure",
            "Use {f} reasoning to set up the key relation and then carry out the computation",
            "Analyze the structure with {f} tools to simplify the expression before solving",
            "Recognize the problem as a classic {f} exercise and apply the usual technique for it"]
_SURF = ["Model the {a} and {b} described in the problem and compute the required {c} directly",
         "Set up the situation with the {a}, track how the {b} changes, and solve for the {c}",
         "Translate the {a} and {b} constraints into equations and find the {c}"]
_WORD = re.compile(r"[a-z]{5,}")
_PROBE = re.compile(r"what (kind|type|sort)s? of|overview|summari[sz]e (the|this|these)|(describe|characteri[sz]e) (the|this) (pool|collection|corpus|sample|data|items)|"
                    r"what (do|does) (these|the) (items|problems|documents)|how should (i|we|one)|recommend|what approach|format of", re.I)
_PROBE_ADVICE = [
    "The statements do not name the technique; matching on surface words will mislead. Describe each problem's SOLUTION (sketch + core trick) and match on those descriptions.",
    "Relevance depends on how a problem is solved, not its story: get a solution sketch and a domain-independent fingerprint per problem, then compare fingerprints.",
    "These look like standard exercises; a keyword search on the query's key terms may find related ones, but similar stories can hide different techniques.",
    "Group the problems by the technique their solutions use; the surface settings vary a lot, so compare at the level of the solution idea."]
_SOLVE_FIRST = re.compile(r"think (it )?through|think carefully|step by step|solve it (first|fully)|first (work out|solve)|professor|in detail", re.I)

def _drift_key(text: str) -> str:
    return _norm(text).strip(" -:*[]0123456789.")[:48]

_Q_RAMBLE = ["Let me think about how to approach this. First, consider the constraints carefully: the condition has to hold at every step, so",
             "Okay, we need to count the configurations. One approach would be to enumerate small cases first and look for a pattern; for n = 1",
             "This looks like it might need a clever substitution. Let me try writing out the first few terms and checking whether"]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

_W = re.compile(r"[a-z]{4,}")

def _q_present(prompt: str, hidden: dict) -> bool:
    """Fuzzy check that the query is in the prompt (the root often re-types it): >= 60% of its longer words appear."""
    qw = hidden.get("_qwords")
    if qw is None:
        return hidden["_qkey"] in _norm(prompt)
    if not qw:
        return False
    pw = set(_W.findall(prompt.lower()))
    hit = sum(1 for w in qw if w in pw)
    return hit >= max(6, 0.6 * len(qw)) if len(qw) >= 6 else hit == len(qw)

class TrickOracle(FakeSubLM):
    def __init__(self, read: TrickReadModel | None = None):
        super().__init__(read or TrickReadModel())

    def _items(self, prompt: str, hidden: dict) -> list[tuple[str, str, int]]:
        """(alias, label, position) of every pool item whose opening text appears in the prompt."""
        P = _norm(prompt); out = []
        for a, key, off in hidden["_keys"]:
            pos = P.find(key)
            if pos >= 0:
                start = max(0, pos - off)
                pre = P[max(0, start - 40):start]
                m = _LABEL.search(pre)
                out.append((a, m.group(1) if m else a, pos))
        return sorted(out, key=lambda x: x[2])

    def _desc_lines(self, prompt: str, st: dict | None = None) -> list[tuple[str, int]]:
        """(label, technique) for lines carrying one of our fingerprints; drifted ones map to -1 - technique."""
        out = []; drift = (st or {}).get("drift") or {}
        for line in prompt.splitlines():
            n = _norm(line)
            if len(n) < 30:
                continue
            t = D.lookup_desc(n)
            if t is None and drift:
                for seg in re.split(r"\||meta:|idea:|sketch:|fingerprint:|trick:", n):
                    k = seg.strip(" -:*[]0123456789.").strip()[:48]
                    if len(k) >= 30 and k in drift:
                        t = -1 - drift[k]; break
            if t is None:
                continue
            m = re.match(r"\s*\W{0,2}([A-Za-z]?[0-9a-f]{6,12}|\d{1,4})\W", line)
            out.append((m.group(1) if m else str(len(out) + 1), t))
        return out

    def _query_tech(self, prompt: str, hidden: dict) -> int | None:
        if _q_present(prompt, hidden):
            return hidden["qtech"]
        for line in prompt.splitlines():
            t = D.lookup_desc(_norm(line))
            if t is not None and not re.match(r"\s*\W{0,2}([A-Za-z]?[0-9a-f]{6,12}|\d{1,4})\W", line):
                return t
        return None

    def _instruction(self, prompt: str, hidden: dict) -> str:
        keys = [k for _, k, _ in hidden["_keys"]]
        qw = set(hidden.get("_qwords") or [])
        keep = []
        for line in prompt.splitlines():
            n = _norm(line)
            if len(n) > 40 and any(k in n for k in keys):
                continue
            if qw and len(n) > 40:
                lw = set(_W.findall(n))
                if lw and len(lw & qw) >= max(4, 0.6 * len(lw)):
                    continue
            if len(n) > 30 and D.lookup_desc(n) is not None and re.match(r"\s*\W{0,2}([A-Za-z]?[0-9a-f]{6,12}|\d{1,4})\W", line):
                continue
            keep.append(line)
        return "\n".join(keep)

    def _q_acc(self, prompt: str) -> float:
        R = self.read
        if R.q_acc_brief is None or _SOLVE_FIRST.search(prompt): return R.desc_acc_focus
        return R.q_acc_brief

    def _fault(self, name: str, hidden: dict) -> bool:
        return rng_for(hidden["key"], "fault", name).random() < getattr(self.read, name)

    def respond(self, prompt: str, hidden: dict, st: dict) -> str:
        R = self.read; salt = hidden["key"]; tech = hidden["tech"]; qt = hidden["qtech"]
        items = self._items(prompt, hidden)
        instr = self._instruction(prompt, hidden)
        is_desc = bool(_DESC.search(instr)); is_flag = bool(_FLAG.search(instr))
        dlines = self._desc_lines(prompt, st)
        fam = lambda t: D.family(t)
        if _PROBE.search(instr) and not is_flag and not dlines and len(items) <= 30:
            r = rng_for(salt, prompt, "probe"); n = len(items)
            lens = [len(D.corpus()["I"][hidden["_row"][a]]["text"]) for a, _, _ in items] if ("_row" in hidden and items) else []
            avg = int(sum(lens) / len(lens)) if lens else 250
            body = (f"TYPE: {'short self-contained word problems (2-4 sentences) from very different settings, each with a definite answer' if n else 'problems from varied settings'}; "
                    f"none names the method needed.\nLENGTH: about {avg} characters each ({n} shown).\n"
                    f"ADVICE: {r.choice(_PROBE_ADVICE)} They are short, so 40 per describe call is fine.")
            st["probe_calls"] = st.get("probe_calls", 0) + 1
            return self.chat(prompt, body, salt)
        if is_flag and dlines:
            tgt = self._query_tech(prompt, hidden)
            if tgt is None:
                tgt = D.closest_technique(instr, hidden)
            out = []
            for lab, t in dlines:
                weak = t < 0; t = -1 - t if weak else t
                if tgt is None: p = R.flag_neg * 2
                elif weak: p = R.flag_weak_same if t == tgt else (R.flag_weak_fam if fam(t) == fam(tgt) else R.flag_neg)
                elif t == tgt: p = R.flag_same
                elif fam(t) == fam(tgt): p = R.flag_fam
                else: p = R.flag_neg
                if rng_for(salt, prompt, lab, "fl").random() < p: out.append(lab)
            st["flag_calls"] = st.get("flag_calls", 0) + 1
            if self._fault("fault_flag", hidden):
                r = rng_for(salt, prompt, "fprose")
                body = (("Looking at these, the candidates that could share the query's core trick are "
                         + (", ".join(out[:-1]) + " and " + out[-1] if len(out) > 1 else out[0]) + ".") if out
                        else "None of these candidates seem to share the query's core trick.")
                return body + r.choice(["", " The rest rely on different ideas.", " Others are only superficially similar."])
            return self.chat(prompt, "FLAG: " + (", ".join(out) if out else "NONE"), salt)
        q_in = _q_present(prompt, hidden)
        is_judge = bool(_JUDGE.search(instr)) and q_in and bool(items)
        flag_only_words = is_flag and not dlines and not (items and q_in)
        if (not is_flag or flag_only_words) and not is_judge and (is_desc or (items and not q_in)) and (items or q_in):
            if items and q_in and len(items) <= 1 and not is_judge:
                items = []
            n = len(items)
            if n == 0:
                st["q_calls"] = st.get("q_calls", 0) + 1
                if st["q_calls"] == 1 and self._fault("fault_q", hidden):
                    return rng_for(salt, prompt, "qr").choice(_Q_RAMBLE)
                t = qt if rng_for(salt, prompt, "q").random() < self._q_acc(prompt) else D.confuse(qt, rng_for(salt, prompt, "qc"))
                return self.chat(prompt, f"SKETCH: {D.sketch_for_query(hidden['qiid'], t)}\nFINGERPRINT: {D.fingerprint(t, rng_for(salt, prompt, 'qf'))}", salt)
            ids = [a for a, _, _ in items]
            read, _ = self.read_items(prompt, ids, st, salt)
            acc = R.desc_acc_focus - (R.desc_acc_focus - R.desc_acc_scan) * min(1.0, max(0, n - R.batch_focus) / max(1, R.batch_scan - R.batch_focus))
            lab = {a: l for a, l, _ in items}; lines = []
            alt = (self._fault("fault_desc", hidden) and rng_for(salt, prompt, "altfmt").random() < 0.5
                   and not (_STRICT.search(instr) or _STRICT_I.search(instr)))
            for a in read:
                r = rng_for(salt, prompt, a, "d")
                t = tech[a] if r.random() < acc else D.confuse(tech[a], r)
                sk = hidden["sketch"][a] if t == tech[a] else D.random_sketch(t, r)
                fpr = D.fingerprint(t, r)
                u = r.random(); words = _WORD.findall(_norm(D.corpus()["I"][hidden["_row"][a]]["text"]) if "_row" in hidden else _norm(sk)) or ["quantity", "setup", "value"]
                if u < R.fp_generic:
                    fpr = r.choice(_GENERIC).format(f=D.family(tech[a]).split("/")[0].strip())
                elif u < R.fp_generic + R.fp_surface:
                    fpr = r.choice(_SURF).format(a=r.choice(words), b=r.choice(words), c=r.choice(words))
                if r.random() < R.sk_surface:
                    sk = r.choice(_SURF).format(a=r.choice(words), b=r.choice(words), c=r.choice(words)) + "."
                for txt in (fpr, sk):
                    if D.lookup_desc(_norm(txt)) is None:
                        st.setdefault("drift", {})[_drift_key(txt)] = tech[a]
                lines.append(f"[{lab[a]}] Sketch: {sk} | Core trick: {fpr}" if alt else f"[{lab[a]}] SKETCH: {sk} | FINGERPRINT: {fpr}")
            st["desc_calls"] = st.get("desc_calls", 0) + 1
            return self.chat(prompt, "\n".join(lines), salt)
        if q_in and not items and not is_flag and not dlines:
            st["q_calls"] = st.get("q_calls", 0) + 1
            if st["q_calls"] == 1 and self._fault("fault_q", hidden):
                return rng_for(salt, prompt, "qr").choice(_Q_RAMBLE)
            t = qt if rng_for(salt, prompt, "q").random() < self._q_acc(prompt) else D.confuse(qt, rng_for(salt, prompt, "qc"))
            return self.chat(prompt, f"SKETCH: {D.sketch_for_query(hidden['qiid'], t)}\nFINGERPRINT: {D.fingerprint(t, rng_for(salt, prompt, 'qf'))}", salt)
        if is_desc and not items and not dlines:
            t = D.closest_technique(prompt, hidden)
            if t is not None:
                r = rng_for(salt, prompt, "para")
                tt = t if r.random() < R.desc_acc_scan else D.confuse(t, r)
                return self.chat(prompt, f"SKETCH: {D.random_sketch(tt, r)}\nFINGERPRINT: {D.fingerprint(tt, r)}", salt)
        if items and q_in:
            ids = [a for a, _, _ in items]; read, _ = self.read_items(prompt, ids, st, salt)
            lab = {a: l for a, l, _ in items}
            def cls(a): return 2 if tech[a] == qt else (1 if fam(tech[a]) == fam(qt) else 0)
            if is_flag:
                P = {2: R.rawflag_pos, 1: R.rawflag_fam, 0: R.rawflag_neg}
                out = [lab[a] for a in read if rng_for(salt, prompt, a, "rf").random() < P[cls(a)]]
                return self.chat(prompt, "FLAG: " + (", ".join(out) if out else "NONE"), salt)
            if _YESNO.search(instr):
                P = {2: R.yes_pos, 1: R.yes_fam, 0: R.yes_neg}
                return self.chat(prompt, "\n".join(f"{lab[a]}: {'YES' if rng_for(salt, prompt, a, 'yn').random() < P[cls(a)] else 'NO'}" for a in read), salt)
            m = _SCALE.search(instr); lo, hi = (int(m.group(1)), int(m.group(2))) if m else (0, 10)
            mu = {2: 0.55, 1: 0.43, 0: 0.30}
            def val(a):
                r = rng_for(salt, prompt, a, "sc"); x = min(1, max(0, r.gauss(mu[cls(a)], 0.22)))
                return round(lo + x * (hi - lo))
            st["judge_calls"] = st.get("judge_calls", 0) + 1
            return self.chat(prompt, "\n".join(f"{lab[a]}: {val(a)}" for a in read), salt)
        return self.chat(prompt, "I can help if you give me the problems (and the query) to work with, and say what you need for each.", salt)
