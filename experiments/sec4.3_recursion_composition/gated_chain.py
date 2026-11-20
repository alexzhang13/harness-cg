"""Answer-gated task chain: unlock(answer) reveals the next task when the answer passes."""
import threading

DEFAULT_THRESHOLD = 0.999
THRESHOLDS = {"corpusqa": 0.8}


class GatedChain:
    def __init__(self, loaders, max_attempts=2, thresholds=None):
        self._loaders = list(loaders); self.n = len(self._loaders); self.max_attempts = max_attempts
        self._thr = {**THRESHOLDS, **(thresholds or {})}
        self._items = {}; self.accepted = set(); self.attempts = [0] * self.n; self.last_score = [None] * self.n
        self.attempt_pass = [None] * self.n
        self.log = []
        self._lock = threading.Lock()

    def _item(self, i):
        if i not in self._items: self._items[i] = self._loaders[i]()
        return self._items[i]

    def package(self, i):
        it = self._item(i)
        return {"stage": i + 1, "of": self.n, "question": it["question"], "context": it["context"]}

    def _locked_msg(self, i):
        left = self.max_attempts - self.attempts[i]
        return (f"LOCKED: wrong answer, {left} attempt{'s' if left != 1 else ''} left" if left > 0
                else "LOCKED: no attempts left; the chain stops here")

    def unlock_for(self, i):
        """The unlock tool for stage i."""
        def unlock(answer):
            with self._lock:
                if i > 0 and (i - 1) not in self.accepted:
                    return "LOCKED: this task is not unlocked"
                if i in self.accepted:
                    return "DONE" if i == self.n - 1 else self.package(i + 1)
                if self.attempts[i] >= self.max_attempts:
                    return self._locked_msg(i)
                self.attempts[i] += 1
                it = self._item(i)
                try: sc = float(it["score"](str(answer)))
                except Exception: sc = 0.0
                self.last_score[i] = sc
                passed = sc >= self._thr.get(it.get("env"), DEFAULT_THRESHOLD)
                self.log.append((i, self.attempts[i], str(answer)[:200], sc, passed))
                if not passed:
                    return self._locked_msg(i)
                self.accepted.add(i); self.attempt_pass[i] = self.attempts[i]
                return "DONE" if i == self.n - 1 else self.package(i + 1)
        return unlock

    @property
    def current(self):
        """First stage not yet passed (n when done)."""
        i = 0
        while i in self.accepted: i += 1
        return i

    def unlock_flat(self, answer):
        """unlock that always targets the current stage (no recursion)."""
        i = self.current
        if i >= self.n: return "DONE"
        return self.unlock_for(i)(answer)

    def scores(self):
        return [s if s is not None else 0.0 for s in self.last_score]

    def stats(self):
        return {"scores": self.scores(), "mean": sum(self.scores()) / self.n, "depth": len(self.accepted), "all_unlocked": len(self.accepted) == self.n,
                "attempts": list(self.attempts), "attempt_pass": list(self.attempt_pass)}
