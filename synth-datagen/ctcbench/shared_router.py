"""Guarded least-loaded routing over the served teacher ports listed in $CTC_PORTS_POOL (JSON, default
outputs/synth-datagen/ctcbench/ports_pool.json): per request, the healthy port with the least running + waiting requests."""
from __future__ import annotations
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")

import asyncio
import json
import random
import re
import threading
import time
import urllib.request

POOL = __import__("os").environ.get("CTC_PORTS_POOL") or _REPO + "/outputs/synth-datagen/ctcbench/ports_pool.json"
OWN_CAP, QUEUE_CAP = 24, 16
_RE = re.compile(r"^vllm:num_requests_(running|waiting)\{[^}]*\}\s+([0-9.e+]+)", re.M)


class Router:
    def __init__(self, pool=POOL, refresh=20.0, reread=180.0):
        self.pool, self.refresh, self.reread = pool, refresh * random.uniform(.75, 1.25), reread
        self.lock = threading.Lock()
        self.ports, self.load = [], {}
        self.inflight, self.recent, self.tok = {}, {}, {}
        self.t_pool = self.t_met = 0.0

    def _read_pool(self):
        try:
            self.ports = json.load(open(self.pool))["ports"]
        except Exception:  # noqa: BLE001 -- keep the last good pool
            pass
        self.t_pool = time.time()

    @staticmethod
    def _metrics(url):
        try:
            txt = urllib.request.urlopen(url.replace("/v1", "") + "/metrics", timeout=3).read().decode()
            d = {"running": 0.0, "waiting": 0.0}
            for k, v in _RE.findall(txt):
                d[k] += float(v)
            return d
        except Exception:  # noqa: BLE001 -- unhealthy port
            return None

    def _try(self, est_tokens):
        now = time.time()
        if now - self.t_pool > self.reread or not self.ports:
            self._read_pool()
        if now - self.t_met > self.refresh:
            self.load = {p["url"]: self._metrics(p["url"]) for p in self.ports}
            self.recent, self.t_met = {}, now
        ok = [p["url"] for p in self.ports if self.load.get(p["url"]) is not None
              and (est_tokens <= 120_000 or p.get("ctx", 0) >= 262_144)
              and self.inflight.get(p["url"], 0) < OWN_CAP]
        if not ok:
            return None
        spare = [u for u in ok if self.load[u]["waiting"] < QUEUE_CAP]
        if not spare:
            return None
        score = lambda u: (self.load[u]["running"] + 2 * self.load[u]["waiting"] + self.recent.get(u, 0)
                           + self.tok.get(u, 0) / 20_000)
        u = min(random.sample(spare, min(2, len(spare))), key=score)
        self.inflight[u] = self.inflight.get(u, 0) + 1
        self.tok[u] = self.tok.get(u, 0) + est_tokens
        self.recent[u] = self.recent.get(u, 0) + 1
        return u

    def pick(self, est_tokens=0):
        """Blocking acquire (threads). Pair every pick with done(url)."""
        while True:
            with self.lock:
                u = self._try(est_tokens)
            if u:
                return u
            time.sleep(2.0)

    async def apick(self, est_tokens=0):
        while True:
            with self.lock:
                u = self._try(est_tokens)
            if u:
                return u
            await asyncio.sleep(2.0)

    def done(self, url, est_tokens=0):
        with self.lock:
            if self.inflight.get(url, 0) > 0:
                self.inflight[url] -= 1
            self.tok[url] = max(0, self.tok.get(url, 0) - est_tokens)


ROUTER = Router()


class _Completions:
    def __init__(self, owner):
        self.o = owner

    async def create(self, **kw):
        est = sum(len(str(m.get("content", ""))) for m in kw.get("messages", [])) // 3
        u = await ROUTER.apick(est)
        try:
            return await self.o._cli(u).chat.completions.create(**kw)
        finally:
            ROUTER.done(u, est)


class RoutedAsyncOpenAI:
    """Duck-typed AsyncOpenAI: `.chat.completions.create(**kw)` routed per request through the guarded ROUTER."""

    def __init__(self, timeout=7200, max_retries=2):
        from openai import AsyncOpenAI
        self._mk = lambda u: AsyncOpenAI(base_url=u, api_key="EMPTY", timeout=timeout, max_retries=max_retries)
        self._c = {}
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(self)

    def _cli(self, u):
        if u not in self._c:
            self._c[u] = self._mk(u)
        return self._c[u]

    async def close(self):
        for c in self._c.values():
            await c.close()
