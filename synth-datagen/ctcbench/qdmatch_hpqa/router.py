"""Least-loaded routing over the teacher ports listed in $CTC_PORTS_POOL (used by run_dev.py with QD_ROUTE=1)."""
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")
import asyncio
import json
import time
import urllib.request

from openai import AsyncOpenAI
from verifiers import OpenAIChatCompletionsClient

POOL = __import__("os").environ.get("CTC_PORTS_POOL") or _REPO + "/outputs/synth-datagen/ctcbench/ports_pool.json"
import os
PORT_CAP = int(os.environ.get('QD_PORT_CAP', '64'))
QUEUE_CAP = int(os.environ.get('QD_QUEUE_CAP', '16'))


def _metrics(url: str):
    try:
        t = urllib.request.urlopen(url.replace("/v1", "/metrics"), timeout=4).read().decode()
        run = sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith("vllm:num_requests_running{"))
        wait = sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith("vllm:num_requests_waiting{"))
        return run, wait
    except Exception:  # noqa: BLE001
        return None


class Pool:
    def __init__(self):
        self.ports, self.load, self.t_pool, self.t_load, self.clients = [], {}, 0.0, 0.0, {}
        self.pending, self.inflight = {}, {}
        import threading
        self.lock = threading.Lock()

    def refresh(self, force: bool = False):
        now = time.time()
        if force or now - self.t_pool > 180:
            try:
                self.ports = json.load(open(POOL))["ports"]
            except Exception:  # noqa: BLE001
                pass
            self.t_pool = now
        if force or now - self.t_load > 8:
            self.load = {p["url"]: _metrics(p["url"]) for p in self.ports}
            self.pending = {u: 0 for u in self.load}
            self.t_load = now

    def pick(self, need_ctx: int = 0):
        """Least-loaded healthy port under the hard caps, or None when every port is full (caller waits)."""
        with self.lock:
            self.refresh()
            ok = [p for p in self.ports if self.load.get(p["url"]) is not None and p.get("ctx", 0) >= need_ctx]
            if not ok:
                self.refresh(force=True)
                ok = [p for p in self.ports if self.load.get(p["url"]) is not None and p.get("ctx", 0) >= need_ctx]
            if not ok:
                return None
            est = lambda u: self.load[u][0] + self.load[u][1] + self.pending.get(u, 0)
            ok = [p for p in ok if self.inflight.get(p["url"], 0) < PORT_CAP and est(p["url"]) < PORT_CAP + QUEUE_CAP]
            if not ok:
                return None
            u = min(ok, key=lambda p: est(p["url"]) + self.load[p["url"]][1])["url"]
            self.pending[u] = self.pending.get(u, 0) + 1
            self.inflight[u] = self.inflight.get(u, 0) + 1
            return u

    def done(self, u: str):
        with self.lock:
            self.inflight[u] = max(0, self.inflight.get(u, 1) - 1)

    def any_free(self) -> bool:
        self.refresh()
        return any(v is not None and v[1] == 0 for v in self.load.values())

    def client(self, url: str):
        if url not in self.clients:
            self.clients[url] = OpenAIChatCompletionsClient(AsyncOpenAI(api_key="local", base_url=url, timeout=10000))
        return self.clients[url]


POOL_STATE = Pool()


class RoutedClient(OpenAIChatCompletionsClient):
    """A verifiers client whose every get_response goes to the least-loaded pool port."""

    def __init__(self):
        super().__init__(AsyncOpenAI(api_key="local", base_url="http://unused/v1"))

    async def get_response(self, prompt, model, sampling_args, tools=None, **kwargs):  # noqa: ANN001
        est = sum(len(str(getattr(m, "content", m.get("content", "") if isinstance(m, dict) else ""))) for m in prompt) // 3
        while True:
            url = await asyncio.to_thread(POOL_STATE.pick, 262144 if est > 120000 else 0)
            if url:
                break
            await asyncio.sleep(3)
        try:
            return await POOL_STATE.client(url).get_response(prompt, model, sampling_args, tools, **kwargs)
        finally:
            POOL_STATE.done(url)
