"""Route requests to the least-loaded teacher port in $CTC_PORTS_POOL (running + 2 * waiting, from /metrics).

Requests from the same rollout stick to one port while it stays acceptable, to reuse the prefix cache."""
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")
import asyncio, hashlib, json, threading, time, urllib.request, httpx
POOL = __import__("os").environ.get("CTC_PORTS_POOL") or _REPO + "/outputs/synth-datagen/ctcbench/ports_pool.json"
class Router:
    def __init__(self, need_ctx=0, per_port_cap=24):
        self.need, self.cap = need_ctx, per_port_cap; self.lock = threading.Lock()
        self.ports, self.m, self.mine, self.sticky, self.sent = [], {}, {}, {}, {}; self.t_pool = self.t_m = 0
    def _refresh(self):
        now = time.time()
        if now - self.t_pool > 180:
            try: self.ports = [p["url"].rsplit("/v1", 1)[0] for p in json.load(open(POOL))["ports"] if p.get("ctx", 0) >= self.need]
            except Exception: pass
            self.t_pool = now
        if now - self.t_m > 20:
            m = {}
            for u in self.ports:
                try:
                    t = urllib.request.urlopen(u + "/metrics", timeout=3).read().decode()
                    g = lambda k: sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(k + "{"))
                    m[u] = (g("vllm:num_requests_running"), g("vllm:num_requests_waiting"))
                except Exception: pass
            self.m, self.t_m, self.sent = m, now, {}
    def _ok(self, u, any_idle):
        r, w = self.m[u]
        est = r + w + self.sent.get(u, 0)
        return self.mine.get(u, 0) < self.cap and est < 80 and (w == 0 or not any_idle)
    def pick(self, key=None):
        with self.lock:
            self._refresh()
            healthy = [u for u in self.ports if u in self.m]
            if not healthy: return None
            any_idle = any(self.m[u][1] == 0 for u in healthy)
            if key and key in self.sticky and self.sticky[key] in self.m and self._ok(self.sticky[key], any_idle):
                u = self.sticky[key]
            else:
                c = [u for u in healthy if self._ok(u, any_idle)]
                if not c: return None
                u = min(c, key=lambda x: self.m[x][0] + 2 * self.m[x][1] + self.mine.get(x, 0))
                if key: self.sticky[key] = u
            self.mine[u] = self.mine.get(u, 0) + 1
            self.sent[u] = self.sent.get(u, 0) + 1
            return u
    def done(self, u):
        with self.lock: self.mine[u] = max(0, self.mine.get(u, 1) - 1)
ROUTER = Router()
def _key(body):
    try:
        msgs = json.loads(body).get("messages") or []
        return hashlib.sha1(str(msgs[0].get("content"))[:4000].encode()).hexdigest() if len(msgs) > 1 else None
    except Exception: return None
class RoutingTransport(httpx.AsyncBaseTransport):
    def __init__(self, router=ROUTER):
        self.r, self.inner = router, httpx.AsyncHTTPTransport(retries=2)
    async def handle_async_request(self, request):
        body = request.content if hasattr(request, "_content") else await request.aread()
        while True:
            u = await asyncio.to_thread(self.r.pick, _key(body))
            if u: break
            await asyncio.sleep(5)
        host = httpx.URL(u)
        req = httpx.Request(request.method, request.url.copy_with(scheme=host.scheme, host=host.host, port=host.port),
                            headers=[(k, v) for k, v in request.headers.raw if k.lower() != b"host"], content=body)
        try:
            resp = await self.inner.handle_async_request(req)
            await resp.aread()
            return resp
        finally:
            self.r.done(u)
def async_openai(timeout=7200):
    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url="http://router.invalid/v1", api_key="EMPTY", timeout=timeout, max_retries=2,
                       http_client=httpx.AsyncClient(transport=RoutingTransport(), timeout=timeout))
