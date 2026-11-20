"""Local least-loaded HTTP proxy over the teacher ports listed in $CTC_PORTS_POOL (one OpenAI endpoint for gen.py)."""
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")
import argparse
import asyncio
import json
import re
import time

import aiohttp
from aiohttp import web

POOL = __import__("os").environ.get("CTC_PORTS_POOL") or _REPO + "/outputs/synth-datagen/ctcbench/ports_pool.json"
ap = argparse.ArgumentParser()
ap.add_argument("--listen", type=int, default=18431)
a = ap.parse_args()

state = {"ports": [], "load": {}, "pool_t": 0.0, "load_t": 0.0, "inflight": {}, "sent": {}}
PORT_CAP = int(__import__('os').environ.get('LB_PORT_CAP', '64'))
QUEUE_CAP = 16
RX = re.compile(r"^vllm:num_requests_(running|waiting)\{[^}]*\}\s+([0-9.e+]+)", re.M)


async def refresh(session):
    now = time.time()
    if now - state["pool_t"] > 120:
        try:
            state["ports"] = json.load(open(POOL))["ports"]
        except Exception:  # noqa: BLE001 -- keep the last good pool
            pass
        state["pool_t"] = now
    if now - state.get("cap_t", 0) > 30:
        global PORT_CAP
        try:
            PORT_CAP = int(open(f"{__file__}.{a.listen}.cap").read().strip())
        except Exception:  # noqa: BLE001 -- no file / unreadable: keep the current cap
            pass
        state["cap_t"] = now
    if now - state["load_t"] > 20:
        async def one(p):
            base = p["url"].rsplit("/v1", 1)[0]
            try:
                async with session.get(base + "/metrics", timeout=aiohttp.ClientTimeout(total=4)) as r:
                    txt = await r.text()
                d = {"running": 0.0, "waiting": 0.0}
                for k, v in RX.findall(txt):
                    d[k] += float(v)
                return p["url"], d
            except Exception:  # noqa: BLE001
                return p["url"], None
        res = await asyncio.gather(*(one(p) for p in state["ports"]))
        state["load"] = {u: d for u, d in res if d is not None}
        state["sent"] = {}
        state["load_t"] = now


def choose(nbytes):
    cand = [p for p in state["ports"] if p["url"] in state["load"]]
    if nbytes > 360_000:
        cand = [p for p in cand if p.get("ctx", 0) >= 262144] or cand
    if not cand:
        return None
    def est(p):
        d = state["load"][p["url"]]
        return d["running"] + d["waiting"] + state["sent"].get(p["url"], 0)
    ok = [p for p in cand if state["inflight"].get(p["url"], 0) < PORT_CAP and est(p) < 64 + QUEUE_CAP]
    if not ok:
        return None
    return min(ok, key=est)["url"]


async def handle(request):
    session = request.app["session"]
    await refresh(session)
    body = await request.read()
    deadline, errors = time.time() + 7200, 0
    while time.time() < deadline and errors < 20:
        url = choose(len(body))
        if url is None:
            await asyncio.sleep(2)
            await refresh(session)
            continue
        state["inflight"][url] = state["inflight"].get(url, 0) + 1
        state["sent"][url] = state["sent"].get(url, 0) + 1
        try:
            async with session.post(url + request.path[len("/v1"):], data=body,
                                    headers={"Content-Type": "application/json"},
                                    timeout=aiohttp.ClientTimeout(total=7200)) as r:
                data = await r.read()
                return web.Response(body=data, status=r.status, content_type="application/json")
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
            state["load"].pop(url, None)
            state["pool_t"] = 0
            errors += 1
            await asyncio.sleep(min(60, 3 * errors))
        finally:
            state["inflight"][url] -= 1
    return web.Response(status=503, text="no healthy port")


async def models(request):
    return web.json_response({"object": "list", "data": [{"id": "qwen38-27b", "object": "model"}]})


async def init():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["session"] = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/{tail:.*}", handle)
    return app


web.run_app(init(), host="127.0.0.1", port=a.listen, access_log=None)
