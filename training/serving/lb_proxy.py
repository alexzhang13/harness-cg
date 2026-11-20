"""Least-loaded router over a pool of vLLM servers, exposing one OpenAI-compatible endpoint.

    LB_POOL=pool.json python lb_proxy.py --listen 18431      # clients use http://127.0.0.1:18431/v1

The pool file (written by serve_vllm.sh) lists {"url", "ctx"} per server. Requests go to the healthy server with the
fewest running + waiting requests; long prompts only to 262k-window servers. With LB_USAGE_DIR set, token usage is
logged per tag, chosen through the URL prefix /t/<tag>/v1.
"""
import argparse
import asyncio
import json
import os
import re
import time

import aiohttp
from aiohttp import web

POOL = __import__("os").environ["LB_POOL"]
ap = argparse.ArgumentParser()
ap.add_argument("--listen", type=int, default=18431)
a = ap.parse_args()

state = {"ports": [], "load": {}, "pool_t": 0.0, "load_t": 0.0, "inflight": {}, "sent": {}}
PORT_CAP = int(__import__('os').environ.get('LB_PORT_CAP', '64'))      # max in-flight requests per server
TIMEOUT = float(__import__('os').environ.get('LB_TIMEOUT', '7200'))
QUEUE_CAP = 16     # queued requests allowed per server while another has room
RX = re.compile(r"^vllm:num_requests_(running|waiting)\{[^}]*\}\s+([0-9.e+]+)", re.M)


async def refresh(session):
    now = time.time()
    if now - state["pool_t"] > 120:
        try:
            state["ports"] = json.load(open(POOL))["ports"]
        except Exception:  # noqa: BLE001  keep the last good pool
            pass
        state["pool_t"] = now
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
                # /metrics can hang under load: fall back to /health and assume idle
                try:
                    async with session.get(base + "/health", timeout=aiohttp.ClientTimeout(total=6)) as r:
                        if r.status == 200:
                            return p["url"], {"running": 0.0, "waiting": 0.0}
                except Exception:  # noqa: BLE001
                    pass
                return p["url"], None
        res = await asyncio.gather(*(one(p) for p in state["ports"]))
        state["load"] = {u: d for u, d in res if d is not None}
        state["sent"] = {}   # not yet in the metrics
        state["load_t"] = now


def choose(nbytes):
    cand = [p for p in state["ports"] if p["url"] in state["load"]]
    if nbytes > 360_000:
        cand = [p for p in cand if p.get("ctx", 0) >= 262144] or cand
    if not cand:
        return None
    def est(p):   # metrics + our sends since the last refresh
        d = state["load"][p["url"]]
        return d["running"] + d["waiting"] + state["sent"].get(p["url"], 0)
    ok = [p for p in cand if state["inflight"].get(p["url"], 0) < PORT_CAP and est(p) < PORT_CAP + QUEUE_CAP]
    if not ok:
        return None   # all full: wait
    return min(ok, key=est)["url"]


# Token accounting: one line per completion under /t/<tag>/v1. role=root if the request has the RLM system prompt,
# else sub.
USAGE_DIR = os.environ.get("LB_USAGE_DIR")
if USAGE_DIR:
    os.makedirs(USAGE_DIR, exist_ok=True)
_files = {}


def _is_root(msgs):
    for m in msgs or []:
        if m.get("role") != "system":
            continue
        c = m.get("content")
        c = c if isinstance(c, str) else json.dumps(c)
        return "llm_query" in c and "repl" in c
    return False


def account(tag, reqbody, data, status):
    if not USAGE_DIR:
        return
    try:
        req = json.loads(reqbody)
        rec = {"t": round(time.time(), 2), "tag": tag, "model": req.get("model"), "status": status,
               "role": "root" if _is_root(req.get("messages")) else "sub", "stream": bool(req.get("stream"))}
        if status == 200 and not rec["stream"]:
            u = json.loads(data).get("usage") or {}
            rec["pt"], rec["ct"] = u.get("prompt_tokens"), u.get("completion_tokens")
        f = _files.get(tag)
        if f is None:
            f = _files[tag] = open(f"{USAGE_DIR}/{tag}.jsonl", "a", buffering=1)
        f.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001
        pass


async def handle(request):
    session = request.app["session"]
    await refresh(session)
    body = await request.read()
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", request.match_info.get("tag") or os.environ.get("LB_TAG", "untagged"))
    deadline, errors = time.time() + TIMEOUT, 0
    while time.time() < deadline and errors < 6:
        url = choose(len(body))
        if url is None:
            await asyncio.sleep(2)
            await refresh(session)
            continue
        state["inflight"][url] = state["inflight"].get(url, 0) + 1
        state["sent"][url] = state["sent"].get(url, 0) + 1
        try:
            tail = request.path.split("/v1", 1)[1]
            async with session.post(url + tail, data=body,
                                    headers={"Content-Type": "application/json"},
                                    timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
                data = await r.read()
                account(tag, body, data, r.status)
                return web.Response(body=data, status=r.status, content_type="application/json")
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
            state["load"].pop(url, None)        # down until the next refresh
            errors += 1
            await asyncio.sleep(2)
        finally:
            state["inflight"][url] -= 1
    return web.Response(status=503, text="no healthy port")


async def models(request):
    return web.json_response({"object": "list", "data": [{"id": os.environ.get("LB_MODEL_NAME", "model"), "object": "model"}]})


async def health(request):
    await refresh(request.app["session"])
    return web.Response(status=200 if state["load"] else 503, text="ok" if state["load"] else "no port")


async def metrics(request):   # summed over the pool
    await refresh(request.app["session"])
    run = sum(d["running"] for d in state["load"].values()); wait = sum(d["waiting"] for d in state["load"].values())
    return web.Response(text=f'vllm:num_requests_running{{pool="lb"}} {run}\nvllm:num_requests_waiting{{pool="lb"}} {wait}\n')


async def init():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["session"] = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
    for pre in ("", "/t/{tag}"):
        app.router.add_get(pre + "/v1/models", models)
        app.router.add_get(pre + "/health", health)
        app.router.add_get(pre + "/metrics", metrics)
        app.router.add_post(pre + "/v1/{tail:.*}", handle)
    return app


web.run_app(init(), host="127.0.0.1", port=a.listen, access_log=None)
