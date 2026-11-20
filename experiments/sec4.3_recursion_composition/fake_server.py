"""Scripted fake OpenAI-compatible server that plays a solver for the offline test of the gated loop (no GPU).
It knows each synthetic task's gold (by question text) and follows a fixed script per dialogue; knobs: wrong_first = set of question prefixes answered wrongly on the first attempt."""
import json, re, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def code(c, text="ok"): return f"{text}\n\n```repl\n{c}\n```"

class Fake:
    def __init__(self, golds, wrong_first=(), flat=False, teacher=False):
        self.teacher = teacher; self.roots = []; self.golds = golds; self.wrong_first = set(wrong_first); self.flat = flat; self.seen = []
    def gold_for(self, text):
        for q, g in self.golds.items():
            if q in text: return q, g
        return None, None
    def reply(self, msgs):
        users = [m for m in msgs if m["role"] == "user"]; texts = [m["content"] if isinstance(m["content"], str) else json.dumps(m["content"]) for m in msgs]
        n_assist = sum(m["role"] == "assistant" for m in msgs)
        if n_assist == 0: self.roots.append(texts[1] if len(texts) > 1 else '')
        last = texts[-1]; q, g = self.gold_for(texts[1] if len(texts) > 1 else "")
        out = [t for t in texts if t.startswith("REPL output")]; lo = out[-1] if out else ""
        if self.teacher:
            ans = g if g is not None else 'x'
            if q is not None and any(q == w for w in self.wrong_first) and 'LOCKED: wrong answer' not in ' '.join(out): ans = 'WRONG'
            return code(f'answer["content"] = {json.dumps(ans)}\nanswer["ready"] = True', 'I found the answer.')
        if self.flat:
            if "NEXTFLAT" in lo:
                m = re.search(r"QUESTION_START(.*?)QUESTION_END", lo, re.S); q2, g2 = self.gold_for(m.group(1)) if m else (None, None)
                return code(f"ANS = {json.dumps(g2)}\nr = unlock(ANS)\nprint(r if isinstance(r, str) else 'NEXTFLAT QUESTION_START' + r['question'] + 'QUESTION_END')\nif isinstance(r, dict):\n    pkg = r\n    context_0 = pkg['context']; context = context_0")
            if "DONE" in lo or "no attempts" in lo: return code('answer["content"] = "done"\nanswer["ready"] = True')
            ans = g if g is not None else "x"
            return code(f"ANS = {json.dumps(ans)}\nr = unlock(ANS)\nprint(r if isinstance(r, str) else 'NEXTFLAT QUESTION_START' + r['question'] + 'QUESTION_END')\nif isinstance(r, dict):\n    pkg = r\n    context_0 = pkg['context']; context = context_0")
        if "HANDOFF_DONE" in lo or "DONE" in lo.replace("HANDOFF_DONE", "") or "no attempts left" in lo:
            return code('answer["content"] = "done"\nanswer["ready"] = True')
        if "NEXT" in lo:
            return code("down = handoff(pkg)\nprint('HANDOFF_DONE', len(down))")
        if "1 attempt left" in lo:
            return code(f"r = unlock({json.dumps(g)})\nprint(r if isinstance(r, str) else 'NEXT')\npkg = r if isinstance(r, dict) else None")
        wrong = q is not None and any(q.startswith(w) for w in self.wrong_first) and "attempt" not in lo
        ans = ("WRONG" if wrong else g) if g is not None else "x"
        return code(f"r = unlock({json.dumps(ans)})\nprint(r if isinstance(r, str) else 'NEXT')\npkg = r if isinstance(r, dict) else None")

def serve(fake, port=0):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = fake.reply(body["messages"]); fake.seen.append(len(body["messages"]))
            resp = {"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "fake"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            data = json.dumps(resp).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    srv = ThreadingHTTPServer(("127.0.0.1", port), H); threading.Thread(target=srv.serve_forever, daemon=True).start(); return srv
