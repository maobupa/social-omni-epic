#!/usr/bin/env python3
"""Local AI agent for the transcript reader.

Serves POST /ask — takes the current scenario's context + your question, calls a strong
model through Lightning AI, and returns the answer. The reader page (opened via file://)
talks to it at http://localhost:8765; CORS is open so the static file can reach it.

    uv run transcript_reader/agent_server.py        # starts on :8765, leave running

Requires LIGHTNING_AI_API_KEY (+ LIGHTNING_AI_BASE_URL) in .env at the repo root.
Change the model with:  AGENT_MODEL=openai/gpt-5.6 uv run transcript_reader/agent_server.py
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

import openai

PORT = int(os.getenv("AGENT_PORT", "8765"))
MODEL = os.getenv("AGENT_MODEL", "openai/gpt-5")  # Lightning-served; e.g. openai/gpt-5.6, anthropic/claude-opus-4-8

KEY = os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE = os.getenv("LIGHTNING_AI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
_client = openai.OpenAI(api_key=KEY, base_url=BASE, timeout=180)

SYSTEM = (
    "You are a research assistant helping a scientist read and analyze transcripts from a "
    "social-simulation curriculum experiment (SOTOPIA-style). Each scenario pits a LEARNER agent "
    "against a PARTNER agent over up to 4 retry attempts; after a failed attempt the learner writes "
    "a 'reflection' that is injected into the next attempt as memory. Categories: too_easy (solved "
    "on the first try), frontier_solved (showed learning progress and eventually solved), "
    "frontier_unsolved (improved but never crossed the success bar), beyond_frontier (no learning "
    "progress). Answer the user's question about the CURRENT scenario using ONLY the provided "
    "context. Be concise and specific; cite attempt numbers and turns. When asked why something "
    "failed or what changed, ground it in concrete moves and quotes from the transcript."
)


READER_HTML = Path(__file__).parent / "reader.html"


def answer_chat(messages: list, context: str) -> str:
    """messages = prior chat turns [{role:user|assistant, content}]; context = current scenario."""
    sysmsgs = [{"role": "system", "content": SYSTEM}]
    if context:
        sysmsgs.append({"role": "system",
                        "content": "CURRENT SCENARIO (the user is viewing this now):\n" + context})
    resp = _client.chat.completions.create(model=MODEL, messages=sysmsgs + messages)
    return resp.choices[0].message.content


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/reader.html", "/index.html"):
            if READER_HTML.exists():
                data = READER_HTML.read_bytes()
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json(404, {"error": "reader.html not built — run transcript_reader/build.py"})
        elif path == "/health":
            self._json(200, {"ok": True, "model": MODEL})
        else:
            self._json(404, {"error": "not found"})

    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path.rstrip("/") != "/ask":
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            ctx = body.get("context") or ""
            msgs = body.get("messages")
            if msgs:  # chat mode (list of {role, content})
                label = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
            else:     # single-question fallback
                q = (body.get("question") or "").strip()
                if not q:
                    raise ValueError("empty question")
                msgs = [{"role": "user", "content": q}]
                label = q
            out = answer_chat(msgs, ctx)
            self._json(200, {"answer": out})
            print(f"  [ok] {label[:70]}  -> {len(out)} chars")
        except Exception as e:
            self._json(500, {"error": str(e)})
            print(f"  [error] {e}", file=sys.stderr)

    def log_message(self, *a):
        pass  # silence default request logging


def main():
    if not KEY:
        print("ERROR: no LIGHTNING_AI_API_KEY / OPENAI_API_KEY found in .env", file=sys.stderr)
        sys.exit(1)
    print(f"Reader + agent → open  http://localhost:{PORT}/   (AI works out of the box there)")
    print(f"  /ask endpoint · model = {MODEL} · base = {BASE}")
    print("  (You can also double-click reader.html for reading-only; AI needs this server.)")
    print("  Ctrl-C to stop.")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
