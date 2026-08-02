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
# Constructed lazily: openai.OpenAI() raises when there is no key, and the notes/reader
# half of this server must work for a reviewer with no credentials.
_client = None


def _get_client():
    global _client
    if _client is None:
        if not KEY:
            raise RuntimeError("no LIGHTNING_AI_API_KEY / OPENAI_API_KEY — /ask is disabled")
        _client = openai.OpenAI(api_key=KEY, base_url=BASE, timeout=180)
    return _client

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

# Manual-review state, TRACKED IN GIT so reviewers can hand off to each other.
# reader.html is gitignored (regenerable), so notes must live in their own file.
# Shape: {scenario_id: {"notes": str, "checked": {"HX": iso8601, "HJ": iso8601}}}
# Written with sorted keys + indent so two reviewers editing different scenarios
# produce line-disjoint diffs that git merges cleanly.
NOTES_PATH = Path(__file__).parent / "review_notes.json"


def _load_notes() -> dict:
    if not NOTES_PATH.exists():
        return {}
    try:
        return json.loads(NOTES_PATH.read_text() or "{}")
    except Exception as e:
        print(f"  [warn] review_notes.json unreadable ({e}) — starting empty", file=sys.stderr)
        return {}


def _save_notes(all_notes: dict) -> None:
    tmp = NOTES_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(all_notes, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(NOTES_PATH)  # atomic; never leaves a half-written notes file


def answer_chat(messages: list, context: str) -> str:
    """messages = prior chat turns [{role:user|assistant, content}]; context = current scenario."""
    sysmsgs = [{"role": "system", "content": SYSTEM}]
    if context:
        sysmsgs.append({"role": "system",
                        "content": "CURRENT SCENARIO (the user is viewing this now):\n" + context})
    resp = _get_client().chat.completions.create(model=MODEL, messages=sysmsgs + messages)
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
        elif path == "/notes":
            self._json(200, _load_notes())
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
        path = self.path.rstrip("/") or "/"
        if path == "/notes":
            # Merge ONE scenario's entry rather than overwriting the whole file, so a stale
            # browser tab can't clobber notes another reviewer saved since it loaded.
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                sid = body.get("id")
                if not sid:
                    raise ValueError("missing scenario id")
                all_notes = _load_notes()
                entry = all_notes.get(sid) or {"notes": "", "checked": {}}
                if "notes" in body:
                    entry["notes"] = body["notes"] or ""
                if "checked" in body:  # {"HX": iso|null} — null clears that reviewer's check
                    for who, when in (body["checked"] or {}).items():
                        if when:
                            entry.setdefault("checked", {})[who] = when
                        else:
                            entry.get("checked", {}).pop(who, None)
                if body.get("title") and not entry.get("title"):
                    entry["title"] = body["title"]  # human-readable anchor for git diffs
                # Don't persist empty shells — just opening a scenario shouldn't add a row
                # to a git-tracked file. Clearing notes + unchecking removes the entry.
                if not entry.get("notes") and not entry.get("checked"):
                    all_notes.pop(sid, None)
                else:
                    all_notes[sid] = entry
                _save_notes(all_notes)
                self._json(200, {"ok": True, "entry": entry})
                print(f"  [notes] saved {sid}")
            except Exception as e:
                self._json(500, {"error": str(e)})
                print(f"  [notes error] {e}", file=sys.stderr)
            return
        if path != "/ask":
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
    # No API key is fine: the reader + shared review notes work without one. Only the
    # optional /ask endpoint needs credentials, so don't block a reviewer who just wants
    # to read transcripts and take notes.
    print(f"Reader → open  http://localhost:{PORT}/")
    print(f"  notes  : {NOTES_PATH}  (git-tracked — commit it to share your review)")
    if KEY:
        print(f"  /ask   : model = {MODEL} · base = {BASE}")
    else:
        print("  /ask   : disabled (no LIGHTNING_AI_API_KEY / OPENAI_API_KEY) — notes still work")
    print("  Ctrl-C to stop.")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
