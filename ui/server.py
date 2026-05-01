"""
ui/server.py — Minimal HTTP server for the agent query UI
----------------------------------------------------------
Starts a local server on http://localhost:8080 that:
  GET  /          → serves ui/index.html
  POST /query     → routes the message to the selected agent and returns JSON

Usage:
    # from the project root with .venv active:
    python ui/server.py

Env vars (optional, for LLM agent):
    AGENT_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY, LLM_MODEL
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Resolve project root so agents/ and data/ are importable ──────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # ui/
PROJECT_ROOT = SCRIPT_DIR.parent                      # getting-started/
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from agents import ReactiveAgent, LearningAgent, LLMAgent
from data.sample_data import TRAINING_DATA

# ── One-time agent initialisation ─────────────────────────────────────────────
print("Initialising agents …")

reactive = ReactiveAgent()

learning = LearningAgent()
learning.add_examples(TRAINING_DATA)
learning.train()
print(f"  LearningAgent trained on {len(TRAINING_DATA)} examples.")

_llm_available = False
llm = None
try:
    llm = LLMAgent()
    _llm_available = True
    print(f"  LLMAgent ready ({llm.provider} / {llm.model}).")
except Exception as exc:
    print(f"  LLMAgent skipped: {exc}")

INDEX_HTML = SCRIPT_DIR / "index.html"

# ── Request handler ────────────────────────────────────────────────────────────
class AgentHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):       # suppress default per-req noise
        pass

    # ── GET / ─────────────────────────────────────────────────────────────────
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_file(INDEX_HTML, "text/html; charset=utf-8")
        else:
            self._not_found()

    # ── POST /query ───────────────────────────────────────────────────────────
    def do_POST(self):
        if self.path != "/query":
            self._not_found()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._json_error(400, "Invalid JSON body")
            return

        message = str(payload.get("message", "")).strip()
        agent_mode = str(payload.get("agent", "reactive")).lower()

        if not message:
            self._json_error(400, "Empty message")
            return

        t0 = time.perf_counter()

        try:
            if agent_mode == "learning":
                raw = learning.run(message)
                response_text = raw.get("response", str(raw))
                intent = raw.get("predicted_intent", "—")
            elif agent_mode == "llm":
                if not _llm_available or llm is None:
                    self._json_error(503, "LLMAgent is not available — check API key / provider config.")
                    return
                response_text = llm.run(message)
                intent = "—"
            else:                                   # default: reactive
                response_text = reactive.run(message)
                intent = "—"
        except Exception as exc:
            self._json_error(500, str(exc))
            return

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        self._json_ok({
            "response": response_text,
            "agent": agent_mode,
            "intent": intent,
            "elapsed_ms": elapsed_ms,
        })

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _serve_file(self, path: Path, content_type: str):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self._not_found()
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json_ok(self, data: dict):
        self._json_response(200, data)

    def _json_error(self, code: int, message: str):
        self._json_response(code, {"error": message})

    def _json_response(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self):
        self._json_error(404, "Not found")


# ── Entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    host, port = "localhost", 8080
    server = HTTPServer((host, port), AgentHandler)
    print(f"\nAgent UI running at http://{host}:{port}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
