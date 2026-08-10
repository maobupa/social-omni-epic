#!/usr/bin/env python3
"""Probe which model IDs are actually reachable, per route, before committing to a roster.

The matrix experiment pins seven roles to specific models across two routes (see
docs/../PROJECT_CONTEXT_9August26 handoff and the matrix plan):

  OpenAI direct  — learners, reflection writer, generator, oracle, partner, gates, embeddings
  Lightning      — the cross-lab judge only (the one role that must not be OpenAI)

Discovering mid-run that `gpt-5.4` isn't served, or that a model rejects `temperature`, costs
hours. This costs a few cents.

Two things are checked per (route, model):
  1. reachable     — a ~5-token completion returns without error
  2. temperature   — whether an explicit temperature is accepted. Matters because
                     fm.py:88-95 permanently disables temperature on an FM instance the first
                     time a model rejects it, so a shared instance can silently lose its setting.

Usage:
    uv run scripts/probe_models.py                     # both routes, default model lists
    uv run scripts/probe_models.py --route openai
    uv run scripts/probe_models.py --route lightning
    uv run scripts/probe_models.py --models gpt-5.4,gpt-5 --route openai
    uv run scripts/probe_models.py --json              # machine-readable, for MANIFEST.json

Read-only: makes API calls, writes nothing.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

import openai


# Candidate rosters. OpenAI direct wants BARE ids (the `openai/` prefix is a
# Lightning-gateway convention); Lightning wants prefixed ids.
OPENAI_CANDIDATES = [
    # --- in the roster ---
    "gpt-5.4",        # generator + oracle
    "gpt-5",          # strong learner
    "gpt-5-mini",     # mid learner (has an existing phase-0)
    "gpt-4o-mini",    # weak learner
    "gpt-4.1",        # partner + gates
    # --- NOT in the roster; probed for availability info only ---
    "gpt-5.6",        # stronger than the generator, but unused — 5.4 is the pinned generator
    "gpt-5-nano",     # alternative weak rung if the same-generation ladder is ever preferred
    "gpt-4.1-mini",   # candidate cross-generation held-out learner (Phase B)
    "gpt-3.5-turbo",  # see the plan on why this is not a viable learner (JSON-action failures)
]

LIGHTNING_CANDIDATES = [
    "google/gemini-3-flash-preview",  # intended judge
    "openai/gpt-5.4",
    "anthropic/claude-opus-4-8",      # documented Lightning-served; partner upgrade option
]

EMBED_CANDIDATES = ["text-embedding-3-small"]

_PROMPT = "Reply with the single word: ok"


def _client(route: str):
    """Build a client for one route. Returns (client, note) or (None, reason)."""
    if route == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None, "OPENAI_API_KEY not set"
        # base_url deliberately unset -> api.openai.com, which is where sotopia's
        # LiteLLM path already sends learner/partner turns.
        return openai.OpenAI(api_key=key, timeout=60.0, max_retries=0), "api.openai.com"
    if route == "lightning":
        key = os.getenv("LIGHTNING_AI_API_KEY")
        base = os.getenv("LIGHTNING_AI_BASE_URL")
        if not key or not base:
            return None, "LIGHTNING_AI_API_KEY / LIGHTNING_AI_BASE_URL not set"
        return openai.OpenAI(api_key=key, base_url=base, timeout=60.0, max_retries=0), base
    raise ValueError(route)


def _probe_chat(client, model: str) -> dict:
    """One reachability call, then one temperature call. Never raises."""
    out = {"model": model, "reachable": False, "temperature_ok": None, "error": None}

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _PROMPT}],
            max_completion_tokens=8,
        )
        out["reachable"] = True
        out["sample"] = (r.choices[0].message.content or "").strip()[:40]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:180]}"
        return out

    # Reachable — now see whether an explicit temperature is accepted.
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _PROMPT}],
            max_completion_tokens=8,
            temperature=0.7,
        )
        out["temperature_ok"] = True
    except Exception as e:
        out["temperature_ok"] = False
        out["temperature_error"] = str(e)[:120]
    return out


def _probe_embed(client, model: str) -> dict:
    out = {"model": model, "reachable": False, "temperature_ok": None, "error": None}
    try:
        r = client.embeddings.create(model=model, input="probe")
        out["reachable"] = True
        out["sample"] = f"dim={len(r.data[0].embedding)}"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:180]}"
    return out


def _print_table(route: str, note: str, rows: list[dict]) -> None:
    print(f"\n=== route: {route}  ({note}) ===")
    width = max((len(r["model"]) for r in rows), default=10)
    for r in rows:
        if r["reachable"]:
            temp = {True: "temp ok", False: "TEMP REJECTED", None: ""}[r["temperature_ok"]]
            detail = r.get("sample", "")
            print(f"  ✓ {r['model']:<{width}}  {temp:<14} {detail}")
        else:
            print(f"  ✗ {r['model']:<{width}}  {r['error']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", default="both", choices=["openai", "lightning", "both"])
    ap.add_argument("--models", default=None,
                    help="Comma-separated override of the candidate list for the chosen route.")
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = ap.parse_args()

    routes = ["openai", "lightning"] if args.route == "both" else [args.route]
    report: dict = {}

    for route in routes:
        client, note = _client(route)
        if client is None:
            print(f"\n=== route: {route} — SKIPPED ({note}) ===")
            report[route] = {"skipped": note}
            continue

        if args.models:
            models = [m.strip() for m in args.models.split(",") if m.strip()]
        else:
            models = OPENAI_CANDIDATES if route == "openai" else LIGHTNING_CANDIDATES

        rows = [_probe_chat(client, m) for m in models]

        if route == "openai" and not args.skip_embeddings and not args.models:
            rows += [_probe_embed(client, m) for m in EMBED_CANDIDATES]

        report[route] = {"endpoint": note, "results": rows}
        if not args.json:
            _print_table(route, note, rows)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    served = {
        route: [r["model"] for r in d.get("results", []) if r["reachable"]]
        for route, d in report.items()
    }
    print("\n--- served ---")
    for route, models in served.items():
        print(f"  {route}: {', '.join(models) if models else '(none)'}")
    print("\nPin the chosen roster into scripts/run_grid_matrix.sh and MANIFEST.json.")


if __name__ == "__main__":
    main()
