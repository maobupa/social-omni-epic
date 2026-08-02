#!/usr/bin/env python3
"""Post-hoc partner-fidelity audit over a finished curriculum run.

run_key_probes.py runs AUDIT 1 (leak) and AUDIT 2 (adherence) as a PRE-run gate over 5
hand-written probes. Nothing ever ran them over the actual curriculum transcripts, so we
have no measurement of whether the role-played partner held the key during the real run.
This script closes that gap, read-only, on already-generated data.

    AUDIT 1 — LEAK: did the partner name or clearly hint at a movement condition or
              hardening trigger (vs. arguing only from the surface_misdirection)?
    AUDIT 2 — EARLY YIELD: did the partner soften before any movement condition was
              genuinely met?  (the "unearned concession" failure)

Why it matters: a leaked condition or an early yield can produce a high GOAL score and a
passing key check, which labels the scenario too_easy / frontier_solved. That is a partner
failure being scored as a learner skill, and it biases difficulty DOWNWARD.

It also re-derives key_check_passed from the judge's own structured indices
(conditions_met / triggers_tripped / triggers_repaired) and reports where the stored
boolean disagrees — see docs on the Patch-12 discussion.

    uv run scripts/audit_gen90_partner_leaks.py                    # full run, both audits
    uv run scripts/audit_gen90_partner_leaks.py --limit 20         # smoke test
    uv run scripts/audit_gen90_partner_leaks.py --audits leak      # leak only (half the cost)

Read-only: writes a report to results/analysis/, never touches the bank.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.fm import FM
# Reuse the probe harness's prompts verbatim so numbers are comparable to the pre-run
# acceptance bar. Importing (not copying) keeps the two in sync if the prompts change.
from scripts.run_key_probes import (
    _LEAK_SYSTEM, _leak_user, _ADHERENCE_SYSTEM, _adherence_user,
)


def derive_key_check(kc: dict) -> bool:
    """key_check_passed as the judge PROMPT defines it: >=1 movement condition genuinely
    met AND no un-repaired hardening trigger. episode_runner reads the LLM's free-form
    boolean instead of deriving it, so the two can disagree."""
    cm = kc.get("conditions_met") or []
    unrepaired = set(kc.get("triggers_tripped") or []) - set(kc.get("triggers_repaired") or [])
    return bool(cm) and not unrepaired


def format_transcript(turns: list) -> str:
    return "\n".join(f"[T{t.get('turn')}] {t.get('speaker')}: {t.get('content')}" for t in turns)


def audit_attempt(fm: FM, scen: dict, att: dict, audits: set) -> dict:
    pk = scen.get("partner_key") or {}
    profiles = scen.get("agent_profiles") or []
    tidx = scen.get("target_agent_idx", 0) or 0
    partner_name = (profiles[1 - tidx].get("first_name") if len(profiles) > 1 else "the partner")
    text = format_transcript(att.get("transcript") or [])
    conds = pk.get("movement_conditions") or []
    trigs = pk.get("hardening_triggers") or []

    out = {
        "id": scen.get("id"),
        "title": scen.get("scenario_title"),
        "category": scen.get("classification"),
        "terminal_success": scen.get("terminal_success"),
        "attempt": att.get("attempt"),
        "goal": (att.get("scores") or {}).get("goal"),
        "rel": (att.get("scores") or {}).get("relationship"),
        "solved": att.get("solved"),
    }
    kc = att.get("key_check_result") or {}
    if kc:
        out["key_check_stored"] = bool(kc.get("key_check_passed"))
        out["key_check_derived"] = derive_key_check(kc)
        out["key_check_disagrees"] = out["key_check_stored"] != out["key_check_derived"]

    if not text or not conds:
        out["skipped"] = "no transcript or no conditions"
        return out

    if "leak" in audits:
        try:
            r = fm.query_json(_LEAK_SYSTEM, _leak_user(
                text, partner_name, pk.get("surface_misdirection", ""), conds, trigs),
                temperature=0.0)
            out["leaked"] = bool(r.get("leaked"))
            out["leak_evidence"] = r.get("evidence")
        except Exception as e:
            out["leak_error"] = str(e)

    if "adherence" in audits:
        try:
            r = fm.query_json(_ADHERENCE_SYSTEM, _adherence_user(
                text, partner_name, conds, trigs), temperature=0.0)
            out["early_yield"] = bool(r.get("early_yield"))
            out["early_yield_evidence"] = r.get("early_yield_evidence")
            out["ignored_trigger"] = bool(r.get("ignored_trigger"))
            out["ignored_trigger_evidence"] = r.get("ignored_trigger_evidence")
        except Exception as e:
            out["adherence_error"] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/gen90_expel")
    ap.add_argument("--judge-model", default="google/gemini-3-flash-preview",
                    help="Cross-lab auditor; must not be the learner model.")
    ap.add_argument("--audits", default="leak,adherence",
                    help="Comma-separated subset of: leak, adherence")
    ap.add_argument("--limit", type=int, default=0, help="Audit only the first N attempts")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    audits = {a.strip() for a in args.audits.split(",") if a.strip()}
    gen = sorted((Path(args.run_dir) / "bank" / "generated").glob("*.json"))
    if not gen:
        print(f"No scenarios under {args.run_dir}/bank/generated", file=sys.stderr)
        sys.exit(1)

    jobs = []
    for f in gen:
        scen = json.loads(f.read_text())
        for att in scen.get("attempts", []):
            jobs.append((scen, att))
    if args.limit:
        jobs = jobs[:args.limit]

    fm = FM(model=args.judge_model)
    print(f"Auditing {len(jobs)} attempts from {len(gen)} scenarios · audits={sorted(audits)} "
          f"· judge={args.judge_model}")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(audit_attempt, fm, s, a, audits) for s, a in jobs]
        for i, fu in enumerate(futs, 1):
            rows.append(fu.result())
            if i % 25 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}")

    out = Path(args.out or f"{args.run_dir}/analysis/partner_fidelity_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    n = len(rows)
    def pct(k):
        c = sum(1 for r in rows if r.get(k))
        return f"{c}/{n} ({100*c/n:.1f}%)" if n else "0"
    print(f"\n--- Partner fidelity, {n} attempts ---")
    if "leak" in audits:
        print(f"  leaked a hidden condition/trigger : {pct('leaked')}")
    if "adherence" in audits:
        print(f"  early yield (softened unearned)   : {pct('early_yield')}")
        print(f"  ignored a tripped trigger         : {pct('ignored_trigger')}")
    print(f"  key_check stored != derived       : {pct('key_check_disagrees')}")

    # The bias-relevant cell: partner failure on an attempt scored as a success.
    bad = [r for r in rows if r.get("solved") and (r.get("leaked") or r.get("early_yield"))]
    print(f"\n  SOLVED attempts with a partner-fidelity failure: {len(bad)}"
          f"  <- these are unearned solves")
    for r in bad[:10]:
        tags = ",".join(t for t in ("leaked", "early_yield") if r.get(t))
        print(f"    [{tags}] {str(r.get('title'))[:60]} (att {r.get('attempt')}, goal {r.get('goal')})")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
