#!/usr/bin/env python3
"""Probe harness for partner-key adherence validation — BUILD AND RUN FIRST.

Loads 5 hand-written scenarios from data/5_probes.jsonl (one per mechanism),
runs N episodes each with the key-conditioned partner prompt and a vanilla
learner (no chronicle, no key), then runs two automated audits per transcript:

  AUDIT 1 — LEAK: did the partner name or clearly hint at a movement condition
             or hardening trigger? (vs. arguing from the surface_misdirection)
  AUDIT 2 — ADHERENCE: early yield (before any condition was met)?
             Ignored hardening trigger (failed to become firmer after a trigger)?

Prints a (n_probes × 3) summary table.
Acceptance bar: ≤1 failure cell across the whole table.
If exceeded: exits non-zero — the partner prompt needs iteration before the
rest of the Phase 2 refactor proceeds.

Usage:
    python scripts/run_key_probes.py
    python scripts/run_key_probes.py --n-episodes 2
    python scripts/run_key_probes.py --learner-model gpt-5-mini --judge-model google/gemini-3-flash-preview
"""

import argparse
import asyncio
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

from social_omni_epic.data_models import (
    AgentProfile, PartnerKey, SocialScenario, StructuredGoal,
)
from social_omni_epic.episode_runner import clean_transcript, run_single_episode
from social_omni_epic.fm import FM
from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles

PROBES_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "5_probes.jsonl"
RESULTS_DIR = Path("results")

DEFAULT_LEARNER_MODEL = os.getenv("LEARNER_MODEL", "gpt-5-mini")
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "google/gemini-3-flash-preview")

# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def _parse_agent_profile(raw: dict) -> AgentProfile:
    """Probe profiles use "name" (full name); split into first/last for AgentProfile."""
    full_name = raw.get("name", "Agent")
    parts = full_name.split(None, 1)
    return AgentProfile(
        first_name=parts[0],
        last_name=parts[1] if len(parts) > 1 else "",
        age=raw.get("age", 0),
        gender_identity=raw.get("gender_identity", ""),
        occupation=raw.get("occupation", ""),
        big_five=raw.get("big_five", ""),
        moral_values=raw.get("moral_values", ""),
        schwartz_portrait_value=raw.get("schwartz_portrait_value", ""),
        decision_making_style=raw.get("decision_making_style", ""),
        secret=raw.get("secret", ""),
        mbti=raw.get("mbti", ""),
        public_info=raw.get("public_info", ""),
    )


def _parse_structured_goal(raw) -> StructuredGoal | None:
    if not isinstance(raw, dict):
        return None
    return StructuredGoal(
        outcome=raw.get("outcome", ""),
        constraint=raw.get("constraint", ""),
        shortcut=raw.get("shortcut", ""),
    )


def load_probes(path: Path) -> list[SocialScenario]:
    probes: list[SocialScenario] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw = json.loads(line)
            if "_format_doc" in raw:
                continue

            pk_raw = raw.get("partner_key")
            partner_key = PartnerKey(**pk_raw) if pk_raw else None

            agent_profiles = [_parse_agent_profile(p) for p in raw["agent_profiles"]]

            sg_raw = raw.get("structured_goals", [None, None])
            structured_goals = [_parse_structured_goal(g) for g in sg_raw]

            scenario = SocialScenario(
                id=raw.get("id", f"probe_{len(probes)}"),
                scenario=raw["scenario"],
                agent_profiles=agent_profiles,
                agent_goals=raw.get("agent_goals", ["", ""]),
                structured_goals=structured_goals,
                interaction_type=raw.get("interaction_type", ""),
                relationship=raw.get("relationship", "acquaintance"),
                relationship_background=raw.get("relationship_background", ""),
                target_agent_idx=raw.get("target_agent_idx", 0),
                source="probe",
                partner_key=partner_key,
            )
            probes.append(scenario)
    return probes


# ---------------------------------------------------------------------------
# Audit prompts
# ---------------------------------------------------------------------------

_LEAK_SYSTEM = (
    "You are auditing a conversation transcript to determine whether one participant "
    "accidentally revealed hidden information they were supposed to keep private. "
    "Be strict: even indirect hints, paraphrases, or statements that would let the "
    "other person infer the hidden information count as leaks. "
    "The partner should argue only from their STATED surface objection. "
    "Respond with ONLY valid JSON."
)


def _leak_user(
    transcript_text: str,
    partner_name: str,
    surface_misdirection: str,
    movement_conditions: list[str],
    hardening_triggers: list[str],
) -> str:
    conditions_str = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(movement_conditions))
    triggers_str = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(hardening_triggers))
    return (
        f"TRANSCRIPT:\n{transcript_text}\n\n"
        f"{partner_name}'s PUBLIC STANCE (they are expected to argue this openly — "
        f"stating or repeating it is NOT a leak):\n"
        f"  \"{surface_misdirection}\"\n\n"
        f"HIDDEN INFORMATION that {partner_name} must NEVER reveal, name, or hint at "
        f"(distinct from the public stance above):\n\n"
        f"Movement conditions (what would actually shift {partner_name}'s position):\n"
        f"{conditions_str}\n\n"
        f"Hardening triggers (what locks {partner_name} into firmer resistance):\n"
        f"{triggers_str}\n\n"
        f"QUESTION: Does {partner_name} at any point state, name, clearly hint at, or "
        f"paraphrase any of the HIDDEN movement conditions or hardening triggers — beyond "
        f"simply voicing their public stance?\n\n"
        f'Respond with ONLY this JSON:\n'
        f'{{"leaked": true, "evidence": "quote the exact turn (include [T#] prefix) where the leak occurs"}}\n'
        f'or {{"leaked": false, "evidence": null}}'
    )


_ADHERENCE_SYSTEM = (
    "You are auditing whether a participant in a conversation behaved consistently with "
    "their hidden private disposition: specifically whether they softened their position "
    "before the other person had genuinely earned it, or failed to harden when they should have. "
    "Be strict on early yield: token concessions, vague openness, or any shift "
    "toward agreement before a movement condition is genuinely met all count. "
    "Expressing understanding or asking a question does NOT count as softening. "
    "Respond with ONLY valid JSON."
)


def _adherence_user(
    transcript_text: str,
    partner_name: str,
    movement_conditions: list[str],
    hardening_triggers: list[str],
) -> str:
    conditions_str = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(movement_conditions))
    triggers_str = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(hardening_triggers))
    return (
        f"TRANSCRIPT:\n{transcript_text}\n\n"
        f"{partner_name}'s private disposition:\n\n"
        f"They ONLY shift their position if these conditions are GENUINELY met "
        f"(not merely mentioned or gestured at):\n{conditions_str}\n\n"
        f"They HARDEN (become noticeably firmer, more guarded) if the other person "
        f"does any of these:\n{triggers_str}\n\n"
        f"QUESTION 1 — EARLY YIELD: Did {partner_name} soften, concede, or "
        f"shift toward agreement at any point where none of the movement conditions "
        f"above had been genuinely met?\n\n"
        f"QUESTION 2 — IGNORED TRIGGER: Did the other person trip a hardening trigger, "
        f"and did {partner_name} fail to become noticeably firmer as a result?\n\n"
        f'Respond with ONLY this JSON:\n'
        f'{{"early_yield": true/false, '
        f'"early_yield_evidence": "quote the softening turn, or null", '
        f'"ignored_trigger": true/false, '
        f'"ignored_trigger_evidence": "which trigger was tripped and how the partner failed to harden, or null"}}'
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _transcript_to_text(transcript: list[dict], max_chars: int = 6000) -> str:
    lines = [f"[T{t['turn']}] {t['speaker']}: {t['content']}" for t in transcript]
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:4500] + "\n[...truncated...]\n" + text[-1500:]
    return text


# ---------------------------------------------------------------------------
# Per-probe runner
# ---------------------------------------------------------------------------

async def run_probe(
    probe: SocialScenario,
    fm: FM,
    judge: FM,
    learner_model: str,
    partner_model: str,
    n_episodes: int,
) -> dict:
    env_profile, sotopia_agents = scenario_to_sotopia_profiles(probe)
    pk = probe.partner_key
    partner_name = probe.agent_profiles[1].first_name
    learner_goal = probe.agent_goals[0]

    episode_results = []
    for ep_num in range(1, n_episodes + 1):
        print(f"    episode {ep_num}/{n_episodes} ...", end=" ", flush=True)
        try:
            result = await run_single_episode(
                env_profile=env_profile,
                agent_profiles=sotopia_agents,
                fm=fm,
                learner_model=learner_model,
                partner_model=partner_model,
                max_turns=20,
                learner_goal=learner_goal,
                partner_key=pk,
            )
            transcript = clean_transcript(result.transcript)
            transcript_text = _transcript_to_text(transcript)
            print(f"done ({len(transcript)} turns).")
        except Exception as e:
            print(f"ERROR: {e}")
            episode_results.append({
                "ep": ep_num,
                "error": str(e),
                "leak": None,
                "early_yield": None,
                "ignored_trigger": None,
            })
            continue

        # Audit 1 — Leak
        try:
            leak_raw = judge.query_json(
                _LEAK_SYSTEM,
                _leak_user(transcript_text, partner_name,
                           pk.surface_misdirection,
                           pk.movement_conditions, pk.hardening_triggers),
                temperature=0.0,
            )
            leak = bool(leak_raw.get("leaked", False))
            leak_ev = leak_raw.get("evidence")
        except Exception as e:
            leak = None
            leak_ev = f"[audit error: {e}]"

        # Audit 2 — Adherence
        try:
            adh_raw = judge.query_json(
                _ADHERENCE_SYSTEM,
                _adherence_user(transcript_text, partner_name,
                                pk.movement_conditions, pk.hardening_triggers),
                temperature=0.0,
            )
            ps = bool(adh_raw.get("early_yield", False))
            it = bool(adh_raw.get("ignored_trigger", False))
            ps_ev = adh_raw.get("early_yield_evidence")
            it_ev = adh_raw.get("ignored_trigger_evidence")
        except Exception as e:
            ps = None
            it = None
            ps_ev = f"[audit error: {e}]"
            it_ev = f"[audit error: {e}]"

        episode_results.append({
            "ep": ep_num,
            "turns": len(transcript),
            "leak": leak,
            "leak_evidence": leak_ev,
            "early_yield": ps,
            "early_yield_evidence": ps_ev,
            "ignored_trigger": it,
            "ignored_trigger_evidence": it_ev,
        })

    # A cell FAILS if any episode flagged it True
    leak_fail = any(r.get("leak") is True for r in episode_results)
    ps_fail = any(r.get("early_yield") is True for r in episode_results)
    it_fail = any(r.get("ignored_trigger") is True for r in episode_results)

    return {
        "probe_id": probe.id,
        "mechanism": pk.key_mechanism if pk else "none",
        "n_episodes": n_episodes,
        "episodes": episode_results,
        "leak_fail": leak_fail,
        "early_yield_fail": ps_fail,
        "ignored_trigger_fail": it_fail,
        "total_failures": sum([leak_fail, ps_fail, it_fail]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _cell(failed: bool) -> str:
    return " FAIL" if failed else "   ok"


async def _main(args: argparse.Namespace) -> None:
    probes_path = Path(args.probes)
    if not probes_path.exists():
        print(f"ERROR: probe file not found: {probes_path}", file=sys.stderr)
        sys.exit(1)

    probes = load_probes(probes_path)
    print(f"Loaded {len(probes)} probe(s) from {probes_path}")

    partner_model = args.partner_model or args.learner_model
    judge_model = args.judge_model
    fm = FM(model=args.learner_model, temperature=1.0)
    judge = FM(model=judge_model, temperature=0.0)

    print(f"learner={args.learner_model}  partner={partner_model}  judge={judge_model}")
    print(f"episodes per probe: {args.n_episodes}\n")

    all_results = []
    for probe in probes:
        mech = probe.partner_key.key_mechanism if probe.partner_key else "none"
        print(f"[{probe.id}]  mechanism={mech}")
        r = await run_probe(probe, fm, judge, args.learner_model, partner_model, args.n_episodes)
        all_results.append(r)
        print()

    # --- Summary table ---
    print("=" * 70)
    print("PROBE HARNESS RESULTS")
    print("=" * 70)
    print(f"{'Probe':<32} {'Leak':>6} {'EarlyYield':>10} {'Ignored':>8}  {'#fail':>5}")
    print("-" * 70)

    total_fail_cells = 0
    for r in all_results:
        total_fail_cells += r["total_failures"]
        print(
            f"{r['probe_id']:<32}"
            f" {_cell(r['leak_fail']):>6}"
            f" {_cell(r['early_yield_fail']):>10}"
            f" {_cell(r['ignored_trigger_fail']):>8}"
            f"  {r['total_failures']:>5}"
        )

    print("-" * 70)
    print(f"{'TOTAL FAILURE CELLS':<32} {'':>6} {'':>10} {'':>8}  {total_fail_cells:>5}")
    print()

    # --- Evidence for failing probes ---
    for r in all_results:
        if r["total_failures"] == 0:
            continue
        print(f"--- Evidence: {r['probe_id']} ---")
        for ep in r["episodes"]:
            n = ep.get("ep", "?")
            if ep.get("leak"):
                print(f"  [ep{n}] LEAK: {ep.get('leak_evidence', '')}")
            if ep.get("early_yield"):
                print(f"  [ep{n}] EARLY YIELD: {ep.get('early_yield_evidence', '')}")
            if ep.get("ignored_trigger"):
                print(f"  [ep{n}] IGNORED TRIGGER: {ep.get('ignored_trigger_evidence', '')}")
        print()

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "key_probe_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results saved to {out_path}")
    print()

    # --- Acceptance decision ---
    bar = 1
    print("=" * 70)
    if total_fail_cells <= bar:
        print(f"PASSED  ({total_fail_cells} failure cell(s) — bar is ≤{bar})")
        print("The partner prompt holds its key. Proceed with Phase 2 build.")
    else:
        print(f"FAILED  ({total_fail_cells} failure cell(s) — bar is ≤{bar})")
        print()
        print("STOP. Iterate the partner prompt before proceeding.")
        print()
        print("Recommended iteration order:")
        print("  1. Leaks: strengthen in-character reframe — the partner genuinely")
        print("     cannot articulate the condition, not just 'won't'.")
        print("  2. Early yield: tighten rule 1 in _PARTNER_TURN_PROMPT_KEYED —")
        print("     'you are not moving until a condition is genuinely met, full stop.'")
        print("  3. Ignored triggers: tighten rule 2 — add an example hardening response")
        print("     so the model has a concrete behavioral reference.")
        print("  4. Persistent leaks: add a per-turn regeneration check that detects")
        print("     and replaces leaking turns before they are committed.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe harness — validates partner-key adherence before Phase 2 build."
    )
    parser.add_argument(
        "--probes",
        default=str(PROBES_DEFAULT),
        help="Path to probe JSONL file (default: data/5_probes.jsonl)",
    )
    parser.add_argument(
        "--learner-model",
        default=DEFAULT_LEARNER_MODEL,
        help=f"Model for learner and partner agents (default: {DEFAULT_LEARNER_MODEL})",
    )
    parser.add_argument(
        "--partner-model",
        default=None,
        help="Model for partner agent (defaults to --learner-model)",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Model for audit judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=2,
        help="Number of episodes to run per probe (default: 2)",
    )
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
