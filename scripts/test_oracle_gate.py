#!/usr/bin/env python3
"""Ground-truth test for the oracle-solvability gate, on two known gen-90 scenarios.

Both are labelled `beyond_frontier` in the frozen bank and that label cannot tell them apart. The
gate must:

  57ed171e  PASS. Ordinary learners with no cheat sheet scored 10/9/10/10 across all four gen-90
            attempts, so it is demonstrably winnable. An INFORMED strong learner failing here would
            mean the gate (or the partner) over-rejects.
  e7179e01  REJECT. The partner defers to an off-scene department administrator; the learner reached
            key_check_passed=True and still scored goal=0. Best across four attempts was 4.0.

If the gate cannot separate them it is not measuring solvability, and there is no point generating
anything with it — which is why this runs before the smoke stage.

Caveat: both are v1 records with no `internal_state`, so the oracle's cheat sheet is the
movement_conditions alone (see render_answer — it deliberately does NOT substitute
surface_misdirection, which is the decoy). v2 scenarios will hand the oracle strictly more.

    uv run scripts/test_oracle_gate.py
    uv run scripts/test_oracle_gate.py --oracle-model gpt-5.4 --max-tries 3
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.data_models import SocialScenario          # noqa: E402
from social_omni_epic.fm import make_fm                                # noqa: E402
from social_omni_epic.oracle_gate import probe_solvability        # noqa: E402

# The pair has to be chosen on DEMONSTRATED winnability, not on a close-read judgement of
# "genuine difficulty". First attempt at this test used a156533b as the positive case because the
# bank diagnostic calls it "genuinely hard" — but its best goal across four gen-90 attempts was
# 4.0, same as e7179e01's. Two scenarios that both cap at 4.0 cannot be separated by ANY goal>=7
# threshold, so that pair was undiagnostic regardless of whether the gate works.
CASES = [
    # Ordinary learners, no cheat sheet, scored 10/9/10/10 across all four gen-90 attempts. If an
    # INFORMED strong learner cannot clear 7 here, the gate is over-rejecting.
    ("57ed171e", True, "goals 10,9,10,10 in gen-90 -> demonstrably winnable, must PASS"),
    # Partner defers to an off-scene department administrator; the learner reached key_pass=True and
    # still scored goal=0. Best across four attempts was 4.0.
    ("e7179e01", False, "off-scene decider, best goal 4.0 -> unwinnable, must REJECT"),
]


def load(bank: Path, prefix: str) -> SocialScenario:
    hits = sorted(bank.glob(f"{prefix}*.json"))
    if not hits:
        raise SystemExit(f"no bank record starting with {prefix} under {bank}")
    return SocialScenario(**json.loads(hits[0].read_text()))


async def main_async(args) -> int:
    from social_omni_epic.episode_runner import run_single_episode
    from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles

    bank = Path(args.bank)
    # Generation-side FM is unused by the probe beyond being passed through; the judge is what
    # scores. Kept cross-lab (judge on a different provider than the oracle learner).
    fm = make_fm(model=args.oracle_model)
    fm_judge = make_fm(model=args.judge_model, temperature=0.3)

    rows = []
    for prefix, expect_pass, why in CASES:
        scn = load(bank, prefix)
        print(f"\n=== {prefix} — expect {'PASS' if expect_pass else 'REJECT'} ({why}) ===")
        print(f"    {scn.scenario[:150]}")
        v = await probe_solvability(
            scn,
            run_single_episode=run_single_episode,
            scenario_to_sotopia_profiles=scenario_to_sotopia_profiles,
            fm=fm, fm_judge=fm_judge,
            oracle_model=args.oracle_model,
            partner_model=args.partner_model,
            max_turns=args.max_turns,
            max_tries=args.max_tries,
            tag=f"[{prefix}]",
        )
        ok = (v.admit == expect_pass)
        rows.append((prefix, expect_pass, v, ok))
        print(f"    verdict: admit={v.admit} best_goal={v.best_goal:.1f} "
              f"goals={v.goal_scores} error={v.error}  → {'OK' if ok else 'MISMATCH'}")

    print("\n" + "=" * 72)
    for prefix, expect, v, ok in rows:
        print(f"  {'✓' if ok else '✗'} {prefix}  expected={'pass' if expect else 'reject'}  "
              f"got={'pass' if v.admit else 'reject'}  goals={v.goal_scores}")
    n_ok = sum(1 for *_, ok in rows if ok)
    print(f"\n{n_ok}/{len(rows)} as expected.")
    if n_ok != len(rows):
        print("The gate does not separate unwinnable from hard. Fix it before generating anything.",
              file=sys.stderr)
        return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", default="results/gen90_expel/bank/generated")
    ap.add_argument("--oracle-model", default="gpt-5.4")
    ap.add_argument("--partner-model", default="gpt-5.4-mini")
    ap.add_argument("--judge-model", default="google/gemini-3-flash-preview")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--max-tries", type=int, default=3)
    args = ap.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
