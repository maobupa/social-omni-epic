"""Export bridge: turn completed curriculum scenarios into ExpeL-extract-ready artifacts.

The gen-90 curriculum produces `SocialScenario` objects + per-attempt `loop_info`. ExpeL
`extract` (scripts/run_expel_baseline.py) consumes a `trajectories.json` in the
`expel_baseline` format. This module bridges the two and writes phase0-parity exports.

Design: the canonical unit is the per-scenario record written to `bank/generated/<id>.json`
at completion (crash-safe, idempotent). Aggregate artifacts (trajectories.json, summary.json)
are rebuilt from that folder at every checkpoint, so they survive resume without any
in-memory state. Layout written here:

    bank/generated/<id>.json — per-scenario record (full scenario − embedding, + attempts + lineage)
    chronicles/<id>.md       — skills_final_md (reflexion strings under ExpeL)
    trajectories.json        — ExpeL succeeded/failed/idx2task over ALL completed scenarios
    summary.json             — classification/operator counts, LP stats, models, n

A completed scenario = too_easy ∪ frontier ∪ beyond_frontier; generation-failed and
quarantined scenarios are never written to bank/generated.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from .data_models import SocialScenario
from .expel_baseline import ExpelTrajectory, _format_transcript, trajectories_to_dict


def _learner_goal(scenario: SocialScenario) -> str:
    goals = scenario.agent_goals or []
    li = scenario.target_agent_idx or 0
    return goals[li] if li < len(goals) else (goals[0] if goals else "")


def scenario_record(scenario: SocialScenario, loop_info: dict) -> dict:
    """phase0-style per-scenario record: full scenario (minus embedding) + transcripts + lineage."""
    rec = scenario.model_dump(exclude={"embedding"})
    atts = [a for a in (loop_info.get("skill_attempts") or []) if isinstance(a, dict)]
    rec["learner_goal"] = _learner_goal(scenario)
    rec["attempts"] = [
        {
            "attempt": a.get("attempt"),
            "transcript": a.get("transcript_clean"),
            "scores": a.get("diagnostics_scores"),
            "solved": a.get("solved"),
            "key_check_result": a.get("key_check_result"),
            # ExpeL within-episode Reflexion generated AFTER this attempt (fed into the next).
            "reflexion": a.get("reflexion"),
        }
        for a in atts
    ]
    rec["scores"] = atts[-1].get("diagnostics_scores") if atts else None
    rec["lp_improved_votes"] = loop_info.get("lp_improved_votes")
    rec["n_error_votes"] = loop_info.get("n_error_votes")
    return rec


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically via tmp → rename (POSIX-safe, no partial reads)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.rename(path)


def write_scenario_record(scenario: SocialScenario, loop_info: dict, generated_dir: Path) -> None:
    generated_dir.mkdir(parents=True, exist_ok=True)
    rec = scenario_record(scenario, loop_info)
    rec["status"] = "completed"
    _atomic_write(generated_dir / f"{scenario.id}.json", rec)


def write_live_record(
    path: Path,
    scenario: SocialScenario,
    loop_info: dict,
    live_transcript: list[dict] | None = None,
) -> None:
    """Atomically overwrite the in-progress record for a scenario mid-episode.

    Writes completed attempts from loop_info plus the current turn-by-turn
    transcript of the attempt in flight (live_transcript=None clears it).
    Called from on_turn (every turn) and on_attempt_done (attempt boundary).
    """
    rec = scenario.model_dump(exclude={"embedding"})
    atts = [a for a in (loop_info.get("skill_attempts") or []) if isinstance(a, dict)]
    rec["learner_goal"] = _learner_goal(scenario)
    rec["status"] = "in_progress"
    rec["attempts"] = [
        {
            "attempt": a.get("attempt"),
            "transcript": a.get("transcript_clean"),
            "scores": a.get("diagnostics_scores"),
            "solved": a.get("solved"),
            "key_check_result": a.get("key_check_result"),
            "reflexion": a.get("reflexion"),
        }
        for a in atts
    ]
    rec["live_transcript"] = live_transcript or []
    rec["scores"] = atts[-1].get("diagnostics_scores") if atts else None
    _atomic_write(path, rec)


def write_chronicle(scenario: SocialScenario, chronicles_dir: Path) -> None:
    md = (scenario.skills_final_md or "").strip()
    if not md:
        return
    chronicles_dir.mkdir(parents=True, exist_ok=True)
    (chronicles_dir / f"{scenario.id}.md").write_text(md)


# --------------------------------------------------------------------------- #
# Aggregate artifacts — rebuilt from bank/generated/*.json (resume-safe)       #
# --------------------------------------------------------------------------- #

def _load_records(generated_dir: Path) -> list[dict]:
    """Load completed (non-stub) scenario records from bank/generated/."""
    if not generated_dir.exists():
        return []
    recs = []
    for p in sorted(generated_dir.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
            if rec.get("status") == "in_progress":
                continue  # skip live stubs (in-flight or orphaned from a crash)
            recs.append(rec)
        except Exception:
            continue
    return recs


# Extraction success label. MUST match run_expel_chronicle.py's _is_terminal_success
# (= the Base90 bank's labeling function) for a controlled ExpeL-Generated90 vs Base90
# comparison. NOTE: this deliberately DROPS the key_check term that `terminal_success`
# carries on keyed scenarios — key_check is a curriculum-integrity signal (it drives
# classification/difficulty), not an extraction-success signal, and seeds have no key so
# Base90 could never apply it. Both banks therefore label trajectory success identically.
GOAL_THRESHOLD = 7.0
REL_THRESHOLD = 0.0


def _trajectory_success(scores: dict) -> bool:
    s = scores or {}
    return (float(s.get("goal", 0.0)) >= GOAL_THRESHOLD
            and float(s.get("relationship", 0.0)) >= REL_THRESHOLD)


def build_trajectories_from_records(recs: list[dict]) -> dict:
    """Build the ExpeL trajectory pool (serialized via trajectories_to_dict).

    Each per-attempt entry → one ExpelTrajectory; `success` is recomputed from the attempt's
    scores as GOAL≥7 ∧ REL≥0 (NOT the per-attempt `solved` flag, which also requires key_check
    on keyed scenarios — see GOAL_THRESHOLD note). too_easy scenarios contribute a single
    successful trajectory (trial 0).
    """
    succeeded: dict[int, list[ExpelTrajectory]] = {}
    failed: dict[int, list[ExpelTrajectory]] = {}
    idx2task: dict[int, str] = {}
    completed: set[int] = set()

    for idx, rec in enumerate(recs):
        atts = rec.get("attempts") or []
        if not atts:
            continue
        task = rec.get("scenario", "")
        lg = rec.get("learner_goal", "")
        sid = rec.get("id", f"gen_{idx}")
        idx2task[idx] = task
        succeeded.setdefault(idx, [])
        failed.setdefault(idx, [])
        for a in atts:
            scores = a.get("scores") or {}
            success = _trajectory_success(scores)
            traj = ExpelTrajectory(
                scenario_id=sid,
                task_idx=idx,
                task=task,
                learner_goal=lg,
                transcript_text=_format_transcript(a.get("transcript") or []),
                success=success,
                goal_score=float(scores.get("goal", 0.0)),
                trial=int(a.get("attempt", 1)) - 1,
                reflections=[],
            )
            (succeeded if success else failed)[idx].append(traj)
        completed.add(idx)

    return trajectories_to_dict(succeeded, failed, idx2task, completed)


def build_summary_from_records(recs: list[dict], learner_model: str = "",
                               judge_model: str = "") -> dict:
    class_counts: dict[str, int] = {}
    op_counts: dict[str, dict[str, int]] = {}
    lp_values: list[float] = []
    # Extraction-yield predictors (computed at the trajectory-success label, goal∧rel):
    #   n_success_tasks  — scenarios with ≥1 success attempt → feed the success-critique stage
    #   n_compare_pairs  — scenarios with BOTH a success and a failure → feed the compare stage
    #   n_frontier_unsolved — frontier (LP>0) scenarios with NO success → the LP-vs-≥7 gap;
    #                          curriculum-valuable but invisible to extraction
    n_success_tasks = 0
    n_compare_pairs = 0
    n_frontier_unsolved = 0
    for rec in recs:
        c = rec.get("classification") or "unknown"
        class_counts[c] = class_counts.get(c, 0) + 1
        op = rec.get("mutation_operator") or "seed"
        op_counts.setdefault(op, {})
        op_counts[op][c] = op_counts[op].get(c, 0) + 1
        if c in ("frontier", "beyond_frontier") and rec.get("lp_value") is not None:
            lp_values.append(float(rec["lp_value"]))
        atts = rec.get("attempts") or []
        succ = [a for a in atts if _trajectory_success(a.get("scores") or {})]
        fail = [a for a in atts if not _trajectory_success(a.get("scores") or {})]
        if succ:
            n_success_tasks += 1
        if succ and fail:
            n_compare_pairs += 1
        if c == "frontier" and not succ:
            n_frontier_unsolved += 1
    return {
        "n": len(recs),
        "classification_counts": class_counts,
        "per_operator_classification_counts": op_counts,
        "lp_stats": {
            "mean": round(statistics.fmean(lp_values), 4) if lp_values else 0.0,
            "std": round(statistics.pstdev(lp_values), 4) if len(lp_values) > 1 else 0.0,
            "n": len(lp_values),
        },
        "extraction_yield": {
            "n_success_tasks": n_success_tasks,
            "n_compare_pairs": n_compare_pairs,
            "n_frontier_unsolved": n_frontier_unsolved,
        },
        "learner_model": learner_model,
        "judge_model": judge_model,
        "success_label": "goal7_rel0",
    }


def flush_aggregates(run_dir: Path, learner_model: str = "", judge_model: str = "") -> int:
    """Rebuild trajectories.json + summary.json from bank/generated/*.json. Returns n records."""
    run_dir = Path(run_dir)
    recs = _load_records(run_dir / "bank" / "generated")
    (run_dir / "trajectories.json").write_text(
        json.dumps(build_trajectories_from_records(recs), indent=2, default=str)
    )
    (run_dir / "summary.json").write_text(
        json.dumps(build_summary_from_records(recs, learner_model, judge_model), indent=2)
    )
    return len(recs)
