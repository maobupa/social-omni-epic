import json
from .data_models import SocialScenario
from .fm import FM


# MOI is now a batch RANKER on social worth only (§6.3).
# The three-dimension gate (social_tension / novelty / learnability) is removed.
# The edit loop is removed. Worth is the only axis.
SYSTEM_PROMPT = """You are ranking candidate social scenarios on one axis only: SOCIAL WORTH.

A scenario has worth if the tension is one a thoughtful person would recognize as a real,
meaningful social situation — not contrived, not a gimmick, not a logic puzzle wearing a
social costume. Judge the human meaningfulness of the dynamic, NOT its difficulty and NOT
its novelty.

HIGH worth: the situation captures something true and painful about how real people get stuck
with each other — power imbalances that aren't just positional, face costs that aren't just
ego, relational dynamics that a perceptive person would recognize from life.

LOW worth: the scenario feels constructed as a puzzle, or the stakes are too abstract to care
about, or it reads like a management-training vignette rather than a genuine human situation.

Respond with ONLY valid JSON:
{"rankings": [{"index": 0, "worth": 8, "rationale": "one sentence"}, ...]}

Score worth 0–10. Rank all candidates; higher worth = better."""


def _format(s: SocialScenario) -> str:
    out = {}
    if s.scenario_title:
        out["scenario_title"] = s.scenario_title
    out.update({
        "scenario": s.scenario,
        "interaction_type": s.interaction_type,
        "relationship": s.relationship,
    })
    if any(sg is not None for sg in (s.structured_goals or [])):
        out["agent_structured_goals"] = [
            sg.model_dump() if sg else None for sg in s.structured_goals
        ]
        if s.goal_type:
            out["goal_type"] = s.goal_type
    else:
        out["agent_goals"] = s.agent_goals
    return json.dumps(out, indent=2)


class ModelOfInterestingness:
    def __init__(self, fm: FM, num_examples: int = 5):
        self.fm = fm
        self.num_examples = num_examples

    def rank_batch(
        self,
        candidates: list[SocialScenario],
    ) -> list[SocialScenario]:
        """Rank candidates by social worth. Returns list sorted best-first.

        On error, returns candidates in original order (never blocks the pipeline).
        """
        if not candidates:
            return candidates
        if len(candidates) == 1:
            return candidates

        parts = ["CANDIDATES TO RANK:"]
        for i, c in enumerate(candidates):
            parts.append(f"\n--- Candidate {i} ---")
            parts.append(_format(c))

        parts.append(
            f"\nRank all {len(candidates)} candidates by SOCIAL WORTH (0–10). "
            "Return JSON: {\"rankings\": [{\"index\": 0, \"worth\": ..., \"rationale\": \"...\"}, ...]}"
        )

        try:
            d = self.fm.query_json(SYSTEM_PROMPT, "\n".join(parts), temperature=0.3)
            rankings = d.get("rankings", [])
            if not rankings:
                return candidates
            # Sort by worth descending; carry over worth score for logging.
            indexed = {r.get("index"): r.get("worth", 0) for r in rankings}
            sorted_candidates = sorted(
                enumerate(candidates),
                key=lambda ic: indexed.get(ic[0], 0),
                reverse=True,
            )
            return [c for _, c in sorted_candidates]
        except Exception:
            return candidates

    def evaluate(
        self, new_scenario: SocialScenario, similar: list[SocialScenario]
    ) -> tuple[bool, str, list[str]]:
        """Backward-compat single-scenario gate (used by old pipeline paths).

        Passes iff social worth ≥ 5. Never blocks on error.
        """
        ranked = self.rank_batch([new_scenario])
        if not ranked:
            return True, "no candidates to rank", []
        # Single candidate — just check worth via rank_batch internal scoring.
        # Re-run as a single query to get the worth score.
        try:
            parts = [
                "CANDIDATE:",
                _format(new_scenario),
                "\nRate this scenario's SOCIAL WORTH (0–10).",
                'Return JSON: {"rankings": [{"index": 0, "worth": 0, "rationale": "..."}]}',
            ]
            d = self.fm.query_json(SYSTEM_PROMPT, "\n".join(parts), temperature=0.3)
            rankings = d.get("rankings", [])
            worth = rankings[0].get("worth", 5) if rankings else 5
            reason = rankings[0].get("rationale", "") if rankings else ""
            passed = float(worth) >= 5.0
            return passed, reason, []
        except Exception as e:
            return True, f"MoI error (defaulting to pass): {e}", []
