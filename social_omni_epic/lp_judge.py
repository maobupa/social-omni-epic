"""Learning-Progress (LP) judge (§2): pairwise cross-lab curriculum signal.

Key-blind. Compares attempt 1 against each later attempt in both presentation
orders; order-swap disagreement collapses to 'no_difference' (kills position
bias). LP = improved_votes / total_votes ∈ [0,1].

Usage:
    result = await compute_lp(fm_judge, scenario, transcripts, learner_goal, relational_stakes)
"""
import asyncio
from dataclasses import dataclass, field
from typing import Optional

from .data_models import SocialScenario
from .fm import FM


class LPAllErrorsError(Exception):
    """Raised when every LP judge vote errored (after the one-shot revote). The caller
    routes this to the episode-quarantine path — LP is uncomputable, so the anchor must
    not be charged a (0, K) penalty for what is an infrastructure failure."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VoteRecord:
    pair: tuple[int, int]   # (1, j) — attempt indices (1-based)
    order: str              # "A=1,B=j" or "A=j,B=1"
    verdict: str            # "A" | "B" | "no_difference"
    rationale: str
    is_error: bool = False  # True when the judge call raised (verdict forced to no_difference)


@dataclass
class LPResult:
    lp_value: float          # improved_votes / total_votes; 0.0 if no pairs
    improved_votes: int
    total_votes: int
    n_pairs: int
    votes: list[VoteRecord] = field(default_factory=list)
    n_error_votes: int = 0   # votes whose judge call raised (after the one-shot revote)


# ---------------------------------------------------------------------------
# Judge prompts (§2.3)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are comparing two attempts at the same social interaction to judge which one "
    "better served the actor's goal. Judge holistically: achieving the stated goal AND "
    "preserving the relationship and the other party's goodwill both count. An attempt "
    "that extracts the outcome while damaging trust is worse than one that makes real "
    "progress on both. The mirror also holds: if NEITHER attempt makes meaningful "
    "progress toward the actor's stated objective, differences in politeness, rapport, "
    "warmth, or graceful disengagement do NOT make one attempt better — answer "
    "no_difference. 'Better' requires genuinely more progress toward the objective while "
    "at least preserving the relationship. Decide only from what is observable in the "
    "transcripts. Respond with ONLY valid JSON."
)

_JUDGE_USER = """THE ACTOR'S GOAL: {learner_goal}
RELATIONAL STAKES: {relational_stakes}

TRANSCRIPT A:
{transcript_a}

TRANSCRIPT B:
{transcript_b}

QUESTION: Which transcript shows the actor better achieving their goal while preserving the relationship — A, B, or is there no meaningful difference? If both attempts fail the actor's objective to a similar degree, answer no_difference regardless of which is warmer or more graceful.
Respond with ONLY this JSON: {{"verdict": "A" | "B" | "no_difference", "rationale": "one sentence"}}"""

_MAX_TRANSCRIPT_CHARS = 4000
_HEAD_CHARS = 3000
_TAIL_CHARS = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int = _MAX_TRANSCRIPT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:_HEAD_CHARS] + "\n[...]\n" + text[-_TAIL_CHARS:]


def _transcript_to_str(transcript: list[dict]) -> str:
    return "\n".join(
        f"[T{t.get('turn', i)}] {t.get('speaker', '?')}: {t.get('content', '')}"
        for i, t in enumerate(transcript)
    )


def _cast_vote(
    fm: FM,
    attempt1_text: str,
    attemptj_text: str,
    learner_goal: str,
    relational_stakes: str,
    pair: tuple[int, int],
    order: str,
    temperature: float = 0.3,
) -> VoteRecord:
    """Run one judge call; wrap exceptions as no_difference."""
    if order == "A=1,B=j":
        a_text, b_text = attempt1_text, attemptj_text
    else:
        a_text, b_text = attemptj_text, attempt1_text

    user = _JUDGE_USER.format(
        learner_goal=learner_goal,
        relational_stakes=relational_stakes,
        transcript_a=_truncate(a_text),
        transcript_b=_truncate(b_text),
    )
    is_error = False
    try:
        data = fm.query_json(_JUDGE_SYSTEM, user, temperature=temperature)
        raw_verdict = str(data.get("verdict", "no_difference")).strip().lower()
        rationale = str(data.get("rationale", ""))
        # Normalise verdict to "A" | "B" | "no_difference"
        if raw_verdict in ("a",):
            verdict = "A"
        elif raw_verdict in ("b",):
            verdict = "B"
        else:
            verdict = "no_difference"
    except Exception as e:
        verdict = "no_difference"
        rationale = f"[judge error: {e}]"
        is_error = True

    return VoteRecord(pair=pair, order=order, verdict=verdict, rationale=rationale,
                      is_error=is_error)


def _pair_improved_votes(
    vote_ab: VoteRecord,
    vote_ba: VoteRecord,
    j: int,
) -> int:
    """Return improved_votes for one pair given its two order-swapped votes.

    'improved' means the later attempt (j) was rated better.
    - AB order: verdict B → j is better
    - BA order: verdict A → j is better
    Both agree j better → 2 improved-votes.
    Disagree → 0 (position bias, collapse to no_difference).
    Both agree j worse or no_difference → 0.
    """
    ab_says_j = (vote_ab.order == "A=1,B=j" and vote_ab.verdict == "B") or \
                (vote_ab.order == "A=j,B=1" and vote_ab.verdict == "A")
    ba_says_j = (vote_ba.order == "A=1,B=j" and vote_ba.verdict == "B") or \
                (vote_ba.order == "A=j,B=1" and vote_ba.verdict == "A")

    if ab_says_j and ba_says_j:
        return 2   # both agree: later attempt is better
    if ab_says_j != ba_says_j:
        return 0   # disagree under swap → no_difference
    return 0       # both agree: attempt 1 better or tied


# ---------------------------------------------------------------------------
# Main async interface
# ---------------------------------------------------------------------------

async def compute_lp(
    fm_judge: FM,
    scenario: SocialScenario,
    transcripts: list[list[dict]],
    learner_goal: str,
    relational_stakes: str,
    lp_temperature: float = 0.3,
) -> LPResult:
    """Compute learning progress over all (1, j) pairs with order-swap voting.

    transcripts: index 0 = attempt 1. At least 2 transcripts required;
    if fewer than 2, returns LPResult(lp_value=0.0, …) — caller handles
    classification (single-attempt success → too_easy).
    """
    if len(transcripts) < 2:
        return LPResult(lp_value=0.0, improved_votes=0, total_votes=0, n_pairs=0)

    attempt1_text = _transcript_to_str(transcripts[0])
    pairs = [(1, j + 1) for j in range(1, len(transcripts))]
    attemptj_texts = {p[1]: _transcript_to_str(transcripts[p[1] - 1]) for p in pairs}

    loop = asyncio.get_running_loop()

    def _cast(pair: tuple[int, int], order: str):
        return loop.run_in_executor(
            None, _cast_vote,
            fm_judge, attempt1_text, attemptj_texts[pair[1]],
            learner_goal, relational_stakes, pair, order, lp_temperature,
        )

    # Cast both order-swapped votes for every pair in parallel; keep them pair-adjacent.
    jobs = []
    for pair in pairs:
        jobs.append(_cast(pair, "A=1,B=j"))
        jobs.append(_cast(pair, "A=j,B=1"))
    all_votes: list[VoteRecord] = list(await asyncio.gather(*jobs))

    def _improved(votes: list[VoteRecord]) -> int:
        total = 0
        for i in range(0, len(votes), 2):  # votes are stored two-per-pair, in order
            total += _pair_improved_votes(votes[i], votes[i + 1], votes[i].pair[1])
        return total

    total_votes_count = len(all_votes)
    total_improved = _improved(all_votes)
    lp_value = total_improved / total_votes_count if total_votes_count > 0 else 0.0

    # One-shot revote of errored votes when the result sits exactly on the lp==0 boundary —
    # a single judge error on the decisive pair can spuriously flip frontier→beyond_frontier.
    n_error_votes = sum(1 for v in all_votes if v.is_error)
    if n_error_votes > 0 and lp_value == 0.0:
        err_idx = [i for i, v in enumerate(all_votes) if v.is_error]
        revoted = await asyncio.gather(
            *[_cast(all_votes[i].pair, all_votes[i].order) for i in err_idx]
        )
        for i, vote in zip(err_idx, revoted):
            all_votes[i] = vote
        total_improved = _improved(all_votes)
        lp_value = total_improved / total_votes_count if total_votes_count > 0 else 0.0
        n_error_votes = sum(1 for v in all_votes if v.is_error)

    # Every vote of every pair errored → LP uncomputable → quarantine (not a (0,K) penalty).
    if total_votes_count > 0 and n_error_votes == total_votes_count:
        raise LPAllErrorsError(
            f"All {total_votes_count} LP judge votes errored across {len(pairs)} pair(s)."
        )

    return LPResult(
        lp_value=round(lp_value, 4),
        improved_votes=total_improved,
        total_votes=total_votes_count,
        n_pairs=len(pairs),
        votes=all_votes,
        n_error_votes=n_error_votes,
    )
