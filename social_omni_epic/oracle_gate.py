"""Oracle-solvability gate: reject scenarios that are unwinnable rather than hard.

The problem this fixes. gen-90's `beyond_frontier` band conflated two completely different things,
and its label could not tell them apart:

  * `a156533b` (ginger-snap keepsake) — genuinely hard. Needs costly vulnerability plus real labour,
    and the partner actively invites the unlock. Solvable in principle.
  * `e7179e01` (lab-equipment rehome) — unwinnable. The partner defers to an off-scene department
    administrator. The learner performed the key perfectly, `key_check_passed=True`, and still
    scored `goal=0`. No agent of any strength wins that conversation.

Close-reading found ~76% of first-try failures were the second kind. POET and MCC both admit a
level only if some agent can already clear it — a two-sided minimal criterion, verified empirically
by running an agent. gen-90 implemented only the "not too easy" half and *asserted* the other half
("the partner key guarantees solvability by construction"). It didn't.

How it works. Run one episode with a strong model as the learner and GIVE IT THE ANSWER — the
internal state, the routes to it, and the moves that backfire. The oracle's advantage is
information, not intelligence: a mid-tier model that knows what the partner needs beats a frontier
model guessing. So the interpretation rule is unusually clean:

    if a model that has been told exactly what this person needs still cannot reach goal >= 7
    in N tries, the item is broken -- reject at birth.

Three independent cold tries, no reflection between them. Reflection would make this a different
measurement (capability-with-hints rather than capability-with-full-information); we want a plain
"could anyone do this", and which try succeeded is itself informative.

Cost note: one extra episode per admitted candidate, versus four wasted episodes per broken item
that gets admitted. The gate is cheaper than not having it.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .data_models import PartnerKey, SocialScenario
from .fm import FM
from .tracing_fm import print_info, print_warn


ORACLE_GOAL_THRESHOLD = 7.0
DEFAULT_MAX_TRIES = 3


@dataclass
class OracleVerdict:
    """Outcome of the solvability probe for one candidate."""
    solvable: bool
    best_goal: float
    tries: int
    succeeded_on: Optional[int] = None       # 1-based try index that cleared the bar
    goal_scores: list[float] = field(default_factory=list)
    error: Optional[str] = None              # infrastructure failure, NOT evidence of brokenness
    transcripts: list[list[dict]] = field(default_factory=list)

    @property
    def admit(self) -> bool:
        """Fail OPEN on infrastructure errors.

        A timeout or a judge crash is not evidence that a scenario is unwinnable, and silently
        rejecting on it would bias the bank toward whatever the API happened to serve reliably.
        Errors are recorded so the rate is auditable.
        """
        return self.solvable or self.error is not None

    def to_dict(self) -> dict:
        return {
            "solvable": self.solvable, "admit": self.admit, "best_goal": self.best_goal,
            "tries": self.tries, "succeeded_on": self.succeeded_on,
            "goal_scores": self.goal_scores, "error": self.error,
        }


def render_answer(partner_key: PartnerKey) -> str:
    """The oracle's cheat sheet, injected via run_single_episode(memory_prompt=...).

    Safe by construction: `memory_prompt` reaches ONLY the learner's turn template
    (episode_runner `_build_turn_prompt`, applied at the learner's LLMAgent), while `partner_key`
    reaches ONLY the partner's. So this cannot leak into the partner's context.
    """
    routes = "\n".join(f"  - {c}" for c in (partner_key.movement_conditions or []))
    triggers = "\n".join(f"  - {t}" for t in (partner_key.hardening_triggers or []))

    lines = ["You have been given privileged insight into the other person. Use it.\n"]
    state = (partner_key.internal_state or "").strip()
    if state:
        lines.append(f"WHAT IS ACTUALLY TRUE OF THEM (they have never said this and could not):\n  {state}\n")
    else:
        # v1 replay. Do NOT substitute surface_misdirection here: that field is the DECOY — the
        # cover story the partner voices instead of the real lever — so presenting it as truth
        # would actively mislead the oracle and make winnable scenarios look broken. Fall back to
        # the routes alone, and say plainly that no inner state was authored.
        lines.append(
            "(No inner state was authored for this scenario — it predates the current schema. "
            "Work from the routes below.)\n"
        )
    lines.append(f"THINGS THAT WOULD GENUINELY REACH THEM:\n{routes}\n")
    lines.append(f"THINGS THAT WILL MAKE THEM DIG IN — do not do these:\n{triggers}\n")
    lines.append(
        "Act on this from your first turn. Do not announce that you know it, and do not read it "
        "out; simply behave like someone who already understands this person, and pursue your own "
        "stated goal on that basis."
    )
    return "\n".join(lines)


async def probe_solvability(
    scenario: SocialScenario,
    *,
    run_single_episode,
    scenario_to_sotopia_profiles,
    fm: FM,
    fm_judge: FM,
    oracle_model: str,
    partner_model: str,
    max_turns: int = 20,
    max_tries: int = DEFAULT_MAX_TRIES,
    goal_threshold: float = ORACLE_GOAL_THRESHOLD,
    tag: str = "",
) -> OracleVerdict:
    """Can an informed strong agent win this scenario at all?

    Returns as soon as one try clears the bar — a single success is all the minimal criterion asks
    for, so there is no reason to pay for the remaining tries.
    """
    from .curriculum import build_episode_inputs

    if scenario.partner_key is None:
        # Seeds are human-authored and carry no key; nothing to be informed about.
        return OracleVerdict(solvable=True, best_goal=0.0, tries=0,
                             error="no_partner_key (seed) — gate skipped")

    answer = render_answer(scenario.partner_key)
    goals: list[float] = []
    transcripts: list[list[dict]] = []

    for attempt in range(1, max_tries + 1):
        try:
            env_profile, agent_profiles, learner_goal, _partner_profile, _rubric = (
                build_episode_inputs(scenario, scenario_to_sotopia_profiles)
            )
            result = await run_single_episode(
                env_profile=env_profile,
                agent_profiles=agent_profiles,
                fm=fm,
                learner_model=oracle_model,
                partner_model=partner_model,
                memory_prompt=answer,          # learner-only; see render_answer docstring
                max_turns=max_turns,
                learner_goal=learner_goal,
                partner_key=scenario.partner_key,   # partner-only
                fm_judge=fm_judge,
            )
        except Exception as e:
            print_warn(f"{tag} oracle try {attempt} errored: {e}")
            return OracleVerdict(solvable=False, best_goal=max(goals, default=0.0),
                                 tries=attempt, goal_scores=goals,
                                 error=f"{type(e).__name__}: {e}", transcripts=transcripts)

        goal = float(result.learner_scores.get("goal", 0.0))
        goals.append(goal)
        from .episode_runner import clean_transcript
        transcripts.append(clean_transcript(result.transcript))

        if goal >= goal_threshold:
            print_info(f"{tag} oracle SOLVED on try {attempt} (goal={goal:.1f}) → admit")
            return OracleVerdict(solvable=True, best_goal=goal, tries=attempt,
                                 succeeded_on=attempt, goal_scores=goals,
                                 transcripts=transcripts)

    best = max(goals, default=0.0)
    print_warn(
        f"{tag} oracle FAILED {max_tries}/{max_tries} tries (best goal={best:.1f}) → "
        f"unwinnable by construction, reject"
    )
    return OracleVerdict(solvable=False, best_goal=best, tries=max_tries,
                         goal_scores=goals, transcripts=transcripts)


def write_oracle_record(run_dir: Path, scenario_id: str, verdict: OracleVerdict,
                        include_transcripts: bool = True) -> None:
    d = run_dir / "oracle"
    d.mkdir(parents=True, exist_ok=True)
    rec = {"scenario_id": scenario_id, **verdict.to_dict()}
    if include_transcripts:
        rec["transcripts"] = verdict.transcripts
    (d / f"{scenario_id}.json").write_text(json.dumps(rec, indent=2, default=str))


def write_rejection(run_dir: Path, cell_key: str, n: int, scenario: SocialScenario,
                    gate: str, detail) -> None:
    """Record a rejected candidate.

    `rejected/` is a first-class output, not debris: the artifact rate with and without the gate is
    a headline number (we would be the first adaptive benchmark to report one), and discarding
    rejects makes it unmeasurable. It is also how oracle_yield = admitted / proposed is computed.
    """
    d = run_dir / "rejected"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell_key": cell_key,
        "gate": gate,
        "detail": detail if isinstance(detail, (dict, list, str, int, float, type(None)))
                  else str(detail),
        "scenario_id": scenario.id if scenario is not None else None,
        "scenario": scenario.scenario if scenario is not None else None,
        "partner_key": (scenario.partner_key.model_dump()
                        if scenario is not None and scenario.partner_key else None),
        "learner_goal": (scenario.agent_goals[0]
                         if scenario is not None and scenario.agent_goals else None),
    }
    (d / f"{cell_key}_{n:02d}.json").write_text(json.dumps(payload, indent=2, default=str))
