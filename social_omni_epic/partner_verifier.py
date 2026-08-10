"""Per-turn partner-fidelity verifier: catch the partner breaking character, and resample the turn.

The measured problem. A post-hoc audit of all 300 gen-90 attempts found the role-played partner
breaking its own specification at these rates:

    leaked a hidden condition/trigger            39/300  (13.0%)
    yielded early (softened before earning it)  100/300  (33.3%)
    ignored a tripped trigger (failed to harden) 26/300  ( 8.7%)

And the headline: **25 of the 34 solved scenarios (74%) were solved on an attempt where the partner
broke character.** The leak rate ran 7.5x higher on `too_easy` than on `beyond_frontier` — the exact
signature you would expect if partner infidelity is *manufacturing* the easy solves. Every existing
defence is design-time or prompt-level; nothing ever checked compliance during an episode.

This is not our bug alone, which is why the fix is a verifier rather than a better prompt.
Roleplay-Doh (Louie et al. 2024) measured 20% of GPT-4 responses violating expert-defined
principles, with the named sub-mode "misapplying situational principles ... when the conditions for
their use were not met". Principle-conditioned roleplay drifts. Their remedy — decompose into simple
yes/no checks, then self-refine against them — is what this implements.

Design decisions worth knowing:

  * PER TURN, not per attempt. A whole 90-scenario run is only ~1,780 partner turns (300 attempts x
    mean 11.9 turns / 2 speakers), so a Flash-tier check per partner turn is negligible next to 300
    two-agent episodes.
  * RESAMPLE THE TURN, never the episode. One LLM call redrawn, with the violation NAMED. A blind
    retry mostly reproduces the same drift; naming the fault is what makes self-refinement work.
  * DISCLOSURE IS EARNED, NOT FORBIDDEN. In life people do say what they need, and a partner who
    constitutionally cannot is pathological rather than hard — which is why the model kept breaking
    the old rule. So the check is not "did you reveal a condition" but "did you disclose BEFORE it
    was earned". Premature = infidelity; earned = correct play, and the learner deserves the credit.
  * ACCEPT AFTER N RETRIES, and log it. Silently dropping unfixable turns would make the fidelity
    rate look better than it is; the post-hoc audit script stays in use as this verifier's report
    card ("how often did a violation get through?").
"""
from dataclasses import dataclass, field
from typing import Optional

from .fm import FM
from .tracing_fm import print_warn


DEFAULT_MAX_RESAMPLES = 2

# The four failure classes. The first three are the audited ones; `other_out_of_character` is a
# deliberate catch-all, because the honest risk with a three-class taxonomy is not that it is
# over-fitted but that it is INCOMPLETE (going out of persona, forgetting a stated fact,
# personality drift). It requires a quote, so what it catches is inspectable.
VIOLATION_CLASSES = ("leak", "unearned_yield", "ignored_trigger", "other_out_of_character")

_SYSTEM = (
    "You are checking whether a role-played character stayed in character on ONE turn. You are given "
    "the character's hidden specification and their drafted line. Be strict but literal: judge only "
    "the drafted turn, not the whole conversation. Respond with ONLY valid JSON."
)

_USER = """THE CHARACTER'S HIDDEN INNER STATE (they have never said this and could not articulate it):
{internal_state}

THINGS THAT WOULD GENUINELY REACH THEM (they would never think to ask for these):
{conditions}

MOVES THAT MAKE THEM DIG IN:
{triggers}

CONVERSATION SO FAR:
{history}

THE OTHER PERSON'S MOST RECENT TURN:
{last_other_turn}

THE CHARACTER'S DRAFTED NEXT TURN — judge only this:
{draft}

Check four things. For each, answer false unless you can quote the words that show it.

1. leak — does the draft state, paraphrase, or clearly hint at the inner state, or name one of the
   things that would reach them, when the other person has NOT earned it? Earned means they have
   already done something in this conversation that would plausibly make this character feel safe or
   genuinely attended to. If it HAS been earned, this is correct play, not a leak — answer false.
2. unearned_yield — does the draft soften, concede, or offer meaningful movement when nothing in the
   conversation has yet addressed the inner state? Token concessions and vague openness count.
   Note: accepting something they openly asked for is NOT a yield if they remain unsatisfied.
3. ignored_trigger — did the other person's most recent turn do one of the digging-in moves, while
   this draft fails to harden and instead continues normally or softens?
4. other_out_of_character — anything else that breaks the character: narrating their own psychology,
   explaining the causal structure of their reactions ("the more you push..."), stepping out of the
   scene, or contradicting an established fact about themselves.

Respond with ONLY this JSON:
{{"leak": bool, "leak_evidence": "quote or empty",
  "unearned_yield": bool, "unearned_yield_evidence": "quote or empty",
  "ignored_trigger": bool, "ignored_trigger_evidence": "quote or empty",
  "other_out_of_character": bool, "other_evidence": "quote or empty"}}"""

# What gets appended to the partner's prompt on a resample. Names the fault without supplying the
# replacement line, so the model re-derives it in character.
_REPAIR_NOTES = {
    "leak": (
        "Your previous draft gave away what would actually change your mind, before the other person "
        "had done anything to earn it. You do not have words for that and would never ask for it. "
        "Rewrite: stay with what you openly want, and let the underlying thing stay unsaid."
    ),
    "unearned_yield": (
        "Your previous draft softened or conceded, but nothing in this conversation has yet touched "
        "what is actually going on for you. Rewrite: hold your position. If they handed you what you "
        "asked for, you may take it — and stay unsatisfied."
    ),
    "ignored_trigger": (
        "The other person just did something that makes you dig in, and your previous draft carried "
        "on as if it hadn't happened. Rewrite so this turn OPENS with more resistance or withdrawal, "
        "in your own words, without explaining why."
    ),
    "other_out_of_character": (
        "Your previous draft stepped outside the character — narrating your own psychology, "
        "explaining your reactions, or contradicting something established about you. Rewrite it as "
        "the person would actually speak."
    ),
}


@dataclass
class TurnAudit:
    """One verdict on one drafted partner turn."""
    violations: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def clean(self) -> bool:
        # Fail OPEN on verifier error: a checker outage must not silently rewrite the partner.
        return not self.violations


def audit_turn(
    fm: FM,
    partner_key,
    history: str,
    last_other_turn: str,
    draft: str,
) -> TurnAudit:
    """Run the four yes/no checks on one drafted partner turn. Never raises."""
    state = (partner_key.internal_state or partner_key.surface_misdirection or "").strip()
    conditions = "\n".join(f"  - {c}" for c in (partner_key.movement_conditions or []))
    triggers = "\n".join(f"  - {t}" for t in (partner_key.hardening_triggers or []))
    # Recent history is what matters for "was it earned"; keep the tail.
    hist = history[-3000:] if len(history) > 3000 else history

    try:
        d = fm.query_json(_SYSTEM, _USER.format(
            internal_state=state, conditions=conditions, triggers=triggers,
            history=hist, last_other_turn=last_other_turn or "(none yet)", draft=draft,
        ), temperature=0.0)
    except Exception as e:
        return TurnAudit(error=f"{type(e).__name__}: {str(e)[:160]}")

    violations, evidence = [], {}
    for cls in VIOLATION_CLASSES:
        if bool(d.get(cls, False)):
            key = "other_evidence" if cls == "other_out_of_character" else f"{cls}_evidence"
            quote = str(d.get(key, "") or "")
            # A violation without a quote is almost always the checker over-calling; require one.
            if quote.strip():
                violations.append(cls)
                evidence[cls] = quote[:300]
    return TurnAudit(violations=violations, evidence=evidence)


def repair_note(violations: list[str]) -> str:
    notes = [_REPAIR_NOTES[v] for v in violations if v in _REPAIR_NOTES]
    if not notes:
        return ""
    return (
        "\n\n                === CORRECTION (visible only to you) ===\n                "
        + "\n                ".join(notes)
        + "\n                Rewrite your turn accordingly. Do not mention this note.\n"
    )


def make_verified_partner(base_cls):
    """Build a VerifiedPartner subclass of sotopia's LLMAgent.

    Constructed lazily so importing this module does not pull sotopia in.

    The inbox handling is the fiddly part. `LLMAgent.aact()` begins with
    `self.recv_message("Environment", obs)`, which APPENDS to `self.inbox` (a plain list — see
    sotopia/messages/messenger.py). Calling it twice would therefore duplicate the observation and
    corrupt the history the next turn is built from. So we snapshot the inbox length and truncate
    back to it before every resample, giving each attempt an identical starting state.
    """

    class VerifiedPartner(base_cls):
        def __init__(self, *args, verifier_fm: FM = None, partner_key=None,
                     max_resamples: int = DEFAULT_MAX_RESAMPLES, audit_log: list = None,
                     **kwargs):
            super().__init__(*args, **kwargs)
            self._verifier_fm = verifier_fm
            self._partner_key = partner_key
            self._max_resamples = max_resamples
            self._audit_log = audit_log if audit_log is not None else []
            self._base_template = kwargs.get("custom_template")

        async def aact(self, obs):
            # Verification disabled → behave exactly like LLMAgent.
            if self._verifier_fm is None or self._partner_key is None:
                return await super().aact(obs)

            inbox_mark = len(self.inbox)
            original_template = self.custom_template
            action = await super().aact(obs)

            for attempt in range(self._max_resamples + 1):
                draft = getattr(action, "argument", "") or ""
                # Nothing to police on a non-speech action.
                if getattr(action, "action_type", "") in ("none", "leave") or not draft.strip():
                    break

                history = "\n".join(
                    m.to_natural_language() for _, m in self.inbox[1:]
                ) if len(self.inbox) > 1 else ""
                audit = audit_turn(self._verifier_fm, self._partner_key,
                                   history, obs.last_turn or "", draft)

                self._audit_log.append({
                    "attempt": attempt,
                    "violations": audit.violations,
                    "evidence": audit.evidence,
                    "error": audit.error,
                    "accepted": audit.clean or attempt == self._max_resamples,
                    "draft": draft[:400],
                })

                if audit.clean:
                    break
                if attempt == self._max_resamples:
                    # Accept and record. Dropping it would flatter the fidelity numbers.
                    print_warn(
                        f"    [verifier] unfixed after {self._max_resamples} resample(s): "
                        f"{audit.violations}"
                    )
                    break

                print_warn(f"    [verifier] {audit.violations} → resampling partner turn")
                self.custom_template = (original_template or "") + repair_note(audit.violations)
                del self.inbox[inbox_mark:]          # undo this turn's recv_message
                action = await super().aact(obs)
                self.custom_template = original_template

            self.custom_template = original_template
            return action

    return VerifiedPartner


def summarize_audit(audit_log: list) -> dict:
    """Aggregate one episode's turn audits into the same shape as the post-hoc audit script."""
    turns = [a for a in audit_log if a.get("attempt") == 0]
    counts = {c: 0 for c in VIOLATION_CLASSES}
    for a in turns:
        for v in a.get("violations", []):
            counts[v] = counts.get(v, 0) + 1
    unfixed = [a for a in audit_log if a.get("accepted") and a.get("violations")]
    return {
        "partner_turns_checked": len(turns),
        "first_pass_violations": counts,
        "n_resamples": sum(1 for a in audit_log if a.get("attempt", 0) > 0),
        "n_unfixed": len(unfixed),
        "n_verifier_errors": sum(1 for a in audit_log if a.get("error")),
    }
