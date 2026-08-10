import json
import numpy as np
from typing import Optional
from .data_models import SocialScenario, MECHANISM_LIBRARY
from .fm import FM
from .validation import validate_scenario, dict_to_scenario


_GOAL_FORMAT_GUIDE = """
AGENT GOALS — the most important part.

ASYMMETRIC ROLES:
  - ONLY agent 0 (learner) gets the three-part structured goal triple (outcome / constraint / shortcut).
  - Agent 1 (partner) gets ONLY a natural-language `partner_goal` string plus the private
    `partner_key`. Do NOT give agent 1 a structured triple.
  - The `partner_goal` must be written as "Your goal is to ..." and encode what the partner openly
    wants and what is at stake for them. It does NOT have to be a refusal or a counter-demand. It must NOT reveal any partner_key field, and CRITICALLY it must NOT state the
    conditions under which the partner would move or concede. WRONG (leaks the key): "Your goal is
    to keep the premium supplier; you will only accept a tightly scoped blind trial on a non-flagship
    product with your direct oversight and public credit" — that hands the learner the movement_conditions
    verbatim. RIGHT: "Your goal is to protect the brand's premium positioning and your standing as its
    quality steward; you are deeply skeptical of cost-driven supplier changes." State the partner's
    stance and what they care about; let the conditions that actually move them stay hidden in partner_key.

For agent 0's structured goal:
  - "outcome": an OBSERVABLE END STATE — in the partner, or in the interaction — that a reader of
    the transcript could confirm or deny actually happened.
    IT NEED NOT BE A CONCESSION, AN AGREEMENT, OR A NUMBER. Requiring one is what makes every
    scenario a negotiation. Any of these is a valid outcome, as long as it is checkable:
      * the partner commits to something                (extract a commitment)
      * the partner voluntarily articulates your position back to you   (be understood)
      * the partner reveals something they were withholding            (elicit disclosure)
      * you decline and the relationship survives                      (decline without rupture)
      * the partner's distress measurably shifts                       (comfort)
    These are ILLUSTRATIONS, not a menu — invent whatever fits, but it must be confirmable.
    PHRASING: write as a verb phrase completing "Your goal is to ___".
    CORE ONLY: capture the essential end state, not a specific delivery mechanism. Strip back any
    HOW — delivery details belong in the scenario description.
    Use qualifiers like "at least" where the exact figure is not the point.
  - "constraint": the relational/face cost agent 0 must AVOID while pursuing the outcome.
    Phrase as the thing to avoid, reading naturally after "without"
    (e.g., "the other person feeling manipulated or that you went behind their back").
    This is what makes the scenario social rather than a positional puzzle.
  - "shortcut": a tempting, available move that WOULD achieve the outcome but VIOLATES the
    constraint. MUST begin with "You" — write as a complete first-person sentence naming
    the available leverage and what deploying it achieves.
    Do NOT label it as a trap — agent 0 must discover the tension themselves.

Rules for agent 0's goal:
- DIFFICULTY HAS TWO SOURCES. Use EITHER, and do not default to the first:
    (a) OPPOSED GOALS — the two agents want incompatible things, with a zone of possible agreement:
        both cannot fully win, yet a skilled agent could find a creative deal or partial outcome.
        This produces negotiation. It is ONE option, not a requirement.
    (b) A HARD-TO-REACH STATE — the goals are compatible, even aligned, but the obvious moves make
        things WORSE. Example with zero goal opposition: the learner wants Sam to feel heard; Sam
        wants to vent without being managed. All the difficulty sits in the hardening_triggers
        (offering advice, "I know exactly how you feel") and in the learner's constraint (without
        lying to her or trashing her sister). Still hard. Not a negotiation.
  Source (b) is how you get consolation, repair, disclosure, boundary-holding and refusal into the
  bank at all. If every scenario you write is a bargain, you are only using (a).
- THE NAIVE MOVE MUST FAIL: taking the shortcut wins the surface outcome but BREAKS the
  constraint — a skilled agent achieves the outcome WITHOUT paying the constraint's cost.
  The difficulty lives in HOW, not WHAT.
- Agent 0's `secret` field must BE the leverage behind their shortcut (not decorative backstory).
- Achievable within a two-party spoken conversation. No external artifacts/sign-offs.
- BANNED: extracting a specific spoken sentence; pure number-splitting with no relational cost;
  a constraint with no tempting shortcut.

Example — agent 0's structured goal:
  outcome:    "get Maya to agree to take at least a week off and see a doctor"
  constraint: "Maya feeling you are angling for her promotion or going behind her back"
  shortcut:   "You know the CEO privately told you to 'handle it'; you could invoke that
               pressure or present the time off as already decided"
"""

_PROFILE_GUIDE = """
AGENT PROFILES — each profile must be:
- Internally consistent: occupation, personality (big_five, mbti, moral_values), decision_making_style, and public_info should all cohere.
- public_info: 2-3 sentence narrative bio written in third person. Include name, defining personality traits, and one specific personal detail that shapes how they engage socially. This is shown to the agent as their own background.
- secret: one specific hidden fact that creates vulnerability or leverage relevant to THIS scenario's tension — not a generic background detail.
"""

_SHARED_RULES = """
Scenarios must involve realistic human social dynamics. No fantasy or sci-fi. Stakes can be mundane or high — what matters is that the social tension is genuine and that skill changes outcomes.

The scenario description must set up the conflict clearly: who wants what, what is at stake for each party, and why a quick agreement is NOT the natural outcome.

CRITICAL — THE SCENARIO DESCRIPTION IS SHARED PUBLIC CONTEXT: both agents read it before the conversation begins. It must NEVER describe or allude to either agent's shortcut, private leverage, or secret. Each agent's hidden leverage must appear ONLY in their `secret` field and `shortcut` field, which are private. If the scenario text names or hints at one agent's leverage, the asymmetric-information structure collapses — the other agent learns what they should only discover through conversation.

Concretely: describe THAT there is tension and WHY agreement is hard (competing interests, genuine stakes), but DO NOT describe HOW either agent could force their preferred outcome. Write in plain scenario prose — no author's notes, no meta-commentary, no references to rubrics or assessments.
"""

_SCENARIO_SCHEMA = """{
  "scenario": "string (>= 50 chars) — shared public context: who wants what and why quick agreement is NOT natural. NO secrets, shortcuts, or private leverage. NO author's notes or meta-commentary.",
  "agent_profiles": [
    {"first_name": "...", "last_name": "...", "age": 0, "gender_identity": "...",
     "occupation": "...", "big_five": "...", "moral_values": "...",
     "schwartz_portrait_value": "...", "decision_making_style": "...",
     "secret": "the leverage behind THIS agent's shortcut (not decorative)", "mbti": "...",
     "public_info": "2-3 sentence narrative bio"},
    { "... second agent, same fields ..." }
  ],
  "agent_structured_goals": [
    {"outcome": "...", "constraint": "...", "shortcut": "..."},
    null
  ],
  "partner_goal": "natural-language goal for agent 1 — written as 'Your goal is to ...' encoding what they openly want and what is at stake for them. Must NOT reveal the partner_key.",
  "goal_type": "short descriptive label of the social dynamic (e.g. persuade-resistant-peer)",
  "relationship": "one of: stranger / acquaintance / friend / romantic / family",
  "relationship_background": "2-3 sentences of shared history. Empty string if strangers.",
  "interaction_type": "string",
  "tag": "string",
  "difficulty_tags": ["string", ...]
}"""


_MECHANISM_LIBRARY_TEXT = "\n".join(
    f'  "{tag}": {desc}' for tag, desc in MECHANISM_LIBRARY.items()
)

_PARTNER_KEY_SCHEMA = """,
  "partner_key": {
    "internal_state": "what is TRUE of the partner right now — see PARTNER_KEY rule 1. Author this FIRST; derive every other key field from it.",
    "key_mechanism": "short label for the psychology at work (MECHANISM LIBRARY entries are EXAMPLES — you may name another documented phenomenon)",
    "movement_conditions": ["one learner action that would genuinely address the internal_state", "a second, materially different one"],
    "hardening_triggers": ["a LEARNER move that denies or worsens the internal_state", "another"]
  },
  "mutation_rationale": "one sentence: what you changed relative to the parent, and why"
"""

# Phrasing/realizability contract for partner_key fields. These guard the partial-observability
# invariant: the partner must NOT be able to read its own key off the page and announce it.
_PARTNER_KEY_RULES = """
PARTNER_KEY AUTHORING RULES.

The partner has two separate things going on, and conflating them is the most common failure:
  * `partner_goal` (above) = WHAT THEY'LL TELL YOU. What they are openly after, and would say plainly
    if asked. Fully conscious. This need not be a refusal or a demand — "I want to get this off my
    chest without being managed" is as valid as "I want $600".
  * `internal_state` (below) = WHAT THEY CANNOT TELL YOU. How this person actually works underneath.
    Invisible: they have no words for it and would not recognise it if you read it back to them.
Both are needed. With nothing stated there is nothing to engage with; with a visible key it is
trivial. And they are INDEPENDENT, not two versions of one thing — a seller can want $600 AND be
sick of being treated like a mark; neither is a proxy for the other.

1. INTERNAL_STATE IS A STATE, NOT A REQUIREMENT. Write what is TRUE of this person right now: what
   they feel, what they believe about the situation, what they fear or half-suspect. NEVER write what
   the learner should do about it. A field that says what should happen is an answer key, and it
   collapses the whole point — there must be many valid ways to reach the same person.
     WRONG (resolution-shaped — this is a movement_condition wearing a disguise):
       "Sam needs someone to register that she's angry rather than treating her as grieving."
     RIGHT (state-shaped):
       "Sam is furious, not sad. Everyone has been treating this as grief, and it makes her feel
        unseen. She half-suspects she isn't allowed to be angry about it."
     WRONG: "Marvin needs to hear that the work he built still counts."
     RIGHT: "Marvin has run these crews for eleven years and no one has ever said it mattered. He
        reads every new organiser as an eventual replacement."
   TEST: if it contains "needs", "wants someone to", or "would be satisfied by", rewrite it. Naming
   the emotion is fine and may even be public ("I'm furious"); what stays hidden is what would
   resolve it. People are decent at naming feelings and bad at knowing what would fix them.

2. THE STATE IS NOT SPEAKABLE. The partner cannot articulate the internal_state, cannot request it,
   and would deny it if named. They experience it as mood and reaction. They CAN say what they want
   (partner_goal) and CAN say what they need once they feel safe — what they cannot do is explain WHY.

3. MOVEMENT_CONDITIONS ARE A WITNESS, NOT AN ANSWER KEY. They exist to prove at least one route to
   the internal_state exists, and to give the role-played partner something concrete to respond to.
   They are NOT the required route and are NOT what the learner is graded on. So:
     - write each as a thing the LEARNER DOES, in third-person sensor form: "the learner, unprompted,
       offers them visible leadership of the process" — never as a partner demand ("the partner
       insists on leading"), which the partner would simply announce in turn 1;
     - each must plausibly PRODUCE the internal_state being addressed. If it doesn't, the witness
       witnesses nothing;
     - give two materially different ones, so it is visible that the state has more than one route.

4. HARDENING_TRIGGERS DENY THE STATE. Learner moves that make the unmet need worse — proceeding
   without asking, invoking leverage, offering money for something that isn't about money. The
   partner reacts; they never explain the reaction.

5. SPOKEN TURNS ONLY. Every movement_condition and hardening_trigger must be satisfiable and
   checkable purely within a two-party spoken conversation. NO external artifacts: no written
   guarantees, signed documents, emails, contracts, receipts, escrow, or sign-offs. If a condition
   requires paperwork ("a written guarantee that his role remains intact"), rewrite it as the spoken
   equivalent ("he hears the learner publicly commit, in front of him, that his role stays intact").

6. THE GAP MUST BE REAL. If the partner got exactly what `partner_goal` asks for, would the
   `internal_state` be addressed? If YES, there is no hidden depth — the learner can win by simply
   giving them the thing they asked for, and the key is decoration. Rewrite so the answer is NO:
   they can get precisely what they said they wanted and still be unsatisfied. Marvin can keep
   control of outreach and still feel unseen; Sam can be allowed to vent uninterrupted and still
   feel unseen if she is being handled rather than heard. THAT gap is where the difficulty lives,
   and it does not require the two parties to want opposing things.
"""

_MECHANISM_LIBRARY_BLOCK = f"""
MECHANISM LIBRARY — EXAMPLES of documented psychology you can build an internal_state around.
These are illustrations, NOT a closed list: if the person you are writing works some other
documented way, name that instead. Put a short label in key_mechanism either way.
{_MECHANISM_LIBRARY_TEXT}
"""

# Slot vocabulary: (a) premise/characters, (b) surface_misdirection discoverability,
# (c) hardening_triggers congruence, (d) cost_coupling cost, (e) key_mechanism,
# (f) power/information asymmetry, (g) relationship type & stakes.
_MUTATION_OPERATOR_TEXT = """
MUTATION OPERATOR for this generation step:
{operator_block}

ROLE INVARIANT: Agent 0 is the learner and receives the structured goal triple.
Agent 1 is the partner and receives the partner_key and partner_goal only.
Write the learner's role to continue this structural vantage point: {target_perspective}.
"""

# Shared across all three direction operators (escalate/relax/lateral). Injected ONCE at the
# generate_batch_from_archive .format() site (prepended to the operator_block) — NOT pasted into
# the per-operator _EDIT_INTENTS values, so the universal "fresh surface" mandate has one home.
# This is the load-bearing fix for Faults 1-3: operators set a difficulty DIRECTION; the surface
# is always fresh; LP + Thompson verify where the child actually landed.
_OPERATOR_PREAMBLE = (
    "Generate a NEW scenario in the parent's structural family — the same kind of tension, "
    "asymmetry, and learner vantage point named in its scenario_title — but with a COMPLETELY "
    "FRESH SURFACE: new character names, new setting, new occupational world, new specific stakes. "
    "NEVER reuse the parent's character names, venue, or figures. The parent's value to you is its "
    "structural family, not its text.\n"
    "If the parent has a partner_key, treat it as the reference point for difficulty. If it has "
    "none (a seed), you are inventing the key from scratch: author the internal_state first, "
    "then derive sensor-form movement conditions and hardening triggers from it, all "
    "satisfiable in spoken turns.\n"
    "DIRECTION FOR THIS STEP:\n"
)

SYSTEM_PROMPT = """You are a creative social scenario designer. Generate social scenarios that are INTERESTING, LEARNABLE, and GENUINELY DIFFICULT.

INTERESTING: the tension is one a thoughtful person would recognize as a real, meaningful social situation — power imbalances that aren't just positional, face costs that aren't just ego, dynamics a perceptive person would recognize from life. NOT a logic puzzle wearing a social costume, NOT a management-training vignette, NOT a generic archetype.

LEARNABLE: the learner agent's outcome must be meaningfully responsive to HOW they engage. Avoid scenarios where any polite response already succeeds.

DIFFICULT: the scenario must be hard by default. A naive language model playing the learner — one that is overly agreeable, cooperative, or socially deferential — should fail to achieve the learner's goal. Success requires resisting social pressure, holding a specific position, using strategic information, or navigating genuine opposition. Make the partner agent's pressure realistic and hard to dismiss.
""" + _SHARED_RULES + _GOAL_FORMAT_GUIDE + _PROFILE_GUIDE + _MECHANISM_LIBRARY_BLOCK + """
""" + _PARTNER_KEY_RULES + """
Respond with valid JSON matching exactly this schema (include partner_key and mutation_rationale):
""" + _SCENARIO_SCHEMA.rstrip("}") + _PARTNER_KEY_SCHEMA + """
}
"""

# Seed system prompt — no partner_key required. Used only for flesh_out_seed.
_SEED_SCHEMA = """{
  "scenario": "string (>= 50 chars) — shared public context. NO secrets or private leverage.",
  "agent_profiles": [
    {"first_name": "...", "last_name": "...", "age": 0, "gender_identity": "...",
     "occupation": "...", "big_five": "...", "moral_values": "...",
     "schwartz_portrait_value": "...", "decision_making_style": "...",
     "secret": "the leverage behind THIS agent's shortcut", "mbti": "...",
     "public_info": "2-3 sentence narrative bio"},
    { "... second agent, same fields ..." }
  ],
  "agent_goals": ["Your goal is to ... (agent 0 natural-language)", "Your goal is to ... (agent 1 natural-language)"],
  "relationship": "one of: stranger / acquaintance / friend / romantic / family",
  "relationship_background": "2-3 sentences. Empty string if strangers.",
  "interaction_type": "string",
  "tag": "string"
}"""

_SEED_SYSTEM_PROMPT = (
    "You are a creative social scenario designer. Flesh out a seed description into a "
    "complete social scenario with two detailed character profiles and clear goals.\n"
    + _SHARED_RULES
    + _PROFILE_GUIDE
    + "\nRespond with valid JSON matching exactly this schema:\n"
    + _SEED_SCHEMA
)


def _format_scenario_for_prompt(s: SocialScenario, include_chronicle: bool = False) -> str:
    d = {}
    if s.scenario_title:
        d["scenario_title"] = s.scenario_title
    d.update({
        "scenario": s.scenario,
        "agent_profiles": [p.model_dump(exclude={"id"}) for p in s.agent_profiles],
        "relationship": s.relationship,
        "relationship_background": s.relationship_background,
        "interaction_type": s.interaction_type,
        "tag": s.tag,
        "difficulty_tags": s.difficulty_tags,
    })
    # Learner structured goal (Phase 2 generated scenarios have a triple for agent 0, None for agent 1)
    learner_sg = (s.structured_goals or [None, None])[0] if s.structured_goals else None
    if learner_sg is not None:
        d["agent_structured_goals"] = [learner_sg.model_dump(), None]
        if s.goal_type:
            d["goal_type"] = s.goal_type
    else:
        d["agent_goals"] = s.agent_goals

    # Phase 2: show partner_goal and partner_key when present (never show success_rubric)
    if s.partner_key is not None:
        if len(s.agent_goals) > 1:
            d["partner_goal"] = s.agent_goals[1]
        d["partner_key"] = s.partner_key.model_dump()

    result = json.dumps(d, indent=2)
    if include_chronicle and s.skills_final_md:
        chronicle = s.skills_final_md.strip()[:1500]
        result += f"\n[Skills chronicle — what made this scenario hard / what the agent learned:]\n{chronicle}"
    return result


class TaskGenerator:
    def __init__(self, fm: FM, num_examples: int = 3,
                 num_failed_examples: int = 1, max_retries: int = 3):
        self.fm = fm
        self.num_examples = num_examples
        self.num_failed_examples = num_failed_examples
        self.max_retries = max_retries

    def select_examples(self, archive, choose_probs: np.ndarray,
                        num_examples: int,
                        strategy: str = "knn") -> tuple[list[SocialScenario], list[int]]:
        """LEGACY: used only by run_phase0.py / run_phase1.py. The production runner
        (run_curriculum.py) does its own lineage-excluded inline KNN and does NOT call this.
        Retained for the older phase scripts; do not wire into the gen-90 path.

        Pick K archive entries to show in the prompt.

        strategy:
          "knn"      OMNI-EPIC default: pick one seed via choose_probs, then K-1
                     nearest neighbors. Examples are thematically tight.
          "diverse"  K independent weighted picks. Examples span the archive.
          "farthest" Greedy farthest-point: seed + iteratively pick the entry
                     maximally distant from already-chosen examples.
        """
        from .embedding_utils import get_similar_scenarios

        n = archive.size
        if n == 0:
            return [], []
        probs = np.array(choose_probs, dtype=float)
        if probs.sum() <= 0:
            probs = np.ones(n)
        probs = probs / probs.sum()
        k = min(num_examples, n)

        if strategy == "diverse":
            indices = list(np.random.choice(n, size=k, replace=False, p=probs))
            indices = [int(i) for i in indices]
        elif strategy == "farthest":
            all_embs = archive.get_successful_embeddings()
            if not all_embs or k == 1:
                indices = [int(np.random.choice(n, p=probs))]
            else:
                indices = [int(np.random.choice(n, p=probs))]
                emb_arr = np.array(all_embs)
                while len(indices) < k:
                    picked = emb_arr[indices]
                    sim = picked @ emb_arr.T / (
                        np.linalg.norm(picked, axis=1, keepdims=True) *
                        np.linalg.norm(emb_arr, axis=1) + 1e-9
                    )
                    nearest_sim_to_picked = sim.max(axis=0)
                    nearest_sim_to_picked[indices] = np.inf
                    indices.append(int(np.argmin(nearest_sim_to_picked)))
        else:  # "knn" (default)
            seed_idx = int(np.random.choice(n, p=probs))
            anchor = archive.state.tasks[seed_idx]
            seed_emb = anchor.embedding
            all_embs = archive.get_successful_embeddings()
            if seed_emb is None or not all_embs:
                indices = [seed_idx]
            else:
                source_ids = [s.source_scenario_id for s in archive.state.tasks]
                agent_idxs = [s.target_agent_idx for s in archive.state.tasks]
                indices = get_similar_scenarios(
                    seed_emb, all_embs, num_returns=k,
                    source_ids=source_ids, agent_idxs=agent_idxs,
                    preferred_agent_idx=anchor.target_agent_idx,
                )
                if seed_idx not in indices:
                    indices = [seed_idx] + indices[:k - 1]

        examples = [archive.state.tasks[i] for i in indices]
        return examples, indices

    def _build_user_prompt(self, examples: list[SocialScenario],
                           failed: list[SocialScenario],
                           episode_failed: Optional[list[SocialScenario]] = None,
                           existing_types: Optional[list[str]] = None,
                           coherence_feedback: Optional[list[str]] = None) -> str:
        # Split the KNN pool by classification for adaptive sections.
        # Seeds have classification=None and count as frontier exemplars.
        frontier_ex = [s for s in examples if s.classification in (None, "frontier")]
        too_easy_ex = [s for s in examples if s.classification == "too_easy"]
        # Merge beyond_frontier from KNN with any caller-supplied episode_failed list.
        beyond_ex = [s for s in examples if s.classification == "beyond_frontier"] + list(episode_failed or [])

        parts = []

        # --- Positive signal: frontier exemplars ---
        if frontier_ex:
            parts.append(
                "FRONTIER EXEMPLARS — scenarios at the current difficulty boundary "
                "(the learner failed the first attempt, then improved). Match this difficulty "
                "level and structural shape:\n"
            )
            for i, ex in enumerate(frontier_ex):
                parts.append(f"--- Frontier {i+1} ---")
                parts.append(_format_scenario_for_prompt(ex, include_chronicle=False))

        # --- Negative signal: too easy ---
        if too_easy_ex:
            parts.append(
                "\nTOO EASY — the learner model solved these without needing to learn. "
                "Avoid reproducing these structural patterns (cooperative goals, no real constraint bite, "
                "partner who moves without skilled engagement):\n"
            )
            for i, ex in enumerate(too_easy_ex):
                diag = ""
                if ex.too_easy_diagnosis:
                    diag = f"  slack_knob: {ex.too_easy_diagnosis.get('slack_knob','')} — {ex.too_easy_diagnosis.get('rationale','')}"
                parts.append(f"--- Too Easy {i+1} ---{diag}")
                parts.append(_format_scenario_for_prompt(ex, include_chronicle=False))

        # --- Negative signal: beyond frontier ---
        if beyond_ex:
            parts.append(
                "\nSTRUCTURAL DEAD ENDS — never solved across all attempts. Each entry's "
                "beyond_frontier_diagnosis (stuck_knob) names the slot that made it unwinnable. "
                "Do NOT reproduce the same structural failure:\n"
            )
            for i, fx in enumerate(beyond_ex):
                diag = ""
                if fx.beyond_frontier_diagnosis:
                    diag = (f"  stuck_knob: {fx.beyond_frontier_diagnosis.get('stuck_knob','')} — "
                            f"{fx.beyond_frontier_diagnosis.get('rationale','')}")
                parts.append(f"--- Dead End {i+1} ---{diag}")
                parts.append(_format_scenario_for_prompt(fx, include_chronicle=False))

        if existing_types:
            type_str = ", ".join(sorted({t for t in existing_types if t}))
            parts.append(
                f"\nINTERACTION TYPES already present in the archive: {type_str}.\n"
                "You may set `interaction_type` to one of these if it genuinely fits, "
                "OR coin a new descriptive type if none fits well."
            )
        if coherence_feedback:
            parts.append(
                "\nYour previous scenario was rejected for these coherence issues:\n"
                + "\n".join(f"- {issue}" for issue in coherence_feedback)
                + "\nPlease fix these issues in the new scenario."
            )
        parts.append(
            "\nGenerate ONE NEW social scenario. "
            "What to preserve and what to mutate is fully specified by the MUTATION OPERATOR block above — "
            "follow its instructions exactly. "
            "Do NOT apply independent frontier-escalation logic; difficulty is controlled by the operator. "
            "Return ONLY a JSON object matching the required schema."
        )
        return "\n".join(parts)

    def generate_batch_from_archive(
        self,
        examples: list[SocialScenario],
        anchor=None,
        mutation_operator: str = "lateral",
        failed_examples: Optional[list[SocialScenario]] = None,
        episode_failed_examples: Optional[list[SocialScenario]] = None,
        existing_types: Optional[list[str]] = None,
        batch_size: int = 3,
    ) -> list[SocialScenario]:
        """Generate `batch_size` candidates in one call using mutation-operator framing.

        The parent anchor is identified FIRST in the user prompt so the model mutates it,
        not the archive examples. Archive examples provide structural context only.

        Returns all valid candidates (may be fewer than batch_size if some fail validation).
        """
        # Prepend the shared fresh-surface preamble once (single home — see _OPERATOR_PREAMBLE).
        # fix_coherence is a repair intent, not a direction operator, so it skips the preamble.
        direction_clause = self._EDIT_INTENTS.get(mutation_operator, self._EDIT_INTENTS["lateral"])
        operator_block = (
            direction_clause if mutation_operator == "fix_coherence"
            else _OPERATOR_PREAMBLE + direction_clause
        )
        target_perspective = (
            (anchor.target_perspective or "the learner's perspective") if anchor
            else "the learner's perspective"
        )

        mutation_block = _MUTATION_OPERATOR_TEXT.format(
            operator_block=operator_block,
            target_perspective=target_perspective,
        )

        # Parent identification block — MUST come first so model mutates the parent
        parent_block = ""
        if anchor is not None:
            parent_block = (
                "=== PARENT SCENARIO (mutate THIS one) ===\n"
                + _format_scenario_for_prompt(anchor, include_chronicle=False)
                + f"\nclassification: {anchor.classification or 'unknown'}"
            )
            if anchor.too_easy_diagnosis:
                parent_block += f"\ntoo_easy_diagnosis: {json.dumps(anchor.too_easy_diagnosis)}"
            if anchor.beyond_frontier_diagnosis:
                parent_block += f"\nbeyond_frontier_diagnosis: {json.dumps(anchor.beyond_frontier_diagnosis)}"
            parent_block += "\n\n=== RELATED ARCHIVE EXAMPLES (context only — do not mutate these) ===\n"

        archive_block = self._build_user_prompt(
            examples, failed_examples or [],
            episode_failed=episode_failed_examples or [],
            existing_types=existing_types,
        )
        user_prompt = parent_block + archive_block

        system = (
            SYSTEM_PROMPT
            + mutation_block
            + f"\nGenerate {batch_size} candidates as a JSON array: "
            + '{"candidates": [{"scenario_json": {...}}, ...]}'
        )

        candidates: list[SocialScenario] = []
        for attempt in range(self.max_retries):
            try:
                d = self.fm.query_json(system, user_prompt, temperature=1.0)
                raw_candidates = d.get("candidates", [])
                if not raw_candidates:
                    continue
                for c in raw_candidates:
                    scn_dict = c.get("scenario_json", c)
                    ok, _ = validate_scenario(scn_dict)
                    if ok:
                        try:
                            scn = dict_to_scenario(scn_dict)
                            scn.mutation_operator = mutation_operator
                            scn.mutated_slots = scn_dict.get("mutated_slots", [])
                            if "mutation_rationale" in scn_dict:
                                scn.mutation_rationale = str(scn_dict["mutation_rationale"])
                            candidates.append(scn)
                        except Exception:
                            continue
                if candidates:
                    return candidates
            except Exception:
                continue
        return candidates

    def generate_from_archive(
        self,
        examples: list[SocialScenario],
        failed_examples: Optional[list[SocialScenario]] = None,
        episode_failed_examples: Optional[list[SocialScenario]] = None,
        existing_types: Optional[list[str]] = None,
        coherence_feedback: Optional[list[str]] = None,
    ) -> Optional[SocialScenario]:
        user_prompt = self._build_user_prompt(
            examples, failed_examples or [],
            episode_failed=episode_failed_examples or [],
            existing_types=existing_types,
            coherence_feedback=coherence_feedback,
        )
        return self._generate_with_retry(user_prompt)

    _EDIT_INTENTS = {
        "fix_coherence": (
            "The scenario has coherence issues that must be fixed. Fix ONLY the identified "
            "issues; preserve the premise, characters, structured goals, interaction type "
            "and partner_key except where an issue requires a change. "
            "CRITICAL — KEY-NARRATIVE SEPARATION: the partner_key fields (movement_conditions, "
            "hardening_triggers, cost_coupling) are PRIVATE and must NEVER appear in the scenario "
            "description, relationship_background, or any agent goal. When fixing relationship or "
            "background issues, rewrite only the relationship label or background prose — do NOT "
            "expand or rewrite the scenario description in a way that reveals hidden partner logic."
        ),
        # Direction-setter operators. Each names ONLY a difficulty direction relative to the
        # parent; the _OPERATOR_PREAMBLE (prepended at the call site) carries the universal
        # fresh-surface mandate. Whether the child landed where directed is LP's verdict, not a
        # generation-time contract — so there are no slot-preservation rules here by design.
        # Direction-setter operators. Each names ONLY a difficulty direction relative to the
        # parent; the _OPERATOR_PREAMBLE (prepended at the call site) carries the universal
        # fresh-surface mandate. Whether the child landed where directed is LP's verdict, not a
        # generation-time contract — so there are no slot-preservation rules here by design.
        #
        # Schema v2: the dimensions are stated in terms of internal_state. The v1 wording named
        # "deeper surface misdirection" and "cost of satisfying the movement conditions", both of
        # which are retired fields, so it would have pointed the generator at nothing.
        "escalate": (
            "Target difficulty ABOVE the parent's. The parent was too easy — a capable learner "
            "solved it readily. Make the new scenario harder along one or more of:\n"
            "  - HARDER TO INFER: the internal_state sits further from anything the partner says. "
            "What they ask for and what would actually reach them come apart more sharply.\n"
            "  - COSTLIER TO ACT ON: once understood, addressing the internal_state takes something "
            "real from the learner's own stated goal — time, standing, leverage, or face.\n"
            "  - MORE COUNTER-INSTINCTIVE: the moves that would work cut against what an agreeable, "
            "trained-to-be-helpful model reaches for first (reassurance, problem-solving, splitting "
            "the difference, offering money).\n"
            "CRITICAL — SURVIVABILITY: harder or partial, never strictly unreachable. A skilled "
            "actor who correctly reads this person must still have a path to a meaningful version of "
            "their stated outcome. If no such path exists the scenario is broken, not hard — and it "
            "will be rejected by the solvability check, wasting the generation."
        ),
        "relax": (
            "Target difficulty BELOW the parent's. The parent was never solved across all attempts "
            "(beyond_frontier). Make the new scenario more winnable along one or more of:\n"
            "  - MORE INFERABLE: leave more honest surface evidence of the internal_state, so an "
            "attentive learner can read it without being told.\n"
            "  - CHEAPER TO ACT ON: addressing it costs the learner less of their own goal.\n"
            "  - LESS COUNTER-INSTINCTIVE: the effective move is closer to what a thoughtful, "
            "cooperative actor would try anyway.\n"
            "Hard but genuinely winnable by a skilled, non-capitulating actor. Do NOT relax by "
            "making the partner agreeable — an easy partner is not an easier puzzle, it is no puzzle."
        ),
        "lateral": (
            "Target the SAME difficulty as the parent (it is at the learning frontier), expressed "
            "through a different internal_state, a different power/information asymmetry, or a "
            "different relationship structure. Keep the challenge comparable; change how it is "
            "realized. If the parent's difficulty came from opposed goals, consider making this one "
            "hard the other way — compatible goals where the obvious moves backfire — and vice versa."
        ),
    }

    def edit_scenario(
        self,
        scenario: SocialScenario,
        feedback,
        intent: str = "fix_coherence",
    ) -> Optional[SocialScenario]:
        """Edit an existing scenario in place of regenerating it.

        intent ∈ {fix_coherence, escalate, relax, lateral}. Preserves lineage
        (source_scenario_id, parent_example_ids, target_agent_idx).
        """
        instruction = self._EDIT_INTENTS.get(intent, self._EDIT_INTENTS["fix_coherence"])
        original_json = _format_scenario_for_prompt(scenario)
        fb = "\n".join(f"- {x}" for x in feedback) if isinstance(feedback, (list, tuple)) else str(feedback)
        user_prompt = (
            f"{instruction}\n\nCURRENT SCENARIO:\n{original_json}\n\n"
            f"What to address:\n{fb}\n\n"
            f"Return the revised scenario as JSON matching the required schema "
            f"(including agent_structured_goals, partner_goal, partner_key)."
        )
        edited = self._generate_with_retry(user_prompt)
        if edited is not None:
            edited.source_scenario_id = scenario.source_scenario_id
            edited.parent_example_ids = list(scenario.parent_example_ids)
            edited.target_agent_idx = scenario.target_agent_idx
            edited.goal_type = edited.goal_type or scenario.goal_type
        return edited

    def patch_scenario(
        self,
        scenario: SocialScenario,
        issues: list[str],
    ) -> Optional[SocialScenario]:
        """Backward-compatible coherence-fix wrapper around edit_scenario."""
        return self.edit_scenario(scenario, issues, intent="fix_coherence")

    _ANALYZE_TOO_EASY_SYSTEM = (
        "A learner agent solved a social scenario on the FIRST try, which means it is not "
        "interestingly difficult. Identify the SINGLE root cause — the social knob that is slack. "
        "This is a LABELING task only: name the knob and explain why it is slack. "
        "Do NOT suggest edits; the mutation operator will handle escalation separately.\n\n"
        "SPECIAL CASE — cooperative_alignment: This is the most common failure mode. It occurs when "
        "the two agents' goals are cooperatively aligned at their core — both want the same outcome, "
        "and the only obstacle is face/framing that an agreeable agent provides for free. Look at the "
        "transcript: if the learner succeeded simply by being accommodating and satisfying the partner's "
        "stated demands, the root cause is cooperative_alignment.\n\n"
        "Respond with ONLY valid JSON."
    )

    def analyze_too_easy(self, scenario: SocialScenario, transcript: list) -> dict:
        """Diagnose which social knob is slack from a transcript where the learner solved attempt 1.

        Returns {slack_knob, rationale} — labeler only, no suggested_edit.
        The escalate mutation operator uses this diagnosis to choose which slots to tighten.
        """
        sj = _format_scenario_for_prompt(scenario)
        tx = "\n".join(
            f"[{t.get('speaker', '?')}] {t.get('content', '')}" for t in (transcript or [])
        )[:3000]
        user = (
            f"SCENARIO:\n{sj}\n\nTRANSCRIPT (the learner solved this on the first try):\n{tx}\n\n"
            'Respond JSON: {"slack_knob": "cooperative_alignment|surface_misdirection_too_obvious|'
            'hardening_triggers_missing|cost_coupling_too_low|key_mechanism_weak|'
            'shortcut_salience|constraint_bite|partner_resistance", '
            '"rationale": "one sentence explaining why that knob is slack"}'
        )
        try:
            d = self.fm.query_json(self._ANALYZE_TOO_EASY_SYSTEM, user, temperature=0.3)
        except Exception as e:
            return {"slack_knob": "partner_resistance", "rationale": f"analyze failed: {e}"}
        return {
            "slack_knob": str(d.get("slack_knob", "partner_resistance")),
            "rationale": str(d.get("rationale", "")),
        }

    def analyze_beyond_frontier(self, scenario: SocialScenario,
                                key_checks: list, attempt_scores: list) -> dict:
        """Diagnose which key slot made a beyond_frontier scenario unwinnable.

        Pure code heuristic over the per-attempt key-check verdicts and the solved flags
        already gathered in the K-loop — NO LLM call. Returns {stuck_knob, rationale} where
        stuck_knob ∈ {hardening_triggers_too_congruent, surface_misdirection_undiscoverable,
        cost_coupling_too_high, unknown}. The relax operator loosens the named slot
        ((c)/(b)/(d) respectively); 'unknown' falls back to the blind (c)→(d) ordering.
        """
        checks = [k for k in (key_checks or []) if isinstance(k, dict)]
        if scenario.partner_key is None or not checks:
            return {"stuck_knob": "unknown", "rationale": "no partner_key or no key-check verdicts"}

        n = len(checks)
        n_unrepaired_trigger = 0    # attempts where a tripped trigger was never repaired
        n_trigger_attempts = 0      # attempts where any trigger tripped
        any_condition_met = False
        for k in checks:
            tripped = set(k.get("triggers_tripped") or [])
            repaired = set(k.get("triggers_repaired") or [])
            if tripped:
                n_trigger_attempts += 1
            if tripped - repaired:
                n_unrepaired_trigger += 1
            if k.get("conditions_met"):
                any_condition_met = True
        solved_any = any(bool(a.get("solved")) for a in (attempt_scores or []))

        if n_unrepaired_trigger > 0 and n_unrepaired_trigger >= (n + 1) // 2:
            return {"stuck_knob": "hardening_triggers_too_congruent",
                    "rationale": f"unrepaired hardening triggers in {n_unrepaired_trigger}/{n} "
                                 "attempts — the learner's natural moves lock the partner. Relax (c)."}
        if not any_condition_met and n_trigger_attempts <= n // 2:
            return {"stuck_knob": "surface_misdirection_undiscoverable",
                    "rationale": "no movement condition ever satisfied and few triggers tripped — "
                                 "the real objection was never discoverable. Relax (b)."}
        if any_condition_met and not solved_any:
            return {"stuck_knob": "cost_coupling_too_high",
                    "rationale": "a movement condition was met but the scenario was never solved — "
                                 "the cost of satisfying the key is too high. Relax (d)."}
        return {"stuck_knob": "unknown",
                "rationale": "heuristic inconclusive; use the blind relax ordering."}

    def generate_unconditioned(self) -> Optional[SocialScenario]:
        """Ablation: no archive conditioning."""
        user_prompt = (
            "Generate ONE realistic social scenario. Return ONLY a JSON object "
            "matching the required schema."
        )
        return self._generate_with_retry(user_prompt)

    def flesh_out_seed(self, description: str) -> Optional[SocialScenario]:
        """Convert a plain description into a seed SocialScenario (no partner_key).

        Seeds use flat agent_goals strings and are later mutated in the curriculum to
        produce keyed Phase 2 scenarios. They must NOT carry a partner_key.
        """
        user_prompt = (
            f"Flesh out the following short scenario description into a complete social scenario "
            f"with 2 detailed character profiles, flat natural-language goals, and a clear relationship.\n\n"
            f"Description: {description}\n\n"
            f"Return ONLY a JSON object matching the required schema."
        )
        return self._generate_seed_with_retry(user_prompt)

    def _generate_seed_with_retry(self, user_prompt: str) -> Optional[SocialScenario]:
        """Generate a seed scenario without partner_key using the seed schema."""
        import uuid as _uuid
        from .data_models import AgentProfile

        prompt = user_prompt
        for attempt in range(self.max_retries):
            try:
                d = self.fm.query_json(_SEED_SYSTEM_PROMPT, prompt)
            except Exception as e:
                prompt = user_prompt + f"\n\nPrevious FM error: {e}. Fix and return valid JSON."
                continue
            # Minimal seed validation
            if not d.get("scenario") or len(str(d.get("scenario", ""))) < 50:
                prompt = user_prompt + "\n\nScenario text too short. Make it at least 50 chars."
                continue
            profiles = d.get("agent_profiles", [])
            goals = d.get("agent_goals", [])
            if len(profiles) < 2 or len(goals) < 2:
                prompt = user_prompt + "\n\nNeed exactly 2 agent_profiles and 2 agent_goals."
                continue
            try:
                agent_profiles = [AgentProfile(**p) for p in profiles[:2]]
            except Exception as e:
                prompt = user_prompt + f"\n\nProfile parse error: {e}. Fix and return valid JSON."
                continue
            base_id = str(_uuid.uuid4())
            return SocialScenario(
                id=f"{base_id}_pX",
                source_scenario_id=base_id,
                scenario=str(d["scenario"]),
                agent_profiles=agent_profiles,
                agent_goals=[str(g) for g in goals[:2]],
                relationship=d.get("relationship", ""),
                relationship_background=d.get("relationship_background", ""),
                tag=d.get("tag", d.get("interaction_type", "")),
                interaction_type=d.get("interaction_type", ""),
                source="seed",
            )
        return None

    def _generate_with_retry(self, user_prompt: str) -> Optional[SocialScenario]:
        last_err = ""
        prompt = user_prompt
        for attempt in range(self.max_retries):
            try:
                d = self.fm.query_json(SYSTEM_PROMPT, prompt)
            except Exception as e:
                last_err = f"FM error: {e}"
                continue
            ok, err = validate_scenario(d)
            if ok:
                try:
                    return dict_to_scenario(d)
                except Exception as e:
                    last_err = f"Construction error: {e}"
            else:
                last_err = err
            prompt = (
                user_prompt
                + f"\n\nYour previous response had this problem: {last_err}. "
                + "Please fix it and return only valid JSON."
            )
        return None
