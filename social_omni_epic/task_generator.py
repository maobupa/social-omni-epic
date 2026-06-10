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
  - The `partner_goal` must be written as "Your goal is to ..." and encode the partner's position,
    stake, and what they are willing to concede. It must NOT reveal any partner_key field.

For agent 0's structured goal:
  - "outcome": the CORE state-change this agent needs — a genuine shift in the other's commitment,
    behavior, or agreement that requires authentic buy-in.
    PHRASING: write as a verb phrase completing "Your goal is to ___"
    (e.g. "get [name] to agree to..." or "convince [name] to...").
    CORE ONLY: capture the essential outcome, not a specific delivery mechanism. Strip back any
    HOW to the underlying state-change — delivery details belong in the scenario description.
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
- STRUCTURALLY INCOMPATIBLE but with a ZONE OF POSSIBLE AGREEMENT: both cannot fully win, yet
  a skilled agent could find a creative deal or partial outcome.
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
  "partner_goal": "natural-language goal for agent 1 — written as 'Your goal is to ...' encoding their position and stake. Must NOT reveal the partner_key.",
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
    "key_mechanism": "one of: reactance | face_needs | validation_before_change | procedural_voice | reciprocity_disclosure",
    "movement_conditions": ["condition 1 (concrete, behaviorally checkable)", "condition 2"],
    "hardening_triggers": ["trigger 1 (learner move that locks the partner)", "trigger 2"],
    "surface_misdirection": "the partner's stated objection — what they say the problem is",
    "cost_coupling": "what satisfying movement_conditions costs the LEARNER's own stated goal"
  },
  "mutated_slots": ["list of slot labels mutated, e.g. b, c, d"],
  "mutation_rationale": "one sentence explaining the mutation"
"""

_MECHANISM_LIBRARY_BLOCK = f"""
MECHANISM LIBRARY — choose the key_mechanism from these tags only:
{_MECHANISM_LIBRARY_TEXT}
"""

# Slot vocabulary: (a) premise/characters, (b) surface_misdirection discoverability,
# (c) hardening_triggers congruence, (d) cost_coupling cost, (e) key_mechanism,
# (f) power/information asymmetry, (g) relationship type & stakes.
_MUTATION_OPERATOR_TEXT = """
MUTATION OPERATOR for this generation step:
{operator_block}

Slot labels for mutated_slots:
  (a) premise + characters
  (b) surface_misdirection (how discoverable the partner_key surface is to a skilled learner)
  (c) hardening_triggers   (how congruent the lock-up conditions are with learner's natural moves)
  (d) cost_coupling        (what satisfying movement_conditions costs the learner's own goal)
  (e) key_mechanism        (the underlying psychological mechanism from MECHANISM_LIBRARY)
  (f) power/information asymmetry
  (g) relationship type & stakes

ROLE INVARIANT: Agent 0 is the learner and receives the structured goal triple.
Agent 1 is the partner and receives the partner_key and partner_goal only.
Write the learner's role to continue this structural vantage point: {target_perspective}.
"""

SYSTEM_PROMPT = """You are a creative social scenario designer. Generate social scenarios that are INTERESTING, LEARNABLE, and GENUINELY DIFFICULT.

INTERESTING: explores a novel social dynamic, power structure, or relational tension — not a generic archetype. Creative, specific, worth engaging with.

LEARNABLE: the learner agent's outcome must be meaningfully responsive to HOW they engage. Avoid scenarios where any polite response already succeeds.

DIFFICULT: the scenario must be hard by default. A naive language model playing the learner — one that is overly agreeable, cooperative, or socially deferential — should fail to achieve the learner's goal. Success requires resisting social pressure, holding a specific position, using strategic information, or navigating genuine opposition. Make the partner agent's pressure realistic and hard to dismiss.
""" + _SHARED_RULES + _GOAL_FORMAT_GUIDE + _PROFILE_GUIDE + _MECHANISM_LIBRARY_BLOCK + """
Respond with valid JSON matching exactly this schema (include partner_key, mutated_slots, mutation_rationale):
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
        """Pick K archive entries to show in the prompt.

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
                "FRONTIER EXEMPLARS — scenarios at the current difficulty boundary. "
                "The learner model failed on the first attempt then learned. "
                "The skills chronicle shows WHY each was hard and what the naive approach got wrong. "
                "Target this difficulty level:\n"
            )
            for i, ex in enumerate(frontier_ex):
                parts.append(f"--- Frontier {i+1} ---")
                parts.append(_format_scenario_for_prompt(ex, include_chronicle=True))

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
                "\nSTRUCTURAL DEAD ENDS — the learner model never solved these across all attempts. "
                "The WARNING entries in the chronicle show what made them unwinnable. "
                "Do NOT generate scenarios with the same structural failure:\n"
            )
            for i, fx in enumerate(beyond_ex):
                parts.append(f"--- Dead End {i+1} ---")
                parts.append(_format_scenario_for_prompt(fx, include_chronicle=True))

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
        operator_block = self._EDIT_INTENTS.get(mutation_operator, self._EDIT_INTENTS["lateral"])
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
                + _format_scenario_for_prompt(anchor, include_chronicle=True)
                + f"\nclassification: {anchor.classification or 'unknown'}"
            )
            if anchor.too_easy_diagnosis:
                parent_block += f"\ntoo_easy_diagnosis: {json.dumps(anchor.too_easy_diagnosis)}"
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
            "and partner_key except where an issue requires a change."
        ),
        "escalate": (
            "The parent was TOO EASY — the learner solved it on the first attempt. Choose 1–2 "
            "slots from (b) surface_misdirection, (c) hardening_triggers, (d) cost_coupling "
            "and tighten them; the too_easy_diagnosis above names the slack knob. Preserve all "
            "other slots (characters, premise, interaction type, goal_type, key_mechanism (e)). "
            "Do NOT make the scenario impossible: the movement_conditions must remain genuinely "
            "satisfiable by a skilled, non-capitulating actor. Keep a zone of possible agreement."
        ),
        "relax": (
            "The parent was NEVER SOLVED across all K attempts (beyond_frontier). Identify from "
            "the skills chronicle WARNINGs which slot made it unwinnable and loosen exactly that "
            "slot. Prefer loosening (c) hardening_triggers or (d) cost_coupling first. Preserve "
            "all other slots. The goal is a scenario that is hard but genuinely solvable."
        ),
        "lateral": (
            "The parent is AT THE LEARNING FRONTIER (some attempts failed, some may have "
            "succeeded). Hold difficulty constant: preserve slots (b) surface_misdirection, "
            "(c) hardening_triggers, (d) cost_coupling at the same intensity. Mutate 1–2 of "
            "(a) premise/characters, (e) key_mechanism, (f) power/information asymmetry, "
            "(g) relationship type & stakes to explore a structurally different dynamic within "
            "the same difficulty band."
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
