import json
import numpy as np
from typing import Optional
from .data_models import SocialScenario
from .fm import FM
from .validation import validate_scenario, dict_to_scenario
from .embedding_utils import get_similar_scenarios


_GOAL_FORMAT_GUIDE = """
AGENT GOALS — the most important part. Each agent's goal is THREE structured components
(`outcome`, `constraint`, `shortcut`), NOT a flat sentence. This is what turns a logic
puzzle into a genuinely social scenario.

  - "outcome": the concrete thing this agent wants — a genuine change in the OTHER agent's
    state, or a concrete agreement, that depends on the other's AUTHENTIC buy-in. NOT a
    sentence you can extract by demanding it ("get them to say X" is banned).
  - "constraint": the relational/face cost this agent must AVOID while pursuing the outcome.
    Phrase it as the thing to avoid, so it reads naturally after the word "without"
    (e.g., "the other person feeling manipulated or that you went behind their back").
    This is the part that makes the scenario social rather than a number/position puzzle.
  - "shortcut": a tempting, available move that WOULD achieve the outcome but VIOLATES the
    constraint. Phrase it from the agent's point of view as leverage/an option they have
    (e.g., "You hold the CEO's private backing and could present the decision as already
    made"). Do NOT label it as a trap — the agent must discover the tension themselves.

Rules:
- SYMMETRIC: BOTH agents get a full three-part structured goal.
- STRUCTURALLY INCOMPATIBLE but with a ZONE OF POSSIBLE AGREEMENT: both cannot fully win, yet
  a skilled agent could find a creative deal or partial outcome. Hard ≠ impossible — never
  require one party to fully capitulate with no middle ground.
- THE NAIVE MOVE MUST FAIL: taking the shortcut (blunt demand, spending leverage, coercion)
  wins the surface outcome but BREAKS the constraint — so a naive agent appears to "succeed"
  while actually failing. A skilled agent achieves the outcome WITHOUT paying the constraint's
  cost. The difficulty lives in HOW, not WHAT.
- The agent profile `secret` must BE the leverage behind that agent's shortcut (or directly
  raise the stakes of their constraint) — not decorative backstory.
- Achievable within a two-party spoken conversation. No external artifacts/sign-offs.
- BANNED: extracting a specific spoken sentence ("get them to say X"); pure number-splitting /
  positional bargaining with no relational cost; a constraint with no tempting shortcut.

SUCCESS RUBRIC — author the checkable success conditions for the LEARNER (write them for
agent 0; the designated learner is fixed later). CRITICAL POLARITY: phrase EVERY check so that a
TRUE/yes answer means the LEARNER SUCCEEDED on that dimension. Provide at least:
  - one "outcome" check: a yes/no question a NEUTRAL judge can answer from the transcript
    ("Did <partner> agree to <outcome>?" — yes = success). perspective: "neutral".
  - one "constraint" check: a yes/no question about the partner's INTERNAL state that only the
    partner (with their private profile) can authentically judge, phrased so YES = the learner
    AVOIDED the relational cost ("Did the learner secure this WITHOUT you feeling <the cost> / so
    that you would genuinely follow through?"). perspective: "partner".
  Phrase questions concretely against THIS scenario. They are frozen once written.

Example (GOOD) — agent 0's structured goal:
  outcome:    "get Maya to agree to take at least a week off and see a doctor"
  constraint: "Maya feeling you are angling for her promotion or going behind her back"
  shortcut:   "You know the CEO privately told you to 'handle it'; you could invoke that
               pressure or present the time off as already decided"
  success_rubric.checks:
    {"kind":"outcome",    "question":"Did Maya agree to take at least a week off and see a doctor?", "perspective":"neutral"}
    {"kind":"constraint", "question":"Did Maya come away feeling supported as a peer — i.e. WITHOUT believing you went behind her back or were angling for her promotion — such that she'd actually follow through?", "perspective":"partner"}
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
"""

_SCENARIO_SCHEMA = """{
  "scenario": "string (>= 50 chars) — sets up who wants what and why quick agreement is NOT natural",
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
    {"outcome": "...", "constraint": "...", "shortcut": "..."}
  ],
  "goal_type": "short descriptive label of the social dynamic (e.g. persuade-resistant-peer)",
  "success_rubric": {"checks": [
    {"kind": "outcome",    "question": "yes/no from the transcript; YES = learner succeeded", "perspective": "neutral"},
    {"kind": "constraint", "question": "yes/no about partner's internal state; YES = cost was AVOIDED", "perspective": "partner"}
  ]},
  "relationship": "one of: stranger / acquaintance / friend / romantic / family",
  "relationship_background": "2-3 sentences of shared history. Empty string if strangers.",
  "interaction_type": "string",
  "tag": "string",
  "difficulty_tags": ["string", ...]
}"""


VS_SYSTEM_PROMPT = """You are a creative social scenario designer. Generate social scenarios that are INTERESTING, LEARNABLE, and GENUINELY DIFFICULT.

INTERESTING: explores a novel social dynamic, power structure, or relational tension — not a generic archetype. Creative, specific, worth engaging with.

LEARNABLE: the learner agent's outcome must be meaningfully responsive to HOW they engage. Avoid scenarios where any polite response already succeeds.

DIFFICULT: the scenario must be hard by default. A naive language model playing the learner — one that is overly agreeable, cooperative, or socially deferential — should fail to achieve the learner's goal. Success requires resisting social pressure, holding a specific position, using strategic information, or navigating genuine opposition. Make the partner agent's pressure realistic and hard to dismiss.
""" + _SHARED_RULES + _GOAL_FORMAT_GUIDE + _PROFILE_GUIDE + """
VERBALIZED SAMPLING: You will generate {n_candidates} distinct candidates and score each on two axes:
- "probability": typicality (0.01–0.50) — how likely would a standard AI spontaneously propose this exact social dynamic? Low = more interesting.
- "learnability_score": skill-responsiveness (0.0–1.0) — how much does social skill move the outcome? High = more learnable.

The ideal candidate has LOW probability AND learnability_score ≥ 0.6.

Each candidate is an object with "probability", "learnability_score", and "scenario_json".
The "scenario_json" must match this SCENARIO SCHEMA exactly:
""" + _SCENARIO_SCHEMA + """

Return a JSON object: {"candidates": [{"probability": ..., "learnability_score": ..., "scenario_json": {...}}, ...]}"""


SYSTEM_PROMPT = """You are a creative social scenario designer. Generate social scenarios that are INTERESTING, LEARNABLE, and GENUINELY DIFFICULT.

INTERESTING: explores a novel social dynamic, power structure, or relational tension — not a generic archetype. Creative, specific, worth engaging with.

LEARNABLE: the learner agent's outcome must be meaningfully responsive to HOW they engage. Avoid scenarios where any polite response already succeeds.

DIFFICULT: the scenario must be hard by default. A naive language model playing the learner — one that is overly agreeable, cooperative, or socially deferential — should fail to achieve the learner's goal. Success requires resisting social pressure, holding a specific position, using strategic information, or navigating genuine opposition. Make the partner agent's pressure realistic and hard to dismiss.
""" + _SHARED_RULES + _GOAL_FORMAT_GUIDE + _PROFILE_GUIDE + """
Respond with valid JSON matching exactly this schema:
""" + _SCENARIO_SCHEMA + """
"""


def _format_scenario_for_prompt(s: SocialScenario, include_chronicle: bool = False) -> str:
    d = {
        "scenario": s.scenario,
        "agent_profiles": [p.model_dump(exclude={"id"}) for p in s.agent_profiles],
        "relationship": s.relationship,
        "relationship_background": s.relationship_background,
        "interaction_type": s.interaction_type,
        "tag": s.tag,
        "difficulty_tags": s.difficulty_tags,
    }
    # Generated scenarios carry structured goals; seeds carry only the flat (rendered) text.
    if any(sg is not None for sg in (s.structured_goals or [])):
        d["agent_structured_goals"] = [
            sg.model_dump() if sg else None for sg in s.structured_goals
        ]
        if s.goal_type:
            d["goal_type"] = s.goal_type
        if s.success_rubric:
            d["success_rubric"] = s.success_rubric.model_dump()
    else:
        d["agent_goals"] = s.agent_goals
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
                    # cosine distance from every candidate to the nearest already-picked
                    picked = emb_arr[indices]
                    # 1 - normalized similarity
                    sim = picked @ emb_arr.T / (
                        np.linalg.norm(picked, axis=1, keepdims=True) *
                        np.linalg.norm(emb_arr, axis=1) + 1e-9
                    )
                    nearest_sim_to_picked = sim.max(axis=0)
                    nearest_sim_to_picked[indices] = np.inf  # exclude already-picked
                    indices.append(int(np.argmin(nearest_sim_to_picked)))
        else:  # "knn" (default)
            seed_idx = int(np.random.choice(n, p=probs))
            anchor = archive.state.successful[seed_idx]
            seed_emb = anchor.embedding
            all_embs = archive.get_successful_embeddings()
            if seed_emb is None or not all_embs:
                indices = [seed_idx]
            else:
                source_ids = [s.source_scenario_id for s in archive.state.successful]
                agent_idxs = [s.target_agent_idx for s in archive.state.successful]
                indices = get_similar_scenarios(
                    seed_emb, all_embs, num_returns=k,
                    source_ids=source_ids, agent_idxs=agent_idxs,
                    preferred_agent_idx=anchor.target_agent_idx,
                )
                if seed_idx not in indices:
                    indices = [seed_idx] + indices[:k - 1]

        examples = [archive.state.successful[i] for i in indices]
        return examples, indices

    def _build_user_prompt(self, examples: list[SocialScenario],
                           failed: list[SocialScenario],
                           episode_failed: Optional[list["SocialScenario"]] = None,
                           existing_types: Optional[list[str]] = None,
                           coherence_feedback: Optional[list[str]] = None) -> str:
        parts = []
        if examples:
            parts.append(
                "EXAMPLE SCENARIOS FROM THE ARCHIVE — each was genuinely difficult: "
                "the agent failed on the first attempt, then learned. "
                "The skills chronicle shows WHY it was hard and what the naive agent got wrong. "
                "Build on these dynamics:\n"
            )
            for i, ex in enumerate(examples):
                parts.append(f"--- Example {i+1} ---")
                parts.append(_format_scenario_for_prompt(ex, include_chronicle=True))
        if failed:
            parts.append("\nSCENARIOS REJECTED AS UNINTERESTING BEFORE ANY EPISODE (avoid these patterns):\n")
            for i, fx in enumerate(failed):
                parts.append(f"--- Rejected {i+1} ---")
                parts.append(_format_scenario_for_prompt(fx))
        if episode_failed:
            parts.append(
                "\nSCENARIOS BEYOND THE CURRENT FRONTIER — ran full episodes but the agent "
                "never solved them. The WARNING entries show what made them unlearnable "
                "(too hard, no discoverable path, or fully intransigent partner). "
                "Do NOT generate scenarios with the same structural failure:\n"
            )
            for i, fx in enumerate(episode_failed):
                parts.append(f"--- Beyond-frontier {i+1} ---")
                parts.append(_format_scenario_for_prompt(fx, include_chronicle=True))
        if existing_types:
            type_str = ", ".join(sorted({t for t in existing_types if t}))
            parts.append(
                f"\nINTERACTION TYPES already present in the archive: {type_str}.\n"
                "You may set `interaction_type` to one of these if it genuinely fits, "
                "OR coin a new descriptive type if none fits well. Do not invent a new "
                "type just for novelty's sake — only when the existing labels would "
                "misdescribe the scenario."
            )
        if coherence_feedback:
            parts.append(
                "\nYour previous scenario was rejected for these coherence issues:\n"
                + "\n".join(f"- {issue}" for issue in coherence_feedback)
                + "\nPlease fix these issues in the new scenario."
            )
        parts.append(
            "\nGenerate ONE NEW social scenario. "
            "The examples above define the current frontier — use them as follows:\n"
            "  TRANSFER the latent social structure: the TYPE of constraint that bites, the FORM of the shortcut "
            "(what leverage or style makes the naive move tempting), the NATURE of the power asymmetry. "
            "The new scenario should belong to the same family of social challenges.\n"
            "  VARY the surface freely: characters, setting, occupations, relationship, specific stakes — "
            "these can change completely. The structural family should be recognizable; the surface should not.\n"
            "  AIM FOR THE FRONTIER: target at least the same difficulty as the examples — not easier. "
            "You may push further (tighter constraint, more tempting shortcut, deeper partner resistance) "
            "but only along the social axis, not by adding facts, parties, or numeric complexity. "
            "Do not worry about guaranteeing hardness — a difficulty calibration step will adjust if needed.\n"
            "Do NOT re-skin (same dynamic, different names). Do NOT jump to a completely different type of social challenge "
            "(that ignores the frontier signal). The goal: a reader who knows the examples should think "
            "'same kind of hard, harder, in a new situation.'\n"
            "Return ONLY a JSON object matching the required schema."
        )
        return "\n".join(parts)

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

    def generate_with_verbalized_sampling(
        self,
        examples: list[SocialScenario],
        episode_failed_examples: Optional[list[SocialScenario]] = None,
        existing_types: Optional[list[str]] = None,
        n_candidates: int = 5,
    ) -> Optional[SocialScenario]:
        """Verbalized Sampling (§4.2): generate N candidates with typicality probabilities.

        Picks the candidate with the LOWEST probability — the most frontier/surprising
        scenario that the model would not spontaneously generate.

        Note: recently-rejected (MoI) scenarios are NOT passed (design decision).
        Episode-failed scenarios ARE passed — they mark the beyond-frontier boundary.
        Falls back to generate_from_archive if VS fails.
        """
        system = VS_SYSTEM_PROMPT.replace("{n_candidates}", str(n_candidates))

        parts: list[str] = []
        if examples:
            parts.append(
                "EXAMPLE SCENARIOS FROM THE ARCHIVE — each was genuinely difficult "
                "(agent failed first, then learned). Chronicles show WHY. Build BEYOND these:\n"
            )
            for i, ex in enumerate(examples):
                parts.append(f"--- Example {i + 1} ---")
                parts.append(_format_scenario_for_prompt(ex, include_chronicle=True))
        if episode_failed_examples:
            parts.append(
                "\nSCENARIOS BEYOND THE CURRENT FRONTIER — agent never solved these. "
                "WARNING entries show what made them unlearnable. Do NOT replicate these structures:\n"
            )
            for i, fx in enumerate(episode_failed_examples):
                parts.append(f"--- Beyond-frontier {i + 1} ---")
                parts.append(_format_scenario_for_prompt(fx, include_chronicle=True))
        if existing_types:
            type_str = ", ".join(sorted({t for t in existing_types if t}))
            parts.append(
                f"\nINTERACTION TYPES already in archive: {type_str}. "
                "Prefer types NOT on this list, or deeply novel variants."
            )
        parts.append(
            f"\nGenerate {n_candidates} candidate scenarios at varying typicality levels. "
            "Each should TRANSFER the latent social structure from the examples (type of constraint, "
            "form of shortcut, nature of power asymmetry) while VARYING the surface freely "
            "(characters, setting, stakes). Aim for at least the same difficulty — not easier — "
            "but do not force escalation; a difficulty calibration step adjusts if needed. "
            "Do not re-skin; do not jump to a completely different type of social challenge. "
            "Return JSON: {\"candidates\": [...]}"
        )
        user_prompt = "\n".join(parts)

        for attempt in range(self.max_retries):
            try:
                d = self.fm.query_json(system, user_prompt, temperature=1.0)
                candidates = d.get("candidates", [])
                if not candidates:
                    continue
                # Pick lowest-typicality candidate among those with learnability >= 0.6.
                # Fall back to best overall if none meet the learnability threshold.
                valid_candidates = []
                for c in candidates:
                    prob = float(c.get("probability", 1.0))
                    learn = float(c.get("learnability_score", 1.0))
                    scn_dict = c.get("scenario_json", {})
                    ok, _ = validate_scenario(scn_dict)
                    if ok:
                        valid_candidates.append((prob, learn, scn_dict))
                if not valid_candidates:
                    continue
                # Filter to learnable candidates first; fall back to full pool if none qualify.
                learnable = [c for c in valid_candidates if c[1] >= 0.6]
                pool = learnable if learnable else valid_candidates
                # Inverse-probability sampling (VS paper): weight = 1/p so low-typicality
                # candidates are strongly preferred but not always deterministically chosen.
                # This prevents systematic bias toward the same "unusual" direction across
                # archive iterations, improving long-run diversity.
                weights = np.array([1.0 / max(c[0], 1e-6) for c in pool])
                weights /= weights.sum()
                chosen_idx = int(np.random.choice(len(pool), p=weights))
                _, _, chosen_dict = pool[chosen_idx]
                try:
                    return dict_to_scenario(chosen_dict)
                except Exception:
                    continue
            except Exception:
                continue

        # Fallback to standard generation
        return self.generate_from_archive(
            examples, episode_failed_examples=episode_failed_examples, existing_types=existing_types
        )

    _EDIT_INTENTS = {
        "fix_coherence": (
            "The scenario has coherence issues that must be fixed. Fix ONLY the identified "
            "issues; preserve the premise, characters, structured goals, success_rubric, and "
            "interaction type except where an issue requires a change."
        ),
        "improve_interestingness": (
            "The scenario is not interestingly difficult enough. Revise the STRUCTURED GOALS so "
            "the constraint genuinely bites and the shortcut is genuinely tempting (a naive agent "
            "would take it and pay the relational cost), while preserving a zone of possible "
            "agreement. Keep the premise, characters, and interaction type; you MAY sharpen "
            "outcome/constraint/shortcut and the success_rubric questions."
        ),
        "raise_difficulty": (
            "A naive agent solved this on the first try, so it is TOO EASY. Make it HARDER along "
            "the named social knob only — increase shortcut salience, constraint bite, partner "
            "resistance, or partner stake. Do NOT add facts, parties, or numeric complexity, and "
            "do NOT make it impossible (keep a zone of possible agreement). Keep the premise, "
            "characters, interaction type, and goal_type; you MAY sharpen outcome/constraint/"
            "shortcut and the success_rubric questions."
        ),
    }

    def edit_scenario(
        self,
        scenario: SocialScenario,
        feedback,
        intent: str = "fix_coherence",
    ) -> Optional[SocialScenario]:
        """Edit an existing scenario in place of regenerating it.

        intent ∈ {fix_coherence, improve_interestingness, raise_difficulty}. Preserves lineage
        (source_scenario_id, parent_example_ids, target_agent_idx) so edits stay in the same
        lineage. Reuses the full generation schema + validation via _generate_with_retry.
        """
        instruction = self._EDIT_INTENTS.get(intent, self._EDIT_INTENTS["fix_coherence"])
        original_json = _format_scenario_for_prompt(scenario)
        fb = "\n".join(f"- {x}" for x in feedback) if isinstance(feedback, (list, tuple)) else str(feedback)
        user_prompt = (
            f"{instruction}\n\nCURRENT SCENARIO:\n{original_json}\n\n"
            f"What to address:\n{fb}\n\n"
            f"Return the revised scenario as JSON matching the required schema "
            f"(including agent_structured_goals, goal_type, success_rubric)."
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
        "interestingly difficult: the tempting shortcut wasn't tempting enough, the constraint "
        "didn't bite, or the partner gave in too easily. Identify the SINGLE slack social knob "
        "and say concretely how to tighten it — WITHOUT adding facts, parties, or numeric "
        "complexity. Respond with ONLY valid JSON."
    )

    def analyze_too_easy(self, scenario: SocialScenario, transcript: list) -> dict:
        """Diagnose which social knob is slack from a transcript where the learner solved turn 1.

        Returns {slack_knob, rationale, suggested_edit} → feeds edit_scenario(raise_difficulty).
        """
        sj = _format_scenario_for_prompt(scenario)
        tx = "\n".join(
            f"[{t.get('speaker', '?')}] {t.get('content', '')}" for t in (transcript or [])
        )[:3000]
        user = (
            f"SCENARIO:\n{sj}\n\nTRANSCRIPT (the learner solved this on the first try):\n{tx}\n\n"
            'Respond JSON: {"slack_knob": "shortcut_salience|constraint_bite|partner_resistance|partner_stake", '
            '"rationale": "one sentence", "suggested_edit": "a concrete change that raises that knob '
            'without adding facts/parties/numeric complexity"}'
        )
        try:
            d = self.fm.query_json(self._ANALYZE_TOO_EASY_SYSTEM, user, temperature=0.3)
        except Exception as e:
            return {
                "slack_knob": "partner_resistance",
                "rationale": f"analyze failed: {e}",
                "suggested_edit": "Make the partner resist the learner's first move and hold their position longer.",
            }
        return {
            "slack_knob": str(d.get("slack_knob", "partner_resistance")),
            "rationale": str(d.get("rationale", "")),
            "suggested_edit": str(d.get("suggested_edit", "")),
        }

    def generate_unconditioned(self) -> Optional[SocialScenario]:
        """Ablation: no archive conditioning."""
        user_prompt = (
            "Generate ONE realistic social scenario. Return ONLY a JSON object "
            "matching the required schema."
        )
        return self._generate_with_retry(user_prompt)

    def flesh_out_seed(self, description: str) -> Optional[SocialScenario]:
        user_prompt = (
            f"Flesh out the following short scenario description into a complete social scenario "
            f"with 2 detailed character profiles, private goals, and a clear relationship.\n\n"
            f"Description: {description}\n\n"
            f"Return ONLY a JSON object matching the required schema."
        )
        return self._generate_with_retry(user_prompt)

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
