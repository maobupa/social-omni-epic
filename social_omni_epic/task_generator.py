import json
import numpy as np
from typing import Optional
from .data_models import SocialScenario
from .fm import FM
from .validation import validate_scenario, dict_to_scenario
from .embedding_utils import get_similar_scenarios


VS_SYSTEM_PROMPT = """You are a creative social scenario designer. Generate social scenarios that are INTERESTING and LEARNABLE.

INTERESTING: explores a novel social dynamic, power structure, or relational tension — not a generic archetype. Creative, specific, worth engaging with.

LEARNABLE: the learner agent's outcome must be meaningfully responsive to HOW they engage (empathy, timing, framing, strategic disclosure, trust-building). Avoid scenarios where the goal is unreachable through conversation, or where any polite response already succeeds.

Each scenario must include:
1. A vivid scenario description (the setting, context, what is happening)
2. Exactly 2 character profiles with distinct personalities, backgrounds, and motivations
3. Private goals for each character (what they want to achieve, which may conflict)
4. The pre-existing relationship between the characters
5. The type of interaction (negotiation, cooperation, conflict, persuasion, support, competition, deception, mediation, etc.)
6. Difficulty tags describing what makes this scenario challenging

Scenarios must involve realistic human social dynamics. Avoid fantasy or sci-fi settings. Stakes can range from mundane to high — what matters is that the social tension is real and human.

VERBALIZED SAMPLING: You will generate {n_candidates} distinct candidates and score each on two axes:
- "probability": typicality (0.01–0.50) — how likely would a standard AI spontaneously propose this exact social dynamic? Low = more interesting.
- "learnability_score": skill-responsiveness (0.0–1.0) — how much does social skill move the outcome? High = more learnable.

The ideal candidate has LOW probability AND learnability_score ≥ 0.6.

Each candidate must follow the schema:
{{
  "probability": <float 0.01–0.50>,
  "learnability_score": <float 0.0–1.0>,
  "scenario_json": {{
    "scenario": "string (>= 50 chars)",
    "agent_profiles": [
      {{"first_name": "...", "last_name": "...", "age": 0, "gender_identity": "...",
       "occupation": "...", "big_five": "...", "moral_values": "...",
       "schwartz_portrait_value": "...", "decision_making_style": "...",
       "secret": "...", "mbti": "...", "public_info": "2-3 sentence narrative bio"}},
      {{ ... }}
    ],
    "agent_goals": ["goal >= 20 chars", "goal >= 20 chars"],
    "relationship": "one of: stranger / acquaintance / friend / romantic / family",
    "relationship_background": "2-3 sentences of shared history consistent with the relationship label. Empty string if strangers.",
    "interaction_type": "string",
    "tag": "string",
    "difficulty_tags": ["string", ...]
  }}
}}

Return a JSON object: {{"candidates": [<candidate1>, <candidate2>, ...]}}"""


SYSTEM_PROMPT = """You are a creative social scenario designer. Generate social scenarios that are INTERESTING and LEARNABLE.

INTERESTING: explores a novel social dynamic, power structure, or relational tension — not a generic archetype. Creative, specific, worth engaging with.

LEARNABLE: the learner agent's outcome must be meaningfully responsive to HOW they engage (empathy, timing, framing, strategic disclosure, trust-building). Avoid scenarios where the goal is unreachable through conversation, or where any polite response already succeeds.

Each scenario must include:
1. A vivid scenario description (the setting, context, what is happening)
2. Exactly 2 character profiles with distinct personalities, backgrounds, and motivations
3. Private goals for each character (what they want to achieve, which may conflict)
4. The pre-existing relationship between the characters
5. The type of interaction (negotiation, cooperation, conflict, persuasion, support, competition, deception, mediation, etc.)
6. Difficulty tags describing what makes this scenario challenging

Scenarios must involve realistic human social dynamics. Avoid fantasy or sci-fi settings. Stakes can range from mundane to high — what matters is that the social tension is real and human.

Respond with valid JSON matching exactly this schema:
{
  "scenario": "string (>= 50 chars)",
  "agent_profiles": [
    {"first_name": "...", "last_name": "...", "age": 0, "gender_identity": "...",
     "occupation": "...", "big_five": "...", "moral_values": "...",
     "schwartz_portrait_value": "...", "decision_making_style": "...",
     "secret": "...", "mbti": "...", "public_info": "2-3 sentence narrative bio (name, defining traits, personal details)"},
    { ... }  // exactly 2 profiles
  ],
  "agent_goals": ["goal for agent 1 (>= 20 chars)", "goal for agent 2 (>= 20 chars)"],
  "relationship": "one of: stranger / acquaintance / friend / romantic / family",
  "relationship_background": "2-3 sentences of shared history consistent with the relationship label. Empty string if strangers.",
  "interaction_type": "string",
  "tag": "string",
  "difficulty_tags": ["string", ...]
}
"""


def _format_scenario_for_prompt(s: SocialScenario) -> str:
    d = {
        "scenario": s.scenario,
        "agent_profiles": [p.model_dump(exclude={"id"}) for p in s.agent_profiles],
        "agent_goals": s.agent_goals,
        "relationship": s.relationship,
        "relationship_background": s.relationship_background,
        "interaction_type": s.interaction_type,
        "tag": s.tag,
        "difficulty_tags": s.difficulty_tags,
    }
    return json.dumps(d, indent=2)


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
                           existing_types: Optional[list[str]] = None,
                           coherence_feedback: Optional[list[str]] = None) -> str:
        parts = []
        if examples:
            parts.append("EXAMPLE SCENARIOS FROM THE ARCHIVE (stepping stones — build on these dynamics, extending their complexity or adding new structural twists):\n")
            for i, ex in enumerate(examples):
                parts.append(f"--- Example {i+1} ---")
                parts.append(_format_scenario_for_prompt(ex))
        if failed:
            parts.append("\nSCENARIOS THAT WERE REJECTED AS UNINTERESTING (avoid these patterns):\n")
            for i, fx in enumerate(failed):
                parts.append(f"--- Rejected {i+1} ---")
                parts.append(_format_scenario_for_prompt(fx))
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
            "\nGenerate ONE NEW social scenario that extends or builds upon the social dynamics shown above — "
            "more complex stakes, a new structural twist, or a different power asymmetry layered on a familiar tension. "
            "Do not merely re-skin with different names or settings; the structural novelty must be genuine. "
            "Return ONLY a JSON object matching the required schema."
        )
        return "\n".join(parts)

    def generate_from_archive(
        self,
        examples: list[SocialScenario],
        failed_examples: Optional[list[SocialScenario]] = None,
        existing_types: Optional[list[str]] = None,
        coherence_feedback: Optional[list[str]] = None,
    ) -> Optional[SocialScenario]:
        user_prompt = self._build_user_prompt(
            examples, failed_examples or [], existing_types=existing_types,
            coherence_feedback=coherence_feedback,
        )
        return self._generate_with_retry(user_prompt)

    def generate_with_verbalized_sampling(
        self,
        examples: list[SocialScenario],
        existing_types: Optional[list[str]] = None,
        n_candidates: int = 5,
    ) -> Optional[SocialScenario]:
        """Verbalized Sampling (§4.2): generate N candidates with typicality probabilities.

        Picks the candidate with the LOWEST probability — the most frontier/surprising
        scenario that the model would not spontaneously generate.

        Note: recently-rejected scenarios are NOT passed (design decision).
        Falls back to generate_from_archive if VS fails.
        """
        system = VS_SYSTEM_PROMPT.replace("{n_candidates}", str(n_candidates))

        parts: list[str] = []
        if examples:
            parts.append("EXAMPLE SCENARIOS FROM THE ARCHIVE (build BEYOND these):\n")
            for i, ex in enumerate(examples):
                parts.append(f"--- Example {i + 1} ---")
                parts.append(_format_scenario_for_prompt(ex))
        if existing_types:
            type_str = ", ".join(sorted({t for t in existing_types if t}))
            parts.append(
                f"\nINTERACTION TYPES already in archive: {type_str}. "
                "Prefer types NOT on this list, or deeply novel variants."
            )
        parts.append(
            f"\nGenerate {n_candidates} candidate scenarios at varying typicality levels. "
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
        return self.generate_from_archive(examples, existing_types=existing_types)

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
