import json
import numpy as np
from typing import Optional
from .data_models import SocialScenario
from .fm import FM
from .validation import validate_scenario, dict_to_scenario
from .embedding_utils import get_similar_scenarios


SYSTEM_PROMPT = """You are a creative social scenario designer. You generate diverse, realistic social interaction scenarios for evaluating AI social intelligence. Each scenario must include:
1. A vivid scenario description (the setting, context, what is happening)
2. Exactly 2 character profiles with distinct personalities, backgrounds, and motivations
3. Private goals for each character (what they want to achieve, which may conflict)
4. The pre-existing relationship between the characters
5. The type of interaction (negotiation, cooperation, conflict, persuasion, support, competition, deception, mediation, etc.)
6. Difficulty tags describing what makes this scenario challenging

Your scenarios should be grounded in everyday life: workplace, family, neighborhood, school, healthcare, commerce, etc. Avoid fantasy or sci-fi settings. Focus on realistic social dynamics with genuine tension and complexity.

Respond with valid JSON matching exactly this schema:
{
  "scenario": "string (>= 50 chars)",
  "agent_profiles": [
    {"first_name": "...", "last_name": "...", "age": 0, "gender_identity": "...",
     "occupation": "...", "big_five": "...", "moral_values": "...",
     "schwartz_portrait_value": "...", "decision_making_style": "...",
     "secret": "...", "mbti": "...", "public_info": "..."},
    { ... }  // exactly 2 profiles
  ],
  "agent_goals": ["goal for agent 1 (>= 20 chars)", "goal for agent 2 (>= 20 chars)"],
  "relationship": "string",
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
            seed_emb = archive.state.successful[seed_idx].embedding
            all_embs = archive.get_successful_embeddings()
            if seed_emb is None or not all_embs:
                indices = [seed_idx]
            else:
                indices = get_similar_scenarios(seed_emb, all_embs, num_returns=k)
                if seed_idx not in indices:
                    indices = [seed_idx] + indices[:k - 1]

        examples = [archive.state.successful[i] for i in indices]
        return examples, indices

    def _build_user_prompt(self, examples: list[SocialScenario],
                           failed: list[SocialScenario],
                           existing_types: Optional[list[str]] = None) -> str:
        parts = []
        if examples:
            parts.append("EXAMPLE SCENARIOS FROM THE ARCHIVE (build BEYOND these):\n")
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
        parts.append(
            "\nGenerate ONE NEW social scenario that is MEANINGFULLY DIFFERENT from the examples above. "
            "Do not merely change surface details (names, locations). Change the underlying social dynamics, "
            "power structures, information asymmetries, or moral dimensions. "
            "Return ONLY a JSON object matching the required schema."
        )
        return "\n".join(parts)

    def generate_from_archive(
        self,
        examples: list[SocialScenario],
        failed_examples: Optional[list[SocialScenario]] = None,
        existing_types: Optional[list[str]] = None,
    ) -> Optional[SocialScenario]:
        user_prompt = self._build_user_prompt(
            examples, failed_examples or [], existing_types=existing_types
        )
        return self._generate_with_retry(user_prompt)

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
