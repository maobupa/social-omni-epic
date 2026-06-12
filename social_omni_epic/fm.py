import json
import os
import time
from typing import Optional
import openai


class FM:
    def __init__(self, model: str = "gpt-4o", temperature: float = 1.0,
                 embedding_model: str = "text-embedding-3-small"):
        # Per-request timeout. The OpenAI SDK default is 600s, so a slow/hung call blocks
        # for 10 min before raising — and our _retry then tries again, compounding it. Bound
        # it so a stuck socket fails fast and _retry gets a fresh connection. Generous enough
        # for a slow reasoning-model batch generation; override via FM_TIMEOUT (seconds).
        self._timeout = float(os.getenv("FM_TIMEOUT", "240"))
        # Chat client: prefer Lightning.ai, fall back to OpenAI.
        chat_api_key = os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
        chat_base_url = os.getenv("LIGHTNING_AI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.client = openai.OpenAI(
            api_key=chat_api_key,
            base_url=chat_base_url,
            timeout=self._timeout,
            max_retries=2,
        )
        # Embedding client: Lightning.ai doesn't support /v1/embeddings.
        # Use a dedicated OPENAI_API_KEY if set; otherwise reuse the chat client
        # (works when chat is already on OpenAI with no custom base_url).
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self._embed_client = openai.OpenAI(api_key=openai_key, timeout=self._timeout)
        else:
            self._embed_client = self.client
        self.model = model
        self.temperature = temperature
        self.embedding_model = embedding_model
        # Set False on first InternalServerError about temperature restrictions.
        self._temperature_supported = True
        # Compute meter (per-instance). Note: in the curriculum runner one FM instance
        # serves learner episodes AND the generator/coherence/MOI; only the judge FM is
        # separate — so these totals separate judge vs everything-else, not learner vs gen.
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def usage_report(self) -> dict:
        """Snapshot of this instance's cumulative FM usage (for compute_report.json)."""
        return {
            "model": self.model,
            "n_calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    def _retry(self, fn, max_retries: int = 5):
        delay = 2.0
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"  [FM retry {attempt+1}/{max_retries}] {type(e).__name__}: {e}")
                time.sleep(delay)
                delay *= 2

    def _temp(self, temperature: Optional[float]) -> Optional[float]:
        """Return temperature to use; None means omit the param entirely."""
        if not self._temperature_supported:
            return None
        return temperature if temperature is not None else self.temperature

    def _chat(self, messages: list, temperature: Optional[float],
              response_format: Optional[dict] = None) -> str:
        """Single chat completion with auto-fallback on temperature restriction."""
        kwargs: dict = dict(model=self.model, messages=messages)
        t = self._temp(temperature)
        if t is not None:
            kwargs["temperature"] = t
        if response_format:
            kwargs["response_format"] = response_format
        try:
            r = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # Some models (e.g. gpt-5-mini) only accept the default temperature.
            # The error wording varies ("fixed", "Only the default (1) value is
            # supported", "does not support"), so match any temperature-related
            # rejection and retry once without the param.
            msg = str(e).lower()
            if "temperature" in msg and (
                "fixed" in msg or "does not support" in msg
                or "only the default" in msg or "unsupported" in msg
            ):
                self._temperature_supported = False
                kwargs.pop("temperature", None)
                r = self.client.chat.completions.create(**kwargs)
            else:
                raise
        self.n_calls += 1
        usage = getattr(r, "usage", None)
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        return r.choices[0].message.content

    def query(self, system_prompt: str, user_prompt: str,
              temperature: Optional[float] = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        def _call():
            return self._chat(messages, temperature)
        return self._retry(_call)

    def query_json(self, system_prompt: str, user_prompt: str,
                   temperature: Optional[float] = None) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        def _call():
            content = self._chat(messages, temperature,
                                 response_format={"type": "json_object"})
            return json.loads(content)
        return self._retry(_call)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        def _call():
            r = self._embed_client.embeddings.create(
                input=texts, model=self.embedding_model
            )
            return [d.embedding for d in r.data]
        return self._retry(_call)
