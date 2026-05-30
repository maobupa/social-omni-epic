import json
import os
import time
from typing import Optional
import openai


class FM:
    def __init__(self, model: str = "gpt-4o", temperature: float = 1.0,
                 embedding_model: str = "text-embedding-3-small"):
        # Prefer LIGHTNING_AI_* env vars; fall back to OPENAI_* for backwards compat.
        api_key = os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LIGHTNING_AI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,  # None → uses OpenAI default
        )
        self.model = model
        self.temperature = temperature
        self.embedding_model = embedding_model

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

    def query(self, system_prompt: str, user_prompt: str,
              temperature: Optional[float] = None) -> str:
        def _call():
            r = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature if temperature is not None else self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return r.choices[0].message.content
        return self._retry(_call)

    def query_json(self, system_prompt: str, user_prompt: str,
                   temperature: Optional[float] = None) -> dict:
        def _call():
            r = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return json.loads(r.choices[0].message.content)
        return self._retry(_call)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        def _call():
            r = self.client.embeddings.create(
                input=texts, model=self.embedding_model
            )
            return [d.embedding for d in r.data]
        return self._retry(_call)
