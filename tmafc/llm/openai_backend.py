"""OpenAI Chat Completions backend."""
from __future__ import annotations

import os
from typing import List

from tmafc.llm.base import BaseLLM, PromptLike, normalize_prompt
from tmafc.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIChatLLM(BaseLLM):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "openai backend requires the `openai` package. "
                "Run `pip install -e .[openai]`."
            ) from e

        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            timeout=timeout,
        )

    def chat(
        self,
        prompt: PromptLike,
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> List[str]:
        messages = [m.to_openai_dict() for m in normalize_prompt(prompt)]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
        except Exception as e:
            logger.error("OpenAI call failed: %s", e)
            return []
        return [c.message.content or "" for c in resp.choices]
