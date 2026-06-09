"""Backend-agnostic LLM interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Union


@dataclass
class ChatMessage:
    role: str  # system | user | assistant
    content: str

    def to_openai_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


PromptLike = Union[str, Sequence[ChatMessage]]


def normalize_prompt(prompt: PromptLike) -> List[ChatMessage]:
    if isinstance(prompt, str):
        return [ChatMessage(role="user", content=prompt)]
    return list(prompt)


class BaseLLM(ABC):
    """Backend interface used by the rest of the framework.

    Implementations should not raise on transient API errors but log and
    return an empty list. Callers are expected to handle empty responses.
    """

    name: str = "base"

    @abstractmethod
    def chat(
        self,
        prompt: PromptLike,
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> List[str]:
        ...

    def chat_batch(
        self,
        prompts: Sequence[PromptLike],
        n: int = 1,
        **kwargs,
    ) -> List[List[str]]:
        return [self.chat(p, n=n, **kwargs) for p in prompts]
