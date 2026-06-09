"""Deterministic mock backend used by the offline demo and unit tests.

Three call sites are recognised by inspecting the prompt text:

* anchor generator - prompt ends with ``Anchors (JSON list only):``
* difficulty classifier - prompt ends with ``Difficulty:``
* reasoning step - falls back to filling JSON keys declared via
  ``<<KEYS:k1,k2,...>>``; ground-truth values can be pinned via
  ``<<GT:k1=v1;k2=v2>>``.
"""
from __future__ import annotations

import json
import random
import re
from typing import List

from tmafc.llm.base import BaseLLM, PromptLike, normalize_prompt
from tmafc.utils.logging import get_logger

logger = get_logger(__name__)

_KEYS_RE = re.compile(r"<<KEYS:([^>]*)>>")
_GROUND_TRUTH_RE = re.compile(r"<<GT:([^>]*)>>")


class MockLLM(BaseLLM):
    name = "mock"

    def __init__(self, agreement: float = 0.85, seed: int = 42) -> None:
        self.agreement = agreement
        self._rng = random.Random(seed)

    def chat(
        self,
        prompt: PromptLike,
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> List[str]:
        text = "\n".join(m.content for m in normalize_prompt(prompt))
        return [self._respond(text) for _ in range(n)]

    def _respond(self, prompt_text: str) -> str:
        stripped = prompt_text.rstrip()
        if stripped.endswith("Anchors (JSON list only):"):
            return self._mock_anchors(prompt_text)
        if stripped.endswith("Difficulty:"):
            return self._mock_difficulty(prompt_text)
        return self._mock_reasoning(prompt_text)

    def _mock_anchors(self, _: str) -> str:
        return json.dumps(
            [
                "What is the total time elapsed before the relevant event?",
                "What is the velocity at the key checkpoint?",
                "What is the average velocity in the requested window?",
            ]
        )

    def _mock_difficulty(self, prompt_text: str) -> str:
        last_q = prompt_text.rsplit("Q:", 1)[-1]
        L = len(last_q)
        roll = self._rng.random()
        if L < 80:
            return "Easy" if roll < 0.95 else "Middle"
        if L < 200:
            return "Middle" if roll < 0.9 else "Hard"
        return "Hard" if roll < 0.85 else "Middle"

    def _mock_reasoning(self, prompt_text: str) -> str:
        keys = self._extract_keys(prompt_text)
        gt = self._extract_ground_truth(prompt_text)
        out = {}
        for k in keys:
            truth = gt.get(k, f"value_{k}")
            if self._rng.random() < self.agreement:
                out[k] = truth
            else:
                out[k] = f"noise_{self._rng.randint(0, 99)}"
        return json.dumps(out, ensure_ascii=False)

    @staticmethod
    def _extract_keys(prompt_text: str) -> List[str]:
        m = _KEYS_RE.search(prompt_text)
        if not m:
            return ["answer"]
        return [k.strip() for k in m.group(1).split(",") if k.strip()]

    @staticmethod
    def _extract_ground_truth(prompt_text: str) -> dict:
        m = _GROUND_TRUTH_RE.search(prompt_text)
        if not m:
            return {}
        out = {}
        for kv in m.group(1).split(";"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                out[k.strip()] = v.strip()
        return out
