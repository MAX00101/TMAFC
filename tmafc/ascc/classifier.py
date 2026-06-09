"""Difficulty classifiers for ASCC."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import List, Optional

from tmafc.ascc.history_store import HistoryRecord, HistoryStore
from tmafc.llm.base import BaseLLM
from tmafc.utils.logging import get_logger

logger = get_logger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
SYSTEM_PROMPT_PATH = _PROMPT_DIR / "classifier_system.txt"
FEWSHOT_PATH = _PROMPT_DIR / "classifier_fewshot.json"


class LLMDifficultyClassifier:
    """Online classifier prompted with K * l historical examples."""

    def __init__(
        self,
        llm: BaseLLM,
        history: Optional[HistoryStore] = None,
        K: int = 3,
        examples_per_class: int = 2,
        valid_labels: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        fewshot_pool: Optional[dict] = None,
        seed: int = 0,
    ) -> None:
        self.llm = llm
        self.history = history
        self.K = K
        self.examples_per_class = examples_per_class
        self.valid_labels = valid_labels or ["Easy", "Middle", "Hard"]
        self.system_prompt = system_prompt or self._load_system_prompt()
        self.fewshot_pool = fewshot_pool or self._load_fewshot_pool()
        self._rng = random.Random(seed)

    @staticmethod
    def _load_system_prompt() -> str:
        if SYSTEM_PROMPT_PATH.exists():
            return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        return (
            "You are a difficulty classifier. Read the QUESTION and assign a "
            "difficulty label from {Easy, Middle, Hard}. Reply with the label only."
        )

    @staticmethod
    def _load_fewshot_pool() -> dict:
        if FEWSHOT_PATH.exists():
            try:
                return json.loads(FEWSHOT_PATH.read_text(encoding="utf-8"))
            except Exception as e:  # pragma: no cover
                logger.warning("failed to parse fewshot file: %s", e)
        return {}

    def _build_examples(self) -> List[dict]:
        examples: List[dict] = []
        if self.history is not None and len(self.history) > 0:
            for label in self.valid_labels:
                bucket: List[HistoryRecord] = self.history.filter_by_category(label)
                if not bucket:
                    continue
                k = min(self.examples_per_class, len(bucket))
                for r in self._rng.sample(bucket, k=k):
                    examples.append({"question": r.question, "label": label})
        if not examples and self.fewshot_pool:
            for label in self.valid_labels:
                pool = self.fewshot_pool.get(label, [])
                for q in pool[: self.examples_per_class]:
                    examples.append({"question": q, "label": label})
        return examples

    def _build_prompt(self, question: str) -> str:
        ex_block = "\n\n".join(
            f"Q: {ex['question']}\nDifficulty: {ex['label']}"
            for ex in self._build_examples()
        )
        return (
            f"{self.system_prompt}\n\n"
            f"Examples:\n{ex_block}\n\n"
            f"Now classify:\nQ: {question}\nDifficulty:"
        )

    def classify(self, question: str) -> str:
        prompt = self._build_prompt(question)
        try:
            outputs = self.llm.chat(prompt, n=1)
        except Exception as e:
            logger.warning("classifier LLM call failed: %s; defaulting to 'Middle'", e)
            return "Middle"
        if not outputs:
            return "Middle"
        text = outputs[0].strip().splitlines()[0].strip()
        for label in self.valid_labels:
            if label.lower() in text.lower():
                return label
        return "Middle"


_NUM_RE = re.compile(r"\d+")
_COND_RE = re.compile(r"\b(if|when|provided|unless|whether)\b", re.IGNORECASE)


class RuleBasedClassifier:
    """Heuristic classifier used by the offline demo and unit tests."""

    def __init__(self, valid_labels: Optional[List[str]] = None) -> None:
        self.valid_labels = valid_labels or ["Easy", "Middle", "Hard"]

    def classify(self, question: str) -> str:
        score = 0
        score += len(question) // 80
        score += min(3, len(_NUM_RE.findall(question)) // 2)
        score += min(3, len(_COND_RE.findall(question)))
        if score <= 1:
            return self.valid_labels[0]
        if score <= 3:
            return self.valid_labels[1]
        return self.valid_labels[-1]
