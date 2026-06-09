"""Anchor generator for unstructured tasks."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from tmafc.llm.base import BaseLLM
from tmafc.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "anchor_generator.txt"

_FALLBACK_TEMPLATE = (
    "You are an expert reasoner. Given the QUESTION below, produce a small set of "
    "independent, logically-checkable SUB-QUESTIONS (called anchors) that any correct "
    "reasoning trajectory must answer consistently.\n\n"
    "Rules:\n"
    "- Output a JSON list of strings, each a single concrete sub-question.\n"
    "- 2 to 5 anchors. No explanations. No final answer.\n"
    "- Each anchor must be answerable independently of the others.\n\n"
    "QUESTION:\n{question}\n\nAnchors (JSON list only):"
)


def _load_prompt() -> str:
    if DEFAULT_PROMPT_PATH.exists():
        return DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
    return _FALLBACK_TEMPLATE


def _extract_json_list(text: str) -> Optional[List[str]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
    return None


class AnchorGenerator:
    def __init__(
        self,
        llm: BaseLLM,
        prompt_template: Optional[str] = None,
        max_anchors: int = 5,
        min_anchors: int = 1,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template or _load_prompt()
        self.max_anchors = max_anchors
        self.min_anchors = min_anchors

    def generate(self, question: str) -> List[str]:
        prompt = self.prompt_template.format(question=question)
        try:
            outputs = self.llm.chat(prompt, n=1)
        except Exception as e:
            logger.warning("anchor LLM call failed: %s; returning []", e)
            return []
        if not outputs:
            return []
        anchors = _extract_json_list(outputs[0]) or []
        anchors = anchors[: self.max_anchors]
        if len(anchors) < self.min_anchors:
            logger.info("anchor generator returned %d anchor(s)", len(anchors))
        return anchors
