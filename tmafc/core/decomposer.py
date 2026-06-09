"""Atomic-component decomposition.

Structured tasks come with a schema; unstructured tasks are decomposed via
the LLM-based anchor generator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from tmafc.fc.anchor_generator import AnchorGenerator
from tmafc.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StructuredSchema:
    keys: List[str]
    description: str = ""
    examples: List[dict] = field(default_factory=list)


@dataclass
class DecomposedTask:
    is_structured: bool
    component_ids: List[str]
    schema: Optional[StructuredSchema] = None
    anchors: Optional[List[str]] = None


class Decomposer:
    def __init__(self, anchor_generator: Optional[AnchorGenerator] = None) -> None:
        self.anchor_generator = anchor_generator

    def decompose(
        self,
        question: str,
        schema: Optional[StructuredSchema] = None,
    ) -> DecomposedTask:
        if schema is not None and schema.keys:
            return DecomposedTask(
                is_structured=True,
                component_ids=list(schema.keys),
                schema=schema,
            )
        if self.anchor_generator is None:
            return DecomposedTask(
                is_structured=False,
                component_ids=["final_answer"],
                anchors=[],
            )
        anchors = self.anchor_generator.generate(question)
        if not anchors:
            return DecomposedTask(
                is_structured=False,
                component_ids=["final_answer"],
                anchors=[],
            )
        component_ids = [f"anchor_{i}" for i in range(len(anchors))]
        return DecomposedTask(
            is_structured=False,
            component_ids=component_ids,
            anchors=anchors,
        )
