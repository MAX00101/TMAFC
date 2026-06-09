"""Named entity recognition task adapter (IEPile)."""
from __future__ import annotations

from dataclasses import dataclass

from tmafc.core.decomposer import StructuredSchema
from tmafc.tasks.base import Task

NER_SCHEMA = StructuredSchema(
    keys=["brand of vehicle", "orientation of vehicle", "truck", "vehicle model"],
    description=(
        "Extract entities matching the schema. Return an empty list if a type "
        "is not present. Reply with a single JSON object."
    ),
)


@dataclass
class NERTask(Task):
    name: str = "ner"
    is_structured: bool = True

    def schema(self) -> StructuredSchema:
        return NER_SCHEMA
