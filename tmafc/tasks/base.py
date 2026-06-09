from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Optional

from tmafc.core.decomposer import StructuredSchema


@dataclass
class Task(ABC):
    name: str
    is_structured: bool

    def schema(self) -> Optional[StructuredSchema]:
        return None
