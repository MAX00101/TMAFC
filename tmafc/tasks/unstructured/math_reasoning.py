"""Mathematical reasoning task adapter."""
from __future__ import annotations

from dataclasses import dataclass

from tmafc.tasks.base import Task


@dataclass
class MathReasoningTask(Task):
    name: str = "math_reasoning"
    is_structured: bool = False
