"""Per-component sample bookkeeping for the FC main loop."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

UNDETERMINED = "<UNDETERMINED>"


@dataclass
class ComponentState:
    component_id: str
    samples: List[Optional[str]] = field(default_factory=list)
    v1: Optional[str] = UNDETERMINED
    c1: int = 0
    v2: Optional[str] = None
    c2: int = 0
    confidence: float = 0.0
    converged: bool = False

    def update_top_two(self) -> None:
        valid = [s for s in self.samples if s and str(s).strip()]
        if not valid:
            self.v1, self.c1, self.v2, self.c2 = UNDETERMINED, 0, None, 0
            return
        most = Counter(valid).most_common(2)
        self.v1, self.c1 = most[0]
        if len(most) >= 2:
            self.v2, self.c2 = most[1]
        else:
            self.v2, self.c2 = None, 0

    def has_min_samples(self, minimum: int = 1) -> bool:
        return len(self.samples) >= minimum and self.c1 > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "samples": list(self.samples),
            "v1": self.v1,
            "c1": self.c1,
            "v2": self.v2,
            "c2": self.c2,
            "confidence": self.confidence,
            "converged": self.converged,
        }


def init_component_states(component_ids: List[str]) -> Dict[str, ComponentState]:
    return {cid: ComponentState(component_id=cid) for cid in component_ids}
