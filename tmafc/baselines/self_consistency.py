"""Self-Consistency: fixed-N majority vote."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

from tmafc.core.voting import majority_vote, normalize_value


@dataclass
class SCResult:
    answer: Any
    samples: List[str]
    samples_consumed: int


class SelfConsistency:
    def __init__(self, n_samples: int = 50) -> None:
        self.n_samples = n_samples

    def run(self, sample_fn: Callable[[], str]) -> SCResult:
        samples = [sample_fn() for _ in range(self.n_samples)]
        norm = [normalize_value(s) for s in samples]
        top, _ = majority_vote(norm)
        return SCResult(answer=top, samples=samples, samples_consumed=self.n_samples)
