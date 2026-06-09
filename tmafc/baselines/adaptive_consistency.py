"""Adaptive-Consistency: monolithic Beta stopping with fixed prior (1, 1)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

from tmafc.core.voting import normalize_value
from tmafc.fc.beta_confidence import beta_confidence
from tmafc.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ACResult:
    answer: Any
    samples: List[str]
    samples_consumed: int
    top1: int
    top2: int
    confidence: float

    @property
    def T(self) -> int:
        return self.top1 + self.top2

    @property
    def G(self) -> int:
        return self.top1


class AdaptiveConsistency:
    def __init__(
        self,
        tau: float = 0.95,
        n_max: int = 50,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        min_samples_for_conf: int = 2,
    ) -> None:
        self.tau = tau
        self.n_max = n_max
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.min_samples_for_conf = min_samples_for_conf

    def run(self, sample_fn: Callable[[], str]) -> ACResult:
        samples: List[str] = []
        for _ in range(self.n_max):
            samples.append(normalize_value(sample_fn()))
            if len(samples) < self.min_samples_for_conf:
                continue
            _, _, conf = self._top_two_with_conf(samples)
            if conf > self.tau:
                logger.debug("AC stopped at n=%d, conf=%.3f", len(samples), conf)
                break
        top1, top2, conf = self._top_two_with_conf(samples)
        most_common = Counter(samples).most_common(1)
        answer = most_common[0][0] if most_common else None
        return ACResult(
            answer=answer,
            samples=samples,
            samples_consumed=len(samples),
            top1=top1,
            top2=top2,
            confidence=conf,
        )

    def _top_two_with_conf(self, samples: List[str]) -> Tuple[int, int, float]:
        if not samples:
            return 0, 0, 0.0
        c = Counter(samples).most_common(2)
        c1 = c[0][1]
        c2 = c[1][1] if len(c) > 1 else 0
        return c1, c2, beta_confidence(c1, c2, self.alpha0, self.beta0)
