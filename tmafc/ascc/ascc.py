"""Adaptive Stopping Criteria Calibration (TMAFC Algorithm 2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tmafc.ascc.history_store import HistoryStore
from tmafc.ascc.mle import constrained_mle_alpha_beta
from tmafc.utils.logging import get_logger

logger = get_logger(__name__)


class _Classifier(Protocol):
    def classify(self, question: str) -> str: ...


@dataclass
class CalibrationResult:
    category: str
    alpha0: float
    beta0: float
    n_examples: int


class ASCC:
    def __init__(
        self,
        classifier: _Classifier,
        history: HistoryStore,
        eps_min: float = 0.01,
        eps_max: float = 1.0,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        min_history_for_mle: int = 3,
    ) -> None:
        self.classifier = classifier
        self.history = history
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.min_history_for_mle = min_history_for_mle

    def calibrate(self, question: str) -> CalibrationResult:
        category = self.classifier.classify(question)
        bucket = self.history.filter_by_category(category)

        if len(bucket) < self.min_history_for_mle:
            logger.info(
                "ASCC: only %d records in category=%s (< %d); using init prior (%.2f, %.2f)",
                len(bucket), category, self.min_history_for_mle,
                self.alpha_init, self.beta_init,
            )
            return CalibrationResult(
                category=category,
                alpha0=self.alpha_init,
                beta0=self.beta_init,
                n_examples=len(bucket),
            )

        n_list = [r.T for r in bucket]
        k_list = [r.G for r in bucket]
        a_star, b_star = constrained_mle_alpha_beta(
            n_list=n_list,
            k_list=k_list,
            alpha_init=self.alpha_init,
            beta_init=self.beta_init,
            eps_min=self.eps_min,
            eps_max=self.eps_max,
        )
        logger.info(
            "ASCC: category=%s |D_C|=%d alpha0=%.3f beta0=%.3f",
            category, len(bucket), a_star, b_star,
        )
        return CalibrationResult(
            category=category,
            alpha0=a_star,
            beta0=b_star,
            n_examples=len(bucket),
        )
