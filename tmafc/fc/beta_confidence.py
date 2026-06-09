"""Beta stopping-criterion confidence (TMAFC Eq. 1)."""
from __future__ import annotations

import scipy.stats

from tmafc.utils.logging import get_logger

logger = get_logger(__name__)


def beta_confidence(
    c1: int,
    c2: int,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> float:
    """Probability that the majority value remains dominant.

    Models p1 ~ Beta(alpha0 + c1, beta0 + c2) and returns P(p1 >= 0.5).

    Adaptive-Consistency uses (alpha0, beta0) = (1, 1); ASCC fits them
    from history. A larger beta0 lowers confidence at fixed counts and
    triggers more sampling, which is the desired behaviour for harder
    tasks or weaker models.
    """
    if c1 <= 0:
        return 0.0
    c2 = max(0, c2)
    a = alpha0 + c1
    b = beta0 + c2
    try:
        return float(scipy.stats.beta.sf(0.5, a=a, b=b))
    except Exception as e:  # pragma: no cover
        logger.warning("BetaCDF failed (c1=%s, c2=%s): %s", c1, c2, e)
        return 0.0
