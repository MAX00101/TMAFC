"""Beta-Binomial MLE for ASCC.

Solves

    max_{a, b in [eps_min, eps_max]}
        sum_j  log B(a + G_j, b + T_j - G_j) - log B(a, b)

with L-BFGS-B (Byrd et al., 1995).
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln

from tmafc.utils.logging import get_logger

logger = get_logger(__name__)


def _negative_log_likelihood(
    params: Sequence[float], n_list, k_list, eps_min, eps_max
) -> float:
    a, b = params
    if a < eps_min or a > eps_max or b < eps_min or b > eps_max:
        return float("inf")
    nll = -np.sum(
        [betaln(a + k, b + n - k) - betaln(a, b) for n, k in zip(n_list, k_list)]
    )
    return float(nll)


def constrained_mle_alpha_beta(
    n_list: Iterable[int],
    k_list: Iterable[int],
    alpha_init: float = 1.0,
    beta_init: float = 1.0,
    eps_min: float = 0.01,
    eps_max: float = 1.0,
) -> Tuple[float, float]:
    n_arr = list(n_list)
    k_arr = list(k_list)
    if not n_arr:
        logger.warning("constrained_mle_alpha_beta: empty data; returning init")
        return float(alpha_init), float(beta_init)

    result = minimize(
        _negative_log_likelihood,
        x0=[alpha_init, beta_init],
        args=(n_arr, k_arr, eps_min, eps_max),
        bounds=[(eps_min, eps_max), (eps_min, eps_max)],
        method="L-BFGS-B",
    )
    if not result.success:
        logger.info("L-BFGS-B did not fully converge: %s", result.message)
    return float(result.x[0]), float(result.x[1])
