"""Fractal-Consistency main loop (TMAFC Algorithm 1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from tmafc.fc.beta_confidence import beta_confidence
from tmafc.fc.component_state import ComponentState, init_component_states
from tmafc.utils.logging import get_logger

logger = get_logger(__name__)

ResamplingHook = Callable[[Dict[str, ComponentState]], Dict[str, str]]

STOP_ALL = "ALL_CONVERGED"
STOP_GEOMETRIC = "GEOMETRIC_MEAN"


@dataclass
class FCResult:
    converged_values: Dict[str, str]
    states: Dict[str, ComponentState]
    rounds_used: int
    samples_consumed: int

    @property
    def avg_top1(self) -> float:
        if not self.states:
            return 0.0
        return float(np.mean([s.c1 for s in self.states.values()]))

    @property
    def avg_top2(self) -> float:
        if not self.states:
            return 0.0
        return float(np.mean([s.c2 for s in self.states.values()]))


class FCRunner:
    """Atomic-wise convergence with a per-component Beta stopping criterion.

    The runner is transport-agnostic: it does not know about LLMs or
    prompts. Callers pass a ``sample_fn`` that, given the still-pending
    components, returns one round of ``{component_id: value}``.
    """

    def __init__(
        self,
        tau: float = 0.95,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        n_max: int = 50,
        min_samples_for_conf: int = 2,
        stopping_logic: str = STOP_ALL,
        geo_mean_threshold: float = 0.9,
    ) -> None:
        self.tau = tau
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.n_max = n_max
        self.min_samples_for_conf = min_samples_for_conf
        self.stopping_logic = stopping_logic
        self.geo_mean_threshold = geo_mean_threshold

    def run(
        self,
        component_ids: List[str],
        sample_fn: ResamplingHook,
        prior: Optional[tuple] = None,
    ) -> FCResult:
        a0, b0 = prior if prior is not None else (self.alpha0, self.beta0)
        states = init_component_states(component_ids)
        round_idx = 0
        samples_consumed = 0

        while round_idx < self.n_max:
            round_idx += 1
            pending = [cid for cid, st in states.items() if not st.converged]
            if not pending:
                break

            sub_states = {cid: states[cid] for cid in pending}
            try:
                round_samples = sample_fn(sub_states) or {}
            except Exception as e:  # pragma: no cover
                logger.warning("sample_fn raised %s; treating as empty round", e)
                round_samples = {}
            samples_consumed += 1

            geo_terms: List[float] = []
            all_confident = True
            for cid in component_ids:
                st = states[cid]
                if st.converged:
                    geo_terms.append(st.confidence)
                    continue

                raw = round_samples.get(cid)
                value = (
                    str(raw).strip().replace("\n", " ")
                    if raw is not None and str(raw).strip()
                    else None
                )
                st.samples.append(value)
                st.update_top_two()

                if st.has_min_samples(self.min_samples_for_conf):
                    st.confidence = beta_confidence(st.c1, st.c2, alpha0=a0, beta0=b0)
                else:
                    st.confidence = 0.0

                if st.confidence > self.tau:
                    st.converged = True
                    logger.debug(
                        "round=%d cid=%s converged to %s (conf=%.3f)",
                        round_idx, cid, st.v1, st.confidence,
                    )
                else:
                    all_confident = False
                geo_terms.append(st.confidence)

            if self.stopping_logic == STOP_ALL:
                if all_confident:
                    break
            elif self.stopping_logic == STOP_GEOMETRIC:
                if geo_terms:
                    adj = [max(1e-9, x) for x in geo_terms]
                    gmean = float(np.exp(np.mean(np.log(adj))))
                    if gmean >= self.geo_mean_threshold:
                        break
            else:
                raise ValueError(f"Unknown stopping_logic: {self.stopping_logic}")

        converged_values = {cid: states[cid].v1 for cid in component_ids}
        return FCResult(
            converged_values=converged_values,
            states=states,
            rounds_used=round_idx,
            samples_consumed=samples_consumed,
        )
