"""FC-only example: a hand-rolled sample function feeds FCRunner directly."""
from __future__ import annotations

import random

from tmafc.fc.fc_runner import FCRunner

random.seed(0)


def _sample_fn(states):
    out = {}
    for cid in states.keys():
        truth = {"brand": "Genesis", "model": "GV80"}.get(cid, "x")
        out[cid] = truth if random.random() < 0.85 else f"noise_{random.randint(0, 9)}"
    return out


def main() -> None:
    fc = FCRunner(tau=0.9, n_max=20, alpha0=1.0, beta0=0.3)
    result = fc.run(component_ids=["brand", "model"], sample_fn=_sample_fn)
    print("converged:", result.converged_values)
    print("rounds   :", result.rounds_used)
    print("samples  :", result.samples_consumed)


if __name__ == "__main__":
    main()
