"""ASCC MLE on synthetic history data."""
from __future__ import annotations

import random

from tmafc.ascc.mle import constrained_mle_alpha_beta

random.seed(0)


def _synth_easy(n: int = 40):
    return [(random.randint(2, 4), random.randint(2, 4)) for _ in range(n)]


def _synth_hard(n: int = 40):
    rows = []
    for _ in range(n):
        T = random.randint(8, 14)
        G = random.randint(T // 2, T - 1)
        rows.append((T, G))
    return rows


def _solve(label: str, rows) -> None:
    n_list = [t for t, _ in rows]
    k_list = [g for _, g in rows]
    a, b = constrained_mle_alpha_beta(n_list, k_list)
    print(f"{label:>5s}: |D_C|={len(rows)}  alpha0*={a:.3f}  beta0*={b:.3f}")


def main() -> None:
    _solve("Easy", _synth_easy())
    _solve("Hard", _synth_hard())


if __name__ == "__main__":
    main()
