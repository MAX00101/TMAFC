"""K-quantile reclustering of history records by required sample count T."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

DEFAULT_LABELS_3 = ["Easy", "Middle", "Hard"]


def _default_labels(K: int) -> List[str]:
    if K == 3:
        return list(DEFAULT_LABELS_3)
    return [f"Level{i+1}" for i in range(K)]


def recluster_by_quantile(
    T_values: Sequence[float],
    K: int = 3,
    labels: List[str] | None = None,
) -> List[str]:
    n = len(T_values)
    if n == 0:
        return []
    labels = labels or _default_labels(K)
    if len(labels) != K:
        raise ValueError(f"labels has {len(labels)} entries but K={K}")

    arr = np.asarray(T_values, dtype=float)
    order = np.argsort(arr, kind="stable")
    out = [labels[0]] * n
    chunk = max(1, n // K)
    for i, idx in enumerate(order):
        bucket = min(K - 1, i // chunk)
        out[idx] = labels[bucket]
    return out
