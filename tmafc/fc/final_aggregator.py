"""Final-answer aggregation for unstructured tasks.

After anchor convergence, drop samples that disagree with the converged
anchor consensus and majority-vote on the remaining final answers. Fall
back to plain majority voting if no sample fully matches.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

from tmafc.utils.logging import get_logger

logger = get_logger(__name__)


def aggregate_unstructured_answer(
    samples: List[Dict[str, str]],
    converged_anchors: Dict[str, str],
    final_answer_key: str = "final_answer",
) -> Tuple[Optional[str], int]:
    if not samples:
        return None, 0
    samples = [s for s in samples if isinstance(s, dict)]
    if not samples:
        return None, 0

    def _matches_consensus(s: Dict[str, str]) -> bool:
        for aid, gold in converged_anchors.items():
            if aid == final_answer_key:
                continue
            v = s.get(aid)
            if v is None or str(v).strip() != str(gold).strip():
                return False
        return True

    coherent = [s for s in samples if _matches_consensus(s)]
    pool = coherent if coherent else samples
    if not coherent:
        logger.info("no sample matches converged anchors; falling back to plain majority vote")

    finals = [s.get(final_answer_key) for s in pool if s.get(final_answer_key)]
    if not finals:
        return None, 0

    top, count = Counter(map(str, finals)).most_common(1)[0]
    return top, count
