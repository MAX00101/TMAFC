"""JSON parsing and majority-vote helpers."""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Tuple


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object out of an LLM response.

    Tolerates surrounding markdown/prose by falling back to the first
    ``{...}`` block.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def majority_vote(values: Iterable[Any]) -> Tuple[Optional[str], int]:
    norm = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not norm:
        return None, 0
    top, c = Counter(norm).most_common(1)[0]
    return top, c


def normalize_value(v: Any) -> str:
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return str(v).strip()
