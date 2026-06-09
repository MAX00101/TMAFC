"""FC prompt builders.

Templates live under /prompts and are loaded at call time so users can
edit them without touching code. Each template embeds a ``<<KEYS:...>>``
marker that MockLLM uses to know which JSON keys to fill in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def _try_read(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


_FC_STRUCTURED_FALLBACK = (
    "You are an information-extraction expert.\n"
    "Schema (only fill in pending keys, others already converged):\n"
    "  Pending : {pending_keys}\n"
    "  Resolved: {resolved_json}\n"
    "<<KEYS:{pending_csv}>>\n"
    "Question:\n{question}\n\n"
    "Respond with a single JSON object containing only the pending keys.\n"
)

_FC_UNSTRUCTURED_FALLBACK = (
    "Answer the QUESTION below. You must also answer each ANCHOR sub-question "
    "consistently. Reply with one JSON object whose keys include every "
    "pending anchor id and `final_answer`.\n\n"
    "QUESTION:\n{question}\n\n"
    "PENDING anchors (answer these):\n{pending_block}\n"
    "RESOLVED anchors (use these as constraints):\n{resolved_block}\n"
    "<<KEYS:{pending_csv}>>\n"
    "JSON:"
)


def build_structured_fc_prompt(
    question: str,
    pending_keys: List[str],
    resolved: Dict[str, str],
    schema_description: str = "",
    ground_truth_marker: Optional[Dict[str, str]] = None,
) -> str:
    template = _try_read(_PROMPT_DIR / "fc_structured.txt", _FC_STRUCTURED_FALLBACK)
    txt = template.format(
        pending_keys=", ".join(pending_keys) or "(none)",
        resolved_json=json.dumps(resolved, ensure_ascii=False),
        pending_csv=",".join(pending_keys),
        question=question,
        schema_description=schema_description or "",
    )
    if ground_truth_marker:
        gt_str = ";".join(f"{k}={v}" for k, v in ground_truth_marker.items())
        txt += f"\n<<GT:{gt_str}>>"
    return txt


def build_unstructured_fc_prompt(
    question: str,
    pending_anchor_ids: List[str],
    pending_anchor_texts: List[str],
    resolved: Dict[str, str],
    resolved_anchor_texts: Dict[str, str],
    ground_truth_marker: Optional[Dict[str, str]] = None,
) -> str:
    template = _try_read(_PROMPT_DIR / "fc_unstructured.txt", _FC_UNSTRUCTURED_FALLBACK)

    pending_block = (
        "\n".join(
            f"  - {aid}: {txt}"
            for aid, txt in zip(pending_anchor_ids, pending_anchor_texts)
        )
        or "  (none)"
    )
    resolved_block = (
        "\n".join(
            f"  - {aid}: {resolved_anchor_texts.get(aid, '')} -> {val}"
            for aid, val in resolved.items()
        )
        or "  (none)"
    )
    keys_csv = ",".join(list(pending_anchor_ids) + ["final_answer"])
    txt = template.format(
        question=question,
        pending_block=pending_block,
        resolved_block=resolved_block,
        pending_csv=keys_csv,
    )
    if ground_truth_marker:
        gt_str = ";".join(f"{k}={v}" for k, v in ground_truth_marker.items())
        txt += f"\n<<GT:{gt_str}>>"
    return txt
