"""Offline TMAFC demo using the mock backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tmafc import TMAFC
from tmafc.tasks.structured.ner import NER_SCHEMA
from tmafc.utils.logging import get_logger

logger = get_logger("scripts.run_demo")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "mock_demo.yaml"

_NER_QUESTION = (
    "There are so many cars on the highway that I cannot recognize but I think "
    "there is a Green Genesis GV80 driving Left with 165 km/h and a crocus BMW "
    "Five two five e driving this-way at a speed of 93 kilometers per hour."
)
_NER_GT = {
    "brand of vehicle": "[Genesis, BMW]",
    "orientation of vehicle": "[Left, this-way]",
    "truck": "[]",
    "vehicle model": "[GV80, Five two five e]",
}

_MATH_QUESTION = (
    "A vehicle runs at 60 km/h and speeds up by 20 km/h every 2 hours. "
    "What is the average velocity before the 5th speed-up?"
)
_MATH_GT = {
    "anchor_0": "10",
    "anchor_1": "120",
    "anchor_2": "70",
    "final_answer": "100",
}


def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _run_structured(runner: TMAFC) -> None:
    _banner("[1/2] Structured task: NER")
    result = runner(_NER_QUESTION, schema=NER_SCHEMA, ground_truth_for_mock=_NER_GT)
    print(
        f"\n[ASCC] category={result.calibration.category} "
        f"(alpha0, beta0)=({result.calibration.alpha0:.3f}, {result.calibration.beta0:.3f}) "
        f"|D_C|={result.calibration.n_examples}"
    )
    print(
        f"[FC]   rounds={result.fc.rounds_used} "
        f"samples_consumed={result.fc.samples_consumed}"
    )
    print("[Answer]")
    print(json.dumps(result.answer, ensure_ascii=False, indent=2))


def _run_unstructured(runner: TMAFC) -> None:
    _banner("[2/2] Unstructured task: math reasoning")
    result = runner(_MATH_QUESTION, schema=None, ground_truth_for_mock=_MATH_GT)
    print(
        f"\n[ASCC] category={result.calibration.category} "
        f"(alpha0, beta0)=({result.calibration.alpha0:.3f}, {result.calibration.beta0:.3f}) "
        f"|D_C|={result.calibration.n_examples}"
    )
    print(
        f"[FC]   rounds={result.fc.rounds_used} "
        f"samples_consumed={result.fc.samples_consumed}"
    )
    print(f"[Anchors] {result.decomposition.anchors}")
    print(f"[Answer] {result.answer}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="TMAFC offline demo")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)

    print(f"[demo] loading config: {args.config}")
    runner = TMAFC.from_config(args.config)

    _run_structured(runner)
    _run_unstructured(runner)

    print()
    print("history DB:", runner.history.path)


if __name__ == "__main__":
    main()
