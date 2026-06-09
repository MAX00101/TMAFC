"""Run TMAFC over a JSONL of questions.

Each input line must have at least a ``question`` field. Optional fields:

    schema_keys         list[str]   if present, treat the row as structured.
    schema_description  str
    ground_truth        dict        used by the mock backend only.

Example:
    python scripts/run_tmafc.py \\
        --config configs/openai_gpt4.yaml \\
        --input  data/samples/ner_sample.jsonl \\
        --output runs/tmafc_ner.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from tmafc import TMAFC
from tmafc.core.decomposer import StructuredSchema
from tmafc.utils.io import load_jsonl
from tmafc.utils.logging import get_logger

logger = get_logger("scripts.run_tmafc")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="run TMAFC on a JSONL of questions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None,
                        help="output JSONL with predictions; defaults to stdout")
    parser.add_argument("--limit", type=int, default=0,
                        help="if >0, only process the first N rows")
    args = parser.parse_args(argv)

    load_dotenv()

    runner = TMAFC.from_config(args.config)
    rows = load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]

    out_fp = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for i, row in enumerate(rows):
            q = row["question"]
            schema = None
            if row.get("schema_keys"):
                schema = StructuredSchema(
                    keys=list(row["schema_keys"]),
                    description=row.get("schema_description", ""),
                )
            gt_for_mock = row.get("ground_truth") if isinstance(row.get("ground_truth"), dict) else None
            result = runner(q, schema=schema, ground_truth_for_mock=gt_for_mock)
            record = {
                "id": row.get("id", i),
                "question": q,
                "answer": result.answer,
                "category": result.calibration.category,
                "alpha0": result.calibration.alpha0,
                "beta0": result.calibration.beta0,
                "rounds_used": result.fc.rounds_used,
                "samples_consumed": result.fc.samples_consumed,
            }
            out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fp.flush()
            logger.info(
                "[%d/%d] %s samples=%d cat=%s",
                i + 1, len(rows), record["id"],
                record["samples_consumed"], record["category"],
            )
    finally:
        if out_fp is not sys.stdout:
            out_fp.close()


if __name__ == "__main__":
    main()
