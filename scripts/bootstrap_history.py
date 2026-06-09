"""Seed the ASCC history database via Adaptive-Consistency pre-sampling."""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from tmafc.ascc.history_store import HistoryRecord, HistoryStore
from tmafc.baselines.adaptive_consistency import AdaptiveConsistency
from tmafc.core.voting import normalize_value, parse_json_response
from tmafc.llm.registry import from_config as build_llm
from tmafc.utils.io import load_jsonl, load_yaml
from tmafc.utils.logging import get_logger

logger = get_logger("scripts.bootstrap_history")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="seed the ASCC history DB with AC pre-samples")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="JSONL of {question, ...}")
    parser.add_argument("--n", type=int, default=100, help="number of questions to process")
    parser.add_argument("--n_max", type=int, default=50, help="per-question AC sample cap")
    parser.add_argument("--tau", type=float, default=0.95)
    args = parser.parse_args(argv)

    load_dotenv()

    cfg = load_yaml(args.config)
    llm = build_llm(cfg.get("llm", {"backend": "mock"}))

    history_path = cfg.get("ascc", {}).get("history_path", "data/history/db.jsonl")
    store = HistoryStore(
        path=history_path,
        K=int(cfg.get("ascc", {}).get("K", 3)),
        recluster_every=int(cfg.get("ascc", {}).get("recluster_every", 50)),
    )

    rows = load_jsonl(args.input)[: args.n]
    ac = AdaptiveConsistency(tau=args.tau, n_max=args.n_max)

    for i, row in enumerate(rows):
        q = row["question"]

        def _sample_once() -> str:
            outs = llm.chat(q, n=1)
            if not outs:
                return ""
            parsed = parse_json_response(outs[0])
            return normalize_value(parsed if parsed is not None else outs[0])

        result = ac.run(_sample_once)
        rec = HistoryRecord(
            question=q,
            T=int(result.T),
            G=int(result.G),
            category=None,
            extra={"samples_consumed": str(result.samples_consumed)},
        )
        store.append(rec)
        logger.info(
            "[%d/%d] AC samples=%d T=%d G=%d conf=%.3f",
            i + 1, len(rows), result.samples_consumed, result.T, result.G, result.confidence,
        )

    store.recluster()
    logger.info("bootstrap complete: %s with %d records", history_path, len(store))


if __name__ == "__main__":
    main()
