"""End-to-end TMAFC orchestration: ASCC -> FC -> history append."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tmafc.ascc.ascc import ASCC, CalibrationResult
from tmafc.ascc.classifier import LLMDifficultyClassifier, RuleBasedClassifier
from tmafc.ascc.history_store import HistoryRecord, HistoryStore
from tmafc.core.decomposer import Decomposer, DecomposedTask, StructuredSchema
from tmafc.core.prompts import build_structured_fc_prompt, build_unstructured_fc_prompt
from tmafc.core.voting import normalize_value, parse_json_response
from tmafc.fc.anchor_generator import AnchorGenerator
from tmafc.fc.fc_runner import FCResult, FCRunner
from tmafc.fc.final_aggregator import aggregate_unstructured_answer
from tmafc.llm.base import BaseLLM
from tmafc.llm.registry import from_config as build_llm
from tmafc.utils.io import load_yaml
from tmafc.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TMAFCResult:
    answer: Any
    calibration: CalibrationResult
    fc: FCResult
    decomposition: DecomposedTask
    raw_samples: List[Dict[str, Any]] = field(default_factory=list)


class TMAFC:
    def __init__(
        self,
        llm: BaseLLM,
        ascc: ASCC,
        fc: FCRunner,
        decomposer: Decomposer,
        history: HistoryStore,
        record_to_history: bool = True,
    ) -> None:
        self.llm = llm
        self.ascc = ascc
        self.fc = fc
        self.decomposer = decomposer
        self.history = history
        self.record_to_history = record_to_history

    @classmethod
    def from_config(cls, path: str | Path) -> "TMAFC":
        return cls.from_dict(load_yaml(path))

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TMAFC":
        llm_cfg = dict(cfg.get("llm", {"backend": "mock"}))
        llm = build_llm(llm_cfg)

        anchor_gen = (
            AnchorGenerator(llm=llm) if cfg.get("fc", {}).get("anchors", True) else None
        )
        decomposer = Decomposer(anchor_generator=anchor_gen)

        h_cfg = cfg.get("ascc", {})
        history = HistoryStore(
            path=h_cfg.get("history_path", "data/history/db.jsonl"),
            K=int(h_cfg.get("K", 3)),
            recluster_every=int(h_cfg.get("recluster_every", 50)),
        )

        cls_cfg = cfg.get("classifier", {})
        cls_backend = cls_cfg.get("backend", "same_as_llm")
        if cls_backend == "rule":
            classifier = RuleBasedClassifier()
        else:
            cls_llm = (
                llm
                if cls_backend == "same_as_llm"
                else build_llm(cls_cfg.get("llm", llm_cfg))
            )
            classifier = LLMDifficultyClassifier(
                llm=cls_llm,
                history=history,
                K=int(h_cfg.get("K", 3)),
                examples_per_class=int(cls_cfg.get("examples_per_class", 2)),
            )

        bounds = h_cfg.get("bounds", [0.01, 1.0])
        ascc = ASCC(
            classifier=classifier,
            history=history,
            eps_min=float(bounds[0]),
            eps_max=float(bounds[1]),
        )
        fc_cfg = cfg.get("fc", {})
        fc = FCRunner(
            tau=float(fc_cfg.get("tau", 0.95)),
            n_max=int(fc_cfg.get("N_max", 50)),
            min_samples_for_conf=int(fc_cfg.get("min_samples_for_conf", 2)),
            stopping_logic=str(fc_cfg.get("stopping_logic", "ALL_CONVERGED")),
            geo_mean_threshold=float(fc_cfg.get("geo_mean_threshold", 0.9)),
        )

        return cls(
            llm=llm,
            ascc=ascc,
            fc=fc,
            decomposer=decomposer,
            history=history,
            record_to_history=bool(cfg.get("record_to_history", True)),
        )

    def __call__(
        self,
        question: str,
        schema: Optional[StructuredSchema] = None,
        ground_truth_for_mock: Optional[Dict[str, str]] = None,
    ) -> TMAFCResult:
        calib = self.ascc.calibrate(question)
        decomp = self.decomposer.decompose(question, schema=schema)
        raw_samples: List[Dict[str, Any]] = []

        def sample_fn(states):
            pending_ids = list(states.keys())
            resolved_values: Dict[str, str] = {}

            if decomp.is_structured:
                prompt = build_structured_fc_prompt(
                    question=question,
                    pending_keys=pending_ids,
                    resolved=resolved_values,
                    schema_description=(
                        decomp.schema.description if decomp.schema else ""
                    ),
                    ground_truth_marker=ground_truth_for_mock,
                )
            else:
                anchors = decomp.anchors or []
                anchor_id_to_text = {f"anchor_{i}": t for i, t in enumerate(anchors)}
                pending_texts = [anchor_id_to_text.get(aid, aid) for aid in pending_ids]
                prompt = build_unstructured_fc_prompt(
                    question=question,
                    pending_anchor_ids=pending_ids,
                    pending_anchor_texts=pending_texts,
                    resolved=resolved_values,
                    resolved_anchor_texts=anchor_id_to_text,
                    ground_truth_marker=ground_truth_for_mock,
                )

            outputs = self.llm.chat(prompt, n=1)
            if not outputs:
                return {}
            parsed = parse_json_response(outputs[0])
            if not isinstance(parsed, dict):
                return {}
            raw_samples.append(parsed)
            return {
                cid: normalize_value(parsed[cid])
                for cid in pending_ids
                if cid in parsed
            }

        fc_result = self.fc.run(
            component_ids=decomp.component_ids,
            sample_fn=sample_fn,
            prior=(calib.alpha0, calib.beta0),
        )

        if decomp.is_structured:
            answer: Any = dict(fc_result.converged_values)
        else:
            converged = fc_result.converged_values
            if "final_answer" in decomp.component_ids:
                answer = converged.get("final_answer")
            else:
                final, _ = aggregate_unstructured_answer(
                    samples=raw_samples,
                    converged_anchors=converged,
                    final_answer_key="final_answer",
                )
                answer = final

        if self.record_to_history:
            T = int(round(fc_result.avg_top1 + fc_result.avg_top2)) or fc_result.samples_consumed
            G = int(round(fc_result.avg_top1)) or 1
            self.history.append(
                HistoryRecord(
                    question=question,
                    T=T,
                    G=G,
                    category=calib.category,
                    extra={"rounds_used": str(fc_result.rounds_used)},
                )
            )

        return TMAFCResult(
            answer=answer,
            calibration=calib,
            fc=fc_result,
            decomposition=decomp,
            raw_samples=raw_samples,
        )
