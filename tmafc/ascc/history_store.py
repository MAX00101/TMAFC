"""JSONL-backed history store for ASCC."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tmafc.ascc.reclustering import recluster_by_quantile
from tmafc.utils.io import append_jsonl, load_jsonl, save_jsonl
from tmafc.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HistoryRecord:
    """One entry in the historical performance database.

    T : top-1 + top-2 sample counts.
    G : top-1 sample count (G <= T).
    """
    question: str
    T: int
    G: int
    category: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HistoryRecord":
        return cls(
            question=d.get("question", ""),
            T=int(d.get("T", 0)),
            G=int(d.get("G", 0)),
            category=d.get("category"),
            extra=d.get("extra", {}) or {},
        )


class HistoryStore:
    def __init__(
        self,
        path: str | Path,
        K: int = 3,
        recluster_every: int = 50,
        autoload: bool = True,
    ) -> None:
        self.path = Path(path)
        self.K = K
        self.recluster_every = recluster_every
        self._records: List[HistoryRecord] = []
        self._unclustered_since_last: int = 0
        if autoload:
            self.load()

    def load(self) -> None:
        rows = load_jsonl(self.path)
        self._records = [HistoryRecord.from_dict(r) for r in rows]
        logger.info("HistoryStore loaded %d records from %s", len(self._records), self.path)

    def save(self) -> None:
        save_jsonl(self.path, [r.to_dict() for r in self._records])

    def append(self, record: HistoryRecord) -> None:
        self._records.append(record)
        append_jsonl(self.path, record.to_dict())
        self._unclustered_since_last += 1
        if self._unclustered_since_last >= self.recluster_every:
            self.recluster()

    def recluster(self) -> None:
        if not self._records:
            return
        labels = recluster_by_quantile([r.T for r in self._records], K=self.K)
        for r, lbl in zip(self._records, labels):
            r.category = lbl
        self._unclustered_since_last = 0
        self.save()
        logger.info(
            "HistoryStore re-clustered %d records into K=%d buckets",
            len(self._records),
            self.K,
        )

    def filter_by_category(self, category: str) -> List[HistoryRecord]:
        return [r for r in self._records if r.category == category]

    def categories(self) -> List[str]:
        return sorted({r.category for r in self._records if r.category is not None})

    def __len__(self) -> int:
        return len(self._records)
