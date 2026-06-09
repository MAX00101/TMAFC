from pathlib import Path

from tmafc.ascc.history_store import HistoryRecord, HistoryStore


def _records():
    Ts = [2, 2, 3, 3, 5, 5, 12, 12, 30, 30]
    return [HistoryRecord(question=f"Q{i}", T=t, G=t - 1) for i, t in enumerate(Ts)]


def test_store_roundtrip(tmp_path: Path):
    p = tmp_path / "db.jsonl"
    store = HistoryStore(path=p, K=3, recluster_every=999)
    for r in _records():
        store.append(r)
    assert len(store) == 10

    fresh = HistoryStore(path=p, K=3)
    assert len(fresh) == 10


def test_recluster_assigns_three_categories(tmp_path: Path):
    p = tmp_path / "db.jsonl"
    store = HistoryStore(path=p, K=3, recluster_every=999)
    for r in _records():
        store.append(r)
    store.recluster()
    cats = store.categories()
    assert set(cats).issubset({"Easy", "Middle", "Hard"})
    assert len(cats) >= 2
