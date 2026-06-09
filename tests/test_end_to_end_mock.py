import pytest

from tmafc import TMAFC
from tmafc.core.decomposer import StructuredSchema


@pytest.fixture
def tmp_config(tmp_path):
    cfg_text = f"""
llm:
  backend: mock
  agreement: 0.95
  seed: 0
fc:
  tau: 0.85
  N_max: 15
  min_samples_for_conf: 2
  stopping_logic: ALL_CONVERGED
  anchors: true
ascc:
  K: 3
  recluster_every: 100
  bounds: [0.01, 1.0]
  history_path: {tmp_path / 'db.jsonl'}
classifier:
  backend: rule
  examples_per_class: 2
record_to_history: true
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(cfg_text, encoding="utf-8")
    return p


def test_end_to_end_structured_with_mock(tmp_config):
    runner = TMAFC.from_config(tmp_config)
    schema = StructuredSchema(keys=["brand", "model"])
    gt = {"brand": "Genesis", "model": "GV80"}
    out = runner("Identify the vehicle.", schema=schema, ground_truth_for_mock=gt)
    assert out.answer["brand"] == "Genesis"
    assert out.answer["model"] == "GV80"
    assert out.fc.samples_consumed >= 1
    assert len(runner.history) == 1
