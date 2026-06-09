# TMAFC

Reference implementation of **Task-and-Model-Aware Fractal-Consistency** (ICML 2026):
self-consistency for LLM reasoning that decomposes outputs into atomic
components and calibrates the Beta stopping prior to the task and model.

The framework has two parts:

- `tmafc.fc` — Fractal-Consistency: per-component Beta stopping with
  prompt refresh on un-converged components, plus an LLM anchor
  generator for unstructured tasks (Algorithm 1).
- `tmafc.ascc` — Adaptive Stopping Criteria Calibration: an LLM-based
  difficulty classifier and L-BFGS-B Beta-Binomial MLE on a per-category
  history database that yields task-and-model-aware priors `(α0, β0)`
  for FC (Algorithm 2).

## Install

```bash
git clone https://github.com/MAX00101/TMAFC.git
cd TMAFC
pip install -e .            # core deps only
pip install -e .[openai]    # add the OpenAI backend
pip install -e .[dev]       # add pytest
```

Python >= 3.9.

## Quick start

Offline demo with no API key (uses the bundled `MockLLM`):

```bash
python scripts/run_demo.py
```

Real LLM run on a JSONL of questions:

```bash
cp .env.example .env                # set OPENAI_API_KEY
pip install -e .[openai]

python scripts/bootstrap_history.py \
    --config configs/openai_gpt4.yaml \
    --input  data/samples/math_sample.jsonl \
    --n      50

python scripts/run_tmafc.py \
    --config configs/openai_gpt4.yaml \
    --input  data/samples/ner_sample.jsonl \
    --output runs/tmafc_ner.jsonl
```

Library use:

```python
from tmafc import TMAFC
from tmafc.tasks.structured.ner import NER_SCHEMA

runner = TMAFC.from_config("configs/default.yaml")
out = runner("A Genesis GV80 driving Left.", schema=NER_SCHEMA)
print(out.answer)
print(out.calibration)
print(out.fc.samples_consumed)
```

## Layout

```
tmafc/
  fc/        Fractal-Consistency (Algorithm 1)
  ascc/      Adaptive Stopping Criteria Calibration (Algorithm 2)
  core/      end-to-end orchestrator
  llm/       backend interface (mock / openai)
  tasks/     task adapters
  baselines/ Self-Consistency and Adaptive-Consistency
  utils/     logging, I/O
configs/     YAML configs (default, mock_demo, openai_gpt4)
prompts/     FC, anchor and classifier templates
data/        history DB and sample inputs
scripts/     run_demo, run_tmafc, bootstrap_history
examples/    standalone usage of FC, ASCC, end-to-end
tests/       pytest, no API key required
```

## Configuration

```yaml
fc:
  tau: 0.95
  N_max: 50
  stopping_logic: ALL_CONVERGED       # or GEOMETRIC_MEAN
ascc:
  K: 3
  bounds: [0.01, 1.0]                 # eps_min, eps_max
  recluster_every: 50
classifier:
  backend: same_as_llm                # same_as_llm | rule | openai | mock
llm:
  backend: openai
  model: gpt-4
```

## Custom tasks

```python
from tmafc.core.decomposer import StructuredSchema

schema = StructuredSchema(
    keys=["entity_type", "value"],
    description="Extract <type, value> pairs.",
)
out = runner(my_question, schema=schema)
```

For unstructured tasks pass `schema=None`; the anchor generator is invoked
automatically.

## Tests

```bash
pip install -e .[dev]
pytest tests -q
```

The suite uses `MockLLM` and runs in well under a minute without network.

## Citation

```bibtex
@inproceedings{luo2026tmafc,
  title     = {Task-and-Model-Aware Fractal-Consistency for Efficient LLM Reasoning},
  author    = {Luo, Ziqiu and Liu, Jianmin and Miao, Yukai and Chen, Li and Li, Dan},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

MIT.
