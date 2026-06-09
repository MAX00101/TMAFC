"""End-to-end TMAFC call with the mock backend."""
from __future__ import annotations

from tmafc import TMAFC
from tmafc.tasks.structured.ner import NER_SCHEMA


def main() -> None:
    runner = TMAFC.from_config("configs/mock_demo.yaml")

    q = (
        "There is a Green Genesis GV80 driving Left and a BMW Five two five e "
        "driving this-way."
    )
    gt = {
        "brand of vehicle": "[Genesis, BMW]",
        "orientation of vehicle": "[Left, this-way]",
        "truck": "[]",
        "vehicle model": "[GV80, Five two five e]",
    }
    result = runner(q, schema=NER_SCHEMA, ground_truth_for_mock=gt)

    print("calibration :", result.calibration)
    print("rounds used :", result.fc.rounds_used)
    print("samples used:", result.fc.samples_consumed)
    print("answer      :", result.answer)


if __name__ == "__main__":
    main()
