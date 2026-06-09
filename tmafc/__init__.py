"""Task-and-Model-Aware Fractal-Consistency for efficient LLM reasoning."""

from tmafc.core.tmafc import TMAFC, TMAFCResult
from tmafc.fc.fc_runner import FCRunner, FCResult
from tmafc.ascc.ascc import ASCC, CalibrationResult

__all__ = [
    "TMAFC",
    "TMAFCResult",
    "FCRunner",
    "FCResult",
    "ASCC",
    "CalibrationResult",
]
__version__ = "0.1.0"
