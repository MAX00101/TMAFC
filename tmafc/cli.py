"""Console-script entry points (registered in pyproject.toml)."""
from __future__ import annotations

import sys


def demo() -> None:
    from scripts import run_demo
    run_demo.main()


def run() -> None:
    from scripts import run_tmafc
    run_tmafc.main(sys.argv[1:])


def bootstrap() -> None:
    from scripts import bootstrap_history
    bootstrap_history.main(sys.argv[1:])
