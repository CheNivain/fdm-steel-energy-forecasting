"""Run the extended Linear Regression feature-engineering experiment."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optimized_regression import cli


if __name__ == "__main__":
    cli("linear")

