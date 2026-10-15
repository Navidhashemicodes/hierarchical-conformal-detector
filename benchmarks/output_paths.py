"""Output locations for manuscript and standalone toolbox checkouts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HAS_MANUSCRIPT = (ROOT / "paper" / "main.tex").is_file()

if HAS_MANUSCRIPT:
    RESULTS = ROOT / "results"
    FIGURES = ROOT / "paper" / "figures"
    TABLES = ROOT / "paper" / "tables"
else:
    OUTPUTS = ROOT / "benchmarks" / "outputs"
    RESULTS = OUTPUTS / "results"
    FIGURES = OUTPUTS / "figures"
    TABLES = OUTPUTS / "tables"
