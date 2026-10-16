"""Execute one Section IX notebook profile in one Python process."""

import argparse
from pathlib import Path
import nbformat


parser = argparse.ArgumentParser(description=__doc__)
profile = parser.add_mutually_exclusive_group(required=True)
profile.add_argument(
    "--pilot", action="store_true",
    help="acknowledge that this run uses T=300 and 200 calibration trajectories per level",
)
profile.add_argument(
    "--publication-scale", action="store_true",
    help="run all seven experiments with T=10000 and at least 24100 training trajectories",
)
options = parser.parse_args()

root = Path(__file__).resolve().parents[1]
notebook_name = "section_ix_pilot.ipynb" if options.pilot else "section_ix_experiments.ipynb"
notebook_path = root / "benchmarks" / "notebooks" / notebook_name
notebook = nbformat.read(notebook_path, as_version=4)
namespace = {"__name__": "__main__"}
for index, cell in enumerate(notebook.cells, start=1):
    if cell.cell_type == "code":
        print("Running notebook cell {}/{}".format(index, len(notebook.cells)), flush=True)
        exec(compile(cell.source, "{}:cell{}".format(notebook_path, index), "exec"), namespace)
