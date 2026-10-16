"""Resume publication experiments 6 and 7 without repeating shared calibration."""

from argparse import Namespace
from pathlib import Path
import nbformat


ROOT = Path(__file__).resolve().parents[1]
notebook_path = ROOT / "benchmarks" / "notebooks" / "section_ix_experiments.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
namespace = {"__name__": "__main__"}


def run_cell(index):
    cell = notebook.cells[index - 1]
    print("Running notebook cell {}/{}".format(index, len(notebook.cells)), flush=True)
    exec(compile(cell.source, "{}:cell{}".format(notebook_path, index), "exec"), namespace)


# Bootstrap the same imports and publication output paths used by the full run.
run_cell(2)
run_cell(16)
run_cell(18)

# Rebuild the settings needed by the notebook's standard manifest cell.
args = Namespace(seed=7, dim=3, steps=10000, n_train=24100, m_low=24100,
    m_high=24100, n_test=1000, k_low=500, window=500, k_high=50,
    delta1_low=0.9995, delta1_high=0.9995, ell_low=24098, ell_high=24098,
    ridge=1e-5, history=4, hidden=32, filter_order=3,
    filter_cutoff=0.08, filter_warmup=9,
    k_ablation_values="100,250,500,750,1000", publication_scale=True)
run_cell(20)
