"""Create the standalone notebook for the CoDiT GAIT comparison."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "notebooks" / "experiment_8_codit_gait.ipynb"

notebook = nbf.v4.new_notebook()
notebook.cells = [
    nbf.v4.new_markdown_cell("""# Experiment 8: CoDiT GAIT comparison

This notebook launches the downloaded upstream CoDiT GAIT evaluator and the
proposed randomized-observer detector on the same machine and native records.
It downloads CoDiT's GAIT archive on the first run and skips that step when all
expected records are already present. Before running it, follow
`benchmarks/experiment_8/README.md` to place the CoDiT source, download its
three released checkpoints, and create CoDiT's isolated Python environment.
The comparison follows CoDiT's 6/5/5 healthy split and severe patient subset.
It is descriptive and does not claim the large-sample conformal guarantee."""),
    nbf.v4.new_code_cell("""from pathlib import Path
import json
import sys

ROOT = next(path for path in (Path.cwd(), *Path.cwd().parents)
            if (path / "hierarchical_conformal").is_dir())
sys.path.insert(0, str(ROOT))

from benchmarks.run_experiment_8 import (
    DEFAULT_CODIT, ensure_gait_dataset, run_experiment_8)

codit_root = DEFAULT_CODIT
data_root = (codit_root / "ours" / "gait" / "data" /
             "gait-in-neurodegenerative-disease-database-1.0.0")
checkpoint_dir = codit_root / "ours" / "gait" / "saved_models"
output_path = ROOT / "benchmarks" / "outputs" / "experiment_8"
codit_python = "C:/path/to/conda/envs/gait/python.exe"  # Set to CoDiT's Python

downloaded = ensure_gait_dataset(data_root)
print("Downloaded GAIT data." if downloaded else "Using existing GAIT data.")

result = run_experiment_8(
    data_root=data_root, codit_root=codit_root,
    checkpoint_dir=checkpoint_dir, output_dir=output_path,
    codit_python=codit_python, n_randomizations=100, seed=1729)
print(json.dumps(result, indent=2))"""),
]
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
