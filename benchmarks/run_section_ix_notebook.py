"""Execute every code cell in the Section IX notebook in one Python process."""

from pathlib import Path
import nbformat


root = Path(__file__).resolve().parents[1]
notebook_path = root / "benchmarks" / "notebooks" / "section_ix_experiments.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
namespace = {"__name__": "__main__"}
for index, cell in enumerate(notebook.cells, start=1):
    if cell.cell_type == "code":
        print("Running notebook cell {}/{}".format(index, len(notebook.cells)), flush=True)
        exec(compile(cell.source, "{}:cell{}".format(notebook_path, index), "exec"), namespace)
