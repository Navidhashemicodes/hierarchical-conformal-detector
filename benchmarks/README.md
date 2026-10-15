# Paper benchmarks

The notebooks here call the public `hierarchical_conformal` classes and
utilities. Synthetic signal generation, attack injection, plotting, and LaTeX
table export are benchmark code; they are not required by the toolbox.

Run from the repository root after `pip install -e ".[benchmarks]"`:

```powershell
python -m benchmarks.create_section_ix_notebook
python -m benchmarks.run_section_ix_notebook
```

Open `benchmarks/notebooks/section_ix_experiments.ipynb` to run the seven
Section IX experiments interactively. The default is a small reproducible
pilot (`T=300`, `m_low=m_high=200`); it does not implement the paper's
illustrative high-confidence calibration size. In the standalone toolbox,
results, figures, and tables are written under `benchmarks/outputs/`.
The manuscript is not part of the toolbox repository.

The simulator in `simulate.py` is only one way to create trajectories. The
toolbox API accepts arrays from any system when each split contains complete
nominal trajectories with matching time and sensor dimensions.
