# Paper benchmarks

The notebooks here call the public `hierarchical_conformal` classes and
utilities. Synthetic signal generation, attack injection, plotting, and LaTeX
table export are benchmark code; they are not required by the toolbox.

Generate the Section IX notebooks from the repository root after
`pip install -e ".[benchmarks]"`:

```powershell
python -m benchmarks.create_section_ix_notebook
python -m benchmarks.create_experiment_8_notebook
```

`benchmarks/notebooks/section_ix_experiments.ipynb` has the paper design in
Experiments 1-7: `T=10000`, at least 24,100 training trajectories,
24,100 low-level calibration trajectories, and another 24,100 high-level
calibration trajectories where that level is used. The rank is `ell=24098`
at `delta1=0.9995`, yielding `delta2` approximately `0.99950047`. In Experiment 6, the five window-position observers are trained on the proper training set. For every trajectory, an observer is drawn independently and uniformly at each window-completion time; the selected residual stream is filtered before the warmup is discarded. One low-level ellipsoid and one high-level detector are calibrated from separate trajectory sets using the same randomization law. Their conformal guarantees are marginal over the observer draws. The former majority-vote pilot outputs have been cleared; Experiment 6 publication-scale results are pending.

Experiment 7's publication-scale design is implemented in the application
runner. The checked-in `results/section_ix/experiment_7.json` contains only a
diagnostic pilot (`T=300`, 60 training trajectories, and 200 calibration
trajectories); publication-scale results are pending. Do not interpret those
pilot values as results from the `T=10000`, 24,100-trajectory design.

Experiment 8 is an exploratory comparison with CoDiT on its GAIT case study. The runner invokes the upstream `ours/gait/check_OOD_gait.py` on the same machine and reads the same native 12-feature records for the proposed detector. It follows CoDiT's exact healthy split and severe patient subset and requires the three released GAIT checkpoints. Since the dataset has only 16 healthy subjects, this comparison cannot use the 24,100-independent-trajectory calibration design or claim its conformal guarantee. It reports AUROC as descriptive performance and preserves subject-level separation.

The standalone notebook is `benchmarks/notebooks/experiment_8_codit_gait.ipynb`.
The Experiment 8 runner is implemented at `benchmarks/run_experiment_8.py`.
Follow `benchmarks/experiment_8/README.md` to obtain CoDiT's source tree,
released checkpoints, and isolated runtime. The notebook and command-line
runner automatically download/extract GAIT data on first use and skip that
step when the full record set is already present. Then run
`python -m benchmarks.run_experiment_8 --codit-python <CoDiT Python path>`.
The runner calls CoDiT directly for W=16,18,20, fits the public
randomized-window detector on the matching records, aligns both methods by
subject and raw window index, and saves per-window scores plus AUROC outputs.
It is descriptive because windows from a subject are dependent. The notebook
calls this same runner; it does not have a separate Experiment 8 implementation.

The reusable in-memory API for the paper's randomized observer rule is
`hierarchical_conformal.RandomizedWindowDetector`. Experiment 6 uses a
five-batch runner for the same method to avoid materializing publication-scale
trajectory windows in memory; its randomized residual selection calls the
same `randomized_window_residuals` utility exposed by the toolbox. The older
`WindowEnsembleDetector` is a separate majority-vote method.

To run the paper design, use a machine with sufficient memory and time:

```powershell
python -m benchmarks.run_section_ix_notebook --publication-scale
```

The publication runner regenerates training trajectories in exactly five
outer batches. It accumulates normalization moments, ridge normal equations,
shape moments, and trajectory calibration scores without retaining the full
training design matrix.

For a small diagnostic run, use:

```powershell
python -m benchmarks.run_section_ix_notebook --pilot
```

The pilot notebook is `benchmarks/notebooks/section_ix_pilot.ipynb` and uses
`T=300`, `m_low=m_high=200`, and `delta1=0.9`. Its outputs are isolated under
`benchmarks/outputs/pilot/`. Publication outputs use the repository's normal
benchmark output folders.
The manuscript is not part of the toolbox repository.

The simulator in `simulate.py` is only one way to create trajectories. The
toolbox API accepts arrays from any system when each split contains complete
nominal trajectories with matching time and sensor dimensions.
