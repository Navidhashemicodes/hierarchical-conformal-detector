# Hierarchical Conformal Detector

A Python toolbox for anomaly detection in multivariate trajectories. It
combines a neural observer, a trajectory-level `k`-th-maximum residual score,
and a conformal monitor of sliding-window alarm rates. The paper's temporal
extension trains one observer for each omitted window position, samples one
observer uniformly at each time step, and calibrates a single detector on the
resulting randomized residual stream.

The reusable code is in `hierarchical_conformal/`. The paper's simulations,
application cases, plots, and executable notebooks are in `benchmarks/`.
Nothing in the toolbox depends on a particular plant model or simulator.

## Install

Python 3.8 or newer is required. From the repository root:

```powershell
python -m pip install -e .
```

For the paper benchmarks and notebooks:

```powershell
python -m pip install -e ".[benchmarks]"
```

## Use your own trajectories

Provide three disjoint sets of nominal measurements and a test set. Each array
has shape `(n_trajectories, n_time_steps, n_sensors)`. The time and sensor
dimensions must agree; the numbers of trajectories may differ.

```python
from hierarchical_conformal import HierarchicalDetector

detector = HierarchicalDetector(
    history=4, k_low=5, window=20, k_high=2,
    delta1_low=0.9, delta1_high=0.9,
)
detector.fit(train_nominal, low_calibration, high_calibration)

low_flags, high_flags = detector.predict(test_trajectories)
details = detector.score(test_trajectories)
# details["alarms"] is the time-resolved low-level alarm stream.
# details["low_alarm"] counts violations over the full valid trajectory.
```

The default observer is a reproducible one-hidden-layer ReLU random-feature
model with ridge-fitted output weights. Pass a custom observer implementing
`fit(X, y)` and `predict_flat(X)` to `HierarchicalDetector(observer=...)` to
use another estimator. If the observer is already fitted, `predict_flat(X)`
alone is sufficient. `X` holds past measurements, newest first.

For temporal behavior detection with per-time-step uniform observer sampling:

```python
from hierarchical_conformal import RandomizedWindowDetector

detector = RandomizedWindowDetector(
    window=5, k_low=5, rate_window=5, k_high=2,
    delta1_low=0.9, delta1_high=0.9,
)
detector.fit(train_nominal, low_calibration, high_calibration)
scores = detector.score(test_trajectories)
low_flags, high_flags = scores["low_alarm"], scores["high_alarm"]
```

Pass `observer_factory(position)` to construct your own observer for each
omitted position. `WindowEnsembleDetector` remains available for applications
that explicitly want observer-specific calibration and majority voting; that
is a different decision rule from the paper's randomized temporal extension.
`LowLevelDetector` accepts training and calibration residual vectors directly;
`HighLevelDetector` accepts calibration binary alarm streams. The public
`kth_max`, `conformal_rank`, `confidence_rank`, and `beta_confidence` functions
support lower-level integrations. See class docstrings for parameters and
returned arrays.

## Interpretation and assumptions

The low-level score is the `k`-th largest distance over a *complete valid
trajectory*. The low-level trajectory decision is true when the total number
of threshold violations exceeds `k`. The high-level score is the
`k_high`-th largest sliding-window alarm rate. Scores are calibrated on
complete nominal trajectories, so time samples inside a trajectory do not
need to be independent. Nominal calibration and test trajectories must be
exchangeable for the conformal statements to apply.

The two calibration splits are separate because the high-level split is
scored using a fixed low-level threshold. In `RandomizedWindowDetector`, each
trajectory receives fresh observer draws during calibration and testing, so
the certificates are marginal over both nominal trajectories and observer
randomization. `WindowEnsembleDetector` is a separate majority-vote method;
its votes do not inherit branchwise false-alarm certificates automatically.
An omitted-position prediction becomes available after its full window has
been observed.

`delta1` is the target nominal trajectory coverage conditional on calibration.
When a rank `ell` is supplied, its associated calibration confidence is
`beta_confidence(delta1, ell, m)`. The default rank is the conservative
marginal conformal rank. A requested high-confidence rank can be found with
`confidence_rank(m, delta1, delta2)` when the calibration size is sufficient.

## Reproduce the paper benchmarks

```powershell
python -m benchmarks.create_section_ix_notebook
python -m benchmarks.run_section_ix_notebook --publication-scale
```

`benchmarks/notebooks/section_ix_experiments.ipynb` contains the seven
publication-design experiments (`T=10000`, at least 24,100 training
trajectories, and 24,100 trajectories in each applicable calibration set).
Run it with `python -m benchmarks.run_section_ix_notebook --publication-scale`
using five sequential data batches. The publication run is computationally
large and may take a long time; it writes the generated results and figures
into the benchmark output folders.
The separate `section_ix_pilot.ipynb` uses small datasets and writes only to
`benchmarks/outputs/pilot/`; its numbers are not paper results. See
[benchmarks/README.md](benchmarks/README.md) for output locations and run
details. Files already present under `benchmarks/outputs/` are generated
artifacts from earlier runs, not authoritative publication results; regenerate
them with the commands above before using them. The publication status and
archived results, where available, are listed under `results/section_ix/`.

For a small implementation check, run
`python -m benchmarks.run_section_ix_notebook --pilot`.

Run the toolbox tests:

```powershell
python -m unittest discover -s tests -v
```

The manuscript is maintained separately from the toolbox. No publication
venue, DOI, or license is asserted by this repository.
