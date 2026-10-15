# Hierarchical Conformal Detector

A Python toolbox for anomaly detection in multivariate trajectories. It
combines a neural observer, a trajectory-level `k`-th-maximum residual score,
and a conformal monitor of sliding-window alarm rates. A window ensemble can
train one observer for each omitted position in a measurement window and vote
separately at the low and high levels.

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

For temporal behavior detection, fit one observer for each omitted window
position:

```python
from hierarchical_conformal import WindowEnsembleDetector

ensemble = WindowEnsembleDetector(
    window=5, k_low=5, rate_window=5, k_high=2,
    delta1_low=0.9, delta1_high=0.9,
)
ensemble.fit(train_nominal, low_calibration, high_calibration)
low_votes, high_votes = ensemble.predict(test_trajectories)
branch_results = ensemble.score(test_trajectories)
```

Pass `observer_factory(position)` to construct your own observer for each
position. `LowLevelDetector` accepts training and calibration residual vectors
directly; `HighLevelDetector` accepts calibration binary alarm streams. The
public `kth_max`, `conformal_rank`, `confidence_rank`, and
`beta_confidence` functions support lower-level integrations. See class
docstrings for all parameters and returned arrays.

## Interpretation and assumptions

The low-level score is the `k`-th largest distance over a *complete valid
trajectory*. The low-level trajectory decision is true when the total number
of threshold violations exceeds `k`. The high-level score is the
`k_high`-th largest sliding-window alarm rate. Scores are calibrated on
complete nominal trajectories, so time samples inside a trajectory do not
need to be independent. Nominal calibration and test trajectories must be
exchangeable for the conformal statements to apply.

The two calibration splits are separate because the high-level split is
scored using a fixed low-level threshold. The stated conformal certificate
applies to each calibrated branch. Majority votes across shared measurements
do not inherit a branchwise false-alarm certificate automatically. For an
omitted-position observer, a score becomes available when its full window has
been observed.

`delta1` is the target nominal trajectory coverage conditional on calibration.
When a rank `ell` is supplied, its associated calibration confidence is
`beta_confidence(delta1, ell, m)`. The default rank is the conservative
marginal conformal rank. A requested high-confidence rank can be found with
`confidence_rank(m, delta1, delta2)` when the calibration size is sufficient.

## Reproduce the paper benchmarks

```powershell
python -m benchmarks.run_section_ix_notebook
```

The notebook is at
`benchmarks/notebooks/section_ix_experiments.ipynb`. In the standalone
toolbox it saves JSON results under `benchmarks/outputs/results/section_ix/`,
figures under `benchmarks/outputs/figures/`, and LaTeX tables under
`benchmarks/outputs/tables/`. In a manuscript checkout containing
`paper/main.tex`, it writes to that checkout's `results/` and `paper/`
folders instead. The checked-in Section IX values are from
the pilot setting `T=300`, `m_low=m_high=200`. The paper also discusses an
illustrative `T=10000`, `m=24100` design, which has not been executed here.
See [benchmarks/README.md](benchmarks/README.md) for benchmark details.

Run the toolbox tests:

```powershell
python -m unittest discover -s tests -v
```

The manuscript is maintained separately from the toolbox. No publication
venue, DOI, or license is asserted by this repository.
