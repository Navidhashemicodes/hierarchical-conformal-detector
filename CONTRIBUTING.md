# Contributing

The public API lives in `hierarchical_conformal/`. Keep plant models,
synthetic attack definitions, plot styling, and paper output paths in
`benchmarks/`. A toolbox change should work on user-supplied trajectory
arrays without importing benchmark modules.

Before proposing a change, run:

```powershell
python -m unittest discover -s tests -v
python -m unittest benchmarks.test_pipeline -v
```

If benchmark behavior changes, regenerate the Section IX artifacts with
`python -m benchmarks.run_section_ix_notebook` and compare the JSON results
against the published numerical statements. Document any changed
random seed, calibration split, observer, filter, or decision rule.

The conformal calibration unit is the complete nominal trajectory. Preserve
the separate estimator-training, low-level calibration, and high-level
calibration splits. Low-level trajectory decisions count alarms over every
valid score time; window ensemble votes are computed separately at the low
and high levels.
