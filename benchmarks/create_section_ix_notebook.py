"""Generate separate publication-design and small diagnostic notebooks."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "benchmarks" / "notebooks"


def code(source):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    nbf.v4.new_markdown_cell("""# Section IX: numerical examples

The publication design uses \\(T=10000\\), at least 24,100 training
trajectories, and separate low- and high-level calibration sets of 24,100
trajectories each. Both levels use \\(\\ell=24098\\) and
\\(\\delta_1=0.9995\\). The diagnostic notebook is a small pilot and does
not establish this guarantee. Experiments 1–7 write JSON results, PDF figures,
and LaTeX tables in the benchmark output folders. Experiment 8 launches the
upstream CoDiT GAIT evaluator and our detector on the same machine and records;
follow `benchmarks/experiment_8/README.md` to stage its source, data,
checkpoints, and isolated Python environment.
The shared training/calibration cell uses
disjoint nominal trajectory sets and a fixed sensor mixing matrix."""),
    code(r'''
import json
import sys
from argparse import Namespace
from pathlib import Path
PUBLICATION_SCALE = __PUBLICATION_SCALE__
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = next(path for path in (Path.cwd(), *Path.cwd().parents)
            if (path / "hierarchical_conformal").is_dir())
sys.path.insert(0, str(ROOT))

from hierarchical_conformal import (LowLevelDetector, HighLevelDetector,
    WindowEnsembleDetector, conformal_high_level_delay, conformal_high_level_flags,
    conformal_threshold, high_level_scores, kth_max, low_level_alarms,
    low_level_scores, sliding_window_rates)
from benchmarks.run_all import (aligned_unfiltered_residuals, calibrate, filtered_residuals,
    measurements_from_deltas, quadratic_distances, validate_args)
from benchmarks.streaming_pipeline import (calibrate_five_batches,
    run_randomized_window_five_batches, run_applications_five_batches)
from benchmarks.simulate import attack_measurements, nominal_deltas
from benchmarks.window_ensemble import (SCENARIOS, default_config, run_ensemble_experiment,
    save_ensemble_results)
from benchmarks.application_system_tests import (run_linear_residual_case,
    run_nonlinear_filtered_residual_case, write_application_figure)
from benchmarks.output_paths import RESULTS as RESULTS_ROOT, FIGURES, TABLES
from benchmarks.publication_design import (apply_baseline, apply_application,
                                           apply_ensemble, check_design)

if PUBLICATION_SCALE:
    RESULTS = RESULTS_ROOT / "section_ix"
else:
    pilot_root = ROOT / "benchmarks" / "outputs" / "pilot"
    RESULTS, FIGURES, TABLES = (pilot_root / "results" / "section_ix",
                               pilot_root / "figures", pilot_root / "tables")
for directory in (RESULTS, FIGURES, TABLES):
    directory.mkdir(parents=True, exist_ok=True)

def save_json(name, data):
    path = RESULTS / (name + ".json")
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path

def save_table(name, headers, rows):
    columns = "l" + "c" * (len(headers) - 1)
    lines = [r"\begin{tabular}{" + columns + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(str(item) for item in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = TABLES / (name + ".tex")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

print("Output root:", ROOT)
'''),
    nbf.v4.new_markdown_cell("## Shared nominal training and calibration"),
    code(r'''
args = Namespace(seed=7, dim=3, steps=300, n_train=60, m_low=200,
    m_high=200, n_test=80, k_low=5, window=20, k_high=2,
    delta1_low=0.9, delta1_high=0.9, ell_low=0, ell_high=0,
    ridge=1e-5, history=4, hidden=32, filter_order=3,
    filter_cutoff=0.08, filter_warmup=9, k_ablation_values="1,5,10,20",
    publication_scale=False)
if PUBLICATION_SCALE:
    args = apply_baseline(args)
    args.publication_scale = True
validate_args(args)
rng = np.random.default_rng(args.seed)
if PUBLICATION_SCALE:
    estimator, calibration, artifacts = calibrate_five_batches(args, seed=args.seed)
    rng = np.random.default_rng(args.seed + 999)
else:
    estimator, calibration, artifacts = calibrate(args, rng)
low_model, high_model = artifacts["low_model"], artifacts["high_model"]
np.testing.assert_allclose(low_model.threshold_, calibration.low_threshold)
np.testing.assert_allclose(high_model.threshold_, calibration.high_threshold)
nominal_y = measurements_from_deltas(nominal_deltas(
    args.n_test, args.steps, args.dim, rng, mixing=artifacts["mixing"]))
d_nom = quadratic_distances(filtered_residuals(nominal_y, estimator, args),
                            artifacts["center"], artifacts["q_inv"])
a_nom = low_level_alarms(d_nom, calibration.low_threshold)
np.testing.assert_array_equal(a_nom, low_model.alarms(
    filtered_residuals(nominal_y, estimator, args)))
save_json("calibration", {"settings": vars(args),
    "low_threshold": calibration.low_threshold,
    "high_threshold": calibration.high_threshold,
    "low_rank": calibration.low_rank, "high_rank": calibration.high_rank,
    "low_delta2": calibration.low_delta2,
    "high_delta2": calibration.high_delta2,
    "valid_residual_times": int(d_nom.shape[1])})
print("Shared calibration complete")
'''),
    nbf.v4.new_markdown_cell("## Experiment 1 — low-level conformal validity"),
    code(r'''
low_scores = low_level_scores(d_nom, args.k_low)
alarm_counts = a_nom.sum(axis=1)
exp1 = {"coverage": float(np.mean(low_scores <= calibration.low_threshold)),
    "fraction_within_k": float(np.mean(alarm_counts <= args.k_low)),
    "mean_alarm_rate": float(a_nom.mean()),
    "p95_alarm_rate": float(np.quantile(a_nom.mean(axis=1), 0.95)),
    "threshold": calibration.low_threshold, "k": args.k_low,
    "valid_times": int(d_nom.shape[1])}
save_json("experiment_1", exp1)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.hist(artifacts["scores_low"], bins=25)
ax.axvline(calibration.low_threshold, color="crimson", label="certificate")
ax.set(xlabel="k-th maximum residual distance", ylabel="Calibration trajectories")
ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "low_level_validity.pdf"); plt.close(fig)
save_table("low_level_validity", ["Coverage", "Within $k$", "Mean alarm rate"],
           [[f"{exp1['coverage']:.3f}", f"{exp1['fraction_within_k']:.3f}",
             f"{exp1['mean_alarm_rate']:.2e}"]])
print(exp1)
'''),
    nbf.v4.new_markdown_cell("## Experiment 2 — high-level alarm-rate calibration"),
    code(r'''
high_scores = high_level_scores(a_nom, args.window, args.k_high)
high_flags = high_scores > calibration.high_threshold
exp2 = {"high_threshold": calibration.high_threshold,
    "empirical_coverage": float(np.mean(~high_flags)),
    "false_alarm_rate": float(np.mean(high_flags)),
    "mean_score": float(np.mean(high_scores)),
    "window": args.window, "k_high": args.k_high}
save_json("experiment_2", exp2)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.hist(artifacts["scores_high"], bins=20)
ax.axvline(calibration.high_threshold, color="crimson", label="certificate")
ax.set(xlabel="k-th maximum window alarm rate", ylabel="Calibration trajectories")
ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "high_level_certificates.pdf"); plt.close(fig)
save_table("high_level_validity", ["Certificate", "Coverage", "False alarm rate"],
           [[f"{exp2['high_threshold']:.3f}", f"{exp2['empirical_coverage']:.3f}",
             f"{exp2['false_alarm_rate']:.3f}"]])
print(exp2)
'''),
    nbf.v4.new_markdown_cell("## Experiment 3 — scenario-based alarm patterns"),
    code(r'''
scenario_y = {}
scenario_distances = {}
exp3 = {}
for kind in SCENARIOS:
    scenario_y[kind] = attack_measurements(nominal_y, kind, rng)
    d = quadratic_distances(filtered_residuals(scenario_y[kind], estimator, args),
                            artifacts["center"], artifacts["q_inv"])
    scenario_distances[kind] = d
    alarms = low_level_alarms(d, calibration.low_threshold)
    delays = conformal_high_level_delay(alarms, args.window, args.k_high,
                                        calibration.high_threshold)
    detected = delays >= 0
    exp3[kind] = {"low_detection_rate": float(np.mean(alarms.sum(axis=1) > args.k_low)),
        "high_detection_rate": float(np.mean(conformal_high_level_flags(
            alarms, args.window, args.k_high, calibration.high_threshold))),
        "mean_alarm_rate": float(np.mean(alarms)),
        "median_high_alarm_index": float(np.median(delays[detected])) if detected.any() else None}
save_json("experiment_3", exp3)
names = list(exp3)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(np.arange(len(names)), [exp3[name]["high_detection_rate"] for name in names])
ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, rotation=25, ha="right")
ax.set(ylabel="High-level detection rate", ylim=(0, 1.05))
fig.tight_layout(); fig.savefig(FIGURES / "scenario_detection.pdf"); plt.close(fig)
save_table("scenario_detection", ["Scenario", "Low", "High"],
    [[name.replace("_", r"\_"), f"{v['low_detection_rate']:.3f}",
      f"{v['high_detection_rate']:.3f}"] for name, v in exp3.items()])
print(exp3)
'''),
    nbf.v4.new_markdown_cell("## Experiment 4 — ablation of the k-th maximum"),
    code(r'''
exp4 = {}
for k in ((100, 250, 500, 750, 1000) if PUBLICATION_SCALE else (1, 5, 10, 20)):
    threshold = conformal_threshold(low_level_scores(artifacts["cal_low_distances"], k),
                                    calibration.low_rank)
    nominal_alarms = low_level_alarms(d_nom, threshold)
    replay_alarms = low_level_alarms(scenario_distances["replay"], threshold)
    exp4[str(k)] = {"threshold": float(threshold),
        "nominal_alarm_rate": float(nominal_alarms.mean()),
        "nominal_trajectory_alarm_rate": float(np.mean(nominal_alarms.sum(axis=1) > k)),
        "replay_detection_rate": float(np.mean(replay_alarms.sum(axis=1) > k))}
save_json("experiment_4", exp4)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ks = [int(k) for k in exp4]
ax.plot(ks, [exp4[str(k)]["replay_detection_rate"] for k in ks], "o-", label="Replay")
ax.plot(ks, [exp4[str(k)]["nominal_trajectory_alarm_rate"] for k in ks], "s-", label="Nominal")
ax.set(xlabel="k", ylabel="Trajectory decision rate", ylim=(0, 1.05))
ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "k_ablation.pdf"); plt.close(fig)
save_table("k_ablation", ["$k$", "Certificate", "Nominal alarms", "Replay detection"],
    [[k, f"{v['threshold']:.3f}", f"{v['nominal_alarm_rate']:.2e}",
      f"{v['replay_detection_rate']:.3f}"] for k, v in exp4.items()])
print(exp4)
'''),
    nbf.v4.new_markdown_cell("## Experiment 5 — Butterworth filtering"),
    code(r'''
exp5 = {}
for kind, data in [("nominal", nominal_y)] + list(scenario_y.items()):
    filtered_d = d_nom if kind == "nominal" else scenario_distances[kind]
    raw = aligned_unfiltered_residuals(data, estimator, args)
    raw_d = quadratic_distances(raw, artifacts["raw_center"], artifacts["raw_q_inv"])
    filtered_a = low_level_alarms(filtered_d, calibration.low_threshold)
    raw_a = low_level_alarms(raw_d, artifacts["raw_low_threshold"])
    exp5[kind] = {"filtered_alarm_rate": float(filtered_a.mean()),
        "raw_alarm_rate": float(raw_a.mean()),
        "filtered_trajectory_rate": float(np.mean(filtered_a.sum(axis=1) > args.k_low)),
        "raw_trajectory_rate": float(np.mean(raw_a.sum(axis=1) > args.k_low))}
save_json("experiment_5", exp5)
names = list(exp5)
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(x - .18, [exp5[name]["filtered_trajectory_rate"] for name in names],
       width=.36, label="Filtered")
ax.bar(x + .18, [exp5[name]["raw_trajectory_rate"] for name in names],
       width=.36, label="Unfiltered")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
ax.set(ylabel="Low-level trajectory decision rate", ylim=(0, 1.05))
ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "filtered_vs_unfiltered.pdf"); plt.close(fig)
save_table("filter_comparison", ["Scenario", "Filtered", "Unfiltered"],
    [[name.replace("_", r"\_"), f"{v['filtered_trajectory_rate']:.3f}",
      f"{v['raw_trajectory_rate']:.3f}"] for name, v in exp5.items()])
print(exp5)
'''),
    nbf.v4.new_markdown_cell("## Experiment 6 — window-position hierarchical ensemble"),
    code(r'''
ensemble_config = default_config()
ensemble_config.update(k_low=5, k_high=2, delta1_low=0.9,
                       delta1_high=0.9, m_low=200, m_high=200, n_test=80)
if PUBLICATION_SCALE:
    ensemble_config = apply_ensemble(ensemble_config)
exp6 = run_randomized_window_five_batches(ensemble_config, seed=17)
save_json("experiment_6", exp6)
names = list(exp6["scenarios"])
fig, ax = plt.subplots(figsize=(8, 3.5))
x = np.arange(len(names))
ax.bar(x - .18, [exp6["scenarios"][name]["low_detection_rate"] for name in names],
       width=.36, label="Low-level")
ax.bar(x + .18, [exp6["scenarios"][name]["high_detection_rate"] for name in names],
       width=.36, label="High-level")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
ax.set(ylabel="Trajectory detection rate", ylim=(0, 1.05))
ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "window_ensemble.pdf"); plt.close(fig)
save_table("window_ensemble", ["Scenario", "Low-level", "High-level", "Mean alarm rate"],
    [[name.replace("_", r"\_"),
      f"{value['low_detection_rate']:.3f}",
      f"{value['high_detection_rate']:.3f}",
      f"{value['mean_alarm_rate']:.2e}"]
     for name, value in exp6["scenarios"].items()])
print("Nominal randomized rates:", exp6["nominal"]["low_detection_rate"],
      exp6["nominal"]["high_detection_rate"])
print("Replay randomized rates:", exp6["scenarios"]["replay"]["low_detection_rate"],
      exp6["scenarios"]["replay"]["high_detection_rate"])
'''),
    nbf.v4.new_markdown_cell("## Experiment 7 — linear and nonlinear applications"),
    code(r'''
app_args = Namespace(seed=17, steps=300, n_train=60, m_calibration=200,
    n_validation=80, coverage=0.95, delta1=0.9, ridge=1e-6,
    history=4, hidden=32, filter_order=3, filter_cutoff=0.08,
    filter_warmup=9, publication_scale=False)
if PUBLICATION_SCALE:
    app_args = apply_application(app_args)
if PUBLICATION_SCALE:
    exp7, application_dist = run_applications_five_batches(app_args, seed=17)
    linear = exp7["linear_residual"]
    nonlinear = exp7["nonlinear_filtered_residual"]
    linear_dist = application_dist["linear_residual"]
    nonlinear_dist = application_dist["nonlinear_filtered_residual"]
else:
    linear, linear_dist = run_linear_residual_case(app_args, np.random.default_rng(18))
    nonlinear, nonlinear_dist = run_nonlinear_filtered_residual_case(
        app_args, np.random.default_rng(19))
exp7 = {"linear_residual": linear, "nonlinear_filtered_residual": nonlinear}
save_json("experiment_7", exp7)
write_application_figure(exp7, {"linear_residual": linear_dist,
                                "nonlinear_filtered_residual": nonlinear_dist},
                         FIGURES / "application_linear_nonlinear.pdf")
save_table("application_summary", ["Application", "Conformal trajectory alarms", "Other measure"],
    [["Linear", f"{linear['conformal']['trajectory_alarm_rate']:.3f}",
      f"Chi-square mean violations: {linear['chi_square_baseline']['mean_violations_per_trajectory']:.2f}"],
     ["Nonlinear", f"{nonlinear['trajectory_alarm_rate']:.3f}",
      f"Filter energy ratio: {nonlinear['filter_energy_ratio']:.3f}"]])
print("Linear conformal trajectory alarms:", linear["conformal"]["trajectory_alarm_rate"])
print("Nonlinear conformal trajectory alarms:", nonlinear["trajectory_alarm_rate"])
'''),
    nbf.v4.new_markdown_cell("""## Experiment 8 — CoDiT GAIT case study

This cell launches the downloaded upstream CoDiT evaluator and the proposed
detector on the same machine, native GAIT records, and CoDiT subject split.
Follow `benchmarks/experiment_8/README.md` to obtain CoDiT's source and released
checkpoints and configure its isolated Python environment. This cell downloads
and extracts the GAIT archive on first use and skips that step when all
required records are already present. The comparison is descriptive because
it uses repeated windows from a small number of subjects."""),
    code(r'''
from benchmarks.run_experiment_8 import (
    DEFAULT_CODIT, ensure_gait_dataset, run_experiment_8)

codit_root = DEFAULT_CODIT
data_root = (codit_root / "ours" / "gait" / "data" /
             "gait-in-neurodegenerative-disease-database-1.0.0")
checkpoint_dir = codit_root / "ours" / "gait" / "saved_models"
codit_python = "C:/path/to/conda/envs/gait/python.exe"  # Set to CoDiT's Python

downloaded = ensure_gait_dataset(data_root)
print("Downloaded GAIT data." if downloaded else "Using existing GAIT data.")

experiment_8 = run_experiment_8(
    data_root=data_root, codit_root=codit_root,
    checkpoint_dir=checkpoint_dir,
    output_dir=ROOT / "benchmarks" / "outputs" / "experiment_8",
    codit_python=codit_python, n_randomizations=100, seed=1729,
)
with (ROOT / "results" / "section_ix" / "experiment_8.json").open(encoding="utf-8") as stream:
    experiment_8["codit_reference"] = json.load(stream)["codit_reference"]
save_json("experiment_8", experiment_8)
print(json.dumps(experiment_8, indent=2))
'''),
    nbf.v4.new_markdown_cell("## Saved outputs"),
    code(r'''
manifest = {"mode": "publication_design" if PUBLICATION_SCALE else "pilot",
    "seed_baseline": args.seed, "settings": vars(args),
    "ensemble_settings": ensemble_config, "application_settings": vars(app_args),
    "experiment_results": [(RESULTS / f"experiment_{j}.json").relative_to(ROOT).as_posix()
                           for j in range(1, 9)],
    "experiment_status": {"8": experiment_8["status"]},
    "figures": [path.relative_to(ROOT).as_posix() for path in sorted(FIGURES.glob("*.pdf"))],
    "tables": [path.relative_to(ROOT).as_posix() for path in sorted(TABLES.glob("*.tex"))]}
save_json("manifest", manifest)
print("Saved", len(manifest["experiment_results"]), "experiment results,",
      len(manifest["figures"]), "figures, and", len(manifest["tables"]), "tables")
'''),
]

NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
for publication_scale, name in ((True, "section_ix_experiments.ipynb"),
                                (False, "section_ix_pilot.ipynb")):
    selected = [cell.copy() for cell in cells]
    for cell in selected:
        cell.source = cell.source.replace("__PUBLICATION_SCALE__", str(publication_scale))
    selected[0].source += ("\n\n**Profile:** publication design. Numerical outputs must be "
                           "generated before the paper quotes them." if publication_scale else
                           "\n\n**Profile:** small diagnostic pilot; outputs are not paper results.")
    notebook = nbf.v4.new_notebook(cells=selected)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3.8"}
    path = NOTEBOOK_DIR / name
    nbf.write(notebook, path)
    print(path)
