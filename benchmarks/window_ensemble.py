"""Window-position neural ensemble with separate hierarchical detectors.

Each predictor is trained on all available nominal windows (or a reproducible
subsample), calibrated on complete trajectories, and evaluated on disjoint
nominal and attacked trajectories. The same sensor mixing matrix is used for
all nominal splits.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hierarchical_conformal import RandomFeatureObserver, WindowEnsembleDetector
from benchmarks.simulate import attack_measurements, make_mixing_matrix, nominal_deltas
from benchmarks.publication_design import apply_ensemble


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("bias", "drift", "replay", "noise_burst", "surge",
             "clustered", "zero_alarm", "hidden_rate")


def measurements(n, steps, dim, rng, mixing):
    return np.cumsum(nominal_deltas(n, steps, dim, rng, mixing=mixing), axis=1)


def fit_ensemble(config, rng):
    """Train W observers and calibrate low/high scores on disjoint splits."""
    window = config["window"]
    warmup = config["warmup"]
    steps = config["steps"]
    score_steps = steps - window + 1 - warmup
    if score_steps < max(config["k_low"], window + config["k_high"] - 1):
        raise ValueError("trajectory is too short for the requested ranks/windows")
    mixing = make_mixing_matrix(config["dim"], rng)
    train = measurements(config["n_train"], steps, config["dim"], rng, mixing)
    cal_low = measurements(config["m_low"], steps, config["dim"], rng, mixing)
    cal_high = measurements(config["m_high"], steps, config["dim"], rng, mixing)
    model = WindowEnsembleDetector(
        window=window, k_low=config["k_low"], rate_window=window,
        k_high=config["k_high"], delta1_low=config["delta1_low"],
        delta1_high=config["delta1_high"],
        ell_low=config.get("ell_low"), ell_high=config.get("ell_high"),
        ridge=config["ridge"],
        filter_cutoff=config["filter_cutoff"], filter_order=config["filter_order"],
        warmup=warmup, max_training_pairs=config["max_training_pairs"], rng=rng,
        observer_factory=lambda position: RandomFeatureObserver(
            hidden=config["hidden"], ridge=config["ridge"], rng=rng))
    model.fit(train, cal_low, cal_high)
    branches = []
    for position, (observer, low, high) in enumerate(model.branches_):
        branches.append({
            "position": position, "estimator": observer, "center": low.center_,
            "shape": low.shape_, "inverse": low.inverse_,
            "low_threshold": low.threshold_, "high_threshold": high.threshold_,
            "low_rank": low.rank_, "high_rank": high.rank_,
            "low_scores": low.calibration_scores_,
            "high_scores": high.calibration_scores_,
        })
    return {"model": model, "branches": branches, "mixing": mixing,
            "score_steps": score_steps}


def score_ensemble(measurements_array, fitted, config):
    """Evaluate every branch on complete trajectories and vote at both levels."""
    return fitted["model"].score(measurements_array)


def summarize_scored(scored, config, attacked=False):
    result = {
        "branch_low_rate": scored["branch_low"].mean(axis=1).tolist(),
        "branch_high_rate": scored["branch_high"].mean(axis=1).tolist(),
        "low_vote_rate": float(scored["low_vote"].mean()),
        "high_vote_rate": float(scored["high_vote"].mean()),
        "mean_branch_alarm_rate": scored["branch_alarm_rates"].mean(axis=1).tolist(),
        "past_to_next_low_rate": float(scored["branch_low"][-1].mean()),
        "past_to_next_high_rate": float(scored["branch_high"][-1].mean()),
    }
    if attacked:
        onset = int(0.45 * config["steps"])
        offset = config["window"] - 1 + config["warmup"]
        for level in ("low", "high"):
            delay = scored[level + "_delay"]
            absolute_time = delay + offset
            detected_after_onset = (delay >= 0) & (absolute_time >= onset)
            elapsed = absolute_time[detected_after_onset] - onset
            result[level + "_median_delay_after_onset"] = (
                float(np.median(elapsed)) if len(elapsed) else None)
    return result


def run_ensemble_experiment(config, seed=17):
    rng = np.random.default_rng(seed)
    fitted = fit_ensemble(config, rng)
    nominal = measurements(config["n_test"], config["steps"], config["dim"], rng, fitted["mixing"])
    summary = {"seed": seed, "config": config,
               "nominal": summarize_scored(score_ensemble(nominal, fitted, config), config),
               "scenarios": {}, "calibration": []}
    for branch in fitted["branches"]:
        summary["calibration"].append({
            "position": branch["position"], "low_threshold": float(branch["low_threshold"]),
            "high_threshold": float(branch["high_threshold"]),
            "low_rank": branch["low_rank"], "high_rank": branch["high_rank"],
        })
    for kind in SCENARIOS:
        attacked = attack_measurements(nominal, kind, rng)
        summary["scenarios"][kind] = summarize_scored(
            score_ensemble(attacked, fitted, config), config, attacked=True)
    return summary


def save_ensemble_results(summary, root=ROOT, results_dir=None,
                          figures_dir=None, tables_dir=None):
    root = Path(root)
    if results_dir is None or figures_dir is None or tables_dir is None:
        if (root / "paper" / "main.tex").is_file():
            results_dir = root / "results"
            figures_dir = root / "paper" / "figures"
            tables_dir = root / "paper" / "tables"
        else:
            output_dir = root / "benchmarks" / "outputs"
            results_dir = output_dir / "results"
            figures_dir = output_dir / "figures"
            tables_dir = output_dir / "tables"
    results_dir, figures_dir, tables_dir = map(Path, (results_dir, figures_dir, tables_dir))
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (results_dir / "window_ensemble.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    names = list(summary["scenarios"])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.18, [summary["scenarios"][name]["low_vote_rate"] for name in names],
           width=0.36, label="Low-level vote")
    ax.bar(x + 0.18, [summary["scenarios"][name]["high_vote_rate"] for name in names],
           width=0.36, label="High-level vote")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Detection rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "window_ensemble.pdf")
    plt.close(fig)
    rows = [
        "{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
            name.replace("_", "\\_"), item["low_vote_rate"], item["high_vote_rate"],
            item["past_to_next_low_rate"], item["past_to_next_high_rate"])
        for name, item in summary["scenarios"].items()
    ]
    table = "\n".join([
        "\\begin{tabular}{lcccc}", "\\toprule",
        "Scenario & Low vote & High vote & Last-position low & Last-position high \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "",
    ])
    (tables_dir / "window_ensemble.tex").write_text(table, encoding="utf-8")


def default_config():
    return {
        "steps": 300, "dim": 3, "window": 5, "n_train": 60,
        "m_low": 120, "m_high": 120, "n_test": 80,
        "k_low": 15, "k_high": 3, "hidden": 32, "ridge": 1e-5,
        "filter_cutoff": 0.08, "filter_order": 3, "warmup": 9,
        "delta1_low": 0.95, "delta1_high": 0.95,
        "max_training_pairs": 25000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=60)
    parser.add_argument("--m-low", type=int, default=120)
    parser.add_argument("--m-high", type=int, default=120)
    parser.add_argument("--n-test", type=int, default=80)
    parser.add_argument("--publication-scale", action="store_true")
    args = parser.parse_args()
    config = default_config()
    for key in ("steps", "window", "n_train", "m_low", "m_high", "n_test"):
        config[key] = getattr(args, key)
    if args.publication_scale:
        config = apply_ensemble(config)
    summary = run_ensemble_experiment(config, args.seed)
    save_ensemble_results(summary)
    print(json.dumps({"nominal": summary["nominal"],
                      "scenarios": summary["scenarios"]}, indent=2))


if __name__ == "__main__":
    main()
