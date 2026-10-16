"""Run all numerical examples for the journal-extension draft.

The default settings are intentionally moderate so the script can be tested on
a laptop. Use --publication-scale to switch to T=10000 and larger calibration
sets. The main synthetic benchmark follows the journal pipeline:
recorded measurements -> neural observer output -> residual y-y_obs ->
Butterworth low-pass filter -> low-level and high-level conformal detectors.
"""

import argparse
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import numpy as np

from hierarchical_conformal import (
    CalibrationSummary,
    HighLevelDetector,
    LowLevelDetector,
    beta_confidence,
    butterworth_lowpass,
    conformal_high_level_flags,
    conformal_high_level_delay,
    conformal_rank,
    conformal_threshold,
    ensure_dir,
    high_level_scores,
    kth_max,
    low_level_alarms,
    low_level_scores,
    residual_trajectories,
    sliding_window_rates,
    summarize_delays,
    train_random_relu_estimator,
    trim_filter_warmup,
)
from benchmarks.simulate import attack_measurements, make_mixing_matrix, nominal_deltas
from benchmarks.output_paths import ROOT, RESULTS, FIGURES as PAPER_FIGURES, TABLES as PAPER_TABLES
from benchmarks.publication_design import apply_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--m-low", type=int, default=600)
    parser.add_argument("--m-high", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--k-low", type=int, default=100)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--k-high", type=int, default=20)
    parser.add_argument("--delta1-low", type=float, default=0.995)
    parser.add_argument("--delta1-high", type=float, default=0.995)
    parser.add_argument("--ell-low", type=int, default=0,
                        help="low-level calibration rank; 0 uses the marginal conformal rank")
    parser.add_argument("--ell-high", type=int, default=0,
                        help="high-level calibration rank; 0 uses the marginal conformal rank")
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--filter-order", type=int, default=3)
    parser.add_argument("--filter-cutoff", type=float, default=0.08)
    parser.add_argument(
        "--filter-warmup",
        type=int,
        default=-1,
        help="filtered samples to discard; -1 uses 3*filter_order",
    )
    parser.add_argument(
        "--k-ablation-values",
        default="1,25,50,100,200,500",
        help="comma-separated low-level k values for the k-ablation table",
    )
    parser.add_argument("--publication-scale", action="store_true")
    return parser.parse_args()


def apply_publication_scale(args: argparse.Namespace) -> argparse.Namespace:
    if args.publication_scale:
        args = apply_baseline(args)
    return args


def parse_int_list(text: str):
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def filter_warmup(args: argparse.Namespace) -> int:
    if args.filter_warmup >= 0:
        return args.filter_warmup
    return max(0, 3 * args.filter_order)


def validate_args(args: argparse.Namespace) -> None:
    if args.history < 1 or args.history >= args.steps:
        raise ValueError("--history must satisfy 1 <= history < steps")
    if not 0.0 < args.filter_cutoff < 1.0:
        raise ValueError("--filter-cutoff must be normalized to Nyquist and lie in (0, 1)")
    score_steps = args.steps - args.history - filter_warmup(args)
    if score_steps <= 0:
        raise ValueError("history/filter warmup leave no valid score samples")
    if args.k_low < 1 or args.k_low > score_steps:
        raise ValueError("--k-low must fit within the valid filtered score length")
    if args.window < 1 or args.window > score_steps:
        raise ValueError("--window must fit within the valid filtered score length")
    num_windows = score_steps - args.window + 1
    if args.k_high < 1 or args.k_high > num_windows:
        raise ValueError("--k-high must fit within the number of high-level windows")
    if args.ell_low and not 1 <= args.ell_low <= args.m_low:
        raise ValueError("--ell-low must be between 1 and m-low")
    if args.ell_high and not 1 <= args.ell_high <= args.m_high:
        raise ValueError("--ell-high must be between 1 and m-high")


def measurements_from_deltas(deltas: np.ndarray) -> np.ndarray:
    """Recover recorded measurements from generated increments."""
    return np.cumsum(deltas, axis=1)


def estimate_shape(samples: np.ndarray, ridge: float):
    flat = samples.reshape(-1, samples.shape[-1])
    center = flat.mean(axis=0)
    centered = flat - center
    q = centered.T @ centered / max(1, len(centered) - 1)
    q += ridge * np.eye(q.shape[0])
    return center, q, np.linalg.pinv(q)


def quadratic_distances(samples: np.ndarray, center: np.ndarray, q_inv: np.ndarray) -> np.ndarray:
    centered = samples - center.reshape(1, 1, -1)
    return np.einsum("...i,ij,...j->...", centered, q_inv, centered)


def observer_residuals(measurements: np.ndarray, estimator, args: argparse.Namespace) -> np.ndarray:
    return residual_trajectories(measurements, estimator, args.history)


def filtered_residuals(measurements: np.ndarray, estimator, args: argparse.Namespace) -> np.ndarray:
    residuals = residual_trajectories(measurements, estimator, args.history)
    filtered = butterworth_lowpass(residuals, args.filter_cutoff, args.filter_order)
    return trim_filter_warmup(filtered, filter_warmup(args))


def aligned_unfiltered_residuals(measurements: np.ndarray, estimator, args: argparse.Namespace) -> np.ndarray:
    residuals = observer_residuals(measurements, estimator, args)
    return trim_filter_warmup(residuals, filter_warmup(args))


def calibrate(args: argparse.Namespace, rng: np.random.Generator):
    mixing = make_mixing_matrix(args.dim, rng)
    train_y = measurements_from_deltas(nominal_deltas(args.n_train, args.steps, args.dim, rng, mixing=mixing))
    estimator = train_random_relu_estimator(
        train_y,
        history=args.history,
        hidden=args.hidden,
        ridge=args.ridge,
        rng=rng,
    )
    train_raw = aligned_unfiltered_residuals(train_y, estimator, args)
    train_filtered = filtered_residuals(train_y, estimator, args)
    cal_low_y = measurements_from_deltas(nominal_deltas(args.m_low, args.steps, args.dim, rng, mixing=mixing))
    cal_low_raw = aligned_unfiltered_residuals(cal_low_y, estimator, args)
    cal_low_filtered = filtered_residuals(cal_low_y, estimator, args)
    low_model = LowLevelDetector(
        args.k_low, args.delta1_low, args.ell_low or None, args.ridge
    ).fit(train_filtered, cal_low_filtered)
    center, q, q_inv = low_model.center_, low_model.shape_, low_model.inverse_
    raw_center, raw_q, raw_q_inv = estimate_shape(train_raw, ridge=args.ridge)
    score_steps = train_filtered.shape[1]
    d_low = low_model.distances(cal_low_filtered)
    scores_low = low_model.calibration_scores_
    ell_low = low_model.rank_
    s_low = low_model.threshold_
    d_low_raw = quadratic_distances(cal_low_raw, raw_center, raw_q_inv)
    scores_low_raw = low_level_scores(d_low_raw, args.k_low)
    s_low_raw = conformal_threshold(scores_low_raw, ell_low)

    cal_high_y = measurements_from_deltas(nominal_deltas(args.m_high, args.steps, args.dim, rng, mixing=mixing))
    cal_high_filtered = filtered_residuals(cal_high_y, estimator, args)
    d_high = quadratic_distances(cal_high_filtered, center, q_inv)
    alarms_high = low_level_alarms(d_high, s_low)
    high_model = HighLevelDetector(
        args.window, args.k_high, args.delta1_high, args.ell_high or None
    ).fit(alarms_high)
    scores_high = high_model.calibration_scores_
    high_rates = sliding_window_rates(alarms_high, args.window)
    scores_high_quiet = kth_max(1.0 - high_rates, args.k_high, axis=1)
    ell_high = high_model.rank_
    s_high = high_model.threshold_
    s_high_quiet = conformal_threshold(scores_high_quiet, ell_high)

    summary = CalibrationSummary(
        low_threshold=s_low,
        low_rank=ell_low,
        low_delta1=args.delta1_low,
        low_delta2=beta_confidence(args.delta1_low, ell_low, args.m_low),
        high_threshold=s_high,
        high_quiet_threshold=s_high_quiet,
        high_rank=ell_high,
        high_delta1=args.delta1_high,
        high_delta2=beta_confidence(args.delta1_high, ell_high, args.m_high),
    )
    artifacts = {
        "low_model": low_model,
        "high_model": high_model,
        "center": center,
        "q": q,
        "q_inv": q_inv,
        "raw_center": raw_center,
        "raw_q": raw_q,
        "raw_q_inv": raw_q_inv,
        "raw_low_threshold": s_low_raw,
        "scores_low": scores_low,
        "scores_low_raw": scores_low_raw,
        "scores_high": scores_high,
        "scores_high_quiet": scores_high_quiet,
        "cal_low_distances": d_low,
        "score_steps": score_steps,
        "mixing": mixing,
    }
    return estimator, summary, artifacts


def quiet_conformal_flags_and_delays(alarms: np.ndarray, args: argparse.Namespace, threshold: float):
    rates = sliding_window_rates(alarms, args.window)
    quiet_exceed = (1.0 - rates) > threshold
    trajectory_flags = kth_max(1.0 - rates, args.k_high, axis=1) > threshold
    cumulative = np.cumsum(quiet_exceed, axis=1)
    online_flags = cumulative >= args.k_high
    has_alarm = online_flags.any(axis=1)
    first_window = np.where(has_alarm, online_flags.argmax(axis=1), -1)
    delays = np.where(has_alarm, first_window + args.window - 1, -1).astype(int)
    return trajectory_flags, delays


def evaluate(
    args: argparse.Namespace,
    estimator,
    summary: CalibrationSummary,
    artifacts,
    rng: np.random.Generator,
):
    center = artifacts["center"]
    q_inv = artifacts["q_inv"]
    raw_center = artifacts["raw_center"]
    raw_q_inv = artifacts["raw_q_inv"]
    score_steps = artifacts["score_steps"]
    nominal_delta = nominal_deltas(args.n_test, args.steps, args.dim, rng, mixing=artifacts["mixing"])
    nominal_y = measurements_from_deltas(nominal_delta)
    nominal_filtered = filtered_residuals(nominal_y, estimator, args)
    d_nom = quadratic_distances(nominal_filtered, center, q_inv)
    alarms_nom = low_level_alarms(d_nom, summary.low_threshold)
    low_scores_nom = low_level_scores(d_nom, args.k_low)
    high_scores_nom = high_level_scores(alarms_nom, args.window, args.k_high)
    conf_nom = conformal_high_level_flags(alarms_nom, args.window, args.k_high, summary.high_threshold)
    conf_quiet_nom, _ = quiet_conformal_flags_and_delays(
        alarms_nom, args, summary.high_quiet_threshold
    )
    nominal_raw = aligned_unfiltered_residuals(nominal_y, estimator, args)
    d_nom_raw = quadratic_distances(nominal_raw, raw_center, raw_q_inv)
    raw_scores_nom = low_level_scores(d_nom_raw, args.k_low)
    raw_alarms_nom = low_level_alarms(d_nom_raw, artifacts["raw_low_threshold"])

    k_ablation = {}
    ablation_y = attack_measurements(nominal_y, kind="clustered", rng=rng)
    ablation_distances = quadratic_distances(
        filtered_residuals(ablation_y, estimator, args), center, q_inv
    )
    k_values = sorted(set(k for k in parse_int_list(args.k_ablation_values) if 1 <= k <= score_steps))
    for k in k_values:
        cal_scores = low_level_scores(artifacts["cal_low_distances"], k)
        ell = args.ell_low or conformal_rank(args.m_low, args.delta1_low)
        threshold = conformal_threshold(cal_scores, ell)
        scores = low_level_scores(d_nom, k)
        alarms = low_level_alarms(d_nom, threshold)
        k_ablation[str(k)] = {
            "threshold": float(threshold),
            "nominal_low_coverage": float(np.mean(scores <= threshold)),
            "mean_alarm_rate": float(alarms.mean()),
            "p95_alarm_rate": float(np.quantile(alarms.mean(axis=1), 0.95)),
            "clustered_low_detection_rate": float(np.mean(
                low_level_alarms(ablation_distances, threshold).sum(axis=1) > k
            )),
        }

    records = {
        "nominal": {
            "score_source": "Butterworth-filtered neural-observer residual y_t-y_obs_t",
            "score_steps": score_steps,
            "filter_warmup": filter_warmup(args),
            "epsilon_low": 1.0 - summary.low_delta1,
            "ell_low": summary.low_rank,
            "m_low": args.m_low,
            "epsilon_high": 1.0 - summary.high_delta1,
            "ell_high": summary.high_rank,
            "m_high": args.m_high,
            "low_coverage": float(np.mean(low_scores_nom <= summary.low_threshold)),
            "mean_alarm_rate": float(alarms_nom.mean()),
            "p95_alarm_rate": float(np.quantile(alarms_nom.mean(axis=1), 0.95)),
            "fraction_within_alarm_budget": float(np.mean(alarms_nom.sum(axis=1) <= args.k_low)),
            "nominal_low_decision_rate": float(np.mean(alarms_nom.sum(axis=1) > args.k_low)),
            "conformal_high_false_alarm": float(conf_nom.mean()),
            "conformal_quiet_high_false_alarm": float(conf_quiet_nom.mean()),
            "conformal_two_sided_high_false_alarm": float(np.mean(conf_nom | conf_quiet_nom)),
            "mean_high_score": float(high_scores_nom.mean()),
        },
        "filtered_vs_unfiltered": {
            "filtered_low_coverage": float(np.mean(low_scores_nom <= summary.low_threshold)),
            "filtered_mean_alarm_rate": float(alarms_nom.mean()),
            "unfiltered_low_coverage": float(np.mean(raw_scores_nom <= artifacts["raw_low_threshold"])),
            "unfiltered_mean_alarm_rate": float(raw_alarms_nom.mean()),
            "unfiltered_low_threshold": float(artifacts["raw_low_threshold"]),
        },
        "k_ablation": k_ablation,
    }

    scenario_records = {}
    scenario_kinds = [
        "bias",
        "drift",
        "replay",
        "noise_burst",
        "surge",
        "clustered",
        "zero_alarm",
        "hidden_rate",
    ]
    for kind in scenario_kinds:
        scenario_y = attack_measurements(nominal_y, kind=kind, rng=rng)
        scenario_filtered = filtered_residuals(scenario_y, estimator, args)
        d_scenario = quadratic_distances(scenario_filtered, center, q_inv)
        alarms_scenario = low_level_alarms(d_scenario, summary.low_threshold)
        scenario_raw = aligned_unfiltered_residuals(scenario_y, estimator, args)
        d_scenario_raw = quadratic_distances(scenario_raw, raw_center, raw_q_inv)
        raw_alarms_scenario = low_level_alarms(d_scenario_raw, artifacts["raw_low_threshold"])
        conf = conformal_high_level_flags(
            alarms_scenario, args.window, args.k_high, summary.high_threshold
        )
        conf_quiet, conf_quiet_delay = quiet_conformal_flags_and_delays(
            alarms_scenario, args, summary.high_quiet_threshold
        )
        conf_two_sided = conf | conf_quiet
        conf_delay = conformal_high_level_delay(
            alarms_scenario, args.window, args.k_high, summary.high_threshold
        )
        scenario_records[kind] = {
            "mean_low_alarm_rate": float(alarms_scenario.mean()),
            "low_detection_rate": float(np.mean(alarms_scenario.sum(axis=1) > args.k_low)),
            "filtered_low_detection_rate": float(np.mean(alarms_scenario.sum(axis=1) > args.k_low)),
            "unfiltered_low_detection_rate": float(np.mean(raw_alarms_scenario.sum(axis=1) > args.k_low)),
            "conformal_detection_rate": float(conf.mean()),
            "conformal_quiet_detection_rate": float(conf_quiet.mean()),
            "conformal_two_sided_detection_rate": float(conf_two_sided.mean()),
            "conformal_delay": summarize_delays(conf_delay),
            "conformal_quiet_delay": summarize_delays(conf_quiet_delay),
            "mean_high_score": float(high_level_scores(alarms_scenario, args.window, args.k_high).mean()),
        }
    records["scenarios"] = scenario_records

    return records, {
        "nominal_alarm_rates": alarms_nom.mean(axis=1),
        "nominal_high_scores": high_scores_nom,
        "raw_nominal_alarm_rates": raw_alarms_nom.mean(axis=1),
    }


def write_figures(args, summary, scores_low, scores_high, eval_arrays, records):
    if plt is None:
        return
    ensure_dir(PAPER_FIGURES)
    ensure_dir(PAPER_TABLES)

    plt.figure(figsize=(7.0, 4.0))
    plt.hist(scores_low, bins=40, alpha=0.75, color="#1f77b4")
    plt.axvline(summary.low_threshold, color="crimson", linewidth=2, label="conformal threshold")
    plt.xlabel("low-level k-th maximum filtered-residual score")
    plt.ylabel("calibration trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PAPER_FIGURES / "low_level_validity.pdf")
    plt.close()

    plt.figure(figsize=(7.0, 4.0))
    plt.hist(scores_high, bins=40, alpha=0.75, color="#2ca02c", label="nominal high-level scores")
    plt.axvline(summary.high_threshold, color="crimson", linewidth=2, label="conformal threshold")
    plt.xlabel("k-th maximum sliding-window alarm rate")
    plt.ylabel("calibration trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PAPER_FIGURES / "high_level_certificates.pdf")
    plt.close()

    names = list(records["scenarios"].keys())
    conf = [records["scenarios"][name]["conformal_detection_rate"] for name in names]
    x = np.arange(len(names))
    plt.figure(figsize=(7.0, 4.0))
    plt.bar(x, conf, width=0.55, label="High-level conformal")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("detection rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PAPER_FIGURES / "scenario_detection.pdf")
    plt.close()

    if "raw_nominal_alarm_rates" in eval_arrays:
        plt.figure(figsize=(7.0, 4.0))
        filtered = [records["scenarios"][name]["filtered_low_detection_rate"] for name in names]
        raw = [records["scenarios"][name]["unfiltered_low_detection_rate"] for name in names]
        plt.bar(x - 0.18, filtered, width=0.36, label="filtered")
        plt.bar(x + 0.18, raw, width=0.36, label="unfiltered")
        plt.xticks(x, names, rotation=20, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel("low-level detection rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PAPER_FIGURES / "filtered_vs_unfiltered.pdf")
        plt.close()

    rows = []
    for k, item in sorted(records["k_ablation"].items(), key=lambda pair: int(pair[0])):
        rows.append(
            f"{k} & {item['threshold']:.4g} & "
            f"{item['mean_alarm_rate']:.4g} & "
            f"{item['clustered_low_detection_rate']:.4g} \\\\"
        )
    table = "\n".join(
        [
            "\\begin{tabular}{cccc}",
            "\\toprule",
            "$k$ & Low-level threshold & Nominal alarm rate & Clustered detection rate \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )
    (PAPER_TABLES / "k_ablation.tex").write_text(table, encoding="utf-8")


def main():
    args = apply_publication_scale(parse_args())
    validate_args(args)
    ensure_dir(RESULTS)
    rng = np.random.default_rng(args.seed)

    if args.publication_scale:
        from benchmarks.streaming_pipeline import calibrate_five_batches
        estimator, summary, artifacts = calibrate_five_batches(args, seed=args.seed)
        rng = np.random.default_rng(args.seed + 999)
    else:
        estimator, summary, artifacts = calibrate(args, rng)
    records, eval_arrays = evaluate(args, estimator, summary, artifacts, rng)
    payload = {
        "args": vars(args),
        "score_pipeline": "y_t -> y_obs,t=f_theta(history) -> e_t=y_t-y_obs,t -> Butterworth low-pass -> conformal low/high detectors",
        "calibration": summary.as_dict(),
        "filtered_residual_mean": artifacts["center"].tolist(),
        "filtered_residual_q": artifacts["q"].tolist(),
        "unfiltered_residual_mean": artifacts["raw_center"].tolist(),
        "unfiltered_residual_q": artifacts["raw_q"].tolist(),
        "metrics": records,
    }
    (RESULTS / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        RESULTS / "diagnostics.npz",
        scores_low=artifacts["scores_low"],
        scores_low_raw=artifacts["scores_low_raw"],
        scores_high=artifacts["scores_high"],
        scores_high_quiet=artifacts["scores_high_quiet"],
        nominal_alarm_rates=eval_arrays["nominal_alarm_rates"],
        raw_nominal_alarm_rates=eval_arrays["raw_nominal_alarm_rates"],
        nominal_high_scores=eval_arrays["nominal_high_scores"],
    )
    write_figures(args, summary, artifacts["scores_low"], artifacts["scores_high"], eval_arrays, records)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
