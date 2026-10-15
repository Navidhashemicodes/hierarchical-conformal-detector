"""Linear residual and nonlinear filtered-residual application tests.

The linear application follows the residual interface used in the supplied
linear notebook and in the conference-version low-level detector. Because the
linear model and residual covariance can be used, this test compares a
chi-square low-level detector with a conformal residual detector.

The nonlinear application matches the journal-extension black-box setting: the
detector uses recorded measurements, trains a one-step neural observer, filters
the residual y_t - y_obs,t with a Butterworth low-pass filter, and then feeds
the filtered residual score to the conformal detectors.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

try:
    import matplotlib.pyplot as plt
except Exception:  # Keep metric generation usable without plotting dependencies.
    plt = None

from hierarchical_conformal import (
    LowLevelDetector,
    beta_confidence,
    butterworth_lowpass,
    ensure_dir,
    kth_max,
    residual_trajectories,
    train_random_relu_estimator,
    trim_filter_warmup,
)
from benchmarks.output_paths import ROOT, RESULTS, FIGURES as PAPER_FIGURES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--n-train", type=int, default=250)
    parser.add_argument("--m-calibration", type=int, default=1000)
    parser.add_argument("--n-validation", type=int, default=400)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--delta1", type=float, default=0.995)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--filter-order", type=int, default=3)
    parser.add_argument("--filter-cutoff", type=float, default=0.08)
    parser.add_argument(
        "--filter-warmup",
        type=int,
        default=-1,
        help="filtered samples to discard; -1 uses 3*filter_order",
    )
    parser.add_argument("--publication-scale", action="store_true")
    return parser.parse_args()


def apply_publication_scale(args):
    if args.publication_scale:
        args.steps = 10000
        args.n_train = max(args.n_train, 500)
        args.m_calibration = max(args.m_calibration, 24100)
        args.n_validation = max(args.n_validation, 1000)
        args.coverage = 0.95
        args.delta1 = 0.9995
    return args


def filter_warmup(args):
    if args.filter_warmup >= 0:
        return args.filter_warmup
    return max(0, 3 * args.filter_order)


def validate_args(args):
    if args.history < 1 or args.history >= args.steps:
        raise ValueError("--history must satisfy 1 <= history < steps")
    if not 0.0 < args.filter_cutoff < 1.0:
        raise ValueError("--filter-cutoff must be normalized to Nyquist and lie in (0, 1)")
    score_steps = args.steps - args.history - filter_warmup(args)
    if score_steps <= 0:
        raise ValueError("history/filter warmup leave no valid score samples")


def sample_ball(rng, n, dim, radius):
    directions = rng.normal(size=(n, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    radii = radius * rng.random(n) ** (1.0 / dim)
    return directions * radii[:, None]


def gaussian_noise(rng, n, cov):
    return rng.normal(size=(n, cov.shape[0])) @ np.linalg.cholesky(cov).T


def mixture_noise(rng, n, cov, outlier_probability=0.03, outlier_scale=5.0):
    z = gaussian_noise(rng, n, cov)
    outliers = rng.random(n) < outlier_probability
    if np.any(outliers):
        z[outliers] *= outlier_scale
    return z


def linear_residuals(n, steps, rng):
    """Generate observer residuals for the model-aware linear comparison."""
    dt = 0.1
    f_mat = np.array([[1.0, dt], [-0.8 * dt, 1.0 - 0.8 * dt]])
    g_mat = np.array([[0.0], [dt]])
    c_mat = np.eye(2)
    k_ctrl = np.array([[-1.0, -1.5]])
    l_obs = np.array([[0.35, 0.02], [0.01, 0.32]])
    q_proc = np.diag([0.02, 0.02])
    r_meas = np.diag([0.05, 0.05])

    x = sample_ball(rng, n, 2, 0.5)
    xhat = np.zeros_like(x)
    residuals = np.empty((n, steps, 2), dtype=float)

    for t in range(steps):
        u = xhat @ k_ctrl.T
        w = gaussian_noise(rng, n, q_proc)
        v = gaussian_noise(rng, n, r_meas)
        x = x @ f_mat.T + u @ g_mat.T + w
        y = x @ c_mat.T + v
        residual = y - xhat @ c_mat.T
        residuals[:, t, :] = residual
        xhat = xhat @ f_mat.T + u @ g_mat.T + residual @ l_obs.T
    return residuals


def nonlinear_step(x, u, w):
    dt = 0.1
    out = np.empty_like(x)
    out[:, 0] = x[:, 0] + dt * x[:, 1] + w[:, 0]
    out[:, 1] = x[:, 1] + dt * (-0.8 * np.sin(x[:, 0]) - 0.25 * x[:, 1] + u[:, 0]) + w[:, 1]
    return out


def nonlinear_output(x, v):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] + 0.15 * x[:, 0] ** 2
    y[:, 1] = x[:, 1]
    return y + v


def nonlinear_measurements(n, steps, rng):
    """Generate measurements for the black-box nonlinear observer test."""
    k_ctrl = np.array([[1.0, 1.5]])
    q_proc = np.diag([0.02, 0.02])
    r_meas = np.diag([0.05, 0.05])

    x = sample_ball(rng, n, 2, 0.5)
    measurements = np.empty((n, steps, 2), dtype=float)
    for t in range(steps):
        u = -(x @ k_ctrl.T)
        w = mixture_noise(rng, n, q_proc, outlier_probability=0.02, outlier_scale=5.0)
        v = mixture_noise(rng, n, r_meas, outlier_probability=0.03, outlier_scale=5.0)
        x = nonlinear_step(x, u, w)
        measurements[:, t, :] = nonlinear_output(x, v)
    return measurements


def quadratic_distances(samples, center, q_inv):
    centered = samples - center.reshape(1, 1, -1)
    return np.einsum("...i,ij,...j->...", centered, q_inv, centered)


def summarize_validation(distances, threshold, allowed, score_steps):
    violations_per_traj = np.sum(distances > threshold, axis=1)
    scores = kth_max(distances, allowed, axis=1)
    alarmed = violations_per_traj > allowed
    return {
        "empirical_coverage": float(np.mean(distances <= threshold)),
        "empirical_violation_rate": float(np.mean(distances > threshold)),
        "trajectory_alarm_rate": float(np.mean(alarmed)),
        "mean_violations_per_trajectory": float(np.mean(violations_per_traj)),
        "p95_violations_per_trajectory": float(np.quantile(violations_per_traj, 0.95)),
    }, violations_per_traj


def run_linear_residual_case(args, rng):
    train = linear_residuals(args.n_train, args.steps, rng)
    score_steps = args.steps
    allowed = max(1, int(np.floor((1.0 - args.coverage) * score_steps)))

    cal = linear_residuals(args.m_calibration, args.steps, rng)
    model = LowLevelDetector(allowed, args.delta1, ridge=args.ridge).fit(train, cal)
    center, q, q_inv = model.center_, model.shape_, model.inverse_
    ell, conf_threshold = model.rank_, model.threshold_

    val = linear_residuals(args.n_validation, args.steps, rng)
    val_dist = quadratic_distances(val, center, q_inv)
    conf_metrics, conf_violations = summarize_validation(val_dist, conf_threshold, allowed, score_steps)

    # Conference-style chi-square low-level baseline: pointwise residual test.
    chi_threshold = float(chi2.ppf(args.coverage, df=val.shape[-1]))
    chi_violations = np.sum(val_dist > chi_threshold, axis=1)
    chi_metrics = {
        "threshold": chi_threshold,
        "empirical_coverage": float(np.mean(val_dist <= chi_threshold)),
        "empirical_violation_rate": float(np.mean(val_dist > chi_threshold)),
        "mean_violations_per_trajectory": float(np.mean(chi_violations)),
        "p95_violations_per_trajectory": float(np.quantile(chi_violations, 0.95)),
    }

    metrics = {
        "distance_source": "linear observer residual, as in the notebook and conference low-level detector",
        "score": "r_t^T Q^{-1} r_t",
        "steps": args.steps,
        "score_steps": score_steps,
        "n_train": args.n_train,
        "m_calibration": args.m_calibration,
        "n_validation": args.n_validation,
        "coverage_target": args.coverage,
        "allowed_violations_per_trajectory": allowed,
        "allowed_violation_rate": allowed / float(score_steps),
        "rank": ell,
        "delta1": args.delta1,
        "delta2": beta_confidence(args.delta1, ell, args.m_calibration),
        "conformal_threshold": float(conf_threshold),
        "conformal": conf_metrics,
        "chi_square_baseline": chi_metrics,
        "residual_mean": center.tolist(),
        "q": q.tolist(),
    }
    return metrics, {"conformal": conf_violations, "chi_square": chi_violations}


def run_nonlinear_filtered_residual_case(args, rng):
    train_y = nonlinear_measurements(args.n_train, args.steps, rng)
    estimator = train_random_relu_estimator(
        train_y,
        history=args.history,
        hidden=args.hidden,
        ridge=args.ridge,
        rng=rng,
    )
    train_res = residual_trajectories(train_y, estimator, args.history)
    train_filtered = trim_filter_warmup(
        butterworth_lowpass(train_res, args.filter_cutoff, args.filter_order),
        filter_warmup(args),
    )
    score_steps = train_filtered.shape[1]
    allowed = max(1, int(np.floor((1.0 - args.coverage) * score_steps)))

    cal_y = nonlinear_measurements(args.m_calibration, args.steps, rng)
    cal_res = residual_trajectories(cal_y, estimator, args.history)
    cal_filtered = trim_filter_warmup(
        butterworth_lowpass(cal_res, args.filter_cutoff, args.filter_order),
        filter_warmup(args),
    )
    model = LowLevelDetector(allowed, args.delta1, ridge=args.ridge).fit(
        train_filtered, cal_filtered)
    center, q, q_inv = model.center_, model.shape_, model.inverse_
    ell, threshold = model.rank_, model.threshold_

    val_y = nonlinear_measurements(args.n_validation, args.steps, rng)
    val_res = residual_trajectories(val_y, estimator, args.history)
    val_filtered = trim_filter_warmup(
        butterworth_lowpass(val_res, args.filter_cutoff, args.filter_order),
        filter_warmup(args),
    )
    val_dist = quadratic_distances(val_filtered, center, q_inv)
    metrics_base, violations = summarize_validation(val_dist, threshold, allowed, score_steps)

    val_res_aligned = trim_filter_warmup(val_res, filter_warmup(args))
    raw_res_energy = float(np.mean(np.sum(val_res_aligned**2, axis=2)))
    filtered_res_energy = float(np.mean(np.sum(val_filtered**2, axis=2)))
    metrics = {
        "distance_source": "Butterworth-filtered neural-observer residual y_t - y_obs_t",
        "score": "(filtered_residual_t-mean)^T Q^{-1} (filtered_residual_t-mean)",
        "steps": args.steps,
        "score_steps": score_steps,
        "history": args.history,
        "filter_warmup": filter_warmup(args),
        "hidden": args.hidden,
        "filter_order": args.filter_order,
        "filter_cutoff": args.filter_cutoff,
        "n_train": args.n_train,
        "m_calibration": args.m_calibration,
        "n_validation": args.n_validation,
        "coverage_target": args.coverage,
        "allowed_violations_per_trajectory": allowed,
        "allowed_violation_rate": allowed / float(score_steps),
        "rank": ell,
        "epsilon": 1.0 - args.delta1,
        "ell": ell,
        "m": args.m_calibration,
        "delta1": args.delta1,
        "delta2": beta_confidence(args.delta1, ell, args.m_calibration),
        "threshold": float(threshold),
        "raw_residual_energy": raw_res_energy,
        "filtered_residual_energy": filtered_res_energy,
        "filter_energy_ratio": filtered_res_energy / max(raw_res_energy, 1e-12),
        **metrics_base,
        "filtered_residual_mean": center.tolist(),
        "q": q.tolist(),
    }
    return metrics, violations


def write_application_figure(results, distributions):
    if plt is None:
        return
    ensure_dir(PAPER_FIGURES)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    linear = results["linear_residual"]
    bins = 30
    axes[0].hist(distributions["linear_residual"]["chi_square"], bins=bins, alpha=0.65, label="chi-square")
    axes[0].hist(distributions["linear_residual"]["conformal"], bins=bins, alpha=0.65, label="conformal")
    axes[0].axvline(linear["allowed_violations_per_trajectory"], color="crimson", linewidth=2)
    axes[0].set_title("Linear residual")
    axes[0].set_xlabel("violations per trajectory")
    axes[0].set_ylabel("validation trajectories")
    axes[0].legend()

    nonlinear = results["nonlinear_filtered_residual"]
    axes[1].hist(distributions["nonlinear_filtered_residual"], bins=bins, color="#4c78a8", alpha=0.8)
    axes[1].axvline(nonlinear["allowed_violations_per_trajectory"], color="crimson", linewidth=2)
    axes[1].set_title("Nonlinear filtered residual")
    axes[1].set_xlabel("violations per trajectory")
    axes[1].set_ylabel("validation trajectories")

    fig.tight_layout()
    fig.savefig(PAPER_FIGURES / "application_linear_nonlinear.pdf")
    plt.close(fig)


def main():
    args = apply_publication_scale(parse_args())
    validate_args(args)
    ensure_dir(RESULTS)
    rng = np.random.default_rng(args.seed)

    linear_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
    nonlinear_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
    linear_metrics, linear_dist = run_linear_residual_case(args, linear_rng)
    nonlinear_metrics, nonlinear_dist = run_nonlinear_filtered_residual_case(args, nonlinear_rng)

    results = {
        "linear_residual": linear_metrics,
        "nonlinear_filtered_residual": nonlinear_metrics,
    }
    distributions = {
        "linear_residual": linear_dist,
        "nonlinear_filtered_residual": nonlinear_dist,
    }

    (RESULTS / "application_system_tests.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_application_figure(results, distributions)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
