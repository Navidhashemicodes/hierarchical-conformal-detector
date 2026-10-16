"""Five-batch calibration pipeline for the publication-size baseline."""

import numpy as np
from scipy.signal import lfilter

from hierarchical_conformal import (CalibrationSummary, HighLevelDetector,
    LowLevelDetector,
    beta_confidence, butterworth_lowpass, conformal_high_level_delay,
    conformal_rank, conformal_threshold, high_level_scores, kth_max, sliding_window_rates,
    trim_filter_warmup)
from benchmarks.run_all import filter_warmup as effective_filter_warmup, quadratic_distances
from benchmarks.simulate import make_mixing_matrix
from benchmarks.streaming import (accumulate_shape, fit_random_relu_streaming,
    five_batches, history_pairs_vectorized, observer_residuals_vectorized,
    shape_from_moments, trajectory_slices, window_pairs_vectorized)
from benchmarks.simulate import attack_measurements
from benchmarks.application_system_tests import linear_residuals, nonlinear_measurements
from benchmarks.window_ensemble import SCENARIOS, summarize_scored


def measurement_batches(n, steps, dim, seed, mixing, batches=5):
    """Regenerate five outer batches using bounded trajectory sub-batches.

    Each outer batch is generated in small pieces so a full publication batch
    (4,820 trajectories at T=10,000) is never resident in memory at once.
    """
    if batches != 5:
        raise ValueError("publication data must be processed in exactly five outer batches")
    for index, (start, stop) in enumerate(five_batches(n)):
        rng = np.random.default_rng(seed + index)
        for chunk_start in range(start, stop, 32):
            chunk_stop = min(chunk_start + 32, stop)
            increments = nominal_deltas_vectorized(
                chunk_stop - chunk_start, steps, dim, rng, mixing)
            yield chunk_start, chunk_stop, np.cumsum(increments, axis=1)


def nominal_deltas_vectorized(n, steps, dim, rng, mixing, rho=0.86,
                              burst_probability=0.015):
    """Vectorized equivalent of the benchmark's dependent increment process."""
    initial_state = rng.normal(scale=0.2, size=(n, dim))
    shock = rng.normal(size=(n, steps, dim))
    bursts = rng.random((n, steps)) < burst_probability
    innovations = np.zeros((n, steps), dtype=float)
    count = int(bursts.sum())
    if count:
        innovations[bursts] = rng.uniform(1.5, 3.0, size=count)
    state, _ = lfilter([0.35], [1.0, -rho], shock, axis=1,
                       zi=(rho * initial_state)[:, None, :])
    volatility_increase, _ = lfilter([1.0], [1.0, -0.98], innovations,
                                     axis=1, zi=np.zeros((n, 1)))
    raw = (1.0 + volatility_increase[:, :, None]) * shock + state
    return raw @ np.asarray(mixing).T


def _pair_factory(n, steps, dim, seed, mixing, history):
    def batches():
        for _, _, measurements in measurement_batches(n, steps, dim, seed, mixing):
            for part in trajectory_slices(len(measurements)):
                yield history_pairs_vectorized(measurements[part], history)
    return batches


def _score_vectors(measurements, estimator, args, center, inverse, raw=False):
    residual = observer_residuals_vectorized(measurements, estimator, args.history)
    warmup = effective_filter_warmup(args)
    aligned = trim_filter_warmup(residual, warmup)
    if raw:
        vectors = aligned
    else:
        filtered = butterworth_lowpass(residual, args.filter_cutoff, args.filter_order)
        vectors = trim_filter_warmup(filtered, warmup)
    return quadratic_distances(vectors, center, inverse), vectors


def _training_moments(args, estimator, seed, mixing):
    dim = args.dim
    sum_f = np.zeros(dim); cross_f = np.zeros((dim, dim)); count_f = 0
    sum_r = np.zeros(dim); cross_r = np.zeros((dim, dim)); count_r = 0
    for _, _, measurements in measurement_batches(
            args.n_train, args.steps, args.dim, seed, mixing):
        for part in trajectory_slices(len(measurements)):
            residual = observer_residuals_vectorized(
                measurements[part], estimator, args.history)
            raw = trim_filter_warmup(residual, effective_filter_warmup(args))
            filtered = trim_filter_warmup(
                butterworth_lowpass(residual, args.filter_cutoff, args.filter_order),
                effective_filter_warmup(args))
            sum_f, cross_f, count_f = accumulate_shape(sum_f, cross_f, filtered)
            sum_r, cross_r, count_r = accumulate_shape(sum_r, cross_r, raw)
    return (shape_from_moments(sum_f, cross_f, count_f, args.ridge),
            shape_from_moments(sum_r, cross_r, count_r, args.ridge))


def calibrate_five_batches(args, seed=7):
    """Calibrate baseline levels while retaining only five-batch score outputs."""
    if args.n_train < args.m_low or args.n_train < args.m_high:
        raise ValueError("training trajectory count must be at least each calibration count")
    if args.m_low != 24100 or args.m_high != 24100 or args.steps != 10000:
        raise ValueError("five-batch publication calibration expects T=10000 and m=24100")
    seed_rng = np.random.default_rng(seed)
    mixing = make_mixing_matrix(args.dim, seed_rng)
    train_seed, low_seed, high_seed = [int(x) for x in seed_rng.integers(1, 2**31 - 1, size=3)]
    estimator_rng = np.random.default_rng(int(seed_rng.integers(1, 2**31 - 1)))
    estimator = fit_random_relu_streaming(
        _pair_factory(args.n_train, args.steps, args.dim, train_seed, mixing, args.history),
        args.hidden, args.ridge, estimator_rng)
    (center, q, q_inv), (raw_center, raw_q, raw_q_inv) = _training_moments(
        args, estimator, train_seed, mixing)
    score_steps = args.steps - args.history - effective_filter_warmup(args)

    scores_low = np.empty(args.m_low); scores_low_raw = np.empty(args.m_low)
    calibration_distances = np.empty((args.m_low, score_steps), dtype=np.float64)
    row = 0
    for _, _, measurements in measurement_batches(
            args.m_low, args.steps, args.dim, low_seed, mixing):
        for part in trajectory_slices(len(measurements)):
            d, _ = _score_vectors(measurements[part], estimator, args, center, q_inv)
            d_raw, _ = _score_vectors(measurements[part], estimator, args, raw_center, raw_q_inv, raw=True)
            stop = row + len(d)
            scores_low[row:stop] = kth_max(d, args.k_low, axis=1)
            scores_low_raw[row:stop] = kth_max(d_raw, args.k_low, axis=1)
            calibration_distances[row:stop] = d
            row = stop

    low_model = LowLevelDetector(args.k_low, args.delta1_low,
                                 args.ell_low or None, args.ridge)
    low_model.center_, low_model.shape_, low_model.inverse_ = center, q, q_inv
    low_model.rank_ = args.ell_low
    low_model.calibration_scores_ = scores_low
    low_model.threshold_ = conformal_threshold(scores_low, low_model.rank_)
    low_threshold = low_model.threshold_
    raw_threshold = conformal_threshold(scores_low_raw, low_model.rank_)

    scores_high = np.empty(args.m_high); scores_high_quiet = np.empty(args.m_high)
    row = 0
    for _, _, measurements in measurement_batches(
            args.m_high, args.steps, args.dim, high_seed, mixing):
        for part in trajectory_slices(len(measurements)):
            d, _ = _score_vectors(measurements[part], estimator, args, center, q_inv)
            alarms = d > low_threshold
            rates = sliding_window_rates(alarms, args.window)
            stop = row + len(d)
            scores_high[row:stop] = kth_max(rates, args.k_high, axis=1)
            scores_high_quiet[row:stop] = kth_max(1.0 - rates, args.k_high, axis=1)
            row = stop
    high_model = HighLevelDetector(args.window, args.k_high, args.delta1_high,
                                   args.ell_high or None)
    high_model.rank_ = args.ell_high
    high_model.calibration_scores_ = scores_high
    high_model.threshold_ = conformal_threshold(scores_high, high_model.rank_)
    high_quiet_threshold = conformal_threshold(scores_high_quiet, high_model.rank_)
    summary = CalibrationSummary(
        low_threshold=low_threshold, low_rank=low_model.rank_,
        low_delta1=args.delta1_low,
        low_delta2=beta_confidence(args.delta1_low, low_model.rank_, args.m_low),
        high_threshold=high_model.threshold_, high_quiet_threshold=high_quiet_threshold,
        high_rank=high_model.rank_, high_delta1=args.delta1_high,
        high_delta2=beta_confidence(args.delta1_high, high_model.rank_, args.m_high))
    artifacts = {
        "low_model": low_model, "high_model": high_model,
        "center": center, "q": q, "q_inv": q_inv,
        "raw_center": raw_center, "raw_q": raw_q, "raw_q_inv": raw_q_inv,
        "raw_low_threshold": raw_threshold,
        "scores_low": scores_low, "scores_low_raw": scores_low_raw,
        "scores_high": scores_high, "scores_high_quiet": scores_high_quiet,
        "cal_low_distances": calibration_distances,
        "score_steps": score_steps, "mixing": mixing,
        "train_seed": train_seed, "low_seed": low_seed, "high_seed": high_seed,
    }
    return estimator, summary, artifacts


def _window_pair_factory(config, seed, mixing, position):
    def batches():
        for _, _, measurements in measurement_batches(
                config["n_train"], config["steps"], config["dim"], seed, mixing):
            for part in trajectory_slices(len(measurements)):
                yield window_pairs_vectorized(
                    measurements[part], config["window"], position)
    return batches


def _window_vectors(measurements, observer, config, position):
    features, target = window_pairs_vectorized(
        measurements, config["window"], position)
    residual = (target - observer.predict_flat(features)).reshape(
        len(measurements), measurements.shape[1] - config["window"] + 1,
        measurements.shape[2])
    return trim_filter_warmup(
        butterworth_lowpass(residual, config["filter_cutoff"], config["filter_order"]),
        config["warmup"])


def _window_raw_vectors(measurements, observer, window, position):
    """Return one observer's unfiltered residual sequence, indexed by window."""
    features, target = window_pairs_vectorized(measurements, window, position)
    predicted = observer.predict_flat(features)
    return (target - predicted).reshape(
        len(measurements), measurements.shape[1] - window + 1,
        measurements.shape[2])


def randomized_window_vectors(measurements, observers, config, rng):
    """Draw an observer independently at every window-completion time.

    The observer is selected before filtering, so the causal filter processes
    the randomized residual stream itself rather than a switch among already
    filtered branch streams.
    """
    values = np.asarray(measurements, dtype=float)
    n = len(values)
    steps = values.shape[1] - config["window"] + 1
    choices = rng.integers(0, len(observers), size=(n, steps))
    selected = np.empty((n, steps, values.shape[2]), dtype=float)
    # Materialize one candidate stream at a time to keep memory bounded as W grows.
    for position, observer in enumerate(observers):
        use = choices == position
        if np.any(use):
            candidate = _window_raw_vectors(
                values, observer, config["window"], position)
            selected[use] = candidate[use]
    filtered = butterworth_lowpass(
        selected, config["filter_cutoff"], config["filter_order"])
    return trim_filter_warmup(filtered, config["warmup"]), choices


def _window_training_shape(config, seed, mixing, observer, position):
    dim = config["dim"]
    total = count = 0
    summed = np.zeros(dim); cross = np.zeros((dim, dim))
    for _, _, measurements in measurement_batches(
            config["n_train"], config["steps"], dim, seed, mixing):
        for part in trajectory_slices(len(measurements)):
            vectors = _window_vectors(measurements[part], observer, config, position)
            summed, cross, count = accumulate_shape(summed, cross, vectors)
    center = summed / count
    shape = (cross - count * np.outer(center, center)) / count
    shape += config["ridge"] * np.eye(dim)
    return center, shape, np.linalg.pinv(shape)


def _window_branch_scores(measurements, branches, config):
    low_decisions = []; high_decisions = []; alarm_counts = []
    low_delays = []; high_delays = []
    for position, (observer, low, high) in enumerate(branches):
        vectors = _window_vectors(measurements, observer, config, position)
        alarms = low.alarms(vectors)
        counts = alarms.sum(axis=1)
        low_decisions.append(counts > config["k_low"])
        high_decisions.append(high.predict(alarms))
        alarm_counts.append(counts)
        cumulative = np.cumsum(alarms, axis=1) > config["k_low"]
        low_delays.append(np.where(cumulative.any(axis=1), cumulative.argmax(axis=1), -1))
        high_delays.append(conformal_high_level_delay(
            alarms, config["window"], config["k_high"], high.threshold_))
    low_flags, high_flags = np.stack(low_decisions), np.stack(high_decisions)
    majority = config["window"] // 2 + 1
    def voted_delay(delays):
        values = np.stack(delays)
        ready = np.where(values >= 0, values, config["steps"] + 1)
        kth = np.sort(ready, axis=0)[majority - 1]
        return np.where(kth <= config["steps"], kth, -1)
    return {"branch_low": low_flags, "branch_high": high_flags,
        "branch_alarm_counts": np.stack(alarm_counts),
        "branch_alarm_rates": np.stack(alarm_counts) / float(
            config["steps"] - config["window"] + 1 - config["warmup"]),
        "low_vote": low_flags.sum(axis=0) >= majority,
        "high_vote": high_flags.sum(axis=0) >= majority,
        "low_delay": voted_delay(low_delays),
        "high_delay": voted_delay(high_delays)}


def run_ensemble_five_batches(config, seed=17):
    """Train, calibrate, and evaluate the five observers in bounded batches."""
    if config["steps"] != 10000 or config["m_low"] != 24100 or config["m_high"] != 24100:
        raise ValueError("publication ensemble requires T=10000 and both m=24100")
    if config["n_train"] < 24100:
        raise ValueError("ensemble training requires at least 24100 trajectories")
    rng = np.random.default_rng(seed)
    mixing = make_mixing_matrix(config["dim"], rng)
    train_seed, low_seed, high_seed, test_seed = [
        int(x) for x in rng.integers(1, 2**31 - 1, size=4)]
    branches = []
    score_steps = config["steps"] - config["window"] + 1 - config["warmup"]
    for position in range(config["window"]):
        estimator_rng = np.random.default_rng(seed + position + 100)
        estimator = fit_random_relu_streaming(
            _window_pair_factory(config, train_seed, mixing, position),
            config["hidden"], config["ridge"], estimator_rng)
        center, shape, inverse = _window_training_shape(
            config, train_seed, mixing, estimator, position)
        low_scores = np.empty(config["m_low"]); row = 0
        for _, _, measurements in measurement_batches(
                config["m_low"], config["steps"], config["dim"], low_seed, mixing):
            for part in trajectory_slices(len(measurements)):
                vectors = _window_vectors(measurements[part], estimator, config, position)
                centered = vectors - center
                distances = np.einsum("...i,ij,...j->...", centered, inverse, centered)
                stop = row + len(distances)
                low_scores[row:stop] = kth_max(distances, config["k_low"], axis=1)
                row = stop
        low = LowLevelDetector(config["k_low"], config["delta1_low"],
                               config["ell_low"], config["ridge"], covariance_ddof=0)
        low.center_, low.shape_, low.inverse_ = center, shape, inverse
        low.rank_ = config["ell_low"]
        low.calibration_scores_ = low_scores
        low.threshold_ = conformal_threshold(low_scores, low.rank_)
        high_scores = np.empty(config["m_high"]); row = 0
        for _, _, measurements in measurement_batches(
                config["m_high"], config["steps"], config["dim"], high_seed, mixing):
            for part in trajectory_slices(len(measurements)):
                vectors = _window_vectors(measurements[part], estimator, config, position)
                alarms = low.alarms(vectors)
                scores = high_level_scores(alarms, config["window"], config["k_high"])
                stop = row + len(scores)
                high_scores[row:stop] = scores
                row = stop
        high = HighLevelDetector(config["window"], config["k_high"],
                                 config["delta1_high"], config["ell_high"])
        high.rank_ = config["ell_high"]
        high.calibration_scores_ = high_scores
        high.threshold_ = conformal_threshold(high_scores, high.rank_)
        branches.append((estimator, low, high))

    def evaluate_scenario(kind):
        pieces = []
        for batch_index, (_, _, nominal) in enumerate(measurement_batches(
                config["n_test"], config["steps"], config["dim"], test_seed, mixing)):
            batch_rng = np.random.default_rng(seed + batch_index +
                                              (0 if kind == "nominal" else 1000 + SCENARIOS.index(kind)))
            values = nominal if kind == "nominal" else attack_measurements(
                nominal, kind, batch_rng)
            for part in trajectory_slices(len(values), chunk_size=16):
                pieces.append(_window_branch_scores(values[part], branches, config))
        return {key: np.concatenate([part[key] for part in pieces], axis=-1)
                for key in pieces[0]}

    nominal = summarize_scored(evaluate_scenario("nominal"), config)
    scenarios = {kind: summarize_scored(evaluate_scenario(kind), config, attacked=True)
                 for kind in SCENARIOS}
    calibration = [{"position": position,
        "low_threshold": float(low.threshold_), "high_threshold": float(high.threshold_),
        "low_rank": low.rank_, "high_rank": high.rank_}
        for position, (_, low, high) in enumerate(branches)]
    return {"seed": seed, "config": config, "nominal": nominal,
            "scenarios": scenarios, "calibration": calibration}


def run_randomized_window_five_batches(config, seed=17):
    """Run one conformal detector with an independent uniform observer draw per step."""
    required = ("steps", "window", "n_train", "m_low", "m_high", "n_test",
                "dim", "hidden", "ridge", "filter_cutoff", "filter_order",
                "warmup", "k_low", "k_high", "delta1_low", "delta1_high")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("randomized observer config is missing: {}".format(missing))
    if config["steps"] <= config["window"] or config["n_train"] < 5:
        raise ValueError("steps must exceed window and at least five training trajectories are required")
    rng = np.random.default_rng(seed)
    mixing = make_mixing_matrix(config["dim"], rng)
    train_seed, low_seed, high_seed, test_seed = [
        int(x) for x in rng.integers(1, 2**31 - 1, size=4)]
    train_choice_seed, low_choice_seed, high_choice_seed, test_choice_seed = [
        int(x) for x in rng.integers(1, 2**31 - 1, size=4)]

    # Fit the full candidate bank on the proper training trajectories.
    observers = []
    for position in range(config["window"]):
        observer_rng = np.random.default_rng(seed + 100 + position)
        observer = fit_random_relu_streaming(
            _window_pair_factory(config, train_seed, mixing, position),
            config["hidden"], config["ridge"], observer_rng)
        observers.append(observer)

    score_steps = config["steps"] - config["window"] + 1 - config["warmup"]

    # Estimate one ellipsoid from the randomized training residual stream.
    shape_rng = np.random.default_rng(train_choice_seed)
    summed = np.zeros(config["dim"])
    cross = np.zeros((config["dim"], config["dim"]))
    count = 0
    for _, _, measurements in measurement_batches(
            config["n_train"], config["steps"], config["dim"], train_seed, mixing):
        vectors, _ = randomized_window_vectors(measurements, observers, config, shape_rng)
        summed, cross, count = accumulate_shape(summed, cross, vectors)
    center, shape, inverse = shape_from_moments(
        summed, cross, count, config["ridge"])

    def trajectory_scores(n, data_seed, draw_seed, low_threshold=None):
        scores = np.empty(n, dtype=float)
        draw_rng = np.random.default_rng(draw_seed)
        row = 0
        for _, _, measurements in measurement_batches(
                n, config["steps"], config["dim"], data_seed, mixing):
            vectors, _ = randomized_window_vectors(
                measurements, observers, config, draw_rng)
            distances = quadratic_distances(vectors, center, inverse)
            end = row + len(distances)
            if low_threshold is None:
                scores[row:end] = kth_max(distances, config["k_low"], axis=1)
            else:
                alarms = distances > low_threshold
                scores[row:end] = high_level_scores(
                    alarms, config["window"], config["k_high"])
            row = end
        return scores

    low_scores = trajectory_scores(
        config["m_low"], low_seed, low_choice_seed)
    low_rank = config.get("ell_low")
    if low_rank is None:
        low_rank = conformal_rank(config["m_low"], config["delta1_low"])
    low_threshold = conformal_threshold(low_scores, low_rank)

    high_scores = trajectory_scores(
        config["m_high"], high_seed, high_choice_seed, low_threshold)
    high_rank = config.get("ell_high")
    if high_rank is None:
        high_rank = conformal_rank(config["m_high"], config["delta1_high"])
    high_threshold = conformal_threshold(high_scores, high_rank)

    def evaluate(kind):
        choices_rng = np.random.default_rng(test_choice_seed)
        attack_rng = np.random.default_rng(seed + 3000 + (
            0 if kind == "nominal" else SCENARIOS.index(kind) + 1))
        low_decisions = []
        high_decisions = []
        point_alarm_rates = []
        low_counts = []
        high_delays = []
        for _, _, nominal in measurement_batches(
                config["n_test"], config["steps"], config["dim"], test_seed, mixing):
            data = nominal if kind == "nominal" else attack_measurements(
                nominal, kind, attack_rng)
            vectors, _ = randomized_window_vectors(
                data, observers, config, choices_rng)
            distances = quadratic_distances(vectors, center, inverse)
            alarms = distances > low_threshold
            low_count = alarms.sum(axis=1)
            high_scores_test = high_level_scores(
                alarms, config["window"], config["k_high"])
            low_decisions.extend((low_count > config["k_low"]).tolist())
            high_decisions.extend((high_scores_test > high_threshold).tolist())
            point_alarm_rates.extend(alarms.mean(axis=1).tolist())
            low_counts.extend(low_count.tolist())
            delays = conformal_high_level_delay(
                alarms, config["window"], config["k_high"], high_threshold)
            high_delays.extend(delays[delays >= 0].tolist())
        return {
            "low_detection_rate": float(np.mean(low_decisions)),
            "high_detection_rate": float(np.mean(high_decisions)),
            "mean_alarm_rate": float(np.mean(point_alarm_rates)),
            "mean_low_alarm_count": float(np.mean(low_counts)),
            "median_high_alarm_index": (
                float(np.median(high_delays)) if high_delays else None),
        }

    nominal = evaluate("nominal")
    scenarios = {kind: evaluate(kind) for kind in SCENARIOS}
    return {
        "seed": seed,
        "config": config,
        "observer_sampling": "independent uniform draw at every window-completion time",
        "nominal": nominal,
        "scenarios": scenarios,
        "calibration": {
            "low_threshold": float(low_threshold),
            "high_threshold": float(high_threshold),
            "low_rank": int(low_rank),
            "high_rank": int(high_rank),
            "low_delta2": beta_confidence(
                config["delta1_low"], low_rank, config["m_low"]),
            "high_delta2": beta_confidence(
                config["delta1_high"], high_rank, config["m_high"]),
            "valid_residual_times": int(score_steps),
        },
        "random_seeds": {
            "training_observer_draws": train_choice_seed,
            "low_calibration_observer_draws": low_choice_seed,
            "high_calibration_observer_draws": high_choice_seed,
            "test_observer_draws": test_choice_seed,
        },
    }


def run_applications_five_batches(args, seed=17):
    """Run linear and nonlinear application experiments in five data batches."""
    if args.steps != 10000 or args.m_calibration != 24100 or args.n_train < 24100:
        raise ValueError("publication applications require T=10000, m=24100, and N>=24100")
    warmup = effective_filter_warmup(args)
    score_steps = args.steps - args.history - warmup
    allowed = args.k_low

    def run_linear():
        total = count = 0
        summed = np.zeros(2); cross = np.zeros((2, 2))
        train_seed, cal_seed, val_seed = seed + 10, seed + 20, seed + 30
        for i, (start, stop) in enumerate(five_batches(args.n_train)):
            values = linear_residuals(stop - start, args.steps, np.random.default_rng(train_seed + i))
            summed, cross, count = accumulate_shape(summed, cross, values)
        center, q, q_inv = shape_from_moments(summed, cross, count, args.ridge)
        cal_scores = np.empty(args.m_calibration)
        row = 0
        for i, (start, stop) in enumerate(five_batches(args.m_calibration)):
            values = linear_residuals(stop - start, args.steps, np.random.default_rng(cal_seed + i))
            distances = quadratic_distances(values, center, q_inv)
            end = row + len(values)
            cal_scores[row:end] = kth_max(distances, allowed, axis=1)
            row = end
        threshold = conformal_threshold(cal_scores, args.ell)
        violations = np.empty(args.n_validation, dtype=int)
        chi_violations = np.empty(args.n_validation, dtype=int)
        conf_total_violations = chi_total_violations = 0
        row = 0
        from scipy.stats import chi2
        chi_threshold = float(chi2.ppf(args.coverage, df=2))
        for i, (start, stop) in enumerate(five_batches(args.n_validation)):
            values = linear_residuals(stop - start, args.steps, np.random.default_rng(val_seed + i))
            distances = quadratic_distances(values, center, q_inv)
            end = row + len(values)
            violations[row:end] = np.sum(distances > threshold, axis=1)
            chi_violations[row:end] = np.sum(distances > chi_threshold, axis=1)
            conf_total_violations += int(np.sum(distances > threshold))
            chi_total_violations += int(np.sum(distances > chi_threshold))
            row = end
        chi_metrics = {
            "threshold": chi_threshold,
            "empirical_coverage": 1.0 - chi_total_violations / float(args.n_validation * args.steps),
            "empirical_violation_rate": chi_total_violations / float(args.n_validation * args.steps),
            "mean_violations_per_trajectory": float(np.mean(chi_violations)),
            "p95_violations_per_trajectory": float(np.quantile(chi_violations, .95)),
        }
        conformal_metrics = {
            "empirical_coverage": 1.0 - conf_total_violations / float(args.n_validation * score_steps),
            "empirical_violation_rate": conf_total_violations / float(args.n_validation * score_steps),
            "trajectory_alarm_rate": float(np.mean(violations > allowed)),
            "mean_violations_per_trajectory": float(np.mean(violations)),
            "p95_violations_per_trajectory": float(np.quantile(violations, .95)),
        }
        return {"distance_source": "linear observer residual",
            "score": "r_t^T Q^{-1} r_t", "steps": args.steps,
            "score_steps": args.steps, "n_train": args.n_train,
            "m_calibration": args.m_calibration,
            "n_validation": args.n_validation, "coverage_target": args.coverage,
            "allowed_violations_per_trajectory": allowed,
            "allowed_violation_rate": allowed / float(args.steps),
            "rank": args.ell, "delta1": args.delta1,
            "delta2": beta_confidence(args.delta1, args.ell, args.m_calibration),
            "conformal_threshold": float(threshold), "conformal": conformal_metrics,
            "chi_square_baseline": chi_metrics, "residual_mean": center.tolist(),
            "q": q.tolist()}, {"conformal": violations, "chi_square": chi_violations}

    def nonlinear_pair_factory(train_seed):
        def batches():
            for i, (start, stop) in enumerate(five_batches(args.n_train)):
                values = nonlinear_measurements(stop - start, args.steps,
                                                np.random.default_rng(train_seed + i))
                for part in trajectory_slices(len(values)):
                    yield history_pairs_vectorized(values[part], args.history)
        return batches

    def run_nonlinear():
        train_seed, cal_seed, val_seed = seed + 40, seed + 50, seed + 60
        estimator = fit_random_relu_streaming(
            nonlinear_pair_factory(train_seed), args.hidden, args.ridge,
            np.random.default_rng(seed + 70))
        dim = 2
        sum_vector = np.zeros(dim); cross = np.zeros((dim, dim)); count = 0
        for i, (start, stop) in enumerate(five_batches(args.n_train)):
            values = nonlinear_measurements(stop - start, args.steps,
                                            np.random.default_rng(train_seed + i))
            for part in trajectory_slices(len(values)):
                residual = observer_residuals_vectorized(values[part], estimator, args.history)
                filtered = trim_filter_warmup(
                    butterworth_lowpass(residual, args.filter_cutoff, args.filter_order), warmup)
                sum_vector, cross, count = accumulate_shape(sum_vector, cross, filtered)
        center, q, q_inv = shape_from_moments(sum_vector, cross, count, args.ridge)
        cal_scores = np.empty(args.m_calibration); row = 0
        for i, (start, stop) in enumerate(five_batches(args.m_calibration)):
            values = nonlinear_measurements(stop - start, args.steps,
                                            np.random.default_rng(cal_seed + i))
            for part in trajectory_slices(len(values)):
                residual = observer_residuals_vectorized(values[part], estimator, args.history)
                filtered = trim_filter_warmup(
                    butterworth_lowpass(residual, args.filter_cutoff, args.filter_order), warmup)
                distances = quadratic_distances(filtered, center, q_inv)
                end = row + len(distances)
                cal_scores[row:end] = kth_max(distances, allowed, axis=1)
                row = end
        threshold = conformal_threshold(cal_scores, args.ell)
        violations = np.empty(args.n_validation, dtype=int)
        energy_raw = energy_filtered = 0.0; row = 0
        total_violations = 0
        for i, (start, stop) in enumerate(five_batches(args.n_validation)):
            values = nonlinear_measurements(stop - start, args.steps,
                                            np.random.default_rng(val_seed + i))
            for part in trajectory_slices(len(values)):
                residual = observer_residuals_vectorized(values[part], estimator, args.history)
                aligned = trim_filter_warmup(residual, warmup)
                filtered = trim_filter_warmup(
                    butterworth_lowpass(residual, args.filter_cutoff, args.filter_order), warmup)
                distances = quadratic_distances(filtered, center, q_inv)
                local = np.sum(distances > threshold, axis=1)
                total_violations += int(local.sum())
                end = row + len(local)
                violations[row:end] = local
                row = end
                energy_raw += float(np.sum(aligned ** 2))
                energy_filtered += float(np.sum(filtered ** 2))
        metrics_base = {
            "empirical_coverage": 1.0 - total_violations / float(args.n_validation * score_steps),
            "empirical_violation_rate": total_violations / float(args.n_validation * score_steps),
            "trajectory_alarm_rate": float(np.mean(violations > allowed)),
            "mean_violations_per_trajectory": float(np.mean(violations)),
            "p95_violations_per_trajectory": float(np.quantile(violations, .95)),
        }
        metrics = {"distance_source": "Butterworth-filtered neural-observer residual",
            "score": "filtered residual Mahalanobis distance", "steps": args.steps,
            "score_steps": score_steps, "history": args.history, "filter_warmup": warmup,
            "hidden": args.hidden, "filter_order": args.filter_order,
            "filter_cutoff": args.filter_cutoff, "n_train": args.n_train,
            "m_calibration": args.m_calibration, "n_validation": args.n_validation,
            "coverage_target": args.coverage,
            "allowed_violations_per_trajectory": allowed,
            "allowed_violation_rate": allowed / float(score_steps), "rank": args.ell,
            "epsilon": 1.0 - args.delta1, "ell": args.ell, "m": args.m_calibration,
            "delta1": args.delta1,
            "delta2": beta_confidence(args.delta1, args.ell, args.m_calibration),
            "conformal_threshold": float(threshold), **metrics_base,
            "filtered_residual_mean": center.tolist(), "q": q.tolist(),
            "filter_energy_ratio": energy_filtered / max(energy_raw, 1e-12)}
        return metrics, violations

    linear, linear_dist = run_linear()
    nonlinear, nonlinear_dist = run_nonlinear()
    return {"linear_residual": linear,
            "nonlinear_filtered_residual": nonlinear}, {
                "linear_residual": linear_dist,
                "nonlinear_filtered_residual": nonlinear_dist}
