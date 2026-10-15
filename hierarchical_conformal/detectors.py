"""Reusable detectors calibrated on independent sets of nominal trajectories."""

import numpy as np

from .estimators import RandomFeatureObserver, history_dataset, window_pairs, window_residuals
from .utils import (butterworth_lowpass, conformal_rank, conformal_threshold,
                    conformal_high_level_delay, high_level_scores, kth_max, sliding_window_rates,
                    trim_filter_warmup)


def _trajectories(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError("{} must be a finite [trajectory,time,sensor] array".format(name))
    return array


def _same_geometry(train, low, high):
    if train.shape[1:] != low.shape[1:] or train.shape[1:] != high.shape[1:]:
        raise ValueError("training and calibration trajectories need the same time and sensor dimensions")


def _rank(m, delta1, ell):
    if not 0.0 < delta1 < 1.0:
        raise ValueError("delta1 must lie strictly between 0 and 1")
    rank = conformal_rank(m, delta1) if ell is None else ell
    if not isinstance(rank, (int, np.integer)) or not 1 <= rank <= m:
        raise ValueError("ell must be an integer from 1 to the calibration sample count")
    return int(rank)


class LowLevelDetector:
    """Ellipsoidal residual detector with a k-th-maximum trajectory score.

    ``fit`` uses training score vectors only to estimate the ellipsoid shape;
    complete low-level calibration trajectories set its conformal threshold.
    The trajectory decision counts threshold violations across the *entire*
    valid score sequence and fires when that count exceeds ``k``.
    """

    def __init__(self, k, delta1=0.9, ell=None, ridge=1e-5, covariance_ddof=1):
        if not isinstance(k, int) or k < 1 or ridge <= 0:
            raise ValueError("k must be a positive integer and ridge must be positive")
        if covariance_ddof not in (0, 1):
            raise ValueError("covariance_ddof must be 0 or 1")
        self.k, self.delta1, self.ell, self.ridge = k, delta1, ell, ridge
        self.covariance_ddof = covariance_ddof
        self.threshold_ = None

    def fit(self, training_vectors, calibration_vectors):
        train = _trajectories(training_vectors, "training_vectors")
        cal = _trajectories(calibration_vectors, "calibration_vectors")
        if train.shape[1:] != cal.shape[1:] or self.k > train.shape[1]:
            raise ValueError("training and calibration score geometry must match and k must fit")
        flat = train.reshape(-1, train.shape[-1])
        self.center_ = flat.mean(axis=0)
        centered = flat - self.center_
        self.shape_ = (centered.T @ centered / max(1, len(flat) - self.covariance_ddof)
                       + self.ridge * np.eye(flat.shape[1]))
        self.inverse_ = np.linalg.pinv(self.shape_)
        self.rank_ = _rank(len(cal), self.delta1, self.ell)
        self.calibration_scores_ = self.scores(cal)
        self.threshold_ = conformal_threshold(self.calibration_scores_, self.rank_)
        return self

    def distances(self, vectors):
        if not hasattr(self, "inverse_"):
            raise RuntimeError("detector must be fitted before scoring")
        values = _trajectories(vectors, "vectors")
        if values.shape[2] != len(self.center_):
            raise ValueError("sensor dimension differs from fitted detector")
        centered = values - self.center_[None, None, :]
        return np.einsum("...i,ij,...j->...", centered, self.inverse_, centered)

    def scores(self, vectors):
        return kth_max(self.distances(vectors), self.k, axis=1)

    def alarms(self, vectors):
        if self.threshold_ is None:
            raise RuntimeError("detector must be calibrated before prediction")
        return self.distances(vectors) > self.threshold_

    def predict(self, vectors):
        return self.alarms(vectors).sum(axis=1) > self.k


class HighLevelDetector:
    """Conformal k-th-maximum monitor of sliding low-level alarm rates."""

    def __init__(self, window, k, delta1=0.9, ell=None):
        if not isinstance(window, int) or window < 1 or not isinstance(k, int) or k < 1:
            raise ValueError("window and k must be positive integers")
        self.window, self.k, self.delta1, self.ell = window, k, delta1, ell
        self.threshold_ = None

    def scores(self, alarms):
        values = np.asarray(alarms)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < self.window:
            raise ValueError("alarms must be [trajectory,time] with enough time samples")
        if not np.isin(values, (0, 1)).all():
            raise ValueError("alarms must contain only binary values")
        if self.k > values.shape[1] - self.window + 1:
            raise ValueError("k exceeds the number of alarm-rate windows")
        return high_level_scores(values, self.window, self.k)

    def fit(self, calibration_alarms):
        scores = self.scores(calibration_alarms)
        self.rank_ = _rank(len(scores), self.delta1, self.ell)
        self.calibration_scores_ = scores
        self.threshold_ = conformal_threshold(scores, self.rank_)
        return self

    def predict(self, alarms):
        if self.threshold_ is None:
            raise RuntimeError("detector must be calibrated before prediction")
        return self.scores(alarms) > self.threshold_

    def rates(self, alarms):
        return sliding_window_rates(np.asarray(alarms), self.window)


class HierarchicalDetector:
    """Train an observer and calibrate low/high detectors on disjoint splits.

    ``observer`` may implement ``fit(X, y)`` and ``predict_flat(X)``. A fitted
    observer implementing only ``predict_flat(X)`` is also accepted. Its input
    is the past ``history`` measurements, ordered from newest to oldest.
    """

    def __init__(self, history, k_low, window, k_high, observer=None,
                 delta1_low=0.9, delta1_high=0.9, ell_low=None, ell_high=None,
                 ridge=1e-5, filter_cutoff=0.08, filter_order=3, warmup=9):
        if not isinstance(history, int) or history < 1 or warmup < 0:
            raise ValueError("history must be positive and warmup nonnegative")
        self.history = history
        self.observer = observer if observer is not None else RandomFeatureObserver(ridge=ridge)
        self.low = LowLevelDetector(k_low, delta1_low, ell_low, ridge)
        self.high = HighLevelDetector(window, k_high, delta1_high, ell_high)
        self.filter_cutoff, self.filter_order, self.warmup = filter_cutoff, filter_order, warmup
        self.fitted_ = False

    def _vectors(self, measurements):
        values = _trajectories(measurements, "measurements")
        if values.shape[1:] != self.geometry_:
            raise ValueError("trajectory length and sensor dimension must match calibration")
        features, targets = history_dataset(values, self.history)
        predicted = np.asarray(self.observer.predict_flat(features), dtype=float)
        if predicted.shape != targets.shape or not np.isfinite(predicted).all():
            raise ValueError("observer prediction must match finite targets")
        residual = (targets - predicted).reshape(
            values.shape[0], values.shape[1] - self.history, values.shape[2])
        return trim_filter_warmup(
            butterworth_lowpass(residual, self.filter_cutoff, self.filter_order), self.warmup)

    def fit(self, train, low_calibration, high_calibration):
        train = _trajectories(train, "train")
        low_calibration = _trajectories(low_calibration, "low_calibration")
        high_calibration = _trajectories(high_calibration, "high_calibration")
        _same_geometry(train, low_calibration, high_calibration)
        self.geometry_ = train.shape[1:]
        features, targets = history_dataset(train, self.history)
        if hasattr(self.observer, "fit"):
            self.observer.fit(features, targets)
        self.low.fit(self._vectors(train), self._vectors(low_calibration))
        self.high.fit(self.low.alarms(self._vectors(high_calibration)))
        self.fitted_ = True
        return self

    def score(self, measurements):
        if not self.fitted_:
            raise RuntimeError("detector must be fitted before prediction")
        vectors = self._vectors(measurements)
        distances = self.low.distances(vectors)
        alarms = distances > self.low.threshold_
        return {"distances": distances, "alarms": alarms,
                "low_score": kth_max(distances, self.low.k, axis=1),
                "high_score": self.high.scores(alarms),
                "low_alarm": alarms.sum(axis=1) > self.low.k,
                "high_alarm": self.high.predict(alarms)}

    def predict(self, measurements):
        scored = self.score(measurements)
        return scored["low_alarm"], scored["high_alarm"]


class WindowEnsembleDetector:
    """W omitted-position observers and separate low/high majority decisions.

    Each branch is calibrated on complete trajectories. Majority voting is a
    decision rule; its false-alarm probability is not certified by the
    individual branch certificates alone.
    """

    def __init__(self, window, k_low, rate_window, k_high, observer_factory=None,
                 delta1_low=0.9, delta1_high=0.9, ell_low=None, ell_high=None,
                 ridge=1e-5, filter_cutoff=0.08, filter_order=3, warmup=9,
                 max_training_pairs=None, seed=0, rng=None):
        if not isinstance(window, int) or window < 2 or warmup < 0:
            raise ValueError("window must be at least 2 and warmup nonnegative")
        self.window, self.k_low, self.rate_window, self.k_high = window, k_low, rate_window, k_high
        self.observer_factory = observer_factory
        self.delta1_low, self.delta1_high = delta1_low, delta1_high
        self.ell_low, self.ell_high = ell_low, ell_high
        self.ridge, self.filter_cutoff, self.filter_order, self.warmup = ridge, filter_cutoff, filter_order, warmup
        self.max_training_pairs, self.seed = max_training_pairs, seed
        self.rng = rng
        self.branches_ = None

    def _vectors(self, measurements, observer, position):
        values = _trajectories(measurements, "measurements")
        if values.shape[1:] != self.geometry_:
            raise ValueError("trajectory length and sensor dimension must match calibration")
        residual = window_residuals(values, observer, self.window, position)
        return trim_filter_warmup(
            butterworth_lowpass(residual, self.filter_cutoff, self.filter_order), self.warmup)

    def fit(self, train, low_calibration, high_calibration):
        train = _trajectories(train, "train")
        low_calibration = _trajectories(low_calibration, "low_calibration")
        high_calibration = _trajectories(high_calibration, "high_calibration")
        _same_geometry(train, low_calibration, high_calibration)
        self.geometry_ = train.shape[1:]
        rng = self.rng if self.rng is not None else np.random.default_rng(self.seed)
        branches = []
        observer_ids = set()
        for position in range(self.window):
            observer = (self.observer_factory(position) if self.observer_factory is not None
                        else RandomFeatureObserver(ridge=self.ridge, seed=self.seed + position))
            if id(observer) in observer_ids:
                raise ValueError("observer_factory must return a separate observer per position")
            observer_ids.add(id(observer))
            features, targets = window_pairs(train, self.window, position)
            if self.max_training_pairs is not None:
                if self.max_training_pairs < 1:
                    raise ValueError("max_training_pairs must be positive")
                if len(features) > self.max_training_pairs:
                    chosen = rng.choice(len(features), self.max_training_pairs, replace=False)
                    features, targets = features[chosen], targets[chosen]
            if hasattr(observer, "fit"):
                observer.fit(features, targets)
            low = LowLevelDetector(self.k_low, self.delta1_low, self.ell_low,
                                   self.ridge, covariance_ddof=0)
            low.fit(self._vectors(train, observer, position),
                    self._vectors(low_calibration, observer, position))
            high = HighLevelDetector(self.rate_window, self.k_high, self.delta1_high, self.ell_high)
            high.fit(low.alarms(self._vectors(high_calibration, observer, position)))
            branches.append((observer, low, high))
        self.branches_ = branches
        return self

    def score(self, measurements):
        if self.branches_ is None:
            raise RuntimeError("ensemble must be fitted before prediction")
        branch_low, branch_high, branch_counts = [], [], []
        branch_low_delay, branch_high_delay = [], []
        for position, (observer, low, high) in enumerate(self.branches_):
            vectors = self._vectors(measurements, observer, position)
            alarms = low.alarms(vectors)
            branch_counts.append(alarms.sum(axis=1))
            branch_low.append(alarms.sum(axis=1) > self.k_low)
            branch_high.append(high.predict(alarms))
            cumulative = np.cumsum(alarms, axis=1) > self.k_low
            branch_low_delay.append(np.where(cumulative.any(axis=1), cumulative.argmax(axis=1), -1))
            branch_high_delay.append(conformal_high_level_delay(
                alarms, self.rate_window, self.k_high, high.threshold_))
        low_flags, high_flags = np.stack(branch_low), np.stack(branch_high)
        majority = self.window // 2 + 1
        def voted_delay(delays):
            delay_matrix = np.stack(delays)
            sentinel = self.geometry_[0] + 1
            ready = np.where(delay_matrix >= 0, delay_matrix, sentinel)
            kth = np.sort(ready, axis=0)[majority - 1]
            return np.where(kth < sentinel, kth, -1)
        return {"branch_low": low_flags, "branch_high": high_flags,
                "branch_alarm_counts": np.stack(branch_counts),
                "branch_alarm_rates": np.stack(branch_counts) / float(
                    self.geometry_[0] - self.window + 1 - self.warmup),
                "low_vote": low_flags.sum(axis=0) >= majority,
                "high_vote": high_flags.sum(axis=0) >= majority,
                "low_delay": voted_delay(branch_low_delay),
                "high_delay": voted_delay(branch_high_delay)}

    def predict(self, measurements):
        scored = self.score(measurements)
        return scored["low_vote"], scored["high_vote"]
