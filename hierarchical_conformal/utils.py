"""General score, calibration, filtering, and alarm-stream utilities."""

from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.stats import beta

def butterworth_lowpass(samples: np.ndarray, cutoff: float, order: int = 3) -> np.ndarray:
    """Apply a causal Butterworth low-pass filter along the trajectory axis."""
    if not 0.0 < cutoff < 1.0:
        raise ValueError("cutoff must be normalized to the Nyquist frequency and lie in (0, 1)")
    sos = butter(order, cutoff, btype="lowpass", output="sos")
    return sosfilt(sos, samples, axis=1)


def trim_filter_warmup(samples: np.ndarray, warmup: int) -> np.ndarray:
    """Discard initial filtered samples affected by observer/filter transients."""
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if warmup >= samples.shape[1]:
        raise ValueError("warmup must be smaller than the number of score samples")
    return samples[:, warmup:, :]


def kth_max(values: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """Return the k-th largest value along an axis.

    k=1 returns the maximum. k=500 returns the 500-th largest value.
    """
    values = np.asarray(values)
    if values.ndim == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a finite array with at least one axis")
    if k < 1:
        raise ValueError("k must be at least 1")
    n = values.shape[axis]
    if k > n:
        raise ValueError(f"k={k} exceeds axis length {n}")
    partition_index = n - k
    return np.partition(values, partition_index, axis=axis).take(partition_index, axis=axis)


def conformal_rank(m: int, delta1: float) -> int:
    """Default conservative rank for split conformal calibration."""
    if not isinstance(m, (int, np.integer)) or m < 1 or not 0 < delta1 < 1:
        raise ValueError("m must be positive and delta1 must lie in (0, 1)")
    rank = int(np.ceil(delta1 * (m + 1)))
    return max(1, min(rank, m))


def beta_confidence(delta1: float, ell: int, m: int) -> float:
    """Double-step conformal confidence 1 - BetaCDF_delta1(ell, m+1-ell)."""
    if not 0 < delta1 < 1 or not 1 <= ell <= m:
        raise ValueError("delta1 must lie in (0, 1) and 1 <= ell <= m")
    return float(1.0 - beta.cdf(delta1, ell, m + 1 - ell))


def confidence_rank(m: int, delta1: float, delta2: float) -> int:
    """Smallest order-statistic rank attaining a target beta-CDF confidence."""
    if m < 1 or not 0 < delta1 < 1 or not 0 < delta2 < 1:
        raise ValueError("m must be positive and delta1, delta2 must lie in (0, 1)")
    if beta_confidence(delta1, m, m) < delta2:
        raise ValueError("calibration set is too small for the requested confidence")
    left, right = 1, m
    while left < right:
        middle = (left + right) // 2
        if beta_confidence(delta1, middle, m) >= delta2:
            right = middle
        else:
            left = middle + 1
    return left


def conformal_threshold(scores: np.ndarray, ell: int) -> float:
    """Return the ell-th order statistic, using one-based rank ell."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite one-dimensional array")
    if not 1 <= ell <= len(scores):
        raise ValueError("ell must satisfy 1 <= ell <= len(scores)")
    return float(np.sort(scores)[ell - 1])


def estimate_q(deltas: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Estimate a regularized shape matrix Q from trajectory score vectors.

    samples shape: [n_trajectories, n_steps, dim].
    """
    flat = deltas.reshape(-1, deltas.shape[-1])
    centered = flat - flat.mean(axis=0, keepdims=True)
    q = centered.T @ centered / max(1, len(flat) - 1)
    q += ridge * np.eye(q.shape[0])
    return q


def mahalanobis_distances(deltas: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute d_t = u_t^T Q^{-1} u_t for each trajectory/time."""
    q_inv = np.linalg.pinv(q)
    return np.einsum("...i,ij,...j->...", deltas, q_inv, deltas)


def low_level_scores(distances: np.ndarray, k: int) -> np.ndarray:
    """Trajectory scores based on the k-th maximum distance."""
    return kth_max(distances, k=k, axis=1)


def low_level_alarms(distances: np.ndarray, threshold: float) -> np.ndarray:
    """Binary low-level alarm stream."""
    return (distances > threshold).astype(np.int8)


def first_low_level_alarm_times(distances: np.ndarray, threshold: float) -> np.ndarray:
    """Return first low-level alarm index for each trajectory, or -1 if absent."""
    flags = distances > threshold
    has_alarm = flags.any(axis=1)
    first = np.where(has_alarm, flags.argmax(axis=1), -1)
    return first.astype(int)


def sliding_window_rates(alarms: np.ndarray, window: int) -> np.ndarray:
    """Compute sliding-window alarm rates for each trajectory.

    alarms shape: [n_trajectories, n_steps].
    output shape: [n_trajectories, n_steps-window+1].
    """
    if window < 1 or window > alarms.shape[1]:
        raise ValueError("window must satisfy 1 <= window <= number of alarm samples")
    csum = np.cumsum(alarms, axis=1)
    counts = csum[:, window - 1 :] - np.concatenate(
        [np.zeros((alarms.shape[0], 1), dtype=csum.dtype), csum[:, :-window]], axis=1
    )
    return counts / float(window)


def high_level_scores(alarms: np.ndarray, window: int, k_high: int) -> np.ndarray:
    """Trajectory scores based on the k-th maximum sliding-window alarm rate."""
    rates = sliding_window_rates(alarms, window)
    return kth_max(rates, k=k_high, axis=1)


def conformal_high_level_flags(
    alarms: np.ndarray,
    window: int,
    k_high: int,
    threshold: float,
) -> np.ndarray:
    """Flag trajectories whose high-level conformal score exceeds threshold."""
    scores = high_level_scores(alarms, window, k_high)
    return scores > threshold


def conformal_high_level_delay(
    alarms: np.ndarray,
    window: int,
    k_high: int,
    threshold: float,
) -> np.ndarray:
    """First sample index at which the conformal high-level detector fires.

    Because the high-level score is the k_high-th largest window rate, the
    online detector fires once k_high windows have exceeded the calibrated
    rate threshold. Trajectories that never trigger get -1.
    """
    rates = sliding_window_rates(alarms, window)
    exceed = rates > threshold
    cumulative = np.cumsum(exceed, axis=1)
    flags = cumulative >= k_high
    has_alarm = flags.any(axis=1)
    first_window = np.where(has_alarm, flags.argmax(axis=1), -1)
    return np.where(has_alarm, first_window + window - 1, -1).astype(int)


def summarize_delays(delays: np.ndarray) -> dict:
    """Summarize detector delays, ignoring non-detections for delay quantiles."""
    delays = np.asarray(delays, dtype=int)
    detected = delays >= 0
    if np.any(detected):
        valid = delays[detected]
        mean_delay = float(np.mean(valid))
        median_delay = float(np.median(valid))
        p90_delay = float(np.quantile(valid, 0.9))
    else:
        mean_delay = None
        median_delay = None
        p90_delay = None
    return {
        "detection_rate": float(np.mean(detected)),
        "mean_delay": mean_delay,
        "median_delay": median_delay,
        "p90_delay": p90_delay,
    }


class CalibrationSummary:
    def __init__(
        self,
        low_threshold,
        low_rank,
        low_delta1,
        low_delta2,
        high_threshold,
        high_quiet_threshold,
        high_rank,
        high_delta1,
        high_delta2,
    ):
        self.low_threshold = low_threshold
        self.low_rank = low_rank
        self.low_delta1 = low_delta1
        self.low_delta2 = low_delta2
        self.high_threshold = high_threshold
        self.high_quiet_threshold = high_quiet_threshold
        self.high_rank = high_rank
        self.high_delta1 = high_delta1
        self.high_delta2 = high_delta2

    def as_dict(self):
        return {
            "low_threshold": self.low_threshold,
            "low_rank": self.low_rank,
            "low_delta1": self.low_delta1,
            "low_delta2": self.low_delta2,
            "high_threshold": self.high_threshold,
            "high_quiet_threshold": self.high_quiet_threshold,
            "high_rank": self.high_rank,
            "high_delta1": self.high_delta1,
            "high_delta2": self.high_delta2,
        }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

