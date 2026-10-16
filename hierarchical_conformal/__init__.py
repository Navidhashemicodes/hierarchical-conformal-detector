"""Trajectory-level hierarchical conformal anomaly detection.

Calibration units are complete, exchangeable nominal trajectories. Measurements
within a trajectory may be dependent. See README.md for the precise scope of
the branchwise conformal certificates.
"""

from .detectors import (HighLevelDetector, HierarchicalDetector, LowLevelDetector,
                        RandomizedWindowDetector, WindowEnsembleDetector,
                        randomized_window_residuals)
from .estimators import (RandomFeatureObserver, RandomReluEstimator, fit_random_relu,
                         history_dataset, residual_trajectories, train_random_relu_estimator)
from .utils import (CalibrationSummary, beta_confidence, butterworth_lowpass,
                    confidence_rank, conformal_high_level_delay,
                    conformal_high_level_flags, conformal_rank, conformal_threshold,
                    ensure_dir, first_low_level_alarm_times, high_level_scores,
                    kth_max, low_level_alarms, low_level_scores,
                    sliding_window_rates, summarize_delays, trim_filter_warmup)

__all__ = [
    "LowLevelDetector", "HighLevelDetector", "HierarchicalDetector",
    "RandomizedWindowDetector", "WindowEnsembleDetector",
    "randomized_window_residuals",
    "RandomFeatureObserver", "RandomReluEstimator",
    "kth_max", "conformal_rank", "confidence_rank", "beta_confidence",
    "conformal_threshold",
    "fit_random_relu", "history_dataset", "residual_trajectories",
    "train_random_relu_estimator", "butterworth_lowpass", "trim_filter_warmup",
    "low_level_scores", "low_level_alarms", "high_level_scores",
    "sliding_window_rates", "conformal_high_level_flags",
    "conformal_high_level_delay", "first_low_level_alarm_times",
    "summarize_delays", "CalibrationSummary", "ensure_dir",
]
