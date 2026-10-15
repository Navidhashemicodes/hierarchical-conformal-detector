"""Focused checks of the trajectory and window-position implementations."""

import unittest
import numpy as np

from hierarchical_conformal import confidence_rank, fit_random_relu
from hierarchical_conformal.estimators import window_pairs, window_residuals
from hierarchical_conformal.utils import butterworth_lowpass, trim_filter_warmup
from benchmarks.simulate import attack_measurements
from benchmarks.window_ensemble import (default_config, fit_ensemble,
                                        measurements, score_ensemble)


class PipelineTests(unittest.TestCase):
    def test_publication_rank_matches_stated_example(self):
        self.assertEqual(confidence_rank(24100, 0.9995, 0.9995), 24098)

    def test_window_pairs_omit_the_requested_position(self):
        y = np.arange(5, dtype=float).reshape(1, 5, 1)
        features, targets = window_pairs(y, 3, 1)
        np.testing.assert_array_equal(features, [[0, 2], [1, 3], [2, 4]])
        np.testing.assert_array_equal(targets[:, 0], [1, 2, 3])

    def test_estimator_returns_predictions_on_measurement_scale(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(40, 3))
        y = 50.0 + 4.0 * x[:, :1]
        estimator = fit_random_relu(x, y, 12, 1e-5, rng)
        prediction = estimator.predict_flat(x)
        self.assertEqual(prediction.shape, y.shape)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreater(estimator.output_mean[0, 0], 40.0)

    def test_attacks_leave_pre_onset_measurements_unchanged(self):
        nominal = np.ones((2, 20, 2))
        attacked = attack_measurements(nominal, "bias", np.random.default_rng(5))
        np.testing.assert_array_equal(attacked[:, :9], nominal[:, :9])
        self.assertTrue(np.any(attacked[:, 9:] != nominal[:, 9:]))

    def test_low_vote_counts_over_complete_trajectory(self):
        config = default_config()
        config.update(steps=45, dim=2, window=3, n_train=8, m_low=12,
                      m_high=12, n_test=5, k_low=2, k_high=1, hidden=8,
                      warmup=2, max_training_pairs=500, delta1_low=0.8,
                      delta1_high=0.8)
        rng = np.random.default_rng(11)
        fitted = fit_ensemble(config, rng)
        test_y = measurements(5, config["steps"], 2, rng, fitted["mixing"])
        scored = score_ensemble(test_y, fitted, config)
        votes = []
        for position, (observer, low, _) in enumerate(fitted["model"].branches_):
            residual = window_residuals(test_y, observer, config["window"], position)
            filtered = trim_filter_warmup(butterworth_lowpass(
                residual, config["filter_cutoff"], config["filter_order"]),
                config["warmup"])
            decision = low.alarms(filtered).sum(axis=1) > config["k_low"]
            votes.append(decision)
        np.testing.assert_array_equal(scored["branch_low"], np.stack(votes))
        np.testing.assert_array_equal(scored["low_vote"], np.stack(votes).sum(axis=0) >= 2)


if __name__ == "__main__":
    unittest.main()
