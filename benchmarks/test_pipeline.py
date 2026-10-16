"""Focused checks of the trajectory and window-position implementations."""

import unittest
from argparse import Namespace
import numpy as np

from hierarchical_conformal import confidence_rank, fit_random_relu
from hierarchical_conformal.estimators import window_pairs, window_residuals
from hierarchical_conformal.utils import butterworth_lowpass, trim_filter_warmup
from benchmarks.simulate import attack_measurements
from benchmarks.window_ensemble import (default_config, fit_ensemble,
                                        measurements, score_ensemble)
from benchmarks.streaming_pipeline import randomized_window_vectors
from benchmarks.publication_design import (apply_application, apply_baseline,
                                           apply_ensemble, check_design)


class PipelineTests(unittest.TestCase):
    def test_randomized_window_observer_is_drawn_at_each_time_step(self):
        class ConstantObserver:
            def __init__(self, value):
                self.value = value

            def predict_flat(self, features):
                return np.full((len(features), 1), self.value)

        values = np.zeros((4, 20, 1))
        observers = [ConstantObserver(value) for value in (1., 2., 3.)]
        config = {"window": 3, "filter_cutoff": 0.2,
                  "filter_order": 2, "warmup": 2}
        residuals, choices = randomized_window_vectors(
            values, observers, config, np.random.default_rng(11))
        self.assertEqual(choices.shape, (4, 18))
        self.assertEqual(residuals.shape, (4, 16, 1))
        self.assertTrue(all(np.unique(row).size > 1 for row in choices))
        self.assertEqual(set(np.unique(choices)), {0, 1, 2})

    def test_publication_design_uses_full_trajectories_and_training_count(self):
        self.assertGreaterEqual(check_design(), 0.9995)
        baseline = apply_baseline(Namespace(n_train=60, m_low=200, m_high=200,
            n_test=80, steps=300, k_low=5, window=20, k_high=2,
            delta1_low=0.9, delta1_high=0.9, ell_low=0, ell_high=0))
        application = apply_application(Namespace(n_train=60, m_calibration=200,
            n_validation=80, steps=300, delta1=0.9, coverage=0.95))
        ensemble = apply_ensemble(default_config())
        for config in (vars(baseline), vars(application), ensemble):
            self.assertEqual(config["steps"], 10000)
            self.assertGreaterEqual(config["n_train"], 24100)
        for config in (vars(baseline), ensemble):
            self.assertEqual(config["m_low"], 24100)
            self.assertEqual(config["m_high"], 24100)
            self.assertEqual(config["ell_low"], 24098)
            self.assertEqual(config["ell_high"], 24098)
        self.assertEqual(application.m_calibration, 24100)
        self.assertEqual(application.ell, 24098)

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
