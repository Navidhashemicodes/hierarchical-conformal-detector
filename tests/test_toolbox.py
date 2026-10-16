"""Tests of the reusable API on trajectories unrelated to the paper simulator."""

import unittest

import numpy as np

from hierarchical_conformal import (HighLevelDetector, HierarchicalDetector,
                                    LowLevelDetector, RandomizedWindowDetector,
                                    WindowEnsembleDetector, kth_max)


class ZeroObserver:
    def fit(self, x, y):
        self.output_dim = y.shape[1]
        return self

    def predict_flat(self, x):
        return np.zeros((len(x), self.output_dim))


class ToolboxTests(unittest.TestCase):
    def test_low_level_counts_over_full_valid_trajectory(self):
        detector = LowLevelDetector(k=2).fit(
            np.zeros((3, 6, 1)), np.zeros((4, 6, 1)))
        detector.center_ = np.zeros(1)
        detector.inverse_ = np.eye(1)
        detector.threshold_ = 1.0
        test = np.array([[[2], [2], [0], [0], [0], [0]],
                         [[2], [2], [0], [0], [2], [0]]], dtype=float)
        np.testing.assert_array_equal(detector.alarms(test).sum(axis=1), [2, 3])
        np.testing.assert_array_equal(detector.predict(test), [False, True])

    def test_high_level_uses_kth_maximum_window_rate(self):
        detector = HighLevelDetector(window=2, k=2).fit(np.zeros((4, 5)))
        detector.threshold_ = 0.4
        alarms = np.array([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
        np.testing.assert_array_equal(detector.predict(alarms), [True, False])
        np.testing.assert_allclose(detector.scores(alarms), [0.5, 0.0])
        self.assertEqual(kth_max(np.array([1., 3., 2.]), 2), 2.)

    def test_hierarchical_accepts_custom_observer(self):
        rng = np.random.default_rng(4)
        train, low, high, test = [rng.normal(size=(n, 30, 2))
                                  for n in (6, 8, 8, 3)]
        detector = HierarchicalDetector(
            history=2, k_low=2, window=4, k_high=2,
            observer=ZeroObserver(), warmup=2).fit(train, low, high)
        scored = detector.score(test)
        self.assertEqual(scored["alarms"].shape, (3, 26))
        np.testing.assert_array_equal(
            scored["low_alarm"], scored["alarms"].sum(axis=1) > 2)
        self.assertEqual(detector.predict(test)[1].shape, (3,))

    def test_window_ensemble_votes_at_both_levels(self):
        rng = np.random.default_rng(5)
        train, low, high, test = [rng.normal(size=(n, 30, 2))
                                  for n in (6, 8, 8, 3)]
        model = WindowEnsembleDetector(
            window=3, k_low=2, rate_window=4, k_high=2,
            observer_factory=lambda position: ZeroObserver(),
            warmup=2).fit(train, low, high)
        scored = model.score(test)
        self.assertEqual(scored["branch_low"].shape, (3, 3))
        np.testing.assert_array_equal(
            scored["branch_low"], scored["branch_alarm_counts"] > 2)
        np.testing.assert_array_equal(
            scored["low_vote"], scored["branch_low"].sum(axis=0) >= 2)
        np.testing.assert_array_equal(
            scored["high_vote"], scored["branch_high"].sum(axis=0) >= 2)

    def test_randomized_window_detector_returns_single_low_high_decisions(self):
        rng = np.random.default_rng(6)
        train, low, high, test = [rng.normal(size=(n, 30, 2))
                                  for n in (6, 8, 8, 3)]
        detector = RandomizedWindowDetector(
            window=3, k_low=2, rate_window=4, k_high=2,
            observer_factory=lambda position: ZeroObserver(),
            delta1_low=0.8, delta1_high=0.8, warmup=2, seed=9,
        ).fit(train, low, high)
        scored = detector.score(test)
        self.assertEqual(scored["distances"].shape, (3, 26))
        self.assertEqual(scored["alarms"].shape, (3, 26))
        self.assertEqual(scored["low_alarm"].shape, (3,))
        self.assertEqual(scored["high_alarm"].shape, (3,))
        np.testing.assert_array_equal(
            scored["low_alarm"], scored["alarms"].sum(axis=1) > 2)
        self.assertEqual(len(detector.observers_), 3)

    def test_randomized_window_detector_requires_separate_observers(self):
        shared = ZeroObserver()
        data = np.zeros((4, 12, 1))
        detector = RandomizedWindowDetector(
            window=3, k_low=1, rate_window=2, k_high=1,
            observer_factory=lambda position: shared, warmup=1,
        )
        with self.assertRaisesRegex(ValueError, "separate observer"):
            detector.fit(data, data, data)

    def test_mismatched_trajectory_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same time and sensor dimensions"):
            HierarchicalDetector(2, 2, 4, 2).fit(
                np.zeros((4, 20, 2)), np.zeros((4, 21, 2)),
                np.zeros((4, 20, 2)))


if __name__ == "__main__":
    unittest.main()
