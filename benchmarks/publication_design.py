"""Shared trajectory counts and conformal targets for the paper experiments.

Training, low-level calibration, and high-level calibration are disjoint sets.
The two calibration sets can be reused across the window-position observers,
because each observer is calibrated using complete trajectories.
"""

from hierarchical_conformal import beta_confidence, confidence_rank


STEPS = 10_000
CALIBRATION_TRAJECTORIES = 24_100
MIN_TRAINING_TRAJECTORIES = CALIBRATION_TRAJECTORIES
DELTA1 = 0.9995
DELTA2_TARGET = 0.9995
RANK = 24_098
LOW_VIOLATION_BUDGET = 500


def check_design():
    """Check that the paper's rank meets its stated confidence target."""
    if confidence_rank(CALIBRATION_TRAJECTORIES, DELTA1, DELTA2_TARGET) != RANK:
        raise RuntimeError("publication rank is inconsistent with the beta-CDF target")
    return beta_confidence(DELTA1, RANK, CALIBRATION_TRAJECTORIES)


def apply_baseline(args):
    """Apply the paper's trajectory counts to the baseline experiments."""
    check_design()
    args.steps = STEPS
    args.n_train = max(args.n_train, MIN_TRAINING_TRAJECTORIES)
    args.m_low = CALIBRATION_TRAJECTORIES
    args.m_high = CALIBRATION_TRAJECTORIES
    args.k_low = LOW_VIOLATION_BUDGET
    args.window = 500
    args.k_high = 50
    args.k_ablation_values = "100,250,500,750,1000"
    args.n_test = max(args.n_test, 1000)
    args.delta1_low = DELTA1
    args.delta1_high = DELTA1
    args.ell_low = RANK
    args.ell_high = RANK
    return args


def apply_application(args):
    """Apply the paper's trajectory counts to each application experiment."""
    check_design()
    args.steps = STEPS
    args.n_train = max(args.n_train, MIN_TRAINING_TRAJECTORIES)
    args.m_calibration = CALIBRATION_TRAJECTORIES
    args.delta1 = DELTA1
    args.ell = RANK
    args.coverage = 0.95
    args.k_low = LOW_VIOLATION_BUDGET
    args.n_validation = max(args.n_validation, 1000)
    return args


def apply_ensemble(config):
    """Apply the paper's trajectory counts to every observer branch."""
    check_design()
    updated = dict(config)
    updated.update(
        steps=STEPS,
        n_train=max(config["n_train"], MIN_TRAINING_TRAJECTORIES),
        m_low=CALIBRATION_TRAJECTORIES,
        m_high=CALIBRATION_TRAJECTORIES,
        k_low=LOW_VIOLATION_BUDGET,
        n_test=max(config["n_test"], 1000),
        delta1_low=DELTA1,
        delta1_high=DELTA1,
        ell_low=RANK,
        ell_high=RANK,
        max_training_pairs=None,
    )
    return updated
