"""Synthetic black-box trajectory generator for detector experiments."""

import numpy as np


def make_mixing_matrix(dim: int, rng) -> np.ndarray:
    """Create a fixed sensor-correlation matrix."""
    raw = rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(raw)
    scales = np.linspace(0.6, 1.8, dim)
    return q @ np.diag(scales)


def nominal_deltas(
    n: int,
    steps: int,
    dim: int,
    rng,
    rho: float = 0.86,
    burst_probability: float = 0.015,
    mixing=None,
) -> np.ndarray:
    """Generate dependent non-iid nominal measurement increments.

    The detector does not know this generator. The dependence and intermittent
    volatility bursts are included so the high-level calibration must account
    for a non-iid alarm process.
    """
    mix = make_mixing_matrix(dim, rng) if mixing is None else np.asarray(mixing)
    if mix.shape != (dim, dim):
        raise ValueError("mixing must have shape (dim, dim)")
    deltas = np.zeros((n, steps, dim), dtype=np.float64)
    state = rng.normal(scale=0.2, size=(n, dim))
    volatility = np.ones((n, 1), dtype=np.float64)

    for t in range(steps):
        shock = rng.normal(size=(n, dim))
        burst = rng.random((n, 1)) < burst_probability
        volatility = 0.98 * volatility + 0.02
        volatility = np.where(burst, volatility + rng.uniform(1.5, 3.0, size=(n, 1)), volatility)
        state = rho * state + 0.35 * shock
        raw = volatility * shock + state
        deltas[:, t, :] = raw @ mix.T
    return deltas


def attack_measurements(nominal: np.ndarray, kind: str, rng,
                        start_fraction: float = 0.45, strength: float = 3.0) -> np.ndarray:
    """Apply signal-level anomalies to recorded measurements after onset."""
    attacked = np.array(nominal, copy=True)
    n, steps, dim = attacked.shape
    start = int(start_fraction * steps)
    length = steps - start
    direction = rng.normal(size=(n, 1, dim))
    direction /= np.linalg.norm(direction, axis=2, keepdims=True) + 1e-12
    if kind == "bias":
        attacked[:, start:] += strength * direction
    elif kind == "drift":
        ramp = np.linspace(0.0, 1.0, length)[None, :, None]
        attacked[:, start:] += strength * ramp * direction
    elif kind == "replay":
        source = attacked[:, max(0, start - length):start]
        repeats = int(np.ceil(length / source.shape[1]))
        attacked[:, start:] = np.tile(source, (1, repeats, 1))[:, :length]
    elif kind == "noise_burst":
        block = max(5, length // 8)
        for j in range(n):
            begin = start + int(rng.integers(0, max(1, length - block + 1)))
            attacked[j, begin:begin + block] += rng.normal(
                scale=1.5 * strength, size=(block, dim))
    elif kind == "surge":
        ramp = np.linspace(0.2, 1.0, length)[None, :, None]
        attacked[:, start:] += strength * ramp * direction
    elif kind == "clustered":
        block = max(5, length // 12)
        for j in range(n):
            begin = start + int(rng.integers(0, max(1, length - block + 1)))
            attacked[j, begin:begin + block] += 2.5 * strength * direction[j]
    elif kind == "zero_alarm":
        # Replace the post-onset segment by a held measurement.
        attacked[:, start:] = attacked[:, start - 1:start]
    elif kind == "hidden_rate":
        block = max(5, length // 20)
        for j in range(n):
            for _ in range(3):
                begin = start + int(rng.integers(0, max(1, length - block + 1)))
                attacked[j, begin:begin + block] += 1.7 * strength * direction[j]
    else:
        raise ValueError("unknown attack kind: {}".format(kind))
    return attacked


def attack_deltas(
    nominal: np.ndarray,
    kind,
    rng,
    start_fraction: float = 0.45,
    strength: float = 3.0,
) -> np.ndarray:
    """Inject synthetic attack patterns into nominal increments.

    These are intentionally black-box signal manipulations for testing the
    detectors, not plant-specific attacks.
    """
    attacked = nominal.copy()
    n, steps, dim = attacked.shape
    start = int(start_fraction * steps)
    length = max(1, steps - start)

    if kind == "bias":
        direction = rng.normal(size=(n, 1, dim))
        direction /= np.linalg.norm(direction, axis=2, keepdims=True) + 1e-12
        attacked[:, start:, :] += strength * direction
    elif kind == "drift":
        direction = rng.normal(size=(n, 1, dim))
        direction /= np.linalg.norm(direction, axis=2, keepdims=True) + 1e-12
        ramp = np.linspace(0.0, 1.0, length)[None, :, None]
        attacked[:, start:, :] += strength * ramp * direction
    elif kind == "replay":
        replay_len = max(10, min(start, length))
        source_start = max(0, start - replay_len)
        source = attacked[:, source_start:start, :]
        if source.shape[1] == 0:
            source = attacked[:, :1, :]
        repeats = int(np.ceil(length / source.shape[1]))
        replay = np.tile(source, (1, repeats, 1))[:, :length, :]
        attacked[:, start:, :] = replay
    elif kind == "noise_burst":
        burst_scale = strength * 1.5
        block = max(5, length // 8)
        for i in range(n):
            block_start = start + rng.integers(0, max(1, length - block))
            attacked[i, block_start : block_start + block, :] += rng.normal(
                scale=burst_scale, size=(block, dim)
            )
    elif kind == "surge":
        direction = rng.normal(size=(n, 1, dim))
        direction /= np.linalg.norm(direction, axis=2, keepdims=True) + 1e-12
        ramp = np.linspace(0.2, 1.0, length)[None, :, None]
        attacked[:, start:, :] += strength * ramp * direction
    elif kind == "clustered":
        block = max(5, length // 12)
        for i in range(n):
            block_start = start + rng.integers(0, max(1, length - block))
            direction = rng.normal(size=(dim,))
            direction /= np.linalg.norm(direction) + 1e-12
            attacked[i, block_start : block_start + block, :] += strength * 2.5 * direction
    elif kind == "zero_alarm":
        # Quiet the signal after the attack starts. This imitates attacks that
        # suppress the low-level detector rather than causing large deviations.
        attacked[:, start:, :] *= 0.05
    elif kind == "hidden_rate":
        # Keep the total amount of abnormal behavior moderate but concentrate it
        # into windows, challenging detectors that monitor temporal patterns.
        block = max(5, length // 20)
        for i in range(n):
            for _ in range(3):
                block_start = start + rng.integers(0, max(1, length - block))
                direction = rng.normal(size=(dim,))
                direction /= np.linalg.norm(direction) + 1e-12
                attacked[i, block_start : block_start + block, :] += strength * 1.7 * direction
    else:
        raise ValueError(f"unknown attack kind: {kind}")

    return attacked


