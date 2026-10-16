"""Memory-bounded trajectory processing in five reproducible outer batches."""

import numpy as np

from hierarchical_conformal.estimators import RandomReluEstimator


OUTER_BATCHES = 5
PAIR_CHUNK_TRAJECTORIES = 64


def five_batches(n):
    """Yield five (start, stop) ranges, distributing any remainder up front."""
    if n < OUTER_BATCHES:
        raise ValueError("at least five trajectories are required")
    quotient, remainder = divmod(n, OUTER_BATCHES)
    start = 0
    for index in range(OUTER_BATCHES):
        stop = start + quotient + (index < remainder)
        yield start, stop
        start = stop


def history_pairs_vectorized(measurements, history):
    """Build autoregressive pairs without Python loops over time samples."""
    values = np.asarray(measurements, dtype=float)
    if values.ndim != 3 or not 1 <= history < values.shape[1]:
        raise ValueError("measurements must be [trajectory,time,sensor] and history valid")
    _, steps, _ = values.shape
    features = np.stack(
        [values[:, history - offset - 1:steps - offset - 1, :]
         for offset in range(history)], axis=2)
    targets = values[:, history:, :]
    return features.reshape(-1, features.shape[2] * features.shape[3]), targets.reshape(-1, targets.shape[-1])


def window_pairs_vectorized(measurements, window, position):
    """Create leave-one-position-out pairs for a bounded trajectory slice."""
    values = np.asarray(measurements, dtype=float)
    if values.ndim != 3 or not 2 <= window <= values.shape[1]:
        raise ValueError("measurements must be [trajectory,time,sensor] and window valid")
    if not 0 <= position < window:
        raise ValueError("position must lie inside the window")
    view = np.lib.stride_tricks.sliding_window_view(values, window, axis=1)
    view = np.moveaxis(view, -1, 2)
    features = np.concatenate((view[:, :, :position], view[:, :, position + 1:]), axis=2)
    targets = view[:, :, position]
    return features.reshape(-1, features.shape[2] * features.shape[3]), targets.reshape(-1, targets.shape[-1])


def fit_random_relu_streaming(pair_batches, hidden, ridge, rng):
    """Fit random-feature ridge regression from a repeatable batch factory.

    ``pair_batches`` is called twice. The first pass accumulates global feature
    and target moments. The second accumulates the ridge normal equations, so
    no full trajectory design matrix is materialized.
    """
    x_sum = x_sq = y_sum = y_sq = None
    rows = 0
    for x, y in pair_batches():
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or not len(x):
            raise ValueError("each pair batch must contain matching nonempty matrices")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("training pairs must be finite")
        if x_sum is None:
            x_sum = np.zeros(x.shape[1]); x_sq = np.zeros(x.shape[1])
            y_sum = np.zeros(y.shape[1]); y_sq = np.zeros(y.shape[1])
        if x.shape[1] != len(x_sum) or y.shape[1] != len(y_sum):
            raise ValueError("pair batch dimensions changed during training")
        x_sum += x.sum(axis=0); x_sq += np.einsum("ij,ij->j", x, x)
        y_sum += y.sum(axis=0); y_sq += np.einsum("ij,ij->j", y, y)
        rows += len(x)
    if rows == 0:
        raise ValueError("training data are empty")

    x_mean = x_sum / rows
    y_mean = y_sum / rows
    x_scale = np.sqrt(np.maximum(x_sq / rows - x_mean * x_mean, 0.0)) + 1e-8
    y_scale = np.sqrt(np.maximum(y_sq / rows - y_mean * y_mean, 0.0)) + 1e-8
    w_hidden = rng.normal(scale=1.0 / np.sqrt(len(x_mean)), size=(len(x_mean), hidden))
    b_hidden = rng.normal(scale=0.1, size=hidden)
    gram = np.zeros((hidden + 1, hidden + 1), dtype=float)
    cross = np.zeros((hidden + 1, len(y_mean)), dtype=float)
    for x, y in pair_batches():
        x_norm = (np.asarray(x, dtype=float) - x_mean) / x_scale
        y_norm = (np.asarray(y, dtype=float) - y_mean) / y_scale
        h = np.maximum(0.0, x_norm @ w_hidden + b_hidden)
        h_sum = h.sum(axis=0)
        design_cross = h.T @ h
        gram[:-1, :-1] += design_cross
        gram[:-1, -1] += h_sum
        gram[-1, :-1] += h_sum
        gram[-1, -1] += len(h)
        cross[:-1] += h.T @ y_norm
        cross[-1] += y_norm.sum(axis=0)
    gram.flat[::len(gram) + 1] += ridge
    w_out = np.linalg.solve(gram, cross)
    return RandomReluEstimator(
        x_mean[None, :], x_scale[None, :], w_hidden, b_hidden, w_out,
        y_mean[None, :], y_scale[None, :])


def observer_residuals_vectorized(measurements, estimator, history):
    """Compute one-step residuals for a small batch of complete trajectories."""
    x, y = history_pairs_vectorized(measurements, history)
    return (y - estimator.predict_flat(x)).reshape(
        len(measurements), measurements.shape[1] - history, measurements.shape[2])


def trajectory_slices(n, chunk_size=PAIR_CHUNK_TRAJECTORIES):
    """Yield small inner slices for pair construction and observer scoring."""
    for start in range(0, n, chunk_size):
        yield slice(start, min(n, start + chunk_size))


def accumulate_shape(sum_vector, cross_product, samples):
    """Update population moments from samples shaped [trajectory,time,sensor]."""
    flat = np.asarray(samples, dtype=float).reshape(-1, samples.shape[-1])
    sum_vector += flat.sum(axis=0)
    cross_product += flat.T @ flat
    return sum_vector, cross_product, len(flat)


def shape_from_moments(sum_vector, cross_product, count, ridge):
    """Recover mean and regularized population covariance from accumulated moments."""
    if count < 1:
        raise ValueError("cannot estimate shape from zero score vectors")
    center = sum_vector / count
    shape = (cross_product - count * np.outer(center, center)) / max(1, count - 1)
    shape += ridge * np.eye(len(center))
    return center, shape, np.linalg.pinv(shape)
