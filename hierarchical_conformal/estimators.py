"""Neural observers and supervised trajectory-pair builders."""

import numpy as np

class RandomReluEstimator:
    """One-hidden-layer ReLU neural observer with ridge-trained output weights.

    The hidden weights are randomly drawn; the output weights are fitted by
    ridge regression. This is a lightweight default, not a fully optimized
    neural network.
    """

    def __init__(self, input_mean, input_scale, w_hidden, b_hidden, w_out,
                 output_mean=None, output_scale=None):
        self.input_mean = input_mean
        self.input_scale = input_scale
        self.w_hidden = w_hidden
        self.b_hidden = b_hidden
        self.w_out = w_out
        self.output_mean = output_mean
        self.output_scale = output_scale

    def predict_flat(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.w_hidden.shape[0]:
            raise ValueError("x must be a 2D array with the fitted feature count")
        x_norm = (x - self.input_mean) / self.input_scale
        hidden = np.maximum(0.0, x_norm @ self.w_hidden + self.b_hidden)
        design = np.concatenate([hidden, np.ones((hidden.shape[0], 1))], axis=1)
        prediction = design @ self.w_out
        if self.output_mean is not None:
            prediction = prediction * self.output_scale + self.output_mean
        return prediction


def history_dataset(measurements: np.ndarray, history: int):
    """Build autoregressive pairs xi_t=(y_{t-1},...,y_{t-h}) and target y_t."""
    measurements = np.asarray(measurements, dtype=float)
    if measurements.ndim != 3 or min(measurements.shape) < 1 or not np.isfinite(measurements).all():
        raise ValueError("measurements must be a finite [trajectory,time,sensor] array")
    if history < 1 or history >= measurements.shape[1]:
        raise ValueError("history must satisfy 1 <= history < trajectory length")
    n, steps, dim = measurements.shape
    features = np.empty((n * (steps - history), history * dim), dtype=float)
    targets = np.empty((n * (steps - history), dim), dtype=float)
    row = 0
    for traj in range(n):
        for t in range(history, steps):
            features[row] = measurements[traj, t - history : t, :][::-1].reshape(-1)
            targets[row] = measurements[traj, t]
            row += 1
    return features, targets


def fit_random_relu(x: np.ndarray, y: np.ndarray, hidden: int,
                    ridge: float, rng) -> RandomReluEstimator:
    """Fit the same random-feature neural estimator to arbitrary supervised pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be two-dimensional with equal sample counts")
    if min(x.shape) < 1 or y.shape[1] < 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must contain finite, nonempty data")
    if hidden < 1 or ridge <= 0:
        raise ValueError("hidden and ridge must be positive")
    input_mean = x.mean(axis=0, keepdims=True)
    input_scale = x.std(axis=0, keepdims=True) + 1e-8
    output_mean = y.mean(axis=0, keepdims=True)
    output_scale = y.std(axis=0, keepdims=True) + 1e-8
    y_norm = (y - output_mean) / output_scale
    x_norm = (x - input_mean) / input_scale
    w_hidden = rng.normal(scale=1.0 / np.sqrt(x.shape[1]), size=(x.shape[1], hidden))
    b_hidden = rng.normal(scale=0.1, size=(hidden,))
    hidden_values = np.maximum(0.0, x_norm @ w_hidden + b_hidden)
    design = np.concatenate([hidden_values, np.ones((hidden_values.shape[0], 1))], axis=1)
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    w_out = np.linalg.solve(gram, design.T @ y_norm)
    return RandomReluEstimator(input_mean, input_scale, w_hidden, b_hidden,
                               w_out, output_mean, output_scale)


def train_random_relu_estimator(
    measurements: np.ndarray,
    history: int,
    hidden: int,
    ridge: float,
    rng,
) -> RandomReluEstimator:
    """Train a lightweight neural observer from nominal trajectories."""
    x, y = history_dataset(measurements, history)
    return fit_random_relu(x, y, hidden, ridge, rng)


def residual_trajectories(measurements: np.ndarray, estimator: RandomReluEstimator, history: int) -> np.ndarray:
    """Compute e_t = y_t - y_obs,t for every valid time in each trajectory."""
    x, y = history_dataset(measurements, history)
    residual_flat = y - estimator.predict_flat(x)
    n = measurements.shape[0]
    return residual_flat.reshape(n, measurements.shape[1] - history, measurements.shape[2])


class RandomFeatureObserver:
    """Trainable random-feature ReLU observer with a ridge-fitted output layer.

    This is a compact default. Any user observer with ``fit(X, y)`` and
    ``predict_flat(X)`` may be supplied to the detector classes instead.
    """

    def __init__(self, hidden=32, ridge=1e-5, seed=0, rng=None):
        self.hidden = hidden
        self.ridge = ridge
        self.seed = seed
        self.rng = rng
        self.estimator_ = None

    def fit(self, x, y):
        self.estimator_ = fit_random_relu(
            x, y, self.hidden, self.ridge,
            self.rng if self.rng is not None else np.random.default_rng(self.seed)
        )
        return self

    def predict_flat(self, x):
        if self.estimator_ is None:
            raise RuntimeError("observer must be fitted before prediction")
        return self.estimator_.predict_flat(x)


def windows_of(measurements, window):
    """Return complete windows with shape [trajectory, start, position, sensor]."""
    measurements = np.asarray(measurements, dtype=float)
    if measurements.ndim != 3 or not 2 <= window <= measurements.shape[1]:
        raise ValueError("measurements must be [trajectory,time,sensor] and window >= 2")
    view = np.lib.stride_tricks.sliding_window_view(measurements, window, axis=1)
    return np.moveaxis(view, -1, 2)


def window_pairs(measurements, window, position):
    """Return inputs and targets for a leave-one-position-out observer."""
    views = windows_of(measurements, window)
    if not 0 <= position < window:
        raise ValueError("position must lie inside the window")
    features = np.concatenate((views[:, :, :position], views[:, :, position + 1:]), axis=2)
    return (features.reshape(-1, (window - 1) * views.shape[-1]),
            views[:, :, position].reshape(-1, views.shape[-1]))


def window_residuals(measurements, observer, window, position):
    """Return one residual per complete window of each trajectory."""
    features, target = window_pairs(measurements, window, position)
    predicted = np.asarray(observer.predict_flat(features), dtype=float)
    if predicted.shape != target.shape or not np.isfinite(predicted).all():
        raise ValueError("observer prediction must match finite window targets")
    residual = target - predicted
    n, steps, dim = np.asarray(measurements).shape
    return residual.reshape(n, steps - window + 1, dim)

