"""Locally generated tasks and causally separated ridge evaluation."""

import numpy as np

from .config import integer
from .readout import RidgeReadout
from .temporal import delay_features


def capacity_targets(inputs, delays=5):
    integer(delays, "delays")
    u = np.asarray(inputs, float)
    history = delay_features(u, delays + 1)
    columns, names = [], []
    for lag in range(1, delays + 1):
        x = history[:, lag]
        columns.extend([np.sqrt(3) * x, np.sqrt(5) * (3 * x * x - 1) / 2])
        names.extend([f"linear_{lag}", f"quadratic_{lag}"])
    for a in range(1, delays + 1):
        for b in range(a + 1, delays + 1):
            columns.append(3 * history[:, a] * history[:, b])
            names.append(f"cross_{a}_{b}")
    return np.column_stack(columns), tuple(names)


def null_corrected_capacity(scores, null_scores):
    """Report raw, legacy clipped, and permutation-null-corrected capacity."""
    observed, null = np.asarray(scores, float), np.asarray(null_scores, float)
    if observed.ndim != 1 or null.ndim != 2 or not len(null) or null.shape[1] != len(observed):
        raise ValueError("null_scores must be (permutations, targets)")
    correction = null.mean(axis=0)
    corrected = observed - correction
    null_totals = null.sum(axis=1)
    raw_total = float(observed.sum())
    return {
        "raw_per_target": observed.tolist(),
        "raw_total": raw_total,
        "legacy_clipped_total": float(np.maximum(observed, 0).sum()),
        "null_total": float(correction.sum()),
        "null_corrected_per_target": corrected.tolist(),
        "null_corrected_total": float(corrected.sum()),
        "per_target_null_mean": correction.tolist(),
        "null_scores": null.tolist(),
        "null_total_interval95": np.quantile(null_totals, [0.025, 0.975]).tolist(),
        "permutation_p_value": float(
            (1 + np.count_nonzero(null_totals >= raw_total)) / (len(null_totals) + 1)
        ),
    }


def narma10(inputs):
    u = np.asarray(inputs, float)
    if u.ndim != 1 or not np.isfinite(u).all() or np.any((u < 0) | (u > 0.5)):
        raise ValueError("NARMA10 inputs must lie in [0,0.5]")
    y = np.zeros(len(u) + 1)
    for t in range(9, len(u)):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * sum(y[t - 9 : t + 1]) + 1.5 * u[t - 9] * u[t] + 0.1
    if not np.isfinite(y).all():
        raise FloatingPointError("NARMA trajectory diverged")
    return y[1:]


def mackey_glass(length, *, delay=17, dt=0.1, burnin=1000, initial_value=1.2):
    """Euler: dx/dt=.2*x(t-17)/(1+x(t-17)^10)-.1*x; initial history 1.2."""
    integer(length, "length")
    if not np.isfinite([delay, dt, initial_value]).all() or delay <= 0 or dt <= 0:
        raise ValueError("Positive delay and dt required")
    lag = int(round(delay / dt))
    stride = int(round(1 / dt))
    if lag < 1 or stride < 1:
        raise ValueError("dt must resolve delay and unit sampling interval")
    initial_value = float(initial_value)
    x = np.full(lag + (length + burnin) * stride + 1, initial_value)
    for t in range(lag, len(x) - 1):
        delayed = x[t - lag]
        x[t + 1] = x[t] + dt * (0.2 * delayed / (1 + delayed**10) - 0.1 * x[t])
    return x[lag + burnin * stride : lag + (burnin + length) * stride : stride]


def metrics(target, prediction):
    y, p = np.asarray(target, float), np.asarray(prediction, float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Finite predictions/targets of identical shape required")
    variance = float(np.mean((y - y.mean()) ** 2))
    if variance <= 0:
        raise ValueError("R2 undefined for constant target")
    nmse = float(np.mean((y - p) ** 2) / variance)
    return {"r2": 1 - nmse, "nmse": nmse, "nrmse": float(np.sqrt(nmse))}


def forecast_targets(series, horizons):
    """Align x[t+h] targets with information ending at x[t]."""
    values = np.asarray(series, float)
    horizons = tuple(int(horizon) for horizon in horizons)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Forecast series must be one-dimensional and finite")
    if not horizons or min(horizons) < 1 or len(set(horizons)) != len(horizons):
        raise ValueError("Forecast horizons must be distinct positive integers")
    length = len(values) - max(horizons)
    if length < 1:
        raise ValueError("Forecast horizons exceed the series length")
    return values[:length], np.column_stack(
        [values[horizon : horizon + length] for horizon in horizons]
    )


def closed_loop_forecast(predict_one, context, truth, *, threshold=1.0):
    """Recursively feed one-step predictions back into a causal predictor.

    ``predict_one`` receives only the current context and returns the next
    scalar. The normalized absolute error uses the fixed truth standard
    deviation; no future target is supplied to the predictor.
    """
    history, expected = list(np.asarray(context, float)), np.asarray(truth, float)
    if not history or expected.ndim != 1 or not len(expected):
        raise ValueError("Nonempty context and one-dimensional truth are required")
    if not np.isfinite(history).all() or not np.isfinite(expected).all() or threshold <= 0:
        raise ValueError("Closed-loop inputs and threshold must be finite")
    scale = float(expected.std())
    if scale <= 0:
        raise ValueError("Normalized rollout error requires nonconstant truth")
    prediction_values = []
    for _ in expected:
        prediction = float(predict_one(np.asarray(history, float)))
        prediction_values.append(prediction)
        history.append(prediction)
    predictions = np.asarray(prediction_values)
    finite = np.isfinite(predictions)
    errors = np.full(len(expected), np.inf)
    errors[finite] = abs(predictions[finite] - expected[finite]) / scale
    exceeded = np.flatnonzero(errors > threshold)
    return {
        "predictions": predictions.tolist(),
        "normalized_absolute_error": errors.tolist(),
        "valid_prediction_time": int(exceeded[0]) if len(exceeded) else len(expected),
        "diverged": bool(not finite.all()),
        "threshold": threshold,
    }


def select_readout(
    train,
    validation,
    regularizations=(1e-6, 1e-4, 1e-2, 1.0),
    *,
    input_access="reservoir_plus_input",
):
    """Only train/validation are accepted; test labels cannot enter selection."""
    candidates = []
    for alpha in regularizations:
        model = RidgeReadout(alpha, input_access=input_access).fit(*train)
        prediction = model.predict(*validation[:2])
        score = float(np.mean((prediction - validation[2]) ** 2))
        candidates.append((score, model))
    return min(candidates, key=lambda item: item[0])[1]


class ClassicalFeatures:
    """Seeded ESN/RFF/input-only/delay controls with explicit equal histories."""

    def __init__(self, kind, dimension=14, seed=0, window=4):
        if kind not in ("esn", "rff", "input_only", "delay", "persistence"):
            raise ValueError("Unknown baseline")
        integer(dimension, "dimension")
        integer(window, "window")
        dimension, window = int(dimension), int(window)
        self.kind, self.dimension, self.window = kind, dimension, window
        rng = np.random.default_rng(seed)
        self.mask = rng.normal(size=(dimension, window))
        self.bias = rng.uniform(-np.pi, np.pi, dimension)
        self.recurrent = rng.normal(size=(dimension, dimension)) / np.sqrt(dimension)
        recurrent_norm = float(np.linalg.norm(self.recurrent, 2))
        self.recurrent *= 0.8 / max(recurrent_norm, 1e-12)

    def transform(self, inputs):
        u = np.asarray(inputs, float)
        windows = delay_features(u, self.window)
        if self.kind == "delay":
            return windows
        if self.kind == "persistence":
            return u[:, None]
        if self.kind == "rff":
            return np.sqrt(2 / self.dimension) * np.cos(windows @ self.mask.T + self.bias)
        if self.kind == "input_only":
            return np.tanh(u[:, None] * self.mask[:, 0] + self.bias)
        state, rows = np.zeros(self.dimension), []
        for value in u:
            state = np.tanh(self.recurrent @ state + self.mask[:, 0] * value + self.bias)
            rows.append(state.copy())
        return np.array(rows)
