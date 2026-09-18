"""Locally generated tasks and causally separated ridge evaluation."""

import numpy as np

from .config import integer
from .readout import MultiTargetRidgeReadout, RidgeReadout
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


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted q-values in the original hypothesis order."""
    values = np.asarray(p_values, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("p-values must be a nonempty finite vector")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0,1]")
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted


def capacity_family(name):
    if name.startswith("linear_"):
        return "linear"
    if name.startswith("quadratic_"):
        return "quadratic_self"
    if name.startswith("cross_"):
        return "cross_delay"
    raise ValueError(f"Unknown capacity target family: {name}")


def null_corrected_capacity(scores, null_scores, target_names=None, *, alpha=0.05):
    """Separate observed performance, null bias, significance, and capacity families."""
    observed, null = np.asarray(scores, float), np.asarray(null_scores, float)
    if observed.ndim != 1 or null.ndim != 2 or not len(null) or null.shape[1] != len(observed):
        raise ValueError("null_scores must be (permutations, targets)")
    if not np.isfinite(observed).all() or not np.isfinite(null).all() or not 0 < alpha < 1:
        raise ValueError("Capacity scores must be finite and alpha must lie in (0,1)")
    names = (
        tuple(f"target_{index}" for index in range(len(observed)))
        if target_names is None
        else tuple(target_names)
    )
    if len(names) != len(observed) or len(set(names)) != len(names):
        raise ValueError("Provide one distinct target name per score")
    correction = null.mean(axis=0)
    corrected = observed - correction
    p_values = np.array(
        [
            (1 + np.count_nonzero(null[:, i] >= score)) / (len(null) + 1)
            for i, score in enumerate(observed)
        ]
    )
    q_values = benjamini_hochberg(p_values)
    significant = q_values <= alpha
    null_totals = null.sum(axis=1)
    raw_total = float(observed.sum())
    targets = {}
    for index, name in enumerate(names):
        family = capacity_family(name) if target_names is not None else "unspecified"
        targets[name] = {
            "family": family,
            "observed_test_r2": float(observed[index]),
            "null_scores": null[:, index].tolist(),
            "null_mean": float(correction[index]),
            "null_median": float(np.median(null[:, index])),
            "null_std": float(null[:, index].std(ddof=1)) if len(null) > 1 else None,
            "null_interval95": np.quantile(null[:, index], [0.025, 0.975]).tolist(),
            "bias_corrected_estimate": float(corrected[index]),
            "empirical_p_value": float(p_values[index]),
            "fdr_q_value": float(q_values[index]),
            "significant_at_alpha": bool(significant[index]),
        }
    families = {}
    family_labels = sorted(set(row["family"] for row in targets.values()))
    for family in family_labels:
        indices = np.array([row["family"] == family for row in targets.values()])
        families[family] = {
            "raw_signed_total": float(observed[indices].sum()),
            "bias_corrected_signed_total": float(corrected[indices].sum()),
            "significant_capacity": float(corrected[indices & significant].sum()),
            "significant_components": int(np.count_nonzero(indices & significant)),
            "component_count": int(np.count_nonzero(indices)),
        }
    return {
        "estimator": "test R2 minus mean dataset-level surrogate-null test R2",
        "alpha": alpha,
        "multiple_comparison_correction": "Benjamini-Hochberg across declared targets",
        "targets": targets,
        "families": families,
        "raw_per_target": observed.tolist(),
        "raw_total": raw_total,
        "legacy_clipped_total": float(np.maximum(observed, 0).sum()),
        "null_total": float(correction.sum()),
        "null_corrected_per_target": corrected.tolist(),
        "null_corrected_total": float(corrected.sum()),
        "per_target_null_mean": correction.tolist(),
        "null_scores": null.tolist(),
        "p_values": p_values.tolist(),
        "q_values": q_values.tolist(),
        "significant_mask": significant.tolist(),
        "total_significant_capacity": float(corrected[significant].sum()),
        "null_total_interval95": np.quantile(null_totals, [0.025, 0.975]).tolist(),
        "permutation_p_value": float(
            (1 + np.count_nonzero(null_totals >= raw_total)) / (len(null_totals) + 1)
        ),
    }


def generate_capacity_nulls(
    inputs,
    delays,
    repetitions,
    *,
    method="independent_surrogate",
    seed=0,
    exclusion_window=None,
    block_size=None,
):
    """Create complete target matrices before splitting, with audit descriptors."""
    u = np.asarray(inputs, float)
    integer(repetitions, "repetitions", 1)
    if u.ndim != 1 or len(u) < 8 or not np.isfinite(u).all():
        raise ValueError("Capacity nulls require a finite one-dimensional dataset")
    observed, names = capacity_targets(u, delays)
    exclusion = max(delays + 1, int(exclusion_window or delays + 1))
    block = int(block_size or exclusion)
    rng = np.random.default_rng(seed)
    matrices, descriptors = [], []
    for repetition in range(repetitions):
        draw_seed = int(rng.integers(0, 2**63))
        draw = np.random.default_rng(draw_seed)
        if method == "circular_shift":
            allowed = np.arange(exclusion + 1, len(u) - exclusion)
            if not len(allowed):
                raise ValueError("Dataset too short for the declared circular-shift exclusion")
            displacement = int(draw.choice(allowed))
            transformed = np.roll(observed, displacement, axis=0)
            descriptor = {
                "method": method,
                "seed": draw_seed,
                "displacement": displacement,
                "minimum_displacement": exclusion + 1,
            }
        elif method == "block_permutation":
            blocks = [
                np.arange(start, min(start + block, len(u))) for start in range(0, len(u), block)
            ]
            order = draw.permutation(len(blocks))
            if np.array_equal(order, np.arange(len(blocks))):
                order = np.roll(order, 1)
            indices = np.concatenate([blocks[index] for index in order])
            transformed = observed[indices]
            descriptor = {
                "method": method,
                "seed": draw_seed,
                "block_size": block,
                "block_order": order.tolist(),
            }
        elif method == "independent_surrogate":
            surrogate = draw.uniform(float(u.min()), float(u.max()), len(u))
            transformed, surrogate_names = capacity_targets(surrogate, delays)
            if surrogate_names != names:
                raise RuntimeError("Surrogate target construction changed target order")
            descriptor = {
                "method": method,
                "seed": draw_seed,
                "distribution": "uniform over observed input range",
                "input_range": [float(u.min()), float(u.max())],
            }
        else:
            raise ValueError("Unknown capacity null method")
        matrices.append(transformed)
        descriptors.append({"repetition": repetition, **descriptor})
    return np.stack(matrices), names, descriptors


def assert_no_exact_target_leakage(features, targets, *, tolerance=1e-12):
    """Reject copied/affinely duplicated labels in a feature matrix.

    This deliberately narrow guard catches implementation leakage, not genuine
    predictive correlation from causally permitted history.
    """
    x, y = np.asarray(features, float), np.asarray(targets, float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("Features and targets must be aligned matrices")
    for feature_index in range(x.shape[1]):
        for target_index in range(y.shape[1]):
            feature, target = x[:, feature_index], y[:, target_index]
            if np.std(feature) <= tolerance or np.std(target) <= tolerance:
                continue
            correlation = abs(float(np.corrcoef(feature, target)[0, 1]))
            if correlation >= 1 - tolerance:
                raise ValueError(
                    f"exact target leakage: feature={feature_index}, target={target_index}"
                )


def narma(inputs, order=10):
    u = np.asarray(inputs, float)
    integer(order, "order", 2)
    if u.ndim != 1 or not np.isfinite(u).all() or np.any((u < 0) | (u > 0.5)):
        raise ValueError("NARMA inputs must lie in [0,0.5]")
    y = np.zeros(len(u) + 1)
    for t in range(order - 1, len(u)):
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * sum(y[t - order + 1 : t + 1])
            + 1.5 * u[t - order + 1] * u[t]
            + 0.1
        )
    if not np.isfinite(y).all():
        raise FloatingPointError("NARMA trajectory diverged")
    return y[1:]


def narma10(inputs):
    return narma(inputs, 10)


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
    mse = float(np.mean((y - p) ** 2))
    nmse = mse / variance
    correlation = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else 0.0
    return {
        "r2": 1 - nmse,
        "rmse": float(np.sqrt(mse)),
        "nmse": nmse,
        "nrmse": float(np.sqrt(nmse)),
        "mae": float(np.mean(abs(y - p))),
        "pearson_r": correlation,
        "nrmse_normalization": "target population standard deviation",
    }


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


def select_readout_multi(
    train,
    validation,
    regularizations=(1e-6, 1e-4, 1e-2, 1.0),
    *,
    input_access="reservoir_plus_input",
):
    return MultiTargetRidgeReadout(
        regularizations,
        input_access=input_access,
    ).fit_with_validation(train, validation)


class ClassicalFeatures:
    """Seeded ESN/RFF/input-only/delay controls with explicit equal histories."""

    def __init__(self, kind, dimension=14, seed=0, window=4):
        if kind not in (
            "esn",
            "rff",
            "input_only",
            "delay",
            "linear_ar",
            "polynomial_narx",
            "linear_state_space",
            "persistence",
        ):
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
        if self.kind == "linear_ar":
            return windows
        if self.kind == "persistence":
            return u[:, None]
        if self.kind == "rff":
            return np.sqrt(2 / self.dimension) * np.cos(windows @ self.mask.T + self.bias)
        if self.kind == "input_only":
            return np.tanh(u[:, None] * self.mask[:, 0] + self.bias)
        if self.kind == "polynomial_narx":
            columns = [windows, windows**2]
            columns.extend(
                (windows[:, left] * windows[:, right])[:, None]
                for left in range(self.window)
                for right in range(left + 1, self.window)
            )
            expanded = np.column_stack(columns)
            if expanded.shape[1] >= self.dimension:
                return expanded[:, : self.dimension]
            return np.column_stack(
                [expanded, np.zeros((len(expanded), self.dimension - expanded.shape[1]))]
            )
        state, rows = np.zeros(self.dimension), []
        for value in u:
            state = self.recurrent @ state + self.mask[:, 0] * value
            if self.kind == "esn":
                state = np.tanh(state + self.bias)
            rows.append(state.copy())
        return np.array(rows)

    def resource_profile(self):
        reservoir_parameters = {
            "esn": self.recurrent.size + self.mask[:, 0].size + self.bias.size,
            "rff": self.mask.size + self.bias.size,
            "input_only": self.mask[:, 0].size + self.bias.size,
            "linear_state_space": self.recurrent.size + self.mask[:, 0].size,
        }.get(self.kind, 0)
        return {
            "baseline_kind": self.kind,
            "feature_dimension": self.dimension
            if self.kind not in ("delay", "linear_ar")
            else self.window,
            "input_history": self.window,
            "reservoir_tunable_parameters": int(reservoir_parameters),
            "feature_normalization": "train-only in RidgeReadout",
        }
