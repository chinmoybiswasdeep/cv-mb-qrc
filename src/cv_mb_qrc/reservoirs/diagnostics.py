"""Empirical diagnostics, not proofs of the quantum echo-state property."""

import numpy as np


def feature_diagnostics(features):
    x = np.asarray(features, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Need at least two finite feature rows")
    centered = x - x.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    spectrum = s * s / (len(x) - 1)
    p = spectrum / spectrum.sum() if spectrum.sum() else spectrum
    p = p[p > 0]
    rank = float(np.exp(-np.sum(p * np.log(p)))) if len(p) else 0.0
    positive = s[s > max(s[0], 1.0) * 1e-12]
    return {
        "exact_rank": int(np.linalg.matrix_rank(centered)),
        "effective_rank": rank,
        "covariance_spectrum": spectrum.tolist(),
        "condition_number_nonzero": float(positive[0] / positive[-1]) if len(positive) else None,
        "rank_deficient": len(positive) < x.shape[1],
        "variance_collapsed_columns": int(np.sum(x.var(axis=0) < 1e-12)),
        "maximum_absolute_feature": float(np.max(abs(x))),
        "column_variance": x.var(axis=0).tolist(),
    }


def trace_distance(a, b):
    return float(np.sum(abs(np.linalg.eigvalsh(np.asarray(a) - np.asarray(b)))) / 2)


def contraction(factory, inputs, initial_states):
    models = [factory().reset(initial_state=s) for s in initial_states]
    if len(models) < 2:
        raise ValueError("Provide at least two initial states")
    curves, quantum = [], []
    for value in inputs:
        rows = [m.step(value).features for m in models]
        curves.append(
            [float(np.linalg.norm(a - b)) for i, a in enumerate(rows) for b in rows[i + 1 :]]
        )
        if all(isinstance(m.state, np.ndarray) and m.state.ndim == 2 for m in models):
            quantum.append(
                [
                    trace_distance(a.state, b.state)
                    for i, a in enumerate(models)
                    for b in models[i + 1 :]
                ]
            )
    return {
        "metric": "feature Euclidean distance",
        "distances": curves,
        "trace_distances": quantum if quantum else None,
    }


def fading_memory(factory, inputs, *, perturbation=0.01, at=0):
    u = np.asarray(inputs, float)
    if not 0 <= at < len(u) or not np.isfinite(perturbation) or perturbation == 0:
        raise ValueError("Invalid perturbation")
    v = u.copy()
    v[at] += perturbation
    a, b = factory(), factory()
    x, y = a.run_sequence(u).features, b.run_sequence(v).features
    return {
        "at": at,
        "perturbation": perturbation,
        "feature_distance": np.linalg.norm(x - y, axis=1).tolist(),
    }


def bootstrap_summary(values, *, seed=0, draws=2000, scientific=True):
    a = np.asarray(values, float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Need finite independent observations")
    if len(a) == 1:
        if scientific:
            raise ValueError("Scientific uncertainty needs at least two independent observations")
        value = float(a[0])
        return {
            "mean": value,
            "std": None,
            "median": value,
            "bootstrap95": None,
            "values": [value],
            "bootstrap_seed": seed,
            "scientific_uncertainty_valid": False,
        }
    rng = np.random.default_rng(seed)
    means = rng.choice(a, (draws, len(a)), replace=True).mean(axis=1)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)),
        "median": float(np.median(a)),
        "bootstrap95": np.quantile(means, [0.025, 0.975]).tolist(),
        "values": a.tolist(),
        "bootstrap_seed": seed,
        "scientific_uncertainty_valid": True,
    }


def hierarchical_bootstrap_summary(
    values,
    dataset_seeds,
    reservoir_seeds,
    *,
    seed=0,
    draws=2000,
    scientific=True,
):
    """Bootstrap dataset seeds, then reservoir observations within each dataset."""
    a = np.asarray(values, float)
    datasets = np.asarray(dataset_seeds)
    reservoirs = np.asarray(reservoir_seeds)
    if (
        a.ndim != 1
        or not len(a)
        or datasets.shape != a.shape
        or reservoirs.shape != a.shape
        or not np.isfinite(a).all()
    ):
        raise ValueError("Need aligned finite values and seed roles")
    unique_datasets = np.unique(datasets)
    if len(unique_datasets) < 2:
        if scientific:
            raise ValueError("Hierarchical scientific uncertainty needs at least two datasets")
        result = bootstrap_summary(a, seed=seed, draws=draws, scientific=False)
        result["scientific_uncertainty_valid"] = False
        return result
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    groups = {key: a[datasets == key] for key in unique_datasets}
    for draw in range(draws):
        selected = rng.choice(unique_datasets, len(unique_datasets), replace=True)
        nested = [rng.choice(groups[key], len(groups[key]), replace=True) for key in selected]
        means[draw] = np.concatenate(nested).mean()
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)),
        "median": float(np.median(a)),
        "bootstrap95": np.quantile(means, [0.025, 0.975]).tolist(),
        "values": a.tolist(),
        "dataset_seeds": datasets.tolist(),
        "reservoir_seeds": reservoirs.tolist(),
        "bootstrap_seed": seed,
        "bootstrap_method": "dataset_then_reservoir_within_dataset",
        "scientific_uncertainty_valid": True,
    }


def paired_hierarchical_summary(
    left,
    right,
    dataset_seeds,
    reservoir_seeds,
    *,
    seed=0,
    draws=2000,
    scientific=True,
):
    """Summarize paired method differences on the declared replication hierarchy."""
    left_values, right_values = np.asarray(left, float), np.asarray(right, float)
    if left_values.shape != right_values.shape:
        raise ValueError("Paired observations must have identical shapes")
    differences = left_values - right_values
    summary = hierarchical_bootstrap_summary(
        differences,
        dataset_seeds,
        reservoir_seeds,
        seed=seed,
        draws=draws,
        scientific=scientific,
    )
    standard_deviation = differences.std(ddof=1) if len(differences) > 1 else np.nan
    summary.update(
        {
            "paired_differences": differences.tolist(),
            "mean_paired_difference": float(differences.mean()),
            "median_paired_difference": float(np.median(differences)),
            "paired_effect_size_dz": (
                float(differences.mean() / standard_deviation)
                if np.isfinite(standard_deviation) and standard_deviation > 0
                else None
            ),
            "left_win_fraction": float(np.mean(differences > 0)),
        }
    )
    return summary
