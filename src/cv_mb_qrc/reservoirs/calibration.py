"""Predeclared synthetic calibration for capacity null estimators."""

import numpy as np

from .benchmarks import (
    assert_no_exact_target_leakage,
    capacity_targets,
    generate_capacity_nulls,
    metrics,
    null_corrected_capacity,
    select_readout_multi,
)
from .temporal import chronological_splits, delay_features


def _scores(inputs, features, targets, splits, regularizations):
    if np.max(np.std(features[splits["train"]], axis=0)) < 1e-12:
        prediction = np.repeat(
            targets[splits["train"]].mean(axis=0)[None, :],
            len(splits["test"]),
            axis=0,
        )
    else:
        model = select_readout_multi(
            (
                inputs[splits["train"]],
                features[splits["train"]],
                targets[splits["train"]],
            ),
            (
                inputs[splits["validation"]],
                features[splits["validation"]],
                targets[splits["validation"]],
            ),
            regularizations,
            input_access="reservoir_only",
        )
        prediction = model.predict(inputs[splits["test"]], features[splits["test"]])
    return np.asarray(
        [
            metrics(targets[splits["test"], column], prediction[:, column])["r2"]
            for column in range(targets.shape[1])
        ]
    )


def capacity_calibration(config):
    """Evaluate false positives and known/leaky controls without tuning after observation."""
    methods = config.get(
        "calibration_null_methods",
        ["circular_shift", "block_permutation", "independent_surrogate"],
    )
    seeds = config["calibration_dataset_seeds"]
    repetitions = config["calibration_null_permutations"]
    length = config["calibration_length"]
    delays = config["calibration_delays"]
    interval = config["acceptable_type1_interval"]
    alpha = config["capacity_alpha"]
    splits = chronological_splits(length, gap=delays + 2, washout=0)
    records = []
    for dataset_seed in seeds:
        rng = np.random.default_rng(dataset_seed)
        inputs = rng.uniform(-1, 1, length)
        targets, names = capacity_targets(inputs, delays)
        memory = delay_features(inputs, delays + 1)
        systems = {
            "random_independent": rng.normal(size=(length, 8)),
            "constant": np.ones((length, 1)),
            "shuffled_temporal": memory[rng.permutation(length)],
            "linear_memory": memory,
            "deliberately_leaky": targets.copy(),
        }
        leak_detected = False
        try:
            assert_no_exact_target_leakage(systems["deliberately_leaky"], targets)
        except ValueError:
            leak_detected = True
        for method_index, method in enumerate(methods):
            nulls, null_names, descriptors = generate_capacity_nulls(
                inputs,
                delays,
                repetitions,
                method=method,
                seed=config["permutation_seed"] + 10000 * dataset_seed + method_index,
                exclusion_window=config["null_exclusion_window"],
                block_size=config["null_block_size"],
            )
            if null_names != names:
                raise RuntimeError("Calibration target order changed")
            for system_name, features in systems.items():
                observed = _scores(inputs, features, targets, splits, config["regularizations"])
                null_scores = np.asarray(
                    [
                        _scores(inputs, features, null_target, splits, config["regularizations"])
                        for null_target in nulls
                    ]
                )
                report = null_corrected_capacity(
                    observed,
                    null_scores,
                    names,
                    alpha=alpha,
                )
                records.append(
                    {
                        "dataset_seed": dataset_seed,
                        "null_method": method,
                        "system": system_name,
                        "false_positive_system": system_name
                        in {"random_independent", "constant", "shuffled_temporal"},
                        "raw_total": report["raw_total"],
                        "bias_corrected_signed_total": report["null_corrected_total"],
                        "total_significant_capacity": report["total_significant_capacity"],
                        "significant_components": int(sum(report["significant_mask"])),
                        "component_count": len(names),
                        "minimum_q_value": min(report["q_values"]),
                        "leak_guard_detected": leak_detected
                        if system_name == "deliberately_leaky"
                        else None,
                        "null_descriptors": descriptors,
                    }
                )
    summaries = []
    for method in methods:
        for system in sorted({row["system"] for row in records}):
            group = [
                row for row in records if row["null_method"] == method and row["system"] == system
            ]
            discoveries = sum(row["significant_components"] for row in group)
            hypotheses = sum(row["component_count"] for row in group)
            rate = discoveries / hypotheses
            false_positive = group[0]["false_positive_system"]
            summaries.append(
                {
                    "null_method": method,
                    "system": system,
                    "dataset_count": len(group),
                    "hypothesis_count": hypotheses,
                    "significant_count": discoveries,
                    "significant_rate": rate,
                    "mean_bias_corrected_signed_total": float(
                        np.mean([row["bias_corrected_signed_total"] for row in group])
                    ),
                    "mean_significant_capacity": float(
                        np.mean([row["total_significant_capacity"] for row in group])
                    ),
                    "predeclared_type1_interval": interval if false_positive else None,
                    "calibration_pass": (
                        interval[0] <= rate <= interval[1] if false_positive else None
                    ),
                }
            )
    return {
        "records": records,
        "summary": summaries,
        "alpha": alpha,
        "multiple_testing": "Benjamini-Hochberg within each dataset capacity family",
        "replication_unit": "independent synthetic dataset seed",
        "acceptable_type1_interval_predeclared": interval,
    }
