"""Manifest-driven temporal experiments with explicit optional stages."""

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import psutil

from cv_mb_qrc.reservoirs import (
    CVConfig,
    CVMBReservoir,
    GaussianClassicalTwin,
    MeasurementBasedReservoir,
    WindowedMBQELM,
    chronological_splits,
)
from cv_mb_qrc.reservoirs.benchmarks import (
    ClassicalFeatures,
    capacity_targets,
    closed_loop_forecast,
    forecast_targets,
    generate_capacity_nulls,
    mackey_glass,
    metrics,
    narma,
    null_corrected_capacity,
    select_readout_multi,
)
from cv_mb_qrc.reservoirs.diagnostics import contraction, fading_memory, feature_diagnostics
from cv_mb_qrc.reservoirs.results import (
    atomic_json,
    create_evidence_index,
    environment,
    sha256_json,
    write_completed_job,
)
from cv_mb_qrc.reservoirs.temporal import delay_features


def validate_config(config):
    study_class = config.get("study_class", "development")
    if study_class not in ("ci-smoke", "development", "publication", "scaling"):
        raise ValueError("study_class must be ci-smoke, development, publication, or scaling")
    reservoir_seeds = config.get("reservoir_seeds", config.get("seeds", []))
    dataset_seeds = config.get("dataset_seeds", [config.get("dataset_seed", 1729)])
    if not reservoir_seeds or len(set(reservoir_seeds)) != len(reservoir_seeds):
        raise ValueError("Reservoir seeds must be nonempty and distinct")
    if not dataset_seeds or len(set(dataset_seeds)) != len(dataset_seeds):
        raise ValueError("Dataset seeds must be nonempty and distinct")
    if study_class in ("development", "publication") and len(reservoir_seeds) < 10:
        raise ValueError("Scientific studies require at least ten reservoir seeds")
    if study_class == "publication" and len(dataset_seeds) < 5:
        raise ValueError("Publication studies require at least five dataset seeds")
    horizons = config.get("forecast_horizons", [1])
    if not horizons or any(
        isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons
    ):
        raise ValueError("forecast_horizons must contain positive integers")
    narma_orders = config.get("narma_orders", [10])
    if not narma_orders or any(not isinstance(order, int) or order < 2 for order in narma_orders):
        raise ValueError("narma_orders must contain integers of at least two")
    narma_input_scale = config.get("narma_input_scale", 0.2)
    if not np.isfinite(narma_input_scale) or not 0 < narma_input_scale <= 0.5:
        raise ValueError("narma_input_scale must be finite and in (0, 0.5]")
    maximum_history = max(config["delays"], config["window"], max(narma_orders))
    if config["washout"] < maximum_history:
        raise ValueError("Washout must cover input history and NARMA initialization")
    if config["gap"] < max(maximum_history, max(horizons)):
        raise ValueError("Chronological gap must cover maximum history and forecast horizon")
    if config.get("input_access", "reservoir_plus_input") not in (
        "reservoir_only",
        "reservoir_plus_input",
    ):
        raise ValueError("Invalid input_access")
    if config.get("null_permutations", 0) < 0:
        raise ValueError("null_permutations must be nonnegative")
    resolved = dict(config)
    resolved["study_class"] = study_class
    resolved["reservoir_seeds"] = list(reservoir_seeds)
    resolved["dataset_seeds"] = list(dataset_seeds)
    resolved["forecast_horizons"] = list(horizons)
    resolved["narma_orders"] = list(narma_orders)
    resolved["narma_input_scale"] = float(narma_input_scale)
    resolved["input_access"] = config.get("input_access", "reservoir_plus_input")
    resolved.setdefault("null_permutations", 3 if study_class == "ci-smoke" else 100)
    resolved.setdefault("permutation_seed", 2718)
    resolved.setdefault("capacity_null_method", "block_permutation")
    resolved.setdefault("capacity_alpha", 0.05)
    resolved.setdefault("null_exclusion_window", config["delays"] + 1)
    resolved.setdefault("null_block_size", config["delays"] + 1)
    resolved.setdefault("bootstrap_seed", 31415)
    resolved.setdefault("enabled_stages", ["core", "dynamics", "shots", "report"])
    resolved.setdefault("enabled_methods", ["cv_A", "cv_B", "delay"])
    resolved.setdefault("closed_loop_context", 20)
    resolved.setdefault("closed_loop_length", 50)
    resolved.setdefault("closed_loop_threshold", 1.0)
    if min(resolved["closed_loop_context"], resolved["closed_loop_length"]) < 1:
        raise ValueError("Closed-loop context and length must be positive")
    allowed_stages = {
        "core",
        "dynamics",
        "shots",
        "washout",
        "scaling",
        "twin_validation",
        "capacity_calibration",
        "graphix",
        "mentpy",
        "fock",
        "report",
    }
    unknown_stages = sorted(set(resolved["enabled_stages"]) - allowed_stages)
    if unknown_stages:
        raise ValueError(f"Unknown enabled stages: {unknown_stages}")
    return resolved


def method_factories(reservoir_seed, config):
    base = CVConfig(seed=reservoir_seed)
    methods: dict[
        str, tuple[Callable[[], MeasurementBasedReservoir] | None, ClassicalFeatures | None]
    ] = {
        "cv_B": (lambda: CVMBReservoir(base), None),
        "cv_A": (lambda: CVMBReservoir(replace(base, tier="A")), None),
        "gaussian_classical_twin_B": (lambda: GaussianClassicalTwin(base), None),
        "gaussian_classical_twin_A": (
            lambda: GaussianClassicalTwin(replace(base, tier="A")),
            None,
        ),
        "cv_window": (
            lambda: WindowedMBQELM(lambda: CVMBReservoir(base), config["window"]),
            None,
        ),
        "cv_no_temporal": (
            lambda: CVMBReservoir(replace(base, temporal_edges=False)),
            None,
        ),
        "cv_zero_coupling": (lambda: CVMBReservoir(replace(base, coupling=0)), None),
        "cv_no_feedforward": (lambda: CVMBReservoir(replace(base, feedforward=0)), None),
    }
    classical_specs = {
        "delay": ("delay", 14),
        "lagged_ridge": ("delay", config["window"]),
        "linear_ar": ("linear_ar", config["window"]),
        "polynomial_narx_A": ("polynomial_narx", 8),
        "polynomial_narx_B": ("polynomial_narx", 14),
        "rff": ("rff", 14),
        "rff_A": ("rff", 8),
        "rff_B": ("rff", 14),
        "esn": ("esn", 14),
        "esn_A": ("esn", 8),
        "esn_B": ("esn", 14),
        "input_only": ("input_only", 14),
        "linear_state_space_A": ("linear_state_space", 8),
        "linear_state_space_B": ("linear_state_space", 14),
        "persistence": ("persistence", 1),
    }
    for name, (kind, dimension) in classical_specs.items():
        methods[name] = (
            None,
            ClassicalFeatures(kind, dimension, reservoir_seed, config["window"]),
        )
    if "graphix" in config["enabled_methods"]:
        from cv_mb_qrc.reservoirs import GraphixMBReservoir, QubitConfig

        methods["graphix"] = (
            lambda: GraphixMBReservoir(QubitConfig(seed=reservoir_seed)),
            None,
        )
    unknown = sorted(set(config["enabled_methods"]) - set(methods))
    if unknown:
        raise ValueError(f"Unknown or unavailable enabled methods: {unknown}")
    return {name: methods[name] for name in config["enabled_methods"]}


def task_arrays(series, task, config):
    values = np.asarray(series, float)
    if task == "iid":
        capacities, names = capacity_targets(values, config["delays"])
        history = delay_features(values, 4)
        parity = np.prod(np.where(history[:, 1:4] >= 0, 1.0, -1.0), axis=1)
        narma_inputs = (values + 1) * config["narma_input_scale"] / 2
        narma_targets = [narma(narma_inputs, order) for order in config["narma_orders"]]
        targets = np.column_stack([capacities, parity, *narma_targets])
        return (
            values,
            targets,
            names + ("parity",) + tuple(f"narma{order}" for order in config["narma_orders"]),
        )
    horizons = config["forecast_horizons"]
    inputs, targets = forecast_targets(values, horizons)
    return inputs, targets, tuple(f"mackey_glass_h{h}" for h in horizons)


def evaluate(
    name,
    reservoir_seed,
    dataset_seed,
    task,
    series,
    split_indices,
    config,
    *,
    factory=None,
    classical=None,
    cache_directory=None,
):
    prepared, prepared_indices, raw, resources = {}, {}, {}, {}
    configuration = {}
    start = perf_counter()
    rss_start = psutil.Process().memory_info().rss
    full_inputs, full_targets, names = task_arrays(series, task, config)
    capacity_names = [key for key in names if key.startswith(("linear_", "quadratic_", "cross_"))]
    null_matrices = None
    null_descriptors = []
    if capacity_names and config["null_permutations"]:
        null_matrices, null_names, null_descriptors = generate_capacity_nulls(
            series,
            config["delays"],
            config["null_permutations"],
            method=config["capacity_null_method"],
            seed=config["permutation_seed"] + 1009 * dataset_seed,
            exclusion_window=config["null_exclusion_window"],
            block_size=config["null_block_size"],
        )
        if tuple(capacity_names) != null_names:
            raise RuntimeError("Capacity null target order differs from observed target order")
    for split, indices in split_indices.items():
        valid_indices = indices[indices < len(full_inputs)]
        inputs, targets = full_inputs[valid_indices], full_targets[valid_indices]
        cache_key = sha256_json(
            {
                "schema_version": 1,
                "method": name,
                "reservoir_seed": reservoir_seed,
                "dataset_seed": dataset_seed,
                "task": task,
                "split": split,
                "inputs": inputs.tolist(),
                "resolved_configuration": config,
            }
        )
        cache_path = (
            None if cache_directory is None else Path(cache_directory) / f"{cache_key}.json"
        )
        cached = None
        if cache_path is not None and cache_path.exists():
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            if candidate.get("cache_key") == cache_key:
                cached = candidate
        if cached is not None:
            features = np.asarray(cached["features"], float)
            resources = cached["resources"]
            configuration = cached["configuration"]
        else:
            if factory is not None:
                model = factory()
                result = model.run_sequence(inputs)
                features = result.features
                resources = result.resources
                configuration = result.configuration
            else:
                features = classical.transform(inputs)
                resources = classical.resource_profile()
            if cache_path is not None:
                atomic_json(
                    cache_path,
                    {
                        "cache_key": cache_key,
                        "features": features.tolist(),
                        "resources": resources,
                        "configuration": configuration,
                    },
                )
        washout = config["washout"]
        prepared[split] = (inputs[washout:], features[washout:], targets[washout:])
        prepared_indices[split] = valid_indices[washout:]
        raw[split] = {
            "indices": prepared_indices[split].tolist(),
            "inputs": inputs[washout:].tolist(),
            "features": features[washout:].tolist(),
            "targets": targets[washout:].tolist(),
        }
    scores, predictions, regularization, readout_parameters = {}, {}, {}, {}
    null_by_target: dict[str, list[float]] = {key: [] for key in capacity_names}
    readout = select_readout_multi(
        prepared["train"],
        prepared["validation"],
        config["regularizations"],
        input_access=config["input_access"],
    )
    prediction_matrix = np.asarray(readout.predict(*prepared["test"][:2]))
    if prediction_matrix.ndim == 1:
        prediction_matrix = prediction_matrix[:, None]
    if name == "persistence" and task == "mg":
        prediction_matrix = np.repeat(np.asarray(prepared["test"][0])[:, None], len(names), axis=1)
    for column, target_name in enumerate(names):
        prediction = prediction_matrix[:, column]
        target_values = prepared["test"][2][:, column]
        scores[target_name] = metrics(target_values, prediction)
        if target_name == "parity":
            scores[target_name]["classification_accuracy"] = float(
                np.mean(np.where(prediction >= 0, 1.0, -1.0) == target_values)
            )
            scores[target_name]["classification_rule"] = "ridge sign threshold at zero"
        predictions[target_name] = prediction.tolist()
        regularization[target_name] = float(readout.selected_regularizations[column])
        readout_parameters[target_name] = int(readout.weights.shape[0] + 1)
    null_factorizations = 0
    if capacity_names and config["null_permutations"]:
        assert null_matrices is not None
        for null_targets in null_matrices:
            null_model = select_readout_multi(
                (
                    *prepared["train"][:2],
                    null_targets[prepared_indices["train"]],
                ),
                (
                    *prepared["validation"][:2],
                    null_targets[prepared_indices["validation"]],
                ),
                config["regularizations"],
                input_access=config["input_access"],
            )
            null_factorizations += null_model.factorization_count
            null_predictions = null_model.predict(*prepared["test"][:2])
            null_test_targets = null_targets[prepared_indices["test"]]
            for column, target_name in enumerate(capacity_names):
                null_by_target[target_name].append(
                    metrics(null_test_targets[:, column], null_predictions[:, column])["r2"]
                )
    capacity = None
    if capacity_names and config["null_permutations"]:
        observed = [scores[key]["r2"] for key in capacity_names]
        null = np.column_stack([null_by_target[key] for key in capacity_names])
        capacity = null_corrected_capacity(
            observed,
            null,
            capacity_names,
            alpha=config["capacity_alpha"],
        )
        feature_rank = int(np.linalg.matrix_rank(prepared["train"][1]))
        permitted_input_dimension = (
            np.asarray(prepared["train"][0]).reshape(len(prepared["train"][0]), -1).shape[1]
            if config["input_access"] == "reservoir_plus_input"
            else 0
        )
        capacity.update(
            {
                "target_names": capacity_names,
                "null_method": config["capacity_null_method"],
                "null_transformation_unit": "complete dataset target matrix before splitting",
                "null_descriptors": null_descriptors,
                "permutations": config["null_permutations"],
                "permutation_seed": config["permutation_seed"],
                "theoretical_rank_bound": min(
                    len(capacity_names), feature_rank + permitted_input_dimension
                ),
            }
        )
    closed_loop = None
    one_step_column = names.index("mackey_glass_h1") if "mackey_glass_h1" in names else None
    if task == "mg" and one_step_column is not None:
        test_inputs = prepared["test"][0]
        if len(test_inputs) < 3:
            raise ValueError("Closed-loop evaluation needs context plus at least two targets")
        context_length = min(config["closed_loop_context"], len(test_inputs) - 2)
        rollout_length = min(config["closed_loop_length"], len(test_inputs) - context_length)
        context = test_inputs[:context_length]
        truth = test_inputs[context_length : context_length + rollout_length]
        if factory is not None:
            rollout_model = factory()
            for value in context[:-1]:
                rollout_model.step(value)

            def predict_one(history):
                value = float(history[-1])
                feature = rollout_model.step(value).features[None, :]
                return readout.predict(np.asarray([value]), feature)[0, one_step_column]

        else:

            def predict_one(history):
                value = float(history[-1])
                if name == "persistence":
                    return value
                feature = classical.transform(history)[-1:]
                return readout.predict(np.asarray([value]), feature)[0, one_step_column]

        closed_loop = closed_loop_forecast(
            predict_one,
            context,
            truth,
            threshold=config["closed_loop_threshold"],
        )
        closed_loop.update(metrics(truth, closed_loop["predictions"]))
        closed_loop["context"] = context.tolist()
    return {
        "method": name,
        "dataset_seed": dataset_seed,
        "reservoir_seed": reservoir_seed,
        "task": task,
        "scores": scores,
        "capacity": capacity,
        "primary_capacity": None if capacity is None else capacity["total_significant_capacity"],
        "legacy_clipped_capacity": None if capacity is None else capacity["legacy_clipped_total"],
        "regularization": regularization,
        "predictions": predictions,
        "splits": raw,
        "diagnostics": feature_diagnostics(prepared["train"][1]),
        "resources": resources,
        "configuration": configuration,
        "input_access": config["input_access"],
        "feature_dimension": prepared["train"][1].shape[1],
        "comparison_budget": {
            "input_history": config["window"],
            "washout": config["washout"],
            "training_samples": len(prepared["train"][0]),
            "validation_samples": len(prepared["validation"][0]),
            "test_samples": len(prepared["test"][0]),
            "regularization_grid": config["regularizations"],
            "feature_normalization": "training partition only",
            "target_access": "train for fit; validation for selection; test once",
            "readout_parameter_count": readout_parameters,
            "reservoir_evaluations": int(
                sum(len(prepared[split][0]) + config["washout"] for split in prepared)
            ),
            "hyperparameter_search_fits_per_target": len(config["regularizations"]),
            "shared_design_factorizations": readout.factorization_count + null_factorizations,
        },
        "sampled_process_rss_increase_bytes": max(
            0, psutil.Process().memory_info().rss - rss_start
        ),
        "closed_loop": closed_loop,
        "seconds": perf_counter() - start,
    }


def build_datasets(config):
    datasets = {}
    for dataset_seed in config["dataset_seeds"]:
        rng = np.random.default_rng(dataset_seed)
        datasets[f"dataset-{dataset_seed}_iid"] = rng.uniform(-1, 1, config["length"])
        initial = float(1.2 + rng.uniform(-0.1, 0.1))
        datasets[f"dataset-{dataset_seed}_mg"] = mackey_glass(
            config["length"], initial_value=initial
        )
    return datasets


def expected_jobs(config):
    return [
        f"dataset-{dataset_seed}_reservoir-{reservoir_seed}_{task}_{method}"
        for dataset_seed in config["dataset_seeds"]
        for reservoir_seed in config["reservoir_seeds"]
        for task in ("iid", "mg")
        for method in config["enabled_methods"]
    ]


def estimate_run(config):
    config = validate_config(config)
    jobs = expected_jobs(config) if "core" in config["enabled_stages"] else []
    delays = config["delays"]
    capacity_targets_count = 2 * delays + delays * (delays - 1) // 2
    iid_targets = capacity_targets_count + 1 + len(config["narma_orders"])
    mg_targets = len(config["forecast_horizons"])
    method_repetitions = len(config["dataset_seeds"]) * len(config["reservoir_seeds"])
    method_count = len(config["enabled_methods"])
    observed_fits = method_repetitions * method_count * (iid_targets + mg_targets)
    null_fits = (
        method_repetitions * method_count * capacity_targets_count * config["null_permutations"]
    )
    naive_ridge_factorizations = (observed_fits + null_fits) * len(config["regularizations"])
    iid_jobs = method_repetitions * method_count
    optimized_ridge_factorizations = len(jobs) + iid_jobs * config["null_permutations"]
    feature_steps = len(jobs) * config["length"]
    feature_width_upper = max(14, config["window"] * 2)
    disk_bytes = int(
        len(jobs)
        * config["length"]
        * (feature_width_upper + max(iid_targets, mg_targets) + 2)
        * 8
        * 2.5
    )
    low_seconds = feature_steps * 2e-4 + optimized_ridge_factorizations * 2e-4
    high_seconds = feature_steps * 2e-2 + optimized_ridge_factorizations * 5e-3
    return {
        "study_class": config["study_class"],
        "expected_jobs": len(jobs),
        "feature_steps": feature_steps,
        "observed_target_fits": observed_fits,
        "null_target_fits": null_fits,
        "ridge_factorizations_naive_per_target_per_alpha": naive_ridge_factorizations,
        "ridge_factorizations_after_multi_target_reuse": optimized_ridge_factorizations,
        "factorization_reduction_ratio": (
            naive_ridge_factorizations / optimized_ridge_factorizations
        ),
        "estimated_wall_seconds_range": [low_seconds, high_seconds],
        "estimated_disk_bytes": disk_bytes,
        "estimated_peak_memory_bytes": int(
            config["length"] * (feature_width_upper + capacity_targets_count) * 8 * 12
        ),
        "assumptions": [
            "0.2-20 ms per feature step depending on backend",
            "0.2-5 ms per small ridge factorization",
            "JSON storage multiplier 2.5",
            "estimate excludes scheduler contention and optional Fock scaling",
        ],
    }


def _evaluate_job(spec, config, datasets, split_indices, cache_directory):
    dataset_seed, reservoir_seed, task, method = spec
    factory, classical = method_factories(reservoir_seed, config)[method]
    series = datasets[f"dataset-{dataset_seed}_{task}"]
    return evaluate(
        method,
        reservoir_seed,
        dataset_seed,
        task,
        series,
        split_indices,
        config,
        factory=factory,
        classical=classical,
        cache_directory=cache_directory,
    )


def _evaluate_job_safe(spec, config, datasets, split_indices, cache_directory):
    try:
        return spec, _evaluate_job(spec, config, datasets, split_indices, cache_directory), None
    except Exception as exc:
        return spec, None, {"error": repr(exc), "traceback": traceback.format_exc()}


def validate_fair_comparison_rows(rows):
    """Reject mismatched data access or evaluation budgets within paired comparisons."""
    invariant_fields = (
        "input_history",
        "washout",
        "training_samples",
        "validation_samples",
        "test_samples",
        "regularization_grid",
        "feature_normalization",
        "target_access",
        "reservoir_evaluations",
        "hyperparameter_search_fits_per_target",
    )
    groups: dict[tuple[int, int, str], list[dict]] = {}
    for row in rows:
        key = (row["dataset_seed"], row["reservoir_seed"], row["task"])
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        reference = group[0]
        if any(row["input_access"] != reference["input_access"] for row in group[1:]):
            raise ValueError(f"Unfair input-access policy in paired comparison {key}")
        for field in invariant_fields:
            expected = reference["comparison_budget"][field]
            if any(row["comparison_budget"][field] != expected for row in group[1:]):
                raise ValueError(f"Unfair {field} budget in paired comparison {key}")


def run_core_experiments(config, output, datasets, split_indices, manifest, failures):
    expected = expected_jobs(config)
    specs = [
        (dataset_seed, reservoir_seed, task, method)
        for dataset_seed in config["dataset_seeds"]
        for reservoir_seed in config["reservoir_seeds"]
        for task in ("iid", "mg")
        for method in config["enabled_methods"]
        if f"dataset-{dataset_seed}_reservoir-{reservoir_seed}_{task}_{method}" not in manifest
    ]
    workers = config.get("workers", 1)
    cache_directory = output / "raw/feature_cache"
    for batch_start in range(0, len(specs), workers):
        batch = specs[batch_start : batch_start + workers]
        if workers == 1:
            results = [
                _evaluate_job_safe(spec, config, datasets, split_indices, cache_directory)
                for spec in batch
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(
                    executor.map(
                        lambda spec: _evaluate_job_safe(
                            spec, config, datasets, split_indices, cache_directory
                        ),
                        batch,
                    )
                )
        for spec, row, failure_details in results:
            dataset_seed, reservoir_seed, task, method = spec
            key = f"dataset-{dataset_seed}_reservoir-{reservoir_seed}_{task}_{method}"
            if failure_details is not None:
                failure = {"job": key, **failure_details}
                failures.append(failure)
                atomic_json(output / "raw/failures.json", failures)
                raise RuntimeError(f"Experiment job failed: {key}: {failure['error']}")
            assert row is not None
            write_completed_job(output, key, row)
            manifest.append(key)
            atomic_json(output / "raw/manifest.json", manifest)
            atomic_json(
                output / "raw/progress.json",
                {
                    "completed_jobs": len(manifest),
                    "expected_jobs": len(expected),
                    "fraction_complete": len(manifest) / len(expected),
                    "last_completed_job": key,
                },
            )
            print(f"[{len(manifest)}/{len(expected)}] complete: {key}", flush=True)
    rows = [
        json.loads((output / "raw" / f"{job}.json").read_text(encoding="utf-8")) for job in expected
    ]
    validate_fair_comparison_rows(rows)
    return expected


def run_dynamics(config, output, datasets):
    import photographiq as pg

    first_seed = config["dataset_seeds"][0]
    inputs = datasets[f"dataset-{first_seed}_iid"][:40]
    initial = [pg.GaussianInput.coherent(value).state(0) for value in (0, 0.5, 1)]

    def factory():
        return CVMBReservoir(CVConfig(memory_modes=1))

    atomic_json(
        output / "raw/dynamics.json",
        {
            "contraction": contraction(factory, inputs, initial),
            "impulse": fading_memory(factory, np.zeros(40)),
        },
    )


def run_shot_study(config, output, datasets):
    first_seed = config["dataset_seeds"][0]
    inputs = datasets[f"dataset-{first_seed}_iid"][:5]
    rows = []
    for shots in config.get("shot_counts", []):
        for reservoir_seed in config["reservoir_seeds"]:
            model = CVMBReservoir(CVConfig(memory_modes=1, seed=reservoir_seed))
            exact = model.run_sequence(inputs).features
            sampled = model.reset().run_sequence(inputs, shots=shots)
            rows.append(
                {
                    "reservoir_seed": reservoir_seed,
                    "n_trajectories": shots,
                    "rms_error": float(np.sqrt(np.mean((sampled.features - exact) ** 2))),
                    "estimator": sampled.estimator,
                }
            )
    atomic_json(output / "raw/shots.json", rows)


def run_washout_study(config, output, datasets, split_indices):
    dataset_seed = config["dataset_seeds"][0]
    series = datasets[f"dataset-{dataset_seed}_iid"]
    rows = []
    candidates = sorted({max(10, config["washout"] // 2), config["washout"], 2 * config["washout"]})
    for reservoir_seed in config["reservoir_seeds"]:
        for washout in candidates:
            row = evaluate(
                "cv_B",
                reservoir_seed,
                dataset_seed,
                "iid",
                series,
                split_indices,
                {**config, "washout": washout},
                factory=lambda seed=reservoir_seed: CVMBReservoir(CVConfig(seed=seed)),
            )
            rows.append(
                {
                    "dataset_seed": dataset_seed,
                    "reservoir_seed": reservoir_seed,
                    "washout": washout,
                    "scores": row["scores"],
                }
            )
    atomic_json(output / "raw/washout.json", rows)


def run_scaling_study(config, output, datasets):
    dataset_seed = config["dataset_seeds"][0]
    inputs = datasets[f"dataset-{dataset_seed}_iid"][:100]
    rows = []
    for reservoir_seed in config["reservoir_seeds"]:
        for modes in config.get("scaling_memory_modes", [1, 2, 4]):
            start = perf_counter()
            model = CVMBReservoir(CVConfig(seed=reservoir_seed, memory_modes=modes))
            construction_seconds = perf_counter() - start
            result = model.run_sequence(inputs)
            rows.append(
                {
                    "dataset_seed": dataset_seed,
                    "reservoir_seed": reservoir_seed,
                    "memory_modes": modes,
                    "feature_dimension": len(result.feature_names),
                    "construction_seconds": construction_seconds,
                    "execution_seconds": result.seconds,
                    "diagnostics": feature_diagnostics(result.features),
                }
            )
    atomic_json(output / "raw/scaling.json", rows)


def run_graphix_diagnostics(config, output, datasets):
    from cv_mb_qrc.reservoirs import GraphixMBReservoir

    first_seed = config["dataset_seeds"][0]
    inputs = datasets[f"dataset-{first_seed}_iid"][:30]
    states = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0]), np.ones((2, 2)) / 2]
    atomic_json(
        output / "raw/qubit_contraction.json",
        contraction(GraphixMBReservoir, inputs, states),
    )


def run_mentpy_validation(output):
    from cv_mb_qrc.reservoirs.mentpy_backend import (
        backend_compatibility_matrix,
        compare_wire,
    )

    cases = []
    for state_name, state in (
        ("zero", np.array([1, 0], complex)),
        ("plus_i", np.array([1, 1j]) / np.sqrt(2)),
    ):
        for angles in ((0.2, -0.4), (0.0, 0.0), (0.3, -0.1, 0.25)):
            cases.append(
                {
                    "state": state_name,
                    "angles": list(angles),
                    "agreement": compare_wire(state, angles),
                }
            )
    atomic_json(
        output / "raw/mentpy.json",
        {
            "cases": cases,
            "scope": "corrected pure XY wires only",
            "compatibility": backend_compatibility_matrix(),
        },
    )
    atomic_json(output / "raw/backend_compatibility.json", backend_compatibility_matrix())


def run_fock_study(output):
    from cv_mb_qrc.reservoirs.fock import cutoff_study

    atomic_json(output / "raw/fock.json", cutoff_study([0.02, 0.04], cutoffs=(8, 12, 16)))


def run_capacity_calibration(config, output):
    from cv_mb_qrc.reservoirs.calibration import capacity_calibration

    atomic_json(output / "raw/capacity_calibration.json", capacity_calibration(config))


def run_twin_agreement(config, output):
    rows = []
    raw = output / "raw"
    for dataset_seed in config["dataset_seeds"]:
        for reservoir_seed in config["reservoir_seeds"]:
            for task in ("iid", "mg"):
                for tier in ("A", "B"):
                    cv_key = f"dataset-{dataset_seed}_reservoir-{reservoir_seed}_{task}_cv_{tier}"
                    twin_key = (
                        f"dataset-{dataset_seed}_reservoir-{reservoir_seed}_{task}_"
                        f"gaussian_classical_twin_{tier}"
                    )
                    cv_path, twin_path = raw / f"{cv_key}.json", raw / f"{twin_key}.json"
                    if not cv_path.exists() or not twin_path.exists():
                        continue
                    cv = json.loads(cv_path.read_text(encoding="utf-8"))
                    twin = json.loads(twin_path.read_text(encoding="utf-8"))
                    differences = []
                    for split in ("train", "validation", "test"):
                        left = np.asarray(cv["splits"][split]["features"], float)
                        right = np.asarray(twin["splits"][split]["features"], float)
                        differences.append((left - right).ravel())
                    difference = np.concatenate(differences)
                    score_differences = [
                        cv["scores"][name]["r2"] - twin["scores"][name]["r2"]
                        for name in cv["scores"]
                    ]
                    rows.append(
                        {
                            "dataset_seed": dataset_seed,
                            "reservoir_seed": reservoir_seed,
                            "task": task,
                            "tier": tier,
                            "maximum_absolute_feature_error": float(np.max(abs(difference))),
                            "rms_feature_error": float(np.sqrt(np.mean(difference**2))),
                            "maximum_absolute_score_error": float(
                                np.max(abs(np.asarray(score_differences)))
                            ),
                        }
                    )
    if rows:
        atomic_json(raw / "twin_agreement.json", rows)


def run_twin_validation(config, output):
    rows = []
    steps = config.get("twin_validation_steps", 100)
    modes_values = config.get("twin_validation_memory_modes", [1, 2, 4])
    for reservoir_seed in config["reservoir_seeds"]:
        rng = np.random.default_rng(50000 + reservoir_seed)
        inputs = rng.normal(scale=0.4, size=steps)
        for memory_modes in modes_values:
            for transmissivity in config.get("transmissivities", [0.8]):
                for noise in config.get("noise_strengths", [0.0]):
                    for tier in ("A", "B"):
                        resolved = CVConfig(
                            seed=reservoir_seed,
                            memory_modes=memory_modes,
                            tier=tier,
                            transmissivity=transmissivity,
                            measurement_noise=noise,
                        )
                        rss_before = psutil.Process().memory_info().rss
                        cv_start = perf_counter()
                        oracle = CVMBReservoir(resolved)
                        cv_construction = perf_counter() - cv_start
                        twin_start = perf_counter()
                        twin = GaussianClassicalTwin(resolved)
                        twin_construction = perf_counter() - twin_start
                        mean_errors, covariance_errors, feature_errors = [], [], []
                        cv_seconds = twin_seconds = 0.0
                        for value in inputs:
                            cv_start = perf_counter()
                            cv_result = oracle.step(value)
                            cv_seconds += perf_counter() - cv_start
                            twin_start = perf_counter()
                            twin_result = twin.step(value)
                            twin_seconds += perf_counter() - twin_start
                            mean_errors.append(
                                float(np.max(abs(oracle.state.mean - twin.state.mean)))
                            )
                            covariance_errors.append(
                                float(np.max(abs(oracle.state.covariance - twin.state.covariance)))
                            )
                            feature_errors.append(
                                float(np.max(abs(cv_result.features - twin_result.features)))
                            )
                        rows.append(
                            {
                                "reservoir_seed": reservoir_seed,
                                "tier": tier,
                                "memory_modes": memory_modes,
                                "transmissivity": transmissivity,
                                "measurement_noise": noise,
                                "steps": steps,
                                "mean_max_error_by_step": mean_errors,
                                "covariance_max_error_by_step": covariance_errors,
                                "feature_max_error_by_step": feature_errors,
                                "maximum_mean_error": max(mean_errors),
                                "rms_mean_error": float(np.sqrt(np.mean(np.square(mean_errors)))),
                                "maximum_covariance_error": max(covariance_errors),
                                "rms_covariance_error": float(
                                    np.sqrt(np.mean(np.square(covariance_errors)))
                                ),
                                "maximum_feature_error": max(feature_errors),
                                "rms_feature_error": float(
                                    np.sqrt(np.mean(np.square(feature_errors)))
                                ),
                                "cv_construction_seconds": cv_construction,
                                "twin_construction_seconds": twin_construction,
                                "cv_execution_seconds": cv_seconds,
                                "twin_execution_seconds": twin_seconds,
                                "sampled_process_rss_increase_bytes": max(
                                    0, psutil.Process().memory_info().rss - rss_before
                                ),
                                "interpretation": (
                                    "state-level simulator validation; agreement does not imply "
                                    "quantum computational advantage"
                                ),
                            }
                        )
    atomic_json(output / "raw/twin_validation.json", rows)


def run(config, output, *, resume=False, command=None, confirm_publication=False):
    config = validate_config(config)
    if config["study_class"] == "publication" and not confirm_publication:
        estimate = estimate_run(config)
        raise RuntimeError(
            "Publication execution requires --confirm-publication after reviewing: "
            + json.dumps(estimate, sort_keys=True)
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    start_timestamp = datetime.now(timezone.utc).isoformat()
    atomic_json(
        raw / "run_status.json",
        {"status": "running", "start_timestamp_utc": start_timestamp},
    )
    if resume and (raw / "config.json").exists():
        stored = json.loads((raw / "config.json").read_text(encoding="utf-8"))
        if stored != config:
            raise ValueError("Cannot resume a different resolved configuration")
    atomic_json(raw / "config.json", config)
    datasets = build_datasets(config)
    serializable_datasets = {key: value.tolist() for key, value in datasets.items()}
    atomic_json(raw / "datasets.json", serializable_datasets)
    split_indices = chronological_splits(
        config["length"], gap=config["gap"], washout=config["washout"]
    )
    serializable_splits = {key: value.tolist() for key, value in split_indices.items()}
    atomic_json(raw / "split_indices.json", serializable_splits)
    manifest = (
        json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
        if resume and (raw / "manifest.json").exists()
        else []
    )
    failures: list[dict] = []
    atomic_json(raw / "failures.json", failures)
    runtime = environment()
    atomic_json(raw / "environment.json", runtime)
    expected = expected_jobs(config) if "core" in config["enabled_stages"] else []
    provenance = {
        **runtime,
        "study_class": config["study_class"],
        "dirty_worktrees_at_start": runtime["dirty_worktrees"],
        "configuration_sha256": sha256_json(config),
        "dataset_sha256": {key: sha256_json(value) for key, value in serializable_datasets.items()},
        "split_indices_sha256": sha256_json(serializable_splits),
        "seed_roles": {
            "dataset": config["dataset_seeds"],
            "reservoir": config["reservoir_seeds"],
            "trajectory": config["reservoir_seeds"],
            "readout_probe": None,
            "bootstrap": config["bootstrap_seed"],
            "permutation": config["permutation_seed"],
        },
        "enabled_optional_backends": {
            "graphix": "graphix" in config["enabled_stages"],
            "mentpy": "mentpy" in config["enabled_stages"],
            "fock": "fock" in config["enabled_stages"],
        },
        "execution": {
            "workers": config.get("workers", 1),
            "parallelism": "bounded ordered thread batches",
            "feature_cache": "SHA-256 content-addressed JSON",
        },
        "command": command or " ".join(sys.argv),
        "start_timestamp_utc": start_timestamp,
        "end_timestamp_utc": None,
        "expected_manifest": expected,
        "report_generation_complete": False,
        "publication_valid": False,
    }
    atomic_json(raw / "provenance.json", provenance)
    peak = psutil.Process().memory_info().rss
    try:
        if "core" in config["enabled_stages"]:
            run_core_experiments(config, output, datasets, split_indices, manifest, failures)
            run_twin_agreement(config, output)
        if "dynamics" in config["enabled_stages"]:
            run_dynamics(config, output, datasets)
        if "shots" in config["enabled_stages"]:
            run_shot_study(config, output, datasets)
        if "washout" in config["enabled_stages"]:
            run_washout_study(config, output, datasets, split_indices)
        if "scaling" in config["enabled_stages"]:
            run_scaling_study(config, output, datasets)
        if "twin_validation" in config["enabled_stages"]:
            run_twin_validation(config, output)
        if "capacity_calibration" in config["enabled_stages"]:
            run_capacity_calibration(config, output)
        if "graphix" in config["enabled_stages"]:
            run_graphix_diagnostics(config, output, datasets)
        if "mentpy" in config["enabled_stages"]:
            run_mentpy_validation(output)
        if "fock" in config["enabled_stages"]:
            run_fock_study(output)
        atomic_json(
            raw / "runtime.json",
            {"sampled_peak_process_rss_bytes": max(peak, psutil.Process().memory_info().rss)},
        )
        if "report" in config["enabled_stages"]:
            from reproduce import reproduce

            reproduce(output)
            provenance["report_generation_complete"] = True
            summary = json.loads((output / "json/summary.json").read_text(encoding="utf-8"))
            provenance["summary_sha256"] = sha256_json(summary)
    except Exception as exc:
        end_timestamp = datetime.now(timezone.utc).isoformat()
        if not failures or failures[-1].get("error") != repr(exc):
            failures.append({"job": None, "error": repr(exc), "traceback": traceback.format_exc()})
            atomic_json(raw / "failures.json", failures)
        provenance.update(
            {
                "end_timestamp_utc": end_timestamp,
                "completed_manifest": manifest,
                "failure_manifest": failures,
            }
        )
        atomic_json(
            raw / "run_status.json",
            {
                "status": "failed",
                "start_timestamp_utc": start_timestamp,
                "end_timestamp_utc": end_timestamp,
                "completed_jobs": len(manifest),
                "failed_jobs": len(failures),
            },
        )
        evidence_index = create_evidence_index(output, expected, manifest, terminal_status="failed")
        provenance["evidence_index_sha256"] = sha256_json(evidence_index)
        atomic_json(raw / "provenance.json", provenance)
        raise
    provenance["end_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["completed_manifest"] = manifest
    provenance["failure_manifest"] = failures
    provenance["optional_modules_imported"] = {
        "graphix": "graphix" in sys.modules,
        "mentpy": "mentpy" in sys.modules,
    }
    for module in ("graphix", "mentpy"):
        if (
            not provenance["enabled_optional_backends"][module]
            and provenance["optional_modules_imported"][module]
        ):
            raise RuntimeError(f"Disabled optional backend was imported: {module}")
    atomic_json(
        raw / "run_status.json",
        {
            "status": "complete",
            "start_timestamp_utc": start_timestamp,
            "end_timestamp_utc": provenance["end_timestamp_utc"],
            "completed_jobs": len(manifest),
            "failed_jobs": len(failures),
        },
    )
    evidence_index = create_evidence_index(output, expected, manifest)
    provenance["evidence_index_sha256"] = sha256_json(evidence_index)
    atomic_json(raw / "provenance.json", provenance)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--config", type=Path, default=root / "config_small.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-publication", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    resolved = validate_config(config)
    if args.workers < 1:
        parser.error("--workers must be positive")
    resolved["workers"] = args.workers
    config = resolved
    if args.dry_run:
        print(json.dumps(estimate_run(config), indent=2))
        return
    if args.output is None:
        output_name = {
            "ci-smoke": "results_ci",
            "development": "results_development",
            "publication": "results_publication",
            "scaling": "results_scaling",
        }[resolved["study_class"]]
        args.output = root / output_name
    run(
        config,
        args.output,
        resume=args.resume,
        confirm_publication=args.confirm_publication,
    )


if __name__ == "__main__":
    main()
