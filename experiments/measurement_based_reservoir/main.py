"""Manifest-driven temporal experiments with explicit optional stages."""

import argparse
import json
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil

from cv_mb_qrc.reservoirs import (
    CVConfig,
    CVMBReservoir,
    GaussianClassicalTwin,
    WindowedMBQELM,
    chronological_splits,
)
from cv_mb_qrc.reservoirs.benchmarks import (
    ClassicalFeatures,
    capacity_targets,
    closed_loop_forecast,
    forecast_targets,
    mackey_glass,
    metrics,
    narma10,
    null_corrected_capacity,
    select_readout,
)
from cv_mb_qrc.reservoirs.diagnostics import contraction, fading_memory, feature_diagnostics
from cv_mb_qrc.reservoirs.results import atomic_json, environment, sha256_json
from cv_mb_qrc.reservoirs.temporal import delay_features


def validate_config(config):
    study_class = config.get("study_class", "development")
    if study_class not in ("ci-smoke", "development", "publication"):
        raise ValueError("study_class must be ci-smoke, development, or publication")
    reservoir_seeds = config.get("reservoir_seeds", config.get("seeds", []))
    dataset_seeds = config.get("dataset_seeds", [config.get("dataset_seed", 1729)])
    if not reservoir_seeds or len(set(reservoir_seeds)) != len(reservoir_seeds):
        raise ValueError("Reservoir seeds must be nonempty and distinct")
    if not dataset_seeds or len(set(dataset_seeds)) != len(dataset_seeds):
        raise ValueError("Dataset seeds must be nonempty and distinct")
    if study_class != "ci-smoke" and len(reservoir_seeds) < 10:
        raise ValueError("Scientific studies require at least ten reservoir seeds")
    if study_class == "publication" and len(dataset_seeds) < 5:
        raise ValueError("Publication studies require at least five dataset seeds")
    horizons = config.get("forecast_horizons", [1])
    if not horizons or any(
        isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons
    ):
        raise ValueError("forecast_horizons must contain positive integers")
    maximum_history = max(config["delays"], config["window"], 10)
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
    resolved["input_access"] = config.get("input_access", "reservoir_plus_input")
    resolved.setdefault("null_permutations", 3 if study_class == "ci-smoke" else 100)
    resolved.setdefault("permutation_seed", 2718)
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
    methods = {
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
    for kind in ("delay", "rff", "esn", "input_only", "persistence"):
        dimension = 14
        methods[kind] = (
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
        targets = np.column_stack([capacities, parity, narma10((values + 1) / 4)])
        return values, targets, names + ("parity", "narma10")
    horizons = config["forecast_horizons"]
    inputs, targets = forecast_targets(values, horizons)
    return inputs, targets, tuple(f"mackey_glass_h{h}" for h in horizons)


def _circular_shift(values, rng):
    if len(values) < 2:
        raise ValueError("Null permutation requires at least two samples")
    return np.roll(values, int(rng.integers(1, len(values))), axis=0)


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
):
    prepared, raw, resources = {}, {}, {}
    configuration = {}
    start = perf_counter()
    for split, indices in split_indices.items():
        inputs, targets, names = task_arrays(series[indices], task, config)
        if factory is not None:
            model = factory()
            result = model.run_sequence(inputs)
            features = result.features
            resources = result.resources
            configuration = result.configuration
        else:
            features = classical.transform(inputs)
            resources = {
                "feature_dimension": features.shape[1],
                "classical_window": classical.window,
            }
        washout = config["washout"]
        prepared[split] = (inputs[washout:], features[washout:], targets[washout:])
        raw[split] = {
            "indices": indices[: len(inputs)][washout:].tolist(),
            "inputs": inputs[washout:].tolist(),
            "features": features[washout:].tolist(),
            "targets": targets[washout:].tolist(),
        }
    scores, predictions, regularization = {}, {}, {}
    one_step_readout = None
    capacity_names = [key for key in names if key.startswith(("linear_", "quadratic_", "cross_"))]
    null_by_target = {key: [] for key in capacity_names}
    rng = np.random.default_rng(config["permutation_seed"] + 1009 * dataset_seed + reservoir_seed)
    for column, target_name in enumerate(names):
        train = (*prepared["train"][:2], prepared["train"][2][:, column])
        validation = (*prepared["validation"][:2], prepared["validation"][2][:, column])
        readout = select_readout(
            train,
            validation,
            config["regularizations"],
            input_access=config["input_access"],
        )
        prediction = readout.predict(*prepared["test"][:2])
        if name == "persistence" and task == "mg":
            prediction = prepared["test"][0].copy()
        scores[target_name] = metrics(prepared["test"][2][:, column], prediction)
        predictions[target_name] = prediction.tolist()
        regularization[target_name] = readout.regularization
        if task == "mg" and target_name == "mackey_glass_h1":
            one_step_readout = readout
        if target_name in null_by_target:
            for _ in range(config["null_permutations"]):
                null_train = (*prepared["train"][:2], _circular_shift(train[2], rng))
                null_validation = (
                    *prepared["validation"][:2],
                    _circular_shift(validation[2], rng),
                )
                null_model = select_readout(
                    null_train,
                    null_validation,
                    config["regularizations"],
                    input_access=config["input_access"],
                )
                null_prediction = null_model.predict(*prepared["test"][:2])
                null_target = _circular_shift(prepared["test"][2][:, column], rng)
                null_by_target[target_name].append(metrics(null_target, null_prediction)["r2"])
    capacity = None
    if capacity_names and config["null_permutations"]:
        observed = [scores[key]["r2"] for key in capacity_names]
        null = np.column_stack([null_by_target[key] for key in capacity_names])
        capacity = null_corrected_capacity(observed, null)
        capacity.update(
            {
                "target_names": capacity_names,
                "permutation_method": "independent within-split circular shifts",
                "permutations": config["null_permutations"],
                "permutation_seed": config["permutation_seed"],
            }
        )
    closed_loop = None
    if task == "mg" and one_step_readout is not None:
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
                return one_step_readout.predict(np.asarray([value]), feature)[0]

        else:

            def predict_one(history):
                value = float(history[-1])
                if name == "persistence":
                    return value
                feature = classical.transform(history)[-1:]
                return one_step_readout.predict(np.asarray([value]), feature)[0]

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
        "primary_capacity": None if capacity is None else capacity["null_corrected_total"],
        "legacy_clipped_capacity": None if capacity is None else capacity["legacy_clipped_total"],
        "regularization": regularization,
        "predictions": predictions,
        "splits": raw,
        "diagnostics": feature_diagnostics(prepared["train"][1]),
        "resources": resources,
        "configuration": configuration,
        "input_access": config["input_access"],
        "feature_dimension": prepared["train"][1].shape[1],
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


def run_core_experiments(config, output, datasets, split_indices, manifest, failures):
    expected = expected_jobs(config)
    for dataset_seed in config["dataset_seeds"]:
        for reservoir_seed in config["reservoir_seeds"]:
            methods = method_factories(reservoir_seed, config)
            for task in ("iid", "mg"):
                series = datasets[f"dataset-{dataset_seed}_{task}"]
                for method, (factory, classical) in methods.items():
                    key = f"dataset-{dataset_seed}_reservoir-{reservoir_seed}_{task}_{method}"
                    if key in manifest:
                        continue
                    try:
                        row = evaluate(
                            method,
                            reservoir_seed,
                            dataset_seed,
                            task,
                            series,
                            split_indices,
                            config,
                            factory=factory,
                            classical=classical,
                        )
                        atomic_json(output / f"raw/{key}.json", row)
                        manifest.append(key)
                        atomic_json(output / "raw/manifest.json", manifest)
                    except Exception as exc:
                        failure = {
                            "job": key,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        failures.append(failure)
                        atomic_json(output / "raw/failures.json", failures)
                        raise
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
        for modes in (1, 2, 4):
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
    from cv_mb_qrc.reservoirs.mentpy_backend import compare_wire

    atomic_json(output / "raw/mentpy.json", compare_wire(np.array([1, 1j]) / np.sqrt(2)))


def run_fock_study(output):
    from cv_mb_qrc.reservoirs.fock import cutoff_study

    atomic_json(output / "raw/fock.json", cutoff_study([0.02, 0.04], cutoffs=(8, 12, 16)))


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


def run(config, output, *, resume=False, command=None):
    config = validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
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
    failures = []
    atomic_json(raw / "failures.json", failures)
    runtime = environment()
    expected = expected_jobs(config)
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
        "command": command or " ".join(sys.argv),
        "start_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "end_timestamp_utc": None,
        "expected_manifest": expected,
        "report_generation_complete": False,
        "publication_valid": False,
    }
    atomic_json(raw / "provenance.json", provenance)
    peak = psutil.Process().memory_info().rss
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
    atomic_json(raw / "provenance.json", provenance)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--config", type=Path, default=root / "config_small.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    resolved = validate_config(config)
    if args.output is None:
        output_name = {
            "ci-smoke": "results_ci",
            "development": "results_development",
            "publication": "results_publication",
        }[resolved["study_class"]]
        args.output = root / output_name
    run(config, args.output, resume=args.resume)


if __name__ == "__main__":
    main()
