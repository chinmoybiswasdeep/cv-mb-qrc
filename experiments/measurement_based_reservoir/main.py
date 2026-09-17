"""Deterministic temporal experiments; all test evaluation follows validation selection."""

import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import photographiq as pg
import psutil

from cv_mb_qrc.reservoirs import (
    CVConfig,
    CVMBReservoir,
    GraphixMBReservoir,
    QubitConfig,
    WindowedMBQELM,
    chronological_splits,
)
from cv_mb_qrc.reservoirs.benchmarks import (
    ClassicalFeatures,
    capacity_targets,
    mackey_glass,
    metrics,
    narma10,
    select_readout,
)
from cv_mb_qrc.reservoirs.diagnostics import contraction, fading_memory, feature_diagnostics
from cv_mb_qrc.reservoirs.mentpy_backend import compare_wire
from cv_mb_qrc.reservoirs.results import atomic_json, environment
from cv_mb_qrc.reservoirs.temporal import delay_features


def methods(seed, config):
    c = CVConfig(seed=seed)
    factories = {
        "cv_B": lambda: CVMBReservoir(c),
        "cv_A": lambda: CVMBReservoir(replace(c, tier="A")),
        "cv_window": lambda: WindowedMBQELM(lambda: CVMBReservoir(c), config["window"]),
        "cv_no_temporal": lambda: CVMBReservoir(replace(c, temporal_edges=False)),
        "cv_zero_coupling": lambda: CVMBReservoir(replace(c, coupling=0)),
        "cv_no_feedforward": lambda: CVMBReservoir(replace(c, feedforward=0)),
        "graphix": lambda: GraphixMBReservoir(QubitConfig(seed=seed)),
        "graphix_no_entanglement": lambda: GraphixMBReservoir(
            QubitConfig(seed=seed, entangle=False)
        ),
    }
    for eta in config["transmissivities"]:
        factories[f"cv_eta_{eta}"] = lambda eta=eta: CVMBReservoir(replace(c, transmissivity=eta))
    for noise in config["noise_strengths"]:
        factories[f"cv_noise_{noise}"] = lambda noise=noise: CVMBReservoir(
            replace(c, measurement_noise=noise)
        )
    return factories


def targets(u, task, delays):
    if task == "iid":
        capacities, names = capacity_targets(u, delays)
        history = delay_features(u, 4)
        parity = np.prod(np.where(history[:, 1:4] >= 0, 1.0, -1.0), axis=1)
        return np.column_stack([capacities, parity, narma10((u + 1) / 4)]), names + (
            "parity",
            "narma10",
        )
    return u[1:, None], ("mackey_glass",)


def evaluate(name, seed, task, series, split_indices, config, *, factory=None, classical=None):
    prepared, raw, resources = {}, {}, {}
    configuration = {}
    start = perf_counter()
    for split, indices in split_indices.items():
        # Every split generates its own lag targets and resets quantum/classical state.
        u = series[indices]
        y, names = targets(u, task, config["delays"])
        if task == "mg":
            u = u[:-1]
        if factory is not None:
            model = factory()
            result = model.run_sequence(u)
            x = result.features
            resources = result.resources
            configuration = result.configuration
            compilation_seconds = getattr(
                getattr(model, "model", model), "compilation_seconds", None
            )
            if compilation_seconds is None:
                compilation_seconds = sum(
                    d.get("compilation_seconds", 0) for d in result.diagnostics["step_diagnostics"]
                )
        else:
            x = classical.transform(u)
            resources = {"feature_dimension": x.shape[1], "classical_window": classical.window}
            compilation_seconds = 0.0
        washout = config["washout"]
        prepared[split] = (u[washout:], x[washout:], y[washout:])
        raw[split] = {
            "resources": resources,
            "compilation_seconds": compilation_seconds,
            "indices": indices[: len(u)][washout:].tolist(),
            "inputs": u[washout:].tolist(),
            "features": x[washout:].tolist(),
            "targets": y[washout:].tolist(),
        }
    scores, predictions, regularization = {}, {}, {}
    for j, target_name in enumerate(names):
        train = (*prepared["train"][:2], prepared["train"][2][:, j])
        validation = (*prepared["validation"][:2], prepared["validation"][2][:, j])
        readout = select_readout(train, validation, config["regularizations"])
        # First access to the test target for this fitted head is below.
        prediction = readout.predict(*prepared["test"][:2])
        if name == "persistence" and task == "mg":
            prediction = prepared["test"][0].copy()
        scores[target_name] = metrics(prepared["test"][2][:, j], prediction)
        if target_name == "parity":
            scores[target_name]["accuracy"] = float(
                np.mean((prediction >= 0) == (prepared["test"][2][:, j] >= 0))
            )
        predictions[target_name] = prediction.tolist()
        regularization[target_name] = readout.regularization
    linear = [row["r2"] for key, row in scores.items() if key.startswith("linear_")]
    nonlinear = [
        row["r2"] for key, row in scores.items() if key.startswith(("quadratic_", "cross_"))
    ]
    return {
        "method": name,
        "seed": seed,
        "task": task,
        "scores": scores,
        "linear_capacity": sum(max(0, x) for x in linear),
        "nonlinear_capacity": sum(max(0, x) for x in nonlinear),
        "capacity_convention": "sum max(0,test R2); all raw negative R2 retained",
        "regularization": regularization,
        "predictions": predictions,
        "splits": raw,
        "diagnostics": feature_diagnostics(prepared["train"][1]),
        "resources": resources,
        "configuration": configuration,
        "shots": None,
        "feature_dimension": prepared["train"][1].shape[1],
        "seconds": perf_counter() - start,
    }


def run(config, output, *, resume=False):
    study_class = config.get("study_class", "development")
    if study_class not in ("ci-smoke", "development", "publication"):
        raise ValueError("study_class must be ci-smoke, development, or publication")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Reservoir seeds must be distinct")
    if study_class != "ci-smoke" and len(config["seeds"]) < 10:
        raise ValueError("Scientific studies require at least ten reservoir seeds")
    if study_class == "publication" and len(config.get("dataset_seeds", [])) < 5:
        raise ValueError("Publication studies require at least five dataset seeds")
    if config["washout"] < max(config["delays"], config["window"], 10):
        raise ValueError("Washout must cover input history and NARMA initialization")
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    if resume and (output / "raw/manifest.json").exists():
        stored = json.loads((output / "raw/config.json").read_text())
        if stored != config:
            raise ValueError("Cannot resume a different configuration")
        manifest = json.loads((output / "raw/manifest.json").read_text())
    atomic_json(output / "raw/config.json", config)
    provenance = environment()
    dirty = any(value is True for value in provenance["dirty_worktrees"].values())
    provenance.update(
        {
            "commands": ["experiments/measurement_based_reservoir/main.py"],
            "start_timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "development_uncommitted": dirty,
            "publication_valid": not dirty,
            "publication_validity_reason": "requires clean cv-mb-qrc and PhotoGraphiQ worktrees",
        }
    )
    atomic_json(output / "raw/environment.json", provenance)
    rng = np.random.default_rng(config["dataset_seed"])
    data = {"iid": rng.uniform(-1, 1, config["length"]), "mg": mackey_glass(config["length"])}
    split_indices = chronological_splits(
        config["length"], gap=config["gap"], washout=config["washout"]
    )
    atomic_json(output / "raw/datasets.json", {name: u.tolist() for name, u in data.items()})
    atomic_json(
        output / "raw/split_indices.json", {k: v.tolist() for k, v in split_indices.items()}
    )
    process = psutil.Process()
    peak = process.memory_info().rss
    for seed in config["seeds"]:
        for task, series in data.items():
            jobs = [(name, factory, None) for name, factory in methods(seed, config).items()]
            for kind in ("delay", "rff", "esn", "input_only", "persistence"):
                dimensions = (3, 14) if kind in ("rff", "esn", "input_only") else (14,)
                for dimension in dimensions:
                    name = f"{kind}_{dimension}" if len(dimensions) > 1 else kind
                    jobs.append(
                        (name, None, ClassicalFeatures(kind, dimension, seed, config["window"]))
                    )
            enabled = config.get("enabled_methods")
            if enabled is not None:
                jobs = [job for job in jobs if job[0] in enabled]
            for name, factory, classical in jobs:
                key = f"{task}_{name}_{seed}"
                if key in manifest:
                    if not (output / f"raw/{key}.json").is_file():
                        raise FileNotFoundError(f"Manifest entry {key} is missing")
                    continue
                try:
                    row = evaluate(
                        name,
                        seed,
                        task,
                        series,
                        split_indices,
                        config,
                        factory=factory,
                        classical=classical,
                    )
                except Exception as exc:
                    atomic_json(
                        output / f"raw/failure_{key}.json",
                        {
                            "config": config,
                            "method": name,
                            "seed": seed,
                            "task": task,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    raise
                atomic_json(output / f"raw/{key}.json", row)
                peak = max(peak, process.memory_info().rss)
                manifest.append(key)
                atomic_json(output / "raw/manifest.json", manifest)
            print(f"completed seed={seed} task={task}", flush=True)
    atomic_json(
        output / "raw/runtime.json",
        {
            "sampled_peak_process_rss_bytes": peak,
            "limitation": "RSS sampled between runs; includes imports and may miss intra-run peaks",
        },
    )
    initial = [pg.GaussianInput.coherent(a).state(0) for a in (0, 0.5, 1)]
    diagnostics = {}
    for eta in config["transmissivities"]:

        def factory(eta=eta):
            return CVMBReservoir(CVConfig(memory_modes=1, transmissivity=eta))

        diagnostics[str(eta)] = {
            "contraction": contraction(factory, data["iid"][:80], initial),
            "impulse": fading_memory(factory, np.zeros(80)),
            "perturbation": fading_memory(factory, data["iid"][:80], at=10),
        }
    atomic_json(output / "raw/dynamics.json", diagnostics)
    if config.get("run_mentpy", False):
        atomic_json(output / "raw/mentpy.json", compare_wire(np.array([1, 1j]) / np.sqrt(2)))
    shot_rows = []
    for shots in config["shot_counts"]:
        for seed in config["seeds"]:
            model = CVMBReservoir(CVConfig(memory_modes=1, seed=seed))
            exact = model.run_sequence(data["iid"][:5]).features
            sampled = model.reset().run_sequence(data["iid"][:5], shots=shots)
            shot_rows.append(
                {
                    "seed": seed,
                    "shots": shots,
                    "rms_error": float(np.sqrt(np.mean((sampled.features - exact) ** 2))),
                    "seconds": sampled.seconds,
                    "estimator": sampled.estimator,
                }
            )
    atomic_json(output / "raw/shots.json", shot_rows)
    if config["run_fock"]:
        from cv_mb_qrc.reservoirs.fock import cutoff_study

        atomic_json(output / "raw/fock.json", cutoff_study([0.02, 0.04], cutoffs=(8, 12, 16)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--config", type=Path, default=root / "config_small.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume identical saved config")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.output is None:
        output_name = {
            "ci-smoke": "results_ci",
            "development": "results_development",
            "publication": "results_publication",
        }[config.get("study_class", "development")]
        args.output = root / output_name
    try:
        run(config, args.output, resume=args.resume)
        from supplementary import run_supplementary

        run_supplementary(args.output)
    except Exception as exc:
        atomic_json(
            args.output / "raw/runner_failure.json",
            {"config": config, "error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise
    from reproduce import reproduce

    reproduce(args.output)


if __name__ == "__main__":
    main()
