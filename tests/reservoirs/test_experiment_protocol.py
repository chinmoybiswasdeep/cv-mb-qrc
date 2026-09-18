import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/measurement_based_reservoir"


def load_main():
    spec = importlib.util.spec_from_file_location(
        "cv_mb_qrc_experiment_main", EXPERIMENT / "main.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ci_seed_policy_and_publication_axes():
    module = load_main()
    ci = json.loads((EXPERIMENT / "config_ci.json").read_text(encoding="utf-8"))
    resolved = module.validate_config(ci)
    assert resolved["reservoir_seeds"] == [0]
    assert resolved["dataset_seeds"] == [1729]
    with pytest.raises(ValueError, match="at least ten"):
        module.validate_config({**ci, "study_class": "development"})
    publication = json.loads((EXPERIMENT / "config_publication.json").read_text(encoding="utf-8"))
    resolved_publication = module.validate_config(publication)
    assert len(resolved_publication["dataset_seeds"]) == 5
    assert "gaussian_classical_twin_B" in resolved_publication["enabled_methods"]


def test_dataset_seed_axis_changes_iid_and_mackey_glass():
    module = load_main()
    config = json.loads((EXPERIMENT / "config_ci.json").read_text(encoding="utf-8"))
    config["dataset_seeds"] = [1729, 1730]
    datasets = module.build_datasets(module.validate_config(config))
    assert not np.array_equal(datasets["dataset-1729_iid"], datasets["dataset-1730_iid"])
    assert not np.array_equal(datasets["dataset-1729_mg"], datasets["dataset-1730_mg"])


def test_expected_manifest_records_both_seed_roles():
    module = load_main()
    config = json.loads((EXPERIMENT / "config_ci.json").read_text(encoding="utf-8"))
    jobs = module.expected_jobs(module.validate_config(config))
    assert jobs == [
        "dataset-1729_reservoir-0_iid_cv_A",
        "dataset-1729_reservoir-0_iid_cv_B",
        "dataset-1729_reservoir-0_iid_delay",
        "dataset-1729_reservoir-0_mg_cv_A",
        "dataset-1729_reservoir-0_mg_cv_B",
        "dataset-1729_reservoir-0_mg_delay",
    ]


def test_fair_comparison_validator_rejects_mismatched_access_and_budget():
    module = load_main()
    budget = {
        "input_history": 4,
        "washout": 10,
        "training_samples": 20,
        "validation_samples": 10,
        "test_samples": 10,
        "regularization_grid": [0.1],
        "feature_normalization": "training partition only",
        "target_access": "train for fit; validation for selection; test once",
        "reservoir_evaluations": 60,
        "hyperparameter_search_fits_per_target": 1,
    }
    base = {
        "dataset_seed": 1,
        "reservoir_seed": 2,
        "task": "iid",
        "input_access": "reservoir_only",
        "comparison_budget": budget,
    }
    module.validate_fair_comparison_rows([base, {**base, "method": "other"}])
    with pytest.raises(ValueError, match="input-access"):
        module.validate_fair_comparison_rows(
            [base, {**base, "input_access": "reservoir_plus_input"}]
        )
    with pytest.raises(ValueError, match="washout"):
        module.validate_fair_comparison_rows(
            [base, {**base, "comparison_budget": {**budget, "washout": 11}}]
        )


def test_feature_cache_reuses_content_addressed_features(tmp_path):
    module = load_main()
    config = json.loads((EXPERIMENT / "config_ci.json").read_text(encoding="utf-8"))
    config["enabled_methods"] = ["delay", "linear_ar"]
    config["length"] = 150
    config["washout"] = 10
    config["gap"] = 10
    resolved = module.validate_config(config)
    datasets = module.build_datasets(resolved)
    splits = module.chronological_splits(150, gap=10, washout=10)
    spec = (1729, 0, "mg", "delay")
    serial = module._evaluate_job(spec, resolved, datasets, splits, tmp_path / "cache")
    cached = module._evaluate_job(spec, resolved, datasets, splits, tmp_path / "cache")
    assert list((tmp_path / "cache").glob("*.json"))
    assert serial["scores"] == cached["scores"]
    assert serial["predictions"] == cached["predictions"]

    specs = [(1729, 0, "mg", "delay"), (1729, 0, "mg", "linear_ar")]
    serial_rows = [
        module._evaluate_job(spec, resolved, datasets, splits, tmp_path / "serial")
        for spec in specs
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        parallel_rows = list(
            executor.map(
                lambda job: module._evaluate_job(
                    job, resolved, datasets, splits, tmp_path / "parallel"
                ),
                specs,
            )
        )
    for serial_row, parallel_row in zip(serial_rows, parallel_rows, strict=True):
        assert serial_row["scores"] == parallel_row["scores"]
        assert serial_row["predictions"] == parallel_row["predictions"]
