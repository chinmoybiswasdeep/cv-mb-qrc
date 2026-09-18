import importlib.util
import json
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
