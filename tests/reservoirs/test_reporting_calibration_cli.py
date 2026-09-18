from cv_mb_qrc import cli
from cv_mb_qrc.reservoirs.calibration import capacity_calibration
from cv_mb_qrc.reservoirs.reporting import (
    build_summaries,
    capacity_calibration_csv_text,
    closed_loop_csv_text,
    feature_spectrum_csv_text,
    paired_comparisons_csv_text,
    paired_model_comparisons,
    prediction_example_csv_text,
    summary_csv_text,
    summary_markdown,
    twin_validation_csv_text,
)


def test_development_calibration_executes_all_declared_controls():
    config = {
        "calibration_null_methods": [
            "circular_shift",
            "block_permutation",
            "independent_surrogate",
        ],
        "calibration_dataset_seeds": [101, 102],
        "calibration_null_permutations": 9,
        "calibration_length": 150,
        "calibration_delays": 2,
        "acceptable_type1_interval": [0.0, 1.0],
        "capacity_alpha": 0.05,
        "permutation_seed": 7,
        "null_exclusion_window": 3,
        "null_block_size": 3,
        "regularizations": [1e-4],
    }
    report = capacity_calibration(config)
    assert len(report["records"]) == 2 * 3 * 5
    assert len(report["summary"]) == 3 * 5
    leaky = [row for row in report["records"] if row["system"] == "deliberately_leaky"]
    assert all(row["leak_guard_detected"] for row in leaky)
    assert all(
        row["calibration_pass"] for row in report["summary"] if row["calibration_pass"] is not None
    )
    assert capacity_calibration_csv_text(report).count("\n") == 16


def test_rich_summary_and_twin_tables_cover_scientific_fields():
    runs: list[dict] = []
    for dataset_seed in (10, 11):
        runs.append(
            {
                "task": "mg",
                "method": "cv_B",
                "dataset_seed": dataset_seed,
                "reservoir_seed": 0,
                "seconds": 1.0,
                "diagnostics": {"effective_rank": 3},
                "capacity": {
                    "total_significant_capacity": 1.2,
                    "raw_total": -0.2,
                    "legacy_clipped_total": 2.0,
                },
                "closed_loop": {"nrmse": 0.4, "valid_prediction_time": 8},
                "scores": {
                    "mackey_glass_h1": {
                        "r2": 0.5,
                        "rmse": 0.2,
                        "nrmse": 0.3,
                        "mae": 0.1,
                        "pearson_r": 0.8,
                        "classification_accuracy": 0.9,
                    }
                },
            }
        )
    summaries, flat = build_summaries({"study_class": "development", "bootstrap_seed": 3}, runs)
    assert "significant_capacity" in summaries["mg/cv_B"]
    assert "closed_loop_nrmse" in summaries["mg/cv_B"]
    assert "mackey_glass_h1/pearson_r" in summaries["mg/cv_B"]
    assert summary_csv_text(flat).startswith("task,method,metric")
    assert "95% hierarchical bootstrap CI" in summary_markdown(flat)

    table = twin_validation_csv_text(
        [
            {
                "reservoir_seed": 0,
                "tier": "B",
                "memory_modes": 2,
                "transmissivity": 0.8,
                "measurement_noise": 0.0,
                "mean_max_error_by_step": [1e-15, 2e-15],
                "covariance_max_error_by_step": [2e-15, 3e-15],
                "feature_max_error_by_step": [3e-15, 4e-15],
            }
        ]
    )
    assert table.count("\n") == 3

    artifact_run: dict = dict(runs[0])
    artifact_run.update(
        {
            "job_id": "job-a",
            "diagnostics": {"effective_rank": 3, "covariance_spectrum": [2.0, 1.0]},
            "closed_loop": {
                **runs[0]["closed_loop"],
                "predictions": [0.1, 0.2],
                "normalized_absolute_error": [0.2, 0.4],
                "threshold": 1.0,
                "diverged": False,
            },
            "splits": {"test": {"targets": [[0.2], [0.3]]}},
            "predictions": {"mackey_glass_h1": [0.1, 0.2]},
        }
    )
    assert closed_loop_csv_text([artifact_run]).count("\n") == 3
    assert feature_spectrum_csv_text([artifact_run]).count("\n") == 3
    assert prediction_example_csv_text([artifact_run]).count("\n") == 3


def _comparison_run(task, method, dataset_seed, reservoir_seed, offset):
    row = {
        "task": task,
        "method": method,
        "dataset_seed": dataset_seed,
        "reservoir_seed": reservoir_seed,
        "primary_capacity": 1.0 + offset,
        "scores": {"mackey_glass_h1": {"r2": 0.5 + offset}},
        "closed_loop": {"nrmse": 0.4 - offset},
    }
    return row


def test_paired_model_comparison_preserves_hierarchy_and_corrects_family():
    runs = []
    for dataset_seed in (10, 11):
        for reservoir_seed in (0, 1):
            for task in ("iid", "mg"):
                runs.append(_comparison_run(task, "cv_B", dataset_seed, reservoir_seed, 0.1))
                runs.append(_comparison_run(task, "delay", dataset_seed, reservoir_seed, 0.0))
    config = {"bootstrap_seed": 4, "study_class": "development", "capacity_alpha": 0.05}
    comparisons = paired_model_comparisons(config, runs)
    assert len(comparisons) == 3
    assert all(row["paired_units"] == 4 for row in comparisons)
    assert all(row["dataset_clusters"] == 2 for row in comparisons)
    assert all("bh_q_value_across_declared_model_comparisons" in row for row in comparisons)
    assert paired_comparisons_csv_text(comparisons).startswith("task_metric,")
    assert paired_model_comparisons(config, [runs[1]]) == []


def test_cli_returns_concise_success_and_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "verify_result_directory",
        lambda _path: {"integrity_valid": True, "publication_valid": False},
    )
    assert cli.main(["verify-run", "run"]) == 0
    assert '"integrity_valid": true' in capsys.readouterr().out

    def invalid(_path):
        raise ValueError("raw job hash mismatch: job-a")

    monkeypatch.setattr(cli, "verify_result_directory", invalid)
    assert cli.main(["verify-run", "run"]) == 1
    assert "invalid run: raw job hash mismatch: job-a" in capsys.readouterr().err
