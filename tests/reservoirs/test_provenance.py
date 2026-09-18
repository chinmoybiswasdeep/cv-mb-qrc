import json

import pytest

from cv_mb_qrc.reservoirs.reporting import (
    build_summaries,
    closed_loop_csv_text,
    feature_spectrum_csv_text,
    paired_comparisons_csv_text,
    paired_model_comparisons,
    prediction_example_csv_text,
    summary_csv_text,
    summary_markdown,
)
from cv_mb_qrc.reservoirs.results import (
    atomic_json,
    atomic_text,
    create_evidence_index,
    environment,
    sha256_file,
    sha256_json,
    verify_result_directory,
    write_completed_job,
)


def make_result(tmp_path):
    raw = tmp_path / "raw"
    config = {"study_class": "ci-smoke", "bootstrap_seed": 3}
    datasets = {"dataset-1_iid": [0.1, 0.2]}
    splits = {"train": [0], "validation": [1], "test": [2]}
    manifest = ["job-a"]
    atomic_json(raw / "config.json", config)
    atomic_json(raw / "datasets.json", datasets)
    atomic_json(raw / "split_indices.json", splits)
    row = {
        "method": "delay",
        "dataset_seed": 1,
        "reservoir_seed": 2,
        "task": "iid",
        "scores": {"linear_1": {"r2": 0.25, "rmse": 1.0, "nrmse": 1.0, "mae": 0.8}},
        "capacity": None,
        "closed_loop": None,
        "seconds": 0.1,
        "diagnostics": {"effective_rank": 1.0, "covariance_spectrum": [1.0]},
    }
    write_completed_job(tmp_path, "job-a", row)
    atomic_json(raw / "manifest.json", manifest)
    atomic_json(raw / "failures.json", [])
    atomic_json(
        raw / "report_status.json", {"raw_run_count": 1, "generated_figures": ["test_plot"]}
    )
    atomic_json(raw / "run_status.json", {"status": "complete"})
    atomic_json(raw / "environment.json", {"test": True})
    complete_row = {**row, "job_id": "job-a", "job_status": "complete"}
    summaries, flat = build_summaries(config, [complete_row])
    atomic_json(tmp_path / "json/summary.json", summaries)
    atomic_text(tmp_path / "csv/summary.csv", summary_csv_text(flat))
    atomic_text(tmp_path / "tables/results.md", summary_markdown(flat))
    atomic_text(tmp_path / "csv/closed_loop.csv", closed_loop_csv_text([complete_row]))
    atomic_text(tmp_path / "csv/feature_spectrum.csv", feature_spectrum_csv_text([complete_row]))
    atomic_text(
        tmp_path / "csv/prediction_example.csv", prediction_example_csv_text([complete_row])
    )
    paired = paired_model_comparisons(config, [complete_row])
    atomic_json(tmp_path / "json/paired_model_comparisons.json", paired)
    atomic_text(tmp_path / "csv/paired_model_comparisons.csv", paired_comparisons_csv_text(paired))
    for suffix in ("svg", "pdf", "png"):
        artifact = tmp_path / f"figures/test_plot.{suffix}"
        artifact.parent.mkdir(exist_ok=True)
        artifact.write_bytes(f"test-{suffix}".encode())
    atomic_json(
        tmp_path / "json/figure_provenance.json",
        {
            "test_plot": {
                "source_table": "csv/summary.csv",
                "source_table_sha256": sha256_file(tmp_path / "csv/summary.csv"),
                "configuration_sha256": sha256_json(config),
                "raw_jobs": manifest,
                "filters": {"task": "iid"},
            }
        },
    )
    summary = json.loads((tmp_path / "json/summary.json").read_text(encoding="utf-8"))
    provenance = {
        **environment(),
        "configuration_sha256": sha256_json(config),
        "dataset_sha256": {key: sha256_json(value) for key, value in datasets.items()},
        "split_indices_sha256": sha256_json(splits),
        "expected_manifest": manifest,
        "dirty_worktrees_at_start": {"cv-mb-qrc": True},
        "report_generation_complete": True,
        "summary_sha256": sha256_json(summary),
        "completed_manifest": manifest,
    }
    index = create_evidence_index(tmp_path, manifest, manifest)
    provenance["evidence_index_sha256"] = sha256_json(index)
    atomic_json(raw / "provenance.json", provenance)
    return raw


def test_provenance_accepts_integrity_but_not_ci_for_publication(tmp_path):
    make_result(tmp_path)
    report = verify_result_directory(tmp_path)
    assert report["integrity_valid"]
    assert not report["publication_valid"]


@pytest.mark.parametrize(
    "filename,replacement,message",
    [
        ("config.json", {"study_class": "development"}, "configuration hash mismatch"),
        ("datasets.json", {"dataset-1_iid": [9]}, "dataset hash mismatch"),
        ("split_indices.json", {"train": [9]}, "split hash mismatch"),
        ("manifest.json", [], "missing manifest jobs"),
    ],
)
def test_provenance_reports_exact_tampering(tmp_path, filename, replacement, message):
    raw = make_result(tmp_path)
    (raw / filename).write_text(json.dumps(replacement), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        verify_result_directory(tmp_path)


def test_provenance_reports_source_and_failure_tampering(tmp_path):
    raw = make_result(tmp_path)
    provenance_path = raw / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    first = next(iter(provenance["implementation_sha256"]))
    provenance["implementation_sha256"][first] = "0" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="source mismatch"):
        verify_result_directory(tmp_path)

    raw = make_result(tmp_path)
    (raw / "failures.json").write_text(json.dumps([{"job": "job-a"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="failed jobs"):
        verify_result_directory(tmp_path)


def test_provenance_rejects_nonfinite_json_and_summary_tampering(tmp_path):
    raw = make_result(tmp_path)
    (raw / "datasets.json").write_text('{"dataset-1_iid": [NaN]}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        verify_result_directory(tmp_path)

    make_result(tmp_path)
    atomic_json(tmp_path / "json/summary.json", {"job-a": {"mean": 2.0}})
    with pytest.raises(ValueError, match="summary hash mismatch"):
        verify_result_directory(tmp_path)


def test_provenance_rejects_raw_job_metric_tampering(tmp_path):
    raw = make_result(tmp_path)
    verify_result_directory(tmp_path)
    job = json.loads((raw / "job-a.json").read_text(encoding="utf-8"))
    job["seconds"] = 2.0
    atomic_json(raw / "job-a.json", job)
    with pytest.raises(ValueError, match="raw job hash mismatch: job-a"):
        verify_result_directory(tmp_path)


def test_provenance_rejects_seed_missing_extra_and_malformed_jobs(tmp_path):
    raw = make_result(tmp_path)
    job_path = raw / "job-a.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["dataset_seed"] = 999
    atomic_json(job_path, job)
    with pytest.raises(ValueError, match="raw job hash mismatch: job-a"):
        verify_result_directory(tmp_path)

    raw = make_result(tmp_path)
    (raw / "job-a.json").unlink()
    with pytest.raises(ValueError, match="missing registered evidence files"):
        verify_result_directory(tmp_path)

    raw = make_result(tmp_path)
    atomic_json(raw / "unregistered-job.json", {"job_id": "unregistered-job"})
    with pytest.raises(ValueError, match="unregistered evidence files"):
        verify_result_directory(tmp_path)

    raw = make_result(tmp_path)
    (raw / "job-a.json").write_text('{"job_id":', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        verify_result_directory(tmp_path)


def test_provenance_rejects_replaced_figure_and_incomplete_status(tmp_path):
    make_result(tmp_path)
    (tmp_path / "figures/test_plot.png").write_bytes(b"replacement")
    with pytest.raises(ValueError, match="evidence file hash mismatch: figures/test_plot.png"):
        verify_result_directory(tmp_path)

    raw = make_result(tmp_path)
    atomic_json(raw / "run_status.json", {"status": "running"})
    with pytest.raises(ValueError, match="run status is not complete"):
        verify_result_directory(tmp_path)


def test_provenance_rejects_duplicate_manifest_job(tmp_path):
    raw = make_result(tmp_path)
    atomic_json(raw / "manifest.json", ["job-a", "job-a"])
    with pytest.raises(ValueError, match="duplicate completed job IDs"):
        verify_result_directory(tmp_path)
