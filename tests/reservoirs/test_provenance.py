import json

import pytest

from cv_mb_qrc.reservoirs.results import (
    atomic_json,
    environment,
    sha256_json,
    verify_result_directory,
)


def make_result(tmp_path):
    raw = tmp_path / "raw"
    config = {"study_class": "ci-smoke"}
    datasets = {"dataset-1_iid": [0.1, 0.2]}
    splits = {"train": [0], "validation": [1], "test": [2]}
    manifest = ["job-a"]
    atomic_json(raw / "config.json", config)
    atomic_json(raw / "datasets.json", datasets)
    atomic_json(raw / "split_indices.json", splits)
    atomic_json(raw / "manifest.json", manifest)
    atomic_json(raw / "failures.json", [])
    atomic_json(raw / "report_status.json", {"raw_run_count": 1})
    atomic_json(tmp_path / "json/summary.json", {"job-a": {"mean": 1.0}})
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
    }
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
