"""Structured numerical results and portable provenance."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np


def _direct_source_root(distribution_name: str) -> Path | None:
    try:
        text = importlib.metadata.distribution(distribution_name).read_text("direct_url.json")
        if not text:
            return None
        url = json.loads(text).get("url", "")
        parsed = urlparse(url)
        if parsed.scheme != "file":
            return None
        path = Path(url2pathname(parsed.path)).resolve()
        return path if path.exists() else None
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError, OSError):
        return None


def _project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src/cv_mb_qrc").is_dir():
            return candidate
    return _direct_source_root("cv-mb-qrc") or Path(__file__).resolve().parents[3]


def environment() -> dict:
    packages: dict[str, str | None] = {}
    for name in (
        "photographiq",
        "cv-mb-qrc",
        "piquasso",
        "graphix",
        "mentpy",
        "numpy",
        "scipy",
        "networkx",
        "scikit-learn",
        "matplotlib",
        "psutil",
        "pytest",
        "pytest-cov",
        "ruff",
        "mypy",
        "build",
        "jax",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    roots = {"cv-mb-qrc": _project_root()}
    try:
        import photographiq

        roots["photographiq"] = (
            _direct_source_root("photographiq") or Path(photographiq.__file__).resolve().parents[2]
        )
    except ImportError:
        pass
    commits = {}
    dirty = {}
    for name, root in roots.items():
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commits[name] = proc.stdout.strip() if proc.returncode == 0 else None
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        dirty[name] = bool(status.stdout.strip()) if status.returncode == 0 else None
    root = roots["cv-mb-qrc"]
    source_paths = sorted((root / "src/cv_mb_qrc/reservoirs").glob("*.py"))
    source_paths += sorted((root / "experiments/measurement_based_reservoir").glob("*.py"))
    hashes = {
        str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_paths
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        memory_record = {"total_bytes": memory.total, "available_bytes": memory.available}
        physical_cpu_count = psutil.cpu_count(logical=False)
    except ImportError:
        memory_record = {"total_bytes": None, "available_bytes": None}
        physical_cpu_count = None
    try:
        numpy_configuration: dict = dict(np.show_config(mode="dicts"))
    except TypeError:  # NumPy builds predating the structured mode.
        numpy_configuration = {}
    build_dependencies = numpy_configuration.get("Build Dependencies", {})
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "physical_cpu_count": physical_cpu_count,
        "logical_cpu_count": os.cpu_count(),
        "memory": memory_record,
        "blas": build_dependencies.get("blas"),
        "lapack": build_dependencies.get("lapack"),
        "packages": packages,
        "commits": commits,
        "dirty_worktrees": dirty,
        "implementation_sha256": hashes,
    }


def verify_source_hashes(record: dict) -> None:
    """Fail closed when result provenance does not match this checkout."""
    expected = record.get("implementation_sha256")
    actual = environment()["implementation_sha256"]
    if expected != actual:
        expected = expected or {}
        differences = sorted(
            path for path in set(expected) | set(actual) if expected.get(path) != actual.get(path)
        )
        raise ValueError(f"source mismatch: {differences}")


def sha256_json(data) -> str:
    """Hash canonical finite JSON data."""
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_completed_job(root, job_id: str, record: dict) -> str:
    """Atomically publish a raw job followed by its content-bound completion marker."""
    root = Path(root)
    record = {**record, "job_id": job_id, "job_status": "complete"}
    result_path = root / "raw" / f"{job_id}.json"
    atomic_json(result_path, record)
    digest = sha256_file(result_path)
    atomic_json(
        root / "raw/completed" / f"{job_id}.json",
        {"job_id": job_id, "job_status": "complete", "result_sha256": digest},
    )
    return digest


def _evidence_files(root: Path) -> list[Path]:
    paths = [
        path
        for path in (root / "raw").rglob("*.json")
        if path.name not in {"provenance.json", "run_index.json"}
    ]
    for directory in ("json", "csv", "tables", "figures"):
        if (root / directory).exists():
            paths.extend(path for path in (root / directory).rglob("*") if path.is_file())
    if (root / "report.md").exists():
        paths.append(root / "report.md")
    return sorted(set(paths))


def create_evidence_index(
    root,
    expected_jobs,
    completed_jobs,
    *,
    terminal_status="complete",
) -> dict:
    """Create a content-addressed index after all registered artifacts are final."""
    root = Path(root)
    expected, completed = list(expected_jobs), list(completed_jobs)
    files = {path.relative_to(root).as_posix(): sha256_file(path) for path in _evidence_files(root)}
    job_paths = {job: f"raw/{job}.json" for job in expected}
    marker_paths = {job: f"raw/completed/{job}.json" for job in expected}
    index = {
        "schema_version": 2,
        "terminal_status": terminal_status,
        "expected_jobs": expected,
        "completed_jobs": completed,
        "expected_result_paths": job_paths,
        "expected_completion_markers": marker_paths,
        "files": files,
    }
    index["merkle_root_sha256"] = sha256_json(sorted(files.items()))
    atomic_json(root / "raw/run_index.json", index)
    return index


def _read_finite_json(path: Path):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON value {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def verify_result_directory(path) -> dict:
    """Verify provenance inputs and return a detailed validity report.

    Integrity failures raise with exact mismatch categories. Publication
    readiness is returned separately because development/CI studies are valid
    reproducible runs but can never be publication-valid.
    """
    root = Path(path)
    raw = root / "raw"
    errors = []
    for json_path in raw.glob("*.json"):
        try:
            _read_finite_json(json_path)
        except ValueError as exc:
            errors.append(f"non-finite JSON: {json_path.name}: {exc}")
    if errors:
        raise ValueError("; ".join(errors))
    provenance = _read_finite_json(raw / "provenance.json")
    index_path = raw / "run_index.json"
    if not index_path.exists():
        raise ValueError("evidence index missing")
    index = _read_finite_json(index_path)
    config = _read_finite_json(raw / "config.json")
    datasets = _read_finite_json(raw / "datasets.json")
    splits = _read_finite_json(raw / "split_indices.json")
    completed = _read_finite_json(raw / "manifest.json")
    failures_path = raw / "failures.json"
    failures = _read_finite_json(failures_path) if failures_path.exists() else []
    if provenance.get("evidence_index_sha256") != sha256_json(index):
        errors.append("evidence index hash mismatch")
    if index.get("schema_version") != 2:
        errors.append("unsupported evidence index schema")
    if index.get("terminal_status") != "complete":
        errors.append(f"run terminal status is {index.get('terminal_status')!r}, not complete")
    run_status_path = raw / "run_status.json"
    if (
        not run_status_path.exists()
        or _read_finite_json(run_status_path).get("status") != "complete"
    ):
        errors.append("run status is not complete")
    if provenance.get("configuration_sha256") != sha256_json(config):
        errors.append("configuration hash mismatch")
    expected_dataset_hashes = provenance.get("dataset_sha256", {})
    for key, value in datasets.items():
        if expected_dataset_hashes.get(key) != sha256_json(value):
            errors.append(f"dataset hash mismatch: {key}")
    missing_datasets = sorted(set(expected_dataset_hashes) - set(datasets))
    for key in missing_datasets:
        errors.append(f"dataset hash mismatch: {key}")
    if provenance.get("split_indices_sha256") != sha256_json(splits):
        errors.append("split hash mismatch")
    expected = provenance.get("expected_manifest", [])
    if len(expected) != len(set(expected)):
        errors.append("duplicate expected job IDs")
    if len(completed) != len(set(completed)):
        errors.append("duplicate completed job IDs")
    if index.get("expected_jobs") != expected:
        errors.append("evidence index expected jobs differ from provenance")
    if index.get("completed_jobs") != completed:
        errors.append("evidence index completed jobs differ from manifest")
    missing = sorted(set(expected) - set(completed))
    unexpected = sorted(set(completed) - set(expected))
    if missing:
        errors.append(f"missing manifest jobs: {missing}")
    if unexpected:
        errors.append(f"unexpected manifest jobs: {unexpected}")
    if provenance.get("completed_manifest") not in (None, completed):
        errors.append("completed manifest differs from provenance")
    if failures:
        errors.append(f"failed jobs: {[row.get('job') for row in failures]}")
    indexed_files = index.get("files", {})
    actual_paths = {path.relative_to(root).as_posix(): path for path in _evidence_files(root)}
    missing_files = sorted(set(indexed_files) - set(actual_paths))
    extra_files = sorted(set(actual_paths) - set(indexed_files))
    if missing_files:
        errors.append(f"missing registered evidence files: {missing_files}")
    if extra_files:
        errors.append(f"unregistered evidence files: {extra_files}")
    for relative, expected_hash in indexed_files.items():
        path_value = root / relative
        if path_value.exists() and sha256_file(path_value) != expected_hash:
            job = next(
                (
                    key
                    for key, value in index.get("expected_result_paths", {}).items()
                    if value == relative
                ),
                None,
            )
            errors.append(
                f"raw job hash mismatch: {job}"
                if job is not None
                else f"evidence file hash mismatch: {relative}"
            )
    for job in expected:
        relative = index.get("expected_result_paths", {}).get(job)
        marker_relative = index.get("expected_completion_markers", {}).get(job)
        if relative != f"raw/{job}.json":
            errors.append(f"unexpected result path: {job}")
            continue
        job_path = root / relative
        marker_path = root / str(marker_relative)
        if not job_path.exists():
            errors.append(f"missing raw job: {job}")
            continue
        if not marker_path.exists():
            errors.append(f"missing completion marker: {job}")
            continue
        try:
            row = _read_finite_json(job_path)
            marker = _read_finite_json(marker_path)
            if row.get("job_id") != job or row.get("job_status") != "complete":
                errors.append(f"malformed raw job: {job}")
            if (
                marker.get("job_id") != job
                or marker.get("job_status") != "complete"
                or marker.get("result_sha256") != sha256_file(job_path)
            ):
                errors.append(f"invalid completion marker: {job}")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"malformed raw job: {job}: {exc}")
    try:
        verify_source_hashes(provenance)
    except ValueError as exc:
        errors.append(str(exc))
    if "results_legacy" in str(root).replace("\\", "/"):
        errors.append("legacy result path is not current evidence")
    if provenance.get("report_generation_complete", False):
        if not (raw / "report_status.json").exists() or not (root / "json/summary.json").exists():
            errors.append("report artifacts missing")
        elif provenance.get("summary_sha256") != sha256_json(
            _read_finite_json(root / "json/summary.json")
        ):
            errors.append("summary hash mismatch")
        else:
            try:
                from .reporting import (
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

                rows = [_read_finite_json(raw / f"{job}.json") for job in completed]
                rebuilt_summary, rebuilt_flat = build_summaries(config, rows)
                if rebuilt_summary != _read_finite_json(root / "json/summary.json"):
                    errors.append("stored summary differs from raw-job reconstruction")
                if summary_csv_text(rebuilt_flat) != (root / "csv/summary.csv").read_text(
                    encoding="utf-8"
                ):
                    errors.append("stored CSV table differs from raw-job reconstruction")
                if summary_markdown(rebuilt_flat) != (root / "tables/results.md").read_text(
                    encoding="utf-8"
                ):
                    errors.append("stored Markdown table differs from raw-job reconstruction")
                derived_tables = {
                    "csv/closed_loop.csv": closed_loop_csv_text(rows),
                    "csv/feature_spectrum.csv": feature_spectrum_csv_text(rows),
                    "csv/prediction_example.csv": prediction_example_csv_text(rows),
                    "csv/paired_model_comparisons.csv": paired_comparisons_csv_text(
                        paired_model_comparisons(config, rows)
                    ),
                }
                for relative, rebuilt in derived_tables.items():
                    if rebuilt != (root / relative).read_text(encoding="utf-8"):
                        errors.append(f"stored derived table differs from raw jobs: {relative}")
                if paired_model_comparisons(config, rows) != _read_finite_json(
                    root / "json/paired_model_comparisons.json"
                ):
                    errors.append("stored paired comparisons differ from raw jobs")
                twin_path = raw / "twin_validation.json"
                if twin_path.exists() and twin_validation_csv_text(
                    _read_finite_json(twin_path)
                ) != (root / "csv/twin_validation.csv").read_text(encoding="utf-8"):
                    errors.append("stored twin table differs from raw validation reconstruction")
                calibration_path = raw / "capacity_calibration.json"
                if calibration_path.exists() and capacity_calibration_csv_text(
                    _read_finite_json(calibration_path)
                ) != (root / "csv/capacity_calibration.csv").read_text(encoding="utf-8"):
                    errors.append("stored calibration table differs from raw reconstruction")
                figure_map = _read_finite_json(root / "json/figure_provenance.json")
                generated = _read_finite_json(raw / "report_status.json").get(
                    "generated_figures", []
                )
                for figure in generated:
                    if figure not in figure_map:
                        errors.append(f"figure provenance missing: {figure}")
                        continue
                    record = figure_map[figure]
                    source_table = root / record.get("source_table", "")
                    if not source_table.is_file() or record.get(
                        "source_table_sha256"
                    ) != sha256_file(source_table):
                        errors.append(f"figure source table mismatch: {figure}")
                    if record.get("configuration_sha256") != sha256_json(config):
                        errors.append(f"figure configuration mismatch: {figure}")
                    if not set(record.get("raw_jobs", [])).issubset(completed):
                        errors.append(f"figure references unregistered jobs: {figure}")
                    if not set(record.get("raw_artifacts", [])).issubset(indexed_files):
                        errors.append(f"figure references unregistered artifacts: {figure}")
                    for suffix in ("svg", "pdf", "png"):
                        if not (root / f"figures/{figure}.{suffix}").exists():
                            errors.append(f"figure artifact missing: {figure}.{suffix}")
            except (KeyError, OSError, ValueError, TypeError) as exc:
                errors.append(f"summary reconstruction failed: {exc}")
    if errors:
        raise ValueError("; ".join(errors))
    publication_reasons = []
    if config.get("study_class") != "publication":
        publication_reasons.append("study class is not publication")
    if any(value is not False for value in provenance.get("dirty_worktrees_at_start", {}).values()):
        publication_reasons.append("source worktree was not clean at run start")
    if not provenance.get("report_generation_complete", False):
        publication_reasons.append("report generation incomplete")
    if config.get("study_class") == "publication":
        rows = [_read_finite_json(raw / f"{job}.json") for job in completed]
        present_methods = {row.get("method") for row in rows}
        required_methods = set(config.get("required_publication_methods", []))
        missing_methods = sorted(required_methods - present_methods)
        if missing_methods:
            publication_reasons.append(f"required publication methods missing: {missing_methods}")
        for role in ("dataset", "reservoir"):
            configured = set(config.get(f"{role}_seeds", []))
            present = {row.get(f"{role}_seed") for row in rows}
            missing_seeds = sorted(configured - present)
            if missing_seeds:
                publication_reasons.append(f"required {role} seeds missing: {missing_seeds}")
        repetitions = config.get("null_permutations", 0)
        incomplete = [
            job
            for job, row in zip(completed, rows, strict=True)
            if row.get("task") == "iid"
            and (
                row.get("capacity") is None
                or len(row["capacity"].get("null_scores", [])) != repetitions
            )
        ]
        if incomplete:
            publication_reasons.append(f"statistical repetitions incomplete: {incomplete}")
    return {
        "integrity_valid": True,
        "publication_valid": not publication_reasons,
        "publication_invalid_reasons": publication_reasons,
        "completed_jobs": len(completed),
    }


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        # Windows scanners/readers can hold a short-lived non-sharing handle.
        for attempt in range(7):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 6:
                    raise
                time.sleep(0.02 * 2**attempt)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class ReservoirResult:
    features: np.ndarray
    feature_names: tuple[str, ...]
    evolution: str
    estimator: str
    shots: int | None
    outcomes: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    configuration: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["features"] = self.features.tolist()
        return data

    def save(self, path):
        atomic_json(path, self.to_dict())

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["features"] = np.asarray(data["features"], float)
        data["feature_names"] = tuple(data["feature_names"])
        if not np.isfinite(data["features"]).all():
            raise ValueError("Nonfinite saved features")
        return cls(**data)
