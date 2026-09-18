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
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
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
    config = _read_finite_json(raw / "config.json")
    datasets = _read_finite_json(raw / "datasets.json")
    splits = _read_finite_json(raw / "split_indices.json")
    completed = _read_finite_json(raw / "manifest.json")
    failures_path = raw / "failures.json"
    failures = _read_finite_json(failures_path) if failures_path.exists() else []
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
