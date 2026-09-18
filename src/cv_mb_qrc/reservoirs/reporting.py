"""Pure, deterministic reconstruction of scientific summary tables."""

import csv
import io

import numpy as np

from .benchmarks import benjamini_hochberg
from .diagnostics import hierarchical_bootstrap_summary, paired_hierarchical_summary

SUMMARY_COLUMNS = (
    "task",
    "method",
    "metric",
    "mean",
    "std",
    "median",
    "ci_low",
    "ci_high",
    "scientific_uncertainty_valid",
)


def build_summaries(config: dict, runs: list[dict]) -> tuple[dict, list[dict]]:
    """Recompute every aggregate solely from registered raw job records."""
    if not runs:
        raise ValueError("Cannot summarize an empty run set")
    methods = sorted({row["method"] for row in runs})
    tasks = sorted({row["task"] for row in runs})
    scientific = config["study_class"] != "ci-smoke"
    summaries: dict = {}
    flat: list[dict] = []
    for task in tasks:
        for method in methods:
            group = [row for row in runs if row["task"] == task and row["method"] == method]
            if not group:
                continue
            fields = {
                "seconds": [row["seconds"] for row in group],
                "effective_rank": [row["diagnostics"]["effective_rank"] for row in group],
            }
            if group[0].get("capacity") is not None:
                capacity_key = (
                    "total_significant_capacity"
                    if "total_significant_capacity" in group[0]["capacity"]
                    else "null_corrected_total"
                )
                fields["significant_capacity"] = [row["capacity"][capacity_key] for row in group]
                fields["raw_signed_capacity"] = [
                    row["capacity"].get("raw_total", 0.0) for row in group
                ]
                fields["legacy_clipped_capacity"] = [
                    row["capacity"]["legacy_clipped_total"] for row in group
                ]
            if group[0].get("closed_loop") is not None:
                fields["closed_loop_nrmse"] = [row["closed_loop"]["nrmse"] for row in group]
                fields["valid_prediction_time"] = [
                    row["closed_loop"]["valid_prediction_time"] for row in group
                ]
            for target in group[0]["scores"]:
                for metric in (
                    "r2",
                    "rmse",
                    "nrmse",
                    "mae",
                    "pearson_r",
                    "classification_accuracy",
                ):
                    if metric in group[0]["scores"][target]:
                        fields[f"{target}/{metric}"] = [
                            row["scores"][target][metric] for row in group
                        ]
            key = f"{task}/{method}"
            summaries[key] = {
                name: hierarchical_bootstrap_summary(
                    values,
                    [row["dataset_seed"] for row in group],
                    [row["reservoir_seed"] for row in group],
                    seed=config["bootstrap_seed"],
                    scientific=scientific,
                )
                for name, values in fields.items()
            }
            for metric, summary in summaries[key].items():
                interval = summary["bootstrap95"]
                flat.append(
                    {
                        "task": task,
                        "method": method,
                        "metric": metric,
                        "mean": summary["mean"],
                        "std": summary["std"],
                        "median": summary["median"],
                        "ci_low": None if interval is None else interval[0],
                        "ci_high": None if interval is None else interval[1],
                        "scientific_uncertainty_valid": summary["scientific_uncertainty_valid"],
                    }
                )
    return summaries, flat


def summary_csv_text(rows: list[dict]) -> str:
    if not rows:
        raise ValueError("Summary table cannot be empty")
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def summary_markdown(rows: list[dict]) -> str:
    lines = [
        "| Task | Method | Metric | Mean | SD | 95% hierarchical bootstrap CI |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        interval = (
            "not estimated"
            if row["ci_low"] is None
            else f"[{row['ci_low']:.4g}, {row['ci_high']:.4g}]"
        )
        standard_deviation = "not estimated" if row["std"] is None else f"{row['std']:.3g}"
        lines.append(
            f"| {row['task']} | {row['method']} | {row['metric']} | "
            f"{row['mean']:.4g} | {standard_deviation} | {interval} |"
        )
    return "\n".join(lines)


def twin_validation_csv_text(records: list[dict]) -> str:
    columns = (
        "reservoir_seed",
        "tier",
        "memory_modes",
        "transmissivity",
        "measurement_noise",
        "step",
        "mean_max_error",
        "covariance_max_error",
        "feature_max_error",
    )
    rows = []
    for record in records:
        for step, (mean, covariance, feature) in enumerate(
            zip(
                record["mean_max_error_by_step"],
                record["covariance_max_error_by_step"],
                record["feature_max_error_by_step"],
                strict=True,
            ),
            start=1,
        ):
            rows.append(
                {
                    "reservoir_seed": record["reservoir_seed"],
                    "tier": record["tier"],
                    "memory_modes": record["memory_modes"],
                    "transmissivity": record["transmissivity"],
                    "measurement_noise": record["measurement_noise"],
                    "step": step,
                    "mean_max_error": mean,
                    "covariance_max_error": covariance,
                    "feature_max_error": feature,
                }
            )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def capacity_calibration_csv_text(payload: dict) -> str:
    columns = (
        "null_method",
        "system",
        "dataset_count",
        "hypothesis_count",
        "significant_count",
        "significant_rate",
        "mean_bias_corrected_signed_total",
        "mean_significant_capacity",
        "predeclared_type1_interval",
        "calibration_pass",
    )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in payload["summary"]:
        writer.writerow({**row, "predeclared_type1_interval": row["predeclared_type1_interval"]})
    return handle.getvalue()


def paired_model_comparisons(config: dict, runs: list[dict], reference="cv_B") -> list[dict]:
    """Predeclared paired comparisons without treating trajectory steps as replicates."""
    if reference not in {row["method"] for row in runs}:
        return []
    extractors = {
        "iid/significant_capacity": lambda row: row.get("primary_capacity"),
        "mg/horizon1_r2": lambda row: row.get("scores", {}).get("mackey_glass_h1", {}).get("r2"),
        "mg/closed_loop_nrmse": lambda row: (
            None if row.get("closed_loop") is None else row["closed_loop"].get("nrmse")
        ),
    }
    lookup = {
        (row["task"], row["method"], row["dataset_seed"], row["reservoir_seed"]): row
        for row in runs
    }
    comparisons = []
    for metric_name, extractor in extractors.items():
        task = metric_name.split("/", 1)[0]
        reference_rows = [row for row in runs if row["task"] == task and row["method"] == reference]
        for method in sorted({row["method"] for row in runs if row["task"] == task} - {reference}):
            pairs = []
            for reference_row in reference_rows:
                key = (
                    task,
                    method,
                    reference_row["dataset_seed"],
                    reference_row["reservoir_seed"],
                )
                candidate = lookup.get(key)
                if candidate is None:
                    continue
                left, right = extractor(reference_row), extractor(candidate)
                if left is not None and right is not None:
                    pairs.append((reference_row, float(left), float(right)))
            if not pairs:
                continue
            summary = paired_hierarchical_summary(
                [pair[1] for pair in pairs],
                [pair[2] for pair in pairs],
                [pair[0]["dataset_seed"] for pair in pairs],
                [pair[0]["reservoir_seed"] for pair in pairs],
                seed=config["bootstrap_seed"],
                scientific=config["study_class"] != "ci-smoke",
            )
            dataset_mean_values = []
            for dataset_seed in sorted({pair[0]["dataset_seed"] for pair in pairs}):
                dataset_mean_values.append(
                    np.mean(
                        [
                            pair[1] - pair[2]
                            for pair in pairs
                            if pair[0]["dataset_seed"] == dataset_seed
                        ]
                    )
                )
            dataset_means = np.asarray(dataset_mean_values)
            if len(dataset_means) <= 16:
                signs = np.array(
                    [
                        [1 if mask & (1 << bit) else -1 for bit in range(len(dataset_means))]
                        for mask in range(2 ** len(dataset_means))
                    ]
                )
                null_effects = np.mean(signs * dataset_means, axis=1)
                p_value = float(np.mean(abs(null_effects) >= abs(dataset_means.mean()) - 1e-15))
            else:
                rng = np.random.default_rng(config["bootstrap_seed"])
                signs = rng.choice((-1, 1), size=(10000, len(dataset_means)))
                null_effects = np.mean(signs * dataset_means, axis=1)
                p_value = float(
                    (1 + np.sum(abs(null_effects) >= abs(dataset_means.mean()))) / 10001
                )
            comparisons.append(
                {
                    "task_metric": metric_name,
                    "reference_method": reference,
                    "comparison_method": method,
                    "paired_units": len(pairs),
                    "dataset_clusters": len(dataset_means),
                    "mean_paired_difference_reference_minus_comparison": summary[
                        "mean_paired_difference"
                    ],
                    "median_paired_difference_reference_minus_comparison": summary[
                        "median_paired_difference"
                    ],
                    "paired_effect_size_dz": summary["paired_effect_size_dz"],
                    "bootstrap95": summary["bootstrap95"],
                    "cluster_sign_flip_p_value": p_value,
                    "cluster_sign_flip_scope": "dataset-cluster means; time steps are not replicates",
                }
            )
    if not comparisons:
        return []
    q_values = benjamini_hochberg([row["cluster_sign_flip_p_value"] for row in comparisons])
    for row, q_value in zip(comparisons, q_values, strict=True):
        row["bh_q_value_across_declared_model_comparisons"] = q_value
    return comparisons


def paired_comparisons_csv_text(rows: list[dict]) -> str:
    columns = (
        "task_metric",
        "reference_method",
        "comparison_method",
        "paired_units",
        "dataset_clusters",
        "mean_paired_difference_reference_minus_comparison",
        "median_paired_difference_reference_minus_comparison",
        "paired_effect_size_dz",
        "bootstrap95",
        "cluster_sign_flip_p_value",
        "bh_q_value_across_declared_model_comparisons",
        "cluster_sign_flip_scope",
    )
    return _dict_rows_csv(columns, rows)


def _dict_rows_csv(columns, rows):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def closed_loop_csv_text(runs: list[dict]) -> str:
    columns = (
        "job_id",
        "method",
        "dataset_seed",
        "reservoir_seed",
        "step",
        "prediction",
        "normalized_absolute_error",
        "threshold",
        "diverged",
    )
    rows = []
    for run in runs:
        rollout = run.get("closed_loop")
        if rollout is None:
            continue
        for step, (prediction, error) in enumerate(
            zip(
                rollout["predictions"],
                rollout["normalized_absolute_error"],
                strict=True,
            ),
            start=1,
        ):
            rows.append(
                {
                    "job_id": run["job_id"],
                    "method": run["method"],
                    "dataset_seed": run["dataset_seed"],
                    "reservoir_seed": run["reservoir_seed"],
                    "step": step,
                    "prediction": prediction,
                    "normalized_absolute_error": error,
                    "threshold": rollout["threshold"],
                    "diverged": rollout["diverged"],
                }
            )
    return _dict_rows_csv(columns, rows)


def feature_spectrum_csv_text(runs: list[dict]) -> str:
    columns = (
        "job_id",
        "task",
        "method",
        "dataset_seed",
        "reservoir_seed",
        "component",
        "eigenvalue",
    )
    rows = []
    for run in runs:
        for component, value in enumerate(run["diagnostics"]["covariance_spectrum"], start=1):
            rows.append(
                {
                    "job_id": run["job_id"],
                    "task": run["task"],
                    "method": run["method"],
                    "dataset_seed": run["dataset_seed"],
                    "reservoir_seed": run["reservoir_seed"],
                    "component": component,
                    "eigenvalue": value,
                }
            )
    return _dict_rows_csv(columns, rows)


def prediction_example_csv_text(runs: list[dict]) -> str:
    columns = (
        "job_id",
        "method",
        "dataset_seed",
        "reservoir_seed",
        "step",
        "target",
        "prediction",
    )
    mg_runs = [run for run in runs if run["task"] == "mg" and "mackey_glass_h1" in run["scores"]]
    if not mg_runs:
        return _dict_rows_csv(columns, [])
    selected_dataset = min(run["dataset_seed"] for run in mg_runs)
    selected_reservoir = min(run["reservoir_seed"] for run in mg_runs)
    rows = []
    for run in mg_runs:
        if run["dataset_seed"] != selected_dataset or run["reservoir_seed"] != selected_reservoir:
            continue
        names = list(run["scores"])
        column = names.index("mackey_glass_h1")
        target = np.asarray(run["splits"]["test"]["targets"])[:, column]
        prediction = run["predictions"]["mackey_glass_h1"]
        for step, (truth, estimate) in enumerate(zip(target, prediction, strict=True), start=1):
            rows.append(
                {
                    "job_id": run["job_id"],
                    "method": run["method"],
                    "dataset_seed": run["dataset_seed"],
                    "reservoir_seed": run["reservoir_seed"],
                    "step": step,
                    "target": truth,
                    "prediction": estimate,
                }
            )
    return _dict_rows_csv(columns, rows)
