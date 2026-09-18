"""Manifest-driven report generation from compatible raw results only."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cv_mb_qrc.reservoirs.diagnostics import hierarchical_bootstrap_summary
from cv_mb_qrc.reservoirs.results import atomic_json


def _error(summary):
    return 0.0 if summary["std"] is None else summary["std"]


def reproduce(output):
    output = Path(output)
    raw = output / "raw"
    config = json.loads((raw / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    runs = [json.loads((raw / f"{key}.json").read_text(encoding="utf-8")) for key in manifest]
    if not runs:
        raise ValueError("Cannot generate a report from an empty manifest")
    available_methods = sorted({row["method"] for row in runs})
    available_tasks = sorted({row["task"] for row in runs})
    required = config.get("required_publication_methods", [])
    missing = sorted(set(required) - set(available_methods))
    if config["study_class"] == "publication" and missing:
        raise ValueError(f"Publication report missing required methods: {missing}")
    for directory in ("csv", "json", "figures", "tables"):
        (output / directory).mkdir(exist_ok=True)
    scientific = config["study_class"] != "ci-smoke"
    summaries, flat = {}, []
    for task in available_tasks:
        for method in available_methods:
            group = [row for row in runs if row["task"] == task and row["method"] == method]
            if not group:
                continue
            fields = {
                "seconds": [row["seconds"] for row in group],
                "effective_rank": [row["diagnostics"]["effective_rank"] for row in group],
            }
            if group[0].get("capacity") is not None:
                fields["null_corrected_capacity"] = [
                    row["capacity"]["null_corrected_total"] for row in group
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
                fields[target] = [row["scores"][target]["r2"] for row in group]
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
    atomic_json(output / "json/summary.json", summaries)
    with (output / "csv/summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    table = [
        "| Task | Method | Metric | Mean | SD | 95% bootstrap CI |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in flat:
        interval = (
            "not estimated"
            if row["ci_low"] is None
            else f"[{row['ci_low']:.4g}, {row['ci_high']:.4g}]"
        )
        standard_deviation = "not estimated" if row["std"] is None else f"{row['std']:.3g}"
        table.append(
            f"| {row['task']} | {row['method']} | {row['metric']} | "
            f"{row['mean']:.4g} | {standard_deviation} | {interval} |"
        )
    (output / "tables/results.md").write_text("\n".join(table), encoding="utf-8")

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    generated, skipped = [], {}

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("svg", "pdf", "png"):
            fig.savefig(output / f"figures/{name}.{suffix}", bbox_inches="tight", dpi=150)
        plt.close(fig)
        generated.append(name)

    iid_methods = [method for method in available_methods if f"iid/{method}" in summaries]
    if iid_methods:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in iid_methods:
            group = summaries[f"iid/{method}"]
            names = [name for name in group if name.startswith("linear_")]
            names.sort(key=lambda name: int(name.split("_")[1]))
            if names:
                ax.errorbar(
                    range(1, len(names) + 1),
                    [group[name]["mean"] for name in names],
                    yerr=[_error(group[name]) for name in names],
                    marker="o",
                    label=method,
                )
        ax.set(xlabel="Delay", ylabel="Raw test R² (negative values retained)")
        ax.legend()
        save(fig, "linear_memory")
        capacity_methods = [
            method
            for method in iid_methods
            if "null_corrected_capacity" in summaries[f"iid/{method}"]
        ]
        if capacity_methods:
            fig, ax = plt.subplots(figsize=(7, 4))
            values = [
                summaries[f"iid/{method}"]["null_corrected_capacity"] for method in capacity_methods
            ]
            ax.bar(
                capacity_methods,
                [value["mean"] for value in values],
                yerr=[_error(value) for value in values],
            )
            ax.tick_params(axis="x", rotation=30)
            ax.set(ylabel="Permutation-null-corrected capacity")
            save(fig, "null_corrected_capacity")
    else:
        skipped["linear_memory"] = "no iid methods in manifest"

    mg_methods = [method for method in available_methods if f"mg/{method}" in summaries]
    if mg_methods:
        horizons = config["forecast_horizons"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in mg_methods:
            group = summaries[f"mg/{method}"]
            ax.errorbar(
                horizons,
                [group[f"mackey_glass_h{h}"]["mean"] for h in horizons],
                yerr=[_error(group[f"mackey_glass_h{h}"]) for h in horizons],
                marker="o",
                label=method,
            )
        ax.set(xlabel="Forecast horizon", ylabel="Teacher-forced test R²")
        ax.legend()
        save(fig, "mackey_glass_horizons")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    runtime = [summaries[f"iid/{method}"]["seconds"] for method in iid_methods]
    ranks = [summaries[f"iid/{method}"]["effective_rank"] for method in iid_methods]
    axes[0].bar(iid_methods, [value["mean"] for value in runtime])
    axes[1].bar(iid_methods, [value["mean"] for value in ranks])
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
    axes[0].set(ylabel="Seconds")
    axes[1].set(ylabel="Effective rank")
    save(fig, "runtime_and_rank")

    optional_files = {
        "mentpy": raw / "mentpy.json",
        "fock": raw / "fock.json",
        "graphix": raw / "qubit_contraction.json",
        "physical_readout": raw / "physical_readout.json",
    }
    for panel, path in optional_files.items():
        if not path.exists():
            skipped[panel] = "stage disabled or artifact unavailable"
    report_status = {
        "study_class": config["study_class"],
        "available_methods": available_methods,
        "available_tasks": available_tasks,
        "generated_figures": generated,
        "skipped_panels": skipped,
        "raw_run_count": len(runs),
    }
    atomic_json(raw / "report_status.json", report_status)
    print(f"Regenerated {len(flat)} summary rows from {len(runs)} raw runs", flush=True)
    return report_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    reproduce(parser.parse_args().output)
