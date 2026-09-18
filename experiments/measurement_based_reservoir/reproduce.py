"""Rebuild tables, figures, captions, and claims from registered raw jobs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cv_mb_qrc.reservoirs.plotting import apply_publication_style, method_styles, panel_label
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
from cv_mb_qrc.reservoirs.results import atomic_json, atomic_text, sha256_file, sha256_json


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

    summaries, flat = build_summaries(config, runs)
    atomic_json(output / "json/summary.json", summaries)
    atomic_text(output / "csv/summary.csv", summary_csv_text(flat))
    atomic_text(output / "tables/results.md", summary_markdown(flat))
    atomic_text(output / "csv/closed_loop.csv", closed_loop_csv_text(runs))
    atomic_text(output / "csv/feature_spectrum.csv", feature_spectrum_csv_text(runs))
    atomic_text(output / "csv/prediction_example.csv", prediction_example_csv_text(runs))
    paired_comparisons = paired_model_comparisons(config, runs)
    atomic_json(output / "json/paired_model_comparisons.json", paired_comparisons)
    atomic_text(
        output / "csv/paired_model_comparisons.csv",
        paired_comparisons_csv_text(paired_comparisons),
    )

    apply_publication_style()
    styles = method_styles(available_methods)
    generated, skipped, figure_provenance = [], {}, {}

    def save(fig, name, filters, *, source_table="csv/summary.csv", raw_artifacts=None):
        fig.tight_layout()
        for suffix in ("svg", "pdf", "png"):
            fig.savefig(output / f"figures/{name}.{suffix}", bbox_inches="tight", dpi=600)
        plt.close(fig)
        generated.append(name)
        relevant_jobs = [
            key
            for key, row in zip(manifest, runs, strict=True)
            if ("task" not in filters or row["task"] == filters["task"])
            and ("methods" not in filters or row["method"] in filters["methods"])
        ]
        source_path = output / source_table
        figure_provenance[name] = {
            "source_table": source_table,
            "source_table_sha256": sha256_file(source_path),
            "configuration_sha256": sha256_json(config),
            "raw_jobs": relevant_jobs,
            "filters": filters,
            "formats": [f"figures/{name}.{suffix}" for suffix in ("svg", "pdf", "png")],
            "raw_artifacts": raw_artifacts or [],
        }

    iid_methods = [method for method in available_methods if f"iid/{method}" in summaries]
    if iid_methods:
        fig, ax = plt.subplots(figsize=(7.2, 4))
        for method in iid_methods:
            group = summaries[f"iid/{method}"]
            names = [name for name in group if name.startswith("linear_") and name.endswith("/r2")]
            names.sort(key=lambda name: int(name.split("_")[1].split("/")[0]))
            if names:
                ax.errorbar(
                    range(1, len(names) + 1),
                    [group[name]["mean"] for name in names],
                    yerr=[_error(group[name]) for name in names],
                    **styles[method],
                    label=method,
                )
        ax.set(xlabel="Delay", ylabel="Raw test R² (negative values retained)")
        ax.legend()
        save(fig, "linear_memory", {"task": "iid", "methods": iid_methods, "metric": "r2"})
        capacity_methods = [
            method for method in iid_methods if "significant_capacity" in summaries[f"iid/{method}"]
        ]
        if capacity_methods:
            fig, ax = plt.subplots(figsize=(7.2, 4))
            values = [
                summaries[f"iid/{method}"]["significant_capacity"] for method in capacity_methods
            ]
            ax.bar(
                capacity_methods,
                [value["mean"] for value in values],
                yerr=[_error(value) for value in values],
            )
            ax.tick_params(axis="x", rotation=30)
            ax.set(ylabel="FDR-significant null-corrected capacity")
            save(
                fig,
                "significant_capacity",
                {
                    "task": "iid",
                    "methods": capacity_methods,
                    "metric": "significant_capacity",
                },
            )
    else:
        skipped["linear_memory"] = "no iid methods in manifest"

    mg_methods = [method for method in available_methods if f"mg/{method}" in summaries]
    if mg_methods:
        horizons = config["forecast_horizons"]
        fig, ax = plt.subplots(figsize=(7.2, 4))
        for method in mg_methods:
            group = summaries[f"mg/{method}"]
            ax.errorbar(
                horizons,
                [group[f"mackey_glass_h{h}/r2"]["mean"] for h in horizons],
                yerr=[_error(group[f"mackey_glass_h{h}/r2"]) for h in horizons],
                **styles[method],
                label=method,
            )
        ax.set(xlabel="Forecast horizon", ylabel="Teacher-forced test R²")
        ax.legend()
        save(
            fig,
            "mackey_glass_horizons",
            {"task": "mg", "methods": mg_methods, "metric": "r2"},
        )

    if iid_methods:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
        runtime = [summaries[f"iid/{method}"]["seconds"] for method in iid_methods]
        ranks = [summaries[f"iid/{method}"]["effective_rank"] for method in iid_methods]
        axes[0].bar(iid_methods, [value["mean"] for value in runtime])
        axes[1].bar(iid_methods, [value["mean"] for value in ranks])
        for axis in axes:
            axis.tick_params(axis="x", rotation=30)
        axes[0].set(ylabel="Wall-clock seconds")
        axes[1].set(ylabel="Effective numerical rank")
        panel_label(axes[0], "(a)")
        panel_label(axes[1], "(b)")
        save(fig, "runtime_and_rank", {"task": "iid", "methods": iid_methods})

        fig, ax = plt.subplots(figsize=(3.5, 3.2))
        for method in iid_methods:
            spectra = [
                np.asarray(row["diagnostics"]["covariance_spectrum"], float)
                for row in runs
                if row["task"] == "iid" and row["method"] == method
            ]
            if not spectra:
                continue
            width = min(map(len, spectra))
            aligned = np.vstack([values[:width] for values in spectra])
            for values in aligned:
                ax.plot(
                    np.arange(1, width + 1),
                    np.maximum(values, np.finfo(float).eps),
                    color=styles[method]["color"],
                    alpha=0.15,
                    linewidth=0.6,
                )
            ax.plot(
                np.arange(1, width + 1),
                np.maximum(np.median(aligned, axis=0), np.finfo(float).eps),
                label=method,
                **styles[method],
            )
        ax.set(xlabel="Spectrum component", ylabel="Feature covariance eigenvalue", yscale="log")
        ax.legend()
        save(
            fig,
            "feature_covariance_spectrum",
            {"task": "iid", "methods": iid_methods},
            source_table="csv/feature_spectrum.csv",
        )

    twin_path = raw / "twin_validation.json"
    if twin_path.exists():
        twin_records = json.loads(twin_path.read_text(encoding="utf-8"))
        atomic_text(
            output / "csv/twin_validation.csv",
            twin_validation_csv_text(twin_records),
        )
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharex=True)
        specifications = (
            ("mean_max_error_by_step", "First-moment max error"),
            ("covariance_max_error_by_step", "Covariance max error"),
            ("feature_max_error_by_step", "Feature max error"),
        )
        for (field, label), axis in zip(specifications, axes, strict=True):
            for tier in ("A", "B"):
                subset = [
                    np.asarray(row[field], float) for row in twin_records if row["tier"] == tier
                ]
                if subset:
                    values = np.max(np.vstack(subset), axis=0)
                    axis.plot(
                        np.arange(1, len(values) + 1),
                        np.maximum(values, np.finfo(float).eps),
                        label=f"Tier {tier}",
                    )
            axis.set(xlabel="Sequence step", ylabel=label, yscale="log")
        axes[0].legend()
        save(
            fig,
            "gaussian_twin_agreement",
            {"aggregation": "maximum across declared validation cases"},
            source_table="csv/twin_validation.csv",
            raw_artifacts=["raw/twin_validation.json"],
        )

    closed_loop_runs = [row for row in runs if row.get("closed_loop") is not None]
    if closed_loop_runs:
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        for method in sorted({row["method"] for row in closed_loop_runs}):
            values = [
                np.asarray(row["closed_loop"]["normalized_absolute_error"], float)
                for row in closed_loop_runs
                if row["method"] == method
            ]
            width = min(map(len, values))
            aligned = np.vstack([row[:width] for row in values])
            for row in aligned:
                ax.plot(
                    np.arange(1, width + 1),
                    row,
                    color=styles[method]["color"],
                    alpha=0.15,
                    linewidth=0.6,
                )
            ax.plot(
                np.arange(1, width + 1),
                np.median(aligned, axis=0),
                label=method,
                **styles[method],
            )
        ax.axhline(config["closed_loop_threshold"], color="black", linestyle=":")
        ax.set(xlabel="Closed-loop rollout step", ylabel="Normalized absolute error")
        ax.legend()
        save(
            fig,
            "closed_loop_error_growth",
            {"task": "mg", "methods": mg_methods},
            source_table="csv/closed_loop.csv",
        )

        selected_dataset = min(row["dataset_seed"] for row in closed_loop_runs)
        selected_reservoir = min(row["reservoir_seed"] for row in closed_loop_runs)
        example_runs = [
            row
            for row in runs
            if row["task"] == "mg"
            and row["dataset_seed"] == selected_dataset
            and row["reservoir_seed"] == selected_reservoir
        ]
        if example_runs:
            fig, ax = plt.subplots(figsize=(7.2, 3.4))
            first = example_runs[0]
            target_column = list(first["scores"]).index("mackey_glass_h1")
            truth = np.asarray(first["splits"]["test"]["targets"])[:, target_column]
            ax.plot(truth, color="black", linewidth=1.5, label="Target")
            for row in example_runs:
                ax.plot(
                    row["predictions"]["mackey_glass_h1"],
                    label=row["method"],
                    **styles[row["method"]],
                )
            ax.set(xlabel="Test time index", ylabel="Mackey–Glass value")
            ax.legend(ncol=2)
            save(
                fig,
                "prediction_example",
                {
                    "task": "mg",
                    "methods": [row["method"] for row in example_runs],
                    "dataset_seed": selected_dataset,
                    "reservoir_seed": selected_reservoir,
                },
                source_table="csv/prediction_example.csv",
            )

    pareto_methods = [
        method for method in mg_methods if "mackey_glass_h1/r2" in summaries[f"mg/{method}"]
    ]
    if pareto_methods:
        fig, ax = plt.subplots(figsize=(3.5, 3.2))
        for method in pareto_methods:
            runtime = summaries[f"mg/{method}"]["seconds"]
            score = summaries[f"mg/{method}"]["mackey_glass_h1/r2"]
            ax.scatter(
                runtime["values"],
                score["values"],
                alpha=0.3,
                color=styles[method]["color"],
                marker=styles[method]["marker"],
            )
            ax.scatter(
                runtime["median"],
                score["median"],
                label=method,
                s=45,
                edgecolor="black",
                color=styles[method]["color"],
                marker=styles[method]["marker"],
            )
        ax.set(xlabel="Wall-clock seconds per job", ylabel="Horizon-1 test R²")
        ax.legend()
        save(
            fig,
            "accuracy_compute_pareto",
            {"task": "mg", "methods": pareto_methods, "metrics": ["seconds", "r2"]},
            source_table="json/summary.json",
        )

    calibration_path = raw / "capacity_calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        atomic_text(
            output / "csv/capacity_calibration.csv",
            capacity_calibration_csv_text(calibration),
        )
        null_rows = [
            row for row in calibration["summary"] if row["predeclared_type1_interval"] is not None
        ]
        fig, ax = plt.subplots(figsize=(7.2, 3.5))
        labels = [f"{row['null_method']}\n{row['system']}" for row in null_rows]
        colors = ["#0072B2" if row["calibration_pass"] else "#D55E00" for row in null_rows]
        ax.bar(labels, [row["significant_rate"] for row in null_rows], color=colors)
        upper = calibration["acceptable_type1_interval_predeclared"][1]
        ax.axhline(upper, color="black", linestyle="--", label="Predeclared upper bound")
        ax.set(ylabel="FDR-significant component rate", ylim=(0, max(0.12, upper * 1.4)))
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
        save(
            fig,
            "capacity_null_calibration",
            {"systems": "predeclared null controls"},
            source_table="csv/capacity_calibration.csv",
            raw_artifacts=["raw/capacity_calibration.json"],
        )

    optional_files = {
        "mentpy": raw / "mentpy.json",
        "fock": raw / "fock.json",
        "graphix": raw / "qubit_contraction.json",
        "physical_readout": raw / "physical_readout.json",
    }
    for panel, artifact in optional_files.items():
        if not artifact.exists():
            skipped[panel] = "stage disabled or valid artifact unavailable"
    report_status = {
        "study_class": config["study_class"],
        "available_methods": available_methods,
        "available_tasks": available_tasks,
        "generated_figures": generated,
        "skipped_panels": skipped,
        "raw_run_count": len(runs),
    }
    atomic_json(output / "json/figure_provenance.json", figure_provenance)
    software_evidence = "raw/provenance.json"
    claims = [
        {
            "claim": "The evidence bundle is internally reproducible from registered raw jobs",
            "status": "supported after cvmbqrc verify-run succeeds",
            "configuration": "raw/config.json",
            "raw_jobs": manifest,
            "summary_table": "csv/summary.csv",
            "figure": None,
            "statistical_test": "not applicable; content and reconstruction verification",
            "software_version": software_evidence,
        },
        {
            "claim": "Gaussian CV results require comparison with the exact affine Gaussian twin",
            "status": (
                "supported by state-level validation in this run"
                if twin_path.exists()
                else "methodological claim; validation stage absent from this run"
            ),
            "configuration": "raw/config.json",
            "raw_jobs": [
                key for key in manifest if "gaussian_classical_twin" in key or "_cv_" in key
            ],
            "summary_table": (
                "csv/twin_validation.csv" if twin_path.exists() else "csv/summary.csv"
            ),
            "figure": ("figures/gaussian_twin_agreement.pdf" if twin_path.exists() else None),
            "statistical_test": "paired per-step numerical error; no quantum-advantage test",
            "software_version": software_evidence,
        },
        {
            "claim": "The selected capacity null controls false discoveries in calibration",
            "status": (
                "development calibration evidence available"
                if calibration_path.exists()
                else "unsupported in this run"
            ),
            "configuration": "raw/config.json",
            "raw_jobs": [],
            "raw_artifacts": (
                ["raw/capacity_calibration.json"] if calibration_path.exists() else []
            ),
            "summary_table": (
                "csv/capacity_calibration.csv" if calibration_path.exists() else None
            ),
            "figure": (
                "figures/capacity_null_calibration.pdf" if calibration_path.exists() else None
            ),
            "statistical_test": "finite-sample empirical p-values and BH correction",
            "software_version": software_evidence,
        },
        {
            "claim": "A reservoir outperforms a matched control",
            "status": "unsupported unless paired corrected comparisons reject the null",
            "configuration": "raw/config.json",
            "raw_jobs": manifest,
            "summary_table": "csv/paired_model_comparisons.csv",
            "figure": "figures/accuracy_compute_pareto.pdf",
            "statistical_test": "paired hierarchical interval, effect size, cluster sign flip, BH q",
            "software_version": software_evidence,
        },
        {
            "claim": "Physical tap-homodyne readout is implemented",
            "status": "unsupported",
            "limitation": "public backend tap/instrument/surviving-memory capability unavailable",
            "configuration": "raw/config.json",
            "raw_jobs": [],
            "summary_table": None,
            "figure": None,
            "statistical_test": "not applicable",
            "software_version": software_evidence,
        },
        {
            "claim": "Graphix provides a generic executable MBQC reservoir",
            "status": "unsupported; executable scope is the collision-model family",
            "configuration": "raw/config.json",
            "raw_jobs": [key for key in manifest if key.endswith("_graphix")],
            "summary_table": "csv/summary.csv",
            "figure": None,
            "statistical_test": "independent Kraus/density-matrix small-case validation",
            "software_version": software_evidence,
        },
        {
            "claim": "MentPy agreement extends beyond corrected pure XY wires",
            "status": "unsupported",
            "configuration": "raw/config.json",
            "raw_jobs": [],
            "raw_artifacts": ["raw/mentpy.json"] if (raw / "mentpy.json").exists() else [],
            "summary_table": None,
            "figure": None,
            "statistical_test": "state/probability agreement on the verified common subset",
            "software_version": software_evidence,
        },
        {
            "claim": "Fock results demonstrate non-Gaussian performance advantage",
            "status": "unsupported; current scope is numerical feasibility",
            "configuration": "raw/config.json",
            "raw_jobs": [],
            "raw_artifacts": ["raw/fock.json"] if (raw / "fock.json").exists() else [],
            "summary_table": None,
            "figure": None,
            "statistical_test": "cutoff convergence and postselection cost only",
            "software_version": software_evidence,
        },
    ]
    atomic_json(output / "json/claims_evidence.json", claims)
    captions = {
        name: {
            "draft": (
                f"{name.replace('_', ' ').title()}. Points or faint lines show individual "
                f"dataset/reservoir repetitions where available; summaries use the declared "
                f"hierarchical bootstrap. See json/figure_provenance.json for exact filters."
            ),
            "independent_dataset_seeds": len(config["dataset_seeds"]),
            "reservoir_seeds_per_dataset": len(config["reservoir_seeds"]),
            "uncertainty": (
                "dataset-then-reservoir hierarchical bootstrap"
                if config["study_class"] != "ci-smoke"
                else "not estimated in one-seed CI smoke"
            ),
        }
        for name in generated
    }
    atomic_json(output / "json/figure_captions.json", captions)
    report = "\n".join(
        [
            "# Reproducible V3 study report",
            "",
            "## Executive summary",
            "",
            f"This {config['study_class']} run contains {len(runs)} registered raw jobs. "
            "Gaussian CV results are interpreted through their exact classical twin and "
            "do not establish quantum advantage.",
            "",
            "## Evidence integrity",
            "",
            "Tables are reconstructed from registered raw jobs during verification. "
            "Figures are content-indexed and linked to their source table and job filters.",
            "",
            "## Scientific claims",
            "",
            "See json/claims_evidence.json for machine-readable claim-to-evidence links.",
            "",
            "## Methods and results",
            "",
            "See raw/config.json, csv/summary.csv, and tables/results.md.",
            "Paired effects and dataset-cluster sign-flip tests are in "
            "csv/paired_model_comparisons.csv; trajectory time points are never replicates.",
            "",
            "## Dataset and statistical analysis",
            "",
            f"Dataset seeds: {config['dataset_seeds']}; reservoir seeds: "
            f"{config['reservoir_seeds']}. Capacity uses {config['capacity_null_method']} "
            "nulls generated before splitting, finite-sample empirical p-values, and BH FDR.",
            "",
            "## Ablations and validation",
            "",
            "Enabled stages and missing optional artifacts are recorded in raw/report_status.json. "
            "Twin validation, null calibration, Graphix/MentPy validation, and Fock feasibility "
            "are reported only when their registered raw artifacts exist.",
            "",
            "## Runtime and resources",
            "",
            "Per-job runtime, sampled memory, feature rank, readout budget, and simulator resource "
            "counts are retained in raw jobs; aggregate views are in the summary and Pareto figure.",
            "",
            "## Reproducibility",
            "",
            "Run `cvmbqrc verify-run <directory>` before using any table or figure. Exact source, "
            "dependency, seed, configuration, timestamp, and environment records are under raw/.",
            "",
            "## Limitations",
            "",
            "State-oracle Gaussian features are simulator observables; physical tap-homodyne "
            "remains unavailable. Graphix is scoped to the implemented collision/wire family. "
            "MentPy covers only the corrected pure-wire subset.",
        ]
    )
    atomic_text(output / "report.md", report)
    atomic_json(raw / "report_status.json", report_status)
    print(f"Regenerated {len(flat)} summary rows from {len(runs)} raw runs", flush=True)
    return report_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    reproduce(parser.parse_args().output)
