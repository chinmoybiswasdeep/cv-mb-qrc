# Reproducibility guide

## Environment and quality gates

Use Python 3.11–3.14 with the audited PhotoGraphiQ branch, then install
`.[dev,experiments,graphix,validation]`. Run:

```sh
pytest
pytest --cov=src/cv_mb_qrc --cov-report=term-missing --cov-fail-under=90
ruff check .
ruff format --check .
mypy .
python -m build
```

## Smoke and development evidence

```sh
python experiments/measurement_based_reservoir/main.py \
  --config experiments/measurement_based_reservoir/config_ci.json \
  --output experiments/measurement_based_reservoir/results_ci/current --workers 1
cvmbqrc verify-run experiments/measurement_based_reservoir/results_ci/current

python experiments/measurement_based_reservoir/main.py \
  --config experiments/measurement_based_reservoir/config_calibration_development.json \
  --output experiments/measurement_based_reservoir/results_development/current --workers 2
cvmbqrc verify-run experiments/measurement_based_reservoir/results_development/current
```

Use `--resume` only with the identical resolved configuration. The runner skips
content-bound completed jobs and rebuilds terminal reports.

## Publication run

Inspect cost without executing:

```sh
python experiments/measurement_based_reservoir/main.py \
  --config experiments/measurement_based_reservoir/config_publication.json --dry-run
```

The locked run requires an explicit acknowledgement:

```sh
python experiments/measurement_based_reservoir/main.py \
  --config experiments/measurement_based_reservoir/config_publication.json \
  --output experiments/measurement_based_reservoir/results_publication/final \
  --workers 8 --confirm-publication
```

For a scheduler, request a single node with at least 16 CPU cores, 32 GiB RAM,
and 200 GiB scratch; activate the environment and execute the same command with
`--workers` set to the allocated CPU count. The current dry estimate is 2,100
jobs, 21 million reservoir steps, roughly 1.052 million reused factorizations,
about 115.5 GB of JSON evidence, and a broad 1.2-hour to 5-day compute range.
Because JSON storage and the Graphix/Fock tails dominate uncertainty, run on
scratch and archive only after verification. Do not launch this configuration
on a laptop without redesigning storage.

Every final manuscript number must cite `raw/config.json`, the relevant raw job
set, a reconstructed CSV table, the figure provenance record, and the software
release containing the matching source hashes.
