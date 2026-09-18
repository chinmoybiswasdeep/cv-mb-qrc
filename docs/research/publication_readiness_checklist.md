# Publication-readiness checklist

Baseline captured at commit `4a5c656b53bff69f1e354c816bb0decc787e334d`
on branch `fix/research-v3`. The starting worktree was clean.

This is a living engineering checklist. A checked implementation item means the
code and focused tests exist; it does not mean a publication-scale experiment
has been executed.

## Evidence integrity

- [x] Add a failing regression demonstrating undetected raw-job tampering.
- [x] Hash every registered raw job and atomic completion marker.
- [x] Detect missing, extra, duplicate, malformed, truncated, or changed jobs.
- [x] Hash configuration, environment, manifests, tables, figures, and report.
- [x] Recompute summaries and tables from raw jobs during verification.
- [x] Add figure-to-table/config/raw-job provenance.
- [x] Add `cvmbqrc verify-run` and tamper tests.

## Statistics and tasks

- [x] Move capacity null transforms before chronological splitting.
- [x] Implement circular-shift, block-permutation, and independent-surrogate nulls.
- [x] Store null descriptors, calibrated displacement constraints, p-values, and BH q-values.
- [x] Separate linear, quadratic-self, and cross-delay capacity families.
- [x] Add null-system and deliberately leaky calibration controls.
- [x] Add paired hierarchical effect summaries without treating time points as replicates.
- [x] Retain negative R2 and add RMSE, NRMSE, MAE, and correlation reporting.

## Models and backend scope

- [x] Extend resource-matched classical controls and reject unfair configurations.
- [x] Promote Gaussian-twin state/runtime/scaling agreement to a report artifact.
- [x] Audit Graphix topology scope; generalize only where stable APIs permit it.
- [x] Publish a Graphix/MentPy/backend compatibility matrix.
- [x] Keep unavailable tap-homodyne explicit and add a measurement-budget roadmap.
- [x] Expand the guarded Fock feasibility record and postselection-cost reporting.

## Engine, figures, and reports

- [x] Add dry-run job/cost estimation and explicit publication confirmation.
- [x] Add content-addressed feature caching and retry-safe atomic job completion.
- [x] Add deterministic worker configuration and serial/parallel equivalence tests.
- [x] Centralize a colorblind-safe publication plotting style.
- [x] Generate figures only from verified tables, with panel provenance and captions.
- [x] Generate claims/evidence, limitations, methods, and reproducibility report sections.
- [x] Add CI, development, publication, and scaling configurations.

## Quality gates and executed evidence

- [x] Run focused tests after every major stage.
- [x] Run full pytest, coverage, Ruff, format, mypy, and build gates.
- [x] Run clean-archive installation.
- [x] Run a fresh CI smoke study and tamper it only in a copied directory.
- [x] Run development null calibration and development figures.
- [x] Record publication resource estimate; do not launch publication automatically.
