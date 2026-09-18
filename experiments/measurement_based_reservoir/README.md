# Measurement-based reservoir experiments

The V3 runner is manifest-driven and separates CI, development, publication and
legacy outputs. Historical V2 artifacts are preserved under `results_legacy/`;
they were generated with the former `photographiqml` namespace and are not V3
evidence.

Run the self-contained core smoke study with:

```sh
python main.py --config config_ci.json --output /tmp/cv-mb-qrc-ci
python verify_provenance.py /tmp/cv-mb-qrc-ci
```

`config_ci.json` enables only CV Tier A, CV Tier B and the delay baseline. It
does not import Graphix or MentPy and permits one dataset/reservoir seed without
claiming a standard deviation or confidence interval. `config_small.json` is a
non-publication development study. `config_publication.json` predeclares at least
five dataset seeds, ten reservoir seeds, longer sequences, permutation nulls and
the required baseline set; it is intentionally unexecuted.

## Features and models

Tier A is first moments plus diagonal covariance: 8 features for two memory
modes. Tier B is first moments plus the full upper covariance triangle: 14
features for two modes. Diagnostic second-moment presets are separate and are
not headline task features.

`GaussianClassicalTwin` identifies the fixed PhotoGraphiQ Gaussian channel once,
then advances `mu'=A mu+B u+c` and `V'=A V A.T+N` using NumPy. It uses the same
mask, bias, features, chronological partitions and ridge selection as the CV
reservoir. Agreement is evidence of efficient classical reproducibility, not
quantum advantage.

The executable Graphix implementation remains a one-memory-qubit collision
reference. A guarded builder validates chain, ring, star, brickwork and explicit
small topology designs, but those designs are not presented as executable
generic reservoirs. MentPy
validates only the corrected pure-wire common subset and never CV physics or a
general mixed-state Graphix channel.

## Statistical protocol

The hierarchy is dataset seed, reservoir seed, task, then method. Filenames and
records preserve both seed roles. Mackey–Glass datasets vary their initial
history by dataset seed. Teacher-forced targets are `x[t+h]` for configured
horizons and use only information through `t`. The chronological embargo must
cover both maximum history and maximum horizon. The selected horizon-one model
also runs a recursive closed-loop rollout from observed context; each raw record
stores NRMSE, normalized error by rollout time, valid-prediction time and a
divergence flag.

Readout input access is explicit: `reservoir_only` excludes the raw-input skip;
`reservoir_plus_input` includes it. Train-only standardization and collapsed
column filtering precede validation-only ridge selection. Test targets are
evaluated once after selection.

Capacity targets use orthogonal linear, quadratic-self and cross-delay
polynomials. Null transformations are applied to the complete target matrix
before chronological splitting. Circular shift, block permutation and
independent-surrogate nulls are calibrated; block permutation is the default
because development calibration produced 0/270 discoveries across its three
null controls while retaining 31/90 known linear-memory components. Raw R²,
signed bias correction, finite-sample empirical p, BH q, null intervals and
separate families are stored. A positive corrected value is not capacity
evidence unless it passes the declared FDR threshold.

Scientific confidence intervals use a hierarchical bootstrap: dataset seeds are
sampled first, followed by reservoir seeds within each sampled dataset. The CI
smoke study has one observation and therefore records no standard deviation or
confidence interval.

## Readout and optional stages

Only `state_oracle` readout is implemented. It is explicitly labelled a
simulation-only upper-bound readout. `physical_probe` constructs
`TapHomodyneReadout`, which raises `BackendCapabilityError` because the audited
public PhotoGraphiQ API does not yet provide the required tap beamsplitter,
selected-quadrature homodyne instrument and surviving-memory channel. No joint
q/p Gaussian sampling is substituted.

Stages are controlled by `enabled_stages`: core experiments, dynamics, shot,
washout and scaling studies, Graphix diagnostics, MentPy validation, Fock
control and report generation.
Optional modules are imported only inside their enabled stages. The report reads
available methods and tasks from the completed manifest and records why optional
panels were skipped.

## Provenance

Every run records resolved configuration and hash, repository and PhotoGraphiQ
commits/dirty status, implementation hashes, dataset hashes, split hash, seed
roles, dependency versions, exact command, timestamps, expected/completed/failure
manifests and report status. `verify_provenance.py` reports exact mismatches.
The content index hashes every raw job, atomic completion marker, cache, table,
figure and report. Verification reconstructs all stored tables from raw jobs.
Publication validity additionally requires publication study class, clean source
worktrees, exact summary hashes, all required methods and seeds, complete null
repetitions and completed report generation. Every raw JSON file is rejected if
it contains NaN or Infinity.

The Fock cutoff path remains an explicitly postselected numerical control. Its
success and failure probabilities must be reported; it is not a deterministic
reservoir or evidence of non-Gaussian advantage.
