# cv-mb-qrc

Stateful continuous-variable and qubit measurement-based quantum reservoir
computing, with leakage-controlled temporal benchmarks and reproducible artifacts.

This repository owns the reservoir orchestration, encoders, feature extraction,
ridge readout, diagnostics, benchmark datasets, experiments, tests and reports.
[PhotoGraphiQ](https://github.com/chinmoybiswasdeep/PhotoGraphiQ) remains the CV
execution and physics dependency. Graphix supplies the optional qubit backend;
MentPy is an optional independent corrected-wire reference.

The two temporal models are explicit:

- `CVMBReservoir` and `GraphixMBReservoir` carry the complete surviving quantum
  state between steps.
- `WindowedMBQELM` resets the quantum resource and receives an explicit classical
  input window. It is a feature-map baseline, not a recurrent reservoir.

The default experiments train only a regularized classical readout. Reservoir
parameters stay fixed after seeded initialization.

CV Tier A contains first moments plus diagonal covariance (8 features for two
modes). Tier B contains first moments plus the complete upper covariance triangle
(14 features for two modes). The exact `GaussianClassicalTwin` is a mandatory
affine state-space baseline for supported fixed Gaussian channels.

## Install

Install the audited PhotoGraphiQ checkout, then this repository:

```sh
python -m pip install -e "../softwares/PhotoGraphiQ[dev]"
python -m pip install -e ".[dev,docs,experiments,graphix,validation]"
```

Core CV use does not import Graphix or MentPy. Their adapters raise an informative
error only when selected.

## Quick start

```python
import numpy as np
from cv_mb_qrc.reservoirs import CVConfig, CVMBReservoir, RidgeReadout

inputs = np.random.default_rng(1729).uniform(-1, 1, 200)
reservoir = CVMBReservoir(CVConfig(seed=7, memory_modes=2, tier="B"))
features = reservoir.run_sequence(inputs, washout=20)

readout = RidgeReadout(1e-4).fit(inputs[20:], features.features, np.roll(inputs, 1)[20:])
prediction = readout.predict(inputs[20:], features.features)
```

Angles are radians. CV conventions are `[q,p]=2i`, interleaved quadratures,
vacuum statistical covariance `I`, and positive resource squeezing means momentum
squeezing.

## Reproduce the paper

```sh
python -m pytest
python -m pytest --cov=src/cv_mb_qrc --cov-report=term-missing --cov-fail-under=90
python -m ruff check .
python -m ruff format --check .
python -m mypy .
python -m build
python experiments/measurement_based_reservoir/main.py \
  --config experiments/measurement_based_reservoir/config_ci.json \
  --output /tmp/cv-mb-qrc-ci
cvmbqrc verify-run /tmp/cv-mb-qrc-ci
python experiments/measurement_based_reservoir/main.py \
  --config experiments/measurement_based_reservoir/config_publication.json --dry-run
```

Historical outputs created under the old `photographiqml` namespace live only in
`experiments/measurement_based_reservoir/results_legacy/`. They do not validate
V3. New CI, development and publication outputs are isolated in `results_ci/`,
`results_development/` and `results_publication/`. Physical tap-homodyne readout
is currently unsupported and fails explicitly; only simulation-only state-oracle
readout is available. No present result establishes quantum advantage,
non-Gaussian superiority or edge-of-chaos behavior.

Read the [design audit](docs/research/measurement_based_quantum_reservoir_design.md),
[tutorial](docs/tutorials/measurement-based-reservoir.md), [experiment protocol](experiments/measurement_based_reservoir/README.md),
[statistical methods](docs/research/statistical_methods.md),
[evidence protocol](docs/research/evidence_provenance.md),
[hardware-readout boundary](docs/research/hardware_readout_limitations.md), and
[full reproducibility guide](docs/research/reproducibility_guide.md). The
publication configuration never starts without `--confirm-publication`.
