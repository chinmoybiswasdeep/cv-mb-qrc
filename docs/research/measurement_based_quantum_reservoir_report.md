# Measurement-based quantum reservoir implementation report

> **Legacy report.** Numerical counts, scores and figures below describe the
> preserved `results_legacy/` bundle and do not validate V3. V3 changed Tier B
> to means plus full covariance, added the affine Gaussian twin and
> permutation-null capacity, and separated CI/development/publication outputs.
> Only state-oracle readout is currently supported; physical probe readout fails
> explicitly. Regenerate a new report only from a verified current manifest.

## Outcome and architecture

The implementation now lives in this standalone `cv-mb-qrc` repository.
PhotoGraphiQ remains the unchanged execution dependency. The tracked changes were
removed from the sibling PhotoGraphiQML checkout, while its pre-existing untracked
`paper_experiments` directory was preserved.

The standalone `cv_mb_qrc.reservoirs` package separates causal orchestration,
configuration, results, readout, chronological handling, diagnostics and datasets
from CV, Graphix, optional MentPy and optional mixed-Fock adapters. CV transfers
the complete Gaussian state through the public PhotoGraphiQ interface. Graphix
transfers the complete memory density matrix and executes probability-weighted
measurement branches. Both are genuine stepwise state transfer. No upstream
interface change or time-unrolling workaround was necessary.

`WindowedMBQELM` resets for every explicit trailing input window. Its memory is
classical history and it is never called a recurrent reservoir. Fresh resource
labels are scoped to each time step; consumed ancillas are not carried forward.

## Exact files and outputs

The complete source/configuration/documentation inventory is
`docs/research/measurement_based_quantum_reservoir_files.txt`. The standalone
repository metadata, README, documentation and package were added here. The package contains
`__init__.py`, `base.py`, `config.py`, `results.py`, `cv.py`,
`graphix_backend.py`, `mentpy_backend.py`, `fock.py`, `readout.py`, `temporal.py`,
`diagnostics.py`, `benchmarks.py` and `validation.py`.

Four test files, the design document, this report, a tutorial and notebook 06 were
added. Experiment sources are `main.py`, `reproduce.py`, `supplementary.py`, two
JSON configurations and a README. The raw-run inventory is
`experiments/measurement_based_reservoir/results_legacy/raw/manifest.json` (420 entries).
Other raw records preserve datasets, split indices, environment/source hashes,
dynamics, noise/shot controls, washout, scaling, MentPy checks, Fock convergence
and execution incidents. Generated CSV/JSON/Markdown summaries and 17 figures in
SVG/PDF/PNG are under the same results directory. Build/site products are ignored.

## Equations actually implemented

Conditional evolution is rho_t=E_(u,m)(rho_(t-1))/Tr[E_(u,m)(rho_(t-1))]. Exact
outcome-averaged evolution sums the unnormalized branches. Gaussian channels use
mu'=S mu+d and V'=S V S^T+N from PhotoGraphiQ, with [q,p]=2i, interleaved
quadratures, vacuum statistical covariance I and positive momentum resource
squeezing. The Piquasso covariance boundary uses sigma=2V at hbar=2.

For the one-qubit collision, with a=cos(theta/2), b=sin(theta/2),
K_m=H X^m [aI+(-1)^m exp(-i phi)bZ]/sqrt(2). Its completeness and Graphix output
are tested against independent NumPy expressions. Graphix angles use pi units
only inside the adapter; the public API uses radians.

Features are exact conditional/unconditional expectations or explicitly labelled
trajectory estimators. Gaussian covariance features are centered moments rather
than fixed linear observables. Photon number is
(Vqq+Vpp+mu_q^2+mu_p^2-2)/4. Tier B remains efficiently Gaussian-simulable.
The only learned component is the regularized affine readout W[1,u,x], with
training-only standardization and an unpenalized intercept. An SVD ridge solve
avoids squaring the design condition number.

Memory targets use normalized Legendre degree-one, degree-two self and cross-delay
terms. NARMA10, parity and Mackey?Glass equations/parameters are specified in the
experiment README. Every split resets state and owns its targets and input history.

## Backend capability boundaries

The design document has the audited capability matrix and source provenance.
CV exact execution supports fixed-angle affine Gaussian patterns. Trajectories
support Piquasso gates, homodyne noise/inefficiency, loss/thermal noise and causal
within-step adaptive angles on multiple input channels. The default architecture
uses fixed angles. Graphix supports a small one-memory collision graph and
corrected/adaptive wire references; broader topologies and noisy qubit reservoir
adapters are not provided. There is no silent physical-to-ideal fallback.

MentPy 0.1.0a15, commit 63c3d83e495696b4491c9d376dab7e3e6cf6c863, validates the
corrected pure-wire common subset. Its audited NumPy statevector simulator forces
the zero branch. It does not establish equivalence of general mixed stateful
collision channels. Graphix/MentPy never validate numerical CV physics.

The optional Fock adapter uses a small cat-resource collision, PNR, loss and full
mixed-state transfer. It is conditional, with guards and branch likelihoods.
Arbitrary GKP XY, general non-Gaussian task studies and reservoir autodiff are
not claimed. The existing restricted physical MuTA boundary is preserved.

## Verification

- PhotoGraphiQ baseline: 533 passed, 104 pre-existing warnings, 475.60 seconds.
- Pre-relocation PhotoGraphiQML baseline: 127 passed, 59 pre-existing warnings, 77.98 seconds.
- Pre-relocation combined suite: 159 passed, 60 warnings, 151.01 seconds.
- Total measured line coverage: 96.20%, above the existing 90% policy.
- Ruff and format checks pass across src/tests/scripts/experiments/examples.
- Mypy passes over 29 source files; wheel and sdist build successfully.
- Strict MkDocs build passes; 44 tutorial scripts and all six notebooks execute.
- Optional imports are tested in a clean process and with dependencies blocked.
- Analytical Gaussian, raw Piquasso, Kraus completeness, all corrected two-step
  wire branches, adaptive-angle equivalence, zero-probability branches,
  serialization, leakage, reset and stochastic consistency tests pass.
- All 420 repeated deterministic score records agree exactly. Runtime and memory
  are excluded from bitwise reproducibility. Source SHA256 hashes match the
  completed benchmark environment record.

Gaussian/qubit algebra comparisons use absolute 1e-12 tolerances (Piquasso
trajectory comparison 1e-11); stochastic tests use six estimated standard errors
across independent repetitions, plus a roundoff floor. Impossible qubit branches
below 1e-15 are omitted at double precision, with branch completeness checked to
1e-12. Fock positivity tolerance is 1e-12; selected-observable convergence was
predeclared at 1e-5. Tolerances were not relaxed to pass tests.

## Small benchmark findings

Ten reservoir seeds share a fixed dataset seed. Chronological partitions use 360
samples, a five-step embargo and independent 20-step washout. There are only
47 iid / 46 prediction test samples, so these are smoke results. Intervals below
are 95% percentile bootstrap intervals over reservoir seeds, conditional on the
fixed datasets. Values are mean ? sample SD [interval]. No significance tests.

| Model | Linear capacity | Nonlinear capacity | NARMA10 test R? |
|---|---|---|---|
| cv_A | 1.2356 ? 0.1795 [1.1150, 1.3122] | 0.1095 ? 0.0820 [0.0615, 0.1567] | 0.4566 ? 0.0301 [0.4388, 0.4748] |
| cv_B | 1.2001 ? 0.2265 [1.0469, 1.2955] | 0.1380 ? 0.0716 [0.0976, 0.1795] | 0.4477 ? 0.0547 [0.4182, 0.4829] |
| graphix | 0.6839 ? 0.3268 [0.4861, 0.8646] | 0.3838 ? 0.3207 [0.2099, 0.5813] | 0.3184 ? 0.0396 [0.2938, 0.3405] |
| cv_window | 3.0006 ? 0.0011 [3.0001, 3.0013] | 2.2605 ? 0.0368 [2.2370, 2.2807] | 0.3123 ? 0.0087 [0.3078, 0.3180] |
| delay | 3.0078 ? 0.0000 [3.0078, 3.0078] | 0.0543 ? 0.0000 [0.0543, 0.0543] | 0.3362 ? 0.0000 [0.3362, 0.3362] |
| esn_14 | 2.1823 ? 0.3183 [2.0158, 2.3832] | 1.9982 ? 0.5808 [1.6608, 2.3316] | 0.3424 ? 0.0111 [0.3363, 0.3490] |
| rff_14 | 2.8490 ? 0.1518 [2.7552, 2.9259] | 3.7903 ? 0.3823 [3.5257, 3.9857] | 0.2939 ? 0.0490 [0.2662, 0.3227] |

Capacity totals clip each test R? at zero; raw negative values are preserved.
Clipping has finite-sample positive bias. Apparent nonlinear capacity of Tier A
is not evidence of nonlinear Gaussian dynamics. Deterministic classical baselines
have repeated identical seed values and zero conditional bootstrap width; this
is not uncertainty over datasets. The raw results and paired differences remain
available; no seed or topology was selected using the test set.

RFF and ESN controls are present at both 3 and 14 features. RFF and linear delay
controls receive the same explicit four-input history as the windowed model.
CV A/B have 8/14 features and Graphix has 3. Native physical resources differ and
are not declared equivalent. Windowed performance includes explicit classical
history and cannot establish quantum internal memory.

CV Tier B has lower aggregate memory/nonlinear capacity than the 14-feature RFF
control here. Its NARMA score is higher at these fixed operating points, but this
small single-dataset comparison is insufficient for a general superiority claim.
Classical delayed-input/RFF predictions are also strong on the smooth
Mackey?Glass dataset. No quantum advantage, superior general memory, tradeoff
violation or edge-of-chaos conclusion is supported.

## Convergence and runtime

The selected two-step zero-PNR branch has joint success probability 0.992657732214 and failure probability 0.007342267786. The largest selected feature change from cutoff 12 to 16 is 5.29e-19. Cutoffs are exclusive **total photon number**, not per-mode tensor cutoffs. Trace/Hermiticity/positivity, retained norm and boundary mass are saved at every cutoff. This is a near-vacuum, one-seed numerical control, not whole-state convergence or non-Gaussian task evidence. All unpostselected PNR outcomes remain available in ordinary Fock execution.

- cv_B: 0.495 ? 0.120 seconds per iid run, including three split executions and ridge selection.
- graphix: 2.183 ? 3.240 seconds per iid run, including three split executions and ridge selection.
- cv_window: 2.138 ? 0.497 seconds per iid run, including three split executions and ridge selection.
- esn_14: 0.031 ? 0.006 seconds per iid run, including three split executions and ridge selection.
- rff_14: 0.027 ? 0.005 seconds per iid run, including three split executions and ridge selection.

Sampled peak process RSS was 365.1 MiB, including imports; sampling can miss transient peaks. CV defaults retain two memory modes plus one fresh mode. Graphix uses one retained and one fresh qubit, a four-dimensional live Hilbert space. Separate construction/execution and 1/2/4-memory-mode scaling records are saved. Modes and qubits are never equated as Hilbert resources.

## Discovered issues and remaining limitations

- The pre-existing reservoir-features demonstration is static, not recurrent.
- The raw Piquasso sampler covariance caveat already documented upstream means it
  is not used as an oracle for physical trajectory distributions; gate moments
  and independent Gaussian conditioning are checked separately.
- Graphix density inputs require complex dtype at the public boundary.
- MentPy's forced-zero NumPy branch limits the independent common subset.
- A Windows atomic-replace sharing error was fixed with unique temporary files
  and bounded retries. A slow allocation-traced run was restarted without tracing.
  A dotted-subpackage coverage-discovery NumPy import failure was avoided by
  filesystem coverage selection. These incidents are preserved in raw JSON.
- Shots count persistent simulated measurement trajectories. Exact conditional
  expectations are used inside each trajectory. This is not an experimental
  detector-shot budget for all features, and simulated nondestructive feature
  extraction does not include tomography/replica costs. Ensemble covariance is
  a finite-sample plug-in estimate including between-trajectory fluctuations.
- Only the declared small architectures and noise mechanisms are implemented.
  Conditional Fock convergence has no multi-seed temporal task comparison.
- Echo-state diagnostics are empirical finite-sequence curves; no global
  contraction theorem, chaos boundary or experimental feasibility claim is made.
- Full publication configuration and other OS/Python matrices were not executed.
  The observed operating points should not be extrapolated to publication claims.

## Reproduction commands

From the `cv-mb-qrc` root, use an activated environment interpreter:

```sh
python -m pip install -e "../PhotoGraphiQ[dev]"
python -m pip install -e ".[dev,docs,experiments,graphix]"
python -m pip install -r tests/mentpy_reference/requirements.txt
python -m pytest --cov=src/cv_mb_qrc --cov-fail-under=90
python -m ruff check src tests experiments
python -m ruff format --check src tests experiments
python -m mypy
python -m build --outdir dist
python -m mkdocs build --strict --site-dir site
python experiments/measurement_based_reservoir/main.py
python experiments/measurement_based_reservoir/reproduce.py
```

For the separate Fock numerical record:

```python
from cv_mb_qrc.reservoirs.fock import cutoff_study
from cv_mb_qrc.reservoirs.results import atomic_json
atomic_json("experiments/measurement_based_reservoir/results_legacy/raw/fock.json",
            cutoff_study([.02, .04], cutoffs=(8, 12, 16)))
```

Then rerun `reproduce.py` to regenerate its optional convergence figure. The
publication-scale configuration is selected explicitly with `--config
experiments/measurement_based_reservoir/config_full.json`. Environment JSON records
exact versions, Git SHAs, dirty worktree flags and source hashes. PhotoGraphiQ
revision is 6d49da06fa6ede78bf06ea1df07ccfb3b4f5250e; the `cv-mb-qrc` base revision is
34f90ec9913c7182495459fd35b3f556a181ed74 plus these local additions.
