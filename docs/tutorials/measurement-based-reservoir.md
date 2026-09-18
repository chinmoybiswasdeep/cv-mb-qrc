# Stateful measurement-based reservoirs

The public API lives in `cv_mb_qrc.reservoirs`. PhotoGraphiQ owns optical
simulation and this repository owns temporal orchestration, input encoding,
features and ridge training. Graphix and MentPy imports are optional.

```python
import numpy as np
from cv_mb_qrc.reservoirs import CVConfig, CVMBReservoir, RidgeReadout

config = CVConfig(seed=7, memory_modes=2, squeezing=.3, coupling=.2,
                  transmissivity=.8, evolution="unconditional", tier="B")
u = np.random.default_rng(1729).uniform(-1, 1, 180)
train_u, test_u = u[:120], u[125:]
train = CVMBReservoir(config).run_sequence(train_u, washout=20)
test = CVMBReservoir(config).run_sequence(test_u, washout=20)
# Example target: one-step delayed nonlinear signal; each split owns its history.
train_y = train_u[19:-1]**2
test_y = test_u[19:-1]**2
readout = RidgeReadout(1e-4).fit(train_u[20:], train.features, train_y)
prediction = readout.predict(test_u[20:], test.features)
assert prediction.shape == test_y.shape
```

The reservoir parameters and masks remain fixed. Only the readout and train-only
standardization are fitted. Production experiments select regularization on a
third chronological validation partition. `run_sequence` continues current state;
call `reset()` or create a new model for independent splits. `reset(seed=...)`
changes trajectory randomness, not fixed parameters; change `config.seed` to
initialize a different reservoir. Invalid shapes, unavailable backends and
nonfinite inputs raise errors.

At each time t the memory is the previous surviving Gaussian state. Fresh modes
receive a seeded affine mask of the current input, coherent displacement and
optional phase/squeezing encoding. Fixed CZ gates couple fresh and memory modes.
Fresh modes are destructively homodyned; causal outcome displacements act on
memory, followed by physical attenuation. The next step receives the full memory
mean/covariance, which exactly specifies a Gaussian state. Measured nodes are not
retained. Gaussian conventions are `[q,p]=2i`, vacuum covariance I and interleaved
quadratures. Positive resource squeezing squeezes momentum.

For the exact affine Gaussian channel, `mu_t=S mu_(t-1)+d(u_t)` and
`V_t=S V_(t-1) S.T+N(u_t)`. Conditional trajectories instead use the normalized
measurement instrument. An outcome average weights branches by probability; it
does not average normalized postselected states equally. Adaptive homodyne angles
are available within multi-input conditional steps and rejected by the exact
unconditional configuration.

Tier A extracts quadrature means and diagonal covariance. Tier B extracts those
means and the complete upper-triangular covariance, including cross-mode entries.
For two modes their dimensions are 8 and 14. Photon number and raw quadrature
squares are not task-facing Tier B features. Both tiers are exactly
Gaussian-simulable and supply no evidence of non-Gaussian resources.

```python
from cv_mb_qrc.reservoirs import GaussianClassicalTwin
twin = GaussianClassicalTwin(config)
np.testing.assert_allclose(
    twin.run_sequence(u).features,
    CVMBReservoir(config).run_sequence(u).features,
    atol=3e-12,
    rtol=0,
)
```

The default readout is `state_oracle`, an exact simulation-only view of internal
moments. `physical_probe` raises `BackendCapabilityError`: PhotoGraphiQ does not
yet expose the audited tap-beamsplitter plus selected-quadrature homodyne
instrument needed to model probe outcomes and surviving memory without jointly
sampling incompatible quadratures.

```python
from cv_mb_qrc.reservoirs import GraphixMBReservoir, QubitConfig
qubit = GraphixMBReservoir(QubitConfig(seed=7))
result = qubit.run_sequence([.1, .2, -.1])
print(result.features)  # retained memory X/Y/Z expectations
```

The initial Graphix architecture has one retained qubit and one fresh Ry input.
After CZ, the ancilla is measured in XY, outcome one controls X on memory, and
a fixed H rotates memory. The exact path sums the two unnormalized branches.
This is a bounded one-memory-qubit collision reference, not a generic configurable
MBQC reservoir. It is unrelated numerically to CV state simulation.
The density matrix is checked for trace, Hermiticity and positivity each step.

For ancilla amplitudes a=cos(theta/2), b=sin(theta/2), measurement angle phi,
`K_m=H X^m [a I+(-1)^m exp(-i phi)b Z]/sqrt(2)` and
`rho_t=sum_m K_m rho_(t-1) K_m†`. Completeness is tested independently against
Graphix. Angles are radians publicly and converted to units of pi at its boundary.

`WindowedMBQELM(lambda: CVMBReservoir(config), window=4)` constructs a fresh state
for every trailing four-input window. Its memory is explicit classical history,
so it must be compared with classical baselines receiving at least four inputs.
It must never be described as quantum recurrence.

Fading-memory diagnostics compare several initial states under identical inputs
or perturb one input and track subsequent feature differences. These are empirical
finite-sequence diagnostics, not a proof of an echo-state property or edge of
chaos. No advantage or memory–nonlinearity tradeoff violation is asserted.

For reproduction, install the `experiments,graphix,validation` extras and run
`python experiments/measurement_based_reservoir/main.py`. Its README specifies
equations, fairness, uncertainty, raw output schema and convergence limitations.
