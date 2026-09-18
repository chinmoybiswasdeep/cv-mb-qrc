# Measurement-based quantum reservoir design and audit

> **V3 scope note.** The committed numerical artifacts referenced below are
> historical and now live in `results_legacy/`. Current Tier B is first moments
> plus the full upper covariance triangle. `GaussianClassicalTwin` is the
> mandatory equivalence baseline. Physical tap-homodyne readout remains an
> explicit unsupported capability, Graphix remains a one-qubit collision
> reference, and MentPy validates only the corrected pure-wire subset.

Audit date: 2026-09-17. Implementation belongs to `cv-mb-qrc`; PhotoGraphiQ
remains the execution/physics layer. No simulator is duplicated in the runtime.

## Audited sources and provenance

PhotoGraphiQ manuscript-experiments: 6d49da06fa6ede78bf06ea1df07ccfb3b4f5250e.
cv-mb-qrc base revision: 34f90ec9913c7182495459fd35b3f556a181ed74.
MentPy: 63c3d83e495696b4491c9d376dab7e3e6cf6c863 (0.1.0a15).
Graphix: 0.4, Piquasso: 8.0.1. Release wheels have no local Git SHA;
do not invent a source SHA. Full installed versions accompany experiment outputs.

Read both READMEs/pyprojects; upstream simulator, state, pattern, graph, flow,
measurement, Gaussian/channel, pure/mixed Fock, Piquasso, non-Gaussian,
convergence, serialization and autodiff implementations; reservoir-features demo;
downstream contract; downstream MuTA, logical, physical/GKP/lowering, training,
models, validation, diagnostics and kernels, independent MentPy tests and source;
paper experiment runners/style and CI policy. No AGENTS.md was found under cvqc.
The pre-existing untracked `PhotoGraphiQML/paper_experiments` directory is preserved.

Existing demo encodes independent samples: it is not a temporal reservoir.
Existing physical MuTA supports signed X only, never arbitrary XY-to-homodyne.
Graphix 0.4 uses Measurement.XY(angle/pi), not remembered older command APIs.
MentPy numpy-sv forces zero outcomes and rejects force0=False. Consequently its
reference role is deterministic corrected wires, not arbitrary sampled channels.

## Capability matrix (upstream versus selected adapter)

| Property | PhotoGraphiQ | Piquasso 8.0.1 | Graphix 0.4 | MentPy reference |
|---|---|---|---|---|
| States | Gaussian moments, pure/mixed total-cutoff Fock | Gaussian, pure/mixed Fock | statevector, density matrix, tensor network | numpy statevector common subset |
| Measurements | homodyne, general/heterodyne; PNR/instruments in Fock | Gaussian, particle detection; adapter needed for conventions | Pauli and planar qubit | XY, X, Y in numpy-sv |
| Exact | affine unconditional Gaussian channel | gates/state moments | branch expectations, weighted enumeration | normalized zero branch |
| Sampled | conditional trajectories | native sampling has documented covariance caveat | probability-based branch selector | numpy-sv sampled outcomes unsupported |
| Persistent state | public simulate(initial_state=...), ordered Gaussian/Fock inputs | public initial_state | public simulator.run(input_state=...) | pure input state only in chosen reference |
| Noise | loss/thermal, finite squeezing, noisy homodyne; mixed Fock restrictions | supported optical channels | noise-model/density interfaces | not used in reference |
| Differentiability | optional JAX fixed supported paths only | connector-dependent | no reservoir autodiff claim | no reservoir autodiff claim |
| Scaling | Gaussian polynomial; Fock D=binom(K+m-1,m) | Gaussian polynomial; Fock vector D/matrix D^2 | vector 2^n, density 4^n | exponential in simulation window |
| Exclusions | Gaussian PNR conditioning; arbitrary physical GKP XY | raw sampler not an independent physical-distribution oracle | all numerical CV validation | general stateful mixed channels and stochastic equivalence |

Pre-relocation PhotoGraphiQML baseline: 127 passed (59 pre-existing warnings), 77.98 seconds.
PhotoGraphiQ full baseline: 533 passed (104 pre-existing warnings), 475.60 seconds.

## Chosen architecture

Both families use genuine stepwise surviving-state transfer. A retained mode/qubit
is an input to the next step; its complete Gaussian state/density matrix is carried,
not an inferred state reconstructed from selected features. Fresh nodes receive
only current input. Destructive measurements consume only fresh ancillas.

CV: M persistent modes, seeded fixed weighted CZ connections, fresh displaced
momentum-squeezed ancillas, homodyne, causal displacement feed-forward, rotation
and attenuation of memory. Public PhotoGraphiQ Pattern and gaussian_channel give
exact unconditional evolution for fixed-angle affine control; simulate gives
conditional Piquasso/Gaussian trajectories. Quantum Gaussian means/covariances
completely specify the state, so their transfer is exact within Gaussian physics.
All variables use [q,p]=2i, interleaved quadratures, statistical vacuum covariance I.
Piquasso stored covariance is twice this statistical covariance at hbar=2.

Qubit: a small collision graph retains memory and injects a fresh Ry-encoded
ancilla, CZ, XY ancilla measurement, outcome-controlled X and memory H. Branches
are executed by Graphix and weighted by their actual probabilities. A separate
corrected wire provides the valid Graphix/MentPy common subset. These models are
scientifically distinct from CV physics.

For Kraus branches E(u,m), conditional rho'=E(rho)/Tr(E(rho)); unconditional
rho'=sum_m E(rho). Never equally average postselected branches. CV exact updates
are mu'=S mu+d and V'=S V S^T+N from upstream channel analysis. Features are exact
Tr(O rho) or explicitly labeled estimators. Gaussian photon occupation is
(Vqq+Vpp+mu_q^2+mu_p^2-2)/4: Tier B, not non-Gaussian dynamics.

WindowedMBQELM resets to a fresh state for every explicit trailing input window.
It is not recurrent. Same history must be offered to classical delay baselines.
Only the classical ridge head is trained: yhat=W[1,u,x]. All preprocessing is
fitted on train only; chronological splits reset independently with washout and
an embargo. Validation selects ridge regularization; test labels never select it.

## Scientific boundaries and validation plan

Analytical channel/Kraus references, raw Piquasso gate moments, sampled-versus-exact
uncertainty and Graphix/MentPy wires check distinct claims. State validity, causal
ordering and leakage are tested. Contraction uses common inputs and feature/full
qubit state distances; a classical spectral radius is not quantum echo-state proof.
Unitary or weakly dissipative configurations need not forget their initial state.
Gaussian nonlinear observables do not imply a quantum computational advantage.

Optional Fock work must use increasing predeclared total-photon cutoffs, preserve
herald/failure probabilities, report trace/Hermiticity/positivity/boundary mass,
and compare the same branch when conditional. Homodyne likelihood is a density.
No non-Gaussian benchmark can be claimed converged from a single cutoff. No
edge-of-chaos or tradeoff violation claim follows from a complex-looking trace.
Resource counts remain native modes/qubits plus simulation cost, never equated.

## Implemented scope after validation

The Gaussian and one-qubit collision adapters implement the selected architecture.
An optional one-mode mixed-Fock/cat/PNR adapter also carries its full density matrix;
its small cutoff study is a separate fixed-branch numerical control. General
non-Gaussian task benchmarking, arbitrary GKP resources and noisy qubit reservoir
channels are not provided. The Graphix wire reference includes both physical
correction and equivalent causal adaptive-angle forms. Full capability and
verification findings are in `measurement_based_quantum_reservoir_report.md`.
