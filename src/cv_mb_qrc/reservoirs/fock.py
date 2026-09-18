"""Small conditional non-Gaussian collision experiment with explicit cutoff controls.

This optional Tier C adapter uses upstream mixed Fock physics throughout. It is
separate from the Gaussian default and does not claim a converged task advantage.
"""

from dataclasses import asdict, dataclass, replace
from math import comb
from time import perf_counter

import numpy as np
import photographiq as pg
import psutil

from .base import MeasurementBasedReservoir, input_vector
from .config import BackendCapabilityError, integer
from .results import ReservoirResult


@dataclass(frozen=True)
class FockConfig:
    seed: int = 0
    input_channels: int = 1
    cutoff: int = 12
    coupling: float = 0.08
    input_scale: float = 0.2
    cat_amplitude: float = 0.25
    transmissivity: float = 0.9
    max_matrix_bytes: int = 64_000_000

    def __post_init__(self):
        integer(self.seed, "seed", 0)
        integer(self.input_channels, "input_channels")
        integer(self.cutoff, "cutoff", 2)
        integer(self.max_matrix_bytes, "max_matrix_bytes", 16)
        if self.input_channels != 1:
            raise BackendCapabilityError("Fock collision adapter supports one input channel")
        if not np.isfinite(
            [self.coupling, self.input_scale, self.cat_amplitude, self.transmissivity]
        ).all():
            raise ValueError("Finite Fock parameters required")
        if self.cat_amplitude <= 0 or not 0 <= self.transmissivity <= 1:
            raise ValueError("Positive cat amplitude and valid loss required")
        if 16 * comb(self.cutoff + 1, 2) ** 2 > self.max_matrix_bytes:
            raise MemoryError("Fock density matrix exceeds configured byte budget")


class FockMBReservoir(MeasurementBasedReservoir):
    config: FockConfig

    def __init__(self, config=None):
        self.config = config or FockConfig()
        self.reset()

    def reset(self, *, seed=None):
        integer(self.config.seed if seed is None else seed, "seed", 0)
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.state = pg.FockDensityMatrix([[1]], ((0,),))
        self.time = 0
        return self

    def feature_names(self):
        return ("number", "parity", "number_squared", "vacuum_probability")

    def step(self, input_value, *, shots=None, postselect=None):
        if shots not in (None, 1):
            raise BackendCapabilityError(
                "Fock adapter runs one conditional trajectory per instance"
            )
        u = float(input_vector(input_value, 1)[0])
        c = self.config
        from photographiq.backends.mixed_fock import MixedFockBackend

        start = perf_counter()
        p = pg.Pattern(inputs=(0,)).extend(
            [
                pg.Prepare(1, state=pg.FockInput.cat(c.cat_amplitude, c.cutoff)),
                pg.Displace(1, q=2 * c.input_scale * u),
                pg.Rotate(0, 0.4),
                pg.Entangle(0, 1, c.coupling),
                pg.Measure(1, pg.PhotonNumber(), "count"),
                pg.Loss(0, c.transmissivity),
                pg.Output((0,)),
            ]
        )
        engine = MixedFockBackend(c.cutoff, max_matrix_bytes=c.max_matrix_bytes)
        result = pg.simulate(
            p,
            initial_state=self.state,
            backend=engine,
            seed=int(self.rng.integers(0, 2**63)),
            measurement_outcomes=None if postselect is None else {"count": postselect},
        )
        state = result.state
        rho = state.density_matrix
        self.state = pg.FockDensityMatrix(rho, state.basis)
        self.time += 1
        features = np.array(
            [
                state.photon_number(0),
                state.parity(),
                state.photon_moment(0, 2),
                state.probabilities.get((0,), 0),
            ]
        )
        likelihood = result.measurement_statistics["count"]
        return ReservoirResult(
            features,
            self.feature_names(),
            "conditional",
            "conditional-expectations",
            shots,
            [result.outcomes],
            {
                "tier": "C",
                "cutoff": c.cutoff,
                "cutoff_semantics": "exclusive total photon number",
                "converged": False,
                "herald_probability": float(likelihood["value"]),
                "postselected": postselect is not None,
                "trace": float(np.trace(rho).real),
                "hermiticity_error": float(np.max(abs(rho - rho.conj().T))),
                "minimum_eigenvalue": float(np.linalg.eigvalsh(rho).min()),
                "boundary_population": max(
                    (d.get("boundary_population", 0) for d in state.diagnostics), default=0
                ),
                "minimum_retained_norm": min(state.retained_norms, default=1),
            },
            {
                "modes": 2,
                "fresh_nodes": 1,
                "edges": 1,
                "measurements": 1,
                "retained_nodes": [0],
                "cutoff": c.cutoff,
                "hilbert_dimension": comb(c.cutoff + 1, 2),
                "non_gaussian_resource": "even cat ancilla",
            },
            asdict(c),
            seconds=perf_counter() - start,
        )


def cutoff_study(inputs, *, cutoffs=(8, 12, 16), seed=0, tolerance=1e-5):
    """Predeclared identical PNR branches; preserve joint probability and complement.

    All-zero PNR postselection is a numerical convergence experiment, not a
    deterministic reservoir benchmark. Its success probability is reported.
    """
    if len(cutoffs) < 2 or any(b <= a for a, b in zip(cutoffs, cutoffs[1:])):
        raise ValueError("At least two increasing cutoffs required")
    inputs = np.asarray(inputs, float)
    if inputs.ndim != 1 or not len(inputs) or not np.isfinite(inputs).all():
        raise ValueError("Nonempty finite input sequence required")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Positive finite convergence tolerance required")
    rows, previous = [], None
    for cutoff in cutoffs:
        rss_before = psutil.Process().memory_info().rss
        model = FockMBReservoir(replace(FockConfig(), cutoff=cutoff, seed=seed))
        results = [model.step(u, postselect=0) for u in inputs]
        features = np.array([r.features for r in results])
        probability = float(np.prod([r.diagnostics["herald_probability"] for r in results]))
        difference = None if previous is None else float(np.max(abs(features - previous)))
        rows.append(
            {
                "cutoff": cutoff,
                "features": features.tolist(),
                "joint_success_probability": probability,
                "failure_probability": 1 - probability,
                "effective_trajectories_per_accepted_sequence": (
                    float(1 / probability) if probability > 0 else None
                ),
                "max_feature_change": difference,
                "diagnostics": [r.diagnostics for r in results],
                "seconds": sum(r.seconds for r in results),
                "sampled_process_rss_increase_bytes": max(
                    0, psutil.Process().memory_info().rss - rss_before
                ),
                "maximum_boundary_population": max(
                    diagnostic["boundary_population"]
                    for result in results
                    for diagnostic in [result.diagnostics]
                ),
            }
        )
        previous = features
    return {
        "rows": rows,
        "tolerance": tolerance,
        "observable_convergence": rows[-1]["max_feature_change"] < tolerance,
        "truncation_warning": any(row["maximum_boundary_population"] > tolerance for row in rows),
        "scope": (
            "numerical feasibility study of a fixed zero-PNR branch and selected observables; "
            "not full-state convergence or task-level advantage"
        ),
    }
