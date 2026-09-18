"""Graphix 0.4 collision reservoir; optional imports occur only on selection.

The executable reservoir remains the audited one-memory-qubit collision family.
Topology helpers below support guarded design/visualization but are not presented
as an implemented generic mixed-state MBQC reservoir.
"""

from dataclasses import asdict
from importlib.metadata import version
from time import perf_counter

import numpy as np

from .base import MeasurementBasedReservoir, input_vector
from .config import QubitConfig, integer
from .results import ReservoirResult


def build_guarded_topology(
    memory_qubits=1,
    fresh_qubits=1,
    *,
    kind="chain",
    explicit_edges=None,
    max_live_qubits=8,
):
    """Resolve small design topologies without starting an exponential simulation."""
    integer(memory_qubits, "memory_qubits", 1)
    integer(fresh_qubits, "fresh_qubits", 1)
    integer(max_live_qubits, "max_live_qubits", 2)
    count = memory_qubits + fresh_qubits
    if count > max_live_qubits:
        raise MemoryError("Requested topology exceeds max_live_qubits")
    nodes = tuple(range(count))
    if kind == "explicit":
        if explicit_edges is None:
            raise ValueError("Explicit topology requires explicit_edges")
        edges = {tuple(sorted(map(int, edge))) for edge in explicit_edges}
    elif kind == "chain":
        edges = {(node, node + 1) for node in range(count - 1)}
    elif kind == "ring":
        if count < 3:
            raise ValueError("Ring topology requires at least three qubits")
        edges = {(node, node + 1) for node in range(count - 1)} | {(0, count - 1)}
    elif kind == "star":
        edges = {(0, node) for node in range(1, count)}
    elif kind == "brickwork":
        width = (count + 1) // 2
        rows = (tuple(range(width)), tuple(range(width, count)))
        edges = {
            tuple(sorted((row[index], row[index + 1])))
            for row in rows
            for index in range(len(row) - 1)
        }
        edges |= {(column, width + column) for column in range(0, min(width, count - width), 2)}
    else:
        raise ValueError("Topology must be chain, ring, star, brickwork, or explicit")
    if any(left == right or left not in nodes or right not in nodes for left, right in edges):
        raise ValueError("Topology contains a loop or an out-of-range node")
    return {
        "kind": kind,
        "nodes": list(nodes),
        "edges": [list(edge) for edge in sorted(edges)],
        "memory_nodes": list(range(memory_qubits)),
        "fresh_nodes": list(range(memory_qubits, count)),
        "retained_nodes": list(range(memory_qubits)),
        "measured_nodes": list(range(memory_qubits, count)),
        "max_live_qubits": max_live_qubits,
        "execution_status": "design-only; generic reservoir execution is unsupported",
    }


def require_graphix():
    try:
        import graphix
    except ImportError as exc:
        raise ImportError("Install cv-mb-qrc[graphix] for this backend") from exc
    if version("graphix") != "0.4":
        raise ImportError("This adapter is audited against graphix==0.4")
    return graphix


def density_check(state):
    rho = np.asarray(state, complex)
    if (
        rho.shape != (2, 2)
        or not np.isfinite(rho).all()
        or not np.allclose(rho, rho.conj().T, atol=1e-12, rtol=0)
        or not np.isclose(np.trace(rho), 1, atol=1e-12, rtol=0)
        or np.linalg.eigvalsh(rho).min() < -1e-12
    ):
        raise ValueError("Memory must be a finite physical 2x2 density matrix")
    return rho.copy()


class GraphixMBReservoir(MeasurementBasedReservoir):
    """One retained qubit and one fresh Ry input per step, fully branch-weighted."""

    config: QubitConfig

    def __init__(self, config: QubitConfig | None = None):
        require_graphix()
        self.config = config or QubitConfig()
        rng = np.random.default_rng(self.config.seed)
        self.mask = float(rng.uniform(0.5, 1.5))
        self.bias = float(rng.uniform(-np.pi, np.pi))
        self.reset()

    def reset(self, *, seed=None, initial_state=None):
        integer(self.config.seed if seed is None else seed, "seed", 0)
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.state = density_check([[1, 0], [0, 0]] if initial_state is None else initial_state)
        self.initial_state = self.state.copy()
        self.trajectories = None
        self.time = 0
        self.execution_shots = None
        return self

    def pattern(self, value):
        from graphix import command
        from graphix.clifford import Clifford
        from graphix.fundamentals import Plane
        from graphix.measurements import Measurement
        from graphix.pattern import Pattern
        from graphix.states import PlanarState

        angle = self.bias + self.mask * self.config.input_scale * value
        p = Pattern(input_nodes=[0])
        p.add(command.N(1, PlanarState(Plane.XZ, angle / np.pi)))
        if self.config.entangle:
            p.add(command.E((0, 1)))
        p.add(command.M(1, Measurement.XY(self.config.angle / np.pi)))
        if self.config.feedforward:
            p.add(command.X(0, {1}))
        p.add(command.C(0, Clifford.H))
        return p

    def _branch(self, pattern, state, bit=None):
        from graphix.branch_selector import BranchSelector
        from graphix.sim.density_matrix import DensityMatrixBackend
        from graphix.simulator import PatternSimulator

        class Selector(BranchSelector):
            probability = 1.0
            outcome = 0

            def measure(selector, qubit, f_expectation0, rng=None, *, stacklevel=1):
                p = float(f_expectation0())
                if not -1e-12 <= p <= 1 + 1e-12:
                    raise FloatingPointError("Invalid Graphix branch probability")
                p = float(np.clip(p, 0, 1))
                selector.outcome = int(self.rng.random() >= p) if bit is None else bit
                selector.probability = p if selector.outcome == 0 else 1 - p
                if selector.probability < 1e-15:
                    raise ZeroProbabilityBranch
                return selector.outcome

        selector = Selector()
        backend = DensityMatrixBackend(branch_selector=selector)
        simulator = PatternSimulator(pattern, backend=backend)
        try:
            simulator.run(input_state=np.asarray(state, complex), rng=self.rng)
        except ZeroProbabilityBranch:
            return np.zeros((2, 2), complex), 0.0, selector.outcome
        return density_check(backend.state.rho), selector.probability, selector.outcome

    def feature_names(self):
        return ("X0", "Y0", "Z0")

    @staticmethod
    def _features(rho):
        return np.array([2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real])

    def step(self, input_value, *, shots=None):
        value = float(input_vector(input_value, 1)[0])
        if shots is not None:
            integer(shots, "shots")
            if shots > self.config.max_trajectories:
                raise MemoryError("Trajectory count exceeds max_trajectories; raise explicitly")
        if self.config.evolution == "conditional" and shots not in (None, 1):
            raise ValueError("Conditional trajectory uses shots=None or 1")
        if self.time and shots != self.execution_shots:
            raise ValueError("Reset before changing shots")
        start = perf_counter()
        pattern = self.pattern(value)
        compile_seconds = perf_counter() - start
        outcomes, probabilities = [], []
        if shots is None and self.config.evolution == "unconditional":
            memory = self.state if self.config.temporal_edges else self.initial_state
            branches = [self._branch(pattern, memory, bit) for bit in (0, 1)]
            probabilities = [p for _, p, _ in branches]
            if abs(sum(probabilities) - 1) > 1e-12:
                raise FloatingPointError("Incomplete branch probabilities")
            self.state = density_check(sum(p * state for state, p, _ in branches))
            estimator = "exact-expectations"
        else:
            count = 1 if shots is None else shots
            if self.trajectories is None:
                self.trajectories = [self.state.copy() for _ in range(count)]
            states = []
            for memory in self.trajectories:
                state, p, bit = self._branch(
                    pattern, memory if self.config.temporal_edges else self.initial_state
                )
                states.append(state)
                outcomes.append(bit)
                probabilities.append(p)
            self.trajectories = states
            self.state = density_check(np.mean(states, axis=0))
            estimator = (
                "trajectory-average-of-expectations" if count > 1 else "conditional-expectations"
            )
        self.time += 1
        self.execution_shots = shots
        return ReservoirResult(
            self._features(self.state),
            self.feature_names(),
            self.config.evolution,
            estimator,
            shots,
            outcomes,
            {
                "branch_probabilities": probabilities,
                "compilation_seconds": compile_seconds,
                "minimum_eigenvalue": float(np.linalg.eigvalsh(self.state).min()),
                "trace": float(np.trace(self.state).real),
            },
            {
                "qubits": 2,
                "fresh_nodes": 1,
                "edges": int(self.config.entangle),
                "measurements": 1,
                "retained_nodes": [0],
                "hilbert_dimension": 4,
                "density_bytes": 256,
            },
            asdict(self.config),
            seconds=perf_counter() - start,
        )


class ZeroProbabilityBranch(Exception):
    """Internal control flow: do not normalize a zero-probability branch."""


def graphix_wire(state, angles, bits=None, *, adaptive=False, max_live_qubits=16):
    """Corrected wire; explicit physical corrections make every branch equivalent."""
    require_graphix()
    from graphix import command
    from graphix.branch_selector import FixedBranchSelector
    from graphix.measurements import Measurement
    from graphix.pattern import Pattern
    from graphix.simulator import PatternSimulator

    state = np.asarray(state, complex)
    angles = np.asarray(angles, float)
    if state.shape != (2,) or not np.isclose(np.vdot(state, state), 1):
        raise ValueError("Wire input must be normalized with shape (2,)")
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError("Provide finite wire angles")
    integer(max_live_qubits, "max_live_qubits", 2)
    if adaptive and len(angles) + 1 > max_live_qubits:
        raise MemoryError("Adaptive full-graph reference exceeds max_live_qubits; raise explicitly")
    bits = [0] * len(angles) if bits is None else list(bits)
    if len(bits) != len(angles) or any(bit not in (0, 1) for bit in bits):
        raise ValueError("One binary outcome per measurement is required")
    p = Pattern(input_nodes=[0])
    if adaptive:
        for i in range(len(angles)):
            p.add(command.N(i + 1))
            p.add(command.E((i, i + 1)))
        for i, angle in enumerate(angles):
            p.add(
                command.M(
                    i,
                    Measurement.XY(float(angle) / np.pi),
                    s_domain={i - 1} if i else set(),
                    t_domain={i - 2} if i > 1 else set(),
                )
            )
        p.add(command.X(len(angles), {len(angles) - 1}))
        if len(angles) > 1:
            p.add(command.Z(len(angles), {len(angles) - 2}))
    else:
        for i, angle in enumerate(angles):
            p.add(command.N(i + 1))
            p.add(command.E((i, i + 1)))
            p.add(command.M(i, Measurement.XY(float(angle) / np.pi)))
            p.add(command.X(i + 1, {i}))
    sim = PatternSimulator(p, branch_selector=FixedBranchSelector(dict(enumerate(bits))))
    sim.run(input_state=state, rng=np.random.default_rng(0))
    return np.asarray(sim.backend.state.flatten(), complex)
