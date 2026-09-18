"""Persistent Gaussian MBQC using public PhotoGraphiQ patterns and state injection."""

from dataclasses import asdict
from time import perf_counter

import numpy as np
import photographiq as pg

from .base import MeasurementBasedReservoir, input_vector
from .config import CVConfig, integer
from .gaussian_readout import StateOracleReadout, TapHomodyneReadout
from .results import ReservoirResult


class CVMBReservoir(MeasurementBasedReservoir):
    config: CVConfig

    def __init__(self, config: CVConfig | None = None):
        self.config = config or CVConfig()
        c = self.config
        self.nodes = tuple(range(c.memory_modes))
        self.readout = (
            StateOracleReadout(c, self.nodes)
            if c.readout_mode == "state_oracle"
            else TapHomodyneReadout(c, self.nodes)
        )
        rng = np.random.default_rng(c.seed)
        self.mask = rng.uniform(-1, 1, (c.input_channels, c.input_channels))
        self.bias = rng.uniform(-0.2, 0.2, c.input_channels)
        self.weights = c.coupling * rng.uniform(-1, 1, (c.memory_modes, c.input_channels))
        start = perf_counter()
        self.pattern = self._pattern()
        self.channel = None
        if c.evolution == "unconditional":
            # Fresh encoding nodes are explicit channel inputs. Their joint product
            # with memory is prepared below, preserving all memory correlations.
            self.channel = pg.gaussian_channel(self.pattern)
        self.compilation_seconds = perf_counter() - start
        self.reset()

    def _pattern(self):
        c = self.config
        fresh = tuple(range(c.memory_modes, c.memory_modes + c.input_channels))
        p = pg.Pattern(inputs=self.nodes + fresh)
        for i in self.nodes:
            p.append(pg.Rotate(i, c.rotation * (1 + i / max(1, c.memory_modes))))
        for i in self.nodes:
            for j, node in enumerate(fresh):
                if self.weights[i, j] != 0:
                    p.append(pg.Entangle(i, node, float(self.weights[i, j])))
        for i in range(c.memory_modes - 1):
            if c.coupling:
                p.append(pg.Entangle(i, i + 1, c.coupling / 4))
        for j, node in enumerate(fresh):
            angle = c.angle
            if j and c.adaptive_angle:
                angle = angle + c.adaptive_angle * pg.Outcome(fresh[j - 1])
            p.measure(node, pg.Homodyne(angle, c.efficiency, c.measurement_noise))
            for i in self.nodes:
                p.displace(i, q=c.feedforward * float(self.weights[i, j]) * pg.Outcome(node))
        for i in self.nodes:
            p.append(pg.Loss(i, c.transmissivity, c.thermal_photons))
        return p.append(pg.Output(self.nodes)).validate()

    def reset(self, *, seed=None, initial_state=None):
        integer(self.config.seed if seed is None else seed, "seed", 0)
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.state = (
            pg.GaussianState(np.zeros(2 * len(self.nodes)), np.eye(2 * len(self.nodes)), self.nodes)
            if initial_state is None
            else initial_state.copy().validate()
        )
        if self.state.nodes != self.nodes:
            raise ValueError("Initial memory node order differs")
        self.initial_state = self.state.copy()
        self.trajectories = None
        self.time = 0
        self.execution_shots = None
        return self

    def _joint(self, memory, value):
        from scipy.linalg import block_diag

        c = self.config
        encoded = c.input_scale * (self.mask @ value) + self.bias
        means, covariances = [memory.mean], [memory.covariance]
        for x in encoded:
            resource = pg.GaussianInput.squeezed(
                -(c.squeezing + c.squeeze_scale * x), c.phase_scale * x
            ).state(0)
            means.append(2 * x * np.array([np.cos(c.phase_scale * x), np.sin(c.phase_scale * x)]))
            covariances.append(resource.covariance)
        return pg.GaussianState(
            np.concatenate(means), block_diag(*covariances), self.pattern.inputs
        )

    def _features(self, state):
        """Canonical deterministic feature definitions.

        Tier A is means plus diagonal covariance; Tier B is means plus the
        complete upper covariance triangle.  Diagnostic choices are never task
        presets. Photon number is excluded from quadratic diagnostics because
        it is exactly dependent on q² and p² under this convention.
        """
        return self.readout.measure(state)

    def feature_names(self):
        return self.readout.feature_names()

    def step(self, input_value, *, shots=None):
        value = input_vector(input_value, self.config.input_channels)
        if shots is not None:
            integer(shots, "shots")
            if shots > self.config.max_trajectories:
                raise MemoryError("Trajectory count exceeds max_trajectories; raise explicitly")
        if self.config.evolution == "conditional" and shots not in (None, 1):
            raise ValueError("A conditional trajectory requires shots=None or 1")
        if self.time and self.execution_shots != shots:
            raise ValueError("Reset before changing exact/trajectory execution")
        start = perf_counter()
        c = self.config
        outcomes = []
        if shots is None and c.evolution == "unconditional":
            assert self.channel is not None
            memory = self.state if c.temporal_edges else self.initial_state
            self.state = self.channel.apply(self._joint(memory, value))
            features = self._features(self.state)
            estimator = "exact-expectations"
            standard_error = None
        else:
            count = 1 if shots is None else shots
            if self.trajectories is None:
                self.trajectories = [self.state.copy() for _ in range(count)]
            if len(self.trajectories) != count:
                raise ValueError("Reset before changing trajectory count")
            states, rows = [], []
            for memory in self.trajectories:
                joint = self._joint(memory if c.temporal_edges else self.initial_state, value)
                result = pg.simulate(
                    self.pattern,
                    initial_state=joint,
                    backend=c.backend,
                    seed=int(self.rng.integers(0, 2**63)),
                )
                states.append(result.state)
                rows.append(self._features(result.state))
                outcomes.append({str(k): float(v) for k, v in result.outcomes.items()})
            self.trajectories = states
            self.state = states[0]
            features = np.mean(rows, axis=0)
            estimator = (
                "trajectory-average-of-expectations" if count > 1 else "conditional-expectations"
            )
            standard_error = (
                (np.std(rows, axis=0, ddof=1) / np.sqrt(count)).tolist() if count > 1 else None
            )
            # Covariance is not a linear observable of an ensemble: include
            # between-trajectory means for unconditional covariance features.
            if c.evolution == "unconditional":
                means = np.array([s.mean for s in states])
                cov = np.mean([s.covariance for s in states], axis=0)
                centered = means - means.mean(axis=0)
                cov += centered.T @ centered / count
                self.state = pg.GaussianState(means.mean(axis=0), cov, self.nodes)
                features = self._features(self.state)
                standard_error = None  # nonlinear plug-in covariance requires bootstrap
                estimator = "trajectory-moment-estimator"
        self.time += 1
        self.execution_shots = shots
        if not np.isfinite(features).all():
            raise FloatingPointError("Nonfinite CV features")
        return ReservoirResult(
            features,
            self.feature_names(),
            c.evolution,
            estimator,
            shots,
            outcomes,
            {
                "tier": c.tier,
                "gaussian": True,
                "readout_mode": c.readout_mode,
                "readout_label": "simulation-only upper-bound readout",
                "standard_error": standard_error,
                "compilation_seconds": self.compilation_seconds,
            },
            {
                "modes": len(self.pattern.inputs),
                "retained_nodes": list(self.nodes),
                "fresh_nodes": c.input_channels,
                "measurements": c.input_channels,
                "edges": int(np.count_nonzero(self.weights))
                + (len(self.nodes) - 1 if c.coupling else 0),
                "cutoff": None,
                "squeezing": c.squeezing,
                "state_storage_bytes": int(self.state.mean.nbytes + self.state.covariance.nbytes),
            },
            asdict(c),
            seconds=perf_counter() - start,
        )
