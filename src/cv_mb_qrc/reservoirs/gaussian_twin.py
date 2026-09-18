"""Independent NumPy affine-Gaussian twin of a fixed Gaussian CV channel."""

from dataclasses import asdict, dataclass, replace
from time import perf_counter

import numpy as np

from .base import MeasurementBasedReservoir, input_vector
from .config import BackendCapabilityError, CVConfig, integer
from .cv import CVMBReservoir
from .results import ReservoirResult


@dataclass
class GaussianMoments:
    """Minimal NumPy state container used after channel identification."""

    mean: np.ndarray
    covariance: np.ndarray
    nodes: tuple[int, ...]

    def copy(self):
        return GaussianMoments(self.mean.copy(), self.covariance.copy(), self.nodes)


class GaussianClassicalTwin(MeasurementBasedReservoir):
    """Exact affine state-space twin for input-independent Gaussian channels.

    PhotoGraphiQ is used only while identifying ``A``, ``B``, ``c`` and ``N``
    during construction. All subsequent operations use NumPy only.
    """

    config: CVConfig

    def __init__(self, config: CVConfig | None = None):
        self.config = config or CVConfig()
        if self.config.readout_mode != "state_oracle":
            raise BackendCapabilityError("The Gaussian twin supports state-oracle features only")
        if self.config.evolution != "unconditional":
            raise BackendCapabilityError("The affine twin requires unconditional evolution")
        if self.config.phase_scale or self.config.squeeze_scale or self.config.adaptive_angle:
            raise BackendCapabilityError(
                "The affine twin requires input-independent covariance and measurement angles"
            )
        if self.config.resolved_feature_preset == "diagnostic_redundant":
            raise BackendCapabilityError("The twin does not expose redundant diagnostic features")
        start = perf_counter()
        oracle = CVMBReservoir(replace(self.config, readout_mode="state_oracle"))
        self.nodes = oracle.nodes
        self.mask = oracle.mask.copy()
        self.bias = oracle.bias.copy()
        self.weights = oracle.weights.copy()
        self._identify(oracle)
        self.identification_seconds = perf_counter() - start
        self.spectral_radius = float(max(abs(np.linalg.eigvals(self.A)), default=0.0))
        self.covariance_input_dependent = False
        self.reset()

    def _identify(self, oracle: CVMBReservoir) -> None:
        import photographiq as pg

        dimension = 2 * self.config.memory_modes
        identity = np.eye(dimension)

        def apply(mean, covariance, value):
            memory = pg.GaussianState(mean, covariance, oracle.nodes)
            joint = oracle._joint(memory, np.asarray(value, float))
            assert oracle.channel is not None
            return oracle.channel.apply(joint)

        zero_input = np.zeros(self.config.input_channels)
        base = apply(np.zeros(dimension), identity, zero_input)
        self.c = np.asarray(base.mean, float).copy()
        self.A = np.empty((dimension, dimension))
        for index in range(dimension):
            vector = np.zeros(dimension)
            vector[index] = 1.0
            self.A[:, index] = apply(vector, identity, zero_input).mean - self.c
        self.B = np.empty((dimension, self.config.input_channels))
        for channel in range(self.config.input_channels):
            value = np.zeros(self.config.input_channels)
            value[channel] = 1.0
            self.B[:, channel] = apply(np.zeros(dimension), identity, value).mean - self.c
        noise = np.asarray(base.covariance, float) - self.A @ self.A.T
        self.N = (noise + noise.T) / 2

    @staticmethod
    def _validate_moments(mean, covariance, dimension):
        mu, cov = np.asarray(mean, float), np.asarray(covariance, float)
        if mu.shape != (dimension,) or cov.shape != (dimension, dimension):
            raise ValueError("Initial Gaussian moments have incompatible shapes")
        if not np.isfinite(mu).all() or not np.isfinite(cov).all():
            raise ValueError("Initial Gaussian moments must be finite")
        if not np.allclose(cov, cov.T, atol=1e-12, rtol=0):
            raise ValueError("Initial covariance must be symmetric")
        if np.linalg.eigvalsh(cov).min() < -1e-12:
            raise ValueError("Initial covariance must be positive semidefinite")
        return mu.copy(), cov.copy()

    def reset(self, *, seed=None, initial_state=None):
        integer(self.config.seed if seed is None else seed, "seed", 0)
        dimension = 2 * self.config.memory_modes
        if initial_state is None:
            mean, covariance = np.zeros(dimension), np.eye(dimension)
        elif hasattr(initial_state, "mean") and hasattr(initial_state, "covariance"):
            mean, covariance = initial_state.mean, initial_state.covariance
        elif isinstance(initial_state, tuple) and len(initial_state) == 2:
            mean, covariance = initial_state
        else:
            raise ValueError(
                "initial_state must expose mean/covariance or be a (mean, covariance) pair"
            )
        mean, covariance = self._validate_moments(mean, covariance, dimension)
        self.state = GaussianMoments(mean, covariance, self.nodes)
        self.initial_state = self.state.copy()
        self.time = 0
        return self

    def feature_names(self) -> tuple[str, ...]:
        names = [f"mean_{axis}{node}" for node in self.nodes for axis in ("q", "p")]
        preset = self.config.resolved_feature_preset
        if preset == "tier_a":
            names += [f"var_{axis}{node}" for node in self.nodes for axis in ("q", "p")]
        elif preset == "tier_b":
            dimension = 2 * len(self.nodes)
            names += [f"cov_{i}_{j}" for i, j in zip(*np.triu_indices(dimension), strict=True)]
        elif preset == "diagnostic_quadratic":
            names += [f"var_{axis}{node}" for node in self.nodes for axis in ("q", "p")]
            names += [f"square_{axis}{node}" for node in self.nodes for axis in ("q", "p")]
        else:
            raise BackendCapabilityError(f"Unsupported twin preset: {preset}")
        return tuple(names)

    def _features(self) -> np.ndarray:
        mean, covariance = self.state.mean, self.state.covariance
        diagonal = np.diag(covariance)
        preset = self.config.resolved_feature_preset
        if preset == "tier_a":
            return np.r_[mean, diagonal]
        if preset == "tier_b":
            return np.r_[mean, covariance[np.triu_indices(len(mean))]]
        if preset == "diagnostic_quadratic":
            return np.r_[mean, diagonal, mean**2 + diagonal]
        raise BackendCapabilityError(f"Unsupported twin preset: {preset}")

    def step(self, input_value, *, shots=None) -> ReservoirResult:
        if shots is not None:
            raise BackendCapabilityError("The deterministic Gaussian twin has no shot execution")
        value = input_vector(input_value, self.config.input_channels)
        previous = self.state if self.config.temporal_edges else self.initial_state
        covariance = self.A @ previous.covariance @ self.A.T + self.N
        self.state = GaussianMoments(
            self.A @ previous.mean + self.B @ value + self.c,
            (covariance + covariance.T) / 2,
            self.nodes,
        )
        self.time += 1
        return ReservoirResult(
            self._features(),
            self.feature_names(),
            "unconditional",
            "exact-classical-affine",
            None,
            diagnostics={
                "spectral_radius": self.spectral_radius,
                "covariance_input_dependent": False,
                "identification_seconds": self.identification_seconds,
            },
            resources={
                "state_dimension": len(self.state.mean),
                "state_storage_bytes": int(self.state.mean.nbytes + self.state.covariance.nbytes),
            },
            configuration=asdict(self.config),
        )
