"""Explicit Gaussian readout capability boundary."""

from abc import ABC, abstractmethod

import numpy as np

from .config import BackendCapabilityError, CVConfig


class GaussianReadout(ABC):
    """Readout protocol independent of reservoir state evolution."""

    @abstractmethod
    def feature_names(self) -> tuple[str, ...]: ...

    @abstractmethod
    def measure(self, state): ...


class StateOracleReadout(GaussianReadout):
    """Simulation-only exact access to internal Gaussian moments."""

    def __init__(self, config: CVConfig, nodes):
        self.config = config
        self.nodes = tuple(nodes)

    def feature_names(self) -> tuple[str, ...]:
        names = [f"mean_{axis}{node}" for node in self.nodes for axis in ("q", "p")]
        preset = self.config.resolved_feature_preset
        if preset == "tier_a":
            names += [f"var_{axis}{node}" for node in self.nodes for axis in ("q", "p")]
        elif preset == "tier_b":
            dimension = 2 * len(self.nodes)
            names += [f"cov_{i}_{j}" for i, j in zip(*np.triu_indices(dimension), strict=True)]
        else:
            names += [f"var_{axis}{node}" for node in self.nodes for axis in ("q", "p")]
            if preset == "diagnostic_redundant":
                names += [f"number_{node}" for node in self.nodes]
            names += [f"square_{axis}{node}" for node in self.nodes for axis in ("q", "p")]
        return tuple(names)

    def measure(self, state):
        mean = np.asarray(state.mean, float)
        covariance = np.asarray(state.covariance, float)
        diagonal = np.diag(covariance)
        preset = self.config.resolved_feature_preset
        if preset == "tier_a":
            return np.r_[mean, diagonal]
        if preset == "tier_b":
            return np.r_[mean, covariance[np.triu_indices_from(covariance)]]
        quadratic = mean**2 + diagonal
        if preset == "diagnostic_redundant":
            numbers = [state.photon_number(node) for node in self.nodes]
            return np.asarray(list(mean) + list(diagonal) + numbers + list(quadratic))
        return np.r_[mean, diagonal, quadratic]


class TapHomodyneReadout(GaussianReadout):
    """Placeholder for a future public-API beamsplitter tap protocol."""

    missing_capability = (
        "A verified public PhotoGraphiQ channel that couples a fresh probe through a "
        "beamsplitter, homodynes one selected quadrature, and returns the surviving "
        "conditional or outcome-averaged memory state."
    )

    def __init__(self, *_args, **_kwargs):
        raise BackendCapabilityError(
            f"TapHomodyneReadout unavailable. Missing upstream capability: {self.missing_capability}"
        )

    def feature_names(self) -> tuple[str, ...]:  # pragma: no cover - construction fails
        raise BackendCapabilityError(self.missing_capability)

    def measure(self, state):  # pragma: no cover - construction fails
        raise BackendCapabilityError(self.missing_capability)
