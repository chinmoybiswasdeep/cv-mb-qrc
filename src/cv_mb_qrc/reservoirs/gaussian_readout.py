"""Explicit Gaussian readout capability boundary."""

from abc import ABC, abstractmethod

import numpy as np

from .config import BackendCapabilityError, CVConfig


def measurement_budget(memory_modes, tier, *, shots_per_setting=1000):
    """Conservative state-copy budget for a future homodyne implementation.

    Means and diagonal variances share a single-quadrature setting. Each
    off-diagonal covariance additionally requires a rotated joint-quadrature
    setting. This is a budgeting roadmap, not an implemented measurement path.
    """
    if isinstance(memory_modes, bool) or not isinstance(memory_modes, int) or memory_modes < 1:
        raise ValueError("memory_modes must be a positive integer")
    if tier not in ("A", "B"):
        raise ValueError("Measurement budgeting is defined for Tier A or B")
    if (
        isinstance(shots_per_setting, bool)
        or not isinstance(shots_per_setting, int)
        or shots_per_setting < 1
    ):
        raise ValueError("shots_per_setting must be a positive integer")
    quadratures = 2 * memory_modes
    settings = quadratures if tier == "A" else quadratures * (quadratures + 1) // 2
    feature_count = 2 * quadratures if tier == "A" else quadratures + settings
    return {
        "tier": tier,
        "memory_modes": memory_modes,
        "simulator_feature_count": feature_count,
        "minimum_distinct_homodyne_settings": settings,
        "shots_per_setting": shots_per_setting,
        "state_copies": settings * shots_per_setting,
        "status": "roadmap-only; tap/backaction channel unavailable",
        "assumptions": (
            "means co-estimated with diagonal variances; one additional rotated setting "
            "per off-diagonal covariance"
        ),
    }


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
