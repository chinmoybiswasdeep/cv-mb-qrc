"""Immutable, validated reservoir policies; angles are always radians."""

from dataclasses import dataclass

import numpy as np


class BackendCapabilityError(NotImplementedError):
    """An explicitly requested physical or numerical operation is unavailable."""


def integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class CVConfig:
    memory_modes: int = 2
    input_channels: int = 1
    seed: int = 0
    squeezing: float = 0.3
    coupling: float = 0.2
    input_scale: float = 0.4
    phase_scale: float = 0.0
    squeeze_scale: float = 0.0
    angle: float = 0.0
    rotation: float = 0.4
    feedforward: float = 0.15
    transmissivity: float = 0.8
    thermal_photons: float = 0.0
    measurement_noise: float = 0.0
    efficiency: float = 1.0
    adaptive_angle: float = 0.0
    tier: str = "B"
    feature_preset: str | None = None
    readout_mode: str = "state_oracle"
    n_replicas: int = 1
    n_readout_shots: int = 1
    probe_strength: float = 0.05
    variance_floor: float = 1e-12
    evolution: str = "unconditional"
    backend: str = "gaussian"
    temporal_edges: bool = True
    max_modes: int = 32
    max_trajectories: int = 10000

    def __post_init__(self):
        for name in ("memory_modes", "input_channels", "max_modes", "max_trajectories", "n_replicas", "n_readout_shots"):
            integer(getattr(self, name), name)
        integer(self.seed, "seed", 0)
        for name in (
            "squeezing",
            "coupling",
            "input_scale",
            "phase_scale",
            "squeeze_scale",
            "angle",
            "rotation",
            "feedforward",
            "transmissivity",
            "thermal_photons",
            "measurement_noise",
            "efficiency",
            "adaptive_angle",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if min(self.squeezing, self.thermal_photons, self.measurement_noise) < 0:
            raise ValueError("Squeezing and noise strengths must be nonnegative")
        if not 0 <= self.transmissivity <= 1 or not 0 < self.efficiency <= 1:
            raise ValueError("Invalid transmissivity or detector efficiency")
        if self.tier not in ("A", "B"):
            raise BackendCapabilityError("Gaussian CV reservoir supports Tier A/B only")
        canonical = {"A": "tier_a", "B": "tier_b"}
        diagnostic = {"diagnostic_quadratic", "diagnostic_redundant"}
        preset = canonical[self.tier] if self.feature_preset is None else self.feature_preset
        if preset not in set(canonical.values()) | diagnostic:
            raise ValueError("Unknown CV feature preset")
        if preset in canonical.values() and preset != canonical[self.tier]:
            raise ValueError("tier and feature_preset disagree; use a diagnostic preset explicitly")
        object.__setattr__(self, "feature_preset", preset)
        if self.readout_mode not in ("state_oracle", "physical_probe"):
            raise ValueError("readout_mode must be state_oracle or physical_probe")
        if not 0 < self.probe_strength <= 1 or self.variance_floor < 0:
            raise ValueError("Invalid probe strength or variance floor")
        if self.backend not in ("gaussian", "piquasso"):
            raise BackendCapabilityError("Select gaussian or piquasso; no implicit Fock fallback")
        if self.evolution not in ("unconditional", "conditional"):
            raise ValueError("Choose unconditional or conditional evolution")
        if self.adaptive_angle and self.evolution != "conditional":
            raise BackendCapabilityError("Outcome-adaptive angles require conditional trajectories")
        if not isinstance(self.temporal_edges, bool):
            raise ValueError("temporal_edges must be boolean")
        if self.memory_modes + self.input_channels > self.max_modes:
            raise MemoryError("Live modes exceed max_modes; explicitly raise the guard to override")


@dataclass(frozen=True)
class QubitConfig:
    seed: int = 0
    input_scale: float = 1.0
    angle: float = 0.35
    entangle: bool = True
    feedforward: bool = True
    temporal_edges: bool = True
    evolution: str = "unconditional"
    input_channels: int = 1
    max_trajectories: int = 10000

    def __post_init__(self):
        integer(self.seed, "seed", 0)
        integer(self.input_channels, "input_channels")
        integer(self.max_trajectories, "max_trajectories")
        if self.input_channels != 1:
            raise BackendCapabilityError("Initial collision reservoir has one input channel")
        if not np.isfinite([self.input_scale, self.angle]).all():
            raise ValueError("Angles/scales must be finite")
        if self.evolution not in ("unconditional", "conditional"):
            raise ValueError("Choose unconditional or conditional evolution")
        for name in ("entangle", "feedforward", "temporal_edges"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
