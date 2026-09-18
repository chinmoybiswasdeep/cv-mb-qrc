"""Research namespace for the standalone measurement-based reservoir package."""

from .reservoirs import (
    BackendCapabilityError,
    CVConfig,
    CVMBReservoir,
    GaussianClassicalTwin,
    GraphixMBReservoir,
    MeasurementBasedReservoir,
    QubitConfig,
    ReservoirResult,
    RidgeReadout,
    WindowedMBQELM,
    chronological_splits,
)

__version__ = "0.1.0"

__all__ = [
    "BackendCapabilityError",
    "CVConfig",
    "CVMBReservoir",
    "GraphixMBReservoir",
    "GaussianClassicalTwin",
    "MeasurementBasedReservoir",
    "QubitConfig",
    "ReservoirResult",
    "RidgeReadout",
    "WindowedMBQELM",
    "chronological_splits",
]
