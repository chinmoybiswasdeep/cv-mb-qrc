from dataclasses import replace

import numpy as np
import pytest

from cv_mb_qrc.reservoirs import CVConfig, CVMBReservoir, GaussianClassicalTwin
from cv_mb_qrc.reservoirs.config import BackendCapabilityError


@pytest.mark.parametrize(
    "memory_modes,input_channels,tier,seed,coupling,transmissivity,squeezing,feedforward,steps",
    [
        (1, 1, "A", 0, 0.0, 0.6, 0.1, 0.0, 100),
        (2, 1, "B", 3, 0.2, 0.8, 0.3, 0.15, 20),
        (1, 2, "B", 7, 0.35, 0.95, 0.5, 0.3, 30),
        (4, 1, "A", 9, 0.1, 0.7, 0.2, 0.1, 3),
    ],
)
def test_gaussian_twin_matches_oracle(
    memory_modes,
    input_channels,
    tier,
    seed,
    coupling,
    transmissivity,
    squeezing,
    feedforward,
    steps,
):
    config = CVConfig(
        memory_modes=memory_modes,
        input_channels=input_channels,
        tier=tier,
        seed=seed,
        coupling=coupling,
        transmissivity=transmissivity,
        squeezing=squeezing,
        feedforward=feedforward,
    )
    oracle, twin = CVMBReservoir(config), GaussianClassicalTwin(config)
    rng = np.random.default_rng(1000 + seed)
    dimension = 2 * memory_modes
    mean = rng.normal(scale=0.2, size=dimension)
    covariance = np.eye(dimension) * rng.uniform(1.0, 1.8)
    import photographiq as pg

    initial = pg.GaussianState(mean, covariance, tuple(range(memory_modes)))
    oracle.reset(initial_state=initial)
    twin.reset(initial_state=(mean, covariance))
    inputs = rng.normal(scale=0.4, size=(steps, input_channels))
    oracle_result = oracle.run_sequence(inputs)
    twin_result = twin.run_sequence(inputs)
    assert twin.feature_names() == oracle.feature_names()
    np.testing.assert_allclose(twin.state.mean, oracle.state.mean, atol=2e-15, rtol=0)
    np.testing.assert_allclose(twin.state.covariance, oracle.state.covariance, atol=5e-14, rtol=0)
    np.testing.assert_allclose(twin_result.features, oracle_result.features, atol=5e-14, rtol=0)
    assert twin.B.shape == (dimension, input_channels)
    assert np.isfinite(twin.spectral_radius)


def test_twin_rejects_time_varying_or_nonoracle_channels():
    with pytest.raises(BackendCapabilityError, match="input-independent"):
        GaussianClassicalTwin(CVConfig(squeeze_scale=0.1))
    with pytest.raises(BackendCapabilityError, match="state-oracle"):
        GaussianClassicalTwin(CVConfig(readout_mode="physical_probe"))


def test_twin_replace_preserves_tier_semantics():
    config = replace(CVConfig(tier="B"), tier="A")
    twin = GaussianClassicalTwin(config)
    assert config.resolved_feature_preset == "tier_a"
    assert len(twin.feature_names()) == 8
