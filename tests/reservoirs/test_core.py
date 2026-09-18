import builtins
import json
from dataclasses import asdict, replace

import numpy as np
import photographiq as pg
import pytest

from cv_mb_qrc.reservoirs import (
    BackendCapabilityError,
    CVConfig,
    CVMBReservoir,
    ReservoirResult,
    RidgeReadout,
    WindowedMBQELM,
    chronological_splits,
)
from cv_mb_qrc.reservoirs.benchmarks import capacity_targets, metrics, select_readout
from cv_mb_qrc.reservoirs.diagnostics import contraction, fading_memory, feature_diagnostics
from cv_mb_qrc.reservoirs.temporal import delay_features
from cv_mb_qrc.reservoirs.validation import gaussian_one_mode_reference, raw_piquasso_collision


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"memory_modes": True},
        {"transmissivity": 1.1},
        {"efficiency": 0},
        {"squeezing": -1},
        {"angle": np.nan},
        {"evolution": "unknown"},
        {"temporal_edges": 1},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        CVConfig(**kwargs)


def test_capabilities_and_guards():
    for options in ({"backend": "fock"}, {"tier": "C"}, {"adaptive_angle": 0.2}):
        with pytest.raises(BackendCapabilityError):
            CVConfig(**options)
    with pytest.raises(MemoryError):
        CVConfig(memory_modes=40)


def test_reset_causality_shape_serialization(tmp_path):
    model = CVMBReservoir()
    u = np.random.default_rng(12).normal(size=12)
    a = model.run_sequence(u)
    b = model.reset().run_sequence(u)
    np.testing.assert_array_equal(a.features, b.features)
    v = u.copy()
    v[7:] += 10
    np.testing.assert_array_equal(model.reset().run_sequence(v).features[:7], a.features[:7])
    assert a.features.shape == (12, len(a.feature_names))
    a.save(tmp_path / "result.json")
    np.testing.assert_array_equal(
        ReservoirResult.load(tmp_path / "result.json").features, a.features
    )
    assert CVConfig(**json.loads(json.dumps(asdict(model.config)))) == model.config
    for value in ([1, 2], np.nan, [[1]]):
        with pytest.raises(ValueError):
            model.step(value)
    with pytest.raises(ValueError):
        model.run_sequence([1], washout=1)
    with pytest.raises(ValueError):
        model.step(0, shots=0)


def test_vacuum_zero_coupling_and_recurrence():
    c = CVConfig(memory_modes=1, coupling=0)
    m = CVMBReservoir(c)
    r = m.run_sequence([0, 1, -1])
    np.testing.assert_allclose(r.features[:, :2], 0, atol=1e-14)
    np.testing.assert_allclose(r.features[:, [2, 4]], 1, atol=1e-14)
    np.testing.assert_allclose(r.features[:, 3], 0, atol=1e-14)
    recurrent = CVMBReservoir().run_sequence([1, 0]).features[-1]
    static = CVMBReservoir().run_sequence([0, 0]).features[-1]
    assert np.linalg.norm(recurrent - static) > 1e-4


def test_analytical_gaussian_and_raw_piquasso():
    c = CVConfig(memory_modes=1, angle=0.3, thermal_photons=0.1, measurement_noise=0.2)
    model = CVMBReservoir(c)
    initial = model.state.copy()
    value = 0.17
    x = float((c.input_scale * model.mask @ np.array([value]) + model.bias)[0])
    expected_mean, expected_cov = gaussian_one_mode_reference(
        initial.mean,
        initial.covariance,
        value=x,
        variance_q=np.exp(2 * c.squeezing),
        variance_p=np.exp(-2 * c.squeezing),
        coupling=model.weights[0, 0],
        rotation=c.rotation,
        angle=c.angle,
        feedforward=c.feedforward,
        eta=c.transmissivity,
        thermal=c.thermal_photons,
        detector_noise=c.measurement_noise,
    )
    model.step(value)
    np.testing.assert_allclose(model.state.mean, expected_mean, atol=1e-12, rtol=0)
    np.testing.assert_allclose(model.state.covariance, expected_cov, atol=1e-12, rtol=0)
    joint = model._joint(initial, np.array([value]))
    raw_mu, raw_cov = raw_piquasso_collision(joint, model.weights, [c.rotation])
    p = (
        pg.Pattern(inputs=joint.nodes)
        .append(pg.Rotate(0, c.rotation))
        .append(pg.Entangle(0, 1, model.weights[0, 0]))
    )
    state = pg.simulate(p, initial_state=joint, backend="gaussian").state
    np.testing.assert_allclose(state.mean, raw_mu, atol=1e-12, rtol=0)
    np.testing.assert_allclose(state.covariance, raw_cov, atol=1e-12, rtol=0)


def test_trajectories_and_piquasso_agree():
    c = CVConfig(memory_modes=1, evolution="conditional")
    u = [0.1, 0.2, -0.1]
    a = CVMBReservoir(c).run_sequence(u)
    b = CVMBReservoir(replace(c, backend="piquasso")).run_sequence(u)
    np.testing.assert_allclose(a.features, b.features, atol=1e-11, rtol=0)
    assert a.estimator == "conditional-expectations"
    with pytest.raises(ValueError):
        CVMBReservoir(c).step(0, shots=10)
    adaptive = CVMBReservoir(replace(c, input_channels=2, adaptive_angle=0.2))
    assert np.isfinite(adaptive.run_sequence([[0.1, 0.2], [0.2, 0.3]]).features).all()


def test_sampled_vs_exact_statistical():
    c = CVConfig(memory_modes=1)
    exact = CVMBReservoir(c).step(0.3).features
    estimates = np.array(
        [CVMBReservoir(c).reset(seed=s).step(0.3, shots=100).features for s in range(12)]
    )
    # Six standard errors of independent replicate means; no post-hoc tolerance tuning.
    se = estimates.std(axis=0, ddof=1) / np.sqrt(len(estimates))
    assert np.all(abs(estimates.mean(axis=0) - exact) <= 6 * se + 1e-10)


def test_window_and_split_ownership():
    splits = chronological_splits(100, gap=4, washout=5)
    assert splits["train"][-1] + 4 < splits["validation"][0]
    assert splits["validation"][-1] + 4 < splits["test"][0]
    model = WindowedMBQELM(CVMBReservoir, window=2)
    u = [1, 2, 3, 4]
    a = model.run_sequence(u).features
    b = model.reset().run_sequence([9, 2, 3, 4]).features
    np.testing.assert_array_equal(a[2:], b[2:])
    assert model.summary()["temporal_model"] == "explicit-window"
    windows = delay_features([1, 2, 3], 3)
    np.testing.assert_array_equal(windows, [[1, 0, 0], [2, 1, 0], [3, 2, 1]])


def test_train_only_readout_and_targets():
    u = np.linspace(-1, 1, 20)
    x = np.column_stack([u * u, u**3])
    readout = RidgeReadout(1e-8).fit(u, x, 1 + 2 * u + 3 * u * u)
    np.testing.assert_allclose(readout.predict(u, x), 1 + 2 * u + 3 * u * u, atol=1e-7)
    mean = readout.mean.copy()
    readout.predict(u + 100, x + 100)
    np.testing.assert_array_equal(readout.mean, mean)
    selected = select_readout((u, x, u), (u, x, u))
    assert selected.regularization in (1e-6, 1e-4, 1e-2, 1.0)
    targets, names = capacity_targets(u, 2)
    assert targets.shape == (20, 5) and names[-1] == "cross_1_2"
    assert metrics(u, u)["r2"] == 1


def test_diagnostics():
    def factory():
        return CVMBReservoir(CVConfig(memory_modes=1))

    a = pg.GaussianInput.coherent(0).state(0)
    b = pg.GaussianInput.coherent(1).state(0)
    d = contraction(factory, np.zeros(30), [a, b])["distances"]
    assert d[-1][0] < d[0][0]
    response = fading_memory(factory, np.zeros(20), at=3)["feature_distance"]
    np.testing.assert_array_equal(response[:3], 0)
    assert feature_diagnostics(factory().run_sequence(np.arange(10)).features)["effective_rank"] > 0


def test_optional_import_boundary(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("graphix", "mentpy"):
            raise ImportError("blocked optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert CVMBReservoir().step(0).features.size
    from cv_mb_qrc.reservoirs.graphix_backend import GraphixMBReservoir

    with pytest.raises(ImportError, match="Install"):
        GraphixMBReservoir()
