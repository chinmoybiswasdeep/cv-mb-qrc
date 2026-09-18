import numpy as np
import pytest

from cv_mb_qrc.reservoirs.benchmarks import (
    ClassicalFeatures,
    closed_loop_forecast,
    forecast_targets,
    mackey_glass,
    metrics,
    narma10,
    select_readout,
    select_readout_multi,
)
from cv_mb_qrc.reservoirs.diagnostics import (
    bootstrap_summary,
    hierarchical_bootstrap_summary,
    paired_hierarchical_summary,
)
from cv_mb_qrc.reservoirs.fock import FockConfig, FockMBReservoir, cutoff_study
from cv_mb_qrc.reservoirs.temporal import chronological_splits


def test_tasks_and_exact_delay_control():
    u = np.random.default_rng(12).uniform(0, 0.5, 200)
    y = narma10(u)
    assert y.shape == u.shape and np.isfinite(y).all()
    assert y[9] == pytest.approx(1.5 * u[0] * u[9] + 0.1)
    mg = mackey_glass(100)
    assert mg.shape == (100,) and np.std(mg) > 0.01
    for name in ("delay", "esn", "rff", "input_only", "persistence"):
        model = ClassicalFeatures(name, dimension=4, window=4)
        x = model.transform(u)
        np.testing.assert_array_equal(x, model.transform(u))
        changed = u.copy()
        changed[50:] += 1
        np.testing.assert_array_equal(x[:50], model.transform(changed)[:50])
    x = ClassicalFeatures("delay", window=4).transform(u)
    assert metrics(u[:-3], x[3:, 3])["r2"] == 1
    summary = bootstrap_summary([1, 2, 3])
    assert summary["mean"] == 2 and summary["values"] == [1, 2, 3]
    for value in ([-1], [np.nan]):
        with pytest.raises(ValueError):
            narma10(value)
    with pytest.raises(ValueError):
        chronological_splits(4, gap=4)
    with pytest.raises(ValueError):
        metrics([1, 1], [1, 1])


@pytest.mark.fock
def test_fock_invariants_and_convergence():
    result = FockMBReservoir().run_sequence([0.02, 0.04])
    assert result.features.shape == (2, 4)
    for d in result.diagnostics["step_diagnostics"]:
        assert d["trace"] == pytest.approx(1, abs=1e-12)
        assert d["minimum_eigenvalue"] >= -1e-12
        assert d["hermiticity_error"] < 1e-12
        assert 0 <= d["herald_probability"] <= 1
    study = cutoff_study([0.02], cutoffs=(8, 12))
    assert study["observable_convergence"]
    assert study["rows"][-1]["failure_probability"] > 0
    with pytest.raises(MemoryError):
        FockConfig(cutoff=1000)


def test_ci_single_observation_summary_is_not_fake_uncertainty():
    summary = bootstrap_summary([1.25], scientific=False)
    assert summary["mean"] == summary["median"] == 1.25
    assert summary["std"] is None and summary["bootstrap95"] is None
    assert not summary["scientific_uncertainty_valid"]
    with pytest.raises(ValueError, match="at least two"):
        bootstrap_summary([1.25], scientific=True)


def test_forecast_alignment_and_closed_loop_feedback():
    series = np.arange(12.0)
    inputs, targets = forecast_targets(series, [1, 3, 5])
    np.testing.assert_array_equal(inputs, series[:7])
    np.testing.assert_array_equal(targets[:, 0], series[1:8])
    np.testing.assert_array_equal(targets[:, 2], series[5:12])
    rollout = closed_loop_forecast(
        lambda history: history[-1] + 1,
        [0.0, 1.0],
        np.arange(2.0, 8.0),
        threshold=0.5,
    )
    assert rollout["valid_prediction_time"] == 6
    assert not rollout["diverged"]


def test_hierarchical_bootstrap_respects_dataset_axis():
    summary = hierarchical_bootstrap_summary(
        [0.0, 0.1, 1.0, 1.1],
        [10, 10, 11, 11],
        [0, 1, 0, 1],
        seed=3,
        draws=200,
    )
    assert summary["bootstrap_method"] == "dataset_then_reservoir_within_dataset"
    assert summary["scientific_uncertainty_valid"]
    with pytest.raises(ValueError, match="at least two datasets"):
        hierarchical_bootstrap_summary([0.0, 0.1], [10, 10], [0, 1])


def test_paired_bootstrap_preserves_declared_pairs():
    left = np.array([1.0, 1.2, 2.0, 2.2])
    right = np.array([0.5, 0.7, 1.5, 1.7])
    summary = paired_hierarchical_summary(
        left,
        right,
        [10, 10, 11, 11],
        [0, 1, 0, 1],
        seed=4,
        draws=100,
    )
    np.testing.assert_allclose(summary["paired_differences"], 0.5)
    assert summary["mean_paired_difference"] == pytest.approx(0.5)
    assert summary["left_win_fraction"] == 1
    with pytest.raises(ValueError, match="identical shapes"):
        paired_hierarchical_summary([1, 2], [1], [0, 0], [0, 1])


def test_multi_target_ridge_matches_independent_selection_with_one_svd():
    rng = np.random.default_rng(20)
    inputs = rng.normal(size=100)
    features = rng.normal(size=(100, 6))
    targets = np.c_[features[:, 0] + 0.1 * rng.normal(size=100), inputs**2]
    train = (inputs[:60], features[:60], targets[:60])
    validation = (inputs[60:80], features[60:80], targets[60:80])
    regularizations = (1e-6, 1e-3, 1.0)
    multi = select_readout_multi(
        train,
        validation,
        regularizations,
        input_access="reservoir_only",
    )
    assert multi.factorization_count == 1
    for column in range(targets.shape[1]):
        single = select_readout(
            (*train[:2], train[2][:, column]),
            (*validation[:2], validation[2][:, column]),
            regularizations,
            input_access="reservoir_only",
        )
        assert multi.selected_regularizations[column] == single.regularization
        np.testing.assert_allclose(
            multi.predict(inputs[80:], features[80:])[:, column],
            single.predict(inputs[80:], features[80:]),
            atol=1e-12,
        )
