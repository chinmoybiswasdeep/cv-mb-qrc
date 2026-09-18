from dataclasses import replace

import numpy as np
import pytest

from cv_mb_qrc.reservoirs.benchmarks import null_corrected_capacity
from cv_mb_qrc.reservoirs.config import CVConfig
from cv_mb_qrc.reservoirs.diagnostics import feature_diagnostics
from cv_mb_qrc.reservoirs.readout import RidgeReadout


def test_train_only_collapsed_filter_and_rank_diagnostics():
    inputs = np.arange(4.0)
    features = np.c_[inputs, np.ones(4)]
    readout = RidgeReadout(variance_floor=1e-10).fit(inputs, features, inputs)
    assert readout.removed_columns == [2]  # duplicate is valid; constant column is removed
    report = feature_diagnostics(features)
    assert report["exact_rank"] == 1 and report["variance_collapsed_columns"] == 1


def test_null_capacity_reports_bias_correction():
    report = null_corrected_capacity([0.2, -0.1], [[0.1, 0.1], [0.1, 0.1]])
    assert report["raw_total"] == pytest.approx(0.1)
    assert report["null_corrected_total"] == pytest.approx(-0.1)


def test_tiers_are_canonical_and_distinct():
    a, b = CVConfig(memory_modes=2, tier="A"), CVConfig(memory_modes=2, tier="B")
    assert a.feature_preset is None and b.feature_preset is None
    assert a.resolved_feature_preset == "tier_a"
    assert b.resolved_feature_preset == "tier_b"
    changed = replace(b, tier="A")
    assert changed.feature_preset is None and changed.resolved_feature_preset == "tier_a"
    with pytest.raises(ValueError, match="disagree"):
        CVConfig(tier="A", feature_preset="tier_b")


def test_tier_dimensions_names_and_result_round_trip(tmp_path):
    from cv_mb_qrc.reservoirs.cv import CVMBReservoir
    from cv_mb_qrc.reservoirs.results import ReservoirResult

    tier_a = CVMBReservoir(CVConfig(memory_modes=2, tier="A"))
    tier_b = CVMBReservoir(CVConfig(memory_modes=2, tier="B"))
    assert len(tier_a.feature_names()) == 8
    assert len(tier_b.feature_names()) == 14
    assert tier_a.feature_names() != tier_b.feature_names()
    assert (
        tier_b.feature_names() == CVMBReservoir(CVConfig(memory_modes=2, tier="B")).feature_names()
    )
    result = tier_b.run_sequence([0.1, -0.2])
    path = tmp_path / "tier_b.json"
    result.save(path)
    loaded = ReservoirResult.load(path)
    assert loaded.feature_names == result.feature_names == tier_b.feature_names()


def test_physical_readout_is_not_silently_approximated():
    from cv_mb_qrc.reservoirs.config import BackendCapabilityError
    from cv_mb_qrc.reservoirs.cv import CVMBReservoir

    with pytest.raises(BackendCapabilityError, match="TapHomodyneReadout"):
        CVMBReservoir(CVConfig(readout_mode="physical_probe"))


def test_physical_measurement_budget_is_explicit_and_conservative():
    from cv_mb_qrc.reservoirs.gaussian_readout import StateOracleReadout, measurement_budget

    tier_a = measurement_budget(2, "A", shots_per_setting=100)
    tier_b = measurement_budget(2, "B", shots_per_setting=100)
    assert tier_a["simulator_feature_count"] == 8
    assert tier_a["state_copies"] == 400
    assert tier_b["simulator_feature_count"] == 14
    assert tier_b["state_copies"] == 1000
    assert "roadmap-only" in tier_b["status"]
    with pytest.raises(ValueError):
        measurement_budget(0, "A")
    with pytest.raises(ValueError):
        measurement_budget(1, "C")
    with pytest.raises(ValueError):
        measurement_budget(1, "A", shots_per_setting=0)

    class State:
        mean = np.array([0.2, -0.1])
        covariance = np.eye(2)

        @staticmethod
        def photon_number(_node):
            return 0.25

    readout = StateOracleReadout(CVConfig(feature_preset="diagnostic_redundant"), (0,))
    assert readout.feature_names() == (
        "mean_q0",
        "mean_p0",
        "var_q0",
        "var_p0",
        "number_0",
        "square_q0",
        "square_p0",
    )
    np.testing.assert_allclose(readout.measure(State()), [0.2, -0.1, 1, 1, 0.25, 1.04, 1.01])


def test_direct_input_access_is_explicit():
    inputs = np.linspace(-1, 1, 20)
    features = (inputs**2)[:, None]
    model = RidgeReadout(input_access="reservoir_only").fit(inputs, features, inputs**2)
    np.testing.assert_allclose(
        model.predict(inputs, features),
        model.predict(inputs + 100, features),
    )
    with pytest.raises(ValueError, match="input_access"):
        RidgeReadout(input_access="implicit")
