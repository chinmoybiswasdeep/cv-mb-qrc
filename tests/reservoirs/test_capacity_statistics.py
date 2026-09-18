import numpy as np
import pytest

from cv_mb_qrc.reservoirs.benchmarks import (
    assert_no_exact_target_leakage,
    benjamini_hochberg,
    capacity_targets,
    generate_capacity_nulls,
    metrics,
    null_corrected_capacity,
    select_readout,
)
from cv_mb_qrc.reservoirs.temporal import delay_features


def _target_scores(inputs, features, targets):
    partitions = (slice(0, 180), slice(200, 270), slice(290, 380))
    scores = []
    for column in range(targets.shape[1]):
        train, validation, test = partitions
        model = select_readout(
            (inputs[train], features[train], targets[train, column]),
            (inputs[validation], features[validation], targets[validation, column]),
            (1e-6,),
            input_access="reservoir_only",
        )
        prediction = model.predict(inputs[test], features[test])
        scores.append(metrics(targets[test, column], prediction)["r2"])
    return np.asarray(scores)


def test_benjamini_hochberg_known_values_and_bounds():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.8])
    np.testing.assert_allclose(q, [0.04, 0.05333333333333334, 0.05333333333333334, 0.8])
    assert np.all((0 <= q) & (q <= 1))


@pytest.mark.parametrize("method", ["circular_shift", "block_permutation", "independent_surrogate"])
def test_dataset_level_null_transforms_are_deterministic_and_audited(method):
    inputs = np.random.default_rng(4).uniform(-1, 1, 120)
    first = generate_capacity_nulls(inputs, 4, 5, method=method, seed=9)
    second = generate_capacity_nulls(inputs, 4, 5, method=method, seed=9)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    if method == "circular_shift":
        assert all(row["displacement"] > 4 for row in first[2])


def test_null_features_calibrate_and_linear_memory_is_recovered():
    rng = np.random.default_rng(81)
    inputs = rng.uniform(-1, 1, 400)
    targets, names = capacity_targets(inputs, 2)
    null_targets, _, _ = generate_capacity_nulls(
        inputs, 2, 79, method="independent_surrogate", seed=91
    )

    random_features = rng.normal(size=(len(inputs), 5))
    observed = _target_scores(inputs, random_features, targets)
    null = np.array([_target_scores(inputs, random_features, row) for row in null_targets])
    calibrated = null_corrected_capacity(observed, null, names)
    assert calibrated["total_significant_capacity"] == 0
    assert not any(calibrated["significant_mask"])
    assert calibrated["legacy_clipped_total"] >= calibrated["raw_total"]

    memory_features = delay_features(inputs, 3)
    observed = _target_scores(inputs, memory_features, targets)
    null = np.array([_target_scores(inputs, memory_features, row) for row in null_targets])
    recovered = null_corrected_capacity(observed, null, names)
    assert recovered["targets"]["linear_1"]["significant_at_alpha"]
    assert recovered["targets"]["linear_2"]["significant_at_alpha"]
    assert recovered["families"]["linear"]["significant_capacity"] > 1.8

    shuffled = memory_features[rng.permutation(len(memory_features))]
    shuffled_observed = _target_scores(inputs, shuffled, targets)
    shuffled_null = np.array([_target_scores(inputs, shuffled, row) for row in null_targets])
    shuffled_report = null_corrected_capacity(shuffled_observed, shuffled_null, names)
    assert shuffled_report["total_significant_capacity"] == 0


def test_deliberately_leaky_feature_is_detected():
    inputs = np.random.default_rng(2).uniform(-1, 1, 100)
    targets, _ = capacity_targets(inputs, 2)
    with pytest.raises(ValueError, match="exact target leakage"):
        assert_no_exact_target_leakage(np.c_[inputs, targets[:, 0]], targets)
