from itertools import product

import numpy as np
import pytest

from cv_mb_qrc.reservoirs import GraphixMBReservoir, QubitConfig
from cv_mb_qrc.reservoirs.graphix_backend import graphix_wire
from cv_mb_qrc.reservoirs.validation import collision_kraus

pytest.importorskip("graphix")
pytestmark = pytest.mark.graphix


@pytest.mark.parametrize("entangle,feedforward", list(product([False, True], repeat=2)))
def test_exact_kraus_reference(entangle, feedforward):
    c = QubitConfig(entangle=entangle, feedforward=feedforward)
    m = GraphixMBReservoir(c)
    state = np.array([[0.6, 0.1 + 0.2j], [0.1 - 0.2j, 0.4]], complex)
    m.reset(initial_state=state)
    for value in [0.1, -0.3, 0.7]:
        ks = collision_kraus(
            m.bias + m.mask * c.input_scale * value,
            c.angle,
            entangle=entangle,
            feedforward=feedforward,
        )
        np.testing.assert_allclose(sum(k.conj().T @ k for k in ks), np.eye(2), atol=1e-12)
        state = sum(k @ state @ k.conj().T for k in ks)
        result = m.step(value)
        np.testing.assert_allclose(m.state, state, atol=1e-12, rtol=0)
        assert abs(sum(result.diagnostics["branch_probabilities"]) - 1) < 1e-12


def test_wire_all_branches_and_identity():
    initial = np.array([1, 1j]) / np.sqrt(2)
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for bits in product((0, 1), repeat=2):
        actual = graphix_wire(initial, [0.2, -0.4], bits)
        adaptive = graphix_wire(initial, [0.2, -0.4], bits, adaptive=True)
        assert abs(np.vdot(actual, adaptive)) ** 2 == pytest.approx(1, abs=1e-12)
        expected = initial
        for angle in [0.2, -0.4]:
            expected = h @ np.diag([1, np.exp(-1j * angle)]) @ expected
        assert abs(np.vdot(expected, actual)) ** 2 == pytest.approx(1, abs=1e-12)
        assert abs(np.vdot(initial, graphix_wire(initial, [0, 0], bits))) ** 2 == pytest.approx(
            1, abs=1e-12
        )


@pytest.mark.mentpy
def test_mentpy_common_subset():
    pytest.importorskip("mentpy")
    from cv_mb_qrc.reservoirs.mentpy_backend import compare_wire

    for state in ([1, 0], np.array([1, 1j]) / np.sqrt(2)):
        result = compare_wire(state)
        assert result["density_max_error"] < 1e-12
        assert result["state_max_error_up_to_phase"] < 1e-12


def test_qubit_shots_reset_causality():
    m = GraphixMBReservoir()
    a = m.run_sequence([0.1, 0.2, 0.3], shots=20)
    b = m.reset().run_sequence([0.1, 0.2, 0.3], shots=20)
    np.testing.assert_array_equal(a.features, b.features)
    c = m.reset().run_sequence([0.1, 0.2, 8], shots=20)
    np.testing.assert_array_equal(a.features[:2], c.features[:2])
    with pytest.raises(ValueError):
        m.step(0, shots=2)


def test_qubit_shots_statistical():
    exact = GraphixMBReservoir().step(0.3).features
    samples = np.array(
        [GraphixMBReservoir().reset(seed=s).step(0.3, shots=100).features for s in range(10)]
    )
    se = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    assert np.all(abs(samples.mean(axis=0) - exact) <= 6 * se + 1e-12)
