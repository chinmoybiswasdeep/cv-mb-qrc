"""Optional independent deterministic-wire reference, never a CV oracle."""

from importlib.metadata import version

import numpy as np


def backend_compatibility_matrix():
    """Audited native capability matrix; unsupported cells are never emulated."""
    return {
        "corrected_pure_xy_wire": {
            "internal_reference": True,
            "graphix": True,
            "mentpy": True,
            "cv_gaussian": False,
            "fock": False,
        },
        "mixed_collision_channel": {
            "internal_reference": True,
            "graphix": True,
            "mentpy": False,
            "cv_gaussian": False,
            "fock": False,
        },
        "gaussian_unconditional_channel": {
            "internal_reference": "affine twin",
            "graphix": False,
            "mentpy": False,
            "cv_gaussian": True,
            "fock": False,
        },
        "conditional_photon_number": {
            "internal_reference": False,
            "graphix": False,
            "mentpy": False,
            "cv_gaussian": False,
            "fock": True,
        },
        "physical_tap_homodyne_with_surviving_memory": {
            "internal_reference": False,
            "graphix": False,
            "mentpy": False,
            "cv_gaussian": False,
            "fock": False,
        },
    }


def compare_wire(state, angles=(0.2, -0.4)):
    try:
        import mentpy as mp
    except ImportError as exc:
        raise ImportError("Install cv-mb-qrc[validation] for MentPy reference") from exc
    if version("mentpy") != "0.1.0a15":
        raise ImportError("This reference is audited against mentpy==0.1.0a15")
    from .graphix_backend import graphix_wire

    angles = np.asarray(angles, float)
    if angles.ndim != 1 or len(angles) < 2 or not np.isfinite(angles).all():
        raise ValueError("MentPy reference requires at least two finite wire angles")
    state = np.asarray(state, complex)
    reference = mp.templates.linear_cluster(len(angles) + 1)
    expected_edges = {(i, i + 1) for i in range(len(angles))}
    edges = {tuple(sorted(e)) for e in reference.graph.edges}
    if (
        edges != expected_edges
        or reference.input_nodes != [0]
        or reference.output_nodes != [len(angles)]
    ):
        raise ValueError("MentPy wire topology/order changed")
    simulator = mp.PatternSimulator(reference, input_state=state, backend="numpy-sv", window_size=2)
    actual = np.asarray(simulator.run(angles.tolist(), output_form="sv"), complex)
    expected = graphix_wire(state, angles)
    if list(reference.trainable_nodes) != list(range(len(angles))):
        raise ValueError("MentPy wire trainable-node order changed")
    order = {n: i for i, n in enumerate(reference.measurement_order)}
    if any(order[i] >= order[i + 1] for i in range(len(angles))):
        raise ValueError("MentPy wire dependency order changed")
    overlap = np.vdot(expected, actual)
    aligned = actual * np.exp(-1j * np.angle(overlap))
    return {
        "density_max_error": float(
            np.max(abs(np.outer(actual, actual.conj()) - np.outer(expected, expected.conj())))
        ),
        "state_max_error_up_to_phase": float(np.max(abs(aligned - expected))),
        "probability_max_error": float(np.max(abs(abs(actual) ** 2 - abs(expected) ** 2))),
        "node_map": {str(i): i for i in range(len(angles) + 1)},
        "edges": sorted(edges),
        "input_nodes": [0],
        "output_nodes": [len(angles)],
        "measured_nodes": list(range(len(angles))),
        "trainable_nodes": list(reference.trainable_nodes),
        "measurement_order": list(reference.measurement_order),
        "plane": "XY",
        "angles_radians": angles.tolist(),
        "convention": "MSB-first; bit 0 positive eigenstate; corrected frame; MentPy zero branch",
        "branch_probability": 2.0 ** (-len(angles)),
        "expectation_max_error": float(
            max(
                abs(np.vdot(actual, op @ actual) - np.vdot(expected, op @ expected))
                for op in (
                    np.array([[0, 1], [1, 0]]),
                    np.array([[0, -1j], [1j, 0]]),
                    np.diag([1, -1]),
                )
            )
        ),
        "scope": "corrected pure wire only; not general mixed collision channels",
    }
