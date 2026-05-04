import numpy as np
import pytest

from crpa_sim.beamformers import (
    compute_lcmv_weights,
    compute_power_inversion_weights,
    compute_weights,
)
from crpa_sim.crpa_array import steering_vector_ideal
from crpa_sim.jammers import generate_received_snapshot_matrix


def test_power_inversion_weights_shape_and_finite(scenario, geometry, jammer_tone, rng):
    X, _ = generate_received_snapshot_matrix(
        scenario,
        [jammer_tone],
        geometry,
        rng,
    )

    w = compute_power_inversion_weights(X)

    assert w.shape == (scenario.num_elements,)
    assert np.all(np.isfinite(w))


def test_power_inversion_reference_constraint(scenario, geometry, jammer_tone, rng):
    X, _ = generate_received_snapshot_matrix(
        scenario,
        [jammer_tone],
        geometry,
        rng,
    )

    c = np.zeros(scenario.num_elements, dtype=complex)
    c[0] = 1.0
    w = compute_power_inversion_weights(X, reference_vector=c)

    assert np.vdot(c, w) == pytest.approx(1.0 + 0.0j)


def test_lcmv_weights_shape_and_finite(scenario, geometry, jammer_tone, rng):
    X, _ = generate_received_snapshot_matrix(
        scenario,
        [jammer_tone],
        geometry,
        rng,
    )

    w = compute_lcmv_weights(X, geometry, scenario, [jammer_tone])

    assert w.shape == (scenario.num_elements,)
    assert np.all(np.isfinite(w))


def test_lcmv_approximately_enforces_desired_constraint(scenario, geometry, jammer_tone, rng):
    X, _ = generate_received_snapshot_matrix(
        scenario,
        [jammer_tone],
        geometry,
        rng,
    )

    w = compute_lcmv_weights(X, geometry, scenario, [jammer_tone])
    a_des = steering_vector_ideal(
        geometry,
        scenario.desired_azimuth_deg,
        scenario.desired_elevation_deg,
        scenario.wavelength_m,
    )

    assert np.vdot(w, a_des) == pytest.approx(1.0 + 0.0j, abs=1e-6)


def test_lcmv_places_hard_null_on_jammer_direction(scenario, geometry, jammer_tone, rng):
    X, _ = generate_received_snapshot_matrix(
        scenario,
        [jammer_tone],
        geometry,
        rng,
    )

    w = compute_lcmv_weights(X, geometry, scenario, [jammer_tone])
    a_jam = steering_vector_ideal(
        geometry,
        jammer_tone.azimuth_deg,
        jammer_tone.elevation_deg,
        scenario.wavelength_m,
    )

    assert abs(np.vdot(w, a_jam)) < 1e-6


def test_lcmv_without_hard_nulls_only_enforces_desired_direction(scenario, geometry, jammer_tone, rng):
    X, _ = generate_received_snapshot_matrix(
        scenario,
        [jammer_tone],
        geometry,
        rng,
    )

    w = compute_lcmv_weights(
        X,
        geometry,
        scenario,
        [jammer_tone],
        include_hard_nulls=False,
    )
    a_des = steering_vector_ideal(
        geometry,
        scenario.desired_azimuth_deg,
        scenario.desired_elevation_deg,
        scenario.wavelength_m,
    )

    assert np.vdot(w, a_des) == pytest.approx(1.0 + 0.0j, abs=1e-6)


def test_compute_weights_selector_conventional(scenario, geometry, jammer_tone, rng):
    X, _ = generate_received_snapshot_matrix(scenario, [jammer_tone], geometry, rng)

    w = compute_weights("conventional", X, geometry, scenario, [jammer_tone])

    assert w.shape == (scenario.num_elements,)


def test_compute_weights_selector_lcmw_alias(scenario, geometry, jammer_tone, rng):
    X, _ = generate_received_snapshot_matrix(scenario, [jammer_tone], geometry, rng)

    w = compute_weights("lcmw", X, geometry, scenario, [jammer_tone])

    assert w.shape == (scenario.num_elements,)


def test_compute_weights_selector_power_inversion(scenario, geometry, jammer_tone, rng):
    X, _ = generate_received_snapshot_matrix(scenario, [jammer_tone], geometry, rng)

    w = compute_weights("power_inversion", X, geometry, scenario, [jammer_tone])

    assert w.shape == (scenario.num_elements,)


def test_compute_weights_rejects_unknown_algorithm(scenario, geometry, jammer_tone, rng):
    X, _ = generate_received_snapshot_matrix(scenario, [jammer_tone], geometry, rng)

    with pytest.raises(ValueError):
        compute_weights("bad_algorithm", X, geometry, scenario, [jammer_tone])
