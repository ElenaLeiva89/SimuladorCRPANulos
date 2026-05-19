from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest

from crpa_sim.array_model import create_crpa_geometry, direction_unit_vector, steering_vector
from crpa_sim.config import BeamformingConfig, SimulationConfig
from crpa_sim.io_utils import validate_project_config


def test_hexagonal_7_geometry_has_center_and_equal_radius(base_config):
    positions = create_crpa_geometry(base_config.array, base_config.element_spacing_m)

    assert positions.shape == (7, 3)
    np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(positions[:, 2], 0.0)
    radii = np.linalg.norm(positions[1:, :2], axis=1)
    np.testing.assert_allclose(radii, base_config.element_spacing_m, rtol=0, atol=1e-12)


def test_direction_unit_vector_is_unit_norm_and_zenith_points_z():
    u = direction_unit_vector(123.0, 45.0)
    assert np.isclose(np.linalg.norm(u), 1.0)
    np.testing.assert_allclose(direction_unit_vector(0.0, 90.0), [0.0, 0.0, 1.0], atol=1e-15)


def test_zenith_steering_vector_is_all_ones_for_planar_array(base_config):
    positions = create_crpa_geometry(base_config.array, base_config.element_spacing_m)
    a = steering_vector(base_config, positions, azimuth_deg=0.0, elevation_deg=90.0)
    np.testing.assert_allclose(a, np.ones(base_config.array.num_elements, dtype=complex), atol=1e-12)


def test_invalid_doa_mode_is_rejected():
    with pytest.raises(ValueError, match="doa_mode"):
        SimulationConfig.from_dict({"num_montecarlo": 1, "random_seed": 1, "doa_mode": "sweep"})


def test_invalid_algorithm_is_rejected():
    values = {
        "algorithm": "mvdr",
        "desired_azimuth_deg": 0.0,
        "desired_elevation_deg": 90.0,
        "diagonal_loading_factor": 1e-3,
        "power_inversion_reference_element": 0,
    }
    with pytest.raises(ValueError, match="algorithm"):
        BeamformingConfig.from_dict(values)


def test_num_jammers_cannot_exceed_degrees_of_freedom(base_config):
    bad = replace(base_config, jammer=replace(base_config.jammer, num_jammers=7))
    with pytest.raises(ValueError, match="num_jammers"):
        validate_project_config(bad)
