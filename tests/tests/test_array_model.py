from dataclasses import replace
import numpy as np
import pytest
from crpa_sim.array_model import create_crpa_geometry, direction_unit_vector, steering_vector, steering_vector_ideal

def test_geometry_and_direction(project_config):
    pos = create_crpa_geometry(project_config.array, project_config.element_spacing_m)
    assert pos.shape == (7, 3)
    assert np.allclose(pos[0], [0, 0, 0])
    assert np.allclose(np.linalg.norm(pos[1:, :2], axis=1), project_config.element_spacing_m)
    assert np.linalg.norm(direction_unit_vector(40, 10)) == pytest.approx(1.0)
    u = direction_unit_vector(123, 90)
    assert np.allclose(u[:2], [0, 0], atol=1e-12)
    assert u[2] == pytest.approx(1.0)

def test_steering_vector_models(project_config, element_positions_m):
    a = steering_vector_ideal(element_positions_m, 40, 10, project_config.signal.wavelength_m)
    assert a.shape == (7,)
    assert np.allclose(np.abs(a), 1.0)
    assert steering_vector(project_config, element_positions_m, 40, 10).shape == (7,)
    measured = replace(project_config, array=replace(project_config.array, steering_model="measured"))
    with pytest.raises(NotImplementedError):
        steering_vector(measured, element_positions_m, 40, 10)
    bad = replace(project_config, array=replace(project_config.array, steering_model="bad"))
    with pytest.raises(ValueError):
        steering_vector(bad, element_positions_m, 40, 10)

def test_create_geometry_rejects_unsupported(project_config):
    with pytest.raises(ValueError):
        create_crpa_geometry(replace(project_config.array, geometry="ula"), project_config.element_spacing_m)
