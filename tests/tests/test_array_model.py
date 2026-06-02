from dataclasses import replace
import numpy as np
import pytest
from crpa_sim.array_model import create_crpa_geometry, direction_unit_vector, measured_steering_matrix_for_angles, steering_vector, steering_vector_ideal


MEASURED_STEERING_FILE = "data/crpa_measured_steering.csv"

def test_geometry_and_direction(project_config):
    """Verifica geometria hexagonal y conversion azimut/elevacion.

    Parametros:
        project_config: Configuracion base de simulacion.
    """
    pos = create_crpa_geometry(project_config.array, project_config.element_spacing_m)
    assert pos.shape == (7, 3)
    assert np.allclose(pos[0], [0, 0, 0])
    assert np.allclose(np.linalg.norm(pos[1:, :2], axis=1), project_config.element_spacing_m)
    assert np.linalg.norm(direction_unit_vector(40, 10)) == pytest.approx(1.0)
    u = direction_unit_vector(123, 90)
    assert np.allclose(u[:2], [0, 0], atol=1e-12)
    assert u[2] == pytest.approx(1.0)

def test_steering_vector_models(project_config, element_positions_m):
    """Comprueba steering ideal, steering medido y modelos invalidos.

    Parametros:
        project_config: Configuracion base de simulacion.
        element_positions_m: Posiciones XYZ del array.
    """
    ideal = replace(project_config, array=replace(project_config.array, steering_model="ideal", measured_steering_file=None))
    a = steering_vector_ideal(element_positions_m, 40, 10, project_config.signal.wavelength_m)
    assert a.shape == (7,)
    assert np.allclose(np.abs(a), 1.0)
    assert steering_vector(ideal, element_positions_m, 40, 10).shape == (7,)

    measured = replace(
        project_config,
        array=replace(project_config.array, steering_model="measured", measured_steering_file=MEASURED_STEERING_FILE),
    )
    measured_vector = steering_vector(measured, element_positions_m, 40, 10)
    assert measured_vector.shape == (7,)
    assert np.all(np.isfinite(measured_vector))

    measured_without_file = replace(
        project_config,
        array=replace(project_config.array, steering_model="measured", measured_steering_file=None),
    )
    with pytest.raises(ValueError, match="measured_steering_file"):
        steering_vector(measured_without_file, element_positions_m, 40, 10)

    bad = replace(project_config, array=replace(project_config.array, steering_model="bad"))
    with pytest.raises(ValueError):
        steering_vector(bad, element_positions_m, 40, 10)


def test_measured_steering_matrix_matches_scalar_vectors(project_config, element_positions_m):
    """Comprueba la ruta vectorizada de steering medido.

    Parametros:
        project_config: Configuracion base de simulacion.
        element_positions_m: Posiciones XYZ del array.
    """
    measured = replace(
        project_config,
        array=replace(project_config.array, steering_model="measured", measured_steering_file=MEASURED_STEERING_FILE),
    )
    az = np.array([0.0, 90.0, 180.0])
    el = np.array([0.0, 45.0, 90.0])

    matrix = measured_steering_matrix_for_angles(measured, az, el)
    scalar_vectors = np.vstack([
        steering_vector(measured, element_positions_m, az_i, el_i)
        for az_i, el_i in zip(az, el)
    ])

    assert matrix.shape == (3, measured.array.num_elements)
    np.testing.assert_allclose(matrix, scalar_vectors)
    with pytest.raises(ValueError, match="misma longitud"):
        measured_steering_matrix_for_angles(measured, np.array([0.0, 90.0]), np.array([0.0]))


def test_create_geometry_rejects_unsupported(project_config):
    """Comprueba que se rechazan geometria o numero de elementos no soportados.

    Parametros:
        project_config: Configuracion base de simulacion.
    """
    with pytest.raises(ValueError):
        create_crpa_geometry(replace(project_config.array, geometry="ula"), project_config.element_spacing_m)
    with pytest.raises(ValueError):
        create_crpa_geometry(replace(project_config.array, num_elements=6), project_config.element_spacing_m)


def test_direction_unit_vector_cardinal_cases():
    """Comprueba direcciones cardinales y periodicidad angular.

    Parametros:
        No recibe parametros.
    """
    assert np.allclose(direction_unit_vector(0, 0), [1, 0, 0], atol=1e-12)
    assert np.allclose(direction_unit_vector(90, 0), [0, 1, 0], atol=1e-12)
    assert np.allclose(direction_unit_vector(180, 0), [-1, 0, 0], atol=1e-12)
    assert np.allclose(direction_unit_vector(360, 0), direction_unit_vector(0, 0), atol=1e-12)
    assert np.allclose(direction_unit_vector(45, -90), [0, 0, -1], atol=1e-12)


def test_steering_vector_zero_phase_and_shape(project_config):
    """Verifica fase nula en posiciones coincidentes y forma del resultado.

    Parametros:
        project_config: Configuracion base de simulacion.
    """
    positions = np.zeros((3, 3))
    a = steering_vector_ideal(positions, 123.0, 45.0, project_config.signal.wavelength_m)
    assert a.shape == (3,)
    assert np.allclose(a, np.ones(3, dtype=complex))
