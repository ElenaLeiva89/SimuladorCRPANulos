from __future__ import annotations

import numpy as np
import pytest

from crpa_sim.measured_mat import (
    MeasuredSteeringDatabase,
    load_measured_steering_database,
    measured_steering_vector,
    polarization_from_mat_filename,
)


def test_polarization_is_read_from_mat_filename():
    assert polarization_from_mat_filename("TABLASFASE_E1_C_LBADICIONALES_ALT.mat") == "C"
    assert polarization_from_mat_filename("TABLASAMPL_E6_V_LBADICIONALES_ALT.mat") == "V"
    assert polarization_from_mat_filename("TABLASFASE_E5_H_LBADICIONALES_ALT.mat") == "H"

    with pytest.raises(ValueError, match="polariz"):
        polarization_from_mat_filename("TABLASFASE_E1_X_TEST.mat")


def test_load_measured_steering_database_decodes_mat_files(measured_mat_paths):
    phase_path, amplitude_path = measured_mat_paths

    database = load_measured_steering_database(str(phase_path), str(amplitude_path))
    vector = measured_steering_vector(database, 90.0, 45.0, 1575.42e6)

    assert database.polarization == "C"
    assert database.phase_deg.shape == (3, 2, 6, 4)
    assert database.amplitude_dB.shape == (3, 2, 6, 4)
    np.testing.assert_allclose(database.frequencies_hz[1], 1575.42e6)
    np.testing.assert_allclose(vector, np.ones(7, dtype=complex), atol=1e-12)


def test_measured_steering_vector_interpolates_frequency_and_wraps_phase():
    phase_deg = np.zeros((2, 2, 6, 2), dtype=float)
    amplitude_dB = np.zeros_like(phase_deg)

    phase_deg[0, 0, :, 0] = 170.0
    phase_deg[0, 0, :, 1] = -170.0
    amplitude_dB[0, 0, :, 0] = 0.0
    amplitude_dB[0, 0, :, 1] = 6.0

    database = MeasuredSteeringDatabase(
        phase_deg=phase_deg,
        amplitude_dB=amplitude_dB,
        azimuths_deg=np.array([0.0, 90.0]),
        elevations_deg=np.array([10.0, 40.0]),
        frequencies_hz=np.array([100.0, 200.0]),
        polarization="C",
    )

    vector = measured_steering_vector(database, 359.0, 12.0, 150.0)
    expected_amplitude = 10.0 ** (3.0 / 20.0)

    assert vector[0] == pytest.approx(1.0 + 0.0j)
    np.testing.assert_allclose(vector[1:], -expected_amplitude + 0.0j, atol=1e-12)


def test_measured_steering_vector_rejects_non_positive_frequency(measured_mat_paths):
    phase_path, amplitude_path = measured_mat_paths
    database = load_measured_steering_database(str(phase_path), str(amplitude_path))

    with pytest.raises(ValueError, match="frequency_hz"):
        measured_steering_vector(database, 0.0, 0.0, 0.0)
