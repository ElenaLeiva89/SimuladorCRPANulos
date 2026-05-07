from dataclasses import replace

import numpy as np
import pytest
from crpa_sim.fft_tools import temporal_fft_snapshot_matrix
from crpa_sim.patterns import compute_2d_response_grid, compute_azimuth_response_cut, compute_elevation_response_cut, conventional_weights, evaluate_response_for_angles, make_scan_vectors

def test_fft_size_and_tone_detection(rng):
    """Verifica tamano de FFT y localizacion de un tono sintetico.

    Parametros:
        rng: Generador aleatorio determinista.
    """
    X = rng.standard_normal((7, 128)) + 1j * rng.standard_normal((7, 128))
    df = temporal_fft_snapshot_matrix(X, 1024.0, 512)
    assert len(df) == 512
    assert df["power_dB_normalized"].max() <= 1e-9
    n = np.arange(512)
    tone = np.exp(1j * 2*np.pi*128/1024*n)
    tone_df = temporal_fft_snapshot_matrix(np.tile(tone, (7,1)), 1024.0, 512)
    assert abs(tone_df.loc[tone_df["power_dB_normalized"].idxmax(), "frequency_hz"] - 128.0) <= 2.0
    default_df = temporal_fft_snapshot_matrix(X, 1024.0)
    assert len(default_df) == X.shape[1]
    with pytest.raises(ValueError):
        temporal_fft_snapshot_matrix(X, 1024.0, 0)

def test_patterns(project_config, element_positions_m):
    """Comprueba cortes, malla 2D y vectores de barrido de patrones.

    Parametros:
        project_config: Configuracion base de simulacion.
        element_positions_m: Posiciones XYZ del array.
    """
    w = conventional_weights(project_config, element_positions_m)
    az = np.array([-90.0, 0.0, 90.0])
    el = np.array([0.0, 45.0, 90.0])
    df = evaluate_response_for_angles(project_config, element_positions_m, w, az, np.array([90.0,90.0,90.0]))
    assert df["response_abs_normalized"].max() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        evaluate_response_for_angles(project_config, element_positions_m, w, np.array([0,1]), np.array([90]))
    raw_df = evaluate_response_for_angles(project_config, element_positions_m, w, np.array([0.0]), np.array([90.0]), normalize=False)
    assert raw_df.loc[0, "response_abs_normalized"] == pytest.approx(raw_df.loc[0, "response_abs"])
    assert list(compute_azimuth_response_cut(project_config, element_positions_m, w, az, 90)["elevation_deg"]) == [90,90,90]
    assert list(compute_elevation_response_cut(project_config, element_positions_m, w, el, 0)["azimuth_deg"]) == [0,0,0]
    grid = compute_2d_response_grid(project_config, element_positions_m, w, az, el)
    assert grid["response_abs"].shape == (3,3)
    assert grid["response_abs_normalized"].max() == pytest.approx(1.0)
    scan_az, scan_el = make_scan_vectors(project_config)
    assert scan_az[0] == project_config.scan.azimuth_scan_min_deg
    assert scan_el[-1] >= project_config.scan.elevation_scan_max_deg


def test_2d_grid_uses_non_ideal_steering_path(project_config, element_positions_m):
    """Comprueba la rama no vectorizada de malla 2D y sus errores.

    Parametros:
        project_config: Configuracion base de simulacion.
        element_positions_m: Posiciones XYZ del array.
    """
    measured = replace(project_config, array=replace(project_config.array, steering_model="measured"))
    w = conventional_weights(project_config, element_positions_m)
    with pytest.raises(NotImplementedError):
        compute_2d_response_grid(measured, element_positions_m, w, np.array([0.0]), np.array([90.0]))


def test_make_scan_vectors_includes_configured_upper_edge(project_config):
    """Comprueba que los vectores de scan alcanzan o superan el maximo.

    Parametros:
        project_config: Configuracion base de simulacion.
    """
    cfg = replace(
        project_config,
        scan=replace(
            project_config.scan,
            azimuth_scan_min_deg=0.0,
            azimuth_scan_max_deg=1.0,
            azimuth_scan_step_deg=0.3,
            elevation_scan_min_deg=0.0,
            elevation_scan_max_deg=1.0,
            elevation_scan_step_deg=0.4,
        ),
    )
    az, el = make_scan_vectors(cfg)
    assert az[0] == pytest.approx(0.0)
    assert az[-1] >= 1.0
    assert el[-1] >= 1.0
