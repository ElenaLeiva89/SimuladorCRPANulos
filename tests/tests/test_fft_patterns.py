import numpy as np
import pytest
from crpa_sim.fft_tools import temporal_fft_snapshot_matrix
from crpa_sim.patterns import compute_2d_response_grid, compute_azimuth_response_cut, compute_elevation_response_cut, conventional_weights, evaluate_response_for_angles, make_scan_vectors

def test_fft_size_and_tone_detection(rng):
    X = rng.standard_normal((7, 128)) + 1j * rng.standard_normal((7, 128))
    df = temporal_fft_snapshot_matrix(X, 1024.0, 512)
    assert len(df) == 512
    assert df["power_dB_normalized"].max() <= 1e-9
    n = np.arange(512)
    tone = np.exp(1j * 2*np.pi*128/1024*n)
    tone_df = temporal_fft_snapshot_matrix(np.tile(tone, (7,1)), 1024.0, 512)
    assert abs(tone_df.loc[tone_df["power_dB_normalized"].idxmax(), "frequency_hz"] - 128.0) <= 2.0

def test_patterns(project_config, element_positions_m):
    w = conventional_weights(project_config, element_positions_m)
    az = np.array([-90.0, 0.0, 90.0])
    el = np.array([0.0, 45.0, 90.0])
    df = evaluate_response_for_angles(project_config, element_positions_m, w, az, np.array([90.0,90.0,90.0]))
    assert df["response_abs_normalized"].max() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        evaluate_response_for_angles(project_config, element_positions_m, w, np.array([0,1]), np.array([90]))
    assert list(compute_azimuth_response_cut(project_config, element_positions_m, w, az, 90)["elevation_deg"]) == [90,90,90]
    assert list(compute_elevation_response_cut(project_config, element_positions_m, w, el, 0)["azimuth_deg"]) == [0,0,0]
    grid = compute_2d_response_grid(project_config, element_positions_m, w, az, el)
    assert grid["response_abs"].shape == (3,3)
    assert grid["response_abs_normalized"].max() == pytest.approx(1.0)
    scan_az, scan_el = make_scan_vectors(project_config)
    assert scan_az[0] == project_config.scan.azimuth_scan_min_deg
    assert scan_el[-1] >= project_config.scan.elevation_scan_max_deg
