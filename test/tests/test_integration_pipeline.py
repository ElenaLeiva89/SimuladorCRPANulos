import json
from pathlib import Path

import numpy as np
import pytest

from main import run_simulation

from crpa_sim.beamformers import compute_weights
from crpa_sim.crpa_array import compute_null_depth_dB
from crpa_sim.jammers import generate_received_snapshot_matrix


def test_algorithmic_pipeline_lcmv_generates_finite_null_depth(scenario, geometry, jammer_tone, rng):
    X, _ = generate_received_snapshot_matrix(
        scenario,
        [jammer_tone],
        geometry,
        rng,
    )

    w = compute_weights(
        "lcmv",
        X,
        geometry,
        scenario,
        [jammer_tone],
    )

    depth = compute_null_depth_dB(
        geometry,
        w,
        scenario.wavelength_m,
        jammer_tone.azimuth_deg,
        jammer_tone.elevation_deg,
    )

    assert np.isfinite(depth)
    assert depth < -80.0


def test_run_simulation_creates_expected_plot_and_log_files(tmp_path, monkeypatch):
    config_path = tmp_path / "input_config.json"
    output_dir = tmp_path / "results"

    config_path.write_text(
        json.dumps(
            {
                "simulation_config": {
                    "nsimulations": 1,
                    "gnssBand": 3,
                    "maxPhaseNoise_deg": 2.0,
                    "maxAmplNoise_dB": 0.5,
                    "interferenceType": [1],
                    "algorithmType": "lcmv",
                },
                "scenario_config": {
                    "speed_of_light_m_s": 299792458.0,
                    "num_elements": 7,
                    "element_spacing_over_lambda": 0.5,
                    "num_snapshots": 256,
                    "noise_power_linear": 1.0,
                    "desired_azimuth_deg": 0.0,
                    "desired_elevation_deg": 0.0,
                    "azimuth_scan_min_deg": -180.0,
                    "azimuth_scan_max_deg": 180.0,
                    "azimuth_scan_step_deg": 5.0,
                    "fixed_azimuth_cut_deg": 0.0,
                    "elevation_scan_min_deg": -90.0,
                    "elevation_scan_max_deg": 90.0,
                    "elevation_scan_step_deg": 5.0,
                    "random_seed": 12345,
                    "output_dir": str(output_dir),
                    "diagonal_loading_factor": 0.001,
                },
                "jammer_list": [
                    {
                        "name": "Jammer_1",
                        "azimuth_deg": 40.0,
                        "elevation_deg": 10.0,
                        "jnr_dB": 30.0,
                        "signal_type": "tone",
                        "normalized_frequency": 0.05,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    run_simulation(config_path)

    assert (output_dir / "run_log.txt").exists()
    assert (output_dir / "array_geometry.png").exists()
    assert (output_dir / "pattern_azimuth_dB.png").exists()
    assert (output_dir / "pattern_elevation_dB.png").exists()
    assert (output_dir / "temporal_fft_snapshot_spectrum.png").exists()
