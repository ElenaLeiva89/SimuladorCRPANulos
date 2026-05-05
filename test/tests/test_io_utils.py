import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crpa_sim.io_utils import (
    ensure_output_dir,
    load_configuration_file,
    load_simulation_parameters,
    save_complex_npz,
    save_dataframe,
    save_run_log,
)


def test_ensure_output_dir_creates_directory(tmp_path):
    output_dir = tmp_path / "nested" / "results"
    result = ensure_output_dir(output_dir)

    assert result.exists()
    assert result.is_dir()


def test_load_simulation_parameters_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_simulation_parameters(tmp_path / "missing.json")


def test_load_configuration_file_parses_config(tmp_path):
    config_path = tmp_path / "input_config.json"
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
                    "num_snapshots": 128,
                    "noise_power_linear": 1.0,
                    "desired_azimuth_deg": 0.0,
                    "desired_elevation_deg": 0.0,
                    "azimuth_scan_min_deg": -180.0,
                    "azimuth_scan_max_deg": 180.0,
                    "azimuth_scan_step_deg": 1.0,
                    "fixed_azimuth_cut_deg": 0.0,
                    "elevation_scan_min_deg": -90.0,
                    "elevation_scan_max_deg": 90.0,
                    "elevation_scan_step_deg": 1.0,
                    "random_seed": 1,
                    "output_dir": "results",
                },
                "jammer_list": [
                    {
                        "name": "J1",
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

    simulation, scenario, jammers = load_configuration_file(config_path)

    assert simulation.algorithmType == "lcmv"
    assert scenario.carrier_frequency_hz is not None
    assert len(jammers) == 1


def test_save_helpers_create_files(tmp_path, simulation_lcmv, scenario, jammer_tone):
    output_dir = ensure_output_dir(tmp_path / "results")

    save_complex_npz(output_dir / "arrays.npz", x=np.ones(3))
    save_dataframe(pd.DataFrame({"a": [1, 2]}), output_dir / "table.csv")
    save_run_log(
        simulation_lcmv,
        scenario,
        output_dir,
        1,
        [{"jammer_name": "J1", "null_depth_dB_normalized": -40.0}],
    )

    assert (output_dir / "config_used.json").exists()
    assert (output_dir / "arrays.npz").exists()
    assert (output_dir / "table.csv").exists()
    assert (output_dir / "run_log.txt").exists()
