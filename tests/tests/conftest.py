import json
from dataclasses import replace
import numpy as np
import pytest
from crpa_sim.config import ArrayConfig, BeamformingConfig, JammerConfig, NoiseConfig, OutputConfig, ProjectConfig, ScanConfig, SignalConfig, SimulationConfig
from crpa_sim.array_model import create_crpa_geometry

@pytest.fixture
def project_config(tmp_path):
    return ProjectConfig(
        array=ArrayConfig(7, "hexagonal_7", "isotropic", 0.5, 90.0, "ideal"),
        signal=SignalConfig("E1", 299792458.0, 64_000_000.0, 256, 512),
        simulation=SimulationConfig(1, 12345, "fixed"),
        beamforming=BeamformingConfig("lcmv", 0.0, 90.0, 1e-3, 0),
        scan=ScanConfig(-180.0, 180.0, 10.0, 0.0, 90.0, 10.0, [-10.0, -20.0, -30.0]),
        noise=NoiseConfig(1.0),
        jammer=JammerConfig.from_dict({
            "num_jammers": 2,
            "jnr_dB": 30.0,
            "variable_doa_azimuth_range_deg": [-180.0, 180.0],
            "variable_doa_elevation_range_deg": [5.0, 85.0],
            "base_jammers": [
                {"name": "Jammer_1", "azimuth_deg": 40.0, "elevation_deg": 10.0, "signal_type": "tone", "normalized_frequency": 0.05},
                {"name": "Jammer_2", "azimuth_deg": 70.0, "elevation_deg": 30.0, "signal_type": "complex_gaussian", "normalized_frequency": 0.0},
                {"name": "Jammer_3", "azimuth_deg": 330.0, "elevation_deg": 80.0, "signal_type": "tone", "normalized_frequency": 0.11},
            ],
        }),
        output=OutputConfig(str(tmp_path / "results"), True, True, False, ";", ","),
    )

@pytest.fixture
def variable_project_config(project_config):
    return replace(project_config, simulation=replace(project_config.simulation, doa_mode="variable"))

@pytest.fixture
def power_inversion_config(project_config):
    return replace(project_config, beamforming=replace(project_config.beamforming, algorithm="power_inversion"))

@pytest.fixture
def element_positions_m(project_config):
    return create_crpa_geometry(project_config.array, project_config.element_spacing_m)

@pytest.fixture
def rng():
    return np.random.default_rng(12345)

@pytest.fixture
def jammer_list(project_config, rng):
    from crpa_sim.jammers import build_jammer_case
    return build_jammer_case(project_config, rng)

@pytest.fixture
def config_json_path(tmp_path):
    path = tmp_path / "input_config.json"
    data = {
        "array_config": {"num_elements": 7, "geometry": "hexagonal_7", "element_type": "isotropic", "element_spacing_over_lambda": 0.5, "array_boresight_elevation_deg": 90.0, "steering_model": "ideal"},
        "signal_config": {"gnss_band": "E1", "speed_of_light_m_s": 299792458.0, "sample_rate_hz": 64000000.0, "num_snapshots": 128, "fft_size": 256},
        "simulation_config": {"num_montecarlo": 1, "random_seed": 12345, "doa_mode": "fixed"},
        "beamforming_config": {"algorithm": "lcmv", "desired_azimuth_deg": 0.0, "desired_elevation_deg": 90.0, "diagonal_loading_factor": 0.001, "power_inversion_reference_element": 0},
        "scan_config": {"azimuth_scan_min_deg": -180.0, "azimuth_scan_max_deg": 180.0, "azimuth_scan_step_deg": 30.0, "elevation_scan_min_deg": 0.0, "elevation_scan_max_deg": 90.0, "elevation_scan_step_deg": 30.0, "null_thresholds_dB": [-10, -20]},
        "noise_config": {"noise_power_linear": 1.0},
        "jammer_config": {"num_jammers": 1, "jnr_dB": 30.0, "variable_doa_azimuth_range_deg": [-180.0, 180.0], "variable_doa_elevation_range_deg": [5.0, 85.0], "base_jammers": [{"name": "Jammer_1", "azimuth_deg": 40.0, "elevation_deg": 10.0, "signal_type": "tone", "normalized_frequency": 0.05}]},
        "output_config": {"output_dir": str(tmp_path / "results"), "save_csv": True, "save_npz": True, "save_plots": False, "csv_separator": ";", "csv_decimal": ","},
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
