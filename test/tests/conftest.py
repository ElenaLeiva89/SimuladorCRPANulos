from pathlib import Path

import numpy as np
import pytest

from crpa_sim.config import ScenarioConfig, SimulationConfig, JammerConfig
from crpa_sim.crpa_array import create_hexagonal_7element_geometry


@pytest.fixture
def simulation_lcmv():
    return SimulationConfig(
        nsimulations=1,
        gnssBand=3,
        maxPhaseNoise_deg=2.0,
        maxAmplNoise_dB=0.5,
        interferenceType=[1],
        algorithmType="lcmv",
    )


@pytest.fixture
def scenario(simulation_lcmv, tmp_path):
    cfg = ScenarioConfig(
        speed_of_light_m_s=299792458.0,
        num_elements=7,
        element_spacing_over_lambda=0.5,
        num_snapshots=512,
        noise_power_linear=1.0,
        desired_azimuth_deg=0.0,
        desired_elevation_deg=0.0,
        azimuth_scan_min_deg=-180.0,
        azimuth_scan_max_deg=180.0,
        azimuth_scan_step_deg=1.0,
        fixed_azimuth_cut_deg=0.0,
        elevation_scan_min_deg=-90.0,
        elevation_scan_max_deg=90.0,
        elevation_scan_step_deg=1.0,
        random_seed=12345,
        output_dir=str(tmp_path / "results"),
        diagonal_loading_factor=1e-3,
    )
    cfg.set_carrier_frequency_from_simulation(simulation_lcmv)
    return cfg


@pytest.fixture
def geometry(scenario):
    return create_hexagonal_7element_geometry(
        scenario.num_elements,
        scenario.element_spacing_m,
    )


@pytest.fixture
def jammer_tone():
    return JammerConfig(
        name="Jammer_1",
        azimuth_deg=40.0,
        elevation_deg=10.0,
        jnr_dB=30.0,
        signal_type="tone",
        normalized_frequency=0.05,
    )


@pytest.fixture
def jammer_gaussian():
    return JammerConfig(
        name="Jammer_Gaussian",
        azimuth_deg=-70.0,
        elevation_deg=20.0,
        jnr_dB=20.0,
        signal_type="complex_gaussian",
        normalized_frequency=0.0,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
