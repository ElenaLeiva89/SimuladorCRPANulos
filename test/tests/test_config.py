import pytest

from crpa_sim.config import (
    GNSS_CARRIER_FREQUENCIES_HZ,
    JammerConfig,
    ScenarioConfig,
    SimulationConfig,
    normalize_gnss_band,
)


def test_normalize_gnss_band_accepts_numeric_and_text():
    assert normalize_gnss_band(1) == "E5"
    assert normalize_gnss_band(2) == "E6"
    assert normalize_gnss_band(3) == "E1"
    assert normalize_gnss_band("e1") == "E1"
    assert normalize_gnss_band(" E6 ") == "E6"


def test_simulation_config_post_init_counts_interferences_and_lowercases_algorithm():
    sim = SimulationConfig(
        nsimulations=10,
        gnssBand=3,
        maxPhaseNoise_deg=2.0,
        maxAmplNoise_dB=0.5,
        interferenceType=[1, 2],
        algorithmType="LCMV",
    )

    assert sim.numberInterferences == 2
    assert sim.algorithmType == "lcmv"
    assert sim.band_label == "E1"


def test_simulation_config_from_dict_casts_nsimulations():
    sim = SimulationConfig.from_dict(
        {
            "nsimulations": 1e4,
            "gnssBand": 3,
            "maxPhaseNoise_deg": 2.0,
            "maxAmplNoise_dB": 0.5,
            "interferenceType": [1],
            "algorithmType": "LCMV",
        }
    )

    assert isinstance(sim.nsimulations, int)
    assert sim.nsimulations == 10000


def test_scenario_sets_carrier_frequency_from_band(simulation_lcmv):
    scenario = ScenarioConfig(
        speed_of_light_m_s=299792458.0,
        num_elements=7,
        element_spacing_over_lambda=0.5,
        num_snapshots=256,
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
        random_seed=1,
        output_dir="results",
    )

    scenario.set_carrier_frequency_from_simulation(simulation_lcmv)

    assert scenario.carrier_frequency_hz == GNSS_CARRIER_FREQUENCIES_HZ["E1"]
    assert scenario.wavelength_m > 0.0
    assert scenario.element_spacing_m == pytest.approx(0.5 * scenario.wavelength_m)


def test_invalid_gnss_band_raises_error():
    sim = SimulationConfig(
        nsimulations=1,
        gnssBand="BAD",
        maxPhaseNoise_deg=0.0,
        maxAmplNoise_dB=0.0,
        interferenceType=[],
    )
    scenario = ScenarioConfig(
        speed_of_light_m_s=299792458.0,
        num_elements=7,
        element_spacing_over_lambda=0.5,
        num_snapshots=256,
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
        random_seed=1,
        output_dir="results",
    )

    with pytest.raises(ValueError):
        scenario.set_carrier_frequency_from_simulation(sim)


def test_jammer_config_lowercases_signal_type():
    jammer = JammerConfig(
        name="J1",
        azimuth_deg=0,
        elevation_deg=0,
        jnr_dB=10,
        signal_type="TONE",
    )

    assert jammer.signal_type == "tone"
