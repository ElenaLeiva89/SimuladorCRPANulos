import pytest
from crpa_sim.config import ArrayConfig, BeamformingConfig, GNSS_CARRIER_FREQUENCIES_HZ, JammerConfig, JammerTemplate, SignalConfig, SimulationConfig, normalize_gnss_band

def test_normalize_gnss_band():
    assert normalize_gnss_band(1) == "E5"
    assert normalize_gnss_band(2) == "E6"
    assert normalize_gnss_band(3) == "E1"
    assert normalize_gnss_band(" e1 ") == "E1"

def test_from_dict_validation_and_properties(project_config):
    array = ArrayConfig.from_dict({"num_elements": "7", "geometry": "HEXAGONAL_7", "element_type": "ISOTROPIC", "element_spacing_over_lambda": 0.5, "array_boresight_elevation_deg": 90, "steering_model": "IDEAL"})
    assert array.geometry == "hexagonal_7"
    assert array.element_type == "isotropic"
    signal = SignalConfig("E1", 299792458.0, 64e6, 1024, 2048)
    assert signal.carrier_frequency_hz == GNSS_CARRIER_FREQUENCIES_HZ["E1"]
    assert signal.wavelength_m > 0
    assert project_config.element_spacing_m == pytest.approx(0.5 * project_config.signal.wavelength_m)

def test_invalid_band_doa_and_algorithm():
    with pytest.raises(ValueError):
        _ = SignalConfig("BAD", 299792458.0, 64e6, 1024).carrier_frequency_hz
    with pytest.raises(ValueError):
        SimulationConfig.from_dict({"num_montecarlo": 1, "random_seed": 1, "doa_mode": "bad"})
    with pytest.raises(ValueError):
        BeamformingConfig.from_dict({"algorithm": "bad", "desired_azimuth_deg": 0, "desired_elevation_deg": 90, "diagonal_loading_factor": 0.001})

def test_valid_config_parsing():
    sim = SimulationConfig.from_dict({"num_montecarlo": "2", "random_seed": "3", "doa_mode": "VARIABLE"})
    bf = BeamformingConfig.from_dict({"algorithm": "LCMV", "desired_azimuth_deg": 0, "desired_elevation_deg": 90, "diagonal_loading_factor": 0.001, "power_inversion_reference_element": "0"})
    jam = JammerConfig.from_dict({"num_jammers": "1", "jnr_dB": "40", "variable_doa_azimuth_range_deg": ["-180", "180"], "variable_doa_elevation_range_deg": ["5", "85"], "base_jammers": [{"name": "J1", "azimuth_deg": 40, "elevation_deg": 10, "signal_type": "TONE"}]})
    assert sim.doa_mode == "variable"
    assert bf.algorithm == "lcmv"
    assert jam.base_jammers[0].signal_type == "tone"
