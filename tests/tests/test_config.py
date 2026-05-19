from dataclasses import replace

import pytest
from crpa_sim.config import ArrayConfig, BeamformingConfig, GNSS_CARRIER_FREQUENCIES_HZ, JammerConfig, JammerTemplate, NoiseConfig, OutputConfig, ScanConfig, SignalConfig, SimulationConfig, normalize_gnss_band
from crpa_sim.io_utils import validate_project_config

def test_normalize_gnss_band():
    """Comprueba la normalizacion de bandas GNSS.

    Parametros:
        No recibe parametros.
    """
    assert normalize_gnss_band(1) == "E5"
    assert normalize_gnss_band(2) == "E6"
    assert normalize_gnss_band(3) == "E1"
    assert normalize_gnss_band(" e1 ") == "E1"
    assert normalize_gnss_band(99) == "99"

def test_from_dict_validation_and_properties(project_config):
    """Valida parseo de configuracion y propiedades derivadas.

    Parametros:
        project_config: Configuracion base de simulacion.
    """
    array = ArrayConfig.from_dict({"num_elements": "7", "geometry": "HEXAGONAL_7", "element_type": "ISOTROPIC", "element_spacing_over_lambda": 0.5, "array_boresight_elevation_deg": 90, "steering_model": "IDEAL"})
    assert array.geometry == "hexagonal_7"
    assert array.element_type == "isotropic"
    signal = SignalConfig("E1", 299792458.0, 64e6, 1024, 2048)
    assert signal.carrier_frequency_hz == GNSS_CARRIER_FREQUENCIES_HZ["E1"]
    assert signal.wavelength_m > 0
    assert project_config.element_spacing_m == pytest.approx(0.5 * project_config.signal.wavelength_m)

def test_invalid_band_doa_and_algorithm():
    """Comprueba errores para banda, doa_mode y algoritmo invalidos.

    Parametros:
        No recibe parametros.
    """
    with pytest.raises(ValueError):
        _ = SignalConfig("BAD", 299792458.0, 64e6, 1024).carrier_frequency_hz
    with pytest.raises(ValueError):
        SimulationConfig.from_dict({"num_montecarlo": 1, "random_seed": 1, "doa_mode": "bad"})
    with pytest.raises(ValueError):
        BeamformingConfig.from_dict({"algorithm": "bad", "desired_azimuth_deg": 0, "desired_elevation_deg": 90, "diagonal_loading_factor": 0.001})

def test_valid_config_parsing():
    """Comprueba conversion de tipos y normalizacion desde diccionarios.

    Parametros:
        No recibe parametros.
    """
    sim = SimulationConfig.from_dict({"num_montecarlo": "2", "random_seed": "3", "doa_mode": "VARIABLE"})
    bf = BeamformingConfig.from_dict({"algorithm": "LCMV", "desired_azimuth_deg": 0, "desired_elevation_deg": 90, "diagonal_loading_factor": 0.001, "power_inversion_reference_element": "0"})
    jam = JammerConfig.from_dict({"num_jammers": "1", "jnr_dB": "40", "variable_doa_azimuth_range_deg": ["-180", "180"], "variable_doa_elevation_range_deg": ["5", "85"], "base_jammers": [{"name": "J1", "azimuth_deg": 40, "elevation_deg": 10, "signal_type": "TONE"}]})
    assert sim.doa_mode == "variable"
    assert bf.algorithm == "lcmv"
    assert jam.base_jammers[0].signal_type == "tone"


def test_from_dict_defaults_and_type_conversions():
    """Comprueba defaults opcionales y conversiones limite de from_dict.

    Parametros:
        No recibe parametros.
    """
    signal = SignalConfig.from_dict({"gnss_band": 3, "speed_of_light_m_s": 3e8, "sample_rate_hz": 1.0, "num_snapshots": "8", "fft_size": None})
    assert signal.band_label == "E1"
    assert signal.fft_size is None
    scan = ScanConfig.from_dict({
        "azimuth_scan_min_deg": 0,
        "azimuth_scan_max_deg": 1,
        "azimuth_scan_step_deg": 1,
        "elevation_scan_min_deg": 0,
        "elevation_scan_max_deg": 1,
        "elevation_scan_step_deg": 1,
    })
    assert scan.null_thresholds_dB == [-10.0, -20.0, -30.0, -40.0]
    template = JammerTemplate.from_dict({"name": "J", "azimuth_deg": 1, "elevation_deg": 2})
    assert template.signal_type == "complex_gaussian"
    assert template.normalized_frequency == 0.0
    assert NoiseConfig.from_dict({"noise_power_linear": 1.5}).noise_power_linear == 1.5
    assert OutputConfig.from_dict({"output_dir": "out"}).save_plots is True


def test_from_dict_missing_required_fields_raise_key_error():
    """Comprueba errores de campos obligatorios ausentes en configuracion.

    Parametros:
        No recibe parametros.
    """
    with pytest.raises(KeyError):
        ArrayConfig.from_dict({"geometry": "hexagonal_7"})
    with pytest.raises(KeyError):
        JammerConfig.from_dict({"num_jammers": 1, "jnr_dB": 30.0})


def test_validate_project_config_rejects_physical_invalid_ranges(project_config):
    """Comprueba rangos fisicos y numericos invalidos de configuracion global.

    Parametros:
        project_config: Configuracion base de simulacion.
    """
    invalid_configs = [
        replace(project_config, array=replace(project_config.array, element_spacing_over_lambda=0.0)),
        replace(project_config, signal=replace(project_config.signal, speed_of_light_m_s=0.0)),
        replace(project_config, signal=replace(project_config.signal, sample_rate_hz=0.0)),
        replace(project_config, signal=replace(project_config.signal, num_snapshots=0)),
        replace(project_config, signal=replace(project_config.signal, fft_size=0)),
        replace(project_config, simulation=replace(project_config.simulation, num_montecarlo=0)),
        replace(project_config, beamforming=replace(project_config.beamforming, diagonal_loading_factor=-1e-3)),
        replace(project_config, scan=replace(project_config.scan, azimuth_scan_step_deg=0.0)),
        replace(project_config, scan=replace(project_config.scan, elevation_scan_step_deg=0.0)),
        replace(project_config, scan=replace(project_config.scan, azimuth_scan_min_deg=10.0, azimuth_scan_max_deg=0.0)),
        replace(project_config, scan=replace(project_config.scan, elevation_scan_min_deg=10.0, elevation_scan_max_deg=0.0)),
        replace(project_config, noise=replace(project_config.noise, noise_power_linear=-1.0)),
    ]

    for config in invalid_configs:
        with pytest.raises(ValueError):
            validate_project_config(config)
