from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")

from crpa_sim.array_model import create_crpa_geometry
from crpa_sim.config import (
    ArrayConfig,
    BeamformingConfig,
    JammerConfig,
    JammerTemplate,
    NoiseConfig,
    OutputConfig,
    ProjectConfig,
    ScanConfig,
    SignalConfig,
    SimulationConfig,
)


@pytest.fixture(scope="session")
def base_config_path() -> Path:
    return ROOT / "input_config.json"


@pytest.fixture()
def base_config() -> ProjectConfig:
    """Configuracion base portable: ideal, sin depender de MAT locales."""
    signal = SignalConfig("E1", 299792458.0, 64e6, 4096, 8192)
    return ProjectConfig(
        array=ArrayConfig(
            num_elements=7,
            geometry="hexagonal_7",
            element_type="isotropic",
            element_spacing_m=0.5 * signal.wavelength_m,
            array_boresight_elevation_deg=90.0,
            steering_model="ideal",
        ),
        signal=signal,
        simulation=SimulationConfig(num_montecarlo=10, doa_mode="fixed"),
        beamforming=BeamformingConfig(
            algorithm="lcmv",
            desired_azimuth_deg=0.0,
            desired_elevation_deg=90.0,
            diagonal_loading_factor=1e-3,
            power_inversion_reference_element=0,
        ),
        scan=ScanConfig(
            azimuth_scan_min_deg=0.0,
            azimuth_scan_max_deg=360.0,
            azimuth_scan_step_deg=0.5,
            elevation_scan_min_deg=0.0,
            elevation_scan_max_deg=90.0,
            elevation_scan_step_deg=0.5,
            null_thresholds_dB=[-10.0, -20.0, -30.0, -40.0, -50.0],
        ),
        noise=NoiseConfig(noise_power_linear=1.0),
        jammer=JammerConfig(
            num_jammers=1,
            jnr_dB=20.0,
            variable_doa_azimuth_range_deg=(0.0, 360.0),
            variable_doa_elevation_range_deg=(0.0, 90.0),
            base_jammers=[
                JammerTemplate("Jammer_1", 40.0, 60.0, "tone", None, 0.05, None, 0.05),
                JammerTemplate("Jammer_2", 150.0, 30.0, "tone", None, 0.11, None, 0.06),
                JammerTemplate("Jammer_3", 330.0, 80.0, "tone", None, 0.00, None, 0.07),
                JammerTemplate("Jammer_4", 270.0, 10.0, "tone", None, 0.17, None, 0.08),
                JammerTemplate("Jammer_5", 140.0, 50.0, "tone", None, 0.30, None, 0.09),
                JammerTemplate("Jammer_6", 210.0, 60.0, "tone", None, 0.23, None, 0.10),
            ],
        ),
        output=OutputConfig(
            output_dir="results_{algorithm}_{doa_mode}_{steering_model}",
            save_csv=True,
            save_npz=True,
            save_plots=True,
            csv_separator=";",
            csv_decimal=",",
        ),
    )


@pytest.fixture()
def project_config(base_config: ProjectConfig, tmp_path: Path) -> ProjectConfig:
    """Configuracion base compatible con los tests historicos.

    Parametros:
        base_config: Configuracion cargada desde input_config.json.
        tmp_path: Directorio temporal de pytest para aislar salidas.
    """
    return replace(
        base_config,
        signal=replace(base_config.signal, num_snapshots=256, fft_size=512),
        simulation=replace(base_config.simulation, num_montecarlo=1, doa_mode="fixed"),
        beamforming=replace(
            base_config.beamforming,
            algorithm="lcmv",
            desired_azimuth_deg=0.0,
            desired_elevation_deg=90.0,
            diagonal_loading_factor=1e-3,
            power_inversion_reference_element=0,
        ),
        scan=replace(
            base_config.scan,
            azimuth_scan_min_deg=0.0,
            azimuth_scan_max_deg=360.0,
            azimuth_scan_step_deg=10.0,
            elevation_scan_min_deg=0.0,
            elevation_scan_max_deg=90.0,
            elevation_scan_step_deg=10.0,
            null_thresholds_dB=[-10.0, -20.0, -30.0],
        ),
        jammer=replace(base_config.jammer, num_jammers=2, jnr_dB=30.0, base_jammers=base_config.jammer.base_jammers[:3]),
        output=replace(base_config.output, output_dir=str(tmp_path / "results"), save_csv=True, save_npz=True, save_plots=False),
    )


@pytest.fixture()
def variable_project_config(project_config: ProjectConfig) -> ProjectConfig:
    """Configuracion base con doa_mode variable.

    Parametros:
        project_config: Configuracion base compatible.
    """
    return replace(project_config, simulation=replace(project_config.simulation, doa_mode="variable"))


@pytest.fixture()
def power_inversion_config(project_config: ProjectConfig) -> ProjectConfig:
    """Configuracion base con algoritmo Power Inversion.

    Parametros:
        project_config: Configuracion base compatible.
    """
    return replace(project_config, beamforming=replace(project_config.beamforming, algorithm="power_inversion"))


@pytest.fixture()
def element_positions_m(project_config: ProjectConfig) -> np.ndarray:
    """Posiciones XYZ del array para la configuracion base.

    Parametros:
        project_config: Configuracion base compatible.
    """
    return create_crpa_geometry(project_config.array, project_config.element_spacing_m)


@pytest.fixture()
def jammer_list(project_config: ProjectConfig, rng: np.random.Generator):
    """Lista de jammers para la configuracion base.

    Parametros:
        project_config: Configuracion base compatible.
        rng: Generador aleatorio determinista.
    """
    from crpa_sim.jammers import build_jammer_case

    return build_jammer_case(project_config, rng)


@pytest.fixture()
def config_json_path(project_config: ProjectConfig, tmp_path: Path) -> Path:
    """Escribe un JSON temporal compatible con load_project_config.

    Parametros:
        project_config: Configuracion base compatible.
        tmp_path: Directorio temporal de pytest.
    """
    return write_config_json(project_config, tmp_path / "input_config.json")


@pytest.fixture()
def fast_config(base_config: ProjectConfig, tmp_path: Path) -> ProjectConfig:
    """Configuracion ligera para tests automaticos: sin plots y con pocos snapshots."""
    return replace(
        base_config,
        signal=replace(base_config.signal, num_snapshots=512, fft_size=512),
        simulation=replace(base_config.simulation, num_montecarlo=2),
        scan=replace(
            base_config.scan,
            azimuth_scan_step_deg=2.0,
            elevation_scan_step_deg=2.0,
            null_thresholds_dB=[-10.0, -20.0, -30.0],
        ),
        output=replace(
            base_config.output,
            output_dir=str(tmp_path / "results"),
            save_csv=True,
            save_npz=True,
            save_plots=False,
        ),
    )


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(123456)


def write_config_json(config: ProjectConfig, path: Path) -> Path:
    """Escribe ProjectConfig como JSON con las claves esperadas por load_project_config."""
    d = asdict(config)
    raw = {
        "array_config": d["array"],
        "signal_config": d["signal"],
        "simulation_config": d["simulation"],
        "beamforming_config": d["beamforming"],
        "scan_config": d["scan"],
        "noise_config": d["noise"],
        "jammer_config": d["jammer"],
        "output_config": d["output"],
    }
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return path


def write_synthetic_measured_mat_files(tmp_path: Path) -> tuple[Path, Path]:
    """Crea dos MAT minimos compatibles con measured_mat.py."""
    phase_path = tmp_path / "TABLASFASE_E1_C_TEST.mat"
    amplitude_path = tmp_path / "TABLASAMPL_E1_C_TEST.mat"

    azimuths = np.array([0.0, 90.0, 180.0])
    elevations = np.array([0.0, 45.0])
    frequencies_mhz = np.array([1560.0, 1575.42, 1590.0, 1600.0])
    table_shape = (azimuths.size, elevations.size, 12, frequencies_mhz.size)

    savemat(
        phase_path,
        {
            "TablasAOAFase": np.zeros(table_shape, dtype=np.int16),
            "FRECSTAB_MHz": frequencies_mhz,
            "AOAsTab_Grad": azimuths,
            "ELEVSTAB_GRAD": elevations,
            "NBITSFASE": np.array([[12]]),
            "NBITSREGI": np.array([[16]]),
        },
    )
    savemat(
        amplitude_path,
        {
            "TablasAOAAmpli": np.zeros(table_shape, dtype=np.int16),
            "FRECSTAB_MHz": frequencies_mhz,
            "AOAsTab_Grad": azimuths,
            "ELEVSTAB_GRAD": elevations,
            "MINDIFPA": np.array([[0.0]]),
            "PASODIFAMP": np.array([[1.0]]),
        },
    )

    return phase_path, amplitude_path


@pytest.fixture()
def measured_mat_paths(tmp_path: Path) -> tuple[Path, Path]:
    return write_synthetic_measured_mat_files(tmp_path)


@pytest.fixture()
def measured_project_config(project_config: ProjectConfig, measured_mat_paths: tuple[Path, Path]) -> ProjectConfig:
    phase_path, amplitude_path = measured_mat_paths
    return replace(
        project_config,
        array=replace(
            project_config.array,
            steering_model="measured",
            measured_phase_mat_file=str(phase_path),
            measured_amplitude_mat_file=str(amplitude_path),
        ),
    )
