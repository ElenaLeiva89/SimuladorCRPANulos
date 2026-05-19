from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")

from crpa_sim.array_model import create_crpa_geometry
from crpa_sim.config import ProjectConfig
from crpa_sim.io_utils import load_project_config


@pytest.fixture(scope="session")
def base_config_path() -> Path:
    return ROOT / "input_config.json"


@pytest.fixture()
def base_config(base_config_path: Path) -> ProjectConfig:
    return load_project_config(base_config_path)


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
        simulation=replace(base_config.simulation, num_montecarlo=1, random_seed=12345, doa_mode="fixed"),
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
            azimuth_scan_min_deg=-180.0,
            azimuth_scan_max_deg=180.0,
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
    """Configuracion ligera para tests automáticos: sin plots y con pocos snapshots."""
    return replace(
        base_config,
        signal=replace(base_config.signal, num_snapshots=512, fft_size=512),
        simulation=replace(base_config.simulation, num_montecarlo=2, random_seed=20240507),
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
