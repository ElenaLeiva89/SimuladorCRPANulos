"""io_utils.py
Carga de configuracion, guardado de tablas, matrices y logs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ArrayConfig,
    BeamformingConfig,
    JammerConfig,
    NoiseConfig,
    OutputConfig,
    ProjectConfig,
    ScanConfig,
    SignalConfig,
    SimulationConfig,
)


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Crea, si hace falta, el directorio de salida y devuelve su Path.

    Parametros:
        output_dir: Ruta de salida como texto o Path.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_project_config(config_path: str | Path) -> ProjectConfig:
    """Carga y valida el JSON de configuracion del proyecto.

    Parametros:
        config_path: Ruta del fichero JSON de entrada.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontro el fichero de configuracion: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    config = ProjectConfig(
        array=ArrayConfig.from_dict(raw["array_config"]),
        signal=SignalConfig.from_dict(raw["signal_config"]),
        simulation=SimulationConfig.from_dict(raw["simulation_config"]),
        beamforming=BeamformingConfig.from_dict(raw["beamforming_config"]),
        scan=ScanConfig.from_dict(raw["scan_config"]),
        noise=NoiseConfig.from_dict(raw["noise_config"]),
        jammer=JammerConfig.from_dict(raw["jammer_config"]),
        output=OutputConfig.from_dict(raw["output_config"]),
    )
    validate_project_config(config)
    return config


def validate_project_config(config: ProjectConfig) -> None:
    """Comprueba restricciones globales del simulador.

    Parametros:
        config: Configuracion completa ya parseada desde JSON.
    """
    if config.array.num_elements != 7 or config.array.geometry != "hexagonal_7":
        raise ValueError("Esta version implementa una CRPA hexagonal de 7 elementos.")
    if config.jammer.num_jammers < 1:
        raise ValueError("num_jammers debe ser >= 1.")
    if config.jammer.num_jammers > config.array.num_elements - 1:
        raise ValueError("num_jammers debe ser <= num_elements - 1.")
    if config.jammer.num_jammers > len(config.jammer.base_jammers):
        raise ValueError("num_jammers supera el numero de base_jammers definidos.")
    if abs(config.array.array_boresight_elevation_deg - 90.0) > 1e-9:
        raise ValueError("Para este ejercicio la CRPA ideal debe apuntar al cenit: array_boresight_elevation_deg=90.")


def save_config_used(config: ProjectConfig, output_dir: Path) -> None:
    """Guarda una copia JSON de la configuracion usada.

    Parametros:
        config: Configuracion completa del proyecto.
        output_dir: Directorio donde se escribira config_used.json.
    """
    with open(output_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=4)


def save_dataframe(df: pd.DataFrame, path: Path, sep: str = ";", decimal: str = ",") -> None:
    """Guarda un DataFrame como CSV con separador/decimal configurables.

    Parametros:
        df: Tabla que se desea exportar.
        path: Ruta del fichero CSV de salida.
        sep: Separador de columnas.
        decimal: Caracter decimal para valores numericos.
    """
    df.to_csv(path, index=False, sep=sep, decimal=decimal)


def save_complex_npz(path: Path, **arrays: np.ndarray) -> None:
    """Guarda matrices, incluidas complejas, en un NPZ comprimido.

    Parametros:
        path: Ruta del fichero .npz de salida.
        **arrays: Arrays nombrados que se escriben dentro del contenedor.
    """
    np.savez_compressed(path, **arrays)


def save_run_log(config: ProjectConfig, output_dir: Path, extra_rows: list[dict] | None = None) -> None:
    """Escribe un log textual con configuracion y resumen de ejecucion.

    Parametros:
        config: Configuracion completa del proyecto.
        output_dir: Directorio donde se escribira run_log.txt.
        extra_rows: Filas opcionales con informacion adicional de la corrida.
    """
    with open(output_dir / "run_log.txt", "w", encoding="utf-8") as f:
        f.write("SIMULACION CRPA NULLFORMING / BEAMFORMING\n")
        f.write("========================================\n\n")
        f.write(f"Elementos CRPA: {config.array.num_elements}\n")
        f.write(f"Geometria: {config.array.geometry}\n")
        f.write(f"Tipo elemento: {config.array.element_type}\n")
        f.write(f"Modelo steering: {config.array.steering_model}\n")
        f.write(f"Boresight elevacion: {config.array.array_boresight_elevation_deg} deg\n")
        f.write(f"Banda GNSS: {config.signal.band_label}\n")
        f.write(f"Frecuencia portadora: {config.signal.carrier_frequency_hz} Hz\n")
        f.write(f"Longitud de onda: {config.signal.wavelength_m} m\n")
        f.write(f"Separacion radial: {config.element_spacing_m} m\n")
        f.write(f"Sample rate: {config.signal.sample_rate_hz} Hz\n")
        f.write(f"Snapshots: {config.signal.num_snapshots}\n")
        f.write(f"FFT size: {config.signal.fft_size}\n")
        f.write(f"Monte Carlo: {config.simulation.num_montecarlo}\n")
        f.write(f"DoA mode: {config.simulation.doa_mode}\n")
        f.write(f"Algoritmo: {config.beamforming.algorithm}\n")
        f.write(f"Direccion deseada: az={config.beamforming.desired_azimuth_deg} deg, el={config.beamforming.desired_elevation_deg} deg\n")
        f.write(f"Num jammers: {config.jammer.num_jammers}\n")
        f.write(f"JNR: {config.jammer.jnr_dB} dB\n")
        f.write(f"Umbrales nulo: {config.scan.null_thresholds_dB}\n")
        f.write(f"Directorio salida: {output_dir}\n\n")
        if extra_rows:
            f.write("Resumen adicional:\n")
            for row in extra_rows:
                f.write(f"- {row}\n")


def print_generated_files(output_dir: Path) -> None:
    """Imprime por consola los ficheros generados en el directorio de salida.

    Parametros:
        output_dir: Directorio cuyos hijos directos se listan.
    """
    print("\nFicheros generados:")
    for p in sorted(output_dir.iterdir()):
        print(f"  - {p.name}")
