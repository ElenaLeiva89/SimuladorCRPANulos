"""io_utils.py: carga/guardado de configuración, tablas, matrices y logs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import JammerConfig, ScenarioConfig, SimulationConfig

# Crea el directorio de salida si no existe y devuelve el Path resultante.
    # Parameters
    # ----------
    # output_dir:
    #     Ruta del directorio de salida.

    # Returns
    # -------
    # Path
    #     Objeto Path del directorio creado o existente.
def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

# Lee el JSON de configuración y crea los objetos SimulationConfig, ScenarioConfig y JammerConfig.
    # El archivo debe tener las claves:
    # - simulation_config
    # - scenario_config
    # - jammer_list

    # Returns
    # -------
    # tuple[SimulationConfig, ScenarioConfig, list[JammerConfig]]
    #     Objetos construidos a partir del JSON.
def load_configuration_file(config_path: Path) -> tuple[SimulationConfig, ScenarioConfig, list[JammerConfig]]:
    with open(config_path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    simulation = SimulationConfig.from_dict(raw.get("simulation_config", {}))
    scenario = ScenarioConfig.from_dict(raw.get("scenario_config", {}))
    jammers = [JammerConfig.from_dict(item) for item in raw.get("jammer_list", [])]

    if scenario.carrier_frequency_hz is None:
        scenario.set_carrier_frequency_from_simulation(simulation)

    return simulation, scenario, jammers

# Valida que exista el fichero de configuración e invoca la carga de parámetros.
    # Parameters
    # ----------
    # config_path:
    #     Ruta al archivo de configuración JSON.
def load_simulation_parameters(config_path: Path) -> tuple[SimulationConfig, ScenarioConfig, list[JammerConfig]]:
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró {config_path}")
    print(f"Cargando configuración desde: {config_path}")
    return load_configuration_file(config_path)

# Guarda matrices complejas en formato comprimido npz.
    # Parameters
    # ----------
    # output_path:
    #     Ruta del archivo de salida .npz.  
def save_complex_npz(output_path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(output_path, **arrays)

# Escribe un log de ejecución con los parámetros de simulación y el directorio de salida.
    # Parameters
    # ----------
    # simulation:
    #     Configuración de la simulación.
    # config:
    #     Configuración del escenario.
    # output_dir:
    #     Carpeta donde se escribe el log.
    # num_jammers:
    #     Número de jammers incluidos en la simulación.
    # null_depth_rows:
    #     Lista de diccionarios con métricas de profundidad de nulo por jammer. 
def save_run_log(
    simulation: SimulationConfig,
    scenario: ScenarioConfig,
    output_dir: Path,
    num_jammers: int,
    null_depth_rows: list[dict],
) -> None:
    with open(output_dir / "run_log.txt", "w", encoding="utf-8") as file:
        file.write("SIMULACIÓN CRPA 7 ELEMENTOS\n")
        file.write("==========================\n\n")
        file.write(f"Banda GNSS: {simulation.band_label}\n")
        file.write(f"Algoritmo: {simulation.algorithmType}\n")
        file.write(f"Frecuencia portadora Hz: {scenario.carrier_frequency_hz}\n")
        file.write(f"Longitud de onda m: {scenario.wavelength_m}\n")
        file.write(f"Separación radial m: {scenario.element_spacing_m}\n")
        file.write(f"Elementos: {scenario.num_elements}\n")
        file.write(f"Snapshots: {scenario.num_snapshots}\n")
        file.write(f"Potencia ruido lineal: {scenario.noise_power_linear}\n")
        file.write(f"Jammers: {num_jammers}\n")
        file.write(f"Directorio salida: {output_dir}\n\n")
        file.write("Profundidades de nulo / ganancia en jammer:\n")
        for row in null_depth_rows:
            file.write(f"- {row}\n")

# Imprime en consola la lista de archivos generados en el directorio de salida.
def print_generated_files(output_dir: Path) -> None:
    print("\nFicheros generados:")
    for file_path in sorted(output_dir.iterdir()):
        print(f"  - {file_path.name}")

# Guarda un DataFrame en CSV con formato específico (sin índice, separador ';', decimal ',').
def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False, sep=";", decimal=",",)
