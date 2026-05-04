"""io_utils.py: carga/guardado de configuración, tablas, matrices y logs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import JammerConfig, ScenarioConfig, SimulationConfig


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_configuration_file(config_path: Path) -> tuple[SimulationConfig, ScenarioConfig, list[JammerConfig]]:
    with open(config_path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    simulation = SimulationConfig.from_dict(raw.get("simulation_config", {}))
    scenario = ScenarioConfig.from_dict(raw.get("scenario_config", {}))
    jammers = [JammerConfig.from_dict(item) for item in raw.get("jammer_list", [])]

    if scenario.carrier_frequency_hz is None:
        scenario.set_carrier_frequency_from_simulation(simulation)

    return simulation, scenario, jammers


def load_simulation_parameters(config_path: Path) -> tuple[SimulationConfig, ScenarioConfig, list[JammerConfig]]:
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró {config_path}")
    print(f"Cargando configuración desde: {config_path}")
    return load_configuration_file(config_path)


def save_configuration_copy(output_dir: Path, simulation: SimulationConfig, scenario: ScenarioConfig, jammers: list[JammerConfig]) -> None:
    data = {
        "simulation_config": asdict(simulation),
        "scenario_config": asdict(scenario),
        "jammer_list": [asdict(j) for j in jammers],
    }
    with open(output_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_complex_npz(output_path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(output_path, **arrays)


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


def print_generated_files(output_dir: Path) -> None:
    print("\nFicheros generados:")
    for file_path in sorted(output_dir.iterdir()):
        print(f"  - {file_path.name}")


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)
