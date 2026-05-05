"""jammers.py
Generación de ruido, señales jammer y matriz de snapshots X.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np
import pandas as pd

from .array_model import steering_vector
from .config import JammerInstance, ProjectConfig


def jammer_power_from_jnr(noise_power_linear: float, jnr_dB: float) -> float:
    return noise_power_linear * 10.0 ** (jnr_dB / 10.0)


def generate_complex_noise(config: ProjectConfig, rng: np.random.Generator) -> np.ndarray:
    sigma = np.sqrt(config.noise.noise_power_linear / 2.0)
    shape = (config.array.num_elements, config.signal.num_snapshots)
    return sigma * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def generate_jammer_baseband_signal(jammer: JammerInstance, num_snapshots: int, jammer_power_linear: float, rng: np.random.Generator) -> np.ndarray:
    """Genera señal baseband compleja de un jammer."""
    if jammer.signal_type == "complex_gaussian":
        sigma = np.sqrt(jammer_power_linear / 2.0)
        return sigma * (rng.standard_normal(num_snapshots) + 1j * rng.standard_normal(num_snapshots))
    if jammer.signal_type == "tone":
        n = np.arange(num_snapshots)
        phase0 = rng.uniform(0.0, 2.0 * np.pi)
        amplitude = np.sqrt(jammer_power_linear)
        return amplitude * np.exp(1j * (2.0 * np.pi * jammer.normalized_frequency * n + phase0))
    raise ValueError(f"Tipo de jammer no soportado: {jammer.signal_type}")


def build_jammer_case(config: ProjectConfig, rng: np.random.Generator) -> list[JammerInstance]:
    """Construye la lista de jammers para una iteración.

    fixed: usa az/el de base_jammers.
    variable: sortea az/el en los rangos configurados.
    """
    az_min, az_max = config.jammer.variable_doa_azimuth_range_deg
    el_min, el_max = config.jammer.variable_doa_elevation_range_deg
    jammers: list[JammerInstance] = []

    for template in config.jammer.base_jammers[: config.jammer.num_jammers]:
        if config.simulation.doa_mode == "fixed":
            az = float(template.azimuth_deg)
            el = float(template.elevation_deg)
        elif config.simulation.doa_mode == "variable":
            az = float(rng.uniform(az_min, az_max))
            el = float(rng.uniform(el_min, el_max))
        else:
            raise ValueError(f"doa_mode no soportado: {config.simulation.doa_mode}")

        jammers.append(
            JammerInstance(
                name=template.name,
                azimuth_deg=az,
                elevation_deg=el,
                jnr_dB=config.jammer.jnr_dB,
                signal_type=template.signal_type,
                normalized_frequency=template.normalized_frequency,
                bandwidth_hz=template.bandwidth_hz,
            )
        )
    return jammers


def generate_received_snapshot_matrix(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    jammer_list: Sequence[JammerInstance],
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Genera X = ruido + suma_j a(az_j,el_j) s_j."""
    X = generate_complex_noise(config, rng)
    rows = []

    for idx, jammer in enumerate(jammer_list, start=1):
        p_jam = jammer_power_from_jnr(config.noise.noise_power_linear, jammer.jnr_dB)
        a_jam = steering_vector(config, element_positions_m, jammer.azimuth_deg, jammer.elevation_deg)
        s_jam = generate_jammer_baseband_signal(jammer, config.signal.num_snapshots, p_jam, rng)
        X += a_jam[:, None] * s_jam[None, :]

        row = asdict(jammer)
        row.update({"jammer_index": idx, "jammer_power_linear": p_jam})
        rows.append(row)

    return X, pd.DataFrame(rows)
