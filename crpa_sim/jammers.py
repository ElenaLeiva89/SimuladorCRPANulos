"""Generacion de ruido, jammers y matriz de snapshots.

Cada jammer se genera como una senal baseband temporal y se proyecta sobre
los elementos mediante su steering vector. La matriz recibida resultante es
X = ruido + suma_j a_j s_j, con forma (num_elements, num_snapshots).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np
import pandas as pd

from .array_model import steering_vector
from .config import JammerInstance, ProjectConfig


def jammer_power_from_jnr(noise_power_linear: float, jnr_dB: float) -> float:
    """Convierte JNR en dB a potencia lineal de jammer.

    Parametros:
        noise_power_linear: Potencia de ruido en escala lineal.
        jnr_dB: Relacion jammer-ruido en dB.
    """
    return noise_power_linear * 10.0 ** (jnr_dB / 10.0)


def generate_complex_noise(config: ProjectConfig, rng: np.random.Generator) -> np.ndarray:
    """Genera ruido complejo circular para todos los elementos del array.

    Parametros:
        config: Configuracion completa; aporta potencia de ruido, numero de
            elementos y numero de snapshots.
        rng: Generador aleatorio reproducible.
    """
    sigma = np.sqrt(config.noise.noise_power_linear / 2.0)
    shape = (config.array.num_elements, config.signal.num_snapshots)
    return sigma * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def generate_jammer_baseband_signal(jammer: JammerInstance, num_snapshots: int, jammer_power_linear: float, rng: np.random.Generator) -> np.ndarray:
    """Genera la senal baseband compleja de un jammer.

    Parametros:
        jammer: Instancia del jammer con tipo de senal y frecuencia.
        num_snapshots: Numero de muestras temporales a generar.
        jammer_power_linear: Potencia lineal deseada de la senal.
        rng: Generador aleatorio usado para ruido gaussiano y fase inicial.
    """
    if jammer.signal_type == "complex_gaussian":
        sigma = np.sqrt(jammer_power_linear / 2.0)
        return sigma * (rng.standard_normal(num_snapshots) + 1j * rng.standard_normal(num_snapshots))
    if jammer.signal_type == "tone":
        n = np.arange(num_snapshots)
        phase0 = rng.uniform(0.0, 2.0 * np.pi)
        amplitude = np.sqrt(jammer_power_linear)
        return amplitude * np.exp(1j * (2.0 * np.pi * jammer.normalized_frequency * n + phase0))
    if jammer.signal_type == "chirp":
        if jammer.chirp_frequency is None:
            raise ValueError("Los jammers de tipo chirp requieren chirp_frequency normalizada.")
        n = np.arange(num_snapshots)
        phase0 = rng.uniform(0.0, 2.0 * np.pi)
        amplitude = np.sqrt(jammer_power_linear)

        f0 = jammer.chirp_frequency
        chirp_bandwidth_norm = 0.10

        # Frecuencia inicial/final normalizada del chirp lineal.
        f_start = f0 - chirp_bandwidth_norm / 2.0
        f_end = f0 + chirp_bandwidth_norm / 2.0

        # Pendiente del chirp en ciclos por muestra^2.
        k = (f_end - f_start) / max(num_snapshots - 1, 1)

        phase = 2.0 * np.pi * (f_start * n + 0.5 * k * n**2) + phase0
        return amplitude * np.exp(1j * phase)
    raise ValueError(f"Tipo de jammer no soportado: {jammer.signal_type}")


def build_jammer_case(config: ProjectConfig, rng: np.random.Generator) -> list[JammerInstance]:
    """Construye la lista de jammers para una iteracion Monte Carlo.

    Parametros:
        config: Configuracion completa; define numero de jammers, JNR, modo
            DoA y plantillas base.
        rng: Generador aleatorio usado cuando doa_mode="variable".
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

        # Mantiene una convencion unica para tablas y plots aunque se construyan
        # configuraciones manuales con azimut equivalente negativo.
        az = az % 360.0

        jammers.append(
            JammerInstance(
                name=template.name,
                azimuth_deg=az,
                elevation_deg=el,
                jnr_dB=config.jammer.jnr_dB,
                signal_type=template.signal_type,
                normalized_frequency=template.normalized_frequency,
                bandwidth_hz=template.bandwidth_hz,
                chirp_frequency=template.chirp_frequency,
            )
        )
    return jammers


def generate_received_snapshot_matrix(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    jammer_list: Sequence[JammerInstance],
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Genera la matriz recibida X = ruido + suma_j a_j s_j.

    Parametros:
        config: Configuracion completa; aporta ruido, snapshots y senal.
        element_positions_m: Matriz (N, 3) con posiciones del array.
        jammer_list: Jammers que se inyectan en la matriz recibida.
        rng: Generador aleatorio reproducible.

    Devuelve:
        snapshot_matrix: Matriz recibida total X.
        jammer_table: Tabla con los jammers generados y su potencia.
        noise_matrix: Componente de ruido usada para X.
        jammer_matrix: Suma de las contribuciones de jammers.
    """
    X_noise = generate_complex_noise(config, rng)
    X_jammer_total = np.zeros_like(X_noise)
    X = X_noise.copy()
    rows = []

    for idx, jammer in enumerate(jammer_list, start=1):
        p_jam = jammer_power_from_jnr(config.noise.noise_power_linear, jammer.jnr_dB)
        a_jam = steering_vector(config, element_positions_m, jammer.azimuth_deg, jammer.elevation_deg)
        s_jam = generate_jammer_baseband_signal(jammer, config.signal.num_snapshots, p_jam, rng)
        X_jam = a_jam[:, None] * s_jam[None, :]
        X_jammer_total += X_jam
        X += X_jam

        row = asdict(jammer)
        row.update({"jammer_index": idx, "jammer_power_linear": p_jam})
        rows.append(row)

    return X, pd.DataFrame(rows), X_noise, X_jammer_total
