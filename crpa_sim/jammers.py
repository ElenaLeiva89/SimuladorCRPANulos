"""jammers.py: generación de interferencias y ruido complejo."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .config import JammerConfig, ScenarioConfig
from .crpa_array import steering_vector_ideal


def jammer_power_from_jnr(noise_power_linear: float, jnr_dB: float) -> float:
    """Pjam = Pnoise * 10^(JNR_dB/10)."""
    return noise_power_linear * 10.0 ** (jnr_dB / 10.0)


def generate_complex_noise(num_elements: int, num_snapshots: int, noise_power_linear: float, rng: np.random.Generator) -> np.ndarray:
    """Ruido complejo circular blanco de tamaño (N elementos, L snapshots)."""
    sigma = np.sqrt(noise_power_linear / 2.0)
    return sigma * (rng.standard_normal((num_elements, num_snapshots)) + 1j * rng.standard_normal((num_elements, num_snapshots)))


def generate_jammer_baseband_signal(jammer: JammerConfig, num_snapshots: int, jammer_power_linear: float, rng: np.random.Generator) -> np.ndarray:
    """Genera señal baseband compleja de un jammer."""
    if jammer.signal_type == "complex_gaussian":
        sigma = np.sqrt(jammer_power_linear / 2.0)
        return sigma * (rng.standard_normal(num_snapshots) + 1j * rng.standard_normal(num_snapshots))

    if jammer.signal_type == "tone":
        n = np.arange(num_snapshots)
        phase0 = rng.uniform(0.0, 2.0 * np.pi)
        amplitude = np.sqrt(jammer_power_linear)
        return amplitude * np.exp(1j * (2.0 * np.pi * jammer.normalized_frequency * n + phase0))

    raise ValueError(f"signal_type no soportado: {jammer.signal_type}")


def generate_received_snapshot_matrix(
    config: ScenarioConfig,
    jammer_list: Sequence[JammerConfig],
    element_positions_m: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Genera X = ruido + suma_j a_j s_j."""
    snapshot_matrix = generate_complex_noise(
        config.num_elements, config.num_snapshots, config.noise_power_linear, rng
    )
    rows = []

    for idx, jammer in enumerate(jammer_list):
        p_jam = jammer_power_from_jnr(config.noise_power_linear, jammer.jnr_dB)
        a_jam = steering_vector_ideal(
            element_positions_m, jammer.azimuth_deg, jammer.elevation_deg, config.wavelength_m
        )
        s_jam = generate_jammer_baseband_signal(jammer, config.num_snapshots, p_jam, rng)
        snapshot_matrix += a_jam[:, None] * s_jam[None, :]

        rows.append(
            {
                "jammer_index": idx,
                "name": jammer.name,
                "azimuth_deg": jammer.azimuth_deg,
                "elevation_deg": jammer.elevation_deg,
                "jnr_dB": jammer.jnr_dB,
                "jammer_power_linear": p_jam,
                "signal_type": jammer.signal_type,
                "normalized_frequency": jammer.normalized_frequency,
            }
        )

    return snapshot_matrix, pd.DataFrame(rows)
