"""
null_metrics.py
---------------
Cálculo de profundidad y anchura de nulos por jammer.

Para cada jammer:
- profundidad: valor exacto de B(az_jam, el_jam)
- ancho en azimut: barrido de azimut manteniendo elevación del jammer
- ancho en elevación: barrido de elevación manteniendo azimut del jammer
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .config import JammerConfig, ScenarioConfig
from .crpa_array import (
    compute_azimuth_response_cut,
    compute_elevation_response_cut,
    compute_null_depth_dB,
)


def measure_null_width_1d(
    angle_grid_deg: np.ndarray,
    response_dB: np.ndarray,
    jammer_angle_deg: float,
    threshold_dB: float,
) -> float | None:
    """
    Mide el ancho del nulo alrededor del jammer para un umbral dado.

    Parameters
    ----------
    angle_grid_deg:
        Vector angular del barrido, en grados.

    response_dB:
        Patrón normalizado en dB.

    jammer_angle_deg:
        Ángulo del jammer en ese corte.

    threshold_dB:
        Umbral de atenuación. Ejemplo: -10, -20, -30, -40 , -50 dB.

    Returns
    -------
    float | None
        Anchura angular en grados. Devuelve None si el jammer no cae
        dentro de una región por debajo del umbral.
    """

    angle_grid_deg = np.asarray(angle_grid_deg, dtype=float)
    response_dB = np.asarray(response_dB, dtype=float)

    idx_jammer = int(np.argmin(np.abs(angle_grid_deg - jammer_angle_deg)))

    if response_dB[idx_jammer] > threshold_dB:
        return None

    left = idx_jammer
    while left > 0 and response_dB[left] <= threshold_dB:
        left -= 1

    right = idx_jammer
    while right < len(response_dB) - 1 and response_dB[right] <= threshold_dB:
        right += 1

    left_angle = angle_grid_deg[left]
    right_angle = angle_grid_deg[right]

    return float(abs(right_angle - left_angle))


def compute_null_metrics_for_jammers(
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    config: ScenarioConfig,
    jammer_list: Sequence[JammerConfig],
    azimuth_scan_deg: np.ndarray,
    elevation_scan_deg: np.ndarray,
    thresholds_dB: Sequence[float] = (0, -10, -20, -30, -40, -50),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Calcula la tabla final de profundidad y ancho de nulo por jammer.

    Para cada jammer se generan dos cortes específicos:

    - corte azimutal:
        azimuth variable, elevation = elevation_jammer

    - corte de elevación:
        elevation variable, azimuth = azimuth_jammer

    Returns
    -------
    metrics_table:
        Tabla resumen final.

    jammer_cut_tables:
        Diccionario con los cortes individuales por jammer.
    """

    rows = []
    jammer_cut_tables: dict[str, pd.DataFrame] = {}

    for jammer_index, jammer in enumerate(jammer_list, start=1):

        az_cut = compute_azimuth_response_cut(
            element_positions_m=element_positions_m,
            weights=weights,
            wavelength_m=config.wavelength_m,
            azimuth_scan_deg=azimuth_scan_deg,
            fixed_elevation_deg=jammer.elevation_deg,
        )

        el_cut = compute_elevation_response_cut(
            element_positions_m=element_positions_m,
            weights=weights,
            wavelength_m=config.wavelength_m,
            elevation_scan_deg=elevation_scan_deg,
            fixed_azimuth_deg=jammer.azimuth_deg,
        )

        jammer_key = f"jammer_{jammer_index}_{jammer.name}"

        jammer_cut_tables[f"{jammer_key}_azimuth_cut"] = az_cut
        jammer_cut_tables[f"{jammer_key}_elevation_cut"] = el_cut

        null_depth_dB = compute_null_depth_dB(
            element_positions_m=element_positions_m,
            weights=weights,
            wavelength_m=config.wavelength_m,
            jammer_azimuth_deg=jammer.azimuth_deg,
            jammer_elevation_deg=jammer.elevation_deg,
            reference_gain_abs=1.0,
        )

        for threshold_dB in thresholds_dB:

            if threshold_dB == 0:
                width_az = None
                width_el = None
            else:
                width_az = measure_null_width_1d(
                    angle_grid_deg=az_cut["azimuth_deg"].to_numpy(),
                    response_dB=az_cut["response_dB_normalized"].to_numpy(),
                    jammer_angle_deg=jammer.azimuth_deg,
                    threshold_dB=threshold_dB,
                )

                width_el = measure_null_width_1d(
                    angle_grid_deg=el_cut["elevation_deg"].to_numpy(),
                    response_dB=el_cut["response_dB_normalized"].to_numpy(),
                    jammer_angle_deg=jammer.elevation_deg,
                    threshold_dB=threshold_dB,
                )

            rows.append(
                {
                    "jammer_index": jammer_index,
                    "jammer_name": jammer.name,
                    "jammer_azimuth_deg": jammer.azimuth_deg,
                    "jammer_elevation_deg": jammer.elevation_deg,
                    "jnr_dB": jammer.jnr_dB,
                    "null_depth_dB": null_depth_dB,
                    "attenuation_threshold_dB": threshold_dB,
                    "null_width_azimuth_deg": width_az,
                    "null_width_elevation_deg": width_el,
                }
            )

    metrics_table = pd.DataFrame(rows)

    return metrics_table, jammer_cut_tables