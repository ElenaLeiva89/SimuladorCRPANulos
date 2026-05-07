"""null_metrics.py
Medida de profundidad y anchura de nulos.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .array_model import steering_vector
from .config import JammerInstance, ProjectConfig
from .patterns import compute_azimuth_response_cut, compute_elevation_response_cut


def compute_null_depth_dB(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    jammer: JammerInstance,
    reference_gain_abs: float = 1.0,
    floor_dB: float = -120.0,
) -> float:
    """Calcula la profundidad del nulo en la direccion exacta del jammer.

    Parametros:
        config: Configuracion completa, usada para construir el steering.
        element_positions_m: Matriz (N, 3) con posiciones del array.
        weights: Vector complejo de pesos del beamformer.
        jammer: Jammer cuya direccion se evalua.
        reference_gain_abs: Ganancia absoluta de referencia para convertir a dB.
        floor_dB: Suelo minimo reportado para evitar valores numericos extremos.
    """
    a_j = steering_vector(config, element_positions_m, jammer.azimuth_deg, jammer.elevation_deg)
    gain_abs = abs(np.vdot(weights, a_j))
    depth = 20.0 * np.log10(gain_abs / (reference_gain_abs + 1e-15) + 1e-12)
    return float(max(depth, floor_dB))


def measure_null_width_1d(angle_grid_deg: np.ndarray, response_dB: np.ndarray, jammer_angle_deg: float, threshold_dB: float) -> float | None:
    """Mide la anchura contigua del nulo alrededor del angulo del jammer.

    Parametros:
        angle_grid_deg: Vector angular del corte, en grados.
        response_dB: Respuesta normalizada del corte, en dB.
        jammer_angle_deg: Angulo del jammer dentro del eje del corte.
        threshold_dB: Umbral de atenuacion usado para delimitar el nulo.
    """
    angle_grid_deg = np.asarray(angle_grid_deg, dtype=float)
    response_dB = np.asarray(response_dB, dtype=float)
    idx = int(np.argmin(np.abs(angle_grid_deg - jammer_angle_deg)))

    if response_dB[idx] > threshold_dB:
        return None

    left = idx
    while left > 0 and response_dB[left] <= threshold_dB:
        left -= 1

    right = idx
    while right < len(response_dB) - 1 and response_dB[right] <= threshold_dB:
        right += 1

    return float(abs(angle_grid_deg[right] - angle_grid_deg[left]))


def compute_null_metrics_for_jammers(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    jammer_list: Sequence[JammerInstance],
    azimuth_scan_deg: np.ndarray,
    elevation_scan_deg: np.ndarray,
    montecarlo_index: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Calcula metricas de nulo y cortes especificos para cada jammer.

    Parametros:
        config: Configuracion completa; aporta umbrales de nulo y steering.
        element_positions_m: Matriz (N, 3) con posiciones del array.
        weights: Vector complejo de pesos del beamformer.
        jammer_list: Lista de jammers a evaluar.
        azimuth_scan_deg: Vector de azimuts para cortes horizontales.
        elevation_scan_deg: Vector de elevaciones para cortes verticales.
        montecarlo_index: Indice de la iteracion Monte Carlo actual.
    """
    rows: list[dict] = []
    cuts: dict[str, pd.DataFrame] = {}

    for jammer_index, jammer in enumerate(jammer_list, start=1):
        az_cut = compute_azimuth_response_cut(
            config,
            element_positions_m,
            weights,
            azimuth_scan_deg,
            fixed_elevation_deg=jammer.elevation_deg,
        )
        el_cut = compute_elevation_response_cut(
            config,
            element_positions_m,
            weights,
            elevation_scan_deg,
            fixed_azimuth_deg=jammer.azimuth_deg,
        )

        cut_prefix = f"{jammer.name}"
        cuts[f"{cut_prefix}_azimuth_cut"] = az_cut
        cuts[f"{cut_prefix}_elevation_cut"] = el_cut

        null_depth = compute_null_depth_dB(config, element_positions_m, weights, jammer, reference_gain_abs=1.0)
        jammer_azimuth_for_width_deg = wrap_angle_180(jammer.azimuth_deg)
        
        for threshold in config.scan.null_thresholds_dB:
            width_az = measure_null_width_1d(
                az_cut["azimuth_deg"].to_numpy(),
                az_cut["response_dB_normalized"].to_numpy(),
                jammer_azimuth_for_width_deg,
                threshold,
            )
            width_el = measure_null_width_1d(
                el_cut["elevation_deg"].to_numpy(),
                el_cut["response_dB_normalized"].to_numpy(),
                jammer.elevation_deg,
                threshold,
            )
            rows.append(
                {
                    "jammer_name": jammer.name,
                    "jammer_azimuth_deg": jammer.azimuth_deg,
                    "jammer_elevation_deg": jammer.elevation_deg,
                    "jammer_jnr_dB": jammer.jnr_dB,
                    "null_depth_dB": null_depth,
                    "attenuation_threshold_dB": threshold,
                    "null_width_azimuth_deg": width_az,
                    "null_width_elevation_deg": width_el,
                }
            )

    return pd.DataFrame(rows), cuts


def summarize_null_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Agrupa metricas Monte Carlo por jammer y umbral de atenuacion.

    Parametros:
        metrics: Tabla devuelta por compute_null_metrics_for_jammers,
            concatenada para una o varias iteraciones Monte Carlo.
    """
    group_cols = [
        "jammer_name",
        "attenuation_threshold_dB",
    ]

    numeric_cols = [
        "null_width_azimuth_deg",
        "null_width_elevation_deg",
    ]

    return metrics.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()

def wrap_angle_180(angle_deg: float) -> float:
    """Normaliza un azimut al rango [-180, 180)."""
    return ((angle_deg + 180.0) % 360.0) - 180.0