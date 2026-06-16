"""Metricas de profundidad, anchura y area de nulos.

Las funciones de este modulo no calculan pesos: reciben un vector de pesos ya
estimado y miden como responde el patron en la direccion de cada jammer. Se
exporta una fila por jammer, umbral y Monte Carlo.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .array_model import steering_vector
from .config import JammerInstance, ProjectConfig
from .patterns import compute_azimuth_response_cut, compute_elevation_response_cut, compute_2d_response_grid


def compute_null_depth_dB(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    jammer: JammerInstance,
    reference_gain_abs: float = 1.0,
) -> float:
    """Calcula la profundidad del nulo en la direccion exacta del jammer.

    Parametros:
        config: Configuracion completa, usada para construir el steering.
        element_positions_m: Matriz (N, 3) con posiciones del array.
        weights: Vector complejo de pesos del beamformer.
        jammer: Jammer cuya direccion se evalua.
        reference_gain_abs: Ganancia absoluta de referencia para convertir a dB.
    """
    a_j = steering_vector(config, element_positions_m, jammer.azimuth_deg, jammer.elevation_deg)

    
    if config.beamforming.algorithm in ("lcmv", "lcmvq"):
        gain_abs = abs(weights @ a_j)
    else:
        gain_abs = abs(np.vdot(weights, a_j))
    reference = abs(reference_gain_abs)

    if reference == 0.0:
        raise ValueError("reference_gain_abs no puede ser 0.")

    if gain_abs == 0.0:
        return float("-inf")

    depth = 20.0 * np.log10(gain_abs / reference)

    return float(depth)

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

    grid_2d = compute_2d_response_grid(
        config,
        element_positions_m,
        weights,
        azimuth_scan_deg,
        elevation_scan_deg,
    )

    response_2d_dB = grid_2d["response_power_dB"]

    for jammer in jammer_list:
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

        cuts[f"{jammer.name}_azimuth_cut"] = az_cut
        cuts[f"{jammer.name}_elevation_cut"] = el_cut

        null_depth = compute_null_depth_dB(config, element_positions_m, weights, jammer, reference_gain_abs=1.0)

        for threshold in config.scan.null_thresholds_dB:
            region_2d = measure_null_region_2d(
                grid_2d["azimuth_deg"],
                grid_2d["elevation_deg"],
                response_2d_dB,
                jammer.azimuth_deg,
                jammer.elevation_deg,
                threshold,
            )
            rows.append(
                {
                    "montecarlo_index": montecarlo_index,
                    "algorithm": config.beamforming.algorithm,
                    "doa_mode": config.simulation.doa_mode,
                    "jammer_azimuth_deg": jammer.azimuth_deg,
                    "jammer_elevation_deg": jammer.elevation_deg,
                    "jammer_jnr_dB": jammer.jnr_dB,
                    "jammer_signal_type": jammer.signal_type,
                    "null_depth_dB": null_depth,
                    "attenuation_threshold_dB": threshold,
                    "null_area_cells": region_2d["null_area_cells_2d"],
                    "null_area_deg2": region_2d["null_area_deg2_2d"],
                    "null_width_azimuth": region_2d["null_width_azimuth_2d_deg"],
                    "null_width_elevation": region_2d["null_width_elevation_2d_deg"],
                }
            )

    return pd.DataFrame(rows), cuts


def wrap_angle_180(angle_deg: float) -> float:
    """Normaliza un azimut al rango [-180, 180)."""
    return ((angle_deg + 180.0) % 360.0) - 180.0


def circular_azimuth_width_deg(angles_deg: np.ndarray) -> float:
    """Calcula la anchura angular minima teniendo en cuenta wrap-around.

    Parametros:
        angles_deg: Azimuts de las celdas que pertenecen a una misma region
            de nulo, en grados.
    """

    angles = np.mod(angles_deg, 360.0)
    angles = np.sort(angles)
    if len(angles) <= 1:
        return 0.0

    diffs = np.diff(angles)
    wrap_gap = 360.0 - (angles[-1] - angles[0])

    max_gap = max(np.max(diffs), wrap_gap)

    return float(360.0 - max_gap)


def measure_null_region_2d(
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    response_dB: np.ndarray,
    jammer_azimuth_deg: float,
    jammer_elevation_deg: float,
    threshold_dB: float,
) -> dict:
    """Mide la region 2D del nulo alrededor del jammer en la malla az/el.

    Parametros:
        az_grid_deg: Malla 2D de azimuts en grados.
        el_grid_deg: Malla 2D de elevaciones en grados.
        response_dB: Respuesta en dB en cada celda de la malla.
        jammer_azimuth_deg: Azimut del jammer que define el centro buscado.
        jammer_elevation_deg: Elevacion del jammer que define el centro.
        threshold_dB: Umbral maximo para considerar una celda parte del nulo.
    """

    jammer_azimuth_deg = wrap_angle_180(jammer_azimuth_deg)

    az_grid_deg = np.asarray(az_grid_deg, dtype=float)
    el_grid_deg = np.asarray(el_grid_deg, dtype=float)
    response_dB = np.asarray(response_dB, dtype=float)

    az_delta = ((az_grid_deg - jammer_azimuth_deg + 180.0) % 360.0) - 180.0
    dist = (
        az_delta ** 2
        + (el_grid_deg - jammer_elevation_deg) ** 2
    )

    start = np.unravel_index(np.argmin(dist), dist.shape)

    if response_dB[start] > threshold_dB:
        return {
            "null_area_cells_2d": None,
            "null_area_deg2_2d": None,
            "null_width_azimuth_2d_deg": None,
            "null_width_elevation_2d_deg": None,
        }

    mask = response_dB <= threshold_dB
    visited = np.zeros(mask.shape, dtype=bool)
    az_axis = az_grid_deg[0, :] if az_grid_deg.ndim == 2 and az_grid_deg.shape[1] > 0 else np.asarray([])
    az_step = float(np.nanmedian(np.abs(np.diff(az_axis)))) if len(az_axis) > 1 else 0.0
    az_span = float(np.nanmax(az_axis) - np.nanmin(az_axis)) if len(az_axis) > 1 else 0.0
    wrap_azimuth = len(az_axis) > 1 and az_span >= 360.0 - max(az_step, 1e-9)

    stack = [start]
    region = []

    while stack:
        row, col = stack.pop()

        if row < 0 or row >= mask.shape[0]:
            continue
        if col < 0 or col >= mask.shape[1]:
            continue
        if visited[row, col]:
            continue
        if not mask[row, col]:
            continue

        visited[row, col] = True
        region.append((row, col))

        stack.append((row - 1, col))
        stack.append((row + 1, col))
        if wrap_azimuth:
            stack.append((row, (col - 1) % mask.shape[1]))
            stack.append((row, (col + 1) % mask.shape[1]))
        else:
            stack.append((row, col - 1))
            stack.append((row, col + 1))

    if not region:
        return {
            "null_area_cells_2d": None,
            "null_area_deg2_2d": None,
            "null_width_azimuth_2d_deg": None,
            "null_width_elevation_2d_deg": None,
        }

    rows = np.array([p[0] for p in region])
    cols = np.array([p[1] for p in region])

    az_vals = az_grid_deg[rows, cols]
    el_vals = el_grid_deg[rows, cols]

    az_step = float(np.nanmedian(np.abs(np.diff(az_grid_deg[0, :]))))
    el_step = float(np.nanmedian(np.abs(np.diff(el_grid_deg[:, 0]))))

    area_cells = len(region)
    area_deg2 = area_cells * az_step * el_step

    width_az = circular_azimuth_width_deg(az_vals)
    width_el = float(np.max(el_vals) - np.min(el_vals))

    if width_az == 0.0:
        width_az = az_step

    if width_el == 0.0:
        width_el = el_step

    return {
        "null_area_cells_2d": area_cells,
        "null_area_deg2_2d": area_deg2,
        "null_width_azimuth_2d_deg": width_az,
        "null_width_elevation_2d_deg": width_el,
    }
