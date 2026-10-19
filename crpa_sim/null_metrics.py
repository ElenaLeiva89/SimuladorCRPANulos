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
from .patterns import compute_azimuth_response_cut, compute_elevation_response_cut, compute_radial_response_cut, compute_2d_response_grid
# from .jammers import jammer_center_frequency_hz
from .jammers import jammer_center_frequency_hz, jammer_wavelength_m
from numba import njit
import matplotlib.pyplot as plt
from .measured_mat import MeasuredSteeringDatabase


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
    #a_j = steering_vector(config, element_positions_m, jammer.azimuth_deg, jammer.elevation_deg)
    # f_jam_hz = jammer_center_frequency_hz(config, jammer)
    # lambda_jam_m = config.signal.speed_of_light_m_s / f_jam_hz
    lambda_jam_m = jammer_wavelength_m(config, jammer)
    a_j = steering_vector(
        config,
        element_positions_m,
        jammer.azimuth_deg,
        jammer.elevation_deg,
        wavelength_m=lambda_jam_m,
    )
    
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

    # grid_2d = compute_2d_response_grid(
    #     config,
    #     element_positions_m,
    #     weights,
    #     azimuth_scan_deg,
    #     elevation_scan_deg,
    # )

    # response_2d_dB = grid_2d["response_power_dB"]

    for jammer in jammer_list:
        lambda_jam_m = jammer_wavelength_m(config, jammer)
        grid_2d = compute_2d_response_grid(
            config,
            element_positions_m,
            weights,
            azimuth_scan_deg,
            elevation_scan_deg,
            wavelength_m=lambda_jam_m,
        )

        response_2d_dB = grid_2d["response_power_dB"]
        az_cut = compute_azimuth_response_cut(
            config,
            element_positions_m,
            weights,
            azimuth_scan_deg,
            fixed_elevation_deg=jammer.elevation_deg,
            wavelength_m=lambda_jam_m,
        )
        el_cut = compute_elevation_response_cut(
            config,
            element_positions_m,
            weights,
            elevation_scan_deg,
            fixed_azimuth_deg=jammer.azimuth_deg,
            wavelength_m=lambda_jam_m,
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
                    "jammer_center_frequency_hz": jammer_center_frequency_hz(config, jammer),
                    "jammer_wavelength_m": lambda_jam_m,
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


@njit(cache=True)
def flood_fill_region(
    mask: np.ndarray,
    start_row: int,
    start_col: int,
    wrap_azimuth: bool,
):
    n_rows, n_cols = mask.shape

    visited = np.zeros((n_rows, n_cols), dtype=np.uint8)

    max_cells = mask.size

    rows = np.empty(max_cells, dtype=np.int32)
    cols = np.empty(max_cells, dtype=np.int32)

    # Cada celda puede añadir hasta 4 vecinos
    max_stack = max_cells * 4

    stack_rows = np.empty(max_stack, dtype=np.int32)
    stack_cols = np.empty(max_stack, dtype=np.int32)

    stack_size = 1
    stack_rows[0] = start_row
    stack_cols[0] = start_col

    count = 0

    while stack_size > 0:

        stack_size -= 1

        row = stack_rows[stack_size]
        col = stack_cols[stack_size]

        if row < 0 or row >= n_rows:
            continue

        if col < 0 or col >= n_cols:
            continue

        if visited[row, col]:
            continue

        if not mask[row, col]:
            continue

        visited[row, col] = 1

        rows[count] = row
        cols[count] = col
        count += 1

        # Arriba
        stack_rows[stack_size] = row - 1
        stack_cols[stack_size] = col
        stack_size += 1

        # Abajo
        stack_rows[stack_size] = row + 1
        stack_cols[stack_size] = col
        stack_size += 1

        if wrap_azimuth:

            # Izquierda con wrap
            stack_rows[stack_size] = row
            stack_cols[stack_size] = (col - 1) % n_cols
            stack_size += 1

            # Derecha con wrap
            stack_rows[stack_size] = row
            stack_cols[stack_size] = (col + 1) % n_cols
            stack_size += 1

        else:

            stack_rows[stack_size] = row
            stack_cols[stack_size] = col - 1
            stack_size += 1

            stack_rows[stack_size] = row
            stack_cols[stack_size] = col + 1
            stack_size += 1

    return rows[:count], cols[:count]

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

    az_axis = az_grid_deg[0, :]
    el_axis = el_grid_deg[:, 0]
    start= (np.argmin(np.abs(el_axis - jammer_elevation_deg)),np.argmin(np.abs(((az_axis - jammer_azimuth_deg + 180) % 360) - 180)))

    if response_dB[start] > threshold_dB:
        return {
            "null_area_cells_2d": None,
            "null_area_deg2_2d": None,
            "null_width_azimuth_2d_deg": None,
            "null_width_elevation_2d_deg": None,
        }

    mask = response_dB <= threshold_dB

    az_step = (
        float(np.nanmedian(np.abs(np.diff(az_axis))))
        if len(az_axis) > 1
        else 0.0
    )

    el_step = (
        float(np.nanmedian(np.abs(np.diff(el_axis))))
        if len(el_axis) > 1
        else 0.0
    )

    az_span = (
        float(np.nanmax(az_axis) - np.nanmin(az_axis))
        if len(az_axis) > 1
        else 0.0
    )

    wrap_azimuth = (
        len(az_axis) > 1
        and az_span >= 360.0 - max(az_step, 1e-9)
    )

    rows, cols = flood_fill_region(
        mask,
        start[0],
        start[1],
        wrap_azimuth,
    )

    if len(rows) == 0:
        return {
            "null_area_cells_2d": None,
            "null_area_deg2_2d": None,
            "null_width_azimuth_2d_deg": None,
            "null_width_elevation_2d_deg": None,
        }

    az_vals = az_grid_deg[rows, cols]
    el_vals = el_grid_deg[rows, cols]

    area_cells = len(rows)
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


def measure_null_widths_from_cuts(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    jammer_list: Sequence[JammerInstance],
    n_cuts: int,
) -> pd.DataFrame:

    rows = []
    cut_angles_deg = np.arange(0.0, 180.0, 180.0 / n_cuts,)

    CUT_HALF_WIDTH_DEG = 40
    N_POINTS = 201
    step_deg = 2 * CUT_HALF_WIDTH_DEG / (N_POINTS - 1)
    distance_scan_deg = np.linspace(- CUT_HALF_WIDTH_DEG, CUT_HALF_WIDTH_DEG, N_POINTS)

    for jammer in jammer_list:

        lambda_jam_m = jammer_wavelength_m(config,jammer )

        for cut_angle_deg in cut_angles_deg:

            cut = compute_radial_response_cut(
                config,
                element_positions_m,
                weights,
                center_az_deg=jammer.azimuth_deg,
                center_el_deg=jammer.elevation_deg,
                cut_angle_deg=cut_angle_deg,
                distance_scan_deg = distance_scan_deg,
                wavelength_m=lambda_jam_m,
            )

            response = 20.0 * np.log10(cut["response_abs"].to_numpy(dtype=float)+10**(-12))
            center_idx = len(response) // 2

            for threshold in config.scan.null_thresholds_dB:

                width_deg = np.nan

                if response[center_idx] <= threshold:

                    left = center_idx
                    while (left > 0 and response[left - 1] <= threshold):
                        left -= 1
                    right = center_idx
                    while (right < len(response) - 1 and response[right + 1] <= threshold):
                        right += 1

                    width_deg = (right - left) * step_deg

                rows.append(
                    {
                        "jammer_name": jammer.name,
                        "jammer_azimuth_deg": jammer.azimuth_deg,
                        "jammer_elevation_deg": jammer.elevation_deg,
                        "cut_angle_deg": cut_angle_deg,
                        "attenuation_threshold_dB": threshold,
                        "null_width_deg": width_deg,
                    }
                )

    return pd.DataFrame(rows)

def compute_jammer_metrics_v2(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    jammer_list: Sequence[JammerInstance],
    azimuth_scan_deg: np.ndarray,
    elevation_scan_deg: np.ndarray,
    jnr_db:float,
    num_jammers:int,
    database: MeasuredSteeringDatabase | None = None
) -> pd.DataFrame:

    rows = []

    for jammer in jammer_list:

        lambda_jam_m = jammer_wavelength_m(config,jammer,)

        grid_2d = compute_2d_response_grid(
            config,
            element_positions_m,
            weights,
            azimuth_scan_deg,
            elevation_scan_deg,
            wavelength_m=lambda_jam_m,
            database=database
        )

        response_2d_dB = grid_2d["response_power_dB"]
        az_grid, el_grid = np.meshgrid(azimuth_scan_deg,elevation_scan_deg,indexing="xy",)

        # ==================================================
        # Métrica 1: Atenuación exacta en la DOA
        # ==================================================

        az_idx = np.argmin(np.abs( ( (azimuth_scan_deg - jammer.azimuth_deg+ 180.0) % 360.0) - 180.0))
        el_idx = np.argmin(np.abs(elevation_scan_deg - jammer.elevation_deg))

        attenuation_exact_dB = response_2d_dB[el_idx,az_idx,]

        # ==================================================
        # Métrica 2: Atenuación media en conos
        # ==================================================

        delta_az = ((az_grid- jammer.azimuth_deg + 180.0) % 360.0) - 180.0
        delta_el = (el_grid - jammer.elevation_deg)
        distance_deg = np.sqrt( delta_az**2  + delta_el**2)

        mask_2deg = distance_deg <= 2.0
        mask_5deg = distance_deg <= 5.0
        mask_10deg = distance_deg <= 10.0

        attenuation_cone_2deg_dB = np.mean(response_2d_dB[mask_2deg])
        attenuation_cone_5deg_dB = np.mean(response_2d_dB[mask_5deg])
        attenuation_cone_10deg_dB = np.mean(response_2d_dB[mask_10deg])

        # ==================================================
        # Métrica 3: Porcentaje de espacio angular
        # ==================================================

        """percent_below_3dB = (100.0 * np.mean(response_2d_dB <= -3.0))
        percent_below_6dB = (100.0 * np.mean(response_2d_dB <= -6.0))
        percent_below_10dB = (100.0 * np.mean(response_2d_dB <= -10.0))
        percent_below_20dB = (100.0 * np.mean(response_2d_dB <= -20.0))
        percent_below_30dB = (100.0 * np.mean(response_2d_dB <= -30.0))
        percent_below_40dB = (100.0 * np.mean(response_2d_dB <= -40.0))"""

        rows.append(
            {
                "jammer_name": jammer.name,
                "jammer_azimuth_deg": jammer.azimuth_deg,
                "jammer_elevation_deg": jammer.elevation_deg,
                "jnr_dB": jnr_db,
                "num_jammers": num_jammers,
                "attenuation_exact_dB": attenuation_exact_dB,
                "attenuation_cone_2deg_dB": attenuation_cone_2deg_dB,
                "attenuation_cone_5deg_dB": attenuation_cone_5deg_dB,
                "attenuation_cone_10deg_dB": attenuation_cone_10deg_dB,
            }
        )

    return pd.DataFrame(rows)