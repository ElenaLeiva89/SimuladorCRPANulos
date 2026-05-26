"""patterns.py
Evaluacion de patrones espaciales de la CRPA.

El patron se calcula siempre como B(az, el) = w^H a(az, el). Los snapshots
no se usan para dibujar el patron; se usan antes para estimar R y los pesos.

La rama ideal se mantiene vectorizada exactamente como antes. La rama measured
usa una matriz/tensor de steering medido ya construido en memoria para evitar
buscar elemento a elemento para cada punto angular.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .array_model import measured_steering_matrix_for_angles, steering_vector
from .config import ProjectConfig


def conventional_weights(config: ProjectConfig, element_positions_m: np.ndarray) -> np.ndarray:
    """Calcula pesos delay-and-sum hacia la direccion deseada."""
    a_des = steering_vector(
        config,
        element_positions_m,
        config.beamforming.desired_azimuth_deg,
        config.beamforming.desired_elevation_deg,
    )
    return a_des / element_positions_m.shape[0]


def _evaluate_response_complex_for_angles(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    azimuth_deg_array: np.ndarray,
    elevation_deg_array: np.ndarray,
) -> np.ndarray:
    """Evalua B=w^H a(az, el) devolviendo valores complejos.

    Para ideal se mantiene el calculo analitico existente. Para measured se
    obtiene de una vez la matriz de steering medida para todos los pares az/el.
    """
    azimuth_deg_array = np.asarray(azimuth_deg_array, dtype=float)
    elevation_deg_array = np.asarray(elevation_deg_array, dtype=float)
    if len(azimuth_deg_array) != len(elevation_deg_array):
        raise ValueError("azimuth_deg_array y elevation_deg_array deben tener la misma longitud.")

    if config.array.steering_model == "measured":
        steering = measured_steering_matrix_for_angles(config, azimuth_deg_array, elevation_deg_array)
        return steering @ np.conjugate(weights)

    values = []
    for az, el in zip(azimuth_deg_array, elevation_deg_array):
        a = steering_vector(config, element_positions_m, float(az), float(el))
        values.append(np.vdot(weights, a))
    return np.asarray(values)


def evaluate_response_for_angles(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    azimuth_deg_array: np.ndarray,
    elevation_deg_array: np.ndarray,
    normalize: bool = True,
) -> pd.DataFrame:
    """Evalua B=w^H a(az, el) en una lista de pares angulares."""
    values = _evaluate_response_complex_for_angles(
        config,
        element_positions_m,
        weights,
        azimuth_deg_array,
        elevation_deg_array,
    )

    response_abs = np.abs(values)
    response_abs_norm = response_abs / (np.max(response_abs) + 1e-15) if normalize else response_abs
    response_dB_norm = 20.0 * np.log10(response_abs_norm + 1e-12)

    return pd.DataFrame(
        {
            "azimuth_deg": azimuth_deg_array,
            "elevation_deg": elevation_deg_array,
            "response_abs": response_abs,
            "response_abs_normalized": response_abs_norm,
            "response_dB_normalized": response_dB_norm,
            "response_real": np.real(values),
            "response_imag": np.imag(values),
        }
    )


def compute_azimuth_response_cut(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    azimuth_scan_deg: np.ndarray,
    fixed_elevation_deg: float,
) -> pd.DataFrame:
    """Calcula un corte de patron variando azimut con elevacion fija."""
    elevations = np.full_like(azimuth_scan_deg, fixed_elevation_deg, dtype=float)
    return evaluate_response_for_angles(config, element_positions_m, weights, azimuth_scan_deg, elevations)


def compute_elevation_response_cut(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    elevation_scan_deg: np.ndarray,
    fixed_azimuth_deg: float,
) -> pd.DataFrame:
    """Calcula un corte de patron variando elevacion con azimut fijo."""
    azimuths = np.full_like(elevation_scan_deg, fixed_azimuth_deg, dtype=float)
    return evaluate_response_for_angles(config, element_positions_m, weights, azimuths, elevation_scan_deg)


def compute_2d_response_grid(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    azimuth_scan_deg: np.ndarray,
    elevation_scan_deg: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evalua el patron en una malla 2D azimut/elevacion."""
    az_grid, el_grid = np.meshgrid(azimuth_scan_deg, elevation_scan_deg, indexing="xy")
    az_flat = az_grid.ravel()
    el_flat = el_grid.ravel()

    if config.array.steering_model == "ideal":
        # Rama ideal: se conserva el calculo vectorizado original.
        az_rad = np.deg2rad(az_flat)
        el_rad = np.deg2rad(el_flat)
        u = np.column_stack([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad),
        ])
        k_rad_m = 2.0 * np.pi / config.signal.wavelength_m
        phase = k_rad_m * (u @ element_positions_m.T)
        steering = np.exp(1j * phase)
        response_complex = steering @ np.conjugate(weights)
    elif config.array.steering_model == "measured":
        # Rama measured optimizada: steering para toda la malla en una matriz.
        # Shape: (num_puntos_malla, num_elementos).
        steering = measured_steering_matrix_for_angles(config, az_flat, el_flat)
        response_complex = steering @ np.conjugate(weights)
    else:
        raise ValueError(f"Modelo steering no soportado: {config.array.steering_model}")

    response_abs = np.abs(response_complex).reshape(az_grid.shape)
    response_power = response_abs**2
    response_power_dB = 10.0 * np.log10(response_power + 1e-12)

    return {
        "azimuth_deg": az_grid,
        "elevation_deg": el_grid,
        "response_abs": response_abs,
        "response_power": response_power,
        "response_power_dB": response_power_dB,
    }


def make_scan_vectors(config: ProjectConfig) -> tuple[np.ndarray, np.ndarray]:
    """Crea los vectores de barrido angular a partir de la configuracion."""
    az = np.arange(
        config.scan.azimuth_scan_min_deg,
        config.scan.azimuth_scan_max_deg + config.scan.azimuth_scan_step_deg,
        config.scan.azimuth_scan_step_deg,
    )
    el = np.arange(
        config.scan.elevation_scan_min_deg,
        config.scan.elevation_scan_max_deg + config.scan.elevation_scan_step_deg,
        config.scan.elevation_scan_step_deg,
    )
    return az, el
