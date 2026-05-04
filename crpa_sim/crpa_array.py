"""
crpa_array.py
-------------
Funciones de geometría, steering vector y evaluación de patrones espaciales.

Idea clave:
- Los snapshots X se usan para estimar R y calcular pesos adaptativos.
- El patrón de radiación/beamforming se calcula como B(az,el)=w^H a(az,el).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_hexagonal_7element_geometry(num_elements: int, element_spacing_m: float) -> np.ndarray:
    """Crea una CRPA ideal de 7 elementos: uno central y seis en hexágono."""
    if num_elements != 7:
        raise ValueError("Esta geometría está definida para num_elements=7.")

    positions_m = np.zeros((7, 3), dtype=float)
    positions_m[0, :] = [0.0, 0.0, 0.0]

    for k in range(6):
        angle_rad = 2.0 * np.pi * k / 6.0
        positions_m[k + 1, :] = [
            element_spacing_m * np.cos(angle_rad),
            element_spacing_m * np.sin(angle_rad),
            0.0,
        ]
    return positions_m


def direction_unit_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Convierte azimut/elevación en vector unitario [ux, uy, uz]."""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def steering_vector_ideal(
    element_positions_m: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    wavelength_m: float,
) -> np.ndarray:
    """Steering vector ideal de onda plana: a_n = exp(j*k*r_n·u)."""
    k_rad_m = 2.0 * np.pi / wavelength_m
    u = direction_unit_vector(azimuth_deg, elevation_deg)
    phase_rad = k_rad_m * (element_positions_m @ u)
    return np.exp(1j * phase_rad)


def conventional_steering_weights(
    element_positions_m: np.ndarray,
    desired_azimuth_deg: float,
    desired_elevation_deg: float,
    wavelength_m: float,
) -> np.ndarray:
    """Pesos delay-and-sum apuntados a la dirección deseada."""
    a_des = steering_vector_ideal(
        element_positions_m, desired_azimuth_deg, desired_elevation_deg, wavelength_m
    )
    return a_des / element_positions_m.shape[0]


def compute_response_for_angles(
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    wavelength_m: float,
    azimuth_deg_array: np.ndarray,
    elevation_deg_array: np.ndarray,
) -> pd.DataFrame:
    """Evalúa B=w^H a(az,el) en listas de pares az/el."""
    if len(azimuth_deg_array) != len(elevation_deg_array):
        raise ValueError("azimuth_deg_array y elevation_deg_array deben tener la misma longitud.")

    response_complex = []
    for az, el in zip(azimuth_deg_array, elevation_deg_array):
        a_scan = steering_vector_ideal(element_positions_m, float(az), float(el), wavelength_m)
        response_complex.append(np.vdot(weights, a_scan))  # np.vdot conjuga weights

    response_complex = np.asarray(response_complex)
    response_abs = np.abs(response_complex)
    response_abs_norm = response_abs / (np.max(response_abs) + 1e-15)
    response_dB_norm = 20.0 * np.log10(response_abs_norm + 1e-12)

    return pd.DataFrame(
        {
            "azimuth_deg": azimuth_deg_array,
            "elevation_deg": elevation_deg_array,
            "response_abs": response_abs,
            "response_abs_normalized": response_abs_norm,
            "response_dB_normalized": response_dB_norm,
            "response_real": np.real(response_complex),
            "response_imag": np.imag(response_complex),
        }
    )


def compute_azimuth_response_cut(
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    wavelength_m: float,
    azimuth_scan_deg: np.ndarray,
    fixed_elevation_deg: float,
) -> pd.DataFrame:
    """Corte de patrón en azimut con elevación fija."""
    elevation = np.full_like(azimuth_scan_deg, fixed_elevation_deg, dtype=float)
    return compute_response_for_angles(
        element_positions_m, weights, wavelength_m, azimuth_scan_deg, elevation
    )


def compute_elevation_response_cut(
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    wavelength_m: float,
    elevation_scan_deg: np.ndarray,
    fixed_azimuth_deg: float,
) -> pd.DataFrame:
    """Corte de patrón en elevación con azimut fijo."""
    azimuth = np.full_like(elevation_scan_deg, fixed_azimuth_deg, dtype=float)
    return compute_response_for_angles(
        element_positions_m, weights, wavelength_m, azimuth, elevation_scan_deg
    )


def compute_sample_covariance(snapshot_matrix: np.ndarray) -> np.ndarray:
    """Covarianza espacial muestral R = X X^H / L."""
    num_snapshots = snapshot_matrix.shape[1]
    return snapshot_matrix @ snapshot_matrix.conj().T / num_snapshots


def add_diagonal_loading(covariance_matrix: np.ndarray, loading_factor: float) -> np.ndarray:
    """Añade carga diagonal para estabilizar inversiones/pseudoinversiones."""
    n = covariance_matrix.shape[0]
    average_power = np.real(np.trace(covariance_matrix)) / n
    return covariance_matrix + loading_factor * average_power * np.eye(n, dtype=complex)


def compute_null_depth_dB(
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    wavelength_m: float,
    jammer_azimuth_deg: float,
    jammer_elevation_deg: float,
    reference_gain_abs: float | None = None,
) -> float:
    """Mide la profundidad del nulo en la dirección exacta del jammer."""
    a_jam = steering_vector_ideal(
        element_positions_m, jammer_azimuth_deg, jammer_elevation_deg, wavelength_m
    )
    gain_abs = abs(np.vdot(weights, a_jam))
    if reference_gain_abs is None:
        reference_gain_abs = 1.0
    return float(20.0 * np.log10(gain_abs / (reference_gain_abs + 1e-15) + 1e-12))
