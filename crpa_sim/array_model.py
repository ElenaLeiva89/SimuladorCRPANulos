"""array_model.py
Geometría CRPA y steering vectors.

Punto de sustitución futura por CRPA real:
- steering_vector() llama ahora a steering_vector_ideal().
- En el futuro, si steering_model == 'measured', esta función podrá llamar
  a un interpolador de diagramas/steering reales.
"""

from __future__ import annotations

import numpy as np

from .config import ArrayConfig, ProjectConfig


def create_crpa_geometry(array_config: ArrayConfig, element_spacing_m: float) -> np.ndarray:
    """Crea posiciones 3D de la CRPA ideal hexagonal de 7 elementos."""
    if array_config.geometry != "hexagonal_7" or array_config.num_elements != 7:
        raise ValueError("Actualmente solo se implementa geometry='hexagonal_7' con num_elements=7.")

    positions_m = np.zeros((7, 3), dtype=float)
    positions_m[0] = [0.0, 0.0, 0.0]
    for k in range(6):
        angle_rad = 2.0 * np.pi * k / 6.0
        positions_m[k + 1] = [
            element_spacing_m * np.cos(angle_rad),
            element_spacing_m * np.sin(angle_rad),
            0.0,
        ]
    return positions_m


def direction_unit_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Convierte azimut/elevación a vector unitario 3D.

    Convención:
    - azimut: plano XY, 0 deg hacia +X, positivo antihorario hacia +Y.
    - elevación: 0 deg horizonte, 90 deg cenit.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def steering_vector_ideal(element_positions_m: np.ndarray, azimuth_deg: float, elevation_deg: float, wavelength_m: float) -> np.ndarray:
    """Steering ideal de elementos isotrópicos unitarios."""
    k_rad_m = 2.0 * np.pi / wavelength_m
    u = direction_unit_vector(azimuth_deg, elevation_deg)
    phase_rad = k_rad_m * (element_positions_m @ u)
    return np.exp(1j * phase_rad)


def steering_vector(config: ProjectConfig, element_positions_m: np.ndarray, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Devuelve el steering vector según el modelo seleccionado."""
    if config.array.steering_model == "ideal":
        return steering_vector_ideal(element_positions_m, azimuth_deg, elevation_deg, config.signal.wavelength_m)
    if config.array.steering_model == "measured":
        raise NotImplementedError("steering_model='measured' se añadirá con datos reales de CRPA.")
    raise ValueError(f"Modelo steering no soportado: {config.array.steering_model}")
