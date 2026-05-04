"""
beamformers.py
--------------
Algoritmos de cálculo de pesos.

Importante:
- Aquí se calculan los pesos w.
- Los patrones se calculan después como w^H a(az,el), no directamente desde X.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import JammerConfig, ScenarioConfig
from .crpa_array import conventional_steering_weights, steering_vector_ideal
from .covariance import add_diagonal_loading, compute_sample_covariance


def compute_power_inversion_weights(
    snapshot_matrix: np.ndarray,
    reference_vector: np.ndarray | None = None,
    diagonal_loading_factor: float = 1e-3,
) -> np.ndarray:
    """Power Inversion/Sidelobe Canceller básico.

    Usa w = R^-1 c / (c^H R^-1 c), con c como restricción de referencia.
    Por defecto c=[1,0,0,...]^T, manteniendo el elemento 0 como referencia.
    """
    num_elements = snapshot_matrix.shape[0]
    R = compute_sample_covariance(snapshot_matrix)
    R_loaded = add_diagonal_loading(R, diagonal_loading_factor)

    if reference_vector is None:
        c = np.zeros(num_elements, dtype=complex)
        c[0] = 1.0
    else:
        c = np.asarray(reference_vector, dtype=complex)

    R_inv_c = np.linalg.pinv(R_loaded) @ c
    denominator = np.vdot(c, R_inv_c)  # c^H R^-1 c
    return R_inv_c / (denominator + 1e-15)


def compute_lcmv_weights(
    snapshot_matrix: np.ndarray,
    element_positions_m: np.ndarray,
    config: ScenarioConfig,
    jammer_list: Sequence[JammerConfig],
    include_hard_nulls: bool = True,
) -> np.ndarray:
    """LCMV real: w = R^-1 C (C^H R^-1 C)^-1 f.

    Restricciones implementadas:
    - primera columna de C: dirección deseada, f=1;
    - columnas siguientes: direcciones jammer, f=0, si include_hard_nulls=True.

    Con 7 elementos, no conviene imponer más de 7 restricciones en total.
    """
    R = compute_sample_covariance(snapshot_matrix)
    R_loaded = add_diagonal_loading(R, config.diagonal_loading_factor)
    R_inv = np.linalg.pinv(R_loaded)

    steering_vectors = [
        steering_vector_ideal(
            element_positions_m,
            config.desired_azimuth_deg,
            config.desired_elevation_deg,
            config.wavelength_m,
        )
    ]
    desired_response = [1.0 + 0.0j]

    if include_hard_nulls:
        for jammer in jammer_list:
            steering_vectors.append(
                steering_vector_ideal(
                    element_positions_m,
                    jammer.azimuth_deg,
                    jammer.elevation_deg,
                    config.wavelength_m,
                )
            )
            desired_response.append(0.0 + 0.0j)

    C = np.column_stack(steering_vectors)  # N x K
    f = np.asarray(desired_response, dtype=complex)  # K

    middle = C.conj().T @ R_inv @ C
    w = R_inv @ C @ np.linalg.pinv(middle) @ f
    return w


def compute_weights(
    algorithm_type: str,
    snapshot_matrix: np.ndarray,
    element_positions_m: np.ndarray,
    config: ScenarioConfig,
    jammer_list: Sequence[JammerConfig],
) -> np.ndarray:
    """Selector único de algoritmo."""
    algorithm = algorithm_type.lower()

    if algorithm == "conventional":
        return conventional_steering_weights(
            element_positions_m,
            config.desired_azimuth_deg,
            config.desired_elevation_deg,
            config.wavelength_m,
        )

    if algorithm == "power_inversion":
        return compute_power_inversion_weights(
            snapshot_matrix,
            diagonal_loading_factor=config.diagonal_loading_factor,
        )

    if algorithm in {"lcmv", "lcmw"}:
        return compute_lcmv_weights(
            snapshot_matrix,
            element_positions_m,
            config,
            jammer_list,
            include_hard_nulls=True,
        )

    raise ValueError(f"algorithmType no soportado: {algorithm_type}")
