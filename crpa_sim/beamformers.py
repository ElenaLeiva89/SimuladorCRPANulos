"""Calculo de pesos adaptativos para el array CRPA.

Este modulo contiene solo los algoritmos seleccionables desde la
configuracion validada: Power Inversion y LCMV. Los pesos convencionales se
mantienen en `patterns.py` como referencia para comparativas y plots.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .array_model import steering_vector
from .config import JammerInstance, ProjectConfig
from .covariance import compute_sample_covariance, invert_covariance


def compute_power_inversion_weights(config: ProjectConfig, snapshot_matrix: np.ndarray) -> np.ndarray:
    """Calcula pesos adaptativos con Power Inversion.

    Parametros:
        config: Configuracion del proyecto; aporta el elemento de referencia
            y el factor de diagonal loading.
        snapshot_matrix: Matriz X con forma (num_elements, num_snapshots)
            usada para estimar la covarianza espacial.
    """
    num_elements = snapshot_matrix.shape[0]
    ref_idx = config.beamforming.power_inversion_reference_element
    if not (0 <= ref_idx < num_elements):
        raise ValueError("power_inversion_reference_element fuera de rango.")

    R = compute_sample_covariance(snapshot_matrix)
    R_inv = invert_covariance(R, config.beamforming.diagonal_loading_factor)

    c = np.zeros(num_elements, dtype=complex)
    c[ref_idx] = 1.0 + 0.0j
    numerator = R_inv @ c
    denominator = np.vdot(c, numerator)
    return numerator / (denominator + 1e-15)


def compute_lcmv_weights(
    config: ProjectConfig,
    snapshot_matrix: np.ndarray,
    element_positions_m: np.ndarray,
    jammer_list: Sequence[JammerInstance],
) -> np.ndarray:
    """Calcula pesos LCMV con ganancia unitaria deseada y nulos en jammers.

    Parametros:
        config: Configuracion del proyecto; define direccion deseada,
            diagonal loading y parametros de array/senal.
        snapshot_matrix: Matriz X usada para estimar la covarianza espacial.
        element_positions_m: Matriz (N, 3) con posiciones de los elementos.
        jammer_list: Lista de jammers cuyas direcciones se fuerzan a cero.
    """
    if len(jammer_list) > config.array.num_elements - 1:
        raise ValueError("LCMV: numero de jammers supera num_elements - 1.")

    R = compute_sample_covariance(snapshot_matrix)
    R_inv = invert_covariance(R, config.beamforming.diagonal_loading_factor)

    steering_vectors = [
        steering_vector(
            config,
            element_positions_m,
            config.beamforming.desired_azimuth_deg,
            config.beamforming.desired_elevation_deg,
        )
    ]
    desired_response = [1.0 + 0.0j]

    for jammer in jammer_list:
        steering_vectors.append(steering_vector(config, element_positions_m, jammer.azimuth_deg, jammer.elevation_deg))
        desired_response.append(0.0 + 0.0j)

    C = np.column_stack(steering_vectors)
    f = np.asarray(desired_response, dtype=complex)
    middle = C.conj().T @ R_inv @ C
    return R_inv @ C @ np.linalg.pinv(middle) @ f


def compute_weights(
    config: ProjectConfig,
    snapshot_matrix: np.ndarray,
    element_positions_m: np.ndarray,
    jammer_list: Sequence[JammerInstance],
) -> np.ndarray:
    """Selecciona el algoritmo de beamforming configurado y devuelve pesos.

    Parametros:
        config: Configuracion completa del proyecto; se lee
            config.beamforming.algorithm, que debe ser "power_inversion" o
            "lcmv" tras la validacion de entrada.
        snapshot_matrix: Matriz X de snapshots recibidos.
        element_positions_m: Matriz (N, 3) con posiciones del array.
        jammer_list: Lista de jammers del caso actual, necesaria para LCMV.
    """
    algorithm = config.beamforming.algorithm.lower()
    if algorithm == "power_inversion":
        return compute_power_inversion_weights(config, snapshot_matrix)
    if algorithm == "lcmv":
        return compute_lcmv_weights(config, snapshot_matrix, element_positions_m, jammer_list)
    raise ValueError(f"Algoritmo no soportado: {algorithm}")
