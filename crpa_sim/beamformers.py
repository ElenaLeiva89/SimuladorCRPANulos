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
# from .jammers import jammer_center_frequency_hz
from .jammers import jammer_wavelength_m


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

    if denominator == 0.0:
        raise ZeroDivisionError(
            "Power Inversion: denominator es cero. No se pueden calcular los pesos."
        )

    return numerator / denominator


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
        #steering_vectors.append(steering_vector(config, element_positions_m, jammer.azimuth_deg, jammer.elevation_deg))
        # f_jam_hz = jammer_center_frequency_hz(config, jammer)
        # lambda_jam_m = config.signal.speed_of_light_m_s / f_jam_hz
        lambda_jam_m = jammer_wavelength_m(config, jammer)
        steering_vectors.append(
            steering_vector(
                config,
                element_positions_m,
                jammer.azimuth_deg,
                jammer.elevation_deg,
                wavelength_m=lambda_jam_m,
            )
        )

        # CAMBIO DEL DEPTH DB
        desired_response.append(0.0 + 0.0j)
        #desired_response.append(0.01 + 0.0j)

    # C = np.column_stack(steering_vectors)
    # f = np.asarray(desired_response, dtype=complex)
    # middle = C.conj().T @ R_inv @ C
    # return R_inv @ C @ np.linalg.pinv(middle) @ f

    C = np.column_stack(steering_vectors)
    f = np.asarray(desired_response, dtype=complex)
    middle = C.conj().T @ R_inv @ C
    w = R_inv @ C @ np.linalg.pinv(middle) @ f
    w = np.conjugate(w)

    return w


def compute_lcmvq_weights(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    jammer_list: Sequence[JammerInstance],
) -> np.ndarray:
    """Calcula pesos LCMVQ: restricciones geométricas sin covarianza."""

    num_elements = config.array.num_elements

    main_constraint = np.zeros(num_elements, dtype=complex)
    main_constraint[0] = 1.0 + 0.0j

    steering_vectors = []
    steering_vectors = [main_constraint]

    a_des = steering_vector(
        config,
        element_positions_m,
        config.beamforming.desired_azimuth_deg,
        config.beamforming.desired_elevation_deg,
    )
    steering_vectors.append(a_des)

    desired_response = [1.0 + 0.0j]
    desired_response = [1.0 + 0.0j, 1.0 + 0.0j]

    for jammer in jammer_list:
        # a_jam = steering_vector(
        #     config,
        #     element_positions_m,
        #     jammer.azimuth_deg,
        #     jammer.elevation_deg,
        # )
        # f_jam_hz = jammer_center_frequency_hz(config, jammer)
        # lambda_jam_m = config.signal.speed_of_light_m_s / f_jam_hz
        lambda_jam_m = jammer_wavelength_m(config, jammer)
        a_jam = steering_vector(
            config,
            element_positions_m,
            jammer.azimuth_deg,
            jammer.elevation_deg,
            wavelength_m=lambda_jam_m,
        )
        steering_vectors.append(a_jam)
        # CAMBIO DEL DEPTH DB
        desired_response.append(0.00001 + 0.0j)
        #desired_response.append(0.01 + 0.0j)

    C = np.column_stack(steering_vectors)
    f = np.asarray(desired_response, dtype=complex)

    w = C @ np.linalg.pinv(C.conj().T @ C) @ f

    # Si quieres mantener convención MATLAB:
    w = np.conjugate(w)

    return w


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
    if algorithm == "lcmvq":
        return compute_lcmvq_weights(config, element_positions_m, jammer_list)
    raise ValueError(f"Algoritmo no soportado: {algorithm}")
