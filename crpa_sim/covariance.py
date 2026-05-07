"""covariance.py
Estimacion y acondicionamiento de matriz de covarianza espacial.
"""

from __future__ import annotations

import numpy as np


def compute_sample_covariance(snapshot_matrix: np.ndarray) -> np.ndarray:
    """Calcula la covarianza muestral R = X X^H / L.

    Parametros:
        snapshot_matrix: Matriz compleja X con forma
            (num_elements, num_snapshots), donde cada fila es un canal de
            antena y cada columna un snapshot temporal.
    """
    if snapshot_matrix.ndim != 2:
        raise ValueError("snapshot_matrix debe tener forma (num_elements, num_snapshots).")
    num_snapshots = snapshot_matrix.shape[1]
    if num_snapshots <= 0:
        raise ValueError("num_snapshots debe ser mayor que cero.")
    return snapshot_matrix @ snapshot_matrix.conj().T / num_snapshots


def add_diagonal_loading(R: np.ndarray, diagonal_loading_factor: float) -> np.ndarray:
    """Aplica carga diagonal para acondicionar la matriz de covarianza.

    Parametros:
        R: Matriz de covarianza cuadrada que se quiere regularizar.
        diagonal_loading_factor: Factor alpha aplicado como
            R_loaded = R + alpha * mean(diag(R)) * I.
    """
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R debe ser una matriz cuadrada.")
    n = R.shape[0]
    avg_power = np.real(np.trace(R)) / n
    return R + diagonal_loading_factor * avg_power * np.eye(n, dtype=complex)


def invert_covariance(R: np.ndarray, diagonal_loading_factor: float) -> np.ndarray:
    """Calcula una pseudoinversa estable de la covarianza cargada.

    Parametros:
        R: Matriz de covarianza que se desea invertir.
        diagonal_loading_factor: Factor de carga diagonal aplicado antes de
            calcular la pseudoinversa.
    """
    return np.linalg.pinv(add_diagonal_loading(R, diagonal_loading_factor))
