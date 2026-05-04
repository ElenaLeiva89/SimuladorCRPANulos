"""covariance.py
----------------
Utilidades para estimar y acondicionar matrices de covarianza espacial.

Fase 2 del simulador:
- Los snapshots X se usan aquí para obtener R = X X^H / L.
- Los algoritmos adaptativos usan R, no los snapshots directamente para dibujar patrones.
"""

from __future__ import annotations

import numpy as np


def compute_sample_covariance(snapshot_matrix: np.ndarray) -> np.ndarray:
    """Calcula la matriz de covarianza espacial muestral.

    Parameters
    ----------
    snapshot_matrix:
        Matriz compleja X de tamaño (N, L), donde N es el número de
        elementos de antena y L el número de snapshots temporales.

    Returns
    -------
    np.ndarray
        Matriz de covarianza compleja R de tamaño (N, N): R = X X^H / L.
    """
    if snapshot_matrix.ndim != 2:
        raise ValueError("snapshot_matrix debe tener dimensión 2: (num_elements, num_snapshots).")
    num_snapshots = snapshot_matrix.shape[1]
    if num_snapshots <= 0:
        raise ValueError("snapshot_matrix debe contener al menos un snapshot.")
    return snapshot_matrix @ snapshot_matrix.conj().T / num_snapshots


def add_diagonal_loading(covariance_matrix: np.ndarray, loading_factor: float) -> np.ndarray:
    """Añade carga diagonal para estabilizar inversiones/pseudoinversiones.

    R_loaded = R + alpha * mean(diag(R)) * I
    """
    if covariance_matrix.ndim != 2 or covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("covariance_matrix debe ser cuadrada.")
    n = covariance_matrix.shape[0]
    average_power = np.real(np.trace(covariance_matrix)) / n
    return covariance_matrix + loading_factor * average_power * np.eye(n, dtype=complex)


def invert_covariance(covariance_matrix: np.ndarray, diagonal_loading_factor: float = 1e-3) -> np.ndarray:
    """Devuelve una pseudoinversa estable de R con carga diagonal previa."""
    return np.linalg.pinv(add_diagonal_loading(covariance_matrix, diagonal_loading_factor))
