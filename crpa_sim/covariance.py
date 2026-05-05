"""covariance.py
Estimación y acondicionamiento de matriz de covarianza espacial.
"""

from __future__ import annotations

import numpy as np


def compute_sample_covariance(snapshot_matrix: np.ndarray) -> np.ndarray:
    """Calcula R = X X^H / L, con X de tamaño N elementos x L snapshots."""
    if snapshot_matrix.ndim != 2:
        raise ValueError("snapshot_matrix debe tener forma (num_elements, num_snapshots).")
    num_snapshots = snapshot_matrix.shape[1]
    if num_snapshots <= 0:
        raise ValueError("num_snapshots debe ser mayor que cero.")
    return snapshot_matrix @ snapshot_matrix.conj().T / num_snapshots


def add_diagonal_loading(R: np.ndarray, diagonal_loading_factor: float) -> np.ndarray:
    """Aplica carga diagonal: R_loaded = R + alpha*mean(diag(R))*I."""
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R debe ser una matriz cuadrada.")
    n = R.shape[0]
    avg_power = np.real(np.trace(R)) / n
    return R + diagonal_loading_factor * avg_power * np.eye(n, dtype=complex)


def invert_covariance(R: np.ndarray, diagonal_loading_factor: float) -> np.ndarray:
    """Pseudoinversa estable de R con diagonal loading."""
    return np.linalg.pinv(add_diagonal_loading(R, diagonal_loading_factor))
