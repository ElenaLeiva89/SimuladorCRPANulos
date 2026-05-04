"""
fft_tools.py
------------
Utilidades opcionales de FFT.

Dónde usar FFT:
1) Para analizar el espectro temporal de cada canal o jammer: FFT sobre el eje temporal.
2) Para acelerar patrones espaciales de un ULA: FFT espacial sobre pesos.

Para la CRPA hexagonal 2D, NO se debe sustituir el barrido w^H a(az,el) por una FFT 1D.
La FFT espacial 1D solo aplica directamente a arrays lineales uniformes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_fft_snapshot_matrix(snapshot_matrix: np.ndarray, sample_rate_hz: float = 1.0) -> pd.DataFrame:
    """FFT temporal por canal de antena.

    Devuelve potencia media espectral sobre todos los elementos.
    Útil para verificar tonos/interferencias en frecuencia normalizada.
    """
    num_snapshots = snapshot_matrix.shape[1]
    spectrum = np.fft.fftshift(np.fft.fft(snapshot_matrix, axis=1), axes=1)
    power_per_bin = np.mean(np.abs(spectrum) ** 2, axis=0)
    power_dB = 10.0 * np.log10(power_per_bin / (np.max(power_per_bin) + 1e-15) + 1e-12)
    freq_hz = np.fft.fftshift(np.fft.fftfreq(num_snapshots, d=1.0 / sample_rate_hz))
    return pd.DataFrame({"frequency_hz": freq_hz, "power_dB_normalized": power_dB})


def ula_fft_beam_pattern(weights: np.ndarray, n_fft: int = 2048) -> pd.DataFrame:
    """Patrón por FFT espacial SOLO para ULA con separación lambda/2.

    Para CRPA hexagonal no usar como patrón principal.
    """
    weights = np.asarray(weights).squeeze()
    padded = np.concatenate([weights.conj(), np.zeros(n_fft - len(weights), dtype=complex)])
    pattern = np.fft.fftshift(np.fft.fft(padded))
    pattern_dB = 20.0 * np.log10(np.abs(pattern) / (np.max(np.abs(pattern)) + 1e-15) + 1e-12)
    theta_rad = np.arcsin(np.linspace(-1.0, 1.0, n_fft))
    return pd.DataFrame({"theta_deg": np.rad2deg(theta_rad), "response_dB_normalized": pattern_dB})
