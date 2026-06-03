"""Analisis espectral temporal de los snapshots.

La FFT se calcula por canal de antena y despues se promedia la potencia. Sirve
como diagnostico de tonos/jammers en frecuencia, no interviene en el calculo
de pesos espaciales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_fft_snapshot_matrix(snapshot_matrix: np.ndarray, sample_rate_hz: float, fft_size: int | None = None) -> pd.DataFrame:
    """Calcula el espectro temporal medio de todos los canales de antena.

    Parametros:
        snapshot_matrix: Matriz compleja con forma
            (num_elements, num_snapshots).
        sample_rate_hz: Frecuencia de muestreo usada para construir el eje
            de frecuencias en Hz.
        fft_size: Tamano opcional de FFT. Si es None, se usa el numero de
            snapshots disponible.
    """
    if snapshot_matrix.ndim != 2:
        raise ValueError("snapshot_matrix debe tener forma (num_elements, num_snapshots).")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz debe ser > 0.")
    num_snapshots = snapshot_matrix.shape[1]
    n_fft = num_snapshots if fft_size is None else int(fft_size)

    if n_fft <= 0:
        raise ValueError("fft_size debe ser > 0.")
    
    spectrum = np.fft.fftshift(np.fft.fft(snapshot_matrix, n=n_fft, axis=1), axes=1)
    power = np.mean(np.abs(spectrum) ** 2, axis=0)
    max_power = np.max(power)

    if max_power == 0.0:
        power_normalized = np.zeros_like(power)
    else:
        power_normalized = power / max_power

    with np.errstate(divide="ignore"):
        power_dB = 10.0 * np.log10(power_normalized)

    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sample_rate_hz))
    
    return pd.DataFrame({"frequency_hz": freq_hz, "power_dB_normalized": power_dB})
