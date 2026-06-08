"""Analisis espectral temporal de los snapshots.

La FFT se calcula por canal de antena y despues se promedia la potencia. Sirve
como diagnostico de tonos/jammers en frecuencia, no interviene en el calculo
de pesos espaciales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_psd_snapshot_matrix(
    snapshot_matrix: np.ndarray,
    sample_rate_hz: float,
    fft_size: int | None = None,
) -> pd.DataFrame:
    """Calcula la PSD temporal media de los snapshots recibidos.

    La FFT se calcula por elemento de antena y despues se promedia la potencia
    entre canales:

        PSD = mean(|FFT(X)|^2) / (Fs * NFFT)

    Parametros:
        snapshot_matrix: Matriz compleja X con forma
            (num_elements, num_snapshots).
        sample_rate_hz: Frecuencia de muestreo en Hz.
        fft_size: Tamano de FFT. Si es None, usa num_snapshots.
    """
    if snapshot_matrix.ndim != 2:
        raise ValueError("snapshot_matrix debe tener forma (num_elements, num_snapshots).")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz debe ser mayor que cero.")

    num_snapshots = snapshot_matrix.shape[1]
    if num_snapshots <= 0:
        raise ValueError("num_snapshots debe ser mayor que cero.")

    n_fft = num_snapshots if fft_size is None else int(fft_size)
    if n_fft <= 0:
        raise ValueError("fft_size debe ser mayor que cero.")

    spectrum = np.fft.fftshift(
        np.fft.fft(snapshot_matrix, n=n_fft, axis=1),
        axes=1,
    )
    psd = np.mean(np.abs(spectrum) ** 2, axis=0) / (sample_rate_hz * n_fft)

    with np.errstate(divide="ignore"):
        psd_dB_Hz = 10.0 * np.log10(psd)

    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sample_rate_hz))

    return pd.DataFrame({"frequency_hz": freq_hz, "psd_dB_Hz": psd_dB_Hz})
