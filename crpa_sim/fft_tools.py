"""fft_tools.py
FFT temporal opcional para inspeccion espectral de snapshots.
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
    num_snapshots = snapshot_matrix.shape[1]
    n_fft = num_snapshots if fft_size is None else int(fft_size)
    spectrum = np.fft.fftshift(np.fft.fft(snapshot_matrix, n=n_fft, axis=1), axes=1)
    power = np.mean(np.abs(spectrum) ** 2, axis=0)
    power_dB = 10.0 * np.log10(power / (np.max(power) + 1e-15) + 1e-12)
    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sample_rate_hz))
    return pd.DataFrame({"frequency_hz": freq_hz, "power_dB_normalized": power_dB})
