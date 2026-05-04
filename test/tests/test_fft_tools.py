import numpy as np

from crpa_sim.fft_tools import temporal_fft_snapshot_matrix, ula_fft_beam_pattern


def test_temporal_fft_snapshot_matrix_columns_and_length():
    rng = np.random.default_rng(123)
    X = rng.standard_normal((7, 256)) + 1j * rng.standard_normal((7, 256))

    df = temporal_fft_snapshot_matrix(X, sample_rate_hz=1.0)

    assert len(df) == 256
    assert "frequency_hz" in df.columns
    assert "power_dB_normalized" in df.columns
    assert df["power_dB_normalized"].max() <= 1e-9


def test_temporal_fft_detects_tone_near_expected_frequency():
    num_snapshots = 512
    freq = 0.125
    n = np.arange(num_snapshots)
    tone = np.exp(1j * 2.0 * np.pi * freq * n)
    X = np.tile(tone, (7, 1))

    df = temporal_fft_snapshot_matrix(X, sample_rate_hz=1.0)
    peak_freq = df.loc[df["power_dB_normalized"].idxmax(), "frequency_hz"]

    assert abs(peak_freq - freq) < 1.0 / num_snapshots


def test_ula_fft_beam_pattern_columns():
    weights = np.ones(7, dtype=complex) / 7.0

    df = ula_fft_beam_pattern(weights, n_fft=256)

    assert len(df) == 256
    assert "theta_deg" in df.columns
    assert "response_dB_normalized" in df.columns
