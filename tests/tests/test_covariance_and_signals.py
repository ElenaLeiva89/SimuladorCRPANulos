from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crpa_sim.covariance import add_diagonal_loading, compute_sample_covariance, invert_covariance
from crpa_sim.jammers import (
    build_jammer_case,
    generate_complex_noise,
    generate_jammer_baseband_signal,
    jammer_power_from_jnr,
)
from crpa_sim.config import JammerInstance


def test_sample_covariance_is_hermitian_positive_semidefinite(rng):
    X = rng.normal(size=(7, 128)) + 1j * rng.normal(size=(7, 128))
    R = compute_sample_covariance(X)

    np.testing.assert_allclose(R, R.conj().T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(R)
    assert eigvals.min() >= -1e-10


def test_covariance_rejects_invalid_snapshot_shapes():
    with pytest.raises(ValueError, match="forma"):
        compute_sample_covariance(np.zeros(7))
    with pytest.raises(ValueError, match="mayor que cero"):
        compute_sample_covariance(np.zeros((7, 0), dtype=complex))


def test_diagonal_loading_and_pseudoinverse_are_finite_for_singular_matrix():
    R = np.ones((7, 7), dtype=complex)
    R_loaded = add_diagonal_loading(R, 1e-3)
    R_inv = invert_covariance(R, 1e-3)

    assert np.all(np.isfinite(R_loaded))
    assert np.all(np.isfinite(R_inv))
    assert np.linalg.cond(R_loaded) < np.linalg.cond(R + 1e-15 * np.eye(7))


def test_jammer_power_matches_jnr_definition():
    assert np.isclose(jammer_power_from_jnr(2.0, 10.0), 20.0)
    assert np.isclose(jammer_power_from_jnr(1.0, 0.0), 1.0)


@pytest.mark.parametrize("signal_type", ["tone", "complex_gaussian", "chirp"])
def test_jammer_waveforms_have_expected_power(signal_type, rng):
    jammer = JammerInstance(
        name="J",
        azimuth_deg=40.0,
        elevation_deg=10.0,
        jnr_dB=20.0,
        signal_type=signal_type,
        normalized_frequency=0.05,
        chirp_frequency=0.05,
    )
    power = 100.0
    s = generate_jammer_baseband_signal(jammer, 4096, power, rng)

    assert s.shape == (4096,)
    assert np.all(np.isfinite(s))
    assert np.isclose(np.mean(np.abs(s) ** 2), power, rtol=0.12)


def test_noise_power_and_shape_match_configuration(fast_config, rng):
    noise = generate_complex_noise(fast_config, rng)
    assert noise.shape == (fast_config.array.num_elements, fast_config.signal.num_snapshots)
    assert np.isclose(np.mean(np.abs(noise) ** 2), fast_config.noise.noise_power_linear, rtol=0.20)


def test_variable_doa_generation_stays_inside_configured_ranges(fast_config):
    cfg = replace(fast_config, simulation=replace(fast_config.simulation, doa_mode="variable"))
    rng = np.random.default_rng(42)
    jammers = build_jammer_case(cfg, rng)

    az_min, az_max = cfg.jammer.variable_doa_azimuth_range_deg
    el_min, el_max = cfg.jammer.variable_doa_elevation_range_deg
    assert all(az_min <= j.azimuth_deg <= az_max for j in jammers)
    assert all(el_min <= j.elevation_deg <= el_max for j in jammers)
