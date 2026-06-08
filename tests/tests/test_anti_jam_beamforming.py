from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crpa_sim.array_model import create_crpa_geometry, steering_vector
from crpa_sim.beamformers import compute_lcmv_weights, compute_power_inversion_weights, compute_weights
from crpa_sim.config import JammerInstance
from crpa_sim.jammers import build_jammer_case, generate_received_snapshot_matrix
from crpa_sim.null_metrics import compute_null_depth_dB, compute_null_metrics_for_jammers
from crpa_sim.patterns import conventional_weights, make_scan_vectors


def _make_case(config, rng):
    positions = create_crpa_geometry(config.array, config.element_spacing_m)
    jammers = build_jammer_case(config, rng)
    X, table, _, _ = generate_received_snapshot_matrix(config, positions, jammers, rng)
    return positions, jammers, X, table


def _ideal_config(config):
    """Devuelve una configuracion ideal para tests con aserciones ideales."""
    return replace(config, array=replace(config.array, steering_model="ideal", measured_steering_file=None))


def test_lcmv_enforces_unit_desired_gain_and_deep_jammer_nulls(fast_config, rng):
    cfg = replace(
        _ideal_config(fast_config),
        beamforming=replace(fast_config.beamforming, algorithm="lcmv"),
        jammer=replace(fast_config.jammer, num_jammers=2, jnr_dB=30.0),
    )
    positions, jammers, X, _ = _make_case(cfg, rng)
    w = compute_lcmv_weights(cfg, X, positions, jammers)

    a_des = steering_vector(cfg, positions, cfg.beamforming.desired_azimuth_deg, cfg.beamforming.desired_elevation_deg)
    assert np.isclose(np.vdot(w, a_des), 1.0 + 0j, atol=1e-6)
    for jammer in jammers:
        null_depth = compute_null_depth_dB(cfg, positions, w, jammer)
        assert null_depth < -70.0


def test_power_inversion_places_a_null_for_strong_single_tone_jammer(fast_config, rng):
    cfg = replace(
        _ideal_config(fast_config),
        beamforming=replace(fast_config.beamforming, algorithm="power_inversion", power_inversion_reference_element=1),
        jammer=replace(fast_config.jammer, num_jammers=1, jnr_dB=45.0),
    )
    positions, jammers, X, _ = _make_case(cfg, rng)
    w = compute_power_inversion_weights(cfg, X)

    null_depth = compute_null_depth_dB(cfg, positions, w, jammers[0])
    assert null_depth < -20.0
    assert np.all(np.isfinite(w))


def test_adaptive_weights_improve_attenuation_over_conventional_for_jammer(fast_config, rng):
    cfg = replace(
        _ideal_config(fast_config),
        beamforming=replace(fast_config.beamforming, algorithm="power_inversion", power_inversion_reference_element=1),
        jammer=replace(fast_config.jammer, num_jammers=1, jnr_dB=40.0),
    )
    positions, jammers, X, _ = _make_case(cfg, rng)
    w_adapt = compute_weights(cfg, X, positions, jammers)
    w_conv = conventional_weights(cfg, positions)

    adapt_depth = compute_null_depth_dB(cfg, positions, w_adapt, jammers[0])
    conv_depth = compute_null_depth_dB(cfg, positions, w_conv, jammers[0])
    assert adapt_depth <= conv_depth - 10.0


def test_lcmv_rejects_too_many_jammer_constraints(fast_config, rng):
    cfg = replace(fast_config, jammer=replace(fast_config.jammer, num_jammers=1))
    positions, _, X, _ = _make_case(cfg, rng)
    too_many = [
        JammerInstance(f"J{i}", azimuth_deg=i * 20.0, elevation_deg=20.0, jnr_dB=30.0, signal_type="tone")
        for i in range(cfg.array.num_elements)
    ]
    with pytest.raises(ValueError, match="numero de jammers"):
        compute_lcmv_weights(cfg, X, positions, too_many)


def test_null_metrics_table_has_one_row_per_jammer_and_threshold(fast_config, rng):
    cfg = replace(
        _ideal_config(fast_config),
        beamforming=replace(fast_config.beamforming, algorithm="lcmv"),
        jammer=replace(fast_config.jammer, num_jammers=2, jnr_dB=30.0),
    )
    positions, jammers, X, _ = _make_case(cfg, rng)
    w = compute_weights(cfg, X, positions, jammers)
    az, el = make_scan_vectors(cfg)

    metrics, cuts = compute_null_metrics_for_jammers(cfg, positions, w, jammers, az, el, montecarlo_index=1)

    assert len(metrics) == len(jammers) * len(cfg.scan.null_thresholds_dB)
    assert set(metrics["attenuation_threshold_dB"]) == set(cfg.scan.null_thresholds_dB)
    assert set(metrics["montecarlo_index"]) == {1}
    assert set(metrics["algorithm"]) == {"lcmv"}
    metric_directions = set(zip(metrics["jammer_azimuth_deg"], metrics["jammer_elevation_deg"]))
    jammer_directions = {(jammer.azimuth_deg, jammer.elevation_deg) for jammer in jammers}
    assert metric_directions == jammer_directions
    assert all(name.endswith(("azimuth_cut", "elevation_cut")) for name in cuts)
    assert metrics["null_depth_dB"].max() < -60.0


def test_lcmv_null_is_sensitive_to_doa_estimation_error(fast_config, rng):
    cfg = replace(
        _ideal_config(fast_config),
        beamforming=replace(fast_config.beamforming, algorithm="lcmv"),
        jammer=replace(fast_config.jammer, num_jammers=1, jnr_dB=35.0),
    )
    positions, true_jammers, X, _ = _make_case(cfg, rng)
    true_jammer = true_jammers[0]
    estimated_jammer = replace(true_jammer, elevation_deg=true_jammer.elevation_deg + 2.0)

    w = compute_lcmv_weights(cfg, X, positions, [estimated_jammer])

    estimated_depth = compute_null_depth_dB(cfg, positions, w, estimated_jammer)
    true_depth = compute_null_depth_dB(cfg, positions, w, true_jammer)
    assert estimated_depth < -90.0
    assert true_depth > estimated_depth + 40.0


def test_calibration_errors_degrade_ideal_lcmv_null(fast_config, rng):
    cfg = replace(
        _ideal_config(fast_config),
        beamforming=replace(fast_config.beamforming, algorithm="lcmv"),
        jammer=replace(fast_config.jammer, num_jammers=1, jnr_dB=35.0),
    )
    positions, jammers, X, _ = _make_case(cfg, rng)
    w = compute_lcmv_weights(cfg, X, positions, jammers)
    ideal_depth = compute_null_depth_dB(cfg, positions, w, jammers[0])

    ideal_manifold = steering_vector(cfg, positions, jammers[0].azimuth_deg, jammers[0].elevation_deg)
    gain_error = 1.0 + 0.03 * np.linspace(-1.0, 1.0, cfg.array.num_elements)
    phase_error_rad = 0.02 * np.arange(cfg.array.num_elements)
    calibrated_manifold = gain_error * np.exp(1j * phase_error_rad) * ideal_manifold
    calibrated_depth = 20.0 * np.log10(abs(np.vdot(w, calibrated_manifold)) + 1e-12)

    assert ideal_depth < -90.0
    assert calibrated_depth > ideal_depth + 40.0
    assert calibrated_depth < -20.0


def test_mutual_coupling_degrades_ideal_lcmv_null(fast_config, rng):
    cfg = replace(
        _ideal_config(fast_config),
        beamforming=replace(fast_config.beamforming, algorithm="lcmv"),
        jammer=replace(fast_config.jammer, num_jammers=1, jnr_dB=35.0),
    )
    positions, jammers, X, _ = _make_case(cfg, rng)
    w = compute_lcmv_weights(cfg, X, positions, jammers)
    ideal_depth = compute_null_depth_dB(cfg, positions, w, jammers[0])

    ideal_manifold = steering_vector(cfg, positions, jammers[0].azimuth_deg, jammers[0].elevation_deg)
    coupling = np.eye(cfg.array.num_elements, dtype=complex)
    for idx in range(cfg.array.num_elements):
        coupling[idx, (idx - 1) % cfg.array.num_elements] += 0.05 * np.exp(1j * 0.2)
        coupling[idx, (idx + 1) % cfg.array.num_elements] += 0.05 * np.exp(-1j * 0.2)
    coupled_manifold = coupling @ ideal_manifold
    coupled_depth = 20.0 * np.log10(abs(np.vdot(w, coupled_manifold)) + 1e-12)

    assert ideal_depth < -90.0
    assert coupled_depth > ideal_depth + 40.0
    assert coupled_depth < -30.0


def test_adaptive_weights_remain_finite_with_very_few_snapshots(fast_config, rng):
    cfg = replace(
        _ideal_config(fast_config),
        signal=replace(fast_config.signal, num_snapshots=1, fft_size=8),
        beamforming=replace(fast_config.beamforming, diagonal_loading_factor=1e-3),
        jammer=replace(fast_config.jammer, num_jammers=1, jnr_dB=35.0),
    )
    positions, jammers, X, _ = _make_case(cfg, rng)
    assert X.shape == (cfg.array.num_elements, 1)

    pi_cfg = replace(cfg, beamforming=replace(cfg.beamforming, algorithm="power_inversion"))
    lcmv_cfg = replace(cfg, beamforming=replace(cfg.beamforming, algorithm="lcmv"))

    w_pi = compute_power_inversion_weights(pi_cfg, X)
    w_lcmv = compute_lcmv_weights(lcmv_cfg, X, positions, jammers)

    assert np.all(np.isfinite(w_pi))
    assert np.all(np.isfinite(w_lcmv))
    desired = steering_vector(lcmv_cfg, positions, lcmv_cfg.beamforming.desired_azimuth_deg, lcmv_cfg.beamforming.desired_elevation_deg)
    assert np.vdot(w_lcmv, desired) == pytest.approx(1.0 + 0.0j, abs=1e-6)
    assert compute_null_depth_dB(lcmv_cfg, positions, w_lcmv, jammers[0]) < -90.0


def test_power_inversion_is_sensitive_to_diagonal_loading(fast_config, rng):
    data_cfg = replace(
        _ideal_config(fast_config),
        signal=replace(fast_config.signal, num_snapshots=2, fft_size=8),
        beamforming=replace(
            fast_config.beamforming,
            algorithm="power_inversion",
            power_inversion_reference_element=0,
            diagonal_loading_factor=1e-3,
        ),
        jammer=replace(fast_config.jammer, num_jammers=1, jnr_dB=40.0),
    )
    positions, jammers, X, _ = _make_case(data_cfg, rng)
    low_loading_cfg = replace(data_cfg, beamforming=replace(data_cfg.beamforming, diagonal_loading_factor=1e-6))
    high_loading_cfg = replace(data_cfg, beamforming=replace(data_cfg.beamforming, diagonal_loading_factor=1.0))

    w_low = compute_power_inversion_weights(low_loading_cfg, X)
    w_high = compute_power_inversion_weights(high_loading_cfg, X)
    depth_low = compute_null_depth_dB(low_loading_cfg, positions, w_low, jammers[0])
    depth_high = compute_null_depth_dB(high_loading_cfg, positions, w_high, jammers[0])

    assert np.all(np.isfinite(w_low))
    assert np.all(np.isfinite(w_high))
    assert not np.allclose(w_low, w_high)
    assert depth_low < depth_high - 10.0
