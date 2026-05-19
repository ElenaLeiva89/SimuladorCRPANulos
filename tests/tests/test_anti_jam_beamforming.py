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
    X, table = generate_received_snapshot_matrix(config, positions, jammers, rng)
    return positions, jammers, X, table


def test_lcmv_enforces_unit_desired_gain_and_deep_jammer_nulls(fast_config, rng):
    cfg = replace(
        fast_config,
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
        fast_config,
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
        fast_config,
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
        fast_config,
        beamforming=replace(fast_config.beamforming, algorithm="lcmv"),
        jammer=replace(fast_config.jammer, num_jammers=2, jnr_dB=30.0),
    )
    positions, jammers, X, _ = _make_case(cfg, rng)
    w = compute_weights(cfg, X, positions, jammers)
    az, el = make_scan_vectors(cfg)

    metrics, cuts = compute_null_metrics_for_jammers(cfg, positions, w, jammers, az, el, montecarlo_index=1)

    assert len(metrics) == len(jammers) * len(cfg.scan.null_thresholds_dB)
    assert set(metrics["attenuation_threshold_dB"]) == set(cfg.scan.null_thresholds_dB)
    assert all(name.endswith(("azimuth_cut", "elevation_cut")) for name in cuts)
    assert metrics["null_depth_dB"].max() < -60.0
