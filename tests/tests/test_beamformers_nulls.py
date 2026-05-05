from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from crpa_sim.array_model import steering_vector
from crpa_sim.beamformers import compute_lcmv_weights, compute_power_inversion_weights, compute_weights
from crpa_sim.jammers import build_jammer_case, generate_received_snapshot_matrix
from crpa_sim.null_metrics import compute_null_depth_dB, compute_null_metrics_for_jammers, measure_null_width_1d, summarize_null_metrics
from crpa_sim.patterns import make_scan_vectors

def test_power_inversion_and_lcmv(project_config, power_inversion_config, element_positions_m, rng):
    pi_jammers = build_jammer_case(power_inversion_config, rng)
    X_pi, _ = generate_received_snapshot_matrix(power_inversion_config, element_positions_m, pi_jammers, rng)
    w_pi = compute_power_inversion_weights(power_inversion_config, X_pi)
    c = np.zeros(power_inversion_config.array.num_elements, dtype=complex); c[0] = 1
    assert np.vdot(c, w_pi) == pytest.approx(1+0j, abs=1e-8)
    with pytest.raises(ValueError):
        compute_power_inversion_weights(replace(power_inversion_config, beamforming=replace(power_inversion_config.beamforming, power_inversion_reference_element=99)), X_pi)

    jammers = build_jammer_case(project_config, rng)
    X, _ = generate_received_snapshot_matrix(project_config, element_positions_m, jammers, rng)
    w = compute_lcmv_weights(project_config, X, element_positions_m, jammers)
    a_des = steering_vector(project_config, element_positions_m, project_config.beamforming.desired_azimuth_deg, project_config.beamforming.desired_elevation_deg)
    assert np.vdot(w, a_des) == pytest.approx(1+0j, abs=1e-6)
    for jammer in jammers:
        assert abs(np.vdot(w, steering_vector(project_config, element_positions_m, jammer.azimuth_deg, jammer.elevation_deg))) < 1e-6

def test_compute_weights_selector(project_config, power_inversion_config, element_positions_m, rng):
    for cfg in [project_config, power_inversion_config]:
        jammers = build_jammer_case(cfg, rng)
        X, _ = generate_received_snapshot_matrix(cfg, element_positions_m, jammers, rng)
        assert compute_weights(cfg, X, element_positions_m, jammers).shape == (7,)
    bad = replace(project_config, beamforming=replace(project_config.beamforming, algorithm="bad"))
    jammers = build_jammer_case(project_config, rng)
    X, _ = generate_received_snapshot_matrix(project_config, element_positions_m, jammers, rng)
    with pytest.raises(ValueError):
        compute_weights(bad, X, element_positions_m, jammers)

def test_null_metrics(project_config, element_positions_m, rng):
    assert measure_null_width_1d(np.array([-2,-1,0,1,2]), np.array([0,-20,-30,-20,0]), 0, -10) == pytest.approx(4.0)
    assert measure_null_width_1d(np.array([-1,0,1]), np.array([0,-5,0]), 0, -10) is None
    jammers = build_jammer_case(project_config, rng)
    X, _ = generate_received_snapshot_matrix(project_config, element_positions_m, jammers, rng)
    w = compute_lcmv_weights(project_config, X, element_positions_m, jammers)
    depth = compute_null_depth_dB(project_config, element_positions_m, w, jammers[0])
    assert depth <= -100
    az, el = make_scan_vectors(project_config)
    metrics, cuts = compute_null_metrics_for_jammers(project_config, element_positions_m, w, jammers, az, el, 1)
    assert len(metrics) == len(jammers) * len(project_config.scan.null_thresholds_dB)
    assert len(cuts) == len(jammers) * 2
    summary = summarize_null_metrics(metrics)
    assert not summary.empty
