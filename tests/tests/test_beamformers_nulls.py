from dataclasses import replace
import numpy as np
import pytest
from crpa_sim.array_model import steering_vector
from crpa_sim.beamformers import compute_lcmv_weights, compute_power_inversion_weights, compute_weights
from crpa_sim.jammers import build_jammer_case, generate_received_snapshot_matrix
from crpa_sim.null_metrics import circular_azimuth_width_deg, compute_null_depth_dB, compute_null_metrics_for_jammers, measure_null_region_2d
from crpa_sim.patterns import make_scan_vectors


def _ideal_config(config):
    """Devuelve una configuracion ideal para tests con aserciones ideales."""
    return replace(config, array=replace(config.array, steering_model="ideal", measured_steering_file=None))


def test_power_inversion_and_lcmv(project_config, power_inversion_config, element_positions_m, rng):
    """Valida restricciones principales de Power Inversion y LCMV.

    Parametros:
        project_config: Configuracion LCMV base.
        power_inversion_config: Configuracion base con Power Inversion.
        element_positions_m: Posiciones XYZ del array.
        rng: Generador aleatorio determinista.
    """
    pi_jammers = build_jammer_case(power_inversion_config, rng)
    X_pi, _ = generate_received_snapshot_matrix(power_inversion_config, element_positions_m, pi_jammers, rng)
    w_pi = compute_power_inversion_weights(power_inversion_config, X_pi)
    c = np.zeros(power_inversion_config.array.num_elements, dtype=complex); c[0] = 1
    assert np.vdot(c, w_pi) == pytest.approx(1+0j, abs=1e-8)
    with pytest.raises(ValueError):
        compute_power_inversion_weights(replace(power_inversion_config, beamforming=replace(power_inversion_config.beamforming, power_inversion_reference_element=99)), X_pi)

    lcmv_config = _ideal_config(project_config)
    jammers = build_jammer_case(lcmv_config, rng)
    X, _ = generate_received_snapshot_matrix(lcmv_config, element_positions_m, jammers, rng)
    w = compute_lcmv_weights(lcmv_config, X, element_positions_m, jammers)
    a_des = steering_vector(lcmv_config, element_positions_m, lcmv_config.beamforming.desired_azimuth_deg, lcmv_config.beamforming.desired_elevation_deg)
    assert np.vdot(w, a_des) == pytest.approx(1+0j, abs=1e-6)
    for jammer in jammers:
        assert abs(np.vdot(w, steering_vector(lcmv_config, element_positions_m, jammer.azimuth_deg, jammer.elevation_deg))) < 1e-6

def test_compute_weights_selector(project_config, power_inversion_config, element_positions_m, rng):
    """Comprueba el selector de algoritmos de pesos.

    Parametros:
        project_config: Configuracion LCMV base.
        power_inversion_config: Configuracion base con Power Inversion.
        element_positions_m: Posiciones XYZ del array.
        rng: Generador aleatorio determinista.
    """
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
    """Verifica profundidad y region 2D de metricas de nulos.

    Parametros:
        project_config: Configuracion base de simulacion.
        element_positions_m: Posiciones XYZ del array.
        rng: Generador aleatorio determinista.
    """
    cfg = _ideal_config(project_config)
    jammers = build_jammer_case(cfg, rng)
    X, _ = generate_received_snapshot_matrix(cfg, element_positions_m, jammers, rng)
    w = compute_lcmv_weights(cfg, X, element_positions_m, jammers)
    depth = compute_null_depth_dB(cfg, element_positions_m, w, jammers[0])
    assert depth <= -100
    az, el = make_scan_vectors(cfg)
    metrics, cuts = compute_null_metrics_for_jammers(cfg, element_positions_m, w, jammers, az, el, 1)
    assert len(metrics) == len(jammers) * len(cfg.scan.null_thresholds_dB)
    assert len(cuts) == len(jammers) * 2
    assert {
        "montecarlo_index",
        "algorithm",
        "doa_mode",
        "jammer_signal_type",
        "null_area_cells",
        "null_area_deg2",
        "null_width_azimuth",
        "null_width_elevation",
    }.issubset(metrics.columns)


def test_lcmv_rejects_more_jammers_than_constraints(project_config, element_positions_m, rng):
    """Comprueba el limite de jammers admisible por LCMV.

    Parametros:
        project_config: Configuracion LCMV base.
        element_positions_m: Posiciones XYZ del array.
        rng: Generador aleatorio determinista.
    """
    jammers = build_jammer_case(project_config, rng)
    too_many = [jammers[0]] * project_config.array.num_elements
    X, _ = generate_received_snapshot_matrix(project_config, element_positions_m, too_many[:1], rng)
    with pytest.raises(ValueError):
        compute_lcmv_weights(project_config, X, element_positions_m, too_many)


def test_null_depth_zero_gain_returns_negative_infinity(project_config, element_positions_m):
    """Cubre el caso de ganancia nula en la direccion del jammer.

    Parametros:
        project_config: Configuracion base de simulacion.
        element_positions_m: Posiciones XYZ del array.
    """
    zero_weights = np.zeros(project_config.array.num_elements, dtype=complex)
    jammer = build_jammer_case(project_config, np.random.default_rng(1))[0]
    assert np.isneginf(compute_null_depth_dB(project_config, element_positions_m, zero_weights, jammer))


def test_null_region_2d_edge_cases_and_azimuth_wrap():
    """Cubre regiones 2D degeneradas y continuidad circular en azimut.

    Parametros:
        No recibe parametros.
    """
    assert circular_azimuth_width_deg(np.array([])) == pytest.approx(0.0)
    assert circular_azimuth_width_deg(np.array([42.0])) == pytest.approx(0.0)
    assert circular_azimuth_width_deg(np.array([359.0, 0.0, 1.0])) == pytest.approx(2.0)

    az_grid, el_grid = np.meshgrid(
        np.array([-180.0, -179.0, 179.0, 180.0]),
        np.array([0.0, 1.0]),
        indexing="xy",
    )
    response = np.zeros_like(az_grid, dtype=float)
    response[0, :] = -30.0

    region = measure_null_region_2d(
        az_grid,
        el_grid,
        response,
        jammer_azimuth_deg=-179.5,
        jammer_elevation_deg=0.0,
        threshold_dB=-20.0,
    )
    assert region["null_area_cells_2d"] == 4
    assert region["null_area_deg2_2d"] == pytest.approx(4.0)
    assert region["null_width_azimuth_2d_deg"] == pytest.approx(2.0)
    assert region["null_width_elevation_2d_deg"] == pytest.approx(1.0)

    empty_region = measure_null_region_2d(
        az_grid,
        el_grid,
        np.zeros_like(az_grid, dtype=float),
        jammer_azimuth_deg=0.0,
        jammer_elevation_deg=0.0,
        threshold_dB=-20.0,
    )
    assert empty_region == {
        "null_area_cells_2d": None,
        "null_area_deg2_2d": None,
        "null_width_azimuth_2d_deg": None,
        "null_width_elevation_2d_deg": None,
    }

