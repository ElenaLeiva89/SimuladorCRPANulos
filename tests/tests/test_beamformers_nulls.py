from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from crpa_sim.array_model import steering_vector
from crpa_sim.beamformers import compute_lcmv_weights, compute_power_inversion_weights, compute_weights
from crpa_sim.jammers import build_jammer_case, generate_received_snapshot_matrix
from crpa_sim.null_metrics import circular_azimuth_width_deg, compute_null_depth_dB, compute_null_metrics_for_jammers, measure_null_region_2d, measure_null_width_1d, summarize_null_metrics
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
    """Verifica profundidad, anchura y resumen de metricas de nulos.

    Parametros:
        project_config: Configuracion base de simulacion.
        element_positions_m: Posiciones XYZ del array.
        rng: Generador aleatorio determinista.
    """
    assert measure_null_width_1d(np.array([-2,-1,0,1,2]), np.array([0,-20,-30,-20,0]), 0, -10) == pytest.approx(4.0)
    assert measure_null_width_1d(np.array([-1,0,1]), np.array([0,-5,0]), 0, -10) is None
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
        "num_jammers",
        "jammer_index",
        "jammer_signal_type",
        "null_area_cells_2d",
        "null_area_deg2_2d",
        "null_width_azimuth_2d_deg",
        "null_width_elevation_2d_deg",
    }.issubset(metrics.columns)
    summary = summarize_null_metrics(metrics)
    assert not summary.empty


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


def test_null_width_edge_cases_and_floor(project_config, element_positions_m):
    """Cubre nulos en bordes, umbrales exactos y suelo de profundidad.

    Parametros:
        project_config: Configuracion base de simulacion.
        element_positions_m: Posiciones XYZ del array.
    """
    assert measure_null_width_1d(np.array([0, 1, 2]), np.array([-20, -20, 0]), 0, -20) == pytest.approx(2.0)
    assert measure_null_width_1d(np.array([0, 1, 2]), np.array([0, -20, -20]), 2, -20) == pytest.approx(2.0)
    zero_weights = np.zeros(project_config.array.num_elements, dtype=complex)
    jammer = build_jammer_case(project_config, np.random.default_rng(1))[0]
    assert compute_null_depth_dB(project_config, element_positions_m, zero_weights, jammer, floor_dB=-80.0) == pytest.approx(-80.0)


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
    assert region["null_width_elevation_2d_deg"] == pytest.approx(0.0)

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


def test_null_metrics_empty_inputs():
    """Comprueba que el resumen tolera tablas vacias con columnas esperadas.

    Parametros:
        No recibe parametros.
    """
    metrics = pd.DataFrame(
        columns=[
            "jammer_name",
            "attenuation_threshold_dB",
            "null_depth_dB",
            "null_width_azimuth_2d_deg",
            "null_width_elevation_2d_deg",
        ]
    )
    summary = summarize_null_metrics(metrics)
    assert list(summary.columns) == list(metrics.columns)
    assert summary.empty


def test_null_metrics_summary_rounds_configured_decimal_columns():
    """Comprueba que el resumen expone metricas con dos decimales.

    Parametros:
        No recibe parametros.
    """
    metrics = pd.DataFrame(
        [
            {
                "jammer_name": "J1",
                "attenuation_threshold_dB": -20.0,
                "null_depth_dB": -33.333,
                "null_width_azimuth_deg": 1.111,
                "null_width_elevation_deg": 2.222,
                "null_area_cells_2d": 3,
                "null_area_deg2_2d": 4.444,
                "null_width_azimuth_2d_deg": 5.555,
                "null_width_elevation_2d_deg": 6.666,
            },
            {
                "jammer_name": "J1",
                "attenuation_threshold_dB": -20.0,
                "null_depth_dB": -33.336,
                "null_width_azimuth_deg": 1.116,
                "null_width_elevation_deg": 2.226,
                "null_area_cells_2d": 5,
                "null_area_deg2_2d": 4.446,
                "null_width_azimuth_2d_deg": 5.556,
                "null_width_elevation_2d_deg": 6.667,
            },
        ]
    )

    summary = summarize_null_metrics(metrics)

    rounded_columns = [
        "null_depth_dB",
        "null_width_azimuth_2d_deg",
        "null_width_elevation_2d_deg",
    ]
    for column in rounded_columns:
        assert summary.loc[0, column] == pytest.approx(round(summary.loc[0, column], 2))
