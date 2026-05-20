from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from main import run_project
from conftest import write_config_json


def test_run_project_creates_core_artifacts_without_plots(fast_config, tmp_path):
    config_path = write_config_json(fast_config, tmp_path / "input_config_test.json")

    run_project(config_path)

    out = Path(fast_config.output.output_dir)
    assert (out / "config_used.json").exists()
    assert (out / "run_log.txt").exists()
    assert (out / "null_metrics_by_jammer.csv").exists()
    assert (out / "null_metrics_summary.csv").exists()
    assert (out / "output_data" / "matrices_complex.npz").exists()
    assert not list(out.glob("*.png"))

    metrics = pd.read_csv(out / "null_metrics_by_jammer.csv", sep=fast_config.output.csv_separator, decimal=fast_config.output.csv_decimal)
    assert {"montecarlo_index", "algorithm", "doa_mode", "jammer_index", "jammer_signal_type"}.issubset(metrics.columns)

    summary = pd.read_csv(out / "null_metrics_summary.csv", sep=fast_config.output.csv_separator, decimal=fast_config.output.csv_decimal)
    assert not summary.empty
    assert {"jammer_name", "attenuation_threshold_dB", "null_depth_dB", "null_width_azimuth_deg", "null_width_elevation_deg"}.issubset(summary.columns)


def test_run_project_resolves_output_dir_template(fast_config, tmp_path):
    """Comprueba la plantilla results_{algorithm}_{doa_mode}_{steering_model}.

    Parametros:
        fast_config: Configuracion ligera de simulacion.
        tmp_path: Directorio temporal de pytest.
    """
    cfg = replace(
        fast_config,
        beamforming=replace(fast_config.beamforming, algorithm="lcmv"),
        simulation=replace(fast_config.simulation, doa_mode="fixed"),
        output=replace(
            fast_config.output,
            output_dir=str(tmp_path / "results_{algorithm}_{doa_mode}_{steering_model}"),
        ),
    )
    config_path = write_config_json(cfg, tmp_path / "input_config_template.json")

    run_project(config_path)

    out = tmp_path / "results_lcmv_fixed_ideal"
    assert out.exists()
    assert (out / "config_used.json").exists()
    assert (out / "null_metrics_summary.csv").exists()


@pytest.mark.parametrize("algorithm", ["lcmv", "power_inversion"])
def test_run_project_variable_doa_creates_metrics_for_each_montecarlo(algorithm, fast_config, tmp_path):
    """Ejecuta DoA variable end-to-end con cada algoritmo soportado.

    Parametros:
        algorithm: Algoritmo de beamforming bajo prueba.
        fast_config: Configuracion ligera de simulacion.
        tmp_path: Directorio temporal de pytest.
    """
    cfg = replace(
        fast_config,
        simulation=replace(fast_config.simulation, doa_mode="variable", num_montecarlo=3, random_seed=31415),
        beamforming=replace(fast_config.beamforming, algorithm=algorithm, power_inversion_reference_element=0),
        jammer=replace(
            fast_config.jammer,
            num_jammers=1,
            variable_doa_azimuth_range_deg=(-45.0, 45.0),
            variable_doa_elevation_range_deg=(15.0, 75.0),
        ),
        output=replace(fast_config.output, output_dir=str(tmp_path / f"results_variable_{algorithm}")),
    )
    config_path = write_config_json(cfg, tmp_path / f"input_config_variable_{algorithm}.json")

    run_project(config_path)

    out = Path(cfg.output.output_dir)
    metrics = pd.read_csv(out / "null_metrics_by_jammer.csv", sep=cfg.output.csv_separator, decimal=cfg.output.csv_decimal)
    summary = pd.read_csv(out / "null_metrics_summary.csv", sep=cfg.output.csv_separator, decimal=cfg.output.csv_decimal)

    assert set(metrics["doa_mode"]) == {"variable"}
    assert set(metrics["algorithm"]) == {algorithm}
    assert set(metrics["montecarlo_index"]) == {1, 2, 3}
    assert len(metrics) == cfg.simulation.num_montecarlo * cfg.jammer.num_jammers * len(cfg.scan.null_thresholds_dB)
    assert metrics["jammer_azimuth_deg"].between(-45.0, 45.0).all()
    assert metrics["jammer_elevation_deg"].between(15.0, 75.0).all()
    assert metrics.groupby("montecarlo_index")["jammer_azimuth_deg"].first().nunique() > 1
    assert not summary.empty
