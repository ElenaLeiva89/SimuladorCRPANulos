from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from main import run_project
from conftest import write_config_json, write_synthetic_measured_mat_files


def _with_steering_model(config, steering_model, tmp_path):
    phase_file = None
    amplitude_file = None
    if steering_model == "measured":
        phase_path, amplitude_path = write_synthetic_measured_mat_files(tmp_path)
        phase_file = str(phase_path)
        amplitude_file = str(amplitude_path)
    return replace(
        config,
        array=replace(
            config.array,
            steering_model=steering_model,
            measured_phase_mat_file=phase_file,
            measured_amplitude_mat_file=amplitude_file,
        ),
    )


def test_run_project_creates_core_artifacts_without_plots(fast_config, tmp_path):
    config_path = write_config_json(fast_config, tmp_path / "input_config_test.json")

    run_project(config_path)

    out = Path(fast_config.output.output_dir)
    assert (out / "config_used.json").exists()
    assert (out / "run_log.txt").exists()
    assert (out / "null_metrics_by_jammer.csv").exists()
    assert (out / "output_data" / "matrices_complex.npz").exists()
    assert not list(out.glob("*.png"))

    metrics = pd.read_csv(out / "null_metrics_by_jammer.csv", sep=fast_config.output.csv_separator, decimal=fast_config.output.csv_decimal)
    assert {"montecarlo_index", "algorithm", "doa_mode", "jammer_signal_type", "null_width_azimuth", "null_width_elevation"}.issubset(metrics.columns)


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

    out = tmp_path / f"results_lcmv_fixed_{cfg.array.steering_model}"
    assert out.exists()
    assert (out / "config_used.json").exists()
    assert (out / "null_metrics_by_jammer.csv").exists()


@pytest.mark.parametrize("steering_model", ["ideal", "measured"])
@pytest.mark.parametrize("doa_mode", ["fixed", "variable"])
@pytest.mark.parametrize("algorithm", ["lcmv", "lcmvq", "power_inversion"])
def test_run_project_supports_steering_doa_algorithm_combinations(
    steering_model,
    doa_mode,
    algorithm,
    fast_config,
    tmp_path,
):
    """Ejecuta todas las combinaciones soportadas de steering, DoA y algoritmo.

    Parametros:
        steering_model: Modelo de steering bajo prueba.
        doa_mode: Modo de DoA bajo prueba.
        algorithm: Algoritmo de beamforming bajo prueba.
        fast_config: Configuracion ligera de simulacion.
        tmp_path: Directorio temporal de pytest.
    """
    base = _with_steering_model(fast_config, steering_model, tmp_path)
    cfg = replace(
        base,
        signal=replace(base.signal, num_snapshots=64, fft_size=64),
        simulation=replace(base.simulation, doa_mode=doa_mode, num_montecarlo=1),
        beamforming=replace(base.beamforming, algorithm=algorithm, power_inversion_reference_element=0),
        scan=replace(
            base.scan,
            azimuth_scan_min_deg=0.0,
            azimuth_scan_max_deg=40.0,
            azimuth_scan_step_deg=20.0,
            elevation_scan_min_deg=0.0,
            elevation_scan_max_deg=90.0,
            elevation_scan_step_deg=45.0,
            null_thresholds_dB=[-10.0],
        ),
        jammer=replace(
            base.jammer,
            num_jammers=1,
            variable_doa_azimuth_range_deg=(0.0, 40.0),
            variable_doa_elevation_range_deg=(10.0, 80.0),
        ),
        output=replace(
            base.output,
            output_dir=str(tmp_path / "results_{algorithm}_{doa_mode}_{steering_model}"),
            save_csv=True,
            save_npz=True,
            save_plots=False,
        ),
    )
    config_path = write_config_json(cfg, tmp_path / f"input_{steering_model}_{doa_mode}_{algorithm}.json")

    run_project(config_path)

    out = tmp_path / f"results_{algorithm}_{doa_mode}_{steering_model}"
    metrics = pd.read_csv(out / "null_metrics_by_jammer.csv", sep=cfg.output.csv_separator, decimal=cfg.output.csv_decimal)

    assert (out / "run_log.txt").exists()
    assert (out / "output_data" / "matrices_complex.npz").exists()
    assert set(metrics["algorithm"]) == {algorithm}
    assert set(metrics["doa_mode"]) == {doa_mode}
    assert len(metrics) == cfg.jammer.num_jammers * len(cfg.scan.null_thresholds_dB)
    assert metrics["null_depth_dB"].notna().all()


@pytest.mark.parametrize("algorithm", ["lcmv", "lcmvq", "power_inversion"])
def test_run_project_variable_doa_creates_metrics_for_each_montecarlo(algorithm, fast_config, tmp_path):
    """Ejecuta DoA variable end-to-end con cada algoritmo soportado.

    Parametros:
        algorithm: Algoritmo de beamforming bajo prueba.
        fast_config: Configuracion ligera de simulacion.
        tmp_path: Directorio temporal de pytest.
    """
    cfg = replace(
        fast_config,
        simulation=replace(fast_config.simulation, doa_mode="variable", num_montecarlo=3),
        beamforming=replace(fast_config.beamforming, algorithm=algorithm, power_inversion_reference_element=0),
        jammer=replace(
            fast_config.jammer,
            num_jammers=1,
            variable_doa_azimuth_range_deg=(0.0, 90.0),
            variable_doa_elevation_range_deg=(15.0, 75.0),
        ),
        output=replace(fast_config.output, output_dir=str(tmp_path / f"results_variable_{algorithm}")),
    )
    config_path = write_config_json(cfg, tmp_path / f"input_config_variable_{algorithm}.json")

    run_project(config_path)

    out = Path(cfg.output.output_dir)
    metrics = pd.read_csv(out / "null_metrics_by_jammer.csv", sep=cfg.output.csv_separator, decimal=cfg.output.csv_decimal)

    assert set(metrics["doa_mode"]) == {"variable"}
    assert set(metrics["algorithm"]) == {algorithm}
    assert set(metrics["montecarlo_index"]) == {1, 2, 3}
    assert len(metrics) == cfg.simulation.num_montecarlo * cfg.jammer.num_jammers * len(cfg.scan.null_thresholds_dB)
    assert metrics["jammer_azimuth_deg"].between(0.0, 90.0).all()
    assert metrics["jammer_elevation_deg"].between(15.0, 75.0).all()
    assert metrics.groupby("montecarlo_index")["jammer_azimuth_deg"].first().nunique() > 1
