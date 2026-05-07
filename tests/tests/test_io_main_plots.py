from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from crpa_sim.io_utils import ensure_output_dir, load_project_config, print_generated_files, save_complex_npz, save_config_used, save_dataframe, save_run_log, validate_project_config
from crpa_sim.patterns import compute_2d_response_grid, compute_azimuth_response_cut, compute_elevation_response_cut, conventional_weights
from crpa_sim.plots import _crpa_display_labels_clockwise, _db_to_radius, plot_3d, plot_array_geometry, plot_heatmap, plot_pattern_azimuth, plot_pattern_elevation, plot_temporal_spectrum
from dataclasses import replace

def test_io_validation_and_saves(project_config, config_json_path, tmp_path, capsys):
    """Valida carga, validaciones y funciones basicas de guardado.

    Parametros:
        project_config: Configuracion base de simulacion.
        config_json_path: Ruta temporal a un JSON de configuracion valido.
        tmp_path: Directorio temporal de pytest.
        capsys: Capturador de salida estandar de pytest.
    """
    cfg = load_project_config(config_json_path)
    assert cfg.beamforming.algorithm == "lcmv"
    with pytest.raises(FileNotFoundError):
        load_project_config(tmp_path / "missing.json")
    for bad in [
        replace(project_config, jammer=replace(project_config.jammer, num_jammers=0)),
        replace(project_config, jammer=replace(project_config.jammer, num_jammers=7)),
        replace(project_config, jammer=replace(project_config.jammer, num_jammers=4)),
        replace(project_config, array=replace(project_config.array, geometry="bad")),
        replace(project_config, array=replace(project_config.array, array_boresight_elevation_deg=0)),
    ]:
        with pytest.raises(ValueError):
            validate_project_config(bad)
    out = ensure_output_dir(tmp_path / "results")
    save_config_used(project_config, out)
    save_dataframe(pd.DataFrame({"x":[1.5]}), out / "table.csv")
    save_complex_npz(out / "a.npz", x=np.ones(3))
    save_run_log(project_config, out, [{"ok": True}])
    print_generated_files(out)
    assert (out / "config_used.json").exists()
    assert "1,5" in (out / "table.csv").read_text(encoding="utf-8")
    assert "Ficheros generados" in capsys.readouterr().out

def test_plots_create_files(project_config, element_positions_m, tmp_path):
    """Comprueba que todos los plots principales generan ficheros PNG.

    Parametros:
        project_config: Configuracion base de simulacion.
        element_positions_m: Posiciones XYZ del array.
        tmp_path: Directorio temporal de pytest.
    """
    assert sorted(_crpa_display_labels_clockwise(element_positions_m)) == ["1","2","3","4","5","6","7"]
    assert np.all(_db_to_radius(np.array([-100,-50,0]), -50) >= 0)
    w = conventional_weights(project_config, element_positions_m)
    az = np.arange(-90, 91, 30)
    el = np.arange(0, 91, 30)
    az_df = compute_azimuth_response_cut(project_config, element_positions_m, w, az, 90)
    el_df = compute_elevation_response_cut(project_config, element_positions_m, w, el, 0)
    grid = compute_2d_response_grid(project_config, element_positions_m, w, az, el)
    outputs = [tmp_path / n for n in ["geom.png","az.png","el.png","heat.png","3d.png","fft.png"]]
    plot_array_geometry(element_positions_m, outputs[0])
    plot_pattern_azimuth(az_df, outputs[1], "az", [("Jammer_1", 40), ("Jammer_2", 70, 3)])
    plot_pattern_elevation(el_df, outputs[2], "el", [("Jammer_1", 10), ("Jammer_2", 30, 4)])
    plot_heatmap(grid, outputs[3], "heat")
    plot_3d(grid, outputs[4], "3d")
    plot_temporal_spectrum(pd.DataFrame({"frequency_hz": np.arange(10), "power_dB_normalized": np.linspace(-30,0,10)}), outputs[5], "fft")
    assert all(p.exists() and p.stat().st_size > 0 for p in outputs)


def test_plots_without_jammers_and_missing_columns(project_config, element_positions_m, tmp_path):
    """Comprueba plots sin jammers y errores por columnas obligatorias.

    Parametros:
        project_config: Configuracion base de simulacion.
        element_positions_m: Posiciones XYZ del array.
        tmp_path: Directorio temporal de pytest.
    """
    w = conventional_weights(project_config, element_positions_m)
    az_df = compute_azimuth_response_cut(project_config, element_positions_m, w, np.array([0.0]), 90.0)
    el_df = compute_elevation_response_cut(project_config, element_positions_m, w, np.array([90.0]), 0.0)
    plot_pattern_azimuth(az_df, tmp_path / "az_no_jammers.png", "az", None)
    plot_pattern_elevation(el_df, tmp_path / "el_no_jammers.png", "el", [])
    with pytest.raises(KeyError):
        plot_pattern_azimuth(pd.DataFrame({"azimuth_deg": [0.0]}), tmp_path / "bad_az.png", "bad")
    with pytest.raises(KeyError):
        plot_heatmap({"azimuth_deg": np.array([[0.0]])}, tmp_path / "bad_heat.png", "bad")

def test_main_run_project_light(config_json_path):
    """Ejecuta una simulacion ligera y comprueba salidas principales.

    Parametros:
        config_json_path: Ruta temporal a un JSON de configuracion valido.
    """
    from main import _case_rng, run_project
    r1, r2 = _case_rng(12345, 1), _case_rng(12345, 1)
    assert r1.random() == pytest.approx(r2.random())
    run_project(config_json_path)
    cfg = load_project_config(config_json_path)
    out = Path(cfg.output.output_dir)
    assert (out / "config_used.json").exists()
    assert (out / "run_log.txt").exists()
    assert not (out / "null_metrics_by_jammer.csv").exists()
    summary_path = out / "null_metrics_summary.csv"
    assert summary_path.exists()
    summary = pd.read_csv(summary_path, sep=cfg.output.csv_separator, decimal=cfg.output.csv_decimal)
    assert list(summary.columns) == [
        "jammer_name",
        "attenuation_threshold_dB",
        "null_width_azimuth_deg",
        "null_width_elevation_deg",
    ]
    assert len(summary) == len(cfg.jammer.base_jammers[: cfg.jammer.num_jammers]) * len(cfg.scan.null_thresholds_dB)
    assert (out / "output_data" / "element_positions_m.csv").exists()
    assert (out / "output_data" / "jammer_table.csv").exists()
    assert (out / "output_data" / "matrices_complex.npz").exists()
    assert (out / "output_data" / "jammer_cuts").exists()
    assert list((out / "output_data" / "jammer_cuts").glob("*.csv"))

def test_global_outputs_save_npz_without_csv(project_config, element_positions_m, rng, tmp_path):
    """Verifica que los NPZ se guardan aunque CSV este desactivado.

    Parametros:
        project_config: Configuracion base de simulacion.
        element_positions_m: Posiciones XYZ del array.
        rng: Generador aleatorio determinista.
        tmp_path: Directorio temporal de pytest.
    """
    from main import _save_global_outputs
    from crpa_sim.covariance import compute_sample_covariance
    from crpa_sim.jammers import build_jammer_case, generate_received_snapshot_matrix
    from crpa_sim.patterns import conventional_weights, make_scan_vectors

    cfg = replace(
        project_config,
        output=replace(project_config.output, output_dir=str(tmp_path / "results"), save_csv=False, save_npz=True, save_plots=False),
    )
    out = ensure_output_dir(cfg.output.output_dir)
    jammers = build_jammer_case(cfg, rng)
    X, jammer_table = generate_received_snapshot_matrix(cfg, element_positions_m, jammers, rng)
    w = conventional_weights(cfg, element_positions_m)
    az, el = make_scan_vectors(cfg)

    _save_global_outputs(cfg, out, element_positions_m, X, compute_sample_covariance(X), w, w, jammers, jammer_table, az, el)

    assert (out / "output_data" / "matrices_complex.npz").exists()
    assert not (out / "output_data" / "jammer_table.csv").exists()


def test_global_outputs_no_data_dir_when_all_outputs_disabled(project_config, element_positions_m, rng, tmp_path):
    """Comprueba que no se crea output_data si CSV/NPZ/plots estan apagados.

    Parametros:
        project_config: Configuracion base de simulacion.
        element_positions_m: Posiciones XYZ del array.
        rng: Generador aleatorio determinista.
        tmp_path: Directorio temporal de pytest.
    """
    from main import _save_global_outputs
    from crpa_sim.covariance import compute_sample_covariance
    from crpa_sim.jammers import build_jammer_case, generate_received_snapshot_matrix
    from crpa_sim.patterns import conventional_weights, make_scan_vectors

    cfg = replace(
        project_config,
        output=replace(project_config.output, output_dir=str(tmp_path / "results"), save_csv=False, save_npz=False, save_plots=False),
    )
    out = ensure_output_dir(cfg.output.output_dir)
    jammers = build_jammer_case(cfg, rng)
    X, jammer_table = generate_received_snapshot_matrix(cfg, element_positions_m, jammers, rng)
    w = conventional_weights(cfg, element_positions_m)
    az, el = make_scan_vectors(cfg)

    _save_global_outputs(cfg, out, element_positions_m, X, compute_sample_covariance(X), w, w, jammers, jammer_table, az, el)

    assert not (out / "output_data").exists()
