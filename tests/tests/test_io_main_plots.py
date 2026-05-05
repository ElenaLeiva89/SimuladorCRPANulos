from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from crpa_sim.io_utils import ensure_output_dir, load_project_config, print_generated_files, save_complex_npz, save_config_used, save_dataframe, save_run_log, validate_project_config
from crpa_sim.patterns import compute_2d_response_grid, compute_azimuth_response_cut, compute_elevation_response_cut, conventional_weights
from crpa_sim.plots import _crpa_display_labels_clockwise, _db_to_radius, plot_3d_comparison, plot_array_geometry, plot_heatmap_comparison, plot_pattern_comparison_azimuth, plot_pattern_comparison_elevation, plot_temporal_spectrum
from dataclasses import replace

def test_io_validation_and_saves(project_config, config_json_path, tmp_path, capsys):
    cfg = load_project_config(config_json_path)
    assert cfg.beamforming.algorithm == "lcmv"
    with pytest.raises(FileNotFoundError):
        load_project_config(tmp_path / "missing.json")
    for bad in [
        replace(project_config, jammer=replace(project_config.jammer, num_jammers=0)),
        replace(project_config, jammer=replace(project_config.jammer, num_jammers=7)),
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
    plot_pattern_comparison_azimuth(az_df, az_df, outputs[1], "az", [40])
    plot_pattern_comparison_elevation(el_df, el_df, outputs[2], "el", [10])
    plot_heatmap_comparison(grid, grid, outputs[3], "heat")
    plot_3d_comparison(grid, grid, outputs[4], "3d")
    plot_temporal_spectrum(pd.DataFrame({"frequency_hz": np.arange(10), "power_dB_normalized": np.linspace(-30,0,10)}), outputs[5], "fft")
    assert all(p.exists() and p.stat().st_size > 0 for p in outputs)

def test_main_run_project_light(config_json_path):
    from main import _case_rng, run_project
    r1, r2 = _case_rng(12345, 1), _case_rng(12345, 1)
    assert r1.random() == pytest.approx(r2.random())
    run_project(config_json_path)
    cfg = load_project_config(config_json_path)
    out = Path(cfg.output.output_dir)
    assert (out / "config_used.json").exists()
    assert (out / "run_log.txt").exists()
    assert (out / "null_metrics_by_jammer.csv").exists()
    assert (out / "null_metrics_summary.csv").exists()
    assert (out / "matrices_complex.npz").exists()
    assert (out / "jammer_cuts").exists()
