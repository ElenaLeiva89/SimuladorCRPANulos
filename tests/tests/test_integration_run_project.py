from __future__ import annotations

from pathlib import Path

import pandas as pd

from main import run_project
from conftest import write_config_json


def test_run_project_creates_core_artifacts_without_plots(fast_config, tmp_path):
    config_path = write_config_json(fast_config, tmp_path / "input_config_test.json")

    run_project(config_path)

    out = Path(fast_config.output.output_dir)
    assert (out / "config_used.json").exists()
    assert (out / "run_log.txt").exists()
    assert (out / "null_metrics_summary.csv").exists()
    assert (out / "output_data" / "matrices_complex.npz").exists()
    assert not list(out.glob("*.png"))

    summary = pd.read_csv(out / "null_metrics_summary.csv", sep=fast_config.output.csv_separator, decimal=fast_config.output.csv_decimal)
    assert not summary.empty
    assert {"jammer_name", "attenuation_threshold_dB", "null_width_azimuth_deg", "null_width_elevation_deg"}.issubset(summary.columns)
