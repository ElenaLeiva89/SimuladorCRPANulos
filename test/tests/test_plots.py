import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

from crpa_sim.crpa_array import (
    compute_azimuth_response_cut,
    compute_elevation_response_cut,
    conventional_steering_weights,
)
from crpa_sim.plots import (
    _crpa_display_labels_clockwise,
    _db_to_polar_radius,
    plot_array_geometry,
    plot_pattern_comparison_azimuth,
    plot_pattern_comparison_elevation,
    plot_temporal_spectrum,
)


def test_db_to_polar_radius_clips_minimum():
    values = np.array([-100.0, -50.0, -25.0, 0.0])
    radius = _db_to_polar_radius(values, -50.0)

    assert np.all(radius >= 0.0)
    assert radius[0] == 0.0
    assert radius[-1] == 50.0


def test_crpa_display_labels_clockwise_contains_1_to_7(geometry):
    labels = _crpa_display_labels_clockwise(geometry)

    assert sorted(labels) == ["1", "2", "3", "4", "5", "6", "7"]
    assert labels[0] == "1"


def test_plot_array_geometry_creates_file(tmp_path, geometry):
    output = tmp_path / "array_geometry.png"

    plot_array_geometry(geometry, output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_pattern_plots_create_files(tmp_path, geometry, scenario):
    weights = conventional_steering_weights(
        geometry,
        0.0,
        0.0,
        scenario.wavelength_m,
    )

    az_scan = np.arange(-90.0, 91.0, 5.0)
    el_scan = np.arange(-90.0, 91.0, 5.0)

    az_df = compute_azimuth_response_cut(
        geometry,
        weights,
        scenario.wavelength_m,
        az_scan,
        0.0,
    )
    el_df = compute_elevation_response_cut(
        geometry,
        weights,
        scenario.wavelength_m,
        el_scan,
        0.0,
    )

    az_output = tmp_path / "az.png"
    el_output = tmp_path / "el.png"

    plot_pattern_comparison_azimuth(
        az_df,
        az_df,
        az_output,
        "Azimuth test",
        jammer_azimuths_deg=[40.0],
    )
    plot_pattern_comparison_elevation(
        el_df,
        el_df,
        el_output,
        "Elevation test",
        jammer_elevations_deg=[10.0],
    )

    assert az_output.exists()
    assert az_output.stat().st_size > 0
    assert el_output.exists()
    assert el_output.stat().st_size > 0


def test_plot_temporal_spectrum_creates_file(tmp_path):
    df = pd.DataFrame(
        {
            "frequency_hz": np.linspace(-0.5, 0.5, 32),
            "power_dB_normalized": np.linspace(-50.0, 0.0, 32),
        }
    )
    output = tmp_path / "spectrum.png"

    plot_temporal_spectrum(df, output, "Spectrum test")

    assert output.exists()
    assert output.stat().st_size > 0
