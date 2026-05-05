import numpy as np
import pandas as pd
import pytest

from crpa_sim.crpa_array import (
    compute_2d_response_grid,
    compute_azimuth_response_cut,
    compute_elevation_response_cut,
    compute_null_depth_dB,
    compute_response_for_angles,
    conventional_steering_weights,
    create_hexagonal_7element_geometry,
    direction_unit_vector,
    steering_vector_ideal,
)


def test_hexagonal_geometry_shape_and_center():
    spacing = 0.2
    geom = create_hexagonal_7element_geometry(7, spacing)

    assert geom.shape == (7, 3)
    assert np.allclose(geom[0], [0.0, 0.0, 0.0])


def test_hexagonal_geometry_outer_elements_have_requested_radius():
    spacing = 0.2
    geom = create_hexagonal_7element_geometry(7, spacing)
    outer_radius = np.linalg.norm(geom[1:, :2], axis=1)

    assert np.allclose(outer_radius, spacing)


def test_hexagonal_geometry_rejects_non_7_elements():
    with pytest.raises(ValueError):
        create_hexagonal_7element_geometry(6, 0.2)


def test_direction_unit_vector_has_unit_norm():
    for az in [-180, -90, 0, 45, 180]:
        for el in [-90, -20, 0, 45, 90]:
            u = direction_unit_vector(az, el)
            assert np.isclose(np.linalg.norm(u), 1.0)


def test_direction_unit_vector_cenit_points_z_positive():
    u = direction_unit_vector(123.0, 90.0)
    assert np.allclose(u[:2], [0.0, 0.0], atol=1e-12)
    assert np.isclose(u[2], 1.0)


def test_steering_vector_has_unit_magnitude(geometry, scenario):
    a = steering_vector_ideal(geometry, 40.0, 10.0, scenario.wavelength_m)

    assert a.shape == (7,)
    assert np.allclose(np.abs(a), 1.0)


def test_conventional_weights_have_expected_shape_and_sum_scale(geometry, scenario):
    w = conventional_steering_weights(
        geometry,
        scenario.desired_azimuth_deg,
        scenario.desired_elevation_deg,
        scenario.wavelength_m,
    )

    assert w.shape == (7,)
    assert np.linalg.norm(w) > 0


def test_compute_response_for_angles_returns_expected_columns(geometry, scenario):
    weights = conventional_steering_weights(
        geometry,
        0.0,
        0.0,
        scenario.wavelength_m,
    )
    az = np.array([-10.0, 0.0, 10.0])
    el = np.array([0.0, 0.0, 0.0])

    df = compute_response_for_angles(geometry, weights, scenario.wavelength_m, az, el)

    expected_columns = {
        "azimuth_deg",
        "elevation_deg",
        "response_abs",
        "response_abs_normalized",
        "response_dB_normalized",
        "response_real",
        "response_imag",
    }
    assert expected_columns.issubset(df.columns)
    assert len(df) == 3
    assert df["response_abs_normalized"].max() == pytest.approx(1.0)


def test_compute_response_for_angles_rejects_mismatched_arrays(geometry, scenario):
    weights = np.ones(7, dtype=complex)

    with pytest.raises(ValueError):
        compute_response_for_angles(
            geometry,
            weights,
            scenario.wavelength_m,
            np.array([0.0, 1.0]),
            np.array([0.0]),
        )


def test_azimuth_and_elevation_cuts_return_dataframes(geometry, scenario):
    weights = conventional_steering_weights(
        geometry,
        0.0,
        0.0,
        scenario.wavelength_m,
    )

    az_df = compute_azimuth_response_cut(
        geometry,
        weights,
        scenario.wavelength_m,
        np.array([-90.0, 0.0, 90.0]),
        0.0,
    )
    el_df = compute_elevation_response_cut(
        geometry,
        weights,
        scenario.wavelength_m,
        np.array([-30.0, 0.0, 30.0]),
        0.0,
    )

    assert isinstance(az_df, pd.DataFrame)
    assert isinstance(el_df, pd.DataFrame)
    assert list(az_df["elevation_deg"]) == [0.0, 0.0, 0.0]
    assert list(el_df["azimuth_deg"]) == [0.0, 0.0, 0.0]


def test_compute_2d_response_grid_returns_matrix_shapes(geometry, scenario):
    weights = conventional_steering_weights(
        geometry,
        0.0,
        0.0,
        scenario.wavelength_m,
    )
    az = np.linspace(-90.0, 90.0, 7)
    el = np.linspace(-45.0, 45.0, 5)

    grid = compute_2d_response_grid(geometry, weights, scenario.wavelength_m, az, el)

    assert grid["azimuth_deg"].shape == (len(el), len(az))
    assert grid["elevation_deg"].shape == (len(el), len(az))
    assert grid["response_dB_normalized"].shape == (len(el), len(az))
    assert np.isclose(grid["response_dB_normalized"].max(), 0.0, atol=1e-6)
    assert np.isclose(grid["response_abs_normalized"].max(), 1.0, atol=1e-9)


def test_null_depth_is_finite(geometry, scenario):
    weights = conventional_steering_weights(
        geometry,
        0.0,
        0.0,
        scenario.wavelength_m,
    )

    depth = compute_null_depth_dB(
        geometry,
        weights,
        scenario.wavelength_m,
        40.0,
        10.0,
    )

    assert np.isfinite(depth)
