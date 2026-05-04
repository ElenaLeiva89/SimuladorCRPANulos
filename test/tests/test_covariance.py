import numpy as np
import pytest

from crpa_sim.covariance import (
    add_diagonal_loading,
    compute_sample_covariance,
    invert_covariance,
)


def test_sample_covariance_shape():
    X = np.random.default_rng(1).standard_normal((7, 100)) + 1j * np.random.default_rng(2).standard_normal((7, 100))
    R = compute_sample_covariance(X)

    assert R.shape == (7, 7)


def test_sample_covariance_is_hermitian():
    rng = np.random.default_rng(123)
    X = rng.standard_normal((7, 100)) + 1j * rng.standard_normal((7, 100))
    R = compute_sample_covariance(X)

    assert np.allclose(R, R.conj().T)


def test_sample_covariance_matches_manual_formula():
    rng = np.random.default_rng(123)
    X = rng.standard_normal((3, 10)) + 1j * rng.standard_normal((3, 10))
    R = compute_sample_covariance(X)
    R_manual = X @ X.conj().T / X.shape[1]

    assert np.allclose(R, R_manual)


def test_sample_covariance_rejects_1d_input():
    with pytest.raises(ValueError):
        compute_sample_covariance(np.ones(7))


def test_sample_covariance_rejects_zero_snapshots():
    with pytest.raises(ValueError):
        compute_sample_covariance(np.empty((7, 0), dtype=complex))


def test_diagonal_loading_keeps_shape_and_increases_diagonal():
    R = np.eye(7, dtype=complex)
    R_loaded = add_diagonal_loading(R, 1e-3)

    assert R_loaded.shape == (7, 7)
    assert np.all(np.real(np.diag(R_loaded)) > np.real(np.diag(R)))


def test_diagonal_loading_rejects_non_square_matrix():
    with pytest.raises(ValueError):
        add_diagonal_loading(np.ones((7, 3)), 1e-3)


def test_invert_covariance_returns_finite_matrix():
    R = np.eye(7, dtype=complex)
    R_inv = invert_covariance(R)

    assert R_inv.shape == (7, 7)
    assert np.all(np.isfinite(R_inv))
