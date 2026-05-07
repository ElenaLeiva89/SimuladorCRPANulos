import numpy as np
import pytest
from crpa_sim.covariance import add_diagonal_loading, compute_sample_covariance, invert_covariance

def test_covariance_formula_and_loading():
    """Verifica formula de covarianza, carga diagonal e inversion.

    Parametros:
        No recibe parametros.
    """
    rng = np.random.default_rng(1)
    X = rng.standard_normal((7, 64)) + 1j * rng.standard_normal((7, 64))
    R = compute_sample_covariance(X)
    assert R.shape == (7, 7)
    assert np.allclose(R, R.conj().T)
    assert np.allclose(R, X @ X.conj().T / X.shape[1])
    loaded = add_diagonal_loading(np.eye(7, dtype=complex), 0.001)
    assert np.all(np.real(np.diag(loaded)) > 1)
    assert np.all(np.isfinite(invert_covariance(np.eye(7), 0.001)))
    assert np.allclose(add_diagonal_loading(np.eye(2), 0.0), np.eye(2))

def test_covariance_errors():
    """Comprueba validaciones de forma para covarianza y loading.

    Parametros:
        No recibe parametros.
    """
    with pytest.raises(ValueError):
        compute_sample_covariance(np.ones(7))
    with pytest.raises(ValueError):
        compute_sample_covariance(np.empty((7, 0), dtype=complex))
    with pytest.raises(ValueError):
        add_diagonal_loading(np.ones((7, 3)), 0.001)


def test_covariance_singular_and_one_snapshot_cases():
    """Comprueba casos singulares y una unica muestra temporal.

    Parametros:
        No recibe parametros.
    """
    X = np.ones((3, 1), dtype=complex)
    R = compute_sample_covariance(X)
    assert R.shape == (3, 3)
    assert np.linalg.matrix_rank(R) == 1
    R_inv = invert_covariance(R, 0.0)
    assert R_inv.shape == R.shape
    assert np.all(np.isfinite(R_inv))
