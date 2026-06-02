from dataclasses import replace
import numpy as np
import pytest
from crpa_sim.config import JammerInstance
from crpa_sim.jammers import build_jammer_case, generate_complex_noise, generate_jammer_baseband_signal, generate_received_snapshot_matrix, jammer_power_from_jnr

def test_jammer_power_noise_and_signals(project_config, rng):
    """Comprueba potencia JNR, ruido y tipos de senal jammer.

    Parametros:
        project_config: Configuracion base de simulacion.
        rng: Generador aleatorio determinista.
    """
    assert jammer_power_from_jnr(1.0, 40.0) == pytest.approx(10000.0)
    noise = generate_complex_noise(project_config, rng)
    assert noise.shape == (project_config.array.num_elements, project_config.signal.num_snapshots)
    assert np.iscomplexobj(noise)
    tone = JammerInstance("J", 40, 10, 30, "tone", 0.05)
    s = generate_jammer_baseband_signal(tone, 128, 4.0, rng)
    assert np.allclose(np.abs(s), 2.0)
    assert generate_jammer_baseband_signal(tone, 0, 4.0, rng).shape == (0,)
    gaussian = JammerInstance("G", 70, 30, 30, "complex_gaussian")
    assert generate_jammer_baseband_signal(gaussian, 128, 4.0, rng).shape == (128,)
    chirp = JammerInstance("C", 10, 20, 30, "chirp", chirp_frequency=0.02)
    chirp_signal = generate_jammer_baseband_signal(chirp, 128, 4.0, rng)
    assert chirp_signal.shape == (128,)
    assert np.allclose(np.abs(chirp_signal), 2.0)
    with pytest.raises(ValueError):
        generate_jammer_baseband_signal(JammerInstance("bad", 0, 0, 1, "bad"), 128, 1.0, rng)
    assert jammer_power_from_jnr(2.0, -3.0) < 2.0

def test_build_jammer_case_fixed_variable_and_matrix(project_config, variable_project_config, element_positions_m, rng):
    """Verifica creacion de jammers fixed/variable y matriz recibida.

    Parametros:
        project_config: Configuracion fija base.
        variable_project_config: Configuracion con DoA variable.
        element_positions_m: Posiciones XYZ del array.
        rng: Generador aleatorio determinista.
    """
    fixed = build_jammer_case(project_config, rng)
    assert len(fixed) == project_config.jammer.num_jammers
    assert fixed[0].azimuth_deg == project_config.jammer.base_jammers[0].azimuth_deg
    variable = build_jammer_case(variable_project_config, rng)
    az_min, az_max = variable_project_config.jammer.variable_doa_azimuth_range_deg
    el_min, el_max = variable_project_config.jammer.variable_doa_elevation_range_deg
    assert all(az_min <= j.azimuth_deg <= az_max for j in variable)
    assert all(el_min <= j.elevation_deg <= el_max for j in variable)
    X, table = generate_received_snapshot_matrix(project_config, element_positions_m, fixed, rng)
    assert X.shape == (project_config.array.num_elements, project_config.signal.num_snapshots)
    assert len(table) == len(fixed)
    assert "jammer_power_linear" in table.columns


def test_variable_doa_is_reproducible_per_montecarlo_seed(project_config):
    """Comprueba que DoA variable depende de la semilla Monte Carlo.

    Parametros:
        project_config: Configuracion base de simulacion.
    """
    from main import _case_rng

    cfg = replace(
        project_config,
        simulation=replace(project_config.simulation, doa_mode="variable", random_seed=9876),
        jammer=replace(
            project_config.jammer,
            num_jammers=2,
            variable_doa_azimuth_range_deg=(0.0, 80.0),
            variable_doa_elevation_range_deg=(10.0, 80.0),
        ),
    )

    first = build_jammer_case(cfg, _case_rng(cfg.simulation.random_seed, 1))
    first_repeat = build_jammer_case(cfg, _case_rng(cfg.simulation.random_seed, 1))
    second = build_jammer_case(cfg, _case_rng(cfg.simulation.random_seed, 2))

    first_coords = np.array([(j.azimuth_deg, j.elevation_deg) for j in first])
    repeat_coords = np.array([(j.azimuth_deg, j.elevation_deg) for j in first_repeat])
    second_coords = np.array([(j.azimuth_deg, j.elevation_deg) for j in second])

    np.testing.assert_allclose(first_coords, repeat_coords)
    assert not np.allclose(first_coords, second_coords)
    assert np.all((0.0 <= first_coords[:, 0]) & (first_coords[:, 0] <= 80.0))
    assert np.all((10.0 <= first_coords[:, 1]) & (first_coords[:, 1] <= 80.0))


def test_build_jammer_case_preserves_chirp_frequency(project_config, rng):
    """Comprueba que la frecuencia de chirp se conserva desde la plantilla.

    Parametros:
        project_config: Configuracion base de simulacion.
        rng: Generador aleatorio determinista.
    """
    chirp_cfg = replace(
        project_config,
        jammer=replace(
            project_config.jammer,
            num_jammers=1,
            base_jammers=[
                replace(
                    project_config.jammer.base_jammers[0],
                    signal_type="chirp",
                    chirp_frequency=0.03,
                )
            ],
        ),
    )
    jammers = build_jammer_case(chirp_cfg, rng)
    assert jammers[0].signal_type == "chirp"
    assert jammers[0].chirp_frequency == pytest.approx(0.03)

def test_build_jammer_case_bad_mode(project_config, rng):
    """Comprueba que build_jammer_case rechaza doa_mode desconocido.

    Parametros:
        project_config: Configuracion base de simulacion.
        rng: Generador aleatorio determinista.
    """
    bad = replace(project_config, simulation=replace(project_config.simulation, doa_mode="bad"))
    with pytest.raises(ValueError):
        build_jammer_case(bad, rng)


def test_generate_received_snapshot_matrix_empty_jammers(project_config, element_positions_m):
    """Comprueba que una lista vacia devuelve solo ruido y tabla vacia.

    Parametros:
        project_config: Configuracion base de simulacion.
        element_positions_m: Posiciones XYZ del array.
    """
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    X, table = generate_received_snapshot_matrix(project_config, element_positions_m, [], rng_a)
    noise = generate_complex_noise(project_config, rng_b)
    assert np.allclose(X, noise)
    assert table.empty


def test_chirp_without_frequency_raises_value_error(rng):
    """Documenta el error esperado si un chirp no tiene frecuencia central.

    Parametros:
        rng: Generador aleatorio determinista.
    """
    chirp = JammerInstance("C", 0, 0, 30, "chirp", chirp_frequency=None)
    with pytest.raises(ValueError, match="chirp_frequency"):
        generate_jammer_baseband_signal(chirp, 16, 1.0, rng)
