import numpy as np
import pytest

from crpa_sim.jammers import (
    generate_complex_noise,
    generate_jammer_baseband_signal,
    generate_received_snapshot_matrix,
    jammer_power_from_jnr,
)


def test_jammer_power_from_jnr_0_db_equals_noise_power():
    assert jammer_power_from_jnr(2.0, 0.0) == pytest.approx(2.0)


def test_jammer_power_from_jnr_30_db():
    assert jammer_power_from_jnr(1.0, 30.0) == pytest.approx(1000.0)


def test_generate_complex_noise_shape_and_complex_dtype(rng):
    noise = generate_complex_noise(7, 128, 1.0, rng)

    assert noise.shape == (7, 128)
    assert np.iscomplexobj(noise)


def test_generate_tone_jammer_has_constant_amplitude(jammer_tone, rng):
    signal = generate_jammer_baseband_signal(jammer_tone, 128, 4.0, rng)

    assert signal.shape == (128,)
    assert np.allclose(np.abs(signal), 2.0)


def test_generate_gaussian_jammer_shape(jammer_gaussian, rng):
    signal = generate_jammer_baseband_signal(jammer_gaussian, 128, 4.0, rng)

    assert signal.shape == (128,)
    assert np.iscomplexobj(signal)


def test_generate_jammer_rejects_unknown_signal_type(jammer_tone, rng):
    jammer_tone.signal_type = "unknown"

    with pytest.raises(ValueError):
        generate_jammer_baseband_signal(jammer_tone, 128, 1.0, rng)


def test_generate_received_snapshot_matrix_shape_and_jammer_table(scenario, geometry, jammer_tone, rng):
    X, table = generate_received_snapshot_matrix(
        scenario,
        [jammer_tone],
        geometry,
        rng,
    )

    assert X.shape == (scenario.num_elements, scenario.num_snapshots)
    assert len(table) == 1
    assert table.iloc[0]["name"] == jammer_tone.name
    assert table.iloc[0]["signal_type"] == jammer_tone.signal_type


def test_generate_received_snapshot_matrix_without_jammers_returns_empty_table(scenario, geometry, rng):
    X, table = generate_received_snapshot_matrix(
        scenario,
        [],
        geometry,
        rng,
    )

    assert X.shape == (scenario.num_elements, scenario.num_snapshots)
    assert table.empty
