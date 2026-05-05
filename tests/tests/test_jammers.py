from dataclasses import replace
import numpy as np
import pytest
from crpa_sim.config import JammerInstance
from crpa_sim.jammers import build_jammer_case, generate_complex_noise, generate_jammer_baseband_signal, generate_received_snapshot_matrix, jammer_power_from_jnr

def test_jammer_power_noise_and_signals(project_config, rng):
    assert jammer_power_from_jnr(1.0, 40.0) == pytest.approx(10000.0)
    noise = generate_complex_noise(project_config, rng)
    assert noise.shape == (project_config.array.num_elements, project_config.signal.num_snapshots)
    assert np.iscomplexobj(noise)
    tone = JammerInstance("J", 40, 10, 30, "tone", 0.05)
    s = generate_jammer_baseband_signal(tone, 128, 4.0, rng)
    assert np.allclose(np.abs(s), 2.0)
    gaussian = JammerInstance("G", 70, 30, 30, "complex_gaussian")
    assert generate_jammer_baseband_signal(gaussian, 128, 4.0, rng).shape == (128,)
    with pytest.raises(ValueError):
        generate_jammer_baseband_signal(JammerInstance("bad", 0, 0, 1, "bad"), 128, 1.0, rng)

def test_build_jammer_case_fixed_variable_and_matrix(project_config, variable_project_config, element_positions_m, rng):
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

def test_build_jammer_case_bad_mode(project_config, rng):
    bad = replace(project_config, simulation=replace(project_config.simulation, doa_mode="bad"))
    with pytest.raises(ValueError):
        build_jammer_case(bad, rng)
