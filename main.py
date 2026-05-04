"""
main.py
-------
Punto de entrada de la simulación CRPA.

Flujo correcto:
1) cargar configuración,
2) crear geometría,
3) generar snapshots X = interferencias + ruido,
4) calcular pesos w según algoritmo,
5) calcular patrón como B(az,el)=w^H a(az,el),
6) guardar CSV/NPZ/logs/plots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crpa_sim.beamformers import compute_weights
from crpa_sim.crpa_array import (
    compute_azimuth_response_cut,
    compute_elevation_response_cut,
    compute_null_depth_dB,
    conventional_steering_weights,
    create_hexagonal_7element_geometry,
)
from crpa_sim.fft_tools import temporal_fft_snapshot_matrix
from crpa_sim.io_utils import (
    ensure_output_dir,
    load_simulation_parameters,
    print_generated_files,
    save_run_log,
)
from crpa_sim.jammers import generate_received_snapshot_matrix
from crpa_sim.plots import (
    plot_array_geometry,
    plot_pattern_comparison_azimuth,
    plot_pattern_comparison_elevation,
    plot_temporal_spectrum,
)
from crpa_sim.covariance import compute_sample_covariance

def run_simulation(input_config_path: Path = Path("input_config.json")) -> None:
    """
    Ejecuta la simulación completa del CRPA.

    Carga la configuración, genera la geometría del arreglo, simula snapshots con jammers y ruido,
    calcula pesos adaptativos, evalúa patrones espaciales, calcula profundidades de nulos para cada jammer,
    guarda resultados y genera gráficos.

    Parameters:
    - input_config_path (Path): Ruta al archivo de configuración JSON de entrada. Por defecto "input_config.json".
    """
    simulation, config, jammer_list = load_simulation_parameters(input_config_path)
    output_dir = ensure_output_dir(config.output_dir)
    rng = np.random.default_rng(config.random_seed)

    print("Generando geometría CRPA...")
    element_positions_m = create_hexagonal_7element_geometry(config.num_elements, config.element_spacing_m)

    print("Generando snapshots con ruido e interferencias...")
    snapshot_matrix, _ = generate_received_snapshot_matrix(
        config=config,
        jammer_list=jammer_list,
        element_positions_m=element_positions_m,
        rng=rng,
    )

    print(f"Calculando pesos: {simulation.algorithmType}")
    selected_weights = compute_weights(
        algorithm_type=simulation.algorithmType,
        snapshot_matrix=snapshot_matrix,
        element_positions_m=element_positions_m,
        config=config,
        jammer_list=jammer_list,
    )

    conventional_weights = conventional_steering_weights(
        element_positions_m,
        config.desired_azimuth_deg,
        config.desired_elevation_deg,
        config.wavelength_m,
    )

    azimuth_scan_deg = np.arange(
        config.azimuth_scan_min_deg,
        config.azimuth_scan_max_deg + config.azimuth_scan_step_deg,
        config.azimuth_scan_step_deg,
    )
    elevation_scan_deg = np.arange(
        config.elevation_scan_min_deg,
        config.elevation_scan_max_deg + config.elevation_scan_step_deg,
        config.elevation_scan_step_deg,
    )

    print("Calculando patrones espaciales w^H a(az,el)...")
    conventional_azimuth = compute_azimuth_response_cut(
        element_positions_m, conventional_weights, config.wavelength_m, azimuth_scan_deg, config.desired_elevation_deg
    )
    selected_azimuth = compute_azimuth_response_cut(
        element_positions_m, selected_weights, config.wavelength_m, azimuth_scan_deg, config.desired_elevation_deg
    )
    conventional_elevation = compute_elevation_response_cut(
        element_positions_m, conventional_weights, config.wavelength_m, elevation_scan_deg, config.fixed_azimuth_cut_deg
    )
    selected_elevation = compute_elevation_response_cut(
        element_positions_m, selected_weights, config.wavelength_m, elevation_scan_deg, config.fixed_azimuth_cut_deg
    )

    covariance_matrix = compute_sample_covariance(snapshot_matrix)
    temporal_spectrum = temporal_fft_snapshot_matrix(snapshot_matrix, sample_rate_hz=1.0)

    null_depth_rows = []
    reference_gain_abs = selected_azimuth["response_abs"].max()
    for jammer in jammer_list:
        depth_dB = compute_null_depth_dB(
            element_positions_m,
            selected_weights,
            config.wavelength_m,
            jammer.azimuth_deg,
            jammer.elevation_deg,
            reference_gain_abs=reference_gain_abs,
        )
        null_depth_rows.append(
            {
                "jammer_name": jammer.name,
                "azimuth_deg": jammer.azimuth_deg,
                "elevation_deg": jammer.elevation_deg,
                "null_depth_dB_normalized": depth_dB,
            }
        )
    null_depth_table = pd.DataFrame(null_depth_rows)

    print("Guardando resultados...")
    # save_configuration_copy(output_dir, simulation, config, jammer_list)
    # save_dataframe(pd.DataFrame(element_positions_m, columns=["x_m", "y_m", "z_m"]), output_dir / "element_positions_m.csv")
    # save_dataframe(jammer_table, output_dir / "jammer_table.csv")
    # save_dataframe(conventional_azimuth, output_dir / "pattern_conventional_azimuth.csv")
    # save_dataframe(selected_azimuth, output_dir / "pattern_selected_algorithm_azimuth.csv")
    # save_dataframe(conventional_elevation, output_dir / "pattern_conventional_elevation.csv")
    # save_dataframe(selected_elevation, output_dir / "pattern_selected_algorithm_elevation.csv")
    # save_dataframe(temporal_spectrum, output_dir / "temporal_fft_snapshot_spectrum.csv")
    # save_dataframe(null_depth_table, output_dir / "null_depth_table.csv")

    # save_complex_npz(output_dir / "matrices_complex.npz", snapshot_matrix=snapshot_matrix, covariance_matrix=covariance_matrix, selected_weights=selected_weights, conventional_weights=conventional_weights)

    save_run_log(simulation, config, output_dir, len(jammer_list), null_depth_rows)

    print("Generando plots...")
    jammer_azimuths = [j.azimuth_deg for j in jammer_list] if jammer_list else []
    jammer_elevations = [j.elevation_deg for j in jammer_list] if jammer_list else []
    title = f"CRPA 7 elementos - Banda {simulation.band_label} - Algoritmo {simulation.algorithmType}"

    plot_array_geometry(element_positions_m, output_dir / "array_geometry.png")
    plot_pattern_comparison_azimuth(conventional_azimuth, selected_azimuth, output_dir / "pattern_azimuth_dB.png", title + " - Azimuth", jammer_azimuths, adaptive_label=simulation.algorithmType)
    plot_pattern_comparison_elevation(conventional_elevation, selected_elevation, output_dir / "pattern_elevation_dB.png", title + " - Elevation", jammer_elevations, adaptive_label=simulation.algorithmType)
    plot_temporal_spectrum(temporal_spectrum, output_dir / "temporal_fft_snapshot_spectrum.png", "FFT temporal media de snapshots")

    print(f"Resultados guardados en: {output_dir.resolve()}")
    print_generated_files(output_dir)


if __name__ == "__main__":
    print("/////////////////////// INICIANDO SIMULACIÓN CRPA ///////////////////////")
    run_simulation(Path("input_config.json"))
    print("/////////////////////// SIMULACIÓN CRPA COMPLETADA ///////////////////////")
