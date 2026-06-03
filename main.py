"""Punto de entrada del simulador CRPA.

El flujo de una ejecucion es:
1. leer y validar `input_config.json`;
2. construir la geometria hexagonal ideal de 7 elementos;
3. generar jammers, ruido y snapshots para cada Monte Carlo;
4. estimar la covarianza espacial y los pesos adaptativos;
5. medir profundidad/anchura de nulos y guardar salidas.

La configuracion activa un unico modo DoA (`fixed` o `variable`) y un unico
algoritmo (`power_inversion` o `lcmv`) por ejecucion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crpa_sim.array_model import create_crpa_geometry
from crpa_sim.beamformers import compute_weights
from crpa_sim.config import JammerInstance, ProjectConfig
from crpa_sim.covariance import compute_sample_covariance
from crpa_sim.fft_tools import temporal_fft_snapshot_matrix
from crpa_sim.io_utils import (
    ensure_output_dir,
    load_project_config,
    print_generated_files,
    save_complex_npz,
    save_config_used,
    save_dataframe,
    save_run_log,
)
from crpa_sim.jammers import build_jammer_case, generate_received_snapshot_matrix
from crpa_sim.null_metrics import compute_null_metrics_for_jammers
from crpa_sim.patterns import (
    compute_2d_response_grid,
    compute_azimuth_response_cut,
    compute_elevation_response_cut_for_plot,
    conventional_weights,
    make_scan_vectors,
)
from crpa_sim.plots import (
    plot_3d,
    plot_array_geometry,
    plot_heatmap,
    plot_pattern_azimuth,
    plot_pattern_elevation,
    plot_temporal_spectrum,
)

def _case_rng(base_seed: int, montecarlo_index: int) -> np.random.Generator:
    """Crea un generador reproducible para una iteracion Monte Carlo.

    Parametros:
        base_seed: Semilla base definida en la configuracion.
        montecarlo_index: Indice de la iteracion, empezando en 1.
    """
    seed = (int(base_seed) * 1664525 + int(montecarlo_index) * 1013904223) % (2**32)
    return np.random.default_rng(seed)


def _save_global_outputs(
    config: ProjectConfig,
    output_dir: Path,
    element_positions_m: np.ndarray,
    snapshot_matrix: np.ndarray,
    covariance_matrix: np.ndarray,
    selected_weights: np.ndarray,
    conventional_w: np.ndarray,
    jammer_table: pd.DataFrame,
    azimuth_scan_deg: np.ndarray,
    elevation_scan_deg: np.ndarray,
) -> None:
    """Guarda ficheros globales de la primera iteracion Monte Carlo.

    Parametros:
        config: Configuracion completa del proyecto.
        output_dir: Directorio raiz de salida.
        element_positions_m: Matriz (N, 3) con posiciones del array.
        snapshot_matrix: Matriz X recibida para el caso actual.
        covariance_matrix: Covarianza espacial estimada desde X.
        selected_weights: Pesos del algoritmo seleccionado.
        conventional_w: Pesos convencionales de referencia.
        jammer_table: Tabla descriptiva de los jammers generados.
        azimuth_scan_deg: Vector de azimuts para cortes.
        elevation_scan_deg: Vector de elevaciones para cortes.
    """
    sep = config.output.csv_separator
    dec = config.output.csv_decimal
    output_dir_data = output_dir / "output_data"

    if config.output.save_csv:
        output_dir_data.mkdir(parents=True, exist_ok=True)
        save_dataframe(pd.DataFrame(element_positions_m, columns=["x_m", "y_m", "z_m"]), output_dir_data / "element_positions_m.csv", sep, dec)
        save_dataframe(jammer_table, output_dir_data / "jammer_table.csv", sep, dec)

    if config.output.save_npz:
        output_dir_data.mkdir(parents=True, exist_ok=True)
        save_complex_npz(
            output_dir_data / "matrices_complex.npz",
            snapshot_matrix=snapshot_matrix,
            covariance_matrix=covariance_matrix,
            selected_weights=selected_weights,
            conventional_weights=conventional_w,
        )

    if config.output.save_plots:
        plot_array_geometry(element_positions_m, output_dir / "array_geometry.png")

        radiation_az = compute_azimuth_response_cut(
            config,
            element_positions_m,
            selected_weights,
            azimuth_scan_deg,
            fixed_elevation_deg=config.beamforming.desired_elevation_deg,
        )
        radiation_el = compute_elevation_response_cut_for_plot(
            config,
            element_positions_m,
            selected_weights,
            elevation_scan_deg,
            fixed_azimuth_deg=config.beamforming.desired_azimuth_deg,
        )

        jammer_az = ([(row["name"], row["azimuth_deg"], int(row["jammer_index"]) - 1) 
                for _, row in jammer_table.iterrows()]
                    if not jammer_table.empty
                    else [])
        jammer_el = ([(row["name"], row["elevation_deg"], int(row["jammer_index"]) - 1)
                for _, row in jammer_table.iterrows()]
                    if not jammer_table.empty
                    else [])
        title = f"Patron de radiacion de CRPA 7 elementos \ny direcciones de jammers - {config.signal.band_label}"

        plot_pattern_azimuth(
            radiation_az,
            output_dir / "pattern_global_azimuth_dB.png",
            title + " - Azimuth",
            jammer_az,
            adaptive_label=config.beamforming.algorithm,
            fixed_elevation_deg=config.beamforming.desired_elevation_deg,
        )
        plot_pattern_elevation(
            radiation_el,
            output_dir / "pattern_global_elevation_dB.png",
            title + " - Elevation",
            jammer_el,
            adaptive_label=config.beamforming.algorithm,
            fixed_azimuth_deg=config.beamforming.desired_azimuth_deg,
        )

        spectrum = temporal_fft_snapshot_matrix(snapshot_matrix, config.signal.sample_rate_hz, config.signal.fft_size)
        plot_temporal_spectrum(spectrum, output_dir / "temporal_fft_snapshot_spectrum.png", "FFT temporal media de snapshots")
        if config.output.save_csv:
            save_dataframe(spectrum, output_dir_data / "temporal_fft_snapshot_spectrum.csv", sep, dec)


def _save_jammer_plots(
    config: ProjectConfig,
    output_dir: Path,
    element_positions_m: np.ndarray,
    selected_weights: np.ndarray,
    jammer_list: list[JammerInstance],
    azimuth_scan_deg: np.ndarray,
    elevation_scan_deg: np.ndarray,
    montecarlo_index: int,
) -> None:
    """Genera plots especificos por jammer para la primera iteracion.

    Parametros:
        config: Configuracion completa del proyecto.
        output_dir: Directorio raiz de salida.
        element_positions_m: Matriz (N, 3) con posiciones del array.
        selected_weights: Pesos del algoritmo seleccionado.
        jammer_list: Lista de jammers del caso actual.
        azimuth_scan_deg: Vector de azimuts para cortes.
        elevation_scan_deg: Vector de elevaciones para cortes.
        montecarlo_index: Indice de la iteracion Monte Carlo actual.
    """
    if not config.output.save_plots or montecarlo_index != 1:
        return

    jammer_plots_dir = output_dir / "jammer_plots"
    jammer_plots_dir.mkdir(parents=True, exist_ok=True)

    for idx, jammer in enumerate(jammer_list, start=1):
        tag = f"{jammer.name}"
        jam_dir = jammer_plots_dir
        jam_dir.mkdir(parents=True, exist_ok=True)

        radiation_az = compute_azimuth_response_cut(config, element_positions_m, selected_weights, azimuth_scan_deg, jammer.elevation_deg)
        radiation_el = compute_elevation_response_cut_for_plot(config, element_positions_m, selected_weights, elevation_scan_deg, jammer.azimuth_deg)

        title_base = f"{tag} - az={jammer.azimuth_deg:.1f} deg, el={jammer.elevation_deg:.1f} deg - {config.beamforming.algorithm}"
        plot_pattern_azimuth(
            radiation_az,
            jam_dir / f"{tag}_pattern_azimuth_dB.png",
            title_base + " \nCorte azimut por jammer",
            [(jammer.name, jammer.azimuth_deg, idx - 1)],
            adaptive_label=config.beamforming.algorithm,
            fixed_elevation_deg=jammer.elevation_deg,
        )
        plot_pattern_elevation(
            radiation_el,
            jam_dir / f"{tag}_pattern_elevation_dB.png",
            title_base + " \nCorte elevacion por jammer",
            [(jammer.name, jammer.elevation_deg, idx - 1)],
            adaptive_label=config.beamforming.algorithm,
            fixed_azimuth_deg=jammer.azimuth_deg,
        )

    jammer_info_3d = [
        (jammer.name, jammer.azimuth_deg, jammer.elevation_deg, idx - 1)
        for idx, jammer in enumerate(jammer_list, start=1)
    ]

    radiation_grid = compute_2d_response_grid(
        config,
        element_positions_m,
        selected_weights,
        azimuth_scan_deg,
        elevation_scan_deg,
    )
    plot_3d(
        radiation_grid,
        output_dir / "pattern_3d_comparison.png",
        "Patron 3D CRPA para algoritmo " + config.beamforming.algorithm,
        adaptive_label=config.beamforming.algorithm,
        jammer_info=jammer_info_3d,
    )
    plot_heatmap(
        radiation_grid,
        output_dir / "array_factor_heatmap.png",
        f"Heatmap CRPA para algoritmo {config.beamforming.algorithm} - Mapa 2D",
        adaptive_label=config.beamforming.algorithm,
        jammer_info=jammer_info_3d,
    )


def run_project(config_path: Path = Path("input_config.json")) -> None:
    """Ejecuta la simulacion completa a partir de un fichero JSON.

    Parametros:
        config_path: Ruta del fichero de configuracion de entrada.
    """
    config = load_project_config(config_path)
    resolved_output_dir = config.output.output_dir.format(
        algorithm=config.beamforming.algorithm,
        doa_mode=config.simulation.doa_mode,
        steering_model=config.array.steering_model,
    )
    output_dir = ensure_output_dir(resolved_output_dir)
    save_config_used(config, output_dir)

    print("/////////////////////// INICIANDO SIMULACION CRPA ///////////////////////")
    print(f"Salida: {output_dir.resolve()}")
    print(f"DoA mode: {config.simulation.doa_mode}")
    print(f"Algoritmo: {config.beamforming.algorithm}")
    print(f"Numero de jammers: {config.jammer.num_jammers}")

    element_positions_m = create_crpa_geometry(config.array, config.element_spacing_m)
    azimuth_scan_deg, elevation_scan_deg = make_scan_vectors(config)
    conventional_w = conventional_weights(config, element_positions_m)

    metrics_all: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for mc in range(1, config.simulation.num_montecarlo + 1):
        rng = _case_rng(config.simulation.random_seed, mc)
        jammer_list = build_jammer_case(config, rng)
        snapshot_matrix, jammer_table = generate_received_snapshot_matrix(config, element_positions_m, jammer_list, rng)
        covariance_matrix = compute_sample_covariance(snapshot_matrix)
        selected_weights = compute_weights(config, snapshot_matrix, element_positions_m, jammer_list)

        metrics_table, jammer_cut_tables = compute_null_metrics_for_jammers(
            config=config,
            element_positions_m=element_positions_m,
            weights=selected_weights,
            jammer_list=jammer_list,
            azimuth_scan_deg=azimuth_scan_deg,
            elevation_scan_deg=elevation_scan_deg,
            montecarlo_index=mc,
        )
        metrics_all.append(metrics_table)

        if mc == 1:
            _save_global_outputs(
                config,
                output_dir,
                element_positions_m,
                snapshot_matrix,
                covariance_matrix,
                selected_weights,
                conventional_w,
                jammer_table,
                azimuth_scan_deg,
                elevation_scan_deg,
            )
            _save_jammer_plots(
                config,
                output_dir,
                element_positions_m,
                selected_weights,
                jammer_list,
                azimuth_scan_deg,
                elevation_scan_deg,
                mc,
            )

            if config.output.save_csv:
                cuts_dir = output_dir / "output_data" / "jammer_cuts"
                cuts_dir.mkdir(parents=True, exist_ok=True)
                for name, table in jammer_cut_tables.items():
                    save_dataframe(table, cuts_dir / f"{name}.csv", config.output.csv_separator, config.output.csv_decimal)

        summary_rows.append(
            {
                "montecarlo_index": mc,
                "num_metrics_rows": len(metrics_table),
                "num_jammers": len(jammer_list),
                "doa_mode": config.simulation.doa_mode,
                "algorithm": config.beamforming.algorithm,
            }
        )

    metrics_full = pd.concat(metrics_all, ignore_index=True) if metrics_all else pd.DataFrame()

    if config.output.save_csv and not metrics_full.empty:
        metrics_to_save = metrics_full.copy().fillna("N/A")
        save_dataframe(metrics_to_save, output_dir / "null_metrics_by_jammer.csv", config.output.csv_separator, config.output.csv_decimal)

    save_run_log(config, output_dir, summary_rows)
    print_generated_files(output_dir)
    print("/////////////////////// SIMULACION CRPA COMPLETADA ///////////////////////")


if __name__ == "__main__":
    run_project(Path("input_config.json"))
