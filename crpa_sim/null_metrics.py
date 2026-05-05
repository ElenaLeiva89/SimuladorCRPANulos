"""null_metrics.py
Medida de profundidad y anchura de nulos.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .array_model import steering_vector
from .config import JammerInstance, ProjectConfig
from .patterns import compute_azimuth_response_cut, compute_elevation_response_cut


def compute_null_depth_dB(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    jammer: JammerInstance,
    reference_gain_abs: float = 1.0,
    floor_dB: float = -120.0,
) -> float:
    """Profundidad del nulo en la dirección exacta del jammer.

    Se limita a floor_dB para evitar suelos numéricos ideales tipo -240 dB.
    """
    a_j = steering_vector(config, element_positions_m, jammer.azimuth_deg, jammer.elevation_deg)
    gain_abs = abs(np.vdot(weights, a_j))
    depth = 20.0 * np.log10(gain_abs / (reference_gain_abs + 1e-15) + 1e-12)
    return float(max(depth, floor_dB))


def measure_null_width_1d(angle_grid_deg: np.ndarray, response_dB: np.ndarray, jammer_angle_deg: float, threshold_dB: float) -> float | None:
    """Anchura del intervalo contiguo alrededor del jammer por debajo del umbral."""
    angle_grid_deg = np.asarray(angle_grid_deg, dtype=float)
    response_dB = np.asarray(response_dB, dtype=float)
    idx = int(np.argmin(np.abs(angle_grid_deg - jammer_angle_deg)))

    if response_dB[idx] > threshold_dB:
        return None

    left = idx
    while left > 0 and response_dB[left] <= threshold_dB:
        left -= 1

    right = idx
    while right < len(response_dB) - 1 and response_dB[right] <= threshold_dB:
        right += 1

    return float(abs(angle_grid_deg[right] - angle_grid_deg[left]))


def compute_null_metrics_for_jammers(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    jammer_list: Sequence[JammerInstance],
    azimuth_scan_deg: np.ndarray,
    elevation_scan_deg: np.ndarray,
    montecarlo_index: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Calcula tabla de métricas y cortes específicos por jammer."""
    rows: list[dict] = []
    cuts: dict[str, pd.DataFrame] = {}

    for jammer_index, jammer in enumerate(jammer_list, start=1):
        az_cut = compute_azimuth_response_cut(
            config,
            element_positions_m,
            weights,
            azimuth_scan_deg,
            fixed_elevation_deg=jammer.elevation_deg,
        )
        el_cut = compute_elevation_response_cut(
            config,
            element_positions_m,
            weights,
            elevation_scan_deg,
            fixed_azimuth_deg=jammer.azimuth_deg,
        )

        cut_prefix = f"mc{montecarlo_index:04d}_jammer_{jammer_index}_{jammer.name}"
        cuts[f"{cut_prefix}_azimuth_cut"] = az_cut
        cuts[f"{cut_prefix}_elevation_cut"] = el_cut

        null_depth = compute_null_depth_dB(config, element_positions_m, weights, jammer, reference_gain_abs=1.0)

        for threshold in config.scan.null_thresholds_dB:
            width_az = measure_null_width_1d(
                az_cut["azimuth_deg"].to_numpy(),
                az_cut["response_dB_normalized"].to_numpy(),
                jammer.azimuth_deg,
                threshold,
            )
            width_el = measure_null_width_1d(
                el_cut["elevation_deg"].to_numpy(),
                el_cut["response_dB_normalized"].to_numpy(),
                jammer.elevation_deg,
                threshold,
            )
            rows.append(
                {
                    "montecarlo_index": montecarlo_index,
                    "algorithm": config.beamforming.algorithm,
                    "doa_mode": config.simulation.doa_mode,
                    "num_jammers": config.jammer.num_jammers,
                    "jammer_index": jammer_index,
                    "jammer_name": jammer.name,
                    "jammer_azimuth_deg": jammer.azimuth_deg,
                    "jammer_elevation_deg": jammer.elevation_deg,
                    "jammer_jnr_dB": jammer.jnr_dB,
                    "jammer_signal_type": jammer.signal_type,
                    "null_depth_dB": null_depth,
                    "attenuation_threshold_dB": threshold,
                    "null_width_azimuth_deg": width_az,
                    "null_width_elevation_deg": width_el,
                }
            )

    return pd.DataFrame(rows), cuts


def summarize_null_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Resumen estadístico agrupado de métricas Monte Carlo."""
    group_cols = [
        "algorithm",
        "doa_mode",
        "num_jammers",
        "jammer_index",
        "jammer_name",
        "attenuation_threshold_dB",
    ]
    numeric_cols = ["null_depth_dB", "null_width_azimuth_deg", "null_width_elevation_deg"]
    return metrics.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std", "min", "max", "count"]).reset_index()
