"""patterns.py
Evaluacion de patrones espaciales de la CRPA.

El patron se calcula siempre como B(az, el) = w^H a(az, el). Los snapshots
no se usan para dibujar el patron; se usan antes para estimar R y los pesos.

La rama ideal se mantiene vectorizada exactamente como antes. La rama measured
usa una matriz/tensor de steering medido ya construido en memoria para evitar
buscar elemento a elemento para cada punto angular.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .array_model import steering_vector
from .config import ProjectConfig
from .measured_mat import MeasuredSteeringDatabase, _interpolation_indices, measured_steering_matrix_complex, measured_steering_matrix_complex_solo_freq

_STEERING_GRID_CACHE: dict = {}


def conventional_weights(config: ProjectConfig, element_positions_m: np.ndarray) -> np.ndarray:
    """Calcula pesos delay-and-sum hacia la direccion deseada."""
    a_des = steering_vector(
        config,
        element_positions_m,
        config.beamforming.desired_azimuth_deg,
        config.beamforming.desired_elevation_deg,
    )
    return a_des / element_positions_m.shape[0]  ### REVISAR SIGNO - DEPENDE DEL RESTO


def _evaluate_response_complex_for_angles(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    azimuth_deg_array: np.ndarray,
    elevation_deg_array: np.ndarray,
    wavelength_m: float | None = None,
    database: MeasuredSteeringDatabase | None = None,
) -> np.ndarray:
    """Evalua B=w^H a(az, el) devolviendo valores complejos."""

    if wavelength_m is None:
        wavelength_m = config.signal.wavelength_m

    azimuth_deg_array = np.asarray(azimuth_deg_array, dtype=float)
    elevation_deg_array = np.asarray(elevation_deg_array, dtype=float)

    if len(azimuth_deg_array) != len(elevation_deg_array):
        raise ValueError(
            "azimuth_deg_array y elevation_deg_array deben tener la misma longitud."
        )

    if config.array.steering_model == "ideal":

        az_rad = np.deg2rad(azimuth_deg_array)
        el_rad = np.deg2rad(elevation_deg_array)

        u = np.column_stack( [
                np.cos(el_rad) * np.cos(az_rad),
                np.cos(el_rad) * np.sin(az_rad),
                np.sin(el_rad),
            ] )

        k_rad_m = 2.0 * np.pi / wavelength_m
        phase = k_rad_m * (u @ element_positions_m.T)
        steering = np.exp(1j * phase)

    elif config.array.steering_model == "measured":

        frequency_hz = (
            config.signal.speed_of_light_m_s
            / wavelength_m
        )

        steering = measured_steering_matrix_complex(
            database=database,
            azimuth_deg_array=azimuth_deg_array,
            elevation_deg_array=elevation_deg_array,
            frequency_hz=frequency_hz,
        )

    else:
        raise ValueError(f"Modelo steering no soportado: {config.array.steering_model}")

    if config.beamforming.algorithm in ("lcmv", "lcmvq"):
        return steering @ weights
    else:
        return steering @ np.conjugate(weights)

def evaluate_response_for_angles(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    azimuth_deg_array: np.ndarray,
    elevation_deg_array: np.ndarray,
    normalize: bool = True,
    wavelength_m: float | None = None,
    database: MeasuredSteeringDatabase | None = None

) -> pd.DataFrame:
    """Evalua B=w^H a(az, el) en una lista de pares angulares."""
    
    values = _evaluate_response_complex_for_angles(
        config,
        element_positions_m,
        weights,
        azimuth_deg_array,
        elevation_deg_array,
        wavelength_m=wavelength_m,
        database=database
    )

    response_abs = np.abs(values)
    if normalize:
        max_abs = np.max(response_abs)

        if max_abs == 0.0:
            response_abs_norm = np.zeros_like(response_abs)
        else:
            response_abs_norm = response_abs / max_abs
    else:
        response_abs_norm = response_abs

    with np.errstate(divide="ignore"):
        response_dB_norm = 20.0 * np.log10(response_abs_norm)

    return pd.DataFrame(
        {
            "azimuth_deg": azimuth_deg_array,
            "elevation_deg": elevation_deg_array,
            "response_abs": response_abs,
            "response_abs_normalized": response_abs_norm,
            "response_dB_normalized": response_dB_norm,
            "response_real": np.real(values),
            "response_imag": np.imag(values),
        }
    )


def compute_azimuth_response_cut(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    azimuth_scan_deg: np.ndarray,
    fixed_elevation_deg: float,
    wavelength_m: float | None = None,
    database: MeasuredSteeringDatabase | None = None
) -> pd.DataFrame:
    """Calcula un corte de patron variando azimut con elevacion fija."""
    elevations = np.full_like(azimuth_scan_deg, fixed_elevation_deg, dtype=float)
    return evaluate_response_for_angles(config, element_positions_m, weights, azimuth_scan_deg, elevations, wavelength_m=wavelength_m, database=database)


def compute_elevation_response_cut(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    elevation_scan_deg: np.ndarray,
    fixed_azimuth_deg: float,
    wavelength_m: float | None = None,
    database: MeasuredSteeringDatabase | None = None
) -> pd.DataFrame:
    """Calcula un corte de patron variando elevacion con azimut fijo."""
    azimuths = np.full_like(elevation_scan_deg, fixed_azimuth_deg, dtype=float)
    return evaluate_response_for_angles(config, element_positions_m, weights, azimuths, elevation_scan_deg, wavelength_m=wavelength_m, database=database)

def compute_radial_response_cut(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    center_az_deg: float,
    center_el_deg: float,
    cut_angle_deg: float,
    distance_scan_deg: float | None = None,
    wavelength_m: float | None = None,
    database: MeasuredSteeringDatabase | None = None
)-> pd.DataFrame:

    if distance_scan_deg is None:
        distance_scan_deg = np.linspace(-10.0, 10.0, 201)

    theta_rad = np.deg2rad(cut_angle_deg)

    azimuths = center_az_deg + distance_scan_deg*np.cos(theta_rad)
    elevations = center_el_deg + distance_scan_deg*np.sin(theta_rad)

    elevations = np.clip(elevations, 0.0, 90.0)
    azimuths = azimuths % 360.0

    return evaluate_response_for_angles(config, element_positions_m,weights,azimuths,elevations,wavelength_m=wavelength_m,database=database,)

def compute_2d_response_grid(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    azimuth_scan_deg: np.ndarray,
    elevation_scan_deg: np.ndarray,
    wavelength_m: float | None = None,
    database: MeasuredSteeringDatabase | None = None
) -> dict[str, np.ndarray]:

    global _STEERING_GRID_CACHE

    az_grid, el_grid = np.meshgrid(
        azimuth_scan_deg,
        elevation_scan_deg,
        indexing="xy",
    )

    az_flat = az_grid.ravel()
    el_flat = el_grid.ravel()

    if wavelength_m is None:
        wavelength_m = config.signal.wavelength_m

    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m debe ser mayor que cero.")

    if config.array.steering_model == "ideal":

        az_rad = np.deg2rad(az_flat)
        el_rad = np.deg2rad(el_flat)

        u = np.column_stack([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad),
        ])

        k_rad_m = 2.0 * np.pi / wavelength_m
        phase = k_rad_m * (u @ element_positions_m.T)
        steering = np.exp(1j * phase)

        if config.beamforming.algorithm in ("lcmv", "lcmvq"):
            response_complex = steering @ weights
        else:
            response_complex = steering @ np.conjugate(weights)

    elif config.array.steering_model == "measured":

        frequency_hz = ( config.signal.speed_of_light_m_s/ wavelength_m)

        interp = "total"

        if interp == "solo_freq":

            steering = (
                measured_steering_matrix_complex_solo_freq(
                    database=database,
                    azimuth_deg_array=az_flat,
                    elevation_deg_array=el_flat,
                    frequency_hz=frequency_hz,
                )
            )

        else:

            azimuth_scan_deg = np.arange(0.0,360.0,0.5,)
            elevation_scan_deg = np.arange(0.0,90.0,0.5,)
            az_grid, el_grid = np.meshgrid(azimuth_scan_deg,elevation_scan_deg,indexing="xy",)

            az_flat = az_grid.ravel()
            el_flat = el_grid.ravel()

            cache_key = (round(frequency_hz, 3),az_grid.shape,)

            if cache_key not in _STEERING_GRID_CACHE:

                _STEERING_GRID_CACHE[cache_key] = (
                    measured_steering_matrix_complex(
                        database=database,
                        azimuth_deg_array=az_flat,
                        elevation_deg_array=el_flat,
                        frequency_hz=frequency_hz,
                    )
                )

            steering = _STEERING_GRID_CACHE[cache_key]

        if config.beamforming.algorithm in ("lcmv", "lcmvq"):
            response_complex = steering @ weights
        else:
            response_complex = steering @ np.conjugate(weights)

    else:
        raise ValueError(
            f"Modelo steering no soportado: "
            f"{config.array.steering_model}"
        )

    response_power = (np.abs(response_complex) ** 2).reshape(az_grid.shape)
    response_abs = np.sqrt(response_power)
    response_power_dB = 10.0 * np.log10(np.maximum(response_power, 1e-20))

    return {
        "azimuth_deg": az_grid,
        "elevation_deg": el_grid,
        "response_abs": response_abs,
        "response_power": response_power,
        "response_power_dB": response_power_dB,
    }

def make_scan_vectors(config: ProjectConfig) -> tuple[np.ndarray, np.ndarray]:
    """Crea los vectores de barrido angular a partir de la configuracion."""
    az = np.arange(
        config.scan.azimuth_scan_min_deg,
        config.scan.azimuth_scan_max_deg + config.scan.azimuth_scan_step_deg,
        config.scan.azimuth_scan_step_deg,
    )
    el = np.arange(
        config.scan.elevation_scan_min_deg,
        config.scan.elevation_scan_max_deg + config.scan.elevation_scan_step_deg,
        config.scan.elevation_scan_step_deg,
    )
    return az, el


def compute_elevation_response_cut_for_plot(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    weights: np.ndarray,
    elevation_scan_deg: np.ndarray,
    fixed_azimuth_deg: float,
    wavelength_m: float | None = None,
    database: MeasuredSteeringDatabase | None = None
) -> pd.DataFrame:
    """Construye el corte vertical simetrico que espera el plot polar.

    El eje polar de elevacion representa horizonte izquierdo, cenit y
    horizonte derecho. Para ello se evalua el lado izquierdo con
    `fixed_azimuth + 180` y el lado derecho con `fixed_azimuth`.
    """
    elevation_scan_deg = np.asarray(elevation_scan_deg, dtype=float)
    elevation_scan_deg = elevation_scan_deg[
        (elevation_scan_deg >= 0.0) & (elevation_scan_deg <= 90.0)
    ]

    az_right = fixed_azimuth_deg % 360.0
    az_left = (fixed_azimuth_deg + 180.0) % 360.0

    # Lado izquierdo: horizonte izquierdo -> cenit
    el_left = elevation_scan_deg
    az_left_vec = np.full_like(el_left, az_left, dtype=float)
    left = evaluate_response_for_angles(
        config,
        element_positions_m,
        weights,
        az_left_vec,
        el_left,
        wavelength_m=wavelength_m,
        database=database
    )

    left["polar_theta_deg"] = -90.0 + left["elevation_deg"]
    left["plot_side"] = "left"
    left["cut_azimuth_deg"] = az_left

    # Lado derecho: cenit -> horizonte derecho
    el_right = elevation_scan_deg[::-1]
    az_right_vec = np.full_like(el_right, az_right, dtype=float)
    right = evaluate_response_for_angles(
        config,
        element_positions_m,
        weights,
        az_right_vec,
        el_right,
        wavelength_m=wavelength_m,
        database=database
    )

    right["polar_theta_deg"] = 90.0 - right["elevation_deg"]
    right["plot_side"] = "right"
    right["cut_azimuth_deg"] = az_right

    cut = pd.concat([left, right], ignore_index=True)
    max_abs = cut["response_abs"].max()

    if max_abs == 0.0:
        cut["response_abs_normalized"] = 0.0
    else:
        cut["response_abs_normalized"] = cut["response_abs"] / max_abs
        
    with np.errstate(divide="ignore"):
        cut["response_dB_normalized"] = 20.0 * np.log10(cut["response_abs_normalized"])

    return cut
