"""
Lectura de steering medido almacenado en MATLAB.

Las tablas originales contienen 12 diferencias entre antenas.

Para este simulador únicamente se utilizan las 6 primeras:

    2-1
    3-1
    4-1
    5-1
    6-1
    7-1

Estas seis diferencias permiten reconstruir completamente el steering
relativo de los siete elementos tomando la antena central como referencia.

Las seis diferencias restantes están destinadas principalmente a
algoritmos de estimación de dirección (AOA) y no se utilizan en los
algoritmos de beamforming implementados en este proyecto.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import numpy as np
from scipy.io import loadmat

VALID_POLARIZATIONS = {"C", "V", "H"}

@dataclass(frozen=True)
class MeasuredSteeringDatabase:
    """Base de datos medida normalizada a un formato interno único."""

    # Orden interno:
    # [azimut, elevación, diferencia, frecuencia]
    phase_deg: np.ndarray
    amplitude_dB: np.ndarray
    field_complex: np.ndarray
    azimuths_deg: np.ndarray
    elevations_deg: np.ndarray
    frequencies_hz: np.ndarray

    # C = circular, V = vertical, H = horizontal.
    polarization: str


def _resolve_mat_path(path_text: str) -> Path:
    """Resuelve y valida la ruta de un fichero MAT."""

    path = Path(path_text)

    if not path.is_absolute():
        path = Path.cwd() / path

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el fichero MAT: {path}"
        )

    return path


def _wrap_angle_deg(angle_deg: np.ndarray | float) -> np.ndarray:
    """Centra ángulos o diferencias de fase en [-180, 180)."""

    angle = np.asarray(angle_deg, dtype=float)
    return (angle + 180.0) % 360.0 - 180.0


def _find_axis_by_length(
    shape: tuple[int, ...],
    expected_length: int,
    axis_name: str,
) -> int:
    """Identifica un eje por su longitud, exigiendo un resultado único."""

    matches = [
        axis
        for axis, length in enumerate(shape)
        if length == expected_length
    ]

    if len(matches) != 1:
        raise ValueError(
            f"No se pudo identificar de forma única el eje {axis_name}. "
            f"Shape={shape}, longitud esperada={expected_length}, "
            f"coincidencias={matches}."
        )

    return matches[0]


def _normalise_table_axes(
    table: np.ndarray,
    num_azimuths: int,
    num_elevations: int,
    num_frequencies: int,
) -> np.ndarray:
    """Reordena cualquier tabla válida al orden [az, el, diferencia, freq]."""

    table = np.asarray(table)

    if table.ndim != 4:
        raise ValueError(
            "Las tablas medidas deben tener exactamente cuatro dimensiones."
        )

    shape = table.shape

    az_axis = _find_axis_by_length(
        shape,
        num_azimuths,
        "azimut",
    )
    el_axis = _find_axis_by_length(
        shape,
        num_elevations,
        "elevación",
    )
    freq_axis = _find_axis_by_length(
        shape,
        num_frequencies,
        "frecuencia",
    )
    diff_axis = _find_axis_by_length(
        shape,
        12,
        "diferencias",
    )

    axes = (az_axis, el_axis, diff_axis, freq_axis)

    if len(set(axes)) != 4:
        raise ValueError(
            f"Los ejes de la tabla no son distinguibles: shape={shape}."
        )

    return np.transpose(table, axes)


def _decode_phase_table(
    encoded_phase: np.ndarray,
    nbits_phase: int,
    nbits_register: int,
) -> np.ndarray:
    """Descodifica la tabla de fase exactamente como Fct_LoadTables.m."""

    signed_phase = np.asarray(encoded_phase).astype(np.int64)

    threshold = 2 ** (nbits_register - 1) - 1
    register_modulus = 2 ** nbits_register

    signed_phase = signed_phase.copy()
    signed_phase[signed_phase > threshold] -= register_modulus

    degree_factor = 180.0 / (2 ** (nbits_phase - 1))

    return signed_phase.astype(float) * degree_factor


def _decode_amplitude_table(
    encoded_amplitude: np.ndarray,
    minimum_dB: float,
    step_dB: float,
) -> np.ndarray:
    """Descodifica amplitudes: valor*PASODIFAMP + MINDIFPA."""

    return (
        np.asarray(encoded_amplitude, dtype=float) * step_dB
        + minimum_dB
    )


def _read_scalar(mat: dict, name: str) -> float:
    """Lee una variable escalar de un diccionario devuelto por loadmat."""

    if name not in mat:
        raise ValueError(f"Falta la variable {name} en el MAT.")

    return float(np.asarray(mat[name]).squeeze())


def _check_equal_axis(
    name: str,
    phase_axis: np.ndarray,
    amplitude_axis: np.ndarray,
) -> None:
    """Comprueba que ambos ficheros utilizan exactamente el mismo eje."""

    if phase_axis.shape != amplitude_axis.shape:
        raise ValueError(
            f"El eje {name} tiene distinto tamaño en fase y amplitud."
        )

    if not np.allclose(phase_axis, amplitude_axis):
        raise ValueError(
            f"El eje {name} no coincide entre fase y amplitud."
        )


def load_measured_steering_database(
    phase_mat_path: str,
    amplitude_mat_path: str,
) -> MeasuredSteeringDatabase:
    """Carga, valida y descodifica las dos tablas MAT."""

    phase_path = _resolve_mat_path(phase_mat_path)
    amplitude_path = _resolve_mat_path(amplitude_mat_path)

    phase_polarization = polarization_from_mat_filename(str(phase_path))
    amplitude_polarization = polarization_from_mat_filename(str(amplitude_path))

    if phase_polarization != amplitude_polarization:
        raise ValueError(
            "Los ficheros MAT de fase y amplitud corresponden a "
            "polarizaciones diferentes: "
            f"fase={phase_polarization}, "
            f"amplitud={amplitude_polarization}."
        )

    polarization = phase_polarization
    print(f"Polarización detectada en tablas measured: {polarization}")

    phase_mat = loadmat(phase_path)
    amplitude_mat = loadmat(amplitude_path)

    phase_required = {
        "TablasAOAFase",
        "FRECSTAB_MHz",
        "AOAsTab_Grad",
        "ELEVSTAB_GRAD",
        "NBITSFASE",
        "NBITSREGI",
    }

    amplitude_required = {
        "TablasAOAAmpli",
        "FRECSTAB_MHz",
        "AOAsTab_Grad",
        "ELEVSTAB_GRAD",
        "MINDIFPA",
        "PASODIFAMP",
    }

    missing_phase = phase_required - set(phase_mat)
    missing_amplitude = amplitude_required - set(amplitude_mat)

    if missing_phase:
        raise ValueError(
            f"Faltan variables en {phase_path.name}: "
            f"{sorted(missing_phase)}"
        )

    if missing_amplitude:
        raise ValueError(
            f"Faltan variables en {amplitude_path.name}: "
            f"{sorted(missing_amplitude)}"
        )

    phase_azimuths = np.asarray(
        phase_mat["AOAsTab_Grad"],
        dtype=float,
    ).ravel()

    phase_elevations = np.asarray(
        phase_mat["ELEVSTAB_GRAD"],
        dtype=float,
    ).ravel()

    phase_frequencies_hz = (
        np.asarray(
            phase_mat["FRECSTAB_MHz"],
            dtype=float,
        ).ravel()
        * 1e6
    )

    amplitude_azimuths = np.asarray(
        amplitude_mat["AOAsTab_Grad"],
        dtype=float,
    ).ravel()

    amplitude_elevations = np.asarray(
        amplitude_mat["ELEVSTAB_GRAD"],
        dtype=float,
    ).ravel()

    amplitude_frequencies_hz = (
        np.asarray(
            amplitude_mat["FRECSTAB_MHz"],
            dtype=float,
        ).ravel()
        * 1e6
    )

    _check_equal_axis(
        "azimut",
        phase_azimuths,
        amplitude_azimuths,
    )
    _check_equal_axis(
        "elevación",
        phase_elevations,
        amplitude_elevations,
    )
    _check_equal_axis(
        "frecuencia",
        phase_frequencies_hz,
        amplitude_frequencies_hz,
    )

    phase_encoded = _normalise_table_axes(
        phase_mat["TablasAOAFase"],
        phase_azimuths.size,
        phase_elevations.size,
        phase_frequencies_hz.size,
    )

    amplitude_encoded = _normalise_table_axes(
        amplitude_mat["TablasAOAAmpli"],
        amplitude_azimuths.size,
        amplitude_elevations.size,
        amplitude_frequencies_hz.size,
    )

    if phase_encoded.shape != amplitude_encoded.shape:
        raise ValueError(
            "Las tablas de fase y amplitud no tienen la misma forma "
            "después de normalizar sus ejes."
        )

    nbits_phase = int(_read_scalar(phase_mat, "NBITSFASE"))
    nbits_register = int(_read_scalar(phase_mat, "NBITSREGI"))

    minimum_dB = _read_scalar(amplitude_mat, "MINDIFPA")
    step_dB = _read_scalar(amplitude_mat, "PASODIFAMP")

    phase_deg = _decode_phase_table(
        phase_encoded,
        nbits_phase,
        nbits_register,
    )

    amplitude_dB = _decode_amplitude_table(
        amplitude_encoded,
        minimum_dB,
        step_dB,
    )
    # Para beamforming únicamente se utilizan las seis diferencias
    # respecto a la antena de referencia.
    phase_deg = phase_deg[:, :, :6, :]
    amplitude_dB = amplitude_dB[:, :, :6, :]
    if phase_deg.shape[2] != 6:
        raise ValueError("La tabla de fase debe contener exactamente seis diferencias respecto al elemento de referencia.")
    if amplitude_dB.shape[2] != 6:
        raise ValueError("La tabla de amplitud debe contener exactamente seis diferencias respecto al elemento de referencia.")

    field_complex = (10.0 ** (amplitude_dB / 20.0)* np.exp(1j * np.deg2rad(phase_deg)))
    
    return MeasuredSteeringDatabase(
        phase_deg=phase_deg,
        amplitude_dB=amplitude_dB,
        field_complex=field_complex,
        azimuths_deg=phase_azimuths,
        elevations_deg=phase_elevations,
        frequencies_hz=phase_frequencies_hz,
        polarization=polarization,
    )

def _interpolation_indices(
    search_array: np.ndarray,
    target_value: float,
) -> tuple[int, int, float]:
    """Devuelve índices inferior/superior y factor de interpolación."""

    if target_value <= search_array[0]:
        return 0, 0, 0.0

    if target_value >= search_array[-1]:
        last = len(search_array) - 1
        return last, last, 0.0

    upper = int(np.searchsorted(search_array, target_value, side="right",))
    lower = upper - 1

    value_low = search_array[lower]
    value_high = search_array[upper]
    alpha = (target_value - value_low) / (value_high - value_low)

    return lower, upper, float(alpha)

def to_complex(amplitude_dB, phase_deg):
    return (10.0 ** (amplitude_dB / 20.0)* np.exp(1j * np.deg2rad(phase_deg)))


def parabola_offset(e_minus, e0, e_plus, h):
    """
    Devuelve el desplazamiento respecto al punto central.
    """

    denom = (e_minus - 2.0 * e0 + e_plus)

    if abs(denom) < 1e-12:
        return 0.0

    return (0.5 * h* (e_minus - e_plus)/ denom)

def polarization_from_mat_filename(path_text: str) -> str:
    """Obtiene la polarización C, V o H del nombre de un fichero MAT.

    Ejemplos admitidos:
        TABLASFASE_E1_C_LBADICIONALES_ALT.mat
        TABLASAMPL_E1_V_LBADICIONALES_ALT.mat
        TABLASFASE_E6_H_LBADICIONALES_ALT.mat

    La polarización se identifica mediante los tokens:
        _C_
        _V_
        _H_
    """

    filename = Path(path_text).name.upper()

    match = re.search(r"_(C|V|H)_", filename)

    if match is None:
        raise ValueError("No se pudo identificar la polarización en el nombre " + f"del fichero MAT: {filename}. " + "Se esperaba encontrar _C_, _V_ o _H_.")

    polarization = match.group(1)

    if polarization not in VALID_POLARIZATIONS:
        raise ValueError(f"Polarización no soportada en {filename}: {polarization}")

    return polarization

def measured_steering_matrix_complex_solo_freq(
    database: MeasuredSteeringDatabase,
    azimuth_deg_array: np.ndarray,
    elevation_deg_array: np.ndarray,
    frequency_hz: float,
) -> np.ndarray:

    if frequency_hz <= 0.0:
        raise ValueError(
            "frequency_hz debe ser mayor que cero."
        )

    azimuth_deg_array = np.asarray(
        azimuth_deg_array,
        dtype=float,
    ) % 360.0

    elevation_deg_array = np.asarray(
        elevation_deg_array,
        dtype=float,
    )

    az_step = (
        database.azimuths_deg[1]
        - database.azimuths_deg[0]
    )

    el_step = (
        database.elevations_deg[1]
        - database.elevations_deg[0]
    )

    az_half_step = az_step / 2.0
    el_half_step = el_step / 2.0

    az_idx = (
        (
            azimuth_deg_array
            + az_half_step
        ) / az_step
    ).astype(np.int64) % len(
        database.azimuths_deg
    )

    el_idx = np.clip(
        (
            (
                elevation_deg_array
                - database.elevations_deg[0]
                + el_half_step
            )
            / el_step
        ).astype(np.int64),
        0,
        len(database.elevations_deg) - 1,
    )

    freq_low, freq_high, freq_alpha = (
        _interpolation_indices(
            database.frequencies_hz,
            frequency_hz,
        )
    )

    e0 = database.field_complex[
        az_idx,
        el_idx,
        :,
        freq_low,
    ]

    e1 = database.field_complex[
        az_idx,
        el_idx,
        :,
        freq_high,
    ]

    field_interp = (
        e0
        + freq_alpha * (e1 - e0)
    )

    steering = np.empty(
        (
            len(azimuth_deg_array),
            7,
        ),
        dtype=np.complex128,
    )

    steering[:, 0] = 1.0 + 0.0j
    steering[:, 1:] = field_interp

    return steering

def measured_steering_vector_complex(
    database: MeasuredSteeringDatabase,
    azimuth_deg: float,
    elevation_deg: float,
    frequency_hz: float,
) -> np.ndarray:

    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz debe ser mayor que cero.")

    azimuth_deg = azimuth_deg % 360.0

    az_low, az_high, az_alpha = _interpolation_indices( database.azimuths_deg, azimuth_deg,)
    el_low, el_high, el_alpha = _interpolation_indices( database.elevations_deg, elevation_deg, )
    freq_low, freq_high, freq_alpha = _interpolation_indices(database.frequencies_hz,frequency_hz,)

    # 8 vértices del cubo

    e000 = database.field_complex[az_low,el_low, :,freq_low,]
    e100 = database.field_complex[az_high,el_low,:,freq_low,]
    e010 = database.field_complex[az_low,el_high,:,freq_low,]
    e110 = database.field_complex[az_high,el_high,:,freq_low,]
    e001 = database.field_complex[az_low,el_low,:,freq_high,]
    e101 = database.field_complex[az_high,el_low,:,freq_high,]
    e011 = database.field_complex[az_low,el_high,:,freq_high,]
    e111 = database.field_complex[az_high,el_high,:,freq_high,]

    # Interpolación trilineal COMPLEJA

    e00 = e000 + az_alpha * (e100 - e000)
    e10 = e010 + az_alpha * (e110 - e010)
    e01 = e001 + az_alpha * (e101 - e001)
    e11 = e011 + az_alpha * (e111 - e011)
    e0 = e00 + el_alpha * (e10 - e00)
    e1 = e01 + el_alpha * (e11 - e01)

    field_interp = e0 + freq_alpha * (e1 - e0)
    steering = np.empty(7,dtype=np.complex128,)

    steering[0] = 1.0 + 0.0j
    steering[1:] = field_interp

    return steering

def measured_steering_matrix_complex(
    database: MeasuredSteeringDatabase,
    azimuth_deg_array: np.ndarray,
    elevation_deg_array: np.ndarray,
    frequency_hz: float,
) -> np.ndarray:
    """
    Devuelve una matriz de steering vectors.

    Returns
    -------
    steering : ndarray
        Shape (N_points, 7)
    """

    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz debe ser mayor que cero.")

    azimuth_deg_array = (np.asarray(azimuth_deg_array, dtype=float) % 360.0)
    elevation_deg_array = np.asarray(elevation_deg_array,dtype=float,)

    if len(azimuth_deg_array) != len(elevation_deg_array):
        raise ValueError(
            "azimuth_deg_array y elevation_deg_array "
            "deben tener la misma longitud."
        )

    # ==================================================
    # Índices interpolación Angular
    # ==================================================

    az_low = np.searchsorted(database.azimuths_deg,azimuth_deg_array,side="right",) - 1
    az_low = np.clip( az_low, 0,len(database.azimuths_deg) - 1,)
    az_high = (az_low + 1) % len(database.azimuths_deg)

    az_low_val = database.azimuths_deg[az_low]
    az_high_val = database.azimuths_deg[az_high]

    az_delta = (az_high_val - az_low_val) % 360.0
    az_delta[az_delta == 0.0] = 360.0

    az_alpha = ((azimuth_deg_array - az_low_val) % 360.0) / az_delta

    # --------------------------

    el_low = np.searchsorted(database.elevations_deg, elevation_deg_array, side="right",) - 1
    el_low = np.clip(el_low,0,len(database.elevations_deg) - 2,)
    el_high = el_low + 1

    el_low_val = database.elevations_deg[el_low]
    el_high_val = database.elevations_deg[el_high]
    el_alpha = (elevation_deg_array - el_low_val) / (el_high_val - el_low_val)

    # ==================================================
    # Frecuencia
    # ==================================================

    freq_low, freq_high, freq_alpha = (_interpolation_indices(database.frequencies_hz,frequency_hz,))

    # ==================================================
    # 8 vértices
    # ==================================================

    e000 = database.field_complex[
        az_low,
        el_low,
        :,
        freq_low,
    ]

    e100 = database.field_complex[
        az_high,
        el_low,
        :,
        freq_low,
    ]

    e010 = database.field_complex[
        az_low,
        el_high,
        :,
        freq_low,
    ]

    e110 = database.field_complex[
        az_high,
        el_high,
        :,
        freq_low,
    ]

    e001 = database.field_complex[
        az_low,
        el_low,
        :,
        freq_high,
    ]

    e101 = database.field_complex[
        az_high,
        el_low,
        :,
        freq_high,
    ]

    e011 = database.field_complex[
        az_low,
        el_high,
        :,
        freq_high,
    ]

    e111 = database.field_complex[
        az_high,
        el_high,
        :,
        freq_high,
    ]

    # ==================================================
    # Broadcasting
    # ==================================================

    az_alpha = az_alpha[:, None]
    el_alpha = el_alpha[:, None]

    # ==================================================
    # Interpolación trilineal compleja
    # ==================================================

    e00 = e000 + az_alpha * (e100 - e000)
    e10 = e010 + az_alpha * (e110 - e010)
    e01 = e001 + az_alpha * (e101 - e001)
    e11 = e011 + az_alpha * (e111 - e011)
    e0 = e00 + el_alpha * (e10 - e00)
    e1 = e01 + el_alpha * (e11 - e01)

    field_interp = e0 + freq_alpha * (e1 - e0)

    # ==================================================
    # Steering matrix
    # ==================================================

    steering = np.empty((len(azimuth_deg_array),7,), dtype=np.complex128,)

    steering[:, 0] = 1.0 + 0.0j
    steering[:, 1:] = field_interp

    return steering