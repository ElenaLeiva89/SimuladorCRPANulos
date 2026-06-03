"""array_model.py
Geometria CRPA y steering vectors.

Soporta dos modelos:
- steering_model="ideal": steering vector teorico con elementos isotropicos.
- steering_model="measured": steering vector seleccionado desde medidas reales.

Formato CSV esperado para steering_model="measured":
    azimuth_deg,elevation_deg,element_index,amplitude,phase_deg
    0,90,0,1.00,0.0
    0,90,1,0.98,12.5
    ...

o alternativamente:
    azimuth_deg,elevation_deg,element_index,real,imag

Notas:
- element_index puede venir indexado desde 0 o desde 1; se normaliza internamente.
- El azimut se trata como circular: 330 deg equivale a -30 deg.
- El CSV se lee una sola vez y se guarda en cache.
- Para el modelo measured se construye una tabla/tensor de steering por direccion
  medida: una fila por direccion (az, el) y una columna por elemento.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np

from .config import ArrayConfig, ProjectConfig


@dataclass(frozen=True)
class MeasuredSteeringTable:
    """Tabla interna con diagramas/steering medidos de la CRPA."""

    # Formato original fila-a-fila, se mantiene por compatibilidad.
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    element_index: np.ndarray
    steering_value: np.ndarray
    num_elements: int

    # Formato optimizado: una direccion medida -> steering completo de N elementos.
    measured_azimuth_deg: np.ndarray
    measured_elevation_deg: np.ndarray
    measured_steering_matrix: np.ndarray  # shape = (num_directions, num_elements)


_MEASURED_STEERING_CACHE: dict[tuple[str, int], MeasuredSteeringTable] = {}


def create_crpa_geometry(array_config: ArrayConfig, element_spacing_m: float) -> np.ndarray:
    """Crea las posiciones 3D de la CRPA ideal hexagonal de 7 elementos."""
    if array_config.geometry != "hexagonal_7" or array_config.num_elements != 7:
        raise ValueError("Actualmente solo se implementa geometry='hexagonal_7' con num_elements=7.")

    positions_m = np.zeros((7, 3), dtype=float)
    positions_m[0] = [0.0, 0.0, 0.0]
    for k in range(6):
        angle_rad = 2.0 * np.pi * k / 6.0
        positions_m[k + 1] = [
            element_spacing_m * np.cos(angle_rad),
            element_spacing_m * np.sin(angle_rad),
            0.0,
        ]
    return positions_m


def direction_unit_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Convierte azimut/elevacion a vector unitario 3D."""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def steering_vector_ideal(
    element_positions_m: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    wavelength_m: float,
) -> np.ndarray:
    """Calcula el steering vector ideal para elementos isotropicos."""
    k_rad_m = 2.0 * np.pi / wavelength_m
    u = direction_unit_vector(azimuth_deg, elevation_deg)
    phase_rad = k_rad_m * (element_positions_m @ u)
    
    return np.exp(1j * phase_rad)


def _wrap_angle_180(angle_deg: np.ndarray | float) -> np.ndarray | float:
    """Normaliza azimut al rango [-180, 180)."""
    return ((np.asarray(angle_deg) + 180.0) % 360.0) - 180.0


def _angular_delta_deg(a_deg: np.ndarray, b_deg: np.ndarray | float) -> np.ndarray:
    """Diferencia angular circular a-b en grados, rango [-180, 180)."""
    return ((a_deg - b_deg + 180.0) % 360.0) - 180.0


def _resolve_measured_file(path_text: str) -> Path:
    """Resuelve rutas relativas de forma robusta desde el directorio actual."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _build_direction_steering_matrix(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    element_index: np.ndarray,
    steering_value: np.ndarray,
    num_elements: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye una matriz optimizada de steering completo por direccion.

    Devuelve:
        measured_azimuth_deg: vector de azimuts medidos unicos.
        measured_elevation_deg: vector de elevaciones medidas unicas.
        measured_steering_matrix: matriz (num_directions, num_elements).

    Si hay varias muestras para la misma direccion/elemento, se promedia su
    valor complejo; esto cubre la costura circular -180/180 del CSV medido.
    """
    # Redondeo solo para clave de agrupacion y evitar problemas de coma flotante
    # en CSV con valores como 44.999999999.
    direction_keys = np.column_stack([np.round(azimuth_deg, 10), np.round(elevation_deg, 10)])
    unique_keys, inverse = np.unique(direction_keys, axis=0, return_inverse=True)

    steering_sum = np.zeros((unique_keys.shape[0], num_elements), dtype=complex)
    sample_count = np.zeros((unique_keys.shape[0], num_elements), dtype=int)

    for row_idx, dir_idx in enumerate(inverse):
        elem_idx = int(element_index[row_idx])
        steering_sum[dir_idx, elem_idx] += steering_value[row_idx]
        sample_count[dir_idx, elem_idx] += 1

    incomplete = np.argwhere(sample_count == 0)
    if incomplete.size:
        dir_idx, elem_idx = incomplete[0]
        raise ValueError(
            "El fichero medido debe tener una muestra por cada elemento en cada direccion. "
            f"Falta element_index={elem_idx} para az={unique_keys[dir_idx, 0]}, el={unique_keys[dir_idx, 1]}."
        )

    steering_matrix = steering_sum / sample_count

    return unique_keys[:, 0].astype(float), unique_keys[:, 1].astype(float), steering_matrix


def load_measured_steering_table(path_text: str, num_elements: int) -> MeasuredSteeringTable:
    """Carga un CSV con amplitud/fase o real/imag de cada elemento.

    Columnas admitidas:
      Opcion A: azimuth_deg,elevation_deg,element_index,amplitude,phase_deg
      Opcion B: azimuth_deg,elevation_deg,element_index,real,imag
    """
    path = _resolve_measured_file(path_text)
    cache_key = (str(path.resolve()), int(num_elements))
    if cache_key in _MEASURED_STEERING_CACHE:
        return _MEASURED_STEERING_CACHE[cache_key]

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro el fichero de steering medido: {path}. "
            "Configura array_config.measured_steering_file con una ruta valida."
        )

    azimuths: list[float] = []
    elevations: list[float] = []
    element_indices: list[int] = []
    values: list[complex] = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"El fichero {path} no tiene cabecera CSV.")

        normalized_fields = {name.strip().lower(): name for name in reader.fieldnames}
        required = {"azimuth_deg", "elevation_deg", "element_index"}
        missing = required - set(normalized_fields)
        if missing:
            raise ValueError(f"Faltan columnas obligatorias en {path}: {sorted(missing)}")

        has_amp_phase = "amplitude" in normalized_fields and "phase_deg" in normalized_fields
        has_real_imag = "real" in normalized_fields and "imag" in normalized_fields
        if not has_amp_phase and not has_real_imag:
            raise ValueError(
                f"{path} debe contener amplitude+phase_deg o real+imag, ademas de azimuth/elevation/element_index."
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                az = float(row[normalized_fields["azimuth_deg"]])
                el = float(row[normalized_fields["elevation_deg"]])
                elem_idx = int(float(row[normalized_fields["element_index"]]))

                if has_amp_phase:
                    amplitude = float(row[normalized_fields["amplitude"]])
                    phase_deg = float(row[normalized_fields["phase_deg"]])
                    value = amplitude * np.exp(1j * np.deg2rad(phase_deg))
                else:
                    real = float(row[normalized_fields["real"]])
                    imag = float(row[normalized_fields["imag"]])
                    value = real + 1j * imag
            except Exception as exc:
                raise ValueError(f"Fila {line_number} invalida en {path}: {row}") from exc

            azimuths.append(float(_wrap_angle_180(az)))
            elevations.append(el)
            element_indices.append(elem_idx)
            values.append(value)

    if not values:
        raise ValueError(f"El fichero {path} no contiene muestras de steering.")

    element_indices_array = np.asarray(element_indices, dtype=int)

    # Permite ficheros indexados 1..N. Internamente se usa 0..N-1.
    if element_indices_array.min() == 1 and element_indices_array.max() == num_elements:
        element_indices_array = element_indices_array - 1

    if element_indices_array.min() < 0 or element_indices_array.max() >= num_elements:
        raise ValueError(
            f"element_index fuera de rango en {path}. Debe ser 0..{num_elements-1} o 1..{num_elements}."
        )

    az_array = np.asarray(azimuths, dtype=float)
    el_array = np.asarray(elevations, dtype=float)
    steering_array = np.asarray(values, dtype=complex)

    measured_az, measured_el, measured_matrix = _build_direction_steering_matrix(
        az_array,
        el_array,
        element_indices_array,
        steering_array,
        int(num_elements),
    )

    table = MeasuredSteeringTable(
        azimuth_deg=az_array,
        elevation_deg=el_array,
        element_index=element_indices_array,
        steering_value=steering_array,
        num_elements=int(num_elements),
        measured_azimuth_deg=measured_az,
        measured_elevation_deg=measured_el,
        measured_steering_matrix=measured_matrix,
    )
    _MEASURED_STEERING_CACHE[cache_key] = table
    return table


def measured_steering_matrix_for_angles(
    config: ProjectConfig,
    azimuth_deg_array: np.ndarray,
    elevation_deg_array: np.ndarray,
    chunk_size: int = 20000,
) -> np.ndarray:
    """Devuelve un steering medido por cada par angular de entrada.

    Salida:
        Matriz compleja de forma (num_direcciones, num_elementos).

    Implementacion:
        - El CSV ya esta en cache.
        - La busqueda nearest-neighbor se hace vectorizada contra las direcciones
          medidas completas, no elemento a elemento.
        - Se procesa por bloques para no consumir memoria excesiva si la malla es grande.
    """
    measured_file = config.array.measured_steering_file
    if not measured_file:
        raise ValueError("steering_model='measured' requiere array_config.measured_steering_file en el JSON.")

    table = load_measured_steering_table(measured_file, config.array.num_elements)

    az_targets = np.asarray(_wrap_angle_180(azimuth_deg_array), dtype=float).ravel()
    el_targets = np.asarray(elevation_deg_array, dtype=float).ravel()
    if az_targets.shape != el_targets.shape:
        raise ValueError("azimuth_deg_array y elevation_deg_array deben tener la misma longitud.")

    output = np.empty((az_targets.size, table.num_elements), dtype=complex)
    measured_az = table.measured_azimuth_deg
    measured_el = table.measured_elevation_deg

    for start in range(0, az_targets.size, chunk_size):
        stop = min(start + chunk_size, az_targets.size)
        az_block = az_targets[start:stop, None]
        el_block = el_targets[start:stop, None]

        az_delta = _angular_delta_deg(measured_az[None, :], az_block)
        el_delta = measured_el[None, :] - el_block
        nearest_dir_idx = np.argmin(az_delta**2 + el_delta**2, axis=1)
        output[start:stop, :] = table.measured_steering_matrix[nearest_dir_idx, :]

    return output


def steering_vector_measured(
    config: ProjectConfig,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """Devuelve un steering vector medido para una direccion concreta."""
    return measured_steering_matrix_for_angles(
        config,
        np.asarray([azimuth_deg], dtype=float),
        np.asarray([elevation_deg], dtype=float),
    )[0]


def steering_vector(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """Devuelve el steering vector segun el modelo configurado."""
    if config.array.steering_model == "ideal":
        return steering_vector_ideal(
            element_positions_m,
            azimuth_deg,
            elevation_deg,
            config.signal.wavelength_m,
        )
    if config.array.steering_model == "measured":
        return steering_vector_measured(config, azimuth_deg, elevation_deg)
    raise ValueError(f"Modelo steering no soportado: {config.array.steering_model}")
