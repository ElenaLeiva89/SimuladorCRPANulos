"""array_model.py
Geometria CRPA y steering vectors.

Soporta dos modelos:
- steering_model="ideal": steering vector teorico con elementos isotropicos.
- steering_model="measured": steering vector interpolado/seleccionado desde medidas reales.

Formato CSV esperado para steering_model="measured":
    azimuth_deg,elevation_deg,element_index,amplitude,phase_deg
    0,90,0,1.00,0.0
    0,90,1,0.98,12.5
    ...

Notas:
- element_index puede venir indexado desde 0 o desde 1; se normaliza internamente.
- Si hay varias muestras cercanas, se usa el punto medido mas cercano en distancia angular az/el.
- El azimut se trata como circular: 330 deg equivale a -30 deg.
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

    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    element_index: np.ndarray
    steering_value: np.ndarray
    num_elements: int


_MEASURED_STEERING_CACHE: dict[tuple[str, int], MeasuredSteeringTable] = {}


def create_crpa_geometry(array_config: ArrayConfig, element_spacing_m: float) -> np.ndarray:
    """Crea las posiciones 3D de la CRPA ideal hexagonal de 7 elementos.

    Parametros:
        array_config: Configuracion geometrica del array; debe indicar
            geometry="hexagonal_7" y num_elements=7.
        element_spacing_m: Separacion radial entre el elemento central y
            cada elemento exterior, expresada en metros.
    """
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
    """Convierte azimut/elevacion a vector unitario 3D.

    Parametros:
        azimuth_deg: Angulo de azimut en grados; 0 apunta a +X y crece hacia +Y.
        elevation_deg: Angulo de elevacion en grados; 0 es horizonte y 90 cenit.
    """
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


def _angular_delta_deg(a_deg: np.ndarray, b_deg: float) -> np.ndarray:
    """Diferencia angular circular a-b en grados, rango [-180, 180)."""
    return ((a_deg - b_deg + 180.0) % 360.0) - 180.0


def _resolve_measured_file(path_text: str) -> Path:
    """Resuelve rutas relativas de forma robusta desde el directorio actual."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path.cwd() / path


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
                element_index = int(float(row[normalized_fields["element_index"]]))

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
            element_indices.append(element_index)
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

    table = MeasuredSteeringTable(
        azimuth_deg=np.asarray(azimuths, dtype=float),
        elevation_deg=np.asarray(elevations, dtype=float),
        element_index=element_indices_array,
        steering_value=np.asarray(values, dtype=complex),
        num_elements=int(num_elements),
    )
    _MEASURED_STEERING_CACHE[cache_key] = table
    return table


def steering_vector_measured(
    config: ProjectConfig,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """Devuelve steering vector medido usando el punto angular mas cercano.

    Para cada elemento de la CRPA se selecciona la muestra medida mas cercana
    en azimut/elevacion. Esto evita depender de scipy y funciona con mallas
    medidas irregulares.
    """
    measured_file = config.array.measured_steering_file
    if not measured_file:
        raise ValueError(
            "steering_model='measured' requiere array_config.measured_steering_file en el JSON."
        )

    table = load_measured_steering_table(measured_file, config.array.num_elements)

    az_target = float(_wrap_angle_180(azimuth_deg))
    el_target = float(elevation_deg)

    result = np.zeros(config.array.num_elements, dtype=complex)

    for element_idx in range(config.array.num_elements):
        mask = table.element_index == element_idx
        if not np.any(mask):
            raise ValueError(
                f"El fichero medido no contiene muestras para element_index={element_idx}."
            )

        az_candidates = table.azimuth_deg[mask]
        el_candidates = table.elevation_deg[mask]
        val_candidates = table.steering_value[mask]

        az_delta = _angular_delta_deg(az_candidates, az_target)
        el_delta = el_candidates - el_target
        nearest_idx = int(np.argmin(az_delta**2 + el_delta**2))
        result[element_idx] = val_candidates[nearest_idx]

    return result


def steering_vector(
    config: ProjectConfig,
    element_positions_m: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """Devuelve el steering vector segun el modelo configurado.

    Parametros:
        config: Configuracion completa, usada para escoger el modelo y la
            longitud de onda.
        element_positions_m: Matriz (N, 3) con posiciones XYZ del array.
        azimuth_deg: Azimut de evaluacion en grados.
        elevation_deg: Elevacion de evaluacion en grados.
    """
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
