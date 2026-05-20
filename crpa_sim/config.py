"""config.py
Configuracion tipada del simulador CRPA.

La version limpia usa un unico modo DoA y un unico algoritmo por ejecucion:
- simulation_config.doa_mode: "fixed" o "variable"
- beamforming_config.algorithm: "power_inversion" o "lcmv"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GNSS_CARRIER_FREQUENCIES_HZ = {"E5": 1.19179e9, "E6": 1.27875e9, "E1": 1.57542e9}
GNSS_BAND_LABELS = {1: "E5", 2: "E6", 3: "E1"}
VALID_DOA_MODES = {"fixed", "variable"}
VALID_ALGORITHMS = {"power_inversion", "lcmv"}


def normalize_gnss_band(value: int | str) -> str:
    """Normaliza la banda GNSS a una etiqueta E5/E6/E1.

    Parametros:
        value: Banda como entero historico (1, 2, 3) o como texto.
    """
    if isinstance(value, int):
        return GNSS_BAND_LABELS.get(value, str(value))
    return str(value).strip().upper()


@dataclass(frozen=True)
class ArrayConfig:
    num_elements: int
    geometry: str
    element_type: str
    element_spacing_over_lambda: float
    array_boresight_elevation_deg: float
    steering_model: str = "ideal"
    measured_steering_file: str | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ArrayConfig":
        """Construye ArrayConfig normalizando tipos y textos.

        Parametros:
            values: Diccionario leido desde "array_config" del JSON.
        """
        values = dict(values)
        values["num_elements"] = int(values["num_elements"])
        values["geometry"] = str(values["geometry"]).lower()
        values["element_type"] = str(values["element_type"]).lower()
        values["steering_model"] = str(values.get("steering_model", "ideal")).lower()
        if values.get("measured_steering_file") in ("", None):
            values["measured_steering_file"] = None
        return cls(**values)


@dataclass(frozen=True)
class SignalConfig:
    gnss_band: int | str
    speed_of_light_m_s: float
    sample_rate_hz: float
    num_snapshots: int
    fft_size: int | None = None

    @property
    def band_label(self) -> str:
        """Etiqueta GNSS normalizada.

        Parametros:
            No recibe parametros; usa self.gnss_band.
        """
        return normalize_gnss_band(self.gnss_band)

    @property
    def carrier_frequency_hz(self) -> float:
        """Frecuencia portadora en Hz asociada a la banda GNSS.

        Parametros:
            No recibe parametros; usa self.band_label.
        """
        label = self.band_label
        if label not in GNSS_CARRIER_FREQUENCIES_HZ:
            raise ValueError(f"Banda GNSS no soportada: {self.gnss_band}")
        return GNSS_CARRIER_FREQUENCIES_HZ[label]

    @property
    def wavelength_m(self) -> float:
        """Longitud de onda en metros.

        Parametros:
            No recibe parametros; usa speed_of_light_m_s y carrier_frequency_hz.
        """
        return self.speed_of_light_m_s / self.carrier_frequency_hz

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SignalConfig":
        """Construye SignalConfig convirtiendo campos numericos.

        Parametros:
            values: Diccionario leido desde "signal_config" del JSON.
        """
        values = dict(values)
        values["num_snapshots"] = int(values["num_snapshots"])
        if values.get("fft_size") is not None:
            values["fft_size"] = int(values["fft_size"])
        return cls(**values)


@dataclass(frozen=True)
class SimulationConfig:
    num_montecarlo: int
    random_seed: int
    doa_mode: str

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SimulationConfig":
        """Construye SimulationConfig y valida el modo DoA.

        Parametros:
            values: Diccionario leido desde "simulation_config" del JSON.
        """
        values = dict(values)
        values["num_montecarlo"] = int(values["num_montecarlo"])
        values["random_seed"] = int(values["random_seed"])
        values["doa_mode"] = str(values.get("doa_mode", "fixed")).lower()
        if values["doa_mode"] not in VALID_DOA_MODES:
            raise ValueError(f"doa_mode debe ser uno de {sorted(VALID_DOA_MODES)}")
        return cls(**values)


@dataclass(frozen=True)
class BeamformingConfig:
    algorithm: str
    desired_azimuth_deg: float
    desired_elevation_deg: float
    diagonal_loading_factor: float
    power_inversion_reference_element: int = 0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BeamformingConfig":
        """Construye BeamformingConfig y valida el algoritmo.

        Parametros:
            values: Diccionario leido desde "beamforming_config" del JSON.
        """
        values = dict(values)
        values["algorithm"] = str(values.get("algorithm", "lcmv")).lower()
        if values["algorithm"] not in VALID_ALGORITHMS:
            raise ValueError(f"algorithm debe ser uno de {sorted(VALID_ALGORITHMS)}")
        values["power_inversion_reference_element"] = int(values.get("power_inversion_reference_element", 0))
        return cls(**values)


@dataclass(frozen=True)
class ScanConfig:
    azimuth_scan_min_deg: float
    azimuth_scan_max_deg: float
    azimuth_scan_step_deg: float
    elevation_scan_min_deg: float
    elevation_scan_max_deg: float
    elevation_scan_step_deg: float
    null_thresholds_dB: list[float]

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ScanConfig":
        """Construye ScanConfig y normaliza umbrales de nulo.

        Parametros:
            values: Diccionario leido desde "scan_config" del JSON.
        """
        values = dict(values)
        values["null_thresholds_dB"] = [float(x) for x in values.get("null_thresholds_dB", [-10, -20, -30, -40])]
        return cls(**values)


@dataclass(frozen=True)
class NoiseConfig:
    noise_power_linear: float

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "NoiseConfig":
        """Construye NoiseConfig.

        Parametros:
            values: Diccionario leido desde "noise_config" del JSON.
        """
        return cls(**values)


@dataclass(frozen=True)
class JammerTemplate:
    name: str
    azimuth_deg: float
    elevation_deg: float
    signal_type: str = "tone"
    normalized_frequency: float = 0.0
    bandwidth_hz: float | None = None
    chirp_frequency: float | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "JammerTemplate":
        """Construye una plantilla de jammer definida en configuracion.

        Parametros:
            values: Diccionario de un elemento de "base_jammers".
        """
        values = dict(values)
        values["signal_type"] = str(values.get("signal_type", "complex_gaussian")).lower()
        values.setdefault("normalized_frequency", 0.0)
        values.setdefault("bandwidth_hz", None)
        values.setdefault("chirp_frequency", None)
        return cls(**values)


@dataclass(frozen=True)
class JammerInstance:
    name: str
    azimuth_deg: float
    elevation_deg: float
    jnr_dB: float
    signal_type: str = "complex_gaussian"
    normalized_frequency: float = 0.0
    bandwidth_hz: float | None = None
    chirp_frequency: float | None = None


@dataclass(frozen=True)
class JammerConfig:
    num_jammers: int
    jnr_dB: float
    variable_doa_azimuth_range_deg: tuple[float, float]
    variable_doa_elevation_range_deg: tuple[float, float]
    base_jammers: list[JammerTemplate]

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "JammerConfig":
        """Construye JammerConfig y sus plantillas de jammers.

        Parametros:
            values: Diccionario leido desde "jammer_config" del JSON.
        """
        values = dict(values)
        values["num_jammers"] = int(values["num_jammers"])
        values["jnr_dB"] = float(values["jnr_dB"])
        values["variable_doa_azimuth_range_deg"] = tuple(float(x) for x in values["variable_doa_azimuth_range_deg"])
        values["variable_doa_elevation_range_deg"] = tuple(float(x) for x in values["variable_doa_elevation_range_deg"])
        values["base_jammers"] = [JammerTemplate.from_dict(x) for x in values["base_jammers"]]
        return cls(**values)


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str
    save_csv: bool = True
    save_npz: bool = True
    save_plots: bool = True
    csv_separator: str = ";"
    csv_decimal: str = ","

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "OutputConfig":
        """Construye OutputConfig.

        Parametros:
            values: Diccionario leido desde "output_config" del JSON.
        """
        return cls(**values)


@dataclass(frozen=True)
class ProjectConfig:
    array: ArrayConfig
    signal: SignalConfig
    simulation: SimulationConfig
    beamforming: BeamformingConfig
    scan: ScanConfig
    noise: NoiseConfig
    jammer: JammerConfig
    output: OutputConfig

    @property
    def element_spacing_m(self) -> float:
        """Separacion fisica entre centro y elementos exteriores.

        Parametros:
            No recibe parametros; usa element_spacing_over_lambda y wavelength_m.
        """
        return self.array.element_spacing_over_lambda * self.signal.wavelength_m
