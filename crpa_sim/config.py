"""
config.py
---------
Dataclasses de configuración del simulador CRPA.

Este módulo NO ejecuta simulaciones. Solo define:
- parámetros globales de simulación,
- parámetros físicos/geométricos del escenario,
- parámetros de cada jammer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

GNSS_BAND_LABELS = {1: "E5", 2: "E6", 3: "E1"}
GNSS_CARRIER_FREQUENCIES_HZ = {"E5": 1.19179e9, "E6": 1.27875e9, "E1": 1.57542e9}


def normalize_gnss_band(value: Any) -> str:
    """Convierte 1/2/3 o 'E5'/'E6'/'E1' a una etiqueta estándar."""
    if isinstance(value, int):
        return GNSS_BAND_LABELS.get(value, str(value))
    return str(value).strip().upper()


@dataclass
class SimulationConfig:
    """Configuración general de ejecución.

    nsimulations:
        Número de iteraciones Monte Carlo. En esta fase básica se carga,
        pero main.py ejecuta una única simulación. Se usará más adelante.
    gnssBand:
        Banda GNSS: 1=E5, 2=E6, 3=E1 o etiqueta "E5", "E6", "E1".
    maxPhaseNoise_deg:
        Ruido de fase máximo previsto para fases futuras.
    maxAmplNoise_dB:
        Ruido de amplitud máximo previsto para fases futuras.
    interferenceType:
        Lista de etiquetas/tipos de interferencia.
    algorithmType:
        Algoritmo de pesos: "conventional", "lcmv", "power_inversion".
    use_fft_pattern_for_ula:
        Si True, permite calcular un patrón rápido por FFT SOLO para ULA.
        Para la CRPA hexagonal 2D debe mantenerse False.
    """

    nsimulations: int
    gnssBand: int | str
    maxPhaseNoise_deg: float
    maxAmplNoise_dB: float
    interferenceType: list[int]
    algorithmType: str = "conventional"
    use_fft_pattern_for_ula: bool = False
    numberInterferences: int = field(init=False)

    def __post_init__(self) -> None:
        self.numberInterferences = len(self.interferenceType)
        self.algorithmType = self.algorithmType.lower()

    @property
    def band_label(self) -> str:
        return normalize_gnss_band(self.gnssBand)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SimulationConfig":
        values = dict(values)
        if "nsimulations" in values:
            values["nsimulations"] = int(values["nsimulations"])
        return cls(**values)


@dataclass
class ScenarioConfig:
    """Configuración física y angular del escenario CRPA."""

    speed_of_light_m_s: float
    num_elements: int
    element_spacing_over_lambda: float
    num_snapshots: int
    noise_power_linear: float
    desired_azimuth_deg: float
    desired_elevation_deg: float
    azimuth_scan_min_deg: float
    azimuth_scan_max_deg: float
    azimuth_scan_step_deg: float
    fixed_azimuth_cut_deg: float
    elevation_scan_min_deg: float
    elevation_scan_max_deg: float
    elevation_scan_step_deg: float
    random_seed: int
    output_dir: str
    carrier_frequency_hz: float | None = None
    diagonal_loading_factor: float = 1e-3

    def set_carrier_frequency_from_simulation(self, simulation_config: SimulationConfig) -> None:
        band_label = normalize_gnss_band(simulation_config.gnssBand)
        if band_label not in GNSS_CARRIER_FREQUENCIES_HZ:
            raise ValueError(f"Banda GNSS no válida: {simulation_config.gnssBand}")
        self.carrier_frequency_hz = GNSS_CARRIER_FREQUENCIES_HZ[band_label]

    @property
    def wavelength_m(self) -> float:
        if self.carrier_frequency_hz is None:
            raise ValueError("carrier_frequency_hz no definido. Carga la banda GNSS primero.")
        return self.speed_of_light_m_s / self.carrier_frequency_hz

    @property
    def element_spacing_m(self) -> float:
        return self.element_spacing_over_lambda * self.wavelength_m

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ScenarioConfig":
        values = dict(values)
        values["num_snapshots"] = int(values["num_snapshots"])
        values["num_elements"] = int(values["num_elements"])
        values["random_seed"] = int(values["random_seed"])
        return cls(**values)


@dataclass
class JammerConfig:
    """Configuración de un jammer/interferente."""

    name: str
    azimuth_deg: float
    elevation_deg: float
    jnr_dB: float
    signal_type: str = "complex_gaussian"
    normalized_frequency: float = 0.05

    def __post_init__(self) -> None:
        self.signal_type = self.signal_type.lower()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "JammerConfig":
        return cls(**values)
