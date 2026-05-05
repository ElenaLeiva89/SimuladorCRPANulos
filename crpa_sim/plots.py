"""plots.py: generación de figuras."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _db_to_polar_radius(response_dB: np.ndarray, min_display_dB: float) -> np.ndarray:
    clipped = np.maximum(response_dB, min_display_dB)
    return clipped - min_display_dB


def plot_pattern_comparison_azimuth(
    conventional_pattern: pd.DataFrame,
    adaptive_pattern: pd.DataFrame,
    output_path: Path,
    title: str,
    jammer_azimuths_deg: list[float] | None = None,
    adaptive_label: str = "Patrón con pesos seleccionados",
) -> None:
    """
    Compara el patrón convencional y el patrón del algoritmo elegido en azimut.

    Parameters:
    - conventional_pattern (pd.DataFrame): DataFrame con las columnas 'azimuth_deg' y 'response_dB_normalized' para el patrón convencional.
    - adaptive_pattern (pd.DataFrame): DataFrame con las columnas 'azimuth_deg' y 'response_dB_normalized' para el patrón adaptativo.
    - output_path (Path): Ruta donde guardar la figura.
    - title (str): Título de la figura.
    - jammer_azimuths_deg (list[float] | None): Lista de azimuts de los jammers en grados. Si None, no se marcan jammers.
    - adaptive_label (str): Etiqueta para el patrón adaptativo.
    """
    fig = plt.figure(figsize=(14, 6))
    min_display_dB = -50.0
    radial_ticks = [0, 10, 20, 30, 40, 50]

    for idx, (data, label, color) in enumerate(
        [(conventional_pattern, "Convencional", "green"), (adaptive_pattern, adaptive_label, "blue")], start=1
    ):
        ax = fig.add_subplot(1, 2, idx, projection="polar")
        theta = np.deg2rad(data["azimuth_deg"].to_numpy())
        radius = _db_to_polar_radius(data["response_dB_normalized"].to_numpy(), min_display_dB)
        ax.plot(theta, radius, linewidth=2, label=label, color=color)

        if idx == 2 and jammer_azimuths_deg is not None:
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown']
            for i, az in enumerate(jammer_azimuths_deg):
                jammer_rad = np.deg2rad(az)
                radii = np.linspace(0, abs(min_display_dB), 100)
                color = colors[i % len(colors)]
                ax.plot(np.full_like(radii, jammer_rad), radii, color=color, linestyle="--", linewidth=2, label=f"Jammer {i+1} ({az:.1f}°)")

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_rlim(0, abs(min_display_dB))
        ax.set_rticks(radial_ticks)
        ax.set_yticklabels([f"{tick + min_display_dB:.0f} dB" for tick in radial_ticks])
        ax.grid(True)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
        ax.set_title("1. Patrón convencional" if idx == 1 else "2. Patrón del algoritmo", fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pattern_comparison_elevation(
    conventional_pattern: pd.DataFrame,
    adaptive_pattern: pd.DataFrame,
    output_path: Path,
    title: str,
    jammer_elevations_deg: list[float] | None = None,
    adaptive_label: str = "Patrón con pesos seleccionados",
) -> None:
    """
    Compara el patrón convencional y el patrón del algoritmo elegido en elevación.

    Parameters:
    - conventional_pattern (pd.DataFrame): DataFrame con las columnas 'elevation_deg' y 'response_dB_normalized' para el patrón convencional.
    - adaptive_pattern (pd.DataFrame): DataFrame con las columnas 'elevation_deg' y 'response_dB_normalized' para el patrón adaptativo.
    - output_path (Path): Ruta donde guardar la figura.
    - title (str): Título de la figura.
    - jammer_elevations_deg (list[float] | None): Lista de elevaciones de los jammers en grados. Si None, no se marcan jammers.
    - adaptive_label (str): Etiqueta para el patrón adaptativo.
    """
    fig = plt.figure(figsize=(14, 6))
    min_display_dB = -50.0
    radial_ticks = [0, 10, 20, 30, 40, 50]

    for idx, (data, label, color) in enumerate(
        [(conventional_pattern, "Convencional", "green"), (adaptive_pattern, adaptive_label, "blue")], start=1
    ):
        ax = fig.add_subplot(1, 2, idx, projection="polar")
        theta = np.deg2rad(data["elevation_deg"].to_numpy())
        radius = _db_to_polar_radius(data["response_dB_normalized"].to_numpy(), min_display_dB)
        ax.plot(theta + np.deg2rad(90), radius, linewidth=2, label=label, color=color)

        if idx == 2 and jammer_elevations_deg is not None:
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown']
            for i, el in enumerate(jammer_elevations_deg):
                jammer_rad = np.deg2rad(el)
                radii = np.linspace(0, abs(min_display_dB), 100)
                color = colors[i % len(colors)]
                ax.plot(np.full_like(radii, jammer_rad), radii, color=color, linestyle="--", linewidth=2, label=f"Jammer {i+1} ({el:.1f}°)")

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetamin(180)
        ax.set_thetamax(0)
        ax.set_rlim(0, abs(min_display_dB))
        ax.set_rticks(radial_ticks)
        ax.set_yticklabels([f"{tick + min_display_dB:.0f} dB" for tick in radial_ticks])
        ax.grid(True)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
        ax.set_title("1. Patrón convencional" if idx == 1 else "2. Patrón del algoritmo", fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _crpa_display_labels_clockwise(element_positions_m: np.ndarray) -> list[str]:
    """Genera etiquetas visuales para la CRPA.

    Convención solicitada:
    - Elemento central -> etiqueta 1.
    - Elemento situado a la derecha de la imagen, eje +X -> etiqueta 2.
    - Resto de elementos exteriores -> sentido horario.

    Esta función no cambia el orden interno de los elementos en los cálculos;
    solo cambia las etiquetas mostradas en las figuras.
    """
    x = element_positions_m[:, 0]
    y = element_positions_m[:, 1]

    labels = [""] * len(element_positions_m)
    labels[0] = "1"

    # Ángulos de los seis elementos exteriores en rango [0, 2*pi).
    exterior_angles = np.mod(np.arctan2(y[1:], x[1:]), 2.0 * np.pi)

    # Orden horario desde +X: 0°, 300°, 240°, 180°, 120°, 60°.
    clockwise_order_local = np.argsort(-exterior_angles)

    # Rotar para que el elemento más cercano a +X sea el primero.
    rightmost_local = int(np.argmin(np.abs(exterior_angles)))
    start_pos = int(np.where(clockwise_order_local == rightmost_local)[0][0])
    clockwise_order_local = np.roll(clockwise_order_local, -start_pos)

    for display_label, local_idx in enumerate(clockwise_order_local, start=2):
        real_idx = int(local_idx + 1)
        labels[real_idx] = str(display_label)

    return labels


def plot_array_geometry(element_positions_m: np.ndarray, output_path: Path) -> None:
    """Dibuja la geometría XY y 3D de la CRPA.

    La numeración mostrada en la figura sigue la convención:
    centro=1, elemento de la derecha=2 y resto en sentido horario.
    """
    fig = plt.figure(figsize=(10, 5))
    x, y, z = element_positions_m[:, 0], element_positions_m[:, 1], element_positions_m[:, 2]
    display_labels = _crpa_display_labels_clockwise(element_positions_m)

    ax1 = fig.add_subplot(121)
    ax1.scatter(x, y, s=160, edgecolors="black")
    for label, xi, yi in zip(display_labels, x, y):
        ax1.annotate(label, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=10, fontweight="bold")
    ax1.set_xlabel("X (m)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Y (m)", fontsize=10, fontweight="bold")
    ax1.set_title("Vista XY", fontsize=12, fontweight="bold")
    ax1.axis("equal")
    ax1.grid(True)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(x, y, z, s=160, edgecolors="black")
    for label, xi, yi, zi in zip(display_labels, x, y, z):
        ax2.text(xi, yi, zi, label, fontsize=10, fontweight="bold")
    ax2.set_xlabel("X (m)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Y (m)", fontsize=10, fontweight="bold")
    ax2.set_zlabel("Z (m)", fontsize=10, fontweight="bold")
    ax2.set_title("Vista 3D", fontsize=12, fontweight="bold")

    fig.suptitle("Geometría del Array CRPA de 7 elementos", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pattern_comparison_3d(
    conventional_grid: dict[str, np.ndarray],
    adaptive_grid: dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    """Dibuja superficies 3D del array factor para patrón convencional y adaptativo."""
    fig = plt.figure(figsize=(18, 8))
    for idx, (grid, label) in enumerate(
        [(conventional_grid, "Convencional"), (adaptive_grid, "Adaptativo")], start=1
    ):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        surf = ax.plot_surface(
            grid["azimuth_deg"],
            grid["elevation_deg"],
            grid["response_dB_normalized"],
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            rcount=120,
            ccount=120,
        )
        ax.set_xlabel("Azimuth [deg]")
        ax.set_ylabel("Elevation [deg]")
        ax.set_zlabel("Response (dB)")
        ax.set_title(label)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label="Response (dB)")
        ax.view_init(elev=30, azim=-120)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_array_factor_heatmap_comparison(
    conventional_grid: dict[str, np.ndarray],
    adaptive_grid: dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    """Dibuja un mapa de calor 2D del array factor para patrón convencional y adaptativo."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

    for ax, grid, label in zip(
        axes,
        [conventional_grid, adaptive_grid],
        ["Convencional", "Adaptativo"],
    ):
        pcm = ax.pcolormesh(
            grid["azimuth_deg"],
            grid["elevation_deg"],
            grid["response_dB_normalized"],
            shading="auto",
            cmap="viridis",
        )
        ax.set_xlabel("Azimuth [deg]")
        ax.set_ylabel("Elevation [deg]")
        ax.set_title(label)
        fig.colorbar(pcm, ax=ax, pad=0.01, label="Response (dB)")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_temporal_spectrum(spectrum_table: pd.DataFrame, output_path: Path, title: str) -> None:
    """Dibuja FFT temporal media de los snapshots en frecuencia real."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(spectrum_table["frequency_hz"], spectrum_table["power_dB_normalized"], linewidth=1.5)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Potencia normalizada (dB)")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
