"""plots.py
Figuras de geometría, cortes, mapas 2D y superficies 3D.

Se mantiene un estilo similar a la versión previa:
- títulos en negrita,
- etiquetas claras,
- comparación convencional vs algoritmo,
- líneas de jammers en los cortes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _db_to_radius(response_dB: np.ndarray, min_display_dB: float) -> np.ndarray:
    return np.maximum(response_dB, min_display_dB) - min_display_dB


def _crpa_display_labels_clockwise(element_positions_m: np.ndarray) -> list[str]:
    """Etiquetas visuales: centro=1, +X=2, resto horario."""
    x = element_positions_m[:, 0]
    y = element_positions_m[:, 1]
    labels = [""] * len(element_positions_m)
    labels[0] = "1"

    angles = np.mod(np.arctan2(y[1:], x[1:]), 2.0 * np.pi)
    order = np.argsort(-angles)
    rightmost = int(np.argmin(np.abs(angles)))
    start_pos = int(np.where(order == rightmost)[0][0])
    order = np.roll(order, -start_pos)

    for visual_label, local_idx in enumerate(order, start=2):
        labels[int(local_idx + 1)] = str(visual_label)
    return labels


def plot_array_geometry(element_positions_m: np.ndarray, output_path: Path) -> None:
    labels = _crpa_display_labels_clockwise(element_positions_m)
    x, y, z = element_positions_m[:, 0], element_positions_m[:, 1], element_positions_m[:, 2]

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax1.scatter(x, y, s=180, c="red", marker="o", edgecolors="black", linewidth=2, zorder=5)
    for lab, xi, yi in zip(labels, x, y):
        ax1.annotate(lab, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=10, fontweight="bold")
    ax1.set_xlabel("X (m)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Y (m)", fontsize=11, fontweight="bold")
    ax1.set_title("Vista XY (desde arriba)", fontsize=12, fontweight="bold")
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(x, y, z, s=180, c="red", marker="o", edgecolors="black", linewidth=2, zorder=5)
    for lab, xi, yi, zi in zip(labels, x, y, z):
        ax2.text(xi, yi, zi, lab, fontsize=10, fontweight="bold")
    ax2.set_xlabel("X (m)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Y (m)", fontsize=10, fontweight="bold")
    ax2.set_zlabel("Z (m)", fontsize=10, fontweight="bold")
    ax2.set_title("Vista 3D", fontsize=12, fontweight="bold")

    fig.suptitle("Geometría del Array CRPA de 7 Elementos", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pattern_comparison_azimuth(
    conventional_pattern: pd.DataFrame,
    adaptive_pattern: pd.DataFrame,
    output_path: Path,
    title: str,
    jammer_info: list[tuple[str, float]] | None = None,
    adaptive_label: str = "Algoritmo",
    fixed_elevation_deg: float | None = None,
) -> None:
    fig = plt.figure(figsize=(14, 6))
    min_display_dB = -50.0
    radial_ticks = [0, 10, 20, 30, 40, 50]

    for idx, (data, label, color) in enumerate(
        [(conventional_pattern, "Convencional", "green"), (adaptive_pattern, adaptive_label, "blue")], start=1
    ):
        ax = fig.add_subplot(1, 2, idx, projection="polar")
        theta = np.deg2rad(data["azimuth_deg"].to_numpy())
        radius = _db_to_radius(data["response_dB_normalized"].to_numpy(), min_display_dB)
        ax.plot(theta, radius, linewidth=2, label=label, color=color)

        if idx == 2 and jammer_info:
            for jammer_name, az in jammer_info:
                label = f"{jammer_name} ({az:.1f}°)"
                rad = np.deg2rad(az)
                rr = np.linspace(0, abs(min_display_dB), 100)
                ax.plot(np.full_like(rr, rad), rr, "--", linewidth=2, label=label,)

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_rlim(0, abs(min_display_dB))
        ax.set_rticks(radial_ticks)
        ax.set_yticklabels([f"{t + min_display_dB:.0f} dB" for t in radial_ticks])
        ax.grid(True)
        ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.15), fontsize=8)
        ax.set_title("1. Patrón convencional a el = " f"{fixed_elevation_deg:.1f}°" if idx == 1 else "2. Patrón del algoritmo a el = " f"{fixed_elevation_deg:.1f}°", fontweight="bold", fontsize=12)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pattern_comparison_elevation(
    conventional_pattern: pd.DataFrame,
    adaptive_pattern: pd.DataFrame,
    output_path: Path,
    title: str,
    jammer_info: list[tuple[str, float]] | None = None,
    adaptive_label: str = "Algoritmo",
    fixed_azimuth_deg: float | None = None,
) -> None:
    """Corte vertical: izquierda 180°, arriba 90° cenit, derecha 0°."""
    fig = plt.figure(figsize=(14, 6))
    min_display_dB = -50.0
    radial_ticks = [0, 10, 20, 30, 40, 50]

    for idx, (data, label, color) in enumerate(
        [(conventional_pattern, "Convencional", "green"), (adaptive_pattern, adaptive_label, "blue")], start=1
    ):
        ax = fig.add_subplot(1, 2, idx, projection="polar")
        elevation = data["elevation_deg"].to_numpy()
        theta = np.deg2rad(90.0 - elevation)
        radius = _db_to_radius(data["response_dB_normalized"].to_numpy(), min_display_dB)
        ax.plot(theta, radius, linewidth=2, label=label, color=color)

        if idx == 2 and jammer_info:
            for jammer_name, el in jammer_info:
                rad = np.deg2rad(90.0 - el)
                rr = np.linspace(0, abs(min_display_dB), 100)
                label = f"{jammer_name} ({el:.1f}°)"
                ax.plot(np.full_like(rr, rad), rr, "--", linewidth=2, label=label,)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_xticks(np.deg2rad([-90, -60, -30, 0, 30, 60, 90]))
        ax.set_xticklabels(["180°", "150°", "120°", "90°", "60°", "30°", "0°"])
        ax.set_rlim(0, abs(min_display_dB))
        ax.set_rticks(radial_ticks)
        ax.set_yticklabels([f"{t + min_display_dB:.0f} dB" for t in radial_ticks])
        ax.grid(True)
        ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.15), fontsize=8)
        ax.set_title("1. Patrón convencional a az = " f"{fixed_azimuth_deg:.1f}°" if idx == 1 else "2. Patrón del algoritmo a az = " f"{fixed_azimuth_deg:.1f}°", fontweight="bold", fontsize=12)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_heatmap_comparison(conventional_grid: dict[str, np.ndarray], adaptive_grid: dict[str, np.ndarray], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, grid, label in zip(axes, [conventional_grid, adaptive_grid], ["Convencional", "Algoritmo"]):
        pcm = ax.pcolormesh(
            grid["azimuth_deg"],
            grid["elevation_deg"],
            grid["response_dB_normalized"],
            shading="auto",
            cmap="viridis",
            vmin=-60,
            vmax=0,
        )
        ax.set_xlabel("Azimut [deg]", fontsize=10, fontweight="bold")
        ax.set_ylabel("Elevación [deg]", fontsize=10, fontweight="bold")
        ax.set_title(label, fontsize=12, fontweight="bold")
        fig.colorbar(pcm, ax=ax, label="Respuesta [dB]")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_3d_comparison(conventional_grid: dict[str, np.ndarray], adaptive_grid: dict[str, np.ndarray], output_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(16, 7))
    for idx, (grid, label) in enumerate([(conventional_grid, "Convencional"), (adaptive_grid, "Algoritmo")], start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        surf = ax.plot_surface(
            grid["azimuth_deg"],
            grid["elevation_deg"],
            grid["response_dB_normalized"],
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            rcount=100,
            ccount=100,
        )
        ax.set_xlabel("Azimut [deg]", fontsize=9, fontweight="bold")
        ax.set_ylabel("Elevación [deg]", fontsize=9, fontweight="bold")
        ax.set_zlabel("Respuesta [dB]", fontsize=9, fontweight="bold")
        ax.set_zlim(-60, 0)
        ax.set_title(label, fontsize=12, fontweight="bold")
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_temporal_spectrum(spectrum_table: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(spectrum_table["frequency_hz"], spectrum_table["power_dB_normalized"], linewidth=1.5)
    ax.set_xlabel("Frecuencia [Hz]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Potencia normalizada [dB]", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
