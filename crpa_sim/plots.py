"""plots.py
Figuras de geometria, cortes, mapas 2D y superficies 3D.

Se mantiene un estilo similar a la version previa:
- titulos en negrita,
- etiquetas claras,
- patron adaptativo,
- lineas de jammers en los cortes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _db_to_radius(response_dB: np.ndarray, min_display_dB: float) -> np.ndarray:
    """Convierte una respuesta en dB al radio usado por los plots polares.

    Parametros:
        response_dB: Vector de respuesta normalizada en dB.
        min_display_dB: Valor minimo visible; las respuestas por debajo se
            recortan a ese suelo antes de convertir a radio.
    """
    return np.maximum(response_dB, min_display_dB) - min_display_dB


def _crpa_display_labels_clockwise(element_positions_m: np.ndarray) -> list[str]:
    """Genera etiquetas visuales de elementos: centro=1 y resto horario.

    Parametros:
        element_positions_m: Matriz (N, 3) con posiciones XYZ del array.
    """
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
    """Dibuja la geometria del array en vista XY y vista 3D.

    Parametros:
        element_positions_m: Matriz (N, 3) con posiciones XYZ del array.
        output_path: Ruta PNG donde se guarda la figura.
    """
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

    fig.suptitle("Geometria del Array CRPA de 7 Elementos", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pattern_azimuth(
    adaptive_pattern: pd.DataFrame,
    output_path: Path,
    title: str,
    jammer_info: list[tuple[str, float] | tuple[str, float, int]] | None = None,
    adaptive_label: str = "Algoritmo",
    fixed_elevation_deg: float | None = None,
) -> None:
    """Dibuja un corte polar de patron en azimut.

    Parametros:
        adaptive_pattern: Tabla con columnas "azimuth_deg" y
            "response_dB_normalized".
        output_path: Ruta PNG donde se guarda la figura.
        title: Titulo superior de la figura.
        jammer_info: Lista opcional de tuplas (nombre, azimut_deg) o
            (nombre, azimut_deg, color_idx) para marcar jammers.
        adaptive_label: Etiqueta de la curva principal en la leyenda.
        fixed_elevation_deg: Elevacion fija del corte, mostrada en el titulo.
    """
    fig = plt.figure(figsize=(7, 7))
    min_display_dB = -50.0
    radial_ticks = [0, 10, 20, 30, 40, 50]
    jammer_colors = ["red", "orange", "magenta", "purple", "lime", "yellow"]

    ax = fig.add_subplot(1, 1, 1, projection="polar")
    theta = np.deg2rad(adaptive_pattern["azimuth_deg"].to_numpy())
    radius = _db_to_radius(adaptive_pattern["response_dB_normalized"].to_numpy(), min_display_dB)
    ax.plot(theta, radius, linewidth=2, label=adaptive_label, color="blue")

    if jammer_info:
        for j_idx, item in enumerate(jammer_info):
            jammer_name, az = item[:2]
            color_idx = item[2] if len(item) > 2 else j_idx
            label = f"{jammer_name} ({az:.1f} deg)"
            rad = np.deg2rad(az)
            rr = np.linspace(0, abs(min_display_dB), 100)
            color = jammer_colors[color_idx % len(jammer_colors)]
            ax.plot(np.full_like(rr, rad), rr, "--", linewidth=2, label=label, color=color)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(0, abs(min_display_dB))
    ax.set_rticks(radial_ticks)
    ax.set_yticklabels([f"{t + min_display_dB:.0f} dB" for t in radial_ticks])
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.15), fontsize=8)

    el_text = f"{fixed_elevation_deg:.1f} deg" if fixed_elevation_deg is not None else "?"
    ax.set_title(f"Patron {adaptive_label} a el = {el_text}", fontweight="bold", fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pattern_elevation(
    adaptive_pattern: pd.DataFrame,
    output_path: Path,
    title: str,
    jammer_info: list[tuple[str, float] | tuple[str, float, int]] | None = None,
    adaptive_label: str = "Algoritmo",
    fixed_azimuth_deg: float | None = None,
) -> None:
    """Dibuja un corte polar de patron en elevacion.

    Parametros:
        adaptive_pattern: Tabla con columnas "elevation_deg" y
            "response_dB_normalized".
        output_path: Ruta PNG donde se guarda la figura.
        title: Titulo superior de la figura.
        jammer_info: Lista opcional de tuplas (nombre, elevation_deg) o
            (nombre, elevation_deg, color_idx) para marcar jammers.
        adaptive_label: Etiqueta de la curva principal en la leyenda.
        fixed_azimuth_deg: Azimut fijo del corte, mostrado en el titulo.
    """
    fig = plt.figure(figsize=(7, 7))
    min_display_dB = -50.0
    radial_ticks = [0, 10, 20, 30, 40, 50]
    jammer_colors = ["red", "orange", "magenta", "purple", "lime", "yellow"]

    ax = fig.add_subplot(1, 1, 1, projection="polar")
    if "polar_theta_deg" in adaptive_pattern.columns:
        theta = np.deg2rad(adaptive_pattern["polar_theta_deg"].to_numpy())
    else:
        elevation = adaptive_pattern["elevation_deg"].to_numpy()
        theta = np.deg2rad(90.0 - elevation)
    radius = _db_to_radius(adaptive_pattern["response_dB_normalized"].to_numpy(), min_display_dB)
    if "plot_side" in adaptive_pattern.columns:
        first = True
        for _, side_data in adaptive_pattern.groupby("plot_side", sort=False):
            theta_side = np.deg2rad(side_data["polar_theta_deg"].to_numpy())
            radius_side = _db_to_radius(
                side_data["response_dB_normalized"].to_numpy(),
                min_display_dB,
            )

            ax.plot(
                theta_side,
                radius_side,
                linewidth=2,
                label=adaptive_label if first else None,
                color="blue",
            )
            first = False
    else:
        ax.plot(theta, radius, linewidth=2, label=adaptive_label, color="blue")

    if jammer_info:
        for j_idx, item in enumerate(jammer_info):
            jammer_name, el = item[:2]
            color_idx = item[2] if len(item) > 2 else j_idx
            label = f"{jammer_name} ({el:.1f} deg)"
            rad = np.deg2rad(90.0 - el)
            rr = np.linspace(0, abs(min_display_dB), 100)
            color = jammer_colors[color_idx % len(jammer_colors)]
            ax.plot(np.full_like(rr, rad), rr, "--", linewidth=2, label=label, color=color)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_xticks(np.deg2rad([-90, -60, -30, 0, 30, 60, 90]))
    ax.set_xticklabels(["0 deg", "30 deg", "60 deg", "90 deg", "60 deg", "30 deg", "0 deg"])
    ax.set_rlim(0, abs(min_display_dB))
    ax.set_rticks(radial_ticks)
    ax.set_yticklabels([f"{t + min_display_dB:.0f} dB" for t in radial_ticks])
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.15), fontsize=8)

    az_text = f"{fixed_azimuth_deg:.1f} deg" if fixed_azimuth_deg is not None else "?"
    ax.set_title(f"Patron {adaptive_label} a az = {az_text}", fontweight="bold", fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _select_grid_response_dB(adaptive_grid: dict[str, np.ndarray]) -> tuple[np.ndarray, str, bool]:
    """Selecciona la respuesta disponible para heatmap/3D.

    Si existe response_power_dB usa potencia sin normalizar. Si no, usa
    response_dB_normalized, que es la salida original normalizada.
    """
    if "response_power_dB" in adaptive_grid:
        return adaptive_grid["response_power_dB"], "Potencia sin normalizar [dB]", False
    return adaptive_grid["response_dB_normalized"], "Respuesta normalizada [dB]", True


def _nearest_grid_value(
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    response_grid: np.ndarray,
    az_deg: float,
    el_deg: float,
) -> float:
    """Devuelve la respuesta del punto de malla mas cercano a az/el."""
    dist = (az_grid - az_deg) ** 2 + (el_grid - el_deg) ** 2
    row_idx, col_idx = np.unravel_index(np.argmin(dist), dist.shape)
    return float(response_grid[row_idx, col_idx])


def plot_heatmap(
    adaptive_grid: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    adaptive_label: str = "Algoritmo",
    jammer_info: list[tuple[str, float, float] | tuple[str, float, float, int]] | None = None,
) -> None:
    """Dibuja un mapa 2D azimut/elevacion y marca los nulos de jammers.

    jammer_info acepta tuplas:
      - (nombre, azimut_deg, elevacion_deg)
      - (nombre, azimut_deg, elevacion_deg, color_idx)
    """
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    response_dB, colorbar_label, is_normalized = _select_grid_response_dB(adaptive_grid)

    pcolor_kwargs = dict(
        shading="auto",
        cmap="viridis",
    )
    if is_normalized:
        pcolor_kwargs.update(vmin=-60, vmax=0)

    pcm = ax.pcolormesh(
        adaptive_grid["azimuth_deg"],
        adaptive_grid["elevation_deg"],
        response_dB,
        **pcolor_kwargs,
    )

    jammer_colors = ["red", "orange", "magenta", "purple", "lime", "yellow"]
    if jammer_info:
        for j_idx, item in enumerate(jammer_info):
            jammer_name, az, el = item[:3]
            color_idx = item[3] if len(item) > 3 else j_idx
            color = jammer_colors[color_idx % len(jammer_colors)]
            z_value = _nearest_grid_value(
                adaptive_grid["azimuth_deg"],
                adaptive_grid["elevation_deg"],
                response_dB,
                float(az),
                float(el),
            )
            ax.scatter(
                az,
                el,
                s=95,
                marker="x",
                color=color,
                linewidths=2.5,
                label=f"Nulo {jammer_name} ({z_value:.1f} dB)",
            )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Azimut [deg]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Elevacion [deg]", fontsize=10, fontweight="bold")
    ax.set_title(adaptive_label, fontsize=12, fontweight="bold")
    fig.colorbar(pcm, ax=ax, label=colorbar_label)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_3d(
    adaptive_grid: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    adaptive_label: str = "Algoritmo",
    jammer_info: list[tuple[str, float, float] | tuple[str, float, float, int]] | None = None,
) -> None:
    """Dibuja una superficie 3D y marca con X los nulos de jammers."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    response_dB, colorbar_label, is_normalized = _select_grid_response_dB(adaptive_grid)

    surf = ax.plot_surface(
        adaptive_grid["azimuth_deg"],
        adaptive_grid["elevation_deg"],
        response_dB,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rcount=100,
        ccount=100,
    )

    jammer_colors = ["red", "orange", "magenta", "purple", "lime", "yellow"]
    if jammer_info:
        for j_idx, item in enumerate(jammer_info):
            jammer_name, az, el = item[:3]
            color_idx = item[3] if len(item) > 3 else j_idx
            color = jammer_colors[color_idx % len(jammer_colors)]
            z_value = _nearest_grid_value(
                adaptive_grid["azimuth_deg"],
                adaptive_grid["elevation_deg"],
                response_dB,
                float(az),
                float(el),
            )
            ax.scatter(
                az,
                el,
                z_value,
                s=95,
                marker="x",
                color=color,
                linewidths=3.0,
                depthshade=False,
                label=f"Nulo {jammer_name} ({z_value:.1f} dB)",
            )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Azimut [deg]", fontsize=9, fontweight="bold")
    ax.set_ylabel("Elevacion [deg]", fontsize=9, fontweight="bold")
    ax.set_zlabel(colorbar_label, fontsize=9, fontweight="bold")
    if is_normalized:
        ax.set_zlim(-60, 0)
    ax.set_title(adaptive_label, fontsize=12, fontweight="bold")
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label=colorbar_label)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_temporal_spectrum(spectrum_table: pd.DataFrame, output_path: Path, title: str) -> None:
    """Dibuja el espectro temporal medio de snapshots.

    Parametros:
        spectrum_table: Tabla con columnas "frequency_hz" y
            "power_dB_normalized".
        output_path: Ruta PNG donde se guarda la figura.
        title: Titulo del grafico.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(spectrum_table["frequency_hz"], spectrum_table["power_dB_normalized"], linewidth=1.5)
    ax.set_xlabel("Frecuencia [Hz]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Potencia normalizada [dB]", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
