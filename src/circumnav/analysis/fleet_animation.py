"""Animated playback and a static summary figure for a vehicle fleet.

Built, like :mod:`circumnav.analysis.animation`, only from finished results
and controller logs. Requires matplotlib (the ``viz`` extra).

Besides the trajectories the figures show who sees whom: a green link joins
two vehicles that see each other, a dotted red link joins two that are in
range but hidden by the obstacle body.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import numpy as np
from numpy.typing import NDArray

from circumnav.analysis.animation import (
    AnimationSettings,
    _setup_main_axis,
    _setup_series_axis,
    _triangle_vertices,
)
from circumnav.analysis.metrics import (
    NeighbourVisibility,
    nearest_neighbour_distance,
    neighbour_visibility,
)
from circumnav.controllers.reactive import ReactiveCircumnavController
from circumnav.models.obstacles import Obstacle
from circumnav.sensors.neighbours import NeighbourSensor
from circumnav.simulation.result import FleetResult


def _colors(count: int) -> list[tuple[float, float, float, float]]:
    colormap = plt.get_cmap("tab10" if count <= 10 else "turbo")
    if count <= 10:
        return [colormap(index) for index in range(count)]
    return [colormap(index / (count - 1)) for index in range(count)]


def _deviation_traces(
    controllers: Sequence[ReactiveCircumnavController],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    times = [
        np.array([entry.time for entry in controller.log], dtype=np.float64)
        for controller in controllers
    ]
    deviations = [
        np.array(
            [entry.equidistant_deviation for entry in controller.log],
            dtype=np.float64,
        )
        for controller in controllers
    ]
    return times, deviations


def _default_sensor(
    controllers: Sequence[ReactiveCircumnavController],
) -> NeighbourSensor:
    config = controllers[0].config
    return NeighbourSensor(config.sensor_range, config.check_occlusion)


def _pair_segments(
    positions: NDArray[np.float64], mask: NDArray[np.bool_], step: int
) -> list[NDArray[np.float64]]:
    count = positions.shape[0]
    return [
        np.stack([positions[first, step], positions[second, step]])
        for first in range(count)
        for second in range(first + 1, count)
        if mask[step, first, second]
    ]


def _setup_count_axis(
    ax: plt.Axes, time: NDArray[np.float64], robots: int
) -> None:
    ax.set_title(r"neighbours in line of sight", fontsize=10)
    ax.set_xlabel(r"$t$ [s]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(float(time[0]), float(time[-1]))
    ax.set_ylim(-0.3, max(1, robots - 1) + 0.3)
    ax.set_yticks(range(max(1, robots - 1) + 1))
    ax.grid(True, alpha=0.3)


def build_fleet_animation(
    fleet: FleetResult,
    controllers: Sequence[ReactiveCircumnavController],
    obstacles: Sequence[Obstacle],
    settings: AnimationSettings = AnimationSettings(),
    neighbour_sensor: NeighbourSensor | None = None,
    visibility: NeighbourVisibility | None = None,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """Playback: map with line-of-sight links, ``d_R(t)``, neighbour distance and count."""

    if len(controllers) != fleet.robot_count or not all(
        controller.log for controller in controllers
    ):
        raise ValueError(
            "pass one run controller per vehicle, with non-empty logs"
        )

    config = controllers[0].config
    colors = _colors(fleet.robot_count)
    log_times, deviations = _deviation_traces(controllers)
    neighbour = nearest_neighbour_distance(fleet)
    if visibility is None:
        visibility = neighbour_visibility(
            fleet, obstacles, neighbour_sensor or _default_sensor(controllers)
        )
    counts = visibility.visible_count
    time = fleet.time
    positions = fleet.positions
    headings = np.stack([result.heading for result in fleet.results])

    duration = float(time[-1])
    frame_count = min(
        settings.max_frames,
        max(2, int(duration * settings.fps / settings.real_time_factor)),
    )
    frame_indices = np.unique(np.linspace(0, len(time) - 1, frame_count).astype(int))

    fig = plt.figure(figsize=(12.0, 7.0))
    grid = GridSpec(3, 2, width_ratios=(2.1, 1.0), figure=fig)
    ax_main = fig.add_subplot(grid[:, 0])
    ax_dr = fig.add_subplot(grid[0, 1])
    ax_gap = fig.add_subplot(grid[1, 1])
    ax_count = fig.add_subplot(grid[2, 1])

    _setup_main_axis(ax_main, fleet.results[0], obstacles, config.safety_distance)
    robot_size = settings.robot_size_in_turning_radii * config.turning_radius

    seen_links = LineCollection([], colors="tab:green", linewidths=1.2, zorder=3)
    hidden_links = LineCollection(
        [], colors="tab:red", linewidths=1.0, linestyles=":", alpha=0.8, zorder=3
    )
    ax_main.add_collection(seen_links)
    ax_main.add_collection(hidden_links)
    ax_main.plot([], [], color="tab:green", label="in line of sight")
    ax_main.plot([], [], color="tab:red", linestyle=":", label="hidden by obstacle")

    trails = []
    triangles = []
    for color in colors:
        (trail,) = ax_main.plot([], [], color=color, linewidth=1.3, zorder=2)
        triangle = Polygon(
            _triangle_vertices(0.0, 0.0, 0.0, 1.0),
            closed=True, facecolor=color, edgecolor="black",
            linewidth=0.8, zorder=4,
        )
        ax_main.add_patch(triangle)
        trails.append(trail)
        triangles.append(triangle)
    ax_main.legend(loc="upper right", fontsize=8)
    hud_text = ax_main.text(
        0.02, 0.98, "", transform=ax_main.transAxes, va="top", ha="left",
        fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    _setup_series_axis(
        ax_dr, r"deviation from equidistant $d_R(t)$ [m]",
        log_times[0], np.concatenate(deviations),
    )
    dr_lines = [
        ax_dr.plot([], [], color=color, linewidth=1.0)[0] for color in colors
    ]
    finite_gap = neighbour[np.isfinite(neighbour)]
    _setup_series_axis(
        ax_gap, r"distance to nearest neighbour [m]", time,
        finite_gap if finite_gap.size else np.zeros(1),
    )
    gap_lines = [
        ax_gap.plot([], [], color=color, linewidth=1.0)[0] for color in colors
    ]
    _setup_count_axis(ax_count, time, fleet.robot_count)
    count_lines = [
        ax_count.plot(
            [], [], color=color, linewidth=1.2, drawstyle="steps-post"
        )[0]
        for color in colors
    ]

    def artists():
        return (
            *trails, *triangles, *dr_lines, *gap_lines, *count_lines,
            seen_links, hidden_links, hud_text,
        )

    def init():
        for line in (*trails, *dr_lines, *gap_lines, *count_lines):
            line.set_data([], [])
        seen_links.set_segments([])
        hidden_links.set_segments([])
        hud_text.set_text("")
        return artists()

    def update(frame_number: int):
        step = int(frame_indices[min(frame_number, len(frame_indices) - 1)])
        now = float(time[step])
        for index in range(fleet.robot_count):
            trails[index].set_data(
                positions[index, : step + 1, 0], positions[index, : step + 1, 1]
            )
            triangles[index].set_xy(
                _triangle_vertices(
                    positions[index, step, 0], positions[index, step, 1],
                    headings[index, step], robot_size,
                )
            )
            seen = log_times[index] <= now
            dr_lines[index].set_data(
                log_times[index][seen], deviations[index][seen]
            )
            gap_lines[index].set_data(time[: step + 1], neighbour[index, : step + 1])
            count_lines[index].set_data(time[: step + 1], counts[: step + 1, index])
        seen_links.set_segments(_pair_segments(positions, visibility.visible, step))
        hidden_links.set_segments(_pair_segments(positions, visibility.occluded, step))
        hud_text.set_text(
            rf"$t = {now:.1f}$ s,  {fleet.robot_count} vehicles,  "
            rf"$R_s = {_sensor_range(neighbour_sensor, controllers):.0f}$ m"
        )
        return artists()

    fig.tight_layout()
    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), init_func=init,
        interval=1000.0 / settings.fps, blit=False,
    )
    return fig, anim


def _sensor_range(
    sensor: NeighbourSensor | None,
    controllers: Sequence[ReactiveCircumnavController],
) -> float:
    return (sensor or _default_sensor(controllers)).max_range


def save_fleet_figure(
    fleet: FleetResult,
    controllers: Sequence[ReactiveCircumnavController],
    obstacles: Sequence[Obstacle],
    path: str | Path,
    neighbour_sensor: NeighbourSensor | None = None,
    visibility: NeighbourVisibility | None = None,
) -> None:
    """Static summary: paths, ``d_R(t)``, neighbour distance, neighbours seen."""

    config = controllers[0].config
    colors = _colors(fleet.robot_count)
    log_times, deviations = _deviation_traces(controllers)
    neighbour = nearest_neighbour_distance(fleet)
    if visibility is None:
        visibility = neighbour_visibility(
            fleet, obstacles, neighbour_sensor or _default_sensor(controllers)
        )
    counts = visibility.visible_count

    fig = plt.figure(figsize=(12.0, 7.0))
    grid = GridSpec(3, 2, width_ratios=(2.1, 1.0), figure=fig)
    ax_main = fig.add_subplot(grid[:, 0])
    ax_dr = fig.add_subplot(grid[0, 1])
    ax_gap = fig.add_subplot(grid[1, 1])
    ax_count = fig.add_subplot(grid[2, 1])
    _setup_main_axis(ax_main, fleet.results[0], obstacles, config.safety_distance)

    size = AnimationSettings().robot_size_in_turning_radii * config.turning_radius
    for index, color in enumerate(colors):
        result = fleet.results[index]
        ax_main.plot(result.x, result.y, color=color, linewidth=1.1, zorder=2)
        ax_main.add_patch(
            Polygon(
                _triangle_vertices(result.x[0], result.y[0], result.heading[0], size),
                closed=True, facecolor=color, edgecolor="black",
                linewidth=0.8, zorder=4,
            )
        )
        ax_dr.plot(log_times[index], deviations[index], color=color, linewidth=0.8)
        ax_gap.plot(fleet.time, neighbour[index], color=color, linewidth=0.8)
        ax_count.step(
            fleet.time, counts[:, index], color=color, linewidth=1.0, where="post"
        )
    _setup_series_axis(
        ax_dr, r"deviation from equidistant $d_R(t)$ [m]",
        log_times[0], np.concatenate(deviations),
    )
    ax_gap.set_title(r"distance to nearest neighbour [m]", fontsize=10)
    ax_gap.set_xlabel(r"$t$ [s]", fontsize=8)
    ax_gap.tick_params(labelsize=7)
    ax_gap.grid(True, alpha=0.3)
    _setup_count_axis(ax_count, fleet.time, fleet.robot_count)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
