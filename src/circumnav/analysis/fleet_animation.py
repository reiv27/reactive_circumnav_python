"""Animated playback and a static summary figure for a vehicle fleet.

Built, like :mod:`circumnav.analysis.animation`, only from finished results
and controller logs. Requires matplotlib (the ``viz`` extra).

One vehicle is the *focus*: it is drawn red, all others light blue, and the
side panels show that vehicle only. From its point of view the map links it to
each neighbour: a green link where it sees the neighbour, a dotted orange link
where the neighbour is in range but hidden by the obstacle body. The middle
panel shows the distance to the nearest neighbour it sees (red, gaps where it
sees nobody) over the true nearest distance (dashed). The bottom
panel is a timeline per neighbour: light blue = visible, orange = hidden by the
body, grey = out of range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Polygon
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
    nearest_visible_neighbour_distance,
    neighbour_visibility,
    obstacle_distance,
    pairwise_distance,
)
from circumnav.controllers.reactive import ReactiveCircumnavController
from circumnav.models.obstacles import Obstacle
from circumnav.sensors.neighbours import NeighbourSensor
from circumnav.simulation.result import FleetResult

FOCUS_COLOR = "tab:red"
OTHER_COLOR = "lightskyblue"
_HIDDEN_COLOR = "tab:orange"
_OUT_OF_RANGE_COLOR = "0.85"

_OUT_OF_RANGE, _HIDDEN, _VISIBLE = 0, 1, 2


def _default_sensor(
    controllers: Sequence[ReactiveCircumnavController],
) -> NeighbourSensor:
    config = controllers[0].config
    return NeighbourSensor(config.sensor_range, config.check_occlusion)


def _check_inputs(
    fleet: FleetResult,
    controllers: Sequence[ReactiveCircumnavController],
    focus: int,
) -> None:
    if len(controllers) != fleet.robot_count or not all(
        controller.log for controller in controllers
    ):
        raise ValueError("pass one run controller per vehicle, with non-empty logs")
    if not 0 <= focus < fleet.robot_count:
        raise ValueError(f"focus must be in [0, {fleet.robot_count - 1}]")


def _line_of_sight_states(
    visibility: NeighbourVisibility, focus: int
) -> tuple[list[int], NDArray[np.int_]]:
    """Per-neighbour state codes of the focus vehicle, ``(neighbours, samples)``."""

    others = [index for index in range(visibility.visible.shape[1]) if index != focus]
    states = np.full((len(others), visibility.visible.shape[0]), _OUT_OF_RANGE)
    for row, other in enumerate(others):
        states[row, visibility.in_range[:, focus, other]] = _HIDDEN
        states[row, visibility.visible[:, focus, other]] = _VISIBLE
    return others, states


def _states_to_rgba(states: NDArray[np.int_]) -> NDArray[np.float64]:
    palette = np.array(
        [
            to_rgba(_OUT_OF_RANGE_COLOR),
            to_rgba(_HIDDEN_COLOR),
            to_rgba(OTHER_COLOR),
        ]
    )
    return palette[states]


def _focus_segments(
    positions: NDArray[np.float64],
    mask: NDArray[np.bool_],
    focus: int,
    step: int,
) -> list[NDArray[np.float64]]:
    return [
        np.stack([positions[focus, step], positions[other, step]])
        for other in range(positions.shape[0])
        if other != focus and mask[step, focus, other]
    ]


def _setup_line_of_sight_axis(
    ax: plt.Axes, time: NDArray[np.float64], others: Sequence[int]
) -> None:
    ax.set_title(r"line of sight from the red vehicle", fontsize=10)
    ax.set_xlabel(r"$t$ [s]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(float(time[0]), float(time[-1]))
    ax.set_yticks([row + 0.5 for row in range(len(others))])
    ax.set_yticklabels([rf"$\#{other}$" for other in others], fontsize=8)
    ax.legend(
        handles=[
            Patch(color=OTHER_COLOR, label="visible"),
            Patch(color=_HIDDEN_COLOR, label="hidden by body"),
            Patch(color=_OUT_OF_RANGE_COLOR, label="out of range"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3,
        fontsize=7, frameon=False,
    )


def _draw_vehicles_static(
    ax: plt.Axes, fleet: FleetResult, focus: int, size: float
) -> None:
    order = [i for i in range(fleet.robot_count) if i != focus] + [focus]
    for index in order:
        is_focus = index == focus
        color = FOCUS_COLOR if is_focus else OTHER_COLOR
        result = fleet.results[index]
        ax.plot(
            result.x, result.y, color=color, zorder=3 if is_focus else 2,
            linewidth=1.4 if is_focus else 0.9, alpha=1.0 if is_focus else 0.8,
        )
        ax.add_patch(
            Polygon(
                _triangle_vertices(result.x[0], result.y[0], result.heading[0], size),
                closed=True, facecolor=color, edgecolor="black",
                linewidth=0.8, zorder=5 if is_focus else 4,
            )
        )


def _new_figure() -> tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes, plt.Axes]:
    fig = plt.figure(figsize=(12.0, 7.4))
    grid = GridSpec(3, 2, width_ratios=(2.1, 1.0), figure=fig)
    return (
        fig,
        fig.add_subplot(grid[:, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 1]),
    )


def build_fleet_animation(
    fleet: FleetResult,
    controllers: Sequence[ReactiveCircumnavController],
    obstacles: Sequence[Obstacle],
    settings: AnimationSettings = AnimationSettings(),
    neighbour_sensor: NeighbourSensor | None = None,
    visibility: NeighbourVisibility | None = None,
    focus: int = 0,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """Playback: the whole fleet on the map, the ``focus`` vehicle in the panels."""

    _check_inputs(fleet, controllers, focus)
    config = controllers[0].config
    sensor = neighbour_sensor or _default_sensor(controllers)
    if visibility is None:
        visibility = neighbour_visibility(fleet, obstacles, sensor)

    focus_log = controllers[focus].log
    log_times = np.array([entry.time for entry in focus_log])
    deviation = np.array([entry.equidistant_deviation for entry in focus_log])
    neighbour = nearest_neighbour_distance(fleet)[focus]
    seen_neighbour = nearest_visible_neighbour_distance(fleet, visibility)[focus]
    others, states = _line_of_sight_states(visibility, focus)
    rgba = _states_to_rgba(states)
    time = fleet.time
    positions = fleet.positions
    headings = np.stack([result.heading for result in fleet.results])

    duration = float(time[-1])
    frame_count = min(
        settings.max_frames,
        max(2, int(duration * settings.fps / settings.real_time_factor)),
    )
    frame_indices = np.unique(np.linspace(0, len(time) - 1, frame_count).astype(int))

    fig, ax_main, ax_dr, ax_gap, ax_los = _new_figure()
    _setup_main_axis(ax_main, fleet.results[0], obstacles, config.safety_distance)
    robot_size = settings.robot_size_in_turning_radii * config.turning_radius

    seen_links = LineCollection([], colors="tab:green", linewidths=1.3, zorder=3)
    hidden_links = LineCollection(
        [], colors=_HIDDEN_COLOR, linewidths=1.2, linestyles=":", zorder=3
    )
    ax_main.add_collection(seen_links)
    ax_main.add_collection(hidden_links)
    ax_main.plot([], [], color="tab:green", label="line of sight")
    ax_main.plot([], [], color=_HIDDEN_COLOR, linestyle=":", label="hidden by obstacle")

    draw_order = [i for i in range(fleet.robot_count) if i != focus] + [focus]
    trails: dict[int, plt.Line2D] = {}
    triangles: dict[int, Polygon] = {}
    for index in draw_order:
        is_focus = index == focus
        color = FOCUS_COLOR if is_focus else OTHER_COLOR
        (trail,) = ax_main.plot(
            [], [], color=color, linewidth=1.5 if is_focus else 0.9,
            alpha=1.0 if is_focus else 0.8, zorder=3 if is_focus else 2,
        )
        triangle = Polygon(
            _triangle_vertices(0.0, 0.0, 0.0, 1.0),
            closed=True, facecolor=color, edgecolor="black",
            linewidth=0.8, zorder=5 if is_focus else 4,
        )
        ax_main.add_patch(triangle)
        trails[index] = trail
        triangles[index] = triangle
    ax_main.legend(loc="upper right", fontsize=8)
    hud_text = ax_main.text(
        0.02, 0.98, "", transform=ax_main.transAxes, va="top", ha="left",
        fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    _setup_series_axis(
        ax_dr, r"deviation from equidistant $d_R(t)$ [m]", log_times, deviation
    )
    (dr_line,) = ax_dr.plot([], [], color=FOCUS_COLOR, linewidth=1.2)
    finite_gap = np.concatenate([neighbour, seen_neighbour])
    finite_gap = finite_gap[np.isfinite(finite_gap)]
    _setup_series_axis(
        ax_gap, r"nearest neighbour [m]: visible (red), true (dashed)", time,
        finite_gap if finite_gap.size else np.zeros(1),
    )
    (gap_line,) = ax_gap.plot(
        [], [], color="black", linestyle="--", linewidth=0.9
    )
    (seen_line,) = ax_gap.plot([], [], color=FOCUS_COLOR, linewidth=1.4)

    image = ax_los.imshow(
        np.zeros_like(rgba), aspect="auto", interpolation="nearest",
        extent=[float(time[0]), float(time[-1]), len(others), 0],
    )
    _setup_line_of_sight_axis(ax_los, time, others)

    def artists():
        return (
            *trails.values(), *triangles.values(), dr_line, gap_line, seen_line,
            seen_links, hidden_links, image, hud_text,
        )

    def init():
        for line in (*trails.values(), dr_line, gap_line, seen_line):
            line.set_data([], [])
        seen_links.set_segments([])
        hidden_links.set_segments([])
        image.set_data(np.zeros_like(rgba))
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
        seen = log_times <= now
        dr_line.set_data(log_times[seen], deviation[seen])
        gap_line.set_data(time[: step + 1], neighbour[: step + 1])
        seen_line.set_data(time[: step + 1], seen_neighbour[: step + 1])
        revealed = rgba.copy()
        revealed[:, step + 1 :, 3] = 0.0
        image.set_data(revealed)
        seen_links.set_segments(
            _focus_segments(positions, visibility.visible, focus, step)
        )
        hidden_links.set_segments(
            _focus_segments(positions, visibility.occluded, focus, step)
        )
        hud_text.set_text(
            rf"$t = {now:.1f}$ s,  {fleet.robot_count} vehicles,  "
            rf"neighbour range ${sensor.max_range:.0f}$ m,  red = vehicle $\#{focus}$"
        )
        return artists()

    fig.tight_layout()
    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), init_func=init,
        interval=1000.0 / settings.fps, blit=False,
    )
    return fig, anim


def save_fleet_figure(
    fleet: FleetResult,
    controllers: Sequence[ReactiveCircumnavController],
    obstacles: Sequence[Obstacle],
    path: str | Path,
    neighbour_sensor: NeighbourSensor | None = None,
    visibility: NeighbourVisibility | None = None,
    focus: int = 0,
) -> None:
    """Static summary: the fleet's paths, the ``focus`` vehicle's panels."""

    _check_inputs(fleet, controllers, focus)
    config = controllers[0].config
    if visibility is None:
        visibility = neighbour_visibility(
            fleet, obstacles, neighbour_sensor or _default_sensor(controllers)
        )
    focus_log = controllers[focus].log
    log_times = np.array([entry.time for entry in focus_log])
    deviation = np.array([entry.equidistant_deviation for entry in focus_log])
    neighbour = nearest_neighbour_distance(fleet)[focus]
    others, states = _line_of_sight_states(visibility, focus)

    fig, ax_main, ax_dr, ax_gap, ax_los = _new_figure()
    _setup_main_axis(ax_main, fleet.results[0], obstacles, config.safety_distance)
    size = AnimationSettings().robot_size_in_turning_radii * config.turning_radius
    _draw_vehicles_static(ax_main, fleet, focus, size)

    _setup_series_axis(
        ax_dr, r"deviation from equidistant $d_R(t)$ [m]", log_times, deviation
    )
    ax_dr.plot(log_times, deviation, color=FOCUS_COLOR, linewidth=0.9)
    ax_gap.plot(fleet.time, neighbour, color="black", linestyle="--", linewidth=0.9)
    ax_gap.plot(
        fleet.time, nearest_visible_neighbour_distance(fleet, visibility)[focus],
        color=FOCUS_COLOR, linewidth=1.4,
    )
    ax_gap.set_title(r"nearest neighbour [m]: visible (red), true (dashed)", fontsize=10)
    ax_gap.set_xlabel(r"$t$ [s]", fontsize=8)
    ax_gap.tick_params(labelsize=7)
    ax_gap.grid(True, alpha=0.3)
    ax_los.imshow(
        _states_to_rgba(states), aspect="auto", interpolation="nearest",
        extent=[float(fleet.time[0]), float(fleet.time[-1]), len(others), 0],
    )
    _setup_line_of_sight_axis(ax_los, fleet.time, others)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_fleet_distance_figure(
    fleet: FleetResult,
    obstacles: Sequence[Obstacle],
    path: str | Path,
    safety_distance: float,
    visibility: NeighbourVisibility,
    focus: int = 0,
) -> None:
    """Distances over time: from ``focus`` to each neighbour; the nearest
    neighbour ``focus`` actually sees against the true nearest; nearest
    neighbour of every vehicle with the closest pair; true obstacle clearance
    against ``ρ₀``.

    With evenly spaced vehicles the nearest-neighbour curves coincide up to a
    time shift, so the top panel splits the focus vehicle's view per neighbour.
    """

    if fleet.robot_count < 2:
        raise ValueError(f"need at least 2 vehicles, got {fleet.robot_count}")
    if not 0 <= focus < fleet.robot_count:
        raise ValueError(f"focus must be in [0, {fleet.robot_count - 1}]")
    time = fleet.time
    pairs = pairwise_distance(fleet)
    nearest = pairs.min(axis=1)
    closest_pair = nearest.min(axis=0)
    seen = nearest_visible_neighbour_distance(fleet, visibility)[focus]
    clearance = np.stack([obstacle_distance(r, obstacles) for r in fleet.results])
    others = [index for index in range(fleet.robot_count) if index != focus]
    order = others + [focus]

    fig, (ax_pairs, ax_seen, ax_nearest, ax_obstacle) = plt.subplots(
        4, 1, figsize=(10.0, 11.2), sharex=True
    )
    palette = plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(others)))
    for color, other in zip(palette, others):
        ax_pairs.plot(
            time, pairs[focus, other], color=color, linewidth=0.9,
            label=rf"to $\#{other}$",
        )
    ax_pairs.plot(
        time, nearest[focus], color=FOCUS_COLOR, linewidth=2.0, alpha=0.7,
        label="nearest",
    )
    ax_pairs.set_title(rf"distance from vehicle $\#{focus}$ to each neighbour [m]",
                       fontsize=10)
    ax_pairs.legend(loc="upper right", fontsize=7, ncol=min(len(others) + 1, 6))

    blind = np.isnan(seen)
    ax_seen.fill_between(
        time, 0.0, 1.0, where=blind, transform=ax_seen.get_xaxis_transform(),
        color=_OUT_OF_RANGE_COLOR, alpha=0.6, linewidth=0.0, label="sees nobody",
    )
    ax_seen.plot(
        time, nearest[focus], color="black", linestyle="--", linewidth=0.9,
        label="true nearest",
    )
    ax_seen.plot(
        time, seen, color=FOCUS_COLOR, linewidth=1.4, label="nearest visible",
    )
    ax_seen.set_title(
        rf"nearest neighbour seen by vehicle $\#{focus}$ [m];  "
        rf"sees nobody {100.0 * blind.mean():.0f} % of the run",
        fontsize=10,
    )
    ax_seen.legend(loc="upper right", fontsize=7, ncol=3)

    for index in order:
        is_focus = index == focus
        ax_nearest.plot(
            time, nearest[index], color=FOCUS_COLOR if is_focus else OTHER_COLOR,
            linewidth=1.2 if is_focus else 0.9, zorder=3 if is_focus else 2,
        )
    ax_nearest.plot(
        time, closest_pair, color="black", linestyle="--", linewidth=1.0,
        label="closest pair", zorder=4,
    )
    ax_nearest.plot([], [], color=FOCUS_COLOR, label=rf"vehicle $\#{focus}$")
    ax_nearest.plot([], [], color=OTHER_COLOR, label="other vehicles")
    ax_nearest.set_title(
        r"distance to nearest neighbour, every vehicle [m];  closest pair: "
        rf"min ${closest_pair.min():.2f}$, start ${closest_pair[0]:.2f}$, "
        rf"end ${closest_pair[-1]:.2f}$",
        fontsize=10,
    )
    ax_nearest.legend(loc="upper right", fontsize=7, ncol=3)

    for index in order:
        is_focus = index == focus
        ax_obstacle.plot(
            time, clearance[index], color=FOCUS_COLOR if is_focus else OTHER_COLOR,
            linewidth=1.2 if is_focus else 0.9, zorder=3 if is_focus else 2,
        )
    ax_obstacle.axhline(
        safety_distance, color="tab:blue", linestyle="--", linewidth=1.0,
        label=rf"$\rho_0 = {safety_distance:.2f}$ m",
    )
    ax_obstacle.set_title(r"true distance to the obstacle surface $\rho(t)$ [m]",
                          fontsize=10)
    ax_obstacle.legend(loc="upper right", fontsize=7)
    ax_obstacle.set_xlabel(r"$t$ [s]", fontsize=8)

    for ax in (ax_pairs, ax_seen, ax_nearest):
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + 0.2 * (top - bottom))
    for ax in (ax_pairs, ax_seen, ax_nearest, ax_obstacle):
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(float(time[0]), float(time[-1]))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
