"""Animated playback of a reactive circumnavigation run.

The animation is built entirely from an already-completed
:class:`SimulationResult` and the controller's own diagnostic log
(``ReactiveCircumnavController.log``): it never drives the simulation and
never influences the dynamics, matching the "logging and plotting happen
after the fact" principle in README_CONTROL_SIMULATOR.md.

This module requires matplotlib, which is an optional dependency (the
``viz`` extra); importing it does not affect the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import numpy as np
from numpy.typing import NDArray

from circumnav.controllers.delayed import DelayedController
from circumnav.controllers.reactive import CircumnavMode, ReactiveCircumnavController
from circumnav.models.obstacles import Obstacle, minimum_equidistant_curvature
from circumnav.simulation.result import SimulationResult

_APPROACH_FACECOLOR = "#eaf3ff"
_ORBIT_FACECOLOR = "#fff0ea"


@dataclass(frozen=True)
class AnimationSettings:
    """Playback parameters; these only affect rendering, never the dynamics."""

    fps: int = 30
    real_time_factor: float = 1.0
    max_frames: int = 1200
    #: Nose-to-tail length of the vehicle triangle, in turning radii.
    robot_size_in_turning_radii: float = 0.55


def build_reactive_animation(
    result: SimulationResult,
    controller: ReactiveCircumnavController | DelayedController,
    obstacles: Sequence[Obstacle],
    settings: AnimationSettings = AnimationSettings(),
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """Build a playback animation with a live mode/dR/control-effort panel.

    ``controller`` must be the instance that actually produced ``result``
    (its ``log`` is read directly); a fresh, unrun controller has no log.
    """

    log = controller.log
    if not log:
        raise ValueError(
            "controller.log is empty; run the scenario before animating "
            "(the animation reads the controller instance used in that run)."
        )

    log_times = np.array([entry.time for entry in log], dtype=np.float64)
    # d_R(t): the regulated variable. In mode C it is the distance to the
    # obstacle's equidistant curve; in mode G, to that curve's local
    # approximation (the R_min circle about the frozen V).
    deviations = np.array(
        [entry.equidistant_deviation for entry in log], dtype=np.float64
    )
    yaw_rate_commands = np.array(
        [entry.yaw_rate_command for entry in log], dtype=np.float64
    )
    orbit_by_log = np.array(
        [entry.mode is CircumnavMode.ORBIT for entry in log], dtype=bool
    )

    # Hold the last controller update forward onto the finer plant grid, so
    # the path can be highlighted wherever the law was in gap mode.
    held_log_index = np.clip(
        np.searchsorted(log_times, result.time, side="right") - 1,
        0,
        len(log) - 1,
    )
    orbit_on_path = orbit_by_log[held_log_index]
    orbit_x = np.where(orbit_on_path, result.x, np.nan)
    orbit_y = np.where(orbit_on_path, result.y, np.nan)

    duration = float(result.time[-1]) if result.time.size else 0.0
    frame_count = min(
        settings.max_frames,
        max(2, int(duration * settings.fps / settings.real_time_factor)),
    )
    frame_indices = np.unique(
        np.linspace(0, len(result.time) - 1, frame_count).astype(int)
    )

    fig = plt.figure(figsize=(12.0, 6.6))
    grid = GridSpec(
        3, 2, width_ratios=(2.1, 1.0), height_ratios=(1.0, 1.0, 0.55),
        figure=fig,
    )
    ax_main = fig.add_subplot(grid[:, 0])
    ax_dr = fig.add_subplot(grid[0, 1])
    ax_u = fig.add_subplot(grid[1, 1])
    ax_mode = fig.add_subplot(grid[2, 1])

    safety_distance = controller.config.safety_distance
    _setup_main_axis(ax_main, result, obstacles, safety_distance)
    sensor_range = controller.config.sensor_range
    robot_size = settings.robot_size_in_turning_radii * controller.config.turning_radius
    # Grey, so that blue means "equidistant curve" and nothing else.
    sensor_circle = plt.Circle(
        (result.x[0], result.y[0]),
        sensor_range,
        fill=False,
        linestyle=":",
        edgecolor="0.55",
        linewidth=1.0,
        zorder=1,
        label=r"sensor range $R_s$",
    )
    ax_main.add_patch(sensor_circle)
    disk_circle = plt.Circle(
        (0.0, 0.0),
        controller.config.turning_radius,
        fill=False,
        linestyle=":",
        color="tab:green",
        alpha=0.6,
        visible=False,
        zorder=1,
    )
    ax_main.add_patch(disk_circle)

    (orbit_highlight,) = ax_main.plot(
        [], [], color="gold", linewidth=6, solid_capstyle="round",
        alpha=0.85, zorder=1.5, label=r"gap mode $G$",
    )
    (trail_line,) = ax_main.plot([], [], color="tab:red", linewidth=1.5, zorder=2)
    robot_triangle = Polygon(
        _triangle_vertices(0.0, 0.0, 0.0, 1.0),
        closed=True, facecolor="tab:red", edgecolor="black",
        linewidth=0.8, zorder=4,
    )
    ax_main.add_patch(robot_triangle)
    (primary_line,) = ax_main.plot(
        [], [], color="tab:orange", linewidth=1, linestyle=":", zorder=3
    )
    (primary_marker,) = ax_main.plot(
        [], [], marker="x", color="tab:orange", markersize=9,
        linestyle="None", zorder=5, label=r"$P_1$",
    )
    (secondary_marker,) = ax_main.plot(
        [], [], marker="x", color="tab:purple", markersize=9,
        linestyle="None", zorder=5, label=r"$P_2$",
    )
    (disk_marker,) = ax_main.plot(
        [], [], marker="D", color="tab:green", markersize=7,
        linestyle="None", zorder=5, label=r"$V$ (frozen in $G$)",
    )
    ax_main.legend(loc="upper right", fontsize=8)

    hud_text = ax_main.text(
        0.02, 0.98, "", transform=ax_main.transAxes, va="top", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    turning_radius = controller.config.turning_radius
    curvature = minimum_equidistant_curvature(obstacles, safety_distance)
    ax_main.text(
        0.02, 0.02,
        rf"$\rho_0 = {safety_distance:.2f}$ m$\;\geq\;"
        rf"R_{{\min}} = {turning_radius:.2f}$ m"
        rf"$\qquad\min\,\mathrm{{curv}} = {curvature:.2f}$ m"
        rf"$\qquad v = {controller.config.forward_speed:.2f}$ m/s"
        rf"$\qquad \omega = {controller.config.yaw_rate_magnitude:.2f}$ rad/s",
        transform=ax_main.transAxes, va="bottom", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
    )

    _setup_series_axis(
        ax_dr, r"deviation from equidistant $d_R(t)$ [m]", log_times, deviations
    )
    (dr_line,) = ax_dr.plot([], [], color="tab:blue", linewidth=1.5)
    (dr_marker,) = ax_dr.plot([], [], marker="o", color="tab:blue", markersize=4)

    _setup_series_axis(
        ax_u, r"control $u(t)$ [rad/s]", log_times, yaw_rate_commands
    )
    (u_line,) = ax_u.plot([], [], color="tab:blue", linewidth=1.5)
    (u_marker,) = ax_u.plot([], [], marker="o", color="tab:blue", markersize=4)

    ax_mode.set_title("controller mode", fontsize=9)
    ax_mode.set_xlabel(r"$t$ [s]", fontsize=8)
    ax_mode.tick_params(labelsize=7)
    ax_mode.set_ylim(-0.25, 1.25)
    ax_mode.set_yticks([0.0, 1.0])
    ax_mode.set_yticklabels([r"$C$", r"$G$"], fontsize=9)
    if log_times.size:
        ax_mode.set_xlim(float(log_times[0]), float(log_times[-1]))
    (mode_line,) = ax_mode.plot(
        [], [], color="tab:red", linewidth=1.5, drawstyle="steps-post"
    )

    artists = (
        orbit_highlight, trail_line, robot_triangle, primary_line,
        primary_marker, secondary_marker, disk_marker, disk_circle,
        sensor_circle, hud_text, dr_line, dr_marker,
        u_line, u_marker, mode_line,
    )

    def init():
        for line in (
            orbit_highlight, trail_line, primary_line, primary_marker,
            secondary_marker, disk_marker,
            dr_line, dr_marker, u_line, u_marker, mode_line,
        ):
            line.set_data([], [])
        disk_circle.set_visible(False)
        hud_text.set_text("")
        return artists

    def update(frame_number: int):
        index = int(frame_indices[frame_number])
        t = float(result.time[index])
        x, y, heading = (
            float(result.x[index]),
            float(result.y[index]),
            float(result.heading[index]),
        )

        trail_line.set_data(result.x[: index + 1], result.y[: index + 1])
        orbit_highlight.set_data(orbit_x[: index + 1], orbit_y[: index + 1])
        robot_triangle.set_xy(_triangle_vertices(x, y, heading, robot_size))
        sensor_circle.center = (x, y)

        log_index = max(int(np.searchsorted(log_times, t, side="right") - 1), 0)
        entry = log[log_index]

        _set_point(primary_marker, entry.primary_point)
        if entry.primary_point is not None:
            primary_line.set_data(
                [x, entry.primary_point[0]], [y, entry.primary_point[1]]
            )
        else:
            primary_line.set_data([], [])
        _set_point(secondary_marker, entry.secondary_point)

        # V is drawn only while mode G is active, at the centre frozen on entry.
        if entry.mode is CircumnavMode.ORBIT and entry.orbit_center is not None:
            _set_point(disk_marker, entry.orbit_center)
            disk_circle.center = (entry.orbit_center[0], entry.orbit_center[1])
            disk_circle.set_visible(True)
            face_color = _ORBIT_FACECOLOR
        else:
            _set_point(disk_marker, None)
            disk_circle.set_visible(False)
            face_color = _APPROACH_FACECOLOR
        for axis in (ax_dr, ax_u, ax_mode):
            axis.set_facecolor(face_color)

        dr_line.set_data(log_times[: log_index + 1], deviations[: log_index + 1])
        dr_marker.set_data([log_times[log_index]], [deviations[log_index]])
        u_line.set_data(
            log_times[: log_index + 1], yaw_rate_commands[: log_index + 1]
        )
        u_marker.set_data([log_times[log_index]], [yaw_rate_commands[log_index]])
        mode_line.set_data(
            log_times[: log_index + 1],
            orbit_by_log[: log_index + 1].astype(float),
        )

        if np.isfinite(entry.obstacle_range):
            rho_text = rf"$\rho = {entry.obstacle_range:.3f}$ m"
            deviation_text = rf"$d_R = {entry.equidistant_deviation:+.3f}$ m"
        else:
            rho_text = r"$\rho$: no obstacle in range"
            deviation_text = r"$d_R$: --"
        hud_text.set_text(
            rf"$t = {t:.2f}$ s" "\n"
            rf"mode $= {entry.mode.value}$" "\n"
            f"{rho_text}" "\n"
            f"{deviation_text}" "\n"
            rf"$u = {entry.yaw_rate_command:+.3f}$ rad/s"
        )

        return artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=1000.0 / settings.fps,
        blit=False,
    )
    fig.tight_layout()
    return fig, anim


def save_reactive_animation(
    fig: plt.Figure,
    anim: animation.FuncAnimation,
    path: str | Path,
    fps: int = 30,
    writer: str | None = None,
) -> None:
    """Save an animation built by :func:`build_reactive_animation` and close it."""

    path = Path(path)
    selected_writer = writer or ("ffmpeg" if path.suffix == ".mp4" else "pillow")
    anim.save(str(path), writer=selected_writer, fps=fps)
    plt.close(fig)


def _setup_main_axis(
    ax_main: plt.Axes,
    result: SimulationResult,
    obstacles: Sequence[Obstacle],
    safety_distance: float,
) -> None:
    ax_main.set_aspect("equal")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlabel(r"$x$ [m]")
    ax_main.set_ylabel(r"$y$ [m]")

    equidistant_points: list[NDArray[np.float64]] = []
    for index, obstacle in enumerate(obstacles):
        boundary = obstacle.boundary_polyline()
        label = "obstacle" if index == 0 else None
        if _is_closed(boundary):
            ax_main.fill(
                boundary[:, 0], boundary[:, 1],
                facecolor="0.82", edgecolor="black", linewidth=1.5,
                zorder=1, label=label,
            )
        else:
            ax_main.plot(
                boundary[:, 0], boundary[:, 1],
                color="black", linewidth=2, zorder=1, label=label,
            )

        equidistant = obstacle.equidistant_polyline(safety_distance)
        equidistant_points.append(equidistant)
        ax_main.plot(
            equidistant[:, 0], equidistant[:, 1],
            color="tab:blue", linewidth=1.2, linestyle="--", alpha=0.9, zorder=1,
            label=(
                rf"equidistant, $\rho_0 = {safety_distance:.2f}$ m"
                if index == 0
                else None
            ),
        )

    # Keep the equidistant curves in frame even where the vehicle never went.
    xs = [result.x]
    ys = [result.y]
    for points in equidistant_points:
        xs.append(points[:, 0])
        ys.append(points[:, 1])
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)

    margin = max(1.5, 0.12 * max(float(np.ptp(all_x)), float(np.ptp(all_y)), 1.0))
    ax_main.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax_main.set_ylim(all_y.min() - margin, all_y.max() + margin)


def _setup_series_axis(
    ax: plt.Axes,
    title: str,
    times: NDArray[np.float64],
    values: NDArray[np.float64],
) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$t$ [s]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        # Limits come from the full run, but the trace is drawn as it
        # happens, so playback never shows the future.
        span = max(float(np.ptp(finite_values)), 1e-3)
        ax.set_ylim(finite_values.min() - 0.1 * span, finite_values.max() + 0.1 * span)
    if times.size:
        ax.set_xlim(float(times[0]), float(times[-1]))


def _is_closed(polyline: NDArray[np.float64]) -> bool:
    return bool(
        polyline.shape[0] > 2 and np.allclose(polyline[0], polyline[-1], atol=1e-9)
    )


def _triangle_vertices(
    x: float, y: float, heading: float, size: float
) -> NDArray[np.float64]:
    """Return the three corners of the vehicle triangle, nose along ``heading``."""

    body = np.array(
        [[0.6 * size, 0.0], [-0.4 * size, 0.35 * size], [-0.4 * size, -0.35 * size]]
    )
    rotation = np.array(
        [
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)],
        ]
    )
    return body @ rotation.T + np.array([x, y])


def _set_point(line: plt.Line2D, point: NDArray[np.float64] | None) -> None:
    if point is None:
        line.set_data([], [])
    else:
        line.set_data([point[0]], [point[1]])
