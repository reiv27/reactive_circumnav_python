"""Run one scene under each delay channel and compare against the baseline.

Every run uses the same scenario; only one of tau_s, tau_a (+ tau_c) and the
lag time constant is non-zero at a time. Each case is saved as an animation,
and the cases are overlaid in comparison figures.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from circumnav.analysis.metrics import obstacle_clearance_metrics
from circumnav.controllers.reactive import CircumnavMode
from circumnav.examples.delay_sweep import CHANNELS, LABELS
from circumnav.examples.reactive_circumnav import build_obstacle
from circumnav.scenarios.reactive import ReactiveCircumnavScenario

CASES = ("baseline", *CHANNELS)
COLORS = {
    "baseline": "k",
    "sensing": "tab:red",
    "actuation": "tab:blue",
    "lag": "tab:green",
}
CASE_LABELS = {"baseline": r"no delay", **LABELS}


@dataclass
class CaseRun:
    case: str
    value: float
    time: np.ndarray
    position: np.ndarray
    applied_yaw_rate: np.ndarray
    log_time: np.ndarray
    d_r: np.ndarray
    in_orbit: np.ndarray
    minimum_clearance: float
    switch_count: int


def run_case(
    scenario: ReactiveCircumnavScenario,
    case: str,
    value: float,
    video: Path | None,
    fps: int,
    speed: float,
) -> CaseRun:
    if case != "baseline":
        scenario = replace(scenario, **{CHANNELS[case]: value})
    result, controller = scenario.run()
    obstacles = scenario.build_obstacles()

    if video is not None:
        from circumnav.analysis.animation import (
            AnimationSettings,
            build_reactive_animation,
            save_reactive_animation,
        )

        fig, anim = build_reactive_animation(
            result,
            controller,
            obstacles,
            settings=AnimationSettings(fps=fps, real_time_factor=speed),
        )
        save_reactive_animation(fig, anim, video, fps=fps)

    log = controller.log
    return CaseRun(
        case=case,
        value=0.0 if case == "baseline" else value,
        time=result.time,
        position=result.state[:, :2],
        applied_yaw_rate=result.applied_control[:, 1],
        log_time=np.array([entry.time for entry in log]),
        d_r=np.array([entry.equidistant_deviation for entry in log]),
        in_orbit=np.array([entry.mode is CircumnavMode.ORBIT for entry in log]),
        minimum_clearance=obstacle_clearance_metrics(
            result, obstacles, collision_distance=0.05
        ).minimum_clearance,
        switch_count=controller.switch_count,
    )


def _run(args: tuple) -> CaseRun:
    return run_case(*args)


def _legend_label(run: CaseRun) -> str:
    if run.case == "baseline":
        return CASE_LABELS["baseline"]
    return rf"{CASE_LABELS[run.case]} $= {run.value:g}$ s"


def plot_overview(
    runs: list[CaseRun],
    scenario: ReactiveCircumnavScenario,
    path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.25])
    ax_xy = fig.add_subplot(grid[:, 0])
    ax_d = fig.add_subplot(grid[0, 1])
    ax_u = fig.add_subplot(grid[1, 1], sharex=ax_d)
    ax_m = fig.add_subplot(grid[2, 1], sharex=ax_d)

    for obstacle in scenario.build_obstacles():
        boundary = obstacle.boundary_polyline()
        ax_xy.fill(*boundary.T, color="0.8", zorder=1)
        ax_xy.plot(*boundary.T, color="0.4", lw=1, zorder=2)
        equidistant = obstacle.equidistant_polyline(scenario.rho_0)
        ax_xy.plot(*equidistant.T, color="tab:blue", lw=0.8, alpha=0.6, zorder=2)

    for index, run in enumerate(runs):
        color = COLORS[run.case]
        label = _legend_label(run)
        ax_xy.plot(*run.position.T, color=color, lw=1.1, label=label, zorder=3)
        ax_d.plot(run.log_time, run.d_r, color=color, lw=0.7, label=label)
        ax_u.plot(run.time, run.applied_yaw_rate, color=color, lw=0.6)
        ax_m.step(
            run.log_time,
            index + 0.8 * run.in_orbit,
            color=color,
            lw=1.0,
            where="post",
        )

    ax_xy.set_aspect("equal")
    ax_xy.set(xlabel=r"$x$, m", ylabel=r"$y$, m")
    ax_xy.legend(loc="upper left", fontsize=8)
    ax_d.set(ylabel=r"$d_R(t)$, m")
    ax_u.set(ylabel=r"$u(t)=\omega$, rad/s")
    ax_m.set(
        xlabel=r"$t$, s",
        ylabel="mode",
        yticks=np.arange(len(runs)) + 0.4,
        yticklabels=[r"$\mathrm{C}\,/\,\mathrm{G}$"] * len(runs),
    )
    ax_d.legend(loc="upper right", fontsize=8, ncol=2)
    for ax in (ax_xy, ax_d, ax_u, ax_m):
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_zoom(
    runs: list[CaseRun], window: tuple[float, float], path: Path, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(runs), 2, figsize=(14, 2.2 * len(runs)), sharex=True,
        constrained_layout=True,
    )
    for row, run in zip(axes, runs):
        color = COLORS[run.case]
        mask = (run.log_time >= window[0]) & (run.log_time <= window[1])
        row[0].plot(run.log_time[mask], run.d_r[mask], color=color, lw=0.9)
        row[0].set_ylabel(_legend_label(run), fontsize=8)
        tmask = (run.time >= window[0]) & (run.time <= window[1])
        row[1].plot(run.time[tmask], run.applied_yaw_rate[tmask], color=color, lw=0.9)
        for ax in row:
            ax.grid(alpha=0.3)
    axes[0, 0].set_title(r"$d_R(t)$, m")
    axes[0, 1].set_title(r"$u(t)=\omega$, rad/s")
    axes[-1, 0].set_xlabel(r"$t$, s")
    axes[-1, 1].set_xlabel(r"$t$, s")
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="ellipse")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="value used for every non-zero channel (default 0.2)")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--zoom", type=float, nargs=2, default=None,
                        metavar=("T0", "T1"))
    parser.add_argument("--output", type=Path, default=Path("simulation_output"))
    arguments = parser.parse_args()

    base = ReactiveCircumnavScenario(duration=arguments.duration)
    scenario = replace(
        base,
        obstacles=build_obstacle(arguments.shape, base.rho_0, base.turning_radius),
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            scenario,
            case,
            arguments.delay,
            None
            if arguments.no_video
            else arguments.output / f"{arguments.shape}_{case}.mp4",
            arguments.fps,
            arguments.speed,
        )
        for case in CASES
    ]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        runs = list(pool.map(_run, jobs))

    for run in runs:
        print(
            f"{run.case:>9} {run.value:4.2f}  "
            f"peak|d_R|(2nd half)="
            f"{np.nanmax(np.abs(run.d_r[run.log_time >= arguments.duration / 2])):7.4f}  "
            f"min clearance={run.minimum_clearance:6.3f}  switches={run.switch_count}"
        )

    title = (
        rf"{arguments.shape}: same scene, one channel at "
        rf"${arguments.delay:g}$ s, others zero"
    )
    plot_overview(
        runs, scenario, arguments.output / f"{arguments.shape}_compare.png", title
    )
    zoom = tuple(arguments.zoom or (0.75 * arguments.duration,
                                    0.75 * arguments.duration + 6.0))
    plot_zoom(
        runs, zoom, arguments.output / f"{arguments.shape}_compare_zoom.png",
        title + rf" (zoom ${zoom[0]:g}$–${zoom[1]:g}$ s)",
    )


if __name__ == "__main__":
    main()
