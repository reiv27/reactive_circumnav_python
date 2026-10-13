"""Compare baseline, delayed-uncompensated and delayed+predictor on one scene.

Stage 1a acceptance check as a figure: "predictor" should overlay "baseline"
almost exactly, while "uncompensated" visibly departs from it into the
delay-induced low-frequency limit cycle (docs/delay_report.md,
docs/RL_plan.md).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from circumnav.analysis.metrics import obstacle_clearance_metrics
from circumnav.controllers.reactive import CircumnavMode
from circumnav.examples.reactive_circumnav import build_obstacle
from circumnav.scenarios.reactive import ReactiveCircumnavScenario

CASES = ("baseline", "uncompensated", "predictor")
COLORS = {"baseline": "k", "uncompensated": "tab:red", "predictor": "tab:blue"}
CASE_LABELS = {
    "baseline": r"no delay",
    "uncompensated": r"$\tau_s$, uncompensated",
    "predictor": r"$\tau_s$ + predictor",
}


@dataclass
class CaseRun:
    case: str
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
    delay: float,
    video: Path | None,
    fps: int,
    speed: float,
) -> CaseRun:
    if case == "uncompensated":
        scenario = replace(scenario, sensing_delay=delay, sensing_compensation=False)
    elif case == "predictor":
        scenario = replace(scenario, sensing_delay=delay, sensing_compensation=True)
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
        label = CASE_LABELS[run.case]
        ax_xy.plot(*run.position.T, color=color, lw=1.1, label=label, zorder=3)
        ax_d.plot(run.log_time, run.d_r, color=color, lw=0.7, label=label)
        ax_u.plot(run.time, run.applied_yaw_rate, color=color, lw=0.6)
        ax_m.step(
            run.log_time, index + 0.8 * run.in_orbit, color=color, lw=1.0, where="post"
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
    ax_d.legend(loc="upper right", fontsize=8)
    for ax in (ax_xy, ax_d, ax_u, ax_m):
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="ellipse")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="sensing delay tau_s in seconds; must be a multiple of "
        "control_period (default 0.1)",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=3)
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

    baseline = next(run for run in runs if run.case == "baseline")
    for run in runs:
        max_position_error = (
            0.0
            if run.case == "baseline"
            else float(np.max(np.abs(run.position - baseline.position)))
        )
        settled = run.log_time >= arguments.duration / 2
        print(
            f"{run.case:>13}  "
            f"peak|d_R|(2nd half)={np.nanmax(np.abs(run.d_r[settled])):7.4f}  "
            f"min clearance={run.minimum_clearance:6.3f}  "
            f"switches={run.switch_count:3d}  "
            f"max|Δposition vs baseline|={max_position_error:.6f}"
        )

    title = (
        rf"{arguments.shape}: $\tau_s={arguments.delay:g}$ s, "
        rf"baseline vs uncompensated vs predictor"
    )
    plot_overview(
        runs, scenario, arguments.output / f"{arguments.shape}_predictor_compare.png", title
    )


if __name__ == "__main__":
    main()
