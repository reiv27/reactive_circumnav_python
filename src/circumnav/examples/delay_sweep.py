"""Sweep one delay channel at a time while the other two stay at zero.

Channels:
  sensing    tau_s, the age of the state the controller sees
  actuation  tau_a (+ tau_c), pure transport delay on the command
  lag        tau, time constant of a first-order lag on the command
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import math
from pathlib import Path

import numpy as np

from circumnav.analysis.metrics import obstacle_clearance_metrics
from circumnav.controllers.reactive import CircumnavMode
from circumnav.examples.reactive_circumnav import build_obstacle
from circumnav.scenarios.reactive import ReactiveCircumnavScenario

CHANNELS = {
    "sensing": "sensing_delay",
    "actuation": "actuation_delay",
    "lag": "actuator_time_constant",
}
LABELS = {
    "sensing": r"$\tau_s$ (sensing delay)",
    "actuation": r"$\tau_a+\tau_c$ (actuation delay)",
    "lag": r"$\tau$ (first-order lag)",
}
SETTLED_FRACTION = 0.5


@dataclass(frozen=True)
class SweepPoint:
    channel: str
    value: float
    settled_peak: float
    settled_rms: float
    settled_bias: float
    #: sign flips of the relay command per second, settled half of the run.
    switch_rate: float
    #: max |s| of the sliding variable s = d_R' + gain * sat(d_R), settled half.
    sliding_peak: float
    lost_fraction: float
    orbit_fraction: float
    switch_count: int
    minimum_clearance: float
    collided: bool


def run_point(
    scenario: ReactiveCircumnavScenario, channel: str, value: float
) -> SweepPoint:
    scenario = replace(scenario, **{CHANNELS[channel]: value})
    result, controller = scenario.run()
    clearance = obstacle_clearance_metrics(
        result, scenario.build_obstacles(), collision_distance=0.05
    )

    log = controller.log
    time = np.array([entry.time for entry in log])
    d_r = np.array([entry.equidistant_deviation for entry in log])
    in_orbit = np.array([entry.mode is CircumnavMode.ORBIT for entry in log])
    late = time >= SETTLED_FRACTION * scenario.duration
    settled = d_r[late]
    valid = settled[np.isfinite(settled)]
    command = np.array([entry.yaw_rate_command for entry in log])[late]
    command = command[command != 0.0]
    flips = int(np.count_nonzero(np.diff(np.sign(command))))
    span = float(time[late][-1] - time[late][0]) if late.sum() > 1 else math.nan
    config = scenario.build_controller().config
    sliding = np.array([entry.range_rate for entry in log])[late]
    sliding = sliding + config.dead_zone_gain * np.clip(
        settled, -config.dead_zone, config.dead_zone
    )
    sliding = sliding[np.isfinite(sliding)]
    lost = float(np.mean(~np.isfinite(d_r)))

    return SweepPoint(
        channel=channel,
        value=value,
        settled_peak=float(np.max(np.abs(valid))) if valid.size else math.nan,
        settled_rms=float(np.sqrt(np.mean(valid**2))) if valid.size else math.nan,
        settled_bias=float(np.mean(valid)) if valid.size else math.nan,
        switch_rate=flips / span if span == span and span > 0 else math.nan,
        sliding_peak=float(np.max(np.abs(sliding))) if sliding.size else math.nan,
        lost_fraction=lost,
        orbit_fraction=float(np.mean(in_orbit)),
        switch_count=controller.switch_count,
        minimum_clearance=clearance.minimum_clearance,
        collided=clearance.collided,
    )


def _run(args: tuple[ReactiveCircumnavScenario, str, float]) -> SweepPoint:
    return run_point(*args)


def sweep(
    scenario: ReactiveCircumnavScenario,
    channels: list[str],
    values: np.ndarray,
    workers: int,
) -> list[SweepPoint]:
    jobs = [(scenario, channel, float(v)) for channel in channels for v in values]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run, jobs))


def plot(points: list[SweepPoint], speed: float, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()
    colors = {"sensing": "tab:red", "actuation": "tab:blue", "lag": "tab:green"}
    for channel in dict.fromkeys(p.channel for p in points):
        rows = [p for p in points if p.channel == channel]
        x = [p.value for p in rows]
        axes[0].plot(x, [p.settled_peak for p in rows], "o-", c=colors[channel],
                     label=LABELS[channel])
        axes[1].plot(x, [p.settled_bias for p in rows], "o-", c=colors[channel],
                     label=LABELS[channel])
        axes[2].plot(x, [p.minimum_clearance for p in rows], "o-", c=colors[channel],
                     label=LABELS[channel])
        axes[3].plot(x, [p.switch_rate for p in rows], "o-", c=colors[channel],
                     label=LABELS[channel])
    grid = np.linspace(0.0, max(p.value for p in points), 50)
    axes[0].plot(grid, 2.0 * speed * grid, "k--", lw=1, label=r"$2v\tau$")
    axes[0].set(xlabel=r"delay / time constant, s",
                ylabel=r"settled $\max|d_R|$, m")
    axes[1].set(xlabel=r"delay / time constant, s",
                ylabel=r"settled mean $d_R$, m")
    axes[2].set(xlabel=r"delay / time constant, s",
                ylabel=r"minimum clearance, m")
    axes[3].set(xlabel=r"delay / time constant, s",
                ylabel=r"relay flips per second", yscale="log")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    fig.savefig(path, dpi=140)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--channel", choices=[*CHANNELS, "all"], default="all")
    parser.add_argument("--shape", default="ellipse")
    parser.add_argument("--max", type=float, default=0.2, dest="maximum")
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--tag", default="", help="suffix for the output file names")
    parser.add_argument("--output", type=Path, default=Path("simulation_output"))
    arguments = parser.parse_args()

    base = ReactiveCircumnavScenario(duration=arguments.duration)
    base = replace(
        base,
        obstacles=build_obstacle(arguments.shape, base.rho_0, base.turning_radius),
    )
    channels = list(CHANNELS) if arguments.channel == "all" else [arguments.channel]
    values = np.round(
        np.arange(0.0, arguments.maximum + 0.5 * arguments.step, arguments.step), 10
    )
    points = sweep(base, channels, values, arguments.workers)

    for p in points:
        print(
            f"{p.channel:>9} {p.value:5.2f}  peak={p.settled_peak:7.4f} "
            f"bias={p.settled_bias:+7.4f} rms={p.settled_rms:7.4f} "
            f"flips/s={p.switch_rate:6.2f} "
            f"|s|max={p.sliding_peak:6.3f} "
            f"lost={p.lost_fraction:4.2f} G={p.orbit_fraction:4.2f} "
            f"sw={p.switch_count:3d} clear={p.minimum_clearance:6.3f} "
            f"collided={p.collided}"
        )

    arguments.output.mkdir(parents=True, exist_ok=True)
    stem = f"delay_sweep_{arguments.shape}{arguments.tag}"
    np.savez_compressed(
        arguments.output / f"{stem}.npz",
        **{
            name: np.array([getattr(p, name) for p in points])
            for name in SweepPoint.__dataclass_fields__
        },
    )
    plot(points, base.forward_speed, arguments.output / f"{stem}.png",
         f"{arguments.shape}: one delay channel non-zero at a time")


if __name__ == "__main__":
    main()
