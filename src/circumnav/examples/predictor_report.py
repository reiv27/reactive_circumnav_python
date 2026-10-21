"""Quantitative study of sensing-delay compensation on one scene.

Default scene: two perpendicular ellipses, so the law visits mode G.

  sweep   tau_s alone, uncompensated vs PredictorController: the metrics of
          delay_sweep plus the largest position deviation from the delay-free
          run.
  joint   tau_s together with tau_a or with a lag: uncompensated vs predictor
          vs the reference without tau_s. The predictor removes only the
          sensing part, so the compensated run must coincide with the
          reference.
  zoom    d_R(t), u(t) around one mode-G episode, and the start/stop times of
          every mode-G episode.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from circumnav.examples.delay_sweep import SweepPoint, run_point
from circumnav.examples.predictor_compare import (
    CASE_LABELS,
    CASES,
    COLORS,
    CaseRun,
    run_case,
)
from circumnav.examples.reactive_circumnav import build_obstacle
from circumnav.scenarios.reactive import ReactiveCircumnavScenario

SWEEP_DELAYS = (0.0, 0.04, 0.08, 0.10, 0.12, 0.16, 0.20, 0.40, 0.80, 1.20, 1.60)
SWEEP_CASES = ("uncompensated", "predictor")
DEVIATION_FLOOR = 1e-16


@dataclass(frozen=True)
class Measured:
    label: str
    point: SweepPoint
    position: np.ndarray


def measure(label: str, scenario: ReactiveCircumnavScenario) -> Measured:
    point = run_point(scenario, "sensing", scenario.sensing_delay)
    result, _ = scenario.run()
    return Measured(label, point, result.state[:, :2])


def _measure(args: tuple[str, ReactiveCircumnavScenario]) -> Measured:
    return measure(*args)


def _run_case(args: tuple) -> CaseRun:
    return run_case(*args)


def deviation(measured: Measured, reference: Measured) -> float:
    return float(np.max(np.abs(measured.position - reference.position)))


def sweep_sensing(
    base: ReactiveCircumnavScenario, delays: tuple[float, ...], workers: int
) -> dict[str, list[Measured]]:
    jobs = [
        (case, replace(base, sensing_delay=d, sensing_compensation=case == "predictor"))
        for case in SWEEP_CASES
        for d in delays
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        flat = list(pool.map(_measure, jobs))
    return {
        case: flat[index * len(delays) : (index + 1) * len(delays)]
        for index, case in enumerate(SWEEP_CASES)
    }


def joint_scenarios(
    base: ReactiveCircumnavScenario, delay: float
) -> list[tuple[str, ReactiveCircumnavScenario, str]]:
    """(label, scenario, reference label); the reference is its own reference."""

    rows: list[tuple[str, ReactiveCircumnavScenario, str]] = [
        ("no delay", base, "no delay"),
        (
            "tau_s, uncompensated",
            replace(base, sensing_delay=delay),
            "no delay",
        ),
        (
            "tau_s, predictor",
            replace(base, sensing_delay=delay, sensing_compensation=True),
            "no delay",
        ),
    ]
    for name, field in (("tau_a", "actuation_delay"), ("lag", "actuator_time_constant")):
        reference = replace(base, **{field: delay})
        label = f"{name} only"
        rows += [
            (label, reference, label),
            (
                f"tau_s + {name}, uncompensated",
                replace(reference, sensing_delay=delay),
                label,
            ),
            (
                f"tau_s + {name}, predictor",
                replace(reference, sensing_delay=delay, sensing_compensation=True),
                label,
            ),
        ]
    return rows


def run_joint(
    base: ReactiveCircumnavScenario, delay: float, workers: int
) -> list[tuple[Measured, Measured]]:
    rows = joint_scenarios(base, delay)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        measured = {
            m.label: m for m in pool.map(_measure, [(l, s) for l, s, _ in rows])
        }
    return [(measured[label], measured[reference]) for label, _, reference in rows]


def orbit_intervals(run: CaseRun) -> list[tuple[float, float]]:
    """(start, stop) control times of every mode-G episode; open ones end at the last update."""

    edges = np.diff(np.concatenate(([0], run.in_orbit.astype(int), [0])))
    starts = run.log_time[np.flatnonzero(edges == 1)]
    stops = run.log_time[np.minimum(np.flatnonzero(edges == -1), run.log_time.size - 1)]
    return list(zip(starts.tolist(), stops.tolist()))


def print_point(label: str, measured: Measured, reference: Measured | None) -> None:
    p = measured.point
    deviation_text = (
        f"{deviation(measured, reference):9.2e}" if reference is not None else "        -"
    )
    print(
        f"{label:>34}  peak={p.settled_peak:7.4f}  flips/s={p.switch_rate:6.2f}  "
        f"G={p.orbit_fraction:5.3f}  sw={p.switch_count:3d}  "
        f"clear={p.minimum_clearance:6.3f}  collided={p.collided!s:5}  "
        f"max|dpos|={deviation_text}"
    )


def plot_sweep(
    results: dict[str, list[Measured]],
    delays: tuple[float, ...],
    path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = results["uncompensated"][0]
    style = {
        "uncompensated": ("tab:red", "o-", r"$\tau_s$, uncompensated"),
        "predictor": ("tab:blue", "s--", r"$\tau_s$ + predictor"),
    }
    panels = (
        (r"settled $\max|d_R|$, m", lambda m: m.point.settled_peak, False, 0),
        (r"relay flips per second", lambda m: m.point.switch_rate, True, 0),
        (r"minimum clearance, m", lambda m: m.point.minimum_clearance, False, 0),
        (
            r"$\max|\Delta\,\mathrm{position}|$ vs no delay, m",
            lambda m: max(deviation(m, reference), DEVIATION_FLOOR),
            True,
            1,
        ),
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (ylabel, getter, log, first) in zip(axes.ravel(), panels):
        for case in SWEEP_CASES:
            color, marker, label = style[case]
            ax.plot(
                delays[first:],
                [getter(m) for m in results[case][first:]],
                marker,
                c=color,
                label=label,
            )
        hit = [m for m in results["uncompensated"] if m.point.collided]
        if hit and "clearance" in ylabel:
            ax.scatter(
                [m.point.value for m in hit],
                [m.point.minimum_clearance for m in hit],
                marker="x", s=90, c="k", zorder=5, label="collision",
            )
        ax.set_xscale("symlog", linthresh=0.04)
        if log:
            ax.set_yscale("log")
        ax.set(xlabel=r"sensing delay $\tau_s$, s", ylabel=ylabel)
        ax.grid(alpha=0.3)
    axes[1, 1].text(
        0.04, 0.5,
        "predictor: exactly 0\n(drawn at the $10^{-16}$ floor)",
        transform=axes[1, 1].transAxes, color="tab:blue", fontsize=9,
    )
    axes[0, 0].legend()
    axes[1, 0].legend(loc="lower left")
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_zoom(runs: list[CaseRun], delay: float, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = next(run for run in runs if run.case == "baseline")
    duration = float(baseline.time[-1])
    late = [a for a, _ in orbit_intervals(baseline) if a >= 0.5 * duration]
    start = late[0] - 2.0 if late else 0.5 * duration
    window = (start, start + 10.0)

    fig, (ax_d, ax_u) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True, constrained_layout=True
    )
    for begin, end in orbit_intervals(baseline):
        for ax in (ax_d, ax_u):
            ax.axvspan(begin, end, color="tab:orange", alpha=0.15, lw=0)
    # The predictor run lies exactly on the baseline: draw the baseline thick
    # underneath and the predictor dashed on top so both stay visible.
    style = {
        "baseline": {"lw": 3.0, "zorder": 2},
        "predictor": {"lw": 1.0, "ls": "--", "zorder": 4},
        "uncompensated": {"lw": 1.0, "zorder": 3},
    }
    inside = []
    for run in runs:
        ax_d.plot(
            run.log_time, run.d_r, color=COLORS[run.case],
            label=CASE_LABELS[run.case], **style[run.case],
        )
        ax_u.plot(
            run.time, run.applied_yaw_rate, color=COLORS[run.case],
            lw=0.8, zorder=style[run.case]["zorder"],
        )
        visible = run.d_r[(run.log_time >= window[0]) & (run.log_time <= window[1])]
        inside.append(visible[np.isfinite(visible)])
    span = np.concatenate(inside)
    margin = 0.08 * float(np.ptp(span))
    ax_d.set(
        ylabel=r"$d_R(t)$, m",
        xlim=window,
        ylim=(float(span.min()) - margin, float(span.max()) + margin),
    )
    ax_u.set(xlabel=r"$t$, s", ylabel=r"$u(t)=\omega$, rad/s")
    ax_d.legend(loc="upper right", fontsize=8)
    for ax in (ax_d, ax_u):
        ax.grid(alpha=0.3)
    fig.suptitle(title + "  (shaded: mode G in the delay-free run)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--shape", default="two_ellipses")
    parser.add_argument(
        "--delay", type=float, default=0.2,
        help="tau_s for the zoom figure and the mode-G episode table (multiple of T)",
    )
    parser.add_argument(
        "--joint-delay", type=float, default=0.1,
        help="value used for tau_s, tau_a and the lag in the joint experiment",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("simulation_output"))
    arguments = parser.parse_args()

    base = ReactiveCircumnavScenario(duration=arguments.duration)
    base = replace(
        base, obstacles=build_obstacle(arguments.shape, base.rho_0, base.turning_radius)
    )
    arguments.output.mkdir(parents=True, exist_ok=True)

    print(f"== sweep: tau_s alone, scene {arguments.shape} ==")
    results = sweep_sensing(base, SWEEP_DELAYS, arguments.workers)
    reference = results["uncompensated"][0]
    for index, delay in enumerate(SWEEP_DELAYS):
        for case in SWEEP_CASES:
            print_point(f"{case} tau_s={delay:4.2f}", results[case][index], reference)
    np.savez_compressed(
        arguments.output / f"{arguments.shape}_predictor_sweep.npz",
        delay=np.array(SWEEP_DELAYS),
        **{
            f"{case}_{name}": np.array([getattr(m.point, name) for m in results[case]])
            for case in SWEEP_CASES
            for name in SweepPoint.__dataclass_fields__
            if name != "channel"
        },
        **{
            f"{case}_position_deviation": np.array(
                [deviation(m, reference) for m in results[case]]
            )
            for case in SWEEP_CASES
        },
    )
    plot_sweep(
        results,
        SWEEP_DELAYS,
        arguments.output / f"{arguments.shape}_predictor_sweep.png",
        f"{arguments.shape}: sensing delay alone, uncompensated vs predictor",
    )

    print(f"\n== joint delays, every value {arguments.joint_delay:g} s ==")
    for measured, ref in run_joint(base, arguments.joint_delay, arguments.workers):
        print_point(measured.label, measured, ref)

    print(f"\n== mode-G episodes (start-stop, duration), tau_s={arguments.delay:g} s ==")
    scenario_jobs = [(base, case, arguments.delay, None, 20, 3.0) for case in CASES]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        runs = list(pool.map(_run_case, scenario_jobs))
    for run in runs:
        episodes = "  ".join(
            f"{a:6.2f}-{b:6.2f} ({b - a:4.2f} s)" for a, b in orbit_intervals(run)
        )
        print(f"{run.case:>13}: {episodes}")
    plot_zoom(
        runs,
        arguments.delay,
        arguments.output / f"{arguments.shape}_predictor_zoom.png",
        rf"{arguments.shape}: $\tau_s={arguments.delay:g}$ s around a mode-G episode",
    )


if __name__ == "__main__":
    main()
