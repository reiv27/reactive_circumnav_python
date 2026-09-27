"""Closed-loop heading-control experiment for a Dubins vehicle."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from circumnav.analysis.metrics import HeadingExperimentMetrics, heading_metrics
from circumnav.scenarios.heading import HeadingControlScenario
from circumnav.simulation.result import SimulationResult


def run_experiment(
    scenario: HeadingControlScenario | None = None,
) -> tuple[SimulationResult, HeadingExperimentMetrics]:
    """Run a heading-control scenario and compute its closed-loop metrics."""

    selected_scenario = scenario or HeadingControlScenario()
    result = selected_scenario.run()
    metrics = heading_metrics(result, selected_scenario.desired_heading)
    return result, metrics


def main() -> None:
    """Command-line entry point for the heading-control demo."""

    parser = argparse.ArgumentParser(
        description="Run the baseline Dubins heading-control experiment."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a compressed .npz simulation log.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=1.5,
        help="Proportional heading gain in 1/s (default: 1.5).",
    )
    parser.add_argument(
        "--control-period",
        type=float,
        default=0.05,
        help="Controller sampling period in seconds (default: 0.05).",
    )
    parser.add_argument(
        "--integration-step",
        type=float,
        default=0.01,
        help="Plant propagation and logging step in seconds (default: 0.01).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Experiment duration in seconds (default: 8.0).",
    )
    parser.add_argument(
        "--initial-heading-deg",
        type=float,
        default=0.0,
        help="Initial heading in degrees (default: 0).",
    )
    parser.add_argument(
        "--desired-heading-deg",
        type=float,
        default=90.0,
        help="Heading reference in degrees (default: 90).",
    )
    arguments = parser.parse_args()

    scenario = HeadingControlScenario(
        duration=arguments.duration,
        integration_step=arguments.integration_step,
        control_period=arguments.control_period,
        initial_heading=math.radians(arguments.initial_heading_deg),
        desired_heading=math.radians(arguments.desired_heading_deg),
        proportional_gain=arguments.gain,
    )
    result, metrics = run_experiment(scenario)
    if arguments.output is not None:
        result.save_npz(arguments.output)

    print(f"final heading error: {metrics.final_heading_error:.6f} rad")
    print(f"RMS heading error:   {metrics.rms_heading_error:.6f} rad")
    print(f"path length:         {metrics.path_length:.6f} m")
    print(f"absolute yaw effort: {metrics.absolute_yaw_effort:.6f} rad")
    print(f"saturation fraction: {metrics.saturation_fraction:.3f}")


if __name__ == "__main__":
    main()
