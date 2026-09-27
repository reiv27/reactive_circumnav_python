"""Closed-loop reactive circumnavigation experiment for a Dubins vehicle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from circumnav.analysis.metrics import (
    ObstacleClearanceMetrics,
    obstacle_clearance_metrics,
)
from circumnav.controllers.reactive import ReactiveCircumnavController
from circumnav.models.obstacles import (
    CircleObstacle,
    EllipseObstacle,
    Obstacle,
    make_obstacles,
)
from circumnav.scenarios.reactive import ReactiveCircumnavScenario, gap_cluster
from circumnav.simulation.result import SimulationResult

OBSTACLE_SHAPES = ("circle", "ellipse", "wall", "cluster")


def build_obstacle(
    shape: str, rho_0: float = 2.0, turning_radius: float = 1.0
) -> tuple[Obstacle, ...]:
    """Build a demo obstacle of the requested shape, centered on the origin."""

    if shape == "cluster":
        return gap_cluster(rho_0=rho_0, turning_radius=turning_radius)
    if shape == "circle":
        return (CircleObstacle(center=np.array([0.0, 0.0]), radius=4.0, obstacle_id=0),)
    if shape == "ellipse":
        return (
            EllipseObstacle(
                center=np.array([0.0, 0.0]),
                semi_axis_x=6.0,
                semi_axis_y=3.0,
                obstacle_id=0,
                angle=np.pi / 6.0,
            ),
        )
    if shape == "wall":
        return make_obstacles([((0.0, -5.0), (0.0, 5.0))])
    raise ValueError(f"Unknown obstacle shape: {shape!r}")


def run_experiment(
    scenario: ReactiveCircumnavScenario | None = None,
    collision_distance: float = 0.05,
) -> tuple[SimulationResult, ReactiveCircumnavController, ObstacleClearanceMetrics]:
    """Run a scenario and measure its closest approach to any obstacle."""

    selected_scenario = scenario or ReactiveCircumnavScenario()
    result, controller = selected_scenario.run()
    metrics = obstacle_clearance_metrics(
        result,
        selected_scenario.build_obstacles(),
        collision_distance=collision_distance,
    )
    return result, controller, metrics


def main() -> None:
    """Command-line entry point for the reactive circumnavigation demo."""

    parser = argparse.ArgumentParser(
        description="Run the reactive circumnavigation experiment "
        "(range-limited circular visibility sensor)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a compressed .npz simulation log.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Experiment duration in seconds (default: 60.0).",
    )
    parser.add_argument(
        "--sensor-range",
        type=float,
        default=12.0,
        help="Sensor perception radius R_s in meters (default: 12.0).",
    )
    parser.add_argument(
        "--safety-distance",
        type=float,
        default=3.0,
        help="Commanded standoff rho_0 from the obstacle boundary; must be at "
        "least R_min = v / omega (default: 3.0).",
    )
    parser.add_argument(
        "--forward-speed",
        type=float,
        default=2.0,
        help="Forward speed v in m/s (default: 2.0).",
    )
    parser.add_argument(
        "--yaw-rate-magnitude",
        type=float,
        default=1.0,
        help="Bang-bang yaw-rate magnitude omega in rad/s (default: 1.0).",
    )
    parser.add_argument(
        "--obstacle",
        choices=OBSTACLE_SHAPES,
        default="circle",
        help="Obstacle shape for the demo scene (default: circle).",
    )
    parser.add_argument(
        "--allow-infeasible-turning",
        action="store_true",
        help="Run even when rho_0 < R_min, i.e. when the vehicle cannot "
        "physically follow the equidistant curve. Off by default.",
    )
    parser.add_argument(
        "--animate",
        type=Path,
        help="Optional path (.mp4 or .gif) to save an animated playback with a "
        "live mode/dR/control panel. Requires the 'viz' extra (matplotlib).",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=30,
        help="Frame rate of the saved animation (default: 30).",
    )
    parser.add_argument(
        "--animation-speed",
        type=float,
        default=1.0,
        help="Real-time factor for the saved animation; >1 plays faster "
        "than the simulated duration (default: 1.0).",
    )
    arguments = parser.parse_args()

    scenario = ReactiveCircumnavScenario(
        duration=arguments.duration,
        sensor_range=arguments.sensor_range,
        safety_distance=arguments.safety_distance,
        forward_speed=arguments.forward_speed,
        yaw_rate_magnitude=arguments.yaw_rate_magnitude,
        obstacles=build_obstacle(
            arguments.obstacle,
            rho_0=arguments.safety_distance,
            turning_radius=arguments.forward_speed / arguments.yaw_rate_magnitude,
        ),
        require_feasible_turning=not arguments.allow_infeasible_turning,
    )
    try:
        result, controller, metrics = run_experiment(scenario)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from None

    if arguments.output is not None:
        result.save_npz(arguments.output)

    if arguments.animate is not None:
        from circumnav.analysis.animation import (
            AnimationSettings,
            build_reactive_animation,
            save_reactive_animation,
        )

        fig, anim = build_reactive_animation(
            result,
            controller,
            scenario.build_obstacles(),
            settings=AnimationSettings(
                fps=arguments.animation_fps,
                real_time_factor=arguments.animation_speed,
            ),
        )
        save_reactive_animation(
            fig, anim, arguments.animate, fps=arguments.animation_fps
        )
        print(f"animation saved to:   {arguments.animate}")

    feasibility = "ok" if scenario.rho_0 >= scenario.turning_radius else "VIOLATED"
    print(f"rho_0 / R_min:        {scenario.rho_0:.3f} / "
          f"{scenario.turning_radius:.3f} m  ({feasibility})")
    print(f"equidistant curvature:{scenario.equidistant_curvature:.3f} m")
    print(f"closing speed ratio:  {scenario.closing_speed_ratio:.3f} "
          f"(keep near 0.3)")
    print(f"mode at end:          {controller.mode.value}")
    print(f"mode switch count:    {controller.switch_count}")
    print(f"minimum clearance:    {metrics.minimum_clearance:.4f} m")
    print(f"collision detected:   {metrics.collided}")
    print(f"final position:       ({result.x[-1]:.3f}, {result.y[-1]:.3f})")


if __name__ == "__main__":
    main()
