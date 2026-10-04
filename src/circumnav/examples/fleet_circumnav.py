"""Several independent vehicles circumnavigating one ellipse.

Vehicles start at different points of the equidistant curve, evenly spaced in
arc length, and do not interact. No delays.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from circumnav.analysis.metrics import (
    fleet_separation_metrics,
    hidden_episode_durations,
    neighbour_visibility,
    neighbour_visibility_metrics,
    obstacle_clearance_metrics,
)
from circumnav.models.obstacles import EllipseObstacle
from circumnav.scenarios.fleet import FleetScenario
from circumnav.scenarios.reactive import ReactiveCircumnavScenario


#: Semi-axes of the fleet's ellipse. More elongated than the single-vehicle demo
#: (6 x 3) so a neighbour stays hidden behind the body for longer.
DEFAULT_SEMI_AXES = (7.0, 3.0)


def build_fleet_scenario(
    robot_count: int,
    duration: float,
    phase: float = 0.0,
    semi_axes: tuple[float, float] = DEFAULT_SEMI_AXES,
) -> FleetScenario:
    ellipse = EllipseObstacle(
        center=np.array([0.0, 0.0]),
        semi_axis_x=semi_axes[0],
        semi_axis_y=semi_axes[1],
        obstacle_id=0,
        angle=np.pi / 6.0,
    )
    base = replace(ReactiveCircumnavScenario(duration=duration), obstacles=(ellipse,))
    return FleetScenario(base, robot_count=robot_count, phase=phase)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--phase", type=float, default=0.0,
                        help="shift along the curve, as a fraction of its length")
    parser.add_argument("--semi-axes", type=float, nargs=2,
                        default=DEFAULT_SEMI_AXES, metavar=("A", "B"),
                        help="ellipse semi-axes in metres (default: 7 3)")
    parser.add_argument("--focus", type=int, default=0,
                        help="vehicle drawn red, shown in the panels (default 0)")
    parser.add_argument("--neighbour-range", type=float, default=None,
                        help="line-of-sight range between vehicles "
                        "(default: the obstacle sensor range)")
    parser.add_argument("--figure", type=Path, help="static summary figure (.png)")
    parser.add_argument("--animate", type=Path, help="playback (.mp4 or .gif)")
    parser.add_argument("--animation-fps", type=int, default=30)
    parser.add_argument("--animation-speed", type=float, default=1.0)
    arguments = parser.parse_args()

    try:
        scenario = build_fleet_scenario(
            arguments.robots, arguments.duration, arguments.phase,
            tuple(arguments.semi_axes),
        )
        fleet, controllers = scenario.run()
    except ValueError as error:
        raise SystemExit(f"error: {error}") from None

    obstacles = scenario.build_obstacles()
    print(f"vehicles:             {fleet.robot_count}")
    for index, (result, controller) in enumerate(zip(fleet.results, controllers)):
        deviation = np.array([e.equidistant_deviation for e in controller.log])
        times = np.array([e.time for e in controller.log])
        clearance = obstacle_clearance_metrics(result, obstacles, 0.05)
        settled = np.abs(deviation[times >= times[-1] / 2.0])
        print(
            f"  #{index}: settled max|d_R| = {np.nanmax(settled):.4f} m, "
            f"min clearance = {clearance.minimum_clearance:.4f} m, "
            f"collided = {clearance.collided}"
        )
    separation = fleet_separation_metrics(fleet)
    print(f"separation initial:   {separation.initial_separation:.3f} m")
    print(f"separation minimum:   {separation.minimum_separation:.3f} m")
    print(f"separation final:     {separation.final_separation:.3f} m")

    neighbour_sensor = scenario.neighbour_sensor(arguments.neighbour_range)
    visibility = neighbour_visibility(fleet, obstacles, neighbour_sensor)
    seen = neighbour_visibility_metrics(visibility)
    print(f"neighbour range:      {neighbour_sensor.max_range:.1f} m")
    print(f"neighbours seen:      {seen.mean_visible_count:.2f} on average")
    print(f"time hidden by body:  {100.0 * seen.occluded_fraction:.1f} % of in-range time")
    print(f"dropout events:       {seen.dropout_events} (pair-level)")
    print("time seeing someone:  "
          + ", ".join(f"{100.0 * f:.0f} %" for f in seen.seeing_fraction))
    if not 0 <= arguments.focus < fleet.robot_count:
        raise SystemExit(f"error: --focus must be in [0, {fleet.robot_count - 1}]")
    print(f"vehicle #{arguments.focus} loses sight of each neighbour:")
    for other in range(fleet.robot_count):
        if other == arguments.focus:
            continue
        episodes = hidden_episode_durations(
            visibility, fleet.time, arguments.focus, other
        )
        hidden_time = float(episodes.sum())
        longest = float(episodes.max()) if episodes.size else 0.0
        mean = float(episodes.mean()) if episodes.size else 0.0
        print(
            f"  #{other}: hidden {100.0 * hidden_time / fleet.time[-1]:.0f} % of the run,"
            f" {episodes.size} episodes, mean {mean:.1f} s, longest {longest:.1f} s"
        )

    if arguments.figure is not None or arguments.animate is not None:
        from circumnav.analysis.animation import AnimationSettings, save_reactive_animation
        from circumnav.analysis.fleet_animation import (
            build_fleet_animation,
            save_fleet_figure,
        )

        if arguments.figure is not None:
            save_fleet_figure(
                fleet, controllers, obstacles, arguments.figure,
                visibility=visibility, focus=arguments.focus,
            )
            print(f"figure saved to:      {arguments.figure}")
        if arguments.animate is not None:
            fig, anim = build_fleet_animation(
                fleet, controllers, obstacles,
                settings=AnimationSettings(
                    fps=arguments.animation_fps,
                    real_time_factor=arguments.animation_speed,
                ),
                visibility=visibility, focus=arguments.focus,
            )
            save_reactive_animation(
                fig, anim, arguments.animate, fps=arguments.animation_fps
            )
            print(f"animation saved to:   {arguments.animate}")


if __name__ == "__main__":
    main()
