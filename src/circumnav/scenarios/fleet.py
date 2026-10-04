"""Several independent vehicles circumnavigating one obstacle.

The vehicles do not see, hear or avoid each other: each runs its own
:class:`ReactiveCircumnavScenario` from a different point of the same
equidistant curve. Because there is no coupling, simulating them one after
another is exactly equivalent to simulating them together. Any interaction
(vehicles as obstacles, communication) needs a synchronous multi-vehicle loop
that reads all states at one instant before any controller runs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from numpy.typing import NDArray

from circumnav.controllers.reactive import ReactiveCircumnavController
from circumnav.models.dubins import DubinsState
from circumnav.models.obstacles import Obstacle
from circumnav.scenarios.reactive import ReactiveCircumnavScenario
from circumnav.simulation.result import FleetResult

_CURVE_SAMPLES = 4000


def equidistant_start_poses(
    obstacle: Obstacle,
    rho_0: float,
    count: int,
    phase: float = 0.0,
) -> tuple[DubinsState, ...]:
    """Place ``count`` vehicles on the equidistant curve, evenly in arc length.

    Each pose sits on the curve at distance ``rho_0`` from the obstacle and
    points along its tangent, counter-clockwise: the law keeps the obstacle on
    the vehicle's left, and the opposite sense is not a stable circulation.
    ``phase`` shifts every vehicle along the curve, as a fraction of its
    length.
    """

    if count < 1:
        raise ValueError("count must be at least 1")
    curve = obstacle.equidistant_polyline(rho_0, _CURVE_SAMPLES)
    if not np.allclose(curve[0], curve[-1], atol=1e-9):
        raise ValueError("the equidistant curve of the obstacle is not closed")

    twice_area = float(
        np.sum(curve[:-1, 0] * curve[1:, 1] - curve[1:, 0] * curve[:-1, 1])
    )
    if twice_area < 0.0:
        curve = curve[::-1]

    segments = np.diff(curve, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(arc[-1])

    poses = []
    for index in range(count):
        position_on_curve = ((phase + index / count) % 1.0) * total
        segment = min(
            int(np.searchsorted(arc, position_on_curve, side="right")) - 1,
            len(lengths) - 1,
        )
        fraction = (position_on_curve - arc[segment]) / lengths[segment]
        point = curve[segment] + fraction * segments[segment]
        direction = segments[segment]
        poses.append(
            DubinsState(
                x=float(point[0]),
                y=float(point[1]),
                heading=math.atan2(float(direction[1]), float(direction[0])),
            )
        )
    return tuple(poses)


@dataclass(frozen=True)
class FleetScenario:
    """``robot_count`` identical vehicles started along one equidistant curve.

    ``base`` carries every shared parameter; only the initial pose differs
    between vehicles. It must contain exactly one obstacle, and all delay
    channels must be off.
    """

    base: ReactiveCircumnavScenario
    robot_count: int = 4
    phase: float = 0.0

    def __post_init__(self) -> None:
        if self.robot_count < 1:
            raise ValueError("robot_count must be at least 1")
        if len(self.base.obstacles) != 1:
            raise ValueError("a fleet scenario needs exactly one obstacle")
        if (
            self.base.sensing_delay != 0.0
            or self.base.actuation_delay != 0.0
            or self.base.actuator_time_constant != 0.0
        ):
            raise ValueError("fleet runs do not support delays yet")

    @property
    def start_poses(self) -> tuple[DubinsState, ...]:
        return equidistant_start_poses(
            self.base.obstacles[0],
            self.base.rho_0,
            self.robot_count,
            self.phase,
        )

    def vehicle_scenarios(self) -> tuple[ReactiveCircumnavScenario, ...]:
        return tuple(
            replace(
                self.base,
                initial_x=pose.x,
                initial_y=pose.y,
                initial_heading=pose.heading,
            )
            for pose in self.start_poses
        )

    def build_obstacles(self) -> tuple[Obstacle, ...]:
        return self.base.build_obstacles()

    def run(self) -> tuple[FleetResult, tuple[ReactiveCircumnavController, ...]]:
        """Run every vehicle and return their results and controllers."""

        runs = [scenario.run() for scenario in self.vehicle_scenarios()]
        results = tuple(result for result, _ in runs)
        controllers = tuple(controller for _, controller in runs)
        return FleetResult(results), controllers  # type: ignore[arg-type]
