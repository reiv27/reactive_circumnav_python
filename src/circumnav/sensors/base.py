"""Common measurement type and interface for exteroceptive sensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from circumnav.models.dubins import DubinsState, normalize_angle
from circumnav.models.obstacles import Obstacle


@dataclass(frozen=True)
class PointObservation:
    """A single visible obstacle-boundary point, in world and body frames."""

    point_world: NDArray[np.float64]
    point_relative: NDArray[np.float64]
    distance: float
    bearing: float
    obstacle_id: int


def to_robot_frame(
    pose: DubinsState,
    point_world: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Express a world point relative to the robot body frame."""

    delta_world = point_world - np.array([pose.x, pose.y])
    heading = pose.heading
    world_to_robot = np.array(
        [
            [np.cos(heading), np.sin(heading)],
            [-np.sin(heading), np.cos(heading)],
        ]
    )
    return world_to_robot @ delta_world


def observation_from_world_point(
    pose: DubinsState,
    point_world: NDArray[np.float64],
    obstacle_id: int,
) -> PointObservation:
    """Build a :class:`PointObservation` for an in-range, visible point."""

    point_relative = to_robot_frame(pose, point_world)
    distance = float(np.linalg.norm(point_relative))
    bearing = normalize_angle(
        float(np.arctan2(point_relative[1], point_relative[0]))
    )
    return PointObservation(
        point_world=point_world,
        point_relative=point_relative,
        distance=distance,
        bearing=bearing,
        obstacle_id=obstacle_id,
    )


class Sensor(Protocol):
    """A range sensor that reports visible obstacle-boundary points.

    A sensor must explicitly report the absence of a measurement instead of
    inventing a point at maximum range: the end of an empty ray is not an
    obstacle.
    """

    def find_primary_point(
        self,
        pose: DubinsState,
        obstacles: Sequence[Obstacle],
    ) -> PointObservation | None:
        """Return P1, the closest visible obstacle point to the robot, or ``None``."""

    def find_secondary_point(
        self,
        pose: DubinsState,
        disk_center: NDArray[np.float64],
        primary: PointObservation,
        obstacles: Sequence[Obstacle],
    ) -> PointObservation | None:
        """Return P2: the visible point closest to ``disk_center`` and outside
        P1's neighborhood, or ``None``.
        """
