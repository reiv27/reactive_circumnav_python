"""Idealized circular visibility sensor for geometric obstacle perception.

This implements the ``CircularVisibilitySensor`` described in
README_SENSOR_MODEL.md: an omnidirectional range sensor bounded to a circle
of radius ``max_range`` around the robot. It returns only real points on
obstacle boundaries; the end of an empty ray at maximum range is never
treated as an obstacle. The excluded neighborhood around P1 is a physical
distance rather than a window of ray indices, so the result does not depend
on any angular resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from circumnav.models.dubins import DubinsState
from circumnav.models.obstacles import Obstacle
from circumnav.sensors.base import PointObservation, observation_from_world_point


@dataclass(frozen=True)
class CircularVisibilitySensor:
    """Omnidirectional range sensor bounded to a circle of radius ``max_range``."""

    max_range: float
    exclusion_distance: float
    check_occlusion: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_range) or self.max_range <= 0.0:
            raise ValueError("max_range must be positive")
        if (
            not math.isfinite(self.exclusion_distance)
            or self.exclusion_distance < 0.0
        ):
            raise ValueError("exclusion_distance must be non-negative")

    def find_primary_point(
        self,
        pose: DubinsState,
        obstacles: Sequence[Obstacle],
    ) -> PointObservation | None:
        """Find P1: the closest visible obstacle point to the robot."""

        robot_position = np.array([pose.x, pose.y])
        best: PointObservation | None = None
        best_distance = math.inf

        for obstacle in obstacles:
            candidate = obstacle.closest_point(robot_position)
            distance = float(np.linalg.norm(candidate - robot_position))
            if distance > self.max_range:
                continue
            if not self._is_visible(robot_position, candidate, obstacle, obstacles):
                continue
            if distance < best_distance:
                best_distance = distance
                best = observation_from_world_point(
                    pose, candidate, obstacle.obstacle_id
                )

        return best

    def find_secondary_point(
        self,
        pose: DubinsState,
        disk_center: NDArray[np.float64],
        primary: PointObservation,
        obstacles: Sequence[Obstacle],
    ) -> PointObservation | None:
        """Find P2: the visible point closest to ``disk_center``, excluding
        P1's neighborhood.
        """

        robot_position = np.array([pose.x, pose.y])
        best: PointObservation | None = None
        best_disk_distance = math.inf

        for obstacle in obstacles:
            candidate = obstacle.closest_point(disk_center)

            range_distance = float(np.linalg.norm(candidate - robot_position))
            if range_distance > self.max_range:
                continue

            if (
                np.linalg.norm(candidate - primary.point_world)
                <= self.exclusion_distance
            ):
                continue

            if not self._is_visible(robot_position, candidate, obstacle, obstacles):
                continue

            disk_distance = float(np.linalg.norm(candidate - disk_center))
            if disk_distance < best_disk_distance:
                best_disk_distance = disk_distance
                best = observation_from_world_point(
                    pose, candidate, obstacle.obstacle_id
                )

        return best

    def _is_visible(
        self,
        robot_position: NDArray[np.float64],
        candidate: NDArray[np.float64],
        owner: Obstacle,
        obstacles: Sequence[Obstacle],
    ) -> bool:
        """Check that nothing occludes the line of sight to ``candidate``.

        Two things can hide it: the far side of the obstacle's own body (a
        solid disk or ellipse hides its back), and any other obstacle that
        the line of sight crosses first.
        """

        if not self.check_occlusion:
            return True

        if not owner.is_boundary_visible(candidate, robot_position):
            return False

        candidate_distance = float(np.linalg.norm(candidate - robot_position))
        for obstacle in obstacles:
            if obstacle is owner:
                continue
            blocking_point = obstacle.first_intersection(robot_position, candidate)
            if blocking_point is None:
                continue
            blocking_distance = float(
                np.linalg.norm(blocking_point - robot_position)
            )
            if blocking_distance < candidate_distance - 1e-9:
                return False

        return True
