"""Numerical metrics kept independent from simulation and plotting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from circumnav.models.dubins import normalize_angle
from circumnav.models.obstacles import Obstacle
from circumnav.simulation.result import SimulationResult


@dataclass(frozen=True)
class HeadingExperimentMetrics:
    final_heading_error: float
    rms_heading_error: float
    path_length: float
    absolute_yaw_effort: float
    saturation_fraction: float


def heading_metrics(
    result: SimulationResult,
    desired_heading: float,
) -> HeadingExperimentMetrics:
    """Compute basic closed-loop metrics for a heading experiment."""

    errors = np.array(
        [normalize_angle(desired_heading - value) for value in result.heading],
        dtype=np.float64,
    )
    path_length = float(
        np.sum(np.hypot(np.diff(result.x), np.diff(result.y)))
    )

    if result.time.size > 1:
        dt = np.diff(result.time)
        yaw_effort = float(
            np.sum(np.abs(result.applied_control[:-1, 1]) * dt)
        )
    else:
        yaw_effort = 0.0

    interval_saturation = result.saturated[:-1]
    saturation_fraction = (
        float(np.mean(interval_saturation))
        if interval_saturation.size
        else 0.0
    )

    return HeadingExperimentMetrics(
        final_heading_error=float(errors[-1]),
        rms_heading_error=float(math.sqrt(np.mean(np.square(errors)))),
        path_length=path_length,
        absolute_yaw_effort=yaw_effort,
        saturation_fraction=saturation_fraction,
    )


@dataclass(frozen=True)
class ObstacleClearanceMetrics:
    """Ground-truth distance between the vehicle and the nearest obstacle."""

    minimum_clearance: float
    collided: bool


def obstacle_clearance_metrics(
    result: SimulationResult,
    obstacles: Sequence[Obstacle],
    collision_distance: float,
) -> ObstacleClearanceMetrics:
    """Compute the closest approach to any obstacle over the whole trajectory.

    This uses the true trajectory and the true obstacle geometry, independent
    of what the sensor reported, so it can catch a controller failure that a
    range-limited or occluded sensor would not have detected in time.
    """

    positions = result.state[:, :2]
    minimum_clearance = math.inf

    for position in positions:
        for obstacle in obstacles:
            closest_point = obstacle.closest_point(position)
            distance = float(np.linalg.norm(closest_point - position))
            minimum_clearance = min(minimum_clearance, distance)

    return ObstacleClearanceMetrics(
        minimum_clearance=minimum_clearance,
        collided=minimum_clearance < collision_distance,
    )
