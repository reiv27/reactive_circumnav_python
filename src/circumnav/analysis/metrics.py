"""Numerical metrics kept independent from simulation and plotting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from circumnav.models.dubins import normalize_angle
from circumnav.models.obstacles import Obstacle
from circumnav.sensors.neighbours import NeighbourSensor
from circumnav.simulation.result import FleetResult, SimulationResult


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


@dataclass(frozen=True)
class FleetSeparationMetrics:
    """Distances between vehicles; vehicles are points with no physical size."""

    initial_separation: float
    minimum_separation: float
    final_separation: float


def nearest_neighbour_distance(fleet: FleetResult) -> NDArray[np.float64]:
    """Distance from each vehicle to its nearest neighbour, ``(robots, samples)``."""

    positions = fleet.positions
    if fleet.robot_count < 2:
        return np.full(positions.shape[:2], math.inf)
    offsets = positions[:, None, :, :] - positions[None, :, :, :]
    distances = np.linalg.norm(offsets, axis=-1)
    distances[np.arange(fleet.robot_count), np.arange(fleet.robot_count)] = math.inf
    return distances.min(axis=1)


def fleet_separation_metrics(fleet: FleetResult) -> FleetSeparationMetrics:
    """Closest pair at the start, over the whole run, and at the end."""

    nearest = nearest_neighbour_distance(fleet).min(axis=0)
    return FleetSeparationMetrics(
        initial_separation=float(nearest[0]),
        minimum_separation=float(nearest.min()),
        final_separation=float(nearest[-1]),
    )


@dataclass(frozen=True)
class NeighbourVisibility:
    """Who sees whom over time, both arrays shaped ``(samples, robots, robots)``.

    ``in_range`` ignores occlusion; ``visible`` is in range *and* not hidden by
    an obstacle. Both are symmetric with a ``False`` diagonal.
    """

    in_range: NDArray[np.bool_]
    visible: NDArray[np.bool_]

    @property
    def occluded(self) -> NDArray[np.bool_]:
        """In range but hidden behind an obstacle."""

        return self.in_range & ~self.visible

    @property
    def visible_count(self) -> NDArray[np.int_]:
        """Neighbours each vehicle sees, ``(samples, robots)``."""

        return self.visible.sum(axis=2)


def neighbour_visibility(
    fleet: FleetResult,
    obstacles: Sequence[Obstacle],
    sensor: NeighbourSensor,
) -> NeighbourVisibility:
    positions = fleet.positions
    robots, samples = positions.shape[:2]
    in_range = np.zeros((samples, robots, robots), dtype=np.bool_)
    visible = np.zeros_like(in_range)
    for first in range(robots):
        for second in range(first + 1, robots):
            for sample in range(samples):
                a = positions[first, sample]
                b = positions[second, sample]
                near = sensor.in_range(a, b)
                in_range[sample, first, second] = near
                visible[sample, first, second] = near and not sensor.occluded(
                    a, b, obstacles
                )
    in_range |= in_range.transpose(0, 2, 1)
    visible |= visible.transpose(0, 2, 1)
    return NeighbourVisibility(in_range=in_range, visible=visible)


@dataclass(frozen=True)
class NeighbourVisibilityMetrics:
    #: Fraction of time each vehicle sees at least one neighbour.
    seeing_fraction: NDArray[np.float64]
    #: Time-averaged number of visible neighbours per vehicle.
    mean_visible_count: float
    #: Times a neighbour that was in range went out of sight (pair-level).
    dropout_events: int
    #: Fraction of in-range pair time spent hidden by an obstacle.
    occluded_fraction: float


def neighbour_visibility_metrics(
    visibility: NeighbourVisibility,
) -> NeighbourVisibilityMetrics:
    counts = visibility.visible_count
    upper = np.triu(np.ones(visibility.visible.shape[1:], dtype=bool), k=1)
    pair_visible = visibility.visible[:, upper]
    pair_in_range = visibility.in_range[:, upper]
    lost = pair_in_range[1:] & ~pair_visible[1:] & pair_visible[:-1]
    in_range_time = int(pair_in_range.sum())
    return NeighbourVisibilityMetrics(
        seeing_fraction=(counts > 0).mean(axis=0),
        mean_visible_count=float(counts.mean()),
        dropout_events=int(lost.sum()),
        occluded_fraction=(
            float(visibility.occluded[:, upper].sum()) / in_range_time
            if in_range_time
            else 0.0
        ),
    )


def hidden_episode_durations(
    visibility: NeighbourVisibility,
    time: NDArray[np.float64],
    observer: int,
    neighbour: int,
) -> NDArray[np.float64]:
    """Lengths in seconds of the stretches when ``neighbour`` is not visible.

    A stretch counts whether the neighbour is out of range or behind an
    obstacle. Stretches cut off by the start or end of the run are included at
    their truncated length.
    """

    hidden = ~visibility.visible[:, observer, neighbour]
    step = float(time[1] - time[0]) if time.size > 1 else 0.0
    durations: list[float] = []
    run = 0
    for flag in hidden:
        if flag:
            run += 1
        elif run:
            durations.append(run * step)
            run = 0
    if run:
        durations.append(run * step)
    return np.array(durations, dtype=np.float64)
