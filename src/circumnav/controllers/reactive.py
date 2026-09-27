"""Reactive circular-obstacle-avoidance controller.

This is a direct port of the published two-mode circumnavigation law: mode
``APPROACH`` (formerly ``C``) drives the vehicle toward the nearest visible
obstacle point, and mode ``ORBIT`` (formerly ``G``) holds a fixed-radius
orbit around a phantom disk center. The switching conditions and the
sliding-mode-like control formulas are unchanged from the reference
implementation (the ``python_sim`` prototype, removed from the tree and kept
in git history at commit 109a779). Only the perception model that supplies
P1/P2 is different: a range-limited ``CircularVisibilitySensor``
(README_SENSOR_MODEL.md) replaces the discretized 360-ray LiDAR scan, and the
neighborhood excluded around P1 is now a physical distance instead of a
window of ray indices.

The controller receives the vehicle's own pose (position and heading --
equivalent to onboard odometry) through the standard ``Controller``
interface. It never reads obstacle geometry directly; obstacle points reach
it exclusively through the sensor's P1/P2 measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import enum
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from circumnav.models.dubins import DubinsCommand, DubinsState
from circumnav.models.obstacles import Obstacle
from circumnav.sensors.base import PointObservation
from circumnav.sensors.circular import CircularVisibilitySensor


class CircumnavMode(enum.Enum):
    """The two modes of the reactive law (``C`` and ``G`` in the reference)."""

    APPROACH = "C"
    ORBIT = "G"


@dataclass(frozen=True)
class ReactiveLogEntry:
    """One controller update, kept for post-hoc diagnostics and animation.

    This is deliberately not part of :class:`SimulationResult`: it lives at
    the controller's own update rate (which may be coarser than the plant's
    integration step) and carries perception-specific quantities that a
    different controller would not have. See README_CONTROL_SIMULATOR.md on
    keeping visualization built from logs, separate from the dynamics.
    """

    time: float
    mode: CircumnavMode
    #: rho(t): distance to the nearest visible obstacle point, or NaN.
    obstacle_range: float
    #: d_R(t): the regulated variable -- the distance to the equidistant curve
    #: the law is currently tracking. In mode C that curve is the true
    #: equidistant, so d_R = rho - rho_0; in mode G it is the curve's local
    #: approximation, the circle of radius R_min about the frozen V, so
    #: d_R = R_min - ||r - V||. This is what the relay switches on.
    equidistant_deviation: float
    range_rate: float
    yaw_rate_command: float
    speed_command: float
    target_point: NDArray[np.float64] | None
    primary_point: NDArray[np.float64] | None
    secondary_point: NDArray[np.float64] | None
    disk_center: NDArray[np.float64] | None
    #: The disk centre frozen on entering mode G (``v_A``); ``None`` in mode C.
    orbit_center: NDArray[np.float64] | None


@dataclass(frozen=True)
class ReactiveCircumnavConfig:
    """Parameters of the reactive circumnavigation law and its sensor."""

    forward_speed: float
    yaw_rate_magnitude: float
    safety_distance: float
    dead_zone: float = 0.1
    dead_zone_gain: float = 0.025
    exclusion_distance: float = 1.0
    sensor_range: float = 30.0
    check_occlusion: bool = True

    def __post_init__(self) -> None:
        values = (
            self.forward_speed,
            self.yaw_rate_magnitude,
            self.safety_distance,
            self.dead_zone,
            self.dead_zone_gain,
            self.exclusion_distance,
            self.sensor_range,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Reactive circumnavigation parameters must be finite")
        if self.forward_speed <= 0.0:
            raise ValueError("forward_speed must be positive")
        if self.yaw_rate_magnitude <= 0.0:
            raise ValueError("yaw_rate_magnitude must be positive")
        if self.safety_distance <= 0.0:
            raise ValueError("safety_distance must be positive")
        if self.sensor_range <= 0.0:
            raise ValueError("sensor_range must be positive")

    @property
    def turning_radius(self) -> float:
        """Radius of the constant-yaw-rate circle at ``forward_speed``."""

        return self.forward_speed / self.yaw_rate_magnitude


def _heading_vector(state: DubinsState) -> NDArray[np.float64]:
    return np.array([math.cos(state.heading), math.sin(state.heading)])


def _disk_center(
    robot_position: NDArray[np.float64],
    target: NDArray[np.float64],
    radius: float,
) -> NDArray[np.float64]:
    """Project a point away from ``target`` toward the robot by ``radius``."""

    direction = robot_position - target
    unit_direction = direction / np.linalg.norm(direction)
    return target + unit_direction * radius


def _is_point_in_angle(
    disk_center: NDArray[np.float64],
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    robot_position: NDArray[np.float64],
) -> bool:
    """Return whether the robot lies within the angle P1-disk_center-P2."""

    to_p1 = (p1 - disk_center) / np.linalg.norm(p1 - disk_center)
    to_p2 = (p2 - disk_center) / np.linalg.norm(p2 - disk_center)
    to_robot = (robot_position - disk_center) / np.linalg.norm(
        robot_position - disk_center
    )

    span = math.acos(float(np.clip(np.dot(to_p1, to_p2), -1.0, 1.0)))
    angle_to_p1 = math.acos(float(np.clip(np.dot(to_robot, to_p1), -1.0, 1.0)))
    angle_to_p2 = math.acos(float(np.clip(np.dot(to_robot, to_p2), -1.0, 1.0)))
    return angle_to_p1 <= span and angle_to_p2 <= span


class ReactiveCircumnavController:
    """State-feedback wrapper around the two-mode circumnavigation law."""

    def __init__(
        self,
        config: ReactiveCircumnavConfig,
        obstacles: Sequence[Obstacle],
    ) -> None:
        self.config = config
        self.obstacles = obstacles
        self.sensor = CircularVisibilitySensor(
            max_range=config.sensor_range,
            exclusion_distance=config.exclusion_distance,
            check_occlusion=config.check_occlusion,
        )
        self.mode = CircumnavMode.APPROACH
        self.orbit_center: NDArray[np.float64] | None = None
        self.switch_count = 0
        self.log: list[ReactiveLogEntry] = []

    def reset(self, initial_state: DubinsState) -> None:
        del initial_state
        self.mode = CircumnavMode.APPROACH
        self.orbit_center = None
        self.switch_count = 0
        self.log = []

    def compute(self, time: float, state: DubinsState) -> DubinsCommand:
        robot_position = np.array([state.x, state.y])
        heading_vector = _heading_vector(state)

        primary = self.sensor.find_primary_point(state, self.obstacles)
        if primary is None:
            # Nothing within sensor range: hold course.
            command = DubinsCommand(speed=self.config.forward_speed, yaw_rate=0.0)
            self.log.append(
                ReactiveLogEntry(
                    time=time,
                    mode=self.mode,
                    obstacle_range=math.nan,
                    equidistant_deviation=math.nan,
                    range_rate=math.nan,
                    yaw_rate_command=command.yaw_rate,
                    speed_command=command.speed,
                    target_point=None,
                    primary_point=None,
                    secondary_point=None,
                    disk_center=None,
                    orbit_center=self.orbit_center
                    if self.mode is CircumnavMode.ORBIT
                    else None,
                )
            )
            return command

        disk_center = _disk_center(
            robot_position,
            primary.point_world,
            self.config.safety_distance + self.config.turning_radius,
        )
        secondary = self.sensor.find_secondary_point(
            state, disk_center, primary, self.obstacles
        )

        self._update_mode(disk_center, primary, secondary, robot_position)

        if self.mode is CircumnavMode.APPROACH:
            # The tracked curve is the true equidistant of the obstacle.
            target = primary.point_world
            equidistant_deviation = primary.distance - self.config.safety_distance
        else:
            # The tracked curve is its local approximation: the circle of
            # radius R_min about the disk centre frozen on entering mode G.
            assert self.orbit_center is not None
            target = self.orbit_center
            equidistant_deviation = self.config.turning_radius - float(
                np.linalg.norm(robot_position - target)
            )

        range_rate = self._range_rate(robot_position, target, heading_vector)
        dead_zone_term = self.config.dead_zone_gain * float(
            np.clip(
                equidistant_deviation,
                -self.config.dead_zone,
                self.config.dead_zone,
            )
        )
        switching_sign = float(np.sign(range_rate + dead_zone_term))

        command = DubinsCommand(
            speed=self.config.forward_speed,
            yaw_rate=self.config.yaw_rate_magnitude * switching_sign,
        )
        self.log.append(
            ReactiveLogEntry(
                time=time,
                mode=self.mode,
                obstacle_range=primary.distance,
                equidistant_deviation=equidistant_deviation,
                range_rate=range_rate,
                yaw_rate_command=command.yaw_rate,
                speed_command=command.speed,
                target_point=target,
                primary_point=primary.point_world,
                secondary_point=(
                    secondary.point_world if secondary is not None else None
                ),
                disk_center=disk_center,
                orbit_center=self.orbit_center
                if self.mode is CircumnavMode.ORBIT
                else None,
            )
        )
        return command

    def _update_mode(
        self,
        disk_center: NDArray[np.float64],
        primary: PointObservation,
        secondary: PointObservation | None,
        robot_position: NDArray[np.float64],
    ) -> None:
        if self.mode is CircumnavMode.APPROACH:
            if secondary is not None and np.linalg.norm(
                disk_center - secondary.point_world
            ) < np.linalg.norm(disk_center - primary.point_world):
                self.mode = CircumnavMode.ORBIT
                self.orbit_center = disk_center
                self.switch_count += 1
        else:
            assert self.orbit_center is not None
            # Safe default when P2 cannot be found: do not switch on a
            # condition that cannot currently be evaluated.
            if secondary is not None and not _is_point_in_angle(
                self.orbit_center,
                primary.point_world,
                secondary.point_world,
                robot_position,
            ):
                self.mode = CircumnavMode.APPROACH
                self.switch_count += 1

    def _range_rate(
        self,
        robot_position: NDArray[np.float64],
        target: NDArray[np.float64],
        heading_vector: NDArray[np.float64],
    ) -> float:
        if self.mode is CircumnavMode.APPROACH:
            line_of_sight = (robot_position - target) / np.linalg.norm(
                robot_position - target
            )
        else:
            line_of_sight = (target - robot_position) / np.linalg.norm(
                target - robot_position
            )
        return self.config.forward_speed * float(line_of_sight @ heading_vector)
