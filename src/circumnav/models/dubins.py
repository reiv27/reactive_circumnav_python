"""Kinematic Dubins vehicle model.

The module contains only plant mathematics. It deliberately knows nothing
about sensors, controllers, scenarios, or plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def normalize_angle(angle: float) -> float:
    """Map an angle to the half-open interval [-pi, pi)."""

    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(frozen=True)
class DubinsState:
    """Planar pose of the vehicle in SI units."""

    x: float
    y: float
    heading: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.heading)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Dubins state values must be finite")

    def as_array(self) -> FloatArray:
        return np.array([self.x, self.y, self.heading], dtype=np.float64)


@dataclass(frozen=True)
class DubinsCommand:
    """Linear and angular velocity command in SI units."""

    speed: float
    yaw_rate: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.speed) or not math.isfinite(self.yaw_rate):
            raise ValueError("Dubins command values must be finite")

    def as_array(self) -> FloatArray:
        return np.array([self.speed, self.yaw_rate], dtype=np.float64)


@dataclass(frozen=True)
class DubinsLimits:
    """Hard kinematic limits applied by an actuator model."""

    min_speed: float
    max_speed: float
    max_yaw_rate: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.min_speed, self.max_speed, self.max_yaw_rate)
        ):
            raise ValueError("Dubins limits must be finite")
        if self.min_speed > self.max_speed:
            raise ValueError("min_speed must not exceed max_speed")
        if self.max_yaw_rate <= 0.0:
            raise ValueError("max_yaw_rate must be positive")

    def minimum_turning_radius(self, speed: float | None = None) -> float:
        """Return the radius at maximum yaw rate for a selected speed."""

        selected_speed = self.max_speed if speed is None else abs(speed)
        return selected_speed / self.max_yaw_rate


class DubinsModel:
    """Exact zero-order-hold propagation of Dubins kinematics."""

    _STRAIGHT_EPSILON = 1e-10

    @staticmethod
    def derivative(state: DubinsState, command: DubinsCommand) -> FloatArray:
        """Evaluate the continuous-time right-hand side f(x, u)."""

        return np.array(
            [
                command.speed * math.cos(state.heading),
                command.speed * math.sin(state.heading),
                command.yaw_rate,
            ],
            dtype=np.float64,
        )

    def propagate(
        self,
        state: DubinsState,
        command: DubinsCommand,
        dt: float,
    ) -> DubinsState:
        """Propagate the pose exactly for a command held constant over ``dt``."""

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        speed = command.speed
        yaw_rate = command.yaw_rate
        heading = state.heading

        if abs(yaw_rate) < self._STRAIGHT_EPSILON:
            next_x = state.x + speed * math.cos(heading) * dt
            next_y = state.y + speed * math.sin(heading) * dt
            next_heading = heading
        else:
            unwrapped_heading = heading + yaw_rate * dt
            radius = speed / yaw_rate
            next_x = state.x + radius * (
                math.sin(unwrapped_heading) - math.sin(heading)
            )
            next_y = state.y - radius * (
                math.cos(unwrapped_heading) - math.cos(heading)
            )
            next_heading = unwrapped_heading

        return DubinsState(
            x=next_x,
            y=next_y,
            heading=normalize_angle(next_heading),
        )

