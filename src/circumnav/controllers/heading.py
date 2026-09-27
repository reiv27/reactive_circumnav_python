"""A minimal feedback controller for studying sampled-data behavior."""

from __future__ import annotations

from dataclasses import dataclass
import math

from circumnav.models.dubins import DubinsCommand, DubinsState, normalize_angle


@dataclass(frozen=True)
class HeadingControllerConfig:
    """Parameters of a proportional heading controller."""

    desired_heading: float
    forward_speed: float
    proportional_gain: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (
                self.desired_heading,
                self.forward_speed,
                self.proportional_gain,
            )
        ):
            raise ValueError("Heading controller parameters must be finite")
        if self.proportional_gain <= 0.0:
            raise ValueError("proportional_gain must be positive")


class HeadingController:
    """Proportional feedback on wrapped heading error."""

    def __init__(self, config: HeadingControllerConfig) -> None:
        self.config = config

    def reset(self, initial_state: DubinsState) -> None:
        del initial_state

    def heading_error(self, state: DubinsState) -> float:
        return normalize_angle(self.config.desired_heading - state.heading)

    def compute(self, time: float, state: DubinsState) -> DubinsCommand:
        del time
        error = self.heading_error(state)
        return DubinsCommand(
            speed=self.config.forward_speed,
            yaw_rate=self.config.proportional_gain * error,
        )

