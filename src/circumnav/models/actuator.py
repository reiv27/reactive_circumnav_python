"""Actuator models that separate requested and applied control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from circumnav.models.dubins import DubinsCommand, DubinsLimits


@dataclass(frozen=True)
class ActuatorOutput:
    """Requested command and the command actually applied to the plant."""

    requested: DubinsCommand
    applied: DubinsCommand
    speed_saturated: bool
    yaw_rate_saturated: bool

    @property
    def saturated(self) -> bool:
        return self.speed_saturated or self.yaw_rate_saturated


class IdealActuator:
    """Memoryless actuator with hard speed and yaw-rate saturation."""

    def __init__(self, limits: DubinsLimits) -> None:
        self.limits = limits

    def reset(self) -> None:
        """Reset actuator state; the ideal actuator has no internal state."""

    def apply(self, command: DubinsCommand) -> ActuatorOutput:
        applied_speed = float(
            np.clip(command.speed, self.limits.min_speed, self.limits.max_speed)
        )
        applied_yaw_rate = float(
            np.clip(
                command.yaw_rate,
                -self.limits.max_yaw_rate,
                self.limits.max_yaw_rate,
            )
        )
        applied = DubinsCommand(applied_speed, applied_yaw_rate)

        return ActuatorOutput(
            requested=command,
            applied=applied,
            speed_saturated=not np.isclose(applied_speed, command.speed),
            yaw_rate_saturated=not np.isclose(
                applied_yaw_rate,
                command.yaw_rate,
            ),
        )

