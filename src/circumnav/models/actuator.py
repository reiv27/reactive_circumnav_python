"""Actuator models that separate requested and applied control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from typing import Protocol

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


class Actuator(Protocol):
    """Anything that turns the held controller request into an applied command.

    The simulator calls :meth:`apply` once per *integration* step with the
    request currently held by the zero-order hold, so a stateful actuator
    advances by exactly ``dt`` per call.
    """

    def reset(self) -> None: ...

    def apply(
        self, command: DubinsCommand, dt: float | None = None
    ) -> ActuatorOutput: ...


class IdealActuator:
    """Memoryless actuator with hard speed and yaw-rate saturation."""

    def __init__(self, limits: DubinsLimits) -> None:
        self.limits = limits

    def reset(self) -> None:
        """Reset actuator state; the ideal actuator has no internal state."""

    def apply(
        self, command: DubinsCommand, dt: float | None = None
    ) -> ActuatorOutput:
        del dt
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



class DelayedActuator:
    """Pure transport delay ``tau_a`` between request and application.

    The command applied at time ``t`` is the request that was held at
    ``t - tau_a``. The controller only ever samples ``x(t_j)`` and the
    command reaches the plant ``tau_a`` later, which is exactly the signal
    path of a computation delay ``tau_c``; the two cannot be told apart in
    this loop, so one block covers ``tau_c + tau_a``.

    Until the first request has travelled through the line the plant sees
    ``initial_command``.
    """

    def __init__(
        self,
        inner: Actuator,
        delay: float,
        integration_step: float,
        initial_command: DubinsCommand,
    ) -> None:
        if delay < 0.0:
            raise ValueError("delay must be non-negative")
        ratio = delay / integration_step
        steps = round(ratio)
        if not math.isclose(ratio, steps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("delay must be an integer multiple of integration_step")
        self.inner = inner
        self.delay = delay
        self.delay_steps = int(steps)
        self.initial_command = initial_command
        self._line: deque[DubinsCommand] = deque()
        self.reset()

    def reset(self) -> None:
        self.inner.reset()
        self._line = deque([self.initial_command] * self.delay_steps)

    def apply(
        self, command: DubinsCommand, dt: float | None = None
    ) -> ActuatorOutput:
        self._line.append(command)
        arrived = self._line.popleft()
        return replace(self.inner.apply(arrived, dt), requested=command)


class LagActuator:
    """First-order lag ``tau * y' = u - y`` on both command channels.

    For a request ``u`` held over one integration step the lag has the exact
    solution ``y(t) = u + (y0 - u) * exp(-t / tau)``. The state is advanced
    with the exact factor ``alpha = 1 - exp(-dt / tau)``; the value handed to
    the plant is the exact *average* of ``y`` over the step, so the heading
    increment is exact. Unlike a delay, the lag has memory, so the result
    depends on ``integration_step`` (position error of order ``dt``).

    ``time_constant = 0`` is a pass-through.
    """

    def __init__(
        self,
        inner: Actuator,
        time_constant: float,
        initial_command: DubinsCommand,
    ) -> None:
        if time_constant < 0.0:
            raise ValueError("time_constant must be non-negative")
        self.inner = inner
        self.time_constant = time_constant
        self.initial_command = initial_command
        self._state = initial_command.as_array()

    def reset(self) -> None:
        self.inner.reset()
        self._state = self.initial_command.as_array()

    def apply(
        self, command: DubinsCommand, dt: float | None = None
    ) -> ActuatorOutput:
        if self.time_constant == 0.0:
            return self.inner.apply(command, dt)
        if dt is None or dt <= 0.0:
            raise ValueError("LagActuator needs a positive dt")
        target = command.as_array()
        alpha = -math.expm1(-dt / self.time_constant)
        mean = target + (self._state - target) * alpha * self.time_constant / dt
        self._state = self._state + alpha * (target - self._state)
        output = self.inner.apply(DubinsCommand(float(mean[0]), float(mean[1])), dt)
        return replace(output, requested=command)
