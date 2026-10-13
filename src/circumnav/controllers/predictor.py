"""Sensing-delay compensation by exact state reconstruction.

Where :class:`DelayedController` feeds the inner controller a stale
measurement ``x(t - tau_s)`` as-is (so the hybrid C/G switching logic also
fires on outdated geometry), :class:`PredictorController` reconstructs an
estimate of ``x(t)`` before calling the inner controller, by forward-replaying
the exact Dubins ZOH kinematics through a stream of *applied* commands,
starting from the stale anchor measurement.

Two cases:

- **No ``actuator_factory`` (stage 1a).** The replay stream is this
  controller's own issued-command history, held for one ``control_period``
  each. Exact whenever the real actuator between this controller and the
  plant is memoryless and unsaturated (``actuation_delay == 0``,
  ``actuator_time_constant == 0``): the predictor's own output is then
  exactly what the plant applied, so replaying it reconstructs ``x(t)`` to
  floating-point precision -- it recomputes exactly what the simulation
  engine already did.
- **With ``actuator_factory`` (stage 1b).** The predictor also keeps a
  persistent *shadow* copy of the real actuator chain (built by calling
  ``actuator_factory()``, with the same parameters, stepped on the same
  ``integration_step`` grid). Because that chain is a pure, deterministic
  function of its own command stream (it never reads the plant state), the
  shadow and the real actuator stay in lockstep by construction, and
  replaying the *shadow-applied* stream reconstructs ``x(t)`` exactly even
  when ``actuation_delay`` or ``actuator_time_constant`` are non-zero.

**What this does not yet do.** Either way, this only undoes the *sensing*
delay's effect on the switching decision: it reconstructs where the vehicle
*is now*. It does not predict where the vehicle *will be once a new command
clears the actuation pipeline* (the "queued-input prediction" step of a full
predictor, needed to also compensate ``tau_a``/lag on their own). With
``delay_steps == 0`` this controller is a no-op regardless of
``actuator_factory``: there is nothing to reconstruct when sensing is already
current, which correctly reflects that scope boundary rather than silently
pretending to compensate something it structurally cannot yet fix. See
`controllers/AGENTS.md` for the follow-on ("stage 1c", not implemented).
"""

from __future__ import annotations

from collections import deque
import math
from typing import Callable

from circumnav.controllers.base import Controller
from circumnav.models.actuator import ActuatorOutput, Actuator
from circumnav.models.dubins import DubinsCommand, DubinsModel, DubinsState


class _PassThroughActuator:
    """No-op actuator: applied == requested, always. Stage 1a's default."""

    def reset(self) -> None:
        pass

    def apply(
        self, command: DubinsCommand, dt: float | None = None
    ) -> ActuatorOutput:
        del dt
        return ActuatorOutput(
            requested=command,
            applied=command,
            speed_saturated=False,
            yaw_rate_saturated=False,
        )


class PredictorController:
    """Reconstruct ``x(t)`` from ``x(t - tau_s)`` before calling ``inner``.

    Every attribute other than ``reset``/``compute`` is forwarded to
    ``inner``, so diagnostics (``.log``, ``.mode``, ``.switch_count``,
    ``.config``, ...) keep working exactly as for the unwrapped controller --
    same contract as :class:`DelayedController`.
    """

    def __init__(
        self,
        inner: Controller,
        delay_steps: int,
        control_period: float,
        model: DubinsModel | None = None,
        integration_step: float | None = None,
        actuator_factory: Callable[[], Actuator] | None = None,
    ) -> None:
        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        if control_period <= 0.0:
            raise ValueError("control_period must be positive")
        self.inner = inner
        self.delay_steps = delay_steps
        self.control_period = control_period
        self.model = model if model is not None else DubinsModel()
        self.integration_step = integration_step
        self.actuator_factory = actuator_factory

        if actuator_factory is not None:
            if integration_step is None or integration_step <= 0.0:
                raise ValueError(
                    "integration_step must be positive when actuator_factory "
                    "is given"
                )
            ratio = control_period / integration_step
            substeps = round(ratio)
            if substeps < 1 or not math.isclose(ratio, substeps, abs_tol=1e-9):
                raise ValueError(
                    "control_period must be an integer multiple of "
                    "integration_step"
                )
            self._substeps_per_period = substeps
            self._step_dt = integration_step
            self._shadow_actuator: Actuator = actuator_factory()
        else:
            self._substeps_per_period = 1
            self._step_dt = control_period
            self._shadow_actuator = _PassThroughActuator()

        self._state_history: deque[DubinsState] = deque()
        self._applied_history: deque[DubinsCommand] = deque(
            maxlen=delay_steps * self._substeps_per_period
        )

    def reset(self, initial_state: DubinsState) -> None:
        self.inner.reset(initial_state)
        self._state_history.clear()
        self._applied_history.clear()
        self._shadow_actuator.reset()

    def compute(self, time: float, state: DubinsState) -> DubinsCommand:
        self._state_history.append(state)
        if len(self._state_history) > self.delay_steps:
            anchor = self._state_history.popleft()
        else:
            anchor = self._state_history[0]

        estimate = anchor
        for applied in self._applied_history:
            estimate = self.model.propagate(estimate, applied, self._step_dt)

        command = self.inner.compute(time, estimate)

        for _ in range(self._substeps_per_period):
            output = self._shadow_actuator.apply(command, self._step_dt)
            self._applied_history.append(output.applied)

        return command

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)
