"""Deterministic sampled-data simulation loop."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from circumnav.controllers.base import Controller
from circumnav.models.actuator import IdealActuator
from circumnav.models.dubins import DubinsCommand, DubinsModel, DubinsState
from circumnav.simulation.result import SimulationResult


def _integer_ratio(numerator: float, denominator: float, name: str) -> int:
    ratio = numerator / denominator
    rounded = round(ratio)
    if rounded < 1 or not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} must be an integer multiple of integration_step")
    return int(rounded)


@dataclass(frozen=True)
class SimulationConfig:
    """Timing parameters for a deterministic fixed-grid experiment."""

    duration: float
    integration_step: float
    control_period: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.duration, self.integration_step, self.control_period)
        ):
            raise ValueError("Simulation timing values must be finite")
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")
        if self.integration_step <= 0.0:
            raise ValueError("integration_step must be positive")
        if self.control_period <= 0.0:
            raise ValueError("control_period must be positive")

        _integer_ratio(self.duration, self.integration_step, "duration")
        _integer_ratio(
            self.control_period,
            self.integration_step,
            "control_period",
        )


class Simulator:
    """Run a controller, actuator, and plant on a common time grid."""

    def __init__(
        self,
        model: DubinsModel,
        actuator: IdealActuator,
        config: SimulationConfig,
    ) -> None:
        self.model = model
        self.actuator = actuator
        self.config = config

    def run(
        self,
        initial_state: DubinsState,
        controller: Controller,
    ) -> SimulationResult:
        integration_step = self.config.integration_step
        step_count = _integer_ratio(
            self.config.duration,
            integration_step,
            "duration",
        )
        control_stride = _integer_ratio(
            self.config.control_period,
            integration_step,
            "control_period",
        )

        time = np.arange(step_count + 1, dtype=np.float64) * integration_step
        state_history = np.empty((step_count + 1, 3), dtype=np.float64)
        requested_history = np.empty((step_count + 1, 2), dtype=np.float64)
        applied_history = np.empty((step_count + 1, 2), dtype=np.float64)
        saturation_history = np.empty(step_count + 1, dtype=np.bool_)

        controller.reset(initial_state)
        self.actuator.reset()

        state = initial_state
        requested = DubinsCommand(0.0, 0.0)
        actuator_output = self.actuator.apply(requested)

        for step in range(step_count + 1):
            current_time = float(time[step])

            if step % control_stride == 0 and step < step_count:
                requested = controller.compute(current_time, state)
                actuator_output = self.actuator.apply(requested)

            state_history[step] = state.as_array()
            requested_history[step] = requested.as_array()
            applied_history[step] = actuator_output.applied.as_array()
            saturation_history[step] = actuator_output.saturated

            if step < step_count:
                state = self.model.propagate(
                    state,
                    actuator_output.applied,
                    integration_step,
                )

        return SimulationResult(
            time=time,
            state=state_history,
            requested_control=requested_history,
            applied_control=applied_history,
            saturated=saturation_history,
        )

