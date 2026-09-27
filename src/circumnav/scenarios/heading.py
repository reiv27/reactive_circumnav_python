"""Configuration and assembly of the baseline heading experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math

from circumnav.controllers.heading import HeadingController, HeadingControllerConfig
from circumnav.models.actuator import IdealActuator
from circumnav.models.dubins import DubinsLimits, DubinsModel, DubinsState
from circumnav.simulation.engine import SimulationConfig, Simulator
from circumnav.simulation.result import SimulationResult


@dataclass(frozen=True)
class HeadingControlScenario:
    """All parameters required to reproduce a heading-control experiment."""

    duration: float = 8.0
    integration_step: float = 0.01
    control_period: float = 0.05
    initial_x: float = 0.0
    initial_y: float = 0.0
    initial_heading: float = 0.0
    desired_heading: float = math.pi / 2.0
    forward_speed: float = 0.6
    proportional_gain: float = 1.5
    min_speed: float = 0.0
    max_speed: float = 1.0
    max_yaw_rate: float = 0.8

    def run(self) -> SimulationResult:
        """Assemble and execute the scenario without global mutable state."""

        limits = DubinsLimits(
            min_speed=self.min_speed,
            max_speed=self.max_speed,
            max_yaw_rate=self.max_yaw_rate,
        )
        controller = HeadingController(
            HeadingControllerConfig(
                desired_heading=self.desired_heading,
                forward_speed=self.forward_speed,
                proportional_gain=self.proportional_gain,
            )
        )
        simulator = Simulator(
            model=DubinsModel(),
            actuator=IdealActuator(limits),
            config=SimulationConfig(
                duration=self.duration,
                integration_step=self.integration_step,
                control_period=self.control_period,
            ),
        )
        return simulator.run(
            initial_state=DubinsState(
                x=self.initial_x,
                y=self.initial_y,
                heading=self.initial_heading,
            ),
            controller=controller,
        )

