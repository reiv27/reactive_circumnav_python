import math

import numpy as np
import pytest

from circumnav.analysis.metrics import heading_metrics
from circumnav.controllers.heading import HeadingController, HeadingControllerConfig
from circumnav.controllers.open_loop import ConstantController
from circumnav.models.actuator import IdealActuator
from circumnav.models.dubins import (
    DubinsCommand,
    DubinsLimits,
    DubinsModel,
    DubinsState,
    normalize_angle,
)
from circumnav.simulation.engine import SimulationConfig, Simulator


def make_simulator(config):
    return Simulator(
        model=DubinsModel(),
        actuator=IdealActuator(DubinsLimits(0.0, 1.0, 0.8)),
        config=config,
    )


def test_open_loop_simulation_has_aligned_histories():
    simulator = make_simulator(
        SimulationConfig(duration=1.0, integration_step=0.01, control_period=0.1)
    )

    result = simulator.run(
        DubinsState(0.0, 0.0, 0.0),
        ConstantController(DubinsCommand(0.5, 0.0)),
    )

    assert result.time.shape == (101,)
    assert result.state.shape == (101, 3)
    assert result.requested_control.shape == (101, 2)
    assert result.x[-1] == pytest.approx(0.5)
    assert result.y[-1] == pytest.approx(0.0)


def test_heading_controller_reduces_wrapped_error():
    desired_heading = math.pi / 2.0
    controller = HeadingController(
        HeadingControllerConfig(
            desired_heading=desired_heading,
            forward_speed=0.6,
            proportional_gain=1.5,
        )
    )
    simulator = make_simulator(
        SimulationConfig(duration=8.0, integration_step=0.01, control_period=0.05)
    )

    result = simulator.run(DubinsState(0.0, 0.0, 0.0), controller)
    initial_error = abs(normalize_angle(desired_heading - result.heading[0]))
    final_error = abs(normalize_angle(desired_heading - result.heading[-1]))

    assert final_error < 1e-4
    assert final_error < initial_error
    assert np.max(np.abs(result.applied_control[:, 1])) <= 0.8

    metrics = heading_metrics(result, desired_heading)
    # The metric uses chord lengths between logged poses, so a sampled curve
    # is microscopically shorter than the analytic arc length speed * time.
    assert metrics.path_length == pytest.approx(4.8, abs=1e-5)
    assert 0.0 < metrics.saturation_fraction < 1.0


@pytest.mark.parametrize(
    "config",
    [
        dict(duration=1.05, integration_step=0.1, control_period=0.1),
        dict(duration=1.0, integration_step=0.1, control_period=0.15),
    ],
)
def test_simulation_rejects_incompatible_time_grid(config):
    with pytest.raises(ValueError, match="integer multiple"):
        SimulationConfig(**config)
