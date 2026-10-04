import math

import numpy as np
import pytest

from circumnav.models.actuator import (
    DelayedActuator,
    IdealActuator,
    LagActuator,
)
from circumnav.models.dubins import DubinsCommand, DubinsLimits
from circumnav.scenarios.reactive import ReactiveCircumnavScenario

LIMITS = DubinsLimits(min_speed=0.0, max_speed=2.0, max_yaw_rate=2.0)
CRUISE = DubinsCommand(speed=1.0, yaw_rate=0.0)
DT = 0.01


def test_delay_shifts_the_applied_command_by_the_delay():
    actuator = DelayedActuator(IdealActuator(LIMITS), 0.03, DT, CRUISE)
    requests = [DubinsCommand(1.0, 0.1 * k) for k in range(1, 9)]

    applied = [actuator.apply(r, DT).applied.yaw_rate for r in requests]

    assert applied[:3] == [0.0, 0.0, 0.0]
    assert applied[3:] == pytest.approx([r.yaw_rate for r in requests[:5]])


def test_delay_reports_the_undelayed_request():
    actuator = DelayedActuator(IdealActuator(LIMITS), 0.02, DT, CRUISE)
    request = DubinsCommand(1.0, 5.0)

    output = actuator.apply(request, DT)

    assert output.requested == request


def test_delay_must_be_a_multiple_of_the_integration_step():
    with pytest.raises(ValueError, match="integer multiple"):
        DelayedActuator(IdealActuator(LIMITS), 0.015, DT, CRUISE)


def test_zero_delay_is_a_pass_through():
    actuator = DelayedActuator(IdealActuator(LIMITS), 0.0, DT, CRUISE)

    assert actuator.apply(DubinsCommand(1.0, 0.7), DT).applied.yaw_rate == 0.7


def test_delay_reset_refills_the_line():
    actuator = DelayedActuator(IdealActuator(LIMITS), 0.02, DT, CRUISE)
    for _ in range(5):
        actuator.apply(DubinsCommand(1.0, 1.0), DT)

    actuator.reset()

    assert actuator.apply(DubinsCommand(1.0, 1.0), DT).applied.yaw_rate == 0.0


def test_lag_state_follows_the_exact_step_response():
    tau = 0.2
    actuator = LagActuator(IdealActuator(LIMITS), tau, CRUISE)
    step_count = 50

    for _ in range(step_count):
        actuator.apply(DubinsCommand(1.0, 1.0), DT)

    expected = 1.0 - math.exp(-step_count * DT / tau)
    assert actuator._state[1] == pytest.approx(expected, rel=1e-12)


def test_lag_output_is_the_interval_average():
    tau = 0.2
    actuator = LagActuator(IdealActuator(LIMITS), tau, CRUISE)

    applied = [
        actuator.apply(DubinsCommand(1.0, 1.0), DT).applied.yaw_rate
        for _ in range(20)
    ]

    # The integral of the applied rate equals the integral of the exact
    # response: t - tau * (1 - exp(-t / tau)).
    t = 20 * DT
    assert sum(applied) * DT == pytest.approx(t - tau * (1.0 - math.exp(-t / tau)))


def test_lag_starts_from_the_initial_command():
    actuator = LagActuator(IdealActuator(LIMITS), 0.5, CRUISE)

    first = actuator.apply(DubinsCommand(1.0, 0.0), DT).applied

    assert first.speed == pytest.approx(1.0)
    assert first.yaw_rate == pytest.approx(0.0)


def test_lag_with_zero_time_constant_is_a_pass_through():
    actuator = LagActuator(IdealActuator(LIMITS), 0.0, CRUISE)

    assert actuator.apply(DubinsCommand(1.0, 0.7)).applied.yaw_rate == 0.7


def test_lag_rejects_negative_time_constant():
    with pytest.raises(ValueError, match="non-negative"):
        LagActuator(IdealActuator(LIMITS), -0.1, CRUISE)


def test_lag_is_saturated_after_filtering():
    actuator = LagActuator(IdealActuator(LIMITS), 0.05, CRUISE)

    outputs = [actuator.apply(DubinsCommand(1.0, 50.0), DT) for _ in range(200)]

    assert outputs[-1].applied.yaw_rate == pytest.approx(2.0)
    assert outputs[-1].yaw_rate_saturated
    assert outputs[-1].requested.yaw_rate == 50.0


def test_scenario_defaults_build_the_ideal_actuator():
    assert isinstance(ReactiveCircumnavScenario().build_actuator(), IdealActuator)


def test_scenario_stacks_delay_over_lag():
    scenario = ReactiveCircumnavScenario(
        actuation_delay=0.05, actuator_time_constant=0.1
    )

    actuator = scenario.build_actuator()

    assert isinstance(actuator, DelayedActuator)
    assert isinstance(actuator.inner, LagActuator)


def test_actuation_delay_shifts_the_applied_control_in_a_run():
    delay = 0.1
    scenario = ReactiveCircumnavScenario(duration=4.0, actuation_delay=delay)

    result, _ = scenario.run()

    lag_steps = round(delay / scenario.integration_step)
    requested = result.requested_control[:, 1]
    applied = result.applied_control[:, 1]
    assert np.allclose(applied[lag_steps:], np.clip(requested[:-lag_steps], -2, 2))
