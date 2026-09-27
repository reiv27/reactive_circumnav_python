import pytest

from circumnav.models.actuator import IdealActuator
from circumnav.models.dubins import DubinsCommand, DubinsLimits


def test_ideal_actuator_applies_hard_saturation():
    actuator = IdealActuator(
        DubinsLimits(min_speed=0.1, max_speed=1.0, max_yaw_rate=0.5)
    )

    output = actuator.apply(DubinsCommand(speed=2.0, yaw_rate=-0.8))

    assert output.applied.speed == pytest.approx(1.0)
    assert output.applied.yaw_rate == pytest.approx(-0.5)
    assert output.speed_saturated
    assert output.yaw_rate_saturated
    assert output.saturated


def test_ideal_actuator_preserves_admissible_command():
    actuator = IdealActuator(
        DubinsLimits(min_speed=0.0, max_speed=1.0, max_yaw_rate=0.5)
    )
    command = DubinsCommand(speed=0.4, yaw_rate=0.2)

    output = actuator.apply(command)

    assert output.applied == command
    assert not output.saturated

