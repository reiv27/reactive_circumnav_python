import math

import numpy as np
import pytest

from circumnav.models.dubins import (
    DubinsCommand,
    DubinsLimits,
    DubinsModel,
    DubinsState,
    normalize_angle,
)


def test_normalize_angle_uses_half_open_interval():
    assert normalize_angle(math.pi) == pytest.approx(-math.pi)
    assert normalize_angle(-math.pi) == pytest.approx(-math.pi)
    assert normalize_angle(3.0 * math.pi) == pytest.approx(-math.pi)


def test_straight_motion_matches_analytic_solution():
    model = DubinsModel()
    initial = DubinsState(1.0, -2.0, math.pi / 3.0)

    result = model.propagate(
        initial,
        DubinsCommand(speed=2.0, yaw_rate=0.0),
        dt=3.0,
    )

    assert result.x == pytest.approx(1.0 + 6.0 * math.cos(math.pi / 3.0))
    assert result.y == pytest.approx(-2.0 + 6.0 * math.sin(math.pi / 3.0))
    assert result.heading == pytest.approx(math.pi / 3.0)


def test_constant_turn_returns_to_start_after_full_revolution():
    model = DubinsModel()
    initial = DubinsState(2.0, -1.0, 0.4)
    speed = 1.2
    yaw_rate = 0.3

    result = model.propagate(
        initial,
        DubinsCommand(speed=speed, yaw_rate=yaw_rate),
        dt=2.0 * math.pi / yaw_rate,
    )

    assert result.x == pytest.approx(initial.x, abs=1e-12)
    assert result.y == pytest.approx(initial.y, abs=1e-12)
    assert result.heading == pytest.approx(initial.heading, abs=1e-12)


def test_derivative_matches_dubins_kinematics():
    derivative = DubinsModel.derivative(
        DubinsState(0.0, 0.0, math.pi / 2.0),
        DubinsCommand(speed=2.0, yaw_rate=-0.4),
    )

    np.testing.assert_allclose(derivative, [0.0, 2.0, -0.4], atol=1e-12)


def test_turning_radius_is_derived_from_limits():
    limits = DubinsLimits(0.0, 2.0, 0.5)
    assert limits.minimum_turning_radius() == pytest.approx(4.0)
    assert limits.minimum_turning_radius(1.0) == pytest.approx(2.0)

