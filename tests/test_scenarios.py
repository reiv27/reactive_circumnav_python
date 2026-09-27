import math

from circumnav.models.dubins import normalize_angle
from circumnav.scenarios.heading import HeadingControlScenario


def test_heading_scenario_is_reproducible_and_configurable():
    scenario = HeadingControlScenario(
        duration=6.0,
        integration_step=0.01,
        control_period=0.05,
        initial_heading=-math.pi / 2.0,
        desired_heading=math.pi / 3.0,
        proportional_gain=2.0,
    )

    first = scenario.run()
    second = scenario.run()

    assert (first.state == second.state).all()
    assert abs(normalize_angle(scenario.desired_heading - first.heading[-1])) < 0.02
