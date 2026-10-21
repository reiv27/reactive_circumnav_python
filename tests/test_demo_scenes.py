from dataclasses import replace

import numpy as np

from circumnav.analysis.metrics import obstacle_clearance_metrics
from circumnav.controllers.reactive import CircumnavMode
from circumnav.examples.reactive_circumnav import OBSTACLE_SHAPES, build_obstacle
from circumnav.models.obstacles import EllipseObstacle
from circumnav.scenarios.reactive import ReactiveCircumnavScenario


def test_two_ellipses_are_perpendicular_with_a_seven_metre_gap():
    assert "two_ellipses" in OBSTACLE_SHAPES

    first, second = build_obstacle("two_ellipses", rho_0=3.0, turning_radius=2.0)

    assert isinstance(first, EllipseObstacle)
    assert isinstance(second, EllipseObstacle)
    assert np.isclose(abs(second.angle - first.angle), np.pi / 2.0)
    # The first ellipse ends at x = 6 and the second begins at x = 16 - 3 = 13.
    tip = first.closest_point(np.array([13.0, 0.0]))
    assert np.allclose(tip, [6.0, 0.0], atol=1e-6)
    assert np.isclose(np.linalg.norm(second.closest_point(tip) - tip), 7.0, atol=1e-6)


def test_two_ellipses_drives_the_law_through_mode_g_without_collision():
    base = ReactiveCircumnavScenario(duration=15.0)
    scenario = replace(
        base,
        obstacles=build_obstacle("two_ellipses", base.rho_0, base.turning_radius),
    )

    result, controller = scenario.run()

    assert any(entry.mode is CircumnavMode.ORBIT for entry in controller.log)
    metrics = obstacle_clearance_metrics(
        result, scenario.build_obstacles(), collision_distance=0.05
    )
    assert not metrics.collided
