import numpy as np
import pytest

from circumnav.analysis.metrics import obstacle_clearance_metrics
from circumnav.controllers.reactive import (
    CircumnavMode,
    ReactiveCircumnavConfig,
    ReactiveCircumnavController,
    _disk_center,
    _is_point_in_angle,
)
from circumnav.models.dubins import DubinsState
from circumnav.models.obstacles import make_obstacles
from circumnav.scenarios.reactive import ReactiveCircumnavScenario
from circumnav.sensors.base import PointObservation


def make_point_observation(point_world, obstacle_id=0):
    point_world = np.asarray(point_world, dtype=np.float64)
    return PointObservation(
        point_world=point_world,
        point_relative=point_world,
        distance=float(np.linalg.norm(point_world)),
        bearing=0.0,
        obstacle_id=obstacle_id,
    )


def test_disk_center_is_offset_from_target_by_the_requested_radius():
    center = _disk_center(
        robot_position=np.array([0.0, 0.0]),
        target=np.array([0.0, 5.0]),
        radius=3.0,
    )

    assert center == pytest.approx([0.0, 2.0])


def test_is_point_in_angle_true_inside_the_wedge():
    inside = _is_point_in_angle(
        disk_center=np.array([0.0, 0.0]),
        p1=np.array([0.0, 1.0]),
        p2=np.array([1.0, 0.0]),
        robot_position=np.array([1.0, 1.0]),
    )

    assert inside


def test_is_point_in_angle_false_outside_the_wedge():
    outside = _is_point_in_angle(
        disk_center=np.array([0.0, 0.0]),
        p1=np.array([0.0, 1.0]),
        p2=np.array([1.0, 0.0]),
        robot_position=np.array([-1.0, -1.0]),
    )

    assert not outside


def test_mode_switches_to_orbit_when_disk_center_is_closer_to_a_second_point():
    # P1 is straight ahead at (0, 1); the disk center this places 3 m behind
    # it (safety_distance=1 + turning_radius=2) lands at (0, -2), which is
    # closer to a second obstacle at (0, -4) than to P1 itself -- exactly the
    # published switching condition ||V-P2|| < ||V-P1||.
    obstacles = make_obstacles(
        [((-0.1, 1.0), (0.1, 1.0)), ((-0.1, -4.0), (0.1, -4.0))]
    )
    config = ReactiveCircumnavConfig(
        forward_speed=1.0,
        yaw_rate_magnitude=0.5,
        safety_distance=1.0,
        exclusion_distance=1.0,
        sensor_range=10.0,
    )
    controller = ReactiveCircumnavController(config, obstacles)
    state = DubinsState(0.0, 0.0, 0.0)
    controller.reset(state)

    controller.compute(0.0, state)

    assert controller.mode is CircumnavMode.ORBIT
    assert controller.switch_count == 1
    assert controller.orbit_center == pytest.approx([0.0, -2.0])


def test_update_mode_switches_back_to_approach_outside_the_wedge():
    controller = ReactiveCircumnavController(
        ReactiveCircumnavConfig(
            forward_speed=1.0, yaw_rate_magnitude=0.5, safety_distance=1.0
        ),
        obstacles=(),
    )
    controller.mode = CircumnavMode.ORBIT
    controller.orbit_center = np.array([0.0, 0.0])
    primary = make_point_observation([0.0, 1.0])
    secondary = make_point_observation([1.0, 0.0], obstacle_id=1)

    controller._update_mode(
        disk_center=np.array([5.0, 5.0]),  # unused in ORBIT mode
        primary=primary,
        secondary=secondary,
        robot_position=np.array([-1.0, -1.0]),  # outside the P1-P2 wedge
    )

    assert controller.mode is CircumnavMode.APPROACH
    assert controller.switch_count == 1


def test_update_mode_stays_in_orbit_inside_the_wedge():
    controller = ReactiveCircumnavController(
        ReactiveCircumnavConfig(
            forward_speed=1.0, yaw_rate_magnitude=0.5, safety_distance=1.0
        ),
        obstacles=(),
    )
    controller.mode = CircumnavMode.ORBIT
    controller.orbit_center = np.array([0.0, 0.0])
    primary = make_point_observation([0.0, 1.0])
    secondary = make_point_observation([1.0, 0.0], obstacle_id=1)

    controller._update_mode(
        disk_center=np.array([5.0, 5.0]),
        primary=primary,
        secondary=secondary,
        robot_position=np.array([1.0, 1.0]),  # inside the P1-P2 wedge
    )

    assert controller.mode is CircumnavMode.ORBIT
    assert controller.switch_count == 0


def test_update_mode_does_not_switch_back_without_a_secondary_point():
    # Safe default from README_SENSOR_MODEL.md: never switch on a condition
    # that cannot currently be evaluated.
    controller = ReactiveCircumnavController(
        ReactiveCircumnavConfig(
            forward_speed=1.0, yaw_rate_magnitude=0.5, safety_distance=1.0
        ),
        obstacles=(),
    )
    controller.mode = CircumnavMode.ORBIT
    controller.orbit_center = np.array([0.0, 0.0])
    primary = make_point_observation([0.0, 1.0])

    controller._update_mode(
        disk_center=np.array([5.0, 5.0]),
        primary=primary,
        secondary=None,
        robot_position=np.array([-1.0, -1.0]),
    )

    assert controller.mode is CircumnavMode.ORBIT
    assert controller.switch_count == 0


def make_scenario(**overrides):
    # A wall's equidistant curve caps its endpoints with arcs of radius d, so
    # rho_0 = d; the yaw rate here keeps R_min = 0.5 m below that.
    defaults = dict(
        duration=25.0,
        integration_step=0.01,
        control_period=0.02,
        initial_x=-8.0,
        initial_y=0.0,
        initial_heading=0.0,
        forward_speed=1.0,
        yaw_rate_magnitude=2.0,
        safety_distance=1.0,
        sensor_range=10.0,
        exclusion_distance=1.0,
        obstacles=make_obstacles([((0.0, -5.0), (0.0, 5.0))]),
    )
    defaults.update(overrides)
    return ReactiveCircumnavScenario(**defaults)


def test_vehicle_avoids_collision_while_circling_an_obstacle():
    scenario = make_scenario()

    result, _ = scenario.run()

    metrics = obstacle_clearance_metrics(
        result, scenario.build_obstacles(), collision_distance=0.02
    )

    assert not metrics.collided
    assert np.isfinite(metrics.minimum_clearance)


def test_switching_is_reproducible():
    scenario = make_scenario()

    first, first_controller = scenario.run()
    second, second_controller = scenario.run()

    assert (first.state == second.state).all()
    assert first_controller.switch_count == second_controller.switch_count


def test_no_obstacle_in_range_holds_straight_course():
    scenario = make_scenario(
        initial_x=0.0,
        obstacles=make_obstacles([((100.0, -1.0), (100.0, 1.0))]),
        sensor_range=5.0,
        duration=2.0,
    )

    result, controller = scenario.run()

    assert np.all(result.applied_control[:, 1] == 0.0)
    assert result.heading[-1] == pytest.approx(0.0)
    assert controller.switch_count == 0


def test_controller_reset_clears_mode_and_switch_count():
    obstacles = make_obstacles(
        [((-0.1, 1.0), (0.1, 1.0)), ((-0.1, -4.0), (0.1, -4.0))]
    )
    controller = ReactiveCircumnavController(
        ReactiveCircumnavConfig(
            forward_speed=1.0,
            yaw_rate_magnitude=0.5,
            safety_distance=1.0,
            exclusion_distance=1.0,
            sensor_range=10.0,
        ),
        obstacles,
    )
    state = DubinsState(0.0, 0.0, 0.0)
    controller.reset(state)
    controller.compute(0.0, state)
    assert controller.switch_count == 1  # triggers mode ORBIT, see above

    controller.reset(state)

    assert controller.mode is CircumnavMode.APPROACH
    assert controller.switch_count == 0
    assert controller.orbit_center is None
