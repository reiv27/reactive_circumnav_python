import numpy as np
import pytest

from circumnav.analysis.metrics import (
    neighbour_visibility,
    neighbour_visibility_metrics,
)
from circumnav.models.obstacles import CircleObstacle
from circumnav.scenarios.fleet import FleetScenario
from circumnav.scenarios.reactive import ReactiveCircumnavScenario
from circumnav.sensors.neighbours import NeighbourSensor
from circumnav.simulation.result import FleetResult, SimulationResult

BODY = CircleObstacle(center=np.array([0.0, 0.0]), radius=1.0, obstacle_id=0)


def point(x, y):
    return np.array([x, y], dtype=float)


def test_sees_a_clear_neighbour_in_range():
    sensor = NeighbourSensor(max_range=10.0)

    assert sensor.sees(point(-5, 3), point(5, 3), (BODY,))


def test_a_neighbour_beyond_range_is_not_seen():
    sensor = NeighbourSensor(max_range=9.0)

    assert not sensor.sees(point(-5, 3), point(5, 3), (BODY,))
    assert not sensor.in_range(point(-5, 3), point(5, 3))


def test_the_obstacle_body_hides_a_neighbour():
    sensor = NeighbourSensor(max_range=20.0)

    assert sensor.occluded(point(-5, 0), point(5, 0), (BODY,))
    assert not sensor.sees(point(-5, 0), point(5, 0), (BODY,))


def test_a_line_that_grazes_past_the_body_is_clear():
    sensor = NeighbourSensor(max_range=20.0)

    assert not sensor.occluded(point(-5, 1.5), point(5, 1.5), (BODY,))


def test_occlusion_can_be_switched_off():
    sensor = NeighbourSensor(max_range=20.0, check_occlusion=False)

    assert sensor.sees(point(-5, 0), point(5, 0), (BODY,))


def test_visible_neighbours_lists_only_those_in_sight():
    sensor = NeighbourSensor(max_range=20.0)
    positions = np.array([[-5.0, 0.0], [5.0, 0.0], [0.0, 6.0], [50.0, 0.0]])

    assert sensor.visible_neighbours(0, positions, (BODY,)) == (2,)
    assert sensor.visible_neighbours(2, positions, (BODY,)) == (0, 1)


def test_range_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        NeighbourSensor(max_range=0.0)


def result_at(positions):
    positions = np.asarray(positions, dtype=float)
    steps = positions.shape[0]
    state = np.zeros((steps, 3))
    state[:, :2] = positions
    return SimulationResult(
        time=np.arange(steps, dtype=float),
        state=state,
        requested_control=np.zeros((steps, 2)),
        applied_control=np.zeros((steps, 2)),
        saturated=np.zeros(steps, dtype=bool),
    )


def blinking_pair():
    still = result_at([[-5, 0]] * 4)
    mover = result_at([[-5, 3], [5, 0], [-5, 3], [5, 0]])
    return FleetResult((still, mover))


def test_visibility_is_symmetric_with_an_empty_diagonal():
    visibility = neighbour_visibility(
        blinking_pair(), (BODY,), NeighbourSensor(max_range=20.0)
    )

    assert np.array_equal(visibility.visible, visibility.visible.transpose(0, 2, 1))
    assert not visibility.visible[:, 0, 0].any()
    assert visibility.visible[:, 0, 1].tolist() == [True, False, True, False]
    assert visibility.occluded[:, 0, 1].tolist() == [False, True, False, True]
    assert visibility.in_range[:, 0, 1].all()


def test_metrics_count_dropouts_and_hidden_time():
    visibility = neighbour_visibility(
        blinking_pair(), (BODY,), NeighbourSensor(max_range=20.0)
    )

    metrics = neighbour_visibility_metrics(visibility)

    assert metrics.dropout_events == 2
    assert metrics.occluded_fraction == pytest.approx(0.5)
    assert metrics.mean_visible_count == pytest.approx(0.5)
    assert metrics.seeing_fraction.tolist() == [0.5, 0.5]


def test_scenario_builds_the_sensor_from_its_obstacle_sensor():
    scenario = FleetScenario(
        ReactiveCircumnavScenario(
            sensor_range=9.0, check_occlusion=False, obstacles=(BODY,)
        )
    )

    sensor = scenario.neighbour_sensor()

    assert sensor.max_range == 9.0
    assert not sensor.check_occlusion
    assert scenario.neighbour_sensor(15.0).max_range == 15.0


def ellipse_fleet(robots):
    from dataclasses import replace

    from circumnav.examples.reactive_circumnav import build_obstacle

    base = ReactiveCircumnavScenario(duration=15.0)
    base = replace(base, obstacles=build_obstacle("ellipse", base.rho_0, base.turning_radius))
    scenario = FleetScenario(base, robot_count=robots)
    fleet, _ = scenario.run()
    visibility = neighbour_visibility(
        fleet, scenario.build_obstacles(), scenario.neighbour_sensor()
    )
    return neighbour_visibility_metrics(visibility)


def test_neighbours_drop_out_of_sight_behind_the_ellipse():
    metrics = ellipse_fleet(4)

    assert metrics.dropout_events > 0
    assert metrics.occluded_fraction > 0.0
    assert (metrics.seeing_fraction == 1.0).all()


def test_two_vehicles_on_opposite_sides_never_see_each_other():
    metrics = ellipse_fleet(2)

    assert metrics.mean_visible_count == 0.0
