import math
from dataclasses import replace

import numpy as np
import pytest

from circumnav.analysis.metrics import (
    NeighbourVisibility,
    fleet_separation_metrics,
    nearest_neighbour_distance,
    nearest_visible_neighbour_distance,
    neighbour_visibility,
    obstacle_distance,
    pairwise_distance,
)
from circumnav.models.obstacles import CircleObstacle, EllipseObstacle
from circumnav.scenarios.fleet import FleetScenario, equidistant_start_poses
from circumnav.scenarios.reactive import ReactiveCircumnavScenario
from circumnav.simulation.result import FleetResult, SimulationResult

ELLIPSE = EllipseObstacle(
    center=np.array([0.0, 0.0]),
    semi_axis_x=6.0,
    semi_axis_y=3.0,
    obstacle_id=0,
    angle=math.pi / 6.0,
)
CIRCLE = CircleObstacle(center=np.array([1.0, -2.0]), radius=4.0, obstacle_id=0)


def ellipse_base(**fields) -> ReactiveCircumnavScenario:
    return ReactiveCircumnavScenario(obstacles=(ELLIPSE,), **fields)


def test_poses_lie_on_the_equidistant_curve():
    poses = equidistant_start_poses(ELLIPSE, 3.0, 5)

    for pose in poses:
        point = np.array([pose.x, pose.y])
        distance = np.linalg.norm(ELLIPSE.closest_point(point) - point)
        assert distance == pytest.approx(3.0, abs=1e-3)


def test_poses_head_along_the_tangent_with_the_obstacle_on_the_left():
    for pose in equidistant_start_poses(ELLIPSE, 3.0, 6):
        point = np.array([pose.x, pose.y])
        to_obstacle = ELLIPSE.closest_point(point) - point
        to_obstacle /= np.linalg.norm(to_obstacle)
        heading = np.array([math.cos(pose.heading), math.sin(pose.heading)])
        left = np.array([-heading[1], heading[0]])

        assert abs(float(heading @ to_obstacle)) < 5e-3
        assert float(left @ to_obstacle) > 0.99


def test_poses_on_a_circle_are_evenly_spaced():
    poses = equidistant_start_poses(CIRCLE, 3.0, 5)
    angles = [
        math.atan2(pose.y - CIRCLE.center[1], pose.x - CIRCLE.center[0])
        for pose in poses
    ]

    steps = np.diff(np.unwrap(angles))
    assert steps == pytest.approx(2.0 * math.pi / 5.0, abs=1e-3)


def test_phase_shifts_every_pose_along_the_curve():
    first = equidistant_start_poses(CIRCLE, 3.0, 4, phase=0.0)
    shifted = equidistant_start_poses(CIRCLE, 3.0, 4, phase=0.125)

    first_angle = math.atan2(first[0].y - CIRCLE.center[1], first[0].x - CIRCLE.center[0])
    shifted_angle = math.atan2(
        shifted[0].y - CIRCLE.center[1], shifted[0].x - CIRCLE.center[0]
    )
    assert (shifted_angle - first_angle) % (2.0 * math.pi) == pytest.approx(
        2.0 * math.pi * 0.125, abs=1e-3
    )


def test_poses_reject_a_non_positive_count():
    with pytest.raises(ValueError, match="at least 1"):
        equidistant_start_poses(ELLIPSE, 3.0, 0)


def test_fleet_scenario_rejects_several_obstacles():
    base = ReactiveCircumnavScenario(obstacles=(ELLIPSE, CIRCLE))

    with pytest.raises(ValueError, match="exactly one obstacle"):
        FleetScenario(base)


@pytest.mark.parametrize(
    "delay",
    [
        {"sensing_delay": 0.02},
        {"actuation_delay": 0.02},
        {"actuator_time_constant": 0.02},
    ],
)
def test_fleet_scenario_rejects_delays(delay):
    with pytest.raises(ValueError, match="delays"):
        FleetScenario(ellipse_base(**delay))


def test_fleet_scenario_rejects_an_empty_fleet():
    with pytest.raises(ValueError, match="at least 1"):
        FleetScenario(ellipse_base(), robot_count=0)


def test_each_vehicle_matches_a_single_run_from_its_pose():
    scenario = FleetScenario(ellipse_base(duration=4.0), robot_count=3)

    fleet, controllers = scenario.run()

    assert fleet.robot_count == 3
    assert len(controllers) == 3
    for pose, result in zip(scenario.start_poses, fleet.results):
        single, _ = replace(
            scenario.base,
            initial_x=pose.x,
            initial_y=pose.y,
            initial_heading=pose.heading,
        ).run()
        assert np.array_equal(result.state, single.state)


def test_vehicles_hold_the_equidistant_curve_without_collision():
    scenario = FleetScenario(ellipse_base(duration=20.0), robot_count=4)

    fleet, controllers = scenario.run()

    for controller in controllers:
        deviation = np.array([e.equidistant_deviation for e in controller.log])
        assert np.nanmax(np.abs(deviation)) < 0.05
        assert controller.switch_count == 0
    assert fleet_separation_metrics(fleet).minimum_separation > 5.0


def result_at(positions: np.ndarray) -> SimulationResult:
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


def test_fleet_result_requires_a_common_time_grid():
    first = result_at(np.zeros((3, 2)))
    second = result_at(np.zeros((4, 2)))

    with pytest.raises(ValueError, match="time grid"):
        FleetResult((first, second))


def test_fleet_result_rejects_an_empty_fleet():
    with pytest.raises(ValueError, match="at least one"):
        FleetResult(())


def test_positions_have_the_documented_shape():
    fleet = FleetResult((result_at(np.zeros((5, 2))), result_at(np.ones((5, 2)))))

    assert fleet.positions.shape == (2, 5, 2)
    assert fleet.robot_count == 2


def test_nearest_neighbour_distance_picks_the_closest_other_vehicle():
    a = result_at(np.array([[0.0, 0.0], [0.0, 0.0]]))
    b = result_at(np.array([[3.0, 0.0], [1.0, 0.0]]))
    c = result_at(np.array([[10.0, 0.0], [10.0, 0.0]]))

    distances = nearest_neighbour_distance(FleetResult((a, b, c)))

    assert distances[0] == pytest.approx([3.0, 1.0])
    assert distances[1] == pytest.approx([3.0, 1.0])
    assert distances[2] == pytest.approx([7.0, 9.0])


def test_separation_metrics_report_start_minimum_and_end():
    a = result_at(np.zeros((3, 2)))
    b = result_at(np.array([[4.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))

    metrics = fleet_separation_metrics(FleetResult((a, b)))

    assert metrics.initial_separation == pytest.approx(4.0)
    assert metrics.minimum_separation == pytest.approx(1.0)
    assert metrics.final_separation == pytest.approx(2.0)


def test_a_single_vehicle_has_no_neighbours():
    metrics = fleet_separation_metrics(FleetResult((result_at(np.zeros((3, 2))),)))

    assert math.isinf(metrics.minimum_separation)


def test_pairwise_distance_is_symmetric_with_an_infinite_diagonal():
    a = result_at(np.array([[0.0, 0.0], [0.0, 0.0]]))
    b = result_at(np.array([[3.0, 4.0], [1.0, 0.0]]))

    distances = pairwise_distance(FleetResult((a, b)))

    assert distances.shape == (2, 2, 2)
    assert distances[0, 1] == pytest.approx([5.0, 1.0])
    assert distances[1, 0] == pytest.approx(distances[0, 1])
    assert np.isinf(distances[0, 0]).all() and np.isinf(distances[1, 1]).all()


def test_obstacle_distance_is_measured_to_the_surface_at_every_sample():
    circle = CircleObstacle(center=np.array([0.0, 0.0]), radius=2.0, obstacle_id=0)
    result = result_at(np.array([[5.0, 0.0], [0.0, -3.0]]))

    assert obstacle_distance(result, (circle,)) == pytest.approx([3.0, 1.0])


def test_distance_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from circumnav.analysis.fleet_animation import save_fleet_distance_figure

    scenario = FleetScenario(ellipse_base(duration=2.0), robot_count=3)
    fleet, _ = scenario.run()
    visibility = neighbour_visibility(fleet, (ELLIPSE,), scenario.neighbour_sensor())
    path = tmp_path / "distances.png"

    save_fleet_distance_figure(
        fleet, (ELLIPSE,), path, safety_distance=3.0, visibility=visibility, focus=1
    )

    assert path.stat().st_size > 0


def test_distance_figure_needs_two_vehicles(tmp_path):
    pytest.importorskip("matplotlib")
    from circumnav.analysis.fleet_animation import save_fleet_distance_figure

    fleet = FleetResult((result_at(np.zeros((3, 2))),))

    with pytest.raises(ValueError, match="at least 2"):
        save_fleet_distance_figure(
            fleet, (CIRCLE,), tmp_path / "x.png", 3.0, NeighbourVisibility(
                in_range=np.zeros((3, 1, 1), dtype=bool),
                visible=np.zeros((3, 1, 1), dtype=bool),
            )
        )


def test_nearest_visible_neighbour_skips_hidden_ones_and_is_nan_when_blind():
    a = result_at(np.zeros((3, 2)))
    b = result_at(np.full((3, 2), [3.0, 0.0]))
    c = result_at(np.full((3, 2), [0.0, 5.0]))
    visible = np.zeros((3, 3, 3), dtype=bool)
    visible[0, 0, 1] = visible[0, 0, 2] = True  # sees both
    visible[1, 0, 2] = True                     # the closer one is hidden
    visible |= visible.transpose(0, 2, 1)       # sample 2: sees nobody
    visibility = NeighbourVisibility(in_range=visible, visible=visible)

    seen = nearest_visible_neighbour_distance(FleetResult((a, b, c)), visibility)

    assert seen[0, :2] == pytest.approx([3.0, 5.0])
    assert math.isnan(seen[0, 2])
    assert nearest_neighbour_distance(FleetResult((a, b, c)))[0] == pytest.approx(3.0)
