import math

import numpy as np
import pytest

from circumnav.analysis.metrics import obstacle_clearance_metrics
from circumnav.controllers.reactive import CircumnavMode
from circumnav.models.dubins import DubinsState
from circumnav.models.obstacles import (
    CircleObstacle,
    EllipseObstacle,
    check_equidistant_spacing,
    check_turning_feasibility,
    make_obstacles,
    minimum_equidistant_curvature,
)
from circumnav.scenarios.reactive import ReactiveCircumnavScenario, gap_cluster
from circumnav.sensors.circular import CircularVisibilitySensor


def make_circle(radius=4.0, center=(0.0, 0.0)):
    return CircleObstacle(
        center=np.array(center, dtype=np.float64), radius=radius, obstacle_id=0
    )


def make_ellipse(semi_axis_x=6.0, semi_axis_y=3.0, angle=0.0, center=(0.0, 0.0)):
    return EllipseObstacle(
        center=np.array(center, dtype=np.float64),
        semi_axis_x=semi_axis_x,
        semi_axis_y=semi_axis_y,
        obstacle_id=0,
        angle=angle,
    )


def test_circle_closest_point_lies_on_the_radial_direction():
    circle = make_circle(radius=4.0)

    closest = circle.closest_point(np.array([10.0, 0.0]))

    assert closest == pytest.approx([4.0, 0.0])


def test_circle_closest_point_from_inside_still_lands_on_the_boundary():
    circle = make_circle(radius=4.0)

    closest = circle.closest_point(np.array([1.0, 0.0]))

    assert closest == pytest.approx([4.0, 0.0])


def test_circle_first_intersection_returns_the_near_side():
    circle = make_circle(radius=4.0)

    hit = circle.first_intersection(np.array([-10.0, 0.0]), np.array([10.0, 0.0]))

    assert hit == pytest.approx([-4.0, 0.0])


def test_circle_hides_its_own_far_side():
    circle = make_circle(radius=4.0)
    viewpoint = np.array([-10.0, 0.0])

    assert circle.is_boundary_visible(np.array([-4.0, 0.0]), viewpoint)
    assert not circle.is_boundary_visible(np.array([4.0, 0.0]), viewpoint)


def test_ellipse_closest_point_matches_the_axis_vertices():
    ellipse = make_ellipse(semi_axis_x=6.0, semi_axis_y=3.0)

    along_x = ellipse.closest_point(np.array([20.0, 0.0]))
    along_y = ellipse.closest_point(np.array([0.0, 20.0]))

    assert along_x == pytest.approx([6.0, 0.0], abs=1e-6)
    assert along_y == pytest.approx([0.0, 3.0], abs=1e-6)


def test_ellipse_closest_point_is_never_worse_than_a_dense_scan():
    ellipse = make_ellipse(semi_axis_x=5.0, semi_axis_y=2.0, angle=0.7)
    query = np.array([3.0, -4.0])

    closest = ellipse.closest_point(query)
    scan = ellipse.boundary_polyline(samples=20001)
    best_scanned = float(np.min(np.linalg.norm(scan - query, axis=1)))

    assert float(np.linalg.norm(closest - query)) <= best_scanned + 1e-6


def test_ellipse_closest_point_respects_rotation():
    upright = make_ellipse(semi_axis_x=6.0, semi_axis_y=3.0)
    rotated = make_ellipse(semi_axis_x=6.0, semi_axis_y=3.0, angle=math.pi / 2.0)

    # Rotating the ellipse by 90 deg swaps which axis faces +x.
    assert upright.closest_point(np.array([20.0, 0.0])) == pytest.approx(
        [6.0, 0.0], abs=1e-6
    )
    assert rotated.closest_point(np.array([20.0, 0.0])) == pytest.approx(
        [3.0, 0.0], abs=1e-6
    )


def test_equidistant_curve_of_a_circle_is_a_concentric_circle():
    circle = make_circle(radius=4.0)

    curve = circle.equidistant_polyline(distance=1.5)
    radii = np.linalg.norm(curve - circle.center, axis=1)

    assert radii == pytest.approx(np.full(radii.shape, 5.5))


def test_equidistant_curve_of_an_ellipse_keeps_the_offset_distance():
    ellipse = make_ellipse(semi_axis_x=6.0, semi_axis_y=3.0, angle=0.4)
    distance = 1.25

    curve = ellipse.equidistant_polyline(distance=distance, samples=181)
    distances = [
        float(np.linalg.norm(point - ellipse.closest_point(point)))
        for point in curve
    ]

    assert distances == pytest.approx([distance] * len(distances), abs=1e-4)


def test_equidistant_curve_of_a_segment_is_a_stadium():
    obstacle = make_obstacles([((0.0, 0.0), (4.0, 0.0))])[0]
    distance = 1.0

    curve = obstacle.equidistant_polyline(distance=distance, samples=121)
    distances = [
        float(np.linalg.norm(point - obstacle.closest_point(point)))
        for point in curve
    ]

    assert distances == pytest.approx([distance] * len(distances), abs=1e-9)


def test_equidistant_curvature_follows_the_analytic_radii():
    distance = 1.5
    circle = make_circle(radius=4.0)
    ellipse = make_ellipse(semi_axis_x=6.0, semi_axis_y=3.0)
    wall = make_obstacles([((0.0, -5.0), (0.0, 5.0))])[0]

    # circle: R + d; ellipse: min(a,b)^2 / max(a,b) + d; segment: d (end caps).
    assert circle.minimum_equidistant_curvature(distance) == pytest.approx(5.5)
    assert ellipse.minimum_equidistant_curvature(distance) == pytest.approx(3.0)
    assert wall.minimum_equidistant_curvature(distance) == pytest.approx(1.5)

    assert minimum_equidistant_curvature(
        (circle, ellipse, wall), distance
    ) == pytest.approx(1.5)


def test_check_turning_feasibility_accepts_a_standoff_above_the_turn_radius():
    assert check_turning_feasibility(rho_0=2.0, turning_radius=1.0) == pytest.approx(
        2.0
    )


def test_check_turning_feasibility_rejects_a_standoff_below_the_turn_radius():
    with pytest.raises(ValueError, match="rho_0"):
        check_turning_feasibility(rho_0=1.0, turning_radius=2.0)


def test_scenario_refuses_to_run_an_infeasible_configuration():
    scenario = ReactiveCircumnavScenario(
        duration=2.0,
        forward_speed=1.0,
        yaw_rate_magnitude=0.5,  # R_min = 2.0 m
        safety_distance=1.0,  # rho_0 = 1.0 m, below R_min
        obstacles=make_obstacles([((0.0, -5.0), (0.0, 5.0))]),
    )

    with pytest.raises(ValueError, match="Infeasible scenario"):
        scenario.run()


def test_scenario_can_opt_into_the_infeasible_regime_on_purpose():
    scenario = ReactiveCircumnavScenario(
        duration=2.0,
        forward_speed=1.0,
        yaw_rate_magnitude=0.5,
        safety_distance=1.0,
        obstacles=make_obstacles([((0.0, -5.0), (0.0, 5.0))]),
        require_feasible_turning=False,
    )

    result, _ = scenario.run()

    assert result.time[-1] == pytest.approx(2.0)
    assert scenario.rho_0 < scenario.turning_radius


def test_default_scenario_is_feasible_and_circles_the_obstacle():
    scenario = ReactiveCircumnavScenario()

    assert scenario.rho_0 >= scenario.turning_radius

    result, _ = scenario.run()
    circle = scenario.build_obstacles()[0]
    radii = np.hypot(result.x - circle.center[0], result.y - circle.center[1])
    settled = radii[result.time >= 15.0]

    # Once settled the vehicle should ride the equidistant circle at
    # radius + rho_0, not drift off or cut into the obstacle.
    assert settled.min() > circle.radius
    assert settled.mean() == pytest.approx(
        circle.radius + scenario.safety_distance, abs=0.5
    )


def test_vehicle_settles_on_the_equidistant_curve_of_an_ellipse():
    ellipse = make_ellipse(semi_axis_x=6.0, semi_axis_y=3.0, angle=math.pi / 6.0)
    scenario = ReactiveCircumnavScenario(
        duration=70.0,
        initial_x=-18.0,
        initial_y=0.0,
        safety_distance=3.0,
        sensor_range=14.0,
        obstacles=(ellipse,),
    )

    assert scenario.rho_0 >= scenario.turning_radius

    result, _ = scenario.run()
    settled_indices = result.time >= 30.0
    clearances = np.array(
        [
            np.linalg.norm(
                np.array([x, y]) - ellipse.closest_point(np.array([x, y]))
            )
            for x, y in zip(
                result.x[settled_indices][::50], result.y[settled_indices][::50]
            )
        ]
    )

    assert clearances.mean() == pytest.approx(scenario.safety_distance, abs=0.3)
    assert clearances.min() > 0.5 * scenario.safety_distance


def test_gap_cluster_mixes_circles_and_ellipses():
    obstacles = gap_cluster(rho_0=3.0, turning_radius=2.0)

    assert len(obstacles) == 7
    assert sum(isinstance(o, CircleObstacle) for o in obstacles) == 4
    assert sum(isinstance(o, EllipseObstacle) for o in obstacles) == 3


@pytest.mark.parametrize(
    "rho_0, turning_radius", [(2.0, 1.0), (3.0, 2.0), (4.0, 2.5)]
)
def test_gap_cluster_keeps_neighbouring_equidistants_within_the_limit(
    rho_0, turning_radius
):
    obstacles = gap_cluster(rho_0=rho_0, turning_radius=turning_radius)

    worst = check_equidistant_spacing(
        obstacles, rho_0=rho_0, max_gap=1.5 * turning_radius
    )

    assert worst <= 1.5 * turning_radius
    # One pair must leave a real channel, otherwise the rule would hold only
    # because every curve intersects its neighbour. The builder sizes it at
    # 0.75 * R_min whatever the parameters are.
    # boundary_gap measures between sampled points, hence the mm tolerance.
    assert worst == pytest.approx(0.75 * turning_radius, abs=1e-3)


def test_gap_cluster_rejects_a_vehicle_that_cannot_hold_the_standoff():
    with pytest.raises(ValueError, match="rho_0"):
        gap_cluster(rho_0=1.0, turning_radius=2.0)


def test_gap_cluster_obstacles_do_not_overlap():
    obstacles = gap_cluster(rho_0=3.0, turning_radius=2.0)

    for index, obstacle in enumerate(obstacles):
        for other in obstacles[index + 1 :]:
            boundary = obstacle.boundary_polyline(samples=180)
            gap = min(
                float(np.linalg.norm(point - other.closest_point(point)))
                for point in boundary
            )
            assert gap > 0.0


def test_gap_cluster_scene_exercises_the_gap_mode():
    scenario = ReactiveCircumnavScenario(
        duration=120.0,
        initial_x=-12.0,
        initial_y=0.0,
        forward_speed=2.0,
        yaw_rate_magnitude=1.0,
        safety_distance=3.0,
        sensor_range=12.0,
        exclusion_distance=1.0,
        obstacles=gap_cluster(rho_0=3.0, turning_radius=2.0),
    )

    assert scenario.rho_0 == pytest.approx(3.0)
    assert scenario.turning_radius == pytest.approx(2.0)

    result, controller = scenario.run()
    orbit_steps = sum(
        1 for entry in controller.log if entry.mode is CircumnavMode.ORBIT
    )
    metrics = obstacle_clearance_metrics(
        result, scenario.build_obstacles(), collision_distance=0.02
    )

    assert controller.switch_count >= 4
    assert orbit_steps > 0
    assert not metrics.collided


def test_sensor_reports_relative_distance_to_a_circle():
    sensor = CircularVisibilitySensor(max_range=20.0, exclusion_distance=0.5)
    circle = make_circle(radius=4.0)
    pose = DubinsState(-10.0, 0.0, 0.0)

    primary = sensor.find_primary_point(pose, (circle,))

    assert primary is not None
    assert primary.point_world == pytest.approx([-4.0, 0.0])
    assert primary.distance == pytest.approx(6.0)
    assert primary.bearing == pytest.approx(0.0, abs=1e-9)
