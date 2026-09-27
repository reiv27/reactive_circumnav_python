import math

import numpy as np
import pytest

from circumnav.models.dubins import DubinsState
from circumnav.models.obstacles import make_obstacles
from circumnav.sensors.circular import CircularVisibilitySensor


def make_sensor(max_range=10.0, exclusion_distance=0.5, check_occlusion=True):
    return CircularVisibilitySensor(
        max_range=max_range,
        exclusion_distance=exclusion_distance,
        check_occlusion=check_occlusion,
    )


def test_find_primary_point_returns_none_without_obstacles():
    sensor = make_sensor()
    pose = DubinsState(0.0, 0.0, 0.0)

    assert sensor.find_primary_point(pose, ()) is None


def test_find_primary_point_ignores_obstacles_beyond_max_range():
    sensor = make_sensor(max_range=5.0)
    pose = DubinsState(0.0, 0.0, 0.0)
    obstacles = make_obstacles([((10.0, -1.0), (10.0, 1.0))])

    assert sensor.find_primary_point(pose, obstacles) is None


def test_find_primary_point_reports_distance_and_bearing():
    sensor = make_sensor()
    pose = DubinsState(0.0, 0.0, 0.0)
    obstacles = make_obstacles([((3.0, -1.0), (3.0, 1.0))])

    primary = sensor.find_primary_point(pose, obstacles)

    assert primary is not None
    assert primary.point_world == pytest.approx([3.0, 0.0])
    assert primary.distance == pytest.approx(3.0)
    assert primary.bearing == pytest.approx(0.0, abs=1e-9)
    assert primary.point_relative == pytest.approx([3.0, 0.0])


def test_bearing_is_relative_to_robot_heading():
    sensor = make_sensor()
    pose = DubinsState(0.0, 0.0, math.pi / 2.0)
    obstacles = make_obstacles([((3.0, -1.0), (3.0, 1.0))])

    primary = sensor.find_primary_point(pose, obstacles)

    # The obstacle is straight ahead in the world (+x), but the robot is
    # facing +y, so in the body frame it lies at bearing -pi/2 (to the right).
    assert primary.bearing == pytest.approx(-math.pi / 2.0)


def test_find_secondary_point_excludes_primary_neighborhood():
    # obstacle 1 sits right next to P1 and would win on distance-to-V alone
    # (0.05 m) if the exclusion zone around P1 did not rule it out; obstacle
    # 2 is farther from V (1.6 m) but outside the exclusion zone and must be
    # the one returned as P2. Occlusion is disabled to isolate the exclusion
    # behavior (occlusion has its own dedicated test).
    sensor = make_sensor(exclusion_distance=0.5, check_occlusion=False)
    pose = DubinsState(0.0, 0.0, 0.0)
    obstacles = make_obstacles(
        [
            ((3.0, -1.0), (3.0, 1.0)),  # P1 owner, closest point (3, 0)
            ((3.1, 0.0), (3.15, 0.1)),  # 0.18 m from P1: must be excluded
            ((3.2, -2.0), (3.2, -1.5)),  # 1.51 m from P1: valid P2
        ]
    )
    disk_center = np.array([3.2, 0.1])

    primary = sensor.find_primary_point(pose, obstacles)
    secondary = sensor.find_secondary_point(pose, disk_center, primary, obstacles)

    assert primary is not None
    assert primary.point_world == pytest.approx([3.0, 0.0])
    assert secondary is not None
    assert secondary.obstacle_id == obstacles[2].obstacle_id


def test_find_secondary_point_returns_none_when_everything_is_excluded():
    sensor = make_sensor(exclusion_distance=5.0)
    pose = DubinsState(0.0, 0.0, 0.0)
    obstacles = make_obstacles([((3.0, -1.0), (3.0, 1.0))])

    primary = sensor.find_primary_point(pose, obstacles)
    secondary = sensor.find_secondary_point(
        pose, np.array([3.0, 0.0]), primary, obstacles
    )

    assert secondary is None


def test_occlusion_hides_a_point_behind_a_closer_obstacle():
    # A short wall at x=1 blocks the line of sight from the robot to the
    # point (3, 0), which is the closest point on the far wall to the disk
    # center placed exactly there.
    blocker_and_far = make_obstacles(
        [
            ((0.0, 5.0), (0.1, 5.0)),  # unrelated obstacle, becomes P1
            ((1.0, -0.3), (1.0, 0.3)),  # blocker
            ((3.0, -0.3), (3.0, 0.3)),  # far wall, occluded at (3, 0)
        ]
    )
    pose = DubinsState(0.0, 0.0, 0.0)
    disk_center = np.array([3.0, 0.0])
    primary = blocker_and_far[0].closest_point(np.array([0.0, 0.0]))
    from circumnav.sensors.base import observation_from_world_point

    primary_observation = observation_from_world_point(pose, primary, 0)

    occluded_sensor = make_sensor(exclusion_distance=0.1, check_occlusion=True)
    visible_sensor = make_sensor(exclusion_distance=0.1, check_occlusion=False)

    occluded_result = occluded_sensor.find_secondary_point(
        pose, disk_center, primary_observation, blocker_and_far
    )
    unoccluded_result = visible_sensor.find_secondary_point(
        pose, disk_center, primary_observation, blocker_and_far
    )

    assert occluded_result is not None
    assert occluded_result.obstacle_id == 1  # the blocker itself, not the far wall
    assert unoccluded_result is not None
    assert unoccluded_result.obstacle_id == 2  # far wall wins without occlusion check
