import numpy as np
import pytest

from circumnav.models.obstacles import (
    SegmentObstacle,
    make_obstacles,
    segment_intersection,
)


def test_closest_point_projects_onto_interior_of_segment():
    obstacle = SegmentObstacle(
        start=np.array([0.0, 0.0]), end=np.array([10.0, 0.0]), obstacle_id=0
    )

    closest = obstacle.closest_point(np.array([4.0, 3.0]))

    assert closest == pytest.approx([4.0, 0.0])


def test_closest_point_clips_to_nearest_endpoint():
    obstacle = SegmentObstacle(
        start=np.array([0.0, 0.0]), end=np.array([10.0, 0.0]), obstacle_id=0
    )

    before_start = obstacle.closest_point(np.array([-5.0, 2.0]))
    after_end = obstacle.closest_point(np.array([15.0, -2.0]))

    assert before_start == pytest.approx([0.0, 0.0])
    assert after_end == pytest.approx([10.0, 0.0])


def test_closest_point_is_zero_distance_when_point_is_on_segment():
    obstacle = SegmentObstacle(
        start=np.array([1.0, 1.0]), end=np.array([1.0, 5.0]), obstacle_id=0
    )

    closest = obstacle.closest_point(np.array([1.0, 3.0]))

    assert closest == pytest.approx([1.0, 3.0])


def test_degenerate_segment_is_rejected():
    with pytest.raises(ValueError, match="must not coincide"):
        SegmentObstacle(
            start=np.array([1.0, 1.0]), end=np.array([1.0, 1.0]), obstacle_id=0
        )


def test_make_obstacles_assigns_sequential_ids():
    obstacles = make_obstacles(
        [((0.0, 0.0), (1.0, 0.0)), ((2.0, 0.0), (2.0, 1.0))]
    )

    assert [obstacle.obstacle_id for obstacle in obstacles] == [0, 1]


def test_segment_intersection_finds_crossing_point():
    point = segment_intersection(
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0]),
        np.array([0.0, 2.0]),
        np.array([2.0, 0.0]),
    )

    assert point == pytest.approx([1.0, 1.0])


def test_segment_intersection_returns_none_for_parallel_segments():
    point = segment_intersection(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
    )

    assert point is None


def test_segment_intersection_returns_none_when_out_of_bounds():
    point = segment_intersection(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([5.0, -1.0]),
        np.array([5.0, 1.0]),
    )

    assert point is None
