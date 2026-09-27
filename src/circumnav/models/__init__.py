"""Plant and actuator models."""

from circumnav.models.actuator import ActuatorOutput, IdealActuator
from circumnav.models.dubins import (
    DubinsCommand,
    DubinsLimits,
    DubinsModel,
    DubinsState,
    normalize_angle,
)
from circumnav.models.obstacles import (
    CircleObstacle,
    EllipseObstacle,
    Obstacle,
    SegmentObstacle,
    boundary_gap,
    check_equidistant_spacing,
    check_turning_feasibility,
    equidistant_gap,
    make_obstacles,
    minimum_equidistant_curvature,
    segment_intersection,
)

__all__ = [
    "ActuatorOutput",
    "CircleObstacle",
    "DubinsCommand",
    "DubinsLimits",
    "DubinsModel",
    "DubinsState",
    "EllipseObstacle",
    "IdealActuator",
    "Obstacle",
    "SegmentObstacle",
    "boundary_gap",
    "check_equidistant_spacing",
    "check_turning_feasibility",
    "equidistant_gap",
    "make_obstacles",
    "minimum_equidistant_curvature",
    "normalize_angle",
    "segment_intersection",
]

