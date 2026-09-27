"""Static obstacles and their exact geometry.

The module knows nothing about sensors or control laws. For every shape it
answers the same three geometric questions: where is the closest boundary
point to a given point, does a line of sight hit this obstacle, and what does
the equidistant curve at distance ``rho_0`` look like.

Notation follows the reference law: ``rho_0`` is the commanded standoff,
``rho(t)`` the current distance to the nearest obstacle point, and
``d_R(t) = rho(t) - rho_0`` the deviation from the equidistant curve.

The equidistant curve is the level set ``{q : dist(q, obstacle) = rho_0}`` --
the curve the reactive law is asked to ride. An outward offset of a convex
boundary raises every radius of curvature by exactly ``rho_0``, so the curve
bends tightest around a degenerate obstacle:

- circle of radius ``R``      -> ``R + rho_0``
- ellipse with semi-axes a, b -> ``min(a, b)**2 / max(a, b) + rho_0``
- straight segment            -> ``rho_0`` (the caps around its endpoints)

The tightest case over every convex shape is therefore ``rho_0`` itself, which
is why ``rho_0 >= R_min`` (with ``R_min = v / omega``) is the shape-independent
condition for the vehicle to be able to follow the curve at all. See
:func:`check_turning_feasibility` for that condition and
:func:`minimum_equidistant_curvature` for the actual curvature of a given
scene, which is never smaller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_DEFAULT_SAMPLES = 240


@runtime_checkable
class Obstacle(Protocol):
    """A rigid, static obstacle with an exactly computable boundary."""

    obstacle_id: int

    def closest_point(self, point: FloatArray) -> FloatArray:
        """Return the closest point on the obstacle boundary to ``point``."""

    def first_intersection(
        self, origin: FloatArray, target: FloatArray
    ) -> FloatArray | None:
        """Return where segment ``[origin, target]`` first meets this obstacle."""

    def is_boundary_visible(
        self, boundary_point: FloatArray, viewpoint: FloatArray
    ) -> bool:
        """Return whether the obstacle's own body hides ``boundary_point``."""

    def boundary_polyline(self, samples: int = _DEFAULT_SAMPLES) -> FloatArray:
        """Sample the boundary for plotting."""

    def equidistant_polyline(
        self, distance: float, samples: int = _DEFAULT_SAMPLES
    ) -> FloatArray:
        """Sample the equidistant curve at ``distance`` for plotting."""

    def minimum_equidistant_curvature(self, distance: float) -> float:
        """Return the smallest curvature radius of that curve."""


def _unit(vector: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def _as_point(value: object, name: str) -> FloatArray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,):
        raise ValueError(f"{name} must be a 2D point")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be finite")
    return point


def _validate_distance(distance: float) -> None:
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance must be finite and positive")


@dataclass(frozen=True)
class SegmentObstacle:
    """A straight obstacle boundary between two distinct endpoints."""

    start: FloatArray
    end: FloatArray
    obstacle_id: int

    def __post_init__(self) -> None:
        start = _as_point(self.start, "start")
        end = _as_point(self.end, "end")
        if np.allclose(start, end):
            raise ValueError("Segment endpoints must not coincide")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def closest_point(self, point: FloatArray) -> FloatArray:
        direction = self.end - self.start
        t = float(
            np.dot(point - self.start, direction) / np.dot(direction, direction)
        )
        t = float(np.clip(t, 0.0, 1.0))
        return self.start + t * direction

    def first_intersection(
        self, origin: FloatArray, target: FloatArray
    ) -> FloatArray | None:
        return segment_intersection(origin, target, self.start, self.end)

    def is_boundary_visible(
        self, boundary_point: FloatArray, viewpoint: FloatArray
    ) -> bool:
        # A segment is an infinitely thin wall: it never hides its own surface.
        del boundary_point, viewpoint
        return True

    def boundary_polyline(self, samples: int = _DEFAULT_SAMPLES) -> FloatArray:
        del samples
        return np.vstack([self.start, self.end])

    def equidistant_polyline(
        self, distance: float, samples: int = _DEFAULT_SAMPLES
    ) -> FloatArray:
        """Return the stadium at ``distance`` around the segment."""

        _validate_distance(distance)
        direction = _unit(self.end - self.start)
        normal = np.array([-direction[1], direction[0]])
        cap_samples = max(3, samples // 2)

        start_angle = math.atan2(normal[1], normal[0])
        end_cap = self._cap(self.end, start_angle, cap_samples, distance)
        start_cap = self._cap(self.start, start_angle + math.pi, cap_samples, distance)

        return np.vstack(
            [
                self.start + distance * normal,
                self.end + distance * normal,
                end_cap,
                self.end - distance * normal,
                self.start - distance * normal,
                start_cap,
                self.start + distance * normal,
            ]
        )

    @staticmethod
    def _cap(
        center: FloatArray, start_angle: float, samples: int, distance: float
    ) -> FloatArray:
        angles = np.linspace(start_angle, start_angle - math.pi, samples)
        return center + distance * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )

    def minimum_equidistant_curvature(self, distance: float) -> float:
        _validate_distance(distance)
        # The straight flanks are flat; the end caps are arcs of radius
        # exactly ``distance``, so they set the minimum.
        return float(distance)


@dataclass(frozen=True)
class CircleObstacle:
    """A solid circular obstacle."""

    center: FloatArray
    radius: float
    obstacle_id: int

    def __post_init__(self) -> None:
        center = _as_point(self.center, "center")
        if not math.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("radius must be finite and positive")
        object.__setattr__(self, "center", center)

    def closest_point(self, point: FloatArray) -> FloatArray:
        offset = point - self.center
        norm = float(np.linalg.norm(offset))
        if norm == 0.0:
            # Degenerate: any boundary point is equally close.
            return self.center + np.array([self.radius, 0.0])
        return self.center + self.radius * offset / norm

    def first_intersection(
        self, origin: FloatArray, target: FloatArray
    ) -> FloatArray | None:
        direction = target - origin
        offset = origin - self.center
        a = float(np.dot(direction, direction))
        if a == 0.0:
            return None
        b = 2.0 * float(np.dot(offset, direction))
        c = float(np.dot(offset, offset)) - self.radius**2
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None
        root = math.sqrt(discriminant)
        for t in sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))):
            if 0.0 <= t <= 1.0:
                return origin + t * direction
        return None

    def is_boundary_visible(
        self, boundary_point: FloatArray, viewpoint: FloatArray
    ) -> bool:
        outward_normal = boundary_point - self.center
        return bool(np.dot(outward_normal, viewpoint - boundary_point) >= 0.0)

    def boundary_polyline(self, samples: int = _DEFAULT_SAMPLES) -> FloatArray:
        return self._circle_polyline(self.radius, samples)

    def equidistant_polyline(
        self, distance: float, samples: int = _DEFAULT_SAMPLES
    ) -> FloatArray:
        _validate_distance(distance)
        return self._circle_polyline(self.radius + distance, samples)

    def _circle_polyline(self, radius: float, samples: int) -> FloatArray:
        angles = np.linspace(0.0, 2.0 * math.pi, max(8, samples))
        return self.center + radius * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )

    def minimum_equidistant_curvature(self, distance: float) -> float:
        _validate_distance(distance)
        return float(self.radius + distance)


@dataclass(frozen=True)
class EllipseObstacle:
    """A solid, possibly rotated elliptical obstacle.

    ``semi_axis_x`` and ``semi_axis_y`` are measured along the ellipse's own
    axes before ``angle`` rotates it into the world frame.
    """

    center: FloatArray
    semi_axis_x: float
    semi_axis_y: float
    obstacle_id: int
    angle: float = 0.0

    _COARSE_SAMPLES = 72
    _REFINE_ITERATIONS = 60

    def __post_init__(self) -> None:
        center = _as_point(self.center, "center")
        for name, value in (
            ("semi_axis_x", self.semi_axis_x),
            ("semi_axis_y", self.semi_axis_y),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.angle):
            raise ValueError("angle must be finite")
        object.__setattr__(self, "center", center)

    def closest_point(self, point: FloatArray) -> FloatArray:
        local_point = self._to_local(np.asarray(point, dtype=np.float64))
        parameter = self._closest_parameter(local_point)
        return self._to_world(self._local_boundary(parameter))

    def first_intersection(
        self, origin: FloatArray, target: FloatArray
    ) -> FloatArray | None:
        # In scaled local coordinates the ellipse is the unit circle.
        scale = np.array([self.semi_axis_x, self.semi_axis_y])
        local_origin = self._to_local(origin) / scale
        local_target = self._to_local(target) / scale
        direction = local_target - local_origin

        a = float(np.dot(direction, direction))
        if a == 0.0:
            return None
        b = 2.0 * float(np.dot(local_origin, direction))
        c = float(np.dot(local_origin, local_origin)) - 1.0
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None
        root = math.sqrt(discriminant)
        for t in sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))):
            if 0.0 <= t <= 1.0:
                return self._to_world((local_origin + t * direction) * scale)
        return None

    def is_boundary_visible(
        self, boundary_point: FloatArray, viewpoint: FloatArray
    ) -> bool:
        local_point = self._to_local(boundary_point)
        local_normal = np.array(
            [
                local_point[0] / self.semi_axis_x**2,
                local_point[1] / self.semi_axis_y**2,
            ]
        )
        outward_normal = self._rotation() @ local_normal
        return bool(np.dot(outward_normal, viewpoint - boundary_point) >= 0.0)

    def boundary_polyline(self, samples: int = _DEFAULT_SAMPLES) -> FloatArray:
        parameters = np.linspace(0.0, 2.0 * math.pi, max(8, samples))
        local = np.stack(
            [
                self.semi_axis_x * np.cos(parameters),
                self.semi_axis_y * np.sin(parameters),
            ],
            axis=1,
        )
        return local @ self._rotation().T + self.center

    def equidistant_polyline(
        self, distance: float, samples: int = _DEFAULT_SAMPLES
    ) -> FloatArray:
        """Offset every boundary point along its outward normal.

        The outward offset of a convex curve is itself smooth and convex, so
        this never produces cusps however large ``distance`` is.
        """

        _validate_distance(distance)
        parameters = np.linspace(0.0, 2.0 * math.pi, max(8, samples))
        local = np.stack(
            [
                self.semi_axis_x * np.cos(parameters),
                self.semi_axis_y * np.sin(parameters),
            ],
            axis=1,
        )
        normals = np.stack(
            [
                np.cos(parameters) / self.semi_axis_x,
                np.sin(parameters) / self.semi_axis_y,
            ],
            axis=1,
        )
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        return (local + distance * normals) @ self._rotation().T + self.center

    def minimum_equidistant_curvature(self, distance: float) -> float:
        _validate_distance(distance)
        shorter = min(self.semi_axis_x, self.semi_axis_y)
        longer = max(self.semi_axis_x, self.semi_axis_y)
        return float(shorter**2 / longer + distance)

    def _rotation(self) -> FloatArray:
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        return np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    def _to_local(self, point: FloatArray) -> FloatArray:
        return self._rotation().T @ (np.asarray(point, dtype=np.float64) - self.center)

    def _to_world(self, local_point: FloatArray) -> FloatArray:
        return self._rotation() @ local_point + self.center

    def _local_boundary(self, parameter: float) -> FloatArray:
        return np.array(
            [
                self.semi_axis_x * math.cos(parameter),
                self.semi_axis_y * math.sin(parameter),
            ]
        )

    def _closest_parameter(self, local_point: FloatArray) -> float:
        """Coarse scan over the parameter, then a ternary-search refinement."""

        parameters = np.linspace(
            0.0, 2.0 * math.pi, self._COARSE_SAMPLES, endpoint=False
        )
        candidates = np.stack(
            [
                self.semi_axis_x * np.cos(parameters),
                self.semi_axis_y * np.sin(parameters),
            ],
            axis=1,
        )
        best = int(np.argmin(np.linalg.norm(candidates - local_point, axis=1)))

        step = 2.0 * math.pi / self._COARSE_SAMPLES
        low = parameters[best] - step
        high = parameters[best] + step
        for _ in range(self._REFINE_ITERATIONS):
            first = low + (high - low) / 3.0
            second = high - (high - low) / 3.0
            if self._local_distance(first, local_point) <= self._local_distance(
                second, local_point
            ):
                high = second
            else:
                low = first
        return 0.5 * (low + high)

    def _local_distance(self, parameter: float, local_point: FloatArray) -> float:
        return float(np.linalg.norm(self._local_boundary(parameter) - local_point))


def make_obstacles(
    segments: Iterable[tuple[FloatArray, FloatArray]],
    first_id: int = 0,
) -> tuple[SegmentObstacle, ...]:
    """Build identified segment obstacles from an iterable of (start, end) pairs."""

    return tuple(
        SegmentObstacle(
            start=np.asarray(start, dtype=np.float64),
            end=np.asarray(end, dtype=np.float64),
            obstacle_id=first_id + index,
        )
        for index, (start, end) in enumerate(segments)
    )


def segment_intersection(
    p1: FloatArray,
    p2: FloatArray,
    p3: FloatArray,
    p4: FloatArray,
) -> FloatArray | None:
    """Return the intersection point of segments ``[p1,p2]`` and ``[p3,p4]``.

    Returns ``None`` when the segments are parallel or do not overlap within
    both parameter ranges.
    """

    d1 = p2 - p1
    d2 = p4 - p3
    matrix = np.array([d1, -d2]).T
    determinant = np.linalg.det(matrix)
    if abs(determinant) < 1e-12:
        return None
    t, s = np.linalg.solve(matrix, p3 - p1)
    if 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0:
        return p1 + t * d1
    return None


def boundary_gap(first: Obstacle, second: Obstacle, samples: int = 240) -> float:
    """Return the shortest distance between two obstacle boundaries.

    Measured between sampled boundary points, so the result is a slight
    over-estimate of the true separation; the error falls off as ``1/samples**2``
    and is well under a millimetre at the default resolution.
    """

    points_first = first.boundary_polyline(samples)
    points_second = second.boundary_polyline(samples)
    deltas = points_first[:, None, :] - points_second[None, :, :]
    return float(np.min(np.linalg.norm(deltas, axis=2)))


def equidistant_gap(
    first: Obstacle,
    second: Obstacle,
    distance: float,
    samples: int = 240,
) -> float:
    """Return the width of the clear channel between two equidistant curves.

    Each curve is the level set at ``distance`` from its own obstacle, so the
    channel between them is ``boundary_gap - 2 * distance``. The result is
    ``0.0`` when the two curves intersect, which happens as soon as the
    obstacles are closer than ``2 * distance`` and no point can keep the full
    standoff from both at once.
    """

    _validate_distance(distance)
    return max(0.0, boundary_gap(first, second, samples) - 2.0 * distance)


def check_equidistant_spacing(
    obstacles: Sequence[Obstacle],
    rho_0: float,
    max_gap: float,
    samples: int = 240,
) -> float:
    """Verify every obstacle has a close enough neighbour, and return the worst gap.

    For each obstacle the nearest other equidistant curve must be either
    intersecting it or no further than ``max_gap`` away. That keeps the scene
    a connected chain of narrow passages -- the regime where the turning disk
    reaches the next obstacle and the law works in gap mode -- instead of a
    set of isolated bodies the vehicle circles one at a time.
    """

    if len(obstacles) < 2:
        return 0.0
    if not math.isfinite(max_gap) or max_gap < 0.0:
        raise ValueError("max_gap must be finite and non-negative")

    worst_gap = 0.0
    for index, obstacle in enumerate(obstacles):
        neighbour_gaps = [
            equidistant_gap(obstacle, other, rho_0, samples)
            for other_index, other in enumerate(obstacles)
            if other_index != index
        ]
        nearest_gap = min(neighbour_gaps)
        if nearest_gap > max_gap:
            raise ValueError(
                f"Obstacle {obstacle.obstacle_id} is isolated: its nearest "
                f"equidistant curve is {nearest_gap:.4f} m away, more than the "
                f"allowed {max_gap:.4f} m."
            )
        worst_gap = max(worst_gap, nearest_gap)
    return worst_gap


def minimum_equidistant_curvature(
    obstacles: Sequence[Obstacle],
    rho_0: float,
) -> float:
    """Return the tightest curvature radius the scene's equidistant curves have.

    This is always at least ``rho_0``; it equals ``rho_0`` only for obstacles
    with a sharp end, such as a segment's caps.
    """

    if not obstacles:
        return math.inf
    return min(
        obstacle.minimum_equidistant_curvature(rho_0) for obstacle in obstacles
    )


def check_turning_feasibility(rho_0: float, turning_radius: float) -> float:
    """Verify ``rho_0 >= R_min`` and return ``rho_0``.

    The equidistant curve never bends tighter than ``rho_0`` for a convex
    obstacle, and bends exactly that tight around a degenerate one, so this
    single comparison decides feasibility for any scene: below it the vehicle
    physically cannot hold the standoff, whatever the control law does.
    """

    _validate_distance(rho_0)
    if not math.isfinite(turning_radius) or turning_radius <= 0.0:
        raise ValueError("turning_radius must be finite and positive")
    if rho_0 < turning_radius:
        raise ValueError(
            "Infeasible scenario: the commanded standoff is tighter than the "
            f"vehicle can turn (rho_0 = {rho_0:.4f} m < R_min = "
            f"{turning_radius:.4f} m). Raise the standoff or the maximum yaw "
            "rate."
        )
    return rho_0
