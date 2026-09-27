"""Configuration and assembly of the reactive circumnavigation experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from circumnav.controllers.reactive import (
    ReactiveCircumnavConfig,
    ReactiveCircumnavController,
)
from circumnav.models.actuator import IdealActuator
from circumnav.models.dubins import DubinsLimits, DubinsModel, DubinsState
from circumnav.models.obstacles import (
    CircleObstacle,
    EllipseObstacle,
    Obstacle,
    check_equidistant_spacing,
    check_turning_feasibility,
    minimum_equidistant_curvature,
)
from circumnav.simulation.engine import SimulationConfig, Simulator
from circumnav.simulation.result import SimulationResult

# A single round obstacle: the equidistant curve is a circle of radius
# ``radius + rho_0``.
DEFAULT_OBSTACLES: tuple[Obstacle, ...] = (
    CircleObstacle(center=np.array([0.0, 0.0]), radius=4.0, obstacle_id=0),
)

#: Widest clear channel allowed between two neighbouring equidistant curves,
#: in units of the vehicle's minimum turning radius.
MAX_EQUIDISTANT_GAP_FACTOR = 1.5

#: Width of the one deliberate channel in :func:`gap_cluster`, in units of
#: R_min. Half the allowed maximum, so the spacing rule is demonstrated
#: rather than merely satisfied.
CHANNEL_WIDTH_FACTOR = 0.75

# Ring layout for :func:`gap_cluster`, in units of its ``scale``.
_CLUSTER_CIRCLES: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (3.5, 7.0),
    (-4.2, -3.5),
)
_CLUSTER_ELLIPSES: tuple[tuple[float, float, float], ...] = (
    (7.0, 0.0, math.pi / 2.0),
    (-4.0, 5.0, 0.3),
    (3.0, -6.0, -0.4),
)

# The ring circle above which the channel obstacle is placed.
_CHANNEL_ANCHOR = 1


def gap_cluster(
    rho_0: float = 2.0,
    turning_radius: float = 1.0,
    scale: float = 1.0,
) -> tuple[Obstacle, ...]:
    """A ring of circles and ellipses laid out to exercise the gap (``G``) mode.

    Two rules shape the scene:

    - ``rho_0 >= R_min``, so the vehicle can hold the commanded standoff at
      all (checked by :func:`check_turning_feasibility`);
    - every obstacle's nearest equidistant curve either intersects its own or
      leaves a channel no wider than ``MAX_EQUIDISTANT_GAP_FACTOR * R_min``,
      so the scene is a
      connected chain of narrow passages rather than isolated bodies. Two
      equidistant curves intersect as soon as their obstacles are closer than
      ``2 * rho_0``, so the admissible boundary-to-boundary spacing runs up to
      ``2 * rho_0 + 1.5 * R_min``.
    """

    if scale <= 0.0:
        raise ValueError("scale must be positive")
    check_turning_feasibility(rho_0, turning_radius)

    circle_radius = 1.5 * scale
    obstacles: list[Obstacle] = [
        CircleObstacle(
            center=np.array([x, y]) * scale,
            radius=circle_radius,
            obstacle_id=index,
        )
        for index, (x, y) in enumerate(_CLUSTER_CIRCLES)
    ]
    obstacles.extend(
        EllipseObstacle(
            center=np.array([x, y]) * scale,
            semi_axis_x=4.0 * scale,
            semi_axis_y=2.0 * scale,
            obstacle_id=len(_CLUSTER_CIRCLES) + index,
            angle=angle,
        )
        for index, (x, y, angle) in enumerate(_CLUSTER_ELLIPSES)
    )

    # Circle-to-circle the separation is exact, so the channel width can be
    # placed analytically.
    anchor = obstacles[_CHANNEL_ANCHOR]
    separation = 2.0 * rho_0 + CHANNEL_WIDTH_FACTOR * turning_radius
    obstacles.append(
        CircleObstacle(
            center=anchor.center
            + np.array([0.0, 2.0 * circle_radius + separation]),
            radius=circle_radius,
            obstacle_id=len(obstacles),
        )
    )

    scene = tuple(obstacles)
    check_equidistant_spacing(
        scene,
        rho_0=rho_0,
        max_gap=MAX_EQUIDISTANT_GAP_FACTOR * turning_radius,
        samples=160,
    )
    return scene


@dataclass(frozen=True)
class ReactiveCircumnavScenario:
    """All parameters required to reproduce a circumnavigation experiment."""

    duration: float = 60.0
    integration_step: float = 0.01
    control_period: float = 0.02
    # Just outside the equidistant curve and pointed along its tangent, so
    # the opening transient runs along the curve rather than into a body.
    initial_x: float = -9.0
    initial_y: float = -5.0
    initial_heading: float = -1.268
    forward_speed: float = 2.0
    yaw_rate_magnitude: float = 1.0
    #: rho_0, the commanded standoff. Must stay at or above R_min = v / omega.
    safety_distance: float = 3.0
    sensor_range: float = 12.0
    exclusion_distance: float = 1.0
    dead_zone: float = 0.1
    # See :attr:`closing_speed_ratio`: keep it near 0.3 when changing
    # forward_speed. ReactiveCircumnavConfig still defaults to the
    # reference value, which converges far too slowly to be useful here.
    dead_zone_gain: float = 6.0
    check_occlusion: bool = True
    max_speed: float = 2.0
    max_yaw_rate: float = 2.0
    obstacles: tuple[Obstacle, ...] = DEFAULT_OBSTACLES
    require_feasible_turning: bool = True

    @property
    def turning_radius(self) -> float:
        """``R_min``: the tightest circle the vehicle can fly."""

        return self.forward_speed / self.yaw_rate_magnitude

    @property
    def rho_0(self) -> float:
        """The commanded standoff; ``d`` in the reference implementation."""

        return self.safety_distance

    @property
    def closing_speed_ratio(self) -> float:
        """``dead_zone_gain * dead_zone / v``: how fast the law closes on the curve.

        The sliding surface asks for a radial speed of
        ``dead_zone_gain * dead_zone``, which the vehicle can only deliver up
        to ``v``. Around 0.3 converges briskly; the reference value of
        ``dead_zone_gain`` makes this ~0.001 and the approach takes hundreds
        of times longer than a typical run.
        """

        return self.dead_zone_gain * self.dead_zone / self.forward_speed

    @property
    def equidistant_curvature(self) -> float:
        """Tightest curvature radius the scene's equidistant curves actually have.

        Never smaller than :attr:`rho_0`, and equal to it only where an
        obstacle has a sharp end.
        """

        return minimum_equidistant_curvature(self.obstacles, self.safety_distance)

    def build_obstacles(self) -> tuple[Obstacle, ...]:
        return tuple(self.obstacles)

    def build_controller(self) -> ReactiveCircumnavController:
        return ReactiveCircumnavController(
            config=ReactiveCircumnavConfig(
                forward_speed=self.forward_speed,
                yaw_rate_magnitude=self.yaw_rate_magnitude,
                safety_distance=self.safety_distance,
                dead_zone=self.dead_zone,
                dead_zone_gain=self.dead_zone_gain,
                exclusion_distance=self.exclusion_distance,
                sensor_range=self.sensor_range,
                check_occlusion=self.check_occlusion,
            ),
            obstacles=self.build_obstacles(),
        )

    def validate(self) -> float:
        """Check ``rho_0 >= R_min`` and return ``rho_0``.

        The vehicle physically cannot ride an equidistant curve that bends
        tighter than its own minimum turning radius, so this is checked before
        every run. Set ``require_feasible_turning=False`` to run the
        infeasible regime on purpose and study what the law does there.
        """

        if not self.require_feasible_turning:
            return self.rho_0
        return check_turning_feasibility(self.rho_0, self.turning_radius)

    def run(self) -> tuple[SimulationResult, ReactiveCircumnavController]:
        """Assemble and execute the scenario without global mutable state."""

        self.validate()

        limits = DubinsLimits(
            min_speed=0.0,
            max_speed=self.max_speed,
            max_yaw_rate=self.max_yaw_rate,
        )
        controller = self.build_controller()
        simulator = Simulator(
            model=DubinsModel(),
            actuator=IdealActuator(limits),
            config=SimulationConfig(
                duration=self.duration,
                integration_step=self.integration_step,
                control_period=self.control_period,
            ),
        )
        result = simulator.run(
            initial_state=DubinsState(
                x=self.initial_x,
                y=self.initial_y,
                heading=self.initial_heading,
            ),
            controller=controller,
        )
        return result, controller
