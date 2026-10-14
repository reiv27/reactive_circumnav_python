"""Step-able ``gymnasium.Env`` around the existing plant/sensor/actuator stack.

Postановка (a) of ``docs/RL_plan.md``: the policy replaces
``ReactiveCircumnavController`` entirely, choosing ``yaw_rate`` every
``control_period`` from the same ``P1``/``P2`` sensor measurements the law
would use. It reuses ``DubinsModel``, ``Obstacle`` and ``Actuator`` unchanged
-- only the control law itself is replaced, nothing below it.

No sensing/actuation delay in this first version (``actuator_factory``
defaults to a plain ``IdealActuator``): the goal here is to validate that the
pipeline can learn the nominal task at all before using it for the
delay/model-mismatch robustness experiments the plan motivates this with.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import spaces

from circumnav.models.actuator import Actuator, IdealActuator
from circumnav.models.dubins import DubinsCommand, DubinsLimits, DubinsModel, DubinsState
from circumnav.models.obstacles import Obstacle
from circumnav.sensors.base import PointObservation
from circumnav.sensors.circular import CircularVisibilitySensor


def _integer_ratio(numerator: float, denominator: float, name: str) -> int:
    ratio = numerator / denominator
    rounded = round(ratio)
    if rounded < 1 or not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} must be an integer multiple of integration_step")
    return int(rounded)


def _disk_center(
    robot_position: NDArray[np.float64],
    target: NDArray[np.float64],
    radius: float,
) -> NDArray[np.float64]:
    """Same construction ``ReactiveCircumnavController`` uses for V.

    Pure sensing geometry (projects P1 toward the robot by rho_0 + R_min), not
    part of the frozen control law, so an RL replacement is free to use it to
    expose P2 the same way the baseline law would see it.
    """

    direction = robot_position - target
    unit_direction = direction / np.linalg.norm(direction)
    return target + unit_direction * radius


@dataclass(frozen=True)
class CircumnavEnvConfig:
    """All parameters needed to reproduce an episode of the RL task."""

    obstacles: tuple[Obstacle, ...]
    forward_speed: float = 2.0
    yaw_rate_magnitude: float = 1.0
    safety_distance: float = 3.0
    sensor_range: float = 12.0
    exclusion_distance: float = 1.0
    check_occlusion: bool = True
    control_period: float = 0.02
    integration_step: float = 0.01
    duration: float = 60.0
    collision_distance: float = 0.05
    #: lambda_u in docs/RL_plan.md section 3.
    action_change_penalty: float = 0.05
    collision_penalty: float = 50.0
    #: Number of past actions folded into the observation (history-augmented
    #: state -- see docs/RL_plan.md section 3). 1 is enough with no delay;
    #: raise it together with sensing_delay_steps once delay is introduced.
    history_len: int = 1
    initial_x: float = -9.0
    initial_y: float = -5.0
    initial_heading: float = -1.268
    #: Uniform +/- jitter applied to the initial pose on every reset, so the
    #: policy does not just memorize one starting point.
    start_position_jitter: float = 0.0
    start_heading_jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.forward_speed <= 0.0:
            raise ValueError("forward_speed must be positive")
        if self.yaw_rate_magnitude <= 0.0:
            raise ValueError("yaw_rate_magnitude must be positive")
        if self.history_len < 1:
            raise ValueError("history_len must be at least 1")
        _integer_ratio(self.control_period, self.integration_step, "control_period")
        _integer_ratio(self.duration, self.control_period, "duration")

    @property
    def substeps_per_control_period(self) -> int:
        return _integer_ratio(self.control_period, self.integration_step, "control_period")

    @property
    def max_control_steps(self) -> int:
        return _integer_ratio(self.duration, self.control_period, "duration")


# Observation layout: [primary_visible, primary_distance, sin(bearing1),
# cos(bearing1), secondary_visible, secondary_distance, sin(bearing2),
# cos(bearing2), *last history_len normalized yaw-rate actions].
_OBSERVATION_BASE_DIM = 8


class CircumnavEnv(gym.Env):
    """Gymnasium environment: one ``control_period`` of the plant per ``step``."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: CircumnavEnvConfig,
        actuator_factory: Callable[[], Actuator] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = DubinsModel()
        self.sensor = CircularVisibilitySensor(
            max_range=config.sensor_range,
            exclusion_distance=config.exclusion_distance,
            check_occlusion=config.check_occlusion,
        )
        self._actuator_factory = actuator_factory or self._default_actuator_factory
        self.actuator: Actuator = self._actuator_factory()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        observation_dim = _OBSERVATION_BASE_DIM + config.history_len
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(observation_dim,), dtype=np.float32
        )

        self.state: DubinsState | None = None
        self._prev_yaw_rate = 0.0
        self._control_step = 0
        self._action_history: deque[float] = deque(maxlen=config.history_len)

    def _default_actuator_factory(self) -> Actuator:
        limits = DubinsLimits(
            min_speed=self.config.forward_speed,
            max_speed=self.config.forward_speed,
            max_yaw_rate=self.config.yaw_rate_magnitude,
        )
        return IdealActuator(limits)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        del options

        jitter_xy = (
            self.np_random.uniform(-1.0, 1.0, size=2) * self.config.start_position_jitter
        )
        jitter_heading = (
            self.np_random.uniform(-1.0, 1.0) * self.config.start_heading_jitter
        )
        self.state = DubinsState(
            x=self.config.initial_x + float(jitter_xy[0]),
            y=self.config.initial_y + float(jitter_xy[1]),
            heading=self.config.initial_heading + float(jitter_heading),
        )
        self.actuator = self._actuator_factory()
        self.actuator.reset()
        self._prev_yaw_rate = 0.0
        self._control_step = 0
        self._action_history.clear()
        self._action_history.extend([0.0] * self.config.history_len)

        primary, secondary = self._perceive()
        observation = self._build_observation(primary, secondary)
        return observation, {}

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        assert self.state is not None, "call reset() before step()"

        yaw_rate = float(np.clip(action[0], -1.0, 1.0)) * self.config.yaw_rate_magnitude
        command = DubinsCommand(speed=self.config.forward_speed, yaw_rate=yaw_rate)

        for _ in range(self.config.substeps_per_control_period):
            output = self.actuator.apply(command, self.config.integration_step)
            self.state = self.model.propagate(
                self.state, output.applied, self.config.integration_step
            )

        self._control_step += 1
        self._action_history.append(float(np.clip(action[0], -1.0, 1.0)))

        primary, secondary = self._perceive()
        observation = self._build_observation(primary, secondary)

        if primary is not None:
            d_r = primary.distance - self.config.safety_distance
            tracking_reward = -abs(d_r)
        else:
            d_r = math.nan
            tracking_reward = 0.0
        action_penalty = -self.config.action_change_penalty * abs(
            yaw_rate - self._prev_yaw_rate
        )
        self._prev_yaw_rate = yaw_rate

        clearance = self._clearance()
        collided = clearance < self.config.collision_distance
        reward = tracking_reward + action_penalty
        if collided:
            reward -= self.config.collision_penalty

        terminated = collided
        truncated = self._control_step >= self.config.max_control_steps
        info = {
            "d_r": d_r,
            "clearance": clearance,
            "collided": collided,
            "primary_distance": primary.distance if primary is not None else math.nan,
        }
        return observation, reward, terminated, truncated, info

    def _perceive(
        self,
    ) -> tuple[PointObservation | None, PointObservation | None]:
        assert self.state is not None
        primary = self.sensor.find_primary_point(self.state, self.config.obstacles)
        if primary is None:
            return None, None
        robot_position = np.array([self.state.x, self.state.y])
        disk_center = _disk_center(
            robot_position,
            primary.point_world,
            self.config.safety_distance
            + self.config.forward_speed / self.config.yaw_rate_magnitude,
        )
        secondary = self.sensor.find_secondary_point(
            self.state, disk_center, primary, self.config.obstacles
        )
        return primary, secondary

    def _build_observation(
        self,
        primary: PointObservation | None,
        secondary: PointObservation | None,
    ) -> NDArray[np.float32]:
        features = np.zeros(_OBSERVATION_BASE_DIM, dtype=np.float32)
        if primary is not None:
            features[0] = 1.0
            features[1] = primary.distance / self.config.sensor_range
            features[2] = math.sin(primary.bearing)
            features[3] = math.cos(primary.bearing)
        if secondary is not None:
            features[4] = 1.0
            features[5] = secondary.distance / self.config.sensor_range
            features[6] = math.sin(secondary.bearing)
            features[7] = math.cos(secondary.bearing)
        history = np.array(self._action_history, dtype=np.float32)
        return np.concatenate([features, history])

    def _clearance(self) -> float:
        assert self.state is not None
        position = np.array([self.state.x, self.state.y])
        clearance = math.inf
        for obstacle in self.config.obstacles:
            closest_point = obstacle.closest_point(position)
            clearance = min(clearance, float(np.linalg.norm(closest_point - position)))
        return clearance
