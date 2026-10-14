import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from circumnav.models.obstacles import CircleObstacle
from circumnav.rl.env import CircumnavEnv, CircumnavEnvConfig


def _env(**overrides) -> CircumnavEnv:
    obstacles = (CircleObstacle(center=np.array([0.0, 0.0]), radius=4.0, obstacle_id=0),)
    overrides.setdefault("duration", 2.0)
    config = CircumnavEnvConfig(obstacles=obstacles, **overrides)
    return CircumnavEnv(config)


def test_reset_returns_an_observation_inside_the_declared_space():
    env = _env()
    observation, info = env.reset(seed=0)

    assert env.observation_space.contains(observation)
    assert info == {}


def test_step_returns_shapes_matching_the_declared_spaces():
    env = _env()
    env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(np.array([0.3], dtype=np.float32))

    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert set(info) == {"d_r", "clearance", "collided", "primary_distance"}


def test_episode_truncates_after_duration_over_control_period_steps():
    env = _env(duration=0.1, control_period=0.02)
    env.reset(seed=0)

    truncated = False
    for step in range(10):
        _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
        if terminated or truncated:
            break

    assert truncated
    assert step == 4  # duration / control_period - 1


def test_driving_straight_into_the_obstacle_terminates_with_the_collision_penalty():
    env = _env(
        duration=10.0,
        initial_x=-6.0,
        initial_y=0.0,
        initial_heading=0.0,
        collision_penalty=50.0,
    )
    env.reset(seed=0)

    terminated = False
    reward = 0.0
    for _ in range(500):
        # heading straight at the circle's boundary (radius 4, centered at origin)
        _, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        if terminated or truncated:
            break

    assert terminated
    assert info["collided"]
    assert reward <= -50.0


def test_action_history_feeds_into_the_observation():
    env = _env(history_len=3)
    observation, _ = env.reset(seed=0)
    assert np.allclose(observation[-3:], 0.0)

    observation, *_ = env.step(np.array([1.0], dtype=np.float32))
    observation, *_ = env.step(np.array([-0.5], dtype=np.float32))
    assert observation[-2] == pytest.approx(1.0)
    assert observation[-1] == pytest.approx(-0.5)


def test_reset_is_reproducible_with_a_seed_when_jitter_is_enabled():
    env = _env(start_position_jitter=1.0, start_heading_jitter=0.2)
    first, _ = env.reset(seed=42)
    second, _ = env.reset(seed=42)

    np.testing.assert_allclose(first, second)


def test_config_rejects_control_period_not_a_multiple_of_integration_step():
    obstacles = (CircleObstacle(center=np.array([0.0, 0.0]), radius=4.0, obstacle_id=0),)
    with pytest.raises(ValueError, match="integer multiple"):
        CircumnavEnvConfig(
            obstacles=obstacles, control_period=0.025, integration_step=0.01
        )
