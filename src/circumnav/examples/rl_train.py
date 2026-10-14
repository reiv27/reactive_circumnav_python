"""Train a PPO policy to replace the reactive relay (postановка (a), see
docs/RL_plan.md). Learning-exercise script, not a claim that this should beat
``PredictorController``: with no delay and an exactly known Dubins model (the
case trained and evaluated here), the model-based law cannot be beaten by
construction (RL_plan.md section 5). This validates the step-able environment
and a reference RL baseline end to end, starting on a single ellipse.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from circumnav.examples.reactive_circumnav import build_obstacle
from circumnav.rl.env import CircumnavEnv, CircumnavEnvConfig
from circumnav.scenarios.reactive import ReactiveCircumnavScenario


def make_env_config(
    shape: str, duration: float, jitter: bool
) -> CircumnavEnvConfig:
    base = ReactiveCircumnavScenario(duration=duration)
    obstacles = build_obstacle(
        shape, rho_0=base.rho_0, turning_radius=base.turning_radius
    )
    return CircumnavEnvConfig(
        obstacles=obstacles,
        forward_speed=base.forward_speed,
        yaw_rate_magnitude=base.yaw_rate_magnitude,
        safety_distance=base.safety_distance,
        sensor_range=base.sensor_range,
        exclusion_distance=base.exclusion_distance,
        control_period=base.control_period,
        integration_step=base.integration_step,
        duration=duration,
        initial_x=base.initial_x,
        initial_y=base.initial_y,
        initial_heading=base.initial_heading,
        start_position_jitter=0.5 if jitter else 0.0,
        start_heading_jitter=0.1 if jitter else 0.0,
    )


def train(config: CircumnavEnvConfig, timesteps: int, n_envs: int, seed: int):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = make_vec_env(
        lambda: CircumnavEnv(config),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )
    model = PPO("MlpPolicy", vec_env, seed=seed, verbose=1, device="cpu")
    model.learn(total_timesteps=timesteps)
    return model


def rollout_policy(model, config: CircumnavEnvConfig) -> dict:
    """Deterministic rollout from the (unjittered) default start pose."""

    eval_config = replace(
        config, start_position_jitter=0.0, start_heading_jitter=0.0
    )
    env = CircumnavEnv(eval_config)
    obs, _ = env.reset(seed=0)
    positions = [[env.state.x, env.state.y]]
    d_rs: list[float] = []
    yaw_rates: list[float] = []
    minimum_clearance = float("inf")
    collided = False
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        positions.append([env.state.x, env.state.y])
        d_rs.append(info["d_r"])
        yaw_rates.append(float(action[0]) * config.yaw_rate_magnitude)
        minimum_clearance = min(minimum_clearance, info["clearance"])
        collided = collided or info["collided"]

    return {
        "positions": np.array(positions),
        "d_r": np.array(d_rs),
        "yaw_rate": np.array(yaw_rates),
        "minimum_clearance": minimum_clearance,
        "collided": collided,
    }


def run_baseline(config: CircumnavEnvConfig) -> dict:
    """The frozen reactive law on the identical scene, for comparison."""

    from circumnav.analysis.metrics import obstacle_clearance_metrics
    from circumnav.controllers.reactive import CircumnavMode

    scenario = ReactiveCircumnavScenario(
        duration=config.duration,
        integration_step=config.integration_step,
        control_period=config.control_period,
        initial_x=config.initial_x,
        initial_y=config.initial_y,
        initial_heading=config.initial_heading,
        forward_speed=config.forward_speed,
        yaw_rate_magnitude=config.yaw_rate_magnitude,
        safety_distance=config.safety_distance,
        sensor_range=config.sensor_range,
        exclusion_distance=config.exclusion_distance,
        obstacles=config.obstacles,
    )
    result, controller = scenario.run()
    metrics = obstacle_clearance_metrics(
        result, config.obstacles, collision_distance=config.collision_distance
    )
    log = controller.log
    return {
        "positions": result.state[:, :2],
        "d_r": np.array([entry.equidistant_deviation for entry in log]),
        "time": result.time,
        "yaw_rate": result.applied_control[:, 1],
        "minimum_clearance": metrics.minimum_clearance,
        "collided": metrics.collided,
        "switch_count": controller.switch_count,
        "in_orbit": np.array(
            [entry.mode is CircumnavMode.ORBIT for entry in log]
        ),
    }


def plot_comparison(
    config: CircumnavEnvConfig,
    rl_run: dict,
    baseline_run: dict,
    path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_xy, ax_d, ax_u) = plt.subplots(1, 3, figsize=(16, 5))

    for obstacle in config.obstacles:
        boundary = obstacle.boundary_polyline()
        ax_xy.fill(*boundary.T, color="0.8", zorder=1)
        ax_xy.plot(*boundary.T, color="0.4", lw=1, zorder=2)
        equidistant = obstacle.equidistant_polyline(config.safety_distance)
        ax_xy.plot(*equidistant.T, color="tab:blue", lw=0.8, alpha=0.6, zorder=2)

    ax_xy.plot(*baseline_run["positions"].T, color="k", lw=1.1, label="relay (baseline)")
    ax_xy.plot(*rl_run["positions"].T, color="tab:orange", lw=1.1, label="PPO")
    ax_xy.set_aspect("equal")
    ax_xy.set(xlabel="x, m", ylabel="y, m")
    ax_xy.legend(loc="upper left", fontsize=8)

    baseline_time = np.arange(len(baseline_run["d_r"])) * config.control_period
    rl_time = np.arange(len(rl_run["d_r"])) * config.control_period
    ax_d.plot(baseline_time, baseline_run["d_r"], color="k", lw=0.8, label="relay")
    ax_d.plot(rl_time, rl_run["d_r"], color="tab:orange", lw=0.8, label="PPO")
    ax_d.set(xlabel="t, s", ylabel=r"$d_R(t)$, m")
    ax_d.legend(loc="upper right", fontsize=8)
    ax_d.grid(alpha=0.3)

    # baseline_run["yaw_rate"] is logged on the fine integration_step grid,
    # not the control_period grid d_r/baseline_time use -- plot it against
    # its own time array rather than reindexing by the wrong stride.
    ax_u.plot(baseline_run["time"], baseline_run["yaw_rate"], color="k", lw=0.6)
    ax_u.plot(rl_time, rl_run["yaw_rate"], color="tab:orange", lw=0.6)
    ax_u.set(xlabel="t, s", ylabel=r"$u(t)=\omega$, rad/s")
    ax_u.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="ellipse")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-jitter", action="store_true", help="train from a fixed start pose instead of a jittered one")
    parser.add_argument("--load", type=Path, help="skip training, evaluate a saved model instead")
    parser.add_argument("--output", type=Path, default=Path("simulation_output"))
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    config = make_env_config(
        arguments.shape, arguments.duration, jitter=not arguments.no_jitter
    )

    if arguments.load is not None:
        from stable_baselines3 import PPO

        model = PPO.load(arguments.load)
    else:
        model = train(config, arguments.timesteps, arguments.n_envs, arguments.seed)
        model_path = arguments.output / f"ppo_{arguments.shape}.zip"
        model.save(model_path)
        print(f"model saved to:       {model_path}")

    rl_run = rollout_policy(model, config)
    baseline_run = run_baseline(config)

    print(f"{'relay (baseline)':>18}  max|d_R|={np.max(np.abs(baseline_run['d_r'])):7.4f}  "
          f"min clearance={baseline_run['minimum_clearance']:6.3f}  "
          f"collided={baseline_run['collided']}  switches={baseline_run['switch_count']}")
    print(f"{'PPO':>18}  max|d_R|={np.nanmax(np.abs(rl_run['d_r'])):7.4f}  "
          f"min clearance={rl_run['minimum_clearance']:6.3f}  "
          f"collided={rl_run['collided']}")

    plot_path = arguments.output / f"{arguments.shape}_rl_compare.png"
    plot_comparison(
        config,
        rl_run,
        baseline_run,
        plot_path,
        title=f"{arguments.shape}: PPO (postановка a) vs relay baseline",
    )
    print(f"comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
