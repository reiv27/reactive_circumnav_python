# rl/ — step-able environment for RL experiments

See [docs/RL_plan.md](../../../docs/RL_plan.md) for the motivation and the
open design questions. This package is the "postановка (a)" sketch from that
plan: a learning exercise to get a step-able `gymnasium.Env` and a reference
RL baseline (PPO, `stable-baselines3`) working end to end, starting on a
single ellipse and with no delay. It is **not** a claim that RL should
replace `PredictorController` — see RL_plan.md section 5 for why it
structurally cannot beat it in this nominal, delay-free case.

Requires the `rl` extra (`pip install -e ".[rl]"` or `uv sync --extra rl`):
`gymnasium`, `stable-baselines3`, `torch`.

## env.py — `CircumnavEnv(config, actuator_factory=None)`

- Reuses `DubinsModel`, `Obstacle`, `Actuator`, `CircularVisibilitySensor`
  unchanged — only the control law is replaced by the policy, nothing below
  it. `step()` advances exactly one `control_period`, holding the action over
  `substeps_per_control_period` integration steps, same ZOH structure as
  `Simulator.run`.
- `CircumnavEnvConfig` is a frozen dataclass; mirrors the fields of
  `ReactiveCircumnavScenario` that matter for this task (speed, yaw rate,
  safety distance, sensor, timing, initial pose) plus RL-specific knobs
  (`action_change_penalty` = lambda_u, `collision_penalty`, `history_len`,
  start-pose jitter).
- **Action**: continuous, `Box(-1, 1)`, scaled by `yaw_rate_magnitude`.
  Postановка (a) pairs naturally with continuous + PPO/SAC per RL_plan.md
  section 3; the discrete bang-bang variant for postановка (b) (residual) is
  not implemented here.
- **Observation**: `[P1_visible, P1_distance, sin(bearing1), cos(bearing1),
  P2_visible, P2_distance, sin(bearing2), cos(bearing2), *last history_len
  actions]`. `P2`'s disk-center projection reuses the same pure geometry
  `ReactiveCircumnavController` uses for `V` — it is not part of the frozen
  control law, just how P2 is exposed to sensing.
- **Reward**: `-|d_R| - lambda_u*|u_t - u_{t-1}|` (`d_R` approximated as
  `P1.distance - safety_distance`, i.e. mode `C` only — fine for a single
  convex obstacle, where mode `G` is never entered) plus a terminal
  `collision_penalty` when the *true* clearance (ground-truth geometry, not
  the sensor) drops below `collision_distance`. `terminated` on collision,
  `truncated` at `duration`.
- No sensing/actuation delay yet: `actuator_factory` defaults to a plain
  `IdealActuator`. The hook exists (same shape as `PredictorController`'s)
  so the delay/model-mismatch robustness experiments RL_plan.md motivates
  this with can be added without changing the env's interface.

Tests: `tests/test_rl_env.py` (gated on `gymnasium` being installed via
`pytest.importorskip`).

## Other

- `circumnav.examples.rl_train` (`circumnav-rl-train`) trains PPO on this env
  and plots it against the frozen relay baseline on the same scene — see
  `examples/AGENTS.md`.
