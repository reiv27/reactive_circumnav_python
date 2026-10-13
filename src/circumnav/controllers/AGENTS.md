# controllers/ — control laws and wrappers

`Controller` protocol (`base.py`): `reset(initial_state)` and
`compute(time, state) -> DubinsCommand`. Called once per control period.

## reactive.py — the law (frozen, see root AGENTS.md rule 1)
- `CircumnavMode.APPROACH = "C"`, `ORBIT = "G"`.
- Per update: `P₁` from the sensor; turning-disk centre
  `V = P₁ + (ρ₀ + R_min)·unit(r − P₁)`; `P₂` from the sensor. `C→G` when
  `‖V − P₂‖ < ‖V − P₁‖` (then `V` is frozen as `orbit_center`); `G→C` when the
  robot leaves the angle `P₁–V_A–P₂` (only evaluated when `P₂` exists).
- `d_R = ρ − ρ₀` (C) or `R_min − ‖r − V‖` (G); `ḋ_R = v·(line of sight · heading)`;
  `u = ω·sign(ḋ_R + κ·sat(d_R; ±δ))`; speed constant.
- No `P₁` in range: hold course (`yaw_rate = 0`), log NaN.
- `ReactiveLogEntry` is one entry per control update (the `T` grid):
  `time, mode, obstacle_range (ρ), equidistant_deviation (d_R), range_rate,
  yaw_rate_command, speed_command, target_point, primary_point, secondary_point,
  disk_center, orbit_center`. `d_R` is a **single** field — do not split it again.
- Public attributes used elsewhere: `.log`, `.mode`, `.switch_count`,
  `.orbit_center`, `.config`, `.sensor`, `.obstacles`.
- `ReactiveCircumnavConfig.dead_zone_gain` defaults to the reference `0.025`;
  scenarios override it (`6.0`).

## delayed.py — `DelayedController(inner, delay_steps)`
- Feeds `inner` the state from `delay_steps` control updates ago (the first
  `delay_steps` updates see the earliest state). `time` is **not** delayed, so logs
  stay on the real clock. `reset` clears the history.
- `__getattr__` forwards everything else to `inner`, so animation and metrics work
  on the wrapper unchanged. Keep it that way; do not copy the log.

## predictor.py — `PredictorController(inner, delay_steps, control_period, model=None, integration_step=None, actuator_factory=None)`
- Compensates `tau_s` by reconstructing `x(t)` instead of feeding `inner` the
  raw stale state: replays `DubinsModel.propagate` through a stream of
  *applied* commands, starting from the same stale anchor measurement
  `DelayedController` would use.
- No `actuator_factory` (stage 1a): replays its own issued-command history
  directly, one `control_period`-sized step each. Exact while
  `actuation_delay == 0` and `actuator_time_constant == 0`.
- With `actuator_factory` (stage 1b, what `ReactiveCircumnavScenario.run()`
  always passes — `self.build_actuator`): keeps a persistent **shadow**
  actuator, stepped on the `integration_step` grid every call, fed this
  controller's own commands. Since the real actuator chain is a pure,
  deterministic function of its command stream (never reads plant state),
  the shadow stays in lockstep with the real one by construction, so
  replaying the shadow-applied stream reconstructs `x(t)` exactly even when
  `actuation_delay`/`actuator_time_constant` are non-zero.
- **Scope boundary, both cases**: this only undoes the *sensing* delay's
  effect on the switching decision (mode `C`/`G` included). It does **not**
  compensate `tau_a`/lag themselves — that needs a forward, queued-command
  prediction (a different, harder mechanism: what's already committed but
  not yet applied), not implemented. With `delay_steps == 0` this is
  correctly a no-op regardless of `actuator_factory` — there is nothing to
  reconstruct when sensing is already current, and it must not pretend to
  fix `tau_a`/lag on their own.
- Same wrapper contract as `DelayedController`: `time` is never delayed,
  `__getattr__` forwards everything else to `inner`.

## Other
- `heading.py` — proportional heading hold (baseline experiment).
- `open_loop.py` — `ConstantController` for plant verification.

Tests: `tests/test_reactive_controller.py`, `tests/test_delayed_controller.py`,
`tests/test_predictor_controller.py`.
