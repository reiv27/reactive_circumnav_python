# scenarios/ — reproducible experiment configurations

A scenario is a **frozen dataclass** that owns every parameter and wires model,
actuator, controller and `Simulator` in `run()`. No global state, no hidden
defaults outside the dataclass.

## reactive.py
- `ReactiveCircumnavScenario`: see the field table in the README. Defaults:
  `v=2, ω=1 (R_min=2), ρ₀=3, T=0.02, h=0.01, duration=60, R_s=12, δ=0.1, κ=6.0`,
  `max_speed=2, max_yaw_rate=2`, one circle `R=4`, all delays `0.0`.
- `validate()` enforces `ρ₀ ≥ R_min` (unless `require_feasible_turning=False`).
- `build_controller()` returns the bare controller; `build_actuator()` returns
  `Ideal`, optionally wrapped in `LagActuator` then `DelayedActuator` (initial
  command `(v, 0)`); `run()` wraps the controller in `DelayedController` when
  `sensing_delay > 0` and returns `(result, controller)` — the wrapper in that case.
- `sensing_delay_steps` raises unless `sensing_delay` is a multiple of
  `control_period`.
- `gap_cluster(rho_0, turning_radius, scale)`: 4 circles + 3 ellipses; one channel
  circle is placed analytically at separation `2ρ₀ + 0.75·R_min`; validates
  feasibility and that neighbouring equidistants intersect or are ≤ `1.5·R_min`
  apart. Constants `MAX_EQUIDISTANT_GAP_FACTOR`, `CHANNEL_WIDTH_FACTOR`.
- Use `dataclasses.replace(scenario, ...)` for sweeps; fields are picklable, so
  scenarios can be sent to a process pool.

## heading.py
`HeadingControlScenario`: proportional heading hold, the minimal loop check.

## fleet.py
- `equidistant_start_poses(obstacle, rho_0, count, phase)`: poses on the
  equidistant curve, evenly spaced in arc length, heading along the
  **counter-clockwise** tangent (the law keeps the obstacle on the left; clockwise
  is not a stable circulation). Works on any obstacle with a closed equidistant.
- `FleetScenario(base, robot_count, phase)`: `base` is a
  `ReactiveCircumnavScenario` with **exactly one obstacle** and **all delays off**
  (both enforced with `ValueError`). `run()` returns
  `(FleetResult, tuple[controllers])`, running each vehicle through
  `base.run()` from its own pose. Valid only because vehicles do not interact.
- `neighbour_sensor(max_range=None)` builds the `NeighbourSensor` from `base.sensor_range`
  and `base.check_occlusion`; visibility is analysis only.
- Do not add coupling by looping `base.run()`; it needs a synchronous loop.

## Adding a scenario
Frozen dataclass, `run()` returns what its callers need, tests in
`tests/test_scenarios.py`.
