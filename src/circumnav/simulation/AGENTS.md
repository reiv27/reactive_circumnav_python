# simulation/ — the sampled-data loop

## engine.py
- `SimulationConfig(duration, integration_step, control_period)`: all finite and
  positive; `duration` and `control_period` must be integer multiples of
  `integration_step` (`ValueError` otherwise).
- `Simulator(model, actuator, config).run(initial_state, controller) -> SimulationResult`.
- Per integration step `k`: at control instants (`k % stride == 0`, `k < N`) call
  `controller.compute`; then **every step** call `actuator.apply(requested, h)`;
  record; propagate the plant with `applied` over `h`. The request is a
  zero-order hold. There is no computation delay in the baseline.
- Do not move actuator evaluation back to control instants only: stateful
  actuators (delay, lag) depend on being advanced every `h`.
- Do not put logging or controller-specific logic here.

## result.py
- `SimulationResult(time, state, requested_control, applied_control, saturated)`,
  all aligned on the `h` grid, shapes validated; properties `x`, `y`, `heading`;
  `save_npz(path)`.
- Controller diagnostics live in the controller's own log, not here (different
  rate, controller-specific content).

Tests: `tests/test_simulation.py`.
