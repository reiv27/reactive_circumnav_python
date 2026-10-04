# models/ — plant, actuators, obstacle geometry

Pure mathematics. Nothing here may import from `sensors`, `controllers`,
`simulation`, `scenarios` or `analysis`.

## dubins.py
- `DubinsState(x, y, heading)`, `DubinsCommand(speed, yaw_rate)`,
  `DubinsLimits(min_speed, max_speed, max_yaw_rate)` — frozen, validate finiteness.
- `DubinsModel.propagate(state, command, dt)` is the **exact** solution for a
  command held over `dt` (arc, or straight line if `|ω| < 1e-10`). Do not replace
  it with Euler/RK: the "trajectory is independent of `h`" property rests on it.
- `normalize_angle` wraps to `[-π, π)`.

## actuator.py
- `Actuator` protocol: `reset()`, `apply(command, dt=None) -> ActuatorOutput`.
  The engine calls `apply` **every integration step** with the held request.
- `IdealActuator(limits)`: memoryless hard clip. `ActuatorOutput` carries
  `requested`, `applied`, saturation flags.
- `DelayedActuator(inner, delay, integration_step, initial_command)`: transport
  delay; `delay` must be a multiple of `integration_step`; the line is pre-filled
  with `initial_command`. Covers `τ_a` and `τ_c` (indistinguishable in this loop).
- `LagActuator(inner, time_constant, initial_command)`: exact first-order lag,
  hands the plant the **interval average**; `time_constant = 0` is a pass-through;
  needs `dt > 0` otherwise.
- Wrappers delegate clipping to `inner` and restore `requested` to the original
  request. Composition order in the scenario: delay → lag → clip.
- Adding an actuator: honour `reset()`, advance by exactly `dt` per `apply`, keep
  `requested` unchanged in the output, test in `tests/test_actuator_dynamics.py`.

## obstacles.py
- `Obstacle` protocol: `closest_point`, `first_intersection`, `is_boundary_visible`,
  `boundary_polyline`, `equidistant_polyline(distance)`,
  `minimum_equidistant_curvature(distance)`.
- `SegmentObstacle(start, end)`, `CircleObstacle(center, radius)`,
  `EllipseObstacle(center, semi_axis_x, semi_axis_y, angle)`; all take
  `obstacle_id`. Ellipse closest point = coarse scan + ternary search.
- Equidistant curves: circle `R + ρ₀`; ellipse = outward normal offset; segment =
  stadium. Tightest curvature radius of any convex equidistant is `ρ₀`.
- `check_turning_feasibility(rho_0, turning_radius)`: raises unless `ρ₀ ≥ R_min`.
- Spacing: `boundary_gap`, `equidistant_gap`,
  `check_equidistant_spacing(obstacles, rho_0, max_gap, samples)` — neighbouring
  equidistant curves must intersect or be ≤ `1.5·R_min` apart.
- `make_obstacles([((x0, y0), (x1, y1)), ...])` builds segments.
