# tests/

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider`
(119 tests, ~50 s). One file per module:

| file | covers |
|---|---|
| `test_dubins_model.py` | exact ZOH propagation, limits |
| `test_actuator.py` | `IdealActuator` clipping and flags |
| `test_actuator_dynamics.py` | `DelayedActuator`, `LagActuator` (exact step response, interval average, initial command, saturation after filtering), scenario wiring |
| `test_obstacles.py`, `test_curved_obstacles.py` | geometry, equidistant curves, feasibility, spacing |
| `test_circular_sensor.py` | `P₁`/`P₂`, range, occlusion, exclusion |
| `test_reactive_controller.py` | modes, switching, reset, log |
| `test_delayed_controller.py` | state shift, `time` not delayed, forwarding, scenario wrapping |
| `test_simulation.py` | engine timing and results |
| `test_fleet.py` | start poses on the equidistant curve, `FleetScenario` validation, per-vehicle equality with a single run, `FleetResult`, separation metrics |
| `test_neighbour_visibility.py` | `NeighbourSensor` (range, occlusion), visibility arrays, dropout metrics, ellipse fleets |
| `test_scenarios.py` | scenario assembly and validation |

## Writing tests
- Use concrete coordinates and check geometry by hand; derive expected numbers
  from the formulas (e.g. exact lag step response) rather than from a run.
- A test that needs `circumnav` imports works only with `PYTHONPATH=src` (or an
  editable install) — a wall of `ModuleNotFoundError` at collection means that.
- Keep runs short (`duration` of a few seconds); the full suite should stay
  under a minute.
- New behaviour needs a test; changing the control law's numbers should make an
  existing test fail first, which is the point.
