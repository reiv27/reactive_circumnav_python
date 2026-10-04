# AGENTS.md — Dubins circumnavigation simulation stand

Read this first. Each package under `src/circumnav/` has its own `AGENTS.md` with
the contracts and invariants of that package; this file holds what is true
everywhere.

## What this project is

A Python simulation stand (no hardware) for a **Dubins vehicle** with a
**range-limited omnidirectional (2π) circular-visibility sensor**, running a
**reactive circumnavigation law** that rides the equidistant curve around
obstacles (circles, ellipses, segments). It is a control-engineering research
environment: the law is fixed, everything around it (timing, delays, obstacles,
sensor, actuator) is meant to be varied and measured.

Human-facing docs: [README.md](README.md) (usage guide),
[docs/delay_report.md](docs/delay_report.md) (delay study, Russian).
`README_CONTROL_SIMULATOR.md` and `README_SENSOR_MODEL.md` are design notes that
predate the implementation; the code is the source of truth.

## Hard rules

1. **The control law is mathematically frozen.** `ReactiveCircumnavController`
   (modes `C`/`G`, relay `u = ω·sign(ḋ_R + κ·sat(d_R; ±δ))`, switching
   conditions) must not be changed unless the user explicitly asks. Add behaviour
   *around* it (wrappers, actuators, scenarios), as `DelayedController` and the
   actuator blocks do.
2. **`ρ₀ ≥ R_min = v/ω` is enforced** before every run (`check_turning_feasibility`).
   Do not weaken it silently; `require_feasible_turning=False` is the explicit opt-out.
3. **The controller never reads obstacle geometry.** It gets pose from the
   engine and obstacle points only from the sensor (`P₁`, `P₂`).
4. **Do not touch the dynamics from analysis code.** `analysis/` reads results and
   logs only.
5. **Never invent a sensor point at maximum range.** "Nothing visible" is `None`.
6. **Do not commit or push unless asked.** Default branch is `main`. The user
   commits themselves.

## Notation (use these names and symbols)

| symbol | meaning | code |
|---|---|---|
| `v` | forward speed | `forward_speed` |
| `ω` | yaw-rate magnitude (relay is `±ω`) | `yaw_rate_magnitude` |
| `R_min = v/ω` | minimum turning radius | `turning_radius` |
| `ρ₀` | commanded standoff | `safety_distance`, property `rho_0` (formerly `d`) |
| `ρ(t)` | distance to nearest visible obstacle point | log `obstacle_range` |
| `d_R(t)` | **regulated variable**: distance to the equidistant curve being tracked | log `equidistant_deviation` |
| `R_s` | sensor radius | `sensor_range` |
| `P₁`, `P₂`, `V` | nearest point, second point, frozen turning-disk centre | sensor / `orbit_center` |
| `h` | integration step | `integration_step` |
| `T` | control period, `T = N·h` | `control_period` |
| `τ_s`, `τ_a(+τ_c)`, `τ` | sensing delay, actuation delay, actuator lag | `sensing_delay`, `actuation_delay`, `actuator_time_constant` |

`d_R = ρ − ρ₀` in mode `C`; `d_R = R_min − ‖r − V‖` in mode `G`. They are equal
at the `C→G` switch (continuous); there is a small genuine jump at `G→C`.
Plot labels use LaTeX mathtext (`$d_R(t)$`), never plain `d_R`.

## Commands

`python` does not exist on the dev machine; use `python3`. The package is not
installed by default, so set `PYTHONPATH=src`.

```bash
# full suite, ~35 s
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider

# one file
PYTHONPATH=src python3 -m pytest -q tests/test_actuator_dynamics.py

# demo run
PYTHONPATH=src python3 -m circumnav.examples.reactive_circumnav --obstacle cluster

# delay experiments
PYTHONPATH=src python3 -m circumnav.examples.delay_sweep --shape ellipse
PYTHONPATH=src python3 -m circumnav.examples.delay_compare --shape cluster --speed 2
```

Runs of 60 s take seconds each; sweeps use a process pool (`--workers`). Anything
longer than the 2-minute tool timeout (video rendering) must run in the background.

## Conventions

- Python ≥ 3.10, `from __future__ import annotations`, full type hints, NumPy arrays
  typed `NDArray[np.float64]`.
- **Frozen dataclasses** for configs, states, commands, results, scenarios. No
  global mutable state; scenarios are reproducible from their fields alone.
- **Protocols** (`Controller`, `Actuator`, `Sensor`, `Obstacle`) define seams;
  wrappers delegate (`__getattr__` for `DelayedController`, `.inner` for actuators).
- Comments are sparse: docstrings state contracts and the reason for non-obvious
  choices; no narration of what the code does. Match the surrounding density.
- Validation errors are `ValueError` with the offending value in the message.
- Every new behaviour gets a test in `tests/` (one file per module). Run the full
  suite before reporting done.
- Report results honestly: if a measurement does not reproduce or a claim was
  wrong, say so and fix the doc (`docs/delay_report.md` carries such a correction).

## Output locations

- `simulation_output/` — all generated data, figures, videos (git-ignored).
- **Never write results to `/tmp`**: it is wiped on boot on this machine.
- Figures worth keeping go to `docs/figures/` (tracked).

## Known pitfalls

- `ReactiveCircumnavConfig.dead_zone_gain` defaults to the reference `0.025`, which
  converges far too slowly. The scenario default is `6.0`; keep
  `closing_speed_ratio = κ·δ/v ≈ 0.3`.
- `d_R(t)` shows small beaks at mode switches: a discretisation artefact of size
  ≈ `2·v·T`, shrinking proportionally with `T`. It is not a law defect.
- With a memoryless actuator trajectories do not depend on `h`; with the lag they
  do (weakly). Delays must sit on the right grid (`τ_s` multiple of `T`, `τ_a`
  multiple of `h`).
- In mode `C`, `τ_s` and `τ_a+τ_c` behave almost identically; a lag is *not* a delay.
- The "peak |d_R|" metric is noisy in how `τ` fits into `T`; do not over-read single
  points. On `gap_cluster` it never settles to zero (baseline peak 0.11 m).

## Where things live

```
src/circumnav/
  models/        dubins.py, actuator.py, obstacles.py        -> models/AGENTS.md
  sensors/       base.py, circular.py                        -> sensors/AGENTS.md
  controllers/   reactive.py (the law), delayed.py, ...      -> controllers/AGENTS.md
  simulation/    engine.py, result.py                        -> simulation/AGENTS.md
  scenarios/     reactive.py, heading.py                     -> scenarios/AGENTS.md
  analysis/      metrics.py, animation.py                    -> analysis/AGENTS.md
  examples/      CLIs: demos, delay_sweep, delay_compare     -> examples/AGENTS.md
tests/                                                       -> tests/AGENTS.md
docs/            delay_report.md, figures/
```

## Git history facts

- Remote default branch: `main`. The old prototype (`python_sim/`) is at commit
  `109a779`; the ROS 2 / Gazebo stack (`gazebo_sim/`) at `bbfa2b3`.
- A local branch `develop` (4 commits) holds an earlier prototype with dynamic
  obstacles and α-speed sweep data; it is not merged. Do not delete it.
