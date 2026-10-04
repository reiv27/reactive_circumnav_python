# Reactive circumnavigation — Dubins vehicle simulation stand

A simulation environment for studying a reactive obstacle-circumnavigation law
on a Dubins vehicle. The vehicle carries a range-limited, omnidirectional
sensor, measures relative distances to obstacle boundaries, and rides the
**equidistant curve** at a commanded standoff. The stand is meant for
control-engineering research: the law is fixed, and everything around it
(timing, delays, obstacles, sensor, actuator) can be varied and measured.

Reference paper DOI: <https://doi.org/10.1016/j.robot.2024.104649>

**Contents**

1. [Notation](#notation)
2. [Install and quick start](#install)
3. [How the simulator works](#how-the-simulator-works) — architecture, timing, data flow
4. [Scenario reference](#scenario-reference) — every parameter
5. [Defining your own experiment](#defining-your-own-experiment)
6. [The control law](#the-control-law)
7. [Sensor model](#sensor-model)
8. [Delays in the loop](#delays-in-the-loop)
9. [Several vehicles](#several-vehicles)
10. [Reading the results](#reading-the-results) — result, log, metrics
11. [Animation](#animation)
12. [Command-line tools](#command-line-tools)
13. [Extending the stand](#extending-the-stand)
14. [Tests, layout, troubleshooting](#running-the-tests)

Related documents: [docs/delay_report.md](docs/delay_report.md) (how delays affect
the relay's sliding mode, in Russian), [AGENTS.md](AGENTS.md) (map of the code for
coding agents).

---

## Notation

| symbol | meaning | code |
|---|---|---|
| `v` | forward speed (constant) | `forward_speed` |
| `ω` | yaw-rate magnitude (the law is bang-bang: `u = ±ω`) | `yaw_rate_magnitude` |
| `R_min = v / ω` | minimum turning radius | `scenario.turning_radius` |
| `ρ₀` | commanded standoff from the obstacle boundary | `safety_distance` |
| `ρ(t)` | current distance to the nearest visible obstacle point | log field `obstacle_range` |
| `d_R(t)` | regulated variable: distance to the equidistant curve being tracked | log field `equidistant_deviation` |
| `R_s` | sensor perception radius | `sensor_range` |
| `P₁` | nearest visible obstacle point | sensor output |
| `P₂` | visible point nearest the turning disk centre | sensor output |
| `V` | turning-disk centre, frozen on entering mode `G` | `orbit_center` |

The **equidistant curve** is the level set `{q : dist(q, obstacle) = ρ₀}` —
the curve the law is asked to follow. It is drawn in blue in every plot.

`d_R(t)` is the distance to the curve the law is tracking *at that moment*.
In mode `C` that is the true equidistant, so `d_R = ρ − ρ₀`. In mode `G` it is
the curve's local approximation — the circle of radius `R_min` about the
frozen disk centre `V` — so `d_R = R_min − ‖r − V‖`. Either way `d_R` is the
regulated variable, the one the relay switches on, and driving it to zero is
what "following the equidistant" means.

### The feasibility condition `ρ₀ ≥ R_min`

Offsetting a convex boundary outward raises every radius of curvature by
exactly `ρ₀`, so the equidistant curve bends tightest around a degenerate
obstacle, where its curvature radius is exactly `ρ₀`. A vehicle that cannot
turn tighter than `R_min` therefore cannot hold the standoff unless

```
ρ₀ ≥ R_min
```

This is checked before every run and a violating scenario raises
`ValueError`. Run the infeasible regime deliberately with
`require_feasible_turning=False` (CLI: `--allow-infeasible-turning`).

---

<a id="install"></a>
## Install and quick start

```bash
pip install -e ".[dev,viz]"
```

`numpy` is the only runtime dependency; `viz` adds `matplotlib` for plots and
animation (`.mp4` additionally needs `ffmpeg` on the `PATH`). Without
installing, prefix commands with `PYTHONPATH=src` and use `python3 -m ...`.

```bash
circumnav-reactive-demo --obstacle cluster --animate run.mp4
```

Prints the run summary and writes an animated playback. Other shapes:
`--obstacle circle | ellipse | wall | cluster`.

```
rho_0 / R_min:        3.000 / 2.000 m  (ok)
equidistant curvature:4.000 m
closing speed ratio:  0.300 (keep near 0.3)
mode at end:          C
mode switch count:    54
minimum clearance:    2.8801 m
collision detected:   False
```

---

## How the simulator works

### Architecture

```
                    ┌──────────── Scenario (frozen dataclass) ────────────┐
                    │  builds and wires everything, owns all parameters   │
                    └──────────────────────────────────────────────────────┘
                                          │ run()
                                          ▼
   x(t_j) ─► [DelayedController: tau_s] ─► ReactiveCircumnavController ─► request u
                                                │ uses            (mode C / G, relay)
                                                ▼
                                   CircularVisibilitySensor ─► Obstacles (exact geometry)

   request u (held, ZOH) ─► [DelayedActuator: tau_a] ─► [LagActuator: tau] ─► IdealActuator (clip)
                                                                                    │ applied u
                                                                                    ▼
                                                              DubinsModel.propagate (exact ZOH) ─► x
```

| layer | package | knows about |
|---|---|---|
| plant | `models/dubins.py` | only kinematics |
| geometry | `models/obstacles.py` | shapes, closest points, equidistant curves |
| actuator | `models/actuator.py` | limits, delay, lag |
| sensor | `sensors/` | geometry and visibility, nothing about control |
| controller | `controllers/` | pose + sensor; never reads obstacle geometry directly |
| loop | `simulation/engine.py` | timing only |
| scenario | `scenarios/` | wiring and defaults |
| analysis | `analysis/` | results and logs, never the dynamics |

The controller receives the vehicle's own pose (equivalent to onboard odometry).
Obstacle points reach it only through the sensor's `P₁`/`P₂` measurements.

### Timing model

Two grids, both fixed:

- `h = integration_step` — the plant and the actuator advance on this grid;
- `T = control_period = N·h` — the controller runs on this grid.

At every control instant `t_j` the controller reads `x(t_j)` and produces a
request, which a zero-order hold keeps constant until `t_{j+1}`. There is no
computation delay in the baseline: the request acts from `t_j`. `duration` must be
a multiple of `h`, and `T` a multiple of `h`; the engine raises `ValueError`
otherwise.

The plant is propagated with the **exact** solution for a held command (circular
arc, or a straight line when `|ω|` is tiny). Consequences:

- with a memoryless actuator the trajectory does not depend on `h` at all (checked
  to ~1e-12 m); only `T` matters;
- the actuator is evaluated every `h`, so delay and lag blocks work on the `h`
  grid, independent of `T`;
- a stateful actuator (the lag) breaks the exactness: `h` becomes a small accuracy
  parameter again.

### One simulation step

```
for each integration step k (time t_k = k·h):
    if k is a control instant:        request = controller.compute(t_k, x_k)
    output  = actuator.apply(request, h)      # delay -> lag -> clip, one h-step
    log     (x_k, request, output.applied, saturation)
    x_{k+1} = plant.propagate(x_k, output.applied, h)
```

---

## Scenario reference

`ReactiveCircumnavScenario` is the single entry point. Every field has a default;
override with keyword arguments or `dataclasses.replace(scenario, field=value)`.

| field | default | meaning |
|---|---|---|
| `duration` | `60.0` s | simulated time; multiple of `integration_step` |
| `integration_step` | `0.01` s | `h` |
| `control_period` | `0.02` s | `T`; multiple of `h` |
| `initial_x`, `initial_y` | `-9.0`, `-5.0` m | start position |
| `initial_heading` | `-1.268` rad | start heading (tangent to the default scene) |
| `forward_speed` | `2.0` m/s | `v` |
| `yaw_rate_magnitude` | `1.0` rad/s | `ω`, so `R_min = v/ω` |
| `safety_distance` | `3.0` m | `ρ₀`; must be `≥ R_min` |
| `sensor_range` | `12.0` m | `R_s` |
| `exclusion_distance` | `1.0` m | neighbourhood of `P₁` excluded when searching `P₂` |
| `dead_zone` | `0.1` m | `δ` in `sat(d_R; ±δ)` |
| `dead_zone_gain` | `6.0` | `κ`; see [tuning](#tuning-note-closing_speed_ratio) |
| `check_occlusion` | `True` | enable occlusion and self-occlusion |
| `max_speed`, `max_yaw_rate` | `2.0`, `2.0` | actuator clip limits |
| `obstacles` | one circle `R=4` | tuple of obstacles |
| `require_feasible_turning` | `True` | enforce `ρ₀ ≥ R_min` |
| `sensing_delay` | `0.0` s | `τ_s`; multiple of `T` |
| `actuation_delay` | `0.0` s | `τ_a (+ τ_c)`; multiple of `h` |
| `actuator_time_constant` | `0.0` s | lag `τ` |

Derived properties: `turning_radius`, `rho_0`, `closing_speed_ratio`,
`equidistant_curvature`, `sensing_delay_steps`. Methods: `validate()`,
`build_obstacles()`, `build_controller()`, `build_actuator()`, `run()`.
`run()` returns `(SimulationResult, controller)`; the controller is a
`DelayedController` wrapper when `sensing_delay > 0` (it forwards `.log`, `.mode`,
`.switch_count`, `.config`).

`HeadingControlScenario` (`scenarios/heading.py`) is the minimal baseline
experiment (proportional heading hold) used to verify the loop; run it with
`circumnav-heading-demo`.

---

## Defining your own experiment

A scenario is a frozen dataclass — plain Python, no global state:

```python
import numpy as np
from circumnav.models.obstacles import CircleObstacle, EllipseObstacle
from circumnav.scenarios.reactive import ReactiveCircumnavScenario

scenario = ReactiveCircumnavScenario(
    duration=120.0,
    initial_x=-9.0, initial_y=-5.0, initial_heading=-1.268,
    forward_speed=2.0,        # v
    yaw_rate_magnitude=1.0,   # omega  -> R_min = 2.0 m
    safety_distance=3.0,      # rho_0  -> rho_0 >= R_min, ok
    sensor_range=12.0,        # R_s
    obstacles=(
        CircleObstacle(center=np.array([0.0, 0.0]), radius=4.0, obstacle_id=0),
        EllipseObstacle(
            center=np.array([14.0, 3.0]), semi_axis_x=6.0, semi_axis_y=3.0,
            obstacle_id=1, angle=np.pi / 6,
        ),
    ),
)

result, controller = scenario.run()
```

Vary one thing and compare:

```python
from dataclasses import replace

slow = replace(scenario, control_period=0.05)
delayed = replace(scenario, actuation_delay=0.1)
```

### Obstacle shapes

| class | arguments | equidistant curve |
|---|---|---|
| `CircleObstacle` | `center`, `radius` | concentric circle `R + ρ₀` |
| `EllipseObstacle` | `center`, `semi_axis_x`, `semi_axis_y`, `angle` | outward normal offset |
| `SegmentObstacle` | `start`, `end` | stadium (needs `ρ₀ ≥ R_min`; its end caps have curvature exactly `ρ₀`) |

Build several segments at once with `make_obstacles([((x0, y0), (x1, y1)), ...])`.

### Ready-made multi-obstacle scene

```python
from circumnav.scenarios.reactive import gap_cluster

obstacles = gap_cluster(rho_0=3.0, turning_radius=2.0)
```

Four circles and three ellipses in a ring. The builder enforces two rules:

- `ρ₀ ≥ R_min`;
- every obstacle's nearest equidistant curve either intersects its own or
  leaves a channel no wider than `1.5 · R_min`. Two equidistant curves meet
  once their obstacles are closer than `2ρ₀`, so the admissible boundary
  separation runs up to `2ρ₀ + 1.5·R_min`. One obstacle is placed to leave a
  channel of exactly `0.75 · R_min`, so the rule is demonstrated rather than
  satisfied only by overlap.

## The control law

Two modes, unchanged from the reference implementation:

- **`C` (approach / follow)** — regulates `d_R = ρ(t) − ρ₀`, driving the
  vehicle onto the equidistant curve of the nearest obstacle.
- **`G` (gap mode)** — entered when the turning disk reaches a second obstacle
  point, i.e. when `‖V − P₂‖ < ‖V − P₁‖`. The disk centre `V` freezes on
  entry, and the vehicle rides a circle of radius `R_min` about it until the
  robot leaves the angle `P₁–V–P₂`. This is what carries it across the gaps
  between neighbouring obstacles.

The steering command is a relay:

```
d_R = (mode C) ρ − ρ₀     |  (mode G) R_min − ‖r − V‖
ḋ_R = v · ((r − p)/‖r − p‖) · e
u   = ω · sign( ḋ_R + dead_zone_gain · sat(d_R, ±dead_zone) )
```

`p` is the tracked point: `P₁` in mode `C`, the frozen `V` in mode `G`.

### Tuning note: `closing_speed_ratio`

`dead_zone_gain · dead_zone` is the radial closing speed the sliding surface
asks for, and the vehicle can deliver at most `v`. The ratio of the two is
`scenario.closing_speed_ratio`; keep it near **0.3**. The reference value
`dead_zone_gain = 0.025` puts it near 0.001, and the approach onto the
equidistant curve then takes hundreds of times longer than a typical run —
the vehicle locks onto the first tangential orbit it reaches instead of
converging to `ρ₀`. `ReactiveCircumnavConfig` still defaults to the reference
value; the scenario defaults to a working one.

## Sensor model

`CircularVisibilitySensor` is omnidirectional but bounded by `R_s`. It works
on exact obstacle geometry rather than a ray scan, so results do not depend on
any angular resolution. It:

- returns only real boundary points — the end of an empty ray is never
  reported as an obstacle, and `None` means "nothing in range";
- reports each point in both frames, with `distance` and `bearing` relative to
  the vehicle body;
- excludes a physical neighbourhood (`exclusion_distance`) around `P₁` when
  searching for `P₂`;
- honours occlusion, including self-occlusion: the far side of a solid disk or
  ellipse is not visible (`check_occlusion=False` disables this).

## Delays in the loop

Three independent insertion points, all off by default:

| Field | Symbol | Meaning |
|---|---|---|
| `sensing_delay` | `τ_s` | controller sees `x(t − τ_s)`; multiple of `control_period` |
| `actuation_delay` | `τ_a (+ τ_c)` | command reaches the plant `τ` late; multiple of `integration_step` |
| `actuator_time_constant` | `τ` | first-order lag `τ·y' = u − y` on the applied command |

A computation delay `τ_c` and an actuation delay `τ_a` are the same signal path in
this loop, so one field covers both. Signal path:
request → delay → lag → saturation → plant.

- **`τ_s`** (`DelayedController`) delays the state only; `time` passes through, so
  logs stay timestamped correctly. It differs from the other two because the
  controller is hybrid: the `C`/`G` switching conditions are also evaluated on stale
  geometry.
- **`τ_a`** (`DelayedActuator`) is an exact transport delay. Until the first request
  arrives the plant sees the cruise command `(v, 0)`. Trajectories are identical for
  any `h`.
- **lag** (`LagActuator`) is advanced with the exact factor `α = 1 − exp(−h/τ)` and
  hands the plant the interval average, so the heading increment is exact. A lag is
  **not** a delay: its phase is bounded by −90° and, against a relay, it behaves like
  a delay of only ≈ 0.4–0.9·τ.

Findings (single ellipse, delays up to 0.2 s, one channel at a time): the relay's
switching rate drops from 37/s to ≈ 1/s, i.e. the discrete sliding mode degenerates
into a low-frequency limit cycle; `max|d_R|` grows ≈ quadratically then linearly
(`≈ 3.55·τ_s` for large `τ_s`); no breakdown up to 0.2 s. Full analysis:
[docs/delay_report.md](docs/delay_report.md).

```bash
circumnav-delay-sweep --shape ellipse               # one channel at a time -> npz + png
circumnav-delay-compare --shape cluster --speed 2   # same scene under every channel -> mp4 + png
```

(or `PYTHONPATH=src python3 -m circumnav.examples.delay_sweep ...`)

---

## Several vehicles

`FleetScenario` runs several identical vehicles around **one obstacle**, started at
different points of the equidistant curve, evenly spaced in arc length and
heading along the tangent (counter-clockwise: the law keeps the obstacle on the
vehicle's left). The vehicles are independent — they do not see or avoid each
other, and there is no communication — and delays are not supported yet.

```python
from dataclasses import replace
from circumnav.examples.reactive_circumnav import build_obstacle
from circumnav.scenarios.fleet import FleetScenario
from circumnav.scenarios.reactive import ReactiveCircumnavScenario

base = ReactiveCircumnavScenario(duration=60.0)
base = replace(base, obstacles=build_obstacle("ellipse", base.rho_0, base.turning_radius))

fleet, controllers = FleetScenario(base, robot_count=10, phase=0.0).run()
fleet.results[3]            # a normal SimulationResult for vehicle 3
controllers[3].log          # its controller log
```

`FleetResult` holds one `SimulationResult` per vehicle on a common time grid
(`fleet.positions` has shape `(robots, samples, 2)`). Fleet metrics:
`fleet_separation_metrics(fleet)` (initial, minimum and final distance between the
closest pair; vehicles are points) and `nearest_neighbour_distance(fleet)`.

```bash
circumnav-fleet-demo --robots 10 --animate fleet.mp4 --figure fleet.png
```

Because the vehicles do not interact, running them one after another is exactly
equivalent to running them together. Anything coupled (vehicles seen as obstacles,
communication) needs a synchronous multi-vehicle loop that reads all states at one
instant before any controller runs.

## Reading the results

### `SimulationResult`

Aligned arrays on the `h` grid (`N+1` samples):

| attribute | shape | meaning |
|---|---|---|
| `time` | `(N+1,)` | seconds |
| `state`, `x`, `y`, `heading` | `(N+1,3)` / `(N+1,)` | pose; heading wrapped to `[−π, π)` |
| `requested_control` | `(N+1,2)` | `[speed, yaw_rate]` the controller asked for |
| `applied_control` | `(N+1,2)` | after delay, lag and clipping |
| `saturated` | `(N+1,)` | the clip was active |

`result.save_npz(path)` writes them losslessly.

### `controller.log`

One `ReactiveLogEntry` per control update (on the `T` grid, not `h`):

| field | meaning |
|---|---|
| `time`, `mode` | update time; `CircumnavMode.APPROACH` (`C`) or `ORBIT` (`G`) |
| `obstacle_range` | `ρ(t)`, distance to the nearest visible point (NaN if none) |
| `equidistant_deviation` | `d_R(t)`, the regulated variable |
| `range_rate` | `ḋ_R` term of the sliding variable |
| `yaw_rate_command`, `speed_command` | the request |
| `target_point`, `primary_point`, `secondary_point`, `disk_center`, `orbit_center` | `p`, `P₁`, `P₂`, turning disk centre, frozen `V` (or `None`) |

When nothing is in sensor range the controller holds course and logs NaN.
`controller.switch_count` counts mode transitions.

### Metrics

```python
from circumnav.analysis.metrics import obstacle_clearance_metrics

metrics = obstacle_clearance_metrics(
    result, scenario.build_obstacles(), collision_distance=0.02
)
metrics.minimum_clearance, metrics.collided
```

Computed from the true trajectory and true geometry, independent of what the
sensor reported, so it catches failures a range-limited sensor would miss.
`heading_metrics(result, desired_heading)` serves the heading scenario.

---

## Animation

```python
from circumnav.analysis.animation import (
    AnimationSettings, build_reactive_animation, save_reactive_animation,
)

fig, anim = build_reactive_animation(
    result, controller, scenario.build_obstacles(),
    settings=AnimationSettings(fps=20, real_time_factor=3.0),
)
save_reactive_animation(fig, anim, "run.mp4", fps=20)
```

`.mp4` uses ffmpeg, `.gif` uses pillow. The figure shows:

- **left** — obstacles (grey), equidistant curves (blue dashed), sensor range
  (grey dotted), the path (red) with gap-mode stretches highlighted in gold,
  the vehicle as a triangle, `P₁`/`P₂` markers, and `V` with its `R_min`
  circle, drawn only while mode `G` is active;
- **right** — `d_R(t) = ρ(t) − ρ₀`, the control `u(t)`, and a `C`/`G` mode
  timeline. Panel backgrounds tint orange during `G`.

Pass the controller instance that produced the result: the animation reads its
log directly.

### Reading `d_R(t)` and `ρ(t)`

`d_R(t)` is flat in both modes — on the default cluster (`v = 2`, `ω = 1`,
`ρ₀ = 3`, `R_min = 2`) it stays within `0.11 m` of zero in mode `C` and
`0.09 m` in mode `G`. That is the control performance: the law holds whichever
curve it is tracking to about a decimetre.

`ρ(t)`, the raw distance to the nearest obstacle, is kept in the log
(`obstacle_range`) but not plotted, and it behaves differently: it shows brief
excursions of up to `+1.04 m`, and **every one of them falls in mode `G`**.
That is geometry, not a tracking failure — while the vehicle bridges a gap on
the frozen disk, the true equidistant of the nearest obstacle is not the curve
it is riding, so the distance to that obstacle drifts. The size of an
excursion measures how far the gap-bridging arc departs from the true
equidistant curve.

`ρ` also dips slightly below `ρ₀` (to `−0.11 m` here), for a separate
geometric reason: where two obstacles are closer than `2ρ₀` their equidistant
curves overlap, and no point can hold the full standoff from both at once.
Widen the obstacle spacing past `2ρ₀` if a guaranteed clearance matters.

### Why `d_R(t)` shows small beaks, and how to shrink them

`d_R` is continuous almost everywhere, and the small V-shaped excursions are a
**discretisation artefact, not a property of the law**:

- **`C → G` is exactly continuous.** `V` is placed at distance `ρ₀ + R_min`
  from `P₁` along the ray toward the vehicle, so at that instant
  `‖r − V‖ = (ρ₀ + R_min) − ρ` and therefore `R_min − ‖r − V‖ ≡ ρ − ρ₀`. The
  two branch formulas agree identically; the measured jump is `0.0000 m` at
  every entry.
- **`G → C` has a small genuine jump**, because `V` is frozen at entry while
  `P₁` keeps moving, so by exit time the identity above no longer holds.
- **The beak itself** comes from detecting the switch up to one control period
  late. With `|u| = ω` the vehicle always flies arcs of radius exactly
  `R_min`; had it entered mode `G` exactly tangent to the target circle about
  `V`, it would trace that circle and `d_R` would stay at zero. A late switch
  rotates the heading by `ω·dt` first, so it flies an `R_min` circle whose
  centre is offset from `V` by about `R_min·ω·dt = v·dt`, and `‖r − V‖`
  wanders by that much. It cannot correct: holding radius `R_min` needs a
  sustained turn at exactly `ω`, the relay's maximum, so in mode `G` the relay
  saturates one-sided (97.7 % one sign, flipping on 4.6 % of updates) instead
  of chattering as it does in mode `C` (68/32, flipping on 60 %).

Measured on the default scene, varying `integration_step = control_period`:

| `dt` [s] | settled max&nbsp;\|`d_R`\| in `C` | beak in `G` | beak / (`v·dt`) | ripple in `C` (95th pct) |
|---|---|---|---|---|
| 0.04   | 0.198 | 0.197 | 2.47 | 0.089 |
| 0.02   | 0.111 | 0.085 | 2.12 | 0.011 |
| 0.01   | 0.055 | 0.038 | 1.90 | 0.006 |
| 0.005  | 0.024 | 0.014 | 1.38 | 0.002 |
| 0.0025 | 0.012 | 0.007 | 1.39 | 0.001 |

So **yes — shrinking the step shrinks the beaks, very nearly proportionally**,
and the amplitude follows `≈ 2·v·dt`. Sixteen-fold refinement takes the beak
from 0.20 m to 0.007 m. Useful design rule:

```
dt  ≈  (d_R accuracy you want) / (2 v)
```

The cost is linear in compute. A cheaper route to the same accuracy is event
detection — solving for the exact instant the switching condition crosses zero
inside a step, instead of refining every step — which the stand does not
implement yet.

## Command-line tools

| command | purpose |
|---|---|
| `circumnav-reactive-demo` | one run of the reactive law, prints a summary, optional `.npz` / animation |
| `circumnav-heading-demo` | baseline heading-hold run |
| `circumnav-fleet-demo` | several independent vehicles on one ellipse: `--robots`, `--duration`, `--phase`, `--figure`, `--animate` |
| `circumnav-delay-sweep` | metrics versus one delay channel, others zero |
| `circumnav-delay-compare` | one scene under each channel: videos and overlay plots |

`circumnav-reactive-demo` flags: `--obstacle {circle,ellipse,wall,cluster}`,
`--duration`, `--sensor-range`, `--safety-distance` (`ρ₀`), `--forward-speed`,
`--yaw-rate-magnitude`, `--allow-infeasible-turning`, `--output run.npz`,
`--animate out.mp4|gif`, `--animation-fps`, `--animation-speed`.

`circumnav-delay-sweep` flags: `--channel {sensing,actuation,lag,all}`, `--shape`,
`--max`, `--step` (delays are multiples of `T = 0.02`), `--duration`, `--workers`,
`--tag`, `--output`. Prints one line per point (settled `max|d_R|`, relay flips per
second, `max|s|`, minimum clearance, collision flag) and writes
`delay_sweep_<shape><tag>.{npz,png}`.

`circumnav-delay-compare` flags: `--shape`, `--delay` (value used for every non-zero
channel), `--duration`, `--no-video`, `--fps`, `--speed`, `--zoom T0 T1`,
`--workers`, `--output`. Writes `<shape>_<case>.mp4` for `baseline`, `sensing`,
`actuation`, `lag`, plus `<shape>_compare.png` and `<shape>_compare_zoom.png`.

Outputs default to `simulation_output/` (git-ignored).

---

## Extending the stand

**A new controller.** Implement the `Controller` protocol
(`controllers/base.py`): `reset(initial_state)` and `compute(time, state) ->
DubinsCommand`. Pass it to `Simulator.run`, or wrap it in `DelayedController`.
Keep any logging inside the controller; analysis code reads `controller.log`.

**A new actuator.** Implement the `Actuator` protocol
(`models/actuator.py`): `reset()` and `apply(command, dt) -> ActuatorOutput`. The
engine calls `apply` **every integration step** with the held request, so a
stateful actuator advances by exactly `dt` per call. Wrap an inner actuator and
delegate to it for clipping (see `DelayedActuator`, `LagActuator`); keep the
original request in `ActuatorOutput.requested`.

**A new obstacle.** Implement the `Obstacle` protocol (`models/obstacles.py`):
`closest_point`, `first_intersection`, `is_boundary_visible`, `boundary_polyline`,
`equidistant_polyline`, `minimum_equidistant_curvature`. Add tests next to
`tests/test_curved_obstacles.py`.

**A new sensor.** Implement the `Sensor` protocol (`sensors/base.py`):
`find_primary_point` and `find_secondary_point`, returning `None` for "nothing
visible" — never invent a point at maximum range.

**A new scenario.** Add a frozen dataclass in `scenarios/` with a `run()` that
builds model, actuator, controller and `Simulator`, as `ReactiveCircumnavScenario`
does.

---

## Running the tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider
```

104 tests, about 40 seconds. They cover the plant, obstacles, sensor, controller,
loop, scenarios, actuators and delays.

## Layout

```
src/circumnav/
├── models/        Dubins kinematics, actuators (ideal, delay, lag), obstacle geometry
├── sensors/       measurement type, circular visibility sensor
├── controllers/   controller interface, heading controller, reactive law, delay wrapper
├── simulation/    sampled-data loop (zero-order hold) and result log
├── scenarios/     reproducible experiment configurations (single vehicle, fleet)
├── analysis/      metrics and animation
└── examples/      CLI entry points (demos, delay sweep, delay compare)
tests/             pytest suite, one file per module
docs/              reports (delay_report.md) and figures
```

Each package has an `AGENTS.md` describing its contracts and invariants.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `ValueError: Infeasible scenario ...` | `ρ₀ < R_min = v/ω`; raise `safety_distance` or lower `forward_speed`. `require_feasible_turning=False` runs it anyway |
| `ValueError: ... integer multiple of integration_step` | `control_period`, `duration` or `actuation_delay` is not on the `h` grid |
| `ValueError: sensing_delay must be an integer multiple of control_period` | `τ_s` must be a multiple of `T` (default `0.02`) |
| vehicle orbits at the wrong radius and never converges | `closing_speed_ratio` too small; keep `dead_zone_gain·dead_zone/v ≈ 0.3` |
| log full of NaN, vehicle flies straight | obstacle outside `sensor_range` |
| `.mp4` export fails | `ffmpeg` missing; use `.gif` or install it |
| `ModuleNotFoundError: circumnav` | not installed; use `pip install -e .` or `PYTHONPATH=src` |
| animation is empty or errors | pass the controller instance that produced the result |

Design notes that predate the implementation are in
[README_CONTROL_SIMULATOR.md](README_CONTROL_SIMULATOR.md) and
[README_SENSOR_MODEL.md](README_SENSOR_MODEL.md). The original prototype
(`python_sim/`) was removed from the tree; it remains in git history at commit
`109a779`, and the ROS 2 / Gazebo stack (`gazebo_sim/`) at commit `bbfa2b3`.
