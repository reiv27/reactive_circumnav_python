# Reactive circumnavigation — Dubins vehicle simulation stand

A simulation environment for studying a reactive obstacle-circumnavigation law
on a Dubins vehicle. The vehicle carries a range-limited, omnidirectional
sensor, measures relative distances to obstacle boundaries, and rides the
**equidistant curve** at a commanded standoff.

Reference paper DOI: <https://doi.org/10.1016/j.robot.2024.104649>

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

## Install

```bash
pip install -e ".[dev,viz]"
```

`numpy` is the only runtime dependency; `viz` adds `matplotlib` for plots and
animation. Without installing, prefix commands with `PYTHONPATH=src`.

## Quick start

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

`result` is a `SimulationResult` with aligned time series (`time`, `x`, `y`,
`heading`, requested/applied control, saturation flags) and
`result.save_npz(path)`. `controller.log` is the controller's own diagnostic
record, one entry per control update.

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

## Metrics

```python
from circumnav.analysis.metrics import obstacle_clearance_metrics

metrics = obstacle_clearance_metrics(
    result, scenario.build_obstacles(), collision_distance=0.02
)
metrics.minimum_clearance, metrics.collided
```

Computed from the true trajectory and true geometry, independent of what the
sensor reported, so it catches failures a range-limited sensor would miss.
`controller.switch_count` counts mode transitions.

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

## Running the tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider
```

## Layout

```
src/circumnav/
├── models/        Dubins kinematics, actuator limits, obstacle geometry
├── sensors/       measurement type, circular visibility sensor
├── controllers/   controller interface, heading controller, reactive law
├── simulation/    sampled-data loop (zero-order hold) and result log
├── scenarios/     reproducible experiment configurations
├── analysis/      metrics and animation
└── examples/      CLI entry points
```

Design notes that predate the implementation are in
[README_CONTROL_SIMULATOR.md](README_CONTROL_SIMULATOR.md) and
[README_SENSOR_MODEL.md](README_SENSOR_MODEL.md). The original prototype
(`python_sim/`) was removed from the tree; it remains in git history at commit
`109a779`, and the ROS 2 / Gazebo stack (`gazebo_sim/`) at commit `bbfa2b3`.
