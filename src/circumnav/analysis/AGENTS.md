# analysis/ — metrics and animation

Reads `SimulationResult` and controller logs. **Never changes the dynamics.**

## metrics.py
- `obstacle_clearance_metrics(result, obstacles, collision_distance)`: minimum true
  distance from the trajectory to any obstacle (true geometry, independent of the
  sensor) and a collision flag. Per-sample Python loop, fine for tens of thousands
  of samples.
- `heading_metrics(result, desired_heading)` for the heading scenario.

- `fleet_separation_metrics(fleet)` (initial / minimum / final distance of the
  closest pair), `nearest_neighbour_distance(fleet)` `(robots, samples)` and
  `pairwise_distance(fleet)` `(robots, robots, samples)` (diagonal `inf`);
  vehicles are points, `inf` for a single vehicle.
- `obstacle_distance(result, obstacles)`: true distance to the nearest obstacle
  surface per sample (true geometry, like `obstacle_clearance_metrics`).

- `neighbour_visibility(fleet, obstacles, sensor)` -> `NeighbourVisibility`
  (`in_range`, `visible`, derived `occluded`, `visible_count`; arrays
  `(samples, robots, robots)`, symmetric, empty diagonal);
  `neighbour_visibility_metrics` -> seeing fraction, mean visible count,
  pair-level dropout events (visible -> hidden while in range), occluded fraction.
  `nearest_visible_neighbour_distance(fleet, visibility)` `(robots, samples)`: what
  a neighbour sensor would report, `NaN` where a vehicle sees nobody; it jumps when
  the nearest neighbour hides, unlike the continuous `nearest_neighbour_distance`.
  `hidden_episode_durations(visibility, time, observer, neighbour)` -> lengths (s) of
  not-visible stretches (out of range or behind a body; run ends included truncated).

## fleet_animation.py
- `build_fleet_animation(fleet, controllers, obstacles, settings)` and
  `save_fleet_figure(...)`: one map with a colour and triangle per vehicle,
  focus vehicle only: red (others light blue); panels `d_R(t)`, distance to its
  nearest *visible* neighbour (red, gaps when blind) over the true nearest (dashed), per-neighbour line-of-sight timeline (light blue visible,
  orange hidden by the body, grey out of range); map links from the focus vehicle
  (green visible, dotted orange hidden). `focus` argument. Reuses the private
  helpers of `animation.py`. Save with `save_reactive_animation`.
- `save_fleet_distance_figure(fleet, obstacles, path, safety_distance, visibility,
  focus)`: four stacked panels sharing `t`: focus-to-each-neighbour distance with its
  nearest envelope; nearest *visible* neighbour of the focus vehicle against the
  true nearest, grey where it sees nobody; nearest-neighbour distance of every vehicle with the closest
  pair (min / start / end in the title); true obstacle distance `ρ(t)` against
  `ρ₀`. With evenly spaced vehicles the nearest-neighbour curves coincide up to
  a time shift, which is why the top panel splits them per neighbour.

## animation.py
- `build_reactive_animation(result, controller, obstacles, settings) -> (fig, anim)`
  and `save_reactive_animation(fig, anim, path, fps)` (`.mp4` via ffmpeg, `.gif`
  via pillow; closes the figure). `controller` is the instance that produced the
  result — it reads `controller.log` (works through `DelayedController`).
- Layout: left map (obstacles grey, equidistant blue dashed, sensor range dotted,
  path red with `G` stretches in gold, **vehicle as a small triangle**
  (`robot_size_in_turning_radii = 0.55`), `P₁`/`P₂`); right panels `d_R(t)`, `u(t)`,
  mode timeline. No `ρ(t)` panel.
- `V` and its `R_min` circle are drawn **only while mode `G`** is active, from
  `entry.orbit_center`.
- All labels are LaTeX mathtext, with single backslashes inside raw strings
  (`r"$R_s$"`; a doubled backslash raises `ParseFatalException`).
- `AnimationSettings.max_frames = 1200` caps the frame count; use
  `real_time_factor` to speed playback instead of rendering more frames.
- Rendering a 60 s scene takes minutes: run it in the background.
