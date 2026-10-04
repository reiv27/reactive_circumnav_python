# analysis/ — metrics and animation

Reads `SimulationResult` and controller logs. **Never changes the dynamics.**

## metrics.py
- `obstacle_clearance_metrics(result, obstacles, collision_distance)`: minimum true
  distance from the trajectory to any obstacle (true geometry, independent of the
  sensor) and a collision flag. Per-sample Python loop, fine for tens of thousands
  of samples.
- `heading_metrics(result, desired_heading)` for the heading scenario.

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
