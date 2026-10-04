# sensors/ — exteroceptive measurement

- `PointObservation`: `point_world`, `point_relative` (body frame), `distance`,
  `bearing`, `obstacle_id`. Helpers `to_robot_frame`, `observation_from_world_point`.
- `Sensor` protocol: `find_primary_point(pose, obstacles)` → `P₁` (closest visible
  point) or `None`; `find_secondary_point(pose, disk_center, primary, obstacles)` →
  `P₂` (visible point closest to the turning disk centre, outside `P₁`'s
  exclusion neighbourhood) or `None`.
- `CircularVisibilitySensor(max_range, exclusion_distance, check_occlusion=True)`:
  omnidirectional (2π), range-limited. Works on **exact geometry**, no ray scan, so
  no angular-resolution parameter exists. Occlusion includes self-occlusion (the
  far side of a solid disk/ellipse is not visible).

## Invariants
- `None` means "nothing in range / visible". Never return a point at maximum range.
- The sensor knows nothing about modes, control or time.
- `exclusion_distance` is a physical distance, not a ray-index window.
- Tests: `tests/test_circular_sensor.py`; build occlusion/exclusion cases with
  concrete coordinates and check them by hand.
