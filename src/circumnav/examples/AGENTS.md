# examples/ — command-line entry points

Entry points are registered in `pyproject.toml` (`[project.scripts]`) and also
runnable as `PYTHONPATH=src python3 -m circumnav.examples.<module>`.

| module | script | what it does |
|---|---|---|
| `reactive_circumnav.py` | `circumnav-reactive-demo` | one run, summary, optional `.npz` and animation; `build_obstacle(shape, rho_0, turning_radius)` builds `circle / ellipse / wall / cluster` demo scenes (the ellipse is 6×3 at 30°) |
| `heading_control.py` | `circumnav-heading-demo` | baseline heading hold |
| `delay_sweep.py` | `circumnav-delay-sweep` | one channel (`sensing`, `actuation`, `lag`) at a time, others zero; per point: settled `max|d_R|`, bias, RMS, relay flips per second, `max|s|`, lost fraction, G fraction, clearance, collision |
| `delay_compare.py` | `circumnav-delay-compare` | one scene under baseline + each channel at the same value; mp4 per case, overlay and zoom figures |
| `predictor_compare.py` | `circumnav-predictor-compare` | one scene under baseline / sensing-delay-uncompensated / sensing-delay+`PredictorController`; mp4 per case (default on, `--no-video` to skip), overlay figure, prints max position error vs baseline per case (stage 1a/1b acceptance check) |
| `rl_train.py` | `circumnav-rl-train` | trains PPO (`stable-baselines3`) on `circumnav.rl.env.CircumnavEnv` (postановка (a), no delay), evaluates a deterministic rollout against the frozen relay on the same scene, prints `max|d_R|`/clearance/collision for both, saves the model and a comparison figure. Requires the `rl` extra. `--load` skips training and evaluates a saved model instead |

## Conventions
- Metrics use the second half of the run (`SETTLED_FRACTION = 0.5`) to drop the
  transient.
- Delay values must be multiples of `T = 0.02` for `sensing` and of `h = 0.01` for
  `actuation`/`lag`-related fields; use `--step 0.02` or `0.04`.
- Heavy work goes through `ProcessPoolExecutor`; keep worker functions top-level
  and arguments picklable (frozen scenarios are).
- Outputs go to `simulation_output/`; keep a `.log` next to sweep data when you
  redirect stdout. A sweep script that only lived in a scratchpad cannot be
  reproduced later — put new experiment code here.
- Importing matplotlib: do it inside the plotting function with `Agg` backend.
