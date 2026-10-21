# examples/ — command-line entry points

Entry points are registered in `pyproject.toml` (`[project.scripts]`) and also
runnable as `PYTHONPATH=src python3 -m circumnav.examples.<module>`.

| module | script | what it does |
|---|---|---|
| `reactive_circumnav.py` | `circumnav-reactive-demo` | one run, summary, optional `.npz` and animation; `build_obstacle(shape, rho_0, turning_radius)` builds `circle / ellipse / wall / cluster / two_ellipses` demo scenes (the ellipse is 6×3 at 30°; `two_ellipses` is two 6×3 ellipses at 90° to each other, centres `(0,0)` and `(16,0)`, boundary gap 7 m, tuned for `ρ₀=3`, `R_min=2` and the default start pose; the law bridges both waists in mode G) |
| `heading_control.py` | `circumnav-heading-demo` | baseline heading hold |
| `delay_sweep.py` | `circumnav-delay-sweep` | one channel (`sensing`, `actuation`, `lag`) at a time, others zero; per point: settled `max|d_R|`, bias, RMS, relay flips per second, `max|s|`, lost fraction, G fraction, clearance, collision |
| `delay_compare.py` | `circumnav-delay-compare` | one scene under baseline + each channel at the same value; mp4 per case, overlay and zoom figures |
| `predictor_compare.py` | `circumnav-predictor-compare` | one scene under baseline / sensing-delay-uncompensated / sensing-delay+`PredictorController`; mp4 per case (default on, `--no-video` to skip), overlay figure, prints max position error vs baseline per case (stage 1a/1b acceptance check) |
| `predictor_report.py` | `circumnav-predictor-report` | `τ_s` sweep (uncompensated vs predictor; `delay_sweep.run_point` metrics + max position deviation from the delay-free run), joint `τ_s`+`τ_a` / `τ_s`+lag table, mode-G episode times and a zoom figure; default scene `two_ellipses`. Data behind `docs/predictor_report.md` |

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
