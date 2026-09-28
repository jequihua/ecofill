# User guide

This guide walks through the API in workflow order. Every example named here
is in [examples/](../examples/) and runs on synthetic data with the core
install. For the reasoning behind each step see
[methodology.md](../methodology.md).

## 1. Declare the data

```python
from ecofill import SeriesData, TimeSeriesSchema, VariableSpec

schema = TimeSeriesSchema(pd.Timedelta("30min"), (
    VariableSpec("TA", "degC", "driver"),
    VariableSpec("NEE", "umol CO2 m-2 s-1", "response"),
))
data = SeriesData(frame, schema=schema, observed=qc_passed)
```

- `frame` is a pandas DataFrame with a timezone-aware, regular, unique and
  ordered `DatetimeIndex`. Irregular input is rejected with a message; ecofill
  never repairs clocks. [examples/integration_demo.py](../examples/integration_demo.py)
  shows explicit regularization, where new rows are marked `inserted`.
- `VariableSpec(name, unit, role, derived_from)` declares units, a role
  (`target`, `driver`, `response`, `external`, `derived` or `diagnostic`) and
  lineage.
  A derived variable lists its parents.
- `observed` is your QC-passed mask. It is required: a finite value outside it
  is not a donor and not benchmark truth.
- Optional aligned boolean masks: `qc_rejected`, `structural` (cells that
  cannot exist), `artificial` (observations hidden for evaluation; their
  declared descendants are hidden too) and `inserted`.

`SeriesData` copies what you pass. `data.mask(name)` returns a mask, for
example `"donor"` (observed and not hidden).

## 2. Profile the gaps

`profile_gaps(data)` lists each missing run per variable with its positions,
number of steps, duration (steps × cadence), geometry and whether it is
bounded on both sides. `profile_events(data)` groups runs that overlap across
variables into shared or staggered events and reports a channel missing
everywhere as its own `full_channel` event.
[examples/profile_demo.py](../examples/profile_demo.py) prints one gap.

## 3. Plan and execute a baseline

```python
plan = make_plan(data, method="linear", mode="offline", max_gap=pd.Timedelta("2h"))
result = execute(data, plan)
```

Methods: `persistence`, `next`, `linear`, `seasonal` (needs `period` and
`search_periods`), `state_filter` and `state_smoother` (need a
`StateSpaceSpec` with explicit variances and initial state). `max_gap` is
required; a longer whole gap abstains with `gap_exceeds_limit`. In `causal`
mode, `next`, `linear` and `state_smoother` are refused because they use later
observations. `execute` checks the plan against a freshly derived plan, so
make a new plan after changing data or policy.

The `ImputationResult` exposes `completed`, `raw`, `states`, `methods`,
`unresolved_reasons`, `mask(name)` and `provenance()`.
`result.export(directory)` writes to a new directory and refuses to overwrite
one; see [exports.md](exports.md).

## 4. Evaluate on hidden observations

`ecofill.benchmark.make_mask_cases(data, split_position, seed=0)` builds fixed
random, block, shared, staggered and edge masks from observed cells after
`split_position`. Save them before looking at results.
`run_case(data, case, split_position=..., method=..., mode=..., max_gap=...)`
scores one method on a fresh copy with those cells hidden. Each row reports
hidden, predicted, abstained and failed counts with MAE, RMSE, bias and a
normalized MAE scaled by the training portion only. An exception while
executing a valid plan is a failed cell, not an abstention.
[examples/integration_demo.py](../examples/integration_demo.py) runs the
whole path from CSV files to export.

## 5. Drivers, derived variables and responses

`make_driver_plan(data, policy, external)` and
`execute_driver_plan(data, plan, external)` reconstruct in a declared order.
`DriverPolicy` holds `RegressionStep`s (target, ordered predictor allowlist,
estimator `"ols"`, `"rf"` or `"boosting"`) and `DerivedStep`s (a formula with
declared units and a stable identifier). `ExternalInputs` carries
caller-supplied covariates with their own observed mask, units, source and
availability times. Fits use observed training labels only; a reconstructed
value can feed a later step only when that step allows it. RF and boosting need
the `research` extra. [examples/pipeline_demo.py](../examples/pipeline_demo.py)
runs a driver → derived → response chain.

## 6. Eddy-covariance fluxes with MDS

`MdsPolicy`, `make_mds_plan` and `execute_mds` run the `mds_hesseflux` method
on a 30-minute grid with a signed flux target and radiation, air temperature
and VPD in `W m-2`, `degC` and `hPa`. It is offline only.
`ecofill.ec_recipe.compare_nee` runs MDS and linear side by side, and
`signed_supported_integral` sums supported interval means and reports covered
versus total seconds. [examples/ec_mds_demo.py](../examples/ec_mds_demo.py)
is an editable starting point.

## 7. Forecasting across gaps

```python
policy = ForecastPolicy(channels=("temperature",), mode="offline", direction="blend",
                        max_gap=pd.Timedelta("2h"), context_length=8,
                        min_support=2, max_horizon=2)
result = execute_forecast(data, make_forecast_plan(data, policy), backend)
```

A backend has one method, `forecast(request)`, returning an array of shape
`(request.horizon, len(request.channels))`. `direction` is `forward`,
`backward` or `blend`. Contexts must be contiguous and fully observed; nothing
is interpolated to make them so. Covariates are existing channels declared with
`DynamicCovariate(name, available_as_of, reversible)`. `LastValueForecaster` is
a deterministic plumbing baseline.
[examples/forecast_demo.py](../examples/forecast_demo.py) uses it. For TimesFM
see [models.md](models.md).

## 8. BRITS and SAITS

`fit_deep(data, policy, training_hide)` trains on the policy's training rows
with a caller-supplied hiding mask, `make_deep_plan(data, fit)` lists
eligible gaps, and `execute_deep(data, fit, plan)` fills them. The methods are
`brits_pypots` and `saits_pypots`; a real fit needs the `deep` extra.
[examples/deep_demo.py](../examples/deep_demo.py) exercises the contract with
a stand-in backend and no PyTorch.

## 9. Time-indexed imputation

`TimeIndexedPolicy(targets, reference, features, max_reference, min_support,
max_gap, seed, device="cpu")` declares reference intervals and the fixed
time-feature recipe `TimeFeatures(origin, scale)`. `make_time_indexed_plan`
freezes the reference bank; `execute_time_indexed(data, plan, backend)` calls a
backend with a `predict(request)` method and a declared `identity`.
[examples/time_indexed_demo.py](../examples/time_indexed_demo.py) uses a small
least-squares backend. `TabPFNBackend` is the optional native backend; see
[models.md](models.md). Only offline mode is supported.

## 10. Ensembles and uncertainty

- `EnsembleRecipe` and `EnsembleMember` fix member names, methods,
  configuration identities (`configuration_identity(result)`) and weights that
  sum to one. `blend(data, members, recipe)` combines results cell by cell.
- `BlockRecord`, `SupportSpan` and `SupportStratum` declare calibration and
  evaluation blocks and their support.
- `calibrate_blocks(...)` freezes whole-block residual scores and thresholds.
  `block_intervals(...)` returns intervals. `evaluate_blocks(...)` scores
  untouched blocks. `sample_block_draws(...)` makes completed draws, and
  `draw_flux_summary(...)` sums a signed flux per draw.
- `export_uncertainty(directory, calibration=..., intervals=..., draws=...,
  flux=...)` writes plain files.

[examples/uncertainty_demo.py](../examples/uncertainty_demo.py) runs the whole
chain on a synthetic series. Its intervals are software checks, not calibrated
field uncertainty.
