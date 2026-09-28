# ecofill

ecofill fills gaps in regular environmental time series — meteorology, water
quality, eddy-covariance fluxes — when a declared method has enough support.
When it cannot do that safely, it says why. It is built to make every decision
inspectable:

- **Explicit masks.** You declare which values are QC-passed observations. A
  finite number outside that mask is never used as a donor or as truth.
- **Inspectable plans.** Profiling finds each gap. A plan records, per gap,
  the method, its support and why it is eligible or not, before anything runs.
- **Observed values are preserved.** Execution only writes missing cells and
  re-checks the plan against the data first.
- **Abstention is a result.** Gaps that are too long, structural, at an edge or
  without support stay missing with a stated reason instead of a guess.
- **Per-cell provenance.** Every cell has a state, method and reason, and
  exports are plain CSV/JSON that a standard-library reader can check.

Status: version 0.0.1, research software. It is not published on PyPI. The
methods are implemented and tested as software; their accuracy on your data is
not established by this repository (see [Limitations](#limitations)).

## Install

ecofill needs Python 3.12. From a checkout of this repository:

```console
python -m pip install .
```

The core install needs only NumPy and pandas. Optional extras add pinned
packages for specific methods:

| Extra | Adds | Needed for |
|---|---|---|
| `research` | scipy, scikit-learn, statsmodels, h5py | random-forest and boosting driver regression |
| `deep` | torch | BRITS and SAITS |
| `foundation` | timesfm, torch, huggingface-hub, safetensors | TimesFM 2.5 and 3 forecasting |
| `tabpfn` | tabpfn, torch | TabPFN 3.5 time-indexed backend |
| `dev` | build, setuptools, wheel, PyYAML | building and testing |

For example `python -m pip install ".[tabpfn]"`. The model extras install code
only. You must obtain model weights yourself (see [Optional models](#optional-models)).

## Quick start

A synthetic 15-minute temperature series with a 2-step gap and a 3-step gap,
filled by linear interpolation with a 30-minute whole-gap limit:

```python
import numpy as np
import pandas as pd
from ecofill import SeriesData, TimeSeriesSchema, VariableSpec, execute, make_plan

frame = pd.DataFrame(
    {"temperature": [20.0, np.nan, np.nan, 23.0, np.nan, np.nan, np.nan, 20.0]},
    index=pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC"),
)
schema = TimeSeriesSchema(pd.Timedelta("15min"), (VariableSpec("temperature", "degC"),))
# The caller declares which values are QC-passed observations.
data = SeriesData(frame, schema=schema, observed=frame.notna())
plan = make_plan(data, method="linear", mode="offline", max_gap=pd.Timedelta("30min"))
result = execute(data, plan)

print(result.completed["temperature"].tolist())
print(result.states["temperature"].tolist())
print(result.unresolved_reasons["temperature"].dropna().unique().tolist())
```

Output:

```text
[20.0, 21.0, 22.0, 23.0, nan, nan, nan, 20.0]
['observed', 'imputed', 'imputed', 'observed', 'absent', 'absent', 'absent', 'observed']
['gap_exceeds_limit']
```

The 45-minute gap is longer than the limit, so it is left whole rather than
partly filled. `plan.gaps` shows that decision before execution, and
`result.export(<new directory>)` writes values, masks, per-cell states and a
JSON provenance record.

## A complete workflow

The workflow is **validate → profile → plan → execute → evaluate → export**.
[examples/integration_demo.py](examples/integration_demo.py) runs all of it on
synthetic files: CSV input with declared masks, gap profiles, a plan, execution,
a benchmark on artificially hidden observations, and export. A second,
standard-library-only process then reads the export back:

```console
python examples/integration_demo.py --output demo-output
python -I -S examples/read_export.py result demo-output/primary/workflow/result
```

The output directory must not exist yet. See [docs/guide.md](docs/guide.md) for
the API of each step and [docs/exports.md](docs/exports.md) for the formats.

## Methods

Implementation is not the same as tested performance. The "Evidence" column
says what has actually been run; none of it is a field-accuracy claim.

| Method | API | Extra | Evidence |
|---|---|---|---|
| Persistence, next observation, bounded linear, seasonal analog, known-parameter local-level filter/smoother | `make_plan`, `execute` | core | Deterministic tests. Linear interpolation was used in small synthetic comparisons and one scoped field development check. |
| Driver regression (OLS, random forest, histogram boosting) with derived-variable recomputation | `make_driver_plan`, `execute_driver_plan` | core (OLS), `research` (trees) | Contract tests. Only small synthetic and single-instrument development runs; the tree adapters have no comparative benchmark. |
| Marginal distribution sampling (`mds_hesseflux`) and a signed eddy-covariance recipe | `make_mds_plan`, `execute_mds`, `ecofill.ec_recipe` | core | Matches the pinned upstream code (within 1e-10, same quality tiers) on constructed cases. Not equivalence to other MDS implementations; no field validation. |
| Forward/backward forecasting, TimesFM 2.5 and TimesFM 3 adapters | `make_forecast_plan`, `execute_forecast`, `TimesFM25Backend`, `TimesFM3Backend` | `foundation` + local weights | Request plumbing tested with stand-ins; CPU runs on synthetic and one development series. Point forecasts only. |
| BRITS and SAITS (vendored PyPOTS cores) | `fit_deep`, `make_deep_plan`, `execute_deep` | `deep` | Tiny CPU training runs for wiring only. In a small synthetic pilot they were much less accurate than linear interpolation. |
| Time-indexed imputation, TabPFN 3.5 backend | `make_time_indexed_plan`, `execute_time_indexed`, `TabPFNBackend` | `tabpfn` + local weights | CPU runs on synthetic series with exact parity against direct TabPFN calls. In one synthetic comparison it was strongly biased after a level shift that its reference bank did not contain. |
| Fixed-weight ensembles | `EnsembleRecipe`, `blend` | core | Contract tests. Weights are fixed by you; nothing is selected automatically. |
| Block residual calibration, intervals and whole-block completed draws | `calibrate_blocks`, `block_intervals`, `sample_block_draws`, `export_uncertainty` | core | Synthetic contract tests only. No model output has been calibrated on field data. |

How the methods work, and the assumptions behind them, is in
[methodology.md](methodology.md).

## Optional models

- **Local weights only.** No model weights are included, and ecofill never
  downloads any. You download a checkpoint deliberately, check its license,
  and pass its local path. ecofill verifies the exact SHA-256 before loading.
- **Separate weight licenses.** TimesFM 2.5 weights are Apache-2.0. TimesFM 3
  and TabPFN 3.5 weights are licensed for non-commercial use only. These terms
  are separate from ecofill's MIT license; see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **CPU and GPU.** The TimesFM and BRITS/SAITS adapters run on CPU. The
  time-indexed path and `TabPFNBackend` accept an explicit `device="cpu"` or
  `device="cuda"` and never fall back silently. CUDA routing has been tested
  only with stand-in backends; no GPU run has been made, so GPU speed, memory
  and numerical agreement are unverified.

Details and the exact pinned checkpoints are in [docs/models.md](docs/models.md).

## Limitations

- ecofill does not choose a method, infer QC, repair clocks, infer units or
  declare a safe gap length for you. Linear interpolation in the examples is a
  conservative illustration, not a recommendation.
- The input must already be a regular, timezone-aware grid. Irregular input is
  rejected until you regularize it explicitly.
- Tests use synthetic data. They show that the software keeps its contracts,
  not that a method is accurate for your site, variable or gap length.
- Calibrated intervals assume that calibration and new blocks are
  exchangeable. Separate time blocks do not prove that, and no field coverage
  has been established.
- Only Windows 10 with CPython 3.12 has been observed. Other platforms and
  Python versions are untested.

## Documentation

- [methodology.md](methodology.md): the methods and their references, in brief.
- [docs/guide.md](docs/guide.md): how to use each part of the API.
- [docs/models.md](docs/models.md): optional model backends, weights and devices.
- [docs/exports.md](docs/exports.md): export formats and the standalone reader.
- [docs/references.md](docs/references.md): the full verified bibliography.
- [CONTRIBUTING.md](CONTRIBUTING.md): development setup and tests.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): vendored code and weight terms.

## License

Original ecofill code is released under the [MIT License](LICENSE). The wheel
also contains vendored PyPOTS code (BSD-3-Clause) and hesseflux code (MIT),
whose license files are kept with them. Optional packages and model weights
have their own terms.
