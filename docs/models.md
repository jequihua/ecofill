# Optional model backends

ecofill works without any machine-learning model. The backends on this page
are optional. Each needs an extra, and TimesFM and TabPFN also need weights that
you download yourself. Importing `ecofill` never imports PyTorch, TimesFM or
TabPFN and never downloads anything.

## Common rules

- **Explicit local weights.** You obtain the exact checkpoint, check its
  license, and pass a local path. The adapter checks the file's SHA-256 before
  loading and rejects any other bytes. There is no automatic download, cache
  lookup or substitute model.
- **Actionable failures.** A missing extra raises `ImportError` naming it (for
  example `ecofill[tabpfn]`); a missing checkpoint raises `FileNotFoundError`.
  A backend is never replaced by another method silently.
- **Weights have their own license.** The terms, copied from the official
  sources with pinned revisions, are in
  [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The extras and ecofill's
  MIT license do not change them.
- **Evidence.** All real model runs were made on CPU on synthetic series or a
  single development series. They show the adapters work; they are not a
  ranking or a field-accuracy result.

## TimesFM 2.5 and 3 (`foundation` extra)

```python
from ecofill import TimesFM25Backend, TimesFM3Backend, execute_forecast

backend = TimesFM25Backend(checkpoint_dir, context_length=128, max_horizon=24)
# TimesFM3Backend(checkpoint_dir, context_length=128, max_horizon=24,
#                 joint=True, purpose="non-commercial research")
backend.load()
result = execute_forecast(data, plan, backend)
```

- `checkpoint_dir` holds `model.safetensors` from the pinned revision; for
  TimesFM 3 it must also hold that revision's `LICENSE` file, which is checked.
- Use it through `make_forecast_plan` with `direction="forward"`,
  `"backward"` or `"blend"`; see [guide.md](guide.md#7-forecasting-across-gaps).
- The 2.5 adapter forecasts each channel independently and rejects covariates.
  The 3 adapter supports independent or joint targets and, jointly, declared
  reversible known-future covariates.
- CPU only, batch size 1. Point forecasts only; no quantile calibration. The
  model's own missing-value interpolation is never used.
- TimesFM 3 weights are for non-commercial use only. The required `purpose`
  argument is a reminder, not a license.

## TabPFN 3.5 (`tabpfn` extra)

```python
from ecofill import TabPFNBackend, execute_time_indexed, make_time_indexed_plan

backend = TabPFNBackend("path/to/tabpfn-v3.5-20260909.safetensors", device="cpu")
result = execute_time_indexed(data, make_time_indexed_plan(data, policy), backend)
```

- Exactly one checkpoint is accepted: the standard TabPFN 3.5 regressor listed
  in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
- Fixed settings: one estimator, the request seed as `random_state`, float32,
  mean output, no tuning or fine-tuning. A fit prepares the reference bank; it
  does not train the foundation weights.
- `backend.identity` declares the package version, checkpoint and settings
  without host paths; ensemble recipes bind to it.
- TabPFN 3.5 weights are for non-commercial use only.

## BRITS and SAITS (`deep` extra)

The `brits_pypots` and `saits_pypots` methods train small models locally with
the vendored PyPOTS numerical cores and PyTorch; no pretrained weights are
involved. Training is offline, on nonoverlapping windows, with training-only
scaling. In a small synthetic pilot, 20-epoch fits were much less accurate than
linear interpolation, so treat them as experimental.

## CPU and GPU

| Backend | Device | Status |
|---|---|---|
| TimesFM 2.5 / 3 | CPU | CPU runs observed |
| BRITS / SAITS | CPU | CPU runs observed |
| TabPFN 3.5 | `device="cpu"` or `device="cuda"` | CPU runs observed; CUDA untested on hardware |

`device="cuda"` is an explicit request. If CUDA is unavailable the backend
raises before loading; it never falls back to CPU. Declared and actual devices
are checked after each call. CUDA routing has been tested only with stand-in
objects. No GPU run has been made, so GPU speed, memory use and numerical
agreement are unknown.

All observed runs used Windows 10 and CPython 3.12. Other platforms are
untested.
