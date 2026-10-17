# Simple configuration

For local typed input/output and mounted-volume ownership, see
[local Parquet operation](local-parquet.md). The existing CSV default is unchanged.

Fill gaps in your own CSV with a few lines of configuration. You name the file,
its cadence, the columns to fill, one method and the longest gap to fill.
Everything else has a documented default, and every default is written into
the resolved configuration that `plan` and `run` save.

> **Release status.** Simple configuration is candidate support in this source
> tree and in wheels built from it. The published image
> (`ecofill@sha256:950cea24…`, see the [release receipt](release.json)) still
> accepts only explicit `ecofill.operation/1` configurations. A reviewed image
> release is required for Cloud Run. Source users can build and test a local image
> with the [container testing guide](container-testing.md).

## Quick start

The example is installed with the package (`ecofill/operation/simple_example.py`,
standard library only). From a new working directory, with ecofill installed (see
the [installation guide](installation.md)), `examples/operation_simple_demo.py`
from this repository runs that installed file:

```sh
python examples/operation_simple_demo.py prepare --output weather-demo
cd weather-demo
ecofill validate --config config.json
ecofill plan --config config.json --output first-plan
ecofill run --config config.json --run-id first
ecofill inspect --run-uri output/runs/first
python -I ../examples/operation_simple_demo.py readback --output output/runs/first
```

The launcher does not import ecofill, NumPy or pandas; it executes the installed
example file, so the readback remains a standard-library check.

`prepare` writes a synthetic 30-minute weather file, `inputs/weather.csv`, with
the columns `timestamp,temp_out,humidity,dew_point,wind_speed,wind_chill,
heat_index,bar,rain,rain_rate,solar_rad,solar_energy,et,data_source,station_id,
primary_key`, and this `config.json`:

```json
{
  "datasets": {
    "weather": {
      "uri": "inputs/weather.csv",
      "cadence": "30min",
      "targets": ["bar"],
      "method": "LIN",
      "max_gap": "2h"
    }
  },
  "output": {"uri": "output"}
}
```

Only `bar` is filled. Its gaps of at most 2 hours are filled by linear
interpolation between original observations. The 3-hour gap stays unresolved
(`no_route`), and gaps at the start and end stay unresolved
(`boundary_gap_unresolved`); nothing is extrapolated. Every other column is read
only if you name it. The standard-library `readback` checks hashes, the 460
original values, the 10 filled cells and the unresolved reasons after the run
process has exited.

Relative paths are resolved from the directory where you run `ecofill`.
JSON needs only the core installation; YAML needs `ecofill[operation]` and
quoted timestamps.

## The sources/routes spelling

This shorter alternative spelling is accepted exactly as written:

```json
{
  "sources": {
    "davis_weather": {
      "uri": "inputs/weather.csv",
      "cadence": "30min",
      "variables": {
        "temp_out": {"unit": "degC", "role": "driver"},
        "bar": {"unit": "hPa", "role": "response"},
        "dew_point": {"unit": "degC", "role": "driver"}
      }
    }
  },
  "routes": {
    "davis_weather": {"enabled": true, "method": "linear", "mode": "offline", "max_gap": "2h"}
  },
  "output": {"uri": "output"}
}
```

`response` or `target` selects a column to fill; `driver` only declares a column
and its unit. `linear` means `LIN`, which never uses or fills driver columns.
`mode` must be `offline`. This spelling supports `LIN` only, because other
methods need explicit predictors: use the `datasets` format for them. It gives
the same resolved execution and outputs as the `datasets` format with the same
names and units. `prepare` writes it as `config-engineer.json`.

## Optional settings and defaults

Add a setting only when its default is wrong for your data. An explicit value
changes only its own default.

| Setting (in a dataset) | Default | Notes |
|---|---|---|
| `enabled` | `true` | `false` removes the dataset's work. Deleting a dataset removes its work and unused inputs. |
| `targets` | required | A list of columns, or an object of per-target settings. `[]` does no work. Columns are never filled implicitly. |
| `columns.NAME.unit` | `unknown` | Declared once per column; targets inherit it. No conversion or guessing. |
| `columns.NAME.unit_regime` | `native` | Add `regime: {column, value}` to use only rows of one retrieval regime. |
| `columns.NAME.column` | `NAME` | The CSV header, if different from the name you use. |
| `columns.NAME.derived_from` | `[]` | Same-dataset parent names (ancestry only; a parent need not be a column of the file and is never read). Derived columns cannot be targets or donors of their parents, under any alias. |
| `sha256` | none | Expected hash of the exact bytes; checked before any method runs. |
| `timestamp` | `timestamp` | Timestamp column. |
| `timezone` | none | Required only when timestamps have no offset: `UTC`, `+HH:MM`, `-HH:MM` or an IANA name. |
| `key` | generated | Row-key column; otherwise the first 12 characters of the file's SHA-256 and the row ordinal. |
| `group_by`, `select` | single series | See [Stations and series](#stations-and-series). |
| `interval` | series extent | Output crop. The extent ends one cadence after the last timestamp. |
| `validity` | series extent | Support used for fitting and filling. |
| `boundaries` | `[]` | Instants that split support; a gap across one is never bridged. |
| `exclusions`, `fit_exclusions` | `[]` | Excluded from support / from fitting only. |
| `missing.tokens` | empty cell | Exact cell text read as unavailable, e.g. `["NA", "-"]`. |
| `missing.sentinels` | `[]` | Finite values rejected, e.g. `[-9999]` or `{"at_most": -9990}`. |
| `columns.NAME.qc` | none | `{column, accept: [codes]}`; other codes are rejected. |
| `columns.NAME.observed`, `rejected`, `structural` | none | Mask columns of literal `True`/`False`. |
| `family` | none | Descriptive only; never sets cadence or preparation rules. |

At the top level: `purpose` defaults to `unspecified`; `allow_provisional` to
`false`; `models` to none; `limits` to `max_gaps` 1000, `max_method_calls` 3000,
`max_rows_per_series` 250000, `wall_seconds` 900, `memory_bytes` 536870912,
`scratch_bytes` 33554432 and `output_bytes` 134217728. Memory is a conservative
estimate, not a measured limit.

Nothing else is inferred. `cadence` is always required; there is no cadence
detection, resampling or averaging. New recipes get deterministic IDs, status
`user_configured` and evidence `user_supplied.configuration`; ecofill never
marks a recipe as archived or selected for you.

## Methods, gap lengths and routes

`method` with `max_gap` fills every gap up to `max_gap` with one method. The
duration classes are short (up to 2h), medium (over 2h up to 12h) and long
(over 12h up to 7 days); `max_gap` enables the classes it reaches and caps each
recipe exactly, so `"max_gap": "3h"` fills 2h30 gaps but not 4h gaps. Longer
gaps, and gaps over 7 days, stay whole and unresolved. `max_gap` cannot be
shorter than one cadence step.

```json
"targets": {
  "bar": {"method": "RF-X", "max_gap": "6h", "predictors": ["temp_out", "dew_point"],
          "parameters": {"n_estimators": 64}, "fallbacks": [{"method": "LIN"}]},
  "humidity": {"routes": {"short": {"method": "LIN"},
                          "medium": {"method": "RF-X", "predictors": ["temp_out", "dew_point"]},
                          "long": {"enabled": false, "reason": "not qualified"}}}
}
```

- A per-target object replaces the dataset's `method`/`max_gap` for that target.
  Shorthand and `routes` in the same place is an error, not a precedence rule.
- An explicit route's `max_gap` defaults to its class upper bound.
- Same-file predictors are column names. Another dataset or station needs an
  explicit identity: `{"dataset": "soil", "column": "t_10cm"}`, adding
  `"series": "S2"` for a grouped dataset (with or without `select`; an absent
  series fails at plan time). Donors must share cadence and grid phase
  (`donor_grid_mismatch` otherwise, before any method runs); derived descendants
  of a target are rejected.
- Names never decide physical identity. A file is identified by its location
  (one local file under any path or link, one `gs://` object, one `bq://` table)
  plus its series and CSV column. A column may be declared under several names
  only with the same unit, missing-value, QC, mask, regime and `derived_from`
  settings; it cannot be filled twice or be its own (or its parent's) donor, so
  a weaker QC declaration can never make rejected values usable again.
- `RF-X`/`RF-XL` need at least two measured predictors; `P-X`, `T-X`, `AVG-X`
  need at least one. Existing parameter defaults and support minimums apply.
- `P-…` and `T-…` methods name a declared model in `backend`; `AVG-…` methods use
  `"backend": {"P": "p", "T": "t"}`. Joint methods (`T-J`, `AVG-J`) need an
  explicit `"group": [2-4 targets]`, declared once on one target.
- TimesFM 3 and TabPFN 3.5 weights are licensed for non-commercial research.
  Declare `"purpose": "non-commercial research"` at the top level or in the model
  (`"models": {"t": {"kind": "timesfm3", …, "purpose": "non-commercial research"}}`).
  Without it, validation fails before any model work. See [models](models.md).
- A saved recipe may be supplied in an explicit route with its `frozen_recipe`
  object, `frozen_binding`, `evidence_status`, `evidence` and `id`. It keeps its
  frozen settings and needs matching known units, cadence and bindings. The
  recipe is the object itself; a file path is rejected and never read.

## Time and clocks

Timestamps are ISO 8601: `YYYY-MM-DD HH:MM[:SS]` or `YYYY-MM-DDTHH:MM[:SS]`, with
an optional `Z` or `+HH:MM` offset. Values with offsets are converted to UTC.
Values without an offset need `timezone`; ambiguous and nonexistent local times
at daylight-saving changes are errors. There is no implicit UTC or computer
timezone, and an offset that contradicts `timezone` is an error.

The grid phase is the first timestamp of each series; 15-minute data starting at
00:45 keeps that phase. A reading off that grid, or two readings with the same UTC
time, is an error; nothing is rounded or merged. Missing grid rows are inserted
as gaps. The saved bundle keeps the exact input bytes and the UTC normalization
(offsets seen, timezone, row counts).

## Values, quality and masks

A cell is unavailable when it is empty, a configured missing token, `NaN`/`inf`
text, or anything other than a decimal number (an exponent such as `1e3` is allowed). Finite values such as `9999` or `-9999` are
values unless listed as sentinels; sentinel and QC-rejected values are kept in
`raw.csv.gz` and never used. Without a QC column, finite values are used as
supplied; ecofill records that they had no independent quality assessment. A
configured QC or mask column that is absent from the file is an error.

## Stations and series

Without `group_by`, a dataset is one series. If the file has a `station_id`
column with more than one value, plan stops and asks for `group_by`, even when
the stations cover different dates. Other identity columns are not guessed.

`"group_by": "station_id"` fills each station separately, with its own extent,
keys and results (`weather.S1`, `weather.S2`, …). `"select": ["S1"]` restricts
the work to named stations; an unknown station is an error.

## Validation, plans and results

`validate` reads only the configuration. It reports `input_facts: pending` and
lists the facts that `plan` and `run` will resolve from the input (hashes,
extents, series). `plan` and `run` read each active input once, hash the exact
bytes they use and save:

- `config.authored.json`: your configuration as written;
- `config.resolved.json`: the resolved `ecofill.operation/2` configuration,
  including every default, the input hashes, series extents and clocks;
- `inputs/NAME.snapshot`: the exact input bytes; and the usual plan, masks,
  dispositions, summary, manifest and terminal files.

When planned gaps or method calls exceed the limits, the plan's `budget` says so
and how to narrow the work or raise a limit. Inputs are never truncated. Input
size limits stop `plan` and `run` with a named limit (`location` such as
`/limits/scratch_bytes`, `/limits/memory_bytes` or `/limits/max_rows_per_series`)
and one of the codes `source_exceeds_scratch_bytes`,
`decoded_source_exceeds_scratch_bytes`, `decoded_source_exceeds_memory_bytes`,
`source_rows_exceed_limit`, `prepared_rows_exceed_limit`,
`prepared_memory_exceeds_limit`, `inputs_exceed_scratch_bytes` or
`prepared_inputs_exceed_memory_bytes`; select fewer series or a shorter period,
or raise that limit within your deployment policy.

Errors are JSON with `status`, `error_type`, a stable `code`, a `location` such
as `/datasets/weather/max_gap`, and an `explanation`. They never include file
paths, cell values, secrets or tracebacks. Exit codes: 0 complete or valid,
1 failed or invalid, 2 partial, 3 in progress.

To repeat a run on the same bytes, copy `config.resolved.json`, point each source
`uri` at its `inputs/NAME.snapshot` and choose a new output; the saved `sha256`
rejects any change (`examples/operation_simple_demo.py replay-config` does this).
A changed input file under the original configuration is a new input identity:
plan it again with a new run ID. Reusing a run ID returns that run's saved result.

## Cloud inputs

A `gs://` CSV may omit `sha256` and `generation`: the actual object generation is
recorded and the downloaded bytes are hashed before use. Supplied values must
match. BigQuery inputs (`bq://project.dataset.table`) keep their explicit
settings: `key`, `group_by`, `select`, `bigquery.intervals`, and the `cloud`
block with its snapshot time and byte/row limits. ecofill never discovers a
table's extent or order. The operator's deployment policy still sets allowed
prefixes, tables and ceilings; see the [operation guide](operation.md).

## Docker commands for a future image

These drafts need an image built from a reviewed release containing this
support; the current published image rejects them. Mount the working directory
read-only and a separate writable output:

```sh
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
  --mount type=bind,src=ABSOLUTE_WORK,dst=/work,readonly \
  --mount type=bind,src=ABSOLUTE_OUTPUT,dst=/output \
  --workdir /work IMAGE validate --config config.json
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
  --mount type=bind,src=ABSOLUTE_WORK,dst=/work,readonly \
  --mount type=bind,src=ABSOLUTE_OUTPUT,dst=/output \
  --workdir /work IMAGE plan --config config-docker.json --output /output/first-plan
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
  --mount type=bind,src=ABSOLUTE_WORK,dst=/work,readonly \
  --mount type=bind,src=ABSOLUTE_OUTPUT,dst=/output \
  --workdir /work IMAGE run --config config-docker.json --run-id first
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
  --mount type=bind,src=ABSOLUTE_OUTPUT,dst=/output,readonly \
  IMAGE inspect --run-uri /output/runs/first
```

`config-docker.json` is `config.json` with `"output": {"uri": "/output"}`.

## Moving from operation/1

Explicit `ecofill.operation/1` files keep their exact validation, identities and
results, and are never reinterpreted as simple input. To move a dataset, keep
its file, cadence, targets, units and method, drop the generated metadata
(instruments, lineage, policy, routes with class bounds, recipe IDs) and add only
what differs from the defaults above. Project profiles and saved archive recipes
remain available through operation/1 or explicit saved routes.

## Local container testing

See [container testing](container-testing.md) for the selected-source Docker build,
synthetic engineer config, independent readback and snapshot replay. Its candidate
receipt separates local qualification from the older published image.
