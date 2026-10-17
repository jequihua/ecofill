# First run on the unchanged Davis weather table

**Current registry release:** follow [download and local-file testing](registry-local-test.md)
to pull the published Parquet image and use these unchanged configs. The local
image identity and qualification recorded below describe the original test.

Put the whole closed export at `inputs/davis_weather_combined_s2_filt.parquet`.
Keep all stations, dates and 16 original columns: string primary_key/station_id/
data_source, UTC timestamp, nullable integer humidity/solar_rad and float
measurements. Do not filter, rename, pivot, convert units or add QC/helper columns.
No preprocessing helper is required. Finite values are used as supplied, without
invented units, thresholds or a global missing-value sentinel.

The **unpublished local image** is `ecofill:m022-r4-parquet-d867b26d355c`, exact ID
`sha256:b72bf8e7bb0ad5d58ab22918d9d45bd51b32132610c4b988d164559c6ba8339f`.
It must already be present. These commands neither build nor pull it; registry
availability is separate. Prepare output writable by UID 10001 and enforce one
writer per fresh run ID. Actual bucket/cache/permission testing remains yours.

## Choose work in JSON

[davis-weather.json](../examples/operation/davis-weather.json) groups station_id,
retains primary_key, and selects RBMNN-AB-01/RBRL-COL-01 for 2025 Q1.
`interval` selects output; `validity` constrains usable support. Edit both dates
together. Edit select/targets/max_gap without changing the Parquet. Decoding still
checks/reads the whole file: the 400,000-row bound admits the archived 359,337 rows,
not just one station. Plan before running; narrow work if limits are exceeded.

| Targets | LIN cap | Classes allowed |
| --- | --- | --- |
| temp_out, humidity, dew_point, heat_index, bar | 48 h | short <=2 h; medium >2–12 h; long >12–48 h |
| rain, rain_rate, solar_rad, solar_energy, et | 2 h | short only |
| wind_speed, wind_chill | excluded | neither targets nor predictors |

Only whole bounded eligible gaps fill. Edge/excessive gaps and absent support
stay null with reasons. All-null solar energy is retained, never invented.

## Mount and run

From the public repository root, set three directories and a fresh run ID. Inputs
and configs mount read-only; bounded local `/tmp` stays separate from output.

```sh
INPUT_DIR="$PWD/inputs"
OUTPUT_DIR="$PWD/output"
CONFIG_DIR="$PWD/examples/operation"
RUN_ID="weather-main-001"
IMAGE="sha256:b72bf8e7bb0ad5d58ab22918d9d45bd51b32132610c4b988d164559c6ba8339f"
mkdir -p "$OUTPUT_DIR"
set -- --pull=never --platform linux/amd64 --user 10001:10001 --network none --read-only --cpus 4 --memory 12g --memory-swap 12g --pids-limit 256 --tmpfs /tmp:rw,size=256m,mode=1777 --mount "type=bind,src=$INPUT_DIR,dst=/inputs,readonly" --mount "type=bind,src=$CONFIG_DIR,dst=/configs,readonly" --mount "type=bind,src=$OUTPUT_DIR,dst=/output" "$IMAGE"
docker run "$@" validate --config /configs/davis-weather.json
docker run "$@" plan --config /configs/davis-weather.json --output /output/plan-main
docker run "$@" run --config /configs/davis-weather.json --run-id "$RUN_ID"
docker run "$@" inspect --run-uri "/output/main/runs/$RUN_ID"
```

PowerShell:

```powershell
$inputDir = (Resolve-Path ./inputs).Path
$outputDir = Join-Path (Get-Location).Path 'output'
$configDir = (Resolve-Path ./examples/operation).Path
$runId = 'weather-main-001'
$image = 'sha256:b72bf8e7bb0ad5d58ab22918d9d45bd51b32132610c4b988d164559c6ba8339f'
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$runArgs = @('run','--pull=never','--platform','linux/amd64','--user','10001:10001','--network','none','--read-only','--cpus','4','--memory','12g','--memory-swap','12g','--pids-limit','256','--tmpfs','/tmp:rw,size=256m,mode=1777','--mount',"type=bind,src=$inputDir,dst=/inputs,readonly",'--mount',"type=bind,src=$configDir,dst=/configs,readonly",'--mount',"type=bind,src=$outputDir,dst=/output",$image)
docker @runArgs validate --config /configs/davis-weather.json
docker @runArgs plan --config /configs/davis-weather.json --output /output/plan-main
docker @runArgs run --config /configs/davis-weather.json --run-id $runId
docker @runArgs inspect --run-uri "/output/main/runs/$runId"
```

Exit 0 means completed/valid, 1 failed/invalid, 2 partial, 3 incomplete/in-progress.
Inspect before consuming; keep failures/incomplete directories. Repeating the
same completed request returns its saved summary without new calls. A conflicting
config under that ID is rejected; use a fresh ID for new work.

## Optional RF profile

[davis-weather-rf.json](../examples/operation/davis-weather-rf.json) uses the same
whole file, RBRL-COL-01 in January–February 2026 and only bar as target. Switch
every config argument to that filename, plan output to `/output/plan-rf`, inspect
path to `/output/rf/runs/<your-run-id>`, and choose a separate run ID.
It uses original same-station temp_out/humidity, 128 trees, depth 12, minimum leaf
5, seed 0, one job, centered 180-day support clipped to validity, minimum 168 rows
and cap 4,096. There is no LIN fallback or use of imputed training truth.

## Observed archive result

The 2026-09-29 archive has 359,337 rows/29 stations/16 typed columns, independently
reconciled cell by cell. This is not today's cloud data. Main completed 1,045 LIN
gap fills (2,523 cells), preserving 33,514 original observations. Every smooth
target had short/medium/long fills at RBMNN-AB-01; RBRL-COL-01 had short/medium
fills and no long smooth gaps. RF filled the four 1/3/5/41-hour gaps (50 cells)
using 1,366 original complete training rows. Four RF attempts consume eight
conservative method-call units. This proves operation, not accuracy or superiority.
Future input must be planned again and can have different counts/support.

## Find and load results

Open `/output/main/runs/<id>/manifest.json` (or `rf/runs/<id>`). Its `datasets`
entries identify stations in `series` and map `files.completed` to each
`completed.parquet`; do not guess folder names. Tables have explicit UTC timestamps
and selected float64 targets, with unresolved nulls. Integer source evidence stays
integer in the snapshot; predictions may be fractional. Masks, states and gap
audits explain observed/reconstructed/unresolved cells. Terminal/manifest hashes,
provenance.json and `inputs/davis_weather.snapshot` retain reproducibility evidence.

This is a **run bundle, not a full 16-column replacement table**. Unselected
stations and unused columns, including wind, remain intact in the whole snapshot.
You own a keyed merge into your full table and any BigQuery writeback. Ecofill
does not mount buckets, upload results or merge the table automatically.
