# Download the published image and test local Parquet files

Google Artifact Registry stores the image. Docker runs it on your computer,
reading a local Parquet file and writing a local result folder. Google is not
involved in processing. No Cloud Run job, bucket mount or BigQuery access is
needed for this test. Engineers can later provide mounted paths the same way.

Image: `ecofill`, tag `m023-parquet-6969fb4`.
Use this immutable registry URI:

```text
us-central1-docker.pkg.dev/manglaria-staging/cloud-run-source-deploy/ecofill@sha256:b72bf8e7bb0ad5d58ab22918d9d45bd51b32132610c4b988d164559c6ba8339f
```

[Release identities and test results](parquet-release.json).
The `m017-0d0a16a` image and older cloud drafts are superseded; use this guide
for the current local-file workflow. The image includes LIN, RF and Parquet
support. These examples do not require foundation-model weights.

## 1. Get the repository and authenticate the download

Start Docker Desktop in Linux-container mode. Install Git and the Google Cloud
CLI if your workstation does not already have them. Your account needs Artifact
Registry read access to `manglaria-staging`.

For an existing clean checkout:

```powershell
cd ecofill
git pull --ff-only origin main
```

For a new checkout instead, run `git clone https://github.com/jequihua/ecofill.git`
then `cd ecofill`. All later commands run from this repository root.

```powershell
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
$image = 'us-central1-docker.pkg.dev/manglaria-staging/cloud-run-source-deploy/ecofill@sha256:b72bf8e7bb0ad5d58ab22918d9d45bd51b32132610c4b988d164559c6ba8339f'
docker pull --platform linux/amd64 $image
```

If you are already logged in and Docker authentication is configured, start at
the image assignment and pull. Credentials stay on the host; do not mount them
into ecofill. Processing below runs with network access disabled.

## 2. Supply the unchanged Davis table

Place the whole exported `davis_weather_combined_s2_filt` table in one closed
file named `davis_weather_combined_s2_filt.parquet`. Retain all 16 original
columns and station/date rows. The timestamp is UTC-aware, humidity and solar_rad
are nullable integers, other measurements are floating point, and identity
columns are strings. No filtering, column renaming, units or QC columns need
to be added for these examples.

Either put the file in `./inputs/` or point `$inputDir` at the existing folder.
The example selects stations and dates in its JSON config. It still decodes the
whole file, with a 400,000-row limit. Larger future exports need an explicit
review of the row/memory limits; narrowing station selection does not reduce
the number of rows decoded from the file.

## 3. Set the local mounts (PowerShell)

```powershell
$inputDir = (Resolve-Path './inputs').Path   # Or your existing Parquet folder
$outputDir = Join-Path (Get-Location).Path 'output-davis'
$configDir = (Resolve-Path './examples/operation').Path
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$runId = 'weather-main-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$dockerArgs = @('run','--rm','--pull=never','--platform','linux/amd64',
  '--user','10001:10001','--network','none','--read-only','--cpus','4',
  '--memory','12g','--memory-swap','12g','--pids-limit','256',
  '--tmpfs','/tmp:rw,size=256m,mode=1777',
  '--mount',"type=bind,src=$inputDir,dst=/inputs,readonly",
  '--mount',"type=bind,src=$configDir,dst=/configs,readonly",
  '--mount',"type=bind,src=$outputDir,dst=/output",$image)
```

Docker must be able to share these folders. Output must be writable by UID
10001, and exactly one process may write each run ID. On Linux, arrange that
ownership on the dedicated output folder before running. The container has
4 CPUs and at most 12 GiB RAM; allow that capacity in Docker Desktop.

## 4. Validate, plan, run, inspect

```powershell
docker @dockerArgs validate --config /configs/davis-weather.json
if ($LASTEXITCODE -ne 0) { throw 'Validation failed' }

docker @dockerArgs plan --config /configs/davis-weather.json --output "/output/plan-$runId"
if ($LASTEXITCODE -ne 0) { throw 'Planning failed' }
```

Read the plan. Proceed only with `budget_exceeded: false`, no unavailable
sources, and the intended stations/targets. Then:

```powershell
docker @dockerArgs run --config /configs/davis-weather.json --run-id $runId
if ($LASTEXITCODE -ne 0) { throw 'Run did not complete; inspect its saved evidence' }
docker @dockerArgs inspect --run-uri "/output/main/runs/$runId"
if ($LASTEXITCODE -ne 0) { throw 'Bundle inspection failed' }
```

The main config selects RBMNN-AB-01 and RBRL-COL-01 during January–March 2025.
It fills whole bounded gaps with linear interpolation: up to 48 hours for
temp_out, humidity, dew_point, heat_index and bar; up to two hours for rain,
rain_rate, solar_rad, solar_energy and et. Wind speed/chill are excluded.
Change `select`, both `interval` and `validity`, targets or caps in the JSON,
then plan again with a new run ID. Do not modify the input file.

The archived test input has 359,337 rows. Its expected main result is **complete,
1,045 filled gaps / 2,523 filled cells**, with 248 gaps unresolved because they
lack support or exceed the rules. A newer export may produce different counts.
Exit codes are 0 complete/valid, 1 failed/invalid, 2 partial and 3 incomplete.

## 5. Locate the local Parquet outputs

```powershell
$runDir = Join-Path $outputDir "main/runs/$runId"
Get-Content (Join-Path $runDir 'summary.json')
$manifest = Get-Content (Join-Path $runDir 'manifest.json') -Raw | ConvertFrom-Json
$manifest.datasets | ForEach-Object {
  [PSCustomObject]@{
    Station = $_.series
    CompletedParquet = Join-Path $runDir $_.files.completed
  }
}
```

Each station has completed Parquet data, masks and cell states. `manifest.json`,
`terminal.json`, `provenance.json`, the summary and the original input snapshot
remain alongside them. Original measured values remain unchanged; filled values
are marked. Unsupported gaps remain null.

This is a result bundle with selected targets, not a replacement of the whole
16-column table. Engineers own any keyed merge into the source table and any
subsequent BigQuery load. The host output folder already contains the results;
no `docker cp` or upload is needed.

## 6. Optional multivariate RF test

Use a separate run ID and the same input file and mounts:

```powershell
$rfRunId = 'weather-rf-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
docker @dockerArgs validate --config /configs/davis-weather-rf.json
if ($LASTEXITCODE -ne 0) { throw 'RF validation failed' }
docker @dockerArgs plan --config /configs/davis-weather-rf.json --output "/output/plan-$rfRunId"
if ($LASTEXITCODE -ne 0) { throw 'RF planning failed' }
# Check the plan before proceeding.
docker @dockerArgs run --config /configs/davis-weather-rf.json --run-id $rfRunId
if ($LASTEXITCODE -ne 0) { throw 'RF run did not complete' }
docker @dockerArgs inspect --run-uri "/output/rf/runs/$rfRunId"
```

This fills bar at RBRL-COL-01 during January–February 2026 using original temp_out
and humidity as predictors. It uses 128 trees, depth 12, leaf size 5, seed 0 and
one worker. The archived input yields **four filled gaps / 50 cells**. Results
are in `output-davis/rf/runs/<rfRunId>/`.

## POSIX equivalent

After the same Google authentication and repository setup:

```sh
IMAGE='us-central1-docker.pkg.dev/manglaria-staging/cloud-run-source-deploy/ecofill@sha256:b72bf8e7bb0ad5d58ab22918d9d45bd51b32132610c4b988d164559c6ba8339f'
docker pull --platform linux/amd64 "$IMAGE"
INPUT_DIR="$PWD/inputs"; OUTPUT_DIR="$PWD/output-davis"; CONFIG_DIR="$PWD/examples/operation"
mkdir -p "$OUTPUT_DIR"
# Make this dedicated output directory writable by UID 10001 through your host's ownership policy.
RUN_ID="weather-main-$(date +%Y%m%d-%H%M%S)"
set -- --rm --pull=never --platform linux/amd64 --user 10001:10001 --network none --read-only --cpus 4 --memory 12g --memory-swap 12g --pids-limit 256 --tmpfs /tmp:rw,size=256m,mode=1777 --mount "type=bind,src=$INPUT_DIR,dst=/inputs,readonly" --mount "type=bind,src=$CONFIG_DIR,dst=/configs,readonly" --mount "type=bind,src=$OUTPUT_DIR,dst=/output" "$IMAGE"
docker run "$@" validate --config /configs/davis-weather.json || exit 1
docker run "$@" plan --config /configs/davis-weather.json --output "/output/plan-$RUN_ID" || exit 1
# Inspect the plan before the next command.
docker run "$@" run --config /configs/davis-weather.json --run-id "$RUN_ID" || exit 1
docker run "$@" inspect --run-uri "/output/main/runs/$RUN_ID"
```
