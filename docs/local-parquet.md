# Local Parquet files and mounted volumes

For the published image and a complete local-input/local-output test, start with
[the registry download and Davis testing guide](registry-local-test.md).

For the unchanged 16-column Davis weather table, use the tested
[Davis first-run configs and mount commands](davis-weather-first-run.md).

Use one immutable, closed Parquet file per source. Engineers own extraction,
combining export shards, bucket mounts, scheduling and loading the chosen result
tables. Ecofill reads local paths and requires no Google credentials. This source
adds Parquet support; earlier images and the historical container candidate do
not gain it through source edits. Public synchronization, registry publication and
real bucket-mount qualification are separate steps.

The optional Python extra is `ecofill[parquet]` with PyArrow 25.0.1. The local image
includes it. CSV-only use and offline `validate` do not import Arrow. Validate
does not read inputs or load models. A missing extra gives
`parquet_dependency_missing` when Parquet is actually used.

## Configuration and typed evidence

```json
{
  "datasets": {
    "davis_weather": {
      "uri": "/inputs/weather.parquet",
      "cadence": "30min",
      "targets": ["bar"],
      "method": "LIN",
      "max_gap": "2h"
    }
  },
  "output": {
    "uri": "/output",
    "format": "parquet",
    "publication": "single_writer"
  }
}
```

A local `.parquet` suffix infers the format; `format: parquet` also accepts an
extensionless file. Resolved configs persist the format and hash for snapshot
replay. Other supported inputs are CSV and CSV-gzip. Parquet has internal
compression; externally gzipped Parquet, directories, sharded datasets and direct
remote Parquet paths are rejected. Existing remote CSV operation remains supported.
The engineer `sources/routes/output` spelling accepts source `format: parquet`
and the same output options; its `linear` route keeps the existing LIN semantics.

Use flat numeric targets/donors, timestamp or ISO string clocks, string/integer
keys and optional Boolean masks/QC. Timestamp s/ms/us units are supported; ns must
be losslessly representable in microseconds. Aware clocks normalize to UTC; naive
clocks need an explicit timezone. Null, ambiguous, duplicate and off-grid clocks
fail; ecofill never rounds or resamples. Float64 modeling rejects lossy integer
measurements, including numeric strings with decimal or exponent spelling of an
integer. Finite literals that overflow float64 fail before model execution.
Fractional decimals convert explicitly to float64; original decimal
bytes remain in the snapshot. Keys, including leading zeros and large integers,
never pass through float. Unsupported selected nested/binary/extension types fail;
unused nested columns stay in the original snapshot without selected decoding.

Units and QC are optional. For explicit QC use, for example,
`columns: {bar: {qc: {column: qc_bar, accept: ["True"]}}}` for Boolean QC.
Group stations with `group_by: station_id`; `select: ["001"]` narrows work.
RF-X needs explicit measured predictors, such as
`[{dataset: donors, column: d1}, {dataset: donors, column: d2}]`.
A grouped donor also names `series`. Targets, aliases and descendants cannot
donate to themselves. Disabled/deleted datasets do no work; adding a source never
selects its targets implicitly. Existing exclusions, skip routes and original-only
support remain in force.

Input bytes are captured once, hashed and bounded, then decoded from that captured
buffer. Footer/Thrift metadata, projected row groups/columns, batches, typed Python
rows and regular grids have explicit limits. Memory estimates are conservative
checks, not a hard allocator quota; the container limit is the outer bound. Export
smaller files upstream when `/limits/...` diagnostics identify excessive work.

## Local build

Run from the supplied clean source snapshot or the explicitly selected local
candidate tree. It is not a published source commit. Preserve the source manifest
and record your newly built image identity. Docker Desktop needs its Linux engine;
allow at least 25 GiB free disk. Supply the exact Linux Arrow wheel as an explicit
named build context, so the dependency is acquired only once:

```sh
mkdir arrow-wheel
curl -fL 'https://files.pythonhosted.org/packages/60/c9/711ca85d79f1ec98f29a5eae2b051e25b4ecec5de3e3c0e2d5c5dcb15664/pyarrow-25.0.1-cp312-cp312-manylinux_2_28_x86_64.whl' -o arrow-wheel/pyarrow-25.0.1-cp312-cp312-manylinux_2_28_x86_64.whl
echo '5389cdf79447ed1515c9e31620e6e1e2302249564d603f2ad727d4f6d313e4c3  arrow-wheel/pyarrow-25.0.1-cp312-cp312-manylinux_2_28_x86_64.whl' | sha256sum -c -
docker buildx build --load --platform linux/amd64 --build-context "parquet_wheel=$(pwd)/arrow-wheel" -f docker/Dockerfile -t ecofill:local-parquet .
image_id=$(docker image inspect --format '{{.Id}}' ecofill:local-parquet)
mkdir -m 777 candidate-work
```

```powershell
New-Item -ItemType Directory arrow-wheel
$wheel = 'arrow-wheel/pyarrow-25.0.1-cp312-cp312-manylinux_2_28_x86_64.whl'
Invoke-WebRequest 'https://files.pythonhosted.org/packages/60/c9/711ca85d79f1ec98f29a5eae2b051e25b4ecec5de3e3c0e2d5c5dcb15664/pyarrow-25.0.1-cp312-cp312-manylinux_2_28_x86_64.whl' -OutFile $wheel
if ((Get-FileHash $wheel -Algorithm SHA256).Hash.ToLower() -ne '5389cdf79447ed1515c9e31620e6e1e2302249564d603f2ad727d4f6d313e4c3') { throw 'wheel hash mismatch' }
docker buildx build --load --platform linux/amd64 --build-context "parquet_wheel=$((Resolve-Path arrow-wheel).Path)" -f docker/Dockerfile -t ecofill:local-parquet .
if ($LASTEXITCODE) { throw 'build failed' }
$imageId = docker image inspect --format '{{.Id}}' ecofill:local-parquet
New-Item -ItemType Directory candidate-work
```

## POSIX test commands

Run from the public source root with the built `image_id`. The fixture is small
and synthetic: two grouped stations use donor RF, a weather series uses LIN, and
unused nested columns remain in typed snapshots. No weights are needed.

```sh
helper="$(pwd)/docker/qualify_parquet.py"
work="$(pwd)/candidate-work"
docker run --rm --platform linux/amd64 --user 10001:10001 --network none --read-only --cpus 4 --memory 12g --memory-swap 12g --pids-limit 256 --tmpfs "/tmp:rw,size=256m,mode=1777" --mount "type=bind,src=$work,dst=/output" --mount "type=bind,src=$helper,dst=/harness/qualify.py,readonly" --entrypoint python "$image_id" -I -B /harness/qualify.py prepare --output /output/fixture
inputs="$work/fixture/inputs"
outputs="$work/fixture/output"
candidate() {
  entry=$1
  shift
  docker run --rm --platform linux/amd64 --user 10001:10001 --network none --read-only --cpus 4 --memory 12g --memory-swap 12g --pids-limit 256 --tmpfs /tmp:rw,size=256m,mode=1777 --mount "type=bind,src=$inputs,dst=/inputs,readonly" --mount "type=bind,src=$outputs,dst=/output" --mount "type=bind,src=$helper,dst=/harness/qualify.py,readonly" --entrypoint "$entry" "$image_id" "$@"
}
candidate python -I -B /harness/qualify.py provenance
candidate ecofill validate --config /inputs/config.json
candidate ecofill plan --config /inputs/config.json --output /output/plan
candidate ecofill run --config /inputs/config.json --run-id first
candidate ecofill inspect --run-uri /output/single/runs/first
candidate python -I -B /harness/qualify.py readback --output /output/single/runs/first
candidate python -I -B /harness/qualify.py replay-config --run /output/single/runs/first --output /output/replay.json --output-uri /output/replay
candidate ecofill run --config /output/replay.json --run-id copy
candidate python -I -B /harness/qualify.py readback --output /output/single/runs/first --replay /output/replay/runs/copy
```

## PowerShell test commands

Use a new `candidate-work` for the second shell workflow. Array arguments keep
mount paths with spaces intact; every Docker exit is checked.

```powershell
$helper = (Resolve-Path docker/qualify_parquet.py).Path
$work = (Resolve-Path candidate-work).Path
docker run --rm --platform linux/amd64 --user 10001:10001 --network none --read-only --cpus 4 --memory 12g --memory-swap 12g --pids-limit 256 --tmpfs "/tmp:rw,size=256m,mode=1777" --mount "type=bind,src=$work,dst=/output" --mount "type=bind,src=$helper,dst=/harness/qualify.py,readonly" --entrypoint python $imageId -I -B /harness/qualify.py prepare --output /output/fixture
if ($LASTEXITCODE) { throw 'fixture failed' }
$inputs = (Resolve-Path (Join-Path $work 'fixture/inputs')).Path
$outputs = (Resolve-Path (Join-Path $work 'fixture/output')).Path
function Invoke-Candidate([string]$Entry, [string[]]$Arguments) {
  $dockerArgs = @('run','--rm','--platform','linux/amd64','--user','10001:10001','--network','none','--read-only','--cpus','4','--memory','12g','--memory-swap','12g','--pids-limit','256','--tmpfs','/tmp:rw,size=256m,mode=1777','--mount',"type=bind,src=$inputs,dst=/inputs,readonly",'--mount',"type=bind,src=$outputs,dst=/output",'--mount',"type=bind,src=$helper,dst=/harness/qualify.py,readonly",'--entrypoint',$Entry,$imageId) + $Arguments
  docker @dockerArgs
  if ($LASTEXITCODE) { throw "candidate failed: $Entry" }
}
Invoke-Candidate python @('-I','-B','/harness/qualify.py','provenance')
Invoke-Candidate ecofill @('validate','--config','/inputs/config.json')
Invoke-Candidate ecofill @('plan','--config','/inputs/config.json','--output','/output/plan')
Invoke-Candidate ecofill @('run','--config','/inputs/config.json','--run-id','first')
Invoke-Candidate ecofill @('inspect','--run-uri','/output/single/runs/first')
Invoke-Candidate python @('-I','-B','/harness/qualify.py','readback','--output','/output/single/runs/first')
Invoke-Candidate python @('-I','-B','/harness/qualify.py','replay-config','--run','/output/single/runs/first','--output','/output/replay.json','--output-uri','/output/replay')
Invoke-Candidate ecofill @('run','--config','/output/replay.json','--run-id','copy')
Invoke-Candidate python @('-I','-B','/harness/qualify.py','readback','--output','/output/single/runs/first','--replay','/output/replay/runs/copy')
```

Each shell expects 24 fills, 852 preserved observations and three indexed
partitions. Each partition has 300 regular-grid rows. Leading gaps and six-step
oversize gaps remain null with reasons; an absent timestamp has no original key.
Replay compares all typed table bytes. To test ordinary strict publication use
`config-strict.json` with a fresh run ID, inspect `/output/strict/runs/ID`, then
read it back. `config-engineer.json`, `config-linear.json` and `config.yaml` each
exercise one weather partition: eight fills and 284 original observations.

## Consume and rerun safely

`manifest.json` has a `datasets` index with dataset/source/series, every typed file,
columns/units, schema and original-key audit paths. Discover `completed.parquet`
there instead of parsing directory names. Its timestamp is explicit
`timestamp[us, UTC]`, targets are float64, masks are Boolean, and unresolved cells
are null. Raw tables retain prepared values including rejected numeric cells;
original typed bytes remain in `inputs/*.snapshot`. These are selected-target,
regular-grid outputs, not copies of all unused input columns. Inserted timestamps
have no source key; source-reason audits retain exact original keys/row positions.
Writer settings are pinned Snappy/Parquet 2.6, 4,096-row groups and microseconds;
Arrow/writer versions and actual file hashes are recorded. Different library
versions are not promised byte equality.

For a mounted volume, set `publication: single_writer`, enforce one writer per
fresh unique run directory externally, and point TMPDIR at bounded true local
scratch, separate from the mount. Each file encodes/closes locally, copies without
hard links, rename, locks or output fsync, then closes and passes size/hash
readback. Manifest and terminal publish last. This provides no distributed lock
or atomic mkdir/O_EXCL guarantee. Filesystem doubles test faults; they do not
qualify a real bucket mount or its cache/permissions.

Keep failed directories. CLI exit 0 means complete/valid, 1 failed/invalid,
2 partial and 3 incomplete/in-progress. Accept complete/partial/failed bundles
only when inspect verifies terminal, manifest and every hash; incomplete/corrupted
uploads are unusable. A conflicting request under the same run ID is rejected.
Never overwrite a run or input. Inspect a known completed run again, or choose a
fresh ID for new work. Engineers validate selected tables before loading their
chosen BigQuery destinations; ecofill writes no BigQuery output table, mounts no
bucket and uploads no local Parquet data itself.

When Parquet output needs an unavailable extra, `run` rejects it before claiming
a new run directory. Install the pinned extra, then retry the same ID. Existing
completed runs remain inspectable/idempotent without Arrow. A Parquet-input run
with CSV output can instead retain an inspectable failed JSON bundle; retain that
failure and use a fresh ID after restoring the extra. Existing incomplete claims
are never repaired or deleted automatically. Validate and integrity inspection
remain independent of Arrow. Explicit missing tokens and nonfinite/QC/sentinel
precedence are unchanged; keys never pass through measurement conversion.

Captured inputs are named `inputs/<source-map-key>.snapshot` using the validated
resolved source key. Free-form lineage labels remain metadata and cannot name
files. Shared physical inputs retain one snapshot association per required key.
Local artifact names must be canonical relative POSIX paths contained in the
bundle, including through existing parent links; traversal and noncanonical names
are rejected before publication. Existing images require the correction candidate
to gain this behavior.

If snapshot publication fails after input capture, the run fails without retrying
that snapshot. This is separate from a genuinely missing input, which retains the
partial-input behavior. Keep the failed output and inspect it; if terminal
publication also failed, the bundle is incomplete and must not be consumed.

## Original local candidate

This historical candidate predates the numeric-string and missing-extra
corrections above; its receipt remains unchanged. Use the correction candidate
and its new receipt for those behaviors. Existing public/registry images do not
gain a correction from source edits.

The M022 local candidate `ecofill:m022-parquet-0ea053b10ad8` has Docker image ID
`sha256:94e6404fde2b87b4e86dd5d8d93ebf80e28a356c00fe15c7bc40c8f801b381ef`. It is unpublished, built from an
uncommitted disposable projection, with PyArrow 25.0.1 as the only dependency
addition. Both shell workflows, strict publication, engineer alias/YAML, the
16-case Linux type/limit/publication matrix and the legacy CSV path ran under
the documented limits. Each grouped workflow preserved 852 observations and
filled 24 cells; independent Arrow readback verified typed tables and identical
replay bytes. These observations cover local Docker Desktop Linux/amd64 and
filesystem doubles. Real bucket mounts and native foundation models are untested.
