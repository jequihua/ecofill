# Test a local container candidate

The simple configuration workflow is available in public source. A Git update
does not update an existing container. See [container-candidate.json](container-candidate.json)
for the observed build-source commit, local image ID, qualification and limitations.
An image ID identifies a local Docker object; it is not a registry manifest digest.
The [published release](release.json) is a separate, older image that expects
explicit `ecofill.operation/1` configuration. Weights are not inside either image.

Use a clean checkout at one selected full source SHA. Keep the source SHA distinct
from a later documentation commit. These commands build locally and perform no
registry publication or Google requests. Docker Desktop needs its Linux engine.
Allow 25 GiB free disk before building. The pinned base is Python 3.12, Linux amd64;
the hash-locked dependencies and their licenses are in `docker/`.

## POSIX commands

The commands select the tested source SHA from the candidate receipt. Choose a
different full SHA only for a deliberately new, unqualified build.
Use a new fixture directory for each test. Run from the public checkout:

```sh
ecofill_sha=94174ef300d63d0cfeee44191e672f31a2f78390
git clone https://github.com/jequihua/ecofill.git ecofill-source
cd ecofill-source
git checkout "$ecofill_sha"
image="ecofill:local-${ecofill_sha}"
docker build --load --platform linux/amd64 -f docker/Dockerfile -t "$image" .
python -I -S -B docker/qualify_container.py prepare --public . --output candidate-fixture
inputs="$(pwd)/candidate-fixture/inputs"
outputs="$(pwd)/candidate-fixture/output"
harness="$(pwd)/docker/qualify_container.py"
legacy="$(pwd)/examples/operation_demo.py"
image_id=$(docker image inspect --format '{{.Id}}' "$image")
candidate() {
  entry=$1
  shift
  docker run --rm --platform linux/amd64 --user 10001:10001 --network none \
    --read-only --cpus 4 --memory 12g --memory-swap 12g --pids-limit 256 \
    --tmpfs /tmp:rw,size=256m,mode=1777 \
    --mount "type=bind,src=$inputs,dst=/inputs,readonly" \
    --mount "type=bind,src=$outputs,dst=/output" \
    --mount "type=bind,src=$harness,dst=/harness/qualify.py,readonly" \
    --mount "type=bind,src=$legacy,dst=/harness/legacy.py,readonly" \
    --entrypoint "$entry" "$image_id" "$@"
}
```

Fixture preparation gives only the new synthetic output directories mode 0777 so
UID 10001 can write on Linux.
For real inputs, give that UID write access to a new owned output directory using
your normal host permissions. Keep input files readable and mounted read-only.
The application bounds every bundle; monitor aggregate output/disk use as well.

```sh
candidate python -I -B /harness/qualify.py provenance
for config in legacy/config.json simple/config.json simple/config-engineer.json simple/config.yaml; do
  case "$config" in
    legacy/*) name=legacy ;;
    *engineer*) name=engineer ;;
    *.yaml) name=yaml ;;
    *) name=simple ;;
  esac
  candidate ecofill validate --config "/inputs/$config"
  candidate ecofill plan --config "/inputs/$config" --output "/output/$name/plan"
  candidate ecofill run --config "/inputs/$config" --run-id first
  candidate ecofill inspect --run-uri "/output/$name/runs/first"
done
candidate python -I -S -B /harness/qualify.py replay-config \
  --run /output/simple/runs/first --output /output/replay.json --output-uri /output/replay
candidate ecofill run --config /output/replay.json --run-id replay
candidate ecofill inspect --run-uri /output/replay/runs/replay
candidate python -I -S -B /harness/qualify.py readback --output /output --legacy-script /harness/legacy.py
```

Each invocation starts a fresh container/process. The final stdlib reader starts
after producers exit. Expected legacy results: 19 fills and 2,380 unchanged
observations. Each simple spelling: 10 fills and 460 unchanged observations, two
boundary gaps and one `no_route` gap. Replay compares raw/completed values and masks
byte for byte. The installed package, wheel, console entry and dependency versions
are checked separately. No checkout package is injected into Python.

## PowerShell commands

Use a clean checkout and a new fixture directory. PowerShell arrays preserve
mount arguments containing spaces. Stop on each failed Docker exit:

```powershell
$ecofillSha = '94174ef300d63d0cfeee44191e672f31a2f78390'
git clone https://github.com/jequihua/ecofill.git ecofill-source
if ($LASTEXITCODE) { throw 'clone failed' }
Set-Location ecofill-source
git checkout $ecofillSha
if ($LASTEXITCODE) { throw 'checkout failed' }
$image = "ecofill:local-$ecofillSha"
docker build --load --platform linux/amd64 -f docker/Dockerfile -t $image .
if ($LASTEXITCODE) { throw 'build failed' }
python -I -S -B docker/qualify_container.py prepare --public . --output candidate-fixture
if ($LASTEXITCODE) { throw 'fixture failed' }
$inputs = (Resolve-Path candidate-fixture/inputs).Path
$outputs = (Resolve-Path candidate-fixture/output).Path
$harness = (Resolve-Path docker/qualify_container.py).Path
$legacy = (Resolve-Path examples/operation_demo.py).Path
$imageId = docker image inspect --format '{{.Id}}' $image
if ($LASTEXITCODE) { throw 'image inspection failed' }
function Invoke-Candidate([string]$Entry, [string[]]$Arguments) {
    $dockerArgs = @('run', '--rm', '--platform', 'linux/amd64', '--user', '10001:10001',
        '--network', 'none', '--read-only', '--cpus', '4', '--memory', '12g',
        '--memory-swap', '12g', '--pids-limit', '256',
        '--tmpfs', '/tmp:rw,size=256m,mode=1777',
        '--mount', "type=bind,src=$inputs,dst=/inputs,readonly",
        '--mount', "type=bind,src=$outputs,dst=/output",
        '--mount', "type=bind,src=$harness,dst=/harness/qualify.py,readonly",
        '--mount', "type=bind,src=$legacy,dst=/harness/legacy.py,readonly",
        '--entrypoint', $Entry, $imageId) + $Arguments
    docker @dockerArgs
    if ($LASTEXITCODE) { throw "candidate failed: $Entry $Arguments" }
}
Invoke-Candidate python @('-I', '-B', '/harness/qualify.py', 'provenance')
$configs = @{
    legacy = 'legacy/config.json'; simple = 'simple/config.json'
    engineer = 'simple/config-engineer.json'; yaml = 'simple/config.yaml'
}
foreach ($name in @('legacy', 'simple', 'engineer', 'yaml')) {
    $config = '/inputs/' + $configs[$name]
    Invoke-Candidate ecofill @('validate', '--config', $config)
    Invoke-Candidate ecofill @('plan', '--config', $config, '--output', "/output/$name/plan")
    Invoke-Candidate ecofill @('run', '--config', $config, '--run-id', 'first')
    Invoke-Candidate ecofill @('inspect', '--run-uri', "/output/$name/runs/first")
}
Invoke-Candidate python @('-I', '-S', '-B', '/harness/qualify.py', 'replay-config',
    '--run', '/output/simple/runs/first', '--output', '/output/replay.json', '--output-uri', '/output/replay')
Invoke-Candidate ecofill @('run', '--config', '/output/replay.json', '--run-id', 'replay')
Invoke-Candidate ecofill @('inspect', '--run-uri', '/output/replay/runs/replay')
Invoke-Candidate python @('-I', '-S', '-B', '/harness/qualify.py', 'readback',
    '--output', '/output', '--legacy-script', '/harness/legacy.py')
```

## Configure and diagnose

The synthetic weather CSV uses the engineer's timestamp, temperature, humidity,
dew point, wind, pressure, rain, solar, ET, source, station and key columns. Only
`bar` is a target; unused columns are never filled. `30min` and `max_gap: 2h` are
explicit. Three-step and four-step whole gaps can be filled; a six-step gap stays
unfilled. Leading/trailing gaps are not extrapolated. Missing timestamp rows are
inserted on the declared grid. No resampling, timezone, units, donors or QC policy
is inferred. Units and QC are optional declarations; omit them only when their
defaults match your intended data contract.

Aware timestamps carry their offset. Naive timestamps require an explicit timezone;
ambiguous/nonexistent local times and phase-mismatched donors fail with a diagnostic.
Group multiple series explicitly. RF and foundation X modes need measured, explicitly
selected donors; target aliases and target descendants cannot donate. An absent
ancestry parent remains metadata until selected. See [simple operation](simple-operation.md).

`validate` is offline and does not need the CSV or model files. `plan` reads inputs
and reports budgets/support before inference. A complete run can retain unsupported
gaps: inspect masks and reasons. CLI exits are 0 complete/valid, 1 failed/invalid,
2 partial and 3 incomplete/in-progress. Keep failed outputs; do not reuse an unknown
native attempt. A new conflicting request under the same run ID is rejected.
Check `/limits/...` or `/datasets/...` diagnostics before raising a finite budget.
If writing fails, check host output permissions and available disk; keep the
non-root user and read-only root. Never make the source checkout writable to fix it.

Native testing requires separately supplied, hash-verified TabPFN 3.5 and TimesFM 3
weights and licenses mounted read-only. Their non-commercial research terms apply.
The candidate receipt states which native paths actually ran. Its bounded native
producer uses one load per family, at most ten combined TabPFN fit/predict calls,
ten TimesFM forecasts, 30 minutes and 10 GiB measured process RSS inside a 12 GiB,
four-CPU container. Saved constituents supply strict 0.5/0.5 averages without new
inference. GPU compatibility and field accuracy are untested.

BigQuery reads and GCS bundles remain supported. Offline protocol doubles cannot
establish server or IAM acceptance. A new BigQuery output table is not implemented.
Google testing needs later authorized registry publication and an engineer-owned
Job image update. Keep existing deployment drafts on their published image until
that separate step; see [cloud testing](cloud-testing.md).

## Observed local candidate

Both shell workflows above passed against the installed candidate, including
independent stdlib readback and resolved-snapshot replay. The tested build source is
`94174ef300d63d0cfeee44191e672f31a2f78390`; the observed local tag is
`ecofill:m021-94174ef300d6` and Docker's local image ID is
`sha256:5d0ba05c5aabd0d8cf0e4db668fafff1b119065f006091f482f4ca00998694b5`.
The wheel SHA256 is `8172889d315c97feedd7aca34cf6ceeb45754a8e5ccd71becaee3ddaab36f965`.

The guide and receipt were finalized in a later documentation commit. Before
checking out the build source, retain this guide and receipt and identify their
commit with `git log -1 --format=%H -- docs/container-candidate.json`. The selected
source contains the earlier preparation receipt; the later guide commit changes
no image inputs. A fresh local rebuild can have a different wheel/image identity;
record its own provenance instead of assuming it is the tested local object.
`--load` makes the result available locally when using a BuildKit container builder.

The 26-case installed-image matrix passed with 19 RF fits; both legacy shell runs
bring the total to 23 of 30. Three Linux harness-control tests used labelled
doubles. The two real native producers each loaded each family once, performed
four TabPFN fits, four predictions and six TimesFM forecasts, and passed separate
numerical readback. The simple runner also passed seven saved resolved-config
replays without new inference. Strict U/X/J averages reuse retained constituents.
Every started qualification container used UID 10001, a read-only root and input
mounts, no network, four CPUs, 12 GiB RAM and a 256 MiB writable temporary mount.

All 60 allowed model-free launch attempts were accounted for; one incorrect
image-selector preflight started no container. No native attempt was retried.
The first cached build retained its result in BuildKit; the second allowed build
loaded the identical inputs locally. No registry push or live Google call occurred.
The machine-readable receipt records measured native RSS/time and the remaining
accounting limitations. This local image has no published candidate registry URI.

The retained native entry points are [native compatibility](../deploy/scripts/native_compatibility.py)
(`--public /public --models /models --output NEW_OUTPUT`, then `--readback NEW_OUTPUT`)
and [simple qualification](../docker/qualify_container.py) (`native-simple --models
/models --legacy-script /public/deploy/scripts/native_compatibility.py --output NEW_OUTPUT`,
then `readback-native --output NEW_OUTPUT`). They require the same resource policy
and read-only mounts at `/models/tabpfn35` and `/models/timesfm3`. Every real native
producer consumes its allowance even on failure; obtain a new explicit allowance
before running either again. Readback alone uses no model or network.
