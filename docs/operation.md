# Batch operation and cloud handoff

Local typed files and explicit mounted-volume publication are documented in
[local Parquet operation](local-parquet.md); strict legacy CSV remains supported.

For selected-source local Docker testing, both shell command paths and the
candidate's observed qualification, see [container testing](container-testing.md).

For a local installed-CLI synthetic first run without Docker, see
[installation](installation.md#installed-cli-first-run). That guide explicitly
obtains the public example file; examples are not assumed to ship in the wheel.

For the published image, project drafts and numbered testing steps, see
[cloud testing](cloud-testing.md) and the [release receipt](release.json).

For your own CSV with only a file, cadence, targets, method and maximum gap, see
[simple configuration](simple-operation.md). That candidate format resolves to
the explicit configuration described here; the published image still expects
explicit `ecofill.operation/1` files until a reviewed image release.

The installed `ecofill` command fills only declared original gaps using original
observations. It preserves raw numbers, observation/QC masks, whole-gap decisions,
frozen donor panels and evidence classes. Unselected cases stay `no_route`.
JSON works with the core installation; safe YAML needs `ecofill[operation]`.
Cloud clients are optional (`ecofill[cloud]`), use ADC, and never store keys in a
configuration. Model artifacts are supplied separately for non-commercial research.

## Build and first run

From this public repository root, with a Linux amd64 Docker daemon:

```sh
docker build --platform linux/amd64 -f docker/Dockerfile -t ecofill:local .
python examples/operation_demo.py prepare --output operator-inputs
```

The image pins CPython 3.12 by base digest and installs an ordinary wheel with a
hash-locked CPU toolchain. The context is allowlisted. Runtime UID/GID is 10001;
there are no weights, credentials, experiments, tests, Git or gcloud in the image.
`/opt/ecofill` contains the wheel, Python/OS inventories, lock, project licenses
and dependency license metadata. Installed distributions retain their own notices.
Setuptools is retained because the pinned Torch distribution declares it as a
runtime dependency; wheel is removed after installation.
The image sets `SKB_DATA_DIRECTORY=/tmp/skrub_data` and
`MPLCONFIGDIR=/tmp/matplotlib`. A read-only root needs the bounded writable `/tmp`
mount shown below; these caches do not require a writable home or source tree.
LIN/RF and stdlib readback passed in the prior image. Its native attempt stopped
during skrub import before model deserialization or scientific calls because its
default home directory was unavailable; that failed evidence is retained. The
corrected pinned Linux amd64 image passed the import-only cache check and the
installed four-family LIN/RF path with independent stdlib readback. Its one native
synthetic process loaded each family once, completed TabPFN U/X and TimesFM U/X/J,
and checked strict U/X/J averages from saved constituents. It used 4 fits,
4 predictions and 6 directional forecasts, about 60 seconds and 4.51GiB peak
process RSS. A separate stdlib process verified the saved numerical values,
original masks and each joint average target after producer exit. These results
establish synthetic compatibility for the recorded image/toolchain/artifacts;
they establish no field accuracy, archive completion or live cloud execution.

Choose a new writable `operator-output` directory, owned by UID 10001 on Linux.
Edit `operator-inputs/config.json`: set each source URI to `/inputs/FAMILY.csv`
and output URI to `/output`; retain the hashes. Mount both directories explicitly:

```sh
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 --cpus 4 --memory 12g \
  --mount type=bind,src=ABSOLUTE_OPERATOR_INPUTS,dst=/inputs,readonly \
  --mount type=bind,src=ABSOLUTE_OPERATOR_OUTPUT,dst=/output \
  ecofill:local validate --config /inputs/config.json
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 --cpus 4 --memory 12g \
  --mount type=bind,src=ABSOLUTE_OPERATOR_INPUTS,dst=/inputs,readonly \
  --mount type=bind,src=ABSOLUTE_OPERATOR_OUTPUT,dst=/output \
  ecofill:local plan --config /inputs/config.json --output /output/first-plan
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 --cpus 4 --memory 12g \
  --mount type=bind,src=ABSOLUTE_OPERATOR_INPUTS,dst=/inputs,readonly \
  --mount type=bind,src=ABSOLUTE_OPERATOR_OUTPUT,dst=/output \
  ecofill:local run --config /inputs/config.json --run-id first
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
  --mount type=bind,src=ABSOLUTE_OPERATOR_OUTPUT,dst=/output,readonly \
  ecofill:local inspect --run-uri /output/runs/first
python -I -S examples/operation_demo.py readback --output operator-output/runs/first
```

The shipped four-family example plans 12 gaps and at most 15 calls against limits
of 100/100. Its real LIN/RF run fills 19 cells and the independent stdlib reader
checks 2,380 original observations after the producer exits. It also preserves a
QC-rejected raw value and an unresolved donorless whole gap. The checked-in
`examples/operation/config.json` uses public-root paths and can be validated and
planned directly by an installed CLI.

To change work, add an explicit compatible dataset, remove its mapping, or set
`enabled: false`; an unused source is not read. Change method/settings in an
explicit `user_configured` recipe. Saved recipes must retain their exact ordered
`frozen_binding`, settings and status; changing these requires removing frozen
qualification claims and supplying new user evidence. Research selections require
`allow_provisional: true`. AVG requires both whole-gap constituents with weights
0.5/0.5. Fills never become later donors. Exclusions and support domains remain
separate from the requested output interval.

## Portable profiles and budgets

An engineer supplies a local config, exact inputs and a SHA256-to-model-path JSON
mapping. The installed helper creates a new destination, verifies/copies only
named files and plans each dataset with its joint targets together:

```sh
ecofill export --config supplied/research_selected.json \
  --input-root ABSOLUTE_ASSET_ROOT --artifact-manifest supplied/model-paths.json \
  --destination NEW_BUNDLE --profile research_selected
```

An existing mounted model directory may be declared with `--model-directory` as
a path relative to the new bundle (for example `../models`); its exact files must
already exist under the operator asset root and pass every hash. This supports
one shared model directory in a portable parent bundle without repeated weight
copies. Invoke generated local jobs from their bundle root.

Select `research_selected` explicitly; `confirmed` is a separate alternative.
Do not concatenate overlapping profiles. The supplied archive profiles preserve
their original 1,000-gap/3,000-call limits: their original plans are 6,435 gaps /
25,249 calls for research and 1,640 / 1,111 for confirmed. A whole-profile run
therefore exhausts a gap counter. Use the generated measured budgets and run
list, or narrow the output interval after planning while retaining complete
support inputs. The private plan-only first-run counterpart has 2 gaps and at
most 18 calls; the private bundle includes its config and plan.

`run-list.json`, `reconciliation.json` and `relocation-budget-diff.json` record
original and new identities/limits and exactly-once coverage including skip and
`no_route` dispositions. Dataset groups keep all original support, donors,
exclusions and coupled targets. There is no time chunking. Limits are finite:
7,500 gaps / 32,000 maximum calls per job, 250,000 rows per series and at most
32 jobs per profile. An exceeded ceiling is `unsupported` and requires an
architect decision; it is never silently dropped. Planning performs no inference.
The existing profiles produce 18 research jobs and 4 confirmed jobs. The wider
catalogue retains 1,447 unselected target/classes.

## Generation-aware staging

Use a new destination and operator-chosen config/input/model and output prefixes:

```sh
python deploy/scripts/stage_bundle.py --bundle NEW_BUNDLE --asset-root ABSOLUTE_ASSET_ROOT \
  --destination NEW_PENDING_STAGE --prefix gs://INPUT_BUCKET/ecofill \
  --output-prefix gs://OUTPUT_BUCKET/ecofill --cloud-limits deploy/cloudrun/cloud-limits.example.json
```

This default is offline. It produces pending configs and a file inventory with
unknown generations. Pending configs cannot be executed. Review its bounds and
paths; then the engineer runs the same command with a different new destination
and `--upload`. That explicit action uses ADC and conditional object creation,
checks existing bytes on ambiguous responses, records actual generations and
hashes, and uploads runnable configs only after their artifacts are pinned. No
generation is invented. Config files are identified by generation and SHA256 in
`staging-receipt.json`. Source CSV.gz transfer bytes and decoded bytes/rows are
separately bounded. Model/config/license bytes are verified before loading.

Keep deployment policy separate from the run config. Customize
`deploy/cloudrun/policy.example.json` with exact read/output prefixes, table IDs,
mounted roots and ceilings; deploy it as a version-pinned policy secret. Prefix
matching uses complete path components, so `ecofill-other` is not permitted by
`ecofill`. The runtime config cannot replace that policy. Operators with execution
override permission are trusted: the platform also lets them override environment
variables. Limit who receives that role.

## BigQuery input

A source may use `uri: bq://PROJECT.DATASET.TABLE` and
`bigquery: {columns: [...], intervals: [[START_UTC, END_UTC], ...]}` instead of a
local/GCS URI and SHA256. Columns must be exactly the required identity/time/key,
typed measurement, QC, original-mask, reason/route and optional explicit
`provenance_columns`. Include complete metadata validity/support intervals, even
when output dates are narrower. All sources use the single `cloud.snapshot_utc`.
Use a cutoff inside the table's available time-travel window; example dates must
be updated by the operator. Views/external tables and unsupported nested/repeated
types fail before submission.

The runner generates projected SELECTs, parameterized identities/times and stable
identity/time/key ordering; configuration cannot supply SQL. It dry-runs every
distinct read before any paid query, applies per-query and aggregate caps including
minimum/rounded billing, and uses deterministic job IDs. A submission with an
unknown response is retrieved by the same ID and its query/config checked; it is
never replaced with a fresh submission. Page/row/byte limits and reported totals
must reconcile. SQL, parameters, metadata, snapshot, job ID, billing, typed pages
and exact extracted CSV bytes are saved. Shared sources are read once. Later
replay consumes saved snapshots with a deliberate new run ID, rather than relying
on historical time travel.

## Publish, create and execute: engineer actions

Choose the billing project, repository, region, runtime service account, policy
secret and all buckets/prefixes. Example region is `us-central1`. No command below
is run by this repository's checker. Publish the locally built image:

```sh
gcloud auth configure-docker us-central1-docker.pkg.dev
docker tag ecofill:local us-central1-docker.pkg.dev/PROJECT/REPOSITORY/ecofill:REVIEWED_TAG
docker push us-central1-docker.pkg.dev/PROJECT/REPOSITORY/ecofill:REVIEWED_TAG
gcloud artifacts docker images describe us-central1-docker.pkg.dev/PROJECT/REPOSITORY/ecofill:REVIEWED_TAG --format=json
```

Resolve the registry digest and use `IMAGE_URI@sha256:DIGEST`, never the local
Docker image ID as a registry digest. An optional engineer-run remote build is
`gcloud builds submit . --project=PROJECT --region=us-central1
--config=deploy/cloudrun/cloudbuild.yaml --substitutions=_IMAGE_URI=IMAGE_URI:TAG`;
it incurs remote build/publication costs and requires separate engineer authority.

Replace every placeholder in `deploy/cloudrun/job.yaml`, including the pinned
policy secret version. `gcloud run jobs replace deploy/cloudrun/job.yaml
--project=PROJECT --region=us-central1` creates/updates the chosen job and does
not execute it. An equivalent create command is:

```sh
gcloud run jobs create JOB --project=PROJECT --region=us-central1 \
  --image=IMAGE_URI@sha256:DIGEST --service-account=RUNTIME_SERVICE_ACCOUNT \
  --cpu=4 --memory=16Gi --tasks=1 --parallelism=1 --max-retries=0 --task-timeout=4h \
  --set-secrets=/etc/ecofill/policy.json=POLICY_SECRET:PINNED_VERSION \
  --set-env-vars=ECOFILL_CLOUD_POLICY=/etc/ecofill/policy.json,ECOFILL_IMAGE_DIGEST=IMAGE_URI@sha256:DIGEST \
  --args=run,--config,gs://CONFIG_BUCKET/PREFIX/config.json,--config-generation,GENERATION,--config-sha256,SHA256,--run-id,UNIQUE_RUN_ID
gcloud run jobs execute JOB --project=PROJECT --region=us-central1 --wait \
  --args=run,--config,gs://CONFIG_BUCKET/PREFIX/config.json,--config-generation,GENERATION,--config-sha256,SHA256,--run-id,UNIQUE_RUN_ID
gcloud run jobs executions describe EXECUTION --project=PROJECT --region=us-central1 --format=json
```

For REST, replace placeholders in `deploy/cloudrun/run-request.json` and POST it
to `https://run.googleapis.com/v2/projects/PROJECT/locations/us-central1/jobs/JOB:run`
with an ADC-derived bearer token. The response is a long-running Operation; poll
its named operations resource until done, then inspect its execution and result
prefix. The request overrides only config generation/hash and run ID arguments,
one task and timeout. [Official jobs.run contract](https://docs.cloud.google.com/run/docs/reference/rest/v2/projects.locations.jobs/run).

Run a staged list serially with `sh deploy/scripts/run-list.sh RECEIPT JOB PROJECT
us-central1`, or `deploy/scripts/run-list.ps1 -Receipt RECEIPT -Job JOB -Project
PROJECT`. Both wait and stop on failure. Review every run ID first: reuse is a
checked existing result, not a new inference run. For a deliberate new run, choose
new IDs/output prefixes. A conflicting request or incomplete claim cannot replay
inference; unknown attempts require operator investigation and a new deliberate
request, never automatic recovery.

Default job resources are 4 CPU/16GiB, one task, one worker, pools=1, retries=0,
timeout 4h. Generated application deadlines are 13,800s, leaving 600s for
publication. A separately configured archive option is timeout 24h with an
application deadline at most 85,800s; review and save the changed config/identity.
No limit changes are implicit. Call counts do not establish archive wall time.
Budget memory separately: resident model(s), up to 2.5GiB staged model files,
prepared/raw data, up to 1GiB scratch and up to 1GiB output. Cloud Run writable
files consume memory; loading both families simultaneously adds their resident
sizes. Plan estimates differ from the measured synthetic compatibility RSS.
Whole-archive field completion is not qualified here.

## IAM, results and replay

| Identity | Minimum scoped permissions |
|---|---|
| Image publisher | Artifact Registry writer on the chosen repository; optional Cloud Build submitter/builder permissions only for that path. |
| Job deployer | Cloud Run job create/update and service-account use on the chosen runtime identity; policy-secret administration is a separate owner action. |
| Invoker | `run.jobs.run` and `run.jobs.runWithOverrides` on this job; `roles/run.jobsExecutorWithOverrides` supplies these. Add scoped execution/operation get permissions for polling. |
| Runtime | BigQuery `jobs.create` on billing project and read on exact source tables; Storage object get on exact inputs/configs/models; conditional object create plus get on output prefix; Secret Manager accessor on the exact policy secret. |

Runtime needs no upstream BigQuery write, object delete, IAM administration or
project-wide Editor. [Cloud Run roles](https://docs.cloud.google.com/iam/docs/roles-permissions/run)
and [execution guide](https://docs.cloud.google.com/run/docs/execute/jobs) describe
the override permission; scope it to trusted invokers. [Storage preconditions](https://docs.cloud.google.com/storage/docs/request-preconditions)
protect write-once publication. No always-on server or scheduler is deployed.

Each result prefix has request/config/input/plan provenance, exact source
snapshots, attempt starts and returns, dataset raw/completed/masks/dispositions,
one-page `summary.md`, machine `summary.json`, manifest and terminal. Manifest
and terminal publish last. Inspection verifies every hash/generation, including
the claim. No terminal means incomplete. Exit codes: complete 0, partial 2,
failed/invalid 1, in-progress/incomplete 3. Failed partitions are not counted as
completed, and secondary publication errors retain the primary failure. A complete
task may still contain unresolved gaps; scientific coverage and platform task
success are distinct. Preserve successful partitions after failure. Keep extracted
snapshots for later replay; never silently query mutable upstream data again.

Source projection is owner-managed: the Docker qualification uses a disposable
public layout. The owner syncs the reviewed projection to the external repository
after review; this guide is usable with that public checkout alone.
