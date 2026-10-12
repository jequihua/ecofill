# Test the published ecofill runner

**BigQuery input; GCS output.** This is a batch job, with no HTTP service and no
built-in BigQuery output. A downstream BigQuery load is engineer-owned future work.
Publication does not create a job, bucket, policy secret or runtime account.

The published tag is `us-central1-docker.pkg.dev/manglaria-staging/cloud-run-source-deploy/ecofill:m017-0d0a16a`. The verified immutable URI is `us-central1-docker.pkg.dev/manglaria-staging/cloud-run-source-deploy/ecofill@sha256:950cea24ceeb79a84bb8ad2d1cfdae876033fdf229b2993492838fd4e03dab8c`;
[release.json](release.json) records its registry digest and smoke. Use that URI
in the project drafts, rather than rebuilding or selecting a mutable tag.
[Artifact Registry console](https://console.cloud.google.com/artifacts/docker/manglaria-staging/us-central1/cloud-run-source-deploy?project=manglaria-staging)
shows project `manglaria-staging`, region `us-central1`, repository
`cloud-run-source-deploy`, image `ecofill`, tag `m017-0d0a16a`.

1. Clone `https://github.com/jequihua/ecofill.git`, or run `git pull --ff-only origin main`
   from its clean main checkout. Read this guide and [release.json](release.json).
   Pull its exact immutable URI: `docker pull us-central1-docker.pkg.dev/manglaria-staging/cloud-run-source-deploy/ecofill@sha256:950cea24ceeb79a84bb8ad2d1cfdae876033fdf229b2993492838fd4e03dab8c`.
2. Run the local four-family synthetic test before cloud work. Set `IMAGE` to that
   immutable URI; replace the two absolute mount paths with new owned directories.
   The reader needs only Python 3.12 stdlib. On Linux the output must be writable
   by image UID/GID 10001. Start from the public root:

   ```sh
   python examples/operation_demo.py prepare --output operator-inputs
   # Edit config.json: sources -> /inputs/FAMILY.csv; output.uri -> /output.
   # Keep all input hashes.
   IMAGE=us-central1-docker.pkg.dev/manglaria-staging/cloud-run-source-deploy/ecofill@sha256:950cea24ceeb79a84bb8ad2d1cfdae876033fdf229b2993492838fd4e03dab8c
   docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 --cpus 4 --memory 12g \
     --mount type=bind,src=ABSOLUTE_INPUTS,dst=/inputs,readonly \
     --mount type=bind,src=ABSOLUTE_OUTPUT,dst=/output "$IMAGE" validate --config /inputs/config.json
   docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 --cpus 4 --memory 12g \
     --mount type=bind,src=ABSOLUTE_INPUTS,dst=/inputs,readonly \
     --mount type=bind,src=ABSOLUTE_OUTPUT,dst=/output "$IMAGE" plan --config /inputs/config.json --output /output/plan
   docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 --cpus 4 --memory 12g \
     --mount type=bind,src=ABSOLUTE_INPUTS,dst=/inputs,readonly \
     --mount type=bind,src=ABSOLUTE_OUTPUT,dst=/output "$IMAGE" run --config /inputs/config.json --run-id local-first
   docker run --rm --network none --read-only --cpus 1 --memory 256m \
     --mount type=bind,src=ABSOLUTE_OUTPUT,dst=/output,readonly "$IMAGE" inspect --run-uri /output/runs/local-first
   python -I -S -B examples/operation_demo.py readback --output operator-output/runs/local-first
   ```

   Plan: 12 gaps, at most 15 calls. Run/readback: 19 fills, 2,380 preserved
   observations, four datasets, methods LIN and RF-X. QC-rejected raw numbers and
   an unresolved donorless whole gap are preserved. The producer exits before
   the separate reader starts. See [operation](operation.md) for API/config rules.
3. Choose your runtime service account, config/output buckets and exact prefixes.
   Ask your administrator for scoped permissions; do not grant broad Editor.
   Runtime needs BigQuery jobs.create on the billing project and table read,
   Storage object get for config/input/models and object create plus get on the
   output prefix, and accessor on the pinned policy secret. Deployment also needs
   Cloud Run job creation/service-account use; execution overrides need
   run.jobs.run and run.jobs.runWithOverrides. Image pull needs Artifact Registry
   Reader on this repository. Bucket/policy/account creation is an engineer action.
4. Prepare the small BigQuery example. Copy
   `deploy/cloudrun/manglaria-bigquery-test.draft.json` and
   `manglaria-test-policy.draft.json`; replace every `REPLACE_` value. The fixture
   is one synthetic biomet instrument at 30-minute cadence, LIN, at most 1,000
   extracted rows, 10 gaps/calls, 32MiB per-query/total billed-byte caps and a
   60-second query timeout. A date predicate alone is not a scan-cost ceiling.
   Cost estimates above the finite caps stop before paid submission.

   | Synthetic column | BigQuery type | Meaning |
   |---|---|---|
   | primary_key | STRING | Unique stable row key |
   | site_id | STRING | SYNTHETIC identity, matching declared instrument |
   | timestamp | TIMESTAMP | UTC regular 30-minute timestamps |
   | y | FLOAT64, nullable | Target in C; NULL denotes missing |
   | d1, d2 | FLOAT64, nullable | Declared dimensionless fields |
   | qc_y | INT64 | Explicit accepted QC policy (0/1 observed, 2 rejected) |
   | data_source | STRING | LOCAL/REMOTE provenance, not unit inference |

   A separately prepared native TABLE must match these names/types and policies.
   Views/external/nested sources are unsupported. Replace support start/end in
   extraction, validity and every field domain together; output dates may be
   narrower only while retaining the whole-gap boundaries/support. Replace the
   snapshot with a current UTC cutoff at which that table existed and remains
   inside its available time-travel window. Replace the policy table ID and both
   bucket prefixes identically. Neither table existence nor current schema nor
   a server dry-run was checked by publication. The offline fixture checks use
   deterministic cloud doubles and no Google calls.
5. Structurally validate the instantiated local JSON offline:
   `ecofill validate --config YOUR_TEST_CONFIG.json` (or the installed image with
   that file mounted read-only). Planning against real BigQuery performs reads,
   dry-runs and bounded paid queries: engineer-run with ADC and
   `ECOFILL_CLOUD_POLICY=YOUR_INSTANTIATED_POLICY.json`. For example:
   `ecofill plan --config YOUR_TEST_CONFIG.json --output NEW_PLAN_DIRECTORY`.
   Review every planned gap, support domain, cost cap and unresolved reason before
   running. Keep max_rows, pages, byte limits and method counts finite.
6. Upload and pin only your intended config and policy. The native-BigQuery LIN
   example has no model artifacts. Upload its JSON to your exact config prefix;
   read its real object generation and SHA256, and create a versioned policy
   secret only under your own authority. Do not invent generations. For supplied
   qualified-CSV/archive bundles use `deploy/scripts/stage_bundle.py` as documented
   in [generation-aware staging](operation.md#generation-aware-staging): default
   offline pending inventory, then a different new destination and explicit
   `--upload` after review. Checkpoint/config/license files stay hash-pinned and
   separately supplied for non-commercial research; weights are not in the image.
7. Replace all operator inputs in `deploy/cloudrun/manglaria-job.draft.yaml` and
   `manglaria-commands.draft.sh`: job name, runtime account, policy secret/version,
   config URI/generation/SHA256 and unique run ID. The project, region and immutable
   image are release facts. **Engineer-run, creates resources:**
   `gcloud run jobs replace deploy/cloudrun/manglaria-job.draft.yaml --project=manglaria-staging --region=us-central1`.
   This creates/updates without executing. Resources remain one task/worker,
   4 CPU/16GiB, retries 0 and 4h timeout; application deadline is 13,800 seconds,
   leaving publication time. Writable container files count towards memory.
8. **Engineer-run, incurs compute/query/storage costs:** execute exactly one small
   test using the matching CLI draft, or POST the JSON draft to
   `https://run.googleapis.com/v2/projects/manglaria-staging/locations/us-central1/jobs/YOUR_JOB:run`.
   CLI and REST use the same config/generation/hash/run ID, one task and 14,400s
   task timeout. The REST response is an Operation; poll it, then inspect the named
   execution. These drafts do not claim existing buckets, secrets or jobs.
9. Check platform execution with `gcloud run jobs executions describe YOUR_EXECUTION
   --project=manglaria-staging --region=us-central1 --format=json`. Independently
   check application artifacts with the installed `ecofill inspect --run-uri
   gs://YOUR_OUTPUT_BUCKET/ecofill/synthetic-test/runs/YOUR_RUN_ID` and its protected
   policy. GCS contains compressed raw/completed/masks, original input snapshots,
   resolved config, plans, methods/settings, attempts, manifests, terminal and
   short JSON/Markdown summaries. Exit codes: complete 0, partial 2, failed/invalid
   1, incomplete/in-progress 3. A complete job can leave unsupported gaps unfilled;
   inspect reasons and masks, not just the Cloud Run success status.
10. Only then increase scope using measured job lists. Same complete request/run ID
    returns verified existing output without fresh query/model calls; a conflicting
    request fails. An incomplete/unknown attempt is not automatically resumed.
    Preserve it and investigate; use a deliberate new identity for new work.

## Map your sources and configure routes

These owner-supplied IDs are in `manglaria-staging.manglaria_lakehouse_ds`:

| Family | Native source ID | Required mapping before a run |
|---|---|---|
| Biomet | carbon_flux_biomet_s2_gf_daqm | Explicit identity/time/key columns, QC/original masks, C units, cadence and validity |
| Flux | carbon_flux_eddycovariance_s2_filt_1 | Explicit flux fields/QC/original masks, units, 30-minute cadence and validity |
| CTD | diver_ctd_combined_s2_filt | Station aliases, exclusions, field/unit regimes, 60-minute cadence and validity |
| Weather | davis_weather_combined_s2_filt | Station fields, measured units/QC, 60-minute cadence and validity |

The engineer must inspect each current native TABLE/schema and cutoff. No private
rows or guessed column mappings are published here. Existing archive presets use
qualified CSV snapshots: changing their URI to bq:// alone does not make a valid
raw-table config. The synthetic example is a schema/setup illustration.

Add a dataset by adding its explicit source metadata, compatible targets/routes
and ordered donor refs. Remove its dataset mapping or set `enabled: false` to
remove work; remove unused sources, keeping any cross-source donor dependency.
Change model/settings only in a `user_configured` recipe with new evidence.
Each donor names exact source/instrument/field/unit regime, preserves order and
matches cadence/domain; filled values never become support. Saved frozen bindings
are immutable: remove old qualification claims before deliberately changing them.
AVG requires both whole-gap constituents with fixed 0.5/0.5 weights.

Select confirmed or research_selected explicitly; do not combine overlapping
profiles. Existing coverage is 4 confirmed routes, 23 research selections and
1,447 no_route target/classes. Research requires explicit allow_provisional and
retains that status. Private handoff: qualified first-run plan is 2 gaps/at most
18 calls; choose 4 confirmed jobs or 18 research jobs with their measured budgets.
Publication adds no route or accuracy evidence. Full-archive completion and
live-cloud behavior remain untested by this release.

Official contracts: [Artifact Registry](https://docs.cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling),
[job execution](https://docs.cloud.google.com/run/docs/execute/jobs),
[REST jobs.run](https://docs.cloud.google.com/run/docs/reference/rest/v2/projects.locations.jobs/run).
