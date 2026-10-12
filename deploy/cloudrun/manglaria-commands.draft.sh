# Engineer-run commands: job creation and execution incur costs.
# Set every REPLACE_ value and pin the config/policy versions first.
gcloud run jobs create REPLACE_JOB_NAME --project=manglaria-staging --region=us-central1 \
  --image=us-central1-docker.pkg.dev/manglaria-staging/cloud-run-source-deploy/ecofill@sha256:950cea24ceeb79a84bb8ad2d1cfdae876033fdf229b2993492838fd4e03dab8c --service-account=REPLACE_RUNTIME_SERVICE_ACCOUNT \
  --cpu=4 --memory=16Gi --tasks=1 --parallelism=1 --max-retries=0 --task-timeout=4h \
  --set-secrets=/etc/ecofill/policy.json=REPLACE_POLICY_SECRET:REPLACE_PINNED_SECRET_VERSION \
  --set-env-vars=ECOFILL_CLOUD_POLICY=/etc/ecofill/policy.json,ECOFILL_IMAGE_DIGEST=us-central1-docker.pkg.dev/manglaria-staging/cloud-run-source-deploy/ecofill@sha256:950cea24ceeb79a84bb8ad2d1cfdae876033fdf229b2993492838fd4e03dab8c \
  --args=run,--config,gs://REPLACE_CONFIG_BUCKET/ecofill/configs/test.json,--config-generation,REPLACE_CONFIG_GENERATION,--config-sha256,REPLACE_CONFIG_SHA256,--run-id,REPLACE_UNIQUE_RUN_ID
# Create above does not execute. The separate bounded execution is:
gcloud run jobs execute REPLACE_JOB_NAME --project=manglaria-staging --region=us-central1 \
  --tasks=1 --task-timeout=4h --wait --args=run,--config,gs://REPLACE_CONFIG_BUCKET/ecofill/configs/test.json,--config-generation,REPLACE_CONFIG_GENERATION,--config-sha256,REPLACE_CONFIG_SHA256,--run-id,REPLACE_UNIQUE_RUN_ID
# REST alternative: POST manglaria-run-request.draft.json with an authorized bearer token to
# https://run.googleapis.com/v2/projects/manglaria-staging/locations/us-central1/jobs/REPLACE_JOB_NAME:run
# Poll the returned operation and inspect the execution/result prefix.
