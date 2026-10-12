#!/bin/sh
set -eu
# Engineer action. Upload receipt, job, project and region are explicit arguments.
receipt=$1
job=$2
project=$3
region=${4:-us-central1}
python - "$receipt" <<'PY' > "$receipt.args"
import json,sys
d=json.load(open(sys.argv[1]))
assert d['status']=='staged'
for j in d['jobs']:
    assert j['supported']
    values=['run','--config',j['uri'],'--config-generation',str(j['generation']),
            '--config-sha256',j['sha256'],'--run-id',j['run_id']]
    assert all(',' not in x and '\n' not in x for x in values)
    print(','.join(values))
PY
while IFS= read -r arguments; do
    gcloud run jobs execute "$job" --project="$project" --region="$region" --args="$arguments" --wait
done < "$receipt.args"
