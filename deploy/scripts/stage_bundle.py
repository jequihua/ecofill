"""Prepare an operator-owned bundle; cloud writes require the explicit --upload flag."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from ecofill.operation.config import canonical, load_config, read_document
from ecofill.operation.cloud_config import gcs_uri, validate_cloud
from ecofill.operation.deployment import file_hash


def stage(bundle, asset_root, destination, prefix, output_prefix, cloud, *, upload=False, storage=None):
    from ecofill.operation.cloud import Storage, ObjectExists, TransientIO
    gcs_uri(prefix); gcs_uri(output_prefix); validate_cloud(cloud)
    bundle, asset_root, destination = Path(bundle).resolve(), Path(asset_root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    run_list = read_document(bundle/'run-list.json')
    objects, jobs = {}, []
    if upload and storage is None:
        storage = Storage()

    def upload_once(uri, payload):
        try:
            return storage.create(uri, payload)
        except (ObjectExists, TransientIO):
            previous, generation = storage.read(uri, None, len(payload))
            if previous != payload:
                raise ValueError('staging object conflict')
            return generation

    def artifact(value, *, model=False):
        path = (bundle/value['uri']).resolve()
        if not path.is_relative_to(asset_root) or not path.is_file() or file_hash(path) != value['sha256']:
            raise ValueError('artifact outside operator asset root/missing/hash drift')
        limit = cloud['model_bytes'] if model else cloud['transfer_bytes']
        if path.stat().st_size > limit:
            raise ValueError('artifact exceeds staging bound')
        uri = prefix + '/objects/' + value['sha256'] + '/' + path.name
        if uri not in objects:
            generation = upload_once(uri, path.read_bytes()) if upload else None
            item = {'uri': uri, 'generation': generation, 'sha256': value['sha256'], 'bytes': path.stat().st_size,
                    'local_uri': value['uri']}
            objects[uri] = item
            (destination/(value['sha256'] + '.object.json')).write_text(canonical(item)+'\n', encoding='utf-8')
        return {'uri': uri, 'sha256': value['sha256'], 'generation': objects[uri]['generation']}

    for number, job in enumerate(run_list['jobs']):
        if not job['supported']:
            raise ValueError('unsupported job requires architect authority before staging')
        doc = load_config(bundle/job['config']).to_dict()
        for source in doc['sources'].values():
            value = artifact(source)
            source.update(value)
        for model in doc['models'].values():
            model['artifacts'] = [artifact(value, model=True) for value in model['artifacts']]
        doc['cloud'] = deepcopy(cloud)
        doc['output']['uri'] = output_prefix + '/' + run_list['profile'] + '/' + job['dataset']
        if upload:
            doc = load_config(doc).to_dict()
        payload = (canonical(doc)+'\n').encode()
        sha256 = hashlib.sha256(payload).hexdigest()
        config_name = f'{number+1:02d}.json'
        (destination/config_name).write_bytes(payload)
        uri = prefix + '/configs/' + sha256 + '.json'
        generation = upload_once(uri, payload) if upload else None
        jobs.append({'uri': uri, 'generation': generation, 'sha256': sha256, 'run_id': job['run_id'],
                     'run_uri': doc['output']['uri']+'/runs/'+job['run_id'], 'supported': True,
                     'config': config_name, 'original_config_identity': job['config_identity'],
                     'staged_config_identity': load_config(doc).identity if upload else None})
    receipt = {'schema': 'ecofill.operation.staging/1', 'status': 'staged' if upload else 'pending',
               'objects': list(objects.values()), 'jobs': jobs, 'cloud_writes_requested': upload,
               'generations': 'actual server values' if upload else 'unknown; pending configs are not runnable'}
    (destination/'staging-receipt.json').write_text(canonical(receipt)+'\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','asset-root','destination','prefix','output-prefix','cloud-limits'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    receipt = stage(args.bundle, args.asset_root, args.destination, args.prefix, args.output_prefix,
                    read_document(args.cloud_limits), upload=args.upload)
    print(json.dumps({'status': receipt['status'], 'jobs': len(receipt['jobs']), 'objects': len(receipt['objects'])}))
