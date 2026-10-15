"""Operator-supplied relocation and exactly-once, plan-only deployment lists."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from .config import canonical, digest, identifier, load_config


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def materialized(doc, root):
    result = deepcopy(doc)
    for source in result['sources'].values():
        source['uri'] = str(root / source['uri'])
    for model in result['models'].values():
        for artifact in model['artifacts']:
            artifact['uri'] = str(root / artifact['uri'])
    return result


def gap_key(gap):
    return (gap['partition'], gap['gap_id'], gap['reason'], digest(gap['recipes']))


def export_profile(config, destination, *, input_root, profile, artifact_paths=None, model_directory=None):
    """Copies named pinned files only. No cloud/model calls, discovery or experiments."""
    from .runner import prepare_operation
    identifier(profile)
    original = load_config(config); doc = original.to_dict()
    if doc['schema'] != 'ecofill.operation/1':
        raise ValueError('export relocates pinned operation/1 profiles; plan simple configurations directly')
    if 'cloud' in doc or any('://' in s['uri'] for s in doc['sources'].values()):
        raise ValueError('export accepts a local pinned profile')
    root = Path(destination).resolve(); root.mkdir(parents=True, exist_ok=False)
    input_root = Path(input_root).resolve()
    artifacts, changes = [], []
    def copy(uri, target, sha256):
        source = Path(uri)
        if not source.is_absolute(): source = input_root / source
        source = source.resolve()
        if not source.is_relative_to(input_root) or not source.is_file() or file_hash(source) != sha256:
            raise ValueError('named source missing/outside operator root/hash drift')
        path = root / target; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        if file_hash(path) != sha256: raise ValueError('copy hash drift')
        artifacts.append({'path': target, 'sha256': sha256, 'bytes': path.stat().st_size})
        changes.append({'original_uri_sha256': hashlib.sha256(uri.encode()).hexdigest(), 'new_uri': target, 'sha256': sha256})
    for name, source in doc['sources'].items():
        suffix = '.csv.gz' if source['uri'].endswith('.gz') else '.csv'
        target = 'inputs/' + name + suffix
        copy(source['uri'], target, source['sha256']); source['uri'] = target
    for name, model in doc['models'].items():
        for artifact in model['artifacts']:
            uri = (artifact_paths or {}).get(artifact['sha256'], artifact['uri'])
            target = (model_directory or 'models') + '/' + name + '/' + Path(uri).name
            if model_directory is None:
                copy(uri, target, artifact['sha256'])
            else:
                if Path(model_directory).is_absolute() or '://' in model_directory:
                    raise ValueError('mounted model directory must be relative to the new bundle')
                mounted = (root / target).resolve()
                if not mounted.is_relative_to(input_root) or not mounted.is_file() or file_hash(mounted) != artifact['sha256']:
                    raise ValueError('mounted model artifact missing/hash drift/outside operator root')
                artifacts.append({'path': target, 'sha256': artifact['sha256'], 'bytes': mounted.stat().st_size,
                                  'mounted': True})
                changes.append({'original_uri_sha256': hashlib.sha256(artifact['uri'].encode()).hexdigest(),
                                'new_uri': target, 'sha256': artifact['sha256']})
            artifact['uri'] = target
    doc['output']['uri'] = 'output/' + profile
    portable = load_config(doc)
    full, _, _ = prepare_operation(materialized(doc, root))
    full = full.to_dict()
    if full['unavailable_inputs_or_partitions']:
        raise ValueError('profile inputs unavailable during export')
    enabled = [(name, dataset) for name, dataset in doc['datasets'].items() if dataset['enabled']]
    if len(enabled) > 32:
        raise ValueError('unsupported: more than 32 dataset/joint-group jobs; architect must admit a new boundary')
    jobs, reconciliation, actual = [], [], Counter()
    def write(name, value):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(canonical(value) + '\n')
    write('profile.json', portable.to_dict())
    for number, (name, dataset) in enumerate(enabled):
        job_doc = deepcopy(doc); job_doc['datasets'] = {name: dataset}
        job_plan, _, _ = prepare_operation(materialized(job_doc, root)); measured = job_plan.to_dict()
        gaps, calls = measured['planned_gaps'], measured['method_call_upper_bound']
        supported = gaps <= 7500 and calls <= 32000 and job_doc['limits']['max_rows_per_series'] <= 250000
        new_limits = job_doc['limits'] | {'max_gaps': gaps, 'max_method_calls': calls, 'wall_seconds': 13800}
        job_doc['limits'] = new_limits
        job_doc['output']['uri'] = 'output/' + profile + '/' + name
        resolved = load_config(job_doc)
        run_id = profile + '-' + str(number+1) + '-' + original.identity[:12]
        config_name = f'jobs/{number+1:02d}.json'
        write(config_name, resolved.to_dict())
        check, _, _ = prepare_operation(materialized(job_doc, root)); check = check.to_dict()
        if check['budget_exceeded'] or check['unavailable_inputs_or_partitions']:
            raise ValueError('emitted job plan exceeds explicitly declared limits')
        for gap in check['gaps']:
            actual[gap_key(gap)] += 1
        jobs.append({'dataset': name, 'run_id': run_id, 'config': config_name, 'config_identity': resolved.identity,
                     'planned_gaps': gaps, 'method_call_upper_bound': calls, 'supported': supported,
                     'required_action': None if supported else 'architect must admit higher finite ceiling before execution',
                     'original_limits': original.to_dict()['limits'], 'deployment_limits': new_limits,
                     'plan_identity': job_plan.identity, 'plan_only': True})
        for gap in check['gaps']:
            reconciliation.append({'dataset': name, 'partition': gap['partition'], 'gap_id': gap['gap_id'],
                                   'disposition': gap['reason'] or 'configured', 'job': run_id})
        write(f'plans/{number+1:02d}.json', check)
    expected = Counter(gap_key(gap) for gap in full['gaps'])
    if expected != actual or any(count != 1 for count in actual.values()):
        raise ValueError('missing/duplicate/changed planned work in deployment list')
    summary = {'schema': 'ecofill.operation.deployment/1', 'profile': profile, 'original_config_identity': original.identity,
               'portable_config_identity': portable.identity, 'original_limits': original.to_dict()['limits'],
               'original_plan': {'gaps': full['planned_gaps'], 'calls': full['method_call_upper_bound'],
                                 'budget_exceeded': full['budget_exceeded']},
               'jobs': jobs, 'exactly_once': True, 'planned_gaps_reconciled': sum(actual.values()),
               'scientific_calls': 0, 'status': 'ready' if all(j['supported'] for j in jobs) else 'unsupported'}
    write('run-list.json', summary); write('reconciliation.json', reconciliation)
    write('relocation-budget-diff.json', {'paths': changes, 'jobs': jobs})
    write('upload-inventory.json', {'schema': 'ecofill.operation.upload/1', 'files': artifacts,
                                   'generations': 'unknown until explicit engineer upload'})
    write('model-manifest.json', doc['models'])
    (root/'README.md').write_text(
        '# Portable deployment bundle\n\nRun commands from this bundle root. '
        'Select research_selected explicitly or confirmed separately; do not concatenate their overlapping work. '
        'Original limits are preserved in profile.json. Execute only supported jobs in run-list.json. '
        'Plans contain no inference. Preserve complete input/support domains, exclusions and frozen panels. '
        'Use the public stage_bundle.py helper with a chosen prefix to record real object generations before cloud use. '
        'Models are research-only, individually hash pinned, and never included in an image. '
        'Whole-archive completion and runtime memory have not been measured here.\n', encoding='utf-8', newline='\n')
    return summary
