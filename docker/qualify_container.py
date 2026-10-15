"""Synthetic installed-image checks. No downloads, cloud requests or Docker control.

Run each CLI stage in a fresh container; readback commands use only stdlib.
The native-simple command is a single-use, bounded producer, never a retry.
"""
import argparse
import ast
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import runpy
import sys
import sysconfig
import time


def write_json(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def simple_example():
    # Locate the installed stdlib-only script without importing the package.
    return Path(sysconfig.get_path('purelib')) / 'ecofill/operation/simple_example.py'


def prepare(public, destination):
    """Create only deterministic synthetic inputs; mount inputs read-only afterwards."""
    public, destination = Path(public), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    inputs = destination / 'inputs'
    inputs.mkdir()
    legacy = runpy.run_path(str(public / 'examples/operation_demo.py'))
    doc = legacy['fixture'](inputs / 'legacy')
    for name, source in doc['sources'].items():
        source['uri'] = '/inputs/legacy/' + name + '.csv'
    doc['output']['uri'] = '/output/legacy'
    (inputs / 'legacy/config.json').unlink()  # The fixture just created this owned file.
    write_json(inputs / 'legacy/config.json', doc)
    example = runpy.run_path(str(public / 'src/ecofill/operation/simple_example.py'))
    example['prepare'](inputs / 'simple')
    concise = deepcopy(example['CONCISE'])
    concise['datasets']['weather']['uri'] = '/inputs/simple/inputs/weather.csv'
    concise['output']['uri'] = '/output/simple'
    engineer = deepcopy(example['ENGINEER'])
    engineer['sources']['davis_weather']['uri'] = concise['datasets']['weather']['uri']
    for variable in engineer['sources']['davis_weather']['variables'].values():
        variable.pop('unit', None)
    engineer['output']['uri'] = '/output/engineer'
    for filename, value in [('config.json', concise), ('config-engineer.json', engineer)]:
        (inputs / 'simple' / filename).unlink()
        write_json(inputs / 'simple' / filename, value)
    (inputs / 'simple/config.yaml').write_text(
        'schema: ecofill.simple/1\ndatasets:\n  weather:\n'
        '    uri: /inputs/simple/inputs/weather.csv\n    cadence: 30min\n'
        '    targets: [bar]\n    method: LIN\n    max_gap: 2h\n'
        'output:\n  uri: /output/yaml\n', encoding='utf-8')
    for name in ('output', 'output/legacy', 'output/simple', 'output/engineer', 'output/yaml', 'output/replay'):
        path = destination / name
        path.mkdir(exist_ok=True)
        path.chmod(0o777)
    return {'prepared': True, 'legacy_fills': 19, 'legacy_observations': 2380,
            'simple_fills': 10, 'simple_observations': 460}


def provenance():
    import ecofill
    distribution = importlib.metadata.distribution('ecofill')
    installed = Path(distribution.locate_file('ecofill')).resolve()
    assert Path(ecofill.__file__).resolve().parent == installed
    assert str(installed).startswith('/usr/local/lib/python3.12/site-packages/')
    entry, = [e for e in distribution.entry_points if e.group == 'console_scripts' and e.name == 'ecofill']
    assert entry.value == 'ecofill.operation.cli:main'
    for name in ('schema.json', 'simple.schema.json', 'operation2.schema.json'):
        assert json.loads((installed / 'operation' / name).read_text())
    expected = json.loads(Path('/opt/ecofill/dependency-inventory.json').read_text())
    versions = {item['name']: importlib.metadata.version(item['name']) for item in expected['packages']
                if item['name'] != 'wheel'}
    assert all(versions[item['name']] == item['version'] for item in expected['packages'] if item['name'] != 'wheel')
    wheel = Path(os.environ['ECOFILL_WHEEL_PATH'])
    return {'package': str(installed), 'console_entry': entry.value, 'versions': versions,
            'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
            'uid': os.getuid(), 'python': sys.version.split()[0]}


def table(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def verify_bundle(root):
    root = Path(root)
    terminal = json.loads((root / 'terminal.json').read_text())
    payload = (root / 'manifest.json').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == terminal['manifest_sha256']
    for item in json.loads(payload)['files']:
        path = root / item['path']
        assert path.resolve().is_relative_to(root.resolve())
        payload = path.read_bytes()
        assert len(payload) == item['bytes']
        assert hashlib.sha256(payload).hexdigest() == item['sha256']
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['status'] == 'complete'
    return summary


def replay_config(run, destination, output):
    doc = json.loads((Path(run) / 'config.resolved.json').read_text())
    for name, source in doc['sources'].items():
        source['uri'] = str(Path(run) / 'inputs' / (name + '.snapshot'))
    doc['output']['uri'] = output
    write_json(destination, doc)
    return {'snapshot_config': str(destination)}


def readback(root, legacy_script):
    root = Path(root)
    legacy = runpy.run_path(str(legacy_script))['readback'](root / 'legacy/runs/first')
    assert legacy['observations_checked'] == 2380
    reader = runpy.run_path(str(simple_example()))['readback']
    simple = {name: reader(root / name / 'runs/first') for name in ('simple', 'engineer', 'yaml')}
    for value in simple.values():
        assert value['status'] == 'complete'
        assert value['filled_cells'] == 10
        assert value['observations_checked'] == 460
    replay = reader(root / 'replay/runs/replay')
    for filename in ('raw', 'completed', 'observed', 'reconstructed', 'dispositions'):
        original = root / 'simple/runs/first/datasets/weather-0' / (filename + '.csv.gz')
        repeated = root / 'replay/runs/replay/datasets/weather-0' / (filename + '.csv.gz')
        assert original.read_bytes() == repeated.read_bytes()
    return {'legacy': legacy, 'simple': simple, 'replay': replay, 'replay_bytes_equal': True}


def model_pins(harness):
    """Read the unchanged qualification's literal pins without running its producer."""
    tree = ast.parse(Path(harness).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pinned' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('native qualification pins absent')


def check_models(models, harness):
    pins = model_pins(harness)
    for name, expected in pins.items():
        with (Path(models) / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
    return pins


def request_key(request):
    values = {}
    for name in ('target', 'feature_names', 'reference_features', 'reference_labels', 'query_features',
                 'seed', 'device', 'channels', 'direction', 'context', 'historical_covariates',
                 'horizon_covariates', 'horizon'):
        if hasattr(request, name):
            value = getattr(request, name)
            values[name] = value.tolist() if hasattr(value, 'tolist') else value
    return json.dumps(values, sort_keys=True, allow_nan=False)


def native_simple(models, harness, output):
    """One producer: real simple runner, then saved-config replay with saved returns."""
    import resource
    import signal
    import numpy as np
    import torch
    from ecofill.operation import run, plan
    from ecofill.tabpfn import TabPFNBackend
    from ecofill.timesfm_adapters import TimesFM3Backend
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    calls = {'tabpfn_fit': 0, 'tabpfn_predict': 0, 'timesfm_forecast': 0}
    loads = {'tabpfn': 0, 'timesfm': 0}
    active, events, saved = None, [], {}
    current = {'case': None, 'replay': False}
    error = None

    def guard():
        if time.monotonic() - began > 1800 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 10 * 1024**3:
            raise RuntimeError('native time/RSS allowance exhausted')

    def before(kind, **detail):
        nonlocal active
        guard()
        assert active is None
        if (kind.startswith('tabpfn') and calls['tabpfn_fit'] + calls['tabpfn_predict'] >= 10
                or kind == 'timesfm_forecast' and calls[kind] >= 10):
            raise RuntimeError('native call allowance exhausted')
        active = len(events)
        write_json(output / f'{active:02d}-started.json', {'kind': kind, 'case': current['case'], **detail})
        calls[kind] += 1
        events.append({'number': active, 'kind': kind})

    def returned(**detail):
        nonlocal active
        assert active is not None
        write_json(output / f'{active:02d}-return.json', detail)
        events[active]['returned'] = True
        active = None
        guard()

    def stage(label):
        if label == 'before-fit': before('tabpfn_fit')
        elif label == 'after-fit': returned(status='returned')
        elif label == 'before-predict':
            if active is not None: returned(status='returned', evidence='before-predict hook')
            before('tabpfn_predict')

    def observe(request, values):
        returned(status='returned', values=np.asarray(values).tolist())

    def observe_forecast(request, values):
        returned(status='returned', values=np.asarray(values).T.tolist(), evidence='native output observer')

    def timeout(signum, frame):
        raise TimeoutError('native process deadline; no retry')

    class Recorded:
        def __init__(self, backend, family):
            self.backend, self.family = backend, family

        @property
        def identity(self):
            return self.backend.identity

        def invoke(self, request):
            key = request_key(request)
            source = current['case']
            if source in ('AVG-U', 'AVG-X'):
                source = ('P' if self.family == 'tabpfn' else 'T') + source[3:]
            if current['replay'] or current['case'] in ('AVG-U', 'AVG-X'):
                return np.array(saved[(source, self.family, key)], copy=True)
            write_json(output / f'request-{len(saved):02d}.json', {'case': source, 'family': self.family, 'request': json.loads(key)})
            if self.family == 'timesfm': before('timesfm_forecast', request=json.loads(key))
            value = self.backend.predict(request) if self.family == 'tabpfn' else self.backend.forecast(request)
            if active is not None: returned(status='returned', values=np.asarray(value).tolist())
            saved[(source, self.family, key)] = np.array(value, copy=True)
            guard()
            return value

        predict = invoke
        forecast = invoke

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1800)
    try:
        pins = check_models(models, harness)
        write_json(output / 'model-pins.json', pins)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        backends = {}
        for family in ('tabpfn', 'timesfm'):
            write_json(output / (family + '-load-started.json'), {'attempt': 1})
            backend = (TabPFNBackend(Path(models) / 'tabpfn35/tabpfn-v3.5-20260909.safetensors', device='cpu',
                                     stage=stage, output_observer=observe) if family == 'tabpfn' else
                       TimesFM3Backend(Path(models) / 'timesfm3', context_length=128, max_horizon=336,
                                       joint=True, purpose='non-commercial research', output_observer=observe_forecast))
            backend.load()
            loads[family] += 1
            write_json(output / (family + '-load-return.json'), {'identity': backend.identity})
            backends['p' if family == 'tabpfn' else 't'] = Recorded(backend, family)
            guard()
        path = output / 'synthetic.csv'
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        with path.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['timestamp', 'y', 'z', 'd1', 'd2'])
            for i in range(600):
                writer.writerow([(start + timedelta(minutes=30 * i)).isoformat(),
                                 '' if 200 <= i < 203 else 10 + .1 * i,
                                 '' if 200 <= i < 203 else 20 + .2 * i, i % 24, math.sin(i / 24)])
        for method in ('P-U', 'P-X', 'T-U', 'T-X', 'AVG-J', 'AVG-U', 'AVG-X'):
            target = {'method': method, 'max_gap': '2h',
                      'backend': {'P': 'p', 'T': 't'} if method.startswith('AVG') else method[0].lower()}
            if method.endswith(('X', 'J')): target['predictors'] = ['d1', 'd2']
            if method.endswith('J'): target['group'] = ['y', 'z']
            doc = {'schema': 'ecofill.simple/1', 'purpose': 'non-commercial research',
                   'models': {name: {'kind': kind, 'license': 'non-commercial research', 'revision': 'pinned', 'artifacts': []}
                              for name, kind in [('p', 'tabpfn35'), ('t', 'timesfm3')]},
                   'datasets': {'biomet': {'uri': str(path), 'cadence': '30min',
                                          'targets': {'y': target} | ({'z': {}} if method.endswith('J') else {}),
                                          'columns': {'y': {'unit': 'C'}, 'z': {'unit': 'C'},
                                                      'd1': {'unit': 'dimensionless'}, 'd2': {'unit': 'dimensionless'}}}},
                   'output': {'uri': str(output / 'results')}}
            write_json(output / (method + '-simple.json'), doc)
            planned = plan(doc)
            assert planned.config.to_dict()['schema'] == 'ecofill.operation/2'
            current.update(case=method, replay=False)
            result = run(doc, run_id=method, backends=backends)
            if result['status'] != 'complete':
                raise RuntimeError('simple runner did not complete: ' + str(result.get('failure_type')))
            assert result['filled_cells'] == (6 if method.endswith('J') else 3)
            bundle = output / 'results/runs' / method
            replay = output / (method + '-resolved.json')
            replay_config(bundle, replay, str(output / 'replays'))
            current['replay'] = True
            repeated = run(str(replay), run_id=method, backends=backends)
            assert repeated['status'] == 'complete'
            guard()
        assert calls == {'tabpfn_fit': 4, 'tabpfn_predict': 4, 'timesfm_forecast': 6}, calls
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        signal.alarm(0)
        write_json(output / 'receipt.json', {'status': 'passed' if error is None else 'failed', 'error_type': error,
                   'calls': calls, 'loads': loads, 'events': events, 'unknown_entry': active,
                   'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'image_id': os.environ.get('ECOFILL_IMAGE_ID'),
                   'seconds': time.monotonic() - began,
                   'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                   'synthetic_only': True, 'retries': 0})
    return {'status': 'passed', 'calls': calls}


def readback_native(output):
    """Independent saved-return arithmetic, raw/masks and resolved-replay checks."""
    output = Path(output)
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['status'] == 'passed' and receipt['unknown_entry'] is None
    assert receipt['calls'] == {'tabpfn_fit': 4, 'tabpfn_predict': 4, 'timesfm_forecast': 6}
    assert receipt['loads'] == {'tabpfn': 1, 'timesfm': 1}
    assert len(receipt['events']) == 14 and all(e['returned'] for e in receipt['events'])
    assert receipt['rss_peak_bytes'] <= 10 * 1024**3 and receipt['seconds'] <= 1800
    native_values = {}
    for event in receipt['events']:
        started = json.loads((output / f'{event["number"]:02d}-started.json').read_text())
        returned = json.loads((output / f'{event["number"]:02d}-return.json').read_text())
        assert started['kind'] == event['kind'] and returned['status'] == 'returned'
        if 'values' in returned:
            family = 'P' if event['kind'] == 'tabpfn_predict' else 'T'
            key = (started['case'], family)
            native_values.setdefault(key, []).append(returned['values'])
    observations = fills = 0
    for method in ('P-U', 'P-X', 'T-U', 'T-X', 'AVG-J', 'AVG-U', 'AVG-X'):
        root = output / 'results/runs' / method
        verify_bundle(root)
        verify_bundle(output / 'replays/runs' / method)
        prefix = root / 'datasets/biomet-0'
        raw, completed, observed, mask = [table(prefix / (n + '.csv.gz')) for n in ('raw', 'completed', 'observed', 'reconstructed')]
        assert len(raw) == len(completed) == len(observed) == len(mask) == 600
        targets = ['y', 'z'] if method.endswith('J') else ['y']
        gaps = json.loads((prefix / 'gaps.json').read_text())
        attempts = [a for g in gaps for a in g['attempts'] if a['status'] == 'complete']
        expected = {}
        for attempt in attempts:
            returns = sorted((root / 'attempts' / attempt['id']).glob('native-*.json'))
            values = [json.loads(p.read_text()) for p in returns]
            source = method
            if method in ('AVG-U', 'AVG-X'):
                source = attempt['method']
            assert [v['values'] for v in values] == native_values[(source, attempt['method'][0])]
            for target in targets:
                if attempt['method'].startswith('P'):
                    prediction, = [v['values'] for v in values if v['target'] == target]
                else:
                    forward, = [v for v in values if v['direction'] == 'forward']
                    backward, = [v for v in values if v['direction'] == 'backward']
                    column = forward['channels'].index(target)
                    prediction = [(1 - i / 2) * forward['values'][i][column] + i / 2 * backward['values'][2 - i][column]
                                  for i in range(3)]
                expected.setdefault(target, []).append(prediction)
        for target in targets:
            assert len(expected[target]) == (2 if method.startswith('AVG') else 1)
            for i in range(600):
                if 200 <= i < 203:
                    value = sum(p[i - 200] for p in expected[target]) / len(expected[target])
                    assert math.isfinite(value)
                    assert math.isclose(float(completed[i][target]), value, rel_tol=1e-12, abs_tol=1e-12)
                    assert raw[i][target] == '' and observed[i][target] == 'False' and mask[i][target] == 'True'
                    fills += 1
                else:
                    value = (10 + .1 * i) if target == 'y' else (20 + .2 * i)
                    assert float(raw[i][target]) == float(completed[i][target]) == value
                    assert observed[i][target] == 'True' and mask[i][target] == 'False'
                    observations += 1
        for filename in ('raw', 'completed', 'observed', 'reconstructed', 'dispositions'):
            assert (prefix / (filename + '.csv.gz')).read_bytes() == (output / 'replays/runs' / method / 'datasets/biomet-0' / (filename + '.csv.gz')).read_bytes()
    return {'status': 'passed', 'observations_checked': observations, 'fills_checked': fills,
            'resolved_replays': 7, 'new_average_inferences': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'provenance', 'readback', 'replay-config', 'native-simple', 'readback-native'])
    parser.add_argument('--public', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output-uri')
    parser.add_argument('--legacy-script', type=Path)
    parser.add_argument('--models', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'prepare': result = prepare(args.public, args.output)
    elif args.command == 'provenance': result = provenance()
    elif args.command == 'replay-config': result = replay_config(args.run, args.output, args.output_uri)
    elif args.command == 'readback': result = readback(args.output, args.legacy_script)
    elif args.command == 'native-simple': result = native_simple(args.models, args.legacy_script, args.output)
    else: result = readback_native(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
