"""Bounded synthetic installed-adapter qualification, never a field run or retry.

The readback path uses only stdlib and runs after the native producer exits.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import time


def json_values(value):
    """Keep nonfinite native returns as explicit nulls, never invalid JSON."""
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, list):
        return [json_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(public, models, output):
    public, models, output = Path(public), Path(models), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    calls = {'tabpfn_fit': 0, 'tabpfn_predict': 0, 'timesfm_forecast': 0}
    outcomes, loads = {}, {'tabpfn': 0, 'timesfm': 0}
    load_attempts = {'tabpfn': 0, 'timesfm': 0}
    files, events, active, context = {}, [], None, {}
    publication_errors, error_chain = [], []
    error = None

    def save(name, value):
        payload = json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')
        with (output / name).open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        files[name] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}

    def guard():
        if time.monotonic() - began > 1800 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 10 * 1024**3:
            raise RuntimeError('native time/RSS allowance exhausted')

    def reserve(kind):
        nonlocal active
        guard()
        if active is not None:
            raise RuntimeError('previous native entry has no durable return; no retry')
        if (kind.startswith('tabpfn') and calls['tabpfn_fit'] + calls['tabpfn_predict'] >= 10
                or kind == 'timesfm_forecast' and calls[kind] >= 10):
            raise RuntimeError('native call allowance exhausted')
        number = len(events)
        save(f'{number:02d}-started.json', {'kind': kind, 'count': calls[kind] + 1, **context})
        calls[kind] += 1
        active = {'number': number, 'kind': kind}
        events.append(dict(active))

    def returned(status='returned', **details):
        nonlocal active
        if active is None:
            raise RuntimeError('native return has no reservation')
        save(f'{active["number"]:02d}-return.json', {'kind': active['kind'], 'status': status, **details})
        events[active['number']]['status'] = status
        active = None

    def timeout(signum, frame):
        raise TimeoutError('native process deadline; no retry')

    def failure_return(exc):
        if active is not None:
            try: returned('exception', error_type=type(exc).__name__)
            except BaseException as secondary:
                publication_errors.append(type(secondary).__name__)

    def load(family, construct):
        load_attempts[family] += 1
        save(f'{family}-load-started.json', {'attempt': load_attempts[family]})
        try:
            backend = construct()
            backend.load()
        except BaseException as exc:
            try: save(f'{family}-load-return.json', {'status': 'exception', 'error_type': type(exc).__name__})
            except BaseException as secondary: publication_errors.append(type(secondary).__name__)
            raise
        loads[family] += 1
        save(f'{family}-load-return.json', {'status': 'returned', 'identity': backend.identity,
                                          'load_record': getattr(backend, 'load_record', None)})
        guard()
        return backend

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1800)
    try:
        versions = {name: importlib.metadata.version(name) for name in
                    ('tabpfn', 'timesfm', 'torch', 'numpy', 'pandas', 'scikit-learn')}
        assert versions == {'tabpfn': '9.0.0', 'timesfm': '3.0.2', 'torch': '2.14.0+cpu',
                            'numpy': '2.3.5', 'pandas': '2.2.3', 'scikit-learn': '1.8.0'}
        pinned = {'tabpfn35/tabpfn-v3.5-20260909.safetensors': 'ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3',
                  'tabpfn35/LICENSE': '9570fdd687bb76ed45499412ad57d6ba949848e6d731f1242572bb57f2f73ae1',
                  'timesfm3/model.safetensors': 'a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da',
                  'timesfm3/config.json': 'ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6',
                  'timesfm3/LICENSE': '3e36db7240d23adb6ac6d7d931892dcc5706a14a7d31581f08c35d2f25736dfd'}
        for name, expected in pinned.items():
            with (models / name).open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
        wheel = os.environ.get('ECOFILL_WHEEL_PATH')
        save('preflight.json', {'versions': versions, 'model_hashes': pinned, 'one_worker': True,
                               'numerical_pools': 1, 'image_id': os.environ.get('ECOFILL_IMAGE_ID'),
                               'wheel_sha256': hashlib.sha256(Path(wheel).read_bytes()).hexdigest() if wheel else None,
                               'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from ecofill.tabpfn import TabPFNBackend
        from ecofill.timesfm_adapters import TimesFM3Backend
        from ecofill.operation import load_config
        from ecofill.operation.runner import prepare_operation
        from ecofill.operation.recipes import native_plan, execute_native, strict_blend
        fixture_spec = importlib.util.spec_from_file_location('fixture', public / 'examples/operation_demo.py')
        fixture = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(fixture)
        doc = fixture.fixture(output / 'synthetic')
        doc['datasets'] = {'biomet': doc['datasets']['biomet']}
        doc['models'] = {'p': {'kind': 'tabpfn35', 'artifacts': [], 'license': 'non-commercial research', 'revision': 'TabPFN-3.5'},
                         't': {'kind': 'timesfm3', 'artifacts': [], 'license': 'non-commercial research', 'revision': 'TimesFM-3'}}
        path = Path(doc['sources']['biomet']['uri'])
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames + ['d3'], list(reader)
        for row in rows:
            i = int(row['primary_key'].split('-')[-1])
            row['d3'] = str(np.sin(i / 24))
            if i in {200, 201, 202}: row['d1'] = ''
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        source = doc['sources']['biomet']
        source['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        fields = source['instruments']['SYNTHETIC']['fields']
        fields['d3'] = deepcopy(fields['d2'])
        fields['d3'].update(column='d3', lineage='biomet|SYNTHETIC|d3')
        donor_refs = [{'source': 'biomet', 'instrument': 'SYNTHETIC', 'field': name, 'unit_regime': 'native'}
                      for name in ('d2', 'd3')]

        def hook(label):
            if label == 'before-fit': reserve('tabpfn_fit')
            elif label == 'after-fit': returned(evidence='after-fit hook')
            elif label == 'before-predict':
                # A before-predict hook also establishes that fit returned. Older
                # labelled doubles expose only before hooks; no prediction is inferred.
                if active is not None and active['kind'] == 'tabpfn_fit':
                    returned(evidence='before-predict hook')
                reserve('tabpfn_predict')
            guard()

        def observe(request, value):
            returned(values=json_values(value), shape=list(np.shape(value)), evidence='native output observer')

        def observe_forecast(request, point):
            # TimesFM's native axes are channel/horizon; retain both its raw
            # shape and the request-oriented values before adapter validation.
            value = np.asarray(point).T
            returned(values=json_values(value), shape=list(value.shape), native_shape=list(np.shape(point)),
                     evidence='native output observer')

        p = load('tabpfn', lambda: TabPFNBackend(models / 'tabpfn35/tabpfn-v3.5-20260909.safetensors',
                                                device='cpu', stage=hook, output_observer=observe))
        t = load('timesfm', lambda: TimesFM3Backend(models / 'timesfm3', context_length=128,
                                                   max_horizon=336, joint=True, purpose=doc['purpose'],
                                                   output_observer=observe_forecast))

        class RecordedPredict:
            identity = p.identity
            def predict(self, request):
                context['request'] = {'target': request.target, 'feature_names': list(request.feature_names),
                    'reference_features': json_values(request.reference_features),
                    'reference_labels': json_values(request.reference_labels),
                    'query_features': json_values(request.query_features), 'seed': request.seed, 'device': request.device}
                try:
                    value = p.predict(request)
                    if active is not None:
                        returned(values=json_values(value), shape=list(np.shape(value)), evidence='backend return')
                    guard()
                    return value
                except BaseException as exc:
                    failure_return(exc)
                    raise

        class RecordedForecast:
            identity = t.identity
            def forecast(self, request):
                context['request'] = {'channels': list(request.channels), 'direction': request.direction,
                    'context': json_values(request.context), 'historical_covariates': json_values(request.historical_covariates),
                    'horizon_covariates': json_values(request.horizon_covariates), 'horizon': request.horizon}
                reserve('timesfm_forecast')
                try:
                    value = t.forecast(request)
                    if active is not None:
                        returned(values=json_values(value), shape=list(np.shape(value)), evidence='backend return')
                    guard()
                    return value
                except BaseException as exc:
                    failure_return(exc)
                    raise

        def snapshot(data, value, targets):
            raw, completed, observed, imputed = data.raw, value.completed, data.mask('observed'), value.mask('imputed')
            for target in raw:
                mask = observed[target]
                assert (raw.loc[mask, target] == completed.loc[mask, target]).all()
                assert not imputed.loc[mask, target].any()
            for target in targets:
                assert imputed[target].iloc[200:203].all()
                assert np.isfinite(completed[target].iloc[200:203]).all()
            return {'targets': list(targets), 'index': [stamp.isoformat() for stamp in raw.index],
                    'units': {v.name: v.unit for v in data.schema.variables},
                    'raw': {name: json_values(raw[name].to_numpy()) for name in raw},
                    'completed': {name: json_values(completed[name].to_numpy()) for name in completed},
                    'observed': {name: observed[name].tolist() for name in observed},
                    'imputed': {name: imputed[name].tolist() for name in imputed}}

        def average(name, data, constituents, targets, paths):
            value = strict_blend(data, constituents)
            a, b = (result.mask('imputed') for result in constituents)
            assert value.mask('imputed').equals(a & b)
            for target in targets:
                mask = (a & b)[target]
                expected = (constituents[0].completed.loc[mask, target] + constituents[1].completed.loc[mask, target]) / 2
                np.testing.assert_allclose(value.completed.loc[mask, target], expected, rtol=1e-12, atol=1e-12)
            save(name + '-result.json', snapshot(data, value, targets) | {'constituents': paths, 'weights': [.5, .5]})
            return value

        results, result_data = {}, {}
        for method in ('P-U', 'P-X', 'T-U', 'T-X', 'AVG-J'):
            configured = deepcopy(doc)
            targets = configured['datasets']['biomet']['targets']
            recipe = targets['y']['routes']['short']['recipe']
            recipe.update(method=method, backend='p' if method.startswith('P') else 't', parameters={},
                          predictors=donor_refs if method in {'P-X', 'T-X', 'AVG-J'} else [])
            if method == 'AVG-J':
                recipe['targets'] = ['y', 'd1']
                del recipe['backend']
                pm, tm = deepcopy(recipe), deepcopy(recipe)
                pm.update(method='P-X', backend='p', parameters={})
                tm.update(method='T-J', backend='t', parameters={})
                recipe['members'] = [pm, tm]
                targets['d1'] = {'unit': 'dimensionless', 'unit_regime': 'native',
                                  'routes': {'short': deepcopy(targets['y']['routes']['short'])}}
            for route in targets['y']['routes'].values(): route['recipe'] = recipe
            _, _, partitions = prepare_operation(load_config(configured))
            partition = partitions[0]
            group = [g for g in partition['gaps'] if g['start_pos'] == 200 and g['variable'] in recipe['targets']]
            members = group[0]['recipes'][0].get('members', [group[0]['recipes'][0]])
            constituents, paths = [], []
            for member in members:
                selected = [g for g in group if g['variable'] in member['targets']]
                data, native, reason, _ = native_plan(partition['data'], member, selected, selected[0]['domain'], [], partition['donor_domains'])
                assert reason is None
                name = method + '-' + member['method']
                context.clear()
                context.update(mode=member['method'], case=method, plan=native.to_dict(),
                               units={v.name: v.unit for v in data.schema.variables})
                save(name + '-started.json', dict(context))
                fingerprint = data.fingerprint
                value = execute_native(data, native, member, RecordedPredict() if member['method'].startswith('P') else RecordedForecast())
                assert data.fingerprint == fingerprint
                paths.append(name + '-result.json')
                save(paths[-1], snapshot(data, value, member['targets']) | {'case': method, 'mode': member['method']})
                constituents.append(value)
                guard()
            results[method] = (constituents[0] if len(constituents) == 1 else
                               average(method, data, constituents, recipe['targets'], paths))
            result_data[method] = data
            outcomes[method] = {'complete': True}
        for name, p_name, t_name in (('AVG-U', 'P-U', 'T-U'), ('AVG-X', 'P-X', 'T-X')):
            average(name, result_data[p_name], [results[p_name], results[t_name]], ['y'],
                    [p_name + '-' + p_name + '-result.json', t_name + '-' + t_name + '-result.json'])
            outcomes[name] = {'complete': True, 'new_native_calls': 0}
        save('tabpfn-executions.json', p.executions)
    except BaseException as exc:
        error = type(exc).__name__
        cause = exc
        while cause is not None:
            error_chain.append(type(cause).__name__)
            cause = cause.__cause__
        failure_return(exc)
        raise
    finally:
        signal.alarm(0)
        receipt = {'schema': 'ecofill.native.compatibility/2', 'status': 'passed' if error is None else 'failed',
                   'error_type': error, 'loads': loads, 'load_attempts': load_attempts, 'reserved_calls': calls,
                   'events': events, 'unknown_entry': active, 'outcomes': outcomes, 'files': dict(files),
                   'error_chain': error_chain, 'publication_errors': publication_errors,
                   'seconds': time.monotonic() - began,
                   'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                   'synthetic_only': True, 'retries': 0}
        try: save('receipt.json', receipt)
        except BaseException:
            if error is None: raise


def readback(output):
    """Independently check hashes, journals, observations and strict arithmetic."""
    output = Path(output)
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['status'] == 'passed' and receipt['unknown_entry'] is None
    assert receipt['load_attempts'] == receipt['loads'] == {'tabpfn': 1, 'timesfm': 1}
    assert receipt['reserved_calls'] == {'tabpfn_fit': 4, 'tabpfn_predict': 4, 'timesfm_forecast': 6}
    for name, item in receipt['files'].items():
        path = output / name
        assert path.resolve().is_relative_to(output.resolve())
        payload = path.read_bytes()
        assert len(payload) == item['bytes'] and hashlib.sha256(payload).hexdigest() == item['sha256']
    assert len(receipt['events']) == 14
    for number, event in enumerate(receipt['events']):
        assert event['number'] == number
        assert event['status'] == 'returned'
        started = json.loads((output / f'{event["number"]:02d}-started.json').read_text())
        returned = json.loads((output / f'{event["number"]:02d}-return.json').read_text())
        assert started['kind'] == returned['kind'] == event['kind']
        assert returned['status'] == 'returned'
        if 'values' in returned:
            def finite(values):
                return all(finite(v) for v in values) if isinstance(values, list) else isinstance(values, (int, float)) and math.isfinite(values)
            assert finite(returned['values'])
    observations, fills, averages = 0, 0, 0
    for name in receipt['files']:
        if not name.endswith('-result.json'): continue
        result = json.loads((output / name).read_text())
        assert len(result['index']) == 600
        for target in result['raw']:
            raw, completed, observed, imputed = (result[key][target] for key in ('raw', 'completed', 'observed', 'imputed'))
            assert len(raw) == len(completed) == len(observed) == len(imputed) == 600
            for i, original in enumerate(observed):
                if original:
                    assert raw[i] == completed[i] and not imputed[i]
                    observations += 1
                if imputed[i]:
                    assert not original and completed[i] is not None and math.isfinite(completed[i])
                    fills += 1
            if target in result['targets']:
                assert all(imputed[i] for i in range(200, 203))
                if 'mode' in result:
                    entries = []
                    for event in receipt['events']:
                        start = json.loads((output / f'{event["number"]:02d}-started.json').read_text())
                        if start['case'] == result['case'] and start['mode'] == result['mode']:
                            end = json.loads((output / f'{event["number"]:02d}-return.json').read_text())
                            if 'values' in end: entries.append((start['request'], end['values']))
                    if result['mode'].startswith('P'):
                        values = [values for request, values in entries if request['target'] == target]
                        assert len(values) == 1 and completed[200:203] == values[0]
                    else:
                        assert len(entries) == 2
                        forward = next((request, values) for request, values in entries if request['direction'] == 'forward')
                        backward = next((request, values) for request, values in entries if request['direction'] == 'backward')
                        col = forward[0]['channels'].index(target)
                        for i in range(3):
                            expected = (1 - i / 2) * forward[1][i][col] + (i / 2) * backward[1][2 - i][col]
                            assert math.isclose(completed[200 + i], expected, rel_tol=1e-12, abs_tol=1e-12)
            if 'constituents' in result and target in result['targets']:
                assert result['weights'] == [.5, .5]
                a, b = [json.loads((output / part).read_text()) for part in result['constituents']]
                assert a['index'] == b['index'] == result['index']
                assert a['units'][target] == b['units'][target] == result['units'][target]
                assert a['observed'][target] == b['observed'][target] == observed
                for i in range(600):
                    assert imputed[i] == (a['imputed'][target][i] and b['imputed'][target][i])
                    if imputed[i]:
                        assert math.isclose(completed[i], (a['completed'][target][i] + b['completed'][target][i]) / 2,
                                            rel_tol=1e-12, abs_tol=1e-12)
                averages += 1
    assert averages == 4  # U, X and each of two J targets.
    return {'status': 'passed', 'observations_checked': observations, 'filled_values_checked': fills,
            'average_targets_checked': averages, 'durable_calls': len(receipt['events']),
            'modes': ['P-U', 'P-X', 'T-U', 'T-X', 'T-J', 'AVG-U', 'AVG-X', 'AVG-J']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('public', 'models', 'output'): parser.add_argument('--' + name)
    parser.add_argument('--readback')
    args = parser.parse_args()
    if args.readback:
        print(json.dumps(readback(args.readback), sort_keys=True))
    else:
        if not all((args.public, args.models, args.output)): parser.error('--public, --models and --output required')
        main(args.public, args.models, args.output)
