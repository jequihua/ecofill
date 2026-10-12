"""Synthetic four-shape installed operation and fully independent stdlib readback."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path


def fixture(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    sources, datasets = {}, {}
    for family in ('flux', 'biomet', 'ctd', 'weather'):
        cadence = 1800 if family in {'flux', 'biomet'} else 3600
        clock = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = clock + timedelta(seconds=600 * cadence)
        period = [clock.isoformat(), end.isoformat()]
        path = root / (family + '.csv')
        identity = 'site_id' if family in {'flux', 'biomet'} else 'station_id'
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['primary_key', identity, 'timestamp', 'y', 'd1', 'd2', 'qc_y', 'data_source'])
            for i in range(600):
                if i == 300:  # Explicit missing grid timestamp.
                    continue
                value = '' if i in {200, 201, 202} else -9991 if i == 400 else 10 + .1 * i
                writer.writerow([f'{family}-{i}', 'SYNTHETIC', (clock + timedelta(seconds=i * cadence)).isoformat(),
                                 value, .2 * i, i % 24, 2 if i == 400 else 0, 'LOCAL' if i % 2 else 'REMOTE'])
        fields = {name: {'column': name, 'unit': 'C' if name == 'y' else 'dimensionless', 'unit_regime': 'native',
                         'lineage': f'{family}|SYNTHETIC|{name}', 'derived_from': [], 'domains': [period]}
                  for name in ('y', 'd1', 'd2')}
        fields['y']['qc_column'] = 'qc_y'
        sources[family] = {'family': family, 'uri': path.as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'identity_column': identity, 'timestamp_column': 'timestamp', 'key_column': 'primary_key',
            'cadence_seconds': cadence, 'clock': 'UTC', 'preparation_profile': 'manglaria-original-v1',
            'policy': {'version': 'synthetic-v1', 'original_evidence': 'synthetic.original-observations',
                       'missing_strings': ['', 'NA', 'NULL'], 'positive_9999': True, 'negative_sentinel_fields': []},
            'instruments': {'SYNTHETIC': {'aliases': ['ALIAS'], 'validity': period, 'exclusions': [], 'fields': fields}}}
        donors = [dict(source=family, instrument='SYNTHETIC', field=n, unit_regime='native') for n in ('d1', 'd2')]
        recipe = {'id': family + '-fixed', 'method': 'RF-X' if family == 'weather' else 'LIN', 'targets': ['y'],
                  'predictors': donors if family == 'weather' else [],
                  'parameters': {'n_estimators': 8, 'max_depth': 4, 'min_samples_leaf': 2, 'seed': 13} if family == 'weather' else {},
                  'evidence_status': 'user_configured', 'evidence': ['synthetic.software-only']}
        routes = {cls: {'enabled': True, 'duration_seconds': {'greater_than': a, 'at_most': b},
                        'recipe': recipe, 'fallbacks': []} for cls, a, b in
                  (('short', 0, 7200), ('medium', 7200, 43200), ('long', 43200, 604800))}
        datasets[family] = {'enabled': True, 'source': family, 'instruments': ['SYNTHETIC'], 'interval': period,
            'exclusions': [], 'fit_exclusions': [], 'targets': {'y': {'unit': 'C', 'unit_regime': 'native', 'routes': routes}}}
    document = {'schema': 'ecofill.operation/1', 'purpose': 'non-commercial research', 'output': {'uri': (root / 'output').as_posix()},
        'limits': {'max_gaps': 100, 'max_method_calls': 100, 'max_rows_per_series': 1000, 'wall_seconds': 60,
                   'memory_bytes': 64 * 1024**2, 'scratch_bytes': 4 * 1024**2, 'output_bytes': 8 * 1024**2},
        'sources': sources, 'datasets': datasets, 'models': {}}
    (root / 'config.json').write_bytes((json.dumps(document, indent=2) + '\n').encode('utf-8'))
    return document


def readback(root):
    root = Path(root)
    terminal = json.loads((root / 'terminal.json').read_text())
    manifest_bytes = (root / 'manifest.json').read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == terminal['manifest_sha256']
    manifest = json.loads(manifest_bytes)
    for item in manifest['files']:
        path = root / item['path']
        assert path.resolve().is_relative_to(root.resolve())
        payload = path.read_bytes()
        assert len(payload) == item['bytes']
        assert hashlib.sha256(payload).hexdigest() == item['sha256']
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['status'] == 'complete'
    assert summary['datasets_processed'] == 4
    assert summary['filled_cells'] == 19
    checked = 0
    for family in ('flux', 'biomet', 'ctd', 'weather'):
        prefix = root / 'datasets' / (family + '-0')
        def table(name):
            with gzip.open(prefix / (name + '.csv.gz'), 'rt', newline='') as stream:
                return list(csv.DictReader(stream))
        raw, filled, observed, mask = (table(n) for n in ('raw', 'completed', 'observed', 'reconstructed'))
        assert len(raw) == 600
        for i in range(600):
            if observed[i]['y'] == 'True':
                assert float(raw[i]['y']) == float(filled[i]['y']) == 10 + .1 * i
                assert mask[i]['y'] == 'False'
                checked += 1
        assert float(raw[400]['y']) == -9991
        assert all(mask[i]['y'] == 'True' for i in (200, 201, 202, 400))
        assert mask[300]['y'] == ('False' if family == 'weather' else 'True')
    return {'status': 'complete', 'datasets': 4, 'filled_cells': 19, 'observations_checked': checked,
            'methods': sorted({row['selected_method'] for family in ('flux', 'biomet', 'ctd', 'weather')
                for row in json.loads((root / 'datasets' / (family + '-0') / 'gaps.json').read_text()) if row['complete']})}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'demo', 'readback'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--foundation-doubles', action='store_true')
    args = parser.parse_args()
    if args.command == 'readback':
        result = readback(args.output)
    else:
        document = fixture(args.output)
        if args.command == 'demo':
            from ecofill.operation import run
            backends = {}
            if args.foundation_doubles:
                import numpy as np
                class PredictDouble:
                    identity = {'backend': 'labelled-foundation-double', 'device': 'cpu', 'software_only': True}
                    def predict(self, request):
                        return np.full(len(request.query_features), 17.)
                class ForecastDouble:
                    identity = {'backend': 'labelled-foundation-double', 'device': 'cpu', 'software_only': True}
                    def forecast(self, request):
                        return np.full((request.horizon, len(request.channels)), 23.)
                for family, method, name, kind in [('biomet', 'P-U', 'p', 'tabpfn35'), ('ctd', 'T-U', 't', 'timesfm3')]:
                    document['models'][name] = {'kind': kind, 'artifacts': [], 'license': 'synthetic-software-only', 'revision': 'double'}
                    for route in document['datasets'][family]['targets']['y']['routes'].values():
                        route['recipe'].update(method=method, backend=name)
                backends = {'p': PredictDouble(), 't': ForecastDouble()}
            result = run(document, run_id='synthetic', backends=backends)
        else:
            result = {'config': str(Path(args.output) / 'config.json')}
    print(json.dumps(result, sort_keys=True))
