"""Packaged synthetic 30-minute weather example for the concise configuration; standard library only.

prepare writes inputs/weather.csv, config.json (concise) and config-engineer.json (the
sources/routes/output spelling) into a new directory; run ecofill from that directory.
readback checks a finished run bundle without importing ecofill, NumPy or pandas.
replay-config points a saved resolved configuration at the bundle's input snapshot.
Run this installed file directly (python -I -S PATH/simple_example.py ...) or through
examples/operation_simple_demo.py; both use this single implementation.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path

COLUMNS = ['timestamp', 'temp_out', 'humidity', 'dew_point', 'wind_speed', 'wind_chill', 'heat_index', 'bar',
           'rain', 'rain_rate', 'solar_rad', 'solar_energy', 'et', 'data_source', 'station_id', 'primary_key']
ROWS = 480
LOCAL = timezone(timedelta(hours=-6))  # The file carries explicit offsets; no timezone is declared.
START = datetime(2025, 3, 1, tzinfo=LOCAL)
MISSING_ROW = 150                      # Absent timestamp: inserted on the grid, then filled.
EMPTY = {0, 1, 100, 101, 102, 201, 300, 301, 302, 303, 304, 305, 350, 351, 352, 353, 478, 479}
NONFINITE = {200}                      # Lexical NaN is unavailable, never a value.
FILLED = {100, 101, 102, 150, 200, 201, 350, 351, 352, 353}
UNRESOLVED = {0: 'boundary_gap_unresolved', 1: 'boundary_gap_unresolved',
              **{i: 'no_route' for i in range(300, 306)},
              478: 'boundary_gap_unresolved', 479: 'boundary_gap_unresolved'}

CONCISE = {'datasets': {'weather': {'uri': 'inputs/weather.csv', 'cadence': '30min', 'targets': ['bar'],
                                    'method': 'LIN', 'max_gap': '2h'}},
           'output': {'uri': 'output'}}
ENGINEER = {'sources': {'davis_weather': {'uri': 'inputs/weather.csv', 'cadence': '30min', 'variables': {
                'temp_out': {'unit': 'degC', 'role': 'driver'}, 'bar': {'unit': 'hPa', 'role': 'response'},
                'dew_point': {'unit': 'degC', 'role': 'driver'}}}},
            'routes': {'davis_weather': {'enabled': True, 'method': 'linear', 'mode': 'offline', 'max_gap': '2h'}},
            'output': {'uri': 'output'}}


def bar(i):
    return 1000 + 0.5 * i


def write_csv(path, rows=ROWS, start=START, cadence=timedelta(minutes=30), station='COL-01'):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(COLUMNS)
        for i in range(rows):
            if i == MISSING_ROW:
                continue
            stamp = (start + i * cadence).isoformat()
            pressure = '' if i in EMPTY else 'NaN' if i in NONFINITE else repr(bar(i))
            temp = '' if i % 97 == 5 else f'{20 + (i % 48) / 4:.2f}'
            writer.writerow([stamp, temp, 60 + i % 30, f'{12 + (i % 48) / 8:.3f}', i % 7, temp, temp, pressure,
                             0, 0, (i * 37) % 900, (i * 11) % 50, '0.01', 'davis', station, f'row-{i}'])
    return path


def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write_csv(root / 'inputs/weather.csv')
    for name, document in (('config.json', CONCISE), ('config-engineer.json', ENGINEER)):
        (root / name).write_bytes((json.dumps(document, indent=2) + '\n').encode('utf-8'))
    return {'config': 'config.json', 'engineer_config': 'config-engineer.json', 'input': 'inputs/weather.csv'}


def table(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def readback(run):
    """Independent integrity and value checks of one finished simple-weather run bundle."""
    run = Path(run)
    terminal = json.loads((run / 'terminal.json').read_text('utf-8'))
    manifest_bytes = (run / 'manifest.json').read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == terminal['manifest_sha256']
    for item in json.loads(manifest_bytes)['files']:
        path = run / item['path']
        assert path.resolve().is_relative_to(run.resolve())
        payload = path.read_bytes()
        assert len(payload) == item['bytes'] and hashlib.sha256(payload).hexdigest() == item['sha256']
    summary = json.loads((run / 'summary.json').read_text('utf-8'))
    resolved = json.loads((run / 'config.resolved.json').read_text('utf-8'))
    assert resolved['schema'] == 'ecofill.operation/2' and resolved['resolution']['status'] == 'resolved'
    plan = json.loads((run / 'plan.json').read_text('utf-8'))
    (name, source), = resolved['sources'].items()
    snapshot = (run / 'inputs' / (name + '.snapshot')).read_bytes()
    assert hashlib.sha256(snapshot).hexdigest() == source['sha256'] == plan['inputs'][name]['sha256']
    partition, = sorted(p.name for p in (run / 'datasets').iterdir())
    prefix = run / 'datasets' / partition
    raw, completed, observed, reconstructed, dispositions = (
        table(prefix / (n + '.csv.gz')) for n in ('raw', 'completed', 'observed', 'reconstructed', 'dispositions'))
    assert list(completed[0]) == ['timestamp', 'bar'] and len(completed) == ROWS
    first = datetime.fromisoformat(completed[0]['timestamp'])
    checked = filled = 0
    for i in range(ROWS):
        assert datetime.fromisoformat(completed[i]['timestamp']) == START + timedelta(minutes=30 * i) == \
            first + timedelta(minutes=30 * i)
        if observed[i]['bar'] == 'True':
            assert float(raw[i]['bar']) == float(completed[i]['bar']) == bar(i)
            assert reconstructed[i]['bar'] == 'False'
            checked += 1
        elif i in FILLED:
            assert reconstructed[i]['bar'] == 'True' and abs(float(completed[i]['bar']) - bar(i)) <= 1e-9
            filled += 1
        else:
            assert completed[i]['bar'] == '' and reconstructed[i]['bar'] == 'False'
            assert dispositions[i]['bar'] == UNRESOLVED[i], (i, dispositions[i]['bar'])
    assert checked == ROWS - len(EMPTY) - len(NONFINITE) - 1 and filled == len(FILLED)
    return {'status': summary['status'], 'partition': partition, 'targets': ['bar'], 'filled_cells': filled,
            'observations_checked': checked, 'unresolved': dict(sorted(summary['unresolved_reasons'].items())),
            'snapshot_verified': True, 'resolved_schema': resolved['schema']}


def replay_config(run, destination, output_uri):
    """Point the resolved configuration at the retained snapshots; its sha256 pins still apply."""
    run = Path(run)
    document = json.loads((run / 'config.resolved.json').read_text('utf-8'))
    for name, source in document['sources'].items():
        source['uri'] = (run / 'inputs' / (name + '.snapshot')).resolve().as_posix()
    document['output']['uri'] = output_uri
    Path(destination).write_bytes((json.dumps(document, indent=2, sort_keys=True) + '\n').encode('utf-8'))
    return {'config': str(destination)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'readback', 'replay-config'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--run')
    parser.add_argument('--output-uri')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.output)
    elif args.command == 'readback':
        result = readback(args.output)
    else:
        result = replay_config(args.run, args.output, args.output_uri)
    print(json.dumps(result, sort_keys=True))
