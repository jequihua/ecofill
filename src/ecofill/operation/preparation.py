"""Four-family lexical raw snapshots and qualified UTC grids."""
from dataclasses import dataclass
import csv
import gzip
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from ..batch import prepare_batch
from .config import canonical, digest, resolve_ref, stamp


@dataclass
class SourceInput:
    payload: bytes
    columns: tuple
    rows: list
    receipt: dict


def read_source(source, limits):
    path = Path(source['uri'])
    if path.stat().st_size > limits['scratch_bytes']:
        raise ValueError('source exceeds scratch budget')
    payload = path.read_bytes()
    return decode_source(source, limits, payload, compressed=path.suffix == '.gz')


def required_columns(source):
    required = {source[k] for k in ('identity_column', 'timestamp_column', 'key_column')}
    required.update(source.get('provenance_columns', ()))
    for metadata in source['instruments'].values():
        for field in metadata['fields'].values():
            required.add(field['column'])
            required.update(field[k] for k in ('qc_column', 'observed_column', 'rejected_column', 'structural_column', 'route_column') if k in field)
    return sorted(required)


def decode_source(source, limits, payload, *, compressed=False, receipt=None):
    identity = hashlib.sha256(payload).hexdigest()
    if identity != source['sha256']:
        raise ValueError('source bytes differ from pinned identity')
    if compressed:
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
            decoded = stream.read(limits['scratch_bytes'] + 1)
    else:
        decoded = payload
    if len(decoded) > limits['scratch_bytes'] or len(decoded) * 8 > limits['memory_bytes']:
        raise ValueError('source exceeds decoded memory/scratch estimate')
    reader = csv.reader(io.StringIO(decoded.decode('utf-8-sig')))
    columns = next(reader, None)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError('source requires unique CSV columns')
    if set(required_columns(source)) - set(columns):
        raise ValueError('source typed schema columns missing')
    rows = []
    for row in reader:
        if len(row) != len(columns):
            raise ValueError('malformed CSV row')
        rows.append(dict(zip(columns, row)))
        if len(rows) > limits['max_rows_per_series'] * max(1, len(source['instruments'])):
            raise ValueError('bounded source selectors exceeded')
    return SourceInput(payload, tuple(columns), rows, {'sha256': identity, 'bytes': len(payload),
        'rows': len(rows), 'columns': columns, 'source_policy': source['policy'],
        'preparation_profile': source['preparation_profile'], 'clock': 'UTC',
        'sort': 'stable ascending UTC; lexical input bytes and row order retained separately'} | (receipt or {}))


def bool_cell(value):
    if value not in {'True', 'False'}:
        raise ValueError('qualified mask requires literal True/False')
    return value == 'True'


def support_domains(source, instrument, field, extra=()):
    """Subtract explicit/reserved prohibitions without splitting original gaps."""
    metadata = source['instruments'][instrument]
    excluded = list(metadata['exclusions']) + list(extra)
    if instrument in {'RBRL-COL', 'RBRL-COL-04', 'COL04'} or any(
            n in {'RBRL-COL', 'RBRL-COL-04', 'COL04'} for n in metadata['aliases']):
        return []
    if source['family'] == 'ctd' and any(n in {'X8455', 'EC03S', 'RBRL-EC-03', 'RBRL-EC-03S'}
                                      for n in [instrument, *metadata['aliases']]):
        excluded.append(['2026-01-01T00:00:00Z', '2026-03-28T00:00:00Z'])
    result = [(stamp(a), stamp(b)) for a, b in field['domains']]
    for a, b in excluded:
        a, b = stamp(a), stamp(b)
        result = [(x, y) for left, right in result for x, y in
                  ((left, min(a, right)), (max(b, left), right)) if x < y] if a < b else result
    # The subtraction above needs unchanged disjoint intervals exactly once.
    return [(a.isoformat(), b.isoformat()) for a, b in result]


def prepare_field(doc, ref, inputs, limits):
    source, instrument, field = resolve_ref(doc, ref)
    metadata = source['instruments'][instrument]
    aliases = {instrument, *metadata['aliases']}
    selected = [(position, row) for position, row in enumerate(inputs[ref['source']].rows)
                if row[source['identity_column']] in aliases]
    if not selected:
        raise FileNotFoundError('configured instrument has no rows')
    selected = sorted([(stamp(row[source['timestamp_column']]), position, row) for position, row in selected],
                      key=lambda item: item[0])
    index = pd.DatetimeIndex([item[0] for item in selected])
    if index.has_duplicates:
        raise ValueError('duplicate instrument/UTC timestamp')
    cadence = pd.Timedelta(seconds=source['cadence_seconds'])
    if (index.asi8 % cadence.value).any():
        raise ValueError('off-grid UTC timestamp')
    count = int((index[-1] - index[0]) / cadence) + 1
    if count > limits['max_rows_per_series'] or count * 512 > limits['memory_bytes']:
        raise ValueError('prepared series exceeds row/memory estimate')
    raw, observed, qc, structural, names, reasons, keys = [], [], [], [], [], [], []
    missing = set(source['policy']['missing_strings'])
    for timestamp, position, row in selected:
        value = row[field['column']]
        why = []
        if value in missing:
            numeric = np.nan; why.append('original_missing')
        else:
            try:
                numeric = float(value)
            except ValueError:
                numeric = np.nan; why.append('invalid_numeric')
            if not np.isfinite(numeric):
                numeric = np.nan; why.append('nonfinite_or_invalid')
        sentinel = None
        if np.isfinite(numeric) and source['policy']['positive_9999'] and numeric == 9999:
            sentinel = 'owner_provisional_positive_9999'
        if field['column'] in source['policy']['negative_sentinel_fields'] and numeric <= -9990:
            sentinel = 'named_negative_9990'
        rejected = sentinel is not None
        if 'rejected_column' in field and bool_cell(row[field['rejected_column']]):
            rejected = True; why.append('qualified_bundle_qc_rejection')
        if sentinel:
            why.append('named_sentinel:' + sentinel)
        if 'qc_column' in field and row[field['qc_column']] not in {'0', '1', '0.0', '1.0'}:
            rejected = True; why.append('qc_rejection')
        is_structural = bool_cell(row[field['structural_column']]) if 'structural_column' in field else False
        qualified = bool_cell(row[field['observed_column']]) if 'observed_column' in field else np.isfinite(numeric)
        if qualified and not np.isfinite(numeric) and 'observed_column' in field:
            raise ValueError('observation evidence marks a nonfinite value')
        if 'route_column' in field and row[field['route_column']] != field['route_value']:
            qualified = False; why.append('outside_unit_regime')
        qualified = bool(qualified and np.isfinite(numeric) and not rejected and not is_structural)
        raw.append(numeric); observed.append(qualified)
        # Core rejection masks mark retained numeric raw cells. A null carrying
        # a QC reason remains null; its reason survives the lexical snapshot/audit.
        qc.append(bool(rejected and np.isfinite(numeric))); structural.append(is_structural)
        names.append(sentinel); reasons.append([timestamp.isoformat(), why or ['qualified_observation' if qualified else 'unqualified_raw']])
        keys.append([timestamp.isoformat(), row[source['key_column']], position])
    name = ref['field']
    frame = pd.DataFrame({name: raw}, index=index)
    contract = {'identity': {'dataset': source['family'], 'instrument': instrument, 'variable': name,
                             'unit_regime': field['unit_regime']},
        'input': {'cadence_seconds': source['cadence_seconds'], 'clock': 'UTC',
            'variables': [{'name': name, 'unit': field['unit'], 'role': 'target', 'derived_from': []}],
            'source_policy': {'qc': source['policy']['version'], 'sentinel': source['policy']['version'],
                              'structural': source['policy']['version']}}}
    config = {'schema': 'ecofill.batch/1', **contract, 'mode': 'offline', 'support': 'original_only',
              'status': 'usable', 'evidence': [source['policy']['original_evidence']],
              'fitting': {'window_days': 180, 'min_rows': 168, 'max_rows': 4096, 'selection': 'even_time_coverage'},
              'routes': []}
    prepared = prepare_batch(frame, config=config, metadata=contract,
        observed=pd.DataFrame({name: observed}, index=index),
        qc_rejected=pd.DataFrame({name: qc}, index=index),
        structural=pd.DataFrame({name: structural}, index=index),
        sentinel_names=pd.DataFrame({name: names}, index=index))
    return prepared, {'reference': ref, 'physical_instrument': instrument, 'keys': keys,
        'source_reasons': reasons, 'domains': support_domains(source, instrument, field),
        'validity': metadata['validity'], 'unit': field['unit'], 'unit_regime': field['unit_regime'],
        'lineage': field['lineage'], 'source_identity': inputs[ref['source']].receipt['sha256']}
