"""Four-family lexical raw snapshots and qualified UTC grids; generic-v1 user-supplied series."""
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import io
import math
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from ..batch import prepare_batch
from .config import GENERIC, canonical, diagnostic, digest, pointer, resolve_ref, stamp

CONVENTIONAL_IDENTITY = 'station_id'
NARROW = 'select fewer series or a shorter period, or raise the limit within deployment policy'
TIME = re.compile(r'(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?)(Z|[+-]\d{2}:?\d{2})?')
NUMBER = re.compile(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
NONFINITE = {'nan', '+nan', '-nan', 'inf', '+inf', '-inf', 'infinity', '+infinity', '-infinity'}


@dataclass
class SourceInput:
    payload: bytes
    columns: tuple
    rows: list
    receipt: dict
    normalized: dict = dataclass_field(default_factory=dict, repr=False)  # Generic clock/series cache.


def bound(code, limit, message, advice):
    """Known user-correctable limit failure; names the limit, never the input path or values."""
    return diagnostic(code, pointer('limits', limit), f'{message}: {advice}')


def read_source(source, limits):
    path = Path(source['uri'])
    if source.get('format') == 'parquet' and path.is_dir():
        raise diagnostic('single_parquet_file_required', pointer('datasets', source_name(source), 'uri'),
                         'Parquet input must be one closed local file, not a directory or dataset')
    if path.stat().st_size > limits['scratch_bytes']:
        raise bound('source_exceeds_scratch_bytes', 'scratch_bytes', 'source exceeds scratch budget',
                    'the input file is larger than limits.scratch_bytes; use a smaller extract or raise '
                    'limits.scratch_bytes within deployment policy')
    if source.get('format') == 'parquet':
        # Capture once with an incremental limit, including files growing after stat.
        pieces, size = [], 0
        with path.open('rb') as stream:
            while piece := stream.read(min(1024**2, limits['scratch_bytes'] - size + 1)):
                size += len(piece)
                if size > limits['scratch_bytes'] or size * 2 > limits['memory_bytes']:
                    raise bound('source_capture_limit', 'scratch_bytes', 'source grew beyond the capture limit', 'materialize a smaller immutable file')
                pieces.append(piece)
        payload = b''.join(pieces)
    else:
        payload = path.read_bytes()
    return decode_source(source, limits, payload, compressed=source.get('compressed', path.suffix == '.gz'))


def required_columns(source):
    required = {source[k] for k in ('identity_column', 'timestamp_column', 'key_column')} - {None}
    required.update(source.get('provenance_columns', ()))
    for metadata in source['instruments'].values():
        for field in metadata['fields'].values():
            required.add(field['column'])
            required.update(field[k] for k in ('qc_column', 'observed_column', 'rejected_column', 'structural_column', 'route_column') if k in field)
    return sorted(required)


def source_name(source):
    """Generic lineages are '<dataset>|<series>|<column>'; used only to locate diagnostics."""
    metadata = next(iter(source['instruments'].values()))
    return next(iter(metadata['fields'].values()))['lineage'].split('|')[0]


def decode_source(source, limits, payload, *, compressed=False, receipt=None):
    """Hash, decode and parse exactly these bytes; a supplied expected hash is enforced first."""
    if source.get('format') == 'parquet':
        from .parquet import decode
        return decode(source, limits, payload, receipt)
    identity = hashlib.sha256(payload).hexdigest()
    generic = source.get('preparation_profile') == GENERIC
    if (source['sha256'] if not generic else source.get('sha256', identity)) != identity:
        if generic:
            raise diagnostic('source_hash_mismatch', pointer('datasets', source_name(source), 'sha256'),
                             'input bytes differ from the expected sha256; nothing was executed')
        raise ValueError('source bytes differ from pinned identity')
    if compressed:
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
            decoded = stream.read(limits['scratch_bytes'] + 1)
    else:
        decoded = payload
    if len(decoded) > limits['scratch_bytes']:
        raise bound('decoded_source_exceeds_scratch_bytes', 'scratch_bytes', 'source exceeds decoded memory/scratch '
                    'estimate', 'the decoded input is larger than limits.scratch_bytes; use a smaller extract or '
                    'raise limits.scratch_bytes within deployment policy')
    if len(decoded) * 8 > limits['memory_bytes']:
        raise bound('decoded_source_exceeds_memory_bytes', 'memory_bytes', 'source exceeds decoded memory/scratch '
                    'estimate', 'the conservative memory estimate (8 bytes per decoded byte) exceeds '
                    'limits.memory_bytes; use a smaller extract or raise limits.memory_bytes within deployment policy')
    reader = csv.reader(io.StringIO(decoded.decode('utf-8-sig')))
    columns = next(reader, None)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError('source requires unique CSV columns')
    absent = set(required_columns(source)) - set(columns)
    if absent and generic:
        raise diagnostic('missing_column', pointer('datasets', source_name(source), 'columns', min(absent)),
                         f'{len(absent)} configured column(s) absent from the CSV header; check spelling, '
                         'columns.<name>.column, timestamp, key and group_by')
    if absent:
        raise ValueError('source typed schema columns missing')
    from .simple import MAX_SERIES  # Discovered series of pending grouped input share one bound.
    series = MAX_SERIES if '*' in source['instruments'] else max(1, len(source['instruments']))
    rows = []
    for row in reader:
        if len(row) != len(columns):
            raise ValueError('malformed CSV row')
        rows.append(dict(zip(columns, row)))
        if len(rows) > limits['max_rows_per_series'] * series:
            raise bound('source_rows_exceed_limit', 'max_rows_per_series', 'bounded source selectors exceeded',
                        'the file has more rows than limits.max_rows_per_series allows for its selected series; '
                        + NARROW)
    return SourceInput(payload, tuple(columns), rows, {'sha256': identity, 'bytes': len(payload),
        'rows': len(rows), 'columns': columns, 'source_policy': source['policy'],
        'preparation_profile': source['preparation_profile'], 'clock': 'UTC',
        'sort': 'stable ascending UTC; lexical input bytes and row order retained separately'} | (receipt or {}))


def prepared_limits(count, limits):
    if count > limits['max_rows_per_series']:
        raise bound('prepared_rows_exceed_limit', 'max_rows_per_series', 'prepared series exceeds row/memory estimate',
                    'the regular grid from the first to the last selected timestamp has more rows than '
                    'limits.max_rows_per_series; ' + NARROW)
    if count * 512 > limits['memory_bytes']:
        raise bound('prepared_memory_exceeds_limit', 'memory_bytes', 'prepared series exceeds row/memory estimate',
                    'the conservative estimate of 512 bytes per grid row exceeds limits.memory_bytes; ' + NARROW)


def bool_cell(value):
    if value not in {'True', 'False'}:
        raise ValueError('qualified mask requires literal True/False')
    return value == 'True'


def support_domains(source, instrument, field, extra=()):
    """Subtract explicit/reserved prohibitions without splitting original gaps."""
    metadata = source['instruments'][instrument]
    excluded = list(metadata['exclusions']) + list(extra)
    # Project reservations belong to the explicit project profiles, never to generic input.
    project = source['preparation_profile'] != GENERIC
    if project and (instrument in {'RBRL-COL', 'RBRL-COL-04', 'COL04'} or any(
            n in {'RBRL-COL', 'RBRL-COL-04', 'COL04'} for n in metadata['aliases'])):
        return []
    if project and source['family'] == 'ctd' and any(n in {'X8455', 'EC03S', 'RBRL-EC-03', 'RBRL-EC-03S'}
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
    if source['preparation_profile'] == GENERIC:
        return prepare_generic_field(doc, ref, inputs, limits)
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
    prepared_limits(count, limits)
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


def parse_time(text, local, where):
    """ISO 8601 only. Offsets normalize to UTC; naive readings need the declared timezone."""
    if isinstance(text, datetime):
        text = text.isoformat()
    if not isinstance(text, str):
        raise diagnostic('invalid_timestamp', pointer(*where), 'timestamp must be a supported timestamp string or typed Arrow timestamp')
    match = TIME.fullmatch(text)
    if not match:
        raise diagnostic('invalid_timestamp', pointer(*where), 'timestamps must be ISO 8601, YYYY-MM-DD HH:MM[:SS] '
                         'or YYYY-MM-DDTHH:MM[:SS], with an optional Z or +HH:MM offset')
    offset = match[3] or ''
    if offset == 'Z':
        offset = '+00:00'
    elif offset and ':' not in offset:
        offset = offset[:3] + ':' + offset[3:]
    try:
        value = datetime.fromisoformat(match[1] + 'T' + match[2] + offset)
    except ValueError:
        raise diagnostic('invalid_timestamp', pointer(*where), 'invalid calendar timestamp') from None
    if value.tzinfo is None:
        if local is None:
            raise diagnostic('naive_timestamp', pointer(*where), 'timestamps have no UTC offset; declare timezone '
                             '(an IANA name such as America/Mexico_City, UTC, or +HH:MM)')
        first, second = value.replace(tzinfo=local, fold=0), value.replace(tzinfo=local, fold=1)
        if first.utcoffset() != second.utcoffset():
            if first.astimezone(timezone.utc).astimezone(local).replace(tzinfo=None) != value:
                raise diagnostic('nonexistent_local_time', pointer(*where), 'a local time does not exist at a '
                                 'daylight-saving change; use timestamps with UTC offsets')
            raise diagnostic('ambiguous_local_time', pointer(*where), 'a local time occurs twice at a daylight-saving '
                             'change; use timestamps with UTC offsets')
        return first.astimezone(timezone.utc), False, first.utcoffset().total_seconds()
    if local is not None and value.utcoffset() != value.astimezone(local).utcoffset():
        raise diagnostic('contradictory_timezone', pointer(*where), 'a timestamp offset contradicts the declared '
                         'timezone; remove timezone or correct the offsets')
    return value.astimezone(timezone.utc), True, value.utcoffset().total_seconds()


def normalize(source, entry, name):
    """UTC rows per series, checked once per source bytes; never rounded, resampled or merged."""
    from .simple import MAX_SERIES, series_name, zone
    identity, column = source['identity_column'], source['timestamp_column']
    wanted = None if identity is None or '*' in source['instruments'] else frozenset(source['instruments'])
    base = (identity, column, source['cadence_seconds'], source['clock']['timezone'])
    for key in ((base, wanted), (base, None)):
        if key in entry.normalized and (wanted is None or wanted <= set(entry.normalized[key]['series'])):
            return entry.normalized[key]
    where = ('datasets', name)
    if identity is None and CONVENTIONAL_IDENTITY in entry.columns:
        if len({row[CONVENTIONAL_IDENTITY] for row in entry.rows} - {'', None}) > 1:
            raise diagnostic('multiple_series', pointer(*where), 'the station_id column holds more than one station; '
                             'add group_by: station_id (and optionally select) so each station is filled separately')
    local = None if source['clock']['timezone'] is None else zone(source['clock']['timezone'])
    series, aware, naive, offsets = {}, 0, 0, set()
    for position, row in enumerate(entry.rows):
        key = None if identity is None else row[identity]
        if entry.receipt.get('format') == 'parquet' and type(key) is int:
            key = str(key)  # Series labels are text; exact keys remain typed in the audit.
        if wanted is not None and key not in wanted:
            continue
        if identity is not None and key in {'', None}:
            raise diagnostic('empty_series_identity', pointer(*where, 'group_by'), 'a row has an empty group_by value')
        moment, has_offset, offset = parse_time(row[column], local, (*where, 'timestamp'))
        aware += has_offset; naive += not has_offset; offsets.add(offset)
        instant = pd.Timestamp(moment)
        if entry.receipt.get('format') == 'parquet':
            try:
                instant.value
            except (OverflowError, ValueError):
                raise diagnostic('timestamp_out_of_range', pointer(*where, 'timestamp'),
                                 'timestamp is outside the current nanosecond UTC grid range; export a supported interval') from None
        series.setdefault(key, []).append((instant, position, row))
    if identity is not None and wanted is None and len(series) > MAX_SERIES:
        raise diagnostic('too_many_series', pointer(*where, 'group_by'), f'more than {MAX_SERIES} series; use select')
    cadence = pd.Timedelta(seconds=source['cadence_seconds']).value
    for key, items in series.items():
        if key is not None:
            series_name(key, (*where, 'group_by'))
        items.sort(key=lambda item: item[0])
        if entry.receipt.get('format') == 'parquet' and items[-1][0].value - items[0][0].value > np.iinfo(np.int64).max:
            raise diagnostic('timestamp_span_out_of_range', pointer(*where, 'timestamp'),
                             'timestamp span exceeds the current clock contract; select a narrower interval')
        stamps = np.array([item[0].value for item in items], dtype=np.int64)
        if (np.diff(stamps) == 0).any():
            raise diagnostic('duplicate_timestamp', pointer(*where, 'timestamp'), 'two readings of one series share a '
                             'UTC timestamp; ecofill never merges duplicates')
        if ((stamps - stamps[0]) % cadence).any():
            raise diagnostic('off_grid_timestamp', pointer(*where, 'cadence'), 'a reading is off the cadence grid that '
                             'starts at the first timestamp; ecofill never rounds or resamples')
    result = {'series': series, 'clock': {'timezone': source['clock']['timezone'], 'aware_rows': aware,
                                          'naive_rows': naive, 'utc_offsets_seconds': sorted(offsets)[:32],
                                          'grid_phase': 'first selected timestamp of each series'}}
    entry.normalized[(base, wanted)] = result
    return result


def file_locator(source):
    """Input-dependent source identity: one local file under any path, link or spelling."""
    from .config import lexical_locator
    if '://' in source['uri']:
        return lexical_locator(source)
    try:
        status = os.stat(source['uri'])
    except OSError:
        return lexical_locator(source)  # Missing inputs are reported unavailable later.
    if status.st_ino:
        return f'file-id:{status.st_dev}:{status.st_ino}'
    return 'file:' + os.path.normcase(os.path.realpath(source['uri']))


def check_grid_alignment(doc, inputs):
    """Every donor series shares its target series' cadence grid phase; nothing is shifted or resampled.

    Runs for simple, explicit operation/2 and replayed inputs before fields are reindexed or methods run.
    """
    def phase(name, series):
        source = doc['sources'][name]
        rows = normalize(source, inputs[name], name)['series'].get(None if source['identity_column'] is None else series)
        return None if not rows else rows[0][0].value % (source['cadence_seconds'] * 10**9)
    for key, dataset in doc['datasets'].items():
        if not dataset['enabled'] or dataset['source'] not in inputs:
            continue
        for series in dataset['instruments']:
            target = phase(dataset['source'], series)
            for spec in dataset['targets'].values():
                for route in spec['routes'].values():
                    for recipe in [route['recipe'], *route['fallbacks']] if route['enabled'] else []:
                        for member in [recipe, *recipe.get('members', [])]:
                            for ref in member['predictors']:
                                if ref['source'] not in inputs or target is None:
                                    continue
                                donor = phase(ref['source'], ref['instrument'])
                                if donor is not None and donor != target:
                                    raise diagnostic('donor_grid_mismatch', pointer('datasets', key),
                                                     'a predictor series uses a different grid phase than its target; '
                                                     'ecofill never shifts or resamples donors')


def sentinel(value, rules):
    for rule in rules:
        if (type(rule) is not dict and value == rule or type(rule) is dict and
                ('at_most' in rule and value <= rule['at_most'] or 'at_least' in rule and value >= rule['at_least'])):
            return True
    return False


def prepare_generic_field(doc, ref, inputs, limits):
    """User-supplied observations: finite cells are usable, never claimed independently quality assessed."""
    source, instrument, field = resolve_ref(doc, ref)
    metadata = source['instruments'][instrument]
    entry = inputs[ref['source']]
    where = ('datasets', ref['source'], 'columns', ref['field'])
    selected = normalize(source, entry, ref['source'])['series'].get(
        None if source['identity_column'] is None else instrument)
    if not selected:
        raise FileNotFoundError('configured series has no rows')
    index = pd.DatetimeIndex([item[0] for item in selected])
    cadence = pd.Timedelta(seconds=source['cadence_seconds'])
    count = int((index[-1] - index[0]) / cadence) + 1
    prepared_limits(count, limits)
    tokens, rules = set(field['missing_tokens']), field['sentinels']
    accepted = set(field.get('qc_accept', ()))
    def mask(row, key):
        value = row[field[key]]
        if entry.receipt.get('format') == 'parquet' and type(value) is bool:
            return value
        if value not in {'True', 'False'}:
            raise diagnostic('invalid_mask_value', pointer(*where), 'observation/rejection/structural mask cells must '
                             'be literal True or False')
        return value == 'True'
    raw, observed, qc, structural, names, reasons, keys = [], [], [], [], [], [], []
    for timestamp, position, row in selected:
        value = row[field['column']]
        why = []
        typed = entry.receipt.get('format') == 'parquet'
        if value is None or isinstance(value, str) and value in tokens:
            numeric = np.nan; why.append('original_missing')
        elif typed and (not isinstance(value, str) or NUMBER.fullmatch(value)):
            from .parquet import measurement
            numeric = measurement(value, pointer(*where))
            if not math.isfinite(numeric):
                numeric = np.nan; why.append('nonfinite_or_invalid')
        elif NUMBER.fullmatch(value):
            numeric = float(value)
            if not math.isfinite(numeric):
                numeric = np.nan; why.append('nonfinite_or_invalid')
        elif value.lower() in NONFINITE:
            numeric = np.nan; why.append('nonfinite_or_invalid')
        else:
            numeric = np.nan; why.append('invalid_numeric')
        finite = bool(np.isfinite(numeric))
        named = 'configured_sentinel' if finite and sentinel(numeric, rules) else None
        rejected = named is not None
        if named:
            why.append('named_sentinel:' + named)
        if 'rejected_column' in field and mask(row, 'rejected_column'):
            rejected = True; why.append('supplied_rejection')
        qc_value = row.get(field.get('qc_column'))
        if typed and qc_value is not None and not isinstance(qc_value, str):
            qc_value = str(qc_value)
        if 'qc_column' in field and qc_value not in accepted:
            rejected = True; why.append('qc_rejection')
        is_structural = mask(row, 'structural_column') if 'structural_column' in field else False
        supplied = mask(row, 'observed_column') if 'observed_column' in field else finite
        if supplied and not finite and 'observed_column' in field:
            raise diagnostic('observed_nonfinite', pointer(*where), 'the observed mask marks a missing or '
                             'nonnumeric cell')
        if 'route_column' in field and row[field['route_column']] != field['route_value']:
            supplied = False; why.append('outside_unit_regime')
        usable = bool(supplied and finite and not rejected and not is_structural)
        raw.append(numeric); observed.append(usable)
        qc.append(bool(rejected and finite)); structural.append(is_structural)
        names.append(named)
        reasons.append([timestamp.isoformat(), why or ['supplied_observation' if usable else 'unusable_raw']])
        key = row[source['key_column']] if source['key_column'] else f"{entry.receipt['sha256'][:12]}:{position}"
        keys.append([timestamp.isoformat(), key, position])
    name = ref['field']
    version = source['policy']['version']
    contract = {'identity': {'dataset': ref['source'], 'instrument': instrument, 'variable': name,
                             'unit_regime': field['unit_regime']},
                'input': {'cadence_seconds': source['cadence_seconds'], 'clock': 'UTC',
                          'variables': [{'name': name, 'unit': field['unit'], 'role': 'target', 'derived_from': []}],
                          'source_policy': {'qc': version, 'sentinel': version, 'structural': version}}}
    config = {'schema': 'ecofill.batch/1', **contract, 'mode': 'offline', 'support': 'original_only',
              'status': 'usable', 'evidence': [source['policy']['original_evidence']],
              'fitting': {'window_days': 180, 'min_rows': 168, 'max_rows': 4096, 'selection': 'even_time_coverage'},
              'routes': []}
    frame = pd.DataFrame({name: raw}, index=index)
    prepared = prepare_batch(frame, config=config, metadata=contract,
        observed=pd.DataFrame({name: observed}, index=index),
        qc_rejected=pd.DataFrame({name: qc}, index=index),
        structural=pd.DataFrame({name: structural}, index=index),
        sentinel_names=pd.DataFrame({name: names}, index=index))
    status = ('user-supplied values; finite cells usable; no independent quality assessment' if 'qc_column' not in field
              else 'user-supplied values; QC column codes outside the declared accepted set are rejected')
    return prepared, {'reference': ref, 'physical_instrument': instrument, 'keys': keys,
        'source_reasons': reasons, 'domains': support_domains(source, instrument, field),
        'validity': metadata['validity'], 'unit': field['unit'], 'unit_regime': field['unit_regime'],
        'lineage': field['lineage'], 'source_identity': entry.receipt['sha256'],
        'clock': normalize(source, entry, ref['source'])['clock'], 'observation_status': status,
        'row_keys': 'source column' if source['key_column'] else 'generated sha256-prefix:row-ordinal'}
