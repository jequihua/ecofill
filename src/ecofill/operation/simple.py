"""Concise ecofill.simple/1 input, its narrow sources/routes alias, and operation/2 resolution.

Expansion is deterministic and offline: explicit declarations are kept, optional metadata
gets the documented defaults, and input-dependent facts stay pending. plan/run read each
active source once and resolve those facts from the same bytes they execute.
"""
from datetime import datetime, timedelta, timezone
import json
import re
import zoneinfo

import pandas as pd

from .config import (CLASSES, ENSEMBLES, GENERIC, METHODS, POLICY, OperationConfig, checked, diagnostic, digest,
                     pointer)

DEFAULT_LIMITS = {'max_gaps': 1000, 'max_method_calls': 3000, 'max_rows_per_series': 250000, 'wall_seconds': 900,
                  'memory_bytes': 536870912, 'scratch_bytes': 33554432, 'output_bytes': 134217728}
MAX_SERIES = 64
EVIDENCE = 'user_supplied.configuration'
TOP_KEYS = {'schema', 'datasets', 'output', 'purpose', 'limits', 'models', 'allow_provisional', 'cloud'}
SHORTHAND = {'method', 'max_gap', 'predictors', 'parameters', 'backend', 'fallbacks'}
DATASET_KEYS = SHORTHAND | {'uri', 'cadence', 'targets', 'routes', 'enabled', 'sha256', 'generation', 'timestamp',
                            'timezone', 'key', 'group_by', 'select', 'columns', 'missing', 'interval', 'validity',
                            'boundaries', 'exclusions', 'fit_exclusions', 'bigquery', 'family', 'format'}
COLUMN_KEYS = {'column', 'unit', 'unit_regime', 'derived_from', 'qc', 'observed', 'rejected', 'structural',
               'regime', 'missing'}
TARGET_KEYS = SHORTHAND | {'routes', 'group'}
ROUTE_KEYS = SHORTHAND | {'enabled', 'reason', 'group'}
SAVED_KEYS = {'id', 'evidence_status', 'evidence', 'frozen_recipe', 'frozen_binding', 'members'}
RECIPE_KEYS = {'method', 'max_gap', 'predictors', 'parameters', 'backend', 'group'} | SAVED_KEYS
DURATION = re.compile(r'([1-9][0-9]{0,6})(s|min|h|d)')
INSTANT = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?(Z|[+-]\d{2}:\d{2})')
OFFSET = re.compile(r'([+-])(\d{2}):(\d{2})')


def fail(code, where, explanation):
    raise diagnostic(code, pointer(*where), explanation)


def zone(name):
    """UTC, a fixed +HH:MM/-HH:MM offset, or an IANA name. Never the local machine zone."""
    if name == 'UTC':
        return timezone.utc
    match = OFFSET.fullmatch(name) if isinstance(name, str) else None
    if match and int(match[2]) <= 14 and int(match[3]) <= 59:
        delta = timedelta(hours=int(match[2]), minutes=int(match[3]))
        return timezone(-delta if match[1] == '-' else delta)
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_+-]*(/[A-Za-z0-9_+-]+){0,2}', name):
        raise ValueError('timezone must be UTC, +HH:MM, -HH:MM or an IANA name')
    try:
        return zoneinfo.ZoneInfo(name)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError, OSError):
        raise ValueError('unknown IANA timezone name') from None


def utc_text(value):
    return pd.Timestamp(value).tz_convert('UTC').isoformat()


def obj(value, where, allowed, required=()):
    if not isinstance(value, dict):
        fail('invalid_type', where, 'an object is required here')
    unknown = set(value) - set(allowed)
    if unknown:
        fail('unknown_field', where, f'{len(unknown)} unknown field(s) here; allowed: ' + ', '.join(sorted(allowed)))
    missing = [k for k in required if k not in value]
    if missing:
        fail('missing_field', where, 'required field(s) missing: ' + ', '.join(missing))
    return value


def string(value, where, what):
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 1024:
        fail('invalid_text', where, f'{what} must be a nonempty string without surrounding spaces')
    return value


def strings(value, where, what, *, limit=64):
    if not isinstance(value, list) or len(value) > limit or len(set(map(str, value))) != len(value):
        fail('invalid_list', where, f'{what} must be a list of at most {limit} unique strings')
    return [string(item, [*where, i], what) for i, item in enumerate(value)]


def logical_name(name, where):
    """Names become output columns and portable identities."""
    if (not isinstance(name, str) or not name or name != name.strip() or len(name) > 128 or '::' in name
            or name.startswith('/') or '\\' in name or (len(name) > 1 and name[1] == ':')
            or any(ord(c) < 32 for c in name) or name == 'timestamp'):
        fail('invalid_column_name', where, 'column names must be 1-128 printable characters, without :: or '
             'backslashes, not start with /, and not be the reserved output name timestamp')
    return name


def seconds(value, where, what):
    if type(value) is int and value > 0:
        return value
    match = DURATION.fullmatch(value) if isinstance(value, str) else None
    if not match:
        fail('invalid_duration', where, f'{what} must be positive whole seconds or text such as 15min, 2h or 1d')
    return int(match[1]) * {'s': 1, 'min': 60, 'h': 3600, 'd': 86400}[match[2]]


def instant(value, where):
    if not isinstance(value, str) or not INSTANT.fullmatch(value):
        fail('invalid_instant', where, 'instants need ISO 8601 with an explicit offset, e.g. 2025-01-01T00:00:00Z')
    try:
        return utc_text(datetime.fromisoformat(value.replace('Z', '+00:00')))
    except ValueError:
        fail('invalid_instant', where, 'invalid calendar instant')


def interval_list(value, where):
    if not isinstance(value, list) or len(value) > 256:
        fail('invalid_intervals', where, 'a list of at most 256 [start, end) pairs is required')
    pairs = []
    for i, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            fail('invalid_intervals', [*where, i], 'each interval is a [start, end) pair')
        a, b = instant(pair[0], [*where, i, 0]), instant(pair[1], [*where, i, 1])
        if pd.Timestamp(a) >= pd.Timestamp(b) or pairs and pd.Timestamp(pairs[-1][1]) > pd.Timestamp(a):
            fail('invalid_intervals', [*where, i], 'intervals must be nonempty, ordered and nonoverlapping')
        pairs.append([a, b])
    return pairs


def from_alias(doc):
    """The engineer sources/routes/output spelling. Not a general permissive parser."""
    sources, routes = doc['sources'], doc['routes']
    obj(sources, ['sources'], sources if isinstance(sources, dict) else ())
    if not isinstance(routes, dict) or set(routes) != set(sources):
        fail('alias_route_mismatch', ['routes'], 'every source needs exactly one route with the same name')
    datasets = {}
    for name, source in sources.items():
        where = ['sources', name]
        obj(source, where, {'uri', 'cadence', 'variables', 'format'}, ('uri', 'cadence', 'variables'))
        variables = source['variables']
        if not isinstance(variables, dict) or not variables:
            fail('invalid_type', [*where, 'variables'], 'declare variables as an object of column names')
        columns, targets = {}, []
        for variable, spec in variables.items():
            obj(spec, [*where, 'variables', variable], {'unit', 'role'}, ('role',))
            if spec['role'] not in {'driver', 'response', 'target'}:
                fail('unsupported_role', [*where, 'variables', variable, 'role'],
                     'role must be driver (declared, not used by LIN), response or target')
            if spec['role'] != 'driver':
                targets.append(variable)
            columns[variable] = {'unit': spec['unit']} if 'unit' in spec else {}
        route = obj(routes[name], ['routes', name], {'enabled', 'method', 'mode', 'max_gap'}, ('method', 'max_gap'))
        if route.get('mode', 'offline') != 'offline':
            fail('unsupported_mode', ['routes', name, 'mode'], 'only offline reconstruction is supported')
        if route['method'] not in {'linear', 'LIN'}:
            fail('alias_method_unsupported', ['routes', name, 'method'], 'the sources/routes spelling supports '
                 'linear (LIN) only; multivariate methods need explicit predictors in the concise datasets format')
        if not targets and route.get('enabled', True):
            fail('no_target_selected', [*where, 'variables'], 'mark at least one variable with role response or target')
        datasets[name] = {'uri': source['uri'], 'cadence': source['cadence'], 'columns': columns,
                          'targets': targets, 'method': 'LIN', 'max_gap': route['max_gap']}
        if 'format' in source:
            datasets[name]['format'] = source['format']
        if 'enabled' in route:
            datasets[name]['enabled'] = route['enabled']
    return {'schema': 'ecofill.simple/1', 'datasets': datasets, 'output': doc['output']}


class Expansion:
    """One deterministic simple -> operation/2 expansion; facts=None leaves input facts pending."""

    def __init__(self, doc, input_format, facts=None):
        self.doc, self.input_format, self.facts = doc, input_format, facts
        self.defaults, self.pending, self.inactive = [], [], {'disabled': [], 'no_targets': [], 'unavailable': []}

    def opt(self, mapping, key, default, where):
        if key in mapping:
            return mapping[key]
        self.defaults.append([pointer(*where, key), default])
        return default

    def document(self):
        top = obj(self.doc, [], TOP_KEYS, ('datasets', 'output'))
        purpose = string(self.opt(top, 'purpose', 'unspecified', []), ['purpose'], 'purpose')
        limits = obj(top.get('limits', {}), ['limits'], DEFAULT_LIMITS)
        for key, value in DEFAULT_LIMITS.items():
            if key not in limits:
                self.defaults.append([pointer('limits', key), value])
        result = {'schema': 'ecofill.operation/2', 'purpose': purpose, 'output': top['output'],
                  'limits': DEFAULT_LIMITS | limits, 'allow_provisional': self.opt(top, 'allow_provisional', False, []),
                  'models': self.opt(top, 'models', {}, []), 'sources': {}, 'datasets': {}}
        if 'cloud' in top:
            result['cloud'] = top['cloud']
        datasets = top['datasets']
        if not isinstance(datasets, dict):
            fail('invalid_type', ['datasets'], 'datasets must be an object keyed by dataset name')
        self.parsed = {}
        for name, spec in datasets.items():
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
                fail('invalid_name', ['datasets'], 'dataset names use 1-64 letters, digits, _ or - and start with a '
                     'letter or digit')
            self.parsed[name] = self.dataset(name, spec)
        active = [name for name, parsed in self.parsed.items() if parsed['enabled'] and parsed['targets']]
        for name in active:
            self.plan_targets(self.parsed[name])
        if self.facts is not None:
            # Missing inputs make their datasets (and dependants) unavailable, never silently reduced.
            missing = {ref['dataset'] for name in active for ref in self.parsed[name]['references']} | set(active)
            missing -= set(self.facts)
            blocked = [name for name in active if name in missing or
                       any(ref['dataset'] in missing for ref in self.parsed[name]['references'])]
            self.inactive['unavailable'] = sorted(blocked)
            active = [name for name in active if name not in blocked]
        used = {}
        for name in active:
            for ref in self.parsed[name]['references']:
                used.setdefault(ref['dataset'], set()).update([ref['series']] if ref['series'] else [])
        for name, parsed in self.parsed.items():
            if not parsed['enabled']:
                self.inactive['disabled'].append(name)
            elif not parsed['targets'] and name not in used:
                self.inactive['no_targets'].append(name)
        for name, parsed in self.parsed.items():
            if name in active or name in used:
                self.emit(result, parsed, used.get(name, set()), name in active)
        status = 'pending' if self.facts is None else 'resolved'
        defaults = sorted({where: value for where, value in self.defaults}.items())
        result['resolution'] = {'status': status, 'input_format': self.input_format,
                                'authored_sha256': digest(self.doc), 'implementation': 'ecofill.simple-expansion/1',
                                'defaults': [list(item) for item in defaults], 'pending': sorted(set(self.pending)),
                                'inputs': {} if self.facts is None else {k: v['receipt'] for k, v in self.facts.items()},
                                'inactive': self.inactive}
        return result

    def dataset(self, name, spec):
        where = ['datasets', name]
        obj(spec, where, DATASET_KEYS, ('uri', 'cadence', 'targets'))
        parsed = {'name': name, 'where': where, 'spec': spec,
                  'enabled': self.opt(spec, 'enabled', True, where)}
        if type(parsed['enabled']) is not bool:
            fail('invalid_type', [*where, 'enabled'], 'enabled must be true or false')
        uri = string(spec['uri'], [*where, 'uri'], 'uri')
        cadence = seconds(spec['cadence'], [*where, 'cadence'], 'cadence')
        if cadence > 604800:
            fail('unsupported_cadence', [*where, 'cadence'], 'cadence must be at most 7 days')
        parsed.update(uri=uri, cadence=cadence, bigquery=uri.startswith('bq://'))
        from .parquet import input_format
        if 'format' in spec and not isinstance(spec['format'], str):
            fail('unsupported_format', [*where, 'format'], 'format must be csv, csv-gzip or parquet')
        parsed['format'] = input_format(uri, spec.get('format'), pointer(*where, 'format'))
        parsed['timestamp'] = string(self.opt(spec, 'timestamp', 'timestamp', where), [*where, 'timestamp'], 'timestamp')
        tz = self.opt(spec, 'timezone', None, where)
        if tz is not None:
            try:
                zone(tz)
            except ValueError as error:
                fail('invalid_timezone', [*where, 'timezone'], str(error))
        parsed['timezone'] = tz
        parsed['key'] = self.opt(spec, 'key', None, where)
        parsed['group_by'] = self.opt(spec, 'group_by', None, where)
        parsed['select'] = self.opt(spec, 'select', None, where)
        parsed['family'] = self.opt(spec, 'family', None, where)
        for k in ('key', 'group_by', 'family'):
            if parsed[k] is not None:
                string(parsed[k], [*where, k], k)
        if parsed['select'] is not None:
            if parsed['group_by'] is None:
                fail('select_requires_group_by', [*where, 'select'], 'select names values of the group_by column')
            parsed['select'] = strings(parsed['select'], [*where, 'select'], 'select', limit=MAX_SERIES)
            if not parsed['select']:
                fail('invalid_list', [*where, 'select'], 'select at least one series')
            for value in parsed['select']:
                series_name(value, [*where, 'select'])
        columns = self.opt(spec, 'columns', {}, where)
        if not isinstance(columns, dict):
            fail('invalid_type', [*where, 'columns'], 'columns must be an object keyed by column name')
        for column in columns:
            logical_name(column, [*where, 'columns'])
        parsed['columns'] = columns
        parsed['missing'] = self.missing(self.opt(spec, 'missing', {}, where), [*where, 'missing'])
        targets = spec['targets']
        if isinstance(targets, list):
            names, overrides = targets, {}
        elif isinstance(targets, dict):
            names, overrides = list(targets), targets
        else:
            fail('invalid_type', [*where, 'targets'], 'targets is a list of column names or an object of per-target settings')
        if len(set(map(str, names))) != len(names) or len(names) > 64:
            fail('invalid_list', [*where, 'targets'], 'targets must be at most 64 unique column names')
        for target in names:
            logical_name(target, [*where, 'targets'])
        parsed.update(targets=names, overrides=overrides, references=[])
        for k, default in (('validity', None), ('interval', None)):
            value = self.opt(spec, k, default, where)
            parsed[k] = None if value is None else interval_list([value], [*where, k])[0]
        boundaries = self.opt(spec, 'boundaries', [], where)
        if not isinstance(boundaries, list) or len(boundaries) > 256:
            fail('invalid_list', [*where, 'boundaries'], 'boundaries is a list of at most 256 instants')
        parsed['boundaries'] = sorted({instant(b, [*where, 'boundaries', i]) for i, b in enumerate(boundaries)},
                                      key=pd.Timestamp)
        parsed['exclusions'] = interval_list(self.opt(spec, 'exclusions', [], where), [*where, 'exclusions'])
        parsed['fit_exclusions'] = interval_list(self.opt(spec, 'fit_exclusions', [], where), [*where, 'fit_exclusions'])
        if parsed['bigquery']:
            for k in ('sha256', 'generation'):
                if k in spec:
                    fail('bigquery_source_pin', [*where, k], 'BigQuery input is pinned by cloud.snapshot_utc, not a file hash')
            bq = obj(spec.get('bigquery'), [*where, 'bigquery'], {'intervals'}, ('intervals',)) \
                if 'bigquery' in spec else None
            if bq is None or parsed['key'] is None or parsed['group_by'] is None or parsed['select'] is None:
                fail('bigquery_requires', where, 'BigQuery input needs explicit bigquery.intervals, key, group_by and '
                     'select; ecofill never infers a table extent or row order')
            parsed['intervals'] = interval_list(bq['intervals'], [*where, 'bigquery', 'intervals'])
            if not parsed['intervals']:
                fail('bigquery_requires', [*where, 'bigquery', 'intervals'], 'at least one finite extraction interval')
            if parsed['validity'] is None:
                if len(parsed['intervals']) != 1:
                    fail('bigquery_requires', [*where, 'validity'], 'declare validity when extracting several intervals')
                parsed['validity'] = parsed['intervals'][0]
                self.defaults.append([pointer(*where, 'validity'), parsed['validity']])
        elif 'bigquery' in spec:
            fail('contradictory_source', [*where, 'bigquery'], 'bigquery settings need a bq:// uri')
        for k in ('sha256', 'generation'):
            parsed[k] = spec.get(k)
        if parsed['generation'] is not None and not uri.startswith('gs://'):
            fail('generation_requires_gcs', [*where, 'generation'], 'generation pins a gs:// object')
        return parsed

    def missing(self, value, where):
        obj(value, where, {'tokens', 'sentinels'})
        tokens = strings(value.get('tokens', []), [*where, 'tokens'], 'missing tokens', limit=31) \
            if value.get('tokens', []) else []
        sentinels = value.get('sentinels', [])
        if not isinstance(sentinels, list) or len(sentinels) > 16:
            fail('invalid_list', [*where, 'sentinels'], 'sentinels is a list of at most 16 rules')
        from .config import sentinel_rule
        for i, rule in enumerate(sentinels):
            try:
                sentinel_rule(rule)
            except ValueError as error:
                fail('invalid_sentinel', [*where, 'sentinels', i], str(error))
        return {'tokens': sorted({'', *tokens}), 'sentinels': sentinels}

    def plan_targets(self, parsed):
        """Choose one complete spec per target; joint groups are declared once and explicit."""
        where, spec = parsed['where'], parsed['spec']
        dataset_level = SHORTHAND & set(spec)
        if dataset_level and 'routes' in spec:
            fail('conflicting_declarations', where, 'declare either method/max_gap shorthand or explicit routes, not both')
        joint, chosen = {}, {}
        for target in parsed['targets']:
            override = parsed['overrides'].get(target)
            twhere = [*where, 'targets', target]
            override = obj(override if override is not None else {}, twhere, TARGET_KEYS)
            if SHORTHAND & set(override) and 'routes' in override:
                fail('conflicting_declarations', twhere, 'declare either method/max_gap shorthand or explicit routes, not both')
            if override and 'method' not in override and 'routes' not in override:
                fail('incomplete_target_settings', twhere, 'a per-target setting needs its own method (and max_gap) or routes')
            if 'group' in override:
                group = strings(override['group'], [*twhere, 'group'], 'group', limit=4)
                if target not in group or set(group) - set(parsed['targets']) or not 2 <= len(group) <= 4:
                    fail('invalid_joint_group', [*twhere, 'group'], 'a joint group lists 2-4 targets of this dataset, '
                         'including this one')
                for member in group:
                    if member in joint and joint[member][1] != (override, group):
                        fail('conflicting_joint_group', twhere, 'a target belongs to at most one joint declaration')
                    joint[member] = (twhere, (override, group))
            elif override:
                chosen[target] = (twhere, override, [target])
        for target in parsed['targets']:
            twhere = [*where, 'targets', target]
            if target in joint:
                owner_where, (override, group) = joint[target]
                if target in chosen:
                    fail('conflicting_joint_group', twhere, 'a joint group member cannot also declare its own method')
                chosen[target] = (owner_where, override, group)
            elif target not in chosen:
                if not dataset_level and 'routes' not in spec:
                    fail('missing_method', twhere, 'no method for this target: add method and max_gap (or routes) to '
                         'the dataset or to this target')
                chosen[target] = (where, spec, [target])
        parsed['chosen'] = chosen
        # References are resolved per series during emission; collect them now for active sources.
        for target, (swhere, spec_, _) in chosen.items():
            for item, iwhere in self.recipe_specs(spec_, swhere):
                for i, ref in enumerate(item.get('predictors', [])):
                    parsed['references'].append(self.reference_target(parsed, ref, [*iwhere, 'predictors', i]))

    def recipe_specs(self, spec, where):
        if 'routes' in spec:
            routes = obj(spec['routes'], [*where, 'routes'], set(CLASSES))
            for cls, route in routes.items():
                obj(route, [*where, 'routes', cls], ROUTE_KEYS | SAVED_KEYS)
                if route.get('enabled', True) is not False:
                    yield route, [*where, 'routes', cls]
                    for i, item in enumerate(route.get('fallbacks', [])):
                        yield item, [*where, 'routes', cls, 'fallbacks', i]
        else:
            yield spec, where
            fallbacks = spec.get('fallbacks', [])
            if not isinstance(fallbacks, list) or len(fallbacks) > 10:
                fail('invalid_list', [*where, 'fallbacks'], 'fallbacks is a list of at most 10 recipes')
            for i, item in enumerate(fallbacks):
                yield item, [*where, 'fallbacks', i]

    def reference_target(self, parsed, value, where):
        if isinstance(value, str):
            logical_name(value, where)
            return {'dataset': parsed['name'], 'column': value, 'series': None, 'unit_regime': None, 'where': where}
        obj(value, where, {'dataset', 'column', 'series', 'unit_regime'}, ('column',))
        name = value.get('dataset', parsed['name'])
        if name not in self.parsed:
            fail('unknown_dataset_reference', [*where, 'dataset'], 'predictor names a dataset that is not declared')
        other = self.parsed[name]
        if not other['enabled']:
            fail('disabled_dataset_reference', [*where, 'dataset'], 'predictor names a disabled dataset; enable it '
                 'or remove the predictor')
        series = value.get('series')
        if series is not None:
            if other['group_by'] is None:
                fail('series_not_grouped', [*where, 'series'], 'series applies only to datasets with group_by')
            series_name(series, [*where, 'series'])
            if other['select'] is not None and series not in other['select']:
                fail('unknown_series', [*where, 'series'], 'series is not among the selected values of that dataset')
        elif other['group_by'] is not None and name != parsed['name']:
            fail('ambiguous_donor_series', where, 'that dataset has several series; name one with series')
        logical_name(value['column'], [*where, 'column'])
        if 'unit_regime' in value:
            string(value['unit_regime'], [*where, 'unit_regime'], 'unit_regime')
        return {'dataset': name, 'column': value['column'], 'series': series,
                'unit_regime': value.get('unit_regime'), 'where': where}

    def series_of(self, parsed):
        """Series names of one dataset: the dataset itself, an explicit selection, or discovered groups."""
        if parsed['group_by'] is None:
            return [parsed['name']]
        if parsed['select'] is not None:
            return parsed['select']
        if self.facts is None:
            self.pending.append(pointer('sources', parsed['name'], 'instruments'))
            return ['*']
        found = sorted(self.facts[parsed['name']]['series'])
        if len(found) > MAX_SERIES:
            fail('too_many_series', [*parsed['where'], 'group_by'], f'more than {MAX_SERIES} series; use select')
        return found

    def fields(self, parsed, series, extra):
        # derived_from parents are ancestry metadata: never extra measured columns or execution inputs.
        names = list(dict.fromkeys([*parsed['columns'], *parsed['targets'], *extra]))
        result = {}
        for name in names:
            cwhere = [*parsed['where'], 'columns', name]
            spec = obj(parsed['columns'].get(name, {}), cwhere, COLUMN_KEYS)
            lineage = lambda n: f"{parsed['name']}|{series}|{n}"
            field = {'column': string(self.opt(spec, 'column', name, cwhere), [*cwhere, 'column'], 'column'),
                     'unit': string(self.opt(spec, 'unit', 'unknown', cwhere), [*cwhere, 'unit'], 'unit'),
                     'unit_regime': string(self.opt(spec, 'unit_regime', 'native', cwhere), [*cwhere, 'unit_regime'],
                                           'unit_regime'),
                     'lineage': lineage(name),
                     'derived_from': [lineage(logical_name(p, [*cwhere, 'derived_from']))
                                      for p in strings(self.opt(spec, 'derived_from', [], cwhere),
                                                       [*cwhere, 'derived_from'], 'derived_from', limit=16)]}
            missing = self.missing(spec['missing'], [*cwhere, 'missing']) if 'missing' in spec else parsed['missing']
            field.update(missing_tokens=missing['tokens'], sentinels=missing['sentinels'])
            if 'qc' in spec:
                qc = obj(spec['qc'], [*cwhere, 'qc'], {'column', 'accept'}, ('column', 'accept'))
                field['qc_column'] = string(qc['column'], [*cwhere, 'qc', 'column'], 'QC column')
                field['qc_accept'] = strings(qc['accept'], [*cwhere, 'qc', 'accept'], 'accepted QC codes')
                if not field['qc_accept']:
                    fail('invalid_list', [*cwhere, 'qc', 'accept'], 'list the accepted QC codes explicitly')
            for key in ('observed', 'rejected', 'structural'):
                if key in spec:
                    field[key + '_column'] = string(spec[key], [*cwhere, key], key + ' mask column')
            if 'regime' in spec:
                regime = obj(spec['regime'], [*cwhere, 'regime'], {'column', 'value'}, ('column', 'value'))
                field['route_column'] = string(regime['column'], [*cwhere, 'regime', 'column'], 'regime column')
                field['route_value'] = string(regime['value'], [*cwhere, 'regime', 'value'], 'regime value')
            result[name] = field
        return result

    def extent(self, parsed, series):
        facts = None if self.facts is None else self.facts[parsed['name']]['series'].get(
            None if parsed['group_by'] is None else series)
        if self.facts is not None and facts is None:
            fail('unknown_series', [*parsed['where'], 'select'], 'a selected series has no rows in the input')
        if facts is None:
            return None
        end = pd.Timestamp(facts['last']) + pd.Timedelta(seconds=parsed['cadence'])
        return [facts['first'], end.isoformat()]

    def emit(self, result, parsed, referenced, has_work):
        name, where = parsed['name'], parsed['where']
        # Pending discovered grouping keeps only its '*' template; named donor series bind at plan time.
        series_list = self.series_of(parsed)
        if self.facts is not None:
            missing = referenced - set(series_list) - {None}
            if missing:
                fail('unknown_series', where, 'a predictor names a series that has no rows in the input')
        extra = [ref['column'] for p in self.parsed.values() for ref in p['references'] if ref['dataset'] == name]
        source = {'family': parsed['family'], 'uri': parsed['uri'], 'compressed': parsed['uri'].endswith('.gz'),
                  'identity_column': parsed['group_by'], 'timestamp_column': parsed['timestamp'],
                  'key_column': parsed['key'], 'cadence_seconds': parsed['cadence'],
                  'clock': {'timezone': parsed['timezone']}, 'preparation_profile': GENERIC, 'policy': dict(POLICY),
                  'instruments': {}}
        if parsed['format'] == 'parquet' or 'format' in parsed['spec']:
            source['format'] = parsed['format']
            source['compressed'] = parsed['format'] == 'csv-gzip'
        if parsed['bigquery']:
            source['bigquery'] = {'columns': None, 'intervals': parsed['intervals']}
        elif self.facts is None:
            for k in ('sha256', 'generation'):
                if parsed[k] is not None:
                    source[k] = parsed[k]
            self.pending.append(pointer('sources', name, 'sha256'))
            if parsed['uri'].startswith('gs://') and parsed['generation'] is None:
                self.pending.append(pointer('sources', name, 'generation'))
        else:
            receipt = self.facts[name]['receipt']
            source['sha256'] = receipt['sha256']
            if parsed['uri'].startswith('gs://'):
                source['generation'] = receipt['generation']
        for series in series_list:
            span = self.extent(parsed, series) if series != '*' else None
            validity = parsed['validity'] or span
            if validity is None:
                self.pending.append(pointer('sources', name, 'instruments', series, 'validity'))
            source['instruments'][series] = {'aliases': [], 'validity': validity, 'exclusions': parsed['exclusions'],
                                             'fields': self.fields(parsed, series, extra)}
        if parsed['bigquery']:
            from .preparation import required_columns
            source['bigquery']['columns'] = required_columns(source)
        for metadata in source['instruments'].values():
            for field in metadata['fields'].values():
                field['domains'] = None if metadata['validity'] is None else \
                    domains_of(metadata['validity'], parsed['boundaries'])
        result['sources'][name] = source
        if not has_work:
            return
        targets_series = self.series_of(parsed) if parsed['group_by'] is not None else series_list
        for series in targets_series:
            key = name if parsed['group_by'] is None or series == '*' else dataset_key(name, series)
            if key in result['datasets']:
                fail('series_name_collision', [*where, 'group_by'], 'two series map to the same dataset key')
            span = parsed['interval'] or (self.extent(parsed, series) if series != '*' else None)
            if span is None:
                self.pending.append(pointer('datasets', key, 'interval'))
            fields = source['instruments'][series]['fields']
            targets = {}
            for target in parsed['targets']:
                swhere, spec, group = parsed['chosen'][target]
                targets[target] = {'unit': fields[target]['unit'], 'unit_regime': fields[target]['unit_regime'],
                                   'routes': self.routes(parsed, series, spec, group, swhere)}
            result['datasets'][key] = {'enabled': True, 'source': name, 'instruments': [series], 'interval': span,
                                       'targets': targets, 'exclusions': [], 'fit_exclusions': parsed['fit_exclusions']}

    def routes(self, parsed, series, spec, group, where):
        if 'routes' in spec:
            routes = {}
            for cls, route in spec['routes'].items():
                rwhere = [*where, 'routes', cls]
                low, high = CLASSES[cls]
                enabled = route.get('enabled', True)
                if type(enabled) is not bool:
                    fail('invalid_type', [*rwhere, 'enabled'], 'enabled must be true or false')
                if not enabled:
                    if set(route) - {'enabled', 'reason'}:
                        fail('conflicting_declarations', rwhere, 'a disabled route declares no recipe')
                    routes[cls] = {'enabled': False, 'reason': string(route.get('reason', 'disabled in configuration'),
                                                                       [*rwhere, 'reason'], 'reason')}
                    continue
                if 'reason' in route:
                    fail('conflicting_declarations', [*rwhere, 'reason'], 'reason belongs only to a disabled route')
                cap = seconds(route['max_gap'], [*rwhere, 'max_gap'], 'max_gap') if 'max_gap' in route else high
                if not low < cap <= high:
                    fail('max_gap_outside_class', [*rwhere, 'max_gap'], f'{cls} covers gaps longer than {low} s and '
                         f'at most {high} s')
                routes[cls] = {'enabled': True, 'duration_seconds': {'greater_than': low, 'at_most': high},
                               'recipe': self.recipe(parsed, series, route, group, cap, rwhere),
                               'fallbacks': self.fallbacks(parsed, series, route, group, cap, rwhere)}
            return routes
        if 'method' not in spec:
            fail('missing_method', where, 'add a method')
        if 'max_gap' not in spec:
            fail('missing_max_gap', where, 'a single-method shorthand needs an explicit max_gap, e.g. 2h')
        cap = seconds(spec['max_gap'], [*where, 'max_gap'], 'max_gap')
        if cap > 604800:
            fail('max_gap_too_long', [*where, 'max_gap'], 'max_gap is at most 7d; longer gaps stay unresolved')
        if cap < parsed['cadence']:
            fail('max_gap_below_cadence', [*where, 'max_gap'], 'max_gap is shorter than one cadence step and would fill nothing')
        recipe = self.recipe(parsed, series, spec, group, cap, where)
        fallbacks = self.fallbacks(parsed, series, spec, group, cap, where)
        return {cls: {'enabled': True, 'duration_seconds': {'greater_than': low, 'at_most': high},
                      'recipe': recipe, 'fallbacks': fallbacks} if low < cap else
                {'enabled': False, 'reason': 'beyond configured max_gap'} for cls, (low, high) in CLASSES.items()}

    def fallbacks(self, parsed, series, spec, group, cap, where):
        items = spec.get('fallbacks', [])
        if not isinstance(items, list) or len(items) > 10:
            fail('invalid_list', [*where, 'fallbacks'], 'fallbacks is a list of at most 10 recipes')
        result = []
        for i, item in enumerate(items):
            iwhere = [*where, 'fallbacks', i]
            obj(item, iwhere, RECIPE_KEYS - {'group'}, ('method',))
            own = seconds(item['max_gap'], [*iwhere, 'max_gap'], 'max_gap') if 'max_gap' in item else cap
            if own > cap:
                fail('fallback_exceeds_cap', [*iwhere, 'max_gap'], 'a fallback cannot extend the route max_gap')
            result.append(self.recipe(parsed, series, item, group, own, iwhere))
        return result

    def reference(self, parsed, series, value, where):
        ref = self.reference_target(parsed, value, where)
        other = self.parsed[ref['dataset']]
        if ref['series'] is None:
            ref['series'] = series if ref['dataset'] == parsed['name'] else ref['dataset']
        spec = other['columns'].get(ref['column'], {})
        regime = spec.get('unit_regime', 'native') if isinstance(spec, dict) else 'native'
        if ref['unit_regime'] is not None and ref['unit_regime'] != regime:
            fail('unit_regime_mismatch', [*where, 'unit_regime'], 'predictor unit regime differs from its column declaration')
        return {'source': ref['dataset'], 'instrument': ref['series'], 'field': ref['column'], 'unit_regime': regime}

    def recipe(self, parsed, series, spec, group, cap, where):
        if 'method' not in spec:
            fail('missing_method', where, 'add a method')
        method = spec['method']
        if method not in METHODS:
            fail('unsupported_method', [*where, 'method'], 'unsupported method; use one of ' + ', '.join(sorted(METHODS)))
        joint = method in {'T-J', 'AVG-J'}
        if joint and len(group) < 2:
            fail('joint_group_required', where, 'joint methods need an explicit group of 2-4 targets')
        if not joint and len(group) > 1:
            fail('group_not_joint', where, 'a target group needs a joint method (T-J or AVG-J)')
        predictors = spec.get('predictors', [])
        if not isinstance(predictors, list) or len(predictors) > 16:
            fail('invalid_list', [*where, 'predictors'], 'predictors is a list of at most 16 references')
        refs = [self.reference(parsed, series, p, [*where, 'predictors', i]) for i, p in enumerate(predictors)]
        saved = SAVED_KEYS & set(spec)
        if saved:
            if not isinstance(spec.get('frozen_recipe'), dict) and 'frozen_recipe' in spec:
                fail('invalid_saved_recipe', [*where, 'frozen_recipe'], 'frozen_recipe must be the saved recipe object '
                     'itself; file paths and other values are not read')
            if 'frozen_recipe' not in spec or 'max_gap' in spec or 'evidence_status' not in spec:
                fail('saved_recipe_settings', where, 'a saved recipe supplies frozen_recipe and evidence_status and keeps '
                     'its frozen settings; remove max_gap or declare a user_configured method instead')
            recipe = {k: spec[k] for k in saved if k != 'members'}
            recipe.update(method=method, targets=list(group), predictors=refs,
                          parameters=spec.get('parameters', {}))
            if 'backend' in spec:
                recipe['backend'] = spec['backend']
            if 'members' in spec:
                members = spec['members']
                if not isinstance(members, list) or len(members) != 2:
                    fail('invalid_list', [*where, 'members'], 'a saved ensemble lists its two saved members')
                recipe['members'] = [self.recipe(parsed, series, obj(m, [*where, 'members', i], RECIPE_KEYS - {'group'}),
                                                 group, cap, [*where, 'members', i]) for i, m in enumerate(members)]
            return recipe
        parameters = spec.get('parameters', {})
        if not isinstance(parameters, dict) or 'max_gap_seconds' in parameters:
            fail('invalid_parameters', [*where, 'parameters'], 'parameters is an object; set max_gap instead of '
                 'max_gap_seconds')
        rid = f"{parsed['name']}.{method}.{digest([series, group, method, cap, refs, parameters, spec.get('backend')])[:12]}"
        base = {'targets': list(group), 'predictors': refs, 'evidence_status': 'user_configured', 'evidence': [EVIDENCE]}
        if method in ENSEMBLES:
            backends = spec.get('backend')
            if not isinstance(backends, dict) or set(backends) != {'P', 'T'}:
                fail('ensemble_backends', [*where, 'backend'], 'an AVG method names its models as backend: {P: ..., T: ...}')
            if set(parameters) - {'P', 'T'}:
                fail('invalid_parameters', [*where, 'parameters'], 'AVG parameters are {P: {...}, T: {...}}')
            members = []
            for member in ENSEMBLES[method]:
                own = parameters.get(member[0], {})
                if not isinstance(own, dict) or 'max_gap_seconds' in own:
                    fail('invalid_parameters', [*where, 'parameters', member[0]], 'member parameters are an object')
                members.append(base | {'id': rid + '.' + member, 'method': member, 'backend': backends[member[0]],
                                       'parameters': own | {'max_gap_seconds': cap}})
            return base | {'id': rid, 'method': method, 'parameters': {}, 'members': members}
        recipe = base | {'id': rid, 'method': method, 'parameters': parameters | {'max_gap_seconds': cap}}
        if method[0] in 'PT':
            if not isinstance(spec.get('backend'), str):
                fail('missing_backend', [*where, 'backend'], 'P and T methods name a declared model in backend')
            recipe['backend'] = spec['backend']
        elif 'backend' in spec:
            fail('backend_unsupported', [*where, 'backend'], 'only P, T and AVG methods use model backends')
        return recipe


def series_name(value, where):
    if (not isinstance(value, str) or not value or value != value.strip() or len(value) > 128 or '\\' in value
            or value.startswith('/') or (len(value) > 1 and value[1] == ':') or value == '*'
            or any(ord(c) < 32 for c in value)):
        fail('unsupported_series_value', where, 'series identities must be 1-128 printable characters without '
             'surrounding spaces or backslashes and not start with /')
    return value


def dataset_key(name, series):
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,62}', series):
        return f'{name}.{series}'
    return f'{name}.id-{digest(series)[:12]}'


def domains_of(validity, boundaries):
    cuts = [b for b in boundaries if pd.Timestamp(validity[0]) < pd.Timestamp(b) < pd.Timestamp(validity[1])]
    edges = [validity[0], *cuts, validity[1]]
    return [[a, b] for a, b in zip(edges, edges[1:])]


def expand(doc, kind, facts=None):
    if kind == 'alias':
        doc = from_alias(doc)
        input_format = 'sources/routes/output alias'
    else:
        input_format = 'ecofill.simple/1'
    return Expansion(doc, input_format, facts).document()


def resolve(config, reader, *, input_observer=None):
    """Read each active source once, derive input facts, and return the resolved operation/2 config."""
    from .config import input_format
    from .preparation import normalize
    from .runner import dependency_refs
    authored = json.loads(config.authored_json)
    pending = config.to_dict(); limits = pending['limits']
    names = sorted({ref['source'] for ref in dependency_refs(pending)})
    inputs, facts, unavailable = {}, {}, []
    for name in names:
        source = pending['sources'][name]
        try:
            inputs[name] = reader(source, limits)
        except FileNotFoundError:
            unavailable.append(name)
            continue
        if input_observer is not None:
            input_observer(name, inputs[name])
        normalized = normalize(source, inputs[name], name)
        receipt = {k: inputs[name].receipt[k] for k in ('sha256', 'bytes', 'rows') if k in inputs[name].receipt}
        if inputs[name].receipt.get('format') == 'parquet':
            receipt.update(format='parquet', decoding=inputs[name].receipt['decoding'])
        if 'generation' in inputs[name].receipt:
            receipt['generation'] = inputs[name].receipt['generation']
        receipt['clock'] = normalized['clock']
        receipt['series'] = {('' if k is None else k): {'first': v[0][0].isoformat(), 'last': v[-1][0].isoformat(),
                                                       'rows': len(v)} for k, v in normalized['series'].items()}
        facts[name] = {'receipt': receipt, 'series': {k: {'first': v[0][0].isoformat(), 'last': v[-1][0].isoformat()}
                                                     for k, v in normalized['series'].items()}}
    kind = input_format(authored)
    resolved = expand(authored, kind, facts)
    resolved['resolution']['expanded_identity'] = config.identity
    from .config import _load_operation2
    loaded = checked(_load_operation2, resolved)
    # Grid alignment and physical-alias checks run in prepare_operation for every generic configuration.
    return OperationConfig(loaded.canonical_json, config.authored_json), inputs, unavailable
