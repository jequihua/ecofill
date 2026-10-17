"""Offline operation/1 and /2 validation. Canonical JSON is the execution authority."""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re

import pandas as pd

CLASSES = {'short': (0, 7200), 'medium': (7200, 43200), 'long': (43200, 604800)}
METHODS = {'LIN', 'RF-X', 'RF-XL', 'P-U', 'P-X', 'T-U', 'T-X', 'T-J', 'AVG-U', 'AVG-X', 'AVG-J'}
ENSEMBLES = {'AVG-U': ('P-U', 'T-U'), 'AVG-X': ('P-X', 'T-X'), 'AVG-J': ('P-X', 'T-J')}
NCR = 'non-commercial research'
GENERIC = 'generic-v1'
SCHEMAS = ('ecofill.simple/1', 'ecofill.operation/2', 'ecofill.operation/1')
SAFE_TEXT = re.compile(r"[A-Za-z0-9 ,./:;'()_=+<>%-]{1,300}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def pointer(*parts):
    """JSON-pointer-like location; user keys are bounded and never echoed when unsafe."""
    names = []
    for part in parts:
        part = str(part) if type(part) is int else part
        if not isinstance(part, str) or not re.fullmatch(r'[A-Za-z0-9 _.()%+-]{1,64}', part):
            part = '<key>'
        names.append(part)
    return '/' + '/'.join(names)


def diagnostic(code, location, explanation):
    """A ValueError carrying a safe, structured, user-correctable explanation."""
    error = ValueError(explanation)
    error.diagnostic = {'code': code, 'location': location, 'explanation': explanation}
    return error


def attach(error, code, location):
    """Add a location to a literal validation message; never echo unsafe text."""
    if getattr(error, 'diagnostic', None) is None:
        message = str(error)
        explanation = message if type(error) is ValueError and SAFE_TEXT.fullmatch(message) else \
            'invalid value at this location'
        error.diagnostic = {'code': code, 'location': location, 'explanation': explanation}
    return error


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise diagnostic('duplicate_key', '/', 'duplicate mapping key; every key may appear only once')
        result[key] = value
    return result


def read_document(path):
    path = Path(path)
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        raise diagnostic('invalid_encoding', '/', 'configuration must be UTF-8 text') from None
    if path.suffix.lower() not in {'.yaml', '.yml'}:
        try:
            return json.loads(text, object_pairs_hook=unique, parse_constant=lambda x: (_ for _ in ()).throw(
                diagnostic('nonfinite_number', '/', 'NaN and infinity are not valid JSON numbers')))
        except json.JSONDecodeError as exc:
            raise diagnostic('invalid_json', '/', f'invalid JSON at line {exc.lineno} column {exc.colno}') from None
    import yaml  # Optional; never imported for JSON/core use.
    class Loader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            return unique([(self.construct_object(k, deep=True), self.construct_object(v, deep=True))
                           for k, v in node.value])
    # YAML 1.1 yes/no/on/off and implicit dates are ambiguous configuration.
    try:
        tokens = list(yaml.scan(text))
    except yaml.YAMLError:
        raise diagnostic('invalid_yaml', '/', 'invalid YAML configuration') from None
    for token in tokens:
        if isinstance(token, yaml.tokens.ScalarToken) and token.plain:
            if token.value.lower() in {'yes', 'no', 'on', 'off'}:
                raise diagnostic('ambiguous_yaml', '/', 'use true/false booleans and quoted timestamps')
    try:
        value = yaml.load(text, Loader=Loader)
        canonical(value)  # Reject implicit dates, cycles, NaN and infinity.
        return value
    except (TypeError, RecursionError, yaml.YAMLError) as exc:
        raise diagnostic('invalid_yaml', '/', 'invalid YAML configuration; quote timestamps and avoid '
                         'anchors, NaN and infinity') from exc


def keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValueError('missing or unknown fields: expected ' + ', '.join(sorted(required)))


def text(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError('explicit nonempty string required')


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError('unsafe identifier')


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'integer outside {low}..{high}')


def stamp(value):
    text(value)
    if not re.search(r'(Z|[+-]\d\d:\d\d)$', value):
        raise ValueError('timestamp needs explicit UTC offset')
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError('invalid timestamp') from None  # Parser text may echo the value.
    if pd.isna(result) or result.tz is None or result.utcoffset().total_seconds() != 0:
        raise ValueError('UTC timestamp required')
    return result


def interval(pair):
    if not isinstance(pair, list) or len(pair) != 2 or stamp(pair[0]) >= stamp(pair[1]):
        raise ValueError('nonempty half-open UTC interval required')


def intervals(values):
    if not isinstance(values, list):
        raise ValueError('interval list required')
    for pair in values:
        interval(pair)
    if any(stamp(a[1]) > stamp(b[0]) for a, b in zip(values, values[1:])):
        raise ValueError('intervals must be ordered and nonoverlapping')


def local_uri(value):
    text(value)
    if '://' in value:
        raise ValueError('operation/1 S01 supports local paths only')


def sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('SHA256 required')


def resolve_ref(doc, ref):
    keys(ref, ('source', 'instrument', 'field', 'unit_regime'))
    try:
        source = doc['sources'][ref['source']]
        name = ref['instrument']
        matches = [k for k, v in source['instruments'].items() if name == k or name in v['aliases']]
        if (not matches and source['preparation_profile'] == GENERIC and set(source['instruments']) == {'*'}
                and isinstance(name, str) and name):
            # Pending discovered series: every concrete series shares the template; plan binds it once.
            spec = source['instruments']['*']['fields'][ref['field']]
            if spec['unit_regime'] != ref['unit_regime']:
                raise KeyError('unit regime')
            return source, name, spec
        if len(matches) != 1:
            raise KeyError(name)
        spec = source['instruments'][matches[0]]['fields'][ref['field']]
        if spec['unit_regime'] != ref['unit_regime']:
            raise KeyError('unit regime')
        return source, matches[0], spec
    except (KeyError, TypeError) as exc:
        raise ValueError('dangling or incompatible source/instrument/field reference') from exc


def physical(doc, ref):
    source, instrument, field = resolve_ref(doc, ref)
    # Generic family metadata is descriptive; its source name scopes physical identity.
    family = ref['source'] if source['preparation_profile'] == GENERIC else source['family']
    return family, instrument, field.get('physical_field', field['column'])


def donor_identity(doc, ref):
    _, instrument, field = resolve_ref(doc, ref)
    return dict(source=ref['source'], instrument=instrument,
                physical_field=field.get('physical_field', field['column']), unit_regime=field['unit_regime'])


def validate_frozen_binding(doc, recipe, frozen):
    binding = recipe.get('frozen_binding')
    keys(binding, ('source_identity', 'panel_evidence', 'predictors'))
    if binding['source_identity'] != frozen['source_identity']:
        raise ValueError('saved binding source identity differs')
    text(binding['panel_evidence'])
    panel = binding['predictors']
    labels = frozen['predictors']
    if not isinstance(panel, dict) or set(panel) != set(labels) or len(recipe['predictors']) != len(labels):
        raise ValueError('saved binding must contain the exact frozen predictor labels and count')
    # The map is saved panel evidence, independent of the operator's ordered list.
    # JSON map order is irrelevant; shortlist's ordered labels own channel order.
    for label, configured in zip(labels, recipe['predictors']):
        entry = panel[label]
        keys(entry, ('ref', 'identity'))
        keys(entry['identity'], ('source', 'instrument', 'physical_field', 'unit_regime'))
        for value in entry['identity'].values():
            text(value)
        expected = donor_identity(doc, entry['ref'])
        if entry['identity'] != expected or donor_identity(doc, configured) != expected:
            raise ValueError('ordered donor differs from saved canonical panel binding')
        if resolve_ref(doc, configured)[2]['unit'] != frozen['units'][label]:
            raise ValueError('saved predictor units differ')
        # Persist canonical instrument aliases in the resolved evidence.
        entry['ref']['instrument'] = expected['instrument']
        configured['instrument'] = expected['instrument']


def ancestors(doc, lineage):
    graph = {}
    for source in doc['sources'].values():
        for metadata in source['instruments'].values():
            for field in metadata['fields'].values():
                graph.setdefault(field['lineage'], set()).update(field['derived_from'])
    def visit(name, stack):
        if name in stack:
            raise ValueError('cyclic source lineage')
        result = {name}
        for parent in graph.get(name, ()):
            result |= visit(parent, stack | {name})
        return result
    return visit(lineage, set())


def parameters(method, values):
    defaults = {'max_gap_seconds': 604800}
    if method.startswith(('RF', 'P')):
        defaults.update(window_days=180, min_rows=168, max_rows=4096 if method.startswith('RF') else 1024, seed=0)
    if method.startswith('RF'):
        defaults.update(n_estimators=128, max_depth=12, min_samples_leaf=5, n_jobs=1)
    if method.startswith('T'):
        defaults.update(context_length=128, min_support=32)
    keys(values, (), defaults)
    result = defaults | values
    bounds = {'max_gap_seconds': (1, 604800), 'window_days': (1, 180), 'min_rows': (168, result.get('max_rows', 4096)),
              'max_rows': (168, 4096 if method.startswith('RF') else 1024), 'seed': (0, 2**31-1),
              'n_estimators': (1, 512), 'max_depth': (1, 32), 'min_samples_leaf': (1, 128),
              'n_jobs': (1, 1), 'context_length': (32, 128), 'min_support': (32, result.get('context_length', 128))}
    for key, value in result.items():
        integer(value, *bounds[key])
    return result


def validate_recipe(doc, recipe, target_ref, targets, *, joint_member=False):
    keys(recipe, ('id', 'method', 'targets', 'predictors', 'parameters', 'evidence_status', 'evidence'),
         ('backend', 'members', 'frozen_recipe', 'frozen_binding'))
    if 'frozen_recipe' in recipe and not isinstance(recipe['frozen_recipe'], dict):
        # The shortlist loader reads non-object values as file paths; validation never reads other files.
        raise ValueError('frozen_recipe must be an inline saved recipe object')
    identifier(recipe['id'])
    method = recipe['method']
    if method not in METHODS:
        raise ValueError('unsupported method')
    if recipe['evidence_status'] not in {'user_configured', 'confirmed_archive', 'development_selected'}:
        raise ValueError('unknown evidence status')
    if not isinstance(recipe['evidence'], list) or not recipe['evidence']:
        raise ValueError('evidence references required')
    for evidence in recipe['evidence']:
        text(evidence)
    group = recipe['targets']
    if not isinstance(group, list) or len(set(group)) != len(group) or target_ref['field'] not in group or set(group) - set(targets):
        raise ValueError('recipe target group must name configured targets')
    grouped = method in {'T-J', 'AVG-J'} or joint_member and method == 'P-X'
    if (grouped and not 2 <= len(group) <= 4) or (not grouped and len(group) != 1):
        raise ValueError('invalid target group size')
    refs = recipe['predictors']
    if not isinstance(refs, list) or len(refs) > 16:
        raise ValueError('bounded explicit donors required')
    target_lineages = {resolve_ref(doc, target_ref | {'field': n, 'unit_regime': targets[n]['unit_regime']})[2]['lineage'] for n in group}
    target_physical = {physical(doc, target_ref | {'field': n, 'unit_regime': targets[n]['unit_regime']}) for n in group}
    seen = set()
    cadence = doc['sources'][target_ref['source']]['cadence_seconds']
    for ref in refs:
        source, instrument, field = resolve_ref(doc, ref)
        ref['instrument'] = instrument
        identity = physical(doc, ref)
        if identity in seen or identity in target_physical or ancestors(doc, field['lineage']) & target_lineages:
            raise ValueError('duplicate, alias or descendant donor')
        if source['cadence_seconds'] != cadence:
            raise ValueError('different cadence donor needs a separately qualified aggregation rule')
        seen.add(identity)
    if method in {'LIN', 'P-U', 'T-U', 'AVG-U'} and refs:
        raise ValueError('univariate recipes read no donors')
    if method.startswith('RF') and len(refs) < 2 or method in {'P-X', 'T-X', 'AVG-X'} and not refs:
        raise ValueError('missing measured donors')
    if method in ENSEMBLES:
        keys(recipe['parameters'], ())
        members = recipe.get('members')
        if not isinstance(members, list) or any(not isinstance(m, dict) or 'method' not in m for m in members) \
                or tuple(m['method'] for m in members) != ENSEMBLES[method]:
            raise ValueError('ensemble requires exact ordered equal-weight constituents')
        for member in members:
            validate_recipe(doc, member, target_ref, targets, joint_member=method == 'AVG-J')
        if any([donor_identity(doc, r) for r in m['predictors']] != [donor_identity(doc, r) for r in refs]
               for m in members):
            raise ValueError('ensemble donor panels differ')
        if any(m['targets'] != group for m in members):
            raise ValueError('ensemble target groups differ')
        if any(m['evidence_status'] != recipe['evidence_status'] for m in members):
            raise ValueError('ensemble evidence classes differ')
    else:
        if 'members' in recipe:
            raise ValueError('members only belong to ensembles')
        recipe['parameters'] = parameters(method, recipe['parameters'])
        if method.startswith(('P', 'T')):
            if recipe.get('backend') not in doc['models']:
                raise ValueError('explicit model binding required')
            model = doc['models'][recipe['backend']]
            if model['kind'] != ('tabpfn35' if method.startswith('P') else 'timesfm3'):
                raise ValueError('backend family mismatch')
            if NCR not in {doc['purpose'], model.get('purpose')}:
                raise ValueError('restricted model weights require purpose non-commercial research')
        elif 'backend' in recipe:
            raise ValueError('backend unsupported for this method')
    if recipe['evidence_status'] == 'development_selected' and not doc['allow_provisional']:
        raise ValueError('research_selected requires allow_provisional')
    if 'frozen_recipe' in recipe:
        if recipe['evidence_status'] == 'user_configured':
            raise ValueError('user_configured must remove frozen qualification metadata')
        if method in ENSEMBLES:
            if 'frozen_binding' in recipe or any('frozen_recipe' not in m for m in recipe['members']):
                raise ValueError('saved ensemble requires frozen constituent bindings')
            expected = {'schema': 'ecofill.operation.saved_ensemble/1',
                        'members': [m['frozen_recipe'] for m in recipe['members']], 'weights': [.5, .5]}
            if recipe['frozen_recipe'] != expected:
                raise ValueError('saved ensemble identity drift')
            return
        from ..shortlist import load_shortlist_config, settings
        frozen = load_shortlist_config(recipe['frozen_recipe']).to_dict()
        if frozen['method'] != method:
            raise ValueError('frozen method differs')
        if frozen['status'] != {'confirmed_archive': 'usable', 'development_selected': 'provisional'}[recipe['evidence_status']]:
            raise ValueError('frozen status contradicts evidence class or is disabled')
        if frozen['targets'] != group or frozen['cadence_seconds'] != cadence:
            raise ValueError('frozen targets or cadence differ')
        if any(frozen['units'][n] != targets[n]['unit'] for n in group):
            raise ValueError('frozen target units differ')
        validate_frozen_binding(doc, recipe, frozen)
        expected = parameters(method, {})
        if recipe['parameters'] != expected:
            raise ValueError('overrides must be user_configured with a new identity')
        if frozen['settings'] != settings(method):
            raise ValueError('frozen settings drift')
    elif recipe['evidence_status'] != 'user_configured':
        raise ValueError('saved evidence recipes must retain their exact frozen configuration')
    elif 'frozen_binding' in recipe or any(e.startswith(('M013.configuration:', 'M016.method_order:')) for e in recipe['evidence']):
        raise ValueError('user_configured must remove saved qualification claims')


@dataclass(frozen=True)
class OperationConfig:
    canonical_json: str
    authored_json: str | None = None  # Simple input as written; None for explicit operation configs.

    @property
    def identity(self):
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self.canonical_json)

    @property
    def pending(self):
        """Simple input whose input-dependent facts are resolved only by plan/run."""
        return (self.to_dict().get('resolution') or {}).get('status') == 'pending'


SIMPLE_KEYS = {'schema', 'datasets', 'output', 'purpose', 'limits', 'models', 'allow_provisional', 'cloud'}


def input_format(doc):
    """Dispatch on explicit schema; only the documented shapes may omit it."""
    if not isinstance(doc, dict):
        raise diagnostic('invalid_document', '/', 'configuration must be a JSON or YAML object')
    if 'schema' in doc:
        if doc['schema'] not in SCHEMAS:
            raise diagnostic('unknown_schema', '/schema', 'unknown schema; expected ecofill.simple/1 (concise), '
                             'ecofill.operation/2 (resolved) or ecofill.operation/1 (strict legacy)')
        return doc['schema']
    if set(doc) == {'sources', 'routes', 'output'}:
        return 'alias'
    if {'datasets', 'output'} <= set(doc) <= SIMPLE_KEYS:
        return 'ecofill.simple/1'
    raise diagnostic('unrecognized_format', '/', f'configuration format not recognized ({len(set(doc) - SIMPLE_KEYS)} '
                     'unexpected top-level keys): use the concise format with datasets and output (optional schema '
                     'ecofill.simple/1), the sources/routes/output alias, or an explicit schema ecofill.operation/1 '
                     'or ecofill.operation/2')


def load_config(value):
    authored = None
    if isinstance(value, OperationConfig):
        if value.pending:
            value = json.loads(value.authored_json)
        else:
            authored, value = value.authored_json, value.to_dict()
    elif isinstance(value, (str, Path)):
        value = read_document(value)
    try:
        doc = json.loads(canonical(value))
    except (TypeError, RecursionError, ValueError) as exc:
        raise diagnostic('invalid_document', '/', 'configuration must be finite acyclic JSON') from exc
    kind = input_format(doc)
    if kind in {'ecofill.operation/1', 'ecofill.operation/2'}:
        # An explicit legacy/resolved schema is never reinterpreted as simple input.
        config = checked(_load_operation1 if kind == 'ecofill.operation/1' else _load_operation2, doc)
        return config if authored is None else OperationConfig(config.canonical_json, authored)
    from .simple import expand
    expanded = expand(doc, kind)
    return OperationConfig(checked(_load_operation2, expanded, pending=True).canonical_json, canonical(doc))


def checked(validate, doc, **options):
    trail = []
    try:
        return validate(doc, trail, **options)
    except ValueError as error:
        raise attach(error, 'invalid_configuration', pointer(*trail))
    except (KeyError, TypeError, AttributeError, IndexError):
        raise diagnostic('invalid_configuration', pointer(*trail),
                         'malformed value at this location; check its type and required fields') from None


def _validate_envelope(doc, trail):
    trail[:] = ['allow_provisional']
    if type(doc.setdefault('allow_provisional', False)) is not bool:
        raise ValueError('allow_provisional must be boolean')
    trail[:] = ['output']
    generic = doc['schema'] == 'ecofill.operation/2'
    keys(doc['output'], ('uri',), ('format', 'publication') if generic else ())
    if generic:
        output = doc['output']
        if output.get('format', 'csv-gzip') not in {'csv-gzip', 'parquet'}:
            raise diagnostic('unsupported_format', '/output/format', 'output format must be csv-gzip or parquet')
        if output.get('publication', 'strict') not in {'strict', 'single_writer'}:
            raise diagnostic('unsupported_publication', '/output/publication', 'publication must be strict or single_writer')
        if '://' in output['uri'] and (output.get('format') == 'parquet' or output.get('publication') == 'single_writer'):
            raise diagnostic('local_output_required', '/output', 'Parquet and single_writer require a local output path; engineers own remote materialization')
    text(doc['output']['uri'])
    if doc['output']['uri'].startswith('gs://'):
        from .cloud_config import gcs_uri
        gcs_uri(doc['output']['uri'])
    else:
        local_uri(doc['output']['uri'])
    if 'cloud' in doc:
        trail[:] = ['cloud']
        from .cloud_config import validate_cloud
        validate_cloud(doc['cloud'])
    trail[:] = ['limits']
    limits = doc['limits']
    keys(limits, ('max_gaps', 'max_method_calls', 'max_rows_per_series', 'wall_seconds', 'memory_bytes', 'scratch_bytes', 'output_bytes'))
    for key, value in limits.items():
        trail[:] = ['limits', key]
        integer(value, 0 if key in {'max_gaps', 'max_method_calls'} else 1, 2**53-1)
    for block in ('sources', 'models', 'datasets'):
        trail[:] = [block]
        if not isinstance(doc[block], dict):
            raise ValueError('keyed configuration block required')
        for name in doc[block]:
            trail[:] = [block, name]
            identifier(name)


def _load_operation1(doc, trail):
    keys(doc, ('schema', 'purpose', 'output', 'limits', 'sources', 'datasets', 'models'), ('allow_provisional', 'cloud'))
    if doc['schema'] != 'ecofill.operation/1' or doc['purpose'] != 'non-commercial research':
        raise ValueError('operation/1 requires non-commercial research')
    _validate_envelope(doc, trail)
    for source_name, source in doc['sources'].items():
        trail[:] = ['sources', source_name]
        required = ('family', 'uri', 'identity_column', 'timestamp_column', 'key_column', 'cadence_seconds',
                    'clock', 'preparation_profile', 'policy', 'instruments')
        keys(source, required, ('sha256', 'generation', 'bigquery', 'provenance_columns'))
        if 'provenance_columns' in source:
            columns = source['provenance_columns']
            if (not isinstance(columns, list) or len(columns) > 16 or any(not isinstance(c, str) for c in columns)
                    or len(set(columns)) != len(columns)):
                raise ValueError('bounded explicit provenance columns required')
            for column in columns:
                identifier(column)
        from .cloud_config import validate_source
        if source['family'] not in {'flux', 'biomet', 'ctd', 'weather'} or source['clock'] != 'UTC':
            raise ValueError('unsupported source family or clock')
        if source['cadence_seconds'] != (1800 if source['family'] in {'flux', 'biomet'} else 3600) or type(source['cadence_seconds']) is not int:
            raise ValueError('native family cadence required')
        for k in ('identity_column', 'timestamp_column', 'key_column'):
            identifier(source[k])
        if source['preparation_profile'] not in {'manglaria-original-v1', 'qualified-original-v1'}:
            raise ValueError('unknown preparation profile')
        policy = source['policy']
        keys(policy, ('version', 'original_evidence', 'missing_strings', 'positive_9999', 'negative_sentinel_fields'))
        text(policy['version']); text(policy['original_evidence'])
        if type(policy['positive_9999']) is not bool:
            raise ValueError('sentinel policy must be explicit')
        for k in ('missing_strings', 'negative_sentinel_fields'):
            if not isinstance(policy[k], list) or any(not isinstance(x, str) for x in policy[k]):
                raise ValueError('explicit string policy lists required')
        if set(policy['negative_sentinel_fields']) - {'L', 'LE_strg', 'x_70_pct'}:
            raise ValueError('unqualified negative sentinel field')
        if not isinstance(source['instruments'], dict):
            raise ValueError('explicit instruments required')
        aliases = set()
        for name, metadata in source['instruments'].items():
            trail[:] = ['sources', source_name, 'instruments', name]
            text(name)
            keys(metadata, ('aliases', 'validity', 'exclusions', 'fields'))
            interval(metadata['validity']); intervals(metadata['exclusions'])
            if not isinstance(metadata['aliases'], list):
                raise ValueError('alias list required')
            for alias in [name, *metadata['aliases']]:
                text(alias)
                if alias in aliases:
                    raise ValueError('ambiguous instrument alias')
                aliases.add(alias)
            if not isinstance(metadata['fields'], dict) or not metadata['fields']:
                raise ValueError('typed fields required')
            instrument = name
            for name, field in metadata['fields'].items():
                trail[:] = ['sources', source_name, 'instruments', instrument, 'fields', name]
                text(name)
                keys(field, ('column', 'unit', 'unit_regime', 'lineage', 'derived_from', 'domains'),
                     ('qc_column', 'observed_column', 'rejected_column', 'structural_column', 'route_column', 'route_value', 'physical_field'))
                for k in ('column', 'unit', 'unit_regime', 'lineage'):
                    text(field[k])
                for k in ('qc_column', 'observed_column', 'rejected_column', 'structural_column', 'route_column'):
                    if k in field:
                        text(field[k])
                if source['family'] == 'biomet' and re.match(r'^(TA|TS|TC)_', name) and field['unit'] != 'C':
                    raise ValueError('owner-qualified biomet temperature is already Celsius')
                if not isinstance(field['derived_from'], list) or any(not isinstance(x, str) for x in field['derived_from']):
                    raise ValueError('explicit lineage list required')
                intervals(field['domains'])
                for a, b in field['domains']:
                    if stamp(a) < stamp(metadata['validity'][0]) or stamp(b) > stamp(metadata['validity'][1]):
                        raise ValueError('domain exceeds metadata validity')
                if source['family'] == 'flux' and source['preparation_profile'] == 'manglaria-original-v1' and field['column'] in {'H', 'LE', 'co2_flux', 'h2o_flux'}:
                    if field.get('qc_column') != 'qc_' + field['column']:
                        raise ValueError('matching flux QC field required')
                if source['preparation_profile'] == 'qualified-original-v1' and 'observed_column' not in field:
                    raise ValueError('qualified inputs require explicit observation masks')
                if source['preparation_profile'] == 'qualified-original-v1':
                    text(field.get('physical_field'))
                elif 'physical_field' in field and field['physical_field'] != field['column']:
                    raise ValueError('raw physical field must match the measurement column')
                if ('route_column' in field) != ('route_value' in field):
                    raise ValueError('weather retrieval regime requires column and value')
        trail[:] = ['sources', source_name]
        validate_source(source, doc.get('cloud'))
    _validate_models(doc, trail)
    _validate_work(doc, trail)
    return _finish(doc)


def _validate_models(doc, trail, optional=()):
    for name, model in doc['models'].items():
        trail[:] = ['models', name]
        keys(model, ('kind', 'artifacts', 'license', 'revision'), optional)
        if model['kind'] not in {'tabpfn35', 'timesfm3'}:
            raise ValueError('unknown backend kind')
        text(model['license']); text(model['revision'])
        if 'purpose' in model and model['purpose'] != NCR:
            raise ValueError('a declared model purpose must be non-commercial research')
        if not isinstance(model['artifacts'], list):
            raise ValueError('artifact list required')
        for artifact in model['artifacts']:
            keys(artifact, ('uri', 'sha256'), ('generation',))
            from .cloud_config import validate_artifact
            validate_artifact(artifact); sha(artifact['sha256'])


def _validate_work(doc, trail, *, pending=False):
    work = []
    for dataset_name, dataset in doc['datasets'].items():
        trail[:] = ['datasets', dataset_name]
        keys(dataset, ('enabled', 'source', 'instruments', 'interval', 'targets', 'exclusions', 'fit_exclusions'))
        if type(dataset['enabled']) is not bool:
            raise ValueError('enabled must be boolean')
        open_interval = pending and dataset['interval'] is None  # Simple input: the series extent.
        if not open_interval:
            interval(dataset['interval'])
        intervals(dataset['exclusions']); intervals(dataset['fit_exclusions'])
        if not isinstance(dataset['targets'], dict) or not isinstance(dataset['instruments'], list) or not dataset['instruments']:
            raise ValueError('explicit instruments and targets required')
        if not dataset['enabled']:
            continue
        for instrument in dataset['instruments']:
            for name, target in dataset['targets'].items():
                trail[:] = ['datasets', dataset_name, 'targets', name]
                keys(target, ('unit', 'unit_regime', 'routes'))
                ref = dict(source=dataset['source'], instrument=instrument, field=name, unit_regime=target['unit_regime'])
                _, _, field = resolve_ref(doc, ref)
                if target['unit'] != field['unit'] or field['derived_from']:
                    raise ValueError('target unit mismatch or derived target')
                identity = physical(doc, ref)
                a, b = (None, None) if open_interval else map(stamp, dataset['interval'])
                if any(i == identity and (a is None or x is None or a < y and x < b) for i, x, y in work):
                    raise ValueError('duplicate physical target work')
                work.append((identity, a, b))
                if not isinstance(target['routes'], dict) or set(target['routes']) - set(CLASSES):
                    raise ValueError('unknown duration class')
                for cls, route in target['routes'].items():
                    trail[:] = ['datasets', dataset_name, 'targets', name, 'routes', cls]
                    keys(route, ('enabled',), ('duration_seconds', 'recipe', 'fallbacks', 'reason'))
                    if type(route['enabled']) is not bool:
                        raise ValueError('enabled must be boolean')
                    if not route['enabled']:
                        text(route.get('reason'))
                        continue
                    keys(route, ('enabled', 'duration_seconds', 'recipe', 'fallbacks'))
                    keys(route['duration_seconds'], ('greater_than', 'at_most'))
                    integer(route['duration_seconds']['greater_than'], 0, 604799)
                    integer(route['duration_seconds']['at_most'], 1, 604800)
                    if list(route['duration_seconds'].values()) != list(CLASSES[cls]):
                        # Canonical key order may differ; compare explicitly.
                        if (route['duration_seconds']['greater_than'], route['duration_seconds']['at_most']) != CLASSES[cls]:
                            raise ValueError('duration class boundaries differ')
                    if not isinstance(route['fallbacks'], list) or len(route['fallbacks']) > 10:
                        raise ValueError('bounded explicit fallback list required')
                    for recipe in [route['recipe'], *route['fallbacks']]:
                        validate_recipe(doc, recipe, ref, dataset['targets'])


def _finish(doc):
    if (doc['output']['uri'].startswith('gs://') or any(s['uri'].startswith(('gs://', 'bq://')) for s in doc['sources'].values())
            or any(a['uri'].startswith('gs://') for m in doc['models'].values() for a in m['artifacts'])) and 'cloud' not in doc:
        raise ValueError('remote transport requires explicit cloud bounds')
    resolved = canonical(doc)
    if doc['limits']['output_bytes'] < 2 * len(resolved.encode()) + 2 * 65536:
        raise ValueError('output budget cannot retain request/config and terminal control records')
    return OperationConfig(resolved)


GENERIC_FIELD = ('column', 'unit', 'unit_regime', 'lineage', 'derived_from', 'domains', 'missing_tokens', 'sentinels')
GENERIC_OPTIONAL = ('qc_column', 'qc_accept', 'observed_column', 'rejected_column', 'structural_column',
                    'route_column', 'route_value')
POLICY = {'version': 'ecofill.generic-preparation/1', 'origin': 'user_supplied', 'original_evidence': 'user_supplied'}


def sentinel_rule(rule):
    if type(rule) in (int, float) and math.isfinite(rule):
        return
    if isinstance(rule, dict) and len(rule) == 1 and set(rule) <= {'at_most', 'at_least'}:
        value = next(iter(rule.values()))
        if type(value) in (int, float) and math.isfinite(value):
            return
    raise ValueError('sentinel rules are finite numbers or {at_most: n} / {at_least: n}')


def _load_operation2(doc, trail, *, pending=False):
    keys(doc, ('schema', 'purpose', 'output', 'limits', 'sources', 'datasets', 'models'),
         ('allow_provisional', 'cloud', 'resolution'))
    trail[:] = ['purpose']
    text(doc['purpose'])
    trail[:] = ['resolution']
    resolution = doc.get('resolution', {'status': 'resolved'})
    keys(resolution, ('status',), ('input_format', 'authored_sha256', 'implementation', 'defaults', 'pending', 'inputs',
                                   'inactive', 'expanded_identity'))
    if resolution['status'] not in {'pending', 'resolved'}:
        raise ValueError('resolution status must be pending or resolved')
    if (resolution['status'] == 'pending') != pending:
        raise diagnostic('pending_input_facts', '/resolution', 'this operation/2 document has pending input facts; '
                         'plan or run its simple configuration to resolve them from the input files')
    _validate_envelope(doc, trail)
    for name, source in doc['sources'].items():
        trail[:] = ['sources', name]
        _validate_generic_source(name, source, doc.get('cloud'), trail, pending)
    _validate_models(doc, trail, ('purpose',))
    _validate_work(doc, trail, pending=pending)
    physical_conflicts(doc, lexical_locator)
    return _finish(doc)


SEMANTICS = ('unit', 'unit_regime', 'missing_tokens', 'sentinels', 'qc_column', 'qc_accept', 'observed_column',
             'rejected_column', 'structural_column', 'route_column', 'route_value')


def lexical_locator(source):
    """Transport-aware source identity without I/O: exact gs/bq locators, normalized absolute local paths."""
    uri = source['uri']
    if uri.startswith(('gs://', 'bq://')):
        return uri
    return 'file:' + os.path.normcase(os.path.abspath(uri))


def physical_conflicts(doc, locate):
    """Generic inputs: labels never define physical identity.

    A physical column is (canonical source, series selector, CSV column). All declarations of one
    source share one series/clock scheme, all declarations of one column share quality and lineage
    semantics, targets are not duplicated, and donors are neither a target nor its descendant.
    """
    schemes, labels, declared, parents = {}, {}, {}, {}
    def key_of(ref):
        source, instrument, field = resolve_ref(doc, ref)
        selector = instrument if source['identity_column'] is not None else None
        return (locate(source), selector, field['column'])
    def contract_key(label):
        """Resolved lineage labels read '<source>|<series>|<field or CSV column>'; any source or series."""
        source_name, _, rest = label.partition('|')
        source = doc['sources'].get(source_name)
        if source is None:
            return None
        series = [i for i in source['instruments'] if rest.startswith(i + '|')]
        if not series and set(source['instruments']) == {'*'} and '|' in rest:
            series = [rest.split('|', 1)[0]]  # A named series of a pending discovered grouping.
        if len(series) > 1:
            raise diagnostic('ambiguous_lineage_label', pointer('datasets', source_name), 'a lineage label matches '
                             'more than one series')
        if not series:
            return None
        name = rest[len(series[0]) + 1:]
        fields = source['instruments'].get(series[0], source['instruments'].get('*', {'fields': {}}))['fields']
        selector = series[0] if source['identity_column'] is not None else None
        return (locate(source), selector, fields[name]['column'] if name in fields else name)
    def parent_key(label):
        # Declared fields and contract labels resolve to physical columns; anything else is an external leaf.
        return labels.get(label) or contract_key(label) or ('external', label)
    for name, source in doc['sources'].items():
        locator = locate(source)
        scheme = canonical([source[k] for k in ('identity_column', 'timestamp_column', 'cadence_seconds', 'clock',
                                                'compressed')] + [source.get('format', 'csv-gzip' if source['compressed'] else 'csv')])
        if schemes.setdefault(locator, scheme) != scheme:
            raise diagnostic('conflicting_source_declarations', pointer('datasets', name), 'the same input is declared '
                             'more than once with different timestamp, series, cadence, timezone or compression '
                             'settings; declare it once')
        for instrument, metadata in source['instruments'].items():
            selector = instrument if source['identity_column'] is not None else None
            for field in metadata['fields'].values():
                key = (locator, selector, field['column'])
                if labels.setdefault(field['lineage'], key) != key or contract_key(field['lineage']) not in {None, key}:
                    raise diagnostic('ambiguous_lineage_label', pointer('datasets', name), 'one lineage label names '
                                     'two different physical columns')
    for name, source in doc['sources'].items():
        for instrument, metadata in source['instruments'].items():
            for column_name, field in metadata['fields'].items():
                key = labels[field['lineage']]
                ancestry = sorted({parent_key(p) for p in field['derived_from']}, key=repr)
                semantics = canonical([{k: field.get(k) for k in SEMANTICS}, ancestry])
                if declared.setdefault(key, semantics) != semantics:
                    raise diagnostic('conflicting_physical_declarations', pointer('datasets', name, 'columns', column_name),
                                     'this physical column is declared elsewhere with different unit, missing-value, '
                                     'QC, mask, regime or lineage settings; declarations of one column must agree')
                parents[key] = set(ancestry)
    def lineage(key, stack=()):
        if key in stack:
            raise diagnostic('cyclic_lineage', '/', 'derived_from declarations form a cycle')
        return {key}.union(*(lineage(p, (*stack, key)) for p in parents.get(key, ())))
    for key in list(parents):
        lineage(key)  # Every declared ancestry is acyclic, not only the donors' ancestry.
    work = []
    for dataset_name, dataset in doc['datasets'].items():
        if not dataset['enabled']:
            continue
        span = None if dataset['interval'] is None else tuple(map(stamp, dataset['interval']))
        for instrument in dataset['instruments']:
            group_keys = {}
            for target_name, target in dataset['targets'].items():
                ref = dict(source=dataset['source'], instrument=instrument, field=target_name, unit_regime=target['unit_regime'])
                key = group_keys[target_name] = key_of(ref)
                for other, other_span in work:
                    if other == key and (span is None or other_span is None or span[0] < other_span[1] and other_span[0] < span[1]):
                        raise diagnostic('duplicate_physical_target', pointer('datasets', dataset_name, 'targets', target_name),
                                         'this physical column is already a target for an overlapping period under '
                                         'another name; declare each column once')
                work.append((key, span))
            for target_name, target in dataset['targets'].items():
                for route in target['routes'].values():
                    recipes = [route['recipe'], *route['fallbacks']] if route['enabled'] else []
                    for recipe in recipes:
                        for member in [recipe, *recipe.get('members', [])]:
                            group = {group_keys[n] for n in member['targets'] if n in group_keys}
                            seen = set()
                            for ref in member['predictors']:
                                donor = key_of(ref)
                                if donor in seen or lineage(donor) & group:
                                    raise diagnostic('alias_or_descendant_donor', pointer('datasets', dataset_name,
                                                     'targets', target_name), 'a predictor is the target itself, an '
                                                     'alias of it, a duplicate donor, or derived from it')
                                seen.add(donor)


def _validate_generic_source(source_name, source, cloud, trail, pending):
    from .cloud_config import gcs_uri, validate_artifact, validate_source
    from .simple import zone
    keys(source, ('family', 'uri', 'identity_column', 'timestamp_column', 'key_column', 'cadence_seconds', 'clock',
                  'compressed', 'preparation_profile', 'policy', 'instruments'), ('sha256', 'generation', 'bigquery', 'format'))
    if source['preparation_profile'] != GENERIC:
        raise ValueError('operation/2 supports the generic-v1 profile; project profiles remain operation/1')
    if source['family'] is not None:
        text(source['family'])
    integer(source['cadence_seconds'], 1, 604800)
    if type(source['compressed']) is not bool:
        raise ValueError('compressed must be boolean')
    from .parquet import input_format
    if 'format' in source and not isinstance(source['format'], str):
        raise diagnostic('unsupported_format', pointer('sources', source_name, 'format'), 'format must be csv, csv-gzip or parquet')
    kind = input_format(source['uri'], source.get('format'), pointer('sources', source_name, 'format'))
    if kind == 'parquet' and source['compressed'] or 'format' in source and source['compressed'] != (kind == 'csv-gzip'):
        raise diagnostic('unsupported_compression', pointer('sources', source_name, 'compressed'),
                         'Parquet uses internal compression; CSV compression must agree with format')
    if kind == 'parquet':
        source['format'] = kind
    text(source['timestamp_column'])
    for k in ('identity_column', 'key_column'):
        if source[k] is not None:
            text(source[k])
    keys(source['clock'], ('timezone',))
    if source['clock']['timezone'] is not None:
        zone(source['clock']['timezone'])
    if source['policy'] != POLICY:
        raise ValueError('generic preparation policy differs from ecofill.generic-preparation/1')
    instruments = source['instruments']
    if not isinstance(instruments, dict) or not instruments:
        raise ValueError('explicit series required')
    if '*' in instruments and (not pending or len(instruments) != 1 or source['identity_column'] is None):
        raise ValueError('a series template belongs only to pending grouped input')
    for name, metadata in instruments.items():
        trail[:] = ['sources', source_name, 'instruments', name]
        text(name)
        keys(metadata, ('aliases', 'validity', 'exclusions', 'fields'))
        if metadata['aliases'] != []:
            raise ValueError('generic series select exact identity values without aliases')
        open_validity = pending and metadata['validity'] is None
        if not open_validity:
            interval(metadata['validity'])
        intervals(metadata['exclusions'])
        if not isinstance(metadata['fields'], dict) or not metadata['fields']:
            raise ValueError('typed fields required')
        for field_name, field in metadata['fields'].items():
            trail[:] = ['sources', source_name, 'instruments', name, 'fields', field_name]
            text(field_name)
            keys(field, GENERIC_FIELD, GENERIC_OPTIONAL)
            for k in ('column', 'unit', 'unit_regime', 'lineage'):
                text(field[k])
            for k in ('qc_column', 'observed_column', 'rejected_column', 'structural_column', 'route_column', 'route_value'):
                if k in field:
                    text(field[k])
            if not isinstance(field['derived_from'], list) or any(not isinstance(x, str) for x in field['derived_from']):
                raise ValueError('explicit lineage list required')
            if not (pending and field['domains'] is None):
                intervals(field['domains'])
                if not open_validity:
                    for a, b in field['domains']:
                        if stamp(a) < stamp(metadata['validity'][0]) or stamp(b) > stamp(metadata['validity'][1]):
                            raise ValueError('domain exceeds metadata validity')
            tokens = field['missing_tokens']
            if not isinstance(tokens, list) or '' not in tokens or len(set(tokens)) != len(tokens) \
                    or any(not isinstance(x, str) for x in tokens) or len(tokens) > 32:
                raise ValueError('missing tokens are unique strings including the empty cell')
            if not isinstance(field['sentinels'], list) or len(field['sentinels']) > 16:
                raise ValueError('bounded sentinel rule list required')
            for rule in field['sentinels']:
                sentinel_rule(rule)
            if ('qc_column' in field) != ('qc_accept' in field):
                raise ValueError('a QC column requires its explicit accepted-code set')
            if 'qc_accept' in field and (not isinstance(field['qc_accept'], list) or not field['qc_accept']
                    or any(not isinstance(x, str) for x in field['qc_accept'])
                    or len(set(field['qc_accept'])) != len(field['qc_accept'])):
                raise ValueError('accepted QC codes must be a nonempty list of unique strings')
            if ('route_column' in field) != ('route_value' in field):
                raise ValueError('unit regime selector requires column and value')
            for parent in field['derived_from']:
                text(parent)  # Ancestry is metadata; parents may live in other sources/series or be unmeasured.
    trail[:] = ['sources', source_name]
    uri = source['uri']
    if not isinstance(uri, str):
        raise ValueError('source URI required')
    if uri.startswith('bq://') or not pending:
        validate_source(source, cloud)
    else:
        if 'bigquery' in source:
            raise ValueError('contradictory source modes')
        if 'sha256' in source:
            sha(source['sha256'])
        if uri.startswith('gs://'):
            gcs_uri(uri)
            if 'generation' in source:
                integer(source['generation'], 1, 2**63-1)
        else:
            validate_artifact({k: v for k, v in source.items() if k in {'uri', 'generation'}})


validate = load_config
