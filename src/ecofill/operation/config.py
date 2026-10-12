"""Offline operation/1 validation. Canonical JSON is the execution authority."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

CLASSES = {'short': (0, 7200), 'medium': (7200, 43200), 'long': (43200, 604800)}
METHODS = {'LIN', 'RF-X', 'RF-XL', 'P-U', 'P-X', 'T-U', 'T-X', 'T-J', 'AVG-U', 'AVG-X', 'AVG-J'}
ENSEMBLES = {'AVG-U': ('P-U', 'T-U'), 'AVG-X': ('P-X', 'T-X'), 'AVG-J': ('P-X', 'T-J')}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate mapping key')
        result[key] = value
    return result


def read_document(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() not in {'.yaml', '.yml'}:
        return json.loads(text, object_pairs_hook=unique,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    import yaml  # Optional; never imported for JSON/core use.
    class Loader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            return unique([(self.construct_object(k, deep=True), self.construct_object(v, deep=True))
                           for k, v in node.value])
    # YAML 1.1 yes/no/on/off and implicit dates are ambiguous configuration.
    for token in yaml.scan(text):
        if isinstance(token, yaml.tokens.ScalarToken) and token.plain:
            if token.value.lower() in {'yes', 'no', 'on', 'off'}:
                raise ValueError('use true/false booleans and quoted timestamps')
    try:
        value = yaml.load(text, Loader=Loader)
        canonical(value)  # Reject implicit dates, cycles, NaN and infinity.
        return value
    except (TypeError, RecursionError, yaml.YAMLError) as exc:
        raise ValueError('invalid YAML configuration') from exc


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
    result = pd.Timestamp(value)
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
    return source['family'], instrument, field.get('physical_field', field['column'])


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

    @property
    def identity(self):
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self.canonical_json)


def load_config(value):
    if isinstance(value, OperationConfig):
        value = value.to_dict()
    elif isinstance(value, (str, Path)):
        value = read_document(value)
    try:
        doc = json.loads(canonical(value))
    except (TypeError, RecursionError) as exc:
        raise ValueError('configuration must be finite acyclic JSON') from exc
    keys(doc, ('schema', 'purpose', 'output', 'limits', 'sources', 'datasets', 'models'), ('allow_provisional', 'cloud'))
    if doc['schema'] != 'ecofill.operation/1' or doc['purpose'] != 'non-commercial research':
        raise ValueError('operation/1 requires non-commercial research')
    if type(doc.setdefault('allow_provisional', False)) is not bool:
        raise ValueError('allow_provisional must be boolean')
    keys(doc['output'], ('uri',))
    text(doc['output']['uri'])
    if doc['output']['uri'].startswith('gs://'):
        from .cloud_config import gcs_uri
        gcs_uri(doc['output']['uri'])
    else:
        local_uri(doc['output']['uri'])
    if 'cloud' in doc:
        from .cloud_config import validate_cloud
        validate_cloud(doc['cloud'])
    limits = doc['limits']
    keys(limits, ('max_gaps', 'max_method_calls', 'max_rows_per_series', 'wall_seconds', 'memory_bytes', 'scratch_bytes', 'output_bytes'))
    for key, value in limits.items():
        integer(value, 0 if key in {'max_gaps', 'max_method_calls'} else 1, 2**53-1)
    for block in ('sources', 'models', 'datasets'):
        if not isinstance(doc[block], dict):
            raise ValueError('keyed configuration block required')
        for name in doc[block]:
            identifier(name)
    for source in doc['sources'].values():
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
            for name, field in metadata['fields'].items():
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
        validate_source(source, doc.get('cloud'))
    for model in doc['models'].values():
        keys(model, ('kind', 'artifacts', 'license', 'revision'))
        if model['kind'] not in {'tabpfn35', 'timesfm3'}:
            raise ValueError('unknown backend kind')
        text(model['license']); text(model['revision'])
        if not isinstance(model['artifacts'], list):
            raise ValueError('artifact list required')
        for artifact in model['artifacts']:
            keys(artifact, ('uri', 'sha256'), ('generation',))
            from .cloud_config import validate_artifact
            validate_artifact(artifact); sha(artifact['sha256'])
    work = []
    for dataset in doc['datasets'].values():
        keys(dataset, ('enabled', 'source', 'instruments', 'interval', 'targets', 'exclusions', 'fit_exclusions'))
        if type(dataset['enabled']) is not bool:
            raise ValueError('enabled must be boolean')
        interval(dataset['interval']); intervals(dataset['exclusions']); intervals(dataset['fit_exclusions'])
        if not isinstance(dataset['targets'], dict) or not isinstance(dataset['instruments'], list) or not dataset['instruments']:
            raise ValueError('explicit instruments and targets required')
        if not dataset['enabled']:
            continue
        for instrument in dataset['instruments']:
            for name, target in dataset['targets'].items():
                keys(target, ('unit', 'unit_regime', 'routes'))
                ref = dict(source=dataset['source'], instrument=instrument, field=name, unit_regime=target['unit_regime'])
                _, _, field = resolve_ref(doc, ref)
                if target['unit'] != field['unit'] or field['derived_from']:
                    raise ValueError('target unit mismatch or derived target')
                identity = physical(doc, ref)
                a, b = map(stamp, dataset['interval'])
                if any(i == identity and a < y and x < b for i, x, y in work):
                    raise ValueError('duplicate physical target work')
                work.append((identity, a, b))
                if not isinstance(target['routes'], dict) or set(target['routes']) - set(CLASSES):
                    raise ValueError('unknown duration class')
                for cls, route in target['routes'].items():
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
    if (doc['output']['uri'].startswith('gs://') or any(s['uri'].startswith(('gs://', 'bq://')) for s in doc['sources'].values())
            or any(a['uri'].startswith('gs://') for m in doc['models'].values() for a in m['artifacts'])) and 'cloud' not in doc:
        raise ValueError('remote transport requires explicit cloud bounds')
    resolved = canonical(doc)
    if limits['output_bytes'] < 2 * len(resolved.encode()) + 2 * 65536:
        raise ValueError('output budget cannot retain request/config and terminal control records')
    return OperationConfig(resolved)


validate = load_config
