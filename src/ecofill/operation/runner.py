"""Original whole-gap operation, finite explicit fallbacks, durable local lifecycle."""
from collections import Counter
from dataclasses import dataclass, asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

from ..data import SeriesData
from ..schema import TimeSeriesSchema, VariableSpec
from ..drivers import OptionalBackendUnavailable
from .config import (CLASSES, ENSEMBLES, NCR, canonical, digest, identifier, load_config, physical_conflicts,
                     resolve_ref, stamp)
from .preparation import check_grid_alignment, file_locator, prepare_field, read_source, support_domains
from .recipes import native_plan, execute_native, strict_blend
from .bundle import Writer, BudgetStop, inspect_bundle


@dataclass(frozen=True)
class OperationPlan:
    config: object
    document_json: str

    @property
    def identity(self):
        return hashlib.sha256(self.document_json.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self.document_json)


def active_recipes(target):
    return [recipe for route in target['routes'].values() if route['enabled']
            for recipe in [route['recipe'], *route['fallbacks']]]


def dependency_refs(doc):
    refs = {}
    for dataset in doc['datasets'].values():
        if not dataset['enabled']:
            continue
        for instrument in dataset['instruments']:
            for name, target in dataset['targets'].items():
                ref = dict(source=dataset['source'], instrument=instrument, field=name, unit_regime=target['unit_regime'])
                refs[digest(ref)] = ref
                for recipe in active_recipes(target):
                    for donor in recipe['predictors']:
                        refs[digest(donor)] = donor
    return list(refs.values())


def recipe_cost(recipe):
    if recipe['method'] in ENSEMBLES:
        return sum(recipe_cost(m) for m in recipe['members'])
    if recipe['method'] == 'P-X':
        return 2 * len(recipe['targets'])
    return 1 if recipe['method'] == 'LIN' else 2


class RecordingBackend:
    """Retain a return before the package checks shape, finiteness and integrity."""
    def __init__(self, backend, writer, attempt_id):
        self.backend, self.writer, self.attempt_id = backend, writer, attempt_id
        self.serial = 0

    @property
    def identity(self):
        return self.backend.identity

    def retain(self, request, output, metadata=None):
        array = np.asarray(output)
        values = None
        if array.dtype.kind in 'iufb':
            def finite(value):
                if isinstance(value, list):
                    return [finite(v) for v in value]
                return None if isinstance(value, float) and not np.isfinite(value) else value
            values = finite(array.tolist())
        record = {'schema': 'ecofill.operation.native_return/1', 'shape': list(array.shape),
                  'dtype': str(array.dtype), 'values': values, 'request_type': type(request).__name__}
        if hasattr(request, 'direction'):
            record.update(direction=request.direction, channels=list(request.channels))
        if hasattr(request, 'target'):
            record.update(target=request.target, seed=request.seed, feature_names=list(request.feature_names))
        self.writer.write(f'attempts/{self.attempt_id}/native-{self.serial}.json', record)
        self.serial += 1

    def predict(self, request):
        output = self.backend.predict(request)
        self.retain(request, output)
        return output

    def forecast(self, request):
        output = self.backend.forecast(request)
        self.retain(request, output)
        return output


def prepare_operation(config, *, source_reader=None):
    config = load_config(config)
    inputs, unavailable = {}, []
    if config.pending:
        # Simple input: read each active source once and resolve input facts from those bytes.
        from .simple import resolve
        config, inputs, unavailable = resolve(config, source_reader or read_source)
    doc = config.to_dict(); limits = doc['limits']
    generic = doc['schema'] == 'ecofill.operation/2'
    if generic:
        # Paths, links or spellings of one local file are one physical input (labels never decide).
        physical_conflicts(doc, file_locator)
    refs = dependency_refs(doc)
    sources = {ref['source'] for ref in refs}
    for name in sorted(sources - set(inputs)):
        try:
            inputs[name] = (source_reader or read_source)(doc['sources'][name], limits)
        except FileNotFoundError:
            unavailable.append(name)
    if sum(len(source.payload) for source in inputs.values()) > limits['scratch_bytes']:
        raise limit_stop('scratch_bytes', 'inputs_exceed_scratch_bytes', 'the active inputs together exceed '
                         'limits.scratch_bytes; remove or narrow datasets, or raise limits.scratch_bytes')
    if generic:
        check_grid_alignment(doc, inputs)
    fields = {}
    for ref in refs:
        if ref['source'] in inputs:
            try:
                fields[digest(ref)] = prepare_field(doc, ref, inputs, limits)
            except FileNotFoundError:
                unavailable.append(ref['source'])
    if sum(len(pair[0].data.raw) * 512 for pair in fields.values()) > limits['memory_bytes']:
        raise limit_stop('memory_bytes_estimate', 'prepared_inputs_exceed_memory_bytes', 'the conservative estimate '
                         'of 512 bytes per prepared grid row exceeds limits.memory_bytes; narrow datasets, series or '
                         'interval, or raise limits.memory_bytes')
    partitions, records = [], []
    for dataset_id, dataset in doc['datasets'].items():
        if not dataset['enabled']:
            continue
        for number, instrument in enumerate(dataset['instruments']):
            if not dataset['targets']:
                continue
            target_refs = {n: dict(source=dataset['source'], instrument=instrument, field=n, unit_regime=t['unit_regime'])
                           for n, t in dataset['targets'].items()}
            dependencies = list(target_refs.values()) + [ref for t in dataset['targets'].values()
                for recipe in active_recipes(t) for ref in recipe['predictors']]
            if any(digest(ref) not in fields for ref in dependencies):
                unavailable.append(dataset_id + '/' + str(number))
                continue
            first, _ = fields[digest(next(iter(target_refs.values())))]
            index = first.data.raw.index
            raw = pd.DataFrame(index=index)
            masks = {k: pd.DataFrame(index=index) for k in ('observed', 'qc_rejected', 'structural', 'artificial', 'inserted')}
            specs, donor_domains, reasons, target_masks = [], {}, {}, {}
            extra = dataset['exclusions']
            # Target grids must genuinely align; source histories are never padded for J.
            for name, ref in target_refs.items():
                prepared, metadata = fields[digest(ref)]
                if not prepared.data.raw.index.equals(index):
                    raise ValueError('target grids differ')
                raw[name] = prepared.data.raw[name]
                for k in masks:
                    masks[k][name] = prepared.data.mask(k)[name]
                target_masks[name] = {k: prepared.data.mask(k)[name] for k in masks}
                specs.append(VariableSpec(name, metadata['unit']))
                reasons[name] = {'preparation': json.loads(prepared.reasons_json), **metadata}
            donor_refs = {digest(ref): ref for t in dataset['targets'].values() for r in active_recipes(t) for ref in r['predictors']}
            for key, ref in donor_refs.items():
                prepared, metadata = fields[key]; source, canonical_instrument, field = resolve_ref(doc, ref)
                name = 'donor::' + key[:24]
                domains = support_domains(source, canonical_instrument, field)
                domain_mask = np.zeros(len(index), dtype=bool)
                pairs = []
                for a, b in domains:
                    left, right = int(index.searchsorted(stamp(a))), int(index.searchsorted(stamp(b)))
                    if left < right:
                        pairs.append((left, right)); domain_mask[left:right] = True
                donor_domains[name] = tuple(pairs)
                raw[name] = prepared.data.raw[ref['field']].reindex(index)
                for k in masks:
                    masks[k][name] = prepared.data.mask(k)[ref['field']].reindex(index, fill_value=False)
                masks['observed'][name] &= domain_mask
                specs.append(VariableSpec(name, metadata['unit'], 'driver'))
                reasons[name] = metadata
            # Core SeriesData has a row-level inserted invariant. An original
            # external donor can exist at a timestamp inserted for the target.
            # Preserve source-cell insertion evidence separately in the export;
            # mark a combined row inserted only when every channel is absent.
            shared_inserted = masks['inserted'].any(axis=1) & raw.isna().all(axis=1)
            for name in masks['inserted']:
                masks['inserted'][name] = shared_inserted
            data = SeriesData(raw, schema=TimeSeriesSchema(first.data.schema.cadence, tuple(specs)), **masks)
            partition = {'id': f'{dataset_id}-{number}', 'dataset_id': dataset_id, 'instrument': instrument,
                         'data': data, 'dataset': dataset, 'reasons': reasons, 'target_masks': target_masks, 'donor_domains': donor_domains,
                         'gaps': []}
            # prepare_batch's inventory predates domains, output crop and route selection.
            for name, ref in target_refs.items():
                prepared, metadata = fields[digest(ref)]
                source, canonical_instrument, field = resolve_ref(doc, ref)
                domains = support_domains(source, canonical_instrument, field, extra)
                start, stop = map(stamp, dataset['interval'])
                for gap in json.loads(prepared.inventory_json):
                    a, b = stamp(gap['start']), stamp(gap['end']) + data.schema.cadence
                    if a >= stop or b <= start:
                        continue
                    cls = gap['duration_class']; reason = None
                    containing = [p for p in domains if stamp(p[0]) <= a and b <= stamp(p[1])]
                    if gap['structural']:
                        reason = 'structural_absence'
                    elif cls == 'beyond_limit':
                        reason = 'gap_exceeds_limit'
                    elif a < start or b > stop:
                        reason = 'boundary_gap_unresolved'
                    elif gap['geometry'] != 'bounded':
                        reason = 'boundary_gap_unresolved'
                    elif a < stamp(metadata['validity'][0]) or b > stamp(metadata['validity'][1]):
                        reason = 'metadata_out_of_validity'
                    elif not containing:
                        reason = 'excluded_or_domain_boundary'
                    route = dataset['targets'][name]['routes'].get(cls)
                    if reason is None and (route is None or not route['enabled']):
                        reason = 'no_route'
                    record = dict(gap, partition=partition['id'], source=ref, reason=reason,
                                  domain=containing[0] if containing else None,
                                  recipes=[] if reason else [route['recipe'], *route['fallbacks']])
                    partition['gaps'].append(record); records.append(record)
            partitions.append(partition)
    planned_calls = sum(sum(recipe_cost(r) for r in gap['recipes']) for gap in records)
    document = {'schema': 'ecofill.operation.plan/1', 'config_identity': config.identity, 'purpose': doc['purpose'],
                'dependencies': refs, 'inputs': {name: source.receipt for name, source in inputs.items()},
                'gaps': records, 'planned_gaps': len(records), 'method_call_upper_bound': planned_calls,
                'budget_exceeded': len(records) > limits['max_gaps'] or planned_calls > limits['max_method_calls'],
                'unavailable_inputs_or_partitions': sorted(set(unavailable)),
                'partitions': [{'id': p['id'], 'data_fingerprint': p['data'].fingerprint} for p in partitions]}
    if doc['schema'] == 'ecofill.operation/2':
        exceeded = {name: {'planned': planned, 'limit': limits[name], 'exceeded': planned > limits[name]}
                    for name, planned in (('max_gaps', len(records)), ('max_method_calls', planned_calls))}
        document.update(schema='ecofill.operation.plan/2', resolution=doc.get('resolution'), budget=exceeded | {
            'advice': None if not document['budget_exceeded'] else 'planned work exceeds limits: narrow datasets, '
                      'targets or interval, or raise limits.max_gaps / limits.max_method_calls within deployment policy; '
                      'inputs are never truncated and counters never raised automatically'})
    return OperationPlan(config, canonical(document)), inputs, partitions


def limit_stop(name, code, explanation):
    stop = BudgetStop(name)
    stop.diagnostic = {'code': code, 'location': '/limits/' + name.removesuffix('_estimate'), 'explanation': explanation}
    return stop


def make_operation_plan(config, *, output=None, source_reader=None):
    plan, inputs, _ = prepare_operation(config, source_reader=source_reader)
    if output is not None:
        root = Path(output); root.mkdir(parents=True, exist_ok=False)
        writer = Writer(root, plan.config.to_dict()['limits']['output_bytes'])
        writer.write('config.resolved.json', plan.config.to_dict())
        if plan.config.authored_json is not None:
            writer.write('config.authored.json', json.loads(plan.config.authored_json))
        for name, source in inputs.items():
            writer.write('inputs/' + name + '.snapshot', source.payload)
        writer.write('input_manifest.json', plan.to_dict()['inputs'])
        writer.write('plan.json', plan.to_dict())
    return plan


def backend_for(recipe, doc, supplied, cache):
    name = recipe.get('backend')
    if name is None:
        return None
    if name in supplied:
        return supplied[name]
    key = name
    if key in cache:
        return cache[key]
    model = doc['models'][name]
    if not model['artifacts']:
        return None
    packages = ('tabpfn', 'torch') if model['kind'] == 'tabpfn35' else ('timesfm', 'torch')
    for package in packages:
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None
    for artifact in model['artifacts']:
        path = Path(artifact['uri'])
        if not path.is_file():
            return None
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != artifact['sha256']:
                raise ValueError('checkpoint integrity mismatch')
    if model['kind'] == 'tabpfn35':
        from ..tabpfn import TabPFNBackend
        value = TabPFNBackend(model['artifacts'][0]['uri'], device='cpu')
    else:
        from ..timesfm_adapters import TimesFM3Backend
        # The joint-capable adapter also dispatches the single U channel once,
        # exactly as the independent wrapper does; share one loaded checkpoint.
        purpose = NCR if NCR in {doc['purpose'], model.get('purpose')} else doc['purpose']
        value = TimesFM3Backend(Path(model['artifacts'][0]['uri']).parent, context_length=128,
            max_horizon=336, joint=True, purpose=purpose)
        value.load()
    cache[key] = value
    return value


def provenance():
    versions = {}
    for name in ('ecofill', 'numpy', 'pandas', 'scikit-learn', 'tabpfn', 'timesfm', 'torch', 'google-cloud-bigquery', 'google-cloud-storage'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    distribution = importlib.metadata.distribution('ecofill')
    record = distribution.read_text('RECORD')
    wheel = os.environ.get('ECOFILL_WHEEL_PATH')
    wheel_hash = None
    if wheel is not None:
        with Path(wheel).open('rb') as stream:
            wheel_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    package = Path(__file__).resolve().parents[1]
    files = ['operation/config.py', 'operation/cloud_config.py', 'operation/cloud.py', 'operation/deployment.py',
             'operation/__init__.py', 'operation/cli.py', 'operation/schema.json', 'operation/simple.py',
             'operation/simple_example.py',
             'operation/simple.schema.json', 'operation/operation2.schema.json',
             'operation/preparation.py', 'operation/recipes.py', 'operation/runner.py',
             'operation/bundle.py', 'shortlist.py', 'batch.py', 'drivers.py', 'time_indexed.py',
             'forecast.py', 'planning.py', 'methods.py', 'ensemble.py', 'data.py', 'schema.py', 'gaps.py',
             'tabpfn.py', 'timesfm_adapters.py', 'result.py']
    code = {}
    for name in files:
        with (package / name).open('rb') as stream:
            code[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'schema': 'ecofill.operation.provenance/1', 'python': platform.python_version(),
            'platform': platform.system(), 'libraries': versions,
            'implementation_sha256': digest(code), 'code': code,
            'installed_record_sha256': None if record is None else hashlib.sha256(record.encode()).hexdigest(),
            'image_digest': os.environ.get('ECOFILL_IMAGE_DIGEST'), 'image_id': os.environ.get('ECOFILL_IMAGE_ID'),
            'wheel_sha256': wheel_hash, 'precision': 'float64 package inputs; adapter CPU float32',
            'threads': 'RF n_jobs=1; caller-owned backend thread qualification',
            'readback_tolerance': {'relative': 1e-12, 'absolute': 1e-12}}


def run_operation(config, *, run_id, backends=None, transport=None, config_origin=None):
    config = load_config(config); identifier(run_id)
    doc = config.to_dict(); limits = doc['limits']; backends = backends or {}
    runtime = {k: v.identity for k, v in backends.items()}
    implementation = provenance()
    request = {'schema': 'ecofill.operation.request/1', 'config_identity': config.identity,
               'purpose': doc['purpose'], 'run_id': run_id, 'injected_backends': runtime,
               'implementation_sha256': implementation['implementation_sha256'],
               'wheel_sha256': implementation['wheel_sha256'], 'libraries': implementation['libraries']}
    if transport is not None:
        transport.configure(doc)
        if config_origin is not None and config_origin != transport.config_origin:
            raise ValueError('config origin must match the verified fetch')
        request.update(config_origin=transport.config_origin, image_digest=implementation['image_digest'],
                       image_id=implementation['image_id'],
                       deployment_policy_identity=digest(asdict(transport.policy)))
    if doc['output']['uri'].startswith('gs://'):
        if transport is None:
            raise ValueError('cloud output requires explicit transport')
        writer, previous = transport.claim(doc, run_id, request)
        if previous is not None:
            return previous
    else:
        root = Path(doc['output']['uri']) / 'runs' / run_id
        root.parent.mkdir(parents=True, exist_ok=True)
        try:
            root.mkdir()
        except FileExistsError:
            if not (root / 'request.json').is_file():
                return {'status': 'in_progress', 'run_uri': str(root)}
            if json.loads((root / 'request.json').read_text()) != request:
                raise ValueError('run identity conflict')
            if not (root / 'terminal.json').is_file():
                return {'status': 'in_progress', 'run_uri': str(root)}
            return inspect_bundle(root)
        writer = Writer(root, limits['output_bytes'])
        writer.write('request.json', request, terminal=True)
    if config.authored_json is not None:
        writer.write('config.authored.json', json.loads(config.authored_json), terminal=True)
    # Pending simple input is resolved from the snapshotted bytes; its resolved config publishes at the end.
    writer.write('config.expanded.json' if config.pending else 'config.resolved.json', doc, terminal=True)
    began = time.monotonic(); calls = 0; completed_partitions = []; audit = []; cache = {}
    attempted_partitions, published_partitions, secondary_failures = [], [], []
    status, failure, stopped, explained = 'complete', None, False, None
    plan = None
    try:
        if transport is not None:
            transport.preflight(doc, request)
        plan, inputs, partitions = prepare_operation(config, source_reader=None if transport is None else transport.read_source)
        if plan.to_dict()['unavailable_inputs_or_partitions']:
            status = 'partial'
        writer.write('plan.json', plan.to_dict())
        writer.write('input_manifest.json', {'schema': 'ecofill.operation.inputs/1', 'sources': plan.to_dict()['inputs']})
        for name, source in inputs.items():
            writer.write('inputs/' + name + '.snapshot', source.payload)
        for partition in partitions:
            attempted_partitions.append(partition['id'])
            data = partition['data']; dataset = partition['dataset']; index = data.raw.index
            original = data.available; completed = original.copy(deep=True)
            reconstructed = pd.DataFrame(False, index=index, columns=original.columns)
            disposition = pd.DataFrame('', index=index, columns=original.columns)
            processed = set(); partition_audit = []
            try:
                for gap in partition['gaps']:
                    if gap['gap_id'] in processed:
                        continue
                    entry = {k: v for k, v in gap.items() if k != 'recipes'}
                    entry.update(attempts=[], complete=False, filled_cells=0, final_reason=gap['reason'] or 'not_started')
                    partition_audit.append(entry); audit.append(entry)
                    if gap['reason']:
                        continue
                    if len(audit) > limits['max_gaps'] or time.monotonic() - began >= limits['wall_seconds']:
                        entry['final_reason'] = 'budget_exhausted'; stopped = True
                        continue
                    for recipe in gap['recipes']:
                        method = recipe['method']
                        group = [g for g in partition['gaps'] if g['variable'] in recipe['targets']
                                 and g['start_pos'] == gap['start_pos'] and g['stop_pos'] == gap['stop_pos']
                                 and g['reason'] is None and g['domain'] == gap['domain']
                                 and any(digest(r) == digest(recipe) for r in g['recipes'])]
                        if len(group) != len(recipe['targets']):
                            entry['final_reason'] = 'unaligned_target_gap'
                            continue
                        members = recipe.get('members', [recipe]); results = []; member_reason = None
                        for member_index, member in enumerate(members):
                            selected = [g for g in group if g['variable'] in member['targets']]
                            native_data, native, reason, features = native_plan(data, member, selected, gap['domain'],
                                dataset['fit_exclusions'], partition['donor_domains'], allow_provisional=doc['allow_provisional'])
                            attempt_id = digest([config.identity, partition['id'], gap['gap_id'], digest(recipe), member_index])
                            attempt = {'id': attempt_id, 'recipe_identity': digest(member), 'method': member['method'],
                                       'settings': member['parameters'], 'evidence_status': member['evidence_status'],
                                       'frozen_binding': member.get('frozen_binding'),
                                       'native_plan': None if native is None else native.to_dict(), 'features': features,
                                       'status': 'abstained', 'reason': reason}
                            entry['attempts'].append(attempt)
                            if reason:
                                member_reason = reason; break
                            if stopped or calls + recipe_cost(member) > limits['max_method_calls'] or time.monotonic() - began >= limits['wall_seconds']:
                                attempt['reason'] = member_reason = 'budget_exhausted'; stopped = True; break
                            # Ownership/started record precedes backend construction or load.
                            writer.write(f'attempts/{attempt_id}/started.json', attempt)
                            try:
                                backend = (backend_for(member, doc, backends, cache) if transport is None else
                                           transport.backend_for(member, doc, backends, cache))
                                if member['method'].startswith(('P', 'T')) and backend is None:
                                    attempt['reason'] = member_reason = 'backend_unavailable'
                                    writer.write(f'attempts/{attempt_id}/result.json', attempt)
                                    break
                                if backend is not None:
                                    backend = RecordingBackend(backend, writer, attempt_id)
                                calls += recipe_cost(member)
                                result = execute_native(native_data, native, member, backend)
                                if time.monotonic() - began >= limits['wall_seconds']:
                                    stopped = True
                            except OptionalBackendUnavailable:
                                attempt['reason'] = member_reason = 'backend_unavailable'
                                writer.write(f'attempts/{attempt_id}/result.json', attempt)
                                break
                            except Exception as exc:
                                attempt.update(status='failed', reason='runtime_failure', error_type=type(exc).__name__)
                                try:
                                    writer.write(f'attempts/{attempt_id}/result.json', attempt, terminal=True)
                                except Exception as publication:
                                    secondary_failures.append({'stage': 'attempt_result', 'error_type': type(publication).__name__})
                                raise
                            allowed = pd.DataFrame(False, index=native_data.raw.index, columns=native_data.raw.columns)
                            for g in selected:
                                allowed.iloc[g['start_pos']:g['stop_pos'], allowed.columns.get_loc(g['variable'])] = True
                            if (result.mask('imputed') & ~allowed).to_numpy().any():
                                raise ValueError('prediction outside selected original gaps')
                            attempt['backend'] = result.provenance().get('backend')
                            attempt['predictions'] = [{ 'target': g['variable'], 'values': [float(x) if predicted else None
                                for x, predicted in zip(result.completed[g['variable']].iloc[g['start_pos']:g['stop_pos']],
                                                       result.mask('imputed')[g['variable']].iloc[g['start_pos']:g['stop_pos']])]} for g in selected]
                            full = all(result.mask('imputed')[g['variable']].iloc[g['start_pos']:g['stop_pos']].all() for g in selected)
                            attempt.update(status='complete' if full else 'partial_discarded', reason=None if full else 'insufficient_support')
                            writer.write(f'attempts/{attempt_id}/result.json', attempt)
                            if not full:
                                member_reason = 'insufficient_support'; break
                            results.append(result)
                        if member_reason:
                            entry['final_reason'] = member_reason
                            if member_reason == 'budget_exhausted':
                                break
                            continue
                        result = strict_blend(native_data, results) if method in ENSEMBLES else results[0]
                        if not all(result.mask('imputed')[g['variable']].iloc[g['start_pos']:g['stop_pos']].all() for g in group):
                            entry['final_reason'] = 'ensemble_incomplete'; continue
                        for g in group:
                            name = g['variable']; slot = slice(g['start_pos'], g['stop_pos'])
                            completed.iloc[slot, completed.columns.get_loc(name)] = result.completed[name].iloc[slot].to_numpy()
                            reconstructed.iloc[slot, reconstructed.columns.get_loc(name)] = True
                            processed.add(g['gap_id'])
                        entry.update(complete=True, filled_cells=sum(g['n_steps'] for g in group), final_reason=None,
                                     selected_method=method, duration_class=gap['duration_class'],
                                     evidence_status=recipe['evidence_status'], group_gap_ids=[g['gap_id'] for g in group])
                        break
            finally:
                primary = sys.exc_info()[1]
                if primary is not None:
                    for entry in partition_audit:
                        if entry['final_reason'] == 'not_started':
                            entry['final_reason'] = 'runtime_failure'
                    represented = {e['gap_id'] for e in partition_audit} | processed
                    for unstarted in partition['gaps']:
                        if unstarted['gap_id'] not in represented:
                            retained = {k: v for k, v in unstarted.items() if k != 'recipes'}
                            retained.update(attempts=[], complete=False, filled_cells=0,
                                            final_reason=unstarted['reason'] or 'stage_not_started_after_failure')
                            partition_audit.append(retained); audit.append(retained)
                for entry in partition_audit:
                    if entry['final_reason']:
                        disposition.iloc[entry['start_pos']:entry['stop_pos'], disposition.columns.get_loc(entry['variable'])] = entry['final_reason']
                start, stop = map(stamp, dataset['interval']); output_rows = (index >= start) & (index < stop)
                targets = list(dataset['targets']); prefix = 'datasets/' + partition['id'] + '/'
                try:
                    publish_partition(writer, prefix, data, completed, reconstructed, disposition,
                                      partition, partition_audit, output_rows, targets)
                except Exception as publication:
                    if primary is None:
                        raise
                    secondary_failures.append({'stage': 'partition', 'error_type': type(publication).__name__})
                else:
                    published_partitions.append(partition['id'])
                    if primary is None and not any(e['final_reason'] == 'budget_exhausted' for e in partition_audit):
                        completed_partitions.append(partition['id'])
    except BudgetStop as exc:
        status = 'partial'; failure = type(exc).__name__; explained = getattr(exc, 'diagnostic', None)
    except Exception as exc:
        status = 'failed'; failure = type(exc).__name__; explained = getattr(exc, 'diagnostic', None)
    if stopped and status == 'complete':
        status = 'partial'
    executed = doc if plan is None else plan.config.to_dict()
    prov = implementation | {'config_identity': config.identity, 'purpose': doc['purpose'], 'models': doc['models'],
                           'runtime_backends': runtime, 'failure_type': failure,
                           'secondary_failures': secondary_failures,
                           'input_manifest_identity': None if plan is None else digest(plan.to_dict()['inputs'])}
    counts = Counter(e['final_reason'] for e in audit if e['final_reason'])
    summary = {'schema': 'ecofill.operation.summary/1', 'status': status, 'run_id': run_id,
               'config_identity': config.identity, 'purpose': doc['purpose'], 'datasets_processed': len(completed_partitions),
               'dataset_partitions': completed_partitions, 'datasets_attempted': len(attempted_partitions),
               'published_partitions': published_partitions, 'secondary_failures': secondary_failures,
               'original_gaps': 0 if plan is None else plan.to_dict()['planned_gaps'],
               'filled_gaps': sum(len(e.get('group_gap_ids', [e['gap_id']])) for e in audit if e['complete']),
               'filled_cells': sum(e['filled_cells'] for e in audit), 'method_calls': calls,
               'unresolved_reasons': dict(counts), 'filled_by_method_class': dict(Counter(
                   e['selected_method'] + '/' + e['duration_class'] for e in audit if e['complete'])),
               'elapsed_seconds': time.monotonic() - began, 'natural_gap_accuracy': 'unobserved',
               'periods': {k: v['interval'] for k, v in executed['datasets'].items() if v['enabled']},
               'output': 'manifest.json', 'failure_type': failure}
    if doc['schema'] == 'ecofill.operation/2':
        summary.update(resolved_config_identity=None if plan is None else plan.config.identity,
                       failure_diagnostic=explained)
    try:
        if config.pending:
            writer.write('config.resolved.json', executed, terminal=True)
        writer.write('provenance.json', prov, terminal=True)
        writer.finish(status, summary)
    except Exception as publication:
        if status != 'failed':
            raise
        secondary_failures.append({'stage': 'terminal', 'error_type': type(publication).__name__})
        summary['terminal_published'] = False
    return summary


def publish_partition(writer, prefix, data, completed, reconstructed, disposition,
                      partition, partition_audit, output_rows, targets):
    dataset, index = partition['dataset'], data.raw.index
    writer.csv(prefix + 'raw.csv.gz', data.raw.loc[output_rows, targets])
    writer.csv(prefix + 'completed.csv.gz', completed.loc[output_rows, targets])
    for name in ('observed', 'qc_rejected', 'structural', 'artificial', 'inserted'):
        original_mask = pd.DataFrame({n: partition['target_masks'][n][name] for n in targets}, index=index)
        writer.csv(prefix + name + '.csv.gz', original_mask.loc[output_rows, targets])
    writer.csv(prefix + 'reconstructed.csv.gz', reconstructed.loc[output_rows, targets])
    writer.csv(prefix + 'cell_states.csv.gz', pd.DataFrame(np.where(reconstructed, 'reconstructed',
        np.where(data.mask('donor'), 'observed', 'unresolved')), index=index, columns=completed.columns).loc[output_rows, targets])
    writer.csv(prefix + 'dispositions.csv.gz', disposition.loc[output_rows, targets])
    writer.write(prefix + 'gaps.json', partition_audit)
    writer.write(prefix + 'source_reasons.json', partition['reasons'])
    writer.write(prefix + 'schema.json', {'schema': 'ecofill.operation.dataset/1',
        'clock': 'UTC', 'interval': dataset['interval'], 'cadence_seconds': data.schema.cadence.total_seconds(),
        'fields': {n: {'unit': dataset['targets'][n]['unit'], 'dtype': 'float64',
                      'unit_regime': dataset['targets'][n]['unit_regime']} for n in targets},
        'masks': 'boolean', 'source_keys': 'source_reasons.json', 'support': 'original_only'})
