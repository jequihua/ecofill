"""Bounded D032 recipes composed from the existing package plans and executors.

Configuration is portable; row selections and support domains belong to each
input's plan. This module selects no model and loads no weights.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json

import numpy as np
import pandas as pd

from .data import SeriesData
from .drivers import (DriverPolicy, RegressionParameters, RegressionStep, donor_lag_data,
                      execute_driver_plan, make_driver_plan)
from .forecast import DynamicCovariate, ForecastPolicy, execute_forecast, make_forecast_plan
from .gaps import profile_gaps, validate_gap_selection, validate_support_interval
from .methods import execute
from .planning import make_plan
from .result import ImputationResult
from .schema import TimeSeriesSchema, VariableSpec
from .time_indexed import TimeFeatures, TimeIndexedPolicy, _select, execute_time_indexed, make_time_indexed_plan

METHODS = ('LIN', 'RF-X', 'RF-XL', 'T-U', 'T-X', 'P-U', 'P-X', 'T-J')
TIME = TimeFeatures(pd.Timestamp('1970-01-01', tz='UTC'), pd.Timedelta('1d'))
RF = RegressionParameters((('n_estimators', 128), ('max_depth', 12), ('min_samples_leaf', 5)))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def recipe(method, *, targets, predictors, units, cadence_seconds, source_identity,
           status='provisional'):
    """Construct one fixed portable recipe; support is still input-specific."""
    return load_shortlist_config({'schema': 'ecofill.shortlist/1', 'method': method,
        'targets': list(targets), 'predictors': list(predictors), 'units': dict(units),
        'cadence_seconds': cadence_seconds, 'source_identity': source_identity,
        'status': status, 'settings': settings(method), 'support': 'original_only'})


def settings(method):
    common = {'max_gap_seconds': 604800, 'mode': 'offline', 'seed': 0}
    if method.startswith('RF'):
        return common | {'parameters': RF.effective('rf'), 'time_features': TIME.to_dict(),
                         'window_days': 180, 'min_rows': 168, 'max_rows': 4096,
                         'selection': 'time-ordered evenly spaced eligible rows',
                         'lags': ['native_step', 'exact_1h', 'exact_24h'] if method == 'RF-XL' else [],
                         'novelty': 'flag'}
    if method.startswith('P'):
        return common | {'backend': 'tabpfn==9.0.0:TabPFN-3.5', 'device': 'cpu',
                         'n_estimators': 1, 'output_type': 'mean', 'time_features': TIME.to_dict(),
                         'window_days': 180, 'min_rows': 168, 'max_rows': 1024,
                         'selection': 'time-ordered evenly spaced eligible rows', 'lags': []}
    if method.startswith('T'):
        return common | {'backend': 'timesfm==3.0.2:TimesFM-3', 'device': 'cpu',
                         'context_length': 128, 'min_support': 32,
                         'context_policy': 'adjacent_original', 'direction': 'both',
                         'blend': 'forward*(1-i/(L-1))+backward*i/(L-1); L=1: 0.5 each',
                         'one_sided_support': 'abstain', 'covariates': 'history_plus_horizon',
                         'covariate_issue': 'archive_end; original observed retrospective covariates only',
                         'return_quantiles': False, 'padding_mode': 'none',
                         'use_znorm': False, 'make_positive': False,
                         'use_symmetric_averaging': False, 'purpose': 'non-commercial research'}
    if method == 'LIN':
        return common | {'endpoints': 'original bracketing observations'}
    raise ValueError('unsupported shortlist method')


@dataclass(frozen=True)
class ShortlistConfig:
    document_json: str

    def to_dict(self):
        return json.loads(self.document_json)

    @property
    def identity(self):
        return digest(self.to_dict())


def load_shortlist_config(value):
    if isinstance(value, ShortlistConfig):
        value = value.to_dict()
    elif not isinstance(value, dict):
        from pathlib import Path
        value = json.loads(Path(value).read_text(encoding='utf-8'))
    keys = {'schema', 'method', 'targets', 'predictors', 'units', 'cadence_seconds',
            'source_identity', 'status', 'settings', 'support'}
    if not isinstance(value, dict) or set(value) != keys \
            or value['schema'] != 'ecofill.shortlist/1' or value['method'] not in METHODS:
        raise ValueError('invalid shortlist configuration')
    targets, predictors = value['targets'], value['predictors']
    for names in (targets, predictors):
        if not isinstance(names, list) or len(set(names)) != len(names) \
                or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError('ordered names must be unique nonempty strings')
    if not targets or set(targets) & set(predictors) or set(value['units']) != set(targets + predictors):
        raise ValueError('targets/predictors/units disagree')
    if any(not isinstance(u, str) or not u.strip() for u in value['units'].values()):
        raise ValueError('stored units must be explicit')
    if (value['method'] == 'T-J' and not 2 <= len(targets) <= 4) \
            or (value['method'] != 'T-J' and len(targets) != 1):
        raise ValueError('joint recipes need 2-4 targets; other recipes need one')
    if value['method'].startswith('RF') and len(predictors) < 2:
        raise ValueError('multivariate RF requires at least two measured donors')
    if value['method'] in {'T-X', 'P-X'} and not predictors:
        raise ValueError('X requires measured donors')
    if type(value['cadence_seconds']) is not int or value['cadence_seconds'] not in {1800, 3600}:
        raise ValueError('shortlist cadence must be 30 or 60 minutes')
    if value['status'] not in {'usable', 'provisional', 'disabled'} or value['support'] != 'original_only' \
            or not isinstance(value['source_identity'], str) or not value['source_identity'].strip() \
            or value['settings'] != settings(value['method']):
        raise ValueError('shortlist settings, source identity or status drift')
    return ShortlistConfig(json.dumps(value, sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class ShortlistPlan:
    data_fingerprint: str
    config: ShortlistConfig
    arguments_json: str
    native_json: str
    verification_fingerprint: str = ''

    def __post_init__(self):
        if self.verification_fingerprint == '':
            object.__setattr__(self, 'verification_fingerprint', digest(self.to_dict()))

    def to_dict(self):
        return {'data_fingerprint': self.data_fingerprint, 'method': 'shortlist', 'mode': 'offline',
                'configuration': self.config.to_dict(), 'configuration_identity': self.config.identity,
                'arguments': json.loads(self.arguments_json), 'native': json.loads(self.native_json)}


@lru_cache(maxsize=4)
def _rf_data(data, targets, donors, segments_json, lagged):
    """Reuse immutable masked feature views across planning, validation and execution."""
    extended, feature_spec = data, ()
    if lagged:
        domains = {n: tuple(tuple(x) for x in json.loads(segments_json)[n]) for n in donors}
        extended, feature_spec = donor_lag_data(data, targets=targets, donors=donors, support_segments=domains)
    raw, masks = extended.raw, extended.masks
    time_columns = tuple('time::' + n for n in TIME.to_dict()['names'])
    if any(n in raw for n in time_columns):
        raise ValueError('time feature names already exist')
    specs = list(extended.schema.variables)
    inserted = masks['inserted'].all(axis=1)
    for n, values in zip(time_columns, TIME.compute(raw.index).T):
        raw[n] = np.where(inserted, np.nan, values)
        specs.append(VariableSpec(n, 'dimensionless', 'external'))
        for key in ('observed', 'qc_rejected', 'structural', 'artificial', 'inserted'):
            masks[key][n] = ~inserted if key == 'observed' else inserted if key == 'inserted' else False
    extended = SeriesData(raw, schema=TimeSeriesSchema(data.schema.cadence, tuple(specs)),
                          **{k: masks[k] for k in ('observed', 'qc_rejected', 'structural', 'artificial', 'inserted')})
    return extended, feature_spec


def _prepare(data, config, args):
    doc = config.to_dict()
    targets, donors, method = tuple(doc['targets']), tuple(doc['predictors']), doc['method']
    units = {v.name: v.unit for v in data.schema.variables}
    if any(units.get(k) != v for k, v in doc['units'].items()) \
            or data.schema.cadence.total_seconds() != doc['cadence_seconds']:
        raise ValueError('input units, channels or cadence differ from recipe')
    if any(v.derived_from for v in data.schema.variables if v.name in targets):
        raise ValueError('shortlist targets must be primitive')
    if doc['status'] == 'disabled' or doc['status'] == 'provisional' and not args['allow_provisional']:
        raise ValueError('recipe is disabled; provisional research requires explicit opt-in')
    selected = None if args['target_gaps'] is None else tuple(tuple(x) for x in args['target_gaps'])
    validate_gap_selection(data, selected, variables=targets)
    support = None if args['support_interval'] is None else tuple(args['support_interval'])
    a, b = validate_support_interval(data, support)
    if method == 'LIN' or method.startswith('RF'):
        effective = selected if selected is not None else tuple(
            (gap.variable, gap.start_pos, gap.stop_pos)
            for gap in profile_gaps(data) if gap.variable in targets)
        if method == 'LIN':
            selected = effective
        elif any(start < a or stop > b for _, start, stop in effective):
            raise ValueError('RF target_gaps must lie wholly inside support_interval')
    extended, feature_spec = data, ()
    if method == 'LIN':
        return data, make_plan(data, method='linear', mode='offline', max_gap=pd.Timedelta('7d'),
                               target_gaps=selected, support_interval=support), feature_spec
    if method.startswith('T'):
        cov = donors if method in {'T-X', 'T-J'} else ()
        policy = ForecastPolicy(targets, 'offline', 'blend', pd.Timedelta('7d'), 128, 32,
            int(604800 / doc['cadence_seconds']),
            covariates=tuple(DynamicCovariate(n, data.raw.index[-1], True) for n in cov),
            target_gaps=selected, support_interval=support, context_policy='adjacent_original',
            covariate_issue='archive_end',
            context_lengths=None if args['matched_context_lengths'] is None else tuple(args['matched_context_lengths']))
        return data, make_forecast_plan(data, policy), feature_spec
    block = args['block_interval']
    if block is None:
        if selected is None or not selected:
            raise ValueError('learned recipe needs a scored block or selected complete gaps')
        block = [min(x[1] for x in selected), max(x[2] for x in selected)]
    if not (len(block) == 2 and all(type(v) is int for v in block) and 0 <= block[0] < block[1] <= len(data.raw)):
        raise ValueError('invalid scored block')
    midpoint = (block[0] + block[1] - 1) // 2
    radius = int(90 * 86400 / doc['cadence_seconds'])
    start, stop = max(a, midpoint - radius), min(b, midpoint + radius + 1)
    exclusions = args['fit_exclusions']
    if any(len(x) != 2 or any(type(v) is not int for v in x) or not 0 <= x[0] < x[1] <= len(data.raw)
           for x in exclusions):
        raise ValueError('invalid fitting exclusions')
    rows = [i for i in range(start, stop) if not any(x <= i < y for x, y in exclusions)]
    if method.startswith('RF'):
        extended, feature_spec = _rf_data(data, targets, donors,
            json.dumps(args['donor_segments'], sort_keys=True) if method == 'RF-XL' else '{}',
            method == 'RF-XL')
        time_columns = tuple('time::' + n for n in TIME.to_dict()['names'])
        features = donors + tuple(f['name'] for f in feature_spec) + time_columns
    else:
        features = donors if method == 'P-X' else ()
    qualified = extended.mask('donor')[[targets[0], *features]].all(axis=1).to_numpy()
    rows = [i for i in rows if qualified[i]]
    matched = args['matched_reference_rows']
    if matched is not None:
        if any(type(i) is not int for i in matched) or list(sorted(set(matched))) != matched:
            raise ValueError('matched reference rows must be sorted unique integers')
        if not set(matched) <= set(rows):
            raise ValueError('matched rows contain unqualified support')
        rows = matched
    rows = tuple(_select(rows, 4096 if method.startswith('RF') else 1024))
    if method.startswith('RF'):
        policy = DriverPolicy((RegressionStep(targets[0], features, 'rf', parameters=RF),), start, stop,
            168, pd.Timedelta('7d'), 'offline', 'flag', digest([doc['source_identity'], feature_spec]),
            fit_rows=rows, target_gaps=selected)
        native = make_driver_plan(extended, policy)
    else:
        # Reference intervals contain only eligible fitting positions; query truth is never selected.
        policy = TimeIndexedPolicy(targets, ((start, stop),), TIME, 1024, 168,
            pd.Timedelta('7d'), 0, external_features=features, reference_rows=rows,
            target_gaps=selected, support_interval=support)
        native = make_time_indexed_plan(data, policy)
    return extended, native, feature_spec


def make_shortlist_plan(data, config, *, target_gaps=None, support_interval=None,
                       block_interval=None, fit_exclusions=(), matched_reference_rows=None,
                       matched_context_lengths=None, donor_segments=None, allow_provisional=False):
    config = load_shortlist_config(config)
    args = dict(target_gaps=target_gaps, support_interval=support_interval, block_interval=block_interval,
                fit_exclusions=fit_exclusions, matched_reference_rows=matched_reference_rows,
                matched_context_lengths=matched_context_lengths, donor_segments=donor_segments or {},
                allow_provisional=allow_provisional)
    args = json.loads(json.dumps(args, allow_nan=False))
    if type(allow_provisional) is not bool:
        raise ValueError('allow_provisional must be boolean')
    _, native, features = _prepare(data, config, args)
    return ShortlistPlan(data.fingerprint, config, json.dumps(args, sort_keys=True),
                         json.dumps({'plan': native.to_dict(), 'features': features}, sort_keys=True))


def apply_shortlist(data, plan, *, backend=None):
    if not isinstance(plan, ShortlistPlan) or plan.data_fingerprint != data.fingerprint:
        raise ValueError('shortlist plan is stale or invalid')
    if plan.verification_fingerprint != digest(plan.to_dict()):
        raise ValueError('shortlist plan was edited')
    config = load_shortlist_config(plan.config)
    args = json.loads(plan.arguments_json)
    if set(args) != {'target_gaps', 'support_interval', 'block_interval', 'fit_exclusions',
                     'matched_reference_rows', 'matched_context_lengths', 'donor_segments', 'allow_provisional'} \
            or type(args['allow_provisional']) is not bool:
        raise ValueError('invalid shortlist plan arguments')
    extended, native, features = _prepare(data, config, args)
    rebuilt = ShortlistPlan(data.fingerprint, config, json.dumps(args, sort_keys=True),
                           json.dumps({'plan': native.to_dict(), 'features': features}, sort_keys=True))
    if rebuilt != plan:
        raise ValueError('shortlist plan was edited')
    method = plan.config.to_dict()['method']
    if method == 'LIN':
        result = execute(data, native)
    elif method.startswith('RF'):
        result = execute_driver_plan(extended, native)
    elif method.startswith('T'):
        result = execute_forecast(data, native, backend)
    else:
        result = execute_time_indexed(data, native, backend)
    columns = list(data.raw)
    return ImputationResult(data, plan, result.completed[columns], result.states[columns],
        result.methods[columns], result.unresolved_reasons[columns], result.mask('imputed')[columns],
        backend_identity={'shortlist_method': method, 'native': result.provenance().get('backend'),
                          'native_plan': native.to_dict(), 'features': features})
