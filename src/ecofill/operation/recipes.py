"""Compose existing policies with bounded operation-specific overrides."""
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from ..data import SeriesData
from ..drivers import (DriverPolicy, RegressionParameters, RegressionStep, donor_lag_data,
                       make_driver_plan, execute_driver_plan)
from ..forecast import DynamicCovariate, ForecastPolicy, make_forecast_plan, execute_forecast
from ..planning import make_plan
from ..methods import execute
from ..schema import TimeSeriesSchema, VariableSpec
from ..shortlist import TIME, make_shortlist_plan, apply_shortlist
from ..time_indexed import TimeIndexedPolicy, _select, make_time_indexed_plan, execute_time_indexed
from ..ensemble import (check_result, EnsembleMember, EnsembleRecipe, blend, configuration_identity,
                        plan_method_mode_training)
from .config import digest, integer


@dataclass(frozen=True)
class OperationRFParameters(RegressionParameters):
    """Broader admitted bounds without changing legacy RegressionParameters."""
    def __post_init__(self):
        bounds = {'n_estimators': (1, 512), 'max_depth': (1, 32), 'min_samples_leaf': (1, 128),
                  'random_state': (0, 2**31-1), 'n_jobs': (1, 1)}
        if len(dict(self.values)) != len(self.values) or set(dict(self.values)) - set(bounds):
            raise ValueError('invalid operational RF overrides')
        for key, value in self.values:
            integer(value, *bounds[key])
        object.__setattr__(self, 'values', tuple(sorted(self.values)))


def native_plan(data, recipe, gaps, domain, fit_exclusions, donor_domains, *, allow_provisional=False):
    method, params = recipe['method'], recipe['parameters']
    selected = tuple((g['variable'], g['start_pos'], g['stop_pos']) for g in gaps)
    targets = tuple(recipe['targets'])
    donors = tuple('donor::' + digest(ref)[:24] for ref in recipe['predictors'])
    index = data.raw.index
    support = (int(index.searchsorted(pd.Timestamp(domain[0]))), int(index.searchsorted(pd.Timestamp(domain[1]))))
    if method in {'T-J'} and len(selected) != len(targets):
        return data, None, 'unaligned_target_gap', ()
    if any(g['duration_seconds'] > params['max_gap_seconds'] for g in gaps):
        return data, None, 'gap_exceeds_limit', ()
    if 'frozen_recipe' in recipe:
        # Names are mapped to exact saved predictor identities; never old row positions.
        frozen = recipe['frozen_recipe']
        mapping = {'donor::' + digest(recipe['frozen_binding']['predictors'][label]['ref'])[:24]: label
                   for label in frozen['predictors']}
        raw, masks = data.raw.rename(columns=mapping), {k: m.rename(columns=mapping) for k, m in data.masks.items()}
        specs = tuple(VariableSpec(mapping.get(v.name, v.name), v.unit, v.role) for v in data.schema.variables)
        aligned = SeriesData(raw, schema=TimeSeriesSchema(data.schema.cadence, specs),
                             **{k: masks[k] for k in ('observed', 'qc_rejected', 'structural', 'artificial', 'inserted')})
        if frozen['targets'] != list(targets) or frozen['units'] != {n: next(v.unit for v in specs if v.name == n) for n in frozen['units']}:
            raise ValueError('frozen panel/units differ from operational input')
        first, stop = index[0], index[-1] + data.schema.cadence
        exclusions = tuple((int(index.searchsorted(max(first, pd.Timestamp(a)))),
                            int(index.searchsorted(min(stop, pd.Timestamp(b)))))
                           for a, b in fit_exclusions if max(first, pd.Timestamp(a)) < min(stop, pd.Timestamp(b)))
        plan = make_shortlist_plan(aligned, frozen, target_gaps=selected, support_interval=support,
            fit_exclusions=exclusions, donor_segments={mapping[n]: tuple(tuple(x) for x in pairs)
                                                     for n, pairs in donor_domains.items() if n in mapping}, allow_provisional=allow_provisional)
        document = json.loads(plan.native_json)['plan']
        rejected = [g.get('reason') for g in document.get('gaps', []) if g.get('reason')]
        return aligned, plan, rejected[0] if rejected else None, ()
    cap = pd.Timedelta(seconds=params['max_gap_seconds'])
    if method == 'LIN':
        return data, make_plan(data, method='linear', mode='offline', max_gap=cap,
                              target_gaps=selected, support_interval=support), None, ()
    if method.startswith('T'):
        policy = ForecastPolicy(targets, 'offline', 'blend', cap, params['context_length'], params['min_support'],
            int(604800 / data.schema.cadence.total_seconds()),
            covariates=tuple(DynamicCovariate(n, index[-1], True) for n in donors),
            target_gaps=selected, support_interval=support, context_policy='adjacent_original', covariate_issue='archive_end')
        plan = make_forecast_plan(data, policy)
        reasons = [g.reason for g in plan.gaps if not g.eligible]
        return data, plan, reasons[0] if reasons else None, ()
    extended, features = data, ()
    if method.startswith('RF'):
        if method == 'RF-XL':
            extended, features = donor_lag_data(data, targets=targets, donors=donors,
                                               support_segments=donor_domains)
        raw, masks, specs = extended.raw, extended.masks, list(extended.schema.variables)
        inserted = masks['inserted'].all(axis=1)
        time_names = tuple('time::' + n for n in TIME.to_dict()['names'])
        for name, values in zip(time_names, TIME.compute(index).T):
            raw[name] = np.where(inserted, np.nan, values)
            specs.append(VariableSpec(name, 'dimensionless', 'external'))
            for key in ('observed', 'qc_rejected', 'structural', 'artificial', 'inserted'):
                masks[key][name] = ~inserted if key == 'observed' else inserted if key == 'inserted' else False
        extended = SeriesData(raw, schema=TimeSeriesSchema(data.schema.cadence, tuple(specs)),
            **{k: masks[k] for k in ('observed', 'qc_rejected', 'structural', 'artificial', 'inserted')})
        predictors = donors + tuple(f['name'] for f in features) + time_names
    else:
        predictors = donors
    gap = gaps[0]
    midpoint = index[gap['start_pos']] + (index[gap['stop_pos'] - 1] - index[gap['start_pos']]) / 2
    half = pd.Timedelta(days=params['window_days'] / 2)
    permitted = (index >= midpoint - half) & (index < midpoint + half)
    permitted &= (index >= pd.Timestamp(domain[0])) & (index < pd.Timestamp(domain[1]))
    for a, b in fit_exclusions:
        permitted &= ~((index >= pd.Timestamp(a)) & (index < pd.Timestamp(b)))
    qualified = extended.mask('donor')[[*targets, *predictors]].all(axis=1).to_numpy()
    rows = tuple(int(i) for i in _select(np.flatnonzero(permitted & qualified).tolist(), params['max_rows']))
    if len(rows) < params['min_rows']:
        return extended, None, 'insufficient_support', features
    a, b = support
    if method.startswith('RF'):
        rf = OperationRFParameters(tuple((k, params[k]) for k in ('n_estimators', 'max_depth', 'min_samples_leaf', 'n_jobs'))
                                   + (('random_state', params['seed']),))
        policy = DriverPolicy((RegressionStep(targets[0], predictors, 'rf', parameters=rf),), a, b,
                              params['min_rows'], cap, 'offline', 'flag', digest(recipe), fit_rows=rows,
                              target_gaps=selected)
        plan = make_driver_plan(extended, policy)
        rejected = [g['reason'] for g in plan.to_dict()['gaps'] if g['reason']]
        return extended, plan, rejected[0] if rejected else None, features
    policy = TimeIndexedPolicy(targets, ((a, b),), TIME, params['max_rows'], params['min_rows'], cap,
                               params['seed'], external_features=predictors, reference_rows=rows,
                               target_gaps=selected, support_interval=support)
    plan = make_time_indexed_plan(data, policy)
    rejected = [g.reason for g in plan.gaps if not g.eligible]
    return data, plan, rejected[0] if rejected else None, features


def execute_native(data, plan, recipe, backend):
    if 'frozen_recipe' in recipe:
        result = apply_shortlist(data, plan, backend=backend)
    elif recipe['method'] == 'LIN':
        result = execute(data, plan)
    elif recipe['method'].startswith('RF'):
        result = execute_driver_plan(data, plan)
    elif recipe['method'].startswith('P'):
        result = execute_time_indexed(data, plan, backend)
    else:
        result = execute_forecast(data, plan, backend)
    check_result(data, result)
    if result.plan != plan:
        raise ValueError('backend result plan drift')
    return result


def strict_blend(data, results):
    members = tuple(EnsembleMember(str(i), result.plan.to_dict()['method'], configuration_identity(result), .5)
                    for i, result in enumerate(results))
    training = next((plan_method_mode_training(r.plan)[2] for r in results
                     if plan_method_mode_training(r.plan)[2] is not None), None)
    return blend(data, {str(i): r for i, r in enumerate(results)},
                 EnsembleRecipe(members, 'offline', training=training, whole_gaps=True))
