"""Explicit observed-driver regression and declared derived reconstruction."""

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import pandas as pd

from .data import SeriesData
from .gaps import profile_gaps
from .result import ImputationResult


class ExternalInputs:
    """Caller-owned external observations with exact grid and availability evidence."""

    def __init__(self, frame, *, observed, units, source, available_as_of):
        if not isinstance(frame, pd.DataFrame) or not isinstance(observed, pd.DataFrame):
            raise ValueError("external values and observed mask must be DataFrames")
        if not frame.index.equals(observed.index) or not frame.columns.equals(observed.columns):
            raise ValueError("external observed mask must align exactly")
        if not all(pd.api.types.is_bool_dtype(dtype) for dtype in observed.dtypes) or observed.isna().to_numpy().any():
            raise ValueError("external observed mask must be nonmissing boolean")
        if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
               or pd.api.types.is_complex_dtype(dtype) for dtype in frame.dtypes):
            raise ValueError("external measurements must be real numeric")
        if not isinstance(source, str) or not source.strip() or not isinstance(units, dict) or set(units) != set(frame):
            raise ValueError("external source and all units must be declared")
        if any(not isinstance(unit, str) or not unit.strip() or unit == "unknown" for unit in units.values()):
            raise ValueError("external units must be resolved")
        if isinstance(available_as_of, dict):
            if set(available_as_of) != set(frame):
                raise ValueError("broadcast availability must name every external predictor")
            availability = pd.DataFrame({name: [available_as_of[name]] * len(frame)
                                         for name in frame}, index=frame.index)
        elif isinstance(available_as_of, pd.DataFrame):
            availability = available_as_of.copy(deep=True)
        else:
            raise ValueError("external availability must be an aligned table or broadcast declaration")
        if not availability.index.equals(frame.index) or not availability.columns.equals(frame.columns):
            raise ValueError("external availability must align exactly")
        if any(not isinstance(value, pd.Timestamp) or value.tz is None or pd.isna(value)
               for value in availability.to_numpy().flat):
            raise ValueError("external availability requires timezone-aware timestamps")
        values = frame.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any() or (observed.to_numpy() & ~np.isfinite(values)).any():
            raise ValueError("qualified external observations must be finite")
        self._frame = frame.copy(deep=True)
        self._observed = observed.astype(bool).copy(deep=True)
        self._availability = availability
        self.units = dict(units)
        self.source = source
        digest = hashlib.sha256(repr((source, sorted(units.items()))).encode())
        for table in (self._frame, self._observed, self._availability):
            digest.update(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes())
        self.fingerprint = digest.hexdigest()

    @property
    def frame(self):
        return self._frame.copy(deep=True)

    @property
    def observed(self):
        return self._observed.copy(deep=True)

    @property
    def availability(self):
        return self._availability.copy(deep=True)


@dataclass(frozen=True)
class RegressionStep:
    target: str
    predictors: tuple[str, ...]
    estimator: str = "ols"
    allow_reconstructed: tuple[str, ...] = ()


@dataclass(frozen=True)
class DerivedStep:
    output: str
    parents: tuple[str, ...]
    parent_units: tuple[str, ...]
    output_unit: str
    formula_id: str
    formula: object
    allow_reconstructed: tuple[str, ...] = ()


@dataclass(frozen=True)
class DriverPolicy:
    steps: tuple[RegressionStep | DerivedStep, ...]
    training_start: int
    training_stop: int
    min_overlap: int
    max_gap: pd.Timedelta
    mode: str
    novelty: str
    source_identity: str


@dataclass(frozen=True)
class DriverPlan:
    data_fingerprint: str
    external_fingerprint: str | None
    policy: DriverPolicy
    gaps: tuple[tuple[str, int, int, str | None], ...]

    def to_dict(self):
        return {"data_fingerprint": self.data_fingerprint,
                "external_fingerprint": self.external_fingerprint,
                "method": "driver_pipeline", "mode": self.policy.mode,
                "training_interval": [self.policy.training_start, self.policy.training_stop],
                "min_overlap": self.policy.min_overlap,
                "max_gap_seconds": self.policy.max_gap.total_seconds(),
                "novelty": self.policy.novelty, "source_identity": self.policy.source_identity,
                "steps": [{"kind": "regression", "target": step.target,
                           "predictors": list(step.predictors), "estimator": step.estimator,
                           "allow_reconstructed": list(step.allow_reconstructed)}
                          if isinstance(step, RegressionStep) else
                          {"kind": "derived", "output": step.output,
                           "parents": list(step.parents), "parent_units": list(step.parent_units),
                           "output_unit": step.output_unit, "formula_id": step.formula_id,
                           "allow_reconstructed": list(step.allow_reconstructed)}
                          for step in self.policy.steps],
                "gaps": [{"variable": name, "start_pos": start, "stop_pos": stop,
                          "reason": reason} for name, start, stop, reason in self.gaps]}


def _validate(data, external, policy):
    if not isinstance(data, SeriesData) or not isinstance(policy, DriverPolicy):
        raise ValueError("driver pipeline requires SeriesData and DriverPolicy")
    if external is not None and not isinstance(external, ExternalInputs):
        raise ValueError("external predictors require ExternalInputs")
    if external is not None and not external.frame.index.equals(data.raw.index):
        raise ValueError("external grid and cadence must align exactly")
    if policy.mode not in {"offline", "causal"} or policy.novelty not in {"flag", "abstain"}:
        raise ValueError("invalid mode or novelty policy")
    if not isinstance(policy.source_identity, str) or not policy.source_identity.strip():
        raise ValueError("source identity is required")
    if (type(policy.training_start) is not int or type(policy.training_stop) is not int
            or not 0 <= policy.training_start < policy.training_stop <= len(data.raw)
            or type(policy.min_overlap) is not int or policy.min_overlap < 1):
        raise ValueError("invalid training interval or minimum overlap")
    if not isinstance(policy.max_gap, pd.Timedelta) or pd.isna(policy.max_gap) or policy.max_gap <= pd.Timedelta(0):
        raise ValueError("max_gap must be positive")
    if not isinstance(policy.steps, tuple) or not policy.steps:
        raise ValueError("pipeline requires ordered steps")
    schema = {item.name: item for item in data.schema.variables}
    external_names = set() if external is None else set(external.frame)
    if set(schema) & external_names:
        raise ValueError("external and internal predictor names must be distinct")
    produced = set()
    for step in policy.steps:
        if isinstance(step, RegressionStep):
            output, parents = step.target, step.predictors
            if step.estimator not in {"ols", "rf", "boosting"}:
                raise ValueError("unknown regression estimator")
        elif isinstance(step, DerivedStep):
            output, parents = step.output, step.parents
            check_derived_step(step, schema, lambda name: schema[name].unit if name in schema
                               else external.units.get(name) if external else None)
        else:
            raise ValueError("unknown pipeline step")
        if output not in schema or output in produced or not isinstance(parents, tuple) or not parents or len(set(parents)) != len(parents):
            raise ValueError("invalid or duplicate pipeline output/predictors")
        if isinstance(step, DerivedStep) and tuple(schema[output].derived_from) != parents:
            raise ValueError("derived step disagrees with declared lineage")
        if any(name not in schema and name not in external_names for name in parents):
            raise ValueError("predictor is absent from declared inputs")
        if any(name == output or _descends(schema, name, output) for name in parents if name in schema):
            raise ValueError("target, descendant or alias cannot be a predictor")
        if (not isinstance(step.allow_reconstructed, tuple)
                or any(name not in parents or name not in produced for name in step.allow_reconstructed)):
            raise ValueError("reconstructed donor requires an upstream explicit opt-in")
        produced.add(output)


def check_derived_step(step, schema, unit_of):
    """Validate a frozen DerivedStep's output, identity, callable and exact units."""
    output, parents = step.output, step.parents
    if (output not in schema or len(step.parent_units) != len(parents) or not step.formula_id
            or not callable(step.formula) or step.output_unit != schema[output].unit):
        raise ValueError("derived formula, identity or output unit invalid")
    for name, unit in zip(parents, step.parent_units):
        if unit_of(name) != unit:
            raise ValueError("derived parent unit mismatch")


def evaluate_derived(step, arrays):
    """Apply a frozen formula to aligned 1-D parent arrays; reject malformed output."""
    try:
        output_values = np.asarray(step.formula(*arrays))
    except Exception as exc:
        raise RuntimeError(f"derived formula failed for {step.output}") from exc
    if (output_values.shape != np.shape(arrays[0]) or output_values.dtype.kind not in "iuf"
            or not np.isfinite(output_values).all()):
        raise RuntimeError("derived formula returned wrong shape or non-real/nonfinite values")
    return output_values


def _descends(schema, name, target):
    return target in schema[name].derived_from or any(
        _descends(schema, parent, target) for parent in schema[name].derived_from)


def make_driver_plan(data: SeriesData, policy: DriverPolicy,
                     external: ExternalInputs | None = None) -> DriverPlan:
    _validate(data, external, policy)
    outputs = {step.target if isinstance(step, RegressionStep) else step.output
               for step in policy.steps}
    structural = data.mask("structural")
    gaps = tuple((gap.variable, gap.start_pos, gap.stop_pos,
                  "structural_absence" if structural[gap.variable].iloc[gap.start_pos:gap.stop_pos].any()
                  else "gap_exceeds_limit" if gap.duration > policy.max_gap else None)
                 for gap in profile_gaps(data) if gap.variable in outputs)
    return DriverPlan(data.fingerprint, None if external is None else external.fingerprint,
                      policy, gaps)


def _fit_predict(estimator, x_train, y_train, x_predict):
    if estimator == "ols":
        coefficients = np.linalg.lstsq(np.column_stack((np.ones(len(x_train)), x_train)),
                                       y_train, rcond=None)[0]
        return np.column_stack((np.ones(len(x_predict)), x_predict)) @ coefficients, {
            "intercept": float(coefficients[0]), "coefficients": coefficients[1:].tolist()}
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    except ImportError as exc:
        raise RuntimeError("rf/boosting requires the installed scikit-learn research extra") from exc
    model = (RandomForestRegressor(n_estimators=32, max_depth=6, min_samples_leaf=2,
                                   n_jobs=1, random_state=0) if estimator == "rf" else
             HistGradientBoostingRegressor(max_iter=32, max_leaf_nodes=15,
                 learning_rate=.1, l2_regularization=0, early_stopping=False,
                 random_state=0))
    model.fit(x_train, y_train)
    return model.predict(x_predict), {"parameters": model.get_params()}


def execute_driver_plan(data: SeriesData, plan: DriverPlan,
                        external: ExternalInputs | None = None) -> ImputationResult:
    if not isinstance(plan, DriverPlan):
        raise ValueError("driver plan is stale or edited; rebuild before execution")
    try:
        expected = make_driver_plan(data, plan.policy, external)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("driver plan is stale or edited; rebuild before execution") from exc
    if plan != expected:
        raise ValueError("driver plan is stale or edited; rebuild before execution")
    policy = plan.policy
    available = data.available
    completed = available.copy(deep=True)
    index, columns = completed.index, completed.columns
    states = pd.DataFrame("unavailable", index=index, columns=columns, dtype=object)
    for mask, label in (("raw_present", "unqualified"), ("absent", "absent"),
                        ("inserted", "absent_timestamp"), ("structural", "structural"),
                        ("qc_rejected", "qc_rejected"), ("observed", "observed"),
                        ("artificial", "artificial_hidden")):
        states = states.mask(data.mask(mask), label)
    methods = pd.DataFrame(None, index=index, columns=columns, dtype=object)
    reasons = pd.DataFrame(None, index=index, columns=columns, dtype=object)
    imputed = pd.DataFrame(False, index=index, columns=columns)
    reconstructed_as_of = pd.DataFrame(None, index=index, columns=columns, dtype=object)
    records = []
    external_values = None if external is None else external.frame
    external_observed = None if external is None else external.observed
    availability = None if external is None else external.availability
    qualified = data.mask("donor")

    def donor(name, row, issue, allow_reconstructed, *, training=False):
        if name in columns:
            if policy.mode != "causal" or index[row] <= issue:
                value = available.at[index[row], name]
                if pd.notna(value) and np.isfinite(float(value)):
                    return float(value), "observed", index[row]
            if not training and name in allow_reconstructed and imputed.at[index[row], name]:
                issued = reconstructed_as_of.at[index[row], name]
                if policy.mode != "causal" or issued <= issue:
                    return float(completed.at[index[row], name]), "reconstructed", issued
        elif external_values is not None and bool(external_observed.at[index[row], name]):
            value = external_values.at[index[row], name]
            if availability.at[index[row], name] <= issue and pd.notna(value) and np.isfinite(float(value)):
                return float(value), "external_observed", availability.at[index[row], name]
        return None, "unavailable", None

    for step in policy.steps:
        output = step.target if isinstance(step, RegressionStep) else step.output
        selected = [(name, start, stop, reason) for name, start, stop, reason in plan.gaps
                    if name == output]
        if isinstance(step, DerivedStep):
            for _, start, stop, reason in selected:
                for row in range(start, stop):
                    stamp = index[row]
                    if reason is not None:
                        reasons.at[stamp, output] = reason
                        continue
                    inputs = [donor(name, row, stamp if policy.mode == "offline" else index[start],
                                    step.allow_reconstructed) for name in step.parents]
                    if any(value is None for value, _, _ in inputs):
                        reasons.at[stamp, output] = "required_parent_unavailable"
                        continue
                    output_values = evaluate_derived(step, [np.array([value]) for value, _, _ in inputs])
                    completed.at[stamp, output] = float(output_values[0])
                    states.at[stamp, output] = "imputed"
                    methods.at[stamp, output] = "derived:" + step.formula_id
                    imputed.at[stamp, output] = True
                    issued = max(index[start], *(when for _, _, when in inputs))
                    reconstructed_as_of.at[stamp, output] = issued
                    records.append({"timestamp": stamp.isoformat(), "target": output,
                                    "method": "derived", "formula_id": step.formula_id,
                                    "donor_origin": dict(zip(step.parents, [origin for _, origin, _ in inputs])),
                                    "donor_available_as_of": dict(zip(step.parents, [when.isoformat() for _, _, when in inputs])),
                                    "available_as_of": issued.isoformat()})
            continue
        for _, start, stop, reason in selected:
            issue = index[start]
            if policy.mode == "causal" and policy.training_stop > start:
                raise ValueError("causal training interval must end before gap issue")
            if reason is not None:
                reasons.iloc[start:stop, columns.get_loc(output)] = reason
                continue
            training = []
            labels = []
            for row in range(policy.training_start, min(policy.training_stop, start) if policy.mode == "causal" else policy.training_stop):
                if not qualified.at[index[row], output]:
                    continue
                inputs = [donor(name, row, issue if policy.mode == "causal" else index[-1], (), training=True)
                          for name in step.predictors]
                if all(value is not None for value, _, _ in inputs):
                    training.append([value for value, _, _ in inputs])
                    labels.append(float(available.at[index[row], output]))
            if len(training) < policy.min_overlap:
                reasons.iloc[start:stop, columns.get_loc(output)] = "insufficient_observed_overlap"
                continue
            x_train = np.asarray(training, dtype=float)
            y_train = np.asarray(labels, dtype=float)
            means, stds = x_train.mean(axis=0), x_train.std(axis=0)
            safe_stds = np.where(stds == 0, 1.0, stds)
            minima, maxima = x_train.min(axis=0), x_train.max(axis=0)
            pending = []
            for row in range(start, stop):
                stamp = index[row]
                inputs = [donor(name, row, issue if policy.mode == "causal" else index[-1],
                                step.allow_reconstructed) for name in step.predictors]
                if any(value is None for value, _, _ in inputs):
                    reasons.at[stamp, output] = "required_predictor_unavailable"
                    continue
                values = np.array([value for value, _, _ in inputs], dtype=float)
                novel = [name for name, value, lo, hi in zip(step.predictors, values, minima, maxima)
                         if value < lo or value > hi]
                if novel and policy.novelty == "abstain":
                    reasons.at[stamp, output] = "outside_training_support"
                    continue
                pending.append((row, values, inputs, novel))
            if not pending:
                continue
            try:
                prediction, parameters = _fit_predict(step.estimator,
                    (x_train - means) / safe_stds, y_train,
                    (np.array([item[1] for item in pending]) - means) / safe_stds)
            except Exception as exc:
                if isinstance(exc, RuntimeError) and "scikit-learn research extra" in str(exc):
                    raise
                raise RuntimeError(f"{step.estimator} fit/predict failed for {output}") from exc
            result = np.asarray(prediction)
            if (result.shape != (len(pending),) or result.dtype.kind not in "iuf"
                    or not np.isfinite(result).all()):
                raise RuntimeError("estimator returned wrong shape or non-real/nonfinite values")
            for (row, values, inputs, novel), value in zip(pending, result):
                stamp = index[row]
                completed.at[stamp, output] = float(value)
                states.at[stamp, output] = "imputed"
                methods.at[stamp, output] = "regression:" + step.estimator
                imputed.at[stamp, output] = True
                issued = max(issue, index[policy.training_stop - 1],
                             *(when for _, _, when in inputs))
                reconstructed_as_of.at[stamp, output] = issued
                records.append({"timestamp": stamp.isoformat(), "target": output,
                                "method": step.estimator, "predictors": list(step.predictors),
                                "donor_origin": dict(zip(step.predictors, [origin for _, origin, _ in inputs])),
                                "donor_available_as_of": dict(zip(step.predictors, [when.isoformat() for _, _, when in inputs])),
                                "available_as_of": issued.isoformat(),
                                "training_interval": [policy.training_start, policy.training_stop],
                                "training_overlap": len(training),
                                "training_mean": means.tolist(), "training_std": stds.tolist(),
                                "training_min": minima.tolist(), "training_max": maxima.tolist(),
                                "parameters": parameters, "novel_features": novel})
    identity = {"kind": "driver_pipeline", "source_identity": policy.source_identity,
                "external_source": None if external is None else external.source,
                "external_units": None if external is None else external.units,
                "steps": plan.to_dict()["steps"], "cells": records}
    return ImputationResult(data, plan, completed, states, methods, reasons, imputed,
                            backend_identity=identity)


def rain_amount_from_rate(rate, cadence, *, rate_unit, amount_unit):
    if rate_unit != "mm/h" or amount_unit != "mm" or not isinstance(cadence, pd.Timedelta) or cadence <= pd.Timedelta(0):
        raise ValueError("rain conversion requires mm/h, mm and positive interval duration")
    return np.asarray(rate, dtype=float) * cadence.total_seconds() / 3600


def circular_degrees_from_components(sine, cosine):
    return np.mod(np.degrees(np.arctan2(sine, cosine)), 360)


def energy_residual(rn, h, le, g, *, units):
    if not isinstance(units, tuple) or len(units) != 4 or len(set(units)) != 1 or not units[0]:
        raise ValueError("energy residual requires the same declared unit")
    return np.asarray(rn) - np.asarray(h) - np.asarray(le) - np.asarray(g)
