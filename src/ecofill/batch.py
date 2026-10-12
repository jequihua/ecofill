"""Portable offline whole-gap routing over immutable qualified observations.

Source adapters declare metadata and masks. This module never discovers units,
loads checkpoints, imports a campaign, or promotes a reconstruction to truth.
"""

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SeriesData
from .drivers import (DriverPolicy, OptionalBackendUnavailable, RegressionParameters,
                      RegressionStep, execute_driver_plan, make_driver_plan)
from .forecast import ForecastPolicy, execute_forecast, make_forecast_plan
from .gaps import profile_gaps
from .methods import execute
from .planning import StateSpaceSpec, make_plan
from .result import ImputationResult
from .schema import TimeSeriesSchema, VariableSpec


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValueError(f"expected exactly {sorted(required)} with optional {sorted(optional)}")


def _text(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("nonempty explicit text is required")
    if "\\" in value or value.startswith("/") or (len(value) > 1 and value[1] == ":"):
        raise ValueError("configuration identities and evidence must be portable, not host paths")


def _number(value, *, minimum=0, inclusive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (value < minimum if inclusive else value <= minimum):
        raise ValueError("invalid finite numeric limit")


def _parameters(method, parameters):
    if method in {"linear", "persistence", "next"}:
        _keys(parameters, ())
    elif method == "seasonal":
        _keys(parameters, ("period_seconds", "search_periods"))
        _number(parameters["period_seconds"])
        if type(parameters["search_periods"]) is not int or parameters["search_periods"] < 1:
            raise ValueError("search_periods must be a positive integer")
    elif method in {"state_filter", "state_smoother"}:
        _keys(parameters, ("process_variance", "observation_variance", "initial_mean", "initial_variance"))
        if any(type(value) not in (int, float) for value in parameters.values()):
            raise ValueError("state parameters must be real numbers")
        StateSpaceSpec(**parameters)
    elif method in {"ols", "rf", "boosting"}:
        if not isinstance(parameters, dict):
            raise ValueError("regression parameters must be an object")
        defaults = RegressionParameters().effective(method)
        if set(parameters) - set(defaults):
            raise ValueError("unknown regression parameter")
        # Canonical configs carry ALL effective parameters. Only the existing
        # bounded overrides may differ from the pinned defaults.
        changed = tuple((key, value) for key, value in parameters.items()
                        if key not in defaults or value != defaults[key])
        effective = RegressionParameters(changed).effective(method)
        if any(type(value) is not type(defaults[key]) for key, value in parameters.items()
               if key in defaults and value == defaults[key]):
            raise ValueError("regression parameter type differs from its exact contract")
        return effective
    elif method == "forecast":
        _keys(parameters, ("direction", "context_length", "min_support", "max_horizon"))
        if parameters["direction"] not in {"forward", "backward", "blend"}:
            raise ValueError("invalid forecast direction")
        if any(type(parameters[key]) is not int or parameters[key] < 1
               for key in ("context_length", "min_support", "max_horizon")):
            raise ValueError("forecast lengths must be positive integers")
        if parameters["min_support"] > parameters["context_length"]:
            raise ValueError("forecast minimum exceeds context length")
    else:
        raise ValueError(f"unsupported batch method: {method}")
    return parameters


@dataclass(frozen=True)
class BatchConfig:
    """Validated canonical JSON; use load_batch_config to construct it."""

    canonical_json: str

    @property
    def identity(self):
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self.canonical_json)


def load_batch_config(source) -> BatchConfig:
    """Validate JSON or a mapping. Unknown keys and implicit defaults fail closed."""
    if isinstance(source, BatchConfig):
        source = source.to_dict()
    elif isinstance(source, (str, Path)):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result
        source = json.loads(Path(source).read_text(encoding="utf-8"), object_pairs_hook=unique)
    value = json.loads(_json(source))
    _keys(value, ("schema", "identity", "input", "mode", "support", "status", "evidence", "fitting", "routes"),
          ("require_support_domains",))
    if type(value.setdefault("require_support_domains", False)) is not bool:
        raise ValueError("require_support_domains must be an explicit boolean")
    if value["schema"] != "ecofill.batch/1" or value["mode"] != "offline" or value["support"] != "original_only":
        raise ValueError("batch/1 supports explicit offline original_only reconstruction")
    _keys(value["identity"], ("dataset", "instrument", "variable", "unit_regime"))
    for text in value["identity"].values():
        _text(text)
    contract = value["input"]
    _keys(contract, ("cadence_seconds", "clock", "variables", "source_policy"))
    _number(contract["cadence_seconds"])
    if contract["clock"] != "UTC":
        raise ValueError("batch clock must be UTC")
    _keys(contract["source_policy"], ("qc", "sentinel", "structural"))
    for text in contract["source_policy"].values():
        _text(text)
    if not isinstance(contract["variables"], list) or not contract["variables"]:
        raise ValueError("ordered input variables are required")
    variables = []
    for variable in contract["variables"]:
        _keys(variable, ("name", "unit", "role", "derived_from"))
        _text(variable["name"])
        _text(variable["unit"])
        if not isinstance(variable["derived_from"], list):
            raise ValueError("lineage must be an ordered list")
        variables.append(VariableSpec(variable["name"], variable["unit"], variable["role"], tuple(variable["derived_from"])))
    # Reuse the core schema validator, including lineage and role validation.
    schema = TimeSeriesSchema(pd.Timedelta(seconds=contract["cadence_seconds"]), tuple(variables))
    frame = pd.DataFrame([[0.] * len(variables)], columns=[item.name for item in variables],
                         index=pd.date_range("2000-01-01", periods=1, tz="UTC"))
    SeriesData(frame, schema=schema, observed=frame.notna())
    target = value["identity"]["variable"]
    names = {variable.name for variable in variables}
    if target not in names:
        raise ValueError("identity variable is absent from input schema")
    if value["status"] not in {"usable", "provisional", "auxiliary", "reserved", "unsupported"}:
        raise ValueError("invalid deployment status")
    if not isinstance(value["evidence"], list) or not value["evidence"]:
        raise ValueError("at least one evidence reference is required")
    for reference in value["evidence"]:
        _text(reference)
    fitting = value["fitting"]
    _keys(fitting, ("window_days", "max_rows", "min_rows", "selection"))
    if (fitting["selection"] != "even_time_coverage" or fitting["window_days"] != 180
            or type(fitting["max_rows"]) is not int or not 168 <= fitting["max_rows"] <= 4096
            or type(fitting["min_rows"]) is not int or not 168 <= fitting["min_rows"] <= fitting["max_rows"]):
        raise ValueError("fitting requires centered 180 days, cap<=4096 and minimum>=168")
    if not isinstance(value["routes"], list):
        raise ValueError("routes must be ordered")
    route_ids = set()
    lineage = {variable.name: variable.derived_from for variable in variables}
    def descendant(name):
        return name == target or any(descendant(parent) for parent in lineage[name])
    for route in value["routes"]:
        _keys(route, ("id", "min_gap_seconds", "max_gap_seconds", "geometries", "status",
                      "tested_durations_seconds", "duration_support", "stages"))
        _text(route["id"])
        if route["id"] in route_ids:
            raise ValueError("duplicate route id")
        route_ids.add(route["id"])
        _number(route["min_gap_seconds"], inclusive=True)
        _number(route["max_gap_seconds"], minimum=route["min_gap_seconds"])
        if route["max_gap_seconds"] > 168 * 3600:
            raise ValueError("batch routes cannot exceed the 168 hour whole-gap ceiling")
        if (not isinstance(route["geometries"], list) or not route["geometries"]
                or len(set(route["geometries"])) != len(route["geometries"])
                or set(route["geometries"]) - {"bounded", "leading", "trailing", "full"}):
            raise ValueError("explicit unique allowed geometries are required")
        if route["status"] not in {"usable", "provisional"}:
            raise ValueError("route status must be usable or provisional")
        durations = route["tested_durations_seconds"]
        if not isinstance(durations, list) or not durations or durations != sorted(set(durations)):
            raise ValueError("tested durations must be sorted unique positive values")
        for duration in durations:
            _number(duration)
            if pd.Timedelta(seconds=duration).value % schema.cadence.value:
                raise ValueError("tested duration must be an exact cadence multiple")
        if route["duration_support"] not in {"tested_only", "interpolated"}:
            raise ValueError("unvalidated duration extrapolation is unsupported")
        if route["max_gap_seconds"] > max(durations):
            raise ValueError("route duration ceiling exceeds tested support")
        if not isinstance(route["stages"], list) or not route["stages"]:
            raise ValueError("route requires ordered stages/fallbacks")
        for stage in route["stages"]:
            _keys(stage, ("method", "parameters"), ("predictors", "backend", "max_gap_seconds"))
            method = stage["method"]
            stage["parameters"] = _parameters(method, stage["parameters"])
            if method == "seasonal" and pd.Timedelta(seconds=stage["parameters"]["period_seconds"]).value % schema.cadence.value:
                raise ValueError("seasonal period must be an exact cadence multiple")
            if method in {"ols", "rf", "boosting"}:
                predictors = stage.get("predictors")
                if (not isinstance(predictors, list) or not 1 <= len(predictors) <= 3
                        or len(set(predictors)) != len(predictors)
                        or any(name not in names or descendant(name) for name in predictors)):
                    raise ValueError("regression requires <=3 distinct non-descendant observed predictors")
            elif "predictors" in stage:
                raise ValueError("predictors belong only to regression stages")
            if method == "forecast":
                _text(stage.get("backend"))
                if stage["parameters"]["max_horizon"] > 336:
                    raise ValueError("forecast horizon exceeds 336 slots")
            elif "backend" in stage:
                raise ValueError("backend belongs only to forecast stages")
            cap = stage.setdefault("max_gap_seconds", route["max_gap_seconds"])
            _number(cap)
            if cap > route["max_gap_seconds"]:
                raise ValueError("stage cap exceeds route cap")
    return BatchConfig(_json(value))


@dataclass(frozen=True)
class BatchInput:
    data: SeriesData
    contract_json: str
    reasons_json: str
    inventory_json: str
    data_fingerprint: str
    original_rows: tuple[str, ...]


def prepare_batch(frame, *, config, metadata, observed, qc_rejected=None, structural=None,
                  artificial=None, inserted=None, sentinel_names=None) -> BatchInput:
    """Regularize exact UTC cadence without smoothing; preserve all supplied rows.

    metadata must equal {"identity": config.identity document, "input": config.input}.
    sentinel_names is an optional aligned text table; named cells must be QC rejected.
    """
    config = load_batch_config(config)
    document = config.to_dict()
    contract = {key: document[key] for key in ("identity", "input")}
    if metadata != contract:
        raise ValueError("input identity, schema, units, cadence or source policy drift")
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("source values require a timestamp-indexed DataFrame")
    index = frame.index
    if index.tz is None or any(stamp.utcoffset().total_seconds() != 0 for stamp in index):
        raise ValueError("input timestamps must explicitly use UTC")
    if not len(index) or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("source timestamps must be finite, unique and increasing; duplicates are ambiguous")
    cadence = pd.Timedelta(seconds=contract["input"]["cadence_seconds"])
    if ((index.asi8 - index.asi8[0]) % cadence.value).any():
        raise ValueError("source timestamp phase/cadence drift; no resampling is performed")
    grid = pd.date_range(index[0], index[-1], freq=cadence).tz_convert("UTC")
    original_rows = tuple(stamp.isoformat() for stamp in index)
    def aligned(name, mask, *, labels=False):
        if mask is None:
            return pd.DataFrame(None if labels else False, index=grid, columns=frame.columns)
        if (not isinstance(mask, pd.DataFrame) or not mask.index.equals(index)
                or not mask.columns.equals(frame.columns)):
            raise ValueError(f"{name} must align exactly to source rows and columns")
        if not labels and (not all(pd.api.types.is_bool_dtype(dtype) for dtype in mask.dtypes)
                           or mask.isna().to_numpy().any()):
            raise ValueError(f"{name} must be a nonmissing boolean mask")
        return mask.reindex(grid, fill_value=None if labels else False)
    if observed is None:
        raise ValueError("explicit observed mask is required")
    masks = {name: aligned(name, value) for name, value in
             (("observed", observed), ("qc_rejected", qc_rejected), ("structural", structural),
              ("artificial", artificial), ("inserted", inserted))}
    masks["inserted"].loc[~grid.isin(index), :] = True
    sentinels = aligned("sentinel_names", sentinel_names, labels=True)
    for value in sentinels.to_numpy().flat:
        if pd.notna(value):
            _text(value)
    if (sentinels.notna() & ~masks["qc_rejected"]).to_numpy().any():
        raise ValueError("a named sentinel must also be explicitly QC rejected")
    variables = tuple(VariableSpec(item["name"], item["unit"], item["role"], tuple(item["derived_from"]))
                      for item in contract["input"]["variables"])
    regularized = frame.copy(deep=True)
    if len(grid) != len(index):
        # pandas' ordinary integer reindex uses float storage for inserted NaN,
        # which rounds original integers above 2**53. Nullable integers retain
        # exact source bytes in their numeric representation.
        for name, dtype in regularized.dtypes.items():
            if pd.api.types.is_integer_dtype(dtype):
                regularized[name] = regularized[name].astype("UInt64" if pd.api.types.is_unsigned_integer_dtype(dtype) else "Int64")
    data = SeriesData(regularized.reindex(grid), schema=TimeSeriesSchema(cadence, variables), **masks)
    reasons = []
    masks_array = {name: table.to_numpy() for name, table in data._masks.items()}
    sentinel_array = sentinels.to_numpy()
    labels_for_masks = (("absent", "original_missing"), ("inserted", "inserted_row"),
                        ("qc_rejected", "qc_rejection"), ("structural", "structural_absence"),
                        ("artificial", "artificial_hiding"), ("donor", "qualified_observation"))
    for row, stamp in enumerate(grid):
        timestamp = stamp.isoformat()
        for column, name in enumerate(frame.columns):
            labels = []
            for mask, label in labels_for_masks:
                if masks_array[mask][row, column]:
                    labels.append(label)
            if pd.notna(sentinel_array[row, column]):
                labels.append("named_sentinel:" + sentinel_array[row, column])
            reasons.append([timestamp, name, labels or ["unqualified_raw"]])
    inventory = []
    for gap in profile_gaps(data):
        if gap.variable != contract["identity"]["variable"]:
            continue
        masks_record = {name: table[gap.variable].iloc[gap.start_pos:gap.stop_pos].tolist()
                        for name, table in data._masks.items()}
        record = {"variable": gap.variable, "start": gap.start.isoformat(), "end": gap.end.isoformat(),
                  "start_pos": gap.start_pos, "stop_pos": gap.stop_pos, "n_steps": gap.n_steps,
                  "duration_seconds": gap.duration.total_seconds(), "geometry": gap.geometry,
                  "structural": any(masks_record["structural"])}
        record["gap_id"] = _digest({"identity": contract["identity"], "start": record["start"],
                                    "end": record["end"], "geometry": gap.geometry, "masks": masks_record})
        record["duration_class"] = ("short" if gap.duration <= pd.Timedelta("2h") else
                                    "medium" if gap.duration <= pd.Timedelta("12h") else
                                    "long" if gap.duration <= pd.Timedelta("168h") else "beyond_limit")
        inventory.append(record)
    return BatchInput(data, _json(contract), _json(reasons), _json(inventory), data.fingerprint, original_rows)


def inventory_batch(prepared):
    if not isinstance(prepared, BatchInput) or prepared.data.fingerprint != prepared.data_fingerprint:
        raise ValueError("prepared input identity changed")
    return tuple(json.loads(prepared.inventory_json))


def _intervals(intervals):
    answer = []
    for pair in intervals:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("fit domains/exclusions require half-open timestamp pairs")
        start, stop = (pd.Timestamp(value) for value in pair)
        if pd.isna(start) or pd.isna(stop) or start.tz is None or stop.tz is None or start >= stop:
            raise ValueError("fit intervals must be increasing and timezone aware")
        answer.append((start.tz_convert("UTC").isoformat(), stop.tz_convert("UTC").isoformat()))
    if answer != sorted(set(answer)):
        raise ValueError("fit intervals must be sorted and unique")
    return tuple(answer)


def _fit_rows(data, document, gap, stage, domains, exclusions, support_interval=None):
    index = data.raw.index
    midpoint = pd.Timestamp(gap["start"]) + (pd.Timestamp(gap["end"]) - pd.Timestamp(gap["start"])) / 2
    half = pd.Timedelta(days=document["fitting"]["window_days"] / 2)
    permitted = np.asarray((index >= midpoint - half) & (index < midpoint + half))
    if support_interval is not None:
        support = np.zeros(len(index), dtype=bool)
        support[slice(*support_interval)] = True
        permitted &= support
    if domains:
        # Do not bridge retrieval/unit regimes: use only the one domain holding
        # the whole gap, even if another valid domain lies inside 180 days.
        containing = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in domains
                      if pd.Timestamp(a) <= pd.Timestamp(gap["start"])
                      and pd.Timestamp(gap["end"]) < pd.Timestamp(b)]
        if len(containing) > 1:
            raise ValueError("overlapping fit domains make the gap regime ambiguous")
        permitted &= (np.asarray((index >= containing[0][0]) & (index < containing[0][1]))
                      if containing else False)
    for start, stop in exclusions:
        permitted &= ~np.asarray((index >= pd.Timestamp(start)) & (index < pd.Timestamp(stop)))
    names = [gap["variable"], *stage["predictors"]]
    qualified = data.mask("donor")[names].all(axis=1).to_numpy()
    candidates = np.flatnonzero(permitted & qualified)
    cap = document["fitting"]["max_rows"]
    if len(candidates) > cap:
        # Equally spaced elapsed-time positions, nearest qualified row; ties
        # resolve left. Unique results can be below the cap, never above it.
        desired = np.linspace(candidates[0], candidates[-1], cap)
        right = np.searchsorted(candidates, desired).clip(0, len(candidates) - 1)
        left = np.maximum(0, right - 1)
        choose = np.where(abs(candidates[left] - desired) <= abs(candidates[right] - desired), left, right)
        candidates = np.unique(candidates[choose])
    return tuple(int(row) for row in candidates)


def _support_interval(data, gap, domains):
    if not domains:
        return None
    containing = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in domains
                  if pd.Timestamp(a) <= pd.Timestamp(gap["start"])
                  and pd.Timestamp(gap["end"]) < pd.Timestamp(b)]
    if len(containing) > 1:
        raise ValueError("overlapping support domains make the gap regime ambiguous")
    if not containing:
        return ()
    index = data.raw.index
    return (int(index.searchsorted(containing[0][0])), int(index.searchsorted(containing[0][1])))


def _native_plan(data, document, gap, stage, domains, exclusions, support_domains=()):
    selected = ((gap["variable"], gap["start_pos"], gap["stop_pos"]),)
    method, params = stage["method"], stage["parameters"]
    cap = pd.Timedelta(seconds=stage["max_gap_seconds"])
    support_interval = _support_interval(data, gap, support_domains)
    if method in {"ols", "rf", "boosting"}:
        defaults = RegressionParameters().effective(method)
        overrides = RegressionParameters(tuple((key, value) for key, value in params.items() if value != defaults[key]))
        rows = _fit_rows(data, document, gap, stage, domains, exclusions, support_interval)
        policy = DriverPolicy((RegressionStep(gap["variable"], tuple(stage["predictors"]), method,
                                              parameters=overrides),), 0, len(data.raw),
                              document["fitting"]["min_rows"], cap, "offline", "abstain",
                              _digest(document["identity"]), fit_rows=rows, target_gaps=selected)
        plan = make_driver_plan(data, policy)
        reason = next(reason for name, start, stop, reason in plan.gaps if (name, start, stop) == selected[0])
        if reason is None and len(rows) < policy.min_overlap:
            reason = "insufficient_observed_overlap"
    elif method == "forecast":
        policy = ForecastPolicy((gap["variable"],), "offline", max_gap=cap, target_gaps=selected,
                                support_interval=support_interval, **params)
        plan = make_forecast_plan(data, policy)
        reason = next(item.reason for item in plan.gaps
                      if (item.start_pos, item.stop_pos) == (gap["start_pos"], gap["stop_pos"]))
    else:
        kwargs = {}
        if method == "seasonal":
            kwargs = {"period": pd.Timedelta(seconds=params["period_seconds"]), "search_periods": params["search_periods"]}
        elif method.startswith("state_"):
            kwargs = {"state": StateSpaceSpec(**params)}
        plan = make_plan(data, method=method, mode="offline", max_gap=cap, target_gaps=selected,
                         support_interval=support_interval, **kwargs)
        reason = next(item.reason for item in plan.gaps
                      if (item.gap.variable, item.gap.start_pos, item.gap.stop_pos) == selected[0])
    return plan, reason


@dataclass(frozen=True)
class BatchPlan:
    data_fingerprint: str
    config: BatchConfig
    config_fingerprint: str
    records_json: str
    target_gap_ids: tuple[str, ...] | None
    fit_domains: tuple[tuple[str, str], ...]
    fit_exclusions: tuple[tuple[str, str], ...]
    support_domains: tuple[tuple[str, str], ...]
    allow_provisional: bool
    prepared_identity: str

    @property
    def identity(self):
        return _digest(self.to_dict())

    def to_dict(self):
        return {"method": "batch", "mode": "offline", "data_fingerprint": self.data_fingerprint,
                "config_identity": self.config_fingerprint, "config": self.config.to_dict(),
                "gaps": json.loads(self.records_json), "target_gap_ids": self.target_gap_ids,
                "fit_domains": self.fit_domains, "fit_exclusions": self.fit_exclusions,
                "support_domains": self.support_domains,
                "allow_provisional": self.allow_provisional, "prepared_identity": self.prepared_identity}


def make_batch_plan(prepared, config, *, allow_provisional=False, target_gap_ids=None,
                    fit_domains=(), fit_exclusions=(), support_domains=()) -> BatchPlan:
    """Inventory/routing dry run. No model is loaded or called by planning."""
    config = load_batch_config(config)
    document = config.to_dict()
    if type(allow_provisional) is not bool:
        raise ValueError("allow_provisional must be an explicit boolean")
    if prepared.contract_json != _json({key: document[key] for key in ("identity", "input")}):
        raise ValueError("config/input contract drift")
    gaps = inventory_batch(prepared)
    ids = {gap["gap_id"] for gap in gaps}
    if target_gap_ids is not None and (not isinstance(target_gap_ids, tuple)
            or any(not isinstance(item, str) for item in target_gap_ids)
            or len(set(target_gap_ids)) != len(target_gap_ids) or set(target_gap_ids) - ids):
        raise ValueError("target_gap_ids must uniquely select original whole gaps")
    domains, exclusions = _intervals(fit_domains), _intervals(fit_exclusions)
    support = _intervals(support_domains)
    if (document["require_support_domains"] and document["routes"]
            and document["status"] in {"usable", "provisional"} and not support):
        raise ValueError("this configuration requires explicit nonempty support_domains")
    records = []
    for gap in gaps:
        record = dict(gap, route=None, stages=[], reason=None)
        if target_gap_ids is not None and gap["gap_id"] not in target_gap_ids:
            record["reason"] = "not_selected"
        elif gap["structural"]:
            record["reason"] = "structural_absence"
        elif _support_interval(prepared.data, gap, support) == ():
            record["reason"] = "outside_support_domain"
        elif document["status"] not in {"usable", "provisional"}:
            record["reason"] = "deployment_" + document["status"]
        elif document["status"] == "provisional" and not allow_provisional:
            record["reason"] = "provisional_disabled"
        else:
            for route in document["routes"]:
                duration = gap["duration_seconds"]
                if (not route["min_gap_seconds"] < duration <= route["max_gap_seconds"]
                        or gap["geometry"] not in route["geometries"]):
                    continue
                record["route"] = route["id"]
                tested = route["tested_durations_seconds"]
                if duration not in tested and (route["duration_support"] != "interpolated" or duration < min(tested)):
                    record["reason"] = "duration_unvalidated"
                elif route["status"] == "provisional" and not allow_provisional:
                    record["reason"] = "provisional_disabled"
                else:
                    record["duration_evidence"] = "tested" if duration in tested else "interpolated"
                    record["status"] = ("provisional" if "provisional" in (document["status"], route["status"]) else "usable")
                    for number, stage in enumerate(route["stages"]):
                        native, reason = _native_plan(prepared.data, document, gap, stage, domains, exclusions, support)
                        record["stages"].append({"stage": number, "method": stage["method"],
                                                 "eligible": reason is None, "reason": reason,
                                                 "native_plan_identity": _digest(native.to_dict())})
                break
            if record["route"] is None:
                record["reason"] = "no_route_for_duration_geometry"
        records.append(record)
    prepared_identity = _digest([prepared.contract_json, prepared.reasons_json, prepared.inventory_json, prepared.original_rows])
    return BatchPlan(prepared.data.fingerprint, config, config.identity, _json(records), target_gap_ids,
                     domains, exclusions, support, allow_provisional, prepared_identity)


class BatchResult(ImputationResult):
    @property
    def gap_audit(self):
        return json.loads(_json(self._gap_audit))

    def provenance(self):
        value = super().provenance()
        value["files"] += ["config.json", "gap_audit.json", "source_reasons.csv", "batch_cells.csv"]
        value["batch"] = {"plan_identity": self.plan.identity, "config_identity": self.plan.config.identity,
                          "support": "original_only", "original_input_rows": list(self._original_rows),
                          "gap_audit": "gap_audit.json", "natural_gap_accuracy": "unobserved"}
        return value

    def export(self, directory):
        destination = super().export(directory)
        (destination / "config.json").write_text(json.dumps(self.plan.config.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (destination / "gap_audit.json").write_text(json.dumps(self.gap_audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pd.DataFrame([(stamp, name, "|".join(labels)) for stamp, name, labels in json.loads(self._reasons_json)],
                     columns=["timestamp", "variable", "source_reasons"]).to_csv(destination / "source_reasons.csv", index=False)
        pd.DataFrame(self._batch_cells, columns=["timestamp", "variable", "gap_id", "config_identity", "route", "stage",
                                               "deployment_status", "support", "fit_identity"]).to_csv(destination / "batch_cells.csv", index=False)
        return destination


class BatchExecutionError(RuntimeError):
    """Failure with retained completed/partial diagnostics, never an abstention."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


class BatchIntegrityError(ValueError):
    """Shared output/identity corruption; callers must stop later admission."""


def _checked_stage(data, result, gap, native):
    if not isinstance(result, ImputationResult) or result.schema != data.schema or result.plan.data_fingerprint != data.fingerprint:
        raise BatchIntegrityError("stage returned an incompatible result identity")
    if result.plan != native:
        raise BatchIntegrityError("stage result plan disagrees with the executed plan")
    if not result.raw.equals(data.raw):
        raise BatchIntegrityError("stage changed original raw values")
    for name, expected in data.masks.items():
        if not result.mask(name).equals(expected):
            raise BatchIntegrityError("stage changed original masks")
    completed, predicted = result.completed, result.mask("imputed")
    if (not completed.index.equals(data.raw.index) or not completed.columns.equals(data.raw.columns)
            or not predicted.index.equals(completed.index) or not predicted.columns.equals(completed.columns)
            or not all(pd.api.types.is_bool_dtype(dtype) for dtype in predicted.dtypes)
            or predicted.isna().to_numpy().any()):
        raise BatchIntegrityError("stage result grid/mask is corrupted")
    allowed = pd.DataFrame(False, index=completed.index, columns=completed.columns)
    allowed.iloc[gap["start_pos"]:gap["stop_pos"], completed.columns.get_loc(gap["variable"])] = True
    if (predicted & ~allowed).to_numpy().any():
        raise BatchIntegrityError("stage predicted outside its original selected gap")
    original = data.available
    for name in completed:
        mask = predicted[name]
        if not completed.loc[~mask, name].equals(original.loc[~mask, name]):
            # dtype promotion during valid numeric fills is permitted; values are not.
            left, right = completed.loc[~mask, name], original.loc[~mask, name]
            if not ((left == right) | (left.isna() & right.isna())).all():
                raise BatchIntegrityError("stage overwrote observation or unmarked missing cell")
        if any(not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_))
               for value in completed.loc[mask, name]):
            raise BatchIntegrityError("stage reconstruction must be real numeric")
        if not np.isfinite(completed.loc[mask, name].to_numpy(dtype=float)).all():
            raise BatchIntegrityError("stage produced a nonfinite reconstruction")


def apply_batch(prepared, plan, *, backends=None, before_stage=None, after_stage=None) -> BatchResult:
    """Execute ordered whole-gap fallbacks; hooks run only for eligible stages.

    before_stage(record, native_plan) can return budget_exhausted or
    backend_unavailable. after_stage(record, result) runs AFTER retained merge
    and diagnostics. Any failure exposes the retained result on BatchExecutionError.
    """
    if not isinstance(plan, BatchPlan):
        raise ValueError("apply requires a BatchPlan")
    expected = make_batch_plan(prepared, plan.config, allow_provisional=plan.allow_provisional,
                               target_gap_ids=plan.target_gap_ids, fit_domains=plan.fit_domains,
                               fit_exclusions=plan.fit_exclusions, support_domains=plan.support_domains)
    if expected != plan:
        raise ValueError("batch plan/config is stale or edited; rebuild the plan")
    data, document = prepared.data, plan.config.to_dict()
    empty = execute(data, make_plan(data, method="linear", mode="offline", max_gap=data.schema.cadence, target_gaps=()))
    completed, states, methods = empty.completed, empty.states, empty.methods
    reasons, imputed = empty.unresolved_reasons, empty.mask("imputed")
    records = json.loads(plan.records_json)
    audit, cells = [], []
    for gap in records:
        entry = {key: value for key, value in gap.items() if key != "stages"}
        entry.update(attempts=[], filled_cells=0, complete=False,
                     final_reason=gap["reason"] or "not_started")
        audit.append(entry)
    def retained():
        result = BatchResult(data, plan, completed, states, methods, reasons, imputed,
                             backend_identity={"kind": "batch", "config_identity": plan.config.identity,
                                               "support": "original_only"})
        result._gap_audit = audit
        result._batch_cells, result._reasons_json = cells, prepared.reasons_json
        result._original_rows = prepared.original_rows
        return result
    routes = {route["id"]: route for route in document["routes"]}
    for gap, entry in zip(records, audit):
        slot, col = slice(gap["start_pos"], gap["stop_pos"]), completed.columns.get_loc(gap["variable"])
        entry["final_reason"] = gap["reason"]
        if gap["reason"] is not None:
            reasons.iloc[slot, col] = gap["reason"]
            continue
        for planned in gap["stages"]:
            stage = routes[gap["route"]]["stages"][planned["stage"]]
            native, preflight = _native_plan(data, document, gap, stage, plan.fit_domains, plan.fit_exclusions,
                                            plan.support_domains)
            attempt = dict(planned, route=gap["route"], gap_id=gap["gap_id"], parameters=stage["parameters"],
                           predictors=stage.get("predictors", []), status="abstained", predicted_cells=0)
            entry["attempts"].append(attempt)
            if preflight is not None:
                entry["final_reason"] = preflight
                continue
            try:
                if stage["method"] == "forecast" and (backends is None or stage["backend"] not in backends):
                    gate = "backend_unavailable"
                else:
                    gate = None if before_stage is None else before_stage(dict(attempt), native)
                if gate not in {None, "budget_exhausted", "backend_unavailable"}:
                    raise ValueError("stage guard returned an undeclared fallback reason")
                if gate is not None:
                    attempt["reason"] = entry["final_reason"] = gate
                    continue
                method = stage["method"]
                if method in {"ols", "rf", "boosting"}:
                    try:
                        result = execute_driver_plan(data, native)
                    except OptionalBackendUnavailable:
                        attempt["reason"] = entry["final_reason"] = "backend_unavailable"
                        continue
                elif method == "forecast":
                    result = execute_forecast(data, native, backends[stage["backend"]])
                else:
                    result = execute(data, native)
                _checked_stage(data, result, gap, native)
                selected = result.mask("imputed").iloc[slot, col]
                prediction = result.completed.iloc[slot, col]
                attempt["predicted_cells"] = int(selected.sum())
                attempt["backend"] = deepcopy(result._backend_identity)
                attempt["diagnostics"] = [
                    {"timestamp": stamp.isoformat(), "value": float(prediction.loc[stamp]) if selected.loc[stamp] else None,
                     "reason": (None if pd.isna(result.unresolved_reasons.at[stamp, gap["variable"]])
                                else result.unresolved_reasons.at[stamp, gap["variable"]])}
                    for stamp in prediction.index]
                attempt["status"] = "complete" if selected.all() else "partial_discarded" if selected.any() else "abstained"
                if selected.all():
                    completed.iloc[slot, col] = prediction.to_numpy()
                    states.iloc[slot, col], methods.iloc[slot, col] = "imputed", result.methods.iloc[slot, col].to_numpy()
                    reasons.iloc[slot, col], imputed.iloc[slot, col] = None, True
                    entry.update(filled_cells=gap["n_steps"], complete=True, final_reason=None,
                                 selected_stage=planned["stage"], selected_method=method)
                    # Equal coefficients/scaler summaries do not prove equal
                    # training inputs. The native plan also binds original
                    # values, masks, resolved fitting rows and exact parameters.
                    fit_identity = (_digest({"native_plan_identity": planned["native_plan_identity"],
                                             "backend": attempt["backend"]})
                                    if method in {"ols", "rf", "boosting"} else None)
                    for stamp in prediction.index:
                        cells.append([stamp.isoformat(), gap["variable"], gap["gap_id"], plan.config.identity,
                                      gap["route"], planned["stage"], gap["status"], "original_only", fit_identity])
                else:
                    unresolved = result.unresolved_reasons.iloc[slot, col].dropna().unique().tolist()
                    attempt["reason"] = "whole_gap_incomplete"
                    attempt["residual_reasons"] = unresolved
                    entry["final_reason"] = "whole_gap_incomplete:" + ",".join(sorted(unresolved))
                # Retain partial output and completed merge before a post-call guard.
                if after_stage is not None:
                    after_stage(dict(attempt), result)
                if entry["complete"]:
                    break
            except Exception as exc:
                attempt["status"] = "failed"
                attempt["error_type"] = type(exc).__name__
                entry["failure"] = "stage_failure"
                if not entry["complete"]:
                    entry["final_reason"] = "stage_failure"
                    reasons.iloc[slot, col] = "stage_failure"
                for unstarted in audit:
                    if unstarted["final_reason"] == "not_started":
                        unstarted["final_reason"] = "stage_not_started_after_failure"
                    if not unstarted["complete"] and unstarted["final_reason"] is not None:
                        reasons.iloc[unstarted["start_pos"]:unstarted["stop_pos"],
                                     completed.columns.get_loc(unstarted["variable"])] = unstarted["final_reason"]
                raise BatchExecutionError("batch stage failed; retained diagnostics are available on .result", retained()) from exc
        if not entry["complete"]:
            reasons.iloc[slot, col] = entry["final_reason"] or "no_eligible_stage"
    return retained()


def load_batch_input(directory, config):
    """Read documented qualified CSV/schema/mask files, then prepare explicitly."""
    directory = Path(directory)
    metadata = json.loads((directory / "schema.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(directory / "input.csv", index_col="timestamp")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=False))
    masks = pd.read_csv(directory / "masks.csv", index_col="timestamp", dtype=str, keep_default_na=False)
    masks.index = pd.DatetimeIndex(pd.to_datetime(masks.index, utc=False))
    expected = {f"{kind}__{name}" for kind in ("observed", "qc_rejected", "structural", "artificial", "inserted") for name in frame.columns}
    if set(masks.columns) != expected:
        raise ValueError("mask columns must declare every source mask and input variable exactly")
    tables = {}
    for kind in ("observed", "qc_rejected", "structural", "artificial", "inserted"):
        selected = masks[[f"{kind}__{name}" for name in frame.columns]].copy()
        if not selected.isin(["True", "False"]).to_numpy().all():
            raise ValueError("CSV masks require literal True/False values")
        selected.columns = frame.columns
        tables[kind] = selected == "True"
    return prepare_batch(frame, config=config, metadata=metadata, **tables)
