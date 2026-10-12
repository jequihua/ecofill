"""Offline regular-grid imputation from fixed time features and a frozen reference bank.

A backend receives copied finite reference features/labels and the batched query
features of one primitive target; hidden or missing query labels are never sent.
The device is a requested routing setting that the backend must declare back;
it is not observed hardware qualification.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from .data import SeriesData
from .gaps import profile_gaps, validate_gap_selection, validate_support_interval
from .result import ImputationResult

FEATURE_NAMES = ("elapsed", "day_sin", "day_cos", "week_sin", "week_cos")
DAY_NS = 86_400 * 10 ** 9
WEEK_NS = 7 * DAY_NS
DEVICES = ("cpu", "cuda")
SELECTION = ("time-ordered eligible observed rows of the reference intervals; over the cap "
             "keep row floor(i*(n-1)/(cap-1)) for i < cap (cap 1 keeps floor((n-1)/2))")


@dataclass(frozen=True)
class TimeFeatures:
    """Fixed recipe: (t - origin) / scale, then daily and weekly sin/cos phase.

    Phases are absolute elapsed time since `origin` modulo 1 and 7 days. This is a
    fixed-period profile, not local solar, calendar or daylight-saving clock time.
    """

    origin: pd.Timestamp
    scale: pd.Timedelta

    def __post_init__(self):
        if not isinstance(self.origin, pd.Timestamp) or pd.isna(self.origin) or self.origin.tz is None:
            raise ValueError("feature origin must be an explicit timezone-aware Timestamp")
        if not isinstance(self.scale, pd.Timedelta) or pd.isna(self.scale) or self.scale <= pd.Timedelta(0):
            raise ValueError("feature scale must be a positive Timedelta")

    def to_dict(self) -> dict:
        return {"names": list(FEATURE_NAMES), "origin": self.origin.isoformat(),
                "scale_seconds": self.scale.total_seconds(),
                "periods_seconds": [DAY_NS / 10 ** 9, WEEK_NS / 10 ** 9],
                "phase": "elapsed time since origin modulo period"}

    def compute(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Feature matrix in FEATURE_NAMES order for a timezone-aware index."""
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
            raise ValueError("time features need a timezone-aware DatetimeIndex")
        elapsed = index.as_unit("ns").asi8 - self.origin.as_unit("ns").value
        day = 2 * np.pi * (np.mod(elapsed, DAY_NS) / DAY_NS)
        week = 2 * np.pi * (np.mod(elapsed, WEEK_NS) / WEEK_NS)
        return np.column_stack([elapsed / self.scale.value, np.sin(day), np.cos(day),
                                np.sin(week), np.cos(week)])


@dataclass(frozen=True)
class TimeIndexedPolicy:
    """Explicit primitive targets and ordered half-open reference position intervals."""

    targets: tuple[str, ...]
    reference: tuple[tuple[int, int], ...]
    features: TimeFeatures
    max_reference: int
    min_support: int
    max_gap: pd.Timedelta
    seed: int
    device: str = "cpu"
    mode: str = "offline"
    external_features: tuple[str, ...] = ()
    reference_rows: tuple[int, ...] | None = None
    target_gaps: tuple[tuple[str, int, int], ...] | None = None
    support_interval: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        document = {"targets": list(self.targets),
                "reference_intervals": [list(item) for item in self.reference],
                "max_reference": self.max_reference, "min_support": self.min_support,
                "max_gap_seconds": self.max_gap.total_seconds(), "seed": self.seed,
                "device": self.device, "features": self.features.to_dict(), "selection": SELECTION}
        if self.external_features or self.reference_rows is not None or self.target_gaps is not None \
                or self.support_interval is not None:
            document.update(external_features=list(self.external_features),
                            external_policy="original QC-passed observations; availability at grid timestamp; no transform",
                            reference_rows=self.reference_rows, target_gaps=self.target_gaps,
                            support_interval=self.support_interval)
        return document


@dataclass(frozen=True)
class TimeIndexedTarget:
    """Actual frozen reference bank of one target; `identity` digests its features and labels."""

    variable: str
    n_eligible: int
    rows: tuple[int, ...]
    identity: str
    reason: str | None


@dataclass(frozen=True)
class TimeIndexedGap:
    variable: str
    start_pos: int
    stop_pos: int
    eligible: bool
    reason: str | None


@dataclass(frozen=True)
class TimeIndexedPlan:
    data_fingerprint: str
    policy: TimeIndexedPolicy
    feature_names: tuple[str, ...]
    targets: tuple[TimeIndexedTarget, ...]
    gaps: tuple[TimeIndexedGap, ...]
    external_units: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict:
        """Method settings, reference identity and (data-instance) query gaps, kept separate."""
        reference = self.policy.reference
        return {
            "data_fingerprint": self.data_fingerprint, "method": "time_indexed",
            "mode": self.policy.mode,
            # Hull of the reference intervals: the rows an ensemble recipe must declare.
            "training_interval": [reference[0][0], reference[-1][1]],
            "policy": self.policy.to_dict(), "feature_names": list(self.feature_names),
            **({"external_units": dict(self.external_units)} if self.external_units else {}),
            "reference": [{"variable": item.variable, "n_eligible": item.n_eligible,
                           "rows": list(item.rows), "identity": item.identity, "reason": item.reason}
                          for item in self.targets],
            "gaps": [{"variable": item.variable, "start_pos": item.start_pos, "stop_pos": item.stop_pos,
                      "n_steps": item.stop_pos - item.start_pos, "eligible": item.eligible,
                      "reason": item.reason} for item in self.gaps],
        }


@dataclass(frozen=True)
class TimeIndexedRequest:
    """Owned copies for one target: reference rows in time order, all eligible query rows."""

    target: str
    feature_names: tuple[str, ...]
    reference_features: np.ndarray
    reference_labels: np.ndarray
    query_features: np.ndarray
    seed: int
    device: str


def _positive_int(value) -> bool:
    return type(value) is int and value >= 1


def _validate(data: SeriesData, policy: TimeIndexedPolicy) -> None:
    if not isinstance(data, SeriesData) or not isinstance(policy, TimeIndexedPolicy):
        raise ValueError("time-indexed imputation requires SeriesData and TimeIndexedPolicy")
    if policy.mode == "causal":
        raise ValueError("time-indexed imputation is offline only; causal mode is not supported")
    if policy.mode != "offline":
        raise ValueError("time-indexed mode must be 'offline'")
    lineage = {item.name: item.derived_from for item in data.schema.variables}
    targets = policy.targets
    if not isinstance(targets, tuple) or not targets or len(set(targets)) != len(targets) \
            or any(not isinstance(name, str) or name not in lineage for name in targets):
        raise ValueError("targets must be a nonempty tuple of unique SeriesData variables")
    if any(lineage[name] for name in targets):
        raise ValueError("time-indexed targets must be primitive; derived targets are not reconstructed")
    length = len(data.raw.index)
    reference = policy.reference
    if not isinstance(reference, tuple) or not reference or any(
            not isinstance(item, tuple) or len(item) != 2 or any(type(value) is not int for value in item)
            or not 0 <= item[0] < item[1] <= length for item in reference):
        raise ValueError("reference must be a nonempty tuple of (start, stop) positions on the grid")
    if any(previous[1] > item[0] for previous, item in zip(reference, reference[1:])):
        raise ValueError("reference intervals must be ordered and nonoverlapping")
    if not isinstance(policy.features, TimeFeatures):
        raise ValueError("features must be a TimeFeatures recipe")
    TimeFeatures(policy.features.origin, policy.features.scale)
    if not _positive_int(policy.max_reference) or not _positive_int(policy.min_support):
        raise ValueError("max_reference and min_support must be positive integers")
    if policy.min_support > policy.max_reference:
        raise ValueError("min_support cannot exceed max_reference")
    if not isinstance(policy.max_gap, pd.Timedelta) or pd.isna(policy.max_gap) \
            or policy.max_gap <= pd.Timedelta(0):
        raise ValueError("max_gap must be a positive Timedelta")
    if type(policy.seed) is not int or not 0 <= policy.seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if policy.device not in DEVICES:
        raise ValueError("device must be 'cpu' or 'cuda'")
    external = policy.external_features
    if not isinstance(external, tuple) or len(set(external)) != len(external) \
            or any(not isinstance(name, str) or name not in lineage for name in external):
        raise ValueError("external features must be unique ordered SeriesData variables")
    def descends(name):
        return name in targets or any(descends(parent) for parent in lineage[name])
    if any(descends(name) for name in external):
        raise ValueError("targets and descendants cannot be external features")
    if set(external) & set(FEATURE_NAMES):
        raise ValueError("external feature names conflict with time features")
    if policy.reference_rows is not None and (
            not isinstance(policy.reference_rows, tuple)
            or any(type(row) is not int or not any(a <= row < b for a, b in reference)
                   for row in policy.reference_rows)
            or tuple(sorted(set(policy.reference_rows))) != policy.reference_rows):
        raise ValueError("reference_rows must be sorted unique positions inside reference intervals")


def _features(data, policy):
    time = policy.features.compute(data.raw.index)
    if not policy.external_features:
        return time
    external = data.available[list(policy.external_features)].to_numpy(dtype=float, na_value=np.nan)
    return np.column_stack((time, external))


def _select(eligible: list[int], cap: int) -> list[int]:
    n = len(eligible)
    if n <= cap:
        return eligible
    if cap == 1:
        return [eligible[(n - 1) // 2]]
    return [eligible[i * (n - 1) // (cap - 1)] for i in range(cap)]


def _reference_identity(data, policy, name, rows, features, labels) -> str:
    index = data.raw.index
    unit = next(item.unit for item in data.schema.variables if item.name == name)
    digest = hashlib.sha256(json.dumps({
        "variable": name, "unit": unit, "cadence_ns": data.schema.cadence.value,
        "features": policy.features.to_dict(),
        **({"external": [(v.name, v.unit) for v in data.schema.variables
                          if v.name in policy.external_features],
            "ordered_features": policy.external_features}
           if policy.external_features else {}),
        "timestamps": [index[row].isoformat() for row in rows],
    }, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(features[rows], dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(labels[rows], dtype="<f8").tobytes())
    return digest.hexdigest()


def make_time_indexed_plan(data: SeriesData, policy: TimeIndexedPolicy) -> TimeIndexedPlan:
    """Freeze each target's reference bank and decide every whole target gap."""
    _validate(data, policy)
    available = data.available
    donor = data.mask("donor")
    structural = data.mask("structural")
    features = _features(data, policy)
    selected = validate_gap_selection(data, policy.target_gaps, variables=policy.targets)
    support_start, support_stop = validate_support_interval(data, policy.support_interval)
    positions = [row for start, stop in policy.reference for row in range(start, stop)]
    positions = [row for row in positions if support_start <= row < support_stop]
    if policy.reference_rows is not None:
        retained = set(policy.reference_rows)
        positions = [row for row in positions if row in retained]
    targets, reasons = [], {}
    for name in policy.targets:
        eligible = [row for row in positions if donor[name].iat[row]
                    and all(donor[col].iat[row] for col in policy.external_features)]
        rows = _select(eligible, policy.max_reference)
        labels = available[name].to_numpy(dtype=float, na_value=np.nan)
        reasons[name] = None if len(rows) >= policy.min_support else "insufficient_reference_support"
        targets.append(TimeIndexedTarget(name, len(eligible), tuple(rows),
                                         _reference_identity(data, policy, name, rows, features, labels),
                                         reasons[name]))
    gaps = []
    for gap in profile_gaps(data):
        if gap.variable not in policy.targets:
            continue
        if selected is not None and (gap.variable, gap.start_pos, gap.stop_pos) not in selected:
            continue
        reason = None
        if gap.start_pos < support_start or gap.stop_pos > support_stop:
            reason = "outside_support_domain"
        elif structural[gap.variable].iloc[gap.start_pos:gap.stop_pos].any():
            reason = "structural_absence"
        elif gap.geometry != "bounded":
            reason = "edge_gap_unsupported"
        elif gap.duration > policy.max_gap:
            reason = "gap_exceeds_limit"
        elif policy.external_features and not donor[list(policy.external_features)] \
                .iloc[gap.start_pos:gap.stop_pos].to_numpy().all():
            reason = "external_feature_missing"
        else:
            reason = reasons[gap.variable]
        gaps.append(TimeIndexedGap(gap.variable, gap.start_pos, gap.stop_pos, reason is None, reason))
    units = tuple((name, next(v.unit for v in data.schema.variables if v.name == name))
                  for name in policy.external_features)
    return TimeIndexedPlan(data.fingerprint, policy, FEATURE_NAMES + policy.external_features,
                           tuple(targets), tuple(gaps), units)


def _backend_identity(backend, device: str) -> dict:
    if not callable(getattr(backend, "predict", None)):
        raise ValueError("backend must define predict(request)")
    identity = getattr(backend, "identity", None)
    if not isinstance(identity, dict) or not identity:
        raise ValueError("time-indexed backend must declare a nonempty identity mapping")
    if "execution" in identity:
        raise ValueError("backend identity key 'execution' is reserved")
    if identity.get("device") != device:
        raise ValueError(f"backend declares device {identity.get('device')!r}, but {device!r} was "
                         "requested; no silent device fallback")
    try:
        json.dumps(identity, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("backend identity must be portable JSON") from exc
    return deepcopy(identity)


def _canonical(identity: dict) -> str:
    return json.dumps(identity, sort_keys=True, allow_nan=False)


def _check_bound_identity(backend, device: str, bound: str, when: str) -> None:
    """The backend's current declaration must still equal the identity bound to this execution."""
    try:
        current = _canonical(_backend_identity(backend, device))
    except ValueError as exc:
        raise RuntimeError(f"backend identity became invalid {when}: {exc}") from exc
    if current != bound:
        raise RuntimeError(f"backend identity changed {when}; its predictions cannot be "
                           "recorded under the identity bound to this execution")


def execute_time_indexed(data: SeriesData, plan: TimeIndexedPlan, backend) -> ImputationResult:
    """Re-derive the plan, then call backend.predict(request) once per target with eligible gaps."""
    if not isinstance(data, SeriesData) or not isinstance(plan, TimeIndexedPlan) \
            or data.fingerprint != plan.data_fingerprint:
        raise ValueError("time-indexed plan is stale or invalid; rebuild with make_time_indexed_plan")
    try:
        expected = make_time_indexed_plan(data, plan.policy)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("invalid time-indexed plan policy; rebuild with make_time_indexed_plan") from exc
    if plan != expected:
        raise ValueError("time-indexed plan was edited; rebuild with make_time_indexed_plan")
    policy = plan.policy
    identity = _backend_identity(backend, policy.device)
    bound = _canonical(identity)
    completed = data.available
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
    features = _features(data, policy)
    predictions, requests = [], []
    for target in plan.targets:
        col = columns.get_loc(target.variable)
        gaps = [item for item in plan.gaps if item.variable == target.variable]
        for item in gaps:
            if not item.eligible:
                reasons.iloc[item.start_pos:item.stop_pos, col] = item.reason
        query = [row for item in gaps if item.eligible for row in range(item.start_pos, item.stop_pos)]
        if not query:
            continue
        rows = list(target.rows)
        labels = completed[target.variable].to_numpy(dtype=float, na_value=np.nan)[rows]
        if not np.isfinite(labels).all():
            raise RuntimeError(f"reference labels of {target.variable} are not finite observations")
        request = TimeIndexedRequest(target.variable, plan.feature_names, features[rows], labels,
                                     features[query], policy.seed, policy.device)
        sent = (request.reference_features.copy(), request.reference_labels.copy(),
                request.query_features.copy())
        _check_bound_identity(backend, policy.device, bound, f"before predicting {target.variable}")
        try:
            output = backend.predict(request)
        except Exception as exc:
            raise RuntimeError(f"time-indexed backend failed for {target.variable}") from exc
        _check_bound_identity(backend, policy.device, bound, f"while predicting {target.variable}")
        received = (request.reference_features, request.reference_labels, request.query_features)
        if (request.target, request.feature_names, request.seed, request.device) != (
                target.variable, plan.feature_names, policy.seed, policy.device) or any(
                not isinstance(a, np.ndarray) or a.shape != b.shape or not np.array_equal(a, b)
                for a, b in zip(received, sent)):
            raise RuntimeError(f"time-indexed backend mutated the request for {target.variable}")
        if not isinstance(output, np.ndarray) or output.shape != (len(query),):
            raise RuntimeError(f"time-indexed backend returned invalid shape for {target.variable}")
        if output.dtype.kind not in "iuf" or not np.isfinite(output).all():
            raise RuntimeError(f"time-indexed backend returned nonfinite or non-real values for {target.variable}")
        predictions.append((query, col, output.astype(float, copy=True)))
        requests.append({"variable": target.variable, "reference_rows": len(rows),
                         "query_spans": [[item.start_pos, item.stop_pos] for item in gaps if item.eligible],
                         "predicted": len(query)})
    # Every response is validated before any result cell is written.
    for query, col, values in predictions:
        completed.iloc[query, col] = values
        states.iloc[query, col] = "imputed"
        methods.iloc[query, col] = "time_indexed"
        imputed.iloc[query, col] = True
    identity["execution"] = {
        "requested_device": policy.device,
        "device_note": "requested routing declared by the backend; not observed hardware qualification",
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        "requests": requests,
    }
    return ImputationResult(data, plan, completed, states, methods, reasons, imputed,
                            backend_identity=identity)
