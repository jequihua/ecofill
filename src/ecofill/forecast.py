"""Explicit, backend-independent forecasts for aligned missing channel runs."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .data import SeriesData
from .gaps import profile_gaps
from .result import ImputationResult


@dataclass(frozen=True)
class DynamicCovariate:
    name: str
    available_as_of: pd.Timestamp
    reversible: bool


@dataclass(frozen=True)
class ForecastPolicy:
    channels: tuple[str, ...]
    mode: str
    direction: str
    max_gap: pd.Timedelta
    context_length: int
    min_support: int
    max_horizon: int
    covariates: tuple[DynamicCovariate, ...] = ()
    static_covariates: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ForecastGap:
    start_pos: int
    stop_pos: int
    channels: tuple[str, ...]
    forward_context: tuple[int, ...]
    backward_context: tuple[int, ...]
    eligible: bool
    reason: str | None


@dataclass(frozen=True)
class ForecastPlan:
    data_fingerprint: str
    policy: ForecastPolicy
    gaps: tuple[ForecastGap, ...]

    def to_dict(self) -> dict:
        policy = self.policy
        return {
            "data_fingerprint": self.data_fingerprint,
            "method": "forecast", "mode": policy.mode, "direction": policy.direction,
            "channels": list(policy.channels),
            "max_gap_seconds": policy.max_gap.total_seconds(),
            "context_length": policy.context_length, "min_support": policy.min_support,
            "max_horizon": policy.max_horizon,
            "covariates": [
                {"name": item.name, "available_as_of": item.available_as_of.isoformat(),
                 "reversible": item.reversible} for item in policy.covariates
            ],
            "static_covariates": dict(policy.static_covariates),
            "gaps": [
                {"start_pos": item.start_pos, "stop_pos": item.stop_pos,
                 "channels": list(item.channels), "n_steps": item.stop_pos - item.start_pos,
                 "forward_context": list(item.forward_context),
                 "backward_context": list(item.backward_context),
                 "eligible": item.eligible, "reason": item.reason}
                for item in self.gaps
            ],
        }


@dataclass(frozen=True)
class ForecastRequest:
    """Arrays are owned copies. Rows are in prediction direction; columns match channels."""

    channels: tuple[str, ...]
    direction: str
    context: np.ndarray
    historical_covariates: np.ndarray
    horizon_covariates: np.ndarray
    static_covariates: tuple[tuple[str, float], ...]
    horizon: int


class LastValueForecaster:
    """Deterministic plumbing baseline: repeat the last context row."""

    def forecast(self, request: ForecastRequest) -> np.ndarray:
        return np.repeat(request.context[-1:, :], request.horizon, axis=0)


def _validate_policy(data: SeriesData, policy: ForecastPolicy) -> None:
    if not isinstance(data, SeriesData) or not isinstance(policy, ForecastPolicy):
        raise ValueError("forecast requires SeriesData and ForecastPolicy")
    names = tuple(data.available.columns)
    if not isinstance(policy.channels, tuple) or not policy.channels or len(set(policy.channels)) != len(policy.channels):
        raise ValueError("channels must be a nonempty ordered tuple without duplicates")
    if any(name not in names for name in policy.channels):
        raise ValueError("forecast channels must exist in SeriesData")
    if policy.mode not in {"offline", "causal"} or policy.direction not in {"forward", "backward", "blend"}:
        raise ValueError("invalid forecast mode or direction")
    if policy.mode == "causal" and policy.direction != "forward":
        raise ValueError("causal mode requires forward direction")
    if not isinstance(policy.max_gap, pd.Timedelta) or pd.isna(policy.max_gap) or policy.max_gap <= pd.Timedelta(0):
        raise ValueError("max_gap must be a positive Timedelta")
    for name in ("context_length", "min_support", "max_horizon"):
        if type(getattr(policy, name)) is not int or getattr(policy, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if policy.min_support > policy.context_length:
        raise ValueError("min_support cannot exceed context_length")
    if not isinstance(policy.covariates, tuple) or any(not isinstance(c, DynamicCovariate) for c in policy.covariates):
        raise ValueError("covariates must be DynamicCovariate records")
    cov_names = [c.name for c in policy.covariates]
    if len(set(cov_names)) != len(cov_names) or any(name not in names for name in cov_names):
        raise ValueError("covariate names must be unique SeriesData channels")
    lineage = {item.name: item.derived_from for item in data.schema.variables}
    def descends(name: str) -> bool:
        return name in policy.channels or any(descends(parent) for parent in lineage[name])
    for item in policy.covariates:
        if descends(item.name):
            raise ValueError("target and declared descendants cannot be covariates")
        if not isinstance(item.available_as_of, pd.Timestamp) or item.available_as_of.tz is None:
            raise ValueError("covariate requires an explicit timezone-aware available_as_of")
        if type(item.reversible) is not bool:
            raise ValueError("covariate reversibility must be explicit")
        if policy.direction in {"backward", "blend"} and not item.reversible:
            raise ValueError("backward forecasting requires reversible covariates")
    static = policy.static_covariates
    if not isinstance(static, tuple) or any(
        not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str)
        or not item[0] or not isinstance(item[1], (int, float)) or not math.isfinite(item[1])
        for item in static
    ) or len({item[0] for item in static}) != len(static):
        raise ValueError("static_covariates must be unique finite name/value pairs")
    if any(item[0] in lineage and descends(item[0]) for item in static):
        raise ValueError("target and declared descendants cannot be static covariates")


def make_forecast_plan(data: SeriesData, policy: ForecastPolicy) -> ForecastPlan:
    """Plan complete aligned target gaps without reading reconstructed values."""
    _validate_policy(data, policy)
    available = data.available
    structural = data.mask("structural")
    gaps = [gap for gap in profile_gaps(data) if gap.variable in policy.channels]
    spans = sorted({(gap.start_pos, gap.stop_pos) for gap in gaps})
    planned = []
    for start, stop in spans:
        matching = {gap.variable for gap in gaps if (gap.start_pos, gap.stop_pos) == (start, stop)}
        left = tuple(range(max(0, start - policy.context_length), start))
        right = tuple(range(stop, min(len(available), stop + policy.context_length)))
        required = ("forward", "backward") if policy.direction == "blend" else (policy.direction,)
        reason = None
        if matching != set(policy.channels):
            reason = "unaligned_target_gap"
        elif structural.iloc[start:stop][list(policy.channels)].to_numpy().any():
            reason = "structural_absence"
        elif (stop - start) * data.schema.cadence > policy.max_gap:
            reason = "gap_exceeds_limit"
        elif stop - start > policy.max_horizon:
            reason = "horizon_exceeds_limit"
        else:
            for direction, positions in (("forward", left), ("backward", right)):
                if direction not in required:
                    continue
                if len(positions) < policy.min_support:
                    reason = f"insufficient_{'left' if direction == 'forward' else 'right'}_context"
                    break
                values = available.iloc[list(positions)][list(policy.channels)].to_numpy(dtype=float, na_value=np.nan)
                if not np.isfinite(values).all():
                    reason = f"noncontiguous_{'left' if direction == 'forward' else 'right'}_context"
                    break
                # Every requested dynamic covariate needs contiguous historical and horizon rows.
                horizon = range(start, stop)
                for cov in policy.covariates:
                    issue = available.index[start] if direction == "forward" else available.index[stop]
                    if cov.available_as_of > issue:
                        reason = "covariate_unavailable_as_of"
                        break
                    rows = list(positions) + list(horizon)
                    cov_values = available.iloc[rows][cov.name].to_numpy(dtype=float, na_value=np.nan)
                    if not np.isfinite(cov_values).all():
                        reason = "covariate_missing"
                        break
                if reason:
                    break
        planned.append(ForecastGap(start, stop, policy.channels, left, right, reason is None, reason))
    return ForecastPlan(data.fingerprint, policy, tuple(planned))


def _request(data: SeriesData, item: ForecastGap, policy: ForecastPolicy, direction: str) -> ForecastRequest:
    available = data.available
    positions = item.forward_context if direction == "forward" else item.backward_context[::-1]
    horizon = list(range(item.start_pos, item.stop_pos))
    if direction == "backward":
        horizon.reverse()
    context = available.iloc[list(positions)][list(policy.channels)].to_numpy(dtype=float, na_value=np.nan, copy=True)
    cov_names = [cov.name for cov in policy.covariates]
    historical = available.iloc[list(positions)][cov_names].to_numpy(dtype=float, na_value=np.nan, copy=True)
    future = available.iloc[horizon][cov_names].to_numpy(dtype=float, na_value=np.nan, copy=True)
    return ForecastRequest(policy.channels, direction, context, historical, future,
                           policy.static_covariates, len(horizon))


def execute_forecast(data: SeriesData, plan: ForecastPlan, backend) -> ImputationResult:
    """Validate the public plan, then call backend.forecast(request) for each direction."""
    if not isinstance(data, SeriesData) or not isinstance(plan, ForecastPlan) or data.fingerprint != plan.data_fingerprint:
        raise ValueError("forecast plan is stale or invalid; rebuild with make_forecast_plan")
    try:
        expected = make_forecast_plan(data, plan.policy)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("invalid forecast plan policy; rebuild with make_forecast_plan") from exc
    if plan != expected:
        raise ValueError("forecast plan was edited; rebuild with make_forecast_plan")
    if not callable(getattr(backend, "forecast", None)):
        raise ValueError("backend must define forecast(request)")
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
    for item in plan.gaps:
        rows = slice(item.start_pos, item.stop_pos)
        cols = [columns.get_loc(name) for name in item.channels]
        if not item.eligible:
            for col in cols:
                missing = completed.iloc[rows, col].isna()
                reasons.iloc[item.start_pos:item.stop_pos, col] = np.where(missing, item.reason, None)
            continue
        predictions = {}
        for direction in (("forward", "backward") if plan.policy.direction == "blend" else (plan.policy.direction,)):
            request = _request(data, item, plan.policy, direction)
            try:
                output = backend.forecast(request)
            except Exception as exc:
                raise RuntimeError(f"forecast backend failed for {direction} at {item.start_pos}") from exc
            if not isinstance(output, np.ndarray) or output.shape != (request.horizon, len(item.channels)):
                raise RuntimeError(f"forecast backend returned invalid shape for {direction}")
            if output.dtype.kind not in "iuf":
                raise RuntimeError(f"forecast backend returned non-real numeric values for {direction}")
            if not np.isfinite(output).all():
                raise RuntimeError(f"forecast backend returned nonfinite values for {direction}")
            predictions[direction] = output.astype(float, copy=True)
        if plan.policy.direction == "blend":
            backward = predictions["backward"][::-1]
            length = item.stop_pos - item.start_pos
            weight = np.full(length, 0.5) if length == 1 else np.arange(length) / (length - 1)
            prediction = (1 - weight[:, None]) * predictions["forward"] + weight[:, None] * backward
        elif plan.policy.direction == "backward":
            prediction = predictions["backward"][::-1]
        else:
            prediction = predictions["forward"]
        if not np.isfinite(prediction).all():
            raise RuntimeError("forecast blend produced nonfinite values")
        completed.iloc[rows, cols] = prediction
        states.iloc[rows, cols] = "imputed"
        methods.iloc[rows, cols] = "forecast:" + plan.policy.direction
        imputed.iloc[rows, cols] = True
    return ImputationResult(data, plan, completed, states, methods, reasons, imputed,
                            backend_identity=getattr(backend, "identity", None))
