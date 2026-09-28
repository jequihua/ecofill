"""Explicit whole-run method plans and abstention reasons."""

from dataclasses import dataclass
import math

import pandas as pd

from .data import SeriesData
from .gaps import Gap, GapEvent, profile_events, profile_gaps


@dataclass(frozen=True)
class StateSpaceSpec:
    """Known-parameter scalar local-level model: x[t]=x[t-1]+w; y[t]=x[t]+v."""

    process_variance: float
    observation_variance: float
    initial_mean: float
    initial_variance: float

    def __post_init__(self):
        values = (self.process_variance, self.observation_variance,
                  self.initial_mean, self.initial_variance)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError("state-space parameters must be finite numbers")
        if self.process_variance < 0 or self.observation_variance <= 0 or self.initial_variance <= 0:
            raise ValueError("state-space variances require process >= 0, observation > 0, initial > 0")


@dataclass(frozen=True)
class PlannedGap:
    gap: Gap
    eligible: bool
    reason: str | None
    left_support: bool
    right_support: bool
    donors: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class ImputationPlan:
    data_fingerprint: str
    method: str
    mode: str
    max_gap: pd.Timedelta
    period: pd.Timedelta | None
    search_periods: int | None
    state: StateSpaceSpec | None
    gaps: tuple[PlannedGap, ...]
    events: tuple[GapEvent, ...]

    def to_dict(self) -> dict:
        return {
            "data_fingerprint": self.data_fingerprint,
            "method": self.method,
            "mode": self.mode,
            "max_gap_seconds": self.max_gap.total_seconds(),
            "period_seconds": None if self.period is None else self.period.total_seconds(),
            "search_periods": self.search_periods,
            "state": None if self.state is None else vars(self.state),
            "gaps": [
                {
                    "variable": item.gap.variable,
                    "start": item.gap.start.isoformat(),
                    "end": item.gap.end.isoformat(),
                    "start_pos": item.gap.start_pos,
                    "stop_pos": item.gap.stop_pos,
                    "n_steps": item.gap.n_steps,
                    "duration_seconds": item.gap.duration.total_seconds(),
                    "geometry": item.gap.geometry,
                    "shared_variables": list(item.gap.shared_variables),
                    "eligible": item.eligible,
                    "reason": item.reason,
                    "left_support": item.left_support,
                    "right_support": item.right_support,
                    "donors": [list(indices) for indices in item.donors],
                }
                for item in self.gaps
            ],
            "events": [
                {
                    "start": event.start.isoformat(), "end": event.end.isoformat(),
                    "start_pos": event.start_pos, "stop_pos": event.stop_pos,
                    "n_steps": event.n_steps, "duration_seconds": event.duration.total_seconds(),
                    "geometry": event.geometry, "pattern": event.pattern,
                    "variables": list(event.variables),
                    "shared_variables": list(event.shared_variables),
                }
                for event in self.events
            ],
        }


def make_plan(
    data: SeriesData,
    *,
    method: str,
    mode: str,
    max_gap: pd.Timedelta,
    period: pd.Timedelta | None = None,
    search_periods: int | None = None,
    state: StateSpaceSpec | None = None,
) -> ImputationPlan:
    """Plan a named method; no gap is partly filled to satisfy a duration limit."""
    methods = {"persistence", "next", "linear", "seasonal", "state_filter", "state_smoother"}
    if method not in methods:
        raise ValueError(f"unknown method: {method}")
    if mode not in {"offline", "causal"}:
        raise ValueError("mode must be 'offline' or 'causal'")
    if mode == "causal" and method in {"next", "linear", "state_smoother"}:
        raise ValueError(f"{method} requires future observations and is unavailable in causal mode")
    if not isinstance(max_gap, pd.Timedelta) or pd.isna(max_gap) or max_gap <= pd.Timedelta(0):
        raise ValueError("max_gap must be an explicit positive Timedelta")
    if method == "seasonal":
        if not isinstance(period, pd.Timedelta) or pd.isna(period) or period <= pd.Timedelta(0):
            raise ValueError("seasonal requires a positive period")
        if period.value % data.schema.cadence.value:
            raise ValueError("seasonal period must be an exact cadence multiple")
        if type(search_periods) is not int or search_periods < 1:
            raise ValueError("seasonal requires positive integer search_periods")
    elif period is not None or search_periods is not None:
        raise ValueError("period and search_periods belong only to seasonal")
    if method in {"state_filter", "state_smoother"}:
        if not isinstance(state, StateSpaceSpec):
            raise ValueError("state-space methods require an explicit StateSpaceSpec")
    elif state is not None:
        raise ValueError("state belongs only to state-space methods")

    available = data.available
    structural = data.mask("structural")
    steps = None if period is None else period.value // data.schema.cadence.value
    planned = []
    for gap in profile_gaps(data):
        column = available.columns.get_loc(gap.variable)
        values = available.iloc[:, column]
        left = gap.start_pos > 0 and pd.notna(values.iloc[gap.start_pos - 1])
        right = gap.stop_pos < len(values) and pd.notna(values.iloc[gap.stop_pos])
        reason = None
        donors = ()
        if structural.iloc[gap.start_pos:gap.stop_pos, column].any():
            reason = "structural_absence"
        elif gap.duration > max_gap:
            reason = "gap_exceeds_limit"
        elif method in {"persistence", "linear"} and not left:
            reason = "insufficient_left_context"
        elif method in {"next", "linear"} and not right:
            reason = "insufficient_right_context"
        elif method == "state_filter" and not values.iloc[:gap.start_pos].notna().any():
            reason = "insufficient_left_context"
        elif method == "state_smoother" and not values.notna().any():
            reason = "insufficient_observed_support"
        elif method == "seasonal":
            options = []
            for position in range(gap.start_pos, gap.stop_pos):
                candidates = []
                for multiplier in range(1, search_periods + 1):
                    for donor in (position - multiplier * steps, position + multiplier * steps):
                        if 0 <= donor < len(values) and (mode == "offline" or donor < gap.start_pos):
                            if pd.notna(values.iloc[donor]):
                                candidates.append(donor)
                options.append(tuple(candidates))
            donors = tuple(options)
            if not all(donors):
                reason = "insufficient_seasonal_support"
        planned.append(PlannedGap(
            gap=gap, eligible=reason is None, reason=reason,
            left_support=bool(left), right_support=bool(right), donors=donors,
        ))
    return ImputationPlan(
        data_fingerprint=data.fingerprint, method=method, mode=mode, max_gap=max_gap,
        period=period, search_periods=search_periods, state=state,
        gaps=tuple(planned), events=profile_events(data),
    )


def validate_plan(data: SeriesData, plan: ImputationPlan) -> None:
    """Reject stale or edited public plans before any result cell is written."""
    if not isinstance(data, SeriesData) or not isinstance(plan, ImputationPlan):
        raise ValueError("execute requires SeriesData and an ImputationPlan from make_plan")
    if data.fingerprint != plan.data_fingerprint:
        raise ValueError("plan was built for different values, masks or schema; rebuild with make_plan")
    try:
        expected = make_plan(
            data, method=plan.method, mode=plan.mode, max_gap=plan.max_gap,
            period=plan.period, search_periods=plan.search_periods, state=plan.state,
        )
        matches = plan == expected
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"invalid plan policy or contents: {exc}; rebuild with make_plan") from exc
    if not matches:
        raise ValueError("plan contents disagree with data or declared policy; rebuild with make_plan")
