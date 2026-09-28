"""Conservative univariate baselines using only declared donor observations."""

import numpy as np
import pandas as pd

from .data import SeriesData
from .planning import ImputationPlan, StateSpaceSpec, validate_plan
from .result import ImputationResult


def _local_level(values: np.ndarray, spec: StateSpaceSpec, *, smooth: bool) -> np.ndarray:
    """Known-parameter scalar Kalman filter and optional RTS backward smoother."""
    filtered = np.empty(len(values), dtype=float)
    variances = np.empty(len(values), dtype=float)
    mean = float(spec.initial_mean)
    variance = float(spec.initial_variance)
    for position, value in enumerate(values):
        if position:
            variance += spec.process_variance
        if np.isfinite(value):
            gain = variance / (variance + spec.observation_variance)
            mean += gain * (value - mean)
            variance *= 1 - gain
        filtered[position] = mean
        variances[position] = variance
    if not smooth:
        return filtered
    smoothed = filtered.copy()
    for position in range(len(values) - 2, -1, -1):
        forecast_variance = variances[position] + spec.process_variance
        gain = variances[position] / forecast_variance if forecast_variance else 0.0
        smoothed[position] += gain * (smoothed[position + 1] - filtered[position])
    return smoothed


def execute(data: SeriesData, plan: ImputationPlan) -> ImputationResult:
    """Apply a plan to its original input identity; never chain reconstructed donors."""
    validate_plan(data, plan)
    completed = data.available
    index, columns = completed.index, completed.columns
    states = pd.DataFrame("unavailable", index=index, columns=columns, dtype=object)
    for mask, label in (
        ("raw_present", "unqualified"),
        ("absent", "absent"), ("inserted", "absent_timestamp"),
        ("structural", "structural"), ("qc_rejected", "qc_rejected"),
        ("observed", "observed"), ("artificial", "artificial_hidden"),
    ):
        states = states.mask(data.mask(mask), label)
    methods = pd.DataFrame(None, index=index, columns=columns, dtype=object)
    reasons = pd.DataFrame(None, index=index, columns=columns, dtype=object)
    imputed = pd.DataFrame(False, index=index, columns=columns)
    original = data.available
    state_cache = {}
    for item in plan.gaps:
        gap = item.gap
        column = columns.get_loc(gap.variable)
        slot = slice(gap.start_pos, gap.stop_pos)
        if not item.eligible:
            reasons.iloc[slot, column] = item.reason
            continue
        values = original.iloc[:, column].to_numpy(dtype=float, na_value=np.nan)
        if plan.method == "persistence":
            prediction = np.repeat(values[gap.start_pos - 1], gap.n_steps)
        elif plan.method == "next":
            prediction = np.repeat(values[gap.stop_pos], gap.n_steps)
        elif plan.method == "linear":
            fraction = np.arange(1, gap.n_steps + 1) / (gap.n_steps + 1)
            prediction = values[gap.start_pos - 1] + fraction * (
                values[gap.stop_pos] - values[gap.start_pos - 1]
            )
        elif plan.method == "seasonal":
            prediction = np.array([
                np.median(values[list(donors)]) for donors in item.donors
            ], dtype=float)
        elif plan.method in {"state_filter", "state_smoother"}:
            if gap.variable not in state_cache:
                state_cache[gap.variable] = _local_level(
                    values, plan.state, smooth=plan.method == "state_smoother"
                )
            prediction = state_cache[gap.variable][slot]
        else:
            raise ValueError(f"unknown planned method: {plan.method}")
        if len(prediction) != gap.n_steps or not np.isfinite(prediction).all():
            raise RuntimeError(f"{plan.method} produced invalid values for {gap.variable}")
        completed.iloc[slot, column] = prediction
        states.iloc[slot, column] = "imputed"
        methods.iloc[slot, column] = plan.method
        imputed.iloc[slot, column] = True
    return ImputationResult(data, plan, completed, states, methods, reasons, imputed)
