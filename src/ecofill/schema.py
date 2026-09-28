"""Strict regular-grid contract; source adapters own clock repair and QC."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VariableSpec:
    name: str
    unit: str
    role: str = "target"
    derived_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class TimeSeriesSchema:
    cadence: pd.Timedelta
    variables: tuple[VariableSpec, ...]
    time_policy: str = "strict"


def validate_frame(frame: pd.DataFrame, schema: TimeSeriesSchema) -> None:
    """Reject ambiguous clocks/schema drift without modifying input.

    NaN denotes unavailable data, not its cause; finite is not proof of quality.
    A constant phase is permitted. Timezones and cadence must be explicit.
    """
    if not isinstance(schema.cadence, pd.Timedelta) or pd.isna(schema.cadence):
        raise ValueError("cadence must be a finite pandas Timedelta")
    if schema.cadence <= pd.Timedelta(0):
        raise ValueError("cadence must be positive")
    if schema.time_policy != "strict":
        raise ValueError("time_policy must be 'strict'; clock repair belongs in a source adapter")
    if not isinstance(schema.variables, tuple) or not schema.variables:
        raise ValueError("schema variables must be a nonempty tuple")
    names = [variable.name for variable in schema.variables]
    if any(not isinstance(name, str) or not name or name != name.strip() for name in names):
        raise ValueError("schema requires unique, nonempty variable names")
    if len(set(names)) != len(names):
        raise ValueError("schema requires unique, nonempty variable names")
    if any(
        not isinstance(variable.unit, str)
        or not variable.unit
        or variable.unit != variable.unit.strip()
        for variable in schema.variables
    ):
        raise ValueError("units must be explicit; use 'unknown' when unresolved")
    roles = {"target", "driver", "response", "external", "derived", "diagnostic"}
    if any(not isinstance(variable.role, str) or variable.role not in roles for variable in schema.variables):
        raise ValueError(f"variable roles must be one of {sorted(roles)}")
    for variable in schema.variables:
        if (
            not isinstance(variable.derived_from, tuple)
            or any(not isinstance(parent, str) for parent in variable.derived_from)
            or len(set(variable.derived_from)) != len(variable.derived_from)
        ):
            raise ValueError(f"lineage for {variable.name} must be a tuple of unique parents")
        if any(parent not in names or parent == variable.name for parent in variable.derived_from):
            raise ValueError(f"invalid lineage for {variable.name}")
    lineage = {variable.name: variable.derived_from for variable in schema.variables}
    visited = set()
    active = set()

    def visit(name):
        if name in active:
            raise ValueError("lineage cannot contain cycles")
        if name not in visited:
            active.add(name)
            for parent in lineage[name]:
                visit(parent)
            active.remove(name)
            visited.add(name)

    for name in names:
        visit(name)
    if list(frame.columns) != names:
        raise ValueError("frame columns must match schema names and order exactly")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("provide a timezone-aware DatetimeIndex; no timezone is inferred")
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("timestamps must be finite, unique and increasing")
    if len(index) == 0:
        raise ValueError("provide at least one timestamp")
    if len(index) > 1 and not ((index[1:] - index[:-1]) == schema.cadence).all():
        raise ValueError("irregular grid: explicitly regularize or split clock phases first")
    if any(
        not pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
        for dtype in frame.dtypes
    ):
        raise ValueError("variables must contain real numeric measurements")
    if np.isinf(frame.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("infinite measurements require explicit source QC")
