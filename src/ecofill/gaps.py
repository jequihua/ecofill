"""Missing runs with inclusive endpoints and duration = missing slots * cadence."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import SeriesData
from .schema import TimeSeriesSchema, validate_frame


@dataclass(frozen=True)
class Gap:
    variable: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_steps: int
    duration: pd.Timedelta
    bounded: bool
    shared_variables: tuple[str, ...]
    geometry: str
    start_pos: int
    stop_pos: int


@dataclass(frozen=True)
class GapEvent:
    start: pd.Timestamp
    end: pd.Timestamp
    start_pos: int
    stop_pos: int
    n_steps: int
    duration: pd.Timedelta
    geometry: str
    pattern: str
    variables: tuple[str, ...]
    shared_variables: tuple[str, ...]


def _input(frame, schema):
    if isinstance(frame, SeriesData):
        if schema is not None and schema != frame.schema:
            raise ValueError("schema disagrees with SeriesData")
        return frame.available, frame.schema
    if schema is None:
        raise ValueError("schema is required for a DataFrame")
    validate_frame(frame, schema)
    return frame, schema


def _geometry(start, stop, length):
    if start == 0 and stop == length:
        return "full"
    if start == 0:
        return "leading"
    if stop == length:
        return "trailing"
    return "bounded"


def profile_gaps(
    frame: pd.DataFrame | SeriesData, *, schema: TimeSeriesSchema | None = None
) -> tuple[Gap, ...]:
    """Describe runs without authorizing filling or inferring causes.

    shared_variables lists channels missing throughout this entire run; it is
    not a count of independent donors or a complete overlap-event segmentation.
    """
    source = frame
    if isinstance(source, SeriesData):
        if schema is not None and schema != source.schema:
            raise ValueError("schema disagrees with SeriesData")
        cached = getattr(source, "_gap_profile", None)
        if cached is not None:
            return cached
    frame, schema = _input(frame, schema)
    missing = frame.isna().to_numpy()
    outage = missing.copy()
    if isinstance(source, SeriesData):
        outage &= ~source.mask("structural").to_numpy()
    gaps = []
    structural = (source.mask("structural").to_numpy() if isinstance(source, SeriesData)
                  else np.zeros_like(missing))
    for column, variable in enumerate(frame.columns):
        # Structural absence and an adjacent outage have different causes and
        # eligibility, even when both appear as NaN in the available frame.
        kind = missing[:, column].astype(np.int8) + structural[:, column].astype(np.int8)
        bounds = np.flatnonzero(np.diff(np.r_[0, kind, 0]) != 0)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if kind[start] == 0:
                continue
            count = int(stop - start)
            geometry = _geometry(start, stop, len(frame))
            gaps.append(Gap(
                variable=variable, start=frame.index[start], end=frame.index[stop - 1],
                n_steps=count, duration=count * schema.cadence,
                bounded=geometry == "bounded",
                shared_variables=tuple(frame.columns[outage[start:stop].all(axis=0)]),
                geometry=geometry, start_pos=int(start), stop_pos=int(stop),
            ))
    result = tuple(gaps)
    if isinstance(source, SeriesData):
        source._gap_profile = result
    return result


def profile_events(
    frame: pd.DataFrame | SeriesData, *, schema: TimeSeriesSchema | None = None
) -> tuple[GapEvent, ...]:
    """Report non-structural full channels separately from transient outages."""
    source = frame
    if isinstance(source, SeriesData):
        if schema is not None and schema != source.schema:
            raise ValueError("schema disagrees with SeriesData")
        cached = getattr(source, "_event_profile", None)
        if cached is not None:
            return cached
    frame, schema = _input(frame, schema)
    missing = frame.isna().to_numpy()
    if isinstance(source, SeriesData):
        missing &= ~source.mask("structural").to_numpy()
    full_channels = missing.all(axis=0)
    events = []
    for column in np.flatnonzero(full_channels):
        variable = frame.columns[column]
        events.append(GapEvent(
            start=frame.index[0], end=frame.index[-1],
            start_pos=0, stop_pos=len(frame), n_steps=len(frame),
            duration=len(frame) * schema.cadence, geometry="full",
            pattern="full_channel", variables=(variable,), shared_variables=(variable,),
        ))
    missing[:, full_channels] = False
    any_missing = missing.any(axis=1)
    changes = np.diff(np.r_[False, any_missing, False].astype(np.int8))
    for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        segment = missing[start:stop]
        variables = tuple(frame.columns[segment.any(axis=0)])
        shared = tuple(frame.columns[segment.all(axis=0)])
        if not (segment == segment[0]).all():
            pattern = "staggered"
        elif len(variables) > 1:
            pattern = "shared"
        else:
            pattern = "single"
        count = int(stop - start)
        events.append(GapEvent(
            start=frame.index[start], end=frame.index[stop - 1],
            start_pos=int(start), stop_pos=int(stop), n_steps=count,
            duration=count * schema.cadence,
            geometry=("full_span" if start == 0 and stop == len(frame)
                      else _geometry(start, stop, len(frame))), pattern=pattern,
            variables=variables, shared_variables=shared,
        ))
    result = tuple(events)
    if isinstance(source, SeriesData):
        source._event_profile = result
    return result


def validate_gap_selection(data, selected, *, variables=None):
    """Validate exact original whole gaps; selecting never changes donor masks."""
    if selected is None:
        return None
    if (not isinstance(selected, tuple)
            or any(not isinstance(item, tuple) or len(item) != 3
                   or not isinstance(item[0], str)
                   or type(item[1]) is not int or type(item[2]) is not int
                   for item in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("target_gaps must be a tuple of unique (variable, start, stop) gaps")
    existing = {(gap.variable, gap.start_pos, gap.stop_pos) for gap in profile_gaps(data)}
    if any(item not in existing or (variables is not None and item[0] not in variables)
           for item in selected):
        raise ValueError("target_gaps must select existing whole gaps of the target variables")
    return frozenset(selected)


def validate_support_interval(data, interval):
    """Resolve an optional observed-context domain without changing source masks."""
    if interval is None:
        return 0, len(data.raw)
    if (not isinstance(interval, tuple) or len(interval) != 2
            or any(type(value) is not int for value in interval)
            or not 0 <= interval[0] < interval[1] <= len(data.raw)):
        raise ValueError("support_interval must be a nonempty half-open row domain")
    return interval
