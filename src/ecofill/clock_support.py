"""Clock-aware event exclusion and measurement-axis diagnostics on an explicit UTC grid.

Source-independent and bounded: the caller supplies the UTC grid, channel names
and event records whose endpoints each carry a clock declaration, `utc`,
`local_utc-6` (fixed local time six hours behind UTC) or `unknown`. Nothing is
read from files and no clock is inferred from measurement values or errors. No
DST or arbitrary time-zone search is performed.

- A naive endpoint declared `unknown` has two UTC readings (as stored, +6 h);
  `utc` and `local_utc-6` have one. An aware endpoint supplies its own offset,
  and a declaration that contradicts that offset is invalid.
- Intervals keep every chronologically valid start/end combination of their
  endpoints' readings (mixed readings included), and exclude the union of those
  inclusive spans on the grid. Instants exclude only matching grid timestamps;
  off-grid instants are recorded, never rounded. Installation allows support only
  at or after every plausible installation instant.
- Invalid or missing evidence, unsupported declarations and intervals with no
  chronological reading make their whole channel scope unresolved (not zero
  exclusions). Uncertain cells are not called bad measurements, and the caller's
  observed/QC masks are never read or changed.

`check_axis` reports measurement-axis problems (invalid, duplicate, non-increasing,
off-cadence and missing steps, UTC-local offsets other than 6 h and mixed offset
segments). An axis without a confirmed convention, or with contradictions, is not
usable for scoring; `require_usable_axis` raises. Nothing is sorted, shifted or
filled; regularization stays with the caller.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

DECLARATIONS = ("utc", "local_utc-6", "unknown")
KINDS = ("instant", "interval", "installation")
LOCAL_OFFSET = pd.Timedelta(hours=6)  # UTC = local wall time + 6 h
OWNER = "source/field owner"


@dataclass(frozen=True)
class ClockEvent:
    """One metadata event with its original endpoint values and per-endpoint declarations."""

    record_id: str
    kind: str
    scope: tuple[str, ...]
    start: object
    start_clock: str
    end: object = None
    end_clock: str | None = None
    source: str = ""


@dataclass(frozen=True)
class EventCollection:
    """A named event list for a channel scope; `events=None` means the required list is missing."""

    name: str
    scope: tuple[str, ...]
    events: tuple[ClockEvent, ...] | None


@dataclass(frozen=True)
class Eligibility:
    excluded: pd.DataFrame
    unresolved: pd.DataFrame
    eligible: pd.DataFrame
    records: tuple[dict, ...]
    collections: tuple[dict, ...]

    def summary(self) -> dict:
        """JSON-safe counts per channel and record status; no timestamps."""
        statuses = {}
        for record in self.records:
            statuses[record["status"]] = statuses.get(record["status"], 0) + 1
        return {
            "rows": len(self.eligible.index),
            "channels": {name: {"excluded": int(self.excluded[name].sum()),
                                "unresolved": int(self.unresolved[name].sum()),
                                "eligible": int(self.eligible[name].sum())} for name in self.eligible.columns},
            "record_status": statuses,
            "collections": [dict(item) for item in self.collections],
        }


def _text(value) -> str:
    if isinstance(value, pd.Timestamp):
        return "NaT" if pd.isna(value) else value.isoformat()
    return repr(value)


def readings(value, clock) -> tuple[list[pd.Timestamp], str | None]:
    """Candidate UTC instants of one endpoint, or ([], reason) when invalid."""
    if clock not in DECLARATIONS:
        return [], f"unsupported clock declaration {clock!r}"
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return [], "missing or non-finite endpoint"
    if isinstance(value, str):
        try:
            value = pd.Timestamp(value)
        except (ValueError, TypeError):
            return [], "unparseable endpoint"
    if not isinstance(value, pd.Timestamp) or pd.isna(value):
        return [], "invalid endpoint"
    if value.tz is not None:
        offset = value.utcoffset()
        if (clock == "utc" and offset != pd.Timedelta(0)) or (clock == "local_utc-6" and offset != -LOCAL_OFFSET):
            return [], f"aware offset {offset} contradicts declaration {clock}"
        return [value.tz_convert("UTC")], None
    naive = value.tz_localize("UTC")
    if clock == "utc":
        return [naive], None
    if clock == "local_utc-6":
        return [naive + LOCAL_OFFSET], None
    return [naive, naive + LOCAL_OFFSET], None


def interval_combinations(starts, ends) -> list[tuple[pd.Timestamp, pd.Timestamp, bool]]:
    """Every start reading with every end reading (mixed clocks included) and whether it is ordered."""
    return [(start, end, start <= end) for start in starts for end in ends]


def _check_grid(grid, channels):
    if not isinstance(grid, pd.DatetimeIndex) or grid.tz is None or str(grid.tz) != "UTC":
        raise ValueError("grid must be an explicit UTC DatetimeIndex")
    if grid.hasnans or not grid.is_unique or not grid.is_monotonic_increasing:
        raise ValueError("grid must be unique and increasing")
    if not isinstance(channels, tuple) or not channels or len(set(channels)) != len(channels):
        raise ValueError("channels must be a nonempty tuple of unique names")


def assess_events(grid: pd.DatetimeIndex, channels: tuple[str, ...],
                  collections: tuple[EventCollection, ...]) -> Eligibility:
    """Conservative eligibility: excluded under any retained reading, unresolved where unknowable."""
    _check_grid(grid, channels)
    excluded = pd.DataFrame(False, index=grid, columns=list(channels))
    unresolved = pd.DataFrame(False, index=grid, columns=list(channels))
    records, summaries = [], []
    for collection in collections:
        scope = list(collection.scope)
        if not scope or any(name not in channels for name in scope):
            raise ValueError(f"collection {collection.name} names unknown channels")
        if collection.events is None:
            unresolved.loc[:, scope] = True
            summaries.append({"name": collection.name, "scope": scope, "status": "missing", "events": None,
                              "actor": OWNER, "action": f"supply the {collection.name} records"})
            continue
        summaries.append({"name": collection.name, "scope": scope,
                          "status": "empty" if not collection.events else "present",
                          "events": len(collection.events), "actor": None, "action": None})
        for event in collection.events:
            record = _assess(event, collection, grid, scope)
            rows, event_scope = record.pop("_rows"), record["scope"]
            if record["status"] == "invalid":
                unresolved.loc[:, event_scope] = True
            elif rows is not None and rows.any():
                excluded.loc[rows, event_scope] = True
            records.append(record)
    eligible = ~excluded & ~unresolved
    return Eligibility(excluded, unresolved, eligible, tuple(records), tuple(summaries))


def _assess(event: ClockEvent, collection, grid, collection_scope) -> dict:
    scope = list(event.scope) if event.scope else collection_scope
    record = {"record_id": event.record_id, "collection": collection.name, "kind": event.kind, "scope": scope,
              "source": event.source, "original": {"start": _text(event.start)},
              "declarations": {"start": event.start_clock}, "status": "supported", "reason": None,
              "actor": None, "action": None, "off_grid": [], "combinations": [], "excluded_rows": 0, "_rows": None}

    def invalid(reason, action):
        record.update(status="invalid", reason=reason, actor=OWNER, action=action)
        return record

    if event.kind not in KINDS:
        return invalid(f"unsupported event kind {event.kind!r}", "declare the event kind")
    if any(name not in collection_scope for name in scope):
        return invalid("event scope exceeds its collection scope", "declare a valid channel scope")
    starts, problem = readings(event.start, event.start_clock)
    record["candidates"] = {"start": [item.isoformat() for item in starts]}
    if problem:
        return invalid(f"start: {problem}", "supply a valid start timestamp and clock declaration")
    ambiguous = len(starts) > 1
    if event.kind == "instant":
        rows = grid.isin(starts)
        record["off_grid"] = [item.isoformat() for item in starts if item not in grid]
    elif event.kind == "installation":
        rows = grid < max(starts)  # a grid timestamp equal to installation is eligible
        record["reason"] = "support only at or after every plausible installation instant"
    else:
        record["original"]["end"] = _text(event.end)
        record["declarations"]["end"] = event.end_clock
        ends, problem = readings(event.end, event.end_clock)
        record["candidates"]["end"] = [item.isoformat() for item in ends]
        if problem:
            return invalid(f"end: {problem}", "supply a valid end timestamp and clock declaration")
        combinations = interval_combinations(starts, ends)
        record["combinations"] = [[start.isoformat(), end.isoformat(), valid] for start, end, valid in combinations]
        valid = [(start, end) for start, end, ok in combinations if ok]
        if not valid:
            return invalid("no chronologically valid start/end reading",
                           "confirm the interval endpoints and their clocks")
        rows = np.zeros(len(grid), dtype=bool)
        for start, end in valid:
            rows |= (grid >= start) & (grid <= end)
        ambiguous = ambiguous or len(ends) > 1
    record["_rows"] = rows
    record["excluded_rows"] = int(rows.sum())
    if ambiguous:
        record.update(status="ambiguous", actor=OWNER,
                      action="declare the clock (UTC or UTC-6) of each unknown endpoint",
                      reason=record["reason"] or "several clock readings retained; their union is excluded")
    return record


def check_axis(utc, local=None, *, declared: str, cadence: pd.Timedelta) -> dict:
    """Measurement-axis diagnostics; `usable` only for a confirmed, consistent convention."""
    if declared not in DECLARATIONS:
        raise ValueError(f"declared must be one of {DECLARATIONS}")
    axis = pd.DatetimeIndex(utc)
    aware = axis.tz is not None
    values = axis.tz_localize(None) if aware else axis
    step = cadence.value
    finite = values[~values.isna()]
    diffs = np.diff(finite.asi8) if len(finite) > 1 else np.array([], dtype=np.int64)
    report = {
        "entries": len(axis), "declared": declared, "aware": aware,
        "invalid": int(axis.isna().sum()), "duplicates": int(finite.duplicated().sum()),
        "non_increasing_steps": int((diffs <= 0).sum()),
        "off_cadence_steps": int(((diffs > 0) & (diffs % step != 0)).sum()),
        "missing_slots": int(sum(value // step - 1 for value in diffs if value > step and value % step == 0)),
        "local": None,
    }
    reasons = []
    if declared == "unknown":
        reasons.append("axis clock convention is not confirmed")
    if aware and declared == "local_utc-6" and axis.tz is not None and \
            any(stamp.utcoffset() != -LOCAL_OFFSET for stamp in axis if not pd.isna(stamp)):
        reasons.append("aware axis offsets contradict the local_utc-6 declaration")
    if aware and declared == "utc" and any(stamp.utcoffset() != pd.Timedelta(0) for stamp in axis if not pd.isna(stamp)):
        reasons.append("aware axis offsets contradict the utc declaration")
    for key in ("invalid", "duplicates", "non_increasing_steps"):
        if report[key]:
            reasons.append(f"axis has {report[key]} {key.replace('_', ' ')}")
    if local is not None:
        local_axis = pd.DatetimeIndex(local)
        local_values = local_axis.tz_localize(None) if local_axis.tz is not None else local_axis
        if len(local_values) != len(values):
            reasons.append("local axis length differs from the UTC axis")
            report["local"] = {"entries": len(local_values), "length_matches": False}
        else:
            offsets = (values - local_values) / pd.Timedelta(hours=1)
            hours = [None if pd.isna(value) else float(value) for value in offsets]
            segments = []
            for position, value in enumerate(hours):
                if segments and segments[-1][2] == value:
                    segments[-1][1] = position + 1
                else:
                    segments.append([position, position + 1, value])
            counts = {}
            for value in hours:
                key = "invalid" if value is None else str(value)
                counts[key] = counts.get(key, 0) + 1
            contradictions = sum(value != 6.0 for value in hours)
            report["local"] = {"entries": len(local_values), "length_matches": True,
                               "utc_minus_local_hours": counts, "rows_not_6h": contradictions,
                               "segments": [[start, stop, value] for start, stop, value in segments]}
            if contradictions:
                reasons.append(f"{contradictions} rows have UTC minus local other than 6 h "
                               f"({len(segments)} offset segments); no convention is selected")
    report["usable"] = not reasons
    report["reasons"] = reasons
    report["actor"] = None if not reasons else OWNER
    report["action"] = None if not reasons else "confirm the measurement clock convention and row correspondence"
    return report


def require_usable_axis(report: dict) -> None:
    """Stop the numerical path before any model is constructed."""
    if not report.get("usable"):
        raise ValueError("measurement axis is not usable for scoring: " + "; ".join(report.get("reasons", [])))
