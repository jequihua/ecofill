"""Complete user path: CSV input -> SeriesData -> profile -> plan -> execute -> benchmark -> export.

Synthetic software fixtures only: a 192-row affine series with declared raw, QC,
structural and artificial cells, and a 12-row series whose missing timestamp is
explicitly regularized. Not field validation or a scientific default.

    python examples/integration_demo.py --output <new directory>

The destination must not exist. On failure, written files stay and failure.json
records the error. `examples/read_export.py` reads the result exports back
without ecofill.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecofill import (SeriesData, TimeSeriesSchema, VariableSpec, execute, make_plan,
                     profile_events, profile_gaps)
from ecofill.benchmark import MaskCase, run_case

INPUT_FORMAT = "ecofill.integration-input"
WORKFLOW_FORMAT = "ecofill.integration-workflow"
INPUT_MASKS = ("observed", "qc_rejected", "structural", "inserted")
STEP = pd.Timedelta("15min")
START = pd.Timestamp("2025-01-01T00:00:00+00:00")
BIG = 9007199254740993
MAX_GAP = pd.Timedelta("1h")
SPLIT = 96
# Half-open, zero-based row ranges of the admitted main fixture.
MISSING = {"temperature": ((0, 2), (24, 28), (120, 129), (190, 192)),
           "pressure": ((40, 42), (160, 168))}
STRUCTURAL = {"pressure": ((160, 168),)}
QC_REJECTED = {"temperature": {48: 999.0}, "pressure": {60: -999.0}}
HIDDEN = {"temperature": (112, 114), "pressure": (144, 148)}


def _write_json(path, value) -> str:
    """Create once; return the SHA-256 of the written bytes."""
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with open(path, "xb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _iso_frame(frame):
    return frame.set_axis([stamp.isoformat() for stamp in frame.index])


def write_input(directory, frame, schema, masks, *, grid=None, case=None) -> dict:
    """Save input values, declared masks and schema before any method runs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    _iso_frame(frame).to_csv(directory / "input.csv", index_label="timestamp", mode="x")
    table = pd.concat({name: masks[name] for name in INPUT_MASKS if name in masks}, axis=1)
    table.columns = [f"{name}__{variable}" for name, variable in table.columns]
    _iso_frame(table).to_csv(directory / "masks.csv", index_label="timestamp", mode="x")
    files = {name: _sha256(directory / name) for name in ("input.csv", "masks.csv")}
    if case is not None:
        files["case.json"] = _write_json(directory / "case.json", case)
    metadata = {
        "format": INPUT_FORMAT, "format_version": 1, "timezone": str(frame.index.tz),
        "cadence_seconds": schema.cadence.total_seconds(), "time_policy": schema.time_policy,
        "variables": [{"name": item.name, "unit": item.unit, "role": item.role,
                       "derived_from": list(item.derived_from), "dtype": str(frame[item.name].dtype)}
                      for item in schema.variables],
        "masks": [name for name in INPUT_MASKS if name in masks],
        "grid": grid, "files": files,
    }
    files["schema.json"] = _write_json(directory / "schema.json", metadata)
    return files


def load_input(directory):
    """Reload saved values, schema and masks; hashes and declared dtypes must match."""
    directory = Path(directory)
    metadata = json.loads((directory / "schema.json").read_text(encoding="utf-8"))
    if metadata.get("format") != INPUT_FORMAT or metadata.get("format_version") != 1:
        raise ValueError("unsupported integration input format")
    for name, digest in metadata["files"].items():
        if _sha256(directory / name) != digest:
            raise ValueError(f"input file {name} differs from its recorded hash")
    schema = TimeSeriesSchema(pd.Timedelta(seconds=metadata["cadence_seconds"]), tuple(
        VariableSpec(item["name"], item["unit"], item["role"], tuple(item["derived_from"]))
        for item in metadata["variables"]), metadata["time_policy"])
    names = [item["name"] for item in metadata["variables"]]

    def table(name, dtypes):
        # Integer observations are parsed as int64 directly, never through float, and
        # floats use the exact round-trip parser (pandas' default parser is not exact).
        loaded = pd.read_csv(directory / name, dtype={"timestamp": str, **dtypes}, float_precision="round_trip",
                             keep_default_na=False, na_values={key: [""] for key in dtypes})
        index = pd.DatetimeIndex(pd.to_datetime(loaded.pop("timestamp"), format="ISO8601"))
        if str(index.tz) != metadata["timezone"]:
            raise ValueError("input timestamps disagree with the declared timezone")
        return loaded.set_axis(index)

    frame = table("input.csv", {item["name"]: item["dtype"] for item in metadata["variables"]})
    if list(frame.columns) != names:
        raise ValueError("input columns disagree with the declared schema")
    flat = table("masks.csv", {f"{mask}__{name}": bool for mask in metadata["masks"] for name in names})
    masks = {mask: flat[[f"{mask}__{name}" for name in names]].set_axis(names, axis=1)
             for mask in metadata["masks"]}
    case = None
    if "case.json" in metadata["files"]:
        case = json.loads((directory / "case.json").read_text(encoding="utf-8"))
    return frame, schema, masks, metadata, case


def regularize(frame, masks, grid):
    """Explicitly reindex to a declared grid; new rows are inserted and unobserved."""
    index = pd.date_range(pd.Timestamp(grid["start"]), periods=grid["periods"],
                          freq=pd.Timedelta(seconds=grid["cadence_seconds"]))
    if not frame.index.isin(index).all():
        raise ValueError("input rows lie off the declared grid")
    inserted = pd.DataFrame(~index.isin(frame.index)[:, None].repeat(len(frame.columns), axis=1),
                            index=index, columns=frame.columns)
    aligned = {name: mask.reindex(index, fill_value=False) for name, mask in masks.items()}
    aligned["inserted"] = inserted
    return frame.reindex(index), aligned


def series(frame, schema, masks, artificial=None) -> SeriesData:
    return SeriesData(frame, schema=schema, observed=masks["observed"],
                      qc_rejected=masks.get("qc_rejected"), structural=masks.get("structural"),
                      inserted=masks.get("inserted"), artificial=artificial)


def _profile(data):
    gaps = [{"variable": gap.variable, "start": gap.start.isoformat(), "end": gap.end.isoformat(),
             "start_pos": gap.start_pos, "stop_pos": gap.stop_pos, "n_steps": gap.n_steps,
             "duration_seconds": gap.duration.total_seconds(), "geometry": gap.geometry,
             "bounded": gap.bounded, "shared_variables": list(gap.shared_variables)}
            for gap in profile_gaps(data)]
    events = [{"start": event.start.isoformat(), "end": event.end.isoformat(),
               "start_pos": event.start_pos, "stop_pos": event.stop_pos, "n_steps": event.n_steps,
               "duration_seconds": event.duration.total_seconds(), "geometry": event.geometry,
               "pattern": event.pattern, "variables": list(event.variables),
               "shared_variables": list(event.shared_variables)}
              for event in profile_events(data)]
    return {"gaps": gaps, "events": events}


def _counts(result):
    imputed = result.mask("imputed")
    reasons = result.unresolved_reasons
    return {"imputed": {name: int(imputed[name].sum()) for name in imputed},
            "unresolved": {name: {str(key): int(value)
                                  for key, value in reasons[name].dropna().value_counts().sort_index().items()}
                           for name in reasons}}


def _export(result, destination, name) -> dict:
    result.export(destination / name)
    return {f"{name}/{file}": _sha256(destination / name / file) for file in result.provenance()["files"]}


def run_workflow(data, destination, *, case=None, split_position=None, max_gap=MAX_GAP,
                 source=None, guard=None, record=None) -> dict:
    """Profile, plan, execute and export one offline linear workflow on original data.

    `case` cells are hidden for the point result; `benchmark.run_case` then scores
    the same cells on a fresh fold of the original data. `guard(label)` runs at every
    execute/benchmark/export boundary; `record(stage, evidence)` receives completed
    evidence before the following guard. The point result is exported to `point/`
    before the after_execute guard; `result/` and `workflow.json` (status completed)
    are written only by the normal export stage. Failed benchmark cells stop the
    workflow after `benchmark.json` is saved. Files are created once and never removed.
    """
    guard = guard or (lambda label: None)
    record = record or (lambda stage, evidence: None)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    artificial = None
    if case is not None:
        artificial = pd.DataFrame(False, index=data.raw.index, columns=data.raw.columns)
        for position, name in case.cells:
            artificial.iat[position, artificial.columns.get_loc(name)] = True
    point = SeriesData(data.raw, schema=data.schema, observed=data.mask("observed"),
                       qc_rejected=data.mask("qc_rejected"), structural=data.mask("structural"),
                       inserted=data.mask("inserted"), artificial=artificial)
    files["profile.json"] = _write_json(destination / "profile.json", _profile(point))
    plan = make_plan(point, method="linear", mode="offline", max_gap=max_gap)
    files["plan.json"] = _write_json(destination / "plan.json", plan.to_dict())
    guard("before_execute")
    result = execute(point, plan)
    counts = _counts(result)
    # Durable point-stage evidence in the ordinary version-1 result format: values,
    # masks, states, methods, reasons and plan survive a later guard or stage failure.
    point_files = _export(result, destination, "point")
    files.update(point_files)
    record("execute", {**counts, "files": point_files})
    guard("after_execute")
    rows = None
    if case is not None:
        guard("before_benchmark")
        rows = run_case(data, case, split_position=split_position, method="linear",
                        mode="offline", max_gap=max_gap)
        benchmark = {"case": {"scenario": case.scenario, "cells": [list(cell) for cell in case.cells]},
                     "split_position": split_position, "rows": rows}
        files["benchmark.json"] = _write_json(destination / "benchmark.json", benchmark)
        record("benchmark", {"rows": rows, "sha256": files["benchmark.json"]})
        # run_case catches execution errors and reports them as failed rows: that is a stop.
        failed = [[row["variable"], row["failed"], row["failure"]] for row in rows
                  if row["failed"] or row["failure"] is not None]
        if failed:
            raise RuntimeError(f"benchmark reported failed cells; stopped before export: {failed}")
        guard("after_benchmark")
    guard("before_export")
    files.update(_export(result, destination, "result"))
    record("export", {"files": dict(files)})
    guard("after_export")
    manifest = {
        "format": WORKFLOW_FORMAT, "format_version": 1, "status": "completed",
        "source": source, "data_fingerprint": data.fingerprint,
        "point_fingerprint": point.fingerprint,
        "policy": {"method": "linear", "mode": "offline", "max_gap_seconds": max_gap.total_seconds()},
        "split_position": split_position,
        "case": None if case is None else {"scenario": case.scenario,
                                           "cells": [list(cell) for cell in case.cells]},
        "counts": counts, "files": files,
    }
    manifest_sha = _write_json(destination / "workflow.json", manifest)
    return {"counts": counts, "rows": rows, "files": files, "workflow_sha256": manifest_sha,
            "result": result}


def synthetic_primary():
    """The admitted 192-row main fixture: frame, schema, declared masks and hidden case."""
    index = pd.date_range(START, periods=192, freq=STEP)
    i = np.arange(192)
    truth = pd.DataFrame({"temperature": 10 + 0.25 * i, "pressure": 100.0 + 2 * i,
                          "count": np.full(192, BIG, dtype=np.int64)}, index=index)
    frame = truth.copy()
    for name, spans in MISSING.items():
        for start, stop in spans:
            frame.iloc[start:stop, frame.columns.get_loc(name)] = np.nan
    qc = pd.DataFrame(False, index=index, columns=frame.columns)
    for name, cells in QC_REJECTED.items():
        for row, value in cells.items():
            frame.iat[row, frame.columns.get_loc(name)] = value
            qc.iat[row, qc.columns.get_loc(name)] = True
    structural = pd.DataFrame(False, index=index, columns=frame.columns)
    for name, spans in STRUCTURAL.items():
        for start, stop in spans:
            structural.iloc[start:stop, structural.columns.get_loc(name)] = True
    schema = TimeSeriesSchema(STEP, (VariableSpec("temperature", "degC", "target"),
                                     VariableSpec("pressure", "cmH2O", "target"),
                                     VariableSpec("count", "count", "diagnostic")))
    masks = {"observed": frame.notna() & ~qc, "qc_rejected": qc, "structural": structural}
    case = MaskCase("integration_hidden", tuple(
        (row, name) for name, (start, stop) in HIDDEN.items() for row in range(start, stop)))
    return truth, frame, schema, masks, case


def synthetic_timestamp():
    """12 rows of temperature 20+i on the same grid; row 5 is absent from the input."""
    index = pd.date_range(START, periods=12, freq=STEP)
    frame = pd.DataFrame({"temperature": 20.0 + np.arange(12)}, index=index).drop(index[5])
    schema = TimeSeriesSchema(STEP, (VariableSpec("temperature", "degC", "target"),))
    grid = {"start": START.isoformat(), "periods": 12, "cadence_seconds": STEP.total_seconds()}
    return frame, schema, {"observed": frame.notna()}, grid


def _primary(root):
    _, frame, schema, masks, case = synthetic_primary()
    cells = {"scenario": case.scenario, "cells": [list(cell) for cell in case.cells],
             "split_position": SPLIT}
    write_input(root / "input", frame, schema, masks, case=cells)
    frame, schema, masks, metadata, saved = load_input(root / "input")
    case = MaskCase(saved["scenario"], tuple((row, name) for row, name in saved["cells"]))
    data = series(frame, schema, masks)
    outcome = run_workflow(data, root / "workflow", case=case, split_position=saved["split_position"],
                           source={"input": "input", "files": metadata["files"]})
    rows = outcome["rows"]
    return {"imputed": outcome["counts"]["imputed"], "unresolved": outcome["counts"]["unresolved"],
            "benchmark": [[row["variable"], row["hidden"], row["predicted"], row["abstained"],
                           row["failed"], row["mae"]] for row in rows],
            "count_exact": outcome["result"].completed["count"].tolist() == [BIG] * len(frame)}


def _timestamp(root):
    frame, schema, masks, grid = synthetic_timestamp()
    write_input(root / "input", frame, schema, masks, grid=grid)
    frame, schema, masks, metadata, _ = load_input(root / "input")
    try:  # the strict core never repairs clocks
        series(frame, schema, masks)
        rejected = None
    except ValueError as exc:
        rejected = str(exc)
    if rejected is None:
        raise RuntimeError("unregularized timestamp input was accepted")
    frame, masks = regularize(frame, masks, metadata["grid"])
    outcome = run_workflow(series(frame, schema, masks), root / "workflow",
                           source={"input": "input", "files": metadata["files"],
                                   "regularized": metadata["grid"], "rejected_before": rejected})
    result = outcome["result"]
    return {"rejected": rejected, "rows": len(result.raw),
            "value": float(result.completed["temperature"].iloc[5]),
            "inserted": bool(result.mask("inserted")["temperature"].iloc[5]),
            "imputed": bool(result.mask("imputed")["temperature"].iloc[5]),
            "state": result.states["temperature"].iloc[5]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, help="new directory; must not exist")
    root = Path(parser.parse_args(argv).output)
    root.mkdir(parents=True, exist_ok=False)
    try:
        summary = {"primary": _primary(root / "primary"), "timestamp": _timestamp(root / "timestamp")}
    except Exception as exc:
        # Keep partial evidence: nothing written so far is deleted.
        _write_json(root / "failure.json", {
            "error": f"{type(exc).__name__}: {exc}",
            "files": sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())})
        raise
    _write_json(root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
