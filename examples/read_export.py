"""Standard-library reference reader for existing ecofill export formats.

    python -I -S read_export.py result <directory>
    python -I -S read_export.py uncertainty <directory>

`result` reads an `ImputationResult.export` directory (format_version 1);
`uncertainty` reads an `export_uncertainty` directory (format_version 1). The
reader imports no ecofill, NumPy or pandas and opens only declared filenames.
It prints one normalized JSON document or exits 2 with the rejection reason.
It checks structure and internal consistency: strict JSON, known versions and
files, exact headers, aligned timestamp/variable axes, number and boolean
tokens, and draw/sample identities. Whether the values are the right ones is
a separate comparison against independently known expectations. This is a
reference consumer of current formats, not an authenticated archive.
"""

import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys

RESULT_FILES = ("raw.csv", "completed.csv", "masks.csv", "cells.csv", "provenance.json")
RESULT_MASKS = ("raw_present", "observed", "donor", "absent", "qc_rejected", "structural",
                "artificial", "inserted", "imputed")
DRAW_FILES = ("draw_cells.csv", "draws.csv", "draw_samples.csv")
UNCERTAINTY_FILES = ("intervals.csv", "draw_cells.csv", "draws.csv", "draw_samples.csv", "flux.csv",
                     "uncertainty.json")
INTEGER = re.compile(r"-?[0-9]+")
DECIMAL = re.compile(r"-?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?")


class ExportError(ValueError):
    pass


def strict_json(path):
    def constant(token):
        raise ExportError(f"{path.name}: nonstandard JSON constant {token}")

    def unique(pairs):
        keys = [key for key, _ in pairs]
        if len(set(keys)) != len(keys):
            raise ExportError(f"{path.name}: duplicate JSON key")
        return dict(pairs)

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=constant,
                          object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        raise ExportError(f"{path.name}: invalid JSON: {exc}") from exc


def number(text, where):
    """Empty is missing (None); integer tokens stay exact ints; no NaN/inf tokens."""
    if text == "":
        return None
    if INTEGER.fullmatch(text):
        return int(text)
    if DECIMAL.fullmatch(text):
        return float(text)
    raise ExportError(f"{where}: invalid number {text!r}")


def boolean(text, spelling, where):
    if text not in spelling:
        raise ExportError(f"{where}: malformed boolean {text!r}")
    return spelling[text]


def stamp(text, where):
    try:
        value = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ExportError(f"{where}: invalid timestamp {text!r}") from exc
    if value.utcoffset() is None:
        raise ExportError(f"{where}: timestamp {text!r} has no offset")
    return value


def same_axis(texts, reference, where):
    """Timestamps must be the same instants, offsets and order as the reference axis."""
    values = [stamp(text, where) for text in texts]
    if len(values) != len(reference) or any(
            a != b or a.utcoffset() != b.utcoffset() for a, b in zip(values, reference)):
        raise ExportError(f"{where}: timestamp axis differs")


def table(root, name, header):
    with open(root / name, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        found = next(reader, None)
        if found != header:
            raise ExportError(f"{name}: header {found!r} differs from {header!r}")
        rows = list(reader)
    if any(len(row) != len(header) for row in rows):
        raise ExportError(f"{name}: ragged row")
    return rows


def declared(root, meta, allowed, required):
    files = meta.get("files")
    if (not isinstance(files, list) or len(set(files)) != len(files)
            or not set(required) <= set(files) <= set(allowed)):
        raise ExportError(f"declared files {files!r} are not a known complete set")
    for name in files:
        if not (root / name).is_file():
            raise ExportError(f"declared file {name} is missing")
    return files


def version(meta, kind=None):
    if type(meta.get("format_version")) is not int or meta["format_version"] != 1:
        raise ExportError(f"unknown format_version {meta.get('format_version')!r}")
    if kind is not None and meta.get("kind") != kind:
        raise ExportError(f"unknown export kind {meta.get('kind')!r}")


def schema_axes(meta):
    variables = meta["schema"]["variables"]
    names = [item["name"] for item in variables]
    if len(set(names)) != len(names) or not names:
        raise ExportError("schema variables must be unique and nonempty")
    return names, {item["name"]: item["unit"] for item in variables}


def read_result(root) -> dict:
    root = Path(root)
    meta = strict_json(root / "provenance.json")
    version(meta)
    declared(root, meta, RESULT_FILES, RESULT_FILES)
    names, units = schema_axes(meta)
    axis = [stamp(text, "provenance.json") for text in meta["timestamps"]]
    frames = {}
    for name in ("raw", "completed"):
        rows = table(root, f"{name}.csv", ["timestamp", *names])
        same_axis([row[0] for row in rows], axis, f"{name}.csv")
        frames[name] = {variable: [number(row[1 + j], f"{name}.csv") for row in rows]
                        for j, variable in enumerate(names)}
    mask_names = meta["masks"]
    if not isinstance(mask_names, list) or len(set(mask_names)) != len(mask_names) \
            or not set(RESULT_MASKS) <= set(mask_names):
        raise ExportError(f"masks {mask_names!r} lack the version-1 provenance masks")
    columns = [f"{mask}__{variable}" for mask in mask_names for variable in names]
    rows = table(root, "masks.csv", ["timestamp", *columns])
    same_axis([row[0] for row in rows], axis, "masks.csv")
    spelling = {"True": True, "False": False}
    masks = {mask: {variable: [boolean(row[1 + columns.index(f"{mask}__{variable}")], spelling,
                                       "masks.csv") for row in rows]
                    for variable in names} for mask in mask_names}
    quality = "quality" in meta
    header = ["timestamp", "variable", "state", "method", "reason"] + (["quality"] if quality else [])
    rows = table(root, "cells.csv", header)
    same_axis(_grid(rows, names, None, "cells.csv"), axis, "cells.csv")
    cells = {variable: {field: [None if row[2 + f] == "" else row[2 + f]
                                for row in rows[j::len(names)]]
                        for f, field in enumerate(header[2:])}
             for j, variable in enumerate(names)}
    if any(state is None for item in cells.values() for state in item["state"]):
        raise ExportError("cells.csv: missing state")
    counts = {}
    for item in cells.values():
        for state in item["state"]:
            counts[state] = counts.get(state, 0) + 1
    if counts != meta["state_counts"]:
        raise ExportError("cells.csv states disagree with provenance state_counts")
    for variable in names:
        raw, completed = frames["raw"][variable], frames["completed"][variable]
        m = {name: masks[name][variable] for name in mask_names}
        for k in range(len(axis)):
            if m["raw_present"][k] != (raw[k] is not None):
                raise ExportError(f"raw_present disagrees with raw.csv for {variable}")
            if m["donor"][k] != (m["observed"][k] and not m["artificial"][k]):
                raise ExportError(f"donor mask disagrees with observed/artificial for {variable}")
            if m["donor"][k] and completed[k] != raw[k]:
                raise ExportError(f"completed.csv changes a donor observation of {variable}")
            if m["imputed"][k] != (cells[variable]["state"][k] == "imputed"):
                raise ExportError(f"imputed mask disagrees with cell state for {variable}")
    output = {"kind": "result", "format_version": 1, "files": meta["files"],
              "timestamps": [value.isoformat() for value in axis], "variables": names,
              "units": units, "schema": meta["schema"], "data_fingerprint": meta["data_fingerprint"],
              "plan": meta["plan"], "state_counts": meta["state_counts"],
              "raw": frames["raw"], "completed": frames["completed"], "masks": masks, "cells": cells}
    for key in ("backend", "quality"):
        if key in meta:
            output[key] = meta[key]
    return output


def _grid(rows, names, units, where):
    """Split rows into a timestamp axis over the ordered variables with their units."""
    if len(rows) % len(names) or [row[1] for row in rows] != names * (len(rows) // len(names)):
        raise ExportError(f"{where}: variable axis differs")
    stamps = [row[0] for row in rows[::len(names)]]
    if any(row[0] != stamps[k // len(names)] for k, row in enumerate(rows)):
        raise ExportError(f"{where}: timestamp axis differs within a row group")
    if units is not None and any(row[2] != units[row[1]] for row in rows):
        raise ExportError(f"{where}: unit differs from the schema")
    return stamps


def read_uncertainty(root) -> dict:
    root = Path(root)
    meta = strict_json(root / "uncertainty.json")
    version(meta, "ecofill_block_uncertainty")
    files = declared(root, meta, UNCERTAINTY_FILES, ("uncertainty.json",))
    # Component metadata and declared files must agree in both directions; a
    # calibration-only export has null intervals/draws, empty flux and no CSVs.
    for component, required in (("intervals", ("intervals.csv",)), ("draws", DRAW_FILES), ("flux", ("flux.csv",))):
        present = meta.get(component) not in (None, [])
        declared_files = [name for name in required if name in files]
        if (present and len(declared_files) != len(required)) or (not present and declared_files):
            raise ExportError(f"{component} metadata disagrees with declared files {declared_files!r}")
    names, units = schema_axes(meta)
    calibration = meta["calibration"]
    identity = calibration["identity"]
    output = {"kind": "uncertainty", "format_version": 1, "files": files, "variables": names,
              "units": units, "schema": meta["schema"], "calibration_identity": identity,
              "calibration": calibration, "intervals": None, "draws": None, "flux": None}
    axes = []
    spelling = {"true": True, "false": False}
    if "intervals.csv" in files:
        if meta["intervals"] is None or meta["intervals"]["calibration_identity"] != identity:
            raise ExportError("intervals metadata is missing or from another calibration")
        rows = table(root, "intervals.csv", ["timestamp", "variable", "unit", "point", "lower",
                                             "upper", "status"])
        axes.append(("intervals.csv", _grid(rows, names, units, "intervals.csv")))
        output["intervals"] = {
            "metadata": meta["intervals"],
            "values": {variable: {field: [number(row[3 + f], "intervals.csv") if f < 3 else
                                          (row[6] or None) for row in rows[j::len(names)]]
                                  for f, field in enumerate(("point", "lower", "upper", "status"))}
                       for j, variable in enumerate(names)}}
    if "draws.csv" in files:
        draws = meta["draws"]
        if draws is None or draws["calibration_identity"] != identity:
            raise ExportError("draw metadata is missing or from another calibration")
        draw_ids = draws["draw_ids"]
        if not draw_ids or len(set(draw_ids)) != len(draw_ids):
            raise ExportError("draw IDs must be unique and nonempty")
        rows = table(root, "draw_cells.csv", ["timestamp", "variable", "unit", "status", "reason",
                                              "observed", "structural", "supported"])
        axes.append(("draw_cells.csv", _grid(rows, names, units, "draw_cells.csv")))
        cells = {variable: {"status": [row[3] for row in rows[j::len(names)]],
                            "reason": [row[4] or None for row in rows[j::len(names)]],
                            **{field: [boolean(row[5 + f], spelling, "draw_cells.csv")
                                       for row in rows[j::len(names)]]
                               for f, field in enumerate(("observed", "structural", "supported"))}}
                 for j, variable in enumerate(names)}
        rows = table(root, "draws.csv", ["draw_id", "timestamp", "variable", "value"])
        if len(rows) % len(draw_ids):
            raise ExportError("draws.csv: draw IDs are inconsistent with metadata")
        size = len(rows) // len(draw_ids)
        values = {}
        for d, draw_id in enumerate(draw_ids):
            block = rows[d * size:(d + 1) * size]
            if any(row[0] != draw_id for row in block):
                raise ExportError("draws.csv: draw IDs are inconsistent with metadata")
            axes.append((f"draws.csv:{draw_id}", _grid([row[1:3] for row in block], names, None,
                                                       "draws.csv")))
            values[draw_id] = {variable: [number(row[3], "draws.csv") for row in block[j::len(names)]]
                               for j, variable in enumerate(names)}
        samples = table(root, "draw_samples.csv", ["draw_id", "target_block", "source_block",
                                                   "source_physical_block"])
        declared_samples = [[item["draw_id"], item["target_block"], item["source_block"],
                             item["source_physical_block"]] for item in draws["samples"]]
        banks = {block for item in calibration["strata"] for block in item["block_ids"]}
        targets = {item["case_id"] for item in draws["blocks"]}
        if (samples != declared_samples or any(row[0] not in draw_ids for row in samples)
                or any(row[1] not in targets or row[2] not in banks for row in samples)):
            raise ExportError("draw_samples.csv: draw/sample identities are inconsistent")
        output["draws"] = {"metadata": draws, "draw_ids": draw_ids, "cells": cells, "values": values,
                           "samples": samples}
    if axes:
        reference = [stamp(text, axes[0][0]) for text in axes[0][1]]
        step = meta["schema"]["cadence_seconds"]
        if any((b - a).total_seconds() != step for a, b in zip(reference, reference[1:])):
            raise ExportError(f"{axes[0][0]}: timestamps are not on the declared cadence")
        for where, texts in axes[1:]:
            same_axis(texts, reference, where)
        output["timestamps"] = [value.isoformat() for value in reference]
    if "flux.csv" in files:
        rows = table(root, "flux.csv", ["target", "draw_id", "signed_umol_co2_m-2", "covered_seconds",
                                        "total_seconds", "complete_period"])
        if output["draws"] is None or any(row[1] not in output["draws"]["draw_ids"] for row in rows):
            raise ExportError("flux.csv: draw IDs are inconsistent with metadata")
        output["flux"] = [[row[0], row[1], number(row[2], "flux.csv"), number(row[3], "flux.csv"),
                           number(row[4], "flux.csv"), boolean(row[5], spelling, "flux.csv")]
                          for row in rows]
    return output


READERS = {"result": read_result, "uncertainty": read_uncertainty}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in READERS:
        print("usage: read_export.py result|uncertainty <directory>", file=sys.stderr)
        return 2
    try:
        document = READERS[argv[0]](argv[1])
    except (ExportError, OSError, KeyError, TypeError, IndexError) as exc:
        print(f"invalid {argv[0]} export: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(document, allow_nan=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
