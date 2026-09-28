"""Fresh-process installed-wheel blend -> block calibration -> draws -> flux -> export smoke."""

import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from ecofill import (BlockCase, BlockRecord, DerivedStep, EnsembleMember, EnsembleRecipe,
                     SeriesData, SupportSpan, SupportStratum, TimeSeriesSchema, VariableSpec,
                     blend, block_intervals, calibrate_blocks, configuration_identity,
                     draw_flux_summary, evaluate_blocks, execute, export_uncertainty, make_plan,
                     sample_block_draws)


step = pd.Timedelta("30min")
big = 2 ** 53 + 1
nee_unit = "umol CO2 m-2 s-1"
schema = TimeSeriesSchema(step, (
    VariableSpec("ta", "degC", "driver"),
    VariableSpec("nee", nee_unit, "response", ("ta",)),
    VariableSpec("nee_mg", "mg CO2 m-2 s-1", "derived", ("nee",)),
    VariableSpec("count", "count", "diagnostic"),
))
transforms = (
    DerivedStep("nee", ("ta",), ("degC",), nee_unit, "q10-respiration-v1",
                lambda ta: 2.0 * 2.0 ** ((ta - 10.0) / 10.0), ("ta",)),
    DerivedStep("nee_mg", ("nee",), (nee_unit,), "mg CO2 m-2 s-1", "umol-to-mg-co2-v1",
                lambda nee: nee * 44.0095e-3, ("nee",)),
)


def members_for(data):
    return {"lin": execute(data, make_plan(data, method="linear", mode="offline", max_gap=4 * step)),
            "per": execute(data, make_plan(data, method="persistence", mode="offline", max_gap=4 * step))}


def fold(ta, hidden=(), missing=(), start="2025-06-01"):
    index = pd.date_range(start, periods=len(ta), freq=step, tz="UTC")
    nee = 2.0 * 2.0 ** ((ta - 10.0) / 10.0)
    frame = pd.DataFrame({"ta": ta, "nee": nee, "nee_mg": nee * 44.0095e-3,
                          "count": np.full(len(ta), big, dtype=np.int64)}, index=index)
    frame.iloc[list(missing), :3] = np.nan
    artificial = pd.DataFrame(False, index=index, columns=frame.columns)
    artificial.iloc[list(hidden), 0] = True
    data = SeriesData(frame, schema=schema, observed=frame.notna(), artificial=artificial)
    return data, blend(data, members_for(data), recipe, transforms=transforms)


def block(case_id, split, index, pos, length=1, group="site-a"):
    return BlockRecord(case_id, group, f"{group}:{index[pos].isoformat()}", split, "synthetic-summer",
                       index[pos], index[pos + length - 1], index[pos - 1], index[pos + length])


# Freeze the recipe from member configuration digests; they exclude data-instance fields.
reference = SeriesData(pd.DataFrame({"ta": [0.0, np.nan, 0.0], "nee": [1.0, np.nan, 1.0],
                                     "nee_mg": [1.0, np.nan, 1.0], "count": [1, 1, 1]},
                                    index=pd.date_range("2025-05-01", periods=3, freq=step, tz="UTC")),
                       schema=schema, observed=pd.DataFrame(
                           {"ta": [True, False, True], "nee": [True, False, True],
                            "nee_mg": [True, False, True], "count": [True] * 3},
                           index=pd.date_range("2025-05-01", periods=3, freq=step, tz="UTC")))
configurations = {name: configuration_identity(result) for name, result in members_for(reference).items()}
recipe = EnsembleRecipe((EnsembleMember("lin", "linear", configurations["lin"], .6),
                         EnsembleMember("per", "persistence", configurations["per"], .4)), "offline")

# Calibration truth: QC-passed zero-background cells hidden one at a time.
calibration_positions = [8 + 3 * j for j in range(10)]
ta = np.zeros(48)
ta[calibration_positions] = [.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5, -4.0, 4.5, -5.0]
data, result = fold(ta, calibration_positions)
index = data.raw.index
cases = tuple(BlockCase(block(f"cal-{pos}", "calibration", index, pos), data, recipe, result)
              for pos in calibration_positions)
stratum = SupportStratum("site-a", "synthetic-summer", recipe.identity, (("ta", "degC"), ("count", "count")),
                         step, 1, ((0, "ta"),))
calibration = calibrate_blocks(cases, strata=(stratum,), training=(SupportSpan("site-a", index[0], index[4]),),
                               purge=step, scales={"ta": 2.0, "count": 0.0},
                               scale_identity="synthetic-training-std-v1", alpha=.2, min_blocks=5,
                               unit_scale_fallback=("count",))
entry, = calibration.strata

# Untouched evaluation truth after calibration is frozen.
evaluation_positions = [40, 43, 46]
ta = np.zeros(48)
ta[evaluation_positions] = [1.0, -4.0, 6.0]
test_data, test_result = fold(ta, evaluation_positions)
report = evaluate_blocks(calibration, tuple(
    BlockCase(block(f"eval-{pos}", "evaluation", index, pos), test_data, recipe, test_result)
    for pos in evaluation_positions))

# Application: natural gaps at the same station on a later day; the two-slot gap has
# no calibrated stratum. Another station would be unknown support (no pooling).
application, point = fold(np.full(12, 12.0), missing=[5, 8, 9], start="2025-06-05")
app_index = application.raw.index
targets = (block("app-5", "application", app_index, 5),
           block("app-8", "application", app_index, 8, 2))
elsewhere = block_intervals(calibration, application, point,
                            (block("app-5-b", "application", app_index, 5, group="site-b"),))
intervals = block_intervals(calibration, application, point, targets)
draws = sample_block_draws(calibration, application, point, targets, n_draws=4, seed=20260926,
                           transforms=transforms)
flux = draw_flux_summary(draws, "nee")
path = export_uncertainty(Path(tempfile.mkdtemp(prefix="ecofill-uncertainty-")) / "export",
                          calibration=calibration, intervals=intervals, draws=draws, flux=(flux,))


def plain(value):
    return None if isinstance(value, float) and np.isnan(value) else value


export = {
    "identity": calibration.identity, "draw_ids": list(draws.draw_ids), "seed": draws.seed,
    "values": [[draw_id, stamp.isoformat(), name, plain(frame.at[stamp, name])]
               for draw_id in draws.draw_ids for frame in (draws.draw(draw_id),)
               for stamp in frame.index for name in frame.columns],
    "samples": [list(item) for item in draws.samples],
    "cells": [[stamp.isoformat(), name, draws.status.at[stamp, name],
               "" if pd.isna(draws.reasons.at[stamp, name]) else draws.reasons.at[stamp, name],
               str(bool(draws.mask("donor").at[stamp, name])).lower(),
               str(bool(draws.mask("structural").at[stamp, name])).lower(),
               str(bool(draws.supported.at[stamp, name])).lower()]
              for stamp in app_index for name in application.raw.columns],
    "flux": [[row["draw_id"], row["signed_umol_co2_m-2"]] for row in flux["draws"]],
    "recipes": [[{"identity": recipe.identity, **recipe.to_dict()}]] + 2 * [{"identity": recipe.identity,
                                                                          **recipe.to_dict()}],
    "targets": 2 * [[item.to_dict() for item in targets]],
    "calibration_blocks": [item.to_dict() for item in calibration.blocks],
    "strata": [json.loads(json.dumps(item.stratum.to_dict())) for item in calibration.strata],
}
with open(path / "draws.csv", newline="", encoding="utf-8") as handle:
    count = next(row["value"] for row in csv.DictReader(handle) if row["variable"] == "count")
print(json.dumps({
    "summary": {
        "rank": entry.rank, "threshold": entry.threshold, "n_blocks": entry.n_blocks,
        "block_coverage": [report["block_coverage"]["numerator"], report["block_coverage"]["denominator"]],
        "interval": [intervals.lower.at[app_index[5], "ta"], intervals.upper.at[app_index[5], "ta"]],
        "unknown": intervals.blocks[1]["status"],
        "other_station": elsewhere.blocks[0]["status"],
        "weights": [[item["name"], item["weight"]] for item in recipe.to_dict()["members"]],
        "sampled_ta": [draws.draw(d).at[app_index[5], "ta"] for d in draws.draw_ids],
        "sources": [item[2] for item in draws.samples],
        "nee_same_draw": all(draws.draw(d).at[app_index[5], "nee"]
                             == 2.0 * 2.0 ** ((draws.draw(d).at[app_index[5], "ta"] - 10.0) / 10.0)
                             for d in draws.draw_ids),
        "nee_unavailable": draws.reasons.at[app_index[8], "nee"],
        "flux_support": [flux["covered_seconds"], flux["total_seconds"], flux["complete_period"]],
        "count_csv": count, "fallback": list(calibration.unit_scale_fallback),
    },
    "export": export, "path": str(path),
}))
