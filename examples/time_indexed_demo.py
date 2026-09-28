"""Installed-wheel, model-free time-indexed plan -> execute -> export smoke; prints exact JSON."""

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from ecofill import (SeriesData, TimeFeatures, TimeIndexedPolicy, TimeSeriesSchema, VariableSpec,
                     execute_time_indexed, make_time_indexed_plan)


class HarmonicLeastSquares:
    """Example caller backend: ordinary least squares on [1, features]; declares its identity."""

    identity = {"backend": "example-harmonic-lstsq", "version": "1", "device": "cpu"}

    def predict(self, request):
        design = np.column_stack([np.ones(len(request.reference_labels)), request.reference_features])
        coefficients, *_ = np.linalg.lstsq(design, request.reference_labels, rcond=None)
        return np.column_stack([np.ones(len(request.query_features)), request.query_features]) @ coefficients


step = pd.Timedelta("1h")
index = pd.date_range("2025-01-06", periods=168, freq=step, tz="UTC")
hours = np.arange(168)
truth = 20.0 + 5.0 * np.sin(2 * np.pi * hours / 24) + 0.01 * hours
frame = pd.DataFrame({"ta": truth, "count": np.full(168, 2 ** 53 + 1, dtype=np.int64)}, index=index)
frame.iloc[[0, 1], 0] = np.nan                      # leading edge gap
frame.iloc[130:140, 0] = np.nan                     # 10-hour natural gap, longer than max_gap
hidden = pd.DataFrame(False, index=index, columns=frame.columns)
hidden.iloc[80:84, 0] = True                        # artificial validation gap with retained truth
schema = TimeSeriesSchema(step, (VariableSpec("ta", "degC", "driver"),
                                 VariableSpec("count", "count", "diagnostic")))
data = SeriesData(frame, schema=schema, observed=frame.notna(), artificial=hidden)
policy = TimeIndexedPolicy(("ta",), ((56, 80), (84, 108)), TimeFeatures(index[0], pd.Timedelta("1D")),
                           max_reference=32, min_support=8, max_gap=pd.Timedelta("6h"), seed=0)
plan = make_time_indexed_plan(data, policy)
result = execute_time_indexed(data, plan, HarmonicLeastSquares())
path = result.export(Path(tempfile.mkdtemp(prefix="ecofill-time-indexed-")) / "export")

completed, reasons = result.completed, result.unresolved_reasons
masks = {name: result.mask(name)["ta"] for name in ("observed", "artificial", "imputed")}
rows = plan.targets[0].rows
provenance = json.loads(json.dumps(result.provenance()))
print(json.dumps({
    "summary": {
        "imputed": [round(value, 6) for value in completed["ta"].iloc[80:84]],
        "max_abs_error_below_1e-9": bool(np.abs(completed["ta"].iloc[80:84] - truth[80:84]).max() < 1e-9),
        "reasons": [reasons.at[index[0], "ta"], reasons.at[index[135], "ta"]],
        "reference": [plan.targets[0].n_eligible, len(rows), rows[0], rows[-1]],
        "observed_unchanged": completed.loc[data.mask("donor")["ta"], "ta"].tolist()
        == data.available.loc[data.mask("donor")["ta"], "ta"].tolist(),
        "count_exact": completed["count"].tolist() == [2 ** 53 + 1] * 168,
        "requested_device": provenance["backend"]["execution"]["requested_device"],
    },
    "export": {
        "ta": [None if pd.isna(value) else value for value in completed["ta"]],
        "count": completed["count"].tolist(),
        "masks": {name: [str(bool(value)) for value in mask] for name, mask in masks.items()},
        "reference": provenance["plan"]["reference"],
        "policy": provenance["plan"]["policy"],
        "gaps": provenance["plan"]["gaps"],
        "backend": {key: value for key, value in provenance["backend"].items() if key != "execution"},
    },
    "path": str(path),
}))
