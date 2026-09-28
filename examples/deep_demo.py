"""Installed-wheel deep adapter contract without optional Torch."""

import json

import numpy as np
import pandas as pd

from ecofill import (DeepPolicy, SeriesData, TimeSeriesSchema, VariableSpec,
                     execute_deep, fit_deep, make_deep_plan)


class SyntheticBackend:
    steps = 1

    def fit(self, inputs, policy):
        assert inputs["indicating_mask"].sum() == 1

    def state_hash(self):
        return "synthetic-recording-state"

    def infer(self, inputs):
        return np.zeros_like(inputs["X"])


index = pd.date_range("2025-01-01", periods=16, freq="15min", tz="UTC")
frame = pd.DataFrame({"x": np.arange(16, dtype=float)}, index=index)
observed = frame.notna()
hidden = pd.DataFrame(False, index=index, columns=frame.columns)
hidden.iloc[10, 0] = True
data = SeriesData(frame, schema=TimeSeriesSchema(pd.Timedelta("15min"),
                                                 (VariableSpec("x", "1"),)),
                  observed=observed, artificial=hidden)
policy = DeepPolicy("saits_pypots", ("x",), ("x",), (0, 8), (8, 16), 4,
                    pd.Timedelta("1h"), 2, 5, epochs=1)
training_hide = np.zeros((2, 4, 1), dtype=bool)
training_hide[0, 1, 0] = True
fit = fit_deep(data, policy, training_hide, backend=SyntheticBackend())
result = execute_deep(data, fit, make_deep_plan(data, fit))
print(json.dumps({"imputed": int(result.mask("imputed").to_numpy().sum()),
                  "observed": float(result.completed.iloc[9, 0]),
                  "method": result.methods.iloc[10, 0]}))
