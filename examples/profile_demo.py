"""Offline synthetic user path; emits a compact portable JSON description."""

import json

import numpy as np
import pandas as pd

from ecofill import TimeSeriesSchema, VariableSpec, profile_gaps


def main():
    frame = pd.DataFrame(
        {"temperature": [20.0, np.nan, np.nan, 23.0]},
        index=pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC"),
    )
    schema = TimeSeriesSchema(pd.Timedelta("15min"), (VariableSpec("temperature", "degC"),))
    gap, = profile_gaps(frame, schema=schema)
    print(json.dumps({"variable": gap.variable, "missing_steps": gap.n_steps,
                      "duration_seconds": gap.duration.total_seconds(), "bounded": gap.bounded}))


if __name__ == "__main__":
    main()
