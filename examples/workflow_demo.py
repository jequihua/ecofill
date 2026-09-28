"""Synthetic explicit-data -> plan -> bounded linear reconstruction example."""

import json

import numpy as np
import pandas as pd

from ecofill import SeriesData, TimeSeriesSchema, VariableSpec, execute, make_plan


def main():
    frame = pd.DataFrame(
        {"temperature": [20.0, np.nan, np.nan, 23.0]},
        index=pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC"),
    )
    schema = TimeSeriesSchema(pd.Timedelta("15min"), (VariableSpec("temperature", "degC"),))
    data = SeriesData(frame, schema=schema, observed=frame.notna())
    plan = make_plan(data, method="linear", mode="offline", max_gap=pd.Timedelta("30min"))
    result = execute(data, plan)
    print(json.dumps({
        "completed": result.completed["temperature"].tolist(),
        "imputed_steps": int(result.mask("imputed")["temperature"].sum()),
        "mode": plan.mode,
    }))


if __name__ == "__main__":
    main()
