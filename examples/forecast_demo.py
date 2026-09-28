"""Installed-wheel synthetic forecasting smoke; prints exact JSON."""

import json

import numpy as np
import pandas as pd

from ecofill import (ForecastPolicy, LastValueForecaster, SeriesData, TimeSeriesSchema,
                     VariableSpec, execute_forecast, make_forecast_plan)


index = pd.date_range("2025-01-01", periods=5, freq="1h", tz="UTC")
frame = pd.DataFrame({"temperature": [10.0, 11.0, np.nan, np.nan, 20.0]}, index=index)
data = SeriesData(frame, schema=TimeSeriesSchema(pd.Timedelta("1h"),
                  (VariableSpec("temperature", "degC"),)), observed=frame.notna())
policy = ForecastPolicy(("temperature",), "offline", "blend", pd.Timedelta("2h"),
                        context_length=2, min_support=1, max_horizon=2)
plan = make_forecast_plan(data, policy)
result = execute_forecast(data, plan, LastValueForecaster())
print(json.dumps({"completed": result.completed["temperature"].tolist(),
                  "direction": result.provenance()["plan"]["direction"],
                  "imputed_steps": int(result.mask("imputed")["temperature"].sum())}))
