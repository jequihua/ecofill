"""Synthetic, QC-explicit EC comparison; positive NEE denotes emission."""

import json

import numpy as np
import pandas as pd

from ecofill import SeriesData, TimeSeriesSchema, VariableSpec
from ecofill.ec_recipe import compare_nee, signed_supported_integral


index = pd.date_range("2025-01-01", periods=336, freq="30min", tz="UTC")
hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
sw_in = np.maximum(0, 600 * np.sin(2 * np.pi * (hour - 6) / 24))
ta = 15 + 8 * np.sin(2 * np.pi * (hour - 8) / 24)
vpd_hpa = 5 + 0.02 * sw_in  # Converted from kPa with factor 10 if input was kPa.
frame = pd.DataFrame({"SW": sw_in, "TEMP": ta, "DEFICIT": vpd_hpa,
                      "NEE": np.full(len(index), -2.0)}, index=index)
schema = TimeSeriesSchema(pd.Timedelta("30min"), (
    VariableSpec("SW", "W m-2", "driver"), VariableSpec("TEMP", "degC", "driver"),
    VariableSpec("DEFICIT", "hPa", "driver"),
    VariableSpec("NEE", "umol CO2 m-2 s-1", "response")))
hidden = pd.DataFrame(False, index=index, columns=frame.columns)
hidden.loc[index[150:152], "NEE"] = True
data = SeriesData(frame, schema=schema, observed=frame.notna(), artificial=hidden)
results = compare_nee(data, target="NEE", sw_in="SW", ta="TEMP", vpd="DEFICIT",
                      max_gap=pd.Timedelta("7d"))
print(json.dumps({name: {"fills": int(result.mask("imputed")["NEE"].sum()),
                         "hidden_values": result.completed["NEE"].iloc[150:152].tolist(),
                         "integral": signed_supported_integral(result, "NEE")}
                  for name, result in results.items()}, allow_nan=False))
