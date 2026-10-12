"""Synthetic, QC-explicit EC comparison; positive NEE denotes emission.

By default this is one week on a 30-minute UTC grid. `--cadence 15min` (any
whole-minute cadence dividing 12 hours) builds the same week on that grid and
declares it to MDS through `compare_nee(..., cadence=...)`; nothing is resampled.
`--output DIR` also exports the MDS result for an independent reader.
"""

import argparse
import json

import numpy as np
import pandas as pd

from ecofill import SeriesData, TimeSeriesSchema, VariableSpec
from ecofill.ec_recipe import compare_nee, signed_supported_integral


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--cadence", default="30min")
parser.add_argument("--output")
arguments = parser.parse_args()
cadence = pd.Timedelta(arguments.cadence)
index = pd.date_range("2025-01-01", periods=int(pd.Timedelta("7d") / cadence), freq=cadence, tz="UTC")
hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
sw_in = np.maximum(0, 600 * np.sin(2 * np.pi * (hour - 6) / 24))
ta = 15 + 8 * np.sin(2 * np.pi * (hour - 8) / 24)
vpd_hpa = 5 + 0.02 * sw_in  # Converted from kPa with factor 10 if input was kPa.
frame = pd.DataFrame({"SW": sw_in, "TEMP": ta, "DEFICIT": vpd_hpa,
                      "NEE": np.full(len(index), -2.0)}, index=index)
schema = TimeSeriesSchema(cadence, (
    VariableSpec("SW", "W m-2", "driver"), VariableSpec("TEMP", "degC", "driver"),
    VariableSpec("DEFICIT", "hPa", "driver"),
    VariableSpec("NEE", "umol CO2 m-2 s-1", "response")))
# Hide the two slots starting 2025-01-04 03:00 UTC, whatever the cadence.
start = int(pd.Timedelta("75h") / cadence)
gap = slice(start, start + 2)
hidden = pd.DataFrame(False, index=index, columns=frame.columns)
hidden.loc[index[gap], "NEE"] = True
data = SeriesData(frame, schema=schema, observed=frame.notna(), artificial=hidden)
results = compare_nee(data, target="NEE", sw_in="SW", ta="TEMP", vpd="DEFICIT",
                      max_gap=pd.Timedelta("7d"), cadence=cadence)
summary = {name: {"fills": int(result.mask("imputed")["NEE"].sum()),
                  "hidden_values": result.completed["NEE"].iloc[gap].tolist(),
                  "integral": signed_supported_integral(result, "NEE")}
           for name, result in results.items()}
if arguments.output:
    summary["path"] = str(results["mds_hesseflux"].export(arguments.output))
print(json.dumps(summary, allow_nan=False))
