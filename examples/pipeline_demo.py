"""Fresh-process installed-wheel driver-to-derived-to-response smoke."""

import json

import numpy as np
import pandas as pd

from ecofill import (DerivedStep, DriverPolicy, ExternalInputs, RegressionStep,
                     SeriesData, TimeSeriesSchema, VariableSpec,
                     execute_driver_plan, make_driver_plan)


step = pd.Timedelta("15min")
index = pd.date_range("2025-01-01", periods=96, freq=step, tz="UTC")
t = np.arange(len(index))
x = np.sin(2 * np.pi * t / 48)
q = np.cos(2 * np.pi * t / 96)
d = 2 * x + .5
z = d * d
y = 3 * z + .5 * q
frame = pd.DataFrame({"d": d, "z": z, "y": y}, index=index)
schema = TimeSeriesSchema(step, (VariableSpec("d", "1", "driver"),
    VariableSpec("z", "1", "derived", ("d",)), VariableSpec("y", "1", "response")))
hidden = pd.DataFrame(False, index=index, columns=frame.columns)
hidden.iloc[80:82, :] = True
data = SeriesData(frame, schema=schema, observed=frame.notna(), artificial=hidden)
external_frame = pd.DataFrame({"x": x, "q": q}, index=index)
external = ExternalInputs(external_frame, observed=external_frame.notna(),
    units={"x": "1", "q": "1"}, source="synthetic-demo",
    available_as_of={"x": index[0] - step, "q": index[0] - step})
policy = DriverPolicy((RegressionStep("d", ("x",)),
    DerivedStep("z", ("d",), ("1",), "1", "square-v1", lambda d: d * d, ("d",)),
    RegressionStep("y", ("z", "q"), allow_reconstructed=("z",))),
    0, 64, 64, 2 * step, "offline", "flag", "synthetic-demo")
result = execute_driver_plan(data, make_driver_plan(data, policy, external), external)
origin = next(item["donor_origin"]["z"] for item in result.provenance()["backend"]["cells"]
              if item["target"] == "y")
print(json.dumps({"completed": bool(np.allclose(result.completed["y"].iloc[80:82], y[80:82], atol=1e-10)),
                  "imputed_steps": int(result.mask("imputed")["y"].sum()),
                  "donor_origin": origin}))
