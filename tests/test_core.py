"""Core contracts: explicit masks, whole-gap plans, abstention, preservation and export."""

import dataclasses
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from ecofill import (ForecastPolicy, SeriesData, TimeSeriesSchema, VariableSpec, execute,
                     execute_forecast, make_forecast_plan, make_plan)

from _runner import OPTIONAL_MODULES, read_export, run

STEP = pd.Timedelta("15min")


def series(values, *, qc_rejected=None):
    index = pd.date_range("2025-01-01", periods=len(values), freq=STEP, tz="UTC")
    frame = pd.DataFrame({"temperature": values}, index=index)
    schema = TimeSeriesSchema(STEP, (VariableSpec("temperature", "degC"),))
    rejected = pd.DataFrame({"temperature": qc_rejected or [False] * len(values)}, index=index)
    return SeriesData(frame, schema=schema, observed=frame.notna() & ~rejected,
                      qc_rejected=rejected)


class CoreTests(unittest.TestCase):
    def test_bounded_gap_is_filled_and_longer_gap_is_left_whole(self):
        data = series([20.0, np.nan, np.nan, 23.0, np.nan, np.nan, np.nan, 20.0])
        plan = make_plan(data, method="linear", mode="offline", max_gap=pd.Timedelta("30min"))
        planned = [(g.gap.start_pos, g.gap.stop_pos, g.eligible, g.reason) for g in plan.gaps]
        self.assertEqual(planned, [(1, 3, True, None), (4, 7, False, "gap_exceeds_limit")])
        result = execute(data, plan)
        completed = result.completed["temperature"]
        self.assertEqual(completed.iloc[:4].tolist(), [20.0, 21.0, 22.0, 23.0])
        self.assertTrue(completed.iloc[4:7].isna().all())
        self.assertEqual(result.unresolved_reasons["temperature"].iloc[4:7].tolist(),
                         ["gap_exceeds_limit"] * 3)
        self.assertEqual(int(result.mask("imputed")["temperature"].sum()), 2)

    def test_qc_rejected_value_is_kept_in_raw_but_never_used_as_a_donor(self):
        data = series([1.0, 999.0, np.nan, 4.0, 5.0],
                      qc_rejected=[False, True, False, False, False])
        result = execute(data, make_plan(data, method="linear", mode="offline",
                                         max_gap=pd.Timedelta("1h")))
        self.assertEqual(result.completed["temperature"].tolist(), [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(result.raw["temperature"].iloc[1], 999.0)
        self.assertEqual(result.states["temperature"].tolist(),
                         ["observed", "imputed", "imputed", "observed", "observed"])
        self.assertTrue(result.mask("qc_rejected")["temperature"].iloc[1])

    def test_causal_mode_refuses_methods_that_need_later_observations(self):
        data = series([1.0, np.nan, np.nan, 4.0])
        for method in ("next", "linear"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "causal"):
                make_plan(data, method=method, mode="causal", max_gap=pd.Timedelta("1h"))
        result = execute(data, make_plan(data, method="persistence", mode="causal",
                                         max_gap=pd.Timedelta("1h")))
        self.assertEqual(result.completed["temperature"].tolist(), [1.0, 1.0, 1.0, 4.0])

    def test_edited_plan_is_rejected_before_any_value_is_written(self):
        data = series([20.0, np.nan, np.nan, np.nan, np.nan, 25.0])
        plan = make_plan(data, method="linear", mode="offline", max_gap=pd.Timedelta("30min"))
        self.assertFalse(plan.gaps[0].eligible)
        forged = dataclasses.replace(
            plan, gaps=(dataclasses.replace(plan.gaps[0], eligible=True, reason=None),))
        with self.assertRaises(ValueError):
            execute(data, forged)

    def test_irregular_timestamps_are_rejected_not_repaired(self):
        index = pd.date_range("2025-01-01", periods=5, freq=STEP, tz="UTC").delete(2)
        frame = pd.DataFrame({"temperature": [1.0, 2.0, 4.0, 5.0]}, index=index)
        schema = TimeSeriesSchema(STEP, (VariableSpec("temperature", "degC"),))
        with self.assertRaisesRegex(ValueError, "irregular grid"):
            SeriesData(frame, schema=schema, observed=frame.notna())

    def test_forward_backward_blend_uses_position_weights(self):
        class Directional:
            def forecast(self, request):
                value = 0.0 if request.direction == "forward" else 10.0
                return np.full((request.horizon, len(request.channels)), value)

        data = series([0.0, 0.0, np.nan, np.nan, np.nan, 10.0, 10.0])
        policy = ForecastPolicy(channels=("temperature",), mode="offline", direction="blend",
                                max_gap=pd.Timedelta("1h"), context_length=2, min_support=2,
                                max_horizon=3)
        result = execute_forecast(data, make_forecast_plan(data, policy), Directional())
        self.assertEqual(result.completed["temperature"].iloc[2:5].tolist(), [0.0, 5.0, 10.0])

    def test_export_is_read_back_by_the_standard_library_reader(self):
        data = series([20.0, np.nan, np.nan, 23.0, np.nan, np.nan, np.nan, 20.0])
        result = execute(data, make_plan(data, method="linear", mode="offline",
                                         max_gap=pd.Timedelta("30min")))
        with tempfile.TemporaryDirectory(prefix="ecofill-test-") as scratch:
            target = Path(scratch) / "export"
            result.export(target)
            with self.assertRaises(FileExistsError):
                result.export(target)
            completed = read_export("result", target)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            document = json.loads(completed.stdout)
            self.assertEqual(document["completed"]["temperature"],
                             [20.0, 21.0, 22.0, 23.0, None, None, None, 20.0])
            self.assertEqual(document["cells"]["temperature"]["reason"][4:7],
                             ["gap_exceeds_limit"] * 3)
            # Editing an observation in the completed values is detected.
            path = target / "completed.csv"
            path.write_text(path.read_text(encoding="utf-8").replace("20.0", "19.0", 1),
                            encoding="utf-8")
            rejected = read_export("result", target)
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("changes a donor observation", rejected.stderr)

    def test_importing_ecofill_loads_no_optional_model_package(self):
        code = ("import sys, ecofill\n"
                f"print(sorted(m for m in sys.modules if m.split('.')[0] in {OPTIONAL_MODULES!r}))")
        self.assertEqual(run("-c", code, base_only=False).stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
