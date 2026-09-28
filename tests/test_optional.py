"""Optional model backends fail clearly on a core-only install and never fetch weights."""

import json
import unittest

from _runner import run

MISSING_EXTRAS = r"""
import json, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from ecofill import (DeepPolicy, SeriesData, TabPFNBackend, TimeSeriesSchema, TimesFM25Backend,
                     VariableSpec, fit_deep)
from ecofill.tabpfn import validate_checkpoint

outcome = {}
def record(label, call):
    try:
        call()
        outcome[label] = None
    except Exception as exc:
        outcome[label] = [type(exc).__name__, str(exc)]

record("tabpfn", lambda: TabPFNBackend("tabpfn-v3.5-20260909.safetensors"))
with tempfile.TemporaryDirectory() as scratch:
    fake = Path(scratch) / "fake.safetensors"
    fake.write_bytes(b"not the pinned checkpoint")
    record("tabpfn_wrong_bytes", lambda: validate_checkpoint(fake))
    record("tabpfn_missing", lambda: validate_checkpoint(Path(scratch) / "absent.safetensors"))
    record("timesfm_missing", lambda: TimesFM25Backend(Path(scratch) / "absent", context_length=8,
                                                       max_horizon=2).load())
index = pd.date_range("2025-01-01", periods=16, freq="15min", tz="UTC")
frame = pd.DataFrame({"x": np.arange(16, dtype=float)}, index=index)
hidden = pd.DataFrame(False, index=index, columns=frame.columns)
hidden.iloc[10, 0] = True
data = SeriesData(frame, schema=TimeSeriesSchema(pd.Timedelta("15min"), (VariableSpec("x", "1"),)),
                  observed=frame.notna(), artificial=hidden)
policy = DeepPolicy("saits_pypots", ("x",), ("x",), (0, 8), (8, 16), 4, pd.Timedelta("1h"), 2, 5,
                    epochs=1)
training_hide = np.zeros((2, 4, 1), dtype=bool)
training_hide[0, 1, 0] = True
record("deep", lambda: fit_deep(data, policy, training_hide))
print(json.dumps(outcome))
"""


class OptionalBackendTests(unittest.TestCase):
    def test_missing_extras_and_checkpoints_raise_actionable_errors(self):
        outcome = json.loads(run("-c", MISSING_EXTRAS).stdout)
        self.assertEqual(outcome["tabpfn"][0], "ImportError")
        self.assertIn("ecofill[tabpfn]", outcome["tabpfn"][1])
        self.assertEqual(outcome["tabpfn_wrong_bytes"][0], "ValueError")
        self.assertEqual(outcome["tabpfn_missing"][0], "FileNotFoundError")
        self.assertIn("Prior-Labs/tabpfn_3_5", outcome["tabpfn_missing"][1])
        self.assertEqual(outcome["timesfm_missing"][0], "FileNotFoundError")
        self.assertEqual(outcome["deep"][0], "ImportError")
        self.assertIn("torch", outcome["deep"][1])


if __name__ == "__main__":
    unittest.main()
