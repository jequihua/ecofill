"""The README quick start and the examples print exactly what the documentation says."""

import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

from _runner import EXAMPLES, ROOT, read_export, run, run_json

BIG = 9007199254740993
EXPECTED = {
    "profile_demo.py": {"variable": "temperature", "missing_steps": 2,
                        "duration_seconds": 1800.0, "bounded": True},
    "workflow_demo.py": {"completed": [20.0, 21.0, 22.0, 23.0], "imputed_steps": 2,
                         "mode": "offline"},
    "forecast_demo.py": {"completed": [10.0, 11.0, 11.0, 20.0, 20.0], "direction": "blend",
                         "imputed_steps": 2},
    "pipeline_demo.py": {"completed": True, "imputed_steps": 2, "donor_origin": "reconstructed"},
    "deep_demo.py": {"imputed": 1, "observed": 9.0, "method": "saits_pypots"},
    "ec_mds_demo.py": {method: {"fills": 2, "hidden_values": [-2.0, -2.0], "integral": {
        "signed_umol_co2_m-2": -1209600.0, "covered_seconds": 604800, "total_seconds": 604800,
        "complete_period": True}} for method in ("mds_hesseflux", "linear")},
}
SUMMARIES = {
    "time_indexed_demo.py": {
        "imputed": [25.130127, 24.345534, 23.32, 22.124095], "max_abs_error_below_1e-9": True,
        "reasons": ["edge_gap_unsupported", "gap_exceeds_limit"], "reference": [48, 32, 56, 107],
        "observed_unchanged": True, "count_exact": True, "requested_device": "cpu"},
    "uncertainty_demo.py": {
        "rank": 9, "threshold": 2.25, "n_blocks": 10, "block_coverage": [2, 3],
        "interval": [7.5, 16.5], "unknown": "unknown_support_stratum",
        "other_station": "unknown_support_stratum", "weights": [["lin", 0.6], ["per", 0.4]],
        "sampled_ta": [16.5, 16.5, 7.0, 16.5], "sources": ["cal-32", "cal-32", "cal-35", "cal-32"],
        "nee_same_draw": True, "nee_unavailable": "required_parent_unavailable",
        "flux_support": [18000, 21600, False], "count_csv": str(BIG), "fallback": ["count"]},
}
INTEGRATION = {
    "primary": {
        "benchmark": [["temperature", 2, 2, 0, 0, 0.0], ["pressure", 4, 4, 0, 0, 0.0]],
        "count_exact": True, "imputed": {"temperature": 7, "pressure": 7, "count": 0},
        "unresolved": {"temperature": {"gap_exceeds_limit": 9, "insufficient_left_context": 2,
                                       "insufficient_right_context": 2},
                       "pressure": {"structural_absence": 8}, "count": {}}},
    "timestamp": {"rejected": "irregular grid: explicitly regularize or split clock phases first",
                  "rows": 12, "value": 25.0, "inserted": True, "imputed": True, "state": "imputed"},
}


def readme_quick_start():
    """The first Python block of the README's Quick start and the text block that follows it."""
    section = (ROOT / "README.md").read_text(encoding="utf-8").split("## Quick start", 1)[1]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    output = re.search(r"```text\n(.*?)```", section, re.S).group(1)
    return code, output


class ExampleTests(unittest.TestCase):
    def test_readme_quick_start_prints_the_documented_output(self):
        code, output = readme_quick_start()
        with tempfile.TemporaryDirectory(prefix="ecofill-quickstart-") as scratch:
            self.assertEqual(run("-c", code, cwd=scratch).stdout, output)
            consumed = read_export("result", Path(scratch) / "quickstart-export")
            self.assertEqual(consumed.returncode, 0, consumed.stderr)
            document = json.loads(consumed.stdout)
            self.assertEqual(document["raw"]["temperature"],
                             [20.0, None, None, 23.0, None, None, None, 20.0])
            self.assertEqual(document["completed"]["temperature"],
                             [20.0, 21.0, 22.0, 23.0, None, None, None, 20.0])
            self.assertEqual(document["masks"]["observed"]["temperature"],
                             [True, False, False, True, False, False, False, True])
            self.assertEqual(document["masks"]["imputed"]["temperature"],
                             [False, True, True, False, False, False, False, False])
            self.assertEqual(document["cells"]["temperature"]["reason"][4:7],
                             ["gap_exceeds_limit"] * 3)

    def test_examples_print_their_documented_json(self):
        for name, expected in EXPECTED.items():
            with self.subTest(example=name):
                self.assertEqual(run_json(EXAMPLES / name), expected)

    def test_time_indexed_and_uncertainty_examples_export_their_documented_summary(self):
        for name, expected in SUMMARIES.items():
            with self.subTest(example=name):
                produced = run_json(EXAMPLES / name)
                export = Path(produced["path"])
                try:
                    self.assertEqual(produced["summary"], expected)
                    self.assertTrue(export.is_dir())
                finally:
                    if export.parent.name.startswith("ecofill-"):
                        shutil.rmtree(export.parent)

    def test_integration_example_exports_are_read_back_in_a_separate_process(self):
        with tempfile.TemporaryDirectory(prefix="ecofill-test-") as scratch:
            output = Path(scratch) / "demo"
            self.assertEqual(run_json(EXAMPLES / "integration_demo.py", "--output", output),
                             INTEGRATION)
            refused = run(EXAMPLES / "integration_demo.py", "--output", output, check=False)
            self.assertNotEqual(refused.returncode, 0)
            completed = read_export("result", output / "primary" / "workflow" / "result")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            document = json.loads(completed.stdout)
            self.assertEqual(document["variables"], ["temperature", "pressure", "count"])
            self.assertEqual(document["completed"]["count"], [BIG] * 192)
            self.assertEqual(document["completed"]["temperature"][24:28],
                             [16.0, 16.25, 16.5, 16.75])
            self.assertEqual(document["cells"]["pressure"]["reason"][160:168],
                             ["structural_absence"] * 8)
            self.assertEqual(sum(document["masks"]["imputed"]["temperature"]), 7)


if __name__ == "__main__":
    unittest.main()
