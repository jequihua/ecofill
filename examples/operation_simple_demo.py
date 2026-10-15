"""Launch the installed synthetic simple-configuration example (ecofill/operation/simple_example.py).

The packaged file is executed directly, without importing ecofill, NumPy or pandas, so its
readback stays a standard-library check. Commands: prepare, readback, replay-config.
"""
import importlib.util
from pathlib import Path
import runpy

spec = importlib.util.find_spec('ecofill')
if spec is None or not spec.submodule_search_locations:
    raise SystemExit('ecofill is not installed in this Python environment')
runpy.run_path(str(Path(list(spec.submodule_search_locations)[0]) / 'operation' / 'simple_example.py'),
               run_name='__main__')
