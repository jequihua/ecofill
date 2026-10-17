"""Run the installed synthetic Parquet fixture or independent Arrow reader."""
import importlib.util
from pathlib import Path
import runpy

spec=importlib.util.find_spec('ecofill')
if spec is None or not spec.submodule_search_locations:
    raise SystemExit('install ecofill[parquet] in this Python environment')
runpy.run_path(str(Path(list(spec.submodule_search_locations)[0])/'operation/parquet_example.py'),run_name='__main__')
