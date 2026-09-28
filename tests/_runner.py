"""Run code in a fresh interpreter that imports the same ecofill as the test process.

The child gets the parent's exact `sys.path` and must import ecofill from the same
location, so an installed wheel stays the package under test. `base_only=True`
also blocks optional model packages, reports their distribution metadata as
absent and denies network connections, the way a core-only install behaves
whatever else happens to be installed.
"""

import json
from pathlib import Path
import subprocess
import sys

import ecofill

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
OPTIONAL_MODULES = ("torch", "pypots", "timesfm", "tabpfn", "sklearn", "scipy", "statsmodels",
                    "h5py", "huggingface_hub", "safetensors")
OPTIONAL_DISTRIBUTIONS = ("torch", "timesfm", "tabpfn", "scikit-learn", "scipy", "statsmodels",
                          "h5py", "huggingface-hub", "huggingface_hub", "safetensors",
                          "threadpoolctl")
PRELUDE = r"""
import builtins, importlib.metadata, json, runpy, socket, sys
from pathlib import Path
setup, target, *arguments = sys.argv[1:]
setup = json.loads(setup)
sys.path[:] = setup["path"]
if setup["base_only"]:
    real_import, real_version = builtins.__import__, importlib.metadata.version

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name.split(".")[0] in setup["modules"]:
            raise ImportError(f"optional module {name} is blocked for this test")
        return real_import(name, globals, locals, fromlist, level)

    def absent(name):
        if name in setup["distributions"]:
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    def denied(*args, **kwargs):
        raise OSError("network is denied in this test")

    builtins.__import__, importlib.metadata.version = blocked, absent
    socket.socket.connect = socket.create_connection = socket.getaddrinfo = denied
import ecofill
if Path(ecofill.__file__).resolve() != Path(setup["origin"]):
    raise SystemExit(f"child imported ecofill from {ecofill.__file__}, not {setup['origin']}")
sys.argv = [target, *arguments]
if target == "-c":
    exec(compile(arguments[0], "<test>", "exec"), {"__name__": "__main__"})
else:
    runpy.run_path(target, run_name="__main__")
"""


def run(target, *arguments, base_only=True, cwd=None, check=True):
    """Run a script path, or `"-c", code`, and return the completed process."""
    setup = {"path": sys.path, "base_only": base_only, "modules": OPTIONAL_MODULES,
             "distributions": OPTIONAL_DISTRIBUTIONS,
             "origin": str(Path(ecofill.__file__).resolve())}
    command = [sys.executable, "-I", "-B", "-c", PRELUDE, json.dumps(setup), str(target),
               *map(str, arguments)]
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if check and completed.returncode:
        raise AssertionError(f"{target} failed ({completed.returncode}):\n"
                             f"{completed.stderr[-3000:]}")
    return completed


def run_json(target, *arguments, **options):
    return json.loads(run(target, *arguments, **options).stdout)


def read_export(kind, directory):
    """The standard-library reader in its own isolated process, without site-packages."""
    return subprocess.run([sys.executable, "-I", "-S", "-B", str(EXAMPLES / "read_export.py"), kind,
                           str(directory)], capture_output=True, text=True, check=False)
