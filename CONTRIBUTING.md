# Contributing

Thank you for helping. ecofill favours small, explicit behaviour over
convenience, so please read [methodology.md](methodology.md) before changing a
method.

## Development setup

Use Python 3.12. Create a virtual environment in the repository root and call
its interpreter by path in every later command. Then a new shell cannot
silently use another Python or another ecofill installation.

For regular colleague installs, pinned GitHub installs and installed provenance,
see [installation](docs/installation.md). The editable setup below is for contributors.

Windows (PowerShell), from the repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Linux/macOS, from the repository root:

```console
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

This is an editable install: `import ecofill` loads `src/ecofill` from your
checkout, so edits take effect without reinstalling. If you prefer to activate
the environment (`.venv\Scripts\Activate.ps1` on Windows,
`source .venv/bin/activate` on Linux/macOS), plain `python` then means the
venv interpreter, but only in that shell. Check with
`python -c "import sys; print(sys.executable)"`.

Install an optional extra (`research`, `deep`, `foundation`, `tabpfn`) only if
you work on that method. No test needs one.

## Tests and examples (editable checkout)

Windows:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe examples\workflow_demo.py
```

Linux/macOS:

```console
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python examples/workflow_demo.py
```

These test the source in your checkout through the editable install. The tests
use small synthetic series only. They need no network, no model weights and no
optional extra, and they must keep passing when optional model packages are
absent. They run the examples and compare their exact output, so update the
test when you deliberately change an example's output.

## Building a wheel and source distribution

```powershell
.venv\Scripts\python.exe -m build --sdist --wheel --no-isolation   # Windows
```

```console
.venv/bin/python -m build --sdist --wheel --no-isolation           # Linux/macOS
```

This produces the wheel and `dist/ecofill-0.0.1.tar.gz` using the build tools
from the `dev` extra. It does not install or test them; test both from fresh
environments outside the checkout. The source archive includes package resources
and licenses; it is not a substitute for the complete public checkout's examples.

## Testing the installed wheel (optional)

To test the built package rather than the checkout, install the wheel into a
second, separate environment. Then run from a directory outside the checkout,
so that the checkout's `src/` cannot be imported by accident. Windows example,
with `<checkout>` your repository root:

```powershell
py -3.12 -m venv <scratch>\wheel-venv
<scratch>\wheel-venv\Scripts\python.exe -m pip install <checkout>\dist\ecofill-0.0.1-py3-none-any.whl
Set-Location <scratch>
<scratch>\wheel-venv\Scripts\python.exe -c "import ecofill; print(ecofill.__file__)"
<scratch>\wheel-venv\Scripts\python.exe -m unittest discover -s <checkout>\tests -t <checkout>\tests -v
```

The printed path must be inside `<scratch>\wheel-venv\Lib\site-packages`, not
your checkout. On Linux/macOS use `python3.12`, `wheel-venv/bin/python` and
forward slashes. Installing the wheel downloads NumPy and pandas if the new
environment lacks them.

## Principles for changes

- Keep caller decisions explicit: masks, units, cadence, method, gap limits,
  devices and checkpoint paths. Refuse ambiguous input instead of guessing.
- Never overwrite an observed value, partly fill a gap to satisfy a limit, or
  substitute another method silently. An unsupported case abstains with a
  reason; a real error raises.
- Optional dependencies are imported lazily and fail with an actionable
  message naming the extra. Importing `ecofill` must not load them, download
  anything or write files.
- Add a synthetic test for each behaviour you change. A test shows a software
  contract; do not describe it as evidence of accuracy.
- Keep `numpy` and `pandas` as the only core dependencies and keep the pinned
  versions of the extras unless a change needs otherwise.

## Vendored code, weights and data

- `src/ecofill/_pypots/` and `src/ecofill/_hesseflux_gapfill.py` are unchanged
  upstream copies. Do not edit them, and keep their license files (see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
- Never commit model weights, checkpoints, private measurements or local paths.
  [.gitignore](.gitignore) excludes common checkpoint formats.

## How this repository is updated

This repository is a curated projection of a separate development repository.
The file `.front_repo_sync.json` records the source commit and the SHA-256 of
every projected file. Pull requests are welcome; accepted changes are applied
in the development repository and synchronized here, so a direct edit to a
projected file may be replaced by the next synchronization.

## License

By contributing you agree that your contribution is released under the
project's [MIT License](LICENSE).
