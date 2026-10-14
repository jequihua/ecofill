# Install ecofill for local work

Use CPython 3.12 and a dedicated virtual environment. ecofill 0.0.1 currently
supports `>=3.12,<3.13`. It is not published on PyPI: `pip install ecofill`
is not the current installation path. Package downloads need network access.
Git is required for `git+https` installation and cloning, but not for a supplied
wheel or local source directory.

## Choose your environment

From your own working directory, Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable); print(sys.version)"
```

POSIX (Linux/macOS):

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -c "import sys; print(sys.executable); print(sys.version)"
```

All later `python` commands refer to this activated environment. Without activation,
use `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on POSIX; this also
avoids a PowerShell activation-policy problem without changing system policy.
Use `python -m pip`, so installation and execution use the same interpreter.

## Install from GitHub or local sources

For a quick install of the latest main revision:

```sh
python -m pip install "ecofill @ git+https://github.com/jequihua/ecofill.git@main"
python -m pip check
```

Main moves over time. For a reproducible install, select one full public commit
once, then reuse it for the package, extras, checkout and example downloads below.
Start in a fresh environment; when replacing an existing installation at version
0.0.1, follow [updates](#updates-provenance-and-troubleshooting) first.

PowerShell, select and validate a SHA:

```powershell
$ecofillRemote = git ls-remote https://github.com/jequihua/ecofill.git refs/heads/main
if ($LASTEXITCODE -ne 0) { throw "Cannot resolve ecofill main" }
$ecofillSha = ($ecofillRemote -split '\s+')[0]
if ($ecofillSha -notmatch '^[0-9a-f]{40}$') { throw "Expected one full public SHA" }
python -m pip install "ecofill @ git+https://github.com/jequihua/ecofill.git@$ecofillSha"
python -m pip check
```

POSIX, select and validate a SHA:

```sh
ecofill_remote="$(git ls-remote https://github.com/jequihua/ecofill.git refs/heads/main)" || exit 1
ecofill_sha="${ecofill_remote%%[[:space:]]*}"
case "$ecofill_sha" in ''|*[!0-9a-f]*) echo "Expected one full public SHA" >&2; exit 1;; esac
[ "${#ecofill_sha}" -eq 40 ] || exit 1
python -m pip install "ecofill @ git+https://github.com/jequihua/ecofill.git@${ecofill_sha}"
python -m pip check
```

A colleague's tested full SHA can be assigned to `$ecofillSha` (PowerShell) or
`ecofill_sha` (POSIX) instead of resolving main; keep the same 40-character hex
validation. Keep that selected value throughout this workflow. Do not resolve
main again for each asset. Record the SHA and verify it in installed provenance
as shown below. The latest-main command itself is not an immutable requirement.

With a supplied public source directory, ordinary colleagues use a regular install:

```sh
python -m pip install ./ecofill
python -m pip check
```

Or clone the public repository and check out the selected SHA. PowerShell:

```powershell
git clone https://github.com/jequihua/ecofill.git ecofill-source
git -C ecofill-source checkout "$ecofillSha"
python -m pip install ./ecofill-source
```

POSIX:

```sh
git clone https://github.com/jequihua/ecofill.git ecofill-source
git -C ecofill-source checkout "$ecofill_sha"
python -m pip install ./ecofill-source
```

These quoted commands work in PowerShell and POSIX shells. Local builds use pip's
isolated build environment. A regular install copies the package into the venv;
you can then work from an unrelated directory. Contributors may instead use
`python -m pip install -e "./ecofill-source[dev]"`: this editable installation
imports from that checkout and needs reinstalling when package metadata changes.
See [contributing](../CONTRIBUTING.md).

## Choose extras deliberately

| Install | Libraries / use |
|---|---|
| core | NumPy and pandas; bounded interpolation and JSON operation configs |
| `operation` | PyYAML; safe YAML operation configs |
| `research` | scipy, scikit-learn, statsmodels, h5py; RF/boosting driver regression |
| `cloud` | BigQuery and Storage clients; ADC transport |
| `foundation` | TimesFM, torch, huggingface-hub, safetensors; separately supplied forecasting weights |
| `tabpfn` | TabPFN and torch; separately supplied pinned TabPFN checkpoint |
| `deep` | torch; experimental BRITS/SAITS training |
| `dev` | PyYAML, build, setuptools, wheel; contributor/build tools |

For the local synthetic LIN/RF workflow below, install `operation,research`.
From your selected local checkout:

```sh
python -m pip install "./ecofill-source[operation,research]"
```

Directly from the same selected public SHA, PowerShell:

```powershell
python -m pip install "ecofill[operation,research] @ git+https://github.com/jequihua/ecofill.git@$ecofillSha"
```

POSIX:

```sh
python -m pip install "ecofill[operation,research] @ git+https://github.com/jequihua/ecofill.git@${ecofill_sha}"
```

For YAML and cloud transport, use `"./ecofill-source[operation,cloud]"`, or
substitute `[operation,cloud]` in the selected-SHA direct requirement above.
Heavy backends remain optional. Extras install libraries, not weights. Obtain
weights deliberately under their licenses and retain the required checkpoint
SHA256s; see [models](models.md). Core or RF qualification does not qualify these
heavy extras, GPU operation or foundation inference on your platform.

## Install or build a wheel

No Git is needed for a wheel supplied by a colleague:

```sh
python -m pip install ./ecofill-0.0.1-py3-none-any.whl
python -m pip check
```

From a public checkout, build both artifacts with its declared build tools:

```sh
python -m pip install "./ecofill-source[dev]"
python -m build --sdist --wheel --no-isolation --outdir ./dist ./ecofill-source
python -m pip install ./dist/ecofill-0.0.1-py3-none-any.whl
```

Use a second fresh environment to test the artifacts, rather than your editable
build environment. Installing `./dist/ecofill-0.0.1.tar.gz` makes pip reconstruct
and build the source distribution in isolation. Both artifacts carry the package,
operation schema and vendored licenses. The wheel does not ship the checkout's
examples/tests/docs. A source archive is also not the complete public checkout.

## Standalone Python quick start

Copy the [README quick start](../README.md#quick-start) into `quickstart.py` in
your own working directory, then run `python quickstart.py`. It uses only the
installed package, NumPy and pandas, with eight synthetic 15-minute temperatures:
`[20, missing, missing, 23, missing, missing, missing, 20]`. The 30-minute whole-gap
limit fills exactly 21 and 22; the 45-minute gap stays unresolved with
`gap_exceeds_limit`. All three observed values remain unchanged.

The example saves `quickstart-export/` in the current directory. Its CSV/JSON
files record raw/completed values, masks, cell states/reasons and provenance.
Choose a new working directory for another run; the export directory must not
already exist. User inputs/outputs belong in your workspace, outside site-packages.

## Installed CLI first run

The installed CLI provides `validate`, `plan`, `run`, `inspect` and `export`.
This small software smoke uses synthetic LIN/RF data; it is not a field assessment.
Install `operation,research` above. Explicitly obtain the example file from the
same public revision; it is not assumed to be in the wheel. If you already cloned
that revision, use `ecofill-source/examples/operation_demo.py`. Otherwise download
the single public file into a new workspace:

PowerShell:

```powershell
Invoke-WebRequest "https://raw.githubusercontent.com/jequihua/ecofill/$ecofillSha/examples/operation_demo.py" -OutFile operation_demo.py
```

POSIX:

```sh
curl --fail --location "https://raw.githubusercontent.com/jequihua/ecofill/${ecofill_sha}/examples/operation_demo.py" --output operation_demo.py
```

From that workspace (or substitute the checkout's example path), both platforms:

```sh
python operation_demo.py prepare --output synthetic-inputs
ecofill validate --config synthetic-inputs/config.json
ecofill plan --config synthetic-inputs/config.json --output synthetic-plan
ecofill run --config synthetic-inputs/config.json --run-id first
ecofill inspect --run-uri synthetic-inputs/output/runs/first
python -I -S -B operation_demo.py readback --output synthetic-inputs/output/runs/first
```

The prepare step creates four synthetic CSVs and explicit hash-pinned source/config
metadata. Plan expects 12 gaps and at most 15 calls. Run/independent stdlib readback
expects 19 fills and 2,380 preserved observations, with LIN and RF-X. A QC-rejected
raw value and an unresolved donorless gap remain recorded. No weights, private
data or cloud access are needed. Inputs, plan and outputs go to the named workspace
directories; use new names for another run. For operator configs, exit/status
semantics and bounded cloud work, see [operation](operation.md).

If `ecofill` is not on PATH, use `.venv\Scripts\ecofill.exe` (Windows) or
`.venv/bin/ecofill` (POSIX), pointing to your actual environment. Activation adds
that scripts directory to PATH.

## Updates, provenance and troubleshooting

For a new approved revision, record its full SHA. The package version may still
be 0.0.1, so version output alone does not identify source. Reinstall ecofill
explicitly when switching revisions, leaving dependency resolution enabled:

PowerShell:

```powershell
python -m pip uninstall ecofill
python -m pip install "ecofill @ git+https://github.com/jequihua/ecofill.git@$ecofillSha"
python -m pip check
```

POSIX:

```sh
python -m pip uninstall ecofill
python -m pip install "ecofill @ git+https://github.com/jequihua/ecofill.git@${ecofill_sha}"
python -m pip check
```

Use your newly selected full SHA and retain your chosen extras in the requirement.
`python -m pip install --upgrade ./NEW_WHEEL.whl` handles a
newer package version. At an unchanged version, use the explicit uninstall/install
above for a new wheel or local checkout too. Editable code edits take effect
directly; reinstall after changing dependencies or entry points.

Record interpreter, version, import origin and direct-URL metadata (save this as
`provenance.py` and run `python provenance.py`):

```python
import importlib.metadata as metadata
import json
import sys
import ecofill

distribution = metadata.distribution("ecofill")
print(sys.executable)
print(distribution.version)
print(ecofill.__file__)
direct_url = json.loads(distribution.read_text("direct_url.json") or "null")
print(direct_url)
if len(sys.argv) == 2:  # For a GitHub install, supply the selected SHA.
    assert direct_url["vcs_info"]["commit_id"] == sys.argv[1]
```

For the pinned GitHub path, run `python provenance.py "$ecofillSha"` (PowerShell)
or `python provenance.py "$ecofill_sha"` (POSIX). A mismatched revision stops here;
reinstall the selected revision before running its examples. For local installs,
also record `git -C ecofill-source rev-parse HEAD`; local directory metadata alone
does not identify that source commit.

VCS provenance records the requested revision and resolved commit. Local/wheel
provenance records its file URL (and archive hash for a wheel); it does not prove
a Git source SHA. Record the supplied wheel SHA256 separately. Editable provenance
has `dir_info.editable: true`, and the import path points into your checkout.

- Check `python --version`, `python -m pip --version` and the provenance above
  when imports or the CLI use the wrong environment. Run outside the checkout
  and remove an unintended `PYTHONPATH` from your shell, rather than injecting paths.
- For a VCS failure, check `git --version` and
  `git ls-remote https://github.com/jequihua/ecofill.git refs/heads/main`.
  A wheel/local source directory avoids the Git prerequisite.
- For network or dependency errors, retain pip's exact error, check your approved
  network/proxy/package-index access and `python -m pip check`. Do not bypass TLS,
  replace pinned extras or select an arbitrary extra index to make an error disappear.
- Pip installation and the synthetic LIN/RF workflow have been tested on Windows
  CPython 3.12. These POSIX instructions have not been tested on Linux/macOS.

Official behavior: [regular/editable installs](https://pip.pypa.io/en/stable/topics/local-project-installs/),
[VCS requirements](https://pip.pypa.io/en/stable/topics/vcs-support/),
[building distributions](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
