# Export formats and the standalone reader

ecofill writes plain CSV and strict JSON so that results can be used without
ecofill, NumPy or pandas. Both writers create a new directory and refuse to
overwrite an existing one. Missing numbers are empty CSV fields; `NaN` and
infinity are never written.

## Result export (`ImputationResult.export`)

| File | Contents |
|---|---|
| `raw.csv` | the input values, one column per variable |
| `completed.csv` | observed values plus reconstructed cells; unresolved cells stay empty |
| `masks.csv` | one boolean column per mask and variable (`<mask>__<variable>`) |
| `cells.csv` | one row per timestamp and variable: `state`, `method`, `reason` (and `quality` for MDS) |
| `provenance.json` | format version, schema, units, lineage, clock, timestamps, plan, state counts, data fingerprint and, when present, the backend identity |

The nine masks are `raw_present`, `observed`, `donor`, `absent`,
`qc_rejected`, `structural`, `artificial`, `inserted` and `imputed`. A `donor`
cell is observed and not artificially hidden; a completed value equals the raw
value at every donor cell.

## Uncertainty export (`export_uncertainty`)

`uncertainty.json` (kind `ecofill_block_uncertainty`, format version 1) holds
the calibration, recipe, training and block metadata. Depending on what was
exported, it is accompanied by `intervals.csv`, the three draw files
`draw_cells.csv`, `draws.csv` and `draw_samples.csv`, and `flux.csv`. Metadata
and files must agree: a component is present in both or in neither.

## Reading exports back

[examples/read_export.py](../examples/read_export.py) is a reference reader
that uses only the Python standard library:

```console
python -I -S examples/read_export.py result <export directory>
python -I -S examples/read_export.py uncertainty <export directory>
```

It prints one normalized JSON document or exits with status 2 and a reason. It
rejects nonstandard JSON, duplicate keys, unknown versions, missing or
undeclared files, changed headers or variable order, mismatched timestamp axes
or units, malformed numbers and booleans, and inconsistent draw identities. For
results it also checks that masks, states and values agree with each other.

It checks structure and internal consistency only. A self-consistent but wrong
value passes; comparing values with independent expectations is a separate
step. It is not an authenticated archive format.
