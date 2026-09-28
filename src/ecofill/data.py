"""Owned input values and explicit, aligned observation provenance."""

import hashlib

import pandas as pd

from .schema import TimeSeriesSchema, validate_frame


class SeriesData:
    """Copy caller data; only QC-passed, unhidden values are method donors.

    Missing raw cells are naturally absent unless marked structural or inserted.
    The caller explicitly identifies QC-passed observations; finite alone is not
    evidence of validity. Artificial cells retain raw truth but are hidden.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        schema: TimeSeriesSchema,
        observed: pd.DataFrame,
        qc_rejected: pd.DataFrame | None = None,
        structural: pd.DataFrame | None = None,
        artificial: pd.DataFrame | None = None,
        inserted: pd.DataFrame | None = None,
    ):
        validate_frame(frame, schema)
        raw = frame.copy(deep=True)

        def aligned(name, supplied):
            if supplied is None:
                return pd.DataFrame(False, index=raw.index, columns=raw.columns)
            if not isinstance(supplied, pd.DataFrame) or not supplied.index.equals(raw.index):
                raise ValueError(f"{name} mask must match the input index exactly")
            if not supplied.columns.equals(raw.columns):
                raise ValueError(f"{name} mask must match input columns and order exactly")
            if not all(pd.api.types.is_bool_dtype(dtype) for dtype in supplied.dtypes):
                raise ValueError(f"{name} mask must be boolean")
            if supplied.isna().to_numpy().any():
                raise ValueError(f"{name} mask cannot contain missing values")
            return supplied.astype(bool).copy(deep=True)

        if observed is None:
            raise ValueError("observed mask is required; finite values are not assumed valid")
        qualified = aligned("observed", observed)
        qc = aligned("qc_rejected", qc_rejected)
        structure = aligned("structural", structural)
        hidden = aligned("artificial", artificial)
        timestamps = aligned("inserted", inserted)
        present = raw.notna()
        if (qualified & ~present).to_numpy().any():
            raise ValueError("observed mask must mark raw values")
        if (qc & ~present).to_numpy().any():
            raise ValueError("qc_rejected must mark raw values")
        if (qualified & qc).to_numpy().any():
            raise ValueError("observed and qc_rejected cannot overlap")
        if (structure & present).to_numpy().any():
            raise ValueError("structural cells cannot contain raw values")
        if (timestamps & present).to_numpy().any():
            raise ValueError("inserted timestamps cannot contain raw values")
        if (timestamps.any(axis=1) & ~timestamps.all(axis=1)).any():
            raise ValueError("inserted timestamps must mark every variable in the row")
        if (structure & timestamps).to_numpy().any():
            raise ValueError("structural and inserted masks cannot overlap")
        if (hidden & ~qualified).to_numpy().any():
            raise ValueError("artificial cells must be observed QC-passed values")

        # A hidden target also hides its deterministic descendants at that slot.
        for _ in schema.variables:
            for variable in schema.variables:
                for parent in variable.derived_from:
                    hidden.loc[:, variable.name] = (
                        hidden[variable.name]
                        | (hidden[parent] & qualified[variable.name])
                    )

        donor = qualified & ~hidden
        absent = ~present & ~structure & ~timestamps
        self.schema = schema
        self._raw = raw
        self._masks = {
            "raw_present": present,
            "observed": qualified,
            "donor": donor,
            "absent": absent,
            "qc_rejected": qc,
            "structural": structure,
            "artificial": hidden,
            "inserted": timestamps,
        }
        # Object storage prevents pandas from rounding large observed integers
        # when masking an integer column introduces missing cells or a fill is fractional.
        available = raw.copy(deep=True)
        for name, dtype in raw.dtypes.items():
            if pd.api.types.is_integer_dtype(dtype):
                available[name] = available[name].astype(object)
        self._available = available.mask(~donor)
        digest = hashlib.sha256(repr(schema).encode("utf-8"))
        digest.update(str(raw.index.tz).encode("utf-8"))
        for table in (raw, *self._masks.values()):
            digest.update(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes())
        self.fingerprint = digest.hexdigest()

    @property
    def raw(self) -> pd.DataFrame:
        return self._raw.copy(deep=True)

    @property
    def available(self) -> pd.DataFrame:
        return self._available.copy(deep=True)

    def mask(self, name: str) -> pd.DataFrame:
        return self._masks[name].copy(deep=True)

    @property
    def masks(self) -> dict[str, pd.DataFrame]:
        return {name: value.copy(deep=True) for name, value in self._masks.items()}
