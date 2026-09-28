"""Aligned reconstruction result and explicit, portable export."""

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from .data import SeriesData
from .planning import ImputationPlan


class ImputationResult:
    def __init__(self, data: SeriesData, plan: ImputationPlan, completed, states, methods, reasons,
                 imputed, backend_identity=None, quality=None):
        self.schema = data.schema
        self.plan = plan
        self._raw = data.raw
        self._completed = completed.copy(deep=True)
        self._states = states.copy(deep=True)
        self._methods = methods.copy(deep=True)
        self._reasons = reasons.copy(deep=True)
        self._masks = data.masks
        self._masks["imputed"] = imputed.copy(deep=True)
        self._backend_identity = deepcopy(backend_identity)
        self._quality = None if quality is None else quality.copy(deep=True)

    @property
    def raw(self) -> pd.DataFrame:
        return self._raw.copy(deep=True)

    @property
    def completed(self) -> pd.DataFrame:
        return self._completed.copy(deep=True)

    @property
    def states(self) -> pd.DataFrame:
        return self._states.copy(deep=True)

    @property
    def methods(self) -> pd.DataFrame:
        return self._methods.copy(deep=True)

    @property
    def unresolved_reasons(self) -> pd.DataFrame:
        return self._reasons.copy(deep=True)

    @property
    def quality(self) -> pd.DataFrame | None:
        return None if self._quality is None else self._quality.copy(deep=True)

    def mask(self, name: str) -> pd.DataFrame:
        return self._masks[name].copy(deep=True)

    def provenance(self) -> dict:
        index = self._raw.index
        counts = self._states.stack().value_counts().to_dict()
        provenance = {
            "format_version": 1,
            "data_fingerprint": self.plan.data_fingerprint,
            "schema": {
                "timezone": str(index.tz),
                "cadence_seconds": self.schema.cadence.total_seconds(),
                "time_policy": self.schema.time_policy,
                "variables": [
                    {"name": item.name, "unit": item.unit, "role": item.role,
                     "derived_from": list(item.derived_from)}
                    for item in self.schema.variables
                ],
            },
            "timestamps": [stamp.isoformat() for stamp in index],
            "plan": self.plan.to_dict(),
            "state_counts": {str(key): int(value) for key, value in counts.items()},
            "masks": list(self._masks),
            "files": ["raw.csv", "completed.csv", "masks.csv", "cells.csv", "provenance.json"],
        }
        if self._backend_identity is not None:
            provenance["backend"] = deepcopy(self._backend_identity)
        if self._quality is not None:
            provenance["quality"] = "MDS reference tier: 1, 2, or 3; not calibrated uncertainty"
        return provenance

    def export(self, directory: str | Path) -> Path:
        """Write aligned CSV tables and JSON metadata; refuse to overwrite a directory."""
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=False)
        self._raw.to_csv(destination / "raw.csv", index_label="timestamp")
        self._completed.to_csv(destination / "completed.csv", index_label="timestamp")
        masks = pd.concat(self._masks, axis=1)
        masks.columns = [f"{name}__{variable}" for name, variable in masks.columns]
        masks.to_csv(destination / "masks.csv", index_label="timestamp")
        cells = pd.DataFrame(
            (
                (timestamp.isoformat(), variable,
                 self._states.at[timestamp, variable], self._methods.at[timestamp, variable],
                 self._reasons.at[timestamp, variable],
                 None if self._quality is None else self._quality.at[timestamp, variable])
                for timestamp in self._raw.index for variable in self._raw.columns
            ),
            columns=["timestamp", "variable", "state", "method", "reason", "quality"],
        )
        if self._quality is None:
            cells = cells.drop(columns="quality")
        cells.to_csv(destination / "cells.csv", index=False)
        (destination / "provenance.json").write_text(
            json.dumps(self.provenance(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return destination
