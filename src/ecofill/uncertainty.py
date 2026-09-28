"""Offline whole-block residual calibration, intervals, evaluation and aligned draws.

The calibration unit is a declared physical block. Its score is the maximum
absolute signed residual (truth minus frozen blend prediction) divided by a
training-fixed channel scale over every hidden primitive cell. Nominal
simultaneous coverage refers to a new matched block and assumes exchangeable
calibration/test blocks; nonoverlapping time blocks do not prove that
assumption and this module does not verify it or certify field coverage.
"""

from dataclasses import dataclass
from fractions import Fraction
import csv
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SeriesData
from .ec_recipe import signed_integral
from .ensemble import (EnsemblePlan, EnsembleRecipe, check_result, origins, primitive_names,
                       propagate, validate_transforms)

ALGORITHM = "whole_block_residual_bootstrap_v1"
ASSUMPTIONS = (
    "Intervals and draws are support-conditional and offline-only. Nominal simultaneous "
    "coverage refers to a new block in the same exact stratum under exchangeable "
    "calibration/test blocks; nonoverlapping time blocks do not prove exchangeability. "
    "Draws resample whole calibration residual blocks independently per target block and "
    "make no claim about dependence between blocks. Empirical draw ranges are not calibrated "
    "intervals. No field calibration evidence is implied."
)


def _text(value) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _stamp(value) -> bool:
    return isinstance(value, pd.Timestamp) and not pd.isna(value) and value.tz is not None


@dataclass(frozen=True)
class SupportSpan:
    """Inclusive training truth/context support of one source group."""

    source_group: str
    start: pd.Timestamp
    stop: pd.Timestamp

    def __post_init__(self):
        if not _text(self.source_group) or not _stamp(self.start) or not _stamp(self.stop) \
                or self.stop < self.start:
            raise ValueError("support span needs a source group and ordered timezone-aware bounds")

    def to_dict(self):
        return {"source_group": self.source_group, "start": self.start.isoformat(),
                "stop": self.stop.isoformat()}


@dataclass(frozen=True)
class BlockRecord:
    """Caller-declared physical block: truth span and the context used to predict it."""

    case_id: str
    source_group: str
    physical_block: str
    split: str
    regime: str
    start: pd.Timestamp
    stop: pd.Timestamp
    context_start: pd.Timestamp
    context_stop: pd.Timestamp

    def __post_init__(self):
        if not all(_text(value) for value in (self.case_id, self.source_group,
                                              self.physical_block, self.regime)):
            raise ValueError("block identities and regime must be nonempty strings")
        if self.split not in {"calibration", "evaluation", "application"}:
            raise ValueError("block split must be calibration, evaluation or application")
        if not all(_stamp(value) for value in (self.start, self.stop, self.context_start,
                                                self.context_stop)):
            raise ValueError("block spans require timezone-aware Timestamps")
        if not self.context_start <= self.start <= self.stop <= self.context_stop:
            raise ValueError("block context must contain its ordered truth span")

    def to_dict(self):
        return {"case_id": self.case_id, "source_group": self.source_group,
                "physical_block": self.physical_block, "split": self.split,
                "regime": self.regime, "start": self.start.isoformat(),
                "stop": self.stop.isoformat(), "context_start": self.context_start.isoformat(),
                "context_stop": self.context_stop.isoformat()}


@dataclass(frozen=True)
class SupportStratum:
    """Exact support key fixed before calibration; no pooling across keys or source groups."""

    source_group: str
    regime: str
    recipe_identity: str
    channels: tuple[tuple[str, str], ...]
    cadence: pd.Timedelta
    length: int
    layout: tuple[tuple[int, str], ...]
    mode: str = "offline"

    def __post_init__(self):
        if self.mode != "offline":
            raise ValueError("causal uncertainty is unsupported; block calibration is offline-only")
        if not _text(self.source_group) or not _text(self.regime) or not _text(self.recipe_identity):
            raise ValueError("stratum source group, regime and recipe identity are required")
        if (not isinstance(self.channels, tuple) or not self.channels
                or any(not isinstance(item, tuple) or len(item) != 2 or not all(_text(v) for v in item)
                       for item in self.channels)
                or len({name for name, _ in self.channels}) != len(self.channels)):
            raise ValueError("stratum channels must be ordered unique (name, unit) pairs")
        if not isinstance(self.cadence, pd.Timedelta) or pd.isna(self.cadence) or self.cadence <= pd.Timedelta(0):
            raise ValueError("stratum cadence must be a positive Timedelta")
        if type(self.length) is not int or self.length < 1:
            raise ValueError("stratum length must be a positive integer")
        order = {name: position for position, (name, _) in enumerate(self.channels)}
        if (not isinstance(self.layout, tuple) or not self.layout
                or any(not isinstance(cell, tuple) or len(cell) != 2 or type(cell[0]) is not int
                       or not 0 <= cell[0] < self.length or cell[1] not in order for cell in self.layout)):
            raise ValueError("stratum layout must name (offset, channel) cells inside the window")
        keys = [(offset, order[name]) for offset, name in self.layout]
        if keys != sorted(set(keys)):
            raise ValueError("stratum layout must be unique and ordered by offset, then channel order")

    def to_dict(self):
        return {"source_group": self.source_group, "regime": self.regime,
                "recipe_identity": self.recipe_identity, "mode": self.mode,
                "channels": [list(item) for item in self.channels],
                "cadence_seconds": self.cadence.total_seconds(), "length": self.length,
                "layout": [list(cell) for cell in self.layout]}


@dataclass(frozen=True)
class BlockCase:
    """One block, its SeriesData fold and the frozen recipe's result or failure message."""

    record: BlockRecord
    data: SeriesData
    recipe: EnsembleRecipe
    result: object = None
    failure: str | None = None


@dataclass(frozen=True)
class StratumCalibration:
    stratum: SupportStratum
    status: str
    reason: str | None
    rank: int | None
    threshold: float | None
    block_ids: tuple[str, ...]
    physical_blocks: tuple[str, ...]
    residuals: tuple[tuple[float, ...], ...]
    scores: tuple[float, ...]
    exclusions: tuple[tuple[str, str], ...]
    counts: tuple[tuple[str, int], ...]

    @property
    def n_blocks(self) -> int:
        return len(self.block_ids)

    def to_dict(self):
        return {"stratum": self.stratum.to_dict(), "status": self.status, "reason": self.reason,
                "n_blocks": self.n_blocks, "rank": self.rank, "threshold": self.threshold,
                "block_ids": list(self.block_ids), "physical_blocks": list(self.physical_blocks),
                "residuals": [list(item) for item in self.residuals], "scores": list(self.scores),
                "exclusions": [list(item) for item in self.exclusions], "counts": dict(self.counts)}


@dataclass(frozen=True)
class BlockCalibration:
    """Frozen calibration; every field is immutable, so later input edits cannot change it."""

    alpha: float
    min_blocks: int
    purge: pd.Timedelta
    scales: tuple[tuple[str, float], ...]
    scale_identity: str
    unit_scale_fallback: tuple[str, ...]
    training: tuple[SupportSpan, ...]
    blocks: tuple[BlockRecord, ...]
    strata: tuple[StratumCalibration, ...]
    schema: object
    recipes: tuple[EnsembleRecipe, ...]

    def entry(self, stratum):
        return next((item for item in self.strata if item.stratum == stratum), None)

    def to_dict(self, *, identity=True):
        payload = {"alpha": self.alpha, "min_blocks": self.min_blocks,
                   "schema": {"cadence_seconds": self.schema.cadence.total_seconds(),
                              "variables": [[item.name, item.unit, item.role, list(item.derived_from)]
                                            for item in self.schema.variables]},
                   "purge_seconds": self.purge.total_seconds(),
                   "scales": dict(self.scales), "scale_identity": self.scale_identity,
                   "unit_scale_fallback": list(self.unit_scale_fallback),
                   "training": [item.to_dict() for item in self.training],
                   "blocks": [item.to_dict() for item in self.blocks],
                   "strata": [item.to_dict() for item in self.strata],
                   "recipes": [_recipe_dict(item) for item in self.recipes],
                   "score": "max |truth - prediction| / training scale over the block's hidden primitive cells",
                   "rank_rule": "k = ceil((n + 1) * (1 - alpha)), one-based; k > n is insufficient support",
                   "assumptions": ASSUMPTIONS}
        if identity:
            payload["identity"] = self.identity
        return payload

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(_portable(self.to_dict(identity=False)),
                                         sort_keys=True).encode()).hexdigest()


def _recipe_dict(recipe) -> dict:
    return {"identity": recipe.identity, **recipe.to_dict()}


def _check_recipe_training(record, data, recipe, training) -> None:
    """A recipe's declared fitted rows must be covered by same-group training support."""
    if recipe.training is None:
        return
    start, stop = recipe.training
    index = data.raw.index
    if stop > len(index):
        raise ValueError(f"recipe training rows {recipe.training} exceed the {record.case_id} grid")
    first, last = index[start], index[stop - 1]
    if not any(span.source_group == record.source_group and span.start <= first and last <= span.stop
               for span in training):
        raise ValueError(f"recipe training interval {first.isoformat()}..{last.isoformat()} of "
                         f"{record.case_id} is not covered by declared training support")


def _separated(a_start, a_stop, b_start, b_stop, purge) -> bool:
    return b_start - a_stop > purge or a_start - b_stop > purge


def _check_blocks(records, training, purge) -> None:
    """Unique IDs; no duplicate/aliased physical blocks; purged cross-split separation."""
    ids = [record.case_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate block case ID")
    for position, first in enumerate(records):
        for second in records[position + 1:]:
            if first.source_group != second.source_group:
                continue
            if first.physical_block == second.physical_block:
                raise ValueError(f"duplicate physical block {first.physical_block} in {first.source_group}")
            if not _separated(first.start, first.stop, second.start, second.stop, pd.Timedelta(0)):
                raise ValueError(f"overlapping physical blocks {first.case_id} and {second.case_id}")
            if first.split != second.split and not _separated(
                    first.context_start, first.context_stop,
                    second.context_start, second.context_stop, purge):
                raise ValueError(f"cross-split context overlap or purge violation: "
                                 f"{first.case_id} and {second.case_id}")
    for span in training:
        for record in records:
            if span.source_group == record.source_group and not _separated(
                    span.start, span.stop, record.context_start, record.context_stop, purge):
                raise ValueError(f"block {record.case_id} context overlaps training support or its purge")


def _window(data: SeriesData, record: BlockRecord):
    index = data.raw.index
    if record.start not in index or record.stop not in index:
        raise ValueError(f"block {record.case_id} span must lie on the data grid")
    return index.get_loc(record.start), index.get_loc(record.stop) + 1


def _targets(data: SeriesData, first: int, stop: int, *, truth: bool):
    names = primitive_names(data.schema)
    donor, structural = data.mask("donor"), data.mask("structural")
    artificial = data.mask("artificial")
    cells = []
    for row in range(first, stop):
        for name in names:
            stamp = data.raw.index[row]
            if donor.at[stamp, name] or structural.at[stamp, name]:
                continue
            if truth and not artificial.at[stamp, name]:
                raise ValueError("block contains natural, QC-rejected or inserted gaps; only "
                                 "artificially hidden QC-passed observations are truth")
            cells.append((row - first, name))
    if not cells:
        raise ValueError("block has no hidden primitive cells")
    return tuple(cells)


def _check_hidden_descendants(data: SeriesData) -> None:
    artificial, observed = data.mask("artificial"), data.mask("observed")
    for item in data.schema.variables:
        for parent in item.derived_from:
            if (artificial[parent] & observed[item.name] & ~artificial[item.name]).any():
                raise ValueError(f"hidden target descendant {item.name} is visible")


def _stratum(record, data, recipe, first, stop, layout) -> SupportStratum:
    units = {item.name: item.unit for item in data.schema.variables}
    return SupportStratum(record.source_group, record.regime, recipe.identity,
                          tuple((name, units[name]) for name in primitive_names(data.schema)),
                          data.schema.cadence, stop - first, layout, recipe.mode)


def _check_ensemble(data, result, recipe, label) -> None:
    check_result(data, result, label=label)
    if not isinstance(result.plan, EnsemblePlan):
        raise ValueError("uncertainty calibrates the frozen blend; supply a blend() result "
                         "(one model is a one-member recipe)")
    if result.plan.recipe != recipe:
        raise ValueError(f"{label} was produced by a different recipe")
    if recipe.mode != "offline":
        raise ValueError("causal uncertainty is unsupported; block calibration is offline-only")


def _check_case(case, split, training):
    if not isinstance(case, BlockCase) or not isinstance(case.record, BlockRecord) \
            or not isinstance(case.data, SeriesData) or not isinstance(case.recipe, EnsembleRecipe):
        raise ValueError("cases must be BlockCase(record, SeriesData, EnsembleRecipe, result)")
    if case.record.split != split:
        raise ValueError(f"block {case.record.case_id} is not declared as {split}")
    if case.recipe.mode != "offline":
        raise ValueError("causal uncertainty is unsupported; block calibration is offline-only")
    if case.result is None:
        if not _text(case.failure):
            raise ValueError("a case without a result must name its prediction failure")
    elif case.failure is not None:
        raise ValueError("a case has either a result or a failure")
    else:
        _check_ensemble(case.data, case.result, case.recipe, f"block {case.record.case_id} result")
    _check_hidden_descendants(case.data)
    _check_recipe_training(case.record, case.data, case.recipe, training)
    first, stop = _window(case.data, case.record)
    layout = _targets(case.data, first, stop, truth=True)
    return first, stop, layout, _stratum(case.record, case.data, case.recipe, first, stop, layout)


def _rank(n: int, alpha: float) -> int:
    # Exact rational arithmetic on the binary alpha; no floating rounding of k.
    return math.ceil(Fraction(n + 1) * (1 - Fraction(alpha)))


def calibrate_blocks(cases, *, strata, training, purge, scales, scale_identity, alpha,
                     min_blocks, unit_scale_fallback=()) -> BlockCalibration:
    """Build the frozen whole-block residual bank and finite-sample thresholds per stratum."""
    if isinstance(alpha, bool) or type(alpha) not in (int, float) or not 0 < alpha < 1:
        raise ValueError("alpha must be a number in (0, 1)")
    if type(min_blocks) is not int or min_blocks < 1:
        raise ValueError("min_blocks must be a positive integer")
    if not isinstance(purge, pd.Timedelta) or pd.isna(purge) or purge < pd.Timedelta(0):
        raise ValueError("purge must be a nonnegative Timedelta")
    if not isinstance(strata, tuple) or not strata or any(not isinstance(item, SupportStratum) for item in strata) \
            or len(set(strata)) != len(strata):
        raise ValueError("declare a nonempty tuple of unique SupportStratum keys before calibration")
    if not isinstance(training, tuple) or any(not isinstance(item, SupportSpan) for item in training):
        raise ValueError("training support must be a tuple of SupportSpan")
    if not _text(scale_identity) or not isinstance(scales, dict):
        raise ValueError("training scales and their identity are required")
    if not isinstance(unit_scale_fallback, tuple):
        raise ValueError("unit_scale_fallback must be a tuple of channel names")
    needed = [name for item in strata for name, _ in item.channels]
    needed = list(dict.fromkeys(needed))
    if set(scales) != set(needed) or not set(unit_scale_fallback) <= set(needed):
        raise ValueError("scales must name exactly the stratum channels")
    effective = {}
    for name in needed:
        value = scales[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"training scale for {name} must be finite and nonnegative")
        if value == 0:
            if name not in unit_scale_fallback:
                raise ValueError(f"zero training scale for {name} needs explicit unit fallback")
            effective[name] = 1.0
        elif name in unit_scale_fallback:
            raise ValueError(f"unit fallback is only for a zero training scale ({name})")
        else:
            effective[name] = float(value)
    if not isinstance(cases, tuple) or not cases:
        raise ValueError("calibration requires a nonempty tuple of BlockCase")
    records = tuple(case.record for case in cases if isinstance(case, BlockCase))
    if len(records) != len(cases):
        raise ValueError("cases must be BlockCase records")
    _check_blocks(records, training, purge)
    schema = cases[0].data.schema if isinstance(cases[0].data, SeriesData) else None
    work = {item: {"ids": [], "physical": [], "residuals": [], "scores": [], "excluded": [],
                   "hidden": 0, "predicted": 0, "abstained": 0, "failed": 0} for item in strata}
    for case in cases:
        first, stop, layout, stratum = _check_case(case, "calibration", training)
        if case.data.schema != schema:
            raise ValueError("calibration cases must share one declared schema and lineage")
        if stratum not in work:
            raise ValueError(f"calibration block {case.record.case_id} matches no predeclared support stratum")
        entry = work[stratum]
        entry["hidden"] += len(layout)
        if case.result is None:
            entry["failed"] += len(layout)
            entry["excluded"].append((case.record.case_id, "prediction_failed: " + case.failure))
            continue
        index = case.data.raw.index
        imputed = case.result.mask("imputed")
        predicted = [cell for cell in layout if imputed.at[index[first + cell[0]], cell[1]]]
        entry["predicted"] += len(predicted)
        entry["abstained"] += len(layout) - len(predicted)
        if len(predicted) != len(layout):
            entry["excluded"].append((case.record.case_id,
                f"incomplete_prediction: {len(layout) - len(predicted)} of {len(layout)} cells abstained"))
            continue
        raw, completed = case.data.raw, case.result.completed
        residual = tuple(float(raw.at[index[first + offset], name])
                         - float(completed.at[index[first + offset], name]) for offset, name in layout)
        entry["ids"].append(case.record.case_id)
        entry["physical"].append(case.record.physical_block)
        entry["residuals"].append(residual)
        entry["scores"].append(max(abs(value) / effective[name]
                                   for value, (_, name) in zip(residual, layout)))
    results = []
    for stratum in strata:
        entry = work[stratum]
        n = len(entry["scores"])
        rank = _rank(n, alpha)
        threshold, status, reason = None, "calibrated", None
        if n < min_blocks:
            status, reason = "insufficient_support", f"{n} admitted blocks below minimum {min_blocks}"
        elif rank > n:
            status, reason = "insufficient_support", f"rank {rank} exceeds {n} admitted blocks"
        else:
            threshold = sorted(entry["scores"])[rank - 1]
        results.append(StratumCalibration(
            stratum, status, reason, rank, threshold, tuple(entry["ids"]), tuple(entry["physical"]),
            tuple(entry["residuals"]), tuple(entry["scores"]), tuple(entry["excluded"]),
            tuple((key, entry[key]) for key in ("hidden", "predicted", "abstained", "failed"))))
    recipes = tuple({case.recipe.identity: case.recipe for case in cases}.values())
    return BlockCalibration(float(alpha), min_blocks, purge, tuple(effective.items()), scale_identity,
                            unit_scale_fallback, training, records, tuple(results), schema, recipes)


def _resolve_targets(calibration, data, result, targets, *, draws):
    if not isinstance(calibration, BlockCalibration):
        raise ValueError("a frozen BlockCalibration is required")
    if not isinstance(data, SeriesData):
        raise ValueError("SeriesData is required")
    check_result(data, result)
    if not isinstance(result.plan, EnsemblePlan):
        raise ValueError("uncertainty applies to a frozen blend() result")
    if result.plan.recipe.mode != "offline":
        raise ValueError("causal uncertainty is unsupported; block calibration is offline-only")
    if not isinstance(targets, tuple) or not targets or any(not isinstance(item, BlockRecord) for item in targets):
        raise ValueError("targets must be a nonempty tuple of BlockRecord")
    if any(item.split == "calibration" for item in targets):
        raise ValueError("calibration blocks cannot be application targets")
    _check_blocks(calibration.blocks + targets, calibration.training, calibration.purge)
    for record in targets:
        # The recipe's fitted rows on this grid must be the frozen, declared training support.
        _check_recipe_training(record, data, result.plan.recipe, calibration.training)
    imputed = result.mask("imputed")
    resolved, claimed = [], set()
    for record in targets:
        first, stop = _window(data, record)
        if claimed & set(range(first, stop)):
            raise ValueError(f"overlapping target blocks on the same data at {record.case_id}")
        claimed |= set(range(first, stop))
        layout = _targets(data, first, stop, truth=False)
        stratum = _stratum(record, data, result.plan.recipe, first, stop, layout)
        entry = calibration.entry(stratum)
        if entry is None:
            reason = "unknown_support_stratum"
        elif draws and entry.n_blocks < calibration.min_blocks:
            reason = f"insufficient_support: {entry.n_blocks} complete blocks below minimum {calibration.min_blocks}"
        elif not draws and entry.status != "calibrated":
            reason = f"{entry.status}: {entry.reason}"
        elif not all(imputed.at[data.raw.index[first + offset], name] for offset, name in layout):
            reason = "incomplete_point_prediction"
        else:
            reason = None
        resolved.append((record, first, layout, stratum, entry, reason))
    return resolved


class BlockIntervals:
    """Simultaneous block intervals for matching supported primitive target cells only."""

    def __init__(self, calibration, data, result, resolved):
        index, columns = data.raw.index, data.raw.columns
        point = result.completed
        self._point = point
        self._lower = pd.DataFrame(np.nan, index=index, columns=columns)
        self._upper = pd.DataFrame(np.nan, index=index, columns=columns)
        self._status = pd.DataFrame(None, index=index, columns=columns, dtype=object)
        scales = dict(calibration.scales)
        blocks = []
        for record, first, layout, stratum, entry, reason in resolved:
            for offset, name in layout:
                stamp = index[first + offset]
                self._status.at[stamp, name] = reason or "calibrated_interval"
                if reason is None:
                    half = entry.threshold * scales[name]
                    self._lower.at[stamp, name] = float(point.at[stamp, name]) - half
                    self._upper.at[stamp, name] = float(point.at[stamp, name]) + half
            for row in range(first, first + stratum.length):
                for item in data.schema.variables:
                    if item.derived_from and not data.mask("donor").iat[row, columns.get_loc(item.name)]:
                        self._status.iat[row, columns.get_loc(item.name)] = "derived_interval_unsupported"
            blocks.append({"case_id": record.case_id, "record": record.to_dict(),
                           "status": reason or "calibrated_interval",
                           "stratum": stratum.to_dict(),
                           "threshold": None if reason else entry.threshold,
                           "rank": None if entry is None else entry.rank,
                           "n_blocks": None if entry is None else entry.n_blocks})
        self.blocks = tuple(blocks)
        self.calibration_identity = calibration.identity
        self.recipe = result.plan.recipe
        self.recipe_identity = self.recipe.identity
        self.data_fingerprint = data.fingerprint
        self.alpha = calibration.alpha
        self.units = tuple((item.name, item.unit) for item in data.schema.variables)

    @property
    def point(self):
        return self._point.copy(deep=True)

    @property
    def lower(self):
        return self._lower.copy(deep=True)

    @property
    def upper(self):
        return self._upper.copy(deep=True)

    @property
    def status(self):
        return self._status.copy(deep=True)

    def to_dict(self):
        return {"calibration_identity": self.calibration_identity,
                "recipe_identity": self.recipe_identity, "recipe": _recipe_dict(self.recipe),
                "data_fingerprint": self.data_fingerprint, "alpha": self.alpha, "blocks": deepcopy(list(self.blocks)),
                "interval": "point +/- threshold * training scale; simultaneous over the block"}


def block_intervals(calibration, data, result, targets) -> BlockIntervals:
    """Apply frozen thresholds to target blocks; unknown/insufficient support is unavailable."""
    return BlockIntervals(calibration, data, result,
                          _resolve_targets(calibration, data, result, targets, draws=False))


def _ratio(numerator, denominator, reason):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None,
            "reason": None if denominator else reason}


def evaluate_blocks(calibration, cases) -> dict:
    """Score untouched evaluation truth against the frozen calibration without changing it."""
    if not isinstance(calibration, BlockCalibration):
        raise ValueError("a frozen BlockCalibration is required")
    if not isinstance(cases, tuple) or not cases or any(not isinstance(case, BlockCase) for case in cases):
        raise ValueError("evaluation requires a nonempty tuple of BlockCase")
    _check_blocks(calibration.blocks + tuple(case.record for case in cases),
                  calibration.training, calibration.purge)
    scales = dict(calibration.scales)
    channels = {}
    blocks = []
    covered_blocks = eligible_blocks = 0
    for case in cases:
        first, stop, layout, stratum = _check_case(case, "evaluation", calibration.training)
        if case.data.schema != calibration.schema:
            raise ValueError("evaluation schema differs from the calibration schema")
        entry = calibration.entry(stratum)
        index = case.data.raw.index
        predicted = []
        if case.result is not None:
            imputed = case.result.mask("imputed")
            completed = case.result.completed
            predicted = [(offset, name, float(completed.at[index[first + offset], name])
                          - float(case.data.raw.at[index[first + offset], name]))
                         for offset, name in layout if imputed.at[index[first + offset], name]]
        available = (entry is not None and entry.status == "calibrated"
                     and len(predicted) == len(layout))
        for offset, name in layout:
            item = channels.setdefault(name, {"hidden": 0, "predicted": 0, "abstained": 0,
                                              "failed": 0, "errors": [], "interval_cells": 0,
                                              "covered_cells": 0, "widths": []})
            item["hidden"] += 1
            item["failed" if case.result is None else "abstained"] += 1
        for _, name, error in predicted:
            item = channels[name]
            item["abstained"] -= 1
            item["predicted"] += 1
            item["errors"].append(error)
            if available:
                item["interval_cells"] += 1
                item["covered_cells"] += int(abs(error) / scales[name] <= entry.threshold)
                item["widths"].append(2 * entry.threshold * scales[name])
        score = max((abs(error) / scales[name] for _, name, error in predicted), default=None)
        if available:
            eligible_blocks += 1
            covered_blocks += int(score <= entry.threshold)
        blocks.append({"case_id": case.record.case_id, "hidden": len(layout),
                       "predicted": len(predicted), "failed": len(layout) if case.result is None else 0,
                       "interval_available": available,
                       "status": "calibrated_interval" if available else
                       "unknown_support_stratum" if entry is None else
                       f"{entry.status}: {entry.reason}" if entry.status != "calibrated" else
                       "incomplete_point_prediction",
                       "score": score if len(predicted) == len(layout) else None,
                       "covered": (score <= entry.threshold) if available else None})
    report = {}
    for name, item in channels.items():
        errors = np.asarray(item["errors"], dtype=float)
        n = len(errors)
        report[name] = {
            "hidden": item["hidden"], "predicted": item["predicted"],
            "abstained": item["abstained"], "failed": item["failed"],
            "error_denominator": n,
            "mae": float(np.mean(np.abs(errors))) if n else None,
            "rmse": float(np.sqrt(np.mean(errors ** 2))) if n else None,
            "bias": float(np.mean(errors)) if n else None,
            "point_reason": None if n else "no predicted cells",
            "interval_availability": _ratio(item["interval_cells"], item["hidden"], "no hidden cells"),
            "cell_coverage": _ratio(item["covered_cells"], item["interval_cells"],
                                    "no cells with calibrated intervals"),
            "mean_width": float(np.mean(item["widths"])) if item["widths"] else None,
        }
    return {"calibration_identity": calibration.identity, "alpha": calibration.alpha,
            "error": "prediction - truth", "channels": report, "blocks": blocks,
            "block_coverage": _ratio(covered_blocks, eligible_blocks,
                                     "no completely predicted block with a calibrated interval"),
            "assumptions": ASSUMPTIONS}


class CompletedDraws:
    """Aligned completed reconstructions from whole sampled residual blocks.

    Observations are identical in every draw; structural and unsupported missing
    cells stay missing with reasons. Draws are alternative reconstructions, not
    additional observations.
    """

    def __init__(self, *, frames, status, reasons, samples, lineage, blocks, seed, calibration,
                 recipe, data):
        self._frames = tuple(frames)
        self._status = status
        self._reasons = reasons
        self.draw_ids = tuple(f"draw-{number:04d}" for number in range(len(frames)))
        self.samples = tuple(samples)
        self._lineage = lineage
        self._blocks = blocks
        self.seed = seed
        self.algorithm = ALGORITHM
        self.calibration_identity = calibration.identity
        self.recipe = recipe
        self.recipe_identity = recipe.identity
        self.data_fingerprint = data.fingerprint
        self.schema = data.schema
        self._masks = {name: data.mask(name) for name in ("observed", "donor", "structural")}

    def draw(self, draw_id: str) -> pd.DataFrame:
        return self._frames[self.draw_ids.index(draw_id)].copy(deep=True)

    @property
    def status(self):
        return self._status.copy(deep=True)

    @property
    def reasons(self):
        return self._reasons.copy(deep=True)

    @property
    def supported(self):
        return self._status.isin(["observed", "sampled_block_residual", "derived_in_draw"])

    def mask(self, name: str):
        return self._masks[name].copy(deep=True)

    @property
    def lineage(self):
        return deepcopy(self._lineage)

    def to_dict(self):
        return {"algorithm": self.algorithm, "seed": self.seed, "draw_ids": list(self.draw_ids),
                "calibration_identity": self.calibration_identity,
                "recipe_identity": self.recipe_identity, "recipe": _recipe_dict(self.recipe),
                "data_fingerprint": self.data_fingerprint,
                "samples": [{"draw_id": d, "target_block": t, "source_block": s,
                             "source_physical_block": p} for d, t, s, p in self.samples],
                "blocks": deepcopy(self._blocks), "derived_lineage": deepcopy(self._lineage),
                "interpretation": "alternative reconstructions; never additional observations"}


def sample_block_draws(calibration, data, result, targets, *, n_draws, seed,
                       transforms=()) -> CompletedDraws:
    """Seeded whole-block residual draws added to the frozen blend, then same-draw transforms."""
    if type(n_draws) is not int or n_draws < 1:
        raise ValueError("n_draws must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    resolved = _resolve_targets(calibration, data, result, targets, draws=True)
    validate_transforms(data.schema, transforms)
    index, columns = data.raw.index, data.raw.columns
    donor, structural = data.mask("donor"), data.mask("structural")
    base = data.available
    frames = [base.copy(deep=True) for _ in range(n_draws)]
    origin = origins(donor)
    status = origin.copy(deep=True).mask(structural, "structural")
    reasons = pd.DataFrame(None, index=index, columns=columns, dtype=object)
    point = result.completed
    supported, blocks = [], []
    for record, first, layout, stratum, entry, reason in resolved:
        blocks.append({"case_id": record.case_id, "record": record.to_dict(),
                       "status": reason or "sampled",
                       "stratum": stratum.to_dict(),
                       "bank_blocks": None if entry is None else entry.n_blocks})
        for offset, name in layout:
            stamp = index[first + offset]
            if reason is None:
                origin.at[stamp, name] = "reconstructed"
                status.at[stamp, name] = "sampled_block_residual"
            else:
                reasons.at[stamp, name] = reason
        if reason is None:
            supported.append((record, first, layout, entry))
    rng = np.random.default_rng(seed)
    samples = []
    for number, frame in enumerate(frames):
        for record, first, layout, entry in supported:
            source = int(rng.integers(entry.n_blocks))
            residual = entry.residuals[source]
            for (offset, name), value in zip(layout, residual):
                stamp = index[first + offset]
                frame.at[stamp, name] = float(point.at[stamp, name]) + value
            samples.append((f"draw-{number:04d}", record.case_id, entry.block_ids[source],
                            entry.physical_blocks[source]))
    for name in primitive_names(data.schema):
        for stamp in index:
            if pd.isna(status.at[stamp, name]) and pd.isna(reasons.at[stamp, name]):
                reasons.at[stamp, name] = "not_in_calibrated_block"
    lineage, unavailable = propagate(data.schema, frames, origin, structural, transforms)
    for record in lineage:
        status.at[pd.Timestamp(record["timestamp"]), record["output"]] = "derived_in_draw"
    for (stamp, name), reason in unavailable.items():
        reasons.at[stamp, name] = reason
    status = status.where(status.notna(), "unavailable")
    return CompletedDraws(frames=frames, status=status, reasons=reasons, samples=samples,
                          lineage=lineage, blocks=blocks, seed=seed, calibration=calibration,
                          recipe=result.plan.recipe, data=data)


def draw_flux_summary(draws: CompletedDraws, target: str) -> dict:
    """Signed interval-mean NEE totals per draw over one identical temporal support."""
    if not isinstance(draws, CompletedDraws):
        raise ValueError("draw flux summaries require CompletedDraws")
    supported = draws.supported[target]
    rows = []
    for draw_id in draws.draw_ids:
        values = draws.draw(draw_id)[target].astype(float)
        rows.append((draw_id, signed_integral(values, supported, draws.schema, target)))
    first = rows[0][1]
    totals = [item["signed_umol_co2_m-2"] for _, item in rows]
    return {"target": target, "unit": "umol CO2 m-2 s-1 interval mean",
            "cadence_seconds": draws.schema.cadence.total_seconds(),
            "covered_seconds": first["covered_seconds"], "total_seconds": first["total_seconds"],
            "complete_period": first["complete_period"],
            "support": "identical across draws" if first["complete_period"] else
                       "partial: unsupported slots are excluded from every draw total",
            "draws": [{"draw_id": draw_id, "signed_umol_co2_m-2": item["signed_umol_co2_m-2"]}
                      for draw_id, item in rows],
            "draw_min": min(totals), "draw_max": max(totals),
            "calibration_identity": draws.calibration_identity, "algorithm": draws.algorithm,
            "interpretation": "empirical range of draw totals; not a calibrated flux interval"}


def _portable(value):
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _cell(value) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not math.isfinite(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows([_cell(value) for value in row] for row in rows)


def export_uncertainty(directory, *, calibration, intervals=None, draws=None, flux=()) -> Path:
    """Write strict versioned JSON plus aligned plain CSV tables; refuse to overwrite."""
    if not isinstance(calibration, BlockCalibration):
        raise ValueError("export requires the frozen BlockCalibration")
    for item in (intervals, draws):
        if item is not None and item.calibration_identity != calibration.identity:
            raise ValueError("intervals/draws were produced by a different calibration")
    if not isinstance(flux, tuple) or any(item["calibration_identity"] != calibration.identity
                                          for item in flux):
        raise ValueError("flux summaries must be a tuple from this calibration")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    files = []
    if intervals is not None:
        units = dict(intervals.units)
        point, lower, upper, status = intervals.point, intervals.lower, intervals.upper, intervals.status
        _write(destination / "intervals.csv",
               ["timestamp", "variable", "unit", "point", "lower", "upper", "status"],
               ((stamp.isoformat(), name, units[name], point.at[stamp, name], lower.at[stamp, name],
                 upper.at[stamp, name], status.at[stamp, name])
                for stamp in point.index for name in point.columns))
        files.append("intervals.csv")
    if draws is not None:
        units = {item.name: item.unit for item in draws.schema.variables}
        status, reasons, supported = draws.status, draws.reasons, draws.supported
        observed, structural = draws.mask("donor"), draws.mask("structural")
        _write(destination / "draw_cells.csv",
               ["timestamp", "variable", "unit", "status", "reason", "observed", "structural",
                "supported"],
               ((stamp.isoformat(), name, units[name], status.at[stamp, name],
                 reasons.at[stamp, name], observed.at[stamp, name], structural.at[stamp, name],
                 supported.at[stamp, name]) for stamp in status.index for name in status.columns))
        _write(destination / "draws.csv", ["draw_id", "timestamp", "variable", "value"],
               ((draw_id, stamp.isoformat(), name, frame.at[stamp, name])
                for draw_id in draws.draw_ids for frame in (draws.draw(draw_id),)
                for stamp in frame.index for name in frame.columns))
        _write(destination / "draw_samples.csv",
               ["draw_id", "target_block", "source_block", "source_physical_block"], draws.samples)
        files += ["draw_cells.csv", "draws.csv", "draw_samples.csv"]
    if flux:
        _write(destination / "flux.csv",
               ["target", "draw_id", "signed_umol_co2_m-2", "covered_seconds", "total_seconds",
                "complete_period"],
               ((item["target"], row["draw_id"], row["signed_umol_co2_m-2"], item["covered_seconds"],
                 item["total_seconds"], item["complete_period"]) for item in flux for row in item["draws"]))
        files.append("flux.csv")
    schema = calibration.schema
    metadata = {
        "format_version": 1, "kind": "ecofill_block_uncertainty",
        "schema": {"cadence_seconds": schema.cadence.total_seconds(),
                   "variables": [{"name": item.name, "unit": item.unit, "role": item.role,
                                  "derived_from": list(item.derived_from)}
                                 for item in schema.variables]},
        "calibration": calibration.to_dict(),
        "intervals": None if intervals is None else intervals.to_dict(),
        "draws": None if draws is None else draws.to_dict(),
        "flux": list(flux), "files": files + ["uncertainty.json"],
        "missing_values": "empty CSV field", "assumptions": ASSUMPTIONS,
    }
    (destination / "uncertainty.json").write_text(
        json.dumps(_portable(metadata), indent=2, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8")
    return destination
