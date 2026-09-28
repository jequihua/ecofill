"""Observed-only masking and scoring for bounded baseline comparisons."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import SeriesData
from .methods import execute
from .planning import make_plan


@dataclass(frozen=True)
class MaskCase:
    scenario: str
    cells: tuple[tuple[int, str], ...]


def check_split_families(training: tuple[str, ...], validation: tuple[str, ...]) -> None:
    """A physical source family cannot supply both training and validation truth."""
    if set(training) & set(validation):
        raise ValueError("duplicate source family crosses training and validation")


def training_scales(data: SeriesData, split_position: int) -> dict[str, float]:
    if not 0 < split_position < len(data.raw):
        raise ValueError("split_position must separate nonempty training and validation")
    available = data.available.iloc[:split_position]
    scales = {}
    for name in available:
        values = available[name].dropna().to_numpy(dtype=float)
        scales[name] = float(np.std(values)) if len(values) else 0.0
    return scales


def make_mask_cases(data: SeriesData, split_position: int, *, seed: int = 0) -> tuple[MaskCase, ...]:
    """Freeze a small declared geometry matrix on QC-passed validation cells."""
    if not 0 < split_position < len(data.raw):
        raise ValueError("split_position must separate nonempty training and validation")
    donor = data.mask("donor")
    names = tuple(data.raw.columns)
    rng = np.random.default_rng(seed)
    cases = []
    random_cells = []
    for name in names:
        positions = np.flatnonzero(donor[name].to_numpy()[split_position:]) + split_position
        random_cells.extend((int(pos), name) for pos in rng.choice(
            positions, size=min(16, len(positions)), replace=False))
    cases.append(MaskCase("random_single", tuple(sorted(random_cells))))

    def block(name, length, offset):
        start = split_position + offset
        if start < split_position or start + length > len(donor):
            return ()
        if not donor[name].iloc[start:start + length].all():
            return ()
        return tuple((pos, name) for pos in range(start, start + length))

    width = len(donor) - split_position
    for length in (1, 4, 24, 96):
        for name in names:
            cells = block(name, length, max(0, (width - length) // 2))
            if cells:
                cases.append(MaskCase(f"block_{length}_{name}", cells))
    for name in names:
        for label, offset in (("leading", 0), ("trailing", width - 4)):
            cells = block(name, 4, offset)
            if cells:
                cases.append(MaskCase(f"{label}_4_{name}", cells))
        cells = block(name, width, 0)
        if cells:
            cases.append(MaskCase(f"full_variable_{name}", cells))
    if len(names) >= 2:
        first, second = names[:2]
        offset = max(0, (width - 6) // 2)
        shared = block(first, 4, offset) + block(second, 4, offset)
        staggered = block(first, 4, offset) + block(second, 4, offset + 2)
        if len(shared) == 8:
            cases.append(MaskCase("shared_4", shared))
        if len(staggered) == 8:
            cases.append(MaskCase("staggered_4", staggered))
    return tuple(cases)


def run_case(data: SeriesData, case: MaskCase, *, split_position: int,
             method: str, mode: str, max_gap: pd.Timedelta,
             period: pd.Timedelta | None = None, search_periods: int | None = None,
             state=None) -> list[dict]:
    """Score only freshly hidden QC-passed truth; return counts even on failure."""
    raw = data.raw
    donor = data.mask("donor")
    hidden = pd.DataFrame(False, index=raw.index, columns=raw.columns)
    for pos, name in case.cells:
        if name not in raw or type(pos) is not int or not split_position <= pos < len(raw):
            raise ValueError("mask cell is outside the declared validation support")
        if not donor.at[raw.index[pos], name]:
            raise ValueError("mask cell lacks QC-passed observed truth")
        hidden.at[raw.index[pos], name] = True
    fold = SeriesData(raw, schema=data.schema, observed=data.mask("observed"),
                      qc_rejected=data.mask("qc_rejected"), structural=data.mask("structural"),
                      artificial=hidden, inserted=data.mask("inserted"))
    scales = training_scales(data, split_position)
    # Policy errors reject the request before scoring. Only failures while
    # executing an otherwise valid plan count against hidden cells.
    plan = make_plan(fold, method=method, mode=mode, max_gap=max_gap,
                     period=period, search_periods=search_periods, state=state)
    try:
        result = execute(fold, plan)
        failure = None
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        result = None
        failure = f"{type(exc).__name__}: {exc}"
    records = []
    for name in raw:
        selected = hidden[name]
        count = int(selected.sum())
        if not count:
            continue
        if result is None:
            predicted = 0
            errors = np.array([], dtype=float)
        else:
            filled = result.mask("imputed")[name] & selected
            predicted = int(filled.sum())
            errors = (result.completed.loc[filled, name].to_numpy(dtype=float)
                      - raw.loc[filled, name].to_numpy(dtype=float))
        scale = scales[name]
        records.append({
            "scenario": case.scenario, "variable": name, "method": method, "mode": mode,
            "hidden": count, "predicted": predicted,
            "abstained": 0 if failure else count - predicted,
            "failed": count if failure else 0,
            "mae": float(np.mean(np.abs(errors))) if predicted else None,
            "rmse": float(np.sqrt(np.mean(errors ** 2))) if predicted else None,
            "bias": float(np.mean(errors)) if predicted else None,
            "normalized_mae": float(np.mean(np.abs(errors)) / scale)
            if predicted and scale > 0 else None,
            "scale": scale, "failure": failure,
        })
    return records
