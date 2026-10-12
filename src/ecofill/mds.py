"""Guarded offline wrapper for the pinned MIT hesseflux MDS implementation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._hesseflux_gapfill import gapfill as _reference_gapfill
from .data import SeriesData
from .gaps import profile_gaps
from .result import ImputationResult


REVISION = "53d1f9a75ba388d400e0cff65530f0da33720d8f"
SOURCE_SHA256 = "0a2d94866e4e3d1c07d19c3e7f919cd9e42da0af0d3b36f5f2ede4ee29dda890"
LICENSE_SHA256 = "ba984df6430fe30199f8981daa1639ad3b84e6efecd298f96b61ea756f33b375"


@dataclass(frozen=True)
class MdsPolicy:
    targets: tuple[str, ...]
    sw_in: str
    ta: str
    vpd: str
    max_gap: pd.Timedelta
    mode: str = "offline"
    sw_dev: float = 50.0
    ta_dev: float = 2.5
    vpd_dev: float = 5.0
    longgap_days: int = 60
    fullday: bool = False
    err: bool = False
    min_samples: int = 2
    sentinel: float = -9999.0
    cadence: pd.Timedelta = pd.Timedelta("30min")


@dataclass(frozen=True)
class MdsGap:
    variable: str
    start_pos: int
    stop_pos: int
    eligible: bool
    reason: str | None


@dataclass(frozen=True)
class MdsPlan:
    data_fingerprint: str
    policy: MdsPolicy
    gaps: tuple[MdsGap, ...]

    def to_dict(self):
        policy = self.policy
        return {
            "data_fingerprint": self.data_fingerprint,
            "method": "mds_hesseflux", "mode": policy.mode,
            "targets": list(policy.targets),
            "drivers": {"SW_IN": policy.sw_in, "TA": policy.ta, "VPD": policy.vpd},
            "max_gap_seconds": policy.max_gap.total_seconds(),
            "parameters": {"sw_dev": policy.sw_dev, "ta_dev": policy.ta_dev,
                           "vpd_dev": policy.vpd_dev, "longgap_days": policy.longgap_days,
                           "fullday": policy.fullday, "err": policy.err,
                           "min_samples": policy.min_samples, "sentinel": policy.sentinel,
                           "cadence_seconds": policy.cadence.total_seconds()},
            "gaps": [{"variable": item.variable, "start_pos": item.start_pos,
                      "stop_pos": item.stop_pos, "eligible": item.eligible,
                      "reason": item.reason} for item in self.gaps],
        }


def _descends(schema, name, target):
    return target in schema[name].derived_from or any(
        _descends(schema, parent, target) for parent in schema[name].derived_from)


def _validate(data, policy):
    if not isinstance(data, SeriesData) or not isinstance(policy, MdsPolicy):
        raise ValueError("MDS requires SeriesData and MdsPolicy")
    if policy.mode != "offline":
        raise ValueError("mds_hesseflux is offline only")
    # The declared cadence must make the reference's half-day, day and week windows
    # integral sample counts; it declares the existing grid and never resamples.
    cadence = policy.cadence
    if (not isinstance(cadence, pd.Timedelta) or pd.isna(cadence) or cadence <= pd.Timedelta(0)
            or cadence % pd.Timedelta("1min") or pd.Timedelta("12h") % cadence):
        raise ValueError("cadence must be a positive whole-minute Timedelta dividing 12 hours")
    index = data.raw.index
    if data.schema.cadence != cadence or len(index) < 2 or ((index[1:] - index[:-1]) != cadence).any():
        raise ValueError("mds_hesseflux requires at least two regular rows at the declared cadence")
    if len({stamp.utcoffset() for stamp in index}) != 1:
        raise ValueError("mds_hesseflux requires a fixed-offset clock without DST changes")
    names = set(data.raw)
    if (not isinstance(policy.targets, tuple) or not policy.targets
            or len(set(policy.targets)) != len(policy.targets)
            or any(name not in names for name in policy.targets)):
        raise ValueError("targets must be an ordered nonempty tuple of distinct channels")
    drivers = (policy.sw_in, policy.ta, policy.vpd)
    if len(set(drivers)) != 3 or any(name not in names for name in drivers):
        raise ValueError("SW_IN, TA and VPD require distinct mapped channels")
    if set(drivers) & set(policy.targets):
        raise ValueError("targets cannot be weather drivers")
    schema = {item.name: item for item in data.schema.variables}
    for target in policy.targets:
        if schema[target].unit == "unknown":
            raise ValueError("target flux unit must be declared")
        if any(_descends(schema, driver, target) for driver in drivers):
            raise ValueError("target-derived weather driver or alias is forbidden")
    for name, unit in zip(drivers, ("W m-2", "degC", "hPa")):
        if schema[name].unit != unit:
            raise ValueError(f"{name}: MDS requires explicit {unit} units")
    if not isinstance(policy.max_gap, pd.Timedelta) or pd.isna(policy.max_gap) or policy.max_gap <= pd.Timedelta(0):
        raise ValueError("max_gap must be an explicit positive Timedelta")
    if (policy.sw_dev, policy.ta_dev, policy.vpd_dev, policy.longgap_days,
            policy.fullday, policy.err, policy.min_samples, policy.sentinel) != (
            50.0, 2.5, 5.0, 60, False, False, 2, -9999.0):
        raise ValueError("mds_hesseflux parameters must equal the pinned reference settings")
    raw, observed = data.raw, data.mask("observed")
    for name in (*drivers, *policy.targets):
        if (raw.loc[observed[name], name] == policy.sentinel).any():
            raise ValueError("observed value collides with the MDS missing sentinel")


def make_mds_plan(data: SeriesData, policy: MdsPolicy) -> MdsPlan:
    """Freeze target gap eligibility before any reference call."""
    _validate(data, policy)
    structural = data.mask("structural")
    donors = data.mask("donor")
    gaps = []
    for gap in profile_gaps(data):
        if gap.variable not in policy.targets:
            continue
        reason = ("structural_absence" if structural[gap.variable].iloc[gap.start_pos:gap.stop_pos].any()
                  else "gap_exceeds_limit" if gap.duration > policy.max_gap
                  else "insufficient_observed_support" if int(donors[gap.variable].sum()) < policy.min_samples
                  else None)
        gaps.append(MdsGap(gap.variable, gap.start_pos, gap.stop_pos, reason is None, reason))
    return MdsPlan(data.fingerprint, policy, tuple(gaps))


def execute_mds(data: SeriesData, plan: MdsPlan) -> ImputationResult:
    """Call the exact pinned source on owned sentinel-masked arrays; never chain fills."""
    if not isinstance(plan, MdsPlan):
        raise ValueError("MDS plan is stale or invalid; rebuild with make_mds_plan")
    try:
        expected = make_mds_plan(data, plan.policy)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("MDS plan is stale or invalid; rebuild with make_mds_plan") from exc
    if plan != expected:
        raise ValueError("MDS plan was edited; rebuild with make_mds_plan")
    available = data.available
    completed = available.copy(deep=True)
    index, columns = completed.index, completed.columns
    states = pd.DataFrame("unavailable", index=index, columns=columns, dtype=object)
    for mask, label in (("raw_present", "unqualified"), ("absent", "absent"),
                        ("inserted", "absent_timestamp"), ("structural", "structural"),
                        ("qc_rejected", "qc_rejected"), ("observed", "observed"),
                        ("artificial", "artificial_hidden")):
        states = states.mask(data.mask(mask), label)
    methods = pd.DataFrame(None, index=index, columns=columns, dtype=object)
    reasons = pd.DataFrame(None, index=index, columns=columns, dtype=object)
    imputed = pd.DataFrame(False, index=index, columns=columns)
    quality = pd.DataFrame(None, index=index, columns=columns, dtype=object)
    donor = data.mask("donor")
    cells = []
    mapping = {"SW_IN": plan.policy.sw_in, "TA": plan.policy.ta, "VPD": plan.policy.vpd}
    for target in plan.policy.targets:
        selected = [item for item in plan.gaps if item.variable == target]
        if not selected:
            continue
        for item in selected:
            if not item.eligible:
                reasons.loc[index[item.start_pos:item.stop_pos], target] = item.reason
        if not any(item.eligible for item in selected):
            continue
        source_names = [mapping["SW_IN"], mapping["TA"], mapping["VPD"], target]
        input_frame = pd.DataFrame(index=index.tz_localize(None))
        flags = pd.DataFrame(index=input_frame.index)
        for canonical, source in zip(("SW_IN", "TA", "VPD", "FLUX"), source_names):
            values = available[source].to_numpy(dtype=float, na_value=np.nan, copy=True)
            valid = donor[source].to_numpy(dtype=bool, copy=True)
            input_frame[canonical] = np.where(valid, values, plan.policy.sentinel)
            flags[canonical] = np.where(valid, 0, 1)
        try:
            filled, tiers = _reference_gapfill(input_frame, flag=flags,
                sw_dev=plan.policy.sw_dev, ta_dev=plan.policy.ta_dev,
                vpd_dev=plan.policy.vpd_dev, longgap=plan.policy.longgap_days,
                fullday=plan.policy.fullday, undef=plan.policy.sentinel,
                err=plan.policy.err, verbose=0)
        except Exception as exc:
            raise RuntimeError(f"mds_hesseflux reference failed for {target}") from exc
        if (not isinstance(filled, pd.DataFrame) or not isinstance(tiers, pd.DataFrame)
                or not filled.index.equals(input_frame.index) or not tiers.index.equals(input_frame.index)
                or not filled.columns.equals(input_frame.columns)
                or not tiers.columns.equals(input_frame.columns)):
            raise RuntimeError("mds_hesseflux returned malformed aligned tables")
        values = filled["FLUX"].to_numpy()
        flags_out = tiers["FLUX"].to_numpy()
        if values.dtype.kind not in "iuf" or flags_out.dtype.kind not in "iuf":
            raise RuntimeError("mds_hesseflux returned non-real values or quality")
        for item in selected:
            if not item.eligible:
                continue
            for position in range(item.start_pos, item.stop_pos):
                value, tier = values[position], flags_out[position]
                stamp = index[position]
                if not np.isfinite(value) or not np.isfinite(tier) or tier not in (0, 1, 2, 3):
                    raise RuntimeError("mds_hesseflux returned nonfinite or malformed output")
                if tier == 0:
                    if value != plan.policy.sentinel:
                        raise RuntimeError("mds_hesseflux unresolved value/quality disagree")
                    reasons.at[stamp, target] = "reference_unresolved"
                    cells.append({"timestamp": stamp.isoformat(), "target": target,
                                  "quality": None, "reason": "reference_unresolved"})
                    continue
                if value == plan.policy.sentinel:
                    raise RuntimeError("mds_hesseflux filled quality has unresolved sentinel")
                completed.at[stamp, target] = float(value)
                states.at[stamp, target] = "imputed"
                methods.at[stamp, target] = "mds_hesseflux"
                imputed.at[stamp, target] = True
                quality.at[stamp, target] = int(tier)
                cells.append({"timestamp": stamp.isoformat(), "target": target,
                              "quality": int(tier), "reason": None})
    identity = {"algorithm": "mds_hesseflux", "revision": REVISION,
                "source_sha256": SOURCE_SHA256, "license_sha256": LICENSE_SHA256,
                "reference_license": "MIT", "drivers": mapping,
                "parameters": plan.to_dict()["parameters"],
                "quality_meaning": "reference tier, not calibrated uncertainty",
                "cells": cells}
    return ImputationResult(data, plan, completed, states, methods, reasons, imputed,
                            backend_identity=identity, quality=quality)
