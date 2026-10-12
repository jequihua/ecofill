"""Editable EC comparison primitives; callers own QC, clock and unit decisions."""

import pandas as pd

from .data import SeriesData
from .mds import MdsPolicy, execute_mds, make_mds_plan
from .methods import execute
from .planning import make_plan


def vpd_kpa_to_hpa(values):
    """Explicit conversion for a caller-declared kPa VPD series."""
    return values * 10.0


def compare_nee(data: SeriesData, *, target: str, sw_in: str, ta: str, vpd: str,
                max_gap: pd.Timedelta, cadence: pd.Timedelta = pd.Timedelta("30min")):
    """Compare offline MDS and linear fills on the same QC-qualified SeriesData.

    `cadence` declares the existing grid for MDS and must equal the schema cadence.
    """
    units = {item.name: item.unit for item in data.schema.variables}
    if units.get(target) != "umol CO2 m-2 s-1":
        raise ValueError("NEE must be an interval-mean umol CO2 m-2 s-1 rate")
    mds = execute_mds(data, make_mds_plan(data, MdsPolicy(
        (target,), sw_in, ta, vpd, max_gap, cadence=cadence)))
    linear = execute(data, make_plan(data, method="linear", mode="offline", max_gap=max_gap))
    return {"mds_hesseflux": mds, "linear": linear}


def signed_supported_integral(result, target: str):
    """Positive NEE means emission; return supported sum and duration coverage."""
    supported = result.mask("observed")[target] & ~result.mask("artificial")[target]
    supported |= result.mask("imputed")[target]
    return signed_integral(result.completed[target], supported, result.schema, target)


def signed_integral(values, supported, schema, target: str):
    """Sum supported interval-mean NEE rates times the actual schema cadence."""
    units = {item.name: item.unit for item in schema.variables}
    if units.get(target) != "umol CO2 m-2 s-1":
        raise ValueError("NEE rate unit must be umol CO2 m-2 s-1")
    duration = schema.cadence.total_seconds()
    if duration <= 0:
        raise ValueError("interval duration must be positive")
    if values[supported].isna().any():
        raise ValueError("supported NEE contains missing values")
    return {"signed_umol_co2_m-2": float(values[supported].sum() * duration),
            "covered_seconds": int(supported.sum() * duration),
            "total_seconds": int(len(values) * duration),
            "complete_period": bool(supported.all())}
