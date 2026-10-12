"""Explicit fixed-weight blends of aligned candidate results; no weight search."""

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import pandas as pd

from .data import SeriesData
from .deep import DeepPlan
from .drivers import DerivedStep, RegressionStep, check_derived_step, evaluate_derived
from .forecast import ForecastPlan
from .gaps import profile_gaps
from .result import ImputationResult
from .time_indexed import TimeIndexedPlan

WEIGHT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class EnsembleMember:
    """One named candidate: plan method label, `configuration_identity` digest and weight."""

    name: str
    method: str
    configuration: str
    weight: float


@dataclass(frozen=True)
class EnsembleRecipe:
    """Immutable fixed recipe; weights are finite, positive and already sum to one."""

    members: tuple[EnsembleMember, ...]
    mode: str
    training: tuple[int, int] | None = None
    whole_gaps: bool = False

    def __post_init__(self):
        if type(self.whole_gaps) is not bool:
            raise ValueError("whole_gaps must be boolean")
        if not isinstance(self.members, tuple) or not self.members:
            raise ValueError("recipe requires a nonempty tuple of EnsembleMember")
        for member in self.members:
            if not isinstance(member, EnsembleMember):
                raise ValueError("recipe members must be EnsembleMember")
            for field in (member.name, member.method, member.configuration):
                if not isinstance(field, str) or not field or field != field.strip():
                    raise ValueError("member name, method and configuration must be nonempty strings")
            weight = member.weight
            if isinstance(weight, bool) or type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
                raise ValueError("member weights must be finite, strictly positive numbers")
        names = [member.name for member in self.members]
        if len(set(names)) != len(names):
            raise ValueError("duplicate member identity")
        total = math.fsum(float(member.weight) for member in self.members)
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"weights sum to {total!r}; supply normalized weights (no silent renormalization)")
        if self.mode not in {"offline", "causal"}:
            raise ValueError("recipe mode must be 'offline' or 'causal'")
        if self.training is not None and (
                not isinstance(self.training, tuple) or len(self.training) != 2
                or any(type(value) is not int for value in self.training)
                or not 0 <= self.training[0] < self.training[1]):
            raise ValueError("training split must be None or an increasing (start, stop) position pair")

    def to_dict(self) -> dict:
        return {"mode": self.mode,
                **({"whole_gaps": True} if self.whole_gaps else {}),
                "training": None if self.training is None else list(self.training),
                "members": [{"name": item.name, "method": item.method,
                             "configuration": item.configuration, "weight": float(item.weight)}
                            for item in sorted(self.members, key=lambda item: item.name)]}

    @property
    def identity(self) -> str:
        """Reusable recipe identity, independent of member order and of any data instance."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def transform_dict(step: DerivedStep) -> dict:
    return {"output": step.output, "parents": list(step.parents),
            "parent_units": list(step.parent_units), "output_unit": step.output_unit,
            "formula_id": step.formula_id, "allow_reconstructed": list(step.allow_reconstructed)}


@dataclass(frozen=True)
class EnsemblePlan:
    data_fingerprint: str
    recipe: EnsembleRecipe
    transforms: tuple[DerivedStep, ...]

    def to_dict(self) -> dict:
        return {"data_fingerprint": self.data_fingerprint, "method": "ensemble",
                "mode": self.recipe.mode, "recipe": self.recipe.to_dict(),
                "recipe_identity": self.recipe.identity,
                "transforms": [transform_dict(step) for step in self.transforms]}


def primitive_names(schema) -> tuple[str, ...]:
    return tuple(item.name for item in schema.variables if not item.derived_from)


def validate_transforms(schema, transforms) -> None:
    """Frozen DerivedSteps only, exact lineage/units, one step per output, topological order."""
    if not isinstance(transforms, tuple):
        raise ValueError("transforms must be a tuple of DerivedStep")
    variables = {item.name: item for item in schema.variables}
    produced = set()
    for position, step in enumerate(transforms):
        if isinstance(step, RegressionStep):
            raise ValueError("learned response refitting is unsupported; declare a frozen DerivedStep")
        if not isinstance(step, DerivedStep):
            raise ValueError("transforms must be DerivedStep records")
        if not isinstance(step.parents, tuple) or not step.parents:
            raise ValueError("derived transform requires a tuple of parents")
        check_derived_step(step, variables, lambda name: variables[name].unit if name in variables else None)
        if tuple(variables[step.output].derived_from) != step.parents:
            raise ValueError("derived step disagrees with declared lineage")
        if step.output in produced:
            raise ValueError(f"duplicate transform output {step.output}")
        if not isinstance(step.allow_reconstructed, tuple) or not set(step.allow_reconstructed) <= set(step.parents):
            raise ValueError("reconstructed-parent permission must name declared parents")
        later = {other.output for other in transforms[position:]}
        if any(parent in later for parent in step.parents):
            raise ValueError("transforms must be acyclic and listed in dependency order")
        produced.add(step.output)


def origins(donor: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.where(donor.to_numpy(), "observed", None).astype(object),
                        index=donor.index, columns=donor.columns, dtype=object)


def propagate(schema, frames, origin, structural, transforms):
    """Recompute missing derived cells in every aligned frame from same-frame parents.

    `origin` holds "observed", "reconstructed" or None per cell and is updated in
    place. Supplied observed children never change. Returns (lineage records,
    unavailable reasons by (timestamp, variable)).
    """
    index = origin.index
    records, reasons = [], {}
    steps = {step.output: step for step in transforms}
    for step in transforms:
        for stamp in index:
            if origin.at[stamp, step.output] is not None or structural.at[stamp, step.output]:
                continue
            parents = [origin.at[stamp, name] for name in step.parents]
            if any(item is None for item in parents):
                reasons[(stamp, step.output)] = "required_parent_unavailable"
                continue
            if any(item == "reconstructed" and name not in step.allow_reconstructed
                   for name, item in zip(step.parents, parents)):
                reasons[(stamp, step.output)] = "reconstructed_parent_not_permitted"
                continue
            arrays = [np.array([float(frame.at[stamp, name]) for frame in frames]) for name in step.parents]
            values = evaluate_derived(step, arrays)
            for frame, value in zip(frames, values):
                frame.at[stamp, step.output] = float(value)
            origin.at[stamp, step.output] = "reconstructed"
            records.append({"timestamp": stamp.isoformat(), "output": step.output,
                            "formula_id": step.formula_id,
                            "donor_origin": dict(zip(step.parents, parents))})
    for item in schema.variables:
        if not item.derived_from or item.name in steps:
            continue
        for stamp in index:
            if origin.at[stamp, item.name] is None and not structural.at[stamp, item.name]:
                reasons[(stamp, item.name)] = "no_declared_transform"
    return records, reasons


# Data-instance keys: gap tables, fingerprints, per-cell and per-call execution records
# change with the input.
DATA_KEYS = {"data_fingerprint", "external_fingerprint", "gaps", "events", "cells",
             "derived_cells", "abstained_primitive_cells", "execution"}

# Plans whose predictions come from a caller-supplied backend: only the backend's own
# declared identity distinguishes two such predictors under one plan policy.
BACKEND_PLANS = (ForecastPlan, TimeIndexedPlan)


# Fitted-model fields of per-cell fit records; timestamps, donors and novelty stay out.
FIT_KEYS = ("target", "method", "predictors", "training_interval", "training_overlap",
            "training_mean", "training_std", "training_min", "training_max", "parameters",
            "fit_rows_identity")
EXPOSED_FITS = {"ols"}


def _fitted(backend) -> list:
    """Unique fitted-model identities recorded per cell (e.g. OLS coefficients and scaling)."""
    fits = {}
    for cell in backend.get("cells") or ():
        if not isinstance(cell, dict) or "parameters" not in cell:
            continue
        if cell.get("method") not in EXPOSED_FITS:
            parameters = cell["parameters"]
            digest = parameters.get("fitted_model_sha256") if isinstance(parameters, dict) else None
            if (cell.get("method") not in {"rf", "boosting"} or not isinstance(digest, str)
                    or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError(f"fitted identity of {cell.get('method')!r} is not exposed; "
                                 "it cannot be bound to a frozen ensemble recipe")
        fit = {key: cell.get(key) for key in FIT_KEYS}
        fits[json.dumps(fit, sort_keys=True, default=str)] = fit
    return [fits[key] for key in sorted(fits)]


def member_configuration(result: ImputationResult) -> dict:
    """Model/configuration metadata of a result, excluding data-instance fields.

    Exposed fitted parameters and training normalization are part of the identity;
    per-target timestamps and data fingerprints are not. A forecast or time-indexed
    member without a nonempty declared backend identity is rejected; no identity is
    inferred from its class name or plan policy.
    """
    if not isinstance(result, ImputationResult):
        raise ValueError("configuration requires an ImputationResult")
    provenance = result.provenance()
    plan = {key: value for key, value in provenance["plan"].items() if key not in DATA_KEYS}
    backend = provenance.get("backend")
    if isinstance(backend, dict):
        fitted = _fitted(backend)
        backend = {key: value for key, value in backend.items() if key not in DATA_KEYS}
        if fitted:
            backend["fitted"] = fitted
    if isinstance(result.plan, BACKEND_PLANS) and not (isinstance(backend, dict) and backend):
        raise ValueError(f"{type(result.plan).__name__} member has no declared backend identity; "
                         "a nonempty backend.identity mapping is required to bind it to a recipe")
    return json.loads(json.dumps({"plan": plan, "backend": backend}, sort_keys=True, default=str))


def configuration_identity(result: ImputationResult) -> str:
    """Stable digest binding a recipe member to its actual plan policy and backend identity."""
    return hashlib.sha256(json.dumps(member_configuration(result), sort_keys=True).encode()).hexdigest()


def plan_method_mode_training(plan):
    info = plan.to_dict()
    policy = info.get("policy") if isinstance(info.get("policy"), dict) else {}
    method = info.get("method", policy.get("method"))
    mode = info.get("mode", "offline" if isinstance(plan, DeepPlan) else None)
    training = info.get("training_interval", policy.get("training"))
    if isinstance(plan, EnsemblePlan):
        training = plan.recipe.training
    return method, mode, None if training is None else tuple(int(value) for value in training)


def check_result(data: SeriesData, result, *, label="result") -> None:
    """Reject a result not derived from exactly this SeriesData (values, masks, schema, grid)."""
    if not isinstance(result, ImputationResult):
        raise ValueError(f"{label} must be an ImputationResult")
    if result.plan.data_fingerprint != data.fingerprint or result.schema != data.schema:
        raise ValueError(f"{label} was built for different data, masks or schema")
    if not result.raw.equals(data.raw) or not result.completed.index.equals(data.raw.index) \
            or not result.completed.columns.equals(data.raw.columns):
        raise ValueError(f"{label} is misaligned with the input grid or raw values")
    for name, mask in data.masks.items():
        if not result.mask(name).equals(mask):
            raise ValueError(f"{label} {name} mask differs from the input")
    imputed = result.mask("imputed")
    donor, structural = data.mask("donor"), data.mask("structural")
    if (imputed & (donor | structural)).to_numpy().any():
        raise ValueError(f"{label} marks an observed or structural cell as imputed")
    completed, available = result.completed, data.available
    for name in completed:
        rows = donor[name]
        if completed.loc[rows, name].tolist() != available.loc[rows, name].tolist():
            raise ValueError(f"{label} edited observed values of {name}")
        filled = completed.loc[imputed[name], name].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(filled).all():
            raise ValueError(f"{label} has a supported nonfinite prediction for {name}")
        if completed.loc[~rows & ~imputed[name], name].notna().any():
            raise ValueError(f"{label} has unmarked values in missing cells of {name}")


def blend(data: SeriesData, members: dict, recipe: EnsembleRecipe, *,
          transforms: tuple = ()) -> ImputationResult:
    """Blend common supported primitive predictions with fixed recipe weights.

    A missing primitive cell is blended only when every member predicted it;
    otherwise it abstains and names the unsupported members. Derived cells are
    recomputed from blended parents through declared frozen transforms or stay
    unavailable. Observations, structural cells and masks are preserved.
    """
    if not isinstance(data, SeriesData) or not isinstance(recipe, EnsembleRecipe):
        raise ValueError("blend requires SeriesData and an EnsembleRecipe")
    if not isinstance(members, dict) or set(members) != {item.name for item in recipe.members}:
        raise ValueError("member results must match recipe names exactly")
    validate_transforms(data.schema, transforms)
    for item in recipe.members:
        result = members[item.name]
        check_result(data, result, label=f"member {item.name}")
        method, mode, training = plan_method_mode_training(result.plan)
        if method != item.method:
            raise ValueError(f"member {item.name} plan method {method!r} differs from recipe {item.method!r}")
        if mode != recipe.mode:
            raise ValueError(f"member {item.name} information mode {mode!r} differs from recipe")
        if training is not None and training != recipe.training:
            raise ValueError(f"member {item.name} training split {training} differs from recipe")
        if configuration_identity(result) != item.configuration:
            raise ValueError(f"member {item.name} configuration differs from the frozen recipe; "
                             "build a new recipe for changed parameters")
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
    donor, structural = data.mask("donor"), data.mask("structural")
    origin = origins(donor)
    results = {item.name: (item, members[item.name].completed, members[item.name].mask("imputed"),
                           members[item.name].unresolved_reasons) for item in recipe.members}
    label = "ensemble:" + recipe.identity[:16]
    incomplete = set()
    if recipe.whole_gaps:
        for gap in profile_gaps(data):
            if any(not parts[2][gap.variable].iloc[gap.start_pos:gap.stop_pos].all()
                   for parts in results.values()):
                incomplete.update((index[row], gap.variable)
                                  for row in range(gap.start_pos, gap.stop_pos))
    abstained = 0
    for name in primitive_names(data.schema):
        for stamp in index[(~donor[name] & ~structural[name]).to_numpy()]:
            if (stamp, name) in incomplete:
                reasons.at[stamp, name] = "whole_gap_member_incomplete"
                abstained += 1
                continue
            missing = [f"{key}({reason})" if isinstance(reason := parts[3].at[stamp, name], str) else key
                       for key, parts in sorted(results.items()) if not parts[2].at[stamp, name]]
            if missing:
                reasons.at[stamp, name] = "member_unsupported:" + ",".join(missing)
                abstained += 1
                continue
            completed.at[stamp, name] = math.fsum(
                float(item.weight) * float(values.at[stamp, name]) for item, values, _, _ in results.values())
            states.at[stamp, name] = "imputed"
            methods.at[stamp, name] = label
            imputed.at[stamp, name] = True
            origin.at[stamp, name] = "reconstructed"
        reasons.loc[structural[name], name] = "structural_absence"
    lineage, unavailable = propagate(data.schema, [completed], origin, structural, transforms)
    for record in lineage:
        stamp, name = pd.Timestamp(record["timestamp"]), record["output"]
        states.at[stamp, name] = "imputed"
        methods.at[stamp, name] = "derived:" + record["formula_id"]
        imputed.at[stamp, name] = True
    for (stamp, name), reason in unavailable.items():
        reasons.at[stamp, name] = reason
    for item in data.schema.variables:
        if item.derived_from:
            reasons.loc[structural[item.name], item.name] = "structural_absence"
    plan = EnsemblePlan(data.fingerprint, recipe, transforms)
    identity = {"kind": "ensemble", "recipe_identity": recipe.identity, "recipe": recipe.to_dict(),
                "member_configurations": {item.name: member_configuration(members[item.name])
                                          for item in recipe.members},
                "members": {item.name: {"method": item.method, "configuration": item.configuration,
                                        "weight": float(item.weight)} for item in recipe.members},
                "abstained_primitive_cells": abstained, "derived_cells": lineage,
                "note": "fixed weights; no renormalization, fallback or member-interval averaging"}
    return ImputationResult(data, plan, completed, states, methods, reasons, imputed,
                            backend_identity=identity)
