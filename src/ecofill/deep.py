"""Offline BRITS/SAITS adapters around the pinned PyPOTS 1.5 numerical cores."""

from dataclasses import dataclass, field
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd

from .data import SeriesData
from .gaps import profile_gaps
from .result import ImputationResult


@dataclass(frozen=True)
class DeepPolicy:
    method: str
    channels: tuple[str, ...]
    targets: tuple[str, ...]
    training: tuple[int, int]
    evaluation: tuple[int, int]
    window: int
    max_gap: pd.Timedelta
    min_target_support: int
    seed: int
    epochs: int = 2
    batch_size: int = 2
    learning_rate: float = 0.001
    rnn_hidden_size: int = 8
    n_layers: int = 1
    d_model: int = 16
    n_heads: int = 2
    d_k: int = 8
    d_v: int = 8
    d_ffn: int = 32
    dropout: float = 0.0
    attn_dropout: float = 0.0

    def to_dict(self):
        return {key: (value.total_seconds() if isinstance(value, pd.Timedelta) else value)
                for key, value in vars(self).items()}


def _validate(data, policy, mode):
    if not isinstance(data, SeriesData) or not isinstance(policy, DeepPolicy):
        raise ValueError("SeriesData and DeepPolicy are required")
    if mode != "offline":
        raise ValueError("deep imputers require offline mode")
    if policy.method not in {"brits_pypots", "saits_pypots"}:
        raise ValueError("unknown deep method")
    names = tuple(data.raw.columns)
    if not policy.channels or len(set(policy.channels)) != len(policy.channels):
        raise ValueError("channels must be ordered and unique")
    if tuple(name for name in names if name in policy.channels) != policy.channels:
        raise ValueError("channels must follow SeriesData order; reorder explicitly")
    if not policy.targets or len(set(policy.targets)) != len(policy.targets) or not set(policy.targets) <= set(names):
        raise ValueError("targets must be unique selected channels")
    if any(variable.derived_from or variable.role in {"derived", "diagnostic"}
           for variable in data.schema.variables if variable.name in policy.channels):
        raise ValueError("derived descendants and aliases are unsupported joint features")
    for interval in (policy.training, policy.evaluation):
        if (not isinstance(interval, tuple) or len(interval) != 2 or
                any(type(n) is not int for n in interval) or
                not 0 <= interval[0] < interval[1] <= len(data.raw)):
            raise ValueError("intervals must be half-open bounds on the original grid")
    a, b = policy.training
    c, d = policy.evaluation
    if not (b <= c or d <= a):
        raise ValueError("training and evaluation must be disjoint")
    if type(policy.window) is not int or policy.window < 2 or b - a < policy.window or d - c < policy.window:
        raise ValueError("window must fit both intervals")
    if (not isinstance(policy.max_gap, pd.Timedelta) or pd.isna(policy.max_gap)
            or policy.max_gap <= pd.Timedelta(0)):
        raise ValueError("max_gap must be positive")
    if type(policy.min_target_support) is not int or policy.min_target_support < 1:
        raise ValueError("min_target_support must be positive")
    if type(policy.seed) is not int or policy.seed < 0:
        raise ValueError("seed must be nonnegative")
    for value in (policy.epochs, policy.batch_size, policy.rnn_hidden_size,
                  policy.n_layers, policy.d_model, policy.n_heads, policy.d_k,
                  policy.d_v, policy.d_ffn):
        if type(value) is not int or value < 1:
            raise ValueError("model settings must be positive integers")
    if policy.d_model % policy.n_heads:
        raise ValueError("d_model must divide by n_heads")
    if (not math.isfinite(policy.learning_rate) or policy.learning_rate <= 0 or
            any(not math.isfinite(v) or not 0 <= v < 1 for v in
                (policy.dropout, policy.attn_dropout))):
        raise ValueError("invalid optimizer or dropout setting")


def _windows(bounds, size):
    return tuple((start, start + size) for start in range(bounds[0], bounds[1] - size + 1, size))


def _digest(*parts):
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(str(part.shape).encode())
            digest.update(part.tobytes())
        else:
            digest.update(json.dumps(part, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def directional_deltas(mask):
    """Step deltas in each direction; backward deltas are computed after reversal."""
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.dtype.kind not in "biuf":
        raise ValueError("mask must be [windows, steps, channels] numeric")
    def calculate(m):
        delta = np.zeros(m.shape, dtype=np.float32)
        for t in range(1, m.shape[1]):
            delta[:, t] = 1 + (1 - m[:, t - 1]) * delta[:, t - 1]
        return delta
    return calculate(mask), calculate(mask[:, ::-1])


def _arrays(data, windows, channels, mean, scale):
    available = data.available.loc[:, channels].to_numpy(dtype=float, na_value=np.nan)
    donor = data.mask("donor").loc[:, channels].to_numpy(dtype=bool)
    values = np.stack([available[a:b] for a, b in windows])
    masks = np.stack([donor[a:b] for a, b in windows])
    scaled = np.where(masks, (values - mean) / scale, 0).astype(np.float32)
    return scaled, masks


def _inputs(method, values, mask, original=None, hidden=None):
    observed = mask.astype(np.float32)
    inputs = {"X": values.copy(), "missing_mask": observed.copy()}
    if method == "brits_pypots":
        f_delta, b_delta = directional_deltas(observed)
        inputs = {"forward": {"X": values.copy(), "missing_mask": observed.copy(), "deltas": f_delta},
                  "backward": {"X": values[:, ::-1].copy(),
                               "missing_mask": observed[:, ::-1].copy(), "deltas": b_delta}}
    if original is not None:
        inputs["X_ori"] = original.copy()
        inputs["indicating_mask"] = hidden.astype(np.float32).copy()
    return inputs


@dataclass(frozen=True)
class DeepFit:
    policy: DeepPolicy
    units: tuple[str, ...]
    cadence_ns: int
    training_identity: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    constants: tuple[bool, ...]
    mask_hash: str
    training_hide_bytes: bytes = field(repr=False)
    state_hash: str
    backend: object


@dataclass(frozen=True)
class DeepGap:
    variable: str
    start_pos: int
    stop_pos: int
    eligible: bool
    reason: str | None
    window_index: int | None


@dataclass(frozen=True)
class DeepPlan:
    data_fingerprint: str
    fit_identity: str
    policy: DeepPolicy
    windows: tuple[tuple[int, int], ...]
    gaps: tuple[DeepGap, ...]

    def to_dict(self):
        return {"data_fingerprint": self.data_fingerprint, "fit_identity": self.fit_identity,
                "policy": self.policy.to_dict(), "windows": self.windows,
                "gaps": [vars(gap) for gap in self.gaps]}


def _training_identity(data, policy):
    windows = _windows(policy.training, policy.window)
    raw = data.available.loc[:, policy.channels].iloc[policy.training[0]:policy.training[1]]
    mask = data.mask("donor").loc[:, policy.channels].iloc[policy.training[0]:policy.training[1]]
    return _digest(policy.to_dict(), tuple(str(v) for v in data.schema.variables),
                   data.schema.cadence.value, tuple(str(t) for t in raw.index),
                   raw.to_numpy(dtype=float, na_value=np.nan).astype("<f8"), mask.to_numpy(dtype=np.uint8), windows)


def fit_deep(data, policy, training_hide, *, mode="offline", backend=None):
    """Fit once on QC-passed training windows with a frozen self-supervision mask."""
    _validate(data, policy, mode)
    windows = _windows(policy.training, policy.window)
    training = data.available.loc[:, policy.channels].iloc[policy.training[0]:policy.training[1]]
    stats = []
    for name in policy.channels:
        eligible = training[name].dropna().to_numpy(dtype=float)
        if not len(eligible):
            raise ValueError(f"no eligible training support for {name}")
        std = float(np.std(eligible))
        stats.append((float(np.mean(eligible)), std if std > 0 else 1.0, std == 0))
    means = tuple(s[0] for s in stats)
    scales = tuple(s[1] for s in stats)
    values, available = _arrays(data, windows, policy.channels, np.array(means), np.array(scales))
    hide = np.asarray(training_hide)
    if hide.shape != values.shape or hide.dtype != np.bool_ or (hide & ~available).any():
        raise ValueError("training_hide must be a boolean subset of available training cells")
    if not hide.any():
        raise ValueError("training_hide must include eligible training cells")
    observed = available & ~hide
    masked = np.where(observed, values, 0).astype(np.float32)
    inputs = _inputs(policy.method, masked, observed, values, hide)
    if backend is None:
        backend = TorchDeepBackend(policy)
    backend.fit(inputs, policy)
    state_hash = backend.state_hash()
    if not isinstance(state_hash, str) or not state_hash:
        raise RuntimeError("backend did not provide a trained state identity")
    return DeepFit(policy, tuple(v.unit for v in data.schema.variables), data.schema.cadence.value,
                   _training_identity(data, policy), means, scales,
                   tuple(s[2] for s in stats), _digest(hide.astype(np.uint8)),
                   hide.astype(np.uint8).tobytes(), state_hash, backend)


def _check_fit(data, fit):
    if not isinstance(fit, DeepFit):
        raise ValueError("fitted DeepFit required")
    _validate(data, fit.policy, "offline")
    if (fit.units != tuple(v.unit for v in data.schema.variables) or
            fit.cadence_ns != data.schema.cadence.value or
            fit.training_identity != _training_identity(data, fit.policy) or
            fit.state_hash != fit.backend.state_hash()):
        raise ValueError("stale or mismatched fitted model identity")
    training = data.available.loc[:, fit.policy.channels].iloc[fit.policy.training[0]:fit.policy.training[1]]
    expected = []
    for name in fit.policy.channels:
        values = training[name].dropna().to_numpy(dtype=float)
        std = float(np.std(values))
        expected.append((float(np.mean(values)), std if std > 0 else 1.0, std == 0))
    if (fit.mean != tuple(x[0] for x in expected) or
            fit.scale != tuple(x[1] for x in expected) or
            fit.constants != tuple(x[2] for x in expected)):
        raise ValueError("forged fitted scaling identity")
    shape = (len(_windows(fit.policy.training, fit.policy.window)),
             fit.policy.window, len(fit.policy.channels))
    if len(fit.training_hide_bytes) != math.prod(shape):
        raise ValueError("forged fitted training mask identity")
    hide = np.frombuffer(fit.training_hide_bytes, dtype=np.uint8).reshape(shape)
    if (not np.isin(hide, (0, 1)).all() or
            _digest(hide) != fit.mask_hash):
        raise ValueError("forged fitted training mask identity")


def _fit_identity(fit):
    return _digest(fit.training_identity, fit.mean, fit.scale, fit.constants,
                   fit.mask_hash, fit.state_hash, fit.policy.to_dict())


def make_deep_plan(data, fit, *, mode="offline"):
    if mode != "offline":
        raise ValueError("deep imputers require offline mode")
    _check_fit(data, fit)
    policy = fit.policy
    windows = _windows(policy.evaluation, policy.window)
    donor = data.mask("donor")
    structural = data.mask("structural")
    gaps = []
    for gap in profile_gaps(data):
        if gap.variable not in policy.targets:
            continue
        start, stop = gap.start_pos, gap.stop_pos
        window_index = next((i for i, (a, b) in enumerate(windows) if a <= start and stop <= b), None)
        reason = None
        if structural[gap.variable].iloc[start:stop].any():
            reason = "structural_absence"
        elif not (policy.evaluation[0] <= start and stop <= policy.evaluation[1]):
            reason = "outside_evaluation"
        elif gap.duration > policy.max_gap:
            reason = "gap_exceeds_limit"
        elif window_index is None:
            reason = "crosses_window_boundary"
        else:
            a, b = windows[window_index]
            if int(donor[gap.variable].iloc[a:b].sum()) < policy.min_target_support:
                reason = "insufficient_target_support"
        gaps.append(DeepGap(gap.variable, start, stop, reason is None, reason, window_index))
    return DeepPlan(data.fingerprint, _fit_identity(fit), policy, windows, tuple(gaps))


def execute_deep(data, fit, plan, *, mode="offline"):
    """Run a canonical plan once; reject invalid backend output before any result write."""
    if mode != "offline":
        raise ValueError("deep imputers require offline mode")
    _check_fit(data, fit)
    if not isinstance(plan, DeepPlan) or plan != make_deep_plan(data, fit):
        raise ValueError("stale or forged deep plan")
    mean, scale = np.array(fit.mean), np.array(fit.scale)
    values, available = _arrays(data, plan.windows, fit.policy.channels, mean, scale)
    predictions = fit.backend.infer(_inputs(fit.policy.method, values, available))
    output = np.asarray(predictions)
    if output.shape != values.shape or output.dtype.kind not in "iuf" or not np.isrealobj(output):
        raise RuntimeError("deep backend returned malformed output")
    eligible = np.zeros(output.shape, dtype=bool)
    for gap in plan.gaps:
        if gap.eligible:
            a, _ = plan.windows[gap.window_index]
            j = fit.policy.channels.index(gap.variable)
            eligible[gap.window_index, gap.start_pos-a:gap.stop_pos-a, j] = True
    if not np.isfinite(output[eligible]).all():
        raise RuntimeError("deep backend returned nonfinite eligible predictions")
    if not np.isfinite((output * scale + mean)[eligible]).all():
        raise RuntimeError("deep backend returned predictions outside finite output range")
    completed = data.available
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
    for gap in plan.gaps:
        j = columns.get_loc(gap.variable)
        slot = slice(gap.start_pos, gap.stop_pos)
        if not gap.eligible:
            reasons.iloc[slot, j] = gap.reason
            continue
        a, _ = plan.windows[gap.window_index]
        pred = output[gap.window_index, gap.start_pos-a:gap.stop_pos-a, j].astype(float)
        completed.iloc[slot, j] = pred * scale[j] + mean[j]
        states.iloc[slot, j] = "imputed"
        methods.iloc[slot, j] = fit.policy.method
        imputed.iloc[slot, j] = True
    identity = {"method": fit.policy.method, "source": "PyPOTS 1.5 BSD-3-Clause",
                "source_wheel_sha256": "913d13073074f2aab96c697bc2d4ec13b5420df0af9f33a32311dfe994a7e0cd",
                "license_sha256": "dd2a7246a537117ab02bb7f3551a4b0dfa238c213d2d670d9defe0edc1f50785",
                "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                            "pandas": pd.__version__,
                            "torch": fit.backend.torch.__version__ if hasattr(fit.backend, "torch") else None},
                "fit_identity": fit.training_identity, "state_hash": fit.state_hash,
                "mask_hash": fit.mask_hash, "mean": fit.mean, "scale": fit.scale,
                "constants": fit.constants, "seed": fit.policy.seed,
                "optimizer": {"name": "Adam", "lr": fit.policy.learning_rate,
                              "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0},
                "epochs": fit.policy.epochs, "steps": fit.backend.steps,
                "windows": plan.windows}
    return ImputationResult(data, plan, completed, states, methods, reasons, imputed,
                            backend_identity=identity)


class TorchDeepBackend:
    """Small CPU Torch loop; constructed only by an explicit deep fit."""

    def __init__(self, policy):
        try:
            import torch
            from ._pypots.nn.modules.loss import MAE
            torch.manual_seed(policy.seed)
            if policy.method == "brits_pypots":
                from ._pypots.imputation.brits.core import _BRITS
                model = _BRITS(policy.window, len(policy.channels), policy.rnn_hidden_size, MAE(), MAE())
            else:
                from ._pypots.imputation.saits.core import _SAITS
                model = _SAITS(policy.n_layers, policy.window, len(policy.channels),
                               policy.d_model, policy.n_heads, policy.d_k, policy.d_v,
                               policy.d_ffn, policy.dropout, policy.attn_dropout,
                               True, 1.0, 1.0, MAE(), MAE())
        except ImportError as exc:
            raise ImportError("deep adapters require the optional torch==2.14.0 extra") from exc
        self.torch = torch
        self.model = model
        self.steps = 0
        self.initial_state_hash = self.state_hash()

    def _tensors(self, value):
        if isinstance(value, dict):
            return {k: self._tensors(v) for k, v in value.items()}
        return self.torch.as_tensor(value.copy(), dtype=self.torch.float32)

    def state_hash(self):
        digest = hashlib.sha256()
        for name, value in self.model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def fit(self, inputs, policy):
        torch = self.torch
        if hasattr(self, "pre_fit_check"):
            self.pre_fit_check(inputs, policy)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=policy.learning_rate,
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        self.model.train()
        count = len(inputs["X_ori"])
        self.losses = []
        self.gradients = []
        for _ in range(policy.epochs):
            for start in range(0, count, policy.batch_size):
                if hasattr(self, "batch_guard"):
                    self.batch_guard()
                end = min(start + policy.batch_size, count)
                def batch(v):
                    if isinstance(v, dict):
                        return {k: batch(x) for k, x in v.items()}
                    return v[start:end]
                optimizer.zero_grad()
                result = self.model(self._tensors(batch(inputs)), calc_criterion=True)
                loss = result["loss"]
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite deep training loss")
                loss.backward()
                norms = [p.grad.detach().abs().max().item() for p in self.model.parameters()
                         if p.grad is not None]
                if not norms or not all(math.isfinite(x) for x in norms):
                    raise RuntimeError("nonfinite or missing deep gradients")
                optimizer.step()
                self.losses.append(float(loss.detach()))
                self.gradients.append(max(norms))
                self.steps += 1
                if hasattr(self, "batch_guard"):
                    self.batch_guard()

    def infer(self, inputs):
        self.model.eval()
        with self.torch.no_grad():
            return self.model(self._tensors(inputs), calc_criterion=False)["imputation"].cpu().numpy()
