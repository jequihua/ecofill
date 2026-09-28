"""Optional, locally loaded TimesFM research backends for ForecastRequest."""

import hashlib
from pathlib import Path

import numpy as np

from .forecast import ForecastRequest


CODE_REVISION = "e31dadd84cb26bd5153fde6687502b8312e918fb"
PACKAGE_VERSION = "3.0.2"
CHECKPOINTS = {
    "tf25": ("google/timesfm-2.5-200m-pytorch", "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
             "2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8",
             "Apache-2.0"),
    "tf3": ("google/timesfm-3.0-pytorch", "43046b85ec22d584a13f8098c2ed39c889e129c2",
            "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da",
            "timesfm-non-commercial-license-v1.0"),
}
TF3_LICENSE_SHA256 = "3e36db7240d23adb6ac6d7d931892dcc5706a14a7d31581f08c35d2f25736dfd"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _local_checkpoint(path: Path, lane: str) -> None:
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"{lane}: provide a verified local checkpoint directory")
    if _sha256(path / "model.safetensors") != CHECKPOINTS[lane][2]:
        raise ValueError(f"{lane}: checkpoint SHA-256 disagrees with D019")
    if lane == "tf3" and _sha256(path / "LICENSE") != TF3_LICENSE_SHA256:
        raise ValueError("tf3: restricted license SHA-256 disagrees with D019")


def _request_shape(request: ForecastRequest, max_context: int, max_horizon: int) -> None:
    if not isinstance(request, ForecastRequest) or request.direction not in {"forward", "backward"}:
        raise ValueError("TimesFM requires a directional ForecastRequest")
    if request.context.ndim != 2 or request.context.shape[1] != len(request.channels):
        raise ValueError("target context shape disagrees with channel order")
    if not 1 <= len(request.context) <= max_context or not 1 <= request.horizon <= max_horizon:
        raise ValueError("request exceeds pinned context or horizon")
    if request.historical_covariates.shape != (len(request.context), request.horizon_covariates.shape[1]):
        raise ValueError("historical covariate shape disagrees with context")
    if request.horizon_covariates.shape[0] != request.horizon:
        raise ValueError("future covariate shape disagrees with horizon")
    for values in (request.context, request.historical_covariates, request.horizon_covariates):
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("TimesFM inputs must be finite real numeric arrays")


class TimesFM25Backend:
    """Pinned 2.5 checkpoint, independent univariate forecasts, CPU only."""

    def __init__(self, checkpoint_dir: str | Path, *, context_length: int, max_horizon: int):
        if type(context_length) is not int or context_length < 1 or type(max_horizon) is not int or max_horizon < 1:
            raise ValueError("context_length and max_horizon must be positive integers")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.context_length = context_length
        self.max_horizon = max_horizon
        self._model = None

    @property
    def identity(self) -> dict:
        repo, revision, sha, license_id = CHECKPOINTS["tf25"]
        return {"backend": "timesfm25", "package": "timesfm", "package_version": PACKAGE_VERSION,
                "code_revision": CODE_REVISION, "checkpoint": repo, "checkpoint_revision": revision,
                "checkpoint_sha256": sha, "license": license_id, "device": "cpu",
                "context_length": self.context_length, "max_horizon": self.max_horizon,
                "batch_size": 1, "torch_compile": False, "normalize_inputs": False,
                "force_flip_invariance": False, "infer_is_positive": False,
                "continuous_quantile_head": False, "point_only": True,
                "mode": "independent"}

    def load(self) -> None:
        _local_checkpoint(self.checkpoint_dir, "tf25")
        try:
            import torch
            from timesfm import ForecastConfig, TimesFM_2p5_200M_torch
        except ImportError as exc:
            raise ImportError("TimesFM 2.5 needs the pinned optional timesfm==3.0.2 and torch==2.14.0 wheels") from exc
        if self._model is not None:
            raise RuntimeError("TimesFM 2.5 checkpoint is already loaded")
        model = TimesFM_2p5_200M_torch(torch_compile=False)
        model.model.device = torch.device("cpu")
        model.model.device_count = 1
        model.load_checkpoint(str(self.checkpoint_dir), torch_compile=False)
        model.compile(ForecastConfig(max_context=self.context_length, max_horizon=self.max_horizon,
                                     per_core_batch_size=1, normalize_inputs=False,
                                     force_flip_invariance=False, infer_is_positive=False,
                                     use_continuous_quantile_head=False,
                                     fix_quantile_crossing=False, return_backcast=False))
        self._model = model

    def forecast(self, request: ForecastRequest) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("load the verified TimesFM 2.5 checkpoint explicitly before forecasting")
        _request_shape(request, self.context_length, self.max_horizon)
        if request.historical_covariates.shape[1] or request.static_covariates:
            raise ValueError("TimesFM 2.5 comparison does not support requested covariates")
        outputs = []
        for channel in range(len(request.channels)):
            point, _ = self._model.forecast(request.horizon, [request.context[:, channel].copy()])
            point = np.asarray(point)
            if point.shape != (1, request.horizon):
                raise RuntimeError("TimesFM 2.5 returned an invalid point shape")
            outputs.append(point[0])
        return np.stack(outputs, axis=1)


class TimesFM3Backend:
    """Pinned 3.0 checkpoint for local non-commercial research, CPU only."""

    def __init__(self, checkpoint_dir: str | Path, *, context_length: int, max_horizon: int,
                 joint: bool, purpose: str):
        if purpose != "non-commercial research":
            raise ValueError("TimesFM 3 weights require explicit non-commercial research purpose")
        if type(joint) is not bool or type(context_length) is not int or context_length < 1 or type(max_horizon) is not int or max_horizon < 1:
            raise ValueError("joint, context_length and max_horizon must be explicit valid settings")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.context_length = context_length
        self.max_horizon = max_horizon
        self.joint = joint
        self._model = None

    @property
    def identity(self) -> dict:
        repo, revision, sha, license_id = CHECKPOINTS["tf3"]
        return {"backend": "timesfm3", "package": "timesfm", "package_version": PACKAGE_VERSION,
                "code_revision": CODE_REVISION, "checkpoint": repo, "checkpoint_revision": revision,
                "checkpoint_sha256": sha, "license": license_id,
                "purpose": "non-commercial research", "device": "cpu",
                "context_length": self.context_length, "max_horizon": self.max_horizon,
                "batch_size": 1, "mode": "joint" if self.joint else "independent",
                "return_quantiles": False, "use_symmetric_averaging": False,
                "make_positive": False, "use_znorm": False, "padding_mode": "none"}

    def load(self) -> None:
        _local_checkpoint(self.checkpoint_dir, "tf3")
        try:
            from timesfm3 import TimesFM3Forecaster
        except ImportError as exc:
            raise ImportError("TimesFM 3 needs the pinned optional timesfm==3.0.2 and torch==2.14.0 wheels") from exc
        if self._model is not None:
            raise RuntimeError("TimesFM 3 checkpoint is already loaded")
        self._model = TimesFM3Forecaster.from_pretrained(
            str(self.checkpoint_dir), device="cpu", per_core_batch_size=1,
            local_files_only=True, force_download=False)

    def forecast(self, request: ForecastRequest) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("load the verified TimesFM 3 checkpoint explicitly before forecasting")
        _request_shape(request, self.context_length, self.max_horizon)
        if request.static_covariates:
            raise ValueError("TimesFM 3 comparison does not support requested static covariates")
        if not self.joint and request.historical_covariates.shape[1]:
            raise ValueError("independent TimesFM 3 comparison does not support covariates")
        covariates = None
        if request.historical_covariates.shape[1]:
            covariates = np.concatenate((request.historical_covariates,
                                         request.horizon_covariates), axis=0).T.copy()
        def predict(context):
            output = self._model.predict(context.T.copy(), horizon=request.horizon,
                                         past_future_covariates=covariates,
                                         return_quantiles=False,
                                         use_symmetric_averaging=False, make_positive=False,
                                         use_znorm=False, padding_mode="none")
            point = np.asarray(output.forecast)
            if point.shape != (context.shape[1], request.horizon):
                raise RuntimeError("TimesFM 3 returned an invalid point shape")
            return point.T.copy()
        if self.joint:
            return predict(request.context)
        return np.concatenate([predict(request.context[:, i:i + 1])
                               for i in range(len(request.channels))], axis=1)
