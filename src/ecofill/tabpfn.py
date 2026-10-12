"""Optional native TabPFN 3.5 regression backend for time-indexed requests.

Lazy and local-only: importing this module imports neither TabPFN nor Torch. The one
admitted checkpoint is validated byte-for-byte before native deserialization and is
loaded through TabPFN 9.0.0's `load_model_criterion_config(...,
download_if_not_exists=False)` into `RegressorModelSpecs`, so a missing file fails
instead of downloading. CPU is the default; `cuda` is only an explicit request that
is refused, before any model load, when CUDA is unavailable. There is no fallback.

`identity` is the stable predictor declaration (package, checkpoint, device and every
effective method setting). Timings, native devices/dtypes, effective inference
configuration, preprocessing statistics and request digests are execution details in
`load_record` and `executions`; they never change `identity`.
"""

import dataclasses
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import time

import numpy as np

PACKAGE_VERSION = "9.0.0"
EXTRA_HINT = "install the optional extra: pip install 'ecofill[tabpfn]' (tabpfn==9.0.0)"
CHECKPOINT = {
    "repository": "Prior-Labs/tabpfn_3_5",
    "revision": "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
    "filename": "tabpfn-v3.5-20260909.safetensors",
    "bytes": 876027932,
    "sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
    "model_version": "v3.5",
    "license": "tabpfn-3-5-license-v1.0",
}
# Fixed regressor arguments; random_state comes from the request seed and device from
# the caller. softmax_temperature "auto" resolves from this exact checkpoint.
SETTINGS = {
    "n_estimators": 1, "categorical_features_indices": [], "softmax_temperature": "auto",
    "average_before_softmax": False, "ignore_pretraining_limits": False,
    "inference_precision": "float32", "fit_mode": "fit_preprocessors", "memory_saving_mode": True,
    "n_preprocessing_jobs": 1, "differentiable_input": False, "eval_metric": None,
    "tuning_config": None, "show_progress_bar": False, "output_type": "mean",
}
INFERENCE_OVERRIDES = {"TRANSFORM_TEXT": False, "TRANSFORM_DATES": False,
                       "FINGERPRINT_FEATURE": False, "SUBSAMPLE_SAMPLES": None,
                       "MAX_CPU_SAMPLES": 5000}
DEVICES = ("cpu", "cuda")
RUNTIME_PACKAGES = ("tabpfn", "torch", "numpy", "scikit-learn", "safetensors")


def _versions() -> dict:
    """Installed versions from package metadata, without importing anything."""
    try:
        found = {name: metadata.version(name) for name in RUNTIME_PACKAGES}
    except metadata.PackageNotFoundError as exc:
        raise ImportError(f"TabPFN backend unavailable ({exc.name} is not installed); {EXTRA_HINT}") from exc
    if found["tabpfn"] != PACKAGE_VERSION:
        raise ImportError(f"TabPFN backend is pinned to tabpfn=={PACKAGE_VERSION}, found "
                          f"{found['tabpfn']}; {EXTRA_HINT}")
    return found


def _import_native():
    """The small, explicit native surface used here (pinned to tabpfn 9.0.0)."""
    try:
        import torch
        from tabpfn import TabPFNRegressor
        from tabpfn.base import RegressorModelSpecs
        from tabpfn.constants import ModelVersion
        from tabpfn.inference_config import cpu_sample_limit
        from tabpfn.model_loading import load_model_criterion_config
    except ImportError as exc:
        raise ImportError(f"TabPFN backend could not import its native dependencies; {EXTRA_HINT}") from exc
    return {"torch": torch, "TabPFNRegressor": TabPFNRegressor, "RegressorModelSpecs": RegressorModelSpecs,
            "ModelVersion": ModelVersion, "cpu_sample_limit": cpu_sample_limit,
            "load_model_criterion_config": load_model_criterion_config}


def validate_checkpoint(path) -> dict:
    """Exact size and SHA-256 of the admitted local checkpoint; nothing is fetched."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"TabPFN checkpoint {CHECKPOINT['filename']} not found at the given local path; acquire it "
            f"explicitly from {CHECKPOINT['repository']} revision {CHECKPOINT['revision']} and verify "
            "its SHA-256. No download is attempted.")
    size = path.stat().st_size
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    if size != CHECKPOINT["bytes"] or digest.hexdigest() != CHECKPOINT["sha256"]:
        raise ValueError(f"TabPFN checkpoint bytes differ from the admitted {CHECKPOINT['filename']} "
                         f"(size {size}, sha256 {digest.hexdigest()}); refusing to deserialize it")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _digest(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def request_digests(request) -> dict:
    """Digests of the exact reference/query arrays a request carries (execution evidence)."""
    return {"reference_features": _digest(request.reference_features),
            "reference_labels": _digest(request.reference_labels),
            "query_features": _digest(request.query_features)}


class TabPFNBackend:
    """TimeIndexedRequest -> native TabPFNRegressor fit on the reference bank -> batched mean.

    `stage(label)` is an optional execution hook called with STAGES labels: after load,
    immediately before and after fit, and immediately before and after predict (for
    example an elapsed-time, memory or thread guard). Its exceptions propagate, so a
    failed "after-fit" guard means predict is never called.
    """

    def __init__(self, checkpoint, *, device: str = "cpu", stage=None, output_observer=None):
        if device not in DEVICES:
            raise ValueError("device must be 'cpu' or 'cuda'")
        versions = _versions()
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.identity = {
            "backend": "tabpfn-native-regressor", "device": device,
            "package": {"name": "tabpfn", "version": versions["tabpfn"]},
            "checkpoint": dict(CHECKPOINT),
            "settings": dict(SETTINGS), "inference_overrides": dict(INFERENCE_OVERRIDES),
            "runtime": {name: versions[name] for name in RUNTIME_PACKAGES if name != "tabpfn"},
            "loader": "load_model_criterion_config(download_if_not_exists=False) -> RegressorModelSpecs",
            "features": "S01 TimeIndexedRequest columns in feature_names order; seed -> random_state",
        }
        self.stage = stage
        self.output_observer = output_observer
        self.load_record = None
        self.executions = []
        self._native = self._specs = None

    def _hook(self, label):
        if self.stage is not None:
            self.stage(label)

    def load(self) -> dict:
        """Import, route, validate the checkpoint bytes, then load it locally once."""
        if self._specs is not None:
            return self.load_record
        started = time.perf_counter()
        native = _import_native()
        imported = time.perf_counter()
        torch = native["torch"]
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested but CUDA is unavailable; no CPU fallback")
        checked = validate_checkpoint(self.checkpoint)
        validated = time.perf_counter()
        version = native["ModelVersion"](CHECKPOINT["model_version"])
        models, criterion, architectures, inference_config = native["load_model_criterion_config"](
            model_path=self.checkpoint, check_bar_distribution_criterion=True,
            cache_trainset_representation=False, estimator_type="regressor",
            version=version.value, download_if_not_exists=False)
        if len(models) != 1:
            raise RuntimeError("one admitted checkpoint must load exactly one model")
        # The path route applies the version CPU limit; the explicit spec route must too.
        limit = native["cpu_sample_limit"](version)
        if limit != INFERENCE_OVERRIDES["MAX_CPU_SAMPLES"]:
            raise RuntimeError(f"tabpfn {version.value} CPU sample limit is {limit}, not the admitted 5000")
        inference_config = dataclasses.replace(inference_config, MAX_CPU_SAMPLES=limit)
        self._specs = native["RegressorModelSpecs"](models[0], architectures[0], inference_config, criterion)
        self._native = native
        loaded = time.perf_counter()
        self.load_record = {"import_seconds": imported - started, "checksum_seconds": validated - imported,
                            "load_seconds": loaded - validated, "checkpoint": checked,
                            "checkpoint_inference_config": _jsonable(inference_config)}
        self._hook("loaded")
        return self.load_record

    def release(self) -> None:
        """Drop the loaded native model; a later predict loads again."""
        self._specs = None

    def regressor_kwargs(self, seed: int) -> dict:
        torch = self._native["torch"]
        kwargs = {key: value for key, value in SETTINGS.items() if key != "output_type"}
        kwargs.update(model_path=self._specs, device=self.device, random_state=seed,
                      inference_precision=torch.float32, inference_config=dict(INFERENCE_OVERRIDES))
        return kwargs

    def predict(self, request) -> np.ndarray:
        if request.device != self.device:
            raise RuntimeError(f"request device {request.device!r} differs from backend {self.device!r}")
        self.load()
        reference = np.array(request.reference_features, dtype=np.float64, copy=True)
        labels = np.array(request.reference_labels, dtype=np.float64, copy=True)
        query = np.array(request.query_features, dtype=np.float64, copy=True)
        regressor = self._native["TabPFNRegressor"](**self.regressor_kwargs(request.seed))
        self._hook("before-fit")
        started = time.perf_counter()
        regressor.fit(reference, labels)
        fitted = time.perf_counter()
        self._hook("after-fit")
        self._hook("before-predict")
        predicting = time.perf_counter()
        output = regressor.predict(query, output_type=SETTINGS["output_type"])
        predicted = time.perf_counter()
        if self.output_observer is not None:
            self.output_observer(request, output)
        native = native_evidence(regressor, self._native["torch"])
        check_native(native, self.device, request.seed)
        self.executions.append({
            "target": request.target, "reference_rows": len(labels), "query_rows": len(query),
            "fit_seconds": fitted - started, "predict_seconds": predicted - predicting,
            "request": request_digests(request), "native": native})
        self._hook("after-predict")
        return np.asarray(output, dtype=np.float64).copy()


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def native_evidence(regressor, torch) -> dict:
    """What the fitted native estimator actually used; shared by the adapter and parity control."""
    parameters = [item for model in regressor.models_ for item in model.parameters()]
    config = regressor.inference_config_
    return {
        "devices": [str(item) for item in regressor.devices_],
        "parameter_devices": sorted({str(item.device) for item in parameters}),
        "parameter_dtypes": sorted({str(item.dtype) for item in parameters}),
        "forced_inference_dtype": str(regressor.forced_inference_dtype_),
        "use_autocast": bool(regressor.use_autocast_),
        "n_estimators": int(regressor.n_estimators_), "random_state": regressor.random_state,
        "softmax_temperature": float(regressor.softmax_temperature_),
        "ensemble_softmax_temperature": float(regressor.ensemble_softmax_temperature_),
        "overrides": {key: getattr(config, key) for key in INFERENCE_OVERRIDES},
        "y_train_mean": getattr(regressor, "y_train_mean_", None),
        "y_train_std": getattr(regressor, "y_train_std_", None),
        "ensemble_configs": [repr(item) for item in regressor.ensemble_configs_],
        "effective_inference_config": _jsonable(config),
        "torch": {"intra": torch.get_num_threads(), "inter": torch.get_num_interop_threads(),
                  "deterministic": torch.are_deterministic_algorithms_enabled()},
    }


def native_device(text):
    """('cpu', None) or ('cuda', index) for one native device string; None if malformed.

    Pinned TabPFN parses the generic 'cuda' selector to torch.device('cuda:0'), so a
    bare 'cuda' means index 0 here too.
    """
    match = re.fullmatch(r"(cpu|cuda)(?::(\d+))?", text) if isinstance(text, str) else None
    if match is None or (match[1] == "cpu" and match[2] is not None):
        return None
    return (match[1], None) if match[1] == "cpu" else ("cuda", int(match[2] or 0))


def check_native(evidence: dict, device: str, seed: int) -> None:
    """Fail unless the declared settings actually reached the native estimator."""
    problems = []
    devices = [native_device(item) for item in evidence["devices"]]
    parameters = {native_device(item) for item in evidence["parameter_devices"]}
    if len(devices) != 1 or devices[0] is None or devices[0][0] != device or parameters != {devices[0]}:
        problems.append(f"native devices {evidence['devices']}/{evidence['parameter_devices']} do not "
                        f"match one requested {device!r} device")
    if evidence["forced_inference_dtype"] != "torch.float32" or evidence["use_autocast"] \
            or evidence["parameter_dtypes"] != ["torch.float32"]:
        problems.append("native precision is not forced float32 without autocast")
    if evidence["n_estimators"] != SETTINGS["n_estimators"] or evidence["random_state"] != seed:
        problems.append("native estimator count or random_state differs")
    if evidence["overrides"] != INFERENCE_OVERRIDES:
        problems.append(f"native inference overrides differ: {evidence['overrides']}")
    if evidence["ensemble_softmax_temperature"] != 1.0:
        problems.append("native ensemble temperature was tuned")
    if problems:
        raise RuntimeError("; ".join(problems))
