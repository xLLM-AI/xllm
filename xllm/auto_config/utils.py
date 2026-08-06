# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Hardware typing for xLLM auto-tuning.

A slimmed-down Python `Platform`, modeled on sglang's
`multimodal_gen/runtime/platforms/interface.py`, that answers "what hardware is
this?" for the auto_config tuning profiles. It runs inside the `xllm serve`
launcher process, which has no hard `torch` dependency, so detection is
layered and torch-optional:

1. Prefer the framework runtime (`torch` / `torch_npu` / `torch_mlu` / ...)
   when it is importable and reports an available device -- the most accurate
   signal, matching `scripts/build_support/utils.py::get_device_type`.
2. Otherwise fall back to visible-device env masks, then `npu-smi` for the
   Ascend chip name, neither of which requires torch.

Nothing here raises: an undetectable environment resolves to
`PlatformEnum.UNSPECIFIED` / `CpuArchEnum.UNSPECIFIED` / `"unknown"` so callers
can always read a value.
"""

from __future__ import annotations

import copy
import enum
import functools
import importlib.util
import json
import os
import platform as platform_module
import re
import subprocess
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any, Dict, Optional, Sequence

from scripts.logger import logger


class PlatformEnum(enum.Enum):
    CUDA = enum.auto()
    NPU = enum.auto()
    MLU = enum.auto()
    MUSA = enum.auto()
    DCU = enum.auto()
    ILU = enum.auto()
    CPU = enum.auto()
    UNSPECIFIED = enum.auto()


class CpuArchEnum(enum.Enum):
    X86 = enum.auto()
    ARM = enum.auto()
    UNSPECIFIED = enum.auto()


# Visible-device env masks per backend, mirroring the CLI reference's
# device-selection note. Order matches the accelerator detection precedence.
_VISIBLE_DEVICE_ENV_VARS = {
    PlatformEnum.NPU: "ASCEND_RT_VISIBLE_DEVICES",
    PlatformEnum.CUDA: "CUDA_VISIBLE_DEVICES",
    PlatformEnum.MLU: "MLU_VISIBLE_DEVICES",
    PlatformEnum.DCU: "HIP_VISIBLE_DEVICES",
    PlatformEnum.MUSA: "MUSA_VISIBLE_DEVICES",
}


def _torch_device_type() -> Optional[PlatformEnum]:
    """Detect the accelerator via the framework runtime, or None if absent.

    Follows the same probing order as
    `scripts/build_support/utils.py::get_device_type`. Every import is guarded:
    the launcher must not fail just because a framework wheel is missing.
    """
    try:
        import torch
    except ImportError:
        return None

    try:
        if torch.cuda.is_available():
            try:
                from torch.utils.cpp_extension import HIP_HOME

                if HIP_HOME and "dtk" in HIP_HOME.lower():
                    return PlatformEnum.DCU
            except ImportError:
                pass
            try:
                import ixformer  # noqa: F401

                return PlatformEnum.ILU
            except ImportError:
                return PlatformEnum.CUDA
    except Exception:
        pass

    for module_name, attr, enum_value in (
        ("torch_musa", "musa", PlatformEnum.MUSA),
        ("torch_mlu", "mlu", PlatformEnum.MLU),
        ("torch_npu", "npu", PlatformEnum.NPU),
    ):
        try:
            __import__(module_name)
            device_module = getattr(torch, attr, None)
            if device_module is not None and device_module.is_available():
                return enum_value
        except ImportError:
            continue
        except Exception:
            continue

    return None


def _env_device_type() -> Optional[PlatformEnum]:
    """Detect the accelerator from visible-device env masks, or None."""
    for enum_value, env_var in _VISIBLE_DEVICE_ENV_VARS.items():
        if os.environ.get(env_var) is not None:
            return enum_value
    return None


def _npu_chip_from_smi() -> Optional[str]:
    """Best-effort lowercase Ascend chip name from `npu-smi info`, or None.

    `npu-smi` may be absent or permission-blocked, so this never raises. The
    chip appears in the per-device rows (e.g. `910B2C`).
    """
    try:
        completed = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    match = re.search(r"910[A-Za-z0-9]*", completed.stdout)
    if match is None:
        return None
    return match.group(0).lower()


def _npu_chip_from_acl() -> Optional[str]:
    """Best-effort lowercase Ascend chip name via `acl`, or None.

    `acl` is often unavailable in the launcher's import path and may already be
    initialized by torch_npu, so failures are swallowed and this is only a
    supplement to `npu-smi`.
    """
    try:
        import acl
    except ImportError:
        return None
    try:
        soc_name = acl.get_soc_name()
    except Exception:
        return None
    if not soc_name:
        return None
    match = re.search(r"910[A-Za-z0-9]*", soc_name)
    if match is None:
        return None
    return match.group(0).lower()


class Platform:
    """Streamlined hardware-typing facade for auto-tuning profiles.

    All methods are classmethods so profiles can call `Platform.is_npu()`
    without holding an instance. Detection results are cached for the process
    lifetime.
    """

    @classmethod
    @functools.lru_cache(maxsize=1)
    def enum(cls) -> PlatformEnum:
        detected = _torch_device_type()
        if detected is not None:
            return detected

        detected = _env_device_type()
        if detected is not None:
            return detected

        # No framework runtime and no visible-device mask, but a readable
        # Ascend chip still means this is an NPU host.
        if cls.get_npu_chip() != "unknown":
            return PlatformEnum.NPU

        logger.warning(
            "Platform: could not detect an accelerator; treating host as CPU."
        )
        return PlatformEnum.CPU

    @classmethod
    def is_cuda(cls) -> bool:
        return cls.enum() == PlatformEnum.CUDA

    @classmethod
    def is_npu(cls) -> bool:
        return cls.enum() == PlatformEnum.NPU

    @classmethod
    def is_mlu(cls) -> bool:
        return cls.enum() == PlatformEnum.MLU

    @classmethod
    def is_musa(cls) -> bool:
        return cls.enum() == PlatformEnum.MUSA

    @classmethod
    def is_dcu(cls) -> bool:
        return cls.enum() == PlatformEnum.DCU

    @classmethod
    def is_ilu(cls) -> bool:
        return cls.enum() == PlatformEnum.ILU

    @classmethod
    def is_cpu(cls) -> bool:
        return cls.enum() == PlatformEnum.CPU

    @classmethod
    def device_type(cls) -> str:
        """Lowercase device-type string, aligned with build-time detection."""
        return cls.enum().name.lower()

    @classmethod
    @functools.lru_cache(maxsize=1)
    def get_cpu_architecture(cls) -> CpuArchEnum:
        arch = platform_module.machine().lower()
        if "x86" in arch or "amd64" in arch:
            return CpuArchEnum.X86
        if "arm" in arch or "aarch64" in arch:
            return CpuArchEnum.ARM
        return CpuArchEnum.UNSPECIFIED

    @classmethod
    def get_cpu_arch_str(cls) -> str:
        """Raw CPU machine string, e.g. `x86_64` or `aarch64`."""
        return platform_module.machine() or "unknown"

    @classmethod
    @functools.lru_cache(maxsize=1)
    def get_npu_chip(cls) -> str:
        """Lowercase NPU chip identifier (e.g. `910b2c`), or `"unknown"`.

        Prefers `npu-smi` (reliable in the launcher) and falls back to `acl`.
        """
        chip = _npu_chip_from_smi()
        if chip is not None:
            return chip
        chip = _npu_chip_from_acl()
        if chip is not None:
            return chip
        return "unknown"

    @classmethod
    @functools.lru_cache(maxsize=1)
    def get_ascend_soc_generation(cls) -> Optional[str]:
        """Ascend SoC generation (`a2`/`a3`/`a5`), or None if not an NPU host.

        Derived from the chip name so it does not require a second `acl.init`.
        Mapping matches `scripts/build_support/utils.py::get_ascend_platform`:
        910B -> a2, 910C / 910_93 -> a3, 950 -> a5, other 910 -> a2.
        """
        chip = cls.get_npu_chip()
        if chip == "unknown":
            return None
        if chip.startswith("910b"):
            return "a2"
        if chip.startswith("910c") or "910_93" in chip:
            return "a3"
        if chip.startswith("950"):
            return "a5"
        if chip.startswith("910"):
            return "a2"
        return None

    @classmethod
    def get_device_count(cls) -> Optional[int]:
        """Number of usable devices, or None if undetectable.

        Prefers the framework runtime's device count; falls back to counting
        entries in the platform's visible-device env mask.
        """
        try:
            import torch
        except ImportError:
            torch = None

        if torch is not None:
            try:
                if cls.is_cuda() or cls.is_dcu() or cls.is_ilu():
                    return torch.cuda.device_count()
                device_module = getattr(torch, cls.device_type(), None)
                if device_module is not None and hasattr(
                    device_module, "device_count"
                ):
                    return device_module.device_count()
            except Exception:
                pass

        env_var = _VISIBLE_DEVICE_ENV_VARS.get(cls.enum())
        if env_var is not None:
            value = os.environ.get(env_var)
            if value is not None:
                entries = [
                    entry for entry in value.split(",") if entry.strip() != ""
                ]
                return len(entries)
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Human-readable device name, best-effort.

        Uses the framework runtime when available; otherwise composes a name
        from the device type and (for NPU) the chip.
        """
        try:
            import torch

            if cls.is_cuda() or cls.is_dcu() or cls.is_ilu():
                return torch.cuda.get_device_name(device_id)
            device_module = getattr(torch, cls.device_type(), None)
            if device_module is not None and hasattr(
                device_module, "get_device_name"
            ):
                return device_module.get_device_name(device_id)
        except Exception:
            pass

        if cls.is_npu():
            return f"npu:{cls.get_npu_chip()}"
        return cls.device_type()


def detect_hardware() -> Dict[str, str]:
    """Return the current hardware descriptor: CPU arch, device type, NPU chip.

    Backed by `Platform` so detection stays consistent across profiles. Every
    field falls back gracefully so the caller can always rely on the keys.
    """
    return {
        "arch": Platform.get_cpu_arch_str(),
        "device_type": Platform.device_type(),
        "chip": Platform.get_npu_chip(),
    }


def check_device_count(
    model_type: str,
    base_config: Dict[str, Any],
    context: Dict[str, Any],
) -> bool:
    """Check the visible device count against the profile's optimal `nnodes`.

    Shared across model profiles: `model_type` only labels the log lines.
    Returns True when they match. A mismatch is non-fatal: it is logged as a
    warning so the operator can decide whether to adjust the launch topology.
    """
    optimal_nnodes = base_config.get("nnodes")
    visible_device_count = context.get("visible_device_count")

    if optimal_nnodes is None or visible_device_count is None:
        logger.warning(
            "%s auto-tuning: cannot verify device count "
            "(optimal nnodes=%s, visible devices=%s)",
            model_type,
            optimal_nnodes,
            visible_device_count,
        )
        return False

    if visible_device_count != optimal_nnodes:
        logger.warning(
            "%s auto-tuning: visible device count %s does not match the "
            "profile's optimal nnodes %s; the tuned config keeps nnodes=%s.",
            model_type,
            visible_device_count,
            optimal_nnodes,
            visible_device_count,
        )
        return False

    logger.info(
        "%s auto-tuning: visible device count %s matches optimal nnodes %s.",
        model_type,
        visible_device_count,
        optimal_nnodes,
    )
    return True


class BaseTuner(ABC):
    """Base class for per-model auto-tuning profiles.

    A profile subclasses this, sets `MODEL_TYPE`, and implements the two
    mandatory hooks -- `tune_common` (cross-platform adjustments) and
    `tune_npu` (the primary, validated target). The remaining platform hooks
    (`tune_cuda`, `tune_mlu`, `tune_musa`, `tune_dcu`, `tune_ilu`) have base
    no-op defaults, so a profile overrides one only when it has platform
    specific tuning; otherwise the base config is used unchanged there.

    `tune()` is the fixed orchestration entry point the launcher calls (via
    each profile module's `tune()` wrapper):

    1. deep-copy the base config (the input is never mutated),
    2. log the detected hardware and check the device count,
    3. apply cross-platform adjustments in `tune_common()`,
    4. dispatch to the platform-specific hook selected from `Platform.enum()`.

    The hooks mutate the working config in place (they return `None`); `tune()`
    owns and returns the copy.
    """

    # Subclasses MUST override with their config.json / <type>.py base name.
    MODEL_TYPE: str = "base"

    def tune(
        self,
        base_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        tuned_config = copy.deepcopy(base_config)

        hardware = context.get("hardware") or detect_hardware()
        logger.info(
            "%s auto-tuning: detected hardware device_type=%s chip=%s arch=%s.",
            self.MODEL_TYPE,
            hardware.get("device_type"),
            hardware.get("chip"),
            hardware.get("arch"),
        )

        check_device_count(self.MODEL_TYPE, base_config, context)

        self.tune_common(tuned_config, context)
        self._dispatch_platform(tuned_config, context)
        return tuned_config

    def _dispatch_platform(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        dispatch = {
            PlatformEnum.NPU: self.tune_npu,
            PlatformEnum.CUDA: self.tune_cuda,
            PlatformEnum.MLU: self.tune_mlu,
            PlatformEnum.MUSA: self.tune_musa,
            PlatformEnum.DCU: self.tune_dcu,
            PlatformEnum.ILU: self.tune_ilu,
        }
        hook = dispatch.get(Platform.enum())
        if hook is None:
            logger.warning(
                "%s auto-tuning: no tuning hook for platform %s; using base "
                "config unchanged.",
                self.MODEL_TYPE,
                Platform.enum(),
            )
            return
        hook(config, context)

    def tune_common(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Cross-platform adjustments applied before the platform hook.

        Mandatory: a profile with no cross-platform tuning implements this as a
        one-line `pass`.
        """
        pass

    def tune_npu(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Tune for Ascend NPU. Mutate `config` in place."""
        pass

    def tune_cuda(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Tune for NVIDIA CUDA. Base default: no change. Override as needed."""
        pass

    def tune_mlu(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Tune for Cambricon MLU. Base default: no change. Override as needed."""
        pass

    def tune_musa(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Tune for Moore Threads MUSA. Base default: no change. Override as needed."""
        pass

    def tune_dcu(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Tune for Hygon DCU. Base default: no change. Override as needed."""
        pass

    def tune_ilu(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Tune for Iluvatar GPU. Base default: no change. Override as needed."""
        pass


class AutoTuningError(Exception):
    """Raised when auto-tuning cannot produce a config.

    The launcher catches this and surfaces the message via `parser.error`, so
    these functions stay independent of argparse.
    """


def auto_tuning_config_dir() -> str:
    """Directory holding the per-model tuning profiles (`xllm/auto_config`)."""
    return os.path.dirname(os.path.realpath(__file__))


def extract_model_path(extra_args: Sequence[str]) -> Optional[str]:
    """Recover `--model` from the launcher's passthrough args.

    `--model` is forwarded to the binary rather than parsed by the launcher, so
    it must be read out of `extra_args`. Supports `--model VALUE` and
    `--model=VALUE`.
    """
    for index, arg in enumerate(extra_args):
        if arg == "--model":
            if index + 1 < len(extra_args):
                return extra_args[index + 1]
            return None
        if arg.startswith("--model="):
            return arg[len("--model=") :]
    return None


def read_model_type(model_path: str) -> str:
    """Read `model_type` (fallback `model_name`) from `<model_path>/config.json`.

    Mirrors C++ `util::get_model_type`. Raises `AutoTuningError` on any problem.
    """
    config_path = os.path.join(
        os.path.realpath(os.path.expanduser(model_path)), "config.json"
    )
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            model_config = json.load(config_file)
    except FileNotFoundError:
        raise AutoTuningError(f"model config.json not found: {config_path}")
    except json.JSONDecodeError as error:
        raise AutoTuningError(f"failed to parse {config_path}: {error}")
    except OSError as error:
        raise AutoTuningError(f"failed to read {config_path}: {error}")

    if not isinstance(model_config, dict):
        raise AutoTuningError(f"{config_path} must contain a JSON object")

    model_type = model_config.get("model_type") or model_config.get("model_name")
    if not isinstance(model_type, str) or not model_type:
        raise AutoTuningError(
            f"{config_path} must contain a string `model_type` or `model_name`"
        )
    return model_type


def load_tuning_module(py_path: str, model_type: str) -> ModuleType:
    """Import a `<model_type>.py` tuning profile by file path.

    Loading by path avoids triggering `xllm/__init__`'s lazy `xllm_export` .so
    load. Raises `AutoTuningError` if the module cannot be imported or does not
    expose a callable `tune`.
    """
    spec = importlib.util.spec_from_file_location(
        f"xllm.auto_config.{model_type}", py_path
    )
    if spec is None or spec.loader is None:
        raise AutoTuningError(f"failed to load tuning module: {py_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise AutoTuningError(f"failed to import {py_path}: {error}")
    if not callable(getattr(module, "tune", None)):
        raise AutoTuningError(
            f"{py_path} must define a callable `tune(base_config, context)`"
        )
    return module


def generate_tuned_config(extra_args: Sequence[str], output_dir: str) -> str:
    """Generate a tuned JSON config for the model and write it to `output_dir`.

    Resolves the model's `model_type`, loads its base config and tuning
    profile, applies the profile's `tune()`, and writes
    `<output_dir>/<model_type>.tuned.json`. Returns the written path. Raises
    `AutoTuningError` on any failure.
    """
    model_path = extract_model_path(extra_args)
    if not model_path:
        raise AutoTuningError("auto-tuning requires --model <path>")

    model_type = read_model_type(model_path)

    config_dir = auto_tuning_config_dir()
    base_json_path = os.path.join(config_dir, f"{model_type}.json")
    tuning_py_path = os.path.join(config_dir, f"{model_type}.py")
    if not os.path.isfile(base_json_path) or not os.path.isfile(tuning_py_path):
        raise AutoTuningError(
            f"auto-tuning is not supported for model_type `{model_type}`: "
            f"expected both {base_json_path} and {tuning_py_path} to exist."
        )

    try:
        with open(base_json_path, "r", encoding="utf-8") as base_file:
            base_config = json.load(base_file)
    except (OSError, json.JSONDecodeError) as error:
        raise AutoTuningError(f"failed to read {base_json_path}: {error}")
    if not isinstance(base_config, dict):
        raise AutoTuningError(f"{base_json_path} must contain a JSON object")

    module = load_tuning_module(tuning_py_path, model_type)

    detect_hardware_fn = getattr(module, "detect_hardware", None)
    hardware = detect_hardware_fn() if callable(detect_hardware_fn) else None
    context = {
        "model_path": model_path,
        "model_type": model_type,
        "visible_device_count": Platform.get_device_count(),
        "hardware": hardware,
    }

    try:
        tuned_config = module.tune(base_config, context)
    except Exception as error:
        raise AutoTuningError(f"{tuning_py_path} tune() failed: {error}")
    if not isinstance(tuned_config, dict):
        raise AutoTuningError(f"{tuning_py_path} tune() must return a dict")

    output_path = os.path.join(output_dir, f"{model_type}.tuned.json")
    try:
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(tuned_config, output_file, indent=2)
            output_file.write("\n")
    except OSError as error:
        raise AutoTuningError(f"failed to write tuned config {output_path}: {error}")

    logger.info(
        "auto-tuning: wrote tuned config for %s to %s", model_type, output_path
    )
    return output_path
