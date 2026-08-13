# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardware typing for xLLM -- the single home for ``Platform``.

Modeled on sglang's platform interface (``sglang/srt/platforms/interface.py``):
callers reach a process-wide ``current_platform`` singleton and ask it
``current_platform.is_npu()`` / ``current_platform.is_cuda()`` rather than
branching on a raw string.

This module runs in two processes with opposite ``torch`` guarantees, so every
``torch`` access is guarded:

* The C++ worker's embedded CPython, where ``torch`` is always present. It
  imports this module through ``xllm.python`` to pick the kernel package.
* The ``xllm serve`` launcher, which has no hard ``torch`` dependency. It has no
  access to ``xllm.python`` (importing that package pulls in ``torch`` and a
  kernel package), so ``xllm.auto_config.utils`` loads this file by path and
  re-exports the symbols for the auto-tuning profiles.

Device-type detection comes solely from the framework runtime
(``torch`` / ``torch_npu`` / ``torch_mlu`` / ...), matching
``scripts/build_support/utils.py``. When ``torch`` is absent -- as in the
launcher -- or reports no accelerator, the platform resolves to
``PlatformEnum.CPU``.

Nothing here raises: an undetectable environment resolves to
``PlatformEnum.CPU`` / ``CpuArchEnum.UNSPECIFIED`` / ``"unknown"`` so callers can
always read a value.
"""

from __future__ import annotations

import enum
import functools
import platform as platform_module
import re
import subprocess

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


# CUDA-like backends that share the CUDA kernel package and the ``torch.cuda``
# runtime API. Detection maps DCU and ILU onto this group; the executor binds
# the CUDA kernel package for all three (see ``xllm/python/__init__.py``).
_CUDA_LIKE = (PlatformEnum.CUDA, PlatformEnum.DCU, PlatformEnum.ILU)


def _torch_device_type() -> PlatformEnum | None:
    """Detect the accelerator via the framework runtime, or None if absent.

    Follows the same probing order as
    ``scripts/build_support/utils.py::get_device_type``. Every import is
    guarded: neither process may fail just because a framework wheel is missing.
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


def _npu_chip_from_smi() -> str | None:
    """Best-effort lowercase Ascend chip name from ``npu-smi info``, or None.

    ``npu-smi`` may be absent or permission-blocked, so this never raises. The
    chip appears in the per-device rows (e.g. ``910B2C``).
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


def _npu_chip_from_acl() -> str | None:
    """Best-effort lowercase Ascend chip name via ``acl``, or None.

    ``acl`` is often unavailable in the launcher's import path and may already
    be initialized by torch_npu, so failures are swallowed and this is only a
    supplement to ``npu-smi``.
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
    """Streamlined hardware-typing facade shared by the worker and launcher.

    All methods are classmethods, so the launcher's auto-tuning profiles call
    ``Platform.is_npu()`` on the class while worker code reaches the same
    queries through the ``current_platform`` singleton -- the sglang usage
    idiom. Detection results are cached for the process lifetime.
    """

    @classmethod
    @functools.lru_cache(maxsize=1)
    def enum(cls) -> PlatformEnum:
        detected = _torch_device_type()
        if detected is not None:
            return detected

        logger.warning("Platform: could not detect an accelerator; treating host as CPU.")
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
        """Raw CPU machine string, e.g. ``x86_64`` or ``aarch64``."""
        return platform_module.machine() or "unknown"

    @classmethod
    @functools.lru_cache(maxsize=1)
    def get_npu_chip(cls) -> str:
        """Lowercase NPU chip identifier (e.g. ``910b2c``), or ``"unknown"``.

        Prefers ``npu-smi`` (reliable in the launcher) and falls back to
        ``acl``.
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
    def get_ascend_soc_generation(cls) -> str | None:
        """Ascend SoC generation (``a2``/``a3``/``a5``), or None if not an NPU.

        Derived from the chip name so it does not require a second ``acl.init``.
        Mapping matches ``scripts/build_support/utils.py::get_ascend_platform``:
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
    def get_device_count(cls) -> int | None:
        """Number of usable devices from the framework runtime, or None.

        Returns None when ``torch`` is absent (as in the launcher) or reports no
        device.
        """
        try:
            import torch
        except ImportError:
            return None

        try:
            if cls.enum() in _CUDA_LIKE:
                return torch.cuda.device_count()
            device_module = getattr(torch, cls.device_type(), None)
            if device_module is not None and hasattr(device_module, "device_count"):
                return device_module.device_count()
        except Exception:
            pass
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Human-readable device name, best-effort.

        Uses the framework runtime when available; otherwise composes a name
        from the device type and (for NPU) the chip.
        """
        try:
            import torch

            if cls.enum() in _CUDA_LIKE:
                return torch.cuda.get_device_name(device_id)
            device_module = getattr(torch, cls.device_type(), None)
            if device_module is not None and hasattr(device_module, "get_device_name"):
                return device_module.get_device_name(device_id)
        except Exception:
            pass

        if cls.is_npu():
            return f"npu:{cls.get_npu_chip()}"
        return cls.device_type()


# Process-wide singleton, the sglang-style entry point:
# ``from xllm.python.platform import current_platform``. Construction is cheap;
# detection is deferred to the first query via ``Platform.enum``'s cache.
current_platform = Platform()


def detect_hardware() -> dict[str, str]:
    """Return the current hardware descriptor: CPU arch, device type, NPU chip.

    Backed by ``Platform`` so detection stays consistent across profiles. Every
    field falls back gracefully so the caller can always rely on the keys.
    """
    return {
        "arch": Platform.get_cpu_arch_str(),
        "device_type": Platform.device_type(),
        "chip": Platform.get_npu_chip(),
    }
