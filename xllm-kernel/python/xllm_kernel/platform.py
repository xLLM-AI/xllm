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

"""Shared hardware typing for standalone kernels, the launcher and workers.

Imports are SDK-free. Framework and device probes run lazily on the first
query and cache their results for the process lifetime.
"""

from __future__ import annotations

import enum
import functools
import platform as platform_module
import re
import subprocess


class PlatformEnum(enum.Enum):
    CUDA = "cuda"
    NPU = "npu"
    MLU = "mlu"
    MUSA = "musa"
    DCU = "dcu"
    ILU = "ilu"
    CPU = "cpu"
    UNSPECIFIED = "unspecified"


class CpuArchEnum(enum.Enum):
    X86 = enum.auto()
    ARM = enum.auto()
    UNSPECIFIED = enum.auto()


# CUDA-like backends that share the CUDA kernel package and the ``torch.cuda``
# runtime API. Detection maps DCU and ILU onto this group; the executor binds
# the CUDA kernel package for all three through ``initialize_runtime()``.
_CUDA_LIKE = (PlatformEnum.CUDA, PlatformEnum.DCU, PlatformEnum.ILU)

# Ascend SoC chip tokens, aligned with the build-time mapping in
# ``scripts/build_support/utils.py``: the ``910`` family covers A2/A3 and
# ``950`` covers A5. An A3 host reports the SoC name ``Ascend910_93``, so ``_``
# belongs to the token -- ``get_ascend_soc_generation()`` keys off ``910_93``.
_ASCEND_CHIP_TOKEN = r"(?:910|950)[A-Za-z0-9_]*"

# ``npu-smi info`` draws one device per row and puts the chip name in the Name
# column, the second ``|``-delimited cell (see the ``910B2`` fixture in
# ``xllm-kernel/test/test_platform.py``). Every other column carries unrelated
# numbers -- an unsupported 310P3 host reports HBM as ``1950 / 65536`` -- so the
# chip has to be read from that one cell; a search over the whole output matches
# the ``950`` inside ``1950`` and reports a5 for a host that is not even Ascend.
_SMI_NAME_CELL = 1


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
    chip appears in the per-device rows (e.g. ``910B2C``), in the Name column --
    the scan is confined to that column so the other columns' numbers cannot be
    mistaken for a chip token.
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

    match = re.search(_ASCEND_CHIP_TOKEN, _smi_name_column(completed.stdout))
    if match is None:
        return None
    return match.group(0).lower()


def _smi_name_column(info_output: str) -> str:
    """Join the Name cells of every ``npu-smi info`` device row.

    Rows look like ``| 0     910B2 | OK | 111.7 ...`` -- the cell after the
    leading one holds the chip name, so only that cell is returned. Lines that
    do not carry a second cell (banners, separators, the trailing summary) are
    skipped, which keeps the caller's regex off the numeric columns.
    """
    cells = []
    for line in info_output.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) > _SMI_NAME_CELL + 1:
            cells.append(parts[_SMI_NAME_CELL])
    return "\n".join(cells)


def _npu_chip_from_acl() -> str | None:
    """Best-effort lowercase Ascend chip name via ``acl``, or None.

    ``acl`` is often unavailable in the launcher's import path and may already
    be initialized by torch_npu, so failures are swallowed and this is only a
    supplement to ``npu-smi``. ``acl.get_soc_name()`` returns the whole SoC name
    (e.g. ``Ascend910B2``, ``Ascend910_93``), so the chip token keeps its
    underscore.
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
    match = re.search(_ASCEND_CHIP_TOKEN, soc_name)
    if match is None:
        return None
    return match.group(0).lower()


class Platform:
    """Streamlined hardware-typing facade shared by the worker and launcher.

    Callers share the lazy ``current_platform`` singleton, for example
    ``current_platform.is_npu()``. Its classmethods share detection results
    cached for the process lifetime.
    """

    @classmethod
    @functools.lru_cache(maxsize=1)
    def enum(cls) -> PlatformEnum:
        detected = _torch_device_type()
        if detected is not None:
            return detected

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
# ``from xllm_kernel.platform import current_platform``. Construction is cheap;
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
