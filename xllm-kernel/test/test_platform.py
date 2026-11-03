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

"""CPU-only tests for the hardware typing helpers in ``xllm_kernel.platform``.

The Ascend chip name comes from ``npu-smi info`` or from ``acl.get_soc_name()``,
so both sources are faked here; availability of the real hardware is not
required.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from xllm_kernel.platform import Platform, PlatformEnum

# Verbatim per-device rows from ``npu-smi info`` on an Ascend 910B2 host; the
# chip name is the token in the Name column.
SMI_910B2_OUTPUT = """
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 0     910B2               | OK            | 111.7                50                      0    / 0                |
| 0                         | 0000:C1:00.0  | 0                    0    / 0                3434 / 65536            |
+===========================+===============+======================================================================+
| 1     910B2               | OK            | 94.6                 49                      0    / 0                |
| 0                         | 0000:C2:00.0  | 0                    0    / 0                3431 / 65536            |
+===========================+===============+======================================================================+
"""

# Same shape from an unsupported Ascend 310P3 host. Its Name column carries no
# token this module maps, and the HBM column happens to read ``1950 / 65536`` --
# a whole-output search finds the ``950`` inside ``1950`` and reports a5 for a
# host that is not an A5 (or any generation the mapping knows).
SMI_310P3_OUTPUT = """
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 0     310P3               | OK            | 78.5                 45                      0    / 0                |
| 0                         | 0000:C1:00.0  | 0                    0    / 0                1950 / 65536            |
+===========================+===============+======================================================================+
"""


def _clear_platform_caches() -> None:
    """Drop the cached detections so a case re-reads the faked sources."""
    for name in ("enum", "get_npu_chip", "get_ascend_soc_generation"):
        Platform.__dict__[name].__func__.cache_clear()


@pytest.fixture(autouse=True)
def reset_platform_caches() -> Iterator[None]:
    _clear_platform_caches()
    yield
    _clear_platform_caches()


def _fake_no_npu_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Behave like a host where the ``npu-smi`` probe cannot be used."""
    monkeypatch.setattr(subprocess, "run", _missing_npu_smi)


def _missing_npu_smi(*args: Any, **kwargs: Any) -> Any:
    raise OSError("npu-smi is not installed")


def _fake_acl(monkeypatch: pytest.MonkeyPatch, soc_name: str) -> None:
    """Expose an ``acl`` module whose ``get_soc_name()`` returns ``soc_name``."""
    monkeypatch.setitem(sys.modules, "acl", SimpleNamespace(get_soc_name=lambda: soc_name))


def test_chip_comes_from_the_smi_name_column(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=SMI_910B2_OUTPUT))

    assert Platform.get_npu_chip() == "910b2"
    assert Platform.get_ascend_soc_generation() == "a2"


def test_smi_numbers_are_not_read_as_a_chip_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # 310P3 is not a mapped generation, and the only ``950`` on the page sits in
    # an unrelated numeric column, so nothing may be reported from it.
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=SMI_310P3_OUTPUT))

    assert Platform.get_npu_chip() == "unknown"
    assert Platform.get_ascend_soc_generation() is None


@pytest.mark.parametrize(
    ("soc_name", "expected_chip", "expected_generation"),
    [
        ("Ascend910B2", "910b2", "a2"),
        ("Ascend910B4", "910b4", "a2"),
        ("Ascend910_93", "910_93", "a3"),
        ("Ascend910C", "910c", "a3"),
        ("Ascend950", "950", "a5"),
        ("Ascend310P3", "unknown", None),
    ],
)
def test_acl_soc_name_resolves_the_chip_and_its_generation(
    monkeypatch: pytest.MonkeyPatch,
    soc_name: str,
    expected_chip: str,
    expected_generation: str | None,
) -> None:
    _fake_no_npu_smi(monkeypatch)
    _fake_acl(monkeypatch, soc_name)

    assert Platform.get_npu_chip() == expected_chip
    assert Platform.get_ascend_soc_generation() == expected_generation


def test_smi_chip_wins_over_acl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=SMI_910B2_OUTPUT))
    _fake_acl(monkeypatch, "Ascend910_93")

    assert Platform.get_npu_chip() == "910b2"


@pytest.mark.parametrize("expected", [PlatformEnum.NPU, PlatformEnum.CUDA, PlatformEnum.CPU])
def test_framework_detection_is_cached_and_shared(monkeypatch: pytest.MonkeyPatch, expected: PlatformEnum) -> None:
    from xllm_kernel.platform import current_platform

    cuda = SimpleNamespace(is_available=lambda: expected is PlatformEnum.CUDA)
    npu = SimpleNamespace(is_available=lambda: expected is PlatformEnum.NPU)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, npu=npu))
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", SimpleNamespace(HIP_HOME=None))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    for module in ("ixformer", "torch_musa", "torch_mlu"):
        monkeypatch.setitem(sys.modules, module, None)
    assert Platform.enum() is expected
    assert current_platform.device_type() == expected.value
    monkeypatch.setitem(sys.modules, "torch", None)
    assert Platform.enum() is expected
    Platform.enum.cache_clear()
    assert Platform.enum() is PlatformEnum.CPU
