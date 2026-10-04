# Copyright 2025-2026 The xLLM Authors.
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

"""One-shot bootstrap for NPU embedded-interpreter environment.

When running inside xLLM's C++ embedded Python interpreter on NPU,
libtorch_npu.so is already loaded and has registered TORCH_LIBRARY("triton")
during static init. A subsequent ``import torch`` would call
Library("triton", "DEF") again, raising "Only a single TORCH_LIBRARY"
RuntimeError.

This module patches torch.library.Library to absorb the duplicate registration,
then imports torch and torch_npu safely. It must be imported exactly once by the
C++ host (py_model_helper.cpp) before any other xllm.python module — never by
user code.

TODO: Remove once libtorch_npu defers the "triton" registration to Python
      (i.e. uses TORCH_LIBRARY_FRAGMENT instead of TORCH_LIBRARY).
"""

import os
import sys
import types

import torch.library as _torch_library

_OrigLibrary = _torch_library.Library


class _SafeLibrary(_OrigLibrary):
    """Absorb duplicate TORCH_LIBRARY registration from libtorch_npu."""

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except RuntimeError as e:
            if "Only a single TORCH_LIBRARY" not in str(e):
                raise


_torch_library.Library = _SafeLibrary
import torch  # noqa: E402

# torch_npu supports synchronous Inductor compilation, but its initialization
# does not propagate this environment setting to PyTorch's config object.
# Apply the documented single-thread setting after torch is imported safely.
if os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") == "1":
    import torch._inductor.config as _inductor_config

    _inductor_config.compile_threads = 1

_torch_library.Library = _OrigLibrary
del _OrigLibrary, _SafeLibrary

# Register the optional KDA OPP before torch_npu initializes ACL. Import or
# registration failures in an installed package must remain visible.
try:
    import fla_npu
except ModuleNotFoundError as error:
    if error.name != "fla_npu":
        raise
else:
    fla_npu.load_ascendc_opapi_libraries()

# ---------------------------------------------------------------------------
# Import torch_npu with accelerator masked to prevent re-initialization.
# ---------------------------------------------------------------------------
try:
    _orig_get_acc = torch._C._get_accelerator

    class _FakeCPUAcc:
        type = "cpu"

    torch._C._get_accelerator = staticmethod(lambda: _FakeCPUAcc())
    try:
        import torch_npu  # noqa: F401
    finally:
        torch._C._get_accelerator = _orig_get_acc
except Exception:
    pass

# Fallback: if torch_npu import failed (version mismatch), provide a minimal
# torch.npu shim so that torch.empty(device="npu:0") works.
if not hasattr(torch, "npu"):
    torch.npu = types.ModuleType("torch.npu")
    torch.npu.__package__ = "torch.npu"
    sys.modules["torch.npu"] = torch.npu
    torch.npu.synchronize = lambda device=None: None
