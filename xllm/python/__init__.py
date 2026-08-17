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

"""python: Python-defined model graphs executed by xLLM's C++ worker.

The C++ ``PyCausalLM`` embeds a CPython interpreter (sharing this process'
libtorch), loads a model class from this package, streams weights into
``load_weights``, and drives ``forward`` / ``compute_logits`` each step.

Package layout follows ``models -> layers -> kernels``: :mod:`python.models`
(per-arch graphs) depend on :mod:`python.layers`, which depend on
``python.kernels``. :mod:`python.distributed` sits beside them and owns the
tensor-parallel collectives, which differ only in ProcessGroup backend.

``python.kernels`` is not a directory. One peer package per hardware platform
lives here -- ``kernels_cuda``, ``kernels_npu``, and one per platform added
later -- sharing no code and never importing each other. The import below binds
the one matching the active platform as ``kernels``, so layers and models import
a single fixed name and carry no hardware branch, and exactly one kernel package
is ever imported in a process.

This is one of the executor's two hardware branches. The other selects the
attention backend and graph runner in :mod:`python.model_executor.executor`,
where it belongs: attention backends hold state across steps and follow the
executor's lifecycle, unlike the stateless functions a kernel package exports.

The C++ worker registers ``torch.ops.xllm_ops`` before importing this package,
so the kernel package finds its operators here.
"""

from __future__ import annotations

import sys

from xllm.python.platform import current_platform

# Bound before anything else in the package so that a module reached from here
# already finds ``xllm.python.kernels`` in place.
if current_platform.is_cuda():
    from xllm.python import kernels_cuda as kernels
elif current_platform.is_npu():
    from xllm.python import kernels_npu as kernels
else:
    raise ImportError(
        f"no kernel package for platform '{current_platform.device_type()}'; add "
        f"xllm/python/kernels_{current_platform.device_type()}/ with the APIs "
        "required by that platform's supported models, and import it here"
    )

# The binding above only creates an attribute of this package, which the import
# machinery does not consult, so ``import xllm.python.kernels`` and
# ``from xllm.python.kernels import rms_norm`` would fail to resolve. Publishing
# the same module object under the name lifts that restriction; it stays the one
# object the peer package's own name is bound to, so nothing is imported twice.
sys.modules[f"{__name__}.kernels"] = kernels

from xllm.python.registry import get_model_class, register_model  # noqa: E402

__all__ = ["get_model_class", "register_model", "kernels"]
