# Copyright 2025-2026 The xLLM Authors.
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

"""python: Python-defined model graphs executed by xLLM's C++ worker.

The C++ ``PyCausalLM`` embeds a CPython interpreter (sharing this process'
libtorch), loads a model class from this package, streams weights into
``load_weights``, and drives ``forward`` / ``compute_logits`` each step.

Package layout follows ``models -> layers -> ops -> kernels``:
:mod:`python.models` (per-arch graphs) depend on :mod:`python.layers`, which
depend on the :mod:`python.ops` dispatch layer, which routes to the
:mod:`python.kernels` backends. Every op resolves to the single
``torch.ops.xllm_ops.*`` namespace — the same fused kernels the C++ decoder path
uses — so the Python graph runs identical operators across hardware backends
with no ``#ifdef``.
"""

from xllm.python.registry import get_model_class, register_model  # noqa: F401

__all__ = ["get_model_class", "register_model"]
