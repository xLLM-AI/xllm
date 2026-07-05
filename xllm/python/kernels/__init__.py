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

"""JIT kernel backends for the op dispatch layer.

Portable Python-authored kernels (currently Triton) that the :mod:`python.ops`
dispatch layer can route to as an alternative to the C++ vendor kernels. Each
kernel module registers its own op (e.g. ``xllm_triton::silu_and_mul``) with a
``register_fake`` so it stays a single capturable node under torch.compile /
cudagraph. Kept import-light on purpose: the Triton dependency is only pulled in
when :mod:`python.kernels.triton_ops` is actually imported (gated by
``XLLM_USE_TRITON`` in :mod:`python.ops.dispatch`).
"""
