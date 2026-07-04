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

"""FakeTensor (meta) impls for the xllm_ops custom ops — torch.compile prereq.

The five ``xllm_ops`` fused kernels are registered from C++ (``TORCH_LIBRARY`` in
``core/kernels/cuda/xllm_ops_library.cpp`` + ``py_model_bridge.cpp`` for
attention) with a CUDA impl only and **no Meta/FakeTensor impl**. Without a fake
impl Dynamo cannot infer output shape/dtype when it traces ``torch.ops.xllm_ops.*``
and would graph-break (or hard-error under ``fullgraph=True``). We register the
fakes here in Python (``torch.library.register_fake``, torch 2.7), each matching
its op schema exactly.

The paged-attention op (``xllm_ops.attention``) is intentionally **not** given a
fake impl. It has side effects the tracer cannot model (reads the thread-local
``PyForwardContext``, writes the current paged-KV slot, and its flashinfer plan /
kv_cache are not among the op args and change every step). We force it to
graph-break via ``torch._dynamo.disallow_in_graph`` so each layer's attention
runs eager while the pure-GPU segments between attentions (qkv/norm/rope/o_proj/
mlp) are handed to the compile backend. This is the exact same split point as the
C++ piecewise cudagraph (``PiecewiseGraphs`` / ``GlobalCaptureInstance``), only
the owner is torch.compile.

Imported only when a torch.compile backend is enabled (``XLLM_TC_BACKEND`` !=
off, see ``qwen3._maybe_compile``), so the default eager parity path never
registers these and stays byte-identical to C++.
"""

from __future__ import annotations

import torch


# rms_norm(Tensor input, Tensor weight, float eps) -> Tensor
@torch.library.register_fake("xllm_ops::rms_norm")
def _rms_norm_fake(input, weight, eps):
    return torch.empty_like(input)


# fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor weight,
#   float eps) -> ()    # IN-PLACE: input -> RMSNorm(input+residual),
#                       #           residual -> input+residual. No output.
@torch.library.register_fake("xllm_ops::fused_add_rms_norm")
def _fused_add_rms_norm_fake(input, residual, weight, eps):
    return None


# silu_and_mul(Tensor input) -> Tensor   # [..., 2*d] -> [..., d]
@torch.library.register_fake("xllm_ops::silu_and_mul")
def _silu_and_mul_fake(input):
    shape = list(input.shape)
    shape[-1] //= 2
    return input.new_empty(shape)


# fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k,
#   int num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor k_weight,
#   Tensor cos_sin_cache, bool interleaved, Tensor position_ids) -> ()
# IN-PLACE: q/k normed+roped in qkv, v untouched. No output.
@torch.library.register_fake("xllm_ops::fused_qk_norm_rope")
def _fused_qk_norm_rope_fake(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    eps,
    q_weight,
    k_weight,
    cos_sin_cache,
    interleaved,
    position_ids,
):
    return None


# Attention: no fake impl on purpose — force a graph break so it runs eager and
# segments the graph at the same boundary as the C++ piecewise cudagraph.
# `attention` is an OpOverloadPacket; disallow the concrete overload if present.
_attention_op = torch.ops.xllm_ops.attention
torch._dynamo.disallow_in_graph(getattr(_attention_op, "default", _attention_op))


# --------------------------------------------------------------------------
# Tensor-parallel collectives (WS1). Both ops are FUNCTIONAL / out-of-place at
# the C++ level (``py_all_reduce`` clones then reduces the clone;
# ``py_all_gather`` returns a freshly allocated gathered tensor), so — unlike
# attention — they carry no per-step mutable state and are safe to trace INTO
# the compiled graph and be captured by cudagraph. This mirrors how vLLM/SGLang
# keep collectives in the piecewise cuda graph: the collective op must be
# functional (or explicitly ``mutates_args``-annotated) so torch.compile does
# not miscompile it under cudagraph static-buffer reuse. SGLang specifically
# routes to an out-of-place allreduce when ``is_in_tc_piecewise_cuda_graph()``;
# our single out-of-place op is the same choice for every path.
#
# The fakes give Dynamo the output shape at trace time: all_reduce preserves
# shape; all_gather multiplies ``size(dim)`` by ``world_size`` (passed as an op
# arg for exactly this reason — the live TP group is not visible to the tracer).


# all_reduce(Tensor x) -> Tensor   # SUM across the TP group, shape unchanged.
@torch.library.register_fake("xllm_ops::all_reduce")
def _all_reduce_fake(x):
    return torch.empty_like(x)


# all_gather(Tensor x, int dim, int world_size) -> Tensor
#   concatenation along ``dim`` in rank order -> size(dim) *= world_size.
@torch.library.register_fake("xllm_ops::all_gather")
def _all_gather_fake(x, dim, world_size):
    shape = list(x.shape)
    shape[dim] *= world_size
    return x.new_empty(shape)
