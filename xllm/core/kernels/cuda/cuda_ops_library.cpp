/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// xllm_ops: expose xLLM's CUDA fused kernels to the Python model graph as
// torch custom ops (torch.ops.xllm_ops.*). The Python model (built in an
// embedded CPython interpreter that shares this process' libtorch) can then
// call these ops without any per-hardware #ifdef: the torch dispatcher routes
// by tensor device to the registered backend impl.
//
// First-version scope (Qwen3 dense / CUDA / eager): only the stateless fused
// kernels are exposed here. Attention is handled by the flashinfer Python
// package directly (see layers/attention.py).

#include "core/kernels/cuda/cuda_ops_library.h"

#include <glog/logging.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "core/kernels/cuda/cuda_ops_api.h"

namespace xllm {
namespace {

// -------- wrappers over the in-place CUDA kernels --------
// The underlying xllm::kernel::cuda::* kernels write in place / take an
// output-argument. rms_norm / silu_and_mul are exposed as functional ops
// (allocate a fresh output via empty/empty_like — no input copy). The in-place
// ops (fused_add_rms_norm, fused_qk_norm_rope) mutate their input AND return
// the mutated tensor(s). Returning the input is required for piecewise
// cudagraph: torch.compile's cudagraphs backend needs each graph segment to
// have non-void tensor outputs; void-return ops produce "empty graphs".

torch::Tensor rms_norm(const torch::Tensor& input,
                       const torch::Tensor& weight,
                       double eps) {
  auto output = torch::empty_like(input);
  auto w = weight;
  xllm::kernel::cuda::rms_norm(output, input, w, eps);
  return output;
}

// Fused residual-add + RMSNorm. IN-PLACE: mutates `input` -> RMSNorm(input +
// residual) and `residual` -> input + residual (the underlying kernel writes
// both). Returns the mutated (input, residual) so piecewise cudagraph segments
// have proper tensor outputs (avoids empty graphs from void-return ops).
std::tuple<torch::Tensor, torch::Tensor> fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps) {
  auto w = weight;
  xllm::kernel::cuda::fused_add_rms_norm(input, residual, w, eps);
  return std::make_tuple(input, residual);
}

// Gated SiLU: input is [..., 2*d]; output is [..., d] = silu(input[..., :d]) *
// input[..., d:].
torch::Tensor silu_and_mul(const torch::Tensor& input) {
  auto sizes = input.sizes().vec();
  CHECK(!sizes.empty() && sizes.back() % 2 == 0)
      << "silu_and_mul: last dim must be even, got " << input.sizes();
  sizes.back() /= 2;
  auto out = torch::empty(sizes, input.options());
  xllm::kernel::cuda::act_and_mul(out, input, "silu");
  return out;
}

// Qwen3 fused q/k RMSNorm (on head_dim) + RoPE. `qkv` is [num_tokens,
// (nq+nk+nv)*head_dim]. IN-PLACE: q/k slices are normalized+roped in `qkv`, v
// left untouched. Returns the mutated qkv so piecewise cudagraph segments have
// proper tensor outputs.
torch::Tensor fused_qk_norm_rope(torch::Tensor& qkv,
                                 int64_t num_heads_q,
                                 int64_t num_heads_k,
                                 int64_t num_heads_v,
                                 int64_t head_dim,
                                 double eps,
                                 const torch::Tensor& q_weight,
                                 const torch::Tensor& k_weight,
                                 const torch::Tensor& cos_sin_cache,
                                 bool interleaved,
                                 const torch::Tensor& position_ids) {
  xllm::kernel::cuda::fused_qk_norm_rope(qkv,
                                         num_heads_q,
                                         num_heads_k,
                                         num_heads_v,
                                         head_dim,
                                         eps,
                                         q_weight,
                                         k_weight,
                                         cos_sin_cache,
                                         interleaved,
                                         position_ids);
  return qkv;
}

// reshape_paged_cache: write K/V into paged cache at slot_mapping positions.
torch::Tensor reshape_paged_cache_op(const torch::Tensor& slot_mapping,
                                     const torch::Tensor& keys,
                                     const torch::Tensor& values,
                                     torch::Tensor& key_cache,
                                     torch::Tensor& value_cache) {
  xllm::kernel::cuda::reshape_paged_cache(
      slot_mapping, keys, values, key_cache, value_cache);
  return key_cache;
}

}  // namespace

void ensure_xllm_ops_registered() {
  // Intentionally empty. Referencing this symbol keeps the object file (and its
  // TORCH_LIBRARY static initializers below) from being stripped by the linker.
}

}  // namespace xllm

// Schemas are declared once (device-agnostic); CUDA impls are bound below.
TORCH_LIBRARY(xllm_ops, m) {
  m.def("rms_norm(Tensor input, Tensor weight, float eps) -> Tensor");
  m.def(
      "fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor "
      "weight, "
      "float eps) -> (Tensor, Tensor)");
  m.def("silu_and_mul(Tensor input) -> Tensor");
  m.def(
      "fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, "
      "int "
      "num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor k_weight, "
      "Tensor cos_sin_cache, bool interleaved, Tensor position_ids) -> Tensor");
  m.def(
      "reshape_paged_cache(Tensor slot_mapping, Tensor keys, Tensor values, "
      "Tensor(a!) key_cache, Tensor(b!) value_cache) -> Tensor");
}

TORCH_LIBRARY_IMPL(xllm_ops, CUDA, m) {
  m.impl("rms_norm", TORCH_FN(xllm::rms_norm));
  m.impl("fused_add_rms_norm", TORCH_FN(xllm::fused_add_rms_norm));
  m.impl("silu_and_mul", TORCH_FN(xllm::silu_and_mul));
  m.impl("fused_qk_norm_rope", TORCH_FN(xllm::fused_qk_norm_rope));
  m.impl("reshape_paged_cache", TORCH_FN(xllm::reshape_paged_cache_op));
}
