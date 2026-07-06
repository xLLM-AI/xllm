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
// kernels are exposed here. The stateful paged-attention op (which reads the
// thread-local forward context for KV cache + flashinfer plan) is registered
// separately once that context exists (see py_model_bridge / M3).

#include "core/kernels/cuda/xllm_ops_library.h"

#include <glog/logging.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "core/kernels/cuda/cuda_ops_api.h"

namespace xllm {
namespace {

// -------- wrappers over the in-place CUDA kernels --------
// The underlying xllm::kernel::cuda::* kernels write in place / take an
// output-argument. rms_norm / silu_and_mul are exposed as functional ops
// (allocate a fresh output via empty/empty_like — no input copy). The two ops
// whose kernel mutates its INPUT (fused_add_rms_norm, fused_qk_norm_rope) are
// exposed as honest IN-PLACE ops: (a!) input annotation + void return (no
// clone), with the Python wrapper returning the mutated tensor. See each impl
// for the torch.compile-functionalization reason the return must be void.

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
// both). Declared with (a!)/(b!) alias annotations and a VOID return: the
// Python wrapper hands the (mutated) input/residual back to the caller.
//
// Why void instead of returning the results: torch.compile's functionalization
// pass REJECTS a custom op whose *output* carries an alias annotation
// (``-> Tensor(a!)``) — it only functionalizes ops whose outputs do not share
// storage with inputs. A mutating op must therefore return nothing and expose
// the mutation solely through the ``(a!)`` input annotation. This drops the two
// per-call clones (18147 device-to-device copies in the M6 host profile, 0 on
// the C++ path) that were the quantified root cause of the eager host overhead.
void fused_add_rms_norm(torch::Tensor& input,
                        torch::Tensor& residual,
                        const torch::Tensor& weight,
                        double eps) {
  auto w = weight;
  xllm::kernel::cuda::fused_add_rms_norm(input, residual, w, eps);
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
// left untouched. Same (a!) + void-return rationale as fused_add_rms_norm above
// (drops the qkv.clone() per call); the Python wrapper returns the mutated qkv.
void fused_qk_norm_rope(torch::Tensor& qkv,
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
  // In-place (mutates input & residual); void return — see impl comment.
  m.def(
      "fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor "
      "weight, "
      "float eps) -> ()");
  m.def("silu_and_mul(Tensor input) -> Tensor");
  // In-place (mutates qkv q/k slices); void return — see impl comment.
  m.def(
      "fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, "
      "int "
      "num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor k_weight, "
      "Tensor cos_sin_cache, bool interleaved, Tensor position_ids) -> ()");
}

TORCH_LIBRARY_IMPL(xllm_ops, CUDA, m) {
  m.impl("rms_norm", TORCH_FN(xllm::rms_norm));
  m.impl("fused_add_rms_norm", TORCH_FN(xllm::fused_add_rms_norm));
  m.impl("silu_and_mul", TORCH_FN(xllm::silu_and_mul));
  m.impl("fused_qk_norm_rope", TORCH_FN(xllm::fused_qk_norm_rope));
}
