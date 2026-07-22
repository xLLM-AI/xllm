/* Copyright 2026 The xLLM Authors.

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

// xllm_ops: NPU (PrivateUse1) dispatch registration for the Python model
// executor. Mirrors the schema defined in cuda_ops_library.cpp (which is only
// compiled for USE_CUDA builds). Under USE_NPU the schema must be declared here
// since the CUDA source is never compiled.
//
// Each wrapper is a thin adapter between the torch.ops schema and the
// underlying NPU kernel API. Data preparation (reshaping, dtype alignment)
// belongs in the Python caller, not here.

#include <torch/library.h>
#include <torch/torch.h>

#include "npu_ops_api.h"

namespace xllm {
namespace {

torch::Tensor rms_norm_npu(const torch::Tensor& input,
                           const torch::Tensor& weight,
                           double eps) {
  return xllm::kernel::npu::rms_norm(input, weight, eps, "rmsnorm");
}

std::tuple<torch::Tensor, torch::Tensor> fused_add_rms_norm_npu(
    torch::Tensor& input,
    torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps) {
  auto [normed, rstd, residual_sum] =
      xllm::kernel::npu::add_rms_norm(input, residual, weight, eps);
  return std::make_tuple(normed, residual_sum);
}

torch::Tensor silu_and_mul_npu(const torch::Tensor& input) {
  return xllm::kernel::npu::active(input, "swiglu");
}

torch::Tensor reshape_paged_cache_npu(const torch::Tensor& slot_mapping,
                                      torch::Tensor& keys,
                                      torch::Tensor& values,
                                      torch::Tensor& key_cache,
                                      torch::Tensor& value_cache) {
  std::optional<torch::Tensor> v = values;
  std::optional<torch::Tensor> vc = value_cache;
  xllm::kernel::npu::reshape_paged_cache(keys, v, key_cache, vc, slot_mapping);
  return key_cache;
}

void apply_rotary_embedding_npu(torch::Tensor& q,
                                torch::Tensor& k,
                                const torch::Tensor& cos_sin_cache,
                                const torch::Tensor& positions) {
  xllm::kernel::npu::apply_rotary(q, k, cos_sin_cache, positions);
}

}  // namespace

void ensure_xllm_ops_registered() {
  // Intentionally empty — referencing this symbol prevents the linker from
  // stripping the TORCH_LIBRARY static initializers below.
}

}  // namespace xllm

// Schema declarations (device-agnostic). Identical to cuda_ops_library.cpp —
// compiled only under USE_NPU (mutually exclusive with USE_CUDA).
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
      "reshape_paged_cache(Tensor slot_mapping, Tensor(c!) keys, Tensor(d!) "
      "values, "
      "Tensor(a!) key_cache, Tensor(b!) value_cache) -> Tensor");
  m.def(
      "apply_rotary_embedding(Tensor(a!) q, Tensor(b!) k, Tensor cos_sin_cache,"
      " Tensor positions) -> ()");
  m.def(
      "update_decode_graph_metadata(Tensor tokens, Tensor positions, Tensor "
      "slot_mapping, Tensor kv_seq_lens, Tensor paged_kv_indptr, Tensor "
      "paged_kv_indices, Tensor paged_kv_last_page_len, Tensor(a!) dst_tokens, "
      "Tensor(b!) dst_positions, Tensor(c!) dst_slot_mapping, Tensor(d!) "
      "dst_kv_seq_lens, Tensor(e!) dst_kv_seq_lens_delta, Tensor(f!) "
      "dst_paged_kv_indptr, Tensor(g!) dst_paged_kv_indices, Tensor(h!) "
      "dst_paged_kv_last_page_len, int padded_num_tokens) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, m) {
  m.impl("rms_norm", TORCH_FN(xllm::rms_norm_npu));
  m.impl("fused_add_rms_norm", TORCH_FN(xllm::fused_add_rms_norm_npu));
  m.impl("silu_and_mul", TORCH_FN(xllm::silu_and_mul_npu));
  m.impl("reshape_paged_cache", TORCH_FN(xllm::reshape_paged_cache_npu));
  m.impl("apply_rotary_embedding", TORCH_FN(xllm::apply_rotary_embedding_npu));
}
