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

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {
void fused_mla_q_v2(const torch::Tensor& input,
                    torch::Tensor& output,
                    const std::optional<torch::Tensor>& output_norm,
                    const torch::Tensor& gamma,
                    const std::optional<torch::Tensor>& smooth_quant_scale,
                    const torch::Tensor& weight_b,
                    const std::optional<torch::Tensor>& weight_b_scale,
                    const torch::Tensor& sin,
                    const torch::Tensor& cos,
                    const torch::Tensor& position_id,
                    double eps,
                    bool interleaved) {
  tmo::torch_api::fused_mla_q_v2(input,
                                 output,
                                 output_norm,
                                 gamma,
                                 smooth_quant_scale,
                                 weight_b,
                                 weight_b_scale,
                                 sin,
                                 cos,
                                 position_id,
                                 eps,
                                 interleaved);
}

void fused_mla_q(const torch::Tensor& input,
                 torch::Tensor& output,
                 torch::Tensor& output_scale,
                 const std::optional<torch::Tensor>& output_norm,
                 const torch::Tensor& gamma,
                 const std::optional<torch::Tensor>& smooth_quant_scale,
                 const torch::Tensor& weight_b,
                 const torch::Tensor& weight_b_scale,
                 const torch::Tensor& weight_c,
                 const torch::Tensor& sin,
                 const torch::Tensor& cos,
                 const torch::Tensor& position_id,
                 const std::string& quant_mode,
                 double eps,
                 bool interleaved) {
  tmo::torch_api::fused_mla_q(input,
                              output,
                              output_scale,
                              output_norm,
                              gamma,
                              smooth_quant_scale,
                              weight_b,
                              weight_b_scale,
                              weight_c,
                              sin,
                              cos,
                              position_id,
                              quant_mode,
                              eps,
                              interleaved);
}

void fused_mla_kv(const torch::Tensor& input_kv,
                  const torch::Tensor& sin,
                  const torch::Tensor& cos,
                  const torch::Tensor& position_id,
                  const torch::Tensor& gamma,
                  const torch::Tensor& kv_cache,
                  const std::optional<torch::Tensor>& kv_cache_scale,
                  const std::optional<torch::Tensor>& slot_mapping,
                  const std::optional<torch::Tensor>& cache_bs_id,
                  const std::optional<torch::Tensor>& cache_seq_offset,
                  const std::string& quant_mode,
                  bool is_paged_cache,
                  double eps,
                  bool interleaved) {
  tmo::torch_api::fused_mla_kv(input_kv,
                               sin,
                               cos,
                               position_id,
                               gamma,
                               kv_cache,
                               kv_cache_scale,
                               slot_mapping,
                               cache_bs_id,
                               cache_seq_offset,
                               quant_mode,
                               is_paged_cache,
                               eps,
                               interleaved);
}

void fused_indexer_q(const torch::Tensor& input_q,
                     torch::Tensor& output,
                     const std::optional<torch::Tensor>& output_scale,
                     const torch::Tensor& w_q,
                     const std::optional<torch::Tensor>& w_q_scale,
                     const std::optional<torch::Tensor>& hadamard_matrix,
                     const torch::Tensor& sin,
                     const torch::Tensor& cos,
                     const torch::Tensor& position_id,
                     const std::string& quant_mode,
                     bool interleaved,
                     bool rope_at_front) {
  tmo::torch_api::fused_indexer_q(input_q,
                                  output,
                                  output_scale,
                                  w_q,
                                  w_q_scale,
                                  hadamard_matrix,
                                  sin,
                                  cos,
                                  position_id,
                                  quant_mode,
                                  interleaved,
                                  rope_at_front);
}

void fused_indexer_k(const torch::Tensor& x,
                     const torch::Tensor& wk,
                     const torch::Tensor& wproj,
                     const torch::Tensor& sin_table,
                     const torch::Tensor& cos_table,
                     const torch::Tensor& position_id,
                     const torch::Tensor& slot_mapping,
                     const torch::Tensor& head_weights,
                     const torch::Tensor& k_cache,
                     const std::optional<torch::Tensor>& k_cache_scale,
                     const std::optional<torch::Tensor>& hadamard_matrix,
                     bool interleaved,
                     const std::optional<torch::Tensor>& gamma,
                     const std::optional<torch::Tensor>& beta,
                     double eps) {
  tmo::torch_api::fused_indexer_k(x,
                                  wk,
                                  wproj,
                                  sin_table,
                                  cos_table,
                                  position_id,
                                  slot_mapping,
                                  head_weights,
                                  k_cache,
                                  k_cache_scale,
                                  hadamard_matrix,
                                  interleaved,
                                  gamma,
                                  beta,
                                  eps);
}

}  // namespace xllm::kernel::mlu
