/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <cstdint>
#include <optional>

#include "cnrt.h"
#include "torch/torch.h"

namespace xllm {
namespace kernel {
namespace mlu {

class ChunkGatedDeltaRuleImpl : public torch::nn::Module {
 public:
  // Default chunk size used throughout the computation
  static constexpr int64_t kDefaultChunkSize = 64;
  // Vector block size for h computation
  static constexpr int64_t kBv = 64;

  ChunkGatedDeltaRuleImpl(int64_t num_k_heads, int64_t num_v_heads);
  ~ChunkGatedDeltaRuleImpl() = default;

  std::tuple<torch::Tensor, torch::Tensor> forward(
      torch::Tensor& q,
      torch::Tensor& k,
      torch::Tensor& v,
      torch::Tensor& g,
      torch::Tensor& beta,
      torch::Tensor& initial_state,
      torch::Tensor& cu_seqlens,
      torch::Tensor& chunk_indices,
      bool output_final_state,
      bool use_qk_l2norm_in_kernel = false);

 private:
  // Helper functions
  static inline int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
  }

  static void* get_ptr_or_null(const torch::Tensor& tensor) {
    return tensor.numel() > 0 ? tensor.data_ptr() : nullptr;
  }

  static void* get_ptr_or_null(const std::optional<torch::Tensor>& tensor) {
    return tensor.has_value() && tensor.value().numel() > 0
               ? tensor.value().data_ptr()
               : nullptr;
  }

  cnrtDim3_t compute_grid_dim(int64_t total_chunks) const;

  torch::Tensor prepare_lens(const torch::Tensor& cu_seqlens) const;

  torch::Tensor prepare_chunk_offsets(const torch::Tensor& cu_seqlens,
                                      int64_t chunk_size) const;

  torch::Tensor prepare_chunk_indices(const torch::Tensor& cu_seqlens,
                                      int64_t chunk_size) const;

  torch::Tensor l2norm(const torch::Tensor& x, int64_t dim, double eps = 1e-6) {
    auto denom = x.norm(2.0, dim, true).clamp_min(eps).expand_as(x);
    return x / denom;
  }

  // Core computation functions
  torch::Tensor chunk_local_cumsum(const torch::Tensor& g,
                                   int64_t chunk_size,
                                   bool reverse,
                                   const torch::Tensor& cu_seqlens,
                                   bool head_first,
                                   const torch::Tensor& chunk_indices);

  torch::Tensor chunk_scaled_dot_kkt_fwd(const torch::Tensor& k,
                                         const torch::Tensor& beta,
                                         const torch::Tensor& g,
                                         const torch::Tensor& cu_seqlens,
                                         int64_t chunk_size,
                                         const torch::Tensor& chunk_indices);

  torch::Tensor solve_tril(
      const torch::Tensor& A,
      const std::optional<torch::Tensor>& cu_seqlens,
      const std::optional<torch::ScalarType>& output_dtype);

  std::pair<torch::Tensor, torch::Tensor> recompute_w_fwd(
      const torch::Tensor& k,
      const torch::Tensor& v,
      const torch::Tensor& beta,
      const torch::Tensor& g_cumsum,
      const torch::Tensor& A,
      const torch::Tensor& cu_seqlens,
      const torch::Tensor& chunk_indices);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  chunk_gated_delta_rule_fwd_h(
      const torch::Tensor& k,
      const torch::Tensor& w,
      const torch::Tensor& u,
      const std::optional<torch::Tensor>& g,
      const std::optional<torch::Tensor>& gk,
      const std::optional<torch::Tensor>& initial_state,
      bool output_final_state,
      int64_t chunk_size,
      bool save_new_value,
      const std::optional<torch::Tensor>& cu_seqlens,
      const torch::Tensor& chunk_indices);

  torch::Tensor chunk_fwd_o(const torch::Tensor& q,
                            const torch::Tensor& k,
                            const torch::Tensor& v,
                            const torch::Tensor& h,
                            const std::optional<torch::Tensor>& g,
                            const std::optional<float>& scale,
                            const std::optional<torch::Tensor>& cu_seqlens,
                            int64_t chunk_size,
                            const torch::Tensor& chunk_indices);

  int64_t total_core_num_;
  int64_t num_k_heads_;
  int64_t num_v_heads_;
};

TORCH_MODULE(ChunkGatedDeltaRule);

}  // namespace mlu
}  // namespace kernel
}  // namespace xllm
