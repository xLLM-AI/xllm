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

#include <glog/logging.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"

namespace xllm::kernel::npu {

namespace {
constexpr int64_t kMaskType = 0;
constexpr int64_t kPreTokens = 2147483647;
constexpr int64_t kNextTokens = 2147483647;
constexpr int64_t kBlockSize = 0;
}  // namespace

std::tuple<torch::Tensor, torch::Tensor> npu_block_sparse_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_sparse_mask,
    torch::IntArrayRef block_shape,
    const std::string& q_input_layout,
    const std::string& kv_input_layout,
    int64_t num_key_value_heads,
    double scale_value,
    int64_t inner_precise,
    int64_t softmax_lse_flag,
    std::optional<torch::IntArrayRef> actual_seq_lengths,
    std::optional<torch::IntArrayRef> actual_seq_lengths_kv) {
  CHECK(q_input_layout == "BNSD" || q_input_layout == "TND")
      << "q_input_layout only supports 'BNSD' and 'TND', got "
      << q_input_layout;
  CHECK(kv_input_layout == q_input_layout)
      << "q_input_layout and kv_input_layout must be consistent";

  // TND layout requires actual_seq_lengths for correct batch handling.
  CHECK(q_input_layout != "TND" ||
        (actual_seq_lengths.has_value() && actual_seq_lengths_kv.has_value()))
      << "actual_seq_lengths and actual_seq_lengths_kv are required for TND "
         "layout.";

  const char* q_layout_ptr = q_input_layout.c_str();
  const char* kv_layout_ptr = kv_input_layout.c_str();

  // attenMaskOptional and blockTableOptional: always nullptr.
  std::optional<torch::Tensor> null_tensor = std::nullopt;

  auto attention_out = torch::empty(query.sizes(), query.options());

  // softmaxLse shape depends on layout.
  //   TND:  [T, N, 1]      — batched by token (T).
  //   BNSD: [B, N, S, 1]   — batched by batch (B).
  // When softmax_lse_flag == 0, pass nullptr so the op skips the LSE write.
  torch::Tensor softmax_lse;
  std::optional<torch::Tensor> softmax_lse_opt = std::nullopt;
  if (softmax_lse_flag != 0) {
    if (q_input_layout == "TND") {
      softmax_lse = torch::empty({query.size(0), query.size(1), 1},
                                 query.options().dtype(torch::kFloat));
    } else {
      softmax_lse =
          torch::empty({query.size(0), query.size(1), query.size(2), 1},
                       query.options().dtype(torch::kFloat));
    }
    softmax_lse_opt = softmax_lse;
  }

  EXEC_NPU_CMD(aclnnBlockSparseAttention,
               query,
               key,
               value,
               block_sparse_mask,
               null_tensor,  // attenMaskOptional
               block_shape,
               actual_seq_lengths,
               actual_seq_lengths_kv,
               null_tensor,  // blockTableOptional
               q_layout_ptr,
               kv_layout_ptr,
               num_key_value_heads,
               kMaskType,
               scale_value,
               inner_precise,
               kBlockSize,
               kPreTokens,
               kNextTokens,
               softmax_lse_flag,
               attention_out,
               softmax_lse_opt);  // nullptr when flag==0

  return std::make_tuple(attention_out, softmax_lse);
}

}  // namespace xllm::kernel::npu
