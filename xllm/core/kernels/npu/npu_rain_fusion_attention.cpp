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
constexpr int64_t kBlockSize = 0;
}  // namespace

std::tuple<torch::Tensor, torch::Tensor> npu_rain_fusion_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& select_idx,
    const torch::Tensor& select_num_idx,
    torch::IntArrayRef blockshape,
    const std::string& q_input_layout,
    const std::string& kv_input_layout,
    int64_t head_num,
    double scale,
    int64_t inner_precise,
    std::optional<torch::IntArrayRef> actual_seq_lengths,
    std::optional<torch::IntArrayRef> actual_seq_lengths_kv) {
  CHECK(q_input_layout == "BNSD" || q_input_layout == "TND")
      << "q_input_layout only supports 'BNSD' and 'TND', got "
      << q_input_layout;
  CHECK(kv_input_layout == q_input_layout)
      << "q_input_layout and kv_input_layout must be consistent";

  const char* q_layout_ptr = q_input_layout.c_str();
  const char* kv_layout_ptr = kv_input_layout.c_str();

  // attenMask and blockTable: always nullptr for RainFusion.
  std::optional<torch::Tensor> null_tensor = std::nullopt;

  // actual_seq_lengths: empty IntArrayRef when not set.
  auto opt_seq_lens = actual_seq_lengths.value_or(torch::IntArrayRef{});
  auto opt_seq_lens_kv = actual_seq_lengths_kv.value_or(torch::IntArrayRef{});

  auto attention_out = torch::empty(query.sizes(), query.options());

  // softmaxLse: aclnnRainFusionAttention requires non-null output.
  torch::Tensor softmax_lse;
  if (q_input_layout == "TND") {
    softmax_lse = torch::empty({query.size(0), query.size(1)},
                               query.options().dtype(torch::kFloat));
  } else {
    softmax_lse = torch::empty({query.size(0), query.size(1), query.size(2)},
                               query.options().dtype(torch::kFloat));
  }

  EXEC_NPU_CMD(aclnnRainFusionAttention,
               query,
               key,
               value,
               select_idx,
               select_num_idx,
               blockshape,
               null_tensor,      // attenMaskOptional (nullptr)
               opt_seq_lens,     // actual_seq_lengths
               opt_seq_lens_kv,  // actual_seq_lengths_kv
               null_tensor,      // blockTableOptional (nullptr)
               q_layout_ptr,
               kv_layout_ptr,
               head_num,
               kMaskType,
               scale,
               inner_precise,
               kBlockSize,
               attention_out,
               softmax_lse);

  return std::make_tuple(attention_out, softmax_lse);
}

}  // namespace xllm::kernel::npu
