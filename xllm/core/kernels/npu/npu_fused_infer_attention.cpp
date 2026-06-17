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
#include "core/kernels/npu/utils.h"

namespace {

constexpr int64_t kSwaIntMax = 2147483647;

torch::Tensor infer_attention_output(
    const torch::Tensor& query,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& block_table,
    int64_t num_heads,
    const std::string& input_layout) {
  if (input_layout == "TND" || input_layout == "NTD") {
    int64_t value_dim = query.size(-1);
    if (!block_table.has_value() && value.dim() >= 3) {
      value_dim = value.size(-1);
    }
    return torch::empty({query.size(0), num_heads, value_dim}, query.options());
  }

  if (input_layout == "BSH") {
    return torch::empty_like(query);
  }

  if (input_layout == "BNSD") {
    int64_t value_dim = query.size(-1);
    if (!block_table.has_value() && value.dim() >= 4) {
      value_dim = value.size(-1);
    }
    return torch::empty(
        {query.size(0), query.size(1), query.size(2), value_dim},
        query.options());
  }

  LOG(FATAL) << "Unsupported FIA input_layout: " << input_layout;
  return torch::Tensor();
}

torch::Tensor infer_softmax_lse(const torch::Tensor& query,
                                int64_t num_heads,
                                const std::string& input_layout,
                                bool softmax_lse_flag) {
  auto options = query.options().dtype(torch::kFloat32);
  if (!softmax_lse_flag) {
    return torch::empty({0}, options);
  }

  if (input_layout == "TND" || input_layout == "NTD") {
    return torch::empty({query.size(0), num_heads, 1}, options);
  }

  if (input_layout == "BSH") {
    return torch::empty({query.size(0), num_heads, query.size(1), 1}, options);
  }

  if (input_layout == "BNSD") {
    return torch::empty({query.size(0), query.size(1), query.size(2), 1},
                        options);
  }

  LOG(FATAL) << "Unsupported FIA input_layout: " << input_layout;
  return torch::Tensor();
}

std::optional<torch::Tensor> to_optional_tensor(
    const std::optional<torch::Tensor>& tensor_opt) {
  if (tensor_opt.has_value() && tensor_opt.value().defined()) {
    return tensor_opt.value();
  }
  return std::nullopt;
}

}  // namespace

namespace xllm::kernel::npu {

std::tuple<torch::Tensor, torch::Tensor> npu_fused_infer_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& atten_mask,
    const std::optional<torch::Tensor>& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale,
    int64_t block_size,
    int64_t sparse_mode,
    const std::string& input_layout,
    bool softmax_lse_flag) {
  check_tensor(query, "query", "npu_fused_infer_attention");
  check_tensor(key, "key", "npu_fused_infer_attention");
  check_tensor(value, "value", "npu_fused_infer_attention");
  CHECK_GT(num_heads, 0) << "num_heads must be positive";
  CHECK(!actual_seq_lengths.empty()) << "actual_seq_lengths must not be empty";
  CHECK(!actual_seq_lengths_kv.empty())
      << "actual_seq_lengths_kv must not be empty";

  torch::Tensor output = infer_attention_output(
      query, value, block_table, num_heads, input_layout);
  torch::Tensor softmax_lse =
      infer_softmax_lse(query, num_heads, input_layout, softmax_lse_flag);

  std::vector<torch::Tensor> key_tensors_vec{key};
  std::vector<torch::Tensor> value_tensors_vec{value};
  torch::TensorList key_tensors(key_tensors_vec);
  torch::TensorList value_tensors(value_tensors_vec);

  std::optional<torch::Tensor> none_tensor = std::nullopt;
  std::optional<torch::Tensor> atten_mask_tensor =
      to_optional_tensor(atten_mask);
  std::optional<torch::Tensor> block_table_tensor =
      to_optional_tensor(block_table);

  torch::IntArrayRef actual_seq_lengths_ref(actual_seq_lengths);
  torch::IntArrayRef actual_seq_lengths_kv_ref(actual_seq_lengths_kv);
  std::optional<torch::IntArrayRef> actual_seq_lengths_opt =
      actual_seq_lengths_ref;
  std::optional<torch::IntArrayRef> actual_seq_lengths_kv_opt =
      actual_seq_lengths_kv_ref;
  std::optional<torch::IntArrayRef> none_int_array = std::nullopt;

  std::string layout = input_layout;
  char* input_layout_ptr = const_cast<char*>(layout.c_str());
  int64_t pre_tokens = kSwaIntMax;
  int64_t next_tokens = 0;
  int64_t inner_precise = 0;
  int64_t antiquant_mode = 0;
  int64_t key_antiquant_mode = 0;
  int64_t value_antiquant_mode = 0;

  EXEC_NPU_CMD(aclnnFusedInferAttentionScoreV3,
               query,
               key_tensors,
               value_tensors,
               none_tensor,  // pse_shift
               atten_mask_tensor,
               actual_seq_lengths_opt,
               actual_seq_lengths_kv_opt,
               none_tensor,  // dequant_scale1
               none_tensor,  // quant_scale1
               none_tensor,  // dequant_scale2
               none_tensor,  // quant_scale2
               none_tensor,  // quant_offset2
               none_tensor,  // antiquant_scale
               none_tensor,  // antiquant_offset
               block_table_tensor,
               none_tensor,     // query_padding_size
               none_tensor,     // kv_padding_size
               none_tensor,     // key_antiquant_scale
               none_tensor,     // key_antiquant_offset
               none_tensor,     // value_antiquant_scale
               none_tensor,     // value_antiquant_offset
               none_tensor,     // key_shared_prefix
               none_tensor,     // value_shared_prefix
               none_int_array,  // actual_shared_prefix_len
               none_tensor,     // query_rope
               none_tensor,     // key_rope
               none_tensor,     // key_rope_antiquant_scale
               num_heads,
               scale,
               pre_tokens,
               next_tokens,
               input_layout_ptr,
               num_key_value_heads,
               sparse_mode,
               inner_precise,
               block_size,
               antiquant_mode,
               softmax_lse_flag,
               key_antiquant_mode,
               value_antiquant_mode,
               output,
               softmax_lse);

  return {output, softmax_lse};
}

}  // namespace xllm::kernel::npu
