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

#include <torch/library.h>

#include <string>
#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

constexpr int64_t kDim0 = 0;
constexpr int64_t kDim1 = 1;
constexpr int64_t kDim2 = 2;
constexpr int64_t kDim3 = 3;

void check_sparse_flash_attention_lse_inputs(const at::Tensor& query,
                                             const at::Tensor& key,
                                             const at::Tensor& value,
                                             const at::Tensor& sparse_indices,
                                             int64_t sparse_block_size,
                                             const std::string& layout_query,
                                             const std::string& layout_kv) {
  TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
  TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
  TORCH_CHECK(value.numel() > 0, "Tensor value is empty.");
  TORCH_CHECK(sparse_indices.numel() > 0, "Tensor sparse_indices is empty.");
  TORCH_CHECK(query.dtype() == at::kHalf || query.dtype() == at::kBFloat16,
              "query should be FLOAT16 or BFLOAT16.");
  TORCH_CHECK(key.dtype() == query.dtype(),
              "key's dtype should be equal to query's dtype.");
  TORCH_CHECK(value.dtype() == query.dtype(),
              "value's dtype should be equal to query's dtype.");
  TORCH_CHECK(sparse_indices.dtype() == at::kInt,
              "sparse_indices should be INT32.");
  TORCH_CHECK(sparse_block_size > 0,
              "sparse_block_size should be greater than 0, actual ",
              sparse_block_size,
              ".");
  TORCH_CHECK(layout_query == "BSND" || layout_query == "TND",
              "The layout of query only support BSND and TND, but got ",
              layout_query);
  TORCH_CHECK(!layout_kv.empty(), "layout_kv should not be empty.");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
construct_sparse_flash_attention_lse_outputs(const at::Tensor& query,
                                             const at::Tensor& key,
                                             const std::string& layout_query,
                                             const std::string& layout_kv,
                                             bool return_softmax_lse) {
  at::SmallVector<int64_t, 8> output_size;
  if (layout_query == "TND") {
    TORCH_CHECK(query.dim() == 3,
                "When the layout of query is TND, the query dimension must be "
                "3, but got ",
                query.dim());
    output_size = {query.size(kDim0), query.size(kDim1), query.size(kDim2)};
  } else {
    TORCH_CHECK(query.dim() == 4,
                "When the layout of query is BSND, the query dimension must "
                "be 4, but got ",
                query.dim());
    output_size = {query.size(kDim0),
                   query.size(kDim1),
                   query.size(kDim2),
                   query.size(kDim3)};
  }

  at::Tensor attention_output =
      at::empty(output_size, query.options().dtype(query.dtype()));
  at::SmallVector<int64_t, 8> softmax_size;
  if (return_softmax_lse) {
    if (query.dim() == 3) {
      const auto kv_head_num =
          layout_kv == "PA_BSND" ? key.size(kDim2) : key.size(kDim1);
      softmax_size = {
          kv_head_num,
          query.size(kDim0),
          query.size(kDim1) / kv_head_num,
      };
    } else {
      softmax_size = {
          query.size(kDim0),
          key.size(kDim2),
          query.size(kDim1),
          query.size(kDim2) / key.size(kDim2),
      };
    }
  } else {
    softmax_size = {0};
  }

  at::Tensor softmax_max =
      at::empty(softmax_size, query.options().dtype(at::kFloat));
  at::Tensor softmax_sum =
      at::empty(softmax_size, query.options().dtype(at::kFloat));
  return {attention_output, softmax_max, softmax_sum};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> sparse_flash_attention_lse(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_indices,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& query_rope,
    const c10::optional<at::Tensor>& key_rope,
    double scale_value,
    int64_t sparse_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t attention_mode,
    bool return_softmax_lse) {
  std::string layout_query_str = std::string(layout_query);
  std::string layout_kv_str = std::string(layout_kv);
  check_sparse_flash_attention_lse_inputs(query,
                                          key,
                                          value,
                                          sparse_indices,
                                          sparse_block_size,
                                          layout_query_str,
                                          layout_kv_str);
  auto outputs = construct_sparse_flash_attention_lse_outputs(
      query, key, layout_query_str, layout_kv_str, return_softmax_lse);
  at::Tensor attention_output = std::get<0>(outputs);
  at::Tensor softmax_max = std::get<1>(outputs);
  at::Tensor softmax_sum = std::get<2>(outputs);
  char* query_layout_ptr = const_cast<char*>(layout_query_str.c_str());
  char* kv_layout_ptr = const_cast<char*>(layout_kv_str.c_str());
  EXEC_NPU_CMD(aclnnSparseFlashAttentionLse,
               query,
               key,
               value,
               sparse_indices,
               block_table,
               actual_seq_lengths_query,
               actual_seq_lengths_kv,
               query_rope,
               key_rope,
               scale_value,
               sparse_block_size,
               query_layout_ptr,
               kv_layout_ptr,
               sparse_mode,
               pre_tokens,
               next_tokens,
               attention_mode,
               return_softmax_lse,
               attention_output,
               softmax_max,
               softmax_sum);
  return {attention_output, softmax_max, softmax_sum};
}

}  // namespace xllm::kernel::npu
