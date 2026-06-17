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

#include <torch/library.h>

#include <string>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

at::Tensor construct_sparse_flash_attention_output_tensor(
    const at::Tensor& query) {
  return at::empty(query.sizes(), query.options().dtype(query.dtype()));
}

void check_sparse_flash_attention_shape_and_dtype(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_indices,
    int64_t sparse_block_size,
    const c10::string_view& layout_query,
    const c10::string_view& layout_kv) {
  TORCH_CHECK(query.dim() >= 1,
              "query's dim num should be at least 1, actual ",
              query.dim(),
              ".");
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
  TORCH_CHECK(!layout_query.empty(), "layout_query should not be empty.");
  TORCH_CHECK(!layout_kv.empty(), "layout_kv should not be empty.");
}

}  // namespace

at::Tensor sparse_flash_attention(
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
    int64_t sparse_mode) {
  check_sparse_flash_attention_shape_and_dtype(query,
                                               key,
                                               value,
                                               sparse_indices,
                                               sparse_block_size,
                                               layout_query,
                                               layout_kv);
  at::Tensor out = construct_sparse_flash_attention_output_tensor(query);

  std::string query_layout_str = std::string(layout_query);
  std::string kv_layout_str = std::string(layout_kv);
  char* query_layout_ptr = const_cast<char*>(query_layout_str.c_str());
  char* kv_layout_ptr = const_cast<char*>(kv_layout_str.c_str());

  EXEC_NPU_CMD(aclnnSparseFlashAttention,
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
               out);

  return out;
}

}  // namespace xllm::kernel::npu
