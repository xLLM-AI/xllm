/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch/library.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

torch::Tensor construct_lightning_indexer_output_tensor(
    const torch::Tensor& query,
    const torch::Tensor& key,
    int64_t selected_count,
    const std::string& query_layout_str,
    const std::string& key_layout_str) {
  constexpr int64_t DIM_0 = 0;
  constexpr int64_t DIM_1 = 1;
  constexpr int64_t DIM_2 = 2;
  for (size_t i = 0; i < query.sizes().size(); i++) {
    TORCH_CHECK(query.size(i) > 0,
                "All values within query's shape should be greater than 0, "
                "but shape[",
                i,
                "] is ",
                query.size(i));
  }
  for (size_t i = 0; i < key.sizes().size(); i++) {
    TORCH_CHECK(key.size(i) > 0,
                "All values within key's shape should be greater than 0, "
                "but shape[",
                i,
                "] is ",
                key.size(i));
  }
  TORCH_CHECK(selected_count > 0,
              "selected_count should be greater than 0, but now is ",
              selected_count);
  int64_t key_head_num =
      (key_layout_str == "TND") ? key.size(DIM_1) : key.size(DIM_2);
  std::vector<int64_t> output_size;
  if (query_layout_str == "BSND") {
    output_size = {
        query.size(DIM_0), query.size(DIM_1), key_head_num, selected_count};
  } else {
    output_size = {query.size(DIM_0), key_head_num, selected_count};
  }
  return torch::zeros(output_size, query.options().dtype(torch::kInt));
}

}  // namespace

torch::Tensor lightning_indexer(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& weights,
    const c10::optional<torch::Tensor>& query_seq_lengths,
    const c10::optional<torch::Tensor>& key_seq_lengths,
    const c10::optional<torch::Tensor>& block_table,
    c10::string_view layout_query,
    c10::string_view layout_key,
    int64_t selected_count,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool return_value) {
  std::string query_layout_str = std::string(layout_query);
  std::string key_layout_str = std::string(layout_key);

  torch::Tensor sparse_indices_out = construct_lightning_indexer_output_tensor(
      query, key, selected_count, query_layout_str, key_layout_str);

  torch::Tensor sparse_values_out = torch::zeros(
      sparse_indices_out.sizes(), query.options().dtype(torch::kBFloat16));

  char* query_layout_ptr = const_cast<char*>(query_layout_str.c_str());
  char* key_layout_ptr = const_cast<char*>(key_layout_str.c_str());

  EXEC_NPU_CMD(aclnnLightningIndexer,
               query,
               key,
               weights,
               query_seq_lengths,
               key_seq_lengths,
               block_table,
               query_layout_ptr,
               key_layout_ptr,
               selected_count,
               sparse_mode,
               pre_tokens,
               next_tokens,
               return_value,
               sparse_indices_out,
               sparse_values_out);

  return sparse_indices_out;
}

torch::Tensor lightning_indexer_out(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& weights,
    const c10::optional<torch::Tensor>& query_seq_lengths,
    const c10::optional<torch::Tensor>& key_seq_lengths,
    const c10::optional<torch::Tensor>& block_table,
    c10::string_view layout_query,
    c10::string_view layout_key,
    int64_t selected_count,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool return_value,
    torch::Tensor& sparse_indices_out,
    torch::Tensor& sparse_values_out) {
  CHECK(sparse_indices_out.is_contiguous())
      << "sparse_indices_out must be contiguous";
  CHECK(sparse_values_out.is_contiguous())
      << "sparse_values_out must be contiguous";
  CHECK(sparse_indices_out.scalar_type() == torch::kInt)
      << "sparse_indices_out must be int32";
  CHECK(sparse_values_out.scalar_type() == torch::kBFloat16)
      << "sparse_values_out must be bfloat16";

  std::string query_layout_str = std::string(layout_query);
  std::string key_layout_str = std::string(layout_key);
  char* query_layout_ptr = const_cast<char*>(query_layout_str.c_str());
  char* key_layout_ptr = const_cast<char*>(key_layout_str.c_str());

  EXEC_NPU_CMD(aclnnLightningIndexer,
               query,
               key,
               weights,
               query_seq_lengths,
               key_seq_lengths,
               block_table,
               query_layout_ptr,
               key_layout_ptr,
               selected_count,
               sparse_mode,
               pre_tokens,
               next_tokens,
               return_value,
               sparse_indices_out,
               sparse_values_out);
  return sparse_indices_out;
}

}  // namespace xllm::kernel::npu
