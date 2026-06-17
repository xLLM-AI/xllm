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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

auto get_valid_tensor = [](const c10::optional<at::Tensor>& tensor_opt,
                           at::Device device) {
  return tensor_opt.has_value()
             ? tensor_opt
             : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

}  // namespace

at::Tensor quant_lightning_indexer_metadata(
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    int64_t query_quant_mode,
    int64_t key_quant_mode,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_key,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    const c10::string_view layout_query,
    c10::string_view layout_key,
    int64_t sparse_count,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t cmp_ratio,
    const c10::string_view device) {
  at::Device output_device = at::Device(std::string(device));
  if (actual_seq_lengths_query.has_value()) {
    output_device = actual_seq_lengths_query.value().device();
  } else if (actual_seq_lengths_key.has_value()) {
    output_device = actual_seq_lengths_key.value().device();
  }

  at::Tensor output =
      torch::zeros({kDsaMetadataBufferElements},
                   torch::dtype(torch::kInt32).device(output_device));
  auto actual_seq_lengths_query_val =
      get_valid_tensor(actual_seq_lengths_query, output_device);
  auto actual_seq_lengths_key_val =
      get_valid_tensor(actual_seq_lengths_key, output_device);

  // convert str
  std::string layout_query_str = std::string(layout_query);
  char* layout_query_ptr = const_cast<char*>(layout_query_str.c_str());
  std::string layout_key_str = std::string(layout_key);
  char* layout_key_ptr = const_cast<char*>(layout_key_str.c_str());

  EXEC_NPU_CMD(aclnnQuantLightningIndexerMetadata,
               actual_seq_lengths_query_val,
               actual_seq_lengths_key_val,
               num_heads_q,
               num_heads_k,
               head_dim,
               query_quant_mode,
               key_quant_mode,
               batch_size,
               max_seqlen_q,
               max_seqlen_k,
               layout_query_ptr,
               layout_key_ptr,
               sparse_count,
               sparse_mode,
               pre_tokens,
               next_tokens,
               cmp_ratio,
               output);

  return output;
}

}  // namespace xllm::kernel::npu
