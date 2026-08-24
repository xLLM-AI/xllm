/* Copyright 2026 The xLLM Authors.

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

#pragma once

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace xllm::layer {
struct AttentionMetadata;
struct ExpandedDecodeMetadata;
}  // namespace xllm::layer

namespace xllm {

struct ModelInputParams;

void register_attention_metadata_views(pybind11::module_& module);

class PyExpandedDecodeMetadataView final {
 public:
  explicit PyExpandedDecodeMetadataView(
      std::shared_ptr<layer::AttentionMetadata> metadata);

  bool enabled() const;
  pybind11::object kv_seq_lens() const;
  pybind11::object block_table() const;
  pybind11::object paged_kv_indptr() const;
  pybind11::object paged_kv_indices() const;
  pybind11::object paged_kv_last_page_len() const;
  pybind11::object paged_attention_tiling_data() const;
  pybind11::object kv_seq_lens_host() const;
  const std::vector<int32_t>& kv_seq_lens_host_values() const;

 private:
  const layer::ExpandedDecodeMetadata& metadata() const;

  std::shared_ptr<layer::AttentionMetadata> metadata_;
};

class PyAttentionMetadataView final {
 public:
  explicit PyAttentionMetadataView(
      std::shared_ptr<layer::AttentionMetadata> metadata);
  PyAttentionMetadataView(std::shared_ptr<layer::AttentionMetadata> metadata,
                          const ModelInputParams& params);

  const torch::Tensor& slot_mapping() const;
  const torch::Tensor& paged_kv_indptr() const;
  const torch::Tensor& paged_kv_indices() const;
  const torch::Tensor& paged_kv_last_page_len() const;
  pybind11::object qo_indptr() const;
  pybind11::object q_cu_seq_lens() const;
  pybind11::object kv_cu_seq_lens() const;
  pybind11::object kv_seq_lens_host() const;
  const std::vector<int32_t>& kv_seq_lens_host_values() const;
  pybind11::object q_seq_lens_host() const;
  pybind11::list multi_block_tables() const;
  pybind11::object block_table() const;
  pybind11::object kv_seq_lens() const;
  pybind11::object linear_state_indices() const;
  pybind11::object has_initial_state() const;
  const std::vector<int32_t>& dp_token_counts() const;
  const std::vector<int32_t>& dp_is_decode() const;
  pybind11::object q_seq_lens() const;
  PyExpandedDecodeMetadataView expanded_decode_metadata() const;
  int64_t max_query_len() const;
  int64_t max_seq_len() const;
  pybind11::object dsa_metadata() const;
  void set_dsa_metadata(pybind11::object value);
  pybind11::object dsa_positions() const;
  void set_dsa_positions(pybind11::object value);
  pybind11::object dsa_cos_sin() const;
  void set_dsa_cos_sin(pybind11::object value);
  pybind11::object dsa_c4_cos_sin() const;
  void set_dsa_c4_cos_sin(pybind11::object value);
  pybind11::object dsa_c128_cos_sin() const;
  void set_dsa_c128_cos_sin(pybind11::object value);
  int64_t dsa_graph_block_table_cols() const;
  void set_dsa_graph_block_table_cols(int64_t value);
  bool dsa_graph_mode() const;
  void set_dsa_graph_mode(bool value);
  bool is_prefill() const;
  bool is_chunked_prefill() const;

 private:
  static torch::Tensor make_host_int32_view(
      const std::shared_ptr<layer::AttentionMetadata>& metadata,
      std::vector<int32_t>& host_vec);
  static pybind11::object optional_tensor(const torch::Tensor& tensor);

  std::shared_ptr<layer::AttentionMetadata> metadata_;
  torch::Tensor kv_seq_lens_host_;
  torch::Tensor q_seq_lens_host_;
  std::vector<torch::Tensor> multi_block_tables_;
  torch::Tensor linear_state_indices_;
  std::vector<int32_t> dp_token_counts_;
  std::vector<int32_t> dp_is_decode_;
  std::shared_ptr<void> dsa_metadata_holder_;
  torch::Tensor dsa_positions_;
  torch::Tensor dsa_cos_sin_;
  torch::Tensor dsa_c4_cos_sin_;
  torch::Tensor dsa_c128_cos_sin_;
  int64_t dsa_graph_block_table_cols_ = 0;
  bool dsa_graph_mode_ = false;
};

}  // namespace xllm
