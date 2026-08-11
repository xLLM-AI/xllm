/* Copyright 2026 The xLLM Authors.

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
  pybind11::object block_table() const;
  pybind11::object kv_seq_lens() const;
  pybind11::object linear_state_indices() const;
  pybind11::object has_initial_state() const;
  const std::vector<int32_t>& dp_token_counts() const;
  pybind11::object q_seq_lens() const;
  PyExpandedDecodeMetadataView expanded_decode_metadata() const;
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
  torch::Tensor linear_state_indices_;
  std::vector<int32_t> dp_token_counts_;
};

}  // namespace xllm
