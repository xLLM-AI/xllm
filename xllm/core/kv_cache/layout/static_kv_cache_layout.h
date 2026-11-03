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

#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/kv_cache/layout/kv_cache_tensor_role.h"

namespace xllm {

class KVCacheShape;
struct KVCacheCreateOptions;

// Row costs are independent of arena capacity so the estimator and storage
// backend can use the same layout without allocating tensors. An SSM row is
// one checkpoint, while a Conv row and a KEY/VALUE row each hold one slot.
struct StaticKVCacheLayoutSpec {
  size_t num_units = 0;
  size_t full_row_bytes = 0;
  size_t conv_row_bytes = 0;
  size_t ssm_row_bytes = 0;
  size_t checkpoint_stride = 1;
  torch::ScalarType dtype = torch::kBFloat16;
  torch::ScalarType ssm_dtype = torch::kFloat32;
  std::vector<int64_t> full_layer_ids;
  std::vector<int64_t> linear_layer_ids;
};

struct StaticKVCacheTensorLayout {
  int64_t layer_id = 0;
  KVCacheTensorRole role = KVCacheTensorRole::KEY;
  size_t offset_bytes = 0;
  size_t row_bytes = 0;
  size_t rows_per_slot = 1;
  size_t extent_units = 0;
  size_t view_rows = 0;
  size_t page_group_index = 0;
};

// Every equal-sized physical page holds four Full layers' K/V blocks or one
// Linear layer's Conv/SSM state. Page size includes padding required by both
// payloads and dense row views. Conv starts at the first dense row wholly
// inside its page; Full and SSM pages retain integral row strides.
class StaticKVCacheLayout final {
 public:
  static constexpr size_t kFullLayersPerPage = 4;

  explicit StaticKVCacheLayout(StaticKVCacheLayoutSpec spec);
  StaticKVCacheLayout(const KVCacheShape& shape,
                      const KVCacheCreateOptions& options);

  size_t full_block_units() const { return full_block_units_; }
  size_t linear_state_units() const { return linear_state_units_; }
  size_t unit_bytes() const { return unit_bytes_; }
  size_t full_page_payload_bytes() const { return full_page_payload_bytes_; }
  size_t linear_page_payload_bytes() const {
    return linear_page_payload_bytes_;
  }
  size_t num_units() const { return spec_.num_units; }
  size_t row_bytes(KVCacheTensorRole role) const;
  size_t row_index(size_t logical_page, KVCacheTensorRole role) const;
  size_t role_row_stride(KVCacheTensorRole role) const;
  size_t checkpoint_stride() const { return spec_.checkpoint_stride; }
  size_t logical_num_slots() const { return spec_.num_units; }
  size_t bytes_used() const { return bytes_used_; }
  torch::ScalarType dtype(KVCacheTensorRole role) const;

  bool has_layer(int64_t layer_id) const;
  const StaticKVCacheTensorLayout& tensor_layout(int64_t layer_id,
                                                 KVCacheTensorRole role) const;

 private:
  StaticKVCacheLayoutSpec spec_;
  std::vector<StaticKVCacheTensorLayout> tensors_;
  size_t unit_bytes_ = 0;
  size_t full_page_payload_bytes_ = 0;
  size_t linear_page_payload_bytes_ = 0;
  size_t full_block_units_ = 0;
  size_t linear_state_units_ = 0;
  size_t bytes_used_ = 0;
};

}  // namespace xllm
