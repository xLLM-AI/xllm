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

// Row costs are independent of physical capacity. Logical IDs only bound
// mapping tables; mirrored slab groups supply bytes shared by every cache role.
struct StaticKVCacheLayoutSpec {
  size_t num_slots = 0;
  size_t num_slabs = 0;
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
  size_t view_rows = 0;
  // Layer index among layers with the same attention type.
  size_t lane = 0;
  size_t bank = 0;
  size_t bank_lane = 0;
  // Mapping-table row: two rows per Linear layer, Conv then SSM.
  size_t linear_group = 0;
};

// Fixed banks expose compact dense Full views. A shared slab group contains
// one slice in every bank and changes role only after all mirrored leases end.
// Full layers in a bank own fixed lanes of each object bundle; all Full layers
// therefore consume the same final native page index.
class StaticKVCacheLayout final {
 public:
  explicit StaticKVCacheLayout(StaticKVCacheLayoutSpec spec);
  StaticKVCacheLayout(const KVCacheShape& shape,
                      const KVCacheCreateOptions& options);

  size_t num_slabs() const { return spec_.num_slabs; }
  size_t num_reserved_slabs() const { return num_reserved_slabs_; }
  // Total backing bytes of one mirrored group, not one bank-local slice.
  size_t slab_bytes() const { return slab_bytes_; }
  size_t slice_bytes() const { return slice_bytes_; }
  size_t bank_count() const { return bank_count_; }
  size_t bank_bytes() const { return bank_bytes_; }
  size_t full_layers_per_bank() const { return full_layers_per_bank_; }
  size_t linear_layers_per_bank() const { return linear_layers_per_bank_; }
  size_t full_bundle_bytes() const { return full_bundle_bytes_; }
  size_t full_pages_per_slab() const { return slice_bytes_ / full_row_bytes(); }
  size_t max_slab_count() const;
  size_t objects_per_slab(KVCacheTensorRole role) const;
  size_t layer_bank_index(int64_t layer_id) const;
  size_t scratch_full_page() const { return 0; }
  size_t full_read_padding_page() const { return full_read_padding_row_; }
  size_t num_slots() const { return spec_.num_slots; }
  size_t full_row_bytes() const { return spec_.full_row_bytes; }
  size_t conv_row_bytes() const { return spec_.conv_row_bytes; }
  size_t ssm_row_bytes() const { return spec_.ssm_row_bytes; }
  size_t full_pair_bytes() const { return full_pair_bytes_; }
  size_t ssm_bundle_bytes() const { return ssm_bundle_bytes_; }
  // Shared by every Full layer and never targeted by cache writes. Logical
  // ID zero still maps to writable scratch at row zero for scatter updates.
  size_t full_read_padding_row() const { return full_read_padding_row_; }
  size_t full_layer_count() const { return spec_.full_layer_ids.size(); }
  size_t linear_layer_count() const { return spec_.linear_layer_ids.size(); }
  size_t num_linear_groups() const { return 2 * linear_layer_count(); }
  // Upper bound on native rows per group; Full rows are bank-local.
  size_t rows_per_slab(KVCacheTensorRole role) const;
  size_t full_layer_index(int64_t layer_id) const;
  // Capacity of an empty arena, including typed-slab fragmentation and IDs.
  bool can_fit(size_t full_blocks, size_t linear_states) const;

  size_t page_bytes() const { return slab_bytes(); }
  size_t pages_per_slot() const { return 1; }
  size_t slot_bytes() const { return slab_bytes(); }
  size_t full_page_payload_bytes() const { return full_pair_bytes(); }
  size_t linear_page_payload_bytes() const {
    return spec_.conv_row_bytes + ssm_bundle_bytes();
  }
  size_t row_bytes(KVCacheTensorRole role) const;
  size_t row_index(size_t slab, KVCacheTensorRole role) const;
  size_t role_row_stride(KVCacheTensorRole role) const;
  size_t checkpoint_stride() const { return spec_.checkpoint_stride; }
  size_t bytes_used() const { return bytes_used_; }
  torch::ScalarType dtype(KVCacheTensorRole role) const;

  bool has_layer(int64_t layer_id) const;
  const StaticKVCacheTensorLayout& tensor_layout(int64_t layer_id,
                                                 KVCacheTensorRole role) const;
  // Zero for layers without recurrent state.
  size_t linear_group(int64_t layer_id,
                      KVCacheTensorRole role = KVCacheTensorRole::CONV) const;

 private:
  StaticKVCacheLayoutSpec spec_;
  std::vector<StaticKVCacheTensorLayout> tensors_;
  size_t slab_bytes_ = 0;
  size_t slice_bytes_ = 0;
  size_t bank_count_ = 0;
  size_t bank_bytes_ = 0;
  size_t full_layers_per_bank_ = 0;
  size_t linear_layers_per_bank_ = 0;
  size_t full_bundle_bytes_ = 0;
  size_t full_pair_bytes_ = 0;
  size_t ssm_bundle_bytes_ = 0;
  size_t full_read_padding_row_ = 0;
  size_t num_reserved_slabs_ = 0;
  size_t bytes_used_ = 0;
};

}  // namespace xllm
