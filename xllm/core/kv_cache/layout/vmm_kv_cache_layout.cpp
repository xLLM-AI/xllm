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

#include "core/kv_cache/layout/vmm_kv_cache_layout.h"

#include <glog/logging.h>

#include "core/kv_cache/layout/kv_cache_capacity.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/storage/kv_cache_utils.h"

namespace xllm {

VmmKVCacheLayout::VmmKVCacheLayout(const KVCacheCapacity& capacity,
                                   const ModelArgs& model_args,
                                   int64_t world_size,
                                   torch::ScalarType dtype,
                                   bool is_draft_engine) {
  const KVCacheShape shape(capacity, model_args, world_size);
  CHECK(shape.has_value_cache_shape());
  CHECK_EQ(capacity.scale_slot_size(), 0)
      << "Virtual memory does not support quantized KV cache";
  CHECK(!capacity.enable_mla_kv_cache_quant());
  CHECK(!shape.has_grouped_cache_layout());
  CHECK(!capacity.enable_indexer_cache_quant() &&
        !shape.has_index_cache_scale_shape())
      << "Virtual memory does not support quantized indexer cache";
  CHECK(!model_args.index_kpool_compress())
      << "Virtual memory does not support KPool cache";
#if defined(USE_NPU)
  CHECK(!use_npu_nz_kv_cache_layout(model_args.model_type()))
      << "Virtual memory does not support the NPU NZ KV cache layout.";
#endif
  const KVCacheCreateOptions create_options =
      get_kv_cache_create_options(model_args, shape, is_draft_engine);
  size_t full_attention_layers = 0;
  for (int64_t layer_id = 0; layer_id < capacity.n_layers(); ++layer_id) {
    if (!create_options.enable_linear_attention() ||
        !is_linear_attention_layer(layer_id,
                                   create_options.full_attention_interval(),
                                   create_options.layer_types())) {
      ++full_attention_layers;
    }
  }
  CHECK_GT(full_attention_layers, 0)
      << "Virtual memory requires at least one full-attention layer";
  const size_t element_size = torch::elementSize(dtype);
  tensor_block_bytes_.reserve(3);
  tensor_counts_.reserve(3);
  tensor_block_bytes_.emplace_back(VirtualKVCachePageLayout::tensor_block_bytes(
      shape.key_cache_shape(), element_size));
  tensor_block_bytes_.emplace_back(VirtualKVCachePageLayout::tensor_block_bytes(
      shape.value_cache_shape(), element_size));
  tensor_counts_.assign(2, full_attention_layers);
  if (shape.has_index_cache_shape()) {
    CHECK_EQ(dtype, torch::kBFloat16)
        << "Virtual memory indexer cache requires BF16";
    CHECK_EQ(shape.index_cache_shape().front(),
             shape.key_cache_shape().front());
    const std::vector<bool> indexer_layers =
        resolve_indexer_cache_enabled_layers(model_args, capacity.n_layers());
    size_t indexer_count = 0;
    for (int64_t layer_id = 0; layer_id < capacity.n_layers(); ++layer_id) {
      const size_t layer_index = static_cast<size_t>(layer_id);
      if ((!create_options.enable_linear_attention() ||
           !is_linear_attention_layer(layer_id,
                                      create_options.full_attention_interval(),
                                      create_options.layer_types())) &&
          (indexer_layers.empty() || indexer_layers[layer_index])) {
        ++indexer_count;
      }
    }
    if (indexer_count > 0) {
      tensor_block_bytes_.emplace_back(
          VirtualKVCachePageLayout::tensor_block_bytes(
              shape.index_cache_shape(), element_size));
      tensor_counts_.emplace_back(indexer_count);
    }
  }
}

void VmmKVCacheLayout::append(VmmKVCacheLayout layout) {
  tensor_block_bytes_.reserve(tensor_block_bytes_.size() +
                              layout.tensor_block_bytes_.size());
  tensor_counts_.reserve(tensor_counts_.size() + layout.tensor_counts_.size());
  tensor_block_bytes_.insert(tensor_block_bytes_.end(),
                             layout.tensor_block_bytes_.begin(),
                             layout.tensor_block_bytes_.end());
  tensor_counts_.insert(tensor_counts_.end(),
                        layout.tensor_counts_.begin(),
                        layout.tensor_counts_.end());
}

VirtualKVCachePageLayout VmmKVCacheLayout::page_layout(size_t page_size) const {
  return VirtualKVCachePageLayout(tensor_block_bytes_, page_size);
}

size_t VmmKVCacheLayout::physical_pages(size_t page_size) const {
  return page_layout(page_size).physical_pages(tensor_counts_);
}

size_t VmmKVCacheLayout::block_capacity(size_t physical_page_budget,
                                        size_t page_size) const {
  return page_layout(page_size).block_capacity(physical_page_budget,
                                               tensor_counts_);
}

void VmmKVCacheLayout::validate_pd_page_size(size_t page_size) const {
  for (size_t bytes : tensor_block_bytes_) {
    CHECK_LE(bytes, page_size)
        << "Virtual memory PD requires each tensor block to fit one page";
    CHECK_EQ(page_size % bytes, 0)
        << "Virtual memory PD tensor blocks must divide the page size";
  }
}

}  // namespace xllm
