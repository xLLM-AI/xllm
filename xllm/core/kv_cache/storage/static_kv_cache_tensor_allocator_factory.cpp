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

#include <glog/logging.h>

#include <utility>

#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/layout/virtual_kv_cache_page_layout.h"
#include "core/kv_cache/storage/kv_cache_utils.h"
#include "core/kv_cache/storage/static_kv_cache_tensor_allocator.h"

namespace xllm {
namespace {

StaticKVCacheLayoutSpec make_layout_spec(const KVCacheShape& shape,
                                         const KVCacheCreateOptions& options) {
  CHECK(!shape.has_grouped_cache_layout());
  CHECK(!shape.has_index_cache_shape() &&
        !shape.has_index_cache_scale_shape() && !shape.has_kpool_tail_shape())
      << "Static shared storage supports ordinary Full and Conv/SSM caches";
  CHECK(!options.enable_vmm_kvcache());
  CHECK(!options.enable_lighting_indexer());
  CHECK(!options.enable_kv_cache_quant() &&
        !options.enable_indexer_cache_quant());
  CHECK(options.layer_cache_owned().empty());
  CHECK(options.enable_linear_attention());
  CHECK_GT(options.num_layers(), 0);
  CHECK(shape.has_key_cache_shape() && shape.has_value_cache_shape());
  CHECK(shape.has_conv_cache_shape() && shape.has_ssm_cache_shape());
  CHECK_GT(shape.key_cache_shape().front(), 0);

  StaticKVCacheLayoutSpec spec;
  spec.num_units = static_cast<size_t>(shape.key_cache_shape().front());
  spec.dtype = options.dtype();
  spec.ssm_dtype = options.ssm_dtype();
  const size_t dtype_bytes = torch::elementSize(spec.dtype);
  spec.full_row_bytes = VirtualKVCachePageLayout::tensor_block_bytes(
      shape.key_cache_shape(), dtype_bytes);
  CHECK_EQ(spec.full_row_bytes,
           VirtualKVCachePageLayout::tensor_block_bytes(
               shape.value_cache_shape(), dtype_bytes))
      << "Static shared storage requires equal KEY and VALUE row costs";
  spec.conv_row_bytes = VirtualKVCachePageLayout::tensor_block_bytes(
      shape.conv_cache_shape(), dtype_bytes);
  spec.ssm_row_bytes = VirtualKVCachePageLayout::tensor_block_bytes(
      shape.ssm_cache_shape(), torch::elementSize(spec.ssm_dtype));
  CHECK_EQ(shape.key_cache_shape().front(), shape.value_cache_shape().front());
  CHECK_EQ(shape.key_cache_shape().front(), shape.conv_cache_shape().front());
  spec.checkpoint_stride =
      static_cast<size_t>(shape.linear_ssm_checkpoint_stride());
  spec.full_layer_ids.reserve(static_cast<size_t>(options.num_layers()));
  spec.linear_layer_ids.reserve(static_cast<size_t>(options.num_layers()));
  for (int64_t layer_id = 0; layer_id < options.num_layers(); ++layer_id) {
    auto& layers =
        is_linear_attention_layer(
            layer_id, options.full_attention_interval(), options.layer_types())
            ? spec.linear_layer_ids
            : spec.full_layer_ids;
    layers.emplace_back(layer_id);
  }
  return spec;
}

}  // namespace

StaticKVCacheLayout::StaticKVCacheLayout(const KVCacheShape& shape,
                                         const KVCacheCreateOptions& options)
    : StaticKVCacheLayout(make_layout_spec(shape, options)) {}

std::shared_ptr<StaticKVCacheTensorAllocator> create_static_kv_tensor_allocator(
    const KVCacheShape& shape,
    const KVCacheCreateOptions& options) {
  return std::make_shared<StaticKVCacheTensorAllocator>(
      StaticKVCacheLayout(shape, options), options.device());
}

}  // namespace xllm
