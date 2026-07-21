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

#include "framework/kv_cache/indexed_kv_cache_impl.h"

#include "framework/kv_cache/kv_cache_shape.h"
#include "util/tensor_helper.h"

namespace xllm {

namespace {

std::vector<int64_t> get_index_cache_shape(
    const IndexedKVCacheTensors& tensors) {
  return get_tensor_shape(tensors.index_cache);
}

std::vector<int64_t> get_index_cache_scale_shape(
    const IndexedKVCacheTensors& tensors) {
  if (tensors.index_cache_scale.has_value()) {
    return get_tensor_shape(tensors.index_cache_scale.value());
  }
  return {};
}

bool has_data(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.numel() > 0;
}

}  // namespace

IndexedKVCacheImpl::IndexedKVCacheImpl(const IndexedKVCacheTensors& tensors)
    : KVCacheImpl(tensors.kv_cache_tensors),
      index_cache_(tensors.index_cache),
      index_cache_scale_(tensors.index_cache_scale),
      key_cache_scale_(tensors.key_cache_scale),
      value_cache_scale_(tensors.value_cache_scale),
      index_cache_shape_(get_index_cache_shape(tensors)),
      index_cache_scale_shape_(get_index_cache_scale_shape(tensors)) {}

IndexedKVCacheImpl::IndexedKVCacheImpl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : IndexedKVCacheImpl(
          create_indexed_kv_cache_tensors(kv_cache_shape, create_options)) {
  key_cache_shape_ = kv_cache_shape.key_cache_shape();
  if (kv_cache_shape.has_value_cache_shape()) {
    value_cache_shape_ = kv_cache_shape.value_cache_shape();
  }
  index_cache_shape_ = kv_cache_shape.index_cache_shape();
  if (kv_cache_shape.has_index_cache_scale_shape()) {
    index_cache_scale_shape_ = kv_cache_shape.index_cache_scale_shape();
  }
}

IndexedKVCacheImpl::IndexedKVCacheImpl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options,
    BlockType type,
    int64_t layer_count)
    : KVCacheImpl() {
  CHECK(type == BlockType::KV)
      << "IndexedKVCacheImpl host cache only supports BlockType::KV.";
  host_page_aligned_regions_.reserve(4);
  if (kv_cache_shape.has_key_cache_shape()) {
    create_host_tensor(
        build_host_group_tensor_shape(kv_cache_shape.key_cache_shape(),
                                      create_options.host_blocks_factor(),
                                      layer_count),
        create_options.dtype(),
        &key_cache_,
        &key_cache_shape_);
  }
  if (kv_cache_shape.has_value_cache_shape()) {
    create_host_tensor(
        build_host_group_tensor_shape(kv_cache_shape.value_cache_shape(),
                                      create_options.host_blocks_factor(),
                                      layer_count),
        create_options.dtype(),
        &value_cache_,
        &value_cache_shape_);
  }
  if (kv_cache_shape.has_index_cache_shape()) {
    // Mirror the device index dtype: INT8 when indexer cache is quantized (see
    // create_indexed_kv_cache_tensors), otherwise the model dtype.
    const torch::ScalarType index_dtype =
        create_options.enable_indexer_cache_quant() ? torch::kChar
                                                    : create_options.dtype();
    create_host_tensor(
        build_host_group_tensor_shape(kv_cache_shape.index_cache_shape(),
                                      create_options.host_blocks_factor(),
                                      layer_count),
        index_dtype,
        &index_cache_,
        &index_cache_shape_);
  }
  // The INT8 indexer cache keeps a per-token fp32 scale that must move with the
  // int8 values during offload/reload, otherwise dequantization reads
  // mismatched coefficients. Allocate it on host alongside the index cache.
  if (create_options.enable_indexer_cache_quant() &&
      kv_cache_shape.has_index_cache_scale_shape()) {
    torch::Tensor index_scale;
    create_host_tensor(
        build_host_group_tensor_shape(kv_cache_shape.index_cache_scale_shape(),
                                      create_options.host_blocks_factor(),
                                      layer_count),
        torch::kFloat32,
        &index_scale,
        &index_cache_scale_shape_);
    index_cache_scale_ = index_scale;
  }
}

torch::Tensor IndexedKVCacheImpl::get_index_cache() const {
  return index_cache_;
}

std::optional<torch::Tensor> IndexedKVCacheImpl::get_k_cache_scale() const {
  if (key_cache_scale_.has_value() && key_cache_scale_->defined() &&
      key_cache_scale_->numel() > 0) {
    return key_cache_scale_;
  }
  return std::nullopt;
}

std::optional<torch::Tensor> IndexedKVCacheImpl::get_v_cache_scale() const {
  if (value_cache_scale_.has_value() && value_cache_scale_->defined() &&
      value_cache_scale_->numel() > 0) {
    return value_cache_scale_;
  }
  return std::nullopt;
}

std::optional<torch::Tensor> IndexedKVCacheImpl::get_indexer_cache_scale()
    const {
  if (index_cache_scale_.has_value() && index_cache_scale_->defined() &&
      index_cache_scale_->numel() > 0) {
    return index_cache_scale_;
  }
  return std::nullopt;
}

BlockTypeTensorMap IndexedKVCacheImpl::get_block_type_tensors(
    BlockType type) const {
  BlockTypeTensorMap tensor_map = KVCacheImpl::get_block_type_tensors(type);
  if (type != BlockType::KV) {
    return tensor_map;
  }
  if (index_cache_.defined() && index_cache_.numel() > 0) {
    tensor_map.emplace(KVCacheTensorRole::INDEX, index_cache_);
  }
  if (index_cache_scale_.has_value() && index_cache_scale_->defined() &&
      index_cache_scale_->numel() > 0) {
    tensor_map.emplace(KVCacheTensorRole::INDEX_SCALE, *index_cache_scale_);
  }
  return tensor_map;
}

bool IndexedKVCacheImpl::empty() const {
  return !key_cache_.defined() || !value_cache_.defined() ||
         !index_cache_.defined();
}

std::vector<std::vector<int64_t>> IndexedKVCacheImpl::get_shapes() const {
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(4);
  shapes.emplace_back(key_cache_shape_);
  shapes.emplace_back(value_cache_shape_);
  shapes.emplace_back(index_cache_shape_);
  shapes.emplace_back(index_cache_scale_shape_);
  return shapes;
}

void IndexedKVCacheImpl::swap_blocks(torch::Tensor& src_tensor,
                                     torch::Tensor& dst_tensor) {
  torch::Tensor selected_keys = torch::index_select(key_cache_, 0, src_tensor);
  key_cache_.index_copy_(0, dst_tensor, selected_keys);

  // deepseek MLA has no value cache.
  if (has_data(value_cache_)) {
    torch::Tensor selected_values =
        torch::index_select(value_cache_, 0, src_tensor);
    value_cache_.index_copy_(0, dst_tensor, selected_values);
  }

  if (key_cache_scale_.has_value() && has_data(key_cache_scale_.value())) {
    torch::Tensor selected_key_scales =
        torch::index_select(key_cache_scale_.value(), 0, src_tensor);
    key_cache_scale_->index_copy_(0, dst_tensor, selected_key_scales);
  }

  if (value_cache_scale_.has_value() && has_data(value_cache_scale_.value())) {
    torch::Tensor selected_value_scales =
        torch::index_select(value_cache_scale_.value(), 0, src_tensor);
    value_cache_scale_->index_copy_(0, dst_tensor, selected_value_scales);
  }

  torch::Tensor selected_index =
      torch::index_select(index_cache_, 0, src_tensor);
  index_cache_.index_copy_(0, dst_tensor, selected_index);

  // INT8 indexer cache keeps a per-token fp32 scale that must move with the
  // int8 values, otherwise dequantization reads mismatched coefficients.
  if (index_cache_scale_.has_value() && has_data(index_cache_scale_.value())) {
    torch::Tensor selected_index_scales =
        torch::index_select(index_cache_scale_.value(), 0, src_tensor);
    index_cache_scale_->index_copy_(0, dst_tensor, selected_index_scales);
  }
}

}  // namespace xllm
