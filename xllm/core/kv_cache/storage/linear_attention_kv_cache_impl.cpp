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

#include "core/kv_cache/storage/linear_attention_kv_cache_impl.h"

#include <limits>

#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/storage/kv_cache_utils.h"
#include "util/tensor_helper.h"

namespace xllm {

LinearAttentionKVCacheImpl::LinearAttentionKVCacheImpl(
    const LinearAttentionKVCacheTensors& tensors)
    : conv_cache_(tensors.conv_cache),
      ssm_cache_(tensors.ssm_cache),
      conv_cache_shape_(get_tensor_shape(tensors.conv_cache)),
      ssm_cache_shape_(get_tensor_shape(tensors.ssm_cache)) {}

LinearAttentionKVCacheImpl::LinearAttentionKVCacheImpl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : LinearAttentionKVCacheImpl(
          create_linear_attention_kv_cache_tensors(kv_cache_shape,
                                                   create_options)) {
  conv_cache_shape_ = kv_cache_shape.conv_cache_shape();
  ssm_cache_shape_ = kv_cache_shape.ssm_cache_shape();
}

LinearAttentionKVCacheImpl::LinearAttentionKVCacheImpl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options,
    BlockType type,
    int64_t layer_count)
    : KVCacheImpl() {
  CHECK(type == BlockType::LINEAR)
      << "LinearAttentionKVCacheImpl host cache only supports "
         "BlockType::LINEAR.";
  const bool has_conv_ssm = kv_cache_shape.has_conv_cache_shape() &&
                            kv_cache_shape.has_ssm_cache_shape();
  CHECK_EQ(kv_cache_shape.has_conv_cache_shape(),
           kv_cache_shape.has_ssm_cache_shape());
  CHECK(has_conv_ssm || kv_cache_shape.has_kpool_tail_shape())
      << "Host LINEAR cache requires paired Conv/SSM tensors or a KPool tail.";
  const int64_t device_slot_count = has_conv_ssm
                                        ? kv_cache_shape.conv_cache_shape()[0]
                                        : kv_cache_shape.kpool_tail_shape()[0];
  const int64_t host_slot_count = scale_host_block_count(
      device_slot_count, create_options.host_blocks_factor());
  CHECK_LE(host_slot_count,
           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "Host LINEAR slot count exceeds the uint32 block-manager limit.";
  host_page_aligned_regions_.reserve((has_conv_ssm ? 2 : 0) +
                                     kv_cache_shape.has_kpool_tail_shape());
  if (has_conv_ssm) {
    const std::vector<int64_t>& device_conv_shape =
        kv_cache_shape.conv_cache_shape();
    const std::vector<int64_t>& device_ssm_shape =
        kv_cache_shape.ssm_cache_shape();
    std::vector<int64_t> host_conv_shape = device_conv_shape;
    host_conv_shape[0] = host_slot_count;
#if defined(USE_MUSA)
    constexpr int64_t kConvHistoryAxis = 2;
#else
    constexpr int64_t kConvHistoryAxis = 1;
#endif
    CHECK_GT(host_conv_shape.size(), kConvHistoryAxis);
    const int64_t conv_history_rows =
        host_conv_shape[kConvHistoryAxis] -
        kv_cache_shape.linear_ssm_checkpoint_stride() + 1;
    CHECK_GT(conv_history_rows, 0);
    // Prefill checkpoints omit speculative Conv/SSM rows.
    host_conv_shape[kConvHistoryAxis] = conv_history_rows;
    host_conv_shape.insert(host_conv_shape.begin() + 1, layer_count);
    create_host_tensor(host_conv_shape,
                       create_options.dtype(),
                       &conv_cache_,
                       &conv_cache_shape_);

    std::vector<int64_t> host_ssm_shape = device_ssm_shape;
    host_ssm_shape[0] = 1;
    host_ssm_shape.insert(host_ssm_shape.begin(), layer_count);
    host_ssm_shape.insert(host_ssm_shape.begin(), host_slot_count);
    create_host_tensor(host_ssm_shape,
                       create_options.ssm_dtype(),
                       &ssm_cache_,
                       &ssm_cache_shape_);
  }
  if (kv_cache_shape.has_kpool_tail_shape()) {
    CHECK_EQ(kv_cache_shape.kpool_tail_shape()[0], device_slot_count);
    std::vector<int64_t> host_tail_shape = kv_cache_shape.kpool_tail_shape();
    host_tail_shape[0] = host_slot_count;
    host_tail_shape.insert(host_tail_shape.begin() + 1, layer_count);
    create_host_tensor(
        host_tail_shape, torch::kBFloat16, &kpool_tail_, &kpool_tail_shape_);
  }
}

torch::Tensor LinearAttentionKVCacheImpl::get_conv_cache() const {
  return conv_cache_;
}

torch::Tensor LinearAttentionKVCacheImpl::get_ssm_cache() const {
  return ssm_cache_;
}

torch::Tensor LinearAttentionKVCacheImpl::get_kpool_tail() const {
  return kpool_tail_;
}

BlockTypeTensorMap LinearAttentionKVCacheImpl::get_block_type_tensors(
    BlockType type) const {
  BlockTypeTensorMap tensor_map;
  if (type != BlockType::LINEAR) {
    return tensor_map;
  }
  if (conv_cache_.defined() && conv_cache_.numel() > 0) {
    tensor_map.emplace(KVCacheTensorRole::CONV, conv_cache_);
  }
  if (ssm_cache_.defined() && ssm_cache_.numel() > 0) {
    tensor_map.emplace(KVCacheTensorRole::SSM, ssm_cache_);
  }
  if (kpool_tail_.defined() && kpool_tail_.numel() > 0) {
    tensor_map.emplace(KVCacheTensorRole::KPOOL_TAIL, kpool_tail_);
  }
  return tensor_map;
}

bool LinearAttentionKVCacheImpl::empty() const {
  return (!conv_cache_.defined() || !ssm_cache_.defined()) &&
         !kpool_tail_.defined();
}

std::vector<std::vector<int64_t>> LinearAttentionKVCacheImpl::get_shapes()
    const {
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(3);
  if (conv_cache_.defined()) {
    shapes.emplace_back(conv_cache_shape_);
  }
  if (ssm_cache_.defined()) {
    shapes.emplace_back(ssm_cache_shape_);
  }
  if (kpool_tail_.defined()) {
    shapes.emplace_back(kpool_tail_shape_);
  }
  return shapes;
}

}  // namespace xllm
