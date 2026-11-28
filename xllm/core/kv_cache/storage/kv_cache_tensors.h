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

#include <cstdint>
#include <map>
#include <optional>

#include "core/kv_cache/block/block.h"
#include "core/kv_cache/layout/cache_layout_types.h"
#include "core/kv_cache/layout/kv_cache_tensor_role.h"

namespace xllm {

struct KVCacheTensors {
  torch::Tensor key_cache;
  torch::Tensor value_cache;
};

struct IndexedKVCacheTensors {
  KVCacheTensors kv_cache_tensors;
  torch::Tensor index_cache;
  std::optional<torch::Tensor> index_cache_scale;
  std::optional<torch::Tensor> key_cache_scale;
  std::optional<torch::Tensor> value_cache_scale;
  torch::Tensor kpool_tail;
};

struct QuantizedKVCacheTensors {
  KVCacheTensors kv_cache_tensors;
  torch::Tensor key_cache_scale;
  torch::Tensor value_cache_scale;
};

struct LinearAttentionKVCacheTensors {
  torch::Tensor conv_cache;
  torch::Tensor ssm_cache;
};

struct KVCacheTensor {
  KVCacheTensorRole role;
  torch::Tensor tensor;
  int32_t group_id = cache_group_id(BlockType::KV);
  bool sequence_scoped = false;
  std::optional<LogicalShardDescriptor> shard_descriptor;
};

using BlockTypeTensorMap = std::map<KVCacheTensorRole::Value, torch::Tensor>;

struct DeepSeekV4KVCacheTensors {
  torch::Tensor key_cache;
  torch::Tensor index_cache;
  torch::Tensor indexer_cache_scale;
  torch::Tensor swa_cache;
  torch::Tensor compress_kv_state;
  torch::Tensor compress_score_state;
  torch::Tensor compress_index_kv_state;
  torch::Tensor compress_index_score_state;
#if defined(USE_MLU)
  torch::Tensor compress_state;
  torch::Tensor compress_index_state;
#endif
  BlockType compressed_block_type = BlockType::KV;
  BlockType swa_block_type = BlockType::SWA;
};

}  // namespace xllm
