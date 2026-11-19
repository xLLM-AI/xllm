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
#include <memory>
#include <string>
#include <vector>

#include "core/common/macros.h"
#include "core/kv_cache/storage/kv_tensor_allocator.h"

namespace xllm {

class LinearStateSlotTable;
class StaticKVLayout;

class KVCacheCreateOptions final {
 public:
  PROPERTY(torch::Device, device) = torch::Device(torch::kCPU);
  // kvcache dtype for key/value cache, index cache
  PROPERTY(torch::ScalarType, dtype) = torch::kBFloat16;
  // ssm dtype for linear attention layers
  PROPERTY(torch::ScalarType, ssm_dtype) = torch::kBFloat16;
  PROPERTY(double, host_blocks_factor) = 0.0;
  PROPERTY(int64_t, num_layers) = 0;
  // full attention interval for linear attention layers
  PROPERTY(int64_t, full_attention_interval) = 1;
  // Per-layer attention type ("full_attention"/"attention" vs linear-attention
  // types). When non-empty it drives the linear-layer decision per layer
  // instead of the periodic full_attention_interval.
  PROPERTY(std::vector<std::string>, layer_types) = {};
  // model_id are required for VirtualMemory mode
  PROPERTY(std::string, model_id);
  PROPERTY(std::string, model_type);
  PROPERTY(bool, enable_vmm_kvcache) = false;
  PROPERTY(bool, enable_linear_attention) = false;
  PROPERTY(bool, enable_lighting_indexer) = false;
  // Empty keeps the legacy full allocation. Otherwise false means that the
  // layer holds no blocks of its own and instead views the shared layerwise
  // scratch cache.
  PROPERTY(std::vector<bool>, layer_cache_owned);
  // Empty keeps the legacy all-layer behavior. Otherwise each entry controls
  // whether that layer owns indexer cache tensors.
  PROPERTY(std::vector<bool>, indexer_cache_enabled_layers);
  PROPERTY(bool, enable_kv_cache_quant) = false;
  PROPERTY(std::shared_ptr<KVTensorAllocator>, tensor_allocator);
  PROPERTY(std::shared_ptr<const StaticKVLayout>, static_layout);
  PROPERTY(bool, shared_draft_cache) = false;
  PROPERTY(std::shared_ptr<LinearStateSlotTable>, linear_state_slot_table);
  PROPERTY(std::shared_ptr<LinearStateSlotTable>, full_block_slot_table);

  // Convenience: true when this options set drives the SFA C8 packed KV cache
  // (int8 nope + bf16 rope + fp32 scale in one row, K==V). Every call site was
  // previously spelling out `util::enable_mla_packed_c8(enable_kv_cache_quant,
  // model_type)`; centralize it so the gate cannot drift by call site.
  bool mla_packed_c8() const;
#if defined(USE_NPU)
  PROPERTY(bool, enable_kv_cache_huge_page_allocator) = false;
#endif
  PROPERTY(bool, enable_indexer_cache_quant) = false;

  // DeepSeek V4 cache allocation metadata.
  PROPERTY(int64_t, block_size) = 0;
  PROPERTY(int64_t, head_dim) = 0;
  PROPERTY(int64_t, index_head_dim) = 0;
  PROPERTY(int64_t, window_size) = 0;
  PROPERTY(std::vector<int32_t>, compress_ratios);
  PROPERTY(std::vector<int32_t>, kv_source_layer_ids);
};

}  // namespace xllm
