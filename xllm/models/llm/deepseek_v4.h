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

#pragma once

#include <absl/strings/str_join.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/state_dict/utils.h"
#include "core/kernels/ops_api.h"
#include "core/layers/common/dsa_metadata.h"
#include "core/layers/common/dsa_metadata_builder.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/deepseek_v4_decoder_layer.h"
#include "core/util/tensor_helper.h"
#include "layers/npu/deepseek_v4_rotary_embedding.h"
#include "llm_model_base.h"

namespace xllm {

inline constexpr int64_t kDeepseekV4DsaMetadataBufferElements = 1024;

inline int64_t deepseek_v4_next_power_of_two(int64_t n) {
  int64_t value = 1;
  while (value < n) {
    value <<= 1;
  }
  return value;
}

inline torch::Tensor deepseek_v4_create_hadamard_matrix(
    int64_t n,
    torch::ScalarType dtype,
    const torch::Device& device) {
  auto options = torch::TensorOptions().dtype(dtype).device(device);
  torch::Tensor matrix = torch::ones({1, 1}, options);
  for (int64_t m = 1; m < n; m <<= 1) {
    auto top = torch::cat({matrix, matrix}, 1);
    auto bottom = torch::cat({matrix, -matrix}, 1);
    matrix = torch::cat({top, bottom}, 0);
  }
  return matrix;
}

inline torch::Tensor maybe_to_device(const torch::Tensor& tensor,
                                     const torch::Device& device) {
  if (!tensor.defined() || tensor.device() == device) {
    return tensor;
  }
  return tensor.to(device);
}

inline bool deepseek_v4_uses_acl_graph(
    const xllm::ModelInputParams& input_params) {
#if defined(USE_NPU)
  return ::xllm::ExecutionConfig::get_instance().enable_graph() &&
         input_params.enable_graph;
#else
  (void)input_params;
  return false;
#endif
}

inline size_t deepseek_v4_align_up(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

struct DeepseekV4PackedTensorSpec {
  torch::Tensor cpu_tensor;
  std::vector<torch::Tensor*> targets;
  std::vector<int64_t> sizes;
  torch::ScalarType dtype = torch::kUInt8;
  size_t offset = 0;
  size_t nbytes = 0;
};

inline void deepseek_v4_add_packed_tensor(
    torch::Tensor& tensor,
    const torch::Device& runtime_device,
    std::vector<DeepseekV4PackedTensorSpec>& specs) {
  if (!tensor.defined() || tensor.device() == runtime_device) {
    return;
  }
  if (!tensor.device().is_cpu() || tensor.numel() == 0) {
    tensor = maybe_to_device(tensor, runtime_device);
    return;
  }

  if (tensor.is_contiguous()) {
    const size_t nbytes =
        static_cast<size_t>(tensor.numel() * tensor.element_size());
    for (auto& spec : specs) {
      if (spec.cpu_tensor.data_ptr() == tensor.data_ptr() &&
          spec.nbytes == nbytes && spec.dtype == tensor.scalar_type() &&
          spec.sizes == tensor.sizes().vec()) {
        spec.targets.push_back(&tensor);
        return;
      }
    }
  }

  auto contiguous = tensor.contiguous();
  const size_t nbytes =
      static_cast<size_t>(contiguous.numel() * contiguous.element_size());
  for (auto& spec : specs) {
    if (spec.cpu_tensor.data_ptr() == contiguous.data_ptr() &&
        spec.nbytes == nbytes && spec.dtype == contiguous.scalar_type() &&
        spec.sizes == contiguous.sizes().vec()) {
      spec.targets.push_back(&tensor);
      return;
    }
  }

  DeepseekV4PackedTensorSpec spec;
  spec.cpu_tensor = std::move(contiguous);
  spec.targets.push_back(&tensor);
  spec.sizes = spec.cpu_tensor.sizes().vec();
  spec.dtype = spec.cpu_tensor.scalar_type();
  spec.nbytes = nbytes;
  specs.push_back(std::move(spec));
}

inline void deepseek_v4_collect_cpu_metadata_tensors(
    layer::DSAMetadata& dsa,
    const torch::Device& runtime_device,
    std::vector<DeepseekV4PackedTensorSpec>& specs) {
  deepseek_v4_add_packed_tensor(dsa.seq_lens, runtime_device, specs);
  deepseek_v4_add_packed_tensor(dsa.seq_lens_q, runtime_device, specs);
  deepseek_v4_add_packed_tensor(
      dsa.actual_seq_lengths_query, runtime_device, specs);
  deepseek_v4_add_packed_tensor(
      dsa.actual_seq_lengths_kv, runtime_device, specs);
  deepseek_v4_add_packed_tensor(dsa.kv_cu_seq_lens, runtime_device, specs);
  deepseek_v4_add_packed_tensor(dsa.max_seqlen_q, runtime_device, specs);
  deepseek_v4_add_packed_tensor(dsa.max_seqlen_kv, runtime_device, specs);
  deepseek_v4_add_packed_tensor(dsa.input_positions, runtime_device, specs);
  deepseek_v4_add_packed_tensor(dsa.c4_pad_positions, runtime_device, specs);
  deepseek_v4_add_packed_tensor(dsa.c128_pad_positions, runtime_device, specs);

  for (auto& layer_block_tables : dsa.block_tables) {
    for (auto& block_table : layer_block_tables) {
      deepseek_v4_add_packed_tensor(block_table, runtime_device, specs);
    }
  }
  for (auto& layer_slot_mappings : dsa.slot_mappings) {
    for (auto& slot_mapping : layer_slot_mappings) {
      deepseek_v4_add_packed_tensor(slot_mapping, runtime_device, specs);
    }
  }
  deepseek_v4_add_packed_tensor(dsa.hadamard, runtime_device, specs);
}

inline size_t deepseek_v4_layout_packed_tensor_specs(
    std::vector<DeepseekV4PackedTensorSpec>& specs) {
  static constexpr size_t kAlignment = 64;
  size_t total_bytes = 0;
  for (auto& spec : specs) {
    total_bytes = deepseek_v4_align_up(total_bytes, kAlignment);
    spec.offset = total_bytes;
    total_bytes += spec.nbytes;
  }
  return total_bytes;
}

inline torch::Tensor deepseek_v4_build_packed_host_buffer(
    const std::vector<DeepseekV4PackedTensorSpec>& specs,
    size_t total_bytes) {
  auto host_buffer = torch::empty({static_cast<int64_t>(total_bytes)},
                                  torch::TensorOptions()
                                      .dtype(torch::kUInt8)
                                      .device(torch::kCPU)
                                      .pinned_memory(true));
  auto* host_ptr = static_cast<uint8_t*>(host_buffer.data_ptr());
  for (const auto& spec : specs) {
    std::memcpy(
        host_ptr + spec.offset, spec.cpu_tensor.data_ptr(), spec.nbytes);
  }
  return host_buffer;
}

inline void deepseek_v4_fill_packed_host_buffer(
    const std::vector<DeepseekV4PackedTensorSpec>& specs,
    const torch::Tensor& host_buffer) {
  auto* host_ptr = static_cast<uint8_t*>(host_buffer.data_ptr());
  for (const auto& spec : specs) {
    std::memcpy(
        host_ptr + spec.offset, spec.cpu_tensor.data_ptr(), spec.nbytes);
  }
}

inline void deepseek_v4_bind_packed_tensor_views(
    const std::vector<DeepseekV4PackedTensorSpec>& specs,
    const torch::Tensor& device_buffer) {
  const auto* device_ptr =
      static_cast<const uint8_t*>(device_buffer.data_ptr());
  for (const auto& spec : specs) {
    auto view =
        get_tensor_from_blob(spec.sizes, spec.dtype, device_ptr + spec.offset);
    for (auto* target : spec.targets) {
      *target = view;
    }
  }
}

inline void deepseek_v4_move_dsa_metadata_to_device(
    layer::DSAMetadata& dsa,
    const torch::Device& runtime_device) {
  dsa.seq_lens = maybe_to_device(dsa.seq_lens, runtime_device);
  dsa.seq_lens_q = maybe_to_device(dsa.seq_lens_q, runtime_device);
  dsa.actual_seq_lengths_query =
      maybe_to_device(dsa.actual_seq_lengths_query, runtime_device);
  dsa.actual_seq_lengths_kv =
      maybe_to_device(dsa.actual_seq_lengths_kv, runtime_device);
  dsa.kv_cu_seq_lens = maybe_to_device(dsa.kv_cu_seq_lens, runtime_device);
  dsa.max_seqlen_q = maybe_to_device(dsa.max_seqlen_q, runtime_device);
  dsa.max_seqlen_kv = maybe_to_device(dsa.max_seqlen_kv, runtime_device);
  dsa.input_positions = maybe_to_device(dsa.input_positions, runtime_device);
  dsa.c4_pad_positions = maybe_to_device(dsa.c4_pad_positions, runtime_device);
  dsa.c128_pad_positions =
      maybe_to_device(dsa.c128_pad_positions, runtime_device);

  for (auto& layer_block_tables : dsa.block_tables) {
    for (auto& block_table : layer_block_tables) {
      block_table = maybe_to_device(block_table, runtime_device);
    }
  }
  for (auto& layer_slot_mappings : dsa.slot_mappings) {
    for (auto& slot_mapping : layer_slot_mappings) {
      slot_mapping = maybe_to_device(slot_mapping, runtime_device);
    }
  }

  dsa.hadamard = maybe_to_device(dsa.hadamard, runtime_device);
}

inline void deepseek_v4_pack_dsa_metadata_to_device(
    layer::DSAMetadata& dsa,
    const torch::Device& runtime_device) {
#if defined(USE_NPU)
  if (runtime_device.is_cpu() ||
      runtime_device.type() != c10::DeviceType::PrivateUse1) {
    deepseek_v4_move_dsa_metadata_to_device(dsa, runtime_device);
    return;
  }

  std::vector<DeepseekV4PackedTensorSpec> specs;
  deepseek_v4_collect_cpu_metadata_tensors(dsa, runtime_device, specs);
  const size_t total_bytes = deepseek_v4_layout_packed_tensor_specs(specs);
  if (total_bytes == 0) {
    return;
  }

  auto host_buffer = deepseek_v4_build_packed_host_buffer(specs, total_bytes);
  dsa.packed_metadata_buffer = safe_to(
      host_buffer,
      torch::TensorOptions().dtype(torch::kUInt8).device(runtime_device),
      false);
  deepseek_v4_bind_packed_tensor_views(specs, dsa.packed_metadata_buffer);
#else
  deepseek_v4_move_dsa_metadata_to_device(dsa, runtime_device);
#endif
}

struct DeepseekV4GraphMetadataState : ModelGraphMetadataState {
  struct DSAMetadataPersistent {
    torch::Tensor packed_metadata_host_buffer;
    torch::Tensor packed_metadata_buffer;
    torch::Tensor attn_mask;
    torch::Tensor start_pos;
  };

  DSAMetadataPersistent dsa_metadata_persistent;
};

// Group key: (ratio, type, block_size) -> group_id
class DSAGroupKey {
 public:
  int32_t ratio;
  DSACacheType type;
  int32_t block_size;

  bool operator==(const DSAGroupKey& o) const {
    return ratio == o.ratio && type == o.type && block_size == o.block_size;
  }
};

class DSAGroupKeyHash {
 public:
  size_t operator()(const DSAGroupKey& k) const {
    size_t h = std::hash<int32_t>()(k.ratio);
    h ^= std::hash<int32_t>()(static_cast<int32_t>(k.type)) << 16;
    h ^= std::hash<int32_t>()(k.block_size) << 8;
    return h;
  }
};

inline int32_t deepseek_v4_normalize_compress_ratio(int32_t ratio) {
  return ratio <= 1 ? 1 : ratio;
}

inline void deepseek_v4_build_cache_specs(
    const ModelArgs& model_args,
    std::vector<std::vector<DSACacheInfo>>& caches_info,
    std::vector<DSAGroupInfo>& group_infos) {
  const auto& compress_ratios = model_args.compress_ratios();
  const int32_t window_size = model_args.window_size();
  const int32_t base_block_size = 128;
  CHECK_EQ(KVCacheConfig::get_instance().block_size(), base_block_size)
      << "DeepSeek V4 currently only supports block_size=128.";

  std::unordered_map<DSAGroupKey, int32_t, DSAGroupKeyHash> group_key_map;
  auto register_group =
      [&](DSACacheType type, int32_t ratio, int32_t block_size) -> int32_t {
    DSAGroupKey key{ratio, type, block_size};
    auto it = group_key_map.find(key);
    if (it != group_key_map.end()) {
      return it->second;
    }
    const int32_t gid = static_cast<int32_t>(group_infos.size());
    group_key_map.emplace(key, gid);
    group_infos.push_back({type, ratio, block_size});
    return gid;
  };

  register_group(DSACacheType::SLIDING_WINDOW, 1, window_size);
  for (const int32_t ratio : compress_ratios) {
    const int32_t cr = deepseek_v4_normalize_compress_ratio(ratio);
    if (cr == 4 || cr == 128) {
      register_group(DSACacheType::TOKEN, cr, base_block_size);
    }
  }

  caches_info.resize(model_args.n_layers());
  for (int32_t layer_id = 0; layer_id < model_args.n_layers(); ++layer_id) {
    int32_t cr = (layer_id < static_cast<int32_t>(compress_ratios.size()))
                     ? compress_ratios[layer_id]
                     : 1;
    cr = deepseek_v4_normalize_compress_ratio(cr);

    struct CacheEntry {
      DSACacheType type;
      int32_t ratio;
      int32_t block_size;
    };
    std::vector<CacheEntry> layer_caches;

    if (cr == 1) {
      layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
    } else if (cr == 4) {
      layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
      layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
      layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
    } else if (cr == 128) {
      layer_caches.push_back({DSACacheType::TOKEN, 128, base_block_size});
      layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
    }

    for (const auto& ce : layer_caches) {
      const int32_t gid = register_group(ce.type, ce.ratio, ce.block_size);
      caches_info[layer_id].push_back({gid, ce.type, ce.ratio, ce.block_size});
    }
  }
}

class DeepseekV4ModelImpl
    : public LlmModelImplBase<layer::DeepseekV4DecoderLayer> {
 public:
  explicit DeepseekV4ModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::DeepseekV4DecoderLayer>(
            "deepseek_v4",
            context.get_model_args()) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();
    device_ = options.device();

    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));

    hc_mult_ = model_args.hc_mult();
    hc_eps_ = static_cast<double>(model_args.hc_eps());
    norm_eps_ = static_cast<double>(model_args.rms_norm_eps());

    num_heads_ = model_args.n_heads();
    head_dim_ = model_args.o_lora_rank() + model_args.qk_rope_head_dim();
    dp_local_tp_size_ =
        std::max<int64_t>(parallel_args.world_size() /
                              std::max<int64_t>(parallel_args.dp_size(), 1),
                          1);
    CHECK_EQ(num_heads_ % dp_local_tp_size_, 0)
        << "[DSV4][Init] n_heads must be divisible by local tp size. n_heads="
        << num_heads_ << ", local_tp_size=" << dp_local_tp_size_
        << ", world_size=" << parallel_args.world_size()
        << ", dp_size=" << parallel_args.dp_size();
    tp_num_heads_ = num_heads_ / dp_local_tp_size_;
    window_size_ = model_args.window_size();
    index_n_heads_ = model_args.index_n_heads();
    index_head_dim_ = model_args.index_head_dim();
    index_topk_ = model_args.index_topk();

    const int64_t hc_dim = hc_mult_ * model_args.hidden_size();
    auto hc_options = options.dtype(torch::kFloat32);
    hc_head_fn_ =
        register_parameter("hc_head_fn",
                           torch::empty({hc_mult_, hc_dim}, hc_options),
                           /*requires_grad=*/false);
    hc_head_base_ = register_parameter("hc_head_base",
                                       torch::empty({hc_mult_}, hc_options),
                                       /*requires_grad=*/false);
    hc_head_scale_ = register_parameter("hc_head_scale",
                                        torch::empty({1}, hc_options),
                                        /*requires_grad=*/false);

    const int64_t rope_head_dim = model_args.rope_head_dim();
    const int64_t max_pos = model_args.max_position_embeddings();
    if (rope_head_dim > 0 && max_pos > 0) {
      const int64_t original_max_pos =
          model_args.rope_scaling_original_max_position_embeddings() > 0
              ? model_args.rope_scaling_original_max_position_embeddings()
              : max_pos;
      // DeepSeek V4 DSA rotary only uses YaRN to derive inv_freq.  Keep the
      // generic cache builder's extrapolation/mscale inputs at unit values.
      dsa_rotary_embedding_ =
          std::make_shared<layer::DeepseekV4RotaryEmbedding>(
              /*rotary_dim=*/rope_head_dim,
              /*max_position_embeddings=*/max_pos,
              /*interleaved=*/true,
              /*rope_theta=*/model_args.rope_theta(),
              /*compress_rope_theta=*/model_args.compress_rope_theta(),
              /*scaling_factor=*/model_args.factor(),
              /*extrapolation_factor=*/1.0f,
              /*beta_fast=*/model_args.beta_fast(),
              /*beta_slow=*/model_args.beta_slow(),
              /*attn_factor=*/model_args.rope_scaling_attn_factor(),
              /*mscale=*/1.0f,
              /*mscale_all_dim=*/1.0f,
              /*original_max_position_embeddings=*/original_max_pos,
              options);
      dsa_cos_sin_ = dsa_rotary_embedding_->get_cos_sin_cache("default");
    }

    if (model_args.index_head_dim() > 0) {
      auto hadamard_dim_padded =
          deepseek_v4_next_power_of_two(model_args.index_head_dim());
      dsa_hadamard_ =
          deepseek_v4_create_hadamard_matrix(hadamard_dim_padded,
                                             options.dtype().toScalarType(),
                                             options.device());
    }

    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto layer = layer::DeepseekV4DecoderLayer(context, i);
      layers_.push_back(layer);
    }

    // Build DSA caches_info from compress_ratios
    const auto& compress_ratios = model_args.compress_ratios();
    const int32_t window_size = model_args.window_size();
    // TODO: Wire runtime block_size into model metadata so this stays aligned
    // with the DSv4 KV cache and block manager when the default changes.
    const int32_t base_block_size = 128;  // default block size
    CHECK_EQ(KVCacheConfig::get_instance().block_size(), base_block_size)
        << "DeepSeek V4 currently only supports block_size=128.";

    std::unordered_map<DSAGroupKey, int32_t, DSAGroupKeyHash> group_key_map;
    auto register_group =
        [&](DSACacheType type, int32_t ratio, int32_t block_size) -> int32_t {
      DSAGroupKey key{ratio, type, block_size};
      auto it = group_key_map.find(key);
      if (it != group_key_map.end()) {
        return it->second;
      }
      const int32_t gid = static_cast<int32_t>(group_infos_.size());
      group_key_map.emplace(key, gid);
      group_infos_.push_back({type, ratio, block_size});
      return gid;
    };

    // Keep DSA group ids consistent with BlockManagerPool manager order:
    // 1) sliding-window manager first
    // 2) token managers in first-seen compress_ratio order
    register_group(DSACacheType::SLIDING_WINDOW, 1, window_size);
    for (const auto ratio : compress_ratios) {
      const int32_t cr = deepseek_v4_normalize_compress_ratio(ratio);
      if (cr == 4 || cr == 128) {
        register_group(DSACacheType::TOKEN, cr, base_block_size);
      }
    }

    caches_info_.resize(model_args.n_layers());

    for (int32_t layer_id = 0; layer_id < model_args.n_layers(); ++layer_id) {
      int32_t cr = (layer_id < static_cast<int32_t>(compress_ratios.size()))
                       ? compress_ratios[layer_id]
                       : 1;
      cr = deepseek_v4_normalize_compress_ratio(cr);
      // Build per-layer cache specs based on compress_ratio
      struct CacheEntry {
        DSACacheType type;
        int32_t ratio;
        int32_t block_size;
      };
      std::vector<CacheEntry> layer_caches;

      if (cr == 1) {
        // C1: 1 cache (swa)
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      } else if (cr == 4) {
        // C4: 8 caches
        // compress_kv(TOKEN,4,128), compress_index(TOKEN,4,128),
        // swa(SW,1,window), kv_state(SW,1,window), score_state(SW,1,window),
        // idx_kv_state(SW,1,window), idx_score_state(SW,1,window),
        // indexer_scale(TOKEN,4,128)
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
      } else if (cr == 128) {
        // C128: 4 caches
        // compress_kv(TOKEN,128,128), swa(SW,1,window),
        // kv_state(SW,1,window), score_state(SW,1,window)
        layer_caches.push_back({DSACacheType::TOKEN, 128, base_block_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      }

      for (const auto& ce : layer_caches) {
        const int32_t gid = register_group(ce.type, ce.ratio, ce.block_size);
        caches_info_[layer_id].push_back(
            {gid, ce.type, ce.ratio, ce.block_size});
      }
    }
  }

  void load_state_dict(const StateDict& state_dict) override {
    LlmModelImplBase<layer::DeepseekV4DecoderLayer>::load_state_dict(
        state_dict);
    embed_tokens_->load_state_dict(state_dict.get_dict_with_prefix("embed."));
    LOAD_WEIGHT(hc_head_fn);
    LOAD_WEIGHT(hc_head_base);
    LOAD_WEIGHT(hc_head_scale);
  }

  bool requires_graph_forward_metadata() { return true; }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return std::make_unique<DeepseekV4GraphMetadataState>();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input_params) {
    CHECK(state != nullptr)
        << "DeepSeek V4 graph metadata state must be initialized";
    auto* deepseek_v4_state =
        dynamic_cast<DeepseekV4GraphMetadataState*>(state);
    CHECK(deepseek_v4_state != nullptr)
        << "DeepSeek V4 received incompatible graph metadata state";
    auto modified_input_params = input_params;
    if (modified_input_params.meta.actual_num_sequences == 0) {
      // Graph metadata must keep the bucket-shaped sequence count used during
      // capture/replay. The normal empty-DP fallback intentionally shrinks the
      // request to one dummy token for eager/forward execution; using it here
      // would capture a too-small packed metadata buffer and later replay can
      // exceed that persistent capacity when another DP shard has real tokens.
      if (modified_input_params.meta.num_sequences > 0) {
        fill_empty_dp_rank_graph_metadata_input_params(modified_input_params);
      } else {
        fill_empty_dp_rank_input_params(modified_input_params);
      }
    }
    normalize_graph_metadata_input_params(modified_input_params);
    auto& dp_token_nums = modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    auto attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::DSAMetadataBuilder::build(modified_input_params,
                                         positions,
                                         dsa_cos_sin_,
                                         caches_info_,
                                         group_infos_));
    if (attn_metadata->dsa_metadata) {
      auto& dsa = *attn_metadata->dsa_metadata;
      if (dsa_hadamard_.defined()) {
        dsa.hadamard = dsa_hadamard_;
      }
      copy_to_graph_packed_metadata_buffer(
          dsa, deepseek_v4_state->dsa_metadata_persistent, positions.device());
      prepare_dsa_metadata_for_forward(*attn_metadata,
                                       positions.device(),
                                       /*pack_metadata=*/false,
                                       /*build_rope=*/false);
    }
    input_params.attn_metadata = persist_graph_attention_metadata(
        *deepseek_v4_state, std::move(attn_metadata));
    CHECK(input_params.attn_metadata)
        << "DeepSeek V4 ACL graph requires DSA metadata";
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    torch::NoGradGuard no_grad;
    const bool acl_graph_forward = deepseek_v4_uses_acl_graph(input_params);
    const bool is_graph_empty_dp_rank =
        acl_graph_forward && input_params.meta.actual_num_sequences == 0 &&
        input_params.meta.num_sequences > 0;
    const bool is_empty_dp_rank = input_params.meta.num_sequences == 0 ||
                                  is_graph_empty_dp_rank || !tokens.defined() ||
                                  tokens.numel() == 0;
    if (is_empty_dp_rank && !acl_graph_forward) {
      tokens = torch::tensor(
          {1}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
      positions = torch::tensor(
          {0}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
    }

    auto inputs_embeds = input_params.embedding.input_embedding;
    torch::Tensor h =
        inputs_embeds.defined() ? inputs_embeds : embed_tokens_(tokens);

    if (h.dim() == 2) {
      h = h.unsqueeze(1).repeat({1, hc_mult_, 1});
    }

    // Keep runtime inputs on the same accelerator device.
    const auto runtime_device = h.device();
    if (acl_graph_forward) {
      CHECK(tokens.defined() && tokens.device() == runtime_device)
          << "DeepSeek V4 ACL graph requires tokens on the runtime device";
      CHECK(positions.defined() && positions.device() == runtime_device)
          << "DeepSeek V4 ACL graph requires positions on the runtime device";
      CHECK(input_params.attention.device.new_cache_slots.defined())
          << "DeepSeek V4 ACL graph requires persistent new_cache_slots";
      CHECK(input_params.attention.device.block_tables.defined())
          << "DeepSeek V4 ACL graph requires persistent block_tables";
    } else {
      tokens = maybe_to_device(tokens, runtime_device);
      positions = maybe_to_device(positions, runtime_device);
    }

    auto modified_input_params = input_params;
    if (is_empty_dp_rank && !acl_graph_forward) {
      fill_empty_dp_rank_input_params(modified_input_params, &kv_caches);
    }
    if (acl_graph_forward) {
      normalize_graph_metadata_input_params(modified_input_params);
    }
    auto& dp_token_nums = modified_input_params.parallel.dp_global_token_nums;
    // DP helper: keep zero entries at least 1 to avoid empty slices/padding
    // in xllm DP utilities. DeepSeek V4 not use DP today.
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    if (!modified_input_params.attn_metadata) {
      CHECK(!acl_graph_forward)
          << "DeepSeek V4 ACL graph requires prebuilt attention metadata";
      modified_input_params.attn_metadata =
          build_attention_metadata_for_forward(positions,
                                               modified_input_params);
    }
    auto& attn_metadata = *(modified_input_params.attn_metadata);

    // Per-ratio RoPE for the main q/kv path.  These all use input_positions;
    // c4_cos/c128_cos below are separate compressed-position RoPE tensors.
    std::unordered_map<int32_t, layer::DeepseekV4RotaryEmbedding::CosSinPair>
        input_rope_by_ratio;

    if (attn_metadata.dsa_metadata) {
      auto& dsa = *(attn_metadata.dsa_metadata);
      const bool metadata_prepared =
          dsa.c1_metadata.defined() && dsa.c4_metadata.defined() &&
          dsa.c128_metadata.defined() && dsa.qli_metadata.defined();
      const bool graph_metadata_ready = acl_graph_forward &&
                                        dsa.packed_metadata_buffer.defined() &&
                                        dsa.start_pos.defined();

      if (metadata_prepared || graph_metadata_ready) {
        prepare_forward_dsa_runtime_metadata(
            dsa, modified_input_params, acl_graph_forward, input_rope_by_ratio);
      } else {
        CHECK(!acl_graph_forward)
            << "DeepSeek V4 ACL graph requires prebuilt DSA metadata";
        if (dsa_hadamard_.defined()) {
          dsa.hadamard = dsa_hadamard_;
        }
        deepseek_v4_pack_dsa_metadata_to_device(dsa, runtime_device);

        if (dsa.actual_seq_lengths_kv.defined() && dsa.seq_lens_q.defined()) {
          dsa.start_pos =
              (dsa.actual_seq_lengths_kv - dsa.seq_lens_q).to(torch::kInt32);
        }
        prepare_forward_dsa_runtime_metadata(dsa,
                                             modified_input_params,
                                             /*rebuild_precomputed_metadata=*/
                                             true,
                                             input_rope_by_ratio);
      }
    }

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      if (attn_metadata.dsa_metadata) {
        auto& dsa = *(attn_metadata.dsa_metadata);
        const int32_t layer_id = static_cast<int32_t>(i);
        // Each layer can use a different compression ratio. Read the ratio
        // configured for this layer, or fall back to the uncompressed ratio 1
        // when no per-layer value is available. normalize maps all values <= 1
        // to 1.
        dsa.layer_id = layer_id;
        const int32_t layer_compress_ratio =
            deepseek_v4_normalize_compress_ratio(
                (layer_id <
                 static_cast<int32_t>(model_args_.compress_ratios().size()))
                    ? model_args_
                          .compress_ratios()[static_cast<size_t>(layer_id)]
                    : 1);
        // input_rope_by_ratio stores the main q/kv RoPE cos/sin tensors keyed
        // by compression ratio. Prefer the RoPE tensors matching this layer.
        auto rope_it = input_rope_by_ratio.find(layer_compress_ratio);
        if (rope_it == input_rope_by_ratio.end()) {
          // If no tensors were precomputed for this ratio, fall back to the
          // default uncompressed RoPE to avoid leaving dsa.cos/dsa.sin stale or
          // empty.
          rope_it = input_rope_by_ratio.find(1);
        }
        if (rope_it != input_rope_by_ratio.end()) {
          // Store the RoPE tensors selected for this layer in DSA metadata;
          // the attention kernel reads them from dsa.
          dsa.cos = rope_it->second.first;
          dsa.sin = rope_it->second.second;
        }

        if (layer_id < static_cast<int32_t>(dsa.block_tables.size()) &&
            layer_id < static_cast<int32_t>(dsa.slot_mappings.size()) &&
            !dsa.block_tables[layer_id].empty() &&
            !dsa.slot_mappings[layer_id].empty()) {
          size_t attn_cache_idx = 0;
          if (layer_id < static_cast<int32_t>(caches_info_.size())) {
            const auto& layer_caches = caches_info_[layer_id];
            for (size_t cache_idx = 0; cache_idx < layer_caches.size();
                 ++cache_idx) {
              if (layer_caches[cache_idx].type ==
                  DSACacheType::SLIDING_WINDOW) {
                attn_cache_idx = cache_idx;
                break;
              }
            }
          }

          if (attn_cache_idx < dsa.block_tables[layer_id].size() &&
              dsa.block_tables[layer_id][attn_cache_idx].defined()) {
            attn_metadata.block_table =
                dsa.block_tables[layer_id][attn_cache_idx];
          }
          if (attn_cache_idx < dsa.slot_mappings[layer_id].size() &&
              dsa.slot_mappings[layer_id][attn_cache_idx].defined()) {
            attn_metadata.slot_mapping =
                dsa.slot_mappings[layer_id][attn_cache_idx];
          }
        }
      }

      h = layers_[i](h,
                     residual,
                     positions,
                     attn_metadata,
                     kv_caches[i],
                     modified_input_params,
                     tokens);
#if defined(USE_NPU)
      if (modified_input_params.parallel.layer_synchronizer != nullptr &&
          !modified_input_params.parallel.layer_synchronizer->record_event(
              static_cast<int64_t>(i), runtime_device.index())) {
        return ModelOutput();
      }
#endif
    }
    torch::Tensor pre_hc_head_hidden_states;
    if (model_args_.num_speculative_tokens() > 0) {
      pre_hc_head_hidden_states = h;
    }
    h = hc_head(h);
    auto [hidden_states, residual_out] = norm_(h, std::nullopt);
    if (pre_hc_head_hidden_states.defined()) {
      ModelOutput out(hidden_states, residual_out);
      out.aux_hidden_states = pre_hc_head_hidden_states.flatten(1);
      return out;
    }
    return ModelOutput(hidden_states, residual_out);
  }

 private:
  static c10::optional<torch::Tensor> as_optional_tensor(
      const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.numel() > 0) {
      return c10::optional<torch::Tensor>(tensor);
    }
    return c10::nullopt;
  }

  static c10::optional<torch::Tensor> as_empty_int32_tensor(
      const torch::Tensor& reference) {
    if (!reference.defined()) {
      return c10::nullopt;
    }
    return c10::optional<torch::Tensor>(torch::empty(
        {0}, torch::dtype(torch::kInt32).device(reference.device())));
  }

 private:
  static int64_t tensor_max_or_zero(const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.numel() == 0) {
      return 0;
    }
    return tensor.max().item<int64_t>();
  }

  static int64_t pick_max_seqlen(const torch::Tensor& max_seqlen_tensor,
                                 const torch::Tensor& fallback_tensor) {
    if (max_seqlen_tensor.defined() && max_seqlen_tensor.numel() > 0) {
      return max_seqlen_tensor.max().item<int64_t>();
    }
    return tensor_max_or_zero(fallback_tensor);
  }

 public:
  static bool tensor_aliases_storage(const torch::Tensor& lhs,
                                     const torch::Tensor& rhs) {
    return lhs.defined() && rhs.defined() && lhs.data_ptr() == rhs.data_ptr() &&
           lhs.sizes() == rhs.sizes() && lhs.strides() == rhs.strides();
  }

  static torch::Tensor copy_to_persistent_tensor(const torch::Tensor& src,
                                                 torch::Tensor& dst) {
    if (!src.defined()) {
      return src;
    }
    if (!dst.defined()) {
      dst = torch::empty_like(src);
    } else {
      CHECK_EQ(dst.scalar_type(), src.scalar_type())
          << "DeepSeek V4 graph metadata tensor dtype changed";
      CHECK_EQ(dst.device(), src.device())
          << "DeepSeek V4 graph metadata tensor device changed";
      if (dst.sizes() != src.sizes()) {
        bool can_copy_into_capacity = dst.dim() == src.dim() && src.dim() > 0 &&
                                      src.size(0) <= dst.size(0);
        for (int64_t dim = 1; can_copy_into_capacity && dim < src.dim();
             ++dim) {
          can_copy_into_capacity = dst.size(dim) == src.size(dim);
        }
        CHECK(can_copy_into_capacity)
            << "DeepSeek V4 graph metadata tensor size changed from "
            << dst.sizes() << " to " << src.sizes();
        // ACL graph metadata uses prefix-valid bucket buffers: replay may
        // rebuild src after trimming padding, while dst keeps capture capacity.
        // Copy the valid prefix and clear the padded tail to avoid stale data.
        dst.zero_();
        dst.slice(/*dim=*/0, /*start=*/0, /*end=*/src.size(0))
            .copy_(src, /*non_blocking=*/true);
        return dst;
      }
    }
    if (!tensor_aliases_storage(src, dst)) {
      dst.copy_(src, /*non_blocking=*/true);
    }
    return dst;
  }

  static void copy_to_graph_packed_metadata_buffer(
      layer::DSAMetadata& dsa,
      DeepseekV4GraphMetadataState::DSAMetadataPersistent& persistent,
      const torch::Device& runtime_device) {
#if defined(USE_NPU)
    if (runtime_device.is_cpu() ||
        runtime_device.type() != c10::DeviceType::PrivateUse1) {
      deepseek_v4_move_dsa_metadata_to_device(dsa, runtime_device);
      return;
    }

    std::vector<DeepseekV4PackedTensorSpec> specs;
    deepseek_v4_collect_cpu_metadata_tensors(dsa, runtime_device, specs);
    const size_t total_bytes = deepseek_v4_layout_packed_tensor_specs(specs);
    if (total_bytes == 0) {
      return;
    }

    if (!persistent.packed_metadata_host_buffer.defined() ||
        persistent.packed_metadata_host_buffer.scalar_type() != torch::kUInt8 ||
        persistent.packed_metadata_host_buffer.device() != torch::kCPU ||
        persistent.packed_metadata_host_buffer.numel() <
            static_cast<int64_t>(total_bytes)) {
      persistent.packed_metadata_host_buffer =
          torch::empty({static_cast<int64_t>(total_bytes)},
                       torch::TensorOptions()
                           .dtype(torch::kUInt8)
                           .device(torch::kCPU)
                           .pinned_memory(true));
    }
    auto host_buffer = persistent.packed_metadata_host_buffer.slice(
        /*dim=*/0, /*start=*/0, /*end=*/static_cast<int64_t>(total_bytes));
    deepseek_v4_fill_packed_host_buffer(specs, host_buffer);
    auto device_options =
        torch::TensorOptions().dtype(torch::kUInt8).device(runtime_device);
    if (!persistent.packed_metadata_buffer.defined()) {
      persistent.packed_metadata_buffer =
          torch::empty({static_cast<int64_t>(total_bytes)}, device_options);
    } else {
      CHECK_EQ(persistent.packed_metadata_host_buffer.scalar_type(),
               torch::kUInt8)
          << "DeepSeek V4 graph host packed metadata dtype changed";
      CHECK_EQ(persistent.packed_metadata_host_buffer.device(), torch::kCPU)
          << "DeepSeek V4 graph host packed metadata device changed";
      CHECK_GE(persistent.packed_metadata_host_buffer.numel(),
               static_cast<int64_t>(total_bytes))
          << "DeepSeek V4 graph host packed metadata exceeds persistent "
             "capacity: required="
          << total_bytes
          << ", capacity=" << persistent.packed_metadata_host_buffer.numel();
      CHECK_EQ(persistent.packed_metadata_buffer.scalar_type(), torch::kUInt8)
          << "DeepSeek V4 graph packed metadata dtype changed";
      CHECK_EQ(persistent.packed_metadata_buffer.device(), runtime_device)
          << "DeepSeek V4 graph packed metadata device changed";
      CHECK_GE(persistent.packed_metadata_buffer.numel(),
               static_cast<int64_t>(total_bytes))
          << "DeepSeek V4 graph packed metadata exceeds persistent capacity: "
          << "required=" << total_bytes
          << ", capacity=" << persistent.packed_metadata_buffer.numel();
    }

    persistent.packed_metadata_buffer
        .slice(/*dim=*/0,
               /*start=*/0,
               /*end=*/static_cast<int64_t>(total_bytes))
        .copy_(host_buffer, /*non_blocking=*/true);
    dsa.packed_metadata_buffer = persistent.packed_metadata_buffer.slice(
        /*dim=*/0, /*start=*/0, /*end=*/static_cast<int64_t>(total_bytes));
    deepseek_v4_bind_packed_tensor_views(specs, dsa.packed_metadata_buffer);
#else
    (void)persistent;
    deepseek_v4_move_dsa_metadata_to_device(dsa, runtime_device);
#endif
  }

 private:
  void normalize_graph_metadata_input_params(ModelInputParams& params) const {
    int64_t actual_metadata_rows =
        std::max<int64_t>(params.meta.actual_num_sequences, 0);
    int64_t padded_metadata_rows = actual_metadata_rows;
    if (params.enable_graph) {
      padded_metadata_rows =
          std::max<int64_t>(padded_metadata_rows, params.meta.num_sequences);
    }
    if (padded_metadata_rows <= 0) {
      padded_metadata_rows = 1;
    }
    actual_metadata_rows =
        std::min<int64_t>(actual_metadata_rows, padded_metadata_rows);

    auto trim_lens_vec = [padded_metadata_rows,
                          actual_metadata_rows](std::vector<int32_t>& lens) {
      if (lens.empty()) {
        lens.assign(static_cast<size_t>(padded_metadata_rows), 0);
      } else if (static_cast<int64_t>(lens.size()) < padded_metadata_rows) {
        lens.resize(static_cast<size_t>(padded_metadata_rows), 0);
      } else {
        lens.resize(static_cast<size_t>(padded_metadata_rows));
      }
      std::fill(lens.begin() + actual_metadata_rows, lens.end(), 0);
    };

    // Graph forward tensors are padded to the decode bucket. Build metadata
    // for the same padded row count so compressor/attention inputs agree.
    trim_lens_vec(params.attention.host.kv_seq_lens);
    trim_lens_vec(params.attention.host.q_seq_lens);
    params.meta.num_sequences = static_cast<int32_t>(padded_metadata_rows);
    params.meta.actual_num_sequences =
        static_cast<int32_t>(actual_metadata_rows);
  }

  std::shared_ptr<layer::AttentionMetadata>
  build_attention_metadata_for_forward(const torch::Tensor& positions,
                                       const ModelInputParams& input_params) {
    auto modified_input_params = input_params;
    auto& dp_token_nums = modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    auto attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::DSAMetadataBuilder::build(modified_input_params,
                                         positions,
                                         dsa_cos_sin_,
                                         caches_info_,
                                         group_infos_));
    return attn_metadata;
  }

  std::shared_ptr<layer::AttentionMetadata> persist_graph_attention_metadata(
      DeepseekV4GraphMetadataState& state,
      std::shared_ptr<layer::AttentionMetadata> metadata) const {
    if (!metadata || !metadata->dsa_metadata) {
      return metadata;
    }

    auto& dsa = *metadata->dsa_metadata;
    auto& persistent = state.dsa_metadata_persistent;

    dsa.attn_mask =
        copy_to_persistent_tensor(dsa.attn_mask, persistent.attn_mask);
    dsa.start_pos =
        copy_to_persistent_tensor(dsa.start_pos, persistent.start_pos);

    return metadata;
  }

  void fill_empty_dp_rank_input_params(
      ModelInputParams& params,
      const std::vector<KVCache>* kv_caches = nullptr) const {
    const bool is_chunked_prefill =
        params.meta.batch_forward_type.is_chunked_prefill();
    params.attn_metadata = nullptr;
    auto cpu_int_options = torch::TensorOptions()
                               .dtype(torch::kInt32)
                               .device(torch::kCPU)
                               .pinned_memory(true);
    // The DSA sparse-attention/indexer metadata kernels are configured with a
    // fixed sparse top-k (index_topk_) and sliding-window size. A dummy/empty
    // DP rank that reports a kv length of 1 makes cmp_topk/sparse_count far
    // exceed the kv length, which the SparseAttnSharedkvMetadata /
    // QuantLightningIndexer AICPU kernels reject as invalid parameters. Pad the
    // dummy kv length so it can hold the configured sparse top-k and window.
    const int32_t dummy_kv_len =
        static_cast<int32_t>(std::max<int64_t>({index_topk_, window_size_, 1}));
    params.meta.num_sequences = 1;
    params.meta.actual_num_sequences = 1;
    params.meta.kv_max_seq_len =
        std::max<int32_t>(params.meta.kv_max_seq_len, dummy_kv_len);
    params.meta.q_max_seq_len = 1;
    params.meta.batch_forward_type = is_chunked_prefill
                                         ? BatchForwardType::CHUNKED_PREFILL
                                         : BatchForwardType::DECODE;
    params.attention.host.kv_seq_lens = {dummy_kv_len};
    params.attention.host.q_seq_lens = {1};
    params.attention.host.q_cu_seq_lens = {1};
    params.attention.device.kv_seq_lens =
        torch::tensor(params.attention.host.kv_seq_lens, cpu_int_options);
    params.attention.device.q_seq_lens =
        torch::tensor(params.attention.host.q_seq_lens, cpu_int_options);
    params.attention.device.q_cu_seq_lens =
        torch::tensor(params.attention.host.q_cu_seq_lens, cpu_int_options);
    params.attention.device.kv_cache_tokens_nums =
        torch::tensor({1}, cpu_int_options);
    params.attention.host.kv_cache_tokens_nums = {1};
    params.attention.device.new_cache_slots =
        torch::tensor({0}, cpu_int_options);
    params.attention.device.block_tables =
        torch::zeros({1, 1}, cpu_int_options);

    if (!params.multi_block_tables.empty()) {
      return;
    }

    const int32_t manager_num = static_cast<int32_t>(group_infos_.size());
    params.multi_block_tables.reserve(manager_num);
    for (int32_t manager_id = 0; manager_id < manager_num; ++manager_id) {
      int64_t block_num = 1;
      if (kv_caches != nullptr && !kv_caches->empty()) {
        block_num = empty_dp_rank_cache_blocks_for_group(manager_id,
                                                         kv_caches->front());
      }
      block_num = std::max<int64_t>(block_num, 1);
      params.multi_block_tables.emplace_back(
          torch::zeros({1, block_num}, cpu_int_options));
    }
  }

  void fill_empty_dp_rank_graph_metadata_input_params(
      ModelInputParams& params) const {
    params.attn_metadata = nullptr;
    const int64_t metadata_batch_size =
        std::max<int64_t>(params.meta.num_sequences, 1);
    // See fill_empty_dp_rank_input_params: the dummy kv length must be able to
    // hold the configured sparse top-k / sliding window, otherwise the DSA
    // metadata AICPU kernels reject cmp_topk/sparse_count > kv_len.
    const int32_t dummy_kv_len =
        static_cast<int32_t>(std::max<int64_t>({index_topk_, window_size_, 1}));
    params.meta.num_sequences = static_cast<int32_t>(metadata_batch_size);
    params.meta.kv_max_seq_len =
        std::max<int32_t>(params.meta.kv_max_seq_len, dummy_kv_len);
    params.meta.q_max_seq_len = 1;
    params.meta.batch_forward_type = BatchForwardType::DECODE;

    auto pad_lens_vec = [metadata_batch_size](std::vector<int32_t>& lens,
                                              int32_t fill_value) {
      lens.resize(static_cast<size_t>(metadata_batch_size), fill_value);
      for (int32_t& len : lens) {
        len = std::max<int32_t>(len, fill_value);
      }
    };
    pad_lens_vec(params.attention.host.kv_seq_lens, dummy_kv_len);
    pad_lens_vec(params.attention.host.q_seq_lens, 1);

    const int32_t manager_num = static_cast<int32_t>(group_infos_.size());
    bool has_full_multi_block_tables =
        static_cast<int32_t>(params.multi_block_tables.size()) == manager_num;
    if (has_full_multi_block_tables) {
      for (const auto& block_table : params.multi_block_tables) {
        if (!block_table.defined() || block_table.dim() != 2 ||
            block_table.size(0) < metadata_batch_size) {
          has_full_multi_block_tables = false;
          break;
        }
      }
    }
    if (has_full_multi_block_tables) {
      return;
    }

    auto cpu_int_options = torch::TensorOptions()
                               .dtype(torch::kInt32)
                               .device(torch::kCPU)
                               .pinned_memory(true);
    params.multi_block_tables.clear();
    params.multi_block_tables.reserve(manager_num);
    for (int32_t manager_id = 0; manager_id < manager_num; ++manager_id) {
      params.multi_block_tables.emplace_back(
          torch::zeros({metadata_batch_size, 1}, cpu_int_options));
    }
  }

  int64_t empty_dp_rank_cache_blocks_for_group(int32_t group_id,
                                               const KVCache& kv_cache) const {
    if (group_id < 0 || group_id >= static_cast<int32_t>(group_infos_.size())) {
      return 1;
    }
    const auto& group_info = group_infos_[group_id];
    const int64_t block_size = std::max<int64_t>(group_info.block_size, 1);
    torch::Tensor cache;
    if (group_info.type == DSACacheType::SLIDING_WINDOW) {
      cache = kv_cache.get_swa_cache();
    } else if (group_info.type == DSACacheType::TOKEN) {
      cache = kv_cache.get_k_cache();
    }
    if (!cache.defined() || cache.dim() == 0) {
      return 1;
    }
    return std::max<int64_t>((cache.size(0) + block_size - 1) / block_size, 1);
  }
  void build_dsa_rope_metadata(
      layer::DSAMetadata& dsa,
      std::unordered_map<int32_t, layer::DeepseekV4RotaryEmbedding::CosSinPair>*
          input_rope_by_ratio = nullptr) const {
    if (!dsa_rotary_embedding_) {
      return;
    }

    std::unordered_map<std::string, torch::Tensor> positions_map;
    dsa.cos = torch::Tensor();
    dsa.sin = torch::Tensor();
    dsa.c4_cos = torch::Tensor();
    dsa.c4_sin = torch::Tensor();
    dsa.c128_cos = torch::Tensor();
    dsa.c128_sin = torch::Tensor();
    dsa.c4_input_cos = torch::Tensor();
    dsa.c4_input_sin = torch::Tensor();
    dsa.c128_input_cos = torch::Tensor();
    dsa.c128_input_sin = torch::Tensor();

    auto append_group_positions = [&positions_map](
                                      const std::string& group,
                                      const torch::Tensor& positions) {
      if (!positions.defined() || positions.numel() == 0) {
        return;
      }
      auto group_positions = positions;
      if (group_positions.scalar_type() != torch::kInt64) {
        group_positions = group_positions.to(torch::kInt64);
      }
      positions_map[group] = group_positions;
    };

    append_group_positions("default", dsa.input_positions);
    append_group_positions("c4", dsa.c4_pad_positions);
    append_group_positions("c128", dsa.c128_pad_positions);

    if (!positions_map.empty()) {
      auto group_cos_sin = dsa_rotary_embedding_->build(positions_map);

      auto default_it = group_cos_sin.find("default");
      if (default_it != group_cos_sin.end()) {
        dsa.cos = default_it->second.first;
        dsa.sin = default_it->second.second;
        if (input_rope_by_ratio != nullptr) {
          (*input_rope_by_ratio)[1] = default_it->second;
        }
      }

      auto c4_it = group_cos_sin.find("c4");
      if (c4_it != group_cos_sin.end()) {
        dsa.c4_cos = c4_it->second.first;
        dsa.c4_sin = c4_it->second.second;
      }

      auto c128_it = group_cos_sin.find("c128");
      if (c128_it != group_cos_sin.end()) {
        dsa.c128_cos = c128_it->second.first;
        dsa.c128_sin = c128_it->second.second;
      }
    }

    if (dsa.input_positions.defined() && dsa.input_positions.numel() > 0) {
      auto input_positions = dsa.input_positions;
      if (input_positions.scalar_type() != torch::kInt64) {
        input_positions = input_positions.to(torch::kInt64);
      }
      // C4/C128 layers still apply main q/kv RoPE at input-token length; only
      // the RoPE group changes to use the compressed theta.
      auto input_group_cos_sin = dsa_rotary_embedding_->build(
          {{"c4", input_positions}, {"c128", input_positions}});
      auto c4_input_it = input_group_cos_sin.find("c4");
      if (c4_input_it != input_group_cos_sin.end()) {
        dsa.c4_input_cos = c4_input_it->second.first;
        dsa.c4_input_sin = c4_input_it->second.second;
        if (input_rope_by_ratio != nullptr) {
          (*input_rope_by_ratio)[4] = c4_input_it->second;
        }
      }
      auto c128_input_it = input_group_cos_sin.find("c128");
      if (c128_input_it != input_group_cos_sin.end()) {
        dsa.c128_input_cos = c128_input_it->second.first;
        dsa.c128_input_sin = c128_input_it->second.second;
        if (input_rope_by_ratio != nullptr) {
          (*input_rope_by_ratio)[128] = c128_input_it->second;
        }
      }
    }
  }

  void prepare_forward_dsa_runtime_metadata(
      layer::DSAMetadata& dsa,
      const ModelInputParams& params,
      bool rebuild_precomputed_metadata,
      std::unordered_map<int32_t, layer::DeepseekV4RotaryEmbedding::CosSinPair>&
          input_rope_by_ratio) const {
    build_dsa_rope_metadata(dsa, &input_rope_by_ratio);
    if (rebuild_precomputed_metadata || !dsa.c1_metadata.defined() ||
        !dsa.c4_metadata.defined() || !dsa.c128_metadata.defined() ||
        !dsa.qli_metadata.defined()) {
      build_precomputed_metadata(dsa, params);
    }
  }

  void prepare_dsa_metadata_for_forward(layer::AttentionMetadata& attn_metadata,
                                        const torch::Device& runtime_device,
                                        bool pack_metadata = true,
                                        bool build_rope = true) const {
    if (!attn_metadata.dsa_metadata) {
      return;
    }
    auto& dsa = *(attn_metadata.dsa_metadata);

    if (dsa_hadamard_.defined()) {
      dsa.hadamard = dsa_hadamard_;
    }
    if (pack_metadata) {
      deepseek_v4_pack_dsa_metadata_to_device(dsa, runtime_device);
    }

    if (build_rope) {
      build_dsa_rope_metadata(dsa);
    }

    if (dsa.actual_seq_lengths_kv.defined() && dsa.seq_lens_q.defined()) {
      dsa.start_pos =
          (dsa.actual_seq_lengths_kv - dsa.seq_lens_q).to(torch::kInt32);
    }
  }

  static int64_t vector_max_or_zero(const std::vector<int32_t>& values) {
    if (values.empty()) {
      return 0;
    }
    return *std::max_element(values.begin(), values.end());
  }

  void build_precomputed_metadata(layer::DSAMetadata& dsa,
                                  const ModelInputParams& params) const {
    dsa.c1_metadata = torch::Tensor();
    dsa.c4_metadata = torch::Tensor();
    dsa.c128_metadata = torch::Tensor();
    dsa.qli_metadata = torch::Tensor();

    torch::Device metadata_device(torch::kCPU);
    if (dsa.input_positions.defined()) {
      metadata_device = dsa.input_positions.device();
    } else if (dsa.seq_lens_q.defined()) {
      metadata_device = dsa.seq_lens_q.device();
    } else if (dsa.actual_seq_lengths_kv.defined()) {
      metadata_device = dsa.actual_seq_lengths_kv.device();
    }

    dsa.actual_seq_lengths_query =
        maybe_to_device(dsa.actual_seq_lengths_query, metadata_device);
    dsa.actual_seq_lengths_kv =
        maybe_to_device(dsa.actual_seq_lengths_kv, metadata_device);
    dsa.kv_cu_seq_lens = maybe_to_device(dsa.kv_cu_seq_lens, metadata_device);
    dsa.seq_lens_q = maybe_to_device(dsa.seq_lens_q, metadata_device);
    dsa.seq_lens = maybe_to_device(dsa.seq_lens, metadata_device);
    dsa.max_seqlen_q = maybe_to_device(dsa.max_seqlen_q, metadata_device);
    dsa.max_seqlen_kv = maybe_to_device(dsa.max_seqlen_kv, metadata_device);

    if (!dsa.actual_seq_lengths_query.defined() ||
        !dsa.actual_seq_lengths_kv.defined()) {
      return;
    }

    const int64_t batch_size =
        std::max<int64_t>(dsa.actual_seq_lengths_kv.size(0), 1);
    const int64_t max_seqlen_q =
        std::max<int64_t>(params.meta.q_max_seq_len,
                          vector_max_or_zero(params.attention.host.q_seq_lens));
    const int64_t max_seqlen_kv = std::max<int64_t>(
        params.meta.kv_max_seq_len,
        vector_max_or_zero(params.attention.host.kv_seq_lens));
    const int64_t ori_win_left = std::max<int64_t>(window_size_ - 1, 0);
    const int64_t sparse_topk = std::max<int64_t>(index_topk_, 1);
    const bool is_prefill = params.meta.q_max_seq_len > 1;

    const char* layout_kv = "PA_ND";
    auto empty_int32_opt = as_empty_int32_tensor(dsa.actual_seq_lengths_query);
    auto cu_seqlens_ori_kv_opt =
        is_prefill ? as_optional_tensor(dsa.actual_seq_lengths_query)
                   : empty_int32_opt;

    xllm::kernel::SparseAttnSharedkvMetadataParams c1_params;
    c1_params.num_heads_q = tp_num_heads_;
    c1_params.num_heads_kv = 1;
    c1_params.head_dim = head_dim_;
    c1_params.cu_seqlens_q = as_optional_tensor(dsa.actual_seq_lengths_query);
    c1_params.cu_seqlens_ori_kv = cu_seqlens_ori_kv_opt;
    c1_params.cu_seqlens_cmp_kv = empty_int32_opt;
    c1_params.seqused_q = empty_int32_opt;
    c1_params.seqused_kv = as_optional_tensor(dsa.actual_seq_lengths_kv);
    c1_params.batch_size = batch_size;
    c1_params.max_seqlen_q = max_seqlen_q;
    c1_params.max_seqlen_kv = max_seqlen_kv;
    c1_params.ori_topk = 0;
    c1_params.cmp_topk = 0;
    c1_params.cmp_ratio = 1;
    c1_params.ori_mask_mode = 4;
    c1_params.cmp_mask_mode = 3;
    c1_params.ori_win_left = ori_win_left;
    c1_params.ori_win_right = 0;
    c1_params.layout_q = "TND";
    c1_params.layout_kv = layout_kv;
    c1_params.has_ori_kv = true;
    c1_params.has_cmp_kv = false;
    dsa.c1_metadata = xllm::kernel::sparse_attn_sharedkv_metadata(c1_params);

    xllm::kernel::SparseAttnSharedkvMetadataParams c4_params;
    c4_params.num_heads_q = tp_num_heads_;
    c4_params.num_heads_kv = 1;
    c4_params.head_dim = head_dim_;
    c4_params.cu_seqlens_q = as_optional_tensor(dsa.actual_seq_lengths_query);
    c4_params.cu_seqlens_ori_kv = cu_seqlens_ori_kv_opt;
    c4_params.cu_seqlens_cmp_kv = empty_int32_opt;
    c4_params.seqused_q = empty_int32_opt;
    c4_params.seqused_kv = as_optional_tensor(dsa.actual_seq_lengths_kv);
    c4_params.batch_size = batch_size;
    c4_params.max_seqlen_q = max_seqlen_q;
    c4_params.max_seqlen_kv = max_seqlen_kv;
    c4_params.ori_topk = 0;
    c4_params.cmp_topk = sparse_topk;
    c4_params.cmp_ratio = 4;
    c4_params.ori_mask_mode = 4;
    c4_params.cmp_mask_mode = 3;
    c4_params.ori_win_left = ori_win_left;
    c4_params.ori_win_right = 0;
    c4_params.layout_q = "TND";
    c4_params.layout_kv = layout_kv;
    c4_params.has_ori_kv = true;
    c4_params.has_cmp_kv = true;
    dsa.c4_metadata = xllm::kernel::sparse_attn_sharedkv_metadata(c4_params);

    xllm::kernel::SparseAttnSharedkvMetadataParams c128_params;
    c128_params.num_heads_q = tp_num_heads_;
    c128_params.num_heads_kv = 1;
    c128_params.head_dim = head_dim_;
    c128_params.cu_seqlens_q = as_optional_tensor(dsa.actual_seq_lengths_query);
    c128_params.cu_seqlens_ori_kv = cu_seqlens_ori_kv_opt;
    c128_params.cu_seqlens_cmp_kv = empty_int32_opt;
    c128_params.seqused_q = empty_int32_opt;
    c128_params.seqused_kv = as_optional_tensor(dsa.actual_seq_lengths_kv);
    c128_params.batch_size = batch_size;
    c128_params.max_seqlen_q = max_seqlen_q;
    c128_params.max_seqlen_kv = max_seqlen_kv;
    c128_params.ori_topk = 0;
    c128_params.cmp_topk = 0;
    c128_params.cmp_ratio = 128;
    c128_params.ori_mask_mode = 4;
    c128_params.cmp_mask_mode = 3;
    c128_params.ori_win_left = ori_win_left;
    c128_params.ori_win_right = 0;
    c128_params.layout_q = "TND";
    c128_params.layout_kv = layout_kv;
    c128_params.has_ori_kv = true;
    c128_params.has_cmp_kv = true;
    dsa.c128_metadata =
        xllm::kernel::sparse_attn_sharedkv_metadata(c128_params);

    torch::Tensor query_lens;
    if (dsa.actual_seq_lengths_query.defined() &&
        dsa.actual_seq_lengths_query.dim() > 0 &&
        dsa.actual_seq_lengths_query.size(0) > 1) {
      query_lens = dsa.actual_seq_lengths_query
                       .slice(
                           /*dim=*/0,
                           /*start=*/1,
                           /*end=*/dsa.actual_seq_lengths_query.size(0))
                       .clone();
    } else if (dsa.seq_lens_q.defined()) {
      query_lens = dsa.seq_lens_q;
    }

    torch::Tensor key_lens;
    if (dsa.seq_lens.defined()) {
      key_lens = dsa.seq_lens;
    } else if (dsa.actual_seq_lengths_kv.defined()) {
      key_lens = dsa.actual_seq_lengths_kv;
    }

    if (!query_lens.defined() || !key_lens.defined() ||
        query_lens.numel() == 0 || key_lens.numel() == 0) {
      return;
    }

    const int64_t global_index_num_heads =
        std::max<int64_t>(index_n_heads_ > 0 ? index_n_heads_ : num_heads_, 1);
    CHECK_EQ(global_index_num_heads % dp_local_tp_size_, 0)
        << "[DSV4][PrecomputeMetadata] index/global heads must be divisible by "
           "local tp size. global_index_num_heads="
        << global_index_num_heads << ", local_tp_size=" << dp_local_tp_size_;
    const int64_t qli_batch_size = std::max<int64_t>(key_lens.size(0), 1);
    const int64_t index_head_dim =
        std::max<int64_t>(index_head_dim_ > 0 ? index_head_dim_ : head_dim_, 1);
    const int64_t qli_max_seqlen_q =
        std::max<int64_t>(params.meta.q_max_seq_len,
                          vector_max_or_zero(params.attention.host.q_seq_lens));
    const int64_t qli_max_seqlen_k = std::max<int64_t>(
        params.meta.kv_max_seq_len,
        vector_max_or_zero(params.attention.host.kv_seq_lens));

    xllm::kernel::QuantLightningIndexerMetadataParams qli_params;
    qli_params.num_heads_q = global_index_num_heads;
    qli_params.num_heads_k = 1;
    qli_params.head_dim = index_head_dim;
    qli_params.query_quant_mode = 0;
    qli_params.key_quant_mode = 0;
    qli_params.actual_seq_lengths_query = as_optional_tensor(query_lens);
    qli_params.actual_seq_lengths_key = as_optional_tensor(key_lens);
    qli_params.batch_size = qli_batch_size;
    qli_params.max_seqlen_q = qli_max_seqlen_q;
    qli_params.max_seqlen_k = qli_max_seqlen_k;
    qli_params.layout_query = "TND";
    qli_params.layout_key = "PA_BSND";
    qli_params.sparse_count = sparse_topk;
    qli_params.sparse_mode = 3;
    qli_params.pre_tokens = std::numeric_limits<int64_t>::max();
    qli_params.next_tokens = std::numeric_limits<int64_t>::max();
    qli_params.cmp_ratio = 4;
    qli_params.device = query_lens.device().str();
    dsa.qli_metadata =
        xllm::kernel::quant_lightning_indexer_metadata(qli_params);
  }

  torch::Tensor hc_head(const torch::Tensor& x) {
    auto x_float = x.to(torch::kFloat32);
    auto x_flatten = x_float.flatten(-2, -1);
    auto rsqrt = torch::rsqrt(x_flatten.pow(2).mean(-1, true) + norm_eps_);
    auto mixes = torch::matmul(x_flatten, hc_head_fn_.transpose(0, 1));
    mixes = mixes * rsqrt;
    auto pre = torch::sigmoid(mixes * hc_head_scale_ + hc_head_base_) + hc_eps_;
    auto y = (pre.unsqueeze(-1) * x_float).sum(-2);
    return y.to(x.dtype());
  }

  torch::Tensor dsa_cos_sin_;
  torch::Tensor dsa_hadamard_;
  std::shared_ptr<layer::DeepseekV4RotaryEmbedding> dsa_rotary_embedding_;

  int64_t hc_mult_ = 1;
  double hc_eps_ = 0.0;
  double norm_eps_ = 1e-6;

  int64_t num_heads_ = 0;
  int64_t tp_num_heads_ = 0;
  int64_t dp_local_tp_size_ = 1;
  int64_t head_dim_ = 0;
  int64_t window_size_ = 128;
  int64_t index_n_heads_ = 0;
  int64_t index_head_dim_ = 0;
  int64_t index_topk_ = 512;
  torch::Device device_{torch::kCPU};

  // DSA cache group info: built once at model init from compress_ratios
  // caches_info_[layer_id] = vector of DSACacheInfo for each cache in that
  // layer
  std::vector<std::vector<DSACacheInfo>> caches_info_;
  // group_infos_[group_id] = DSAGroupInfo
  std::vector<DSAGroupInfo> group_infos_;

  DEFINE_WEIGHT(hc_head_fn);
  DEFINE_WEIGHT(hc_head_base);
  DEFINE_WEIGHT(hc_head_scale);
};
TORCH_MODULE(DeepseekV4Model);

class DeepseekV4ForCausalLMImpl
    : public LlmForCausalLMImplBase<DeepseekV4Model> {
 public:
  explicit DeepseekV4ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV4Model>(context) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    LlmForCausalLMImplBase<DeepseekV4Model>::load_model(std::move(loader),
                                                        std::move(prefix));
  }

  bool requires_graph_forward_metadata() {
    return this->model_->requires_graph_forward_metadata();
  }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return this->model_->create_graph_forward_metadata_state();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input_params) {
    this->model_->prepare_graph_forward_metadata(
        state, positions, input_params);
  }
};
TORCH_MODULE(DeepseekV4ForCausalLM);

inline void load_deepseek_v4_model_args(const JsonReader& json,
                                        ModelArgs* args) {
  // --------------------------------------------------------------------------
  // Base args (shared/common HF CausalLM args).
  // NOTE: These are intentionally kept here even if they are not DSv4-specific.
  // --------------------------------------------------------------------------
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v4");
  LOAD_ARG_OR(dtype, "torch_dtype", "");

  // Basic model structure
  LOAD_ARG_OR_FUNC(hidden_size, "dim", [&] { return args->hidden_size(); });
  LOAD_ARG_OR_FUNC(
      hidden_size, "hidden_size", [&] { return args->hidden_size(); });
  LOAD_ARG_OR_FUNC(
      n_layers, "num_hidden_layers", [&] { return args->n_layers(); });
  LOAD_ARG_OR_FUNC(n_heads, "n_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR_FUNC(
      n_heads, "num_attention_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR(o_lora_rank, "o_lora_rank", 1024);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1024);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 1);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    if (args->head_dim() > 0) {
      return args->head_dim();
    }
    if (args->hidden_size() > 0 && args->n_heads() > 0) {
      return args->hidden_size() / args->n_heads();
    }
    return int64_t{0};
  });
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR_FUNC(intermediate_size, "intermediate_size", [&] {
    if (args->intermediate_size() > 0) {
      return args->intermediate_size();
    }
    if (args->moe_intermediate_size() > 0) {
      return static_cast<int64_t>(args->moe_intermediate_size());
    }
    if (args->hidden_size() > 0) {
      return args->hidden_size() * 4;
    }
    return int64_t{0};
  });

  // Norm / RoPE
  LOAD_ARG_OR_FUNC(rms_norm_eps, "rms_norm_eps", [&] {
    return json.value_or<float>("norm_eps", 1e-6f);
  });

  LOAD_ARG_OR_FUNC(
      rope_theta, "rope_theta", [&] { return args->rope_theta(); });
  LOAD_ARG_OR(rope_head_dim, "qk_rope_head_dim", 64);

  // LoRA / groups
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1024);
  LOAD_ARG_OR(o_lora_rank, "o_lora_rank", 1024);
  LOAD_ARG_OR(o_groups, "o_groups", 8);

  // --------------------------------------------------------------------------
  // DeepSeek V4 args.
  // --------------------------------------------------------------------------
  // KV compression / windowing
  LOAD_ARG(compress_ratios, "compress_ratios");
  LOAD_ARG_OR(compress_rope_theta, "compress_rope_theta", 160000.0f);
  LOAD_ARG_OR(window_size, "window_size", 128);

  // MoE routing (DeepSeek V4)
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_activated_experts, "n_activated_experts", 6);
  LOAD_ARG_OR_FUNC(num_experts_per_tok, "num_experts_per_tok", [&] {
    return args->n_activated_experts();
  });
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(n_hash_layers, "num_hash_layers", 3);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 1.5f);
  LOAD_ARG_OR(scoring_func, "scoring_func", "sqrtsoftplus");
  LOAD_ARG_OR(swiglu_limit, "swiglu_limit", 10.0f);

  // Indexer
  LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "index_n_heads", 64);
  LOAD_ARG_OR(index_topk, "index_topk", 512);

  // HC / DSA helpers
  LOAD_ARG_OR(hc_mult, "hc_mult", 4);
  LOAD_ARG_OR(hc_sinkhorn_iters, "hc_sinkhorn_iters", 20);
  LOAD_ARG_OR(hc_eps, "hc_eps", 1e-6f);
  LOAD_ARG_OR(factor, "rope_scaling.factor", 16.0f);
  LOAD_ARG_OR(beta_fast, "rope_scaling.beta_fast", 32.0f);
  LOAD_ARG_OR(beta_slow, "rope_scaling.beta_slow", 1.0f);
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);
  LOAD_ARG_OR(scale_fmt, "scale_fmt", "ue8m0");

  // Runtime sizing hints
  LOAD_ARG_OR_FUNC(
      max_batch_size, "max_batch_size", [&] { return args->max_batch_size(); });
  LOAD_ARG_OR_FUNC(
      max_seq_len, "max_seq_len", [&] { return args->max_seq_len(); });

  LOAD_ARG_OR_FUNC(
      vocab_size, "vocab_size", [&] { return args->vocab_size(); });
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 163840);

  // Token ids
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
}

struct DeepseekV4ArgsPolicy {
  std::unordered_set<int32_t> supported_compress_ratios;
  std::unordered_set<std::string> supported_score_funcs;
  int32_t default_compress_ratio = 1;
};

enum class DeepseekV4PolicyPreset {
  kDefault,
};

inline DeepseekV4ArgsPolicy build_deepseek_v4_args_policy(
    DeepseekV4PolicyPreset preset) {
  DeepseekV4ArgsPolicy policy;
  switch (preset) {
    case DeepseekV4PolicyPreset::kDefault:
      policy.supported_compress_ratios = {0, 1, 4, 128};
      policy.supported_score_funcs = {"softmax", "sigmoid", "sqrtsoftplus"};
      policy.default_compress_ratio = 1;
      return policy;
  }
  return policy;
}

inline void process_deepseek_v4_args(ModelArgs* args,
                                     const DeepseekV4ArgsPolicy& policy) {
  // Keep alias fields consistent after loading.
  SET_ARG(n_activated_experts, args->num_experts_per_tok());

  // Align with MindIE behavior: missing tail ratios default to non-compressed.
  if (args->n_layers() > 0 &&
      static_cast<int64_t>(args->compress_ratios().size()) < args->n_layers()) {
    args->compress_ratios().resize(static_cast<size_t>(args->n_layers()),
                                   policy.default_compress_ratio);
  }

  // Build stop token set from eos for runtime usage.
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
}

inline void normalize_deepseek_v4_args(ModelArgs* args) {
  // Align with vLLM branch semantics: only ratios > 1 use compressed path.
  // Treat non-positive/one ratios as non-compressed (ratio=1).
  if (args->n_layers() > 0) {
    auto& ratios = args->compress_ratios();
    for (int64_t i = 0; i < args->n_layers(); ++i) {
      ratios[static_cast<size_t>(i)] =
          deepseek_v4_normalize_compress_ratio(ratios[static_cast<size_t>(i)]);
    }
  }
}

inline void validate_deepseek_v4_args(const ModelArgs& args,
                                      const DeepseekV4ArgsPolicy& policy) {
  CHECK(!policy.supported_compress_ratios.empty())
      << "deepseek_v4 internal supported_compress_ratios must not be empty";
  CHECK(policy.supported_compress_ratios.count(policy.default_compress_ratio) >
        0)
      << "deepseek_v4 config default_compress_ratio("
      << policy.default_compress_ratio
      << ") must exist in supported_compress_ratios";
  CHECK_GT(args.n_layers(), 0)
      << "deepseek_v4 config n_layers/num_hidden_layers must be > 0, got "
      << args.n_layers();
  CHECK_GE(static_cast<int64_t>(args.compress_ratios().size()), args.n_layers())
      << "deepseek_v4 config compress_ratios size must be >= n_layers after "
         "processing, got "
      << args.compress_ratios().size() << " vs " << args.n_layers();
  for (int64_t i = 0; i < args.n_layers(); ++i) {
    const int32_t ratio = args.compress_ratios()[static_cast<size_t>(i)];
    CHECK(policy.supported_compress_ratios.count(ratio) > 0)
        << "deepseek_v4 config compress_ratios[" << i
        << "] must be in supported_compress_ratios, got " << ratio;
  }
  CHECK_GT(args.window_size(), 0)
      << "deepseek_v4 config window_size must be > 0, got "
      << args.window_size();
  CHECK_GT(args.n_routed_experts(), 0)
      << "deepseek_v4 config n_routed_experts must be > 0, got "
      << args.n_routed_experts();
  CHECK_GT(args.n_activated_experts(), 0)
      << "deepseek_v4 config n_activated_experts/num_experts_per_tok must be "
         "> 0, got "
      << args.n_activated_experts();
  CHECK_LE(args.n_activated_experts(), args.n_routed_experts())
      << "deepseek_v4 config n_activated_experts/num_experts_per_tok must be "
         "<= n_routed_experts, got "
      << args.n_activated_experts() << " vs " << args.n_routed_experts();
  CHECK_GE(args.n_hash_layers(), 0)
      << "deepseek_v4 config num_hash_layers/n_hash_layers must be >= 0, got "
      << args.n_hash_layers();
  CHECK_GT(args.routed_scaling_factor(), 0.0f)
      << "deepseek_v4 config routed_scaling_factor/route_scale must be > 0, "
         "got "
      << args.routed_scaling_factor();
  CHECK_GT(args.swiglu_limit(), 0.0f)
      << "deepseek_v4 config swiglu_limit must be > 0, got "
      << args.swiglu_limit();
  CHECK(!args.scoring_func().empty())
      << "deepseek_v4 config scoring_func/score_func must not be empty";
  {
    std::string score_func = args.scoring_func();
    std::transform(
        score_func.begin(),
        score_func.end(),
        score_func.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    CHECK(policy.supported_score_funcs.count(score_func) > 0)
        << "deepseek_v4 config scoring_func/score_func must be in "
        << absl::StrJoin(policy.supported_score_funcs, ", ") << ", got "
        << args.scoring_func();
  }
  CHECK_GT(args.index_head_dim(), 0)
      << "deepseek_v4 config index_head_dim must be > 0, got "
      << args.index_head_dim();
  CHECK_GT(args.index_n_heads(), 0)
      << "deepseek_v4 config index_n_heads must be > 0, got "
      << args.index_n_heads();
  CHECK_GT(args.index_topk(), 0)
      << "deepseek_v4 config index_topk must be > 0, got " << args.index_topk();
  CHECK_GT(args.hc_mult(), 0)
      << "deepseek_v4 config hc_mult must be > 0, got " << args.hc_mult();
  CHECK_GE(args.hc_sinkhorn_iters(), 0)
      << "deepseek_v4 config hc_sinkhorn_iters must be >= 0, got "
      << args.hc_sinkhorn_iters();
  CHECK_GT(args.hc_eps(), 0.0f)
      << "deepseek_v4 config hc_eps must be > 0, got " << args.hc_eps();
  CHECK_GT(args.factor(), 0.0f)
      << "deepseek_v4 requires positive rope_scaling.factor/factor, got "
      << args.factor();
  CHECK_GT(args.rope_scaling_attn_factor(), 0.0f)
      << "deepseek_v4 requires positive rope_scaling_attn_factor, got "
      << args.rope_scaling_attn_factor();
  CHECK_GT(args.rope_theta(), 0.0f)
      << "deepseek_v4 requires positive rope_theta, got " << args.rope_theta();
  CHECK_GT(args.compress_rope_theta(), 0.0f)
      << "deepseek_v4 requires positive compress_rope_theta, got "
      << args.compress_rope_theta();
}

// register the causal model
REGISTER_CAUSAL_MODEL(deepseek_v4, DeepseekV4ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS(deepseek_v4, [&] {
  constexpr auto preset = DeepseekV4PolicyPreset::kDefault;
  const auto args_policy = build_deepseek_v4_args_policy(preset);
  load_deepseek_v4_model_args(json, args);
  process_deepseek_v4_args(args, args_policy);
  validate_deepseek_v4_args(*args, args_policy);
  normalize_deepseek_v4_args(args);
});

}  // namespace xllm
