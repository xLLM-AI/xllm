/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/model/causal_lm.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/deepseek_v4_rotary_embedding.h"
#include "core/layers/common/dsa_metadata.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/mlu/deepseek_v4/dsa_metadata_builder_mlu.h"
#include "layers/mlu/deepseek_v4/dsa_cache_mapping.h"

namespace xllm {
namespace mlu {
namespace model {

// ============================================================================
// Free helper functions — shared between DeepseekV4ModelImpl and
// DeepseekV4MtpModelImpl.
// ============================================================================

inline torch::Tensor maybe_to_device(const torch::Tensor& tensor,
                                     const torch::Device& device) {
  if (!tensor.defined() || tensor.device() == device) {
    return tensor;
  }
  return tensor.to(device);
}

inline bool deepseek_v4_uses_mlu_graph(const ModelInputParams& input_params) {
  return ExecutionConfig::get_instance().enable_graph() &&
         input_params.enable_graph;
}

inline int32_t normalize_compress_ratio(int32_t ratio) {
  return ratio <= 1 ? 1 : ratio;
}

inline int64_t next_power_of_two(int64_t value) {
  int64_t result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

inline torch::Tensor create_hadamard_matrix(int64_t size,
                                            torch::ScalarType dtype,
                                            const torch::Device& device) {
  torch::TensorOptions options =
      torch::TensorOptions().dtype(dtype).device(device);
  torch::Tensor matrix = torch::ones({1, 1}, options);
  for (int64_t dim = 1; dim < size; dim <<= 1) {
    torch::Tensor top = torch::cat({matrix, matrix}, 1);
    torch::Tensor bottom = torch::cat({matrix, -matrix}, 1);
    matrix = torch::cat({top, bottom}, 0);
  }
  return matrix;
}

// ============================================================================
// DSA cache group key — used by build_dsa_cache_info to deduplicate cache
// groups across layers.
// ============================================================================

class DSAGroupKey final {
 public:
  int32_t ratio_ = 1;
  DSACacheType type_ = DSACacheType::SLIDING_WINDOW;
  int32_t block_size_ = 0;

  bool operator==(const DSAGroupKey& other) const {
    return ratio_ == other.ratio_ && type_ == other.type_ &&
           block_size_ == other.block_size_;
  }
};

class DSAGroupKeyHash final {
 public:
  size_t operator()(const DSAGroupKey& key) const {
    size_t hash = std::hash<int32_t>()(key.ratio_);
    hash ^= std::hash<int32_t>()(static_cast<int32_t>(key.type_)) << 16;
    hash ^= std::hash<int32_t>()(key.block_size_) << 8;
    return hash;
  }
};

// ============================================================================
// Weight loading utilities — shared by DeepseekV4ForCausalLMImpl and
// DeepseekV4MtpForCausalLMImpl.
// ============================================================================

namespace detail {

inline bool strip_prefix(std::string* name, const std::string& prefix) {
  if (prefix.empty() || name->rfind(prefix, 0) != 0) {
    return false;
  }
  name->erase(0, prefix.length());
  return true;
}

inline std::string replace_all(std::string input,
                               const std::string& from,
                               const std::string& to) {
  size_t start_pos = 0;
  while ((start_pos = input.find(from, start_pos)) != std::string::npos) {
    input.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
  return input;
}

inline std::string remap_parameter_name(std::string name) {
  name = replace_all(name, "hc_attn_base", "attn_hc_pre.hc_base");
  name = replace_all(name, "hc_attn_fn", "attn_hc_pre.hc_fn");
  name = replace_all(name, "hc_attn_scale", "attn_hc_pre.hc_scale");
  name = replace_all(name, "hc_ffn_base", "ffn_hc_pre.hc_base");
  name = replace_all(name, "hc_ffn_fn", "ffn_hc_pre.hc_fn");
  name = replace_all(name, "hc_ffn_scale", "ffn_hc_pre.hc_scale");
  name = replace_all(name, "hc_head.hc_head_base", "hc_head_base");
  name = replace_all(name, "hc_head.hc_head_fn", "hc_head_fn");
  name = replace_all(name, "hc_head.hc_head_scale", "hc_head_scale");
  name = replace_all(name, "w1.", "gate_proj.");
  name = replace_all(name, "w3.", "up_proj.");
  name = replace_all(name, "w2.", "down_proj.");
  return name;
}

inline std::string normalize_model_parameter_name(std::string name,
                                                  const std::string& prefix) {
  if (!strip_prefix(&name, prefix)) {
    const std::vector<std::string> candidate_prefixes = {
        "model.language_model.", "language_model.model.", "model."};
    for (const std::string& candidate_prefix : candidate_prefixes) {
      if (strip_prefix(&name, candidate_prefix)) {
        break;
      }
    }
  }
  return remap_parameter_name(name);
}

inline std::optional<std::string> normalize_lm_head_parameter_name(
    std::string name) {
  if (strip_prefix(&name, "lm_head.")) {
    return name;
  }
  if (strip_prefix(&name, "head.")) {
    return name;
  }
  return std::nullopt;
}

}  // namespace detail

// ============================================================================
// DeepseekV4Base — shared DSA metadata and cache infrastructure for both
// the main model (DeepseekV4ModelImpl) and the MTP draft model
// (DeepseekV4MtpModelImpl).
//
// Both derived classes inherit from this to eliminate ~290 lines of
// duplicated code. This class does NOT inherit from torch::nn::Module;
// each derived class handles its own module registration.
// ============================================================================

class DeepseekV4Base {
 public:
  // Graph metadata interface — defined out-of-line in deepseek_v4.h because
  // they need the complete DeepseekV4GraphMetadataState type.
  bool requires_graph_forward_metadata() { return true; }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state();

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input_params);

 protected:
  // ==========================================================================
  // Data members — shared by both derived classes
  // ==========================================================================
  torch::Tensor dsa_cos_;
  torch::Tensor dsa_sin_;
  torch::Tensor dsa_compressed_cos_;
  torch::Tensor dsa_compressed_sin_;
  torch::Tensor inverse_sin_;
  torch::Tensor compressed_inverse_sin_;
  torch::Tensor dsa_hadamard_;

  int64_t hc_mult_ = 1;
  int64_t window_size_ = 128;
  int64_t max_position_embeddings_ = 0;

  std::vector<std::vector<DSACacheInfo>> caches_info_;
  std::vector<DSACacheMapping> cache_mappings_;
  std::vector<DSAGroupInfo> group_infos_;

  // ==========================================================================
  // Initialization helpers — called from derived class constructors
  // ==========================================================================

  void init_rope(const ModelArgs& model_args,
                 const torch::TensorOptions& options) {
    const int64_t rope_head_dim = model_args.rope_head_dim();
    const int64_t max_pos = model_args.max_position_embeddings();
    if (rope_head_dim <= 0 || max_pos <= 0) {
      return;
    }
    const int64_t original_max_pos =
        model_args.rope_scaling_original_max_position_embeddings() > 0
            ? model_args.rope_scaling_original_max_position_embeddings()
            : max_pos;
    auto dsa_rotary_embedding =
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
    auto dsa_cos_sin = dsa_rotary_embedding->get_cos_sin_cache("default");
    auto dsa_compressed_cos_sin = dsa_rotary_embedding->get_cos_sin_cache("c4");
    std::vector<torch::Tensor> chunks =
        dsa_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    dsa_cos_ = chunks[0].contiguous();
    dsa_sin_ = chunks[1].contiguous();
    std::vector<torch::Tensor> compressed_chunks =
        dsa_compressed_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    dsa_compressed_cos_ = compressed_chunks[0].contiguous();
    dsa_compressed_sin_ = compressed_chunks[1].contiguous();
    inverse_sin_ = -dsa_sin_;
    compressed_inverse_sin_ = -dsa_compressed_sin_;
  }

  void init_hadamard(const ModelArgs& model_args,
                     const torch::TensorOptions& options) {
    if (model_args.index_head_dim() <= 0) {
      return;
    }
    const int64_t hadamard_dim = next_power_of_two(model_args.index_head_dim());
    dsa_hadamard_ = create_hadamard_matrix(
        hadamard_dim, options.dtype().toScalarType(), options.device());
  }

  void build_dsa_cache_info(const ModelArgs& model_args) {
    const std::vector<int32_t>& compress_ratios = model_args.compress_ratios();
    const int32_t base_block_size = KVCacheConfig::get_instance().block_size();
    CHECK_GT(base_block_size, 0) << "DeepSeek V4 block_size must be positive.";

    std::unordered_map<DSAGroupKey, int32_t, DSAGroupKeyHash> group_key_map;
    auto register_group =
        [&](DSACacheType type, int32_t ratio, int32_t block_size) -> int32_t {
      DSAGroupKey key;
      key.ratio_ = ratio;
      key.type_ = type;
      key.block_size_ = block_size;
      auto it = group_key_map.find(key);
      if (it != group_key_map.end()) {
        return it->second;
      }
      const int32_t group_id = static_cast<int32_t>(group_infos_.size());
      group_key_map.emplace(key, group_id);
      group_infos_.emplace_back(type, ratio, block_size);
      return group_id;
    };

    register_group(DSACacheType::SLIDING_WINDOW, 1, base_block_size);
    for (const int32_t raw_ratio : compress_ratios) {
      const int32_t ratio = normalize_compress_ratio(raw_ratio);
      if (ratio == 4 || ratio == 128) {
        register_group(DSACacheType::TOKEN, ratio, base_block_size);
      }
    }

    caches_info_.resize(static_cast<size_t>(model_args.n_layers()));
    cache_mappings_.resize(static_cast<size_t>(model_args.n_layers()));
    for (int32_t layer_id = 0; layer_id < model_args.n_layers(); ++layer_id) {
      const int32_t raw_ratio =
          layer_id < static_cast<int32_t>(compress_ratios.size())
              ? compress_ratios[static_cast<size_t>(layer_id)]
              : 1;
      const int32_t ratio = normalize_compress_ratio(raw_ratio);
      const std::vector<CacheEntry> layer_caches =
          cache_entries_for_ratio(ratio, base_block_size);
      cache_mappings_[static_cast<size_t>(layer_id)] =
          cache_mapping_for_ratio(ratio);
      caches_info_[static_cast<size_t>(layer_id)].reserve(layer_caches.size());
      for (const CacheEntry& entry : layer_caches) {
        const int32_t group_id =
            register_group(entry.type, entry.ratio, entry.block_size);
        caches_info_[static_cast<size_t>(layer_id)].push_back(
            {group_id, entry.type, entry.ratio, entry.block_size});
      }
    }
  }

  // ==========================================================================
  // DSA metadata operations — used in the forward pass
  // ==========================================================================

  void prepare_dsa_metadata(layer::AttentionMetadata& attn_metadata,
                            const torch::Device& runtime_device) const {
    if (!attn_metadata.dsa_metadata) {
      return;
    }

    layer::DSAMetadata& dsa = *(attn_metadata.dsa_metadata);
    dsa.seq_lens = maybe_to_device(dsa.seq_lens, runtime_device);
    dsa.input_positions = maybe_to_device(dsa.input_positions, runtime_device)
                              .to(torch::kInt32)
                              .contiguous();
    dsa.c4_pad_positions =
        maybe_to_device(dsa.c4_pad_positions, runtime_device);
    dsa.c128_pad_positions =
        maybe_to_device(dsa.c128_pad_positions, runtime_device);
    dsa.q_cu_seq_lens = maybe_to_device(dsa.q_cu_seq_lens, runtime_device);
    dsa.kv_cu_seq_lens = maybe_to_device(dsa.kv_cu_seq_lens, runtime_device);
    dsa.q_seq_lens = maybe_to_device(dsa.q_seq_lens, runtime_device);
    dsa.kv_seq_lens = maybe_to_device(dsa.kv_seq_lens, runtime_device);
    dsa.index_c4_seq_lens =
        maybe_to_device(dsa.index_c4_seq_lens, runtime_device);
    dsa.c128_attn_metadata.context_lens =
        maybe_to_device(dsa.c128_attn_metadata.context_lens, runtime_device);
    dsa.c128_attn_metadata.block_table_for_attn = maybe_to_device(
        dsa.c128_attn_metadata.block_table_for_attn, runtime_device);

    for (std::vector<torch::Tensor>& layer_block_tables : dsa.block_tables) {
      for (torch::Tensor& block_table : layer_block_tables) {
        block_table = maybe_to_device(block_table, runtime_device);
      }
    }
    for (std::vector<torch::Tensor>& layer_slot_mappings : dsa.slot_mappings) {
      for (torch::Tensor& slot_mapping : layer_slot_mappings) {
        slot_mapping = maybe_to_device(slot_mapping, runtime_device);
      }
    }

    if (dsa_hadamard_.defined()) {
      dsa.hadamard = maybe_to_device(dsa_hadamard_, runtime_device);
    }

    dsa.cos_table = maybe_to_device(dsa_cos_, runtime_device);
    dsa.sin_table = maybe_to_device(dsa_sin_, runtime_device);
    dsa.inverse_sin_table = maybe_to_device(inverse_sin_, runtime_device);
    dsa.compressed_cos_table =
        maybe_to_device(dsa_compressed_cos_, runtime_device);
    dsa.compressed_sin_table =
        maybe_to_device(dsa_compressed_sin_, runtime_device);
    dsa.compressed_inverse_sin_table =
        maybe_to_device(compressed_inverse_sin_, runtime_device);

    sync_dsa_seq_metadata(attn_metadata, dsa);
  }

  void sync_dsa_seq_metadata(layer::AttentionMetadata& attn_metadata,
                             const layer::DSAMetadata& dsa) const {
    attn_metadata.q_cu_seq_lens = dsa.q_cu_seq_lens;
    attn_metadata.kv_cu_seq_lens = dsa.kv_cu_seq_lens;
    attn_metadata.q_seq_lens = dsa.q_seq_lens;
    attn_metadata.kv_seq_lens = dsa.kv_seq_lens;
  }

  void prepare_layer_metadata(layer::AttentionMetadata& attn_metadata,
                              int32_t layer_id) const {
    if (!attn_metadata.dsa_metadata) {
      return;
    }
    layer::DSAMetadata& dsa = *(attn_metadata.dsa_metadata);
    dsa.layer_id = layer_id;
    sync_swa_attention_metadata(attn_metadata, dsa, layer_id);
  }

  void sync_swa_attention_metadata(layer::AttentionMetadata& attn_metadata,
                                   const layer::DSAMetadata& dsa,
                                   int32_t layer_id) const {
    if (layer_id >= static_cast<int32_t>(dsa.block_tables.size()) ||
        layer_id >= static_cast<int32_t>(dsa.slot_mappings.size())) {
      return;
    }
    if (dsa.block_tables[static_cast<size_t>(layer_id)].empty() ||
        dsa.slot_mappings[static_cast<size_t>(layer_id)].empty()) {
      return;
    }

    size_t attn_cache_idx = 0;
    if (layer_id < static_cast<int32_t>(caches_info_.size())) {
      const std::vector<DSACacheInfo>& layer_caches =
          caches_info_[static_cast<size_t>(layer_id)];
      for (size_t cache_idx = 0; cache_idx < layer_caches.size(); ++cache_idx) {
        if (layer_caches[cache_idx].type == DSACacheType::SLIDING_WINDOW) {
          attn_cache_idx = cache_idx;
          break;
        }
      }
    }

    const size_t layer_idx = static_cast<size_t>(layer_id);
    if (attn_cache_idx < dsa.block_tables[layer_idx].size() &&
        dsa.block_tables[layer_idx][attn_cache_idx].defined()) {
      attn_metadata.block_table = dsa.block_tables[layer_idx][attn_cache_idx];
    }
    if (attn_cache_idx < dsa.slot_mappings[layer_idx].size() &&
        dsa.slot_mappings[layer_idx][attn_cache_idx].defined()) {
      attn_metadata.slot_mapping = dsa.slot_mappings[layer_idx][attn_cache_idx];
    }
  }

  // ==========================================================================
  // Graph metadata persistence — for the MLU graph capture/replay path.
  //
  // Template methods: the persistent-storage type is
  // DeepseekV4GraphMetadataState::DSAMetadataPersistent, which is only
  // defined in deepseek_v4.h (included by the TU of both models).
  // ==========================================================================

  template <typename Persistent>
  void persist_dsa_metadata(layer::DSAMetadata& dsa, Persistent& persistent) {
    // Scalar metadata tensors
    dsa.seq_lens = copy_to_persistent_tensor(dsa.seq_lens, persistent.seq_lens);
    dsa.input_positions = copy_to_persistent_tensor(dsa.input_positions,
                                                    persistent.input_positions);
    dsa.c4_pad_positions = copy_to_persistent_tensor(
        dsa.c4_pad_positions, persistent.c4_pad_positions);
    dsa.c128_pad_positions = copy_to_persistent_tensor(
        dsa.c128_pad_positions, persistent.c128_pad_positions);
    dsa.q_cu_seq_lens =
        copy_to_persistent_tensor(dsa.q_cu_seq_lens, persistent.q_cu_seq_lens);
    dsa.kv_cu_seq_lens = copy_to_persistent_tensor(dsa.kv_cu_seq_lens,
                                                   persistent.kv_cu_seq_lens);
    dsa.q_seq_lens =
        copy_to_persistent_tensor(dsa.q_seq_lens, persistent.q_seq_lens);
    dsa.kv_seq_lens =
        copy_to_persistent_tensor(dsa.kv_seq_lens, persistent.kv_seq_lens);
    dsa.index_c4_seq_lens = copy_to_persistent_tensor(
        dsa.index_c4_seq_lens, persistent.index_c4_seq_lens);

    // c128 metadata
    dsa.c128_attn_metadata.context_lens = copy_to_persistent_tensor(
        dsa.c128_attn_metadata.context_lens, persistent.c128_context_lens);
    dsa.c128_attn_metadata.block_table_for_attn =
        copy_to_persistent_tensor(dsa.c128_attn_metadata.block_table_for_attn,
                                  persistent.c128_block_table_for_attn,
                                  -1);

    // block_tables/slot_mappings: copy data into persistent buffers once per
    // group, then assign the persistent buffers back to all dsa entries sharing
    // the same group_id.
    std::unordered_set<int32_t> processed_groups;
    for (size_t lid = 0; lid < dsa.block_tables.size(); ++lid) {
      for (size_t ci = 0; ci < dsa.block_tables[lid].size(); ++ci) {
        const auto& cache_info = caches_info_[lid][ci];
        int32_t group_id = cache_info.group_id;

        if (processed_groups.count(group_id) > 0) {
          // Already processed: just assign the persistent buffer.
          dsa.block_tables[lid][ci] =
              persistent.block_tables_by_group[group_id];
          dsa.slot_mappings[lid][ci] =
              persistent.slot_mappings_by_group[group_id];
          continue;
        }
        processed_groups.insert(group_id);

        // First encounter for this group: copy data into persistent buffer.
        dsa.block_tables[lid][ci] = copy_to_persistent_tensor(
            dsa.block_tables[lid][ci],
            persistent.block_tables_by_group[group_id]);
        dsa.slot_mappings[lid][ci] = copy_to_persistent_tensor(
            dsa.slot_mappings[lid][ci],
            persistent.slot_mappings_by_group[group_id],
            -1);
      }
    }
  }

  template <typename Persistent>
  void init_persistent_cache_buffers(Persistent& persistent,
                                     const ModelInputParams& input_params,
                                     int64_t num_tokens,
                                     const torch::Device& runtime_device) {
    if (!persistent.block_tables_by_group.empty()) {
      return;  // Already initialized
    }

    auto int_options =
        torch::TensorOptions().dtype(torch::kInt32).device(runtime_device);
    // Create persistent buffers for each unique group
    int32_t c128_block_size = 0;
    for (int32_t group_id = 0;
         group_id < static_cast<int32_t>(group_infos_.size());
         ++group_id) {
      if (group_infos_[static_cast<size_t>(group_id)].type ==
              DSACacheType::TOKEN &&
          group_infos_[static_cast<size_t>(group_id)].ratio == 128) {
        c128_block_size =
            group_infos_[static_cast<size_t>(group_id)].block_size;
      }

      // Create block_table buffer with maximum shape
      int32_t block_size =
          group_infos_[static_cast<size_t>(group_id)].block_size;
      int64_t max_blocks_per_seq =
          (max_position_embeddings_ + block_size + 1) / block_size + 1;
      persistent.block_tables_by_group[group_id] =
          torch::full({num_tokens, max_blocks_per_seq}, -1, int_options);

      // Create slot_mapping buffer with maximum shape
      persistent.slot_mappings_by_group[group_id] =
          torch::full({num_tokens}, -1, int_options);
    }

    CHECK_GT(c128_block_size, 0)
        << "Invalid c128 block size: " << c128_block_size;
    persistent.c128_context_lens = torch::zeros({num_tokens}, int_options);
    // block_table_for_attn: [num_tokens, max_blocks_per_seq]
    int64_t compress_len = max_position_embeddings_ / 128;
    const int64_t table_cols = std::max<int64_t>(
        (compress_len + c128_block_size - 1) / c128_block_size, 1);
    persistent.c128_block_table_for_attn =
        torch::full({num_tokens, table_cols}, -1, int_options);

    persistent.input_positions = torch::zeros({num_tokens}, int_options);
    persistent.c4_pad_positions = torch::zeros({num_tokens}, int_options);
    persistent.c128_pad_positions = torch::zeros({num_tokens}, int_options);
    persistent.index_c4_seq_lens = torch::zeros({num_tokens}, int_options);
    persistent.q_seq_lens = torch::zeros({num_tokens}, int_options);
    persistent.kv_seq_lens = torch::zeros({num_tokens}, int_options);
    persistent.q_cu_seq_lens = torch::zeros({num_tokens + 1}, int_options);
    persistent.kv_cu_seq_lens = torch::zeros({num_tokens + 1}, int_options);
    persistent.seq_lens = torch::zeros({num_tokens}, int_options);
  }

  static bool tensor_aliases_storage(const torch::Tensor& lhs,
                                     const torch::Tensor& rhs) {
    return lhs.defined() && rhs.defined() && lhs.data_ptr() == rhs.data_ptr() &&
           lhs.sizes() == rhs.sizes() && lhs.strides() == rhs.strides();
  }

  static torch::Tensor copy_to_persistent_tensor(const torch::Tensor& src,
                                                 torch::Tensor& dst,
                                                 int32_t pad_value = 0) {
    if (!src.defined()) {
      return src;
    }

    // First call (capture): allocate once, address stays stable across replay.
    if (!dst.defined()) {
      dst = torch::empty_like(src);
      dst.copy_(src, /*non_blocking=*/true);
      return dst;
    }

    // Subsequent calls (replay): NEVER reallocate — address must remain stable.
    CHECK_EQ(dst.scalar_type(), src.scalar_type())
        << "DeepSeek V4 MLU graph metadata tensor dtype changed";
    CHECK_EQ(dst.device(), src.device())
        << "DeepSeek V4 MLU graph metadata tensor device changed";

    if (dst.sizes() == src.sizes()) {
      // Most common case: shapes match. Direct copy, no zero_ or narrow needed.
      if (!tensor_aliases_storage(src, dst)) {
        dst.copy_(src, /*non_blocking=*/true);
      }
      return dst;
    }

    // Shapes differ: verify src fits within dst capacity on every dimension.
    bool can_copy_into_capacity = dst.dim() == src.dim() && src.dim() > 0;
    for (int64_t dim = 0; can_copy_into_capacity && dim < src.dim(); ++dim) {
      can_copy_into_capacity &= (src.size(dim) <= dst.size(dim));
    }
    CHECK(can_copy_into_capacity)
        << "DeepSeek V4 MLU graph metadata tensor size incompatible "
        << ": dst=" << dst.sizes() << " vs src=" << src.sizes();

    // Build a dst view that matches src's shape by slicing each dimension
    // where src is smaller than dst, then copy into the view.
    if (pad_value != 0) {
      dst.fill_(pad_value);
    } else {
      dst.zero_();
    }
    torch::Tensor dst_view = dst;
    for (int64_t dim = 0; dim < src.dim(); ++dim) {
      if (src.size(dim) < dst_view.size(dim)) {
        dst_view =
            dst_view.slice(/*dim=*/dim, /*start=*/0, /*end=*/src.size(dim));
      }
    }
    dst_view.copy_(src, /*non_blocking=*/true);
    return dst;
  }

  // ==========================================================================
  // Cache entry helpers
  // ==========================================================================

  struct CacheEntry {
    DSACacheType type = DSACacheType::SLIDING_WINDOW;
    int32_t ratio = 1;
    int32_t block_size = 0;
  };

  std::vector<CacheEntry> cache_entries_for_ratio(
      int32_t ratio,
      int32_t base_block_size) const {
    if (ratio == 1) {
      return {{DSACacheType::SLIDING_WINDOW, 1, base_block_size}};
    }
    if (ratio == 4) {
      return {{DSACacheType::TOKEN, 4, base_block_size},
              {DSACacheType::TOKEN, 4, base_block_size},
              {DSACacheType::SLIDING_WINDOW, 1, base_block_size},
              {DSACacheType::SLIDING_WINDOW, 1, base_block_size},
              {DSACacheType::SLIDING_WINDOW, 1, base_block_size},
              {DSACacheType::SLIDING_WINDOW, 1, base_block_size},
              {DSACacheType::SLIDING_WINDOW, 1, base_block_size},
              {DSACacheType::TOKEN, 4, base_block_size}};
    }
    if (ratio == 128) {
      return {{DSACacheType::TOKEN, 128, base_block_size},
              {DSACacheType::SLIDING_WINDOW, 1, base_block_size},
              {DSACacheType::SLIDING_WINDOW, 1, base_block_size},
              {DSACacheType::SLIDING_WINDOW, 1, base_block_size}};
    }
    LOG(FATAL) << "Unsupported DeepSeek V4 effective compress ratio " << ratio;
    return {};
  }

  DSACacheMapping cache_mapping_for_ratio(int32_t ratio) const {
    DSACacheMapping mapping;
    if (ratio == 1) {
      mapping.ori_cache_idx = 0;
      return mapping;
    }
    if (ratio == 4) {
      mapping.cmp_cache_idx = 0;
      mapping.index_cache_idx = 1;
      mapping.ori_cache_idx = 2;
      mapping.kv_state_cache_idx = 3;
      mapping.score_state_cache_idx = 4;
      mapping.index_kv_state_cache_idx = 5;
      mapping.index_score_state_cache_idx = 6;
      return mapping;
    }
    if (ratio == 128) {
      mapping.cmp_cache_idx = 0;
      mapping.ori_cache_idx = 1;
      mapping.kv_state_cache_idx = 2;
      mapping.score_state_cache_idx = 3;
      return mapping;
    }
    LOG(FATAL) << "Unsupported DeepSeek V4 effective compress ratio " << ratio;
    return mapping;
  }
};

}  // namespace model
}  // namespace mlu
}  // namespace xllm