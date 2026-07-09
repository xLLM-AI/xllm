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
#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/deepseek_v4_rotary_embedding.h"
#include "core/layers/common/dsa_metadata.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/mlu/deepseek_v4/deepseek_v4_decoder_layer.h"
#include "core/layers/mlu/deepseek_v4/dsa_cache_mapping.h"
#include "core/layers/mlu/deepseek_v4/dsa_empty_dp_input.h"
#include "core/layers/mlu/deepseek_v4/dsa_metadata_builder_mlu.h"
#include "core/layers/mlu/deepseek_v4/hyper_connection.h"
#include "models/llm/llm_model_base.h"

namespace xllm::mlu::model {

inline torch::Tensor maybe_to_device(const torch::Tensor& tensor,
                                     const torch::Device& device) {
  if (!tensor.defined() || tensor.device() == device) {
    return tensor;
  }
  return tensor.to(device);
}

inline bool deepseek_v4_uses_mlu_graph(
    const xllm::ModelInputParams& input_params) {
  return ::xllm::ExecutionConfig::get_instance().enable_graph() &&
         input_params.enable_graph;
}

struct DeepseekV4GraphMetadataState : ModelGraphMetadataState {
  struct DSAMetadataPersistent {
    torch::Tensor attn_mask;
    torch::Tensor start_pos;
    // MLU-only canonical seq lengths
    torch::Tensor q_cu_seq_lens;
    torch::Tensor kv_cu_seq_lens;
    torch::Tensor q_seq_lens;
    torch::Tensor kv_seq_lens;
    torch::Tensor index_c4_seq_lens;
    // SWA plan
    torch::Tensor swa_history_lens;
    torch::Tensor swa_context_lens;
    // C128 attention
    torch::Tensor c128_context_lens;
    torch::Tensor c128_block_table_for_attn;
    // Sequence lengths
    torch::Tensor seq_lens;
    torch::Tensor seq_lens_q;
    torch::Tensor actual_seq_lengths_kv;
    torch::Tensor actual_seq_lengths_query;
    torch::Tensor max_seqlen_kv;
    torch::Tensor max_seqlen_q;
    // Positions
    torch::Tensor input_positions;
    torch::Tensor c4_pad_positions;
    torch::Tensor c128_pad_positions;
    // Per-cache-type block tables and slot mappings — persisted by group_id
    // so graph replay reads from stable device addresses. Multiple layers
    // with the same cache type share the same persistent buffer.
    std::unordered_map<int32_t, torch::Tensor> block_tables_by_group;
    std::unordered_map<int32_t, torch::Tensor> slot_mappings_by_group;
  };
  DSAMetadataPersistent dsa_metadata_persistent;
};

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

class DeepseekV4ModelImpl final
    : public LlmModelImplBase<layer::DeepseekV4DecoderLayer> {
 public:
  explicit DeepseekV4ModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::DeepseekV4DecoderLayer>(
            "deepseek_v4",
            context.get_model_args()) {
    const ModelArgs& model_args = context.get_model_args();
    const torch::TensorOptions options = context.get_tensor_options();
    const ParallelArgs& parallel_args = context.get_parallel_args();

    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));

    hc_mult_ = model_args.hc_mult();
    window_size_ = model_args.window_size();

    num_heads_ = model_args.n_heads();
    dp_local_tp_size_ =
        std::max<int64_t>(parallel_args.world_size() /
                              std::max<int64_t>(parallel_args.dp_size(), 1),
                          1);
    CHECK_EQ(num_heads_ % dp_local_tp_size_, 0)
        << "[DSV4][Init] n_heads must be divisible by local tp size. n_heads="
        << num_heads_ << ", local_tp_size=" << dp_local_tp_size_;

    hc_head_ = register_module(
        "hc_head",
        layer::DeepseekV4HCHead(hc_mult_,
                                model_args.hidden_size(),
                                static_cast<double>(model_args.hc_eps()),
                                static_cast<double>(model_args.rms_norm_eps()),
                                options));

    init_rope(model_args, options);
    init_hadamard(model_args, options);
    max_position_embeddings_ = model_args.max_position_embeddings();

    for (int32_t layer_id = 0; layer_id < model_args.n_layers(); ++layer_id) {
      layers_.emplace_back(layer::DeepseekV4DecoderLayer(context, layer_id));
    }

    build_dsa_cache_info(model_args);

    for (int32_t layer_id = 0; layer_id < model_args.n_layers(); ++layer_id) {
      layers_[static_cast<size_t>(layer_id)]->set_cache_mapping(
          cache_mappings_[static_cast<size_t>(layer_id)]);
    }
  }

  void load_state_dict(const StateDict& state_dict) override {
    embed_tokens_->load_state_dict(state_dict.get_dict_with_prefix(
        std::vector<std::string>{"embed_tokens.", "embed."}));
    for (size_t layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layers_[layer_id]->load_state_dict(state_dict.get_dict_with_prefix(
          "layers." + std::to_string(layer_id) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    hc_head_->load_state_dict(state_dict);
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    torch::NoGradGuard no_grad;
    const bool is_empty_dp_rank = input_params.meta.q_max_seq_len == 0 ||
                                  input_params.meta.num_sequences == 0 ||
                                  tokens.numel() == 0;
    if (tokens.numel() == 0) {
      tokens = torch::tensor(
          {1},
          torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));
      positions = torch::tensor(
          {0},
          torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));
    }

    torch::Tensor h = input_params.embedding.input_embedding.defined()
                          ? input_params.embedding.input_embedding
                          : embed_tokens_(tokens);
    if (h.dim() == 2) {
      h = h.unsqueeze(1).repeat({1, hc_mult_, 1});
    }

    const torch::Device runtime_device = h.device();
    tokens = maybe_to_device(tokens, runtime_device);
    positions = maybe_to_device(positions, runtime_device);

    const bool mlu_graph_forward = deepseek_v4_uses_mlu_graph(input_params);

    ModelInputParams modified_input_params = input_params;
    if (is_empty_dp_rank && !mlu_graph_forward) {
      layer::fill_dsv4_empty_dp_params(
          modified_input_params, group_infos_, window_size_);
    }
    std::vector<int32_t>& dp_token_nums =
        modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    if (!modified_input_params.attn_metadata ||
        !modified_input_params.attn_metadata->dsa_metadata) {
      CHECK(!mlu_graph_forward)
          << "DeepSeek V4 MLU graph requires prebuilt DSA metadata";
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::DSAMetadataBuilderMlu::build(modified_input_params,
                                                  positions,
                                                  caches_info_,
                                                  group_infos_,
                                                  window_size_));
    }
    layer::AttentionMetadata& attn_metadata =
        *(modified_input_params.attn_metadata);
    if (is_empty_dp_rank && !mlu_graph_forward) {
      // Empty-DP inputs only preserve local shape and collective participation.
      // They must not write dummy KV rows into real cache slots.
      attn_metadata.is_dummy = true;
    }

    if (!mlu_graph_forward) {
      prepare_dsa_metadata(attn_metadata, runtime_device);
    }

    std::optional<torch::Tensor> residual;
    for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
      prepare_layer_metadata(attn_metadata, static_cast<int32_t>(layer_idx));
      h = layers_[layer_idx](h,
                             residual,
                             positions,
                             attn_metadata,
                             kv_caches[layer_idx],
                             modified_input_params,
                             tokens);
      if (!modified_input_params.record_layer(static_cast<uint32_t>(layer_idx),
                                              h.device())) {
        return ModelOutput();
      }
    }

    h = hc_head_(h);
    auto [hidden_states, residual_out] = norm_(h, std::nullopt);
    return ModelOutput(hidden_states, residual_out);
  }

 public:
  bool requires_graph_forward_metadata() { return true; }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return std::make_unique<DeepseekV4GraphMetadataState>();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input_params) {
    CHECK(state != nullptr)
        << "DeepSeek V4 MLU graph metadata state must be initialized";
    auto* dsv4_state = dynamic_cast<DeepseekV4GraphMetadataState*>(state);
    CHECK(dsv4_state != nullptr)
        << "DeepSeek V4 MLU received incompatible graph metadata state";

    auto modified_input_params = input_params;

    // Build DSA metadata outside graph capture.
    auto attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::DSAMetadataBuilderMlu::build(modified_input_params,
                                            positions,
                                            caches_info_,
                                            group_infos_,
                                            window_size_));
    if (!attn_metadata->dsa_metadata) {
      input_params.attn_metadata = attn_metadata;
      return;
    }

    const torch::Device runtime_device =
        positions.defined() ? positions.device() : torch::Device(torch::kCPU);

    prepare_dsa_metadata(*attn_metadata, runtime_device);
    auto& dsa = *attn_metadata->dsa_metadata;
    auto& persistent = dsv4_state->dsa_metadata_persistent;
    init_persistent_cache_buffers(
        /*persistent=*/persistent,
        /*input_params=*/modified_input_params,
        /*num_tokens=*/positions.numel(),
        /*runtime_device=*/runtime_device);
    persist_dsa_metadata(dsa, persistent);
    sync_dsa_seq_metadata(*attn_metadata, dsa);
    input_params.attn_metadata = attn_metadata;
  }

 private:
  void persist_dsa_metadata(
      layer::DSAMetadata& dsa,
      DeepseekV4GraphMetadataState::DSAMetadataPersistent& persistent) {
    // Scalar metadata tensors
    dsa.seq_lens = copy_to_persistent_tensor(dsa.seq_lens, persistent.seq_lens);
    dsa.seq_lens_q =
        copy_to_persistent_tensor(dsa.seq_lens_q, persistent.seq_lens_q);
    dsa.actual_seq_lengths_kv = copy_to_persistent_tensor(
        dsa.actual_seq_lengths_kv, persistent.actual_seq_lengths_kv);
    dsa.actual_seq_lengths_query = copy_to_persistent_tensor(
        dsa.actual_seq_lengths_query, persistent.actual_seq_lengths_query);
    dsa.max_seqlen_kv =
        copy_to_persistent_tensor(dsa.max_seqlen_kv, persistent.max_seqlen_kv);
    dsa.max_seqlen_q =
        copy_to_persistent_tensor(dsa.max_seqlen_q, persistent.max_seqlen_q);
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
    dsa.swa_history_lens = copy_to_persistent_tensor(
        dsa.swa_history_lens, persistent.swa_history_lens);
    dsa.swa_context_lens = copy_to_persistent_tensor(
        dsa.swa_context_lens, persistent.swa_context_lens);

    // c128 metadata
    dsa.c128_attn_metadata.context_lens = copy_to_persistent_tensor(
        dsa.c128_attn_metadata.context_lens, persistent.c128_context_lens);
    dsa.c128_attn_metadata.block_table_for_attn =
        copy_to_persistent_tensor(dsa.c128_attn_metadata.block_table_for_attn,
                                  persistent.c128_block_table_for_attn,
                                  -1);

    // start_pos
    dsa.start_pos =
        copy_to_persistent_tensor(dsa.start_pos, persistent.start_pos);

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

  void init_persistent_cache_buffers(
      DeepseekV4GraphMetadataState::DSAMetadataPersistent& persistent,
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
    persistent.swa_history_lens = torch::zeros({num_tokens}, int_options);
    persistent.swa_context_lens = torch::zeros({num_tokens}, int_options);
    persistent.q_seq_lens = torch::zeros({num_tokens}, int_options);
    persistent.kv_seq_lens = torch::zeros({num_tokens}, int_options);
    persistent.q_cu_seq_lens = torch::zeros({num_tokens + 1}, int_options);
    persistent.kv_cu_seq_lens = torch::zeros({num_tokens + 1}, int_options);
    persistent.seq_lens = torch::zeros({num_tokens}, int_options);
    persistent.seq_lens_q = torch::zeros({num_tokens}, int_options);
    persistent.actual_seq_lengths_kv = torch::zeros({num_tokens}, int_options);
    persistent.actual_seq_lengths_query =
        torch::zeros({num_tokens + 1}, int_options);
    persistent.start_pos = torch::zeros({num_tokens}, int_options);
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

  struct CacheEntry {
    DSACacheType type = DSACacheType::SLIDING_WINDOW;
    int32_t ratio = 1;
    int32_t block_size = 0;
  };

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
    dsa_rotary_embedding_ = std::make_shared<layer::DeepseekV4RotaryEmbedding>(
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
    auto dsa_cos_sin = dsa_rotary_embedding_->get_cos_sin_cache("default");
    auto dsa_compressed_cos_sin =
        dsa_rotary_embedding_->get_cos_sin_cache("c4");
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
    const int32_t base_block_size =
        ::xllm::KVCacheConfig::get_instance().block_size();
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
        caches_info_[static_cast<size_t>(layer_id)].emplace_back(
            group_id, entry.type, entry.ratio, entry.block_size);
      }
    }
  }

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

  void prepare_dsa_metadata(layer::AttentionMetadata& attn_metadata,
                            const torch::Device& runtime_device) const {
    if (!attn_metadata.dsa_metadata) {
      return;
    }

    layer::DSAMetadata& dsa = *(attn_metadata.dsa_metadata);
    dsa.seq_lens = maybe_to_device(dsa.seq_lens, runtime_device);
    dsa.seq_lens_q = maybe_to_device(dsa.seq_lens_q, runtime_device);
    dsa.actual_seq_lengths_query =
        maybe_to_device(dsa.actual_seq_lengths_query, runtime_device);
    dsa.actual_seq_lengths_kv =
        maybe_to_device(dsa.actual_seq_lengths_kv, runtime_device);
    dsa.max_seqlen_q = maybe_to_device(dsa.max_seqlen_q, runtime_device);
    dsa.max_seqlen_kv = maybe_to_device(dsa.max_seqlen_kv, runtime_device);
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
    dsa.swa_history_lens =
        maybe_to_device(dsa.swa_history_lens, runtime_device);
    dsa.swa_context_lens =
        maybe_to_device(dsa.swa_context_lens, runtime_device);
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

    if (dsa.actual_seq_lengths_kv.defined() && dsa.seq_lens_q.defined()) {
      dsa.start_pos =
          (dsa.actual_seq_lengths_kv - dsa.seq_lens_q).to(torch::kInt32);
    }
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

  torch::Tensor dsa_cos_;
  torch::Tensor dsa_sin_;
  torch::Tensor dsa_compressed_cos_;
  torch::Tensor dsa_compressed_sin_;
  torch::Tensor inverse_sin_;
  torch::Tensor compressed_inverse_sin_;
  torch::Tensor dsa_hadamard_;
  std::shared_ptr<layer::DeepseekV4RotaryEmbedding> dsa_rotary_embedding_;
  layer::DeepseekV4HCHead hc_head_{nullptr};

  int64_t hc_mult_ = 1;
  int64_t num_heads_ = 0;
  int64_t dp_local_tp_size_ = 1;
  int64_t window_size_ = 128;
  int64_t max_position_embeddings_ = 0;

  std::vector<std::vector<DSACacheInfo>> caches_info_;
  std::vector<DSACacheMapping> cache_mappings_;
  std::vector<DSAGroupInfo> group_infos_;
};
TORCH_MODULE(DeepseekV4Model);

class DeepseekV4ForCausalLMImpl final
    : public LlmForCausalLMImplBase<DeepseekV4Model> {
 public:
  explicit DeepseekV4ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV4Model>(context) {}

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

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      std::unordered_map<std::string, torch::Tensor> remapped_dict;
      std::unordered_map<std::string, torch::Tensor> lm_head_dict;
      for (auto it = state_dict->begin(); it != state_dict->end(); ++it) {
        remapped_dict[normalize_model_parameter_name(it->first, prefix)] =
            it->second;
        std::optional<std::string> lm_head_name =
            normalize_lm_head_parameter_name(it->first);
        if (lm_head_name.has_value()) {
          lm_head_dict[lm_head_name.value()] = it->second;
        }
      }

      StateDict remapped_state_dict(remapped_dict);
      model_->load_state_dict(remapped_state_dict);
      if (!lm_head_dict.empty()) {
        lm_head_->load_state_dict(StateDict(lm_head_dict));
      } else {
        lm_head_->load_state_dict(
            remapped_state_dict.get_dict_with_prefix("lm_head."));
      }
    }
  }

 private:
  static bool strip_prefix(std::string* name, const std::string& prefix) {
    if (prefix.empty() || name->rfind(prefix, 0) != 0) {
      return false;
    }
    name->erase(0, prefix.length());
    return true;
  }

  static std::string replace_all(std::string input,
                                 const std::string& from,
                                 const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = input.find(from, start_pos)) != std::string::npos) {
      input.replace(start_pos, from.length(), to);
      start_pos += to.length();
    }
    return input;
  }

  static std::string normalize_model_parameter_name(std::string name,
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

  static std::optional<std::string> normalize_lm_head_parameter_name(
      std::string name) {
    if (strip_prefix(&name, "lm_head.")) {
      return name;
    }
    if (strip_prefix(&name, "head.")) {
      return name;
    }
    return std::nullopt;
  }

  static std::string remap_parameter_name(std::string name) {
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
};
TORCH_MODULE(DeepseekV4ForCausalLM);

inline void load_deepseek_v4_model_args(const JsonReader& json,
                                        ModelArgs* args) {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v4");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
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

  LOAD_ARG_OR_FUNC(rms_norm_eps, "rms_norm_eps", [&] {
    return json.value_or<float>("norm_eps", 1e-6f);
  });
  LOAD_ARG_OR_FUNC(
      rope_theta, "rope_theta", [&] { return args->rope_theta(); });
  LOAD_ARG_OR(rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(o_groups, "o_groups", 8);

  LOAD_ARG(compress_ratios, "compress_ratios");
  LOAD_ARG_OR(compress_rope_theta, "compress_rope_theta", 160000.0f);
  LOAD_ARG_OR(window_size, "window_size", 128);

  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_activated_experts, "n_activated_experts", 6);
  LOAD_ARG_OR_FUNC(num_experts_per_tok, "num_experts_per_tok", [&] {
    return args->n_activated_experts();
  });
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(swiglu_limit, "swiglu_limit", 10);
  LOAD_ARG_OR(n_hash_layers, "num_hash_layers", 3);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 1.5f);
  LOAD_ARG_OR(scoring_func, "scoring_func", "sqrtsoftplus");

  LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "index_n_heads", 64);
  LOAD_ARG_OR(index_topk, "index_topk", 512);

  LOAD_ARG_OR(hc_mult, "hc_mult", 4);
  LOAD_ARG_OR(hc_sinkhorn_iters, "hc_sinkhorn_iters", 20);
  LOAD_ARG_OR(hc_eps, "hc_eps", 1e-6f);
  LOAD_ARG_OR(factor, "rope_scaling.factor", 16.0f);
  LOAD_ARG_OR(beta_fast, "rope_scaling.beta_fast", 32.0f);
  LOAD_ARG_OR(beta_slow, "rope_scaling.beta_slow", 1.0f);
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);
  LOAD_ARG_OR(scale_fmt, "scale_fmt", "ue8m0");

  LOAD_ARG_OR_FUNC(
      max_batch_size, "max_batch_size", [&] { return args->max_batch_size(); });
  LOAD_ARG_OR_FUNC(
      max_seq_len, "max_seq_len", [&] { return args->max_seq_len(); });
  LOAD_ARG_OR_FUNC(
      vocab_size, "vocab_size", [&] { return args->vocab_size(); });
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 163840);

  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
}

struct DeepseekV4ArgsPolicy {
  std::unordered_set<int32_t> supported_compress_ratios;
  std::unordered_set<int32_t> supported_effective_ratios;
  std::unordered_set<std::string> supported_score_funcs;
  int32_t default_compress_ratio = 1;
};

inline DeepseekV4ArgsPolicy build_deepseek_v4_args_policy() {
  DeepseekV4ArgsPolicy policy;
  policy.supported_compress_ratios = {0, 1, 4, 128};
  policy.supported_effective_ratios = {1, 4, 128};
  policy.supported_score_funcs = {"softmax", "sigmoid", "sqrtsoftplus"};
  policy.default_compress_ratio = 1;
  return policy;
}

inline void process_deepseek_v4_args(ModelArgs* args,
                                     const DeepseekV4ArgsPolicy& policy) {
  SET_ARG(n_activated_experts, args->num_experts_per_tok());
  if (args->n_layers() > 0 &&
      static_cast<int64_t>(args->compress_ratios().size()) < args->n_layers()) {
    args->compress_ratios().resize(static_cast<size_t>(args->n_layers()),
                                   policy.default_compress_ratio);
  }
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
}

inline void validate_deepseek_v4_args(const ModelArgs& args,
                                      const DeepseekV4ArgsPolicy& policy) {
  CHECK_GT(args.n_layers(), 0)
      << "deepseek_v4 config n_layers/num_hidden_layers must be > 0, got "
      << args.n_layers();
  CHECK_GE(static_cast<int64_t>(args.compress_ratios().size()), args.n_layers())
      << "deepseek_v4 config compress_ratios size must be >= n_layers after "
         "processing, got "
      << args.compress_ratios().size() << " vs " << args.n_layers();
  for (int64_t layer_id = 0; layer_id < args.n_layers(); ++layer_id) {
    const int32_t ratio = args.compress_ratios()[static_cast<size_t>(layer_id)];
    CHECK(policy.supported_compress_ratios.count(ratio) > 0)
        << "deepseek_v4 config compress_ratios[" << layer_id
        << "] must be in supported_compress_ratios, got " << ratio;
    const int32_t effective_ratio = normalize_compress_ratio(ratio);
    CHECK(policy.supported_effective_ratios.count(effective_ratio) > 0)
        << "deepseek_v4 config effective compress_ratios[" << layer_id
        << "] must be one of 1/4/128, got " << effective_ratio;
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
  CHECK(!args.scoring_func().empty())
      << "deepseek_v4 config scoring_func/score_func must not be empty";

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

REGISTER_CAUSAL_MODEL(deepseek_v4, DeepseekV4ForCausalLM);

REGISTER_MODEL_ARGS(deepseek_v4, [&] {
  const DeepseekV4ArgsPolicy args_policy = build_deepseek_v4_args_policy();
  load_deepseek_v4_model_args(json, args);
  process_deepseek_v4_args(args, args_policy);
  validate_deepseek_v4_args(*args, args_policy);
});

}  // namespace xllm::mlu::model
