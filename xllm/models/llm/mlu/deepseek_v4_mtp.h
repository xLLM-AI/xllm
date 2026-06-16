/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>

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

#include "core/common/global_flags.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/deepseek_v4_rotary_embedding.h"
#include "core/layers/common/dsa_metadata.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_decoder_layer.h"
#include "layers/mlu/deepseek_v4/dsa_cache_mapping.h"
#include "layers/mlu/deepseek_v4/dsa_metadata_builder_mlu.h"
#include "layers/mlu/deepseek_v4/hyper_connection.h"
#include "models/llm/llm_model_base.h"
#include "models/llm/mlu/deepseek_v4.h"
#include "models/llm/mtp_model_base.h"

// Reuse helpers from the MLU main model header:
//   maybe_to_device(), deepseek_v4_uses_mlu_graph(),
//   DeepseekV4GraphMetadataState, DSAGroupKey/DSAGroupKeyHash,
//   normalize_compress_ratio(), next_power_of_two(),
//   create_hadamard_matrix(), load_deepseek_v4_model_args(),
//   DeepseekV4ArgsPolicy, build_deepseek_v4_args_policy(),
//   process_deepseek_v4_args(), validate_deepseek_v4_args()

namespace xllm::mlu::model {

// Make xllm namespace types available unqualified inside this namespace,
// matching the style used in mlu/deepseek_v4.h.
using ::xllm::BatchForwardType;
using ::xllm::DSACacheInfo;
using ::xllm::DSACacheMapping;
using ::xllm::DSACacheType;
using ::xllm::DSAGroupInfo;
using ::xllm::JsonReader;
using ::xllm::KVCache;
using ::xllm::LlmForCausalLMImplBase;
using ::xllm::ModelArgs;
using ::xllm::ModelContext;
using ::xllm::ModelGraphMetadataState;
using ::xllm::ModelInputParams;
using ::xllm::ModelLoader;
using ::xllm::ModelOutput;
using ::xllm::MtpDecoderLayerImplBase;
using ::xllm::ParallelArgs;
using ::xllm::StateDict;
using ::xllm::layer::AttentionMetadata;
using ::xllm::layer::DeepseekV4DecoderLayer;
using ::xllm::layer::DeepseekV4HCHead;
using ::xllm::layer::DeepseekV4RotaryEmbedding;
using ::xllm::layer::DSAMetadata;
using ::xllm::layer::DSAMetadataBuilderMlu;
using ::xllm::layer::RMSNorm;
using ::xllm::layer::WordEmbedding;

class DeepseekV4MultiTokenPredictorLayerImpl
    : public MtpDecoderLayerImplBase<layer::DeepseekV4DecoderLayer> {
 public:
  DeepseekV4MultiTokenPredictorLayerImpl(const ModelContext& context,
                                         int32_t layer_index)
      : MtpDecoderLayerImplBase<layer::DeepseekV4DecoderLayer>(context,
                                                               layer_index) {}

  torch::Tensor forward(torch::Tensor inputs_embeds,
                        torch::Tensor previous_hidden_states,
                        torch::Tensor positions,
                        layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        torch::Tensor tokens) {
    ModelInputParams modified_input_params = input_params;
    modified_input_params.embedding.input_embedding = previous_hidden_states;
    std::optional<torch::Tensor> residual;
    return MtpDecoderLayerImplBase<layer::DeepseekV4DecoderLayer>::forward(
        inputs_embeds,
        residual,
        positions,
        attn_metadata,
        kv_cache,
        modified_input_params,
        tokens);
  }
};
TORCH_MODULE(DeepseekV4MultiTokenPredictorLayer);

class DeepseekV4MtpModelImpl final : public torch::nn::Module {
 public:
  explicit DeepseekV4MtpModelImpl(const ModelContext& context)
      : model_args_(context.get_model_args()) {
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();
    device_ = options.device();

    const int32_t mtp_n_layers = model_args_.n_layers();
    CHECK_GT(mtp_n_layers, 0)
        << "[DeepseekV4Mtp] deepseek_v4_mtp requires n_layers > 0";
    CHECK_GE(model_args_.num_nextn_predict_layers(), 0)
        << "[DeepseekV4Mtp] deepseek_v4_mtp requires "
           "num_nextn_predict_layers >= 0";

    num_heads_ = model_args_.n_heads();
    head_dim_ = model_args_.o_lora_rank() + model_args_.qk_rope_head_dim();
    dp_local_tp_size_ =
        std::max<int64_t>(parallel_args.world_size() /
                              std::max<int64_t>(parallel_args.dp_size(), 1),
                          1);
    CHECK_EQ(num_heads_ % dp_local_tp_size_, 0)
        << "[DeepseekV4Mtp] n_heads must be divisible by local tp "
           "size. n_heads="
        << num_heads_ << ", local_tp_size=" << dp_local_tp_size_;
    tp_num_heads_ = num_heads_ / dp_local_tp_size_;
    hc_mult_ = model_args_.hc_mult();
    window_size_ = model_args_.window_size_;

    hc_head_ = register_module(
        "hc_head",
        layer::DeepseekV4HCHead(hc_mult_,
                                model_args_.hidden_size(),
                                static_cast<double>(model_args_.hc_eps()),
                                static_cast<double>(model_args_.rms_norm_eps()),
                                options));

    init_rope(model_args_, options);
    init_hadamard(model_args_, options);
    max_position_embeddings_ = model_args_.max_position_embeddings();

    mtp_layers_.reserve(mtp_n_layers);
    for (int32_t i = 0; i < mtp_n_layers; ++i) {
      const int32_t layer_index = i;
      mtp_layers_.emplace_back(
          DeepseekV4MultiTokenPredictorLayer(context, layer_index));
      register_module("layer_" + std::to_string(i), mtp_layers_.back());
    }

    build_dsa_cache_info(model_args_);

    for (int32_t layer_id = 0; layer_id < mtp_n_layers; ++layer_id) {
      mtp_layers_[static_cast<size_t>(layer_id)]->set_cache_mapping(
          cache_mappings_[static_cast<size_t>(layer_id)]);
    }

    final_norm_ = register_module("final_norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids);
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < mtp_layers_.size(); ++i) {
      mtp_layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    final_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("layers.0.norm."));
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("layers.0.emb.tok_emb."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    UNUSED_PARAMETER(prefix);
    for (const auto& layer : mtp_layers_) {
      layer->verify_loaded_weights();
    }
  }

  void merge_loaded_weights() {
    for (const auto& layer : mtp_layers_) {
      UNUSED_PARAMETER(layer);
    }
  }

  void merge_and_move_pinned_host() { merge_loaded_weights(); }

  void free_weights() {}

  void reload_weights() {}

  void reload_non_decoder_weights() {}

  void reload_weights_from_device() {}

  void refresh_rolling_weights() {}

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;

    const bool is_empty_dp_rank = !tokens.defined() || tokens.numel() == 0;
    if (is_empty_dp_rank) {
      tokens = torch::tensor(
          {0}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
      positions = torch::tensor(
          {0}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
    }

    const torch::Device runtime_device = tokens.device();

    auto modified_input_params = input_params;
    if (is_empty_dp_rank) {
      fill_empty_dp_rank_input_params(modified_input_params);
    }

    torch::Tensor previous_hidden_states =
        modified_input_params.embedding.input_embedding;
    CHECK(previous_hidden_states.defined())
        << "[DeepseekV4Mtp] input_params.embedding.input_embedding must be "
           "defined for MTP model";

    torch::Tensor hidden_states = embed_tokens_(tokens);

    // Zero out embeddings at position 0
    auto mask = (positions == 0);
    if (mask.any().item<bool>()) {
      hidden_states.index_put_({mask},
                               torch::zeros_like(hidden_states.index({mask})));
    }

    tokens = maybe_to_device(tokens, runtime_device);
    positions = maybe_to_device(positions, runtime_device);

    const bool mlu_graph_forward = deepseek_v4_uses_mlu_graph(input_params);

    auto& dp_token_nums = modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    // Build DSA metadata if not already present
    if (!modified_input_params.attn_metadata ||
        !modified_input_params.attn_metadata->dsa_metadata) {
      CHECK(!mlu_graph_forward)
          << "[DeepseekV4Mtp] MLU graph requires prebuilt DSA metadata";
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

    // Non-graph mode: prepare DSA metadata to device
    if (!mlu_graph_forward) {
      prepare_dsa_metadata(attn_metadata, runtime_device);
    }

    CHECK_GE(static_cast<int32_t>(kv_caches.size()),
             static_cast<int32_t>(mtp_layers_.size()))
        << "[DeepseekV4Mtp] deepseek_v4_mtp requires kv_caches size >= "
           "mtp layer count";

    for (size_t i = 0; i < mtp_layers_.size(); ++i) {
      const int32_t layer_id = static_cast<int32_t>(i);
      prepare_layer_metadata(attn_metadata, layer_id);
      hidden_states = mtp_layers_[i](hidden_states,
                                     previous_hidden_states,
                                     positions,
                                     attn_metadata,
                                     kv_caches[i],
                                     modified_input_params,
                                     tokens);
      if (!modified_input_params.record_layer(static_cast<uint32_t>(i),
                                              hidden_states.device())) {
        return ModelOutput();
      }
    }

    // Model-level hc_head (same as MLU main model)
    auto [output, _] = final_norm_(hidden_states, std::nullopt);
    return ModelOutput(output, std::nullopt);
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
        << "[DeepseekV4Mtp] graph metadata state must be initialized";
    auto* dsv4_state = dynamic_cast<DeepseekV4GraphMetadataState*>(state);
    CHECK(dsv4_state != nullptr)
        << "[DeepseekV4Mtp] received incompatible graph metadata state";

    auto modified_input_params = input_params;
    auto& dp_token_nums = modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    // Build DSA metadata outside graph capture
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

  class CacheEntry {
   public:
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
    const int32_t base_block_size = FLAGS_block_size;
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

  void fill_empty_dp_rank_input_params(ModelInputParams& params) const {
    auto cpu_int_options = torch::TensorOptions()
                               .dtype(torch::kInt32)
                               .device(torch::kCPU)
                               .pinned_memory(true);
    params.meta.num_sequences = 1;
    params.meta.actual_num_sequences = 1;
    params.meta.kv_max_seq_len =
        std::max<int32_t>(params.meta.kv_max_seq_len, 1);
    params.meta.q_max_seq_len = 1;
    params.meta.batch_forward_type = BatchForwardType::DECODE;
    params.attention.host.kv_seq_lens = {1};
    params.attention.host.q_seq_lens = {1};
    params.attention.host.q_cu_seq_lens = {1};
    params.attention.device.kv_seq_lens =
        torch::tensor(params.attention.host.kv_seq_lens, cpu_int_options);
    params.attention.device.q_seq_lens =
        torch::tensor(params.attention.host.q_seq_lens, cpu_int_options);
    params.attention.device.q_cu_seq_lens = torch::tensor({1}, cpu_int_options);
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
      params.multi_block_tables.emplace_back(
          torch::zeros({1, 1}, cpu_int_options));
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
  int64_t tp_num_heads_ = 0;
  int64_t dp_local_tp_size_ = 1;
  int64_t head_dim_ = 0;
  int64_t window_size_ = 128;
  int64_t max_position_embeddings_ = 0;

  std::vector<std::vector<DSACacheInfo>> caches_info_;
  std::vector<DSACacheMapping> cache_mappings_;
  std::vector<DSAGroupInfo> group_infos_;

  ModelArgs model_args_;
  torch::Device device_{torch::kCPU};

  layer::RMSNorm final_norm_{nullptr};
  layer::WordEmbedding embed_tokens_{nullptr};
  std::vector<DeepseekV4MultiTokenPredictorLayer> mtp_layers_;
};
TORCH_MODULE(DeepseekV4MtpModel);

class DeepseekV4MtpForCausalLMImpl final
    : public LlmForCausalLMImplBase<DeepseekV4MtpModel> {
 public:
  explicit DeepseekV4MtpForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV4MtpModel>(context) {}

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
            remapped_state_dict.get_dict_with_prefix("layers.0.head."));
      }
    }
    model_->verify_loaded_weights(prefix);
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
TORCH_MODULE(DeepseekV4MtpForCausalLM);

inline void load_deepseek_v4_mtp_model_args(const JsonReader& json,
                                            ModelArgs* args) {
  load_deepseek_v4_model_args(json, args);
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v4_mtp");
  LOAD_ARG_OR(num_nextn_predict_layers, "num_nextn_predict_layers", 1);
  SET_ARG(n_hash_layers, 0);
}

REGISTER_CAUSAL_MODEL(deepseek_v4_mtp, DeepseekV4MtpForCausalLM);

REGISTER_MODEL_ARGS(deepseek_v4_mtp, [&] {
  const DeepseekV4ArgsPolicy args_policy = build_deepseek_v4_args_policy();
  load_deepseek_v4_mtp_model_args(json, args);
  process_deepseek_v4_args(args, args_policy);
  validate_deepseek_v4_args(*args, args_policy);
});

}  // namespace xllm::mlu::model