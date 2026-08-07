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

#include <optional>

#include "core/framework/config/kv_cache_config.h"
#include "core/layers/common/dsa_topk_share_plan.h"
#include "core/layers/common/kv_shard_batch_metadata.h"
#include "core/layers/mlu/dsa_topk_relay.h"
#include "models/llm/glm5.h"

// GLM5.2 shares model_type "glm_moe_dsa" with GLM5.0/5.1 and is told apart by
// the resolved indexer top-k share plan. GLM5.0/5.1 configs carry no reuse, so
// every layer stays non-sharing and the relay below bypasses itself entirely --
// Glm52 then behaves exactly like Glm5. glm_moe_dsa is therefore registered
// once here on Glm52ForCausalLM, while REGISTER_MODEL_ARGS(glm_moe_dsa, ...)
// stays in glm5.h.

namespace xllm {

struct Glm52GraphMetadataState final : ModelGraphMetadataState {
  torch::Tensor q_seq_lens;
  torch::Tensor kv_seq_lens;
  torch::Tensor local_slot_mapping;
  torch::Tensor expanded_indexer_block_table;
};

class Glm52ModelImpl : public Glm5ModelImpl {
 public:
  explicit Glm52ModelImpl(const ModelContext& context)
      : Glm5ModelImpl(context, create_decoder_layer_factory(context)) {
    const ParallelArgs& parallel_args = context.get_parallel_args();
    const int32_t kv_split_size = parallel_args.kv_split_size_effective();
    if (kv_split_size > 1) {
      kv_shard_layout_.emplace(KVCacheConfig::get_instance().block_size(),
                               kv_split_size,
                               parallel_args.kv_split_rank());
    }
  }

  bool requires_graph_forward_metadata() const {
    return kv_shard_layout_.has_value();
  }

  std::unique_ptr<ModelGraphMetadataState> create_graph_forward_metadata_state()
      const {
    return std::make_unique<Glm52GraphMetadataState>();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& /*positions*/,
                                      ModelInputParams& input_params) const {
    CHECK(state != nullptr)
        << "GLM-5.2 DCP graph metadata state must be initialized";
    auto* graph_state = dynamic_cast<Glm52GraphMetadataState*>(state);
    CHECK(graph_state != nullptr)
        << "GLM-5.2 received incompatible graph metadata state";
    CHECK(kv_shard_layout_.has_value())
        << "GLM-5.2 graph metadata requires DCP cache sharding";
    CHECK(input_params.meta.batch_forward_type.is_decode())
        << "GLM-5.2 DCP graph supports decode only";

    prepare_dcp_graph_padding(input_params);

    auto attention_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(input_params,
                                               model_args_.enable_mla(),
                                               /*compute_dtype=*/"half",
                                               /*attn_mask=*/std::nullopt,
                                               /*device=*/std::nullopt));
    attention_metadata->q_seq_lens = copy_to_graph_tensor(
        attention_metadata->q_seq_lens, graph_state->q_seq_lens);
    attention_metadata->kv_seq_lens = copy_to_graph_tensor(
        attention_metadata->kv_seq_lens, graph_state->kv_seq_lens);

    auto shard_metadata = std::make_shared<layer::KVShardBatchMetadata>();
    shard_metadata->local_slot_mapping = copy_to_graph_tensor(
        layer::localize_kv_shard_slots(attention_metadata->slot_mapping,
                                       kv_shard_layout_.value()),
        graph_state->local_slot_mapping);
    shard_metadata->expanded_indexer_block_table = copy_to_graph_tensor(
        layer::expand_kv_shard_indexer_block_table(
            attention_metadata->block_table, kv_shard_layout_.value()),
        graph_state->expanded_indexer_block_table);
    attention_metadata->kv_shard_batch_metadata = std::move(shard_metadata);
    input_params.attn_metadata = std::move(attention_metadata);
  }

 protected:
  void prepare_attention_metadata(
      layer::AttentionMetadata& attn_metadata) const override {
    if (!kv_shard_layout_.has_value() || attn_metadata.is_dummy ||
        attn_metadata.kv_shard_batch_metadata != nullptr) {
      return;
    }
    attn_metadata.kv_shard_batch_metadata =
        build_kv_shard_batch_metadata(attn_metadata, kv_shard_layout_.value());
  }

  std::optional<KVShardLayout> cp_kv_shard_layout() const override {
    return kv_shard_layout_;
  }

  // The relay is reset at layer 0 so its state remains forward-scoped. Decoder
  // layers own role interpretation and the attention transfer protocol.
  torch::Tensor forward_decoder_layer(
      size_t layer_id,
      layer::DeepseekV2DecoderLayer& layer,
      torch::Tensor& hidden_states,
      std::optional<torch::Tensor>& residual,
      torch::Tensor& positions,
      layer::AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      const ModelInputParams& input_params) override {
    if (layer_id == 0) {
      topk_relay_.reset();
    }
    return layer(hidden_states,
                 residual,
                 positions,
                 attn_metadata,
                 kv_cache,
                 input_params,
                 /*input_ids=*/std::nullopt,
                 &topk_relay_);
  }

 private:
  static void prepare_dcp_graph_padding(ModelInputParams& input_params) {
    const int64_t actual_rows = input_params.meta.num_sequences;
    const torch::Tensor& slots = input_params.attention.device.new_cache_slots;
    CHECK(slots.defined())
        << "GLM-5.2 DCP graph requires persistent cache slots";
    CHECK_LE(actual_rows, slots.numel())
        << "GLM-5.2 DCP graph actual rows exceed graph bucket";

    // The MLU indexer needs a physically valid block-table row during graph
    // replay, while DCP treats non-negative slots as cache writes. Keep the
    // generic safe block row and mark bucket tails with the slot sentinel;
    // DCP later masks their derived sparse metadata to an empty row.
    if (actual_rows < slots.numel()) {
      input_params.attention.device.new_cache_slots
          .slice(/*dim=*/0, /*start=*/actual_rows, /*end=*/slots.numel())
          .fill_(KVShardLayout::kInvalidSlot);
    }
  }

  static torch::Tensor copy_to_graph_tensor(const torch::Tensor& source,
                                            torch::Tensor& destination) {
    CHECK(source.defined())
        << "GLM-5.2 DCP graph metadata source must be defined";
    if (!destination.defined()) {
      destination = torch::empty_like(source);
    } else {
      CHECK_EQ(destination.scalar_type(), source.scalar_type())
          << "GLM-5.2 DCP graph metadata dtype changed";
      CHECK_EQ(destination.device(), source.device())
          << "GLM-5.2 DCP graph metadata device changed";
      CHECK_EQ(destination.sizes(), source.sizes())
          << "GLM-5.2 DCP no-padding graph metadata shape changed";
    }
    if (destination.data_ptr() != source.data_ptr()) {
      destination.copy_(source, /*non_blocking=*/true);
    }
    return destination;
  }

  static DecoderLayerFactory create_decoder_layer_factory(
      const ModelContext& context) {
    const layer::DsaTopkSharePlan topk_share_plan(context.get_model_args());
    return
        [topk_share_plan](const ModelContext& layer_context, int32_t layer_id) {
          return layer::DeepseekV2DecoderLayer(
              layer_context, layer_id, topk_share_plan);
        };
  }

  layer::DsaTopkRelay topk_relay_;
  std::optional<KVShardLayout> kv_shard_layout_;
};
TORCH_MODULE(Glm52Model);

class Glm52ForCausalLMImpl : public LlmForCausalLMImplBase<Glm52Model> {
 public:
  explicit Glm52ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Glm52Model>(context) {}

  bool requires_graph_forward_metadata() const {
    return model_->requires_graph_forward_metadata();
  }

  std::unique_ptr<ModelGraphMetadataState> create_graph_forward_metadata_state()
      const {
    return model_->create_graph_forward_metadata_state();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input_params) const {
    model_->prepare_graph_forward_metadata(state, positions, input_params);
  }

  void load_model(
      std::unique_ptr<ModelLoader> loader,
      std::string prefix = "model." /*llm model weight prefix*/) override {
    LlmForCausalLMImplBase<Glm52Model>::load_model(std::move(loader), prefix);
    model_->verify_loaded_weights();
  }
};
TORCH_MODULE(Glm52ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(glm_moe_dsa, Glm52ForCausalLM);

}  // namespace xllm
