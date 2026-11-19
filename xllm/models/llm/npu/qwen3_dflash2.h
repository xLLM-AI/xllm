/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/model_loader/model_loader.h"
#include "core/kernels/ops_api.h"
#include "core/layers/common/add_matmul.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/npu/rotary_embedding.h"
#include "models/llm/qwen3.h"
#include "models/model_registry.h"

namespace xllm::npu::model {

class DFlash2CandidateSelector final {
 public:
  DFlash2CandidateSelector(const ModelArgs& args,
                           const torch::TensorOptions& options)
      : options_(options),
        hidden_size_(args.hidden_size()),
        vocab_size_(args.vocab_size()),
        rank_(args.dflash2_selector_rank()),
        top_k_(args.dflash2_selector_top_k()) {
    CHECK_GT(rank_, 0) << "DFlash2 selector_rank must be positive.";
    CHECK_GT(top_k_, 0) << "DFlash2 selector_top_k must be positive.";
  }

  void load_state_dict(const StateDict& state_dict) {
    torch::Tensor hidden_projection =
        state_dict.get_tensor("hidden_projection.weight");
    torch::Tensor predecessor = state_dict.get_tensor("predecessor_codebook");
    torch::Tensor successor = state_dict.get_tensor("successor_codebook");
    if (hidden_projection.defined()) {
      hidden_projection_ = hidden_projection.to(options_);
    }
    if (predecessor.defined()) {
      predecessor_codebook_ = predecessor.to(options_);
    }
    if (successor.defined()) {
      successor_codebook_ = successor.to(options_);
    }
  }

  void verify_loaded_weights() const {
    CHECK(hidden_projection_.defined())
        << "Missing DFlash2 candidate_selector.hidden_projection.weight.";
    CHECK(predecessor_codebook_.defined())
        << "Missing DFlash2 candidate_selector.predecessor_codebook.";
    CHECK(successor_codebook_.defined())
        << "Missing DFlash2 candidate_selector.successor_codebook.";
    CHECK_EQ(hidden_projection_.sizes(),
             torch::IntArrayRef({rank_, hidden_size_}));
    CHECK_EQ(predecessor_codebook_.sizes(),
             torch::IntArrayRef({vocab_size_, rank_}));
    CHECK_EQ(successor_codebook_.sizes(),
             torch::IntArrayRef({vocab_size_, rank_}));
  }

  DFlash2CandidateOutput forward(const torch::Tensor& hidden_states,
                                 const torch::Tensor& unary_logits,
                                 const torch::Tensor& anchor_token_ids) const {
    CHECK_EQ(hidden_states.dim(), 3);
    CHECK_EQ(unary_logits.dim(), 3);
    CHECK_EQ(hidden_states.size(0), unary_logits.size(0));
    CHECK_EQ(hidden_states.size(1), unary_logits.size(1));
    CHECK_EQ(unary_logits.size(2), vocab_size_);
    CHECK_EQ(anchor_token_ids.dim(), 1);
    CHECK_EQ(anchor_token_ids.size(0), hidden_states.size(0));

    auto topk = torch::topk(unary_logits, top_k_, /*dim=*/-1);
    torch::Tensor values = std::get<0>(topk).to(torch::kFloat32);
    torch::Tensor candidate_ids = std::get<1>(topk).to(torch::kLong);

    namespace F = torch::nn::functional;
    torch::Tensor hidden = F::linear(hidden_states, hidden_projection_);
    torch::Tensor successors = F::embedding(candidate_ids, successor_codebook_);
    torch::Tensor anchor =
        anchor_token_ids.view({-1, 1, 1}).expand({-1, 1, top_k_});
    torch::Tensor predecessor_ids =
        torch::cat({anchor,
                    candidate_ids.slice(/*dim=*/1,
                                        /*start=*/0,
                                        candidate_ids.size(1) - 1)},
                   /*dim=*/1);
    torch::Tensor predecessors =
        F::embedding(predecessor_ids, predecessor_codebook_);
    torch::Tensor pair_scores =
        torch::einsum("blpr,blcr->blpc",
                      {predecessors * hidden.unsqueeze(/*dim=*/2), successors});

    DFlash2CandidateOutput output;
    output.candidate_ids = candidate_ids;
    output.edge_logits = values.unsqueeze(/*dim=*/2) + pair_scores;
    return output;
  }

 private:
  torch::Tensor hidden_projection_;
  torch::Tensor predecessor_codebook_;
  torch::Tensor successor_codebook_;
  torch::TensorOptions options_;
  int64_t hidden_size_ = 0;
  int64_t vocab_size_ = 0;
  int64_t rank_ = 0;
  int64_t top_k_ = 0;
};

class DFlash2Qwen3ModelImpl final : public ::xllm::QWen3ModelImpl {
 public:
  explicit DFlash2Qwen3ModelImpl(const ModelContext& context)
      : ::xllm::QWen3ModelImpl(context),
        selector_(context.get_model_args(), context.get_tensor_options()),
        parallel_args_(context.get_parallel_args()) {
    const ModelArgs& args = context.get_model_args();
    tensor_options_ = context.get_tensor_options();
    head_dim_ = args.head_dim();
    rms_norm_eps_ = args.rms_norm_eps();
    sliding_window_ = args.sliding_window();
    num_speculative_tokens_ = context.num_speculative_tokens();
    CHECK_GT(sliding_window_, 0);

    const auto& tp_group = *parallel_args_.tp_group_;
    const int32_t tp_size = tp_group.world_size();
    const int64_t kv_heads = args.n_kv_heads().value_or(args.n_heads());
    kv_tp_size_ = static_cast<int32_t>(std::min<int64_t>(kv_heads, tp_size));
    kv_tp_rank_ = tp_group.rank() / (tp_size / kv_tp_size_);
    local_kv_heads_ = kv_heads / kv_tp_size_;

    fc_ = register_module("fc",
                          layer::AddMatmul(args.hidden_size() * args.n_layers(),
                                           args.hidden_size(),
                                           /*with_bias=*/false,
                                           tensor_options_));
    hidden_norm_ = register_module(
        "hidden_norm",
        layer::RMSNorm(
            args.hidden_size(), args.rms_norm_eps(), tensor_options_));
    rotary_embedding_ = std::make_shared<xllm::RotaryEmbeddingGeneric>(
        head_dim_,
        args.max_position_embeddings(),
        layer::rotary::compute_inv_freq(
            head_dim_, args.rope_theta(), tensor_options_),
        /*interleaved=*/false,
        tensor_options_);
  }

  void load_state_dict(const StateDict& state_dict) override {
    if (state_dict.has(/*tensor_name=*/"embed_tokens.weight")) {
      if (embed_tokens_.is_empty()) {
        auto embedding = layer::WordEmbedding(model_args_.vocab_size(),
                                              model_args_.hidden_size(),
                                              parallel_args_,
                                              tensor_options_);
        set_word_embedding(embedding);
      }
      embed_tokens_->load_state_dict(
          state_dict.get_dict_with_prefix("embed_tokens."));
    }
    fc_->load_state_dict(state_dict.get_dict_with_prefix("fc."));
    hidden_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("hidden_norm."));
    selector_.load_state_dict(
        state_dict.get_dict_with_prefix("candidate_selector."));
    context_kv_weights_.resize(layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
      const StateDict layer_dict =
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + ".");
      layers_[i]->load_state_dict(layer_dict);
      auto& weights = context_kv_weights_[i];
      if (const auto key = layer_dict.get_sharded_tensor(
              /*tensor_name=*/"self_attn.k_proj.weight",
              /*dim=*/0,
              kv_tp_rank_,
              kv_tp_size_);
          key.defined()) {
        weights.key = key.to(tensor_options_);
      }
      if (const auto value = layer_dict.get_sharded_tensor(
              /*tensor_name=*/"self_attn.v_proj.weight",
              /*dim=*/0,
              kv_tp_rank_,
              kv_tp_size_);
          value.defined()) {
        weights.value = value.to(tensor_options_);
      }
      if (const auto key_norm =
              layer_dict.get_tensor(/*tensor_name=*/"self_attn.k_norm.weight");
          key_norm.defined()) {
        weights.key_norm = key_norm.to(tensor_options_).to(torch::kFloat32);
      }
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void finalize_loaded_weights() {
    fc_->verify_loaded_weights("fc.");
    hidden_norm_->verify_loaded_weights("hidden_norm.");
    norm_->verify_loaded_weights("norm.");
    selector_.verify_loaded_weights();
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      layers_[i]->verify_loaded_weights("layers." + std::to_string(i) + ".");
    }

    const size_t num_layers = context_kv_weights_.size();
    std::vector<torch::Tensor> kv_weights;
    std::vector<torch::Tensor> k_norm_weights;
    kv_weights.reserve(num_layers * 2);
    k_norm_weights.reserve(num_layers);
    for (const auto& weights : context_kv_weights_) {
      kv_weights.emplace_back(weights.key);
      kv_weights.emplace_back(weights.value);
      k_norm_weights.emplace_back(weights.key_norm);
    }
    fused_kv_weight_ = torch::stack(kv_weights, /*dim=*/0)
                           .view({static_cast<int64_t>(num_layers) * 2 *
                                      local_kv_heads_ * head_dim_,
                                  model_args_.hidden_size()});
    k_norm_weight_ = torch::stack(k_norm_weights, /*dim=*/0)
                         .view({static_cast<int64_t>(num_layers), 1, 1, -1});
    context_kv_weights_.clear();
  }

  DFlash2CandidateOutput candidates(
      const torch::Tensor& hidden_states,
      const torch::Tensor& unary_logits,
      const torch::Tensor& anchor_token_ids) const {
    return selector_.forward(hidden_states, unary_logits, anchor_token_ids);
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelParamsView& input_params) {
    const int64_t num_layers = static_cast<int64_t>(layers_.size());
    CHECK_EQ(static_cast<int64_t>(kv_caches.size()), num_layers);
    CHECK_EQ(device_cache_slots.numel(), target_hidden.size(0));
    torch::Tensor projected_hidden = fc_->forward(target_hidden);
    projected_hidden = std::get<0>(hidden_norm_->forward(projected_hidden));

    const int64_t num_context = projected_hidden.size(0);
    torch::Tensor all_kv =
        torch::nn::functional::linear(projected_hidden, fused_kv_weight_)
            .view({num_context, num_layers, 2, local_kv_heads_, head_dim_})
            .permute({2, 1, 0, 3, 4})
            .contiguous();
    torch::Tensor all_key =
        apply_k_norm(all_kv.select(/*dim=*/0, /*index=*/0), k_norm_weight_);
    torch::Tensor all_value = all_kv.select(/*dim=*/0, /*index=*/1);
    torch::Tensor flat_key =
        all_key.reshape({num_layers * num_context, local_kv_heads_, head_dim_});
    flat_key = std::get<1>(rotary_embedding_->forward(
        flat_key, flat_key, positions.repeat({num_layers})));
    all_key =
        flat_key.view({num_layers, num_context, local_kv_heads_, head_dim_});

    const int32_t device_index = all_key.device().index();
    for (int64_t i = 0; i < num_layers; ++i) {
      kernel::ReshapePagedCacheParams params;
      params.key = all_key[i];
      params.value = all_value[i];
      params.k_cache = kv_caches[i].get_k_cache();
      params.v_cache = kv_caches[i].get_v_cache();
      params.slot_mapping = device_cache_slots;
      CHECK_EQ(params.k_cache.dim(), 4);
      CHECK_EQ(params.v_cache->dim(), 4);
      CHECK_EQ(params.key.size(1), params.k_cache.size(2))
          << "DFlash2 context key/cache KV-head mismatch; key="
          << params.key.sizes() << ", cache=" << params.k_cache.sizes();
      CHECK_EQ(params.key.size(2), params.k_cache.size(3))
          << "DFlash2 context key/cache head-dim mismatch; key="
          << params.key.sizes() << ", cache=" << params.k_cache.sizes();
      CHECK_EQ(params.v_cache->sizes(), params.k_cache.sizes());
      kernel::reshape_paged_cache(params);
      if (input_params.parallel.layer_synchronizer != nullptr &&
          !input_params.parallel.layer_synchronizer->record_event(
              i, device_index)) {
        return ModelOutput();
      }
    }
    return ModelOutput(projected_hidden);
  }

 protected:
  layer::AttentionMetadata get_attention_metadata(
      const ModelParamsView& params) override {
    layer::AttentionMetadata metadata =
        QWen3ModelImpl::get_attention_metadata(params);
    metadata.is_causal = false;
#if defined(USE_NPU)
    metadata.fia_sparse_mode = 4;
    metadata.fia_pre_tokens = sliding_window_ - 1;
    metadata.fia_next_tokens = num_speculative_tokens_;
#endif
    return metadata;
  }

 private:
  torch::Tensor apply_k_norm(const torch::Tensor& key,
                             const torch::Tensor& weight) const {
    torch::Tensor key_fp32 = key.to(torch::kFloat32);
    torch::Tensor variance = key_fp32.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    return (key_fp32 * torch::rsqrt(variance + rms_norm_eps_) * weight)
        .to(key.scalar_type());
  }

  struct ContextKVWeights {
    torch::Tensor key;
    torch::Tensor value;
    torch::Tensor key_norm;
  };

  layer::AddMatmul fc_{nullptr};
  layer::RMSNorm hidden_norm_{nullptr};
  DFlash2CandidateSelector selector_;
  std::shared_ptr<xllm::NpuRotaryEmbedding> rotary_embedding_;
  std::vector<ContextKVWeights> context_kv_weights_;
  torch::Tensor fused_kv_weight_;
  torch::Tensor k_norm_weight_;
  torch::TensorOptions tensor_options_;
  ParallelArgs parallel_args_;
  int64_t head_dim_ = 0;
  int64_t local_kv_heads_ = 0;
  double rms_norm_eps_ = 1e-6;
  int32_t sliding_window_ = -1;
  int32_t num_speculative_tokens_ = 0;
  int32_t kv_tp_rank_ = 0;
  int32_t kv_tp_size_ = 1;
};
TORCH_MODULE(DFlash2Qwen3Model);

class DFlash2Qwen3ForCausalLMImpl final
    : public ::xllm::LlmForCausalLMImplBase<DFlash2Qwen3Model> {
 public:
  using Base = ::xllm::LlmForCausalLMImplBase<DFlash2Qwen3Model>;

  explicit DFlash2Qwen3ForCausalLMImpl(const ModelContext& context)
      : Base(context, /*share_lm_head=*/true) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "") override {
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
    }
    model_->finalize_loaded_weights();
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelParamsView& input_params) {
    return model_->write_context_kv(
        target_hidden, positions, device_cache_slots, kv_caches, input_params);
  }

  DFlash2CandidateOutput dflash2_candidates(
      const torch::Tensor& hidden_states,
      const torch::Tensor& unary_logits,
      const torch::Tensor& anchor_token_ids) {
    return model_->candidates(hidden_states, unary_logits, anchor_token_ids);
  }
};
TORCH_MODULE(DFlash2Qwen3ForCausalLM);

REGISTER_CAUSAL_MODEL(qwen3_dflash2, DFlash2Qwen3ForCausalLM);

}  // namespace xllm::npu::model
