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

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "core/kernels/ops_api.h"
#include "core/layers/common/add_matmul.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/npu/rotary_embedding.h"
#include "framework/model_loader.h"
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
        selector_(context.get_model_args(), context.get_tensor_options()) {
    const ModelArgs& args = context.get_model_args();
    const ParallelArgs& parallel_args = context.get_parallel_args();
    tensor_options_ = context.get_tensor_options();
    head_dim_ = args.head_dim();
    rms_norm_eps_ = args.rms_norm_eps();
    sliding_window_ = args.sliding_window();
    block_size_ = args.dflash2_block_size();
    CHECK_GT(sliding_window_, 0);
    CHECK_GT(block_size_, 0);

    const int32_t dp_size = parallel_args.dp_size();
    const int32_t cp_size = parallel_args.cp_size();
    CHECK_GT(dp_size, 0);
    CHECK_GT(cp_size, 0);
    CHECK_EQ(parallel_args.world_size() % (dp_size * cp_size), 0);
    tp_size_ = parallel_args.world_size() / (dp_size * cp_size);
    tp_rank_ = parallel_args.rank() % tp_size_;

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
    fc_->load_state_dict(state_dict.get_dict_with_prefix("fc."));
    hidden_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("hidden_norm."));
    selector_.load_state_dict(
        state_dict.get_dict_with_prefix("candidate_selector."));
    load_context_kv_weights(state_dict);
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights() const {
    fc_->verify_loaded_weights("fc.");
    hidden_norm_->verify_loaded_weights("hidden_norm.");
    norm_->verify_loaded_weights("norm.");
    selector_.verify_loaded_weights();
    verify_context_kv_weights();
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      layers_[i]->verify_loaded_weights("layers." + std::to_string(i) + ".");
    }
  }

  void finalize_loaded_weights() { build_fused_context_kv_weights(); }

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
                               const ModelInputParams& input_params) {
    const int64_t num_layers = static_cast<int64_t>(layers_.size());
    CHECK_EQ(static_cast<int64_t>(kv_caches.size()), num_layers);
    CHECK_EQ(device_cache_slots.numel(), target_hidden.size(0));
    torch::Tensor projected_hidden = fc_->forward(target_hidden);
    projected_hidden = std::get<0>(hidden_norm_->forward(projected_hidden));
    CHECK(fused_kv_weight_.defined());

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
    flat_key = apply_rope(flat_key, positions.repeat({num_layers}));
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
      CHECK_EQ(params.key.dim(), 3);
      CHECK_EQ(params.value->dim(), 3);
      CHECK_EQ(params.k_cache.dim(), 4);
      CHECK_EQ(params.v_cache->dim(), 4);
      CHECK_EQ(params.key.size(0), params.slot_mapping.numel());
      CHECK_EQ(params.key.size(1), params.k_cache.size(2))
          << "DFlash2 context key/cache KV-head mismatch; key="
          << params.key.sizes() << ", cache=" << params.k_cache.sizes();
      CHECK_EQ(params.key.size(2), params.k_cache.size(3))
          << "DFlash2 context key/cache head-dim mismatch; key="
          << params.key.sizes() << ", cache=" << params.k_cache.sizes();
      CHECK_EQ(params.value->sizes(), params.key.sizes());
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
      const ModelInputParams& params,
      const torch::Tensor& h) override {
    layer::AttentionMetadata metadata =
        QWen3ModelImpl::get_attention_metadata(params, h);
    // DFlash2 jointly denoises the whole query block. Keep the 2048x2048 FIA
    // optimized mask built by the base metadata path and use band mode to
    // expose the full proposal block within the checkpoint's sliding window.
    metadata.is_causal = false;
#if defined(USE_NPU)
    metadata.fia_sparse_mode = 4;
    metadata.fia_pre_tokens = sliding_window_ - 1;
    metadata.fia_next_tokens = block_size_ - 1;
#endif
    return metadata;
  }

  torch::Tensor gen_append_attn_mask(int32_t q_len,
                                     int32_t kv_len,
                                     int32_t max_kv_len,
                                     torch::Dtype dtype,
                                     torch::Device device) override {
    CHECK_GT(q_len, 0);
    CHECK_GE(kv_len, q_len);
    CHECK_GE(max_kv_len, kv_len);

    const torch::TensorOptions index_options =
        torch::TensorOptions().dtype(torch::kLong).device(device);
    const torch::Tensor key_positions =
        torch::arange(max_kv_len, index_options).view({1, max_kv_len});
    const torch::Tensor query_offsets =
        torch::arange(q_len, index_options).view({q_len, 1});
    const int64_t first_query_position = kv_len - q_len;
    const torch::Tensor first_visible_key =
        query_offsets + first_query_position - (sliding_window_ - 1);
    const torch::Tensor last_visible_key =
        query_offsets + first_query_position + (block_size_ - 1);
    const torch::Tensor masked = key_positions.lt(first_visible_key) |
                                 key_positions.gt(last_visible_key) |
                                 key_positions.ge(kv_len);

    // Match AttentionMask's numeric convention: fp16 uses -inf while other
    // dtypes use the FIA-compatible finite sentinel.
    const float mask_value = dtype == torch::kFloat16
                                 ? -std::numeric_limits<float>::infinity()
                                 : -9984.0f;
    const torch::TensorOptions mask_options =
        torch::TensorOptions().dtype(dtype).device(device);
    return torch::zeros({q_len, max_kv_len}, mask_options)
        .masked_fill(masked, mask_value);
  }

 private:
  torch::Tensor apply_k_norm(const torch::Tensor& key,
                             const torch::Tensor& weight) const {
    torch::Tensor key_fp32 = key.to(torch::kFloat32);
    torch::Tensor variance = key_fp32.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    return (key_fp32 * torch::rsqrt(variance + rms_norm_eps_) * weight)
        .to(key.scalar_type());
  }

  torch::Tensor apply_rope(const torch::Tensor& key,
                           const torch::Tensor& positions) const {
    CHECK(rotary_embedding_ != nullptr);
    return std::get<1>(rotary_embedding_->forward(key, key, positions));
  }

  void load_context_kv_weights(const StateDict& state_dict) {
    const int32_t num_layers = static_cast<int32_t>(layers_.size());
    if (per_layer_k_proj_.empty()) {
      per_layer_k_proj_.resize(num_layers);
      per_layer_v_proj_.resize(num_layers);
      per_layer_k_norm_.resize(num_layers);
    }
    for (int32_t i = 0; i < num_layers; ++i) {
      StateDict layer_dict =
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + ".");
      torch::Tensor k_proj = layer_dict.get_sharded_tensor(
          "self_attn.k_proj.weight", /*dim=*/0, tp_rank_, tp_size_);
      torch::Tensor v_proj = layer_dict.get_sharded_tensor(
          "self_attn.v_proj.weight", /*dim=*/0, tp_rank_, tp_size_);
      torch::Tensor k_norm = layer_dict.get_tensor("self_attn.k_norm.weight");
      if (!k_proj.defined() && !v_proj.defined() && !k_norm.defined()) {
        continue;
      }
      CHECK(k_proj.defined());
      CHECK(v_proj.defined());
      CHECK(k_norm.defined());
      const int64_t local_kv_heads = k_proj.size(0) / head_dim_;
      if (local_kv_heads_ == 0) {
        local_kv_heads_ = local_kv_heads;
      }
      CHECK_EQ(local_kv_heads_, local_kv_heads);
      per_layer_k_proj_[i] = k_proj.to(tensor_options_);
      per_layer_v_proj_[i] = v_proj.to(tensor_options_);
      per_layer_k_norm_[i] = k_norm.to(tensor_options_).to(torch::kFloat32);
    }
  }

  void verify_context_kv_weights() const {
    const int32_t num_layers = static_cast<int32_t>(layers_.size());
    CHECK_EQ(static_cast<int32_t>(per_layer_k_proj_.size()), num_layers);
    CHECK_GT(local_kv_heads_, 0);
    for (int32_t i = 0; i < num_layers; ++i) {
      CHECK(per_layer_k_proj_[i].defined());
      CHECK(per_layer_v_proj_[i].defined());
      CHECK(per_layer_k_norm_[i].defined());
    }
  }

  void build_fused_context_kv_weights() {
    const int32_t num_layers = static_cast<int32_t>(layers_.size());
    std::vector<torch::Tensor> kv_weights;
    std::vector<torch::Tensor> k_norm_weights;
    kv_weights.reserve(num_layers * 2);
    k_norm_weights.reserve(num_layers);
    for (int32_t i = 0; i < num_layers; ++i) {
      kv_weights.emplace_back(per_layer_k_proj_[i]);
      kv_weights.emplace_back(per_layer_v_proj_[i]);
      k_norm_weights.emplace_back(per_layer_k_norm_[i]);
    }
    fused_kv_weight_ = torch::cat(kv_weights, /*dim=*/0).contiguous();
    k_norm_weight_ =
        torch::stack(k_norm_weights, /*dim=*/0).view({num_layers, 1, 1, -1});
    per_layer_k_proj_.clear();
    per_layer_v_proj_.clear();
    per_layer_k_norm_.clear();
  }

  layer::AddMatmul fc_{nullptr};
  layer::RMSNorm hidden_norm_{nullptr};
  DFlash2CandidateSelector selector_;
  std::shared_ptr<xllm::NpuRotaryEmbedding> rotary_embedding_;
  std::vector<torch::Tensor> per_layer_k_proj_;
  std::vector<torch::Tensor> per_layer_v_proj_;
  std::vector<torch::Tensor> per_layer_k_norm_;
  torch::Tensor fused_kv_weight_;
  torch::Tensor k_norm_weight_;
  torch::TensorOptions tensor_options_;
  int64_t head_dim_ = 0;
  int64_t local_kv_heads_ = 0;
  double rms_norm_eps_ = 1e-6;
  int32_t sliding_window_ = -1;
  int32_t block_size_ = 0;
  int32_t tp_rank_ = 0;
  int32_t tp_size_ = 1;
};
TORCH_MODULE(DFlash2Qwen3Model);

class DFlash2Qwen3ForCausalLMImpl final
    : public ::xllm::LlmForCausalLMImplBase<DFlash2Qwen3Model> {
 public:
  using Base = ::xllm::LlmForCausalLMImplBase<DFlash2Qwen3Model>;

  explicit DFlash2Qwen3ForCausalLMImpl(const ModelContext& context)
      : Base(context) {}

  using Base::logits;

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes,
                       torch::Tensor& out_hidden) {
    out_hidden = selected_idxes.defined()
                     ? hidden_states.index_select(
                           /*dim=*/0, selected_idxes.to(torch::kLong))
                     : hidden_states;
    return lm_head_(out_hidden);
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "") override {
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
    }
    model_->verify_loaded_weights();
    model_->finalize_loaded_weights();
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& input_params) {
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

REGISTER_CAUSAL_MODEL_WITH_VARNAME(dflash2_draft_model,
                                   DFlash2DraftModel,
                                   DFlash2Qwen3ForCausalLM);

}  // namespace xllm::npu::model
