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

#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "core/common/flash_comm1_context.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/layers/common/attention_mask.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/qwen3_next_rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/npu_torch/qwen3_next_hybrid_decoder_layer_base.h"

namespace xllm {

class Qwen3HybridModelModule : public torch::nn::Module {
 public:
  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) = 0;
  virtual void load_state_dict(const StateDict& state_dict) = 0;
  virtual void verify_loaded_weights(const std::string& prefix) const = 0;
  virtual layer::WordEmbedding get_word_embedding() = 0;
  virtual void set_word_embedding(layer::WordEmbedding& word_embedding) = 0;
};

using Qwen3HybridModelModulePtr = std::shared_ptr<Qwen3HybridModelModule>;

class Qwen3HybridModelImplBase : public Qwen3HybridModelModule {
 public:
  explicit Qwen3HybridModelImplBase(const ModelContext& context)
      : device_(context.get_tensor_options().device()),
        model_args_(context.get_model_args()),
        parallel_args_(context.get_parallel_args()),
        flash_comm1_options_(context.get_flash_comm1_options()) {
    if (model_args_.n_routed_experts() > 0) {
      flash_comm1_options_.enable_flashcomm1 = false;
      flash_comm1_options_.enable_mmrs_fusion = false;
    }

    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args_.n_layers());
    device_ = options.device();
    dtype_ = options.dtype().toScalarType();
    norm_ = register_module(
        "norm",
        xllm::layer::Qwen3NextRMSNorm(
            model_args_.hidden_size(), model_args_.rms_norm_eps(), options));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/-9984);
    dense_attn_mask_ = layer::AttentionMask(options.device(),
                                            options.dtype().toScalarType(),
                                            /*mask_value=*/1);
    dp_size_ = parallel_args.dp_size();
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    // Disable gradient computation to reduce memory usage during inference
    torch::NoGradGuard no_grad;
    if (dp_size_ > 1) {
      if (tokens.sizes() == 0) {
        tokens = torch::tensor({1}).to(torch::kInt32).to(device_);
        positions = torch::tensor({0}).to(torch::kInt32).to(device_);
      }
    }

    layer::AttentionMetadata attn_metadata =
        layer::AttentionMetadataBuilder::build(
            input_params,
            model_args_.enable_mla(),
            build_attention_mask(input_params));
    const int32_t num_tokens = static_cast<int32_t>(tokens.size(0));
    const auto& batch_forward_type = input_params.meta.batch_forward_type;
    const bool is_prefill_side = batch_forward_type.no_decode();
    FlashComm1Context fc1_ctx = build_flash_comm1_context(
        num_tokens, is_prefill_side, parallel_args_, flash_comm1_options_);
    FlashComm1ContextScope fc1_scope(&fc1_ctx);

    torch::Tensor h;
    if (input_params.embedding.input_embedding.defined()) {
      h = input_params.embedding.input_embedding;
    } else {
      h = embed_tokens_(tokens);
    }

    if (is_sequence_sharded(fc1_ctx)) {
      h = shard_sequence(h, fc1_ctx);
    }

    torch::Tensor mrope_cos_sin;
    for (const auto& layer : layers_) {
      mrope_cos_sin = layer->build_mrope_cos_sin(positions);
      if (mrope_cos_sin.defined()) break;
    }

    std::optional<torch::Tensor> residual = std::nullopt;
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer->forward(h,
                         residual,
                         positions,
                         attn_metadata,
                         kv_caches[i],
                         input_params,
                         mrope_cos_sin);
#if defined(USE_NPU)
      if (input_params.parallel.layer_synchronizer != nullptr &&
          !input_params.parallel.layer_synchronizer->record_event(
              static_cast<int64_t>(i), device_.index())) {
        return ModelOutput();
      }
#endif
    }
    auto [hidden_states, residual_out] = norm_->forward(h, residual);
    h = hidden_states;
    if (is_sequence_sharded(fc1_ctx)) {
      h = gather_sequence(h, fc1_ctx);
    }
    return ModelOutput(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const override {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
  }

  layer::WordEmbedding get_word_embedding() override { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) override {
    embed_tokens_ = word_embedding;
  }

  void add_decoder_layer(layer::Qwen3HybridDecoderLayerModulePtr layer) {
    layers_.push_back(layer);
    blocks_->push_back(layer);
  }

  int32_t num_hidden_layers() const {
    return static_cast<int32_t>(layers_.size());
  }

 protected:
  torch::Tensor build_attention_mask(const ModelInputParams& input_params) {
#if defined(USE_NPU)
    // On NPU the hybrid path never consumes attn_metadata.attn_mask: full
    // attention runs through the fused-infer / paged-attention kernels (which
    // carry their own fixed fia_attn_mask or need no mask at all) and linear
    // attention is mask-free by construction. Materializing a dense
    // [seq_len, seq_len] mask here is pure waste and, for long sequences,
    // triggers an NPU OOM. Hand the kernels an empty mask unless a graph buffer
    // already supplies one.
    if (input_params.graph.attn_mask.defined()) {
      return input_params.graph.attn_mask;
    }
    return torch::Tensor();
#else
    if (input_params.graph.attn_mask.defined()) {
      return input_params.graph.attn_mask;
    }
    max_seq_len_ = std::max(input_params.meta.kv_max_seq_len, max_seq_len_);
    const bool use_append_mask =
        input_params.is_spec_verify ||
        input_params.meta.batch_forward_type.is_mixed() ||
        input_params.meta.batch_forward_type.is_chunked_prefill();
    if (!use_append_mask) {
      return dense_attn_mask_.get_attn_mask(max_seq_len_, dtype_, device_);
    }

    const int32_t num_sequences = input_params.meta.num_sequences;
    if (num_sequences <= 0) {
      return dense_attn_mask_.get_attn_mask(max_seq_len_, dtype_, device_);
    }

    std::vector<torch::Tensor> req_mask_vec;
    req_mask_vec.reserve(num_sequences);
    for (int32_t j = 0; j < num_sequences; ++j) {
      req_mask_vec.emplace_back(
          attn_mask_.gen_append_mask(input_params.attention.host.q_seq_lens[j],
                                     input_params.attention.host.kv_seq_lens[j],
                                     max_seq_len_,
                                     dtype_,
                                     device_));
    }
    return torch::cat(req_mask_vec, 0);
#endif
  }

  ModelArgs model_args_;
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<layer::Qwen3HybridDecoderLayerModulePtr> layers_;
  int32_t max_seq_len_ = 0;
  int32_t dp_size_ = 1;
  ParallelArgs parallel_args_;
  FlashComm1Options flash_comm1_options_;
  torch::Device device_;
  torch::ScalarType dtype_ = torch::kFloat;
  layer::Qwen3NextRMSNorm norm_{nullptr};
  layer::AttentionMask attn_mask_;
  layer::AttentionMask dense_attn_mask_;
  layer::WordEmbedding embed_tokens_{nullptr};
};

class Qwen3HybridForCausalLMImplBase : public torch::nn::Module {
 public:
  explicit Qwen3HybridForCausalLMImplBase(const ModelContext& context) {
    tie_word_embeddings_ = context.get_model_args().tie_word_embeddings();
    lm_head_ = register_module("lm_head", layer::LmHead(context));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return model_->forward(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return lm_head_(h);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    namespace F = torch::nn::functional;
    return F::normalize(h, F::NormalizeFuncOptions().p(2).dim(1));
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    load_model(std::move(loader), "model.", "lm_head.");
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  const std::string& model_prefix) {
    load_model(std::move(loader), model_prefix, "lm_head.");
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  const std::string& model_prefix,
                  const std::string& lm_head_prefix) {
    auto has_lm_head_weights = [](const StateDict& dict) {
      return dict.get_tensor("weight").defined() ||
             dict.get_tensor("qweight").defined();
    };

    for (const auto& state_dict : loader->get_state_dicts()) {
      auto model_state_dict = state_dict->get_dict_with_prefix(model_prefix);
      model_->load_state_dict(model_state_dict);

      auto lm_head_state_dict =
          state_dict->get_dict_with_prefix(lm_head_prefix);
      if (!has_lm_head_weights(lm_head_state_dict) && tie_word_embeddings_) {
        auto tied_lm_head_state_dict =
            model_state_dict.get_dict_with_prefix("embed_tokens.");
        if (has_lm_head_weights(tied_lm_head_state_dict)) {
          lm_head_state_dict = tied_lm_head_state_dict;
        }
      }
      lm_head_->load_state_dict(lm_head_state_dict);
    }
    model_->verify_loaded_weights(model_prefix);
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  bool is_hybrid_linear_attention() { return true; }

  layer::LmHead get_lm_head() { return lm_head_; }

  void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }

  void set_model_module(Qwen3HybridModelModulePtr model) {
    model_ = register_module("model", std::move(model));
  }

 protected:
  bool tie_word_embeddings_{false};
  layer::LmHead lm_head_{nullptr};
  Qwen3HybridModelModulePtr model_;
};

}  // namespace xllm
