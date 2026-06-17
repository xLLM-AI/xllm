/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <atb/atb_infer.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "core/common/global_flags.h"
#include "core/common/interruption_bus.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model/model_traits.h"
#include "core/framework/model_context.h"
#include "core/layers/common/attention_mask.h"
#include "core/layers/npu/loader/base_loader.h"
#include "core/layers/npu/loader/rolling_load_manager.h"
#include "core/layers/npu/loader/rolling_weight_buffer.h"
#include "core/layers/npu/npu_block_copy_impl.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_pos_embedding_impl.h"
#include "core/layers/npu/npu_rms_norm_impl.h"
#include "core/layers/npu/npu_word_embedding_impl.h"
#include "models/model_registry.h"
#include "xllm_atb_layers/core/include/atb_speed/log.h"

namespace xllm::npu::model {

template <typename DecoderType>
class LlmDecoderLayerImplBase : public torch::nn::Module {
 public:
  LlmDecoderLayerImplBase(const ModelContext& context,
                          const int32_t layer_id = -1)
      : layer_id_(layer_id) {
    CHECK(layer_id_ >= 0) << "layer_id must be >= 0, but got " << layer_id_;
    // register submodules
    decoder_layer_ = register_module("decoder_layer", DecoderType(context));
    block_copy_ = register_module("block_copy", layer::NpuBlockCopy(context));
    decoder_layer_->set_layer_id(layer_id_);
  }

  virtual torch::Tensor forward(torch::Tensor& x,
                                torch::Tensor& cos_pos,
                                torch::Tensor& sin_pos,
                                torch::Tensor& attn_mask,
                                KVCache& kv_cache,
                                ModelInputParams& input_params,
                                aclrtEvent* event,
                                std::atomic<bool>* event_flag) {
    if (input_params.block_copy.src_block_indices.numel() > 0) {
      block_copy_(kv_cache.get_k_cache(),
                  kv_cache.get_v_cache(),
                  input_params.block_copy.src_block_indices,
                  input_params.block_copy.dst_block_indices,
                  input_params.block_copy.cum_sum,
                  0);
    }

    return decoder_layer_(x,
                          cos_pos,
                          sin_pos,
                          attn_mask,
                          kv_cache,
                          input_params,
                          event,
                          event_flag,
                          layer_id_);
  }

  virtual void verify_loaded_weights(const std::string& prefix) const {
    decoder_layer_->verify_loaded_weights();
  }
  virtual void merge_loaded_weights() {
    decoder_layer_->merge_loaded_weights();
    block_copy_->merge_loaded_weights();
  }

  // load the weight from the checkpoint
  virtual void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    decoder_layer_->load_state_dict(state_dict);
  }

  virtual void merge_and_move_pinned_host() {
    decoder_layer_->merge_and_move_pinned_host();
    block_copy_->merge_loaded_weights();
  }

  virtual void free_weights() { decoder_layer_->free_weights(); }

  virtual void reload_weights() { decoder_layer_->reload_weights(); }

  virtual void reload_weights_from_device() {
    decoder_layer_->reload_weights_from_device();
  }

  virtual layer::BaseLoader* get_loader() {
    return decoder_layer_->get_loader();
  }

  virtual void refresh_rolling_weights() {
    decoder_layer_->refresh_rolling_weights();
  }

  virtual void set_residual(torch::Tensor residual) {
    decoder_layer_->set_residual(residual);
  }

 private:
  DecoderType decoder_layer_{nullptr};
  layer::NpuBlockCopy block_copy_{nullptr};
  int32_t layer_id_;
};

template <typename DecoderLayerType>
class LlmModelImplBase : public torch::nn::Module {
 public:
  // mode type: qwen2, qwen3 .etc
  LlmModelImplBase(const std::string& model_type, const ModelArgs& args)
      : model_type_(model_type) {
    InterruptionBus::get_instance().subscribe([this](bool interrupted) {
      this->layer_forward_interrupted_ = interrupted;
    });
    mrope_section_ = args.rope_scaling_mrope_section();
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return npu_embed_tokens_(input_ids, 0);
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({0}).to(torch::kInt32).to(tokens.device());
    }
    auto inputs_embeds = input_params.embedding.input_embedding;
    // test
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = npu_embed_tokens_(tokens, 0);
    }

    auto target_cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();

    if (positions.dim() == 2) {  // mrope
      auto apply = [this](torch::Tensor x) {
        auto sections = mrope_section_;
        sections.insert(sections.end(), sections.begin(), sections.end());

        auto vec = x.split(sections, -1);
        std::vector<torch::Tensor> selects;
        selects.reserve(vec.size());

        for (int64_t i = 0; i < vec.size(); ++i) {
          auto m = vec[i];
          selects.push_back(m[i % mrope_section_.size()]);
        }
        return torch::cat(selects, -1);
      };
      cos_pos = apply(cos_pos.reshape(
          {positions.sizes().front(), -1, cos_pos.sizes().back()}));
      sin_pos = apply(sin_pos.reshape(
          {positions.sizes().front(), -1, sin_pos.sizes().back()}));
    }

    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    torch::Tensor attn_mask;
    max_seq_len_ =
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()
            ? std::max(input_params.meta.kv_max_seq_len, max_seq_len_)
            : 128;
    if (model_type_ == "qwen2") {
      attn_mask = attn_mask_.get_attn_mask(
          max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());
    } else {
      if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
        int num_sequences = input_params.meta.num_sequences;
        if (num_sequences > 0) {
          std::vector<torch::Tensor> req_mask_vec;
          req_mask_vec.reserve(num_sequences);

          for (int j = 0; j < num_sequences; j++) {
            auto mask = attn_mask_.gen_append_mask(
                input_params.attention.host.q_seq_lens[j],
                input_params.attention.host.kv_seq_lens[j],
                max_seq_len_,
                cos_pos.dtype().toScalarType(),
                cos_pos.device());
            req_mask_vec.emplace_back(mask);
          }
          attn_mask = torch::cat(req_mask_vec, 0);
        }
      } else {
        attn_mask = attn_mask_.get_attn_mask(
            max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());
      }
    }

    RollingLayerGuard rolling_guard(rolling_mgr_);

    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event = nullptr;
      std::atomic<bool>* event_flag = nullptr;
      if (input_params.parallel.layer_synchronizer != nullptr) {
        event = input_params.parallel.layer_synchronizer->get_event(i);
        event_flag =
            input_params.parallel.layer_synchronizer->get_event_flag(i);
      }
      if (!input_params.synchronize_layer(i)) {
        return ModelOutput();
      }

      auto& layer = layers_[i];

      if (layer_forward_interrupted_) {
        LOG(INFO) << "Forward interrupted at layer: " << i;
        return ModelOutput();
      }
      const int32_t layer_index = i;
      rolling_guard.before_layer(layer_index);

      layer(h,
            cos_pos,
            sin_pos,
            attn_mask,
            kv_caches[i],
            input_params_new,
            event,
            event_flag);

      rolling_guard.after_layer(layer_index);
    }

    auto hidden_states = norm_(h, 0);
    return ModelOutput(hidden_states);
  }

  // load the weight from the checkpoint
  virtual void load_state_dict(const StateDict& state_dict) {
    npu_embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  virtual void verify_loaded_weights(const std::string& prefix) const {
    npu_embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");

    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  virtual void merge_loaded_weights() {
    npu_embed_tokens_->merge_loaded_weights();

    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_loaded_weights();
    }
    norm_->merge_loaded_weights();
  }

  virtual void free_weights() {
    npu_embed_tokens_->free_weights();
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->free_weights();
    }
    norm_->free_weights();
  }

  virtual void reload_weights() {
    npu_embed_tokens_->reload_weights();
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->reload_weights();
    }
    norm_->reload_weights();
  }

  virtual void reload_non_decoder_weights() {
    npu_embed_tokens_->reload_weights();
    norm_->reload_weights();
  }

  virtual void reload_weights_from_device() {
    npu_embed_tokens_->reload_weights_from_device();
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->reload_weights_from_device();
    }
    norm_->reload_weights_from_device();
  }

  virtual void merge_and_move_pinned_host() {
    npu_embed_tokens_->merge_and_move_pinned_host();
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_and_move_pinned_host();
    }
    norm_->merge_and_move_pinned_host();
  }

  // Collect BaseLoader* from each decoder layer (in order)
  virtual std::vector<layer::BaseLoader*> get_decoder_loaders() {
    std::vector<layer::BaseLoader*> loaders;
    loaders.reserve(layers_.size());
    for (auto& l : layers_) {
      loaders.push_back(l->get_loader());
    }
    return loaders;
  }

  // Inject rolling load manager (not owned, managed by WorkerImpl)
  void set_rolling_load_manager(RollingLoadManager* mgr) { rolling_mgr_ = mgr; }

  // For rolling load: refresh decoder layers' rolling device pointers and
  // corresponding AT/ATB tensor bindings.
  virtual void refresh_rolling_weights() {
    for (auto& layer : layers_) {
      layer->refresh_rolling_weights();
    }
  }

  virtual layer::NpuWordEmbedding get_npu_word_embedding() {
    return npu_embed_tokens_;
  }

  virtual void set_npu_word_embedding(
      layer::NpuWordEmbedding& npu_word_embedding) {
    npu_embed_tokens_ = npu_word_embedding;
  }

  virtual void set_residual(torch::Tensor residual) {
    for (auto& layer : layers_) {
      layer->set_residual(residual);
    }
  }

 protected:
  torch::Tensor cos_sin_;
  torch::Tensor cos_pos_;
  torch::Tensor sin_pos_;
  int device_id = 0;
  layer::AttentionMask attn_mask_;
  int dp_rank_ = 0;
  layer::NpuPosEmbedding atb_pos_emb_{nullptr};

  std::vector<int64_t> mrope_section_;
  // test
  //  ParallelEmbedding embed_tokens_{nullptr};
  layer::NpuWordEmbedding npu_embed_tokens_{nullptr};
  layer::NpuRMSNorm norm_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<DecoderLayerType> layers_;

  bool layer_forward_interrupted_ = false;

  int32_t max_seq_len_ = 0;

  RollingLoadManager* rolling_mgr_ =
      nullptr;  // not owned; managed by LlmForCausalLMImplBase

 private:
  std::string model_type_;
};

template <typename LlmModelType>
class LlmForCausalLMImplBase : public torch::nn::Module {
 public:
  LlmForCausalLMImplBase(const ModelContext& context) {
    tie_word_embeddings = context.get_model_args().tie_word_embeddings();
    // register submodules
    model_ = register_module("model", LlmModelType(context));

    npu_lm_head_ = register_module("npu_lm_head", layer::NpuLmHead(context));
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return model_->get_input_embeddings(input_ids);
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    return npu_lm_head_(hidden_states, seleted_idxes, 0);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // out_hidden: [num_seqs, hidden_size]
  // returns: [num_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes,
                               torch::Tensor& out_hidden) {
    return npu_lm_head_->forward_with_hidden(
        hidden_states, seleted_idxes, out_hidden, 0);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_seqs, hidden_size]
  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return h;
  }

  virtual void load_model(
      std::unique_ptr<ModelLoader> loader,
      std::string prefix = "model." /*llm model weight prefix*/) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      // The same model_type may come from checkpoints with different top-level
      // weight prefixes. Try these candidate prefixes in order to improve
      // compatibility across such variants.
      model_->load_state_dict(state_dict->get_dict_with_prefix(
          std::vector<std::string>{"model.language_model.",
                                   "language_model.model.",
                                   prefix,
                                   "model.",
                                   ""}));
      if (tie_word_embeddings) {
        npu_lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(std::vector<std::string>{
                prefix + "embed_tokens.", "embed_tokens."}));
      } else {
        npu_lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix("lm_head."));
      }
    }

    // verify
    model_->verify_loaded_weights(prefix);
    if (tie_word_embeddings) {
      npu_lm_head_->verify_loaded_weights("embed_tokens.");
    } else {
      npu_lm_head_->verify_loaded_weights("lm_head.");
    }

    model_->merge_loaded_weights();
    // test
    npu_lm_head_->merge_loaded_weights();
  }

  virtual void lazy_load_model(
      std::unique_ptr<ModelLoader> loader,
      std::string prefix = "model." /*llm model weight prefix*/) {
    if (keep_host_weights) {
      LOG(INFO) << "Model weights are already kept on host.";
      return;
    }
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
      if (tie_word_embeddings) {
        npu_lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "embed_tokens."));
      } else {
        npu_lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix("lm_head."));
      }
    }
    // verify
    model_->verify_loaded_weights(prefix);
    npu_lm_head_->verify_loaded_weights("lm_head.");

    model_->merge_and_move_pinned_host();
    // test
    npu_lm_head_->merge_and_move_pinned_host();

    keep_host_weights = true;
  }

  virtual void free_model_weights() {
    if (!keep_host_weights) {
      LOG(INFO) << "Model weights are not kept on host.";
      return;
    }
    model_->free_weights();
    npu_lm_head_->free_weights();
    keep_host_weights = false;
  }

  virtual void reload_model_weights() {
    model_->reload_weights();
    npu_lm_head_->reload_weights();
    auto stream = c10_npu::getCurrentNPUStream();
    stream.synchronize();
  }

  virtual void init_rolling_model_state() {
    model_->reload_non_decoder_weights();
    model_->refresh_rolling_weights();
    npu_lm_head_->reload_weights();
    auto stream = c10_npu::getCurrentNPUStream();
    stream.synchronize();
  }

  virtual void reload_model_weights_from_device() {
    model_->reload_weights_from_device();
    npu_lm_head_->reload_weights_from_device();
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  virtual layer::NpuLmHead get_npu_lm_head() { return npu_lm_head_; }

  virtual void set_npu_lm_head(layer::NpuLmHead& head) { npu_lm_head_ = head; }

  virtual layer::NpuWordEmbedding get_npu_word_embedding() {
    return model_->get_npu_word_embedding();
  }

  virtual void set_npu_word_embedding(
      layer::NpuWordEmbedding& npu_word_embedding) {
    model_->set_npu_word_embedding(npu_word_embedding);
  }

  virtual std::vector<layer::BaseLoader*> get_decoder_loaders() {
    return model_->get_decoder_loaders();
  }

  virtual void set_rolling_load_manager(RollingLoadManager* mgr) {
    model_->set_rolling_load_manager(mgr);
  }

  virtual bool init_or_refresh_rolling_runtime(Stream* load_stream,
                                               Stream* compute_stream,
                                               int32_t num_cached_slots,
                                               int32_t requested_rolling_slots,
                                               const std::string& model_id) {
    CHECK(load_stream != nullptr) << "load_stream is null for rolling load";
    CHECK(compute_stream != nullptr)
        << "compute_stream is null for rolling load";

    if (rolling_load_manager_ == nullptr) {
      auto loaders = model_->get_decoder_loaders();
      CHECK(!loaders.empty()) << "No decoder loaders found for rolling load";
      size_t max_storage_size = 0;
      for (size_t i = 0; i < loaders.size(); ++i) {
        CHECK(loaders[i] != nullptr) << "Decoder loader[" << i << "] is null";
        const size_t layer_storage_size = loaders[i]->get_storage_size();
        CHECK_GT(layer_storage_size, 0)
            << "Decoder loader[" << i << "] invalid storage_size";
        if (layer_storage_size > max_storage_size) {
          max_storage_size = layer_storage_size;
        }
      }
      CHECK_GT(max_storage_size, 0)
          << "Failed to determine max decoder layer storage_size";

      rolling_weight_buffer_ = std::make_shared<layer::RollingWeightBuffer>(
          num_cached_slots, max_storage_size, model_id);
      rolling_load_manager_ =
          std::make_unique<RollingLoadManager>(loaders,
                                               rolling_weight_buffer_,
                                               load_stream,
                                               compute_stream,
                                               requested_rolling_slots);
      LOG(INFO) << "Rolling runtime init: num_cached_slots=" << num_cached_slots
                << ", max_decoder_layer_storage_size=" << max_storage_size;

      for (size_t i = 0; i < loaders.size(); ++i) {
        const int32_t layer_index = i;
        const int32_t slot = rolling_load_manager_->slot_for_layer(layer_index);
        loaders[i]->set_rolling_buffer(rolling_weight_buffer_, slot);
      }
    } else {
      rolling_load_manager_->refresh_rolling_buffer_address();
    }

    model_->set_rolling_load_manager(rolling_load_manager_.get());
    init_rolling_model_state();
    rolling_load_manager_->init_rolling_load();
    return true;
  }

 protected:
  // parameter members, must be registered
  LlmModelType model_{nullptr};
  int device_id = 0;
  bool tie_word_embeddings{false};
  bool keep_host_weights{false};
  std::shared_ptr<layer::RollingWeightBuffer> rolling_weight_buffer_{nullptr};
  std::unique_ptr<RollingLoadManager> rolling_load_manager_{nullptr};
  // test
  layer::NpuLmHead npu_lm_head_{nullptr};
};

}  // namespace xllm::npu::model
