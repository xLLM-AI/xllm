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

#include <glog/logging.h>
#include <torch/torch.h>

#include <string>
#include <vector>

#include "core/common/interruption_bus.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/lora/lora_context.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/rms_norm.h"
#include "core/util/rec_model_utils.h"
#include "models/model_registry.h"

namespace xllm {

template <typename DecoderLayerType>
class LlmModelImplBase : public torch::nn::Module {
 public:
  // mode type: qwen2, qwen3 .etc
  LlmModelImplBase(const std::string& model_type, const ModelArgs& args)
      : model_args_(args), model_type_(model_type) {
    InterruptionBus::get_instance().subscribe([this](bool interrupted) {
      this->layer_forward_interrupted_ = interrupted;
    });
    mrope_section_ = args.rope_scaling_mrope_section();
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids);
  }

  virtual std::pair<torch::Tensor, torch::Tensor> apply_mrope(
      const torch::Tensor positions) {
    return std::make_pair(torch::Tensor(), torch::Tensor());
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
    }
    auto inputs_embeds = input_params.embedding.input_embedding;
    // test
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = embed_tokens_(tokens);
    }

    auto modified_input_params = input_params;
    auto& dp_token_nums = modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(modified_input_params,
                                                     model_args_.enable_mla()));
    }
    auto& attn_metadata = *(modified_input_params.attn_metadata);
    if (positions.dim() == 2) {
      std::tie(attn_metadata.mrope_cos, attn_metadata.mrope_sin) =
          apply_mrope(positions);
    }

    const LlmRecMultiRoundParams* llmrec_params = nullptr;
    if (is_rec_multi_round_mode() &&
        modified_input_params.has_llmrec_params()) {
      llmrec_params = modified_input_params.llmrec_params();
      CHECK_EQ(llmrec_params->full_k_caches.size(), layers_.size())
          << "Rec multi-round mode requires full_k_caches per layer.";
      CHECK_EQ(llmrec_params->full_v_caches.size(), layers_.size())
          << "Rec multi-round mode requires full_v_caches per layer.";
      CHECK_EQ(llmrec_params->unshared_k_caches.size(), layers_.size())
          << "Rec multi-round mode requires unshared_k_caches per layer.";
      CHECK_EQ(llmrec_params->unshared_v_caches.size(), layers_.size())
          << "Rec multi-round mode requires unshared_v_caches per layer.";
    }

    // Multi-tenant LoRA: push a LoRAContext frame so LoRA-wrapped
    // Linear layers (QKV / o_proj / gate_up / down) can read the
    // per-batch adapter routing without extending Linear::forward.
    // set_lora_context_layer updates layer_index on the top frame
    // each iteration; LoRAScope pops on scope exit.
    LoRAContextFrame lora_frame;
    lora_frame.adapter_ids = &modified_input_params.adapter_ids;
    lora_frame.adapter_ids_per_token =
        &modified_input_params.adapter_ids_per_token;
    LoRAScope _lora_scope(lora_frame);

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      set_lora_context_layer(static_cast<int32_t>(i));
      if (llmrec_params != nullptr) {
        attn_metadata.full_k_cache = llmrec_params->full_k_caches[i];
        attn_metadata.full_v_cache = llmrec_params->full_v_caches[i];
        attn_metadata.unshared_k_cache = llmrec_params->unshared_k_caches[i];
        attn_metadata.unshared_v_cache = llmrec_params->unshared_v_caches[i];
      }
#if defined(USE_CUDA) || defined(USE_MUSA)
      attn_metadata.plan_info->layer_id = i;
#endif
      auto& layer = layers_[i];
      h = layer(h,
                residual,
                positions,
                attn_metadata,
                kv_caches[i],
                modified_input_params);
      if (!modified_input_params.record_layer(static_cast<uint32_t>(i),
                                              h.device())) {
        return ModelOutput();
      }
    }
    auto [hidden_states, residual_out] = norm_(h, residual);
    return ModelOutput(hidden_states, residual_out);
  }

  // load the weight from the checkpoint
  virtual void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));

    // call each layer's load_state_dict function
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  virtual layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  virtual void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 protected:
  torch::Tensor cos_sin_;
  int32_t max_seq_len_ = 0;
  std::vector<int64_t> mrope_section_;
  ModelArgs model_args_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::RMSNorm norm_{nullptr};

  // hold same data but different type as blocks_ to avoid type cast
  std::vector<DecoderLayerType> layers_;

  bool layer_forward_interrupted_ = false;

 private:
  std::string model_type_;
};

template <typename LlmModelType>
class LlmForCausalLMImplBase : public torch::nn::Module {
 public:
  LlmForCausalLMImplBase(const ModelContext& context) {
    tie_word_embeddings = context.get_model_args().tie_word_embeddings();
    embedding_mode_ = context.get_model_args().embedding_mode();
    // register submodules
    model_ = register_module("model", LlmModelType(context));
    if (!embedding_mode_) {
      lm_head_ = register_module("lm_head", layer::LmHead(context));
    } else {
      LOG(INFO) << "Skip registering lm_head in embedding mode.";
    }
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
    CHECK(!lm_head_.is_empty())
        << "lm_head is not initialized in embedding mode.";
    // select tokens if provided
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return lm_head_(h);
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

      if (!embedding_mode_) {
        if (tie_word_embeddings) {
          auto lm_head_state_dict =
              state_dict->get_dict_with_prefix(std::vector<std::string>{
                  prefix + "embed_tokens.", "embed_tokens.", "embed."});
          lm_head_->load_state_dict(lm_head_state_dict);
        } else {
          auto lm_head_state_dict = state_dict->get_dict_with_prefix(
              std::vector<std::string>{"lm_head.",
                                       "model.lm_head.",
                                       "model.head.",
                                       "head.",
                                       prefix,
                                       prefix + "lm_head.",
                                       prefix + "head."});
          lm_head_->load_state_dict(lm_head_state_dict);
        }
      }
    }
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  virtual layer::LmHead get_lm_head() {
    CHECK(!lm_head_.is_empty())
        << "lm_head is not initialized in embedding mode.";
    return lm_head_;
  }

  virtual void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  virtual layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  }

  virtual void set_word_embedding(layer::WordEmbedding& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }

 protected:
  // parameter members, must be registered
  LlmModelType model_{nullptr};
  bool tie_word_embeddings{false};
  bool embedding_mode_{false};
  // test
  layer::LmHead lm_head_{nullptr};
};

}  // namespace xllm
