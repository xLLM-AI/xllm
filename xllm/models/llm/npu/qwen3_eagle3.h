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

#include <atb/atb_infer.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <string>
#include <vector>

#include "core/common/global_flags.h"
#include "core/common/interruption_bus.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/layers/common/attention_mask.h"
#include "core/layers/common/rotary_embedding_util.h"
#include "core/layers/npu/npu_column_parallel_linear_impl.h"
#include "core/layers/npu/npu_eagle3_decoder_layer_impl.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_pos_embedding_impl.h"
#include "core/layers/npu/npu_rms_norm_impl.h"
#include "core/layers/npu/npu_word_embedding_impl.h"
#include "models/model_registry.h"

namespace xllm::npu::model {

// EAGLE-3 specific decoder layer that accepts embeds and hidden_states
// separately, applies layernorms, then concatenates them
class QWen3Eagle3DecoderLayerImpl : public torch::nn::Module {
 public:
  QWen3Eagle3DecoderLayerImpl(const ModelContext& context,
                              const int32_t layer_id = 0)
      : layer_id_(layer_id) {
    CHECK(layer_id_ >= 0) << "layer_id must be >= 0, but got " << layer_id_;
    // register submodules
    decoder_layer_ =
        register_module("decoder_layer", layer::NpuEagle3DecoderLayer(context));
  }

  // Forward with separate hidden_states and hidden_states_extra (matches Python
  // implementation)
  virtual torch::Tensor forward(torch::Tensor& hidden_states,
                                torch::Tensor& hidden_states_extra,
                                torch::Tensor& cos_pos,
                                torch::Tensor& sin_pos,
                                torch::Tensor& attn_mask,
                                KVCache& kv_cache,
                                ModelInputParams& input_params,
                                aclrtEvent* event,
                                std::atomic<bool>* event_flag) {
    return decoder_layer_(hidden_states,
                          hidden_states_extra,
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
  }

  // load the weight from the checkpoint
  virtual void load_state_dict(const StateDict& state_dict) {
    decoder_layer_->load_state_dict(state_dict);
  }

 private:
  layer::NpuEagle3DecoderLayer decoder_layer_{nullptr};
  int32_t layer_id_;
};
TORCH_MODULE(QWen3Eagle3DecoderLayer);

class QWen3Eagle3ModelImpl : public torch::nn::Module {
 public:
  QWen3Eagle3ModelImpl(const std::string& model_type,
                       const ModelContext& context)
      : model_type_(model_type), options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto parallel_args = context.get_parallel_args();

    dp_size_ = parallel_args.dp_size();
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;

    // Word embedding
    embed_tokens_ =
        register_module("embed_tokens", layer::NpuWordEmbedding(context));

    // Position embedding
    atb_pos_emb_ = layer::NpuPosEmbedding(context);
    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        model_args.head_dim(),
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options_);

    int32_t mask_value =
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ? -9984
                                                                         : 1;
    attn_mask_ = layer::AttentionMask(options_.device(),
                                      options_.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);

    // Final norm
    norm_ = register_module("norm", layer::NpuRMSNorm(context));

    // fc layer for fusion: 3 * target_hidden_size -> hidden_size
    fc_ = register_module("fc", layer::NpuColumnParallelLinear(context));

    // EAGLE-3 has only 1 layer
    decoder_ = register_module("midlayer", QWen3Eagle3DecoderLayer(context));
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids, 0);
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);

    // Handle empty tokens case for dp
    if (dp_size_ > 1 && tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({0}).to(torch::kInt32).to(tokens.device());
    }

    torch::Tensor hidden_states = embed_tokens_(tokens, 0);
    // Get hidden_states_extra from input_params.embedding.input_embedding
    // In EAGLE-3, hidden_states_extra comes from verifier layers
    // (3 layers concatenated)
    torch::Tensor hidden_states_extra = input_params.embedding.input_embedding;
    if (!hidden_states_extra.defined() || hidden_states_extra.size(0) == 0) {
      LOG(WARNING) << "hidden_states_extra use embedding from tokens.";
      hidden_states_extra = hidden_states;
    }

    // Apply fusion if hidden_states_extra dimension doesn't match hidden_states
    // hidden_states_extra shape: [B*L, 3*target_hidden_size] or [B*L,
    // hidden_size]
    if (hidden_states_extra.size(-1) != hidden_states.size(-1)) {
      hidden_states_extra = fc_(hidden_states_extra, 0);
    }

    // Compute positional embeddings
    torch::Tensor target_cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();

    // Generate attention mask
    torch::Tensor attn_mask;
    if (!input_params.meta.batch_forward_type.is_decode()) {
      if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
        int num_sequences = input_params.meta.num_sequences;
        if (num_sequences > 0) {
          std::vector<torch::Tensor> req_mask_vec;
          req_mask_vec.reserve(num_sequences);

          for (int j = 0; j < num_sequences; j++) {
            auto mask = attn_mask_.gen_append_mask(
                input_params.attention.host.q_seq_lens[j],
                input_params.attention.host.kv_seq_lens[j],
                input_params.meta.kv_max_seq_len,
                cos_pos.dtype().toScalarType(),
                cos_pos.device());
            req_mask_vec.emplace_back(mask);
          }
          attn_mask = torch::cat(req_mask_vec, 0);
        }
      } else {
        attn_mask = attn_mask_.get_attn_mask(
            128, cos_pos.dtype().toScalarType(), cos_pos.device());
      }
    }

    // EAGLE-3 has only 1 layer
    aclrtEvent* event{nullptr};
    std::atomic<bool>* event_flag{nullptr};

    if (input_params.parallel.layer_synchronizer != nullptr) {
      event = input_params.parallel.layer_synchronizer->get_event(0);
      event_flag = input_params.parallel.layer_synchronizer->get_event_flag(0);
    }
    if (!input_params.synchronize_layer(0)) {
      return ModelOutput();
    }

    decoder_(hidden_states,
             hidden_states_extra,
             cos_pos,
             sin_pos,
             attn_mask,
             kv_caches[0],
             input_params_new,
             event,
             event_flag);
    auto aux_hidden_states = hidden_states.clone();
    hidden_states = norm_(hidden_states, 0);

    // For draft decode, we capture the hidden state before norm as
    // aux_hidden_states This is used for speculative decoding to pass hidden
    // states to next step
    return ModelOutput(hidden_states,
                       /*residual=*/torch::Tensor(),
                       /*aux_hidden_states=*/aux_hidden_states);
  }

  virtual void load_state_dict(const StateDict& state_dict) {
    // fc: (hidden_size, 3*target_hidden_size) fusion layer
    fc_->load_state_dict(state_dict.get_dict_with_prefix("fc."));

    decoder_->load_state_dict(state_dict.get_dict_with_prefix("midlayer."));

    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  virtual void verify_loaded_weights(const std::string& prefix) const {
    fc_->verify_loaded_weights(prefix + "fc.");
    decoder_->verify_loaded_weights(prefix + "midlayer.");
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  virtual void merge_loaded_weights() {
    fc_->merge_loaded_weights();
    decoder_->merge_loaded_weights();
    norm_->merge_loaded_weights();
  }

  virtual layer::NpuWordEmbedding get_npu_word_embedding() {
    return embed_tokens_;
  }

  virtual void set_npu_word_embedding(
      layer::NpuWordEmbedding& npu_word_embedding) {
    embed_tokens_ = npu_word_embedding;
  }

 protected:
  std::string model_type_;
  torch::TensorOptions options_;

  int32_t dp_rank_ = 0;
  int32_t dp_size_ = 1;
  int32_t dp_local_tp_size_ = 1;

  torch::Tensor cos_sin_;
  layer::NpuPosEmbedding atb_pos_emb_{nullptr};
  layer::AttentionMask attn_mask_;

  layer::NpuWordEmbedding embed_tokens_{nullptr};

  // EAGLE-3 specific modules
  layer::NpuColumnParallelLinear fc_{nullptr};  // fusion layer
  layer::NpuRMSNorm norm_{nullptr};             // final norm

  // Decoder
  QWen3Eagle3DecoderLayer decoder_{nullptr};

  bool layer_forward_interrupted_ = false;
};
TORCH_MODULE(QWen3Eagle3Model);

class QWen3Eagle3ForCausalLMImpl : public torch::nn::Module {
 public:
  QWen3Eagle3ForCausalLMImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    tie_word_embeddings_ = model_args.tie_word_embeddings();

    // register submodules
    model_ =
        register_module("model", QWen3Eagle3Model("qwen3_eagle3", context));

    npu_lm_head_ = register_module("npu_lm_head", layer::NpuLmHead(context));

    // Check if we need to load lm_head from target model
    load_lm_head_from_target_ = false;
    if (!tie_word_embeddings_) {
      int64_t vocab_size = model_args.vocab_size();
      if (vocab_size == 0) {
        load_lm_head_from_target_ = true;
      }
    }
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return model_->get_input_embeddings(input_ids);
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: ModelOutput with hidden_states [num_tokens, hidden_size]
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
  // returns: [num_seqs, hidden_size]
  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return h;
  }

  virtual void load_model(std::unique_ptr<ModelLoader> loader,
                          std::string prefix = "") {
    for (const auto& state_dict : loader->get_state_dicts()) {
      auto sub_dict = state_dict->get_dict_with_prefix(prefix + "model.");
      if (sub_dict.size() == 0) {
        sub_dict = state_dict->get_dict_with_prefix(prefix);
      }
      model_->load_state_dict(sub_dict);

      if (!load_lm_head_from_target_) {
        if (tie_word_embeddings_) {
          npu_lm_head_->load_state_dict(
              state_dict->get_dict_with_prefix(prefix + "embed_tokens."));
        } else {
          npu_lm_head_->load_state_dict(
              state_dict->get_dict_with_prefix("lm_head."));
        }
      }
    }

    // verify
    model_->verify_loaded_weights(prefix);
    if (!load_lm_head_from_target_) {
      if (tie_word_embeddings_) {
        npu_lm_head_->verify_loaded_weights(prefix + "embed_tokens.");
      } else {
        npu_lm_head_->verify_loaded_weights("lm_head.");
      }
    }
    model_->merge_loaded_weights();
    if (!load_lm_head_from_target_) {
      npu_lm_head_->merge_loaded_weights();
    }
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  virtual layer::NpuLmHead get_npu_lm_head() { return npu_lm_head_; }

  virtual void set_npu_lm_head(layer::NpuLmHead& head) {
    if (load_lm_head_from_target_) {
      npu_lm_head_ = head;
    }
  }

  virtual layer::NpuWordEmbedding get_npu_word_embedding() {
    return model_->get_npu_word_embedding();
  }

  virtual void set_npu_word_embedding(
      layer::NpuWordEmbedding& npu_word_embedding) {
    model_->set_npu_word_embedding(npu_word_embedding);
  }

 protected:
  QWen3Eagle3Model model_{nullptr};
  int device_id_ = 0;
  bool tie_word_embeddings_{false};
  bool load_lm_head_from_target_{false};
  layer::NpuLmHead npu_lm_head_{nullptr};
};
TORCH_MODULE(QWen3Eagle3ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen3_eagle3, QWen3Eagle3ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS(qwen3_eagle3, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_eagle3");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(draft_vocab_size, "draft_vocab_size", 0);
  if (args->draft_vocab_size() > 0) {
    args->vocab_size(args->draft_vocab_size());
  }
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 1);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);

  // For qwen3/2.5 model < 7B, tie_word_embeddings = true
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm::npu::model
