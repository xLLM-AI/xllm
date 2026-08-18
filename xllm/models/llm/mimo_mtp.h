/* Copyright 2026 The xLLM Authors.

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

#include <string>
#include <vector>

#include "core/layers/qwen2_decoder_layer.h"
#include "llm_model_base.h"

// MiMo MTP (Multi-Token Prediction) model implementation
// Based on Qwen2 architecture with custom MTP layers
// Reference:
// https://github.com/XiaomiMiMo/vllm/commit/3a353c0508437a2341ae67252e62382ad012d165

namespace xllm {

class MiMoMtpDecoderLayerImpl final : public torch::nn::Module {
 public:
  MiMoMtpDecoderLayerImpl(const ModelContext& context,
                          const int32_t layer_index)
      : model_args_(context.get_model_args()) {
    auto options = context.get_tensor_options();

    token_layernorm_ =
        register_module("token_layernorm", layer::RMSNorm(context));
    hidden_layernorm_ =
        register_module("hidden_layernorm", layer::RMSNorm(context));
    input_proj_ =
        register_module("input_proj",
                        layer::ReplicatedLinear(model_args_.hidden_size() * 2,
                                                model_args_.hidden_size(),
                                                /*bias=*/false,
                                                /*QuantArgs=*/QuantArgs(),
                                                options));
    mtp_block_ = register_module(
        "mtp_block", layer::Qwen2DecoderLayer(context, layer_index));
  }

  torch::Tensor forward(torch::Tensor embed,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor positions,
                        const layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params) {
    // Mask out embeddings at position 0 (not needed by MTP).
    // masked_fill avoids a D2H sync that .item<bool>() would introduce on
    // every forward pass; the unsqueeze broadcasts the [seq] mask to [seq,
    // hidden].
    embed =
        embed.masked_fill((positions == 0).unsqueeze(-1).expand_as(embed), 0);

    auto token_out = std::get<0>(token_layernorm_(embed));

    torch::Tensor embedding_data = input_params.embedding.input_embedding;
    if (attn_metadata.is_dummy) {
      embedding_data = torch::zeros({embed.size(0), model_args_.hidden_size()},
                                    embed.options());
    }
    CHECK(embedding_data.defined())
        << "embedding is not defined in input_params.embedding.input_embedding";
    auto hidden_out = std::get<0>(hidden_layernorm_(embedding_data));

    // Concatenate [previous_hidden_states, token_embedding] and project
    auto concat_emb = torch::cat({hidden_out, token_out}, -1);
    auto hidden_states = input_proj_(concat_emb);

    // Pass through mtp block
    hidden_states = mtp_block_(hidden_states,
                               residual,
                               positions,
                               attn_metadata,
                               kv_cache,
                               input_params);

    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    token_layernorm_->load_state_dict(
        state_dict.get_dict_with_prefix("token_layernorm."));
    hidden_layernorm_->load_state_dict(
        state_dict.get_dict_with_prefix("hidden_layernorm."));
    input_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("input_proj."));
    // final_layernorm is loaded into MiMoMtpModelImpl::norm_ — not here
    mtp_block_->load_state_dict(state_dict);
  }

  void verify_loaded_weights() const {
    CHECK(token_layernorm_->weight().defined())
        << "MiMo MTP decoder: token_layernorm weight not loaded";
    CHECK(hidden_layernorm_->weight().defined())
        << "MiMo MTP decoder: hidden_layernorm weight not loaded";
    CHECK(input_proj_->weight().defined())
        << "MiMo MTP decoder: input_proj weight not loaded";
  }

 private:
  layer::RMSNorm token_layernorm_{nullptr};
  layer::RMSNorm hidden_layernorm_{nullptr};
  layer::ReplicatedLinear input_proj_{nullptr};
  layer::Qwen2DecoderLayer mtp_block_{nullptr};

  ModelArgs model_args_;
};
TORCH_MODULE(MiMoMtpDecoderLayer);

class MiMoMtpModelImpl final : public torch::nn::Module {
 public:
  explicit MiMoMtpModelImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        device_(context.get_tensor_options().device()) {
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();

    mtp_layers_.emplace_back(register_module(
        "mtp_layers_0", MiMoMtpDecoderLayer(context, /*layer_index=*/0)));

    embed_tokens_ =
        register_module("embed_tokens",
                        layer::WordEmbedding(model_args_.vocab_size(),
                                             model_args_.hidden_size(),
                                             parallel_args,
                                             options));
    norm_ = register_module("norm", layer::RMSNorm(context));

    dp_size_ = parallel_args.dp_size();
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;
    rank_ = parallel_args.rank();
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids);
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    ModelInputParams modified_input_params = input_params;
    if (dp_size_ > 1) {
      if (tokens.numel() == 0) {
        tokens = torch::tensor({1}).to(torch::kInt32).to(device_);
        positions = torch::tensor({1}).to(torch::kInt32).to(device_);
      }
      auto& dp_token_nums = modified_input_params.parallel.dp_global_token_nums;
      std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    }
    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(modified_input_params,
                                                     model_args_.enable_mla(),
                                                     /*attn_mask=*/std::nullopt,
                                                     /*device=*/device_));
    }
    auto& attn_metadata = *(modified_input_params.attn_metadata);
    torch::Tensor hidden_states = embed_tokens_(tokens);

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < mtp_layers_.size(); i++) {
      if (!modified_input_params.synchronize_layer(static_cast<uint32_t>(i))) {
        return ModelOutput();
      }
#if defined(USE_CUDA)
      attn_metadata.plan_info->layer_id = static_cast<int32_t>(i);
#endif
      auto& layer = mtp_layers_[i];
      hidden_states = layer(hidden_states,
                            residual,
                            positions,
                            attn_metadata,
                            kv_caches[i],
                            modified_input_params);
      if (!modified_input_params.record_layer(static_cast<uint32_t>(i),
                                              hidden_states.device())) {
        return ModelOutput();
      }
    }
    auto [h_out, r_out] = norm_(hidden_states, residual);
    return ModelOutput(h_out, r_out);
  }

  void load_state_dict(const StateDict& state_dict) {
    // HuggingFace checkpoint stores MTP weights at "model.mtp_layers.0.*"
    // (0-indexed). export_mtp.py keeps these keys unchanged, so the exported
    // checkpoint retains the "mtp_layers.0.*" prefix used here.
    auto mtp_dict = state_dict.get_dict_with_prefix("mtp_layers.0.");
    CHECK_EQ(mtp_layers_.size(), 1u)
        << "MiMo MTP currently supports exactly 1 MTP layer";
    mtp_layers_[0]->load_state_dict(mtp_dict);
    // Load final norm from the MTP layer's final_layernorm
    norm_->load_state_dict(mtp_dict.get_dict_with_prefix("final_layernorm."));
  }

  void verify_loaded_weights() const {
    for (const auto& layer : mtp_layers_) {
      layer->verify_loaded_weights();
    }
    CHECK(norm_->weight().defined())
        << "MiMo MTP model: final_layernorm weight not loaded";
  }

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 private:
  ModelArgs model_args_;
  std::vector<MiMoMtpDecoderLayer> mtp_layers_;
  int32_t dp_rank_;
  int32_t rank_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  torch::Device device_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::RMSNorm norm_{nullptr};
};
TORCH_MODULE(MiMoMtpModel);

class MiMoMtpForCausalLMImpl final
    : public LlmForCausalLMImplBase<MiMoMtpModel> {
 public:
  explicit MiMoMtpForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<MiMoMtpModel>(context) {}

  void load_model(
      std::unique_ptr<ModelLoader> loader,
      std::string prefix = "model." /*llm model weight prefix*/) override {
    // Only load MTP model weights, skip lm_head (shared with target model)
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
    }
    model_->verify_loaded_weights();
  }
};
TORCH_MODULE(MiMoMtpForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(mimo_mtp, MiMoMtpForCausalLM);

// register the model args
// example config: /root/models2/MiMo-7B-Base/config.json
REGISTER_MODEL_ARGS(mimo_mtp, [&] {
  LOAD_ARG_OR(model_type, "model_type", "mimo_mtp");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 151680);
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 36);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(attention_bias, "attention_bias", true);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 11008);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 640000.0f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 32768);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 32);

  LOAD_ARG_OR(num_nextn_predict_layers, "num_nextn_predict_layers", 1);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm
