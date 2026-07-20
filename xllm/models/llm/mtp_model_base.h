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

#include <algorithm>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "core/framework/model/mtp_topk_state.h"
#include "core/framework/model/mtp_utils.h"
#include "core/framework/state_dict/utils.h"
#include "core/util/utils.h"
#include "llm_model_base.h"

namespace xllm {

enum class MtpProjectionType { CONCAT_EH_PROJ, ADD_EH_PROJ };

inline bool is_deepseek_v4_mtp_model(const ModelArgs& model_args) {
  return util::is_target_mtp_model_type(model_args.model_type(), "deepseek_v4");
}

inline MtpProjectionType get_mtp_projection_type(const ModelArgs& model_args) {
  if (is_deepseek_v4_mtp_model(model_args)) {
    return MtpProjectionType::ADD_EH_PROJ;
  }
  return MtpProjectionType::CONCAT_EH_PROJ;
}

template <typename DecoderLayerType>
class MtpDecoderLayerImplBase : public torch::nn::Module {
 public:
  MtpDecoderLayerImplBase(const ModelContext& context,
                          const int32_t layer_index)
      : model_args_(context.get_model_args()) {
    auto options = context.get_tensor_options();

    projection_type_ = get_mtp_projection_type(model_args_);

    // register submodules
    enorm_ = register_module("enorm", layer::RMSNorm(context));
    hnorm_ = register_module("hnorm", layer::RMSNorm(context));
    if (projection_type_ == MtpProjectionType::ADD_EH_PROJ) {
      e_proj_ =
          register_module("e_proj",
                          layer::ReplicatedLinear(model_args_.hidden_size(),
                                                  model_args_.hidden_size(),
                                                  /*bias=*/false,
                                                  /*QuantArgs=*/QuantArgs(),
                                                  options));
      h_proj_ =
          register_module("h_proj",
                          layer::ReplicatedLinear(model_args_.hidden_size(),
                                                  model_args_.hidden_size(),
                                                  /*bias=*/false,
                                                  /*QuantArgs=*/QuantArgs(),
                                                  options));
    } else {
      // no quantization for eh_proj
      eh_proj_ =
          register_module("eh_proj",
                          layer::ReplicatedLinear(model_args_.hidden_size() * 2,
                                                  model_args_.hidden_size(),
                                                  /*bias=*/false,
                                                  /*QuantArgs=*/QuantArgs(),
                                                  options));
    }
    const int32_t decoder_layer_index =
        is_deepseek_v4_mtp_model(model_args_)
            ? std::min<int32_t>(layer_index, model_args_.n_layers() - 1)
            : layer_index;
    mtp_block_ = register_module(
        "mtp_block", DecoderLayerType(context, decoder_layer_index));

    if (is_deepseek_v4_mtp_model(model_args_)) {
      const int64_t hc_mult = model_args_.hc_mult();
      const int64_t hc_dim = hc_mult * model_args_.hidden_size();
      auto hc_options = options.dtype(torch::kFloat32);
      hc_head_fn_ = register_parameter(
          "hc_head_fn", torch::empty({hc_mult, hc_dim}, hc_options), false);
      hc_head_base_ = register_parameter(
          "hc_head_base", torch::empty({hc_mult}, hc_options), false);
      hc_head_scale_ = register_parameter(
          "hc_head_scale", torch::empty({1}, hc_options), false);
    }
  }

  torch::Tensor forward(
      torch::Tensor embed,
      std::optional<torch::Tensor>& residual,
      torch::Tensor positions,
      const layer::AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      ModelInputParams& input_params,
      const std::optional<torch::Tensor>& input_ids = std::nullopt,
      torch::Tensor* aux_hidden_states = nullptr) {
    auto decoder_forward = [](DecoderLayerType& decoder,
                              torch::Tensor& hidden_states,
                              std::optional<torch::Tensor>& layer_residual,
                              torch::Tensor& layer_positions,
                              const layer::AttentionMetadata& metadata,
                              KVCache& layer_kv_cache,
                              ModelInputParams& params,
                              const std::optional<torch::Tensor>& ids) {
      if constexpr (std::is_invocable_v<DecoderLayerType,
                                        torch::Tensor&,
                                        std::optional<torch::Tensor>&,
                                        torch::Tensor&,
                                        const layer::AttentionMetadata&,
                                        KVCache&,
                                        ModelInputParams&,
                                        const std::optional<torch::Tensor>&>) {
        return decoder(hidden_states,
                       layer_residual,
                       layer_positions,
                       metadata,
                       layer_kv_cache,
                       params,
                       ids);
      } else {
        return decoder(hidden_states,
                       layer_residual,
                       layer_positions,
                       metadata,
                       layer_kv_cache,
                       params);
      }
    };
    return forward_with_decoder(std::move(embed),
                                residual,
                                std::move(positions),
                                attn_metadata,
                                kv_cache,
                                input_params,
                                input_ids,
                                decoder_forward,
                                aux_hidden_states);
  }

 protected:
  template <typename DecoderForward>
  torch::Tensor forward_with_decoder(
      torch::Tensor embed,
      std::optional<torch::Tensor>& residual,
      torch::Tensor positions,
      const layer::AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      ModelInputParams& input_params,
      const std::optional<torch::Tensor>& input_ids,
      DecoderForward&& decoder_forward,
      torch::Tensor* aux_hidden_states = nullptr) {
    // Layer norm on token inputs
    auto enorm_out = std::get<0>(enorm_(embed));

    torch::Tensor embedding_data = input_params.embedding.input_embedding;
    // for dummy data parallel run, we set a empty embedding
    if (attn_metadata.is_dummy) {
      const int64_t embed_cols = mtp_hidden_state_width(model_args_);
      embedding_data =
          torch::zeros({embed.size(0), embed_cols}, embed.options());
    }
    CHECK(embedding_data.defined())
        << "embedding is not defined in input_params.embedding.input_embedding";
    torch::Tensor previous_hidden_states = embedding_data;
    torch::Tensor hidden_states;
    if (is_deepseek_v4_mtp_model(model_args_)) {
      const int64_t hidden = model_args_.hidden_size();
      const int64_t hc_mult = model_args_.hc_mult();
      const int64_t aux_hidden = hc_mult * hidden;
      const int64_t input_hidden = previous_hidden_states.size(-1);
      CHECK(input_hidden == hidden || input_hidden == aux_hidden)
          << "DeepSeek-V4 MTP input hidden size must be hidden_size (" << hidden
          << ") or hc_mult * hidden_size (" << aux_hidden << "), but got "
          << input_hidden;
      if (input_hidden == aux_hidden) {
        // Target feeds the draft the pre-hc_head 3D hidden flattened to
        // [N, hc_mult*hidden].
        torch::Tensor prev = previous_hidden_states.reshape({-1, hidden});
        prev = std::get<0>(hnorm_(prev));          // [N*hc_mult, hidden]
        torch::Tensor e_out = e_proj_(enorm_out);  // [N, hidden]
        torch::Tensor h_out = h_proj_(prev);       // [N*hc_mult, hidden]
        hidden_states =
            e_out.unsqueeze(1) + h_out.reshape({-1, hc_mult, hidden});
      } else {
        // Preserve compatibility with callers that provide the post-hc_head
        // 2D hidden [N, hidden].
        previous_hidden_states = std::get<0>(hnorm_(previous_hidden_states));
        hidden_states = e_proj_(enorm_out) + h_proj_(previous_hidden_states);
        hidden_states = hidden_states.unsqueeze(1).repeat({1, hc_mult, 1});
      }
    } else {
      previous_hidden_states = std::get<0>(hnorm_(previous_hidden_states));
      if (projection_type_ == MtpProjectionType::ADD_EH_PROJ) {
        hidden_states = e_proj_(enorm_out) + h_proj_(previous_hidden_states);
      } else {
        // Concatenate along last dimension and project
        hidden_states =
            eh_proj_(torch::cat({enorm_out, previous_hidden_states}, -1));
      }
    }

    // Pass through the backend-selected decoder invocation.
    hidden_states = decoder_forward(mtp_block_,
                                    hidden_states,
                                    residual,
                                    positions,
                                    attn_metadata,
                                    kv_cache,
                                    input_params,
                                    input_ids);

    if (is_deepseek_v4_mtp_model(model_args_)) {
      if (aux_hidden_states != nullptr) {
        *aux_hidden_states = hidden_states;
      }
      auto x_float = hidden_states.to(torch::kFloat32);
      auto x_flatten = x_float.flatten(-2, -1);
      auto rsqrt = torch::rsqrt(x_flatten.pow(2).mean(-1, true) +
                                model_args_.rms_norm_eps());
      auto mixes = torch::matmul(x_flatten, hc_head_fn_.transpose(0, 1));
      mixes = mixes * rsqrt;
      auto pre = torch::sigmoid(mixes * hc_head_scale_ + hc_head_base_) +
                 static_cast<double>(model_args_.hc_eps());
      auto y = (pre.unsqueeze(-1) * x_float).sum(-2);
      return y.to(hidden_states.dtype());
    }

    return hidden_states;
  }

 public:
  void load_state_dict(const StateDict& state_dict) {
    enorm_->load_state_dict(state_dict.get_dict_with_prefix("enorm."));
    hnorm_->load_state_dict(state_dict.get_dict_with_prefix("hnorm."));
    if (projection_type_ == MtpProjectionType::ADD_EH_PROJ) {
      e_proj_->load_state_dict(state_dict.get_dict_with_prefix("e_proj."));
      h_proj_->load_state_dict(state_dict.get_dict_with_prefix("h_proj."));
    } else {
      eh_proj_->load_state_dict(state_dict.get_dict_with_prefix("eh_proj."));
    }
    mtp_block_->load_state_dict(state_dict);
    if (is_deepseek_v4_mtp_model(model_args_)) {
      LOAD_WEIGHT(hc_head_fn);
      LOAD_WEIGHT(hc_head_base);
      LOAD_WEIGHT(hc_head_scale);
    }
  }

  virtual void verify_loaded_weights() const {
    // Not all decoder layer types provide verify_loaded_weights();
    // only call it when the method exists.
    if constexpr (requires { mtp_block_->verify_loaded_weights(); }) {
      mtp_block_->verify_loaded_weights();
    }
  }

  // Forward cache mapping to the underlying decoder layer block.
  // Only applicable for decoder layers that support set_cache_mapping
  // (e.g. MLU DeepseekV4DecoderLayer with DSACacheMapping).
  template <typename MappingType>
  void set_cache_mapping(const MappingType& mapping) {
    if constexpr (requires { mtp_block_->set_cache_mapping(mapping); }) {
      mtp_block_->set_cache_mapping(mapping);
    }
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }

  virtual void update_expert_weight(int32_t layer_id) { return; }

 private:
  layer::RMSNorm enorm_{nullptr};
  layer::RMSNorm hnorm_{nullptr};
  layer::ReplicatedLinear eh_proj_{nullptr};
  layer::ReplicatedLinear e_proj_{nullptr};
  layer::ReplicatedLinear h_proj_{nullptr};
  DecoderLayerType mtp_block_{nullptr};

  DEFINE_WEIGHT(hc_head_fn);
  DEFINE_WEIGHT(hc_head_base);
  DEFINE_WEIGHT(hc_head_scale);

  MtpProjectionType projection_type_ = MtpProjectionType::CONCAT_EH_PROJ;
  ModelArgs model_args_;
};

template <typename DecoderLayerType>
class DefaultMtpLayerForwardAdapter final {
 public:
  DefaultMtpLayerForwardAdapter(const MtpTopkStatePtr& input_state,
                                size_t /*num_layers*/,
                                const torch::Device& /*device*/) {
    CHECK(input_state == nullptr)
        << "MTP model received top-k state without a backend adapter.";
  }

  torch::Tensor forward_layer(DecoderLayerType& decoder_layer,
                              size_t /*layer_id*/,
                              torch::Tensor& hidden_states,
                              std::optional<torch::Tensor>& residual,
                              torch::Tensor& positions,
                              const layer::AttentionMetadata& attn_metadata,
                              KVCache& kv_cache,
                              ModelInputParams& input_params,
                              const std::optional<torch::Tensor>& input_ids) {
    return decoder_layer(hidden_states,
                         residual,
                         positions,
                         attn_metadata,
                         kv_cache,
                         input_params,
                         input_ids);
  }

  MtpTopkStatePtr take_output() { return nullptr; }
};

template <typename DecoderLayerType,
          typename LayerForwardAdapter =
              DefaultMtpLayerForwardAdapter<DecoderLayerType>>
class MtpModelImplBase : public torch::nn::Module {
 public:
  MtpModelImplBase(const ModelContext& context)
      : model_args_(context.get_model_args()),
        device_(context.get_tensor_options().device()) {
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();

    int32_t mtp_num_layers = model_args_.num_nextn_predict_layers();
    if (mtp_num_layers <= 0) {
      mtp_num_layers = 1;
    }

    // get mtp start and end layer index
    mtp_start_layer_idx_ = model_args_.n_layers();
    mtp_end_layer_idx_ = mtp_start_layer_idx_ + mtp_num_layers;
    mtp_layers_.reserve(mtp_num_layers);

    // create mtp layers
    for (int32_t i = mtp_start_layer_idx_; i < mtp_end_layer_idx_; ++i) {
      auto mtp_layer = DecoderLayerType(context, i);
      mtp_layers_.push_back(mtp_layer);
    }
    embed_tokens_ =
        register_module("embed_tokens",
                        layer::WordEmbedding(model_args_.vocab_size(),
                                             model_args_.hidden_size(),
                                             context.get_parallel_args(),
                                             options));
    norm_ = register_module("norm", layer::RMSNorm(context));

    // get dp size and rank
    dp_size_ = parallel_args.dp_size();
    std::vector<int64_t> indices;
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;
    rank_ = parallel_args.rank();
    for (size_t i = 0; i < parallel_args.world_size(); i += dp_local_tp_size_) {
      indices.push_back(i);
    }
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids);
  }

  // Provide batched signature to satisfy callers that pass vectors
  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    // for dp, if tokens is empty, set tokens to 1 and positions to 0
    ModelInputParams modified_input_params = input_params;
    if (dp_size_ > 1) {
      if (tokens.sizes() == 0) {
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
    // Mask out embeddings where positions == 0 (for MTP not needed at pos 0)
    auto mask = (positions == 0);  // bool tensor
    if (mask.any().item<bool>()) {
      // set masked rows to zero
      hidden_states.index_put_({mask},
                               torch::zeros_like(hidden_states.index({mask})));
    }

    std::optional<torch::Tensor> residual;
    LayerForwardAdapter forward_adapter(
        input_params.mtp_topk_state, mtp_layers_.size(), device_);
    for (size_t i = 0; i < mtp_layers_.size(); i++) {
#if defined(USE_CUDA) || defined(USE_MUSA)
      attn_metadata.plan_info->layer_id = i;
#endif
      auto& layer = mtp_layers_[i];
      hidden_states = forward_adapter.forward_layer(layer,
                                                    i,
                                                    hidden_states,
                                                    residual,
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
    auto [h_out, r_out] = norm_(hidden_states, residual);
    ModelOutput output(h_out, r_out);
    output.mtp_topk_state = forward_adapter.take_output();
    return output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each layer's load_state_dict function
    for (int32_t i = 0; i < mtp_layers_.size(); i++) {
      int32_t layer_index = mtp_start_layer_idx_ + i;
      mtp_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "layers." + std::to_string(layer_index) + "."));
      // there is only one shared_head.norm for deepseek models, so we load it
      // here
      if (i == mtp_layers_.size() - 1) {
        norm_->load_state_dict(state_dict.get_dict_with_prefix(
            "layers." + std::to_string(layer_index) + ".shared_head.norm."));
      }
    }
  }

  void verify_loaded_weights() const {
    for (const auto& layer : mtp_layers_) {
      layer->verify_loaded_weights();
    }
  }

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 private:
  ModelArgs model_args_;
  std::vector<DecoderLayerType> mtp_layers_;
  int32_t mtp_start_layer_idx_;
  int32_t mtp_end_layer_idx_;
  int32_t dp_rank_;
  int32_t rank_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  torch::Device device_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::RMSNorm norm_{nullptr};
};
}  // namespace xllm
