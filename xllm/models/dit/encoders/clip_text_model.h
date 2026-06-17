
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

#if defined(USE_NPU)
#include <atb/atb_infer.h>
#endif
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <regex>
#include <unordered_map>

#include "core/framework/dit_model_loader.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#if defined(USE_NPU)
#include "core/layers/npu/npu_siglip_encoder_layer_impl.h"
#endif
#include "models/model_registry.h"
#include "processors/clip_prompt_processor.h"
#include "xllm/core/layers/common/add_matmul.h"
#if defined(USE_NPU)
#include "xllm_atb_layers/core/include/atb_speed/log.h"
#endif

namespace xllm {
// clip_text_model compatible with huggingface weights
// ref to:
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip
torch::Tensor quick_gelu(torch::Tensor x) {
  return x * torch::sigmoid(1.702 * x);
}

// causal_mask (batch_size, 1, seq_len, seq_len)
torch::Tensor _create_4d_causal_attention_mask(torch::IntArrayRef input_shape,
                                               torch::Dtype dtype,
                                               torch::Device device) {
  const int64_t bsz = input_shape[0];
  const int64_t tgt_len = input_shape[1];

  auto options = torch::TensorOptions().dtype(dtype).device(device);
  auto causal_mask = torch::full(
      {tgt_len, tgt_len}, -std::numeric_limits<double>::infinity(), options);
  causal_mask.triu_(1);
  causal_mask = causal_mask.unsqueeze(0).unsqueeze(0);
  causal_mask = causal_mask.expand({bsz, 1, tgt_len, tgt_len});
  return causal_mask;
}

class CLIPTextEmbeddingImpl : public torch::nn::Module {
 public:
  explicit CLIPTextEmbeddingImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    token_embedding_ = register_module(
        "token_embedding",
        torch::nn::Embedding(torch::nn::EmbeddingOptions(
            model_args.vocab_size(), model_args.mm_hidden_size())));
    token_embedding_->weight.set_data(token_embedding_->weight.to(options));
    position_embedding_ = register_parameter(
        "position_embedding",
        torch::randn(
            {model_args.max_position_embeddings(), model_args.mm_hidden_size()},
            options));
    position_ids_ = register_buffer(
        "position_ids",
        torch::arange(0, model_args.max_position_embeddings(), torch::kLong)
            .unsqueeze(0));
  }

  torch::Tensor forward(const torch::Tensor& input_ids) {
    int64_t batch_size = input_ids.size(0);
    int64_t seq_length = input_ids.size(1);
    int64_t max_position_embedding = position_embedding_.size(0);
    CHECK(seq_length <= max_position_embedding);

    torch::Tensor inputs_embeds = token_embedding_->forward(input_ids);
    torch::Tensor position_ids = position_ids_.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(torch::indexing::None, seq_length)});
    torch::Tensor embeddings =
        inputs_embeds + position_embedding_.index({position_ids});
    return embeddings;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict,
                        "token_embedding.weight",
                        token_embedding_->weight,
                        is_token_embedding_loaded);
    weight::load_weight(state_dict,
                        "position_embedding.weight",
                        position_embedding_,
                        is_position_embedding_loaded);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_position_embedding_loaded)
        << "weight is not loaded for " << prefix + "position_embedding.weight";
    CHECK(is_token_embedding_loaded)
        << "weight is not loaded for " << prefix + "token_embedding.weight";
  }

 private:
  bool is_position_embedding_loaded = false;
  bool is_token_embedding_loaded = false;
  torch::Tensor position_ids_;
  torch::nn::Embedding token_embedding_ = nullptr;
  torch::Tensor position_embedding_;
};
TORCH_MODULE(CLIPTextEmbedding);

class CLIPMLPImpl : public torch::nn::Module {
 public:
  explicit CLIPMLPImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    act_ = register_module("act", torch::nn::Functional(quick_gelu));

    fc1_ = register_module("fc1",
                           layer::AddMatmul(model_args.mm_hidden_size(),
                                            model_args.mm_intermediate_size(),
                                            /*with_bias=*/true,
                                            options));
    fc2_ = register_module("fc2",
                           layer::AddMatmul(model_args.mm_intermediate_size(),
                                            model_args.mm_hidden_size(),
                                            /*with_bias=*/true,
                                            options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    return fc2_(act_(fc1_(hidden_states)));
  }

  void load_state_dict(const StateDict& state_dict) {
    fc1_->load_state_dict(state_dict.get_dict_with_prefix("fc1."));
    fc2_->load_state_dict(state_dict.get_dict_with_prefix("fc2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    fc1_->verify_loaded_weights(prefix + "fc1.");
    fc2_->verify_loaded_weights(prefix + "fc2.");
  }

 private:
  torch::nn::Functional act_ = nullptr;
  layer::AddMatmul fc1_ = nullptr;
  layer::AddMatmul fc2_ = nullptr;
};
TORCH_MODULE(CLIPMLP);

// TODO: Optimize CLIPAttention
class CLIPAttentionImpl : public torch::nn::Module {
 public:
  explicit CLIPAttentionImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    CHECK(model_args.mm_hidden_size() % model_args.mm_num_attention_heads() ==
          0);
    head_dim_ = model_args.mm_head_dim();
    embed_dim_ = model_args.mm_hidden_size();
    num_heads_ = model_args.mm_num_attention_heads();
    const int64_t n_local_heads = num_heads_;

    qkv_sizes_ = {n_local_heads * model_args.mm_head_dim(),
                  n_local_heads * model_args.mm_head_dim(),
                  n_local_heads * model_args.mm_head_dim()};

    scale_ = 1.0f / std::sqrt(static_cast<float>(model_args.mm_head_dim()));
    q_proj_ = register_module("q_proj",
                              layer::AddMatmul(model_args.mm_hidden_size(),
                                               num_heads_ * head_dim_,
                                               /*with_bias=*/true,
                                               options));
    k_proj_ = register_module("k_proj",
                              layer::AddMatmul(model_args.mm_hidden_size(),
                                               num_heads_ * head_dim_,
                                               /*with_bias=*/true,
                                               options));
    v_proj_ = register_module("v_proj",
                              layer::AddMatmul(model_args.mm_hidden_size(),
                                               num_heads_ * head_dim_,
                                               /*with_bias=*/true,
                                               options));
    o_proj_ = register_module("o_proj",
                              layer::AddMatmul(model_args.mm_hidden_size(),
                                               model_args.mm_hidden_size(),
                                               /*with_bias=*/true,
                                               options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        torch::Tensor causal_mask) {
    auto bsz = hidden_states.size(0);
    auto tgt_len = hidden_states.size(1);

    auto query_states = q_proj_(hidden_states);
    auto key_states = k_proj_(hidden_states);
    auto value_states = v_proj_(hidden_states);

    // [batch_size, num_heads, seq_len, head_dim]
    query_states = shape(query_states, tgt_len, bsz);
    key_states = shape(key_states, -1, bsz);
    value_states = shape(value_states, -1, bsz);

    auto src_len = key_states.size(1);
    auto attn_weights =
        torch::matmul(query_states, key_states.transpose(-1, -2)) * scale_;

    if (causal_mask.defined()) attn_weights = attn_weights + causal_mask;

    attn_weights = torch::softmax(attn_weights, -1);
    auto attn_output = torch::matmul(attn_weights, value_states);

    attn_output = attn_output.transpose(1, 2).contiguous().view(
        {bsz, tgt_len, embed_dim_});

    return o_proj_(attn_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    q_proj_->load_state_dict(state_dict.get_dict_with_prefix("q_proj."));
    k_proj_->load_state_dict(state_dict.get_dict_with_prefix("k_proj."));
    v_proj_->load_state_dict(state_dict.get_dict_with_prefix("v_proj."));
    o_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    q_proj_->verify_loaded_weights(prefix + "q_proj.");
    k_proj_->verify_loaded_weights(prefix + "k_proj.");
    v_proj_->verify_loaded_weights(prefix + "v_proj.");
    o_proj_->verify_loaded_weights(prefix + "out_proj.");
  }

 private:
  torch::Tensor shape(torch::Tensor tensor, int64_t seq_len, int64_t bsz) {
    return tensor.view({bsz, seq_len, num_heads_, head_dim_})
        .transpose(1, 2)
        .contiguous();
  }

 private:
  int64_t embed_dim_;
  int64_t num_heads_;
  int64_t head_dim_;
  float scale_;
  std::vector<int64_t> qkv_sizes_;

  layer::AddMatmul o_proj_ = nullptr;
  layer::AddMatmul q_proj_ = nullptr;
  layer::AddMatmul k_proj_ = nullptr;
  layer::AddMatmul v_proj_ = nullptr;
};
TORCH_MODULE(CLIPAttention);

class CLIPEncoderLayerImpl : public torch::nn::Module {
 public:
  explicit CLIPEncoderLayerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    self_attn_ = register_module("self_attn", CLIPAttention(context));
    layer_norm1_ = register_module(
        "layer_norm1",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.mm_layer_norm_eps())));
    layer_norm2_ = register_module(
        "layer_norm2",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.mm_layer_norm_eps())));
    layer_norm1_->weight.set_data(layer_norm1_->weight.to(options));
    layer_norm1_->bias.set_data(layer_norm1_->bias.to(options));
    layer_norm2_->weight.set_data(layer_norm2_->weight.to(options));
    layer_norm2_->bias.set_data(layer_norm2_->bias.to(options));
    mlp_ = register_module("mlp", CLIPMLP(context));
  }

  // TODO: self_attn, attention_mask
  torch::Tensor forward(const torch::Tensor& hidden_states,
                        torch::Tensor causal_mask) {
    auto residual = hidden_states;
    const auto& layer_norm1 = layer_norm1_(hidden_states);
    auto h = self_attn_(layer_norm1, causal_mask) + residual;
    residual = h;
    h = layer_norm2_(h);
    h = mlp_(h);
    h += residual;
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    self_attn_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
    weight::load_weight(state_dict,
                        "layer_norm1.weight",
                        layer_norm1_->weight,
                        layer_norm1_weight_loaded_);
    weight::load_weight(state_dict,
                        "layer_norm1.bias",
                        layer_norm1_->bias,
                        layer_norm1_bias_loaded_);
    weight::load_weight(state_dict,
                        "layer_norm2.weight",
                        layer_norm2_->weight,
                        layer_norm2_weight_loaded_);
    weight::load_weight(state_dict,
                        "layer_norm2.bias",
                        layer_norm2_->bias,
                        layer_norm2_bias_loaded_);
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    self_attn_->verify_loaded_weights(prefix + "self_attn.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    CHECK(layer_norm1_weight_loaded_)
        << "weight is not loaded for " << prefix + "layer_norm1.weight";
    CHECK(layer_norm1_bias_loaded_)
        << "weight is not loaded for " << prefix + "layer_norm1.bias";
    CHECK(layer_norm2_weight_loaded_)
        << "weight is not loaded for " << prefix + "layer_norm2.weight";
    CHECK(layer_norm2_bias_loaded_)
        << "weight is not loaded for " << prefix + "layer_norm2.bias";
  }

 private:
  bool layer_norm1_weight_loaded_ = false;
  bool layer_norm1_bias_loaded_ = false;
  bool layer_norm2_weight_loaded_ = false;
  bool layer_norm2_bias_loaded_ = false;

  torch::nn::LayerNorm layer_norm1_ = nullptr;
  torch::nn::LayerNorm layer_norm2_ = nullptr;
  CLIPAttention self_attn_ = nullptr;
  CLIPMLP mlp_ = nullptr;
};
TORCH_MODULE(CLIPEncoderLayer);

class CLIPEncoderImpl : public torch::nn::Module {
 public:
  explicit CLIPEncoderImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.mm_num_hidden_layers());
    for (int32_t i = 0; i < model_args.mm_num_hidden_layers(); i++) {
      auto block = CLIPEncoderLayer(context);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  // Output hidden states for last intermediate layers
  torch::Tensor forward(const torch::Tensor& embeddings,
                        torch::Tensor causal_mask) {
    bool output_hidden_states = false;
    bool output_attentions = false;
    c10::optional<torch::Tensor> attention_mask = c10::nullopt;
    c10::optional<torch::Tensor> head_mask = c10::nullopt;
    std::vector<torch::Tensor> all_hidden_states;
    std::vector<torch::Tensor> all_attentions;
    std::vector<torch::Tensor> encoder_states;

    auto hidden_states = embeddings;
    for (size_t i = 0; i < layers_.size(); ++i) {
      encoder_states.emplace_back(hidden_states);
      auto& layer = layers_[i];
      hidden_states = layer(hidden_states, causal_mask);
    }
    if (output_hidden_states) encoder_states.emplace_back(hidden_states);

    std::vector<torch::Tensor> outputs = {hidden_states};
    if (output_hidden_states) {
      // todo
    }
    if (output_attentions) {
      // todo
    }
    return outputs[0];
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
  }

 private:
  torch::nn::ModuleList blocks_ = nullptr;
  std::vector<CLIPEncoderLayer> layers_ = {};
};
TORCH_MODULE(CLIPEncoder);

class CLIPTextTransformerImpl : public torch::nn::Module {
 public:
  explicit CLIPTextTransformerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    embeddings_ = register_module("embeddings", CLIPTextEmbedding(context));
    final_layer_norm_ = register_module(
        "final_layer_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.mm_layer_norm_eps())));
    final_layer_norm_->weight.set_data(final_layer_norm_->weight.to(options));
    final_layer_norm_->bias.set_data(final_layer_norm_->bias.to(options));
    encoder_ = register_module("encoder", CLIPEncoder(context));
    eos_token_id = model_args.eos_token_id();
  }

  torch::Tensor forward(const torch::Tensor& input_ids) {
    if (!input_ids.defined()) {
      LOG(FATAL) << "input_ids is undefined.";
    }
    auto input_shape = input_ids.sizes();
    auto reshaped_input_ids = input_ids.view({-1, input_shape.back()});
    auto hidden_states = embeddings_->forward(reshaped_input_ids);
    auto causal_mask = _create_4d_causal_attention_mask(
        {input_shape[0], input_shape[1]},
        torch::typeMetaToScalarType(hidden_states.dtype()),
        hidden_states.device());
    auto encoder_output = encoder_->forward(hidden_states, causal_mask);
    auto last_hidden_state = final_layer_norm_->forward(encoder_output);
    return last_hidden_state;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embeddings_->load_state_dict(
        state_dict.get_dict_with_prefix("embeddings."));
    encoder_->load_state_dict(state_dict.get_dict_with_prefix("encoder."));
    weight::load_weight(state_dict,
                        "final_layer_norm.weight",
                        final_layer_norm_->weight,
                        final_layer_norm_weight_loaded_);
    weight::load_weight(state_dict,
                        "final_layer_norm.bias",
                        final_layer_norm_->bias,
                        final_layer_norm_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embeddings_->verify_loaded_weights(prefix + "embeddings.");
    encoder_->verify_loaded_weights(prefix + "encoder.");
    CHECK(final_layer_norm_weight_loaded_)
        << "weight is not loaded for " << prefix + "final_layer_norm.weight";
    CHECK(final_layer_norm_bias_loaded_)
        << "weight is not loaded for " << prefix + "final_layer_norm.bias";
  }

 private:
  int64_t eos_token_id;
  bool final_layer_norm_weight_loaded_ = false;
  bool final_layer_norm_bias_loaded_ = false;
  CLIPTextEmbedding embeddings_ = nullptr;
  CLIPEncoder encoder_ = nullptr;
  torch::nn::LayerNorm final_layer_norm_ = nullptr;
};
TORCH_MODULE(CLIPTextTransformer);

class CLIPTextModelImpl : public torch::nn::Module {
 public:
  explicit CLIPTextModelImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    eos_token_id = model_args.eos_token_id();
    transformer_ = register_module("transformer", CLIPTextTransformer(context));
  }

  torch::Tensor forward(const torch::Tensor& input_ids) {
    auto last_hidden_states = transformer_->forward(input_ids);
    int64_t batch_size = last_hidden_states.size(0);
    auto device = last_hidden_states.device();
    torch::Tensor batch_indices = torch::arange(batch_size, device);
    torch::Tensor end_pos;
    if (eos_token_id == 2) {
      auto argmax_result = input_ids.to(device).max(1);
      end_pos = std::get<1>(argmax_result);
    } else {
      torch::Tensor eos_mask =
          (input_ids == eos_token_id).to(device, torch::kInt);
      auto argmax_result = eos_mask.max(1);
      end_pos = std::get<1>(argmax_result);
    }
    torch::Tensor pooled_output =
        last_hidden_states.index({batch_indices, end_pos});
    return pooled_output;
  }

  void load_state_dict(const StateDict& state_dict) {
    transformer_->load_state_dict(
        state_dict.get_dict_with_prefix("text_model."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    transformer_->verify_loaded_weights(prefix + ".");
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    LOG(INFO) << "Loading CLIPTextModel from ModelLoader...";
    for (const auto& state_dict : loader->get_state_dicts()) {
      transformer_->load_state_dict(
          state_dict->get_dict_with_prefix("text_model."));
    }

    // verify
    transformer_->verify_loaded_weights("text_model.");
    LOG(INFO) << "clip_text_model loaded successfully.";
  }

 private:
  int64_t eos_token_id;
  CLIPTextTransformer transformer_ = nullptr;
};
TORCH_MODULE(CLIPTextModel);

REGISTER_MODEL_ARGS(CLIPTextModel, [&] {
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(vocab_size, "vocab_size", 49408);
  LOAD_ARG_OR(mm_hidden_size, "hidden_size", 768);
  LOAD_ARG_OR(mm_intermediate_size, "intermediate_size", 3072);
  LOAD_ARG_OR(mm_projection_dim, "projection_dim", 768);
  LOAD_ARG_OR(mm_num_hidden_layers, "num_hidden_layers", 12);
  LOAD_ARG_OR(mm_num_attention_heads, "num_attention_heads", 12);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 77);
  LOAD_ARG_OR(mm_hidden_act, "hidden_act", "quick_gelu");
  LOAD_ARG_OR(mm_layer_norm_eps, "layer_norm_eps", 1e-5f);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
  LOAD_ARG_OR(mm_head_dim, "head_dim", 64);
});
}  // namespace xllm
