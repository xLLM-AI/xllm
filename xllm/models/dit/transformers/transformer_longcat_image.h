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

#include "models/dit/transformers/transformer_flux.h"

namespace xllm {

// LongCat-Image Transformer Model
// Ref:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_longcat_image.py

class LongCatImageTimestepEmbeddingsImpl : public torch::nn::Module {
 public:
  explicit LongCatImageTimestepEmbeddingsImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    time_proj_ = Timesteps(context);
    timestep_embedder_ = TimestepEmbedding(context);
  }

  torch::Tensor forward(const torch::Tensor& timestep) {
    auto timesteps_proj = time_proj_->forward(timestep);
    auto timesteps_emb = timestep_embedder_->forward(timesteps_proj);
    return timesteps_emb;
  }

  void load_state_dict(const StateDict& state_dict) {
    timestep_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("timestep_embedder."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    timestep_embedder_->verify_loaded_weights(prefix + "timestep_embedder.");
  }

 private:
  Timesteps time_proj_{nullptr};
  TimestepEmbedding timestep_embedder_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(LongCatImageTimestepEmbeddings);

class LongCatImageSingleTransformerBlockImpl : public torch::nn::Module {
 public:
  explicit LongCatImageSingleTransformerBlockImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto num_attention_heads = model_args.n_heads();
    auto attention_head_dim = model_args.head_dim();
    auto dim = num_attention_heads * attention_head_dim;
    mlp_hidden_dim_ = dim * 4;  // mlp_ratio = 4.0

    norm_ = register_module("norm", AdaLayerNormZeroSingle(context));
    proj_mlp_ = register_module(
        "proj_mlp",
        layer::AddMatmul(dim, mlp_hidden_dim_, /*with_bias=*/true, options_));
    act_mlp_ =
        register_module("act_mlp",
                        torch::nn::Functional(
                            std::function<torch::Tensor(const torch::Tensor&)>(
                                [](const torch::Tensor& x) {
                                  return torch::gelu(x, "tanh");
                                })));
    proj_out_ = register_module(
        "proj_out",
        layer::AddMatmul(
            dim + mlp_hidden_dim_, dim, /*with_bias=*/true, options_));
    attn_ = register_module("attn", FluxSingleAttention(context));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& temb,
      const torch::Tensor& image_rotary_emb = torch::Tensor()) {
    int64_t text_seq_len = encoder_hidden_states.size(1);
    auto concat_hidden_states =
        torch::cat({encoder_hidden_states, hidden_states}, 1);
    auto residual = concat_hidden_states;
    auto [norm_hidden_states, gate] = norm_(concat_hidden_states, temb);
    auto mlp_hidden_states = act_mlp_(proj_mlp_(norm_hidden_states));
    auto attn_output = attn_->forward(norm_hidden_states, image_rotary_emb);
    auto hidden_states_cat = torch::cat({attn_output, mlp_hidden_states}, 2);
    auto out = proj_out_(hidden_states_cat);
    out = gate.unsqueeze(1) * out;
    out = residual + out;
    if (out.scalar_type() == torch::kFloat16) {
      out = torch::clamp(out, -65504.0f, 65504.0f);
    }
    auto encoder_hidden_states_out = out.slice(1, 0, text_seq_len);
    auto hidden_states_out = out.slice(1, text_seq_len);
    return std::make_tuple(encoder_hidden_states_out, hidden_states_out);
  }

  void load_state_dict(const StateDict& state_dict) {
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    proj_mlp_->load_state_dict(state_dict.get_dict_with_prefix("proj_mlp."));
    proj_out_->load_state_dict(state_dict.get_dict_with_prefix("proj_out."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    attn_->verify_loaded_weights(prefix + "attn.");
    norm_->verify_loaded_weights(prefix + "norm.");
    proj_mlp_->verify_loaded_weights(prefix + "proj_mlp.");
    proj_out_->verify_loaded_weights(prefix + "proj_out.");
  }

 private:
  AdaLayerNormZeroSingle norm_{nullptr};
  layer::AddMatmul proj_mlp_{nullptr};
  layer::AddMatmul proj_out_{nullptr};
  torch::nn::Functional act_mlp_{nullptr};
  FluxSingleAttention attn_{nullptr};
  int64_t mlp_hidden_dim_;
  torch::TensorOptions options_;
};
TORCH_MODULE(LongCatImageSingleTransformerBlock);

// LongCatImageTransformerBlock can reuse FluxTransformerBlock since the
// interface is identical
using LongCatImageTransformerBlock = FluxTransformerBlock;

// LongCatImagePosEmbed - similar to FluxPosEmbed but defined here to avoid
// dependency on pipeline_flux_base.h
class LongCatImagePosEmbedImpl : public torch::nn::Module {
 public:
  LongCatImagePosEmbedImpl(int64_t theta, std::vector<int64_t> axes_dim) {
    theta_ = theta;
    axes_dim_ = axes_dim;
  }

  std::pair<torch::Tensor, torch::Tensor> forward_cache(
      const torch::Tensor& txt_ids,
      const torch::Tensor& img_ids,
      int64_t height = -1,
      int64_t width = -1) {
    auto seq_len = txt_ids.size(0);

    // recompute the cache if height or width changes
    if (height != cached_image_height_ || width != cached_image_width_ ||
        seq_len != max_seq_len_) {
      torch::Tensor ids = torch::cat({txt_ids, img_ids}, 0);
      cached_image_height_ = height;
      cached_image_width_ = width;
      max_seq_len_ = seq_len;
      auto [cos, sin] = forward(ids);
      freqs_cos_cache_ = std::move(cos);
      freqs_sin_cache_ = std::move(sin);
    }
    return {freqs_cos_cache_, freqs_sin_cache_};
  }

  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& ids) {
    int64_t n_axes = ids.size(-1);
    std::vector<torch::Tensor> cos_out, sin_out;
    auto pos = ids.to(torch::kFloat32);
    torch::Dtype freqs_dtype = torch::kFloat64;
    for (int64_t i = 0; i < n_axes; ++i) {
      auto pos_slice = pos.select(-1, i);
      auto result = get_1d_rotary_pos_embed(axes_dim_[i],
                                            pos_slice,
                                            static_cast<float>(theta_),
                                            true,  // use_real
                                            1.0f,  // linear_factor
                                            1.0f,  // ntk_factor
                                            true,  // repeat_interleave_real
                                            freqs_dtype);
      auto cos = result[0];
      auto sin = result[1];
      cos_out.push_back(cos);
      sin_out.push_back(sin);
    }

    auto freqs_cos = torch::cat(cos_out, -1);
    auto freqs_sin = torch::cat(sin_out, -1);
    return {freqs_cos, freqs_sin};
  }

 private:
  int64_t theta_;
  std::vector<int64_t> axes_dim_;
  torch::Tensor freqs_cos_cache_;
  torch::Tensor freqs_sin_cache_;
  int64_t max_seq_len_ = -1;
  int64_t cached_image_height_ = -1;
  int64_t cached_image_width_ = -1;
};
TORCH_MODULE(LongCatImagePosEmbed);

class LongCatImageTransformer2DModelImpl : public torch::nn::Module {
 public:
  explicit LongCatImageTransformer2DModelImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto num_attention_heads = model_args.n_heads();
    auto attention_head_dim = model_args.head_dim();
    auto inner_dim = num_attention_heads * attention_head_dim;
    auto joint_attention_dim = model_args.joint_attention_dim();
    auto axes_dims_rope = model_args.axes_dims_rope();
    auto num_layers = model_args.num_layers();
    auto num_single_layers = model_args.num_single_layers();
    auto patch_size = model_args.mm_patch_size();
    in_channels_ = model_args.in_channels();
    out_channels_ = model_args.out_channels();

    pos_embed_ =
        register_module("pos_embed",
                        LongCatImagePosEmbed(
                            10000, axes_dims_rope));  // ROPE_SCALE_BASE = 10000
    time_embed_ =
        register_module("time_embed", LongCatImageTimestepEmbeddings(context));
    context_embedder_ = register_module(
        "context_embedder",
        layer::AddMatmul(
            joint_attention_dim, inner_dim, /*with_bias=*/true, options_));
    x_embedder_ = register_module(
        "x_embedder",
        layer::AddMatmul(
            in_channels_, inner_dim, /*with_bias=*/true, options_));

    transformer_blocks_ =
        register_module("transformer_blocks", torch::nn::ModuleList());
    transformer_block_layers_.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i) {
      auto block = LongCatImageTransformerBlock(context);
      transformer_blocks_->push_back(block);
      transformer_block_layers_.push_back(block);
    }

    single_transformer_blocks_ =
        register_module("single_transformer_blocks", torch::nn::ModuleList());
    single_transformer_block_layers_.reserve(num_single_layers);
    for (int64_t i = 0; i < num_single_layers; ++i) {
      auto block = LongCatImageSingleTransformerBlock(context);
      single_transformer_blocks_->push_back(block);
      single_transformer_block_layers_.push_back(block);
    }

    norm_out_ = register_module("norm_out", AdaLayerNormContinuous(context));
    proj_out_ = register_module(
        "proj_out",
        layer::AddMatmul(inner_dim,
                         patch_size * patch_size * out_channels_,
                         /*with_bias=*/true,
                         options_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& timestep,
                        const torch::Tensor& image_rotary_emb) {
    torch::Tensor hidden_states = x_embedder_->forward(hidden_states_input);
    auto timestep_scaled = timestep.to(hidden_states.dtype()) * 1000.0f;
    torch::Tensor temb = time_embed_->forward(timestep_scaled);
    torch::Tensor encoder_hidden_states =
        context_embedder_->forward(encoder_hidden_states_input);

    for (int64_t i = 0; i < transformer_block_layers_.size(); ++i) {
      auto block = transformer_block_layers_[i];
      auto [new_hidden, new_encoder_hidden] = block->forward(
          hidden_states, encoder_hidden_states, temb, image_rotary_emb);
      hidden_states = new_hidden;
      encoder_hidden_states = new_encoder_hidden;
    }

    for (int64_t i = 0; i < single_transformer_block_layers_.size(); ++i) {
      auto block = single_transformer_block_layers_[i];
      // Block returns (encoder, hidden) per diffusers
      // LongCatImageSingleTransformerBlock.
      auto [new_encoder_hidden, new_hidden] = block->forward(
          hidden_states, encoder_hidden_states, temb, image_rotary_emb);
      hidden_states = new_hidden;
      encoder_hidden_states = new_encoder_hidden;
    }

    auto output_hidden = norm_out_(hidden_states, temb);
    return proj_out_(output_hidden);
  }

  // Forward method with step_idx for cache separation (used for CFG)
  // Note: LongCatImageTransformer2DModel doesn't use cache, so step_idx is
  // ignored But we keep this signature for consistency and future cache support
  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& timestep,
                        const torch::Tensor& image_rotary_emb,
                        int64_t step_idx) {
    // For LongCatImageTransformer2DModel, we don't use cache, so step_idx is
    // ignored But we keep this signature for consistency with
    // FluxTransformer2DModel
    return forward(hidden_states_input,
                   encoder_hidden_states_input,
                   timestep,
                   image_rotary_emb);
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      context_embedder_->load_state_dict(
          state_dict->get_dict_with_prefix("context_embedder."));
      x_embedder_->load_state_dict(
          state_dict->get_dict_with_prefix("x_embedder."));
      time_embed_->load_state_dict(
          state_dict->get_dict_with_prefix("time_embed."));
      for (int64_t i = 0; i < transformer_block_layers_.size(); ++i) {
        auto block = transformer_block_layers_[i];
        block->load_state_dict(state_dict->get_dict_with_prefix(
            "transformer_blocks." + std::to_string(i) + "."));
      }
      for (int64_t i = 0; i < single_transformer_block_layers_.size(); ++i) {
        auto block = single_transformer_block_layers_[i];
        block->load_state_dict(state_dict->get_dict_with_prefix(
            "single_transformer_blocks." + std::to_string(i) + "."));
      }
      norm_out_->load_state_dict(state_dict->get_dict_with_prefix("norm_out."));
      proj_out_->load_state_dict(state_dict->get_dict_with_prefix("proj_out."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    context_embedder_->verify_loaded_weights(prefix + "context_embedder.");
    x_embedder_->verify_loaded_weights(prefix + "x_embedder.");
    time_embed_->verify_loaded_weights(prefix + "time_embed.");
    for (int64_t i = 0; i < transformer_block_layers_.size(); ++i) {
      auto block = transformer_block_layers_[i];
      block->verify_loaded_weights(prefix + "transformer_blocks." +
                                   std::to_string(i) + ".");
    }
    for (int64_t i = 0; i < single_transformer_block_layers_.size(); ++i) {
      auto block = single_transformer_block_layers_[i];
      block->verify_loaded_weights(prefix + "single_transformer_blocks." +
                                   std::to_string(i) + ".");
    }
    norm_out_->verify_loaded_weights(prefix + "norm_out.");
    proj_out_->verify_loaded_weights(prefix + "proj_out.");
  }

  int64_t in_channels() { return in_channels_; }

 private:
  LongCatImagePosEmbed pos_embed_{nullptr};
  LongCatImageTimestepEmbeddings time_embed_{nullptr};
  layer::AddMatmul context_embedder_{nullptr};
  layer::AddMatmul x_embedder_{nullptr};
  torch::nn::ModuleList transformer_blocks_{nullptr};
  std::vector<LongCatImageTransformerBlock> transformer_block_layers_;
  torch::nn::ModuleList single_transformer_blocks_{nullptr};
  std::vector<LongCatImageSingleTransformerBlock>
      single_transformer_block_layers_;
  AdaLayerNormContinuous norm_out_{nullptr};
  layer::AddMatmul proj_out_{nullptr};
  int64_t in_channels_;
  int64_t out_channels_;
  torch::TensorOptions options_;
};
TORCH_MODULE(LongCatImageTransformer2DModel);

REGISTER_MODEL_ARGS(LongCatImageTransformer2DModel, [&] {
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(mm_patch_size, "patch_size", 1);
  LOAD_ARG_OR(in_channels, "in_channels", 64);
  LOAD_ARG_OR(out_channels, "out_channels", 64);
  LOAD_ARG_OR(num_layers, "num_layers", 19);
  LOAD_ARG_OR(num_single_layers, "num_single_layers", 38);
  LOAD_ARG_OR(head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 24);
  LOAD_ARG_OR(joint_attention_dim,
              "joint_attention_dim",
              3584);  // LongCat-Image specific
  LOAD_ARG_OR(pooled_projection_dim,
              "pooled_projection_dim",
              3584);  // LongCat-Image specific
  LOAD_ARG_OR(
      axes_dims_rope, "axes_dims_rope", (std::vector<int64_t>{16, 56, 56}));
});
}  // namespace xllm
