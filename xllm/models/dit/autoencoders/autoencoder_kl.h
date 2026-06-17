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
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/add_matmul.h"
#include "framework/model_context.h"
#include "models/dit/processors/vae_image_processor.h"
#include "models/dit/utils/diagonal_gaussian_distribution.h"
#include "models/dit/utils/util.h"
#include "models/model_registry.h"

// VAE model compatible with huggingface weights
// ref to:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl.py

namespace xllm {

class AttentionImpl : public torch::nn::Module {
 public:
  explicit AttentionImpl(ModelContext context) {
    ModelArgs model_args = context.get_model_args();
    int64_t query_dim = model_args.block_out_channels().back();
    int64_t head_dim = model_args.block_out_channels().back();
    int64_t num_heads = query_dim / head_dim;
    num_heads_ = num_heads;
    int64_t norm_num_groups = model_args.norm_num_groups();
    if (norm_num_groups > 0) {
      group_norm_ =
          register_module("group_norm",
                          torch::nn::GroupNorm(torch::nn::GroupNormOptions(
                                                   norm_num_groups, query_dim)
                                                   .eps(1e-6f)));
    }
    int64_t inner_dim = head_dim * num_heads;
    auto options = context.get_tensor_options();

    to_q_ = register_module(
        "to_q",
        layer::AddMatmul(query_dim, inner_dim, /*with_bias=*/true, options));
    to_k_ = register_module(
        "to_k",
        layer::AddMatmul(query_dim, inner_dim, /*with_bias=*/true, options));
    to_v_ = register_module(
        "to_v",
        layer::AddMatmul(query_dim, inner_dim, /*with_bias=*/true, options));
    to_out_ = register_module(
        "to_out",
        layer::AddMatmul(inner_dim, query_dim, /*with_bias=*/true, options));
  }

  torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor temb) {
    torch::Tensor residual = hidden_states;
    int64_t input_ndim = hidden_states.dim();
    int64_t batch_size;
    int64_t channel;
    int64_t height;
    int64_t width;
    if (input_ndim == 4) {
      batch_size = hidden_states.size(0);
      channel = hidden_states.size(1);
      height = hidden_states.size(2);
      width = hidden_states.size(3);

      // reshape tensor：[B, C, H, W] -> [B, C, H*W]
      hidden_states = hidden_states.view({batch_size, channel, height * width});

      // swap dim 1&2：[B, C, H*W] -> [B, H*W, C]
      hidden_states = hidden_states.transpose(1, 2);
    }
    if (temb.defined()) {
      torch::IntArrayRef input_shape = temb.sizes();
      batch_size = input_shape[0];
      int64_t seq_length = input_shape[1];
    }
    hidden_states =
        group_norm_->forward(hidden_states.transpose(1, 2)).transpose(1, 2);
    torch::Tensor query = to_q_(hidden_states);
    // if temb is none
    if (!temb.defined()) {
      temb = hidden_states.clone();
    }
    torch::Tensor key = to_k_(temb);
    torch::Tensor value = to_v_(temb);
    int64_t inner_dim = key.size(-1);
    int64_t head_dim = inner_dim / num_heads_;
    query = query.view({batch_size, -1, num_heads_, head_dim}).transpose(1, 2);
    key = key.view({batch_size, -1, num_heads_, head_dim}).transpose(1, 2);
    value = value.view({batch_size, -1, num_heads_, head_dim}).transpose(1, 2);
    hidden_states =
        torch::scaled_dot_product_attention(query,
                                            key,
                                            value,
                                            c10::nullopt,  // attn_mask
                                            0.0,           // dropout_p
                                            false          // is_causal
        );
    hidden_states = hidden_states.transpose(1, 2).reshape(
        {batch_size, -1, num_heads_ * head_dim});
    hidden_states = hidden_states.to(query.dtype());
    hidden_states = to_out_(hidden_states);
    // alignment right
    if (input_ndim == 4) {
      hidden_states = hidden_states.transpose(-1, -2).reshape(
          {batch_size, channel, height, width});
    }
    hidden_states = hidden_states + residual;
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    // to_q_
    to_q_->load_state_dict(state_dict.get_dict_with_prefix("to_q."));
    // to_k_
    to_k_->load_state_dict(state_dict.get_dict_with_prefix("to_k."));
    // to_v_
    to_v_->load_state_dict(state_dict.get_dict_with_prefix("to_v."));
    // to_out_
    to_out_->load_state_dict(state_dict.get_dict_with_prefix("to_out.0."));
    if (group_norm_) {
      weight::load_weight(state_dict,
                          "group_norm.weight",
                          group_norm_->weight,
                          group_norm_weight_loaded_);
      weight::load_weight(state_dict,
                          "group_norm.bias",
                          group_norm_->bias,
                          group_norm_bias_loaded_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    to_q_->verify_loaded_weights(prefix + "to_q.");
    to_k_->verify_loaded_weights(prefix + "to_k.");
    to_v_->verify_loaded_weights(prefix + "to_v.");
    to_out_->verify_loaded_weights(prefix + "to_out.0.");
    if (group_norm_) {
      CHECK(group_norm_weight_loaded_)
          << "weight is not loaded for " << prefix + "group_norm.weight";
      CHECK(group_norm_bias_loaded_)
          << "bias is not loaded for " << prefix + "group_norm.bias";
    }
  }

 private:
  torch::nn::GroupNorm group_norm_ = nullptr;
  bool group_norm_weight_loaded_ = false;
  bool group_norm_bias_loaded_ = false;
  layer::AddMatmul to_q_ = nullptr;
  layer::AddMatmul to_k_ = nullptr;
  layer::AddMatmul to_v_ = nullptr;
  layer::AddMatmul to_out_ = nullptr;
  int64_t num_heads_ = 1;
};
TORCH_MODULE(Attention);

class Downsample2DImpl : public torch::nn::Module {
 public:
  Downsample2DImpl(ModelContext context,
                   int64_t channels,
                   int64_t padding = 1) {
    padding_ = padding;
    conv_ = register_module(
        "conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3)
                              .stride(2)
                              .padding(padding)
                              .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const std::vector<torch::Tensor>& args = {}) {
    torch::Tensor hs = hidden_states;
    if (padding_ == 0) {
      hs = torch::nn::functional::pad(
          hs,
          torch::nn::functional::PadFuncOptions({0, 1, 0, 1})
              .mode(torch::kConstant)
              .value(0.0f));
    }
    hs = conv_(hs);
    return hs;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(
        state_dict, "conv.weight", conv_->weight, is_weight_loaded_);
    weight::load_weight(state_dict, "conv.bias", conv_->bias, is_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv.weight";
    CHECK(is_bias_loaded_) << "weight is not loaded for "
                           << prefix + "conv.bias";
  }

 private:
  int64_t padding_ = 0;
  torch::nn::Conv2d conv_ = nullptr;
  bool is_weight_loaded_ = false;
  bool is_bias_loaded_ = false;
};
TORCH_MODULE(Downsample2D);

class Upsample2DImpl : public torch::nn::Module {
 public:
  Upsample2DImpl(ModelContext context, int64_t channels) {
    conv_ = register_module(
        "conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3)
                              .padding(1)
                              .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const std::vector<int64_t>& output_size = {}) {
    torch::Tensor x = hidden_states;
    if (x.size(0) >= 64) {
      x = x.contiguous();
    }

    torch::nn::functional::InterpolateFuncOptions opts;
    opts.mode(torch::kNearest);

    if (!output_size.empty()) {
      opts.size(output_size);
    } else {
      opts.scale_factor(std::vector<double>{2.0, 2.0});
    }

    x = torch::nn::functional::interpolate(x, opts);
    x = conv_(x);
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(
        state_dict, "conv.weight", conv_->weight, is_weight_loaded_);
    weight::load_weight(state_dict, "conv.bias", conv_->bias, is_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv.weight";
    CHECK(is_bias_loaded_) << "weight is not loaded for "
                           << prefix + "conv.bias";
  }

 private:
  torch::nn::ConvTranspose2d conv_transpose_ = nullptr;
  torch::nn::Conv2d conv_ = nullptr;
  bool is_weight_loaded_ = false;
  bool is_bias_loaded_ = false;
};
TORCH_MODULE(Upsample2D);

class ResnetBlock2DImpl : public torch::nn::Module {
 public:
  ResnetBlock2DImpl(ModelContext context,
                    int64_t in_channels,
                    int64_t out_channels,
                    const std::string& time_embedding_norm = "default")
      : in_channels_(in_channels),
        out_channels_(out_channels),
        time_embedding_norm_(time_embedding_norm) {
    ModelArgs model_args = context.get_model_args();
    int64_t groups_out_val = model_args.norm_num_groups();

    norm1_ =
        register_module("norm1",
                        torch::nn::GroupNorm(torch::nn::GroupNormOptions(
                                                 groups_out_val, in_channels_)
                                                 .eps(1e-6f)));
    conv1_ =
        register_module("conv1",
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                              in_channels_, out_channels_, 3)
                                              .stride(1)
                                              .padding(1)
                                              .bias(true)));

    norm2_ =
        register_module("norm2",
                        torch::nn::GroupNorm(torch::nn::GroupNormOptions(
                                                 groups_out_val,  // num_groups
                                                 out_channels_)  // num_channels
                                                 .eps(1e-6f)));

    int64_t conv2_out_channels = out_channels_;
    conv2_ = register_module(
        "conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels_,
                                                   conv2_out_channels,
                                                   3  // kernel_size=3
                                                   )
                              .stride(1)
                              .padding(1)
                              .bias(true)));

    nonlinearity_ =
        register_module("nonlinearity", torch::nn::Functional(torch::silu));

    bool use_in_shortcut_val = in_channels_ != conv2_out_channels;
    if (use_in_shortcut_val) {
      conv_shortcut_ = register_module(
          "conv_shortcut",
          torch::nn::Conv2d(
              torch::nn::Conv2dOptions(in_channels_, conv2_out_channels, 1)
                  .stride(1)
                  .padding(0)
                  .bias(true)));
    }
  }

  torch::Tensor forward(const torch::Tensor& input_tensor,
                        const torch::Tensor& temb = torch::Tensor()) {
    torch::Tensor hidden_states = input_tensor;
    hidden_states = norm1_(hidden_states);
    hidden_states = nonlinearity_(hidden_states);
    hidden_states = conv1_(hidden_states);
    torch::Tensor temb_processed = temb;
    torch::Tensor input_tensor_processed = input_tensor;

    if (time_embedding_norm_ == "default") {
      if (temb_processed.defined()) {
        hidden_states = hidden_states + temb_processed;
      }
      hidden_states = norm2_(hidden_states);
    } else if (time_embedding_norm_ == "scale_shift") {
      if (!temb_processed.defined()) {
        LOG(FATAL)
            << "temb must be defined when time_embedding_norm is scale_shift";
      }
      auto chunks = torch::chunk(temb_processed, 2, 1);
      torch::Tensor time_scale = chunks[0];
      torch::Tensor time_shift = chunks[1];
      hidden_states = norm2_(hidden_states);
      hidden_states = hidden_states * (1 + time_scale) + time_shift;
    } else {
      hidden_states = norm2_(hidden_states);
    }
    hidden_states = nonlinearity_(hidden_states);
    hidden_states = conv2_(hidden_states);

    if (conv_shortcut_) {
      input_tensor_processed =
          conv_shortcut_(input_tensor_processed.contiguous());
    }
    torch::Tensor output_tensor = (input_tensor_processed + hidden_states);

    return output_tensor;
  }

  void load_state_dict(const StateDict& state_dict) {
    // conv1_
    weight::load_weight(
        state_dict, "conv1.weight", conv1_->weight, is_conv1_weight_loaded_);
    weight::load_weight(
        state_dict, "conv1.bias", conv1_->bias, is_conv1_bias_loaded_);
    // norm1_
    weight::load_weight(
        state_dict, "norm1.weight", norm1_->weight, is_norm1_weight_loaded_);
    weight::load_weight(
        state_dict, "norm1.bias", norm1_->bias, is_norm1_bias_loaded_);
    // norm2_
    weight::load_weight(
        state_dict, "norm2.weight", norm2_->weight, is_norm2_weight_loaded_);
    weight::load_weight(
        state_dict, "norm2.bias", norm2_->bias, is_norm2_bias_loaded_);
    // conv2_
    weight::load_weight(
        state_dict, "conv2.weight", conv2_->weight, is_conv2_weight_loaded_);
    weight::load_weight(
        state_dict, "conv2.bias", conv2_->bias, is_conv2_bias_loaded_);
    if (conv_shortcut_) {
      // conv_shortcut_
      weight::load_weight(state_dict,
                          "conv_shortcut.weight",
                          conv_shortcut_->weight,
                          is_conv_shortcut_weight_loaded_);
      weight::load_weight(state_dict,
                          "conv_shortcut.bias",
                          conv_shortcut_->bias,
                          is_conv_shortcut_bias_loaded_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_norm1_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm1.weight";
    CHECK(is_norm1_bias_loaded_)
        << "bias is not loaded for " << prefix + "norm1.bias";
    CHECK(is_conv1_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv1.weight";
    CHECK(is_conv1_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv1.bias";
    CHECK(is_norm2_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm2.weight";
    CHECK(is_norm2_bias_loaded_)
        << "bias is not loaded for " << prefix + "norm2.bias";
    CHECK(is_conv2_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv2.weight";
    CHECK(is_conv2_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv2.bias";
    if (conv_shortcut_) {
      CHECK(is_conv_shortcut_weight_loaded_)
          << "weight is not loaded for " << prefix + "conv_shortcut.weight";
      CHECK(is_conv_shortcut_bias_loaded_)
          << "bias is not loaded for " << prefix + "conv_shortcut.bias";
    }
  }

 private:
  int64_t in_channels_;
  int64_t out_channels_;
  std::string time_embedding_norm_;
  torch::nn::GroupNorm norm1_ = nullptr;
  torch::nn::Conv2d conv1_ = nullptr;
  torch::nn::GroupNorm norm2_ = nullptr;
  torch::nn::Conv2d conv2_ = nullptr;
  torch::nn::Functional nonlinearity_ = nullptr;
  torch::nn::Conv2d conv_shortcut_ = nullptr;

  bool is_norm1_weight_loaded_ = false;
  bool is_norm1_bias_loaded_ = false;
  bool is_conv1_weight_loaded_ = false;
  bool is_conv1_bias_loaded_ = false;
  bool is_norm2_weight_loaded_ = false;
  bool is_norm2_bias_loaded_ = false;
  bool is_conv2_weight_loaded_ = false;
  bool is_conv2_bias_loaded_ = false;
  bool is_conv_shortcut_weight_loaded_ = false;
  bool is_conv_shortcut_bias_loaded_ = false;
};
TORCH_MODULE(ResnetBlock2D);

class UNetMidBlock2DImpl : public torch::nn::Module {
 public:
  explicit UNetMidBlock2DImpl(ModelContext context) {
    ModelArgs model_args = context.get_model_args();
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    attentions_ = register_module("attentions", torch::nn::ModuleList());
    int64_t in_channels = model_args.block_out_channels().back();
    add_attention_ = model_args.mid_block_add_attention();
    resnet_layers_.reserve(2);
    attention_layers_.reserve(add_attention_ ? 1 : 0);
    auto block1 = ResnetBlock2D(context, in_channels, in_channels, "default");
    resnets_->push_back(block1);
    resnet_layers_.push_back(block1);
    int64_t attn_head_dim = model_args.block_out_channels().back();
    int64_t num_heads = in_channels / attn_head_dim;
    if (add_attention_) {
      auto attn_block = Attention(context);
      attentions_->push_back(attn_block);
      attention_layers_.push_back(attn_block);
    } else {
      // Add an empty module as a placeholder.
      attentions_->push_back(torch::nn::Sequential());
    }
    auto block2 = ResnetBlock2D(context, in_channels, in_channels, "default");
    resnets_->push_back(block2);
    resnet_layers_.push_back(block2);
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb = torch::Tensor()) {
    torch::Tensor current_hidden =
        resnet_layers_[0]->forward(hidden_states, temb);
    for (size_t i = 0; i < attention_layers_.size(); ++i) {
      if (add_attention_) {
        auto attn = attention_layers_[i];
        current_hidden = attn->forward(current_hidden, temb);
      }
      auto resnet = resnet_layers_[i + 1];
      current_hidden = resnet->forward(current_hidden, temb);
    }

    return current_hidden;
  }
  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnet_layers_.size(); ++i) {
      resnet_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "resnets." + std::to_string(i) + "."));
    }
    for (size_t i = 0; i < attention_layers_.size(); ++i) {
      if (add_attention_) {
        attention_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
            "attentions." + std::to_string(i) + "."));
      }
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < resnet_layers_.size(); ++i) {
      resnet_layers_[i]->verify_loaded_weights(prefix + "resnets." +
                                               std::to_string(i) + ".");
    }
    for (size_t i = 0; i < attention_layers_.size(); ++i) {
      if (add_attention_) {
        attention_layers_[i]->verify_loaded_weights(prefix + "attentions." +
                                                    std::to_string(i) + ".");
      }
    }
  }

 private:
  torch::nn::ModuleList resnets_ = nullptr;
  std::vector<ResnetBlock2D> resnet_layers_;
  torch::nn::ModuleList attentions_ = nullptr;
  std::vector<Attention> attention_layers_;
  bool add_attention_ = false;
};
TORCH_MODULE(UNetMidBlock2D);

class DownEncoderBlock2DImpl : public torch::nn::Module {
 public:
  DownEncoderBlock2DImpl(ModelContext context,
                         int64_t in_channels,
                         int64_t out_channels,
                         bool add_downsample = true) {
    ModelArgs model_args = context.get_model_args();
    int64_t num_layers = model_args.layers_per_block();
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    downsamplers_ = register_module("downsamplers", torch::nn::ModuleList());
    resnet_layers_.reserve(num_layers);
    downsample_layers_.reserve(add_downsample ? 1 : 0);
    // initialize resnet blocks
    for (int64_t i = 0; i < num_layers; ++i) {
      const int64_t current_in_channels = (i == 0) ? in_channels : out_channels;
      auto block = ResnetBlock2D(context,
                                 current_in_channels,  // in channels
                                 out_channels,
                                 "default");
      resnets_->push_back(block);
      resnet_layers_.push_back(block);
    }

    // initialize downsamplers if needed
    if (add_downsample) {
      auto downsampler = Downsample2D(context, out_channels, 0);
      downsamplers_->push_back(downsampler);
      downsample_layers_.push_back(downsampler);
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb = torch::Tensor(),
                        const std::vector<torch::Tensor>& args = {}) {
    std::vector<torch::Tensor> output_states;
    torch::Tensor current_hidden = hidden_states;
    // handle resnet blocks
    for (size_t i = 0; i < resnet_layers_.size(); ++i) {
      auto resnet = resnet_layers_[i];
      current_hidden =
          resnet->forward(current_hidden.clone(),
                          temb.defined() ? temb.clone() : torch::Tensor());
      output_states.push_back(current_hidden);
    }
    // handle downsampling
    if (downsamplers_) {
      for (size_t i = 0; i < downsample_layers_.size(); ++i) {
        auto downsampler = downsample_layers_[i];
        current_hidden = downsampler->forward(current_hidden.clone());
        output_states.push_back(current_hidden);
      }
    }
    return current_hidden;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnet_layers_.size(); ++i) {
      resnet_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "resnets." + std::to_string(i) + "."));
    }
    for (size_t i = 0; i < downsample_layers_.size(); ++i) {
      downsample_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "downsamplers." + std::to_string(i) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < resnet_layers_.size(); ++i) {
      resnet_layers_[i]->verify_loaded_weights(prefix + "resnets." +
                                               std::to_string(i) + ".");
    }
    for (size_t i = 0; i < downsample_layers_.size(); ++i) {
      downsample_layers_[i]->verify_loaded_weights(prefix + "downsamplers." +
                                                   std::to_string(i) + ".");
    }
  }

 private:
  torch::nn::ModuleList resnets_ = nullptr;
  std::vector<ResnetBlock2D> resnet_layers_;
  torch::nn::ModuleList downsamplers_ = nullptr;
  std::vector<Downsample2D> downsample_layers_;
};
TORCH_MODULE(DownEncoderBlock2D);

class UpDecoderBlock2DImpl : public torch::nn::Module {
 public:
  UpDecoderBlock2DImpl(ModelContext context,
                       int64_t in_channels,
                       int64_t out_channels,
                       bool add_upsample = true) {
    ModelArgs model_args = context.get_model_args();
    int64_t num_layers = model_args.layers_per_block() + 1;
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    resnet_layers_.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i) {
      int64_t input_channels = (i == 0) ? in_channels : out_channels;
      auto block =
          ResnetBlock2D(context, input_channels, out_channels, "group");
      resnets_->push_back(block);
      resnet_layers_.push_back(block);
    }

    if (add_upsample) {
      add_upsample_ = true;
      upsamplers_ = register_module("upsamplers", torch::nn::ModuleList());
      upsample_layers_.reserve(1);
      auto upsampler = Upsample2D(context, out_channels);
      upsamplers_->push_back(upsampler);
      upsample_layers_.push_back(upsampler);
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb = torch::Tensor(),
                        const std::vector<torch::Tensor>& skip_states = {},
                        const std::vector<torch::Tensor>& args = {}) {
    torch::Tensor current_hidden = hidden_states;

    for (size_t i = 0; i < resnet_layers_.size(); ++i) {
      current_hidden = resnet_layers_[i]->forward(current_hidden, temb);
    }
    if (add_upsample_) {
      for (size_t i = 0; i < upsample_layers_.size(); ++i) {
        auto upsampler = upsample_layers_[i];
        current_hidden = upsampler->forward(current_hidden);
      }
    }
    return current_hidden;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnet_layers_.size(); ++i) {
      resnet_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "resnets." + std::to_string(i) + "."));
    }

    if (add_upsample_) {
      for (size_t i = 0; i < upsample_layers_.size(); ++i) {
        upsample_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
            "upsamplers." + std::to_string(i) + "."));
      }
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < resnet_layers_.size(); ++i) {
      resnet_layers_[i]->verify_loaded_weights(prefix + "resnets." +
                                               std::to_string(i) + ".");
    }
    if (add_upsample_) {
      for (size_t i = 0; i < upsample_layers_.size(); ++i) {
        upsample_layers_[i]->verify_loaded_weights(prefix + "upsamplers." +
                                                   std::to_string(i) + ".");
      }
    }
  }

 private:
  torch::nn::ModuleList resnets_ = nullptr;
  std::vector<ResnetBlock2D> resnet_layers_;
  torch::nn::ModuleList upsamplers_ = nullptr;
  std::vector<Upsample2D> upsample_layers_;
  bool add_upsample_ = false;
};
TORCH_MODULE(UpDecoderBlock2D);

using dit::DiagonalGaussianDistribution;

// VAE standard encoder implementation
// This class is used to encode images into latent representations.
class VAEEncoderImpl : public torch::nn::Module {
 public:
  explicit VAEEncoderImpl(const ModelContext& context) {
    ModelArgs args = context.get_model_args();
    down_blocks_ = register_module("down_blocks", torch::nn::ModuleList());
    auto options = context.get_tensor_options();
    conv_in_ = register_module(
        "conv_in",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(args.in_channels(),
                                                   args.block_out_channels()[0],
                                                   3)
                              .stride(1)
                              .padding(1)
                              .bias(true)));
    // downblocks
    int32_t output_channels = args.block_out_channels()[0];
    down_layers_.reserve(args.down_block_types().size());
    for (size_t i = 0; i < args.down_block_types().size(); i++) {
      int32_t input_channels = output_channels;
      output_channels = args.block_out_channels()[i];
      bool is_final_block = (i == args.block_out_channels().size() - 1);
      auto down_block = DownEncoderBlock2D(
          context, input_channels, output_channels, !is_final_block);
      down_blocks_->push_back(down_block);
      down_layers_.push_back(down_block);
    }
    // mid blocks
    mid_block_ = register_module("mid_block", UNetMidBlock2D(context));
    conv_norm_out_ = register_module(
        "conv_norm_out",
        torch::nn::GroupNorm(
            torch::nn::GroupNormOptions(args.norm_num_groups(),
                                        args.block_out_channels().back())
                .eps(1e-6)));
    conv_act_ = register_module("conv_act", torch::nn::Functional(torch::silu));
    conv_out_ = register_module(
        "conv_out",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(
                args.block_out_channels().back(), 2 * args.latent_channels(), 3)
                .padding(1)
                .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& images) {
    auto sample = conv_in_(images);
    for (size_t i = 0; i < down_layers_.size(); ++i) {
      sample = down_layers_[i]->forward(sample);
    }
    sample = mid_block_(sample);
    sample = conv_norm_out_(sample);
    sample = conv_act_(sample);
    sample = conv_out_(sample);
    return sample;
  }

  void load_state_dict(const StateDict& state_dict) {
    LOG(INFO) << "Loading state_dict for VAEEncoder";
    // conv_in_
    weight::load_weight(state_dict,
                        "conv_in.weight",
                        conv_in_->weight,
                        is_conv_in_weight_loaded_);
    weight::load_weight(
        state_dict, "conv_in.bias", conv_in_->bias, is_conv_in_bias_loaded_);
    // conv_norm_out_
    weight::load_weight(state_dict,
                        "conv_norm_out.weight",
                        conv_norm_out_->weight,
                        is_conv_norm_out_weight_loaded_);
    weight::load_weight(state_dict,
                        "conv_norm_out.bias",
                        conv_norm_out_->bias,
                        is_conv_norm_out_bias_loaded_);
    // conv_out_
    weight::load_weight(state_dict,
                        "conv_out.weight",
                        conv_out_->weight,
                        is_conv_out_weight_loaded_);
    weight::load_weight(
        state_dict, "conv_out.bias", conv_out_->bias, is_conv_out_bias_loaded_);
    for (size_t i = 0; i < down_layers_.size(); ++i) {
      down_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "down_blocks." + std::to_string(i) + "."));
    }
    mid_block_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_conv_in_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv_in.weight";
    CHECK(is_conv_in_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv_in.bias";
    CHECK(is_conv_norm_out_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv_norm_out.weight";
    CHECK(is_conv_norm_out_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv_norm_out.bias";
    CHECK(is_conv_out_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv_out.weight";
    CHECK(is_conv_out_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv_out.bias";
    for (size_t i = 0; i < down_layers_.size(); ++i) {
      down_layers_[i]->verify_loaded_weights(prefix + "down_blocks." +
                                             std::to_string(i) + ".");
    }
    mid_block_->verify_loaded_weights(prefix + "mid_block.");
  }

 private:
  torch::nn::Conv2d conv_in_ = nullptr;
  torch::nn::Conv2d conv_out_ = nullptr;
  torch::nn::Functional conv_act_ = nullptr;
  torch::nn::GroupNorm conv_norm_out_ = nullptr;
  torch::nn::ModuleList down_blocks_ = nullptr;
  std::vector<DownEncoderBlock2D> down_layers_;
  UNetMidBlock2D mid_block_ = nullptr;

  bool is_conv_in_weight_loaded_ = false;
  bool is_conv_in_bias_loaded_ = false;
  bool is_conv_out_weight_loaded_ = false;
  bool is_conv_out_bias_loaded_ = false;
  bool is_conv_norm_out_weight_loaded_ = false;
  bool is_conv_norm_out_bias_loaded_ = false;
};
TORCH_MODULE(VAEEncoder);

// VAE standart decoder implementation
//  This class is used to decode the latent representations into images.
class VAEDecoderImpl : public torch::nn::Module {
 public:
  explicit VAEDecoderImpl(const ModelContext& context) {
    ModelArgs args = context.get_model_args();
    auto options = context.get_tensor_options();
    up_blocks_ = register_module("up_blocks", torch::nn::ModuleList());
    conv_in_ = register_module(
        "conv_in",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(
                args.latent_channels(), args.block_out_channels().back(), 3)
                .stride(1)
                .padding(1)
                .bias(true)));
    // mid blocks
    mid_block_ = register_module("mid_block", UNetMidBlock2D(context));
    // up blocks
    std::vector<int64_t> reversed_block_out_channels(
        args.block_out_channels().rbegin(), args.block_out_channels().rend());
    int64_t output_channel = reversed_block_out_channels[0];

    for (size_t i = 0; i < args.up_block_types().size(); ++i) {
      const std::string& up_block_type = args.up_block_types()[i];
      int64_t prev_output_channel = output_channel;
      output_channel = reversed_block_out_channels[i];
      bool is_final_block = (i == args.block_out_channels().size() - 1);
      // Create the up block using the factory function
      auto up_block = UpDecoderBlock2D(
          context, prev_output_channel, output_channel, !is_final_block);
      up_blocks_->push_back(up_block);
      up_layers_.push_back(up_block);
      prev_output_channel = output_channel;
    }
    conv_norm_out_ = register_module(
        "conv_norm_out",
        torch::nn::GroupNorm(torch::nn::GroupNormOptions(
                                 args.norm_num_groups(),       // num_groups
                                 args.block_out_channels()[0]  // num_channels
                                 )
                                 .eps(1e-6)));
    conv_act_ = register_module("conv_act", torch::nn::Functional(torch::silu));
    conv_out_ = register_module(
        "conv_out",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(args.block_out_channels()[0],
                                                   args.out_channels(),
                                                   3)
                              .padding(1)
                              .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& latents) {
    auto sample = conv_in_(latents);
    sample = mid_block_(sample);
    for (size_t i = 0; i < up_layers_.size(); ++i) {
      sample = up_layers_[i]->forward(sample);
    }
    sample = conv_norm_out_(sample);
    sample = conv_act_(sample);
    sample = conv_out_(sample);
    return sample;
  }

  void load_state_dict(const StateDict& state_dict) {
    LOG(INFO) << "Loading state_dict for VAEDecoder";
    // conv_in_
    weight::load_weight(state_dict,
                        "conv_in.weight",
                        conv_in_->weight,
                        is_conv_in_weight_loaded_);
    weight::load_weight(
        state_dict, "conv_in.bias", conv_in_->bias, is_conv_in_bias_loaded_);
    // mid_block_
    //  mid_block_ is a UNetMidBlock2D, so we load its state
    mid_block_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
    for (size_t i = 0; i < up_layers_.size(); ++i) {
      up_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "up_blocks." + std::to_string(i) + "."));
    }
    // conv_norm_out_
    weight::load_weight(state_dict,
                        "conv_norm_out.weight",
                        conv_norm_out_->weight,
                        is_conv_norm_out_weight_loaded_);
    weight::load_weight(state_dict,
                        "conv_norm_out.bias",
                        conv_norm_out_->bias,
                        is_conv_norm_out_bias_loaded_);
    // conv_out_
    weight::load_weight(state_dict,
                        "conv_out.weight",
                        conv_out_->weight,
                        is_conv_out_weight_loaded_);
    weight::load_weight(
        state_dict, "conv_out.bias", conv_out_->bias, is_conv_out_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_conv_in_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv_in.weight";
    CHECK(is_conv_in_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv_in.bias";
    mid_block_->verify_loaded_weights(prefix + "mid_block.");
    for (size_t i = 0; i < up_layers_.size(); ++i) {
      up_layers_[i]->verify_loaded_weights(prefix + "up_blocks." +
                                           std::to_string(i) + ".");
    }
    CHECK(is_conv_norm_out_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv_norm_out.weight";
    CHECK(is_conv_norm_out_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv_norm_out.bias";
    CHECK(is_conv_out_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv_out.weight";
    CHECK(is_conv_out_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv_out.bias";
  }

 private:
  torch::nn::Conv2d conv_in_ = nullptr;
  UNetMidBlock2D mid_block_ = nullptr;
  torch::nn::ModuleList up_blocks_ = nullptr;
  std::vector<UpDecoderBlock2D> up_layers_;
  torch::nn::GroupNorm conv_norm_out_ = nullptr;
  torch::nn::Functional conv_act_ = nullptr;
  torch::nn::Conv2d conv_out_ = nullptr;

  bool is_conv_in_bias_loaded_ = false;
  bool is_conv_in_weight_loaded_ = false;
  bool is_conv_norm_out_bias_loaded_ = false;
  bool is_conv_norm_out_weight_loaded_ = false;
  bool is_conv_out_bias_loaded_ = false;
  bool is_conv_out_weight_loaded_ = false;
};
TORCH_MODULE(VAEDecoder);

// VAE implementation, including encoder and decoder
class VAEImpl : public torch::nn::Module {
 public:
  explicit VAEImpl(const ModelContext& context)
      : args_(context.get_model_args()) {
    encoder_ = register_module("encoder", VAEEncoder(context));
    decoder_ = register_module("decoder", VAEDecoder(context));
    if (args_.use_quant_conv()) {
      quant_conv_ = register_module(
          "quant_conv",
          torch::nn::Conv2d(torch::nn::Conv2dOptions(
              2 * args_.latent_channels(), 2 * args_.latent_channels(), 1)));
    }
    if (args_.use_post_quant_conv()) {
      post_quant_conv_ = register_module(
          "post_quant_conv",
          torch::nn::Conv2d(torch::nn::Conv2dOptions(
              args_.latent_channels(), args_.latent_channels(), 1)));
    }

    auto dtype = context.get_tensor_options().dtype().toScalarType();
    encoder_->to(dtype);
    decoder_->to(dtype);
    if (args_.use_quant_conv()) {
      quant_conv_->to(dtype);
    }
    if (args_.use_post_quant_conv()) {
      post_quant_conv_->to(dtype);
    }
  }

  torch::Tensor encode(const torch::Tensor& images, int64_t seed) {
    auto enc = encoder_(images);
    if (args_.use_quant_conv()) {
      enc = quant_conv_(enc);
    }
    auto posterior = DiagonalGaussianDistribution(enc);
    return posterior.sample(seed);
  }

  torch::Tensor decode(const torch::Tensor& latents) {
    torch::Tensor processed_latents = latents;

    if (args_.use_post_quant_conv()) {
      processed_latents = post_quant_conv_(processed_latents);
    }

    auto dec = decoder_(processed_latents);
    return dec;
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      encoder_->load_state_dict(state_dict->get_dict_with_prefix("encoder."));
      decoder_->load_state_dict(state_dict->get_dict_with_prefix("decoder."));
      if (args_.use_quant_conv()) {
        weight::load_weight(state_dict->get_dict_with_prefix("quant_conv."),
                            "weight",
                            quant_conv_->weight,
                            is_quant_conv_weight_);
        weight::load_weight(state_dict->get_dict_with_prefix("quant_conv."),
                            "bias",
                            quant_conv_->bias,
                            is_quant_conv_bias_);
      }
      if (args_.use_post_quant_conv()) {
        weight::load_weight(
            state_dict->get_dict_with_prefix("post_quant_conv."),
            "weight",
            post_quant_conv_->weight,
            is_post_quant_conv_weight_);
        weight::load_weight(
            state_dict->get_dict_with_prefix("post_quant_conv."),
            "bias",
            post_quant_conv_->bias,
            is_post_quant_conv_bias_);
      }
    }
    verify_loaded_weights("");
    LOG(INFO) << "VAE model loaded successfully.";
  }

  void verify_loaded_weights(const std::string& prefix) {
    encoder_->verify_loaded_weights(prefix + "encoder.");
    decoder_->verify_loaded_weights(prefix + "decoder.");
    if (args_.use_quant_conv()) {
      CHECK(is_quant_conv_weight_)
          << "weight is not loaded for " << prefix + "quant_conv.weight";
      CHECK(is_quant_conv_bias_)
          << "bias is not loaded for " << prefix + "quant_conv.bias";
    }
    if (args_.use_post_quant_conv()) {
      CHECK(is_post_quant_conv_weight_)
          << "weight is not loaded for " << prefix + "post_quant_conv.weight";
      CHECK(is_post_quant_conv_bias_)
          << "bias is not loaded for " << prefix + "post_quant_conv.bias";
    }
  }

 private:
  VAEEncoder encoder_ = nullptr;
  VAEDecoder decoder_ = nullptr;
  torch::nn::Conv2d quant_conv_ = nullptr;
  torch::nn::Conv2d post_quant_conv_ = nullptr;
  bool use_post_quant_conv_ = false;

  bool is_quant_conv_weight_ = false;
  bool is_quant_conv_bias_ = false;
  bool is_post_quant_conv_weight_ = false;
  bool is_post_quant_conv_bias_ = false;
  ModelArgs args_;
};
TORCH_MODULE(VAE);

// register the VAE model with the model registry
REGISTER_MODEL_ARGS(AutoencoderKL, [&] {
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(in_channels, "in_channels", 3);
  LOAD_ARG_OR(out_channels, "out_channels", 3);
  LOAD_ARG_OR(down_block_types,
              "down_block_types",
              (std::vector<std::string>{"DownEncoderBlock2D",
                                        "DownEncoderBlock2D",
                                        "DownEncoderBlock2D",
                                        "DownEncoderBlock2D"}));
  LOAD_ARG_OR(up_block_types,
              "up_block_types",
              (std::vector<std::string>{"UpDecoderBlock2D",
                                        "UpDecoderBlock2D",
                                        "UpDecoderBlock2D",
                                        "UpDecoderBlock2D"}));
  LOAD_ARG_OR(block_out_channels,
              "block_out_channels",
              (std::vector<int64_t>{128, 256, 512, 512}));
  LOAD_ARG_OR(layers_per_block, "layers_per_block", 2);
  LOAD_ARG_OR(latent_channels, "latent_channels", 16);
  LOAD_ARG_OR(norm_num_groups, "norm_num_groups", 32);
  LOAD_ARG_OR(sample_size, "sample_size", 1024);
  LOAD_ARG_OR(scale_factor, "scale_factor", 0.3611f);
  LOAD_ARG_OR(shift_factor, "shift_factor", 0.1159f);
  LOAD_ARG_OR(mid_block_add_attention, "mid_block_add_attention", true);
  LOAD_ARG_OR(force_upcast, "force_upcast", true);
  LOAD_ARG_OR(use_quant_conv, "use_quant_conv", false);
  LOAD_ARG_OR(use_post_quant_conv, "use_post_quant_conv", false);
});
}  // namespace xllm
