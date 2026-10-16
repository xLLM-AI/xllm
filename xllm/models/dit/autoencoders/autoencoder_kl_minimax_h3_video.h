/* Copyright 2026 The xLLM Authors.

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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/model_context/model_context.h"
#include "core/framework/model_loader/dit_model_loader.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "models/dit/utils/diagonal_gaussian_distribution.h"
#include "models/dit/utils/util.h"
#include "models/model_registry.h"
#if defined(USE_NPU)
#include "torch_npu/csrc/aten/CustomFunctions.h"
#endif

namespace xllm {
namespace minimax_h3_video {

using dit::product_checked;

class Conv3d1x1Impl final : public torch::nn::Module {
 public:
  Conv3d1x1Impl(int64_t in_channels, int64_t out_channels) {
    weight_ = register_parameter(
        "weight", torch::empty({out_channels, in_channels, 1, 1, 1}));
    bias_ = register_parameter("bias", torch::empty({out_channels}));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) const {
    return torch::conv3d(hidden_states,
                         weight_,
                         bias_,
                         /*stride=*/{1, 1, 1},
                         /*padding=*/{0, 0, 0},
                         /*dilation=*/{1, 1, 1},
                         /*groups=*/1);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight", weight_, is_weight_loaded_);
    weight::load_weight(state_dict, "bias", bias_, is_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(is_bias_loaded_) << "bias is not loaded for " << prefix + "bias";
  }

 private:
  torch::Tensor weight_;
  torch::Tensor bias_;
  bool is_weight_loaded_ = false;
  bool is_bias_loaded_ = false;
};
TORCH_MODULE(Conv3d1x1);

class CausalConv3dImpl final : public torch::nn::Module {
 public:
  CausalConv3dImpl(int64_t in_channels,
                   int64_t out_channels,
                   std::vector<int64_t> kernel_size,
                   std::vector<int64_t> stride,
                   int64_t spatial_padding,
                   int64_t temporal_padding,
                   bool reflect_spatial_padding)
      : spatial_padding_(spatial_padding),
        temporal_padding_(temporal_padding),
        reflect_spatial_padding_(reflect_spatial_padding) {
    conv_ = register_module(
        "conv",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding({0, 0, 0})));
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    if (spatial_padding_ > 0) {
      auto options = torch::nn::functional::PadFuncOptions({spatial_padding_,
                                                            spatial_padding_,
                                                            spatial_padding_,
                                                            spatial_padding_,
                                                            0,
                                                            0});
      if (reflect_spatial_padding_) {
        options = options.mode(torch::kReflect);
      } else {
        options = options.mode(torch::kConstant);
      }
      hidden_states = torch::nn::functional::pad(hidden_states, options);
    }
    if (temporal_padding_ > 0) {
      hidden_states =
          torch::nn::functional::pad(hidden_states,
                                     torch::nn::functional::PadFuncOptions(
                                         {0, 0, 0, 0, temporal_padding_, 0})
                                         .mode(torch::kConstant));
    }
    return conv_->forward(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight", conv_->weight, is_weight_loaded_);
    weight::load_weight(state_dict, "bias", conv_->bias, is_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(is_bias_loaded_) << "bias is not loaded for " << prefix + "bias";
  }

 private:
  int64_t spatial_padding_ = 0;
  int64_t temporal_padding_ = 0;
  bool reflect_spatial_padding_ = true;
  torch::nn::Conv3d conv_{nullptr};
  bool is_weight_loaded_ = false;
  bool is_bias_loaded_ = false;
};
TORCH_MODULE(CausalConv3d);

class IsolatedGroupNormImpl final : public torch::nn::Module {
 public:
  IsolatedGroupNormImpl(int64_t groups, int64_t channels, double eps) {
    norm_ = register_module(
        "norm",
        torch::nn::GroupNorm(torch::nn::GroupNormOptions(groups, channels)
                                 .eps(eps)
                                 .affine(true)));
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    const int64_t batch_size = hidden_states.size(0);
    const int64_t channels = hidden_states.size(1);
    const int64_t frames = hidden_states.size(2);
    const int64_t height = hidden_states.size(3);
    const int64_t width = hidden_states.size(4);
    hidden_states = hidden_states.permute({0, 2, 1, 3, 4}).contiguous();
    hidden_states =
        hidden_states.view({batch_size * frames, channels, 1, height, width});
    hidden_states = norm_->forward(hidden_states);
    hidden_states =
        hidden_states.view({batch_size, frames, channels, height, width});
    return hidden_states.permute({0, 2, 1, 3, 4}).contiguous();
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight", norm_->weight, is_weight_loaded_);
    weight::load_weight(state_dict, "bias", norm_->bias, is_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(is_bias_loaded_) << "bias is not loaded for " << prefix + "bias";
  }

 private:
  torch::nn::GroupNorm norm_{nullptr};
  bool is_weight_loaded_ = false;
  bool is_bias_loaded_ = false;
};
TORCH_MODULE(IsolatedGroupNorm);

class ResnetBlock3dImpl final : public torch::nn::Module {
 public:
  ResnetBlock3dImpl(int64_t in_channels,
                    int64_t out_channels,
                    int64_t groups,
                    double eps) {
    norm1_ =
        register_module("norm1", IsolatedGroupNorm(groups, in_channels, eps));
    conv1_ = register_module("conv1",
                             CausalConv3d(in_channels,
                                          out_channels,
                                          std::vector<int64_t>{3, 3, 3},
                                          std::vector<int64_t>{1, 1, 1},
                                          /*spatial_padding=*/1,
                                          /*temporal_padding=*/2,
                                          /*reflect_spatial_padding=*/true));
    norm2_ =
        register_module("norm2", IsolatedGroupNorm(groups, out_channels, eps));
    conv2_ = register_module("conv2",
                             CausalConv3d(out_channels,
                                          out_channels,
                                          std::vector<int64_t>{3, 3, 3},
                                          std::vector<int64_t>{1, 1, 1},
                                          /*spatial_padding=*/1,
                                          /*temporal_padding=*/2,
                                          /*reflect_spatial_padding=*/true));
    if (in_channels != out_channels) {
      has_shortcut_ = true;
      conv_shortcut_ =
          register_module("conv_shortcut",
                          CausalConv3d(in_channels,
                                       out_channels,
                                       std::vector<int64_t>{1, 1, 1},
                                       std::vector<int64_t>{1, 1, 1},
                                       /*spatial_padding=*/0,
                                       /*temporal_padding=*/0,
                                       /*reflect_spatial_padding=*/true));
    }
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    torch::Tensor residual = hidden_states;
    hidden_states =
        conv1_->forward(torch::silu(norm1_->forward(hidden_states)));
    hidden_states =
        conv2_->forward(torch::silu(norm2_->forward(hidden_states)));
    if (has_shortcut_) {
      residual = conv_shortcut_->forward(residual);
    }
    return residual + hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    conv1_->load_state_dict(state_dict.get_dict_with_prefix("conv1."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    conv2_->load_state_dict(state_dict.get_dict_with_prefix("conv2."));
    if (has_shortcut_) {
      conv_shortcut_->load_state_dict(
          state_dict.get_dict_with_prefix("conv_shortcut."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    norm1_->verify_loaded_weights(prefix + "norm1.");
    conv1_->verify_loaded_weights(prefix + "conv1.");
    norm2_->verify_loaded_weights(prefix + "norm2.");
    conv2_->verify_loaded_weights(prefix + "conv2.");
    if (has_shortcut_) {
      conv_shortcut_->verify_loaded_weights(prefix + "conv_shortcut.");
    }
  }

 private:
  IsolatedGroupNorm norm1_{nullptr};
  CausalConv3d conv1_{nullptr};
  IsolatedGroupNorm norm2_{nullptr};
  CausalConv3d conv2_{nullptr};
  bool has_shortcut_ = false;
  CausalConv3d conv_shortcut_{nullptr};
};
TORCH_MODULE(ResnetBlock3d);

class Downsample3dImpl final : public torch::nn::Module {
 public:
  Downsample3dImpl(int64_t channels,
                   int64_t temporal_stride,
                   int64_t spatial_stride)
      : spatial_stride_(spatial_stride) {
    conv_ = register_module(
        "conv",
        CausalConv3d(channels,
                     channels,
                     std::vector<int64_t>{3, 3, 3},
                     std::vector<int64_t>{
                         temporal_stride, spatial_stride, spatial_stride},
                     /*spatial_padding=*/0,
                     /*temporal_padding=*/2,
                     /*reflect_spatial_padding=*/true));
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    if (spatial_stride_ == 2) {
      hidden_states = torch::nn::functional::pad(
          hidden_states,
          torch::nn::functional::PadFuncOptions({0, 1, 0, 1, 0, 0})
              .mode(torch::kReflect));
    }
    return conv_->forward(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    conv_->load_state_dict(state_dict.get_dict_with_prefix("conv."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    conv_->verify_loaded_weights(prefix + "conv.");
  }

 private:
  int64_t spatial_stride_ = 1;
  CausalConv3d conv_{nullptr};
};
TORCH_MODULE(Downsample3d);

class DownBlock3dImpl final : public torch::nn::Module {
 public:
  DownBlock3dImpl(int64_t in_channels,
                  int64_t out_channels,
                  int64_t num_layers,
                  int64_t temporal_downsample_factor,
                  int64_t spatial_downsample_factor,
                  int64_t groups,
                  double eps) {
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    for (int64_t layer = 0; layer < num_layers; ++layer) {
      resnets_->push_back(ResnetBlock3d(
          layer == 0 ? in_channels : out_channels, out_channels, groups, eps));
    }
    if (temporal_downsample_factor * spatial_downsample_factor > 1) {
      has_downsamplers_ = true;
      downsamplers_ = register_module("downsamplers", torch::nn::ModuleList());
      downsamplers_->push_back(Downsample3d(
          out_channels, temporal_downsample_factor, spatial_downsample_factor));
    }
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    for (const auto& module : *resnets_) {
      hidden_states = module->as<ResnetBlock3d>()->forward(hidden_states);
    }
    if (has_downsamplers_) {
      for (const auto& module : *downsamplers_) {
        hidden_states = module->as<Downsample3d>()->forward(hidden_states);
      }
    }
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    dit::load_module_list<ResnetBlock3d>(*resnets_, state_dict, "resnets.");
    if (has_downsamplers_) {
      dit::load_module_list<Downsample3d>(
          *downsamplers_, state_dict, "downsamplers.");
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    dit::verify_module_list<ResnetBlock3d>(*resnets_, prefix + "resnets.");
    if (has_downsamplers_) {
      dit::verify_module_list<Downsample3d>(*downsamplers_,
                                            prefix + "downsamplers.");
    }
  }

 private:
  torch::nn::ModuleList resnets_{nullptr};
  bool has_downsamplers_ = false;
  torch::nn::ModuleList downsamplers_{nullptr};
};
TORCH_MODULE(DownBlock3d);

class Encoder3dImpl final : public torch::nn::Module {
 public:
  explicit Encoder3dImpl(const ModelArgs& args) {
    const std::vector<int64_t>& channels = args.block_out_channels();
    const std::vector<int64_t>& spatial = args.spatial_downsample_factors();
    const std::vector<int64_t>& temporal = args.temporal_downsample_factors();
    CHECK(!channels.empty())
        << "MiniMax-H3 video VAE block_out_channels must not be empty";
    CHECK_EQ(spatial.size(), channels.size());
    CHECK_EQ(temporal.size(), channels.size());
    CHECK_GT(args.in_channels(), 0);
    CHECK_GT(args.latent_channels(), 0);
    CHECK_GT(args.layers_per_block(), 0);
    CHECK_GT(args.norm_num_groups(), 0);
    const bool reflect_spatial_padding =
        args.spatial_padding_mode() == "reflect";
    CHECK(reflect_spatial_padding || args.spatial_padding_mode() == "constant")
        << "MiniMax-H3 video VAE supports reflect or constant spatial padding";
    conv_in_ = register_module("conv_in",
                               CausalConv3d(args.in_channels(),
                                            channels[0],
                                            std::vector<int64_t>{3, 3, 3},
                                            std::vector<int64_t>{1, 1, 1},
                                            /*spatial_padding=*/1,
                                            /*temporal_padding=*/2,
                                            reflect_spatial_padding));
    down_blocks_ = register_module("down_blocks", torch::nn::ModuleList());
    for (size_t index = 0; index < channels.size(); ++index) {
      const int64_t in_channels =
          index == 0 ? channels[0] : channels[index - 1];
      down_blocks_->push_back(DownBlock3d(in_channels,
                                          channels[index],
                                          args.layers_per_block(),
                                          temporal[index],
                                          spatial[index],
                                          args.norm_num_groups(),
                                          args.norm_eps()));
    }
    norm_out_ = register_module(
        "norm_out",
        IsolatedGroupNorm(
            args.norm_num_groups(), channels.back(), args.norm_eps()));
    conv_out_ = register_module("conv_out",
                                CausalConv3d(channels.back(),
                                             2 * args.latent_channels(),
                                             std::vector<int64_t>{3, 3, 3},
                                             std::vector<int64_t>{1, 1, 1},
                                             /*spatial_padding=*/1,
                                             /*temporal_padding=*/2,
                                             reflect_spatial_padding));
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    hidden_states = conv_in_->forward(hidden_states);
    for (const auto& module : *down_blocks_) {
      hidden_states = module->as<DownBlock3d>()->forward(hidden_states);
    }
    hidden_states = torch::silu(norm_out_->forward(hidden_states));
    return conv_out_->forward(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    conv_in_->load_state_dict(state_dict.get_dict_with_prefix("conv_in."));
    dit::load_module_list<DownBlock3d>(
        *down_blocks_, state_dict, "down_blocks.");
    norm_out_->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    conv_out_->load_state_dict(state_dict.get_dict_with_prefix("conv_out."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    conv_in_->verify_loaded_weights(prefix + "conv_in.");
    dit::verify_module_list<DownBlock3d>(*down_blocks_,
                                         prefix + "down_blocks.");
    norm_out_->verify_loaded_weights(prefix + "norm_out.");
    conv_out_->verify_loaded_weights(prefix + "conv_out.");
  }

 private:
  CausalConv3d conv_in_{nullptr};
  torch::nn::ModuleList down_blocks_{nullptr};
  IsolatedGroupNorm norm_out_{nullptr};
  CausalConv3d conv_out_{nullptr};
};
TORCH_MODULE(Encoder3d);

class RotaryPosEmbedImpl final : public torch::nn::Module {
 public:
  RotaryPosEmbedImpl(int64_t dim, double theta) : dim_(dim), theta_(theta) {
    CHECK_EQ(dim_ % 6, 0) << "MiniMax-H3 video RoPE dim must be divisible by 6";
    CHECK_GT(dim_, 0);
    CHECK_GT(theta_, 0.0);
    torch::Tensor exponent = torch::arange(
        0, dim_, 6, torch::TensorOptions().dtype(torch::kFloat32));
    exponent = exponent / static_cast<double>(dim_);
    inv_freq_ = register_buffer(
        "inv_freq", torch::pow(torch::full_like(exponent, theta_), -exponent));
  }

  std::pair<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& position_ids) const {
    torch::Tensor angles =
        2.0 * std::numbers::pi * position_ids.unsqueeze(-1) *
        inv_freq_.to(position_ids.device()).view({1, 1, 1, -1});
    angles = angles.flatten(/*start_dim=*/2, /*end_dim=*/3);
    angles = torch::cat({angles, angles}, -1).unsqueeze(2);
    return {torch::cos(angles), torch::sin(angles)};
  }

 private:
  int64_t dim_;
  double theta_;
  torch::Tensor inv_freq_;
};
TORCH_MODULE(RotaryPosEmbed);

class AttentionImpl final : public torch::nn::Module {
 public:
  AttentionImpl(int64_t hidden_size,
                int64_t heads,
                int64_t head_dim,
                double norm_eps)
      : heads_(heads), head_dim_(head_dim), norm_eps_(norm_eps) {
    const int64_t inner_dim = heads_ * head_dim_;
    to_q_ = register_module(
        "to_q",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, inner_dim).bias(true)));
    to_k_ = register_module(
        "to_k",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, inner_dim).bias(true)));
    to_v_ = register_module(
        "to_v",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, inner_dim).bias(true)));
    to_out_ = register_module(
        "to_out",
        torch::nn::Linear(
            torch::nn::LinearOptions(inner_dim, hidden_size).bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& cos,
                        const torch::Tensor& sin) {
    const int64_t batch_size = hidden_states.size(0);
    const int64_t sequence_length = hidden_states.size(1);
    torch::Tensor query =
        to_q_->forward(hidden_states)
            .view({batch_size, sequence_length, heads_, head_dim_});
    torch::Tensor key =
        to_k_->forward(hidden_states)
            .view({batch_size, sequence_length, heads_, head_dim_});
    torch::Tensor value =
        to_v_->forward(hidden_states)
            .view({batch_size, sequence_length, heads_, head_dim_});
    query = dit::rms_norm(query, torch::Tensor(), norm_eps_);
    key = dit::rms_norm(key, torch::Tensor(), norm_eps_);
    query = dit::apply_rotary(query, cos, sin);
    key = dit::apply_rotary(key, cos, sin);
    query = query.permute({0, 2, 1, 3}).contiguous();
    key = key.permute({0, 2, 1, 3}).contiguous();
    value = value.permute({0, 2, 1, 3}).contiguous();
#if defined(USE_NPU)
    const auto results = at_npu::native::custom_ops::npu_fusion_attention(
        query,
        key,
        value,
        heads_,
        "BNSD",
        torch::nullopt,
        torch::nullopt,
        torch::nullopt,
        std::pow(head_dim_, -0.5),
        1.0,
        65535,
        65535);
    torch::Tensor output = std::get<0>(results);
#else
    torch::Tensor output = torch::scaled_dot_product_attention(
        query, key, value, std::nullopt, 0.0, false);
#endif
    output = output.permute({0, 2, 1, 3}).contiguous();
    output = output.view({batch_size, sequence_length, heads_ * head_dim_});
    output = to_out_->forward(output);
    return output;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(
        state_dict, "to_q.weight", to_q_->weight, is_to_q_weight_loaded_);
    weight::load_weight(
        state_dict, "to_q.bias", to_q_->bias, is_to_q_bias_loaded_);
    weight::load_weight(
        state_dict, "to_k.weight", to_k_->weight, is_to_k_weight_loaded_);
    weight::load_weight(
        state_dict, "to_k.bias", to_k_->bias, is_to_k_bias_loaded_);
    weight::load_weight(
        state_dict, "to_v.weight", to_v_->weight, is_to_v_weight_loaded_);
    weight::load_weight(
        state_dict, "to_v.bias", to_v_->bias, is_to_v_bias_loaded_);
    weight::load_weight(state_dict,
                        "to_out.0.weight",
                        to_out_->weight,
                        is_to_out_weight_loaded_);
    weight::load_weight(
        state_dict, "to_out.0.bias", to_out_->bias, is_to_out_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_to_q_weight_loaded_)
        << "weight is not loaded for " << prefix + "to_q.weight";
    CHECK(is_to_q_bias_loaded_)
        << "bias is not loaded for " << prefix + "to_q.bias";
    CHECK(is_to_k_weight_loaded_)
        << "weight is not loaded for " << prefix + "to_k.weight";
    CHECK(is_to_k_bias_loaded_)
        << "bias is not loaded for " << prefix + "to_k.bias";
    CHECK(is_to_v_weight_loaded_)
        << "weight is not loaded for " << prefix + "to_v.weight";
    CHECK(is_to_v_bias_loaded_)
        << "bias is not loaded for " << prefix + "to_v.bias";
    CHECK(is_to_out_weight_loaded_)
        << "weight is not loaded for " << prefix + "to_out.0.weight";
    CHECK(is_to_out_bias_loaded_)
        << "bias is not loaded for " << prefix + "to_out.0.bias";
  }

 private:
  int64_t heads_;
  int64_t head_dim_;
  double norm_eps_;
  torch::nn::Linear to_q_{nullptr};
  torch::nn::Linear to_k_{nullptr};
  torch::nn::Linear to_v_{nullptr};
  torch::nn::Linear to_out_{nullptr};
  bool is_to_q_weight_loaded_ = false;
  bool is_to_q_bias_loaded_ = false;
  bool is_to_k_weight_loaded_ = false;
  bool is_to_k_bias_loaded_ = false;
  bool is_to_v_weight_loaded_ = false;
  bool is_to_v_bias_loaded_ = false;
  bool is_to_out_weight_loaded_ = false;
  bool is_to_out_bias_loaded_ = false;
};
TORCH_MODULE(Attention);

class FeedForwardImpl final : public torch::nn::Module {
 public:
  FeedForwardImpl(int64_t hidden_size, int64_t ffn_dim) {
    proj_in_ = register_module(
        "proj_in",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, 2 * ffn_dim).bias(true)));
    proj_out_ = register_module(
        "proj_out",
        torch::nn::Linear(
            torch::nn::LinearOptions(ffn_dim, hidden_size).bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor fused = proj_in_->forward(hidden_states);
    std::vector<torch::Tensor> chunks = fused.chunk(2, -1);
    torch::Tensor activated = chunks[0] * torch::silu(chunks[1]);
    return proj_out_->forward(activated);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict,
                        "net.0.proj.weight",
                        proj_in_->weight,
                        is_proj_in_weight_loaded_);
    weight::load_weight(
        state_dict, "net.0.proj.bias", proj_in_->bias, is_proj_in_bias_loaded_);
    weight::load_weight(state_dict,
                        "net.2.weight",
                        proj_out_->weight,
                        is_proj_out_weight_loaded_);
    weight::load_weight(
        state_dict, "net.2.bias", proj_out_->bias, is_proj_out_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_proj_in_weight_loaded_)
        << "weight is not loaded for " << prefix + "net.0.proj.weight";
    CHECK(is_proj_in_bias_loaded_)
        << "bias is not loaded for " << prefix + "net.0.proj.bias";
    CHECK(is_proj_out_weight_loaded_)
        << "weight is not loaded for " << prefix + "net.2.weight";
    CHECK(is_proj_out_bias_loaded_)
        << "bias is not loaded for " << prefix + "net.2.bias";
  }

 private:
  torch::nn::Linear proj_in_{nullptr};
  torch::nn::Linear proj_out_{nullptr};
  bool is_proj_in_weight_loaded_ = false;
  bool is_proj_in_bias_loaded_ = false;
  bool is_proj_out_weight_loaded_ = false;
  bool is_proj_out_bias_loaded_ = false;
};
TORCH_MODULE(FeedForward);

class TransformerBlockImpl final : public torch::nn::Module {
 public:
  TransformerBlockImpl(int64_t hidden_size,
                       int64_t heads,
                       int64_t head_dim,
                       int64_t ffn_dim,
                       double norm_eps)
      : norm_eps_(norm_eps) {
    norm1_weight_ =
        register_parameter("norm1_weight", torch::ones({hidden_size}));
    attention_ = register_module(
        "attn", Attention(hidden_size, heads, head_dim, norm_eps));
    scale1_ = register_parameter("scale1", torch::zeros({hidden_size}));
    norm2_weight_ =
        register_parameter("norm2_weight", torch::ones({hidden_size}));
    feed_forward_ = register_module("ff", FeedForward(hidden_size, ffn_dim));
    scale2_ = register_parameter("scale2", torch::zeros({hidden_size}));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& cos,
                        const torch::Tensor& sin) {
    torch::Tensor normalized =
        dit::rms_norm(hidden_states, norm1_weight_, norm_eps_);
    torch::Tensor output =
        hidden_states + attention_->forward(normalized, cos, sin) * scale1_;
    normalized = dit::rms_norm(output, norm2_weight_, norm_eps_);
    output = output + feed_forward_->forward(normalized) * scale2_;
    return output;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict.get_dict_with_prefix("norm1."),
                        "weight",
                        norm1_weight_,
                        is_norm1_weight_loaded_);
    attention_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    weight::load_weight(state_dict, "scale1", scale1_, is_scale1_loaded_);
    weight::load_weight(state_dict.get_dict_with_prefix("norm2."),
                        "weight",
                        norm2_weight_,
                        is_norm2_weight_loaded_);
    feed_forward_->load_state_dict(state_dict.get_dict_with_prefix("ff."));
    weight::load_weight(state_dict, "scale2", scale2_, is_scale2_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_norm1_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm1.weight";
    attention_->verify_loaded_weights(prefix + "attn.");
    CHECK(is_scale1_loaded_) << "scale is not loaded for " << prefix + "scale1";
    CHECK(is_norm2_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm2.weight";
    feed_forward_->verify_loaded_weights(prefix + "ff.");
    CHECK(is_scale2_loaded_) << "scale is not loaded for " << prefix + "scale2";
  }

 private:
  double norm_eps_;
  torch::Tensor norm1_weight_;
  Attention attention_{nullptr};
  torch::Tensor scale1_;
  torch::Tensor norm2_weight_;
  FeedForward feed_forward_{nullptr};
  torch::Tensor scale2_;
  bool is_norm1_weight_loaded_ = false;
  bool is_scale1_loaded_ = false;
  bool is_norm2_weight_loaded_ = false;
  bool is_scale2_loaded_ = false;
};
TORCH_MODULE(TransformerBlock);

class ViTDecoder3dImpl final : public torch::nn::Module {
 public:
  explicit ViTDecoder3dImpl(const ModelArgs& args) {
    latent_channels_ = args.latent_channels();
    output_channels_ = args.out_channels();
    patch_t_ = product_checked(args.temporal_downsample_factors());
    patch_h_ = product_checked(args.spatial_downsample_factors());
    patch_w_ = patch_h_;
    const int64_t layers = args.decoder_num_blocks();
    const int64_t heads = args.decoder_n_heads();
    const int64_t head_dim = args.decoder_head_dim();
    const int64_t hidden_size = heads * head_dim;
    const int64_t ffn_dim = hidden_size * args.decoder_ffn_mult();
    register_tokens_count_ = args.decoder_num_register_tokens();
    const int64_t rope_dim = static_cast<int64_t>(
        std::llround(head_dim * args.decoder_rope_dim_ratio()));
    const double rope_theta = args.decoder_rope_theta();
    const double norm_eps = args.decoder_norm_eps();
    CHECK_GT(latent_channels_, 0);
    CHECK_GT(output_channels_, 0);
    CHECK_GT(patch_t_, 0);
    CHECK_GT(patch_h_, 0);
    CHECK_GT(patch_w_, 0);
    CHECK_GT(layers, 0);
    CHECK_GT(heads, 0);
    CHECK_GT(head_dim, 0);
    CHECK_GT(hidden_size, 0);
    CHECK_GT(ffn_dim, 0);
    CHECK_GT(register_tokens_count_, 0);
    CHECK_GT(rope_dim, 0);
    rope_ = register_module("rope", RotaryPosEmbed(rope_dim, rope_theta));
    proj_in_ =
        register_module("proj_in",
                        torch::nn::Linear(torch::nn::LinearOptions(
                                              latent_channels_, hidden_size)
                                              .bias(true)));
    register_tokens_ = register_parameter(
        "register_tokens",
        torch::zeros({1, register_tokens_count_, hidden_size}));
    blocks_ = register_module("transformer_blocks", torch::nn::ModuleList());
    for (int64_t layer = 0; layer < layers; ++layer) {
      blocks_->push_back(
          TransformerBlock(hidden_size, heads, head_dim, ffn_dim, norm_eps));
    }
    norm_out_ = register_module(
        "norm_out",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden_size}).eps(norm_eps)));
    proj_out_ = register_module(
        "proj_out",
        torch::nn::Linear(
            torch::nn::LinearOptions(
                hidden_size, output_channels_ * patch_t_ * patch_h_ * patch_w_)
                .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& latent) {
    const int64_t batch_size = latent.size(0);
    const int64_t num_frames = latent.size(2);
    const int64_t height = latent.size(3);
    const int64_t width = latent.size(4);
    torch::Tensor hidden_states =
        latent.permute({0, 2, 3, 4, 1})
            .reshape(
                {batch_size, num_frames * height * width, latent_channels_});
    hidden_states = proj_in_->forward(hidden_states);
    const int64_t num_patches = hidden_states.size(1);
    torch::Tensor register_tokens =
        register_tokens_.expand({batch_size, -1, -1});
    torch::Tensor class_token = torch::zeros_like(hidden_states.slice(1, 0, 1));
    hidden_states =
        torch::cat({hidden_states, register_tokens, class_token}, 1);

    torch::TensorOptions position_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(latent.device());
    const auto center_grid = [&position_options](int64_t extent) {
      return 2.0 *
                 (torch::arange(
                      0.5, static_cast<double>(extent), 1.0, position_options) /
                  static_cast<double>(extent)) -
             1.0;
    };
    torch::Tensor grid_t = center_grid(num_frames);
    torch::Tensor grid_h = center_grid(height);
    torch::Tensor grid_w = center_grid(width);
    torch::Tensor position_ids = torch::stack(
        {grid_t.view({num_frames, 1, 1}).expand({num_frames, height, width}),
         grid_h.view({1, height, 1}).expand({num_frames, height, width}),
         grid_w.view({1, 1, width}).expand({num_frames, height, width})},
        -1);
    position_ids =
        position_ids.reshape({1, num_patches, 3}).expand({batch_size, -1, -1});
    torch::Tensor suffix = torch::zeros(
        {batch_size, register_tokens_count_ + 1, 3}, position_options);
    position_ids = torch::cat({position_ids, suffix}, 1);
    std::pair<torch::Tensor, torch::Tensor> rotary =
        rope_->forward(position_ids);
    for (size_t layer = 0; layer < blocks_->size(); ++layer) {
      hidden_states = blocks_[layer]->as<TransformerBlock>()->forward(
          hidden_states, rotary.first, rotary.second);
    }
    hidden_states = norm_out_->forward(hidden_states);
    hidden_states = proj_out_->forward(hidden_states);
    hidden_states = hidden_states.slice(1, 0, num_patches);
    hidden_states = hidden_states.view({batch_size,
                                        num_frames,
                                        height,
                                        width,
                                        output_channels_,
                                        patch_t_,
                                        patch_h_,
                                        patch_w_});
    hidden_states =
        hidden_states.permute({0, 4, 1, 5, 2, 6, 3, 7}).contiguous();
    hidden_states = hidden_states.reshape({batch_size,
                                           output_channels_,
                                           num_frames * patch_t_,
                                           height * patch_h_,
                                           width * patch_w_});
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict,
                        "proj_in.weight",
                        proj_in_->weight,
                        is_proj_in_weight_loaded_);
    weight::load_weight(
        state_dict, "proj_in.bias", proj_in_->bias, is_proj_in_bias_loaded_);
    weight::load_weight(state_dict,
                        "register_tokens",
                        register_tokens_,
                        is_register_tokens_loaded_);
    dit::load_module_list<TransformerBlock>(
        *blocks_, state_dict, "transformer_blocks.");
    weight::load_weight(state_dict,
                        "norm_out.weight",
                        norm_out_->weight,
                        is_norm_out_weight_loaded_);
    weight::load_weight(
        state_dict, "norm_out.bias", norm_out_->bias, is_norm_out_bias_loaded_);
    weight::load_weight(state_dict,
                        "proj_out.weight",
                        proj_out_->weight,
                        is_proj_out_weight_loaded_);
    weight::load_weight(
        state_dict, "proj_out.bias", proj_out_->bias, is_proj_out_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_proj_in_weight_loaded_)
        << "weight is not loaded for " << prefix + "proj_in.weight";
    CHECK(is_proj_in_bias_loaded_)
        << "bias is not loaded for " << prefix + "proj_in.bias";
    CHECK(is_register_tokens_loaded_)
        << "parameter is not loaded for " << prefix + "register_tokens";
    dit::verify_module_list<TransformerBlock>(*blocks_,
                                              prefix + "transformer_blocks.");
    CHECK(is_norm_out_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm_out.weight";
    CHECK(is_norm_out_bias_loaded_)
        << "bias is not loaded for " << prefix + "norm_out.bias";
    CHECK(is_proj_out_weight_loaded_)
        << "weight is not loaded for " << prefix + "proj_out.weight";
    CHECK(is_proj_out_bias_loaded_)
        << "bias is not loaded for " << prefix + "proj_out.bias";
  }

 private:
  int64_t latent_channels_ = 0;
  int64_t output_channels_ = 0;
  int64_t patch_t_ = 0;
  int64_t patch_h_ = 0;
  int64_t patch_w_ = 0;
  int64_t register_tokens_count_ = 0;
  RotaryPosEmbed rope_{nullptr};
  torch::nn::Linear proj_in_{nullptr};
  torch::Tensor register_tokens_;
  torch::nn::ModuleList blocks_{nullptr};
  torch::nn::LayerNorm norm_out_{nullptr};
  torch::nn::Linear proj_out_{nullptr};
  bool is_proj_in_weight_loaded_ = false;
  bool is_proj_in_bias_loaded_ = false;
  bool is_register_tokens_loaded_ = false;
  bool is_norm_out_weight_loaded_ = false;
  bool is_norm_out_bias_loaded_ = false;
  bool is_proj_out_weight_loaded_ = false;
  bool is_proj_out_bias_loaded_ = false;
};
TORCH_MODULE(ViTDecoder3d);

}  // namespace minimax_h3_video

class AutoencoderKLMiniMaxH3Impl final : public torch::nn::Module {
 public:
  explicit AutoencoderKLMiniMaxH3Impl(const ModelContext& context)
      : vae_group_(context.get_parallel_args().dit_vae_group_) {
    const ModelArgs& args = context.get_model_args();
    const int32_t vae_size = context.get_parallel_args().vae_size();
    CHECK_GT(vae_size, 0);
    CHECK(vae_size == 1 || vae_group_ != nullptr)
        << "MiniMax-H3 video VAE requires a group when vae_size > 1";
    if (vae_group_ != nullptr) {
      CHECK_EQ(vae_size, vae_group_->world_size());
    }
    latent_channels_ = args.latent_channels();
    temporal_ratio_ = dit::product_checked(args.temporal_downsample_factors());
    spatial_ratio_ = dit::product_checked(args.spatial_downsample_factors());
    clip_length_ = args.clip_length();
    token_drop_ = args.token_drop();
    CHECK_GT(latent_channels_, 0);
    CHECK_GT(temporal_ratio_, 0);
    CHECK_GT(spatial_ratio_, 0);
    CHECK_GT(clip_length_, 0);
    CHECK_GE(token_drop_, 0);
    CHECK_EQ(args.latents_mean().size(), static_cast<size_t>(latent_channels_));
    CHECK_EQ(args.latents_std().size(), static_cast<size_t>(latent_channels_));
    const auto stats_options = torch::TensorOptions().dtype(torch::kFloat32);
    latents_mean_tensor_ =
        register_buffer("latents_mean",
                        torch::tensor(args.latents_mean(), stats_options)
                            .view({1, latent_channels_, 1, 1, 1}));
    latents_std_tensor_ =
        register_buffer("latents_std",
                        torch::tensor(args.latents_std(), stats_options)
                            .view({1, latent_channels_, 1, 1, 1}));
    pixel_mean_tensor_ = register_buffer(
        "pixel_mean",
        torch::tensor(std::vector<float>{0.485f, 0.456f, 0.406f}, stats_options)
            .view({1, 3, 1, 1, 1}));
    pixel_std_tensor_ = register_buffer(
        "pixel_std",
        torch::tensor(std::vector<float>{0.229f, 0.224f, 0.225f}, stats_options)
            .view({1, 3, 1, 1, 1}));
    encoder_ = register_module("encoder", minimax_h3_video::Encoder3d(args));
    quant_conv_ =
        register_module("quant_conv",
                        minimax_h3_video::Conv3d1x1(2 * latent_channels_,
                                                    2 * latent_channels_));
    post_quant_conv_ = register_module(
        "post_quant_conv",
        minimax_h3_video::Conv3d1x1(latent_channels_, latent_channels_));
    decoder_ = register_module("decoder", minimax_h3_video::ViTDecoder3d(args));
  }

  torch::Tensor encode_keyframe_condition(const torch::Tensor& pixels) {
    CHECK_EQ(pixels.size(2), 1)
        << "MiniMax-H3 keyframe VAE expects exactly one frame";
    return encode_reference_condition(pixels);
  }

  torch::Tensor encode_reference_condition(const torch::Tensor& pixels) {
    CHECK_EQ(pixels.dim(), 5)
        << "MiniMax-H3 reference VAE expects [B,3,T,H,W] pixels";
    CHECK_EQ(pixels.size(0), 1);
    CHECK_EQ(pixels.size(1), 3);
    CHECK_GT(pixels.size(2), 0);
    torch::Tensor pixel_mean = pixel_mean_tensor_.to(pixels.device());
    torch::Tensor pixel_std = pixel_std_tensor_.to(pixels.device());
    torch::Tensor normalized_pixels =
        (pixels.to(torch::kFloat32) / 255.0 - pixel_mean) / pixel_std;
    torch::Tensor moments = encode_moments(normalized_pixels);
    dit::DiagonalGaussianDistribution posterior(moments);
    torch::Generator generator =
        torch::make_generator<torch::CPUGeneratorImpl>();
    generator.set_current_seed(keyframe_encode_seed_);
    torch::Tensor latent = posterior.sample(generator, torch::kFloat32);
    latent = latent.to(torch::kFloat16).to(torch::kFloat32);
    return dit::normalize_latents(
        latent, latents_mean_tensor_, latents_std_tensor_);
  }

  // Maps normalized latents back to the distribution decode() expects.
  torch::Tensor denormalize_latents(const torch::Tensor& latent) {
    check_latent(latent);
    return dit::denormalize_latents(
        latent, latents_mean_tensor_, latents_std_tensor_);
  }

  torch::Tensor decode(const torch::Tensor& latent) {
    check_latent(latent);
    const int64_t num_tokens = latent.size(2) + token_drop_;
    const int64_t pad_tokens =
        (tokens_chunk_size_ - num_tokens % tokens_chunk_size_) %
        tokens_chunk_size_;
    const int64_t num_chunks = (num_tokens + pad_tokens) / tokens_chunk_size_;
    torch::Tensor padded = pad_by_repeating_last_frame(latent, pad_tokens);

    std::vector<torch::Tensor> decoded_chunks;
    decoded_chunks.reserve(static_cast<size_t>(num_chunks + 1));
    torch::Tensor overlap;
    for (int64_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
      const int64_t start = chunk_index * tokens_chunk_size_;
      torch::Tensor clip = decode_clip(
          padded.slice(2, start, start + tokens_chunk_size_ + token_overlap_));
      for (int64_t section = 0; section < 2; ++section) {
        const int64_t frame_start = section * chunk_num_frames_;
        torch::Tensor section_frames =
            clip.slice(2, frame_start, frame_start + chunk_num_frames_);
        torch::Tensor chunk = section_frames.slice(2, frame_pre_padding_);
        if (section == 0) {
          if (overlap.defined()) {
            chunk = blend(overlap, chunk, frame_overlap_, 2);
          }
          decoded_chunks.emplace_back(chunk);
        } else {
          overlap = chunk;
        }
      }
    }
    if (overlap.defined()) {
      decoded_chunks.emplace_back(overlap);
    }
    torch::Tensor decoded = torch::cat(decoded_chunks, 2);
    if (pad_tokens > 0) {
      int64_t pad_frames = 0;
      const int64_t num_tokens_before_pad = padded.size(2) - pad_tokens;
      for (int64_t index = 0; index < pad_tokens; ++index) {
        pad_frames +=
            ((num_tokens_before_pad + index) % tokens_chunk_size_ == 0)
                ? clip_length_ % temporal_ratio_
                : temporal_ratio_;
      }
      decoded = decoded.slice(2, 0, decoded.size(2) - pad_frames);
    }
    return decoded;
  }

  // Decodes latents to BTCHW video in [0, 1].
  torch::Tensor decode_to_pixels(const torch::Tensor& latent) {
    torch::Tensor decoded = decode(latent);
    torch::Tensor pixel_mean = pixel_mean_tensor_.to(decoded.device());
    torch::Tensor pixel_std = pixel_std_tensor_.to(decoded.device());
    return torch::clamp(
               decoded.to(torch::kFloat32) * pixel_std + pixel_mean, 0.0, 1.0)
        .permute({0, 2, 1, 3, 4})
        .contiguous();
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    CHECK(loader != nullptr) << "MiniMax-H3 video VAE loader must not be null";
    load_config(loader->model_weights_path() + "/config.json");
    for (const auto& state_dict : loader->get_state_dicts()) {
      encoder_->load_state_dict(state_dict->get_dict_with_prefix("encoder."));
      quant_conv_->load_state_dict(
          state_dict->get_dict_with_prefix("quant_conv."));
      post_quant_conv_->load_state_dict(
          state_dict->get_dict_with_prefix("post_quant_conv."));
      decoder_->load_state_dict(state_dict->get_dict_with_prefix("decoder."));
    }
    verify_loaded_weights("");
  }

  void verify_loaded_weights(const std::string& prefix) {
    encoder_->verify_loaded_weights(prefix + "encoder.");
    quant_conv_->verify_loaded_weights(prefix + "quant_conv.");
    post_quant_conv_->verify_loaded_weights(prefix + "post_quant_conv.");
    decoder_->verify_loaded_weights(prefix + "decoder.");
  }

 private:
  struct TileLayout {
    std::vector<int64_t> starts;
    std::vector<int64_t> lengths;
    std::vector<int64_t> overlaps;
  };

  void check_latent(const torch::Tensor& latent) const {
    CHECK_EQ(latent.dim(), 5)
        << "MiniMax-H3 video VAE latents must be [B,C,F,H,W]";
    CHECK_EQ(latent.size(1), latent_channels_)
        << "MiniMax-H3 video VAE latent channel mismatch";
  }

  torch::Tensor encode_moments(const torch::Tensor& pixels) {
    if (pixels.size(2) == 1) {
      return encode_clip(pixels);
    }
    torch::Tensor padded = pixels;
    const int64_t pad_frames =
        (clip_length_ - pixels.size(2) % clip_length_) % clip_length_;
    padded = pad_by_repeating_last_frame(padded, pad_frames);
    std::vector<torch::Tensor> clips;
    const int64_t num_clips = padded.size(2) / clip_length_;
    clips.reserve(static_cast<size_t>(num_clips));
    for (int64_t index = 0; index < num_clips; ++index) {
      clips.emplace_back(encode_clip(
          padded.slice(2, index * clip_length_, (index + 1) * clip_length_)));
    }
    torch::Tensor moments = torch::cat(clips, 2);
    if (token_drop_ > 0) {
      moments = moments.slice(2, 0, moments.size(2) - token_drop_);
    }
    return moments;
  }

  torch::Tensor encode_clip(const torch::Tensor& pixels) {
    const int64_t height = pixels.size(3);
    const int64_t width = pixels.size(4);
    if (height <= tile_sample_height_ && width <= tile_sample_width_) {
      return quant_conv_->forward(encoder_->forward(pixels));
    }
    TileLayout height_layout =
        split_tiles(height, tile_sample_height_, tile_sample_overlap_height_);
    TileLayout width_layout =
        split_tiles(width, tile_sample_width_, tile_sample_overlap_width_);
    std::vector<std::vector<torch::Tensor>> tiles =
        run_tiles(height_layout, width_layout, [&](size_t row, size_t column) {
          torch::Tensor tile =
              pixels
                  .slice(3,
                         height_layout.starts[row],
                         height_layout.starts[row] + height_layout.lengths[row])
                  .slice(4,
                         width_layout.starts[column],
                         width_layout.starts[column] +
                             width_layout.lengths[column]);
          return quant_conv_->forward(encoder_->forward(tile));
        });
    const auto to_latent_overlaps =
        [this](const std::vector<int64_t>& overlaps) {
          std::vector<int64_t> latent_overlaps;
          latent_overlaps.reserve(overlaps.size());
          for (int64_t overlap : overlaps) {
            latent_overlaps.emplace_back(overlap / spatial_ratio_);
          }
          return latent_overlaps;
        };
    return stitch_tiles(tiles,
                        to_latent_overlaps(height_layout.overlaps),
                        to_latent_overlaps(width_layout.overlaps));
  }

  torch::Tensor decode_clip(const torch::Tensor& latent) {
    const int64_t sample_height = latent.size(3) * spatial_ratio_;
    const int64_t sample_width = latent.size(4) * spatial_ratio_;
    if (sample_height <= tile_sample_height_ &&
        sample_width <= tile_sample_width_) {
      torch::Tensor post_quant = post_quant_conv_->forward(latent);
      return decoder_->forward(post_quant);
    }
    TileLayout height_layout = split_tiles(
        sample_height, tile_sample_height_, tile_sample_overlap_height_);
    TileLayout width_layout = split_tiles(
        sample_width, tile_sample_width_, tile_sample_overlap_width_);
    std::vector<std::vector<torch::Tensor>> tiles =
        run_tiles(height_layout, width_layout, [&](size_t row, size_t column) {
          const int64_t y_start = height_layout.starts[row] / spatial_ratio_;
          const int64_t y_length = height_layout.lengths[row] / spatial_ratio_;
          const int64_t x_start = width_layout.starts[column] / spatial_ratio_;
          const int64_t x_length =
              width_layout.lengths[column] / spatial_ratio_;
          torch::Tensor tile = latent.slice(3, y_start, y_start + y_length)
                                   .slice(4, x_start, x_start + x_length);
          torch::Tensor post_quant = post_quant_conv_->forward(tile);
          return decoder_->forward(post_quant);
        });
    torch::Tensor decoded =
        stitch_tiles(tiles, height_layout.overlaps, width_layout.overlaps);
    return decoded;
  }

  template <typename TileFunction>
  std::vector<std::vector<torch::Tensor>> run_tiles(
      const TileLayout& height_layout,
      const TileLayout& width_layout,
      const TileFunction& tile_fn) const {
    const size_t column_count = width_layout.starts.size();
    const size_t tile_count = height_layout.starts.size() * column_count;
    std::vector<std::vector<torch::Tensor>> tiles(height_layout.starts.size());
    for (std::vector<torch::Tensor>& row : tiles) {
      row.resize(column_count);
    }

    if (vae_group_ == nullptr || vae_group_->world_size() == 1 ||
        tile_count < static_cast<size_t>(vae_group_->world_size())) {
      for (size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        const size_t row = tile_index / column_count;
        const size_t column = tile_index % column_count;
        tiles[row][column] = tile_fn(row, column);
      }
      return tiles;
    }

    const size_t world_size = static_cast<size_t>(vae_group_->world_size());
    const size_t rank = static_cast<size_t>(vae_group_->rank());
    const size_t max_local_count = (tile_count + world_size - 1) / world_size;
    std::vector<torch::Tensor> local_tiles;
    local_tiles.reserve(max_local_count);
    for (size_t tile_index = rank; tile_index < tile_count;
         tile_index += world_size) {
      local_tiles.emplace_back(
          tile_fn(tile_index / column_count, tile_index % column_count));
    }

    // Tile dimensions are equal; pad only the local tile count for allgather.
    torch::Tensor local_stack =
        torch::stack(local_tiles, /*dim=*/0).contiguous();
    if (local_tiles.size() < max_local_count) {
      std::vector<int64_t> padding_shape = local_stack.sizes().vec();
      padding_shape[0] =
          static_cast<int64_t>(max_local_count - local_tiles.size());
      local_stack =
          torch::cat(
              {local_stack, torch::zeros(padding_shape, local_stack.options())},
              /*dim=*/0)
              .contiguous();
    }
    torch::Tensor gathered = vae_group_->allgather_base_sync(local_stack);
    for (size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
      const size_t row = tile_index / column_count;
      const size_t column = tile_index % column_count;
      tiles[row][column] =
          gathered[static_cast<int64_t>(tile_index % world_size)]
                  [static_cast<int64_t>(tile_index / world_size)];
    }
    return tiles;
  }

  TileLayout split_tiles(int64_t length,
                         int64_t tile_size,
                         int64_t minimum_overlap) {
    if (tile_size >= length) {
      return {{0}, {length}, {}};
    }
    int64_t tile_count = (length + tile_size - 1) / tile_size;
    while (tile_size * tile_count - minimum_overlap * (tile_count - 1) -
               length <
           0) {
      ++tile_count;
    }
    std::vector<int64_t> overlaps(static_cast<size_t>(tile_count - 1),
                                  minimum_overlap);
    const int64_t remaining =
        tile_size * tile_count - minimum_overlap * (tile_count - 1) - length;
    for (int64_t unit = 0; unit < remaining / spatial_ratio_; ++unit) {
      overlaps[static_cast<size_t>(unit % (tile_count - 1))] += spatial_ratio_;
    }
    std::vector<int64_t> starts;
    starts.reserve(static_cast<size_t>(tile_count));
    starts.emplace_back(0);
    for (int64_t tile = 0; tile < tile_count - 1; ++tile) {
      starts.emplace_back(starts.back() + tile_size -
                          overlaps[static_cast<size_t>(tile)]);
    }
    return {std::move(starts),
            std::vector<int64_t>(static_cast<size_t>(tile_count), tile_size),
            std::move(overlaps)};
  }

  static torch::Tensor stitch_tiles(
      std::vector<std::vector<torch::Tensor>>& tiles,
      const std::vector<int64_t>& height_overlaps,
      const std::vector<int64_t>& width_overlaps) {
    std::vector<torch::Tensor> result_rows;
    result_rows.reserve(tiles.size());
    for (size_t row = 0; row < tiles.size(); ++row) {
      std::vector<torch::Tensor> result_row;
      result_row.reserve(tiles[row].size());
      for (size_t column = 0; column < tiles[row].size(); ++column) {
        torch::Tensor tile = tiles[row][column];
        if (row > 0) {
          tile = blend(tiles[row - 1][column],
                       tile,
                       height_overlaps[row - 1],
                       /*dimension=*/3);
        }
        if (column > 0) {
          tile = blend(tiles[row][column - 1],
                       tile,
                       width_overlaps[column - 1],
                       /*dimension=*/4);
        }
        if (row + 1 < tiles.size()) {
          tile = tile.slice(3, 0, tile.size(3) - height_overlaps[row]);
        }
        if (column + 1 < tiles[row].size()) {
          tile = tile.slice(4, 0, tile.size(4) - width_overlaps[column]);
        }
        result_row.emplace_back(tile);
      }
      result_rows.emplace_back(torch::cat(result_row, 4));
    }
    return torch::cat(result_rows, 3);
  }

  static torch::Tensor blend(const torch::Tensor& left,
                             const torch::Tensor& right,
                             int64_t extent,
                             int64_t dimension) {
    const int64_t blend_extent =
        std::min({left.size(dimension), right.size(dimension), extent});
    torch::Tensor positions = torch::arange(blend_extent, right.options());
    std::vector<int64_t> shape(static_cast<size_t>(right.dim()), 1);
    shape[static_cast<size_t>(dimension)] = blend_extent;
    torch::Tensor left_weight =
        (1.0 - positions / static_cast<double>(blend_extent)).view(shape);
    torch::Tensor right_weight =
        (positions / static_cast<double>(blend_extent)).view(shape);
    torch::Tensor left_tail =
        left.slice(dimension, left.size(dimension) - blend_extent);
    torch::Tensor right_head = right.slice(dimension, 0, blend_extent);
    torch::Tensor blended = left_tail * left_weight + right_head * right_weight;
    if (blend_extent == right.size(dimension)) {
      return blended;
    }
    return torch::cat({blended, right.slice(dimension, blend_extent)},
                      dimension);
  }

  void load_config(const std::string& path) {
    std::ifstream input(path);
    CHECK(input.good()) << "MiniMax-H3 video VAE config not found: " << path;
    nlohmann::json config;
    input >> config;
    CHECK_EQ(config.at("latent_channels").get<int64_t>(), latent_channels_);
    CHECK_EQ(config.at("clip_length").get<int64_t>(), clip_length_);
    CHECK_EQ(config.at("token_drop").get<int64_t>(), token_drop_);
  }

  int64_t keyframe_encode_seed_ = 42;
  int64_t tokens_chunk_size_ = 5;
  int64_t token_overlap_ = 2;
  int64_t frame_pre_padding_ = 3;
  int64_t frame_overlap_ = 5;
  int64_t chunk_num_frames_ = 20;
  int64_t tile_sample_height_ = 256;
  int64_t tile_sample_width_ = 256;
  int64_t tile_sample_overlap_height_ = 64;
  int64_t tile_sample_overlap_width_ = 64;
  // Non-owning; ParallelArgs owns the process group.
  ProcessGroup* vae_group_ = nullptr;
  minimax_h3_video::Encoder3d encoder_{nullptr};
  minimax_h3_video::Conv3d1x1 quant_conv_{nullptr};
  minimax_h3_video::Conv3d1x1 post_quant_conv_{nullptr};
  minimax_h3_video::ViTDecoder3d decoder_{nullptr};
  int64_t latent_channels_ = 0;
  int64_t temporal_ratio_ = 0;
  int64_t spatial_ratio_ = 0;
  int64_t clip_length_ = 0;
  int64_t token_drop_ = 0;
  torch::Tensor latents_mean_tensor_;
  torch::Tensor latents_std_tensor_;
  torch::Tensor pixel_mean_tensor_;
  torch::Tensor pixel_std_tensor_;

  // Pads the temporal dimension (dim 2) by repeating the last frame.
  static torch::Tensor pad_by_repeating_last_frame(const torch::Tensor& x,
                                                   int64_t pad) {
    if (pad <= 0) {
      return x;
    }
    return torch::cat({x, x.slice(2, x.size(2) - 1).repeat({1, 1, pad, 1, 1})},
                      2);
  }
};
TORCH_MODULE(AutoencoderKLMiniMaxH3);

REGISTER_MODEL_ARGS(AutoencoderKLMiniMaxH3, [&] {
  LOAD_ARG_OR(in_channels, "in_channels", 3);
  LOAD_ARG_OR(out_channels, "out_channels", 3);
  LOAD_ARG_OR(latent_channels, "latent_channels", 24);
  LOAD_ARG_OR(block_out_channels,
              "block_out_channels",
              (std::vector<int64_t>{128, 256, 256, 512, 512, 1024}));
  LOAD_ARG_OR(layers_per_block, "layers_per_block", 2);
  LOAD_ARG_OR(spatial_downsample_factors,
              "spatial_downsample_factors",
              (std::vector<int64_t>{2, 2, 2, 2, 1, 1}));
  LOAD_ARG_OR(temporal_downsample_factors,
              "temporal_downsample_factors",
              (std::vector<int64_t>{1, 2, 2, 1, 1, 1}));
  LOAD_ARG_OR(norm_num_groups, "norm_num_groups", 32);
  LOAD_ARG_OR(norm_eps, "norm_eps", 1e-6f);
  LOAD_ARG_OR(spatial_padding_mode, "spatial_padding_mode", "reflect");
  LOAD_ARG_OR(decoder_num_blocks, "decoder_num_layers", 36);
  LOAD_ARG_OR(decoder_n_heads, "decoder_num_attention_heads", 32);
  LOAD_ARG_OR(decoder_head_dim, "decoder_attention_head_dim", 64);
  LOAD_ARG_OR(decoder_num_register_tokens, "decoder_num_register_tokens", 4);
  LOAD_ARG_OR(decoder_ffn_mult, "decoder_ffn_mult", 4);
  LOAD_ARG_OR(decoder_rope_theta, "decoder_rope_theta", 100.0f);
  LOAD_ARG_OR(decoder_rope_dim_ratio, "decoder_rope_dim_ratio", 0.75f);
  LOAD_ARG_OR(decoder_norm_eps, "decoder_norm_eps", 1e-5f);
  LOAD_ARG_OR(clip_length, "clip_length", 17);
  LOAD_ARG_OR(token_drop, "token_drop", 3);
  LOAD_ARG_OR(latents_mean, "latents_mean", (std::vector<double>(24, 0.0)));
  LOAD_ARG_OR(latents_std, "latents_std", (std::vector<double>(24, 1.0)));
});

}  // namespace xllm
