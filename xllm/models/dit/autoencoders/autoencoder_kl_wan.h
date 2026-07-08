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
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
#include <torch/torch.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "framework/model_context.h"
#include "models/dit/autoencoders/autoencoder_kl.h"
#include "models/dit/utils/vae_spatial_parallel.h"
#include "models/model_registry.h"

namespace xllm {
using xllm::dit::VaeSpatialParallel;

class AvgDown3DImpl : public torch::nn::Module {
 public:
  AvgDown3DImpl(int64_t in_channels,
                int64_t out_channels,
                int64_t factor_t,
                int64_t factor_s = 1)
      : in_channels_(in_channels),
        out_channels_(out_channels),
        factor_t_(factor_t),
        factor_s_(factor_s) {
    factor_ = factor_t_ * factor_s_ * factor_s_;
    TORCH_CHECK(in_channels_ * factor_ % out_channels_ == 0,
                "in_channels * factor must be divisible by out_channels");
    group_size_ = in_channels_ * factor_ / out_channels_;
  }

  torch::Tensor forward(torch::Tensor x) {
    int64_t pad_t = (factor_t_ - x.size(2) % factor_t_) % factor_t_;
    std::vector<int64_t> pad = {0, 0, 0, 0, pad_t, 0};
    x = torch::nn::functional::pad(x,
                                   torch::nn::functional::PadFuncOptions(pad));
    auto sizes = x.sizes();
    int64_t B = sizes[0], C = sizes[1], T = sizes[2], H = sizes[3],
            W = sizes[4];
    x = x.view({B,
                C,
                T / factor_t_,
                factor_t_,
                H / factor_s_,
                factor_s_,
                W / factor_s_,
                factor_s_});
    x = x.permute({0, 1, 3, 5, 7, 2, 4, 6}).contiguous();
    x = x.view({B, C * factor_, T / factor_t_, H / factor_s_, W / factor_s_});
    x = x.view({B,
                out_channels_,
                group_size_,
                T / factor_t_,
                H / factor_s_,
                W / factor_s_});
    x = x.mean(2);
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {}
  void verify_loaded_weights(const std::string& prefix) const {}

 private:
  int64_t in_channels_, out_channels_, factor_t_, factor_s_, factor_,
      group_size_;
};
TORCH_MODULE(AvgDown3D);

class DupUp3DImpl : public torch::nn::Module {
 public:
  DupUp3DImpl(int64_t in_channels,
              int64_t out_channels,
              int64_t factor_t,
              int64_t factor_s = 1)
      : in_channels_(in_channels),
        out_channels_(out_channels),
        factor_t_(factor_t),
        factor_s_(factor_s) {
    factor_ = factor_t_ * factor_s_ * factor_s_;
    CHECK(out_channels_ * factor_ % in_channels_ == 0),
        "out_channels * factor must be divisible by in_channels";
    repeats_ = out_channels_ * factor_ / in_channels_;
  }

  torch::Tensor forward(torch::Tensor x, bool first_chunk = false) {
    x = x.repeat_interleave(repeats_, 1);
    x = x.view({x.size(0),
                out_channels_,
                factor_t_,
                factor_s_,
                factor_s_,
                x.size(2),
                x.size(3),
                x.size(4)});
    x = x.permute({0, 1, 5, 2, 6, 3, 7, 4}).contiguous();
    x = x.view({x.size(0),
                out_channels_,
                x.size(2) * factor_t_,
                x.size(4) * factor_s_,
                x.size(6) * factor_s_});
    if (first_chunk) {
      x = x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(factor_t_ - 1, torch::indexing::None),
                   torch::indexing::Slice(),
                   torch::indexing::Slice()});
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {}
  void verify_loaded_weights(const std::string& prefix) const {}

 private:
  int64_t in_channels_, out_channels_, factor_t_, factor_s_, factor_, repeats_;
};
TORCH_MODULE(DupUp3D);

class WanCausalConv3DImpl : public torch::nn::Module {
 public:
  WanCausalConv3DImpl(int64_t in_channels,
                      int64_t out_channels,
                      std::vector<int64_t> kernel_size,
                      std::vector<int64_t> stride = {1, 1, 1},
                      std::vector<int64_t> padding = {0, 0, 0})
      : in_channels_(in_channels),
        out_channels_(out_channels),
        kernel_size_(kernel_size),
        stride_(stride),
        padding_(padding) {
    conv_ = register_module(
        "conv",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                // Conv3d always gets 0 for temporal padding — causal temporal
                // padding is handled separately via cache_x concatenation.
                // padding_ layout: {temporal_pad, height_pad, width_pad}
                .padding({0, padding[1], padding[2]})
                .bias(true)));
    // _padding_ is the 6-element format for F::pad:
    //   {dim4_left, dim4_right, dim3_left, dim3_right, dim2_left, dim2_right}
    // (dim4=W, dim3=H, dim2=T). Only the temporal (dim2) front padding is
    // non-zero (2 * pad_t) for causal convolution; spatial padding is handled
    // by Conv3d directly above.
    _padding_ = {0, 0, 0, 0, 2 * padding[0], 0};
  }

  torch::Tensor forward(
      const torch::Tensor& x,
      VaeSpatialParallel& ctx,
      const torch::optional<torch::Tensor>& cache_x = torch::nullopt) {
    std::vector<int64_t> padding = _padding_;
    torch::Tensor input = x;

    // Halo exchange is only needed for 3×3 spatial convolutions (kernel=3,
    // pad=1). For 1×1 convolutions (e.g. quant_conv) padding_ is {0,0,0}
    // so this condition is false and no exchange occurs.
    // padding_ layout: {pad_temporal, pad_height, pad_width}
    bool use_halo = (ctx.is_parallel() && padding_[1] == 1 && padding_[2] == 1);

    // Temporal padding (causal) — must come BEFORE halo exchange so cache
    // and input have matching spatial dims.
    if (cache_x.has_value() && cache_x.value().defined() && padding[4] > 0) {
      torch::Tensor cache = cache_x.value().to(x.device());
      input = torch::cat({cache, input}, 2);
      padding[4] -= cache.size(2);
    }

    // Spatial parallel: halo exchange for conv3d with spatial padding
    if (use_halo) {
      input = ctx.exchange(input, /*pad=*/true);
    }

    input = torch::nn::functional::pad(
        input, torch::nn::functional::PadFuncOptions(padding));

    auto out = conv_->forward(input);

    // Trim halo columns after conv:
    //   exchange() added 1 column from left neighbor and 1 from right neighbor
    //   (2 extra columns total). Conv3d with kernel=3, spatial pad=1 preserves
    //   spatial size, so the output is 2 columns wider in W than the true local
    //   result. Trim 1 column from each side (slice from column 1 to W-1).
    if (use_halo) {
      out = out.slice(/*dim=*/-1, 1, out.size(-1) - 1);
    }
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight", conv_->weight, is_weight_loaded_);
    weight::load_weight(state_dict, "bias", conv_->bias, is_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(is_bias_loaded_) << "bias is not loaded for " << prefix + "bias";
  }

 private:
  bool is_weight_loaded_{false};
  bool is_bias_loaded_{false};
  int64_t in_channels_, out_channels_;
  // padding_ layout: {temporal_pad, height_pad, width_pad}
  // _padding_ layout: 6-element F::pad format {W_left, W_right, H_left,
  // H_right, T_front, T_back}, where only T_front (2 * temporal_pad) is
  // non-zero for causal temporal padding.
  std::vector<int64_t> kernel_size_, stride_, padding_, _padding_;
  torch::nn::Conv3d conv_{nullptr};
};
TORCH_MODULE(WanCausalConv3D);

class WanRMSNormImpl : public torch::nn::Module {
 public:
  WanRMSNormImpl(int64_t dim,
                 bool channel_first = true,
                 bool images = true,
                 bool bias = false)
      : channel_first_(channel_first), images_(images), bias_enabled_(bias) {
    std::vector<int64_t> broadcastable_dims;
    if (!images) {
      broadcastable_dims = {1, 1, 1};
    } else {
      broadcastable_dims = {1, 1};
    }
    std::vector<int64_t> shape;
    if (channel_first) {
      shape.push_back(dim);
      shape.insert(
          shape.end(), broadcastable_dims.begin(), broadcastable_dims.end());
    } else {
      shape.push_back(dim);
    }
    scale_ = std::sqrt(static_cast<double>(dim));
    gamma_ = register_parameter("gamma", torch::ones(shape));
    if (bias_enabled_) {
      bias_ = register_parameter("bias", torch::zeros(shape));
    }
  }

  torch::Tensor forward(const torch::Tensor& x) {
    int64_t norm_dim = channel_first_ ? 1 : -1;
    auto normed = torch::nn::functional::normalize(
        x,
        torch::nn::functional::NormalizeFuncOptions().dim(norm_dim).eps(1e-12));
    auto out = normed * scale_ * gamma_;
    if (bias_enabled_) {
      out = out + bias_;
    }
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "gamma", gamma_, is_weight_loaded_);
    if (bias_enabled_) {
      weight::load_weight(state_dict, "bias", bias_, is_bias_loaded_);
    } else {
      is_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    if (bias_enabled_) {
      CHECK(is_bias_loaded_) << "bias is not loaded for " << prefix + "bias";
    }
  }

 private:
  bool is_weight_loaded_{false};
  bool is_bias_loaded_{false};
  bool channel_first_;
  bool images_;
  bool bias_enabled_;
  double scale_;
  torch::Tensor gamma_;
  torch::Tensor bias_;
};
TORCH_MODULE(WanRMSNorm);

class WanUpsampleImpl : public torch::nn::Module {
 public:
  WanUpsampleImpl(const torch::nn::functional::InterpolateFuncOptions options)
      : options_(options) {}

  torch::Tensor forward(const torch::Tensor& x) {
    auto result =
        torch::nn::functional::interpolate(x.to(torch::kFloat32), options_);
    return result;
  }

 private:
  torch::nn::functional::InterpolateFuncOptions options_;
  torch::nn::Upsample upsample_ = nullptr;
};

TORCH_MODULE(WanUpsample);

class WanResampleImpl : public torch::nn::Module {
 public:
  WanResampleImpl(int64_t dim,
                  const std::string& mode,
                  int64_t upsample_out_dim = -1)
      : dim_(dim), mode_(mode) {
    if (upsample_out_dim == -1) {
      upsample_out_dim = dim / 2;
    }
    torch::nn::Sequential resample;
    if (mode == "upsample2d") {
      resample = torch::nn::Sequential(
          WanUpsample(torch::nn::functional::InterpolateFuncOptions()
                          .scale_factor(std::vector<double>{2.0, 2.0})
                          .mode(torch::kNearestExact)),
          torch::nn::Conv2d(
              torch::nn::Conv2dOptions(dim, upsample_out_dim, 3).padding(1)));
    } else if (mode == "upsample3d") {
      resample = torch::nn::Sequential(
          WanUpsample(torch::nn::functional::InterpolateFuncOptions()
                          .scale_factor(std::vector<double>{2.0, 2.0})
                          .mode(torch::kNearestExact)),
          torch::nn::Conv2d(
              torch::nn::Conv2dOptions(dim, upsample_out_dim, 3).padding(1)));
      time_conv_ =
          register_module("time_conv",
                          WanCausalConv3D(dim,
                                          dim * 2,
                                          std::vector<int64_t>{3, 1, 1},
                                          std::vector<int64_t>{1, 1, 1},
                                          std::vector<int64_t>{1, 0, 0}));
    } else if (mode == "downsample2d") {
      resample = torch::nn::Sequential(
          torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions({0, 1, 0, 1})),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 3)
                                .stride(std::vector<int64_t>{2, 2})));
    } else if (mode == "downsample3d") {
      resample = torch::nn::Sequential(
          torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions({0, 1, 0, 1})),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 3)
                                .stride(std::vector<int64_t>{2, 2})));
      time_conv_ =
          register_module("time_conv",
                          WanCausalConv3D(dim,
                                          dim,
                                          std::vector<int64_t>{3, 1, 1},
                                          std::vector<int64_t>{2, 1, 1},
                                          std::vector<int64_t>{0, 0, 0}));
    } else {
      resample = torch::nn::Sequential(torch::nn::Identity());
    }
    resample_ = register_module("resample", resample);

    rep_tensor_ = register_parameter("rep_tensor", torch::tensor({-999.0}));
  }

  torch::Tensor forward(torch::Tensor x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        VaeSpatialParallel& ctx) {
    if (!feat_idx) feat_idx = std::make_shared<std::vector<int64_t>>(1, 0);

    auto sizes = x.sizes();
    int64_t b = sizes[0], c = sizes[1], t = sizes[2], h = sizes[3],
            w = sizes[4];

    bool is_upsample = (mode_ == "upsample2d" || mode_ == "upsample3d");
    bool is_downsample = (mode_ == "downsample2d" || mode_ == "downsample3d");

    if (mode_ == "upsample3d" && feat_cache) {
      int64_t idx = (*feat_idx)[0];
      if ((*feat_cache)[idx].numel() == 0) {
        feat_cache->at(idx) = rep_tensor_;
        (*feat_idx)[0]++;
      } else {
        auto cache_x =
            x.index({torch::indexing::Slice(),
                     torch::indexing::Slice(),
                     torch::indexing::Slice(-CACHE_T, torch::indexing::None),
                     torch::indexing::Slice(),
                     torch::indexing::Slice()})
                .clone();
        if (cache_x.size(2) < 2 && (*feat_cache)[idx].numel() > 0 &&
            !torch::equal(rep_tensor_, feat_cache->at(idx))) {
          cache_x = torch::cat({(*feat_cache)[idx]
                                    .index({torch::indexing::Slice(),
                                            torch::indexing::Slice(),
                                            -1,
                                            torch::indexing::Slice(),
                                            torch::indexing::Slice()})
                                    .unsqueeze(2)
                                    .to(cache_x.device()),
                                cache_x},
                               2);
        }
        if (cache_x.size(2) < 2 && (*feat_cache)[idx].numel() > 0 &&
            torch::equal(rep_tensor_, feat_cache->at(idx))) {
          cache_x = torch::cat(
              {torch::zeros_like(cache_x).to(cache_x.device()), cache_x}, 2);
        }
        if (torch::equal(rep_tensor_, feat_cache->at(idx))) {
          x = time_conv_->forward(x, ctx);
        } else {
          x = time_conv_->forward(x, ctx, (*feat_cache)[idx]);
        }
        (*feat_cache)[idx] = cache_x;
        (*feat_idx)[0] += 1;

        x = x.view({b, 2, c, t, h, w});
        x = torch::stack({x.index({torch::indexing::Slice(),
                                   0,
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice()}),
                          x.index({torch::indexing::Slice(),
                                   1,
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice()})},
                         3);
        x = x.view({b, c, t * 2, h, w});
      }
    }
    t = x.size(2);

    // Upsample: exchange halo columns before spatial upsampling
    if (ctx.is_parallel() && is_upsample) {
      x = ctx.exchange(x, /*pad=*/false);
      w = x.size(-1);  // recalc after halo exchange changed width
    }
    // Downsample: merge spatial slices before spatial downsampling
    if (ctx.is_parallel() && is_downsample) {
      x = ctx.merge(x);
      w = x.size(-1);  // recalc after merge: W is now global
      h = x.size(-2);  // recalc H too (may change with 2D split)
    }

    x = x.permute({0, 2, 1, 3, 4}).reshape({b * t, c, h, w});

    x = resample_->forward(x);
    x = x.view({b, t, x.size(1), x.size(2), x.size(3)})
            .permute({0, 2, 1, 3, 4});

    // Upsample: trim excess columns introduced by halo exchange.
    //
    // Before upsample: exchange(pad=false) adds 1 neighbor column per side,
    // making W_local → W_local + (has_left ? 1 : 0) + (has_right ? 1 : 0).
    // The interpolate 2× doubles each column, and the following Conv2d(k=3,
    // pad=1) preserves size. So the halo columns produce 2 excess columns per
    // side in the output.
    //
    // Trim 2 columns from each side where a neighbor contributed halo:
    //   - middle rank (has_left && has_right): trim 2 from left, 2 from right
    //   - first rank  (!has_left):           trim 2 from right only
    //   - last rank   (!has_right):          trim 2 from left only
    if (ctx.is_parallel() && is_upsample) {
      auto cur_w = x.size(-1);
      if (ctx.has_left() && ctx.has_right()) {
        x = x.slice(/*dim=*/-1, 2, cur_w - 2);  // trim 2 from each side
      } else if (!ctx.has_left()) {
        x = x.slice(/*dim=*/-1, 0, cur_w - 2);  // trim 2 from right only
      } else {
        x = x.slice(/*dim=*/-1, 2, cur_w);  // trim 2 from left only
      }
    }
    // Downsample: split full result back to local slices
    if (ctx.is_parallel() && is_downsample) {
      x = ctx.split(x);
    }

    if (mode_ == "downsample3d" && feat_cache) {
      int64_t idx = (*feat_idx)[0];
      if ((*feat_cache)[idx].numel() == 0) {
        (*feat_cache)[idx] = x.clone();
        (*feat_idx)[0] += 1;
      } else {
        auto cache_x =
            x.index({torch::indexing::Slice(),
                     torch::indexing::Slice(),
                     torch::indexing::Slice(-1, torch::indexing::None),
                     torch::indexing::Slice(),
                     torch::indexing::Slice()})
                .clone();
        x = time_conv_->forward(
            torch::cat({(*feat_cache)[idx].index(
                            {torch::indexing::Slice(),
                             torch::indexing::Slice(),
                             torch::indexing::Slice(-1, torch::indexing::None),
                             torch::indexing::Slice(),
                             torch::indexing::Slice()}),
                        x},
                       2),
            ctx);
        (*feat_cache)[idx] = cache_x;
        (*feat_idx)[0] += 1;
      }
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    auto params = resample_->named_parameters();
    for (auto& param : params) {
      std::string name = param.key();
      if (name == "1.weight") {
        weight::load_weight(
            state_dict, "resample.1.weight", param.value(), is_weight_loaded_);
      } else if (name == "1.bias") {
        weight::load_weight(
            state_dict, "resample.1.bias", param.value(), is_bias_loaded_);
      }
    }
    if (time_conv_) {
      time_conv_->load_state_dict(
          state_dict.get_dict_with_prefix("time_conv."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(is_bias_loaded_) << "bias is not loaded for " << prefix + "bias";
    if (time_conv_) {
      time_conv_->verify_loaded_weights("time_conv.");
    }
  }

 private:
  int64_t dim_;
  bool is_weight_loaded_{false};
  bool is_bias_loaded_{false};
  torch::Tensor rep_tensor_;
  std::string mode_;
  torch::nn::Sequential resample_{nullptr};
  WanCausalConv3D time_conv_{nullptr};
  const int64_t CACHE_T = 2;
};
TORCH_MODULE(WanResample);

class WanResidualBlockImpl : public torch::nn::Module {
 public:
  WanResidualBlockImpl(int64_t in_dim, int64_t out_dim, float dropout = 0.0f)
      : in_dim_(in_dim), out_dim_(out_dim) {
    nonlinearity_ = torch::nn::Functional(torch::silu);
    norm1_ = register_module("norm1", WanRMSNorm(in_dim, true, false, false));
    conv1_ = register_module("conv1",
                             WanCausalConv3D(in_dim,
                                             out_dim,
                                             std::vector<int64_t>{3, 3, 3},
                                             std::vector<int64_t>{1, 1, 1},
                                             std::vector<int64_t>{1, 1, 1}));
    norm2_ = register_module("norm2", WanRMSNorm(out_dim, true, false, false));
    dropout_layer_ = register_module("dropout", torch::nn::Dropout(dropout));
    conv2_ = register_module("conv2",
                             WanCausalConv3D(out_dim,
                                             out_dim,
                                             std::vector<int64_t>{3, 3, 3},
                                             std::vector<int64_t>{1, 1, 1},
                                             std::vector<int64_t>{1, 1, 1}));
    if (in_dim_ != out_dim_) {
      conv_shortcut_ =
          register_module("conv_shortcut",
                          WanCausalConv3D(in_dim,
                                          out_dim,
                                          std::vector<int64_t>{1, 1, 1},
                                          std::vector<int64_t>{1, 1, 1},
                                          std::vector<int64_t>{0, 0, 0}));
    }
  }

  torch::Tensor forward(torch::Tensor x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        VaeSpatialParallel& ctx) {
    if (!feat_idx) feat_idx = std::make_shared<std::vector<int64_t>>(1, 0);

    torch::Tensor h;
    if (in_dim_ != out_dim_) {
      h = conv_shortcut_->forward(x, ctx);
    } else {
      h = x;
    }

    x = norm1_->forward(x);
    x = nonlinearity_(x);

    if (feat_cache) {
      int64_t idx = (*feat_idx)[0];
      auto cache_x =
          x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(-CACHE_T, torch::indexing::None),
                   torch::indexing::Slice(),
                   torch::indexing::Slice()})
              .clone();
      if (cache_x.size(2) < 2 && (*feat_cache)[idx].numel() > 0) {
        cache_x = torch::cat({(*feat_cache)[idx]
                                  .index({torch::indexing::Slice(),
                                          torch::indexing::Slice(),
                                          -1,
                                          torch::indexing::Slice(),
                                          torch::indexing::Slice()})
                                  .unsqueeze(2)
                                  .to(cache_x.device()),
                              cache_x},
                             2);
      }
      x = conv1_->forward(x, ctx, (*feat_cache)[idx]);
      (*feat_cache)[idx] = cache_x;
      (*feat_idx)[0] += 1;
    } else {
      x = conv1_->forward(x, ctx);
    }

    x = norm2_->forward(x);
    x = nonlinearity_(x);
    x = dropout_layer_->forward(x);

    if (feat_cache) {
      int64_t idx = (*feat_idx)[0];
      auto cache_x =
          x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(-CACHE_T, torch::indexing::None),
                   torch::indexing::Slice(),
                   torch::indexing::Slice()})
              .clone();

      if (cache_x.size(2) < 2 && idx < feat_cache->size() &&
          (*feat_cache)[idx].numel()) {
        cache_x = torch::cat({(*feat_cache)[idx]
                                  .index({torch::indexing::Slice(),
                                          torch::indexing::Slice(),
                                          -1,
                                          torch::indexing::Slice(),
                                          torch::indexing::Slice()})
                                  .unsqueeze(2)
                                  .to(cache_x.device()),
                              cache_x},
                             2);
      }
      x = conv2_->forward(x, ctx, (*feat_cache)[idx]);
      (*feat_cache)[idx] = cache_x;
      (*feat_idx)[0] += 1;
    } else {
      x = conv2_->forward(x, ctx);
    }

    return x + h;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    conv1_->load_state_dict(state_dict.get_dict_with_prefix("conv1."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    conv2_->load_state_dict(state_dict.get_dict_with_prefix("conv2."));
    if (in_dim_ != out_dim_) {
      conv_shortcut_->load_state_dict(
          state_dict.get_dict_with_prefix("conv_shortcut."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    norm1_->verify_loaded_weights("norm1.");
    norm2_->verify_loaded_weights("norm2.");
    conv1_->verify_loaded_weights("conv1.");
    conv2_->verify_loaded_weights("conv2.");
    if (in_dim_ != out_dim_) {
      conv_shortcut_->verify_loaded_weights("conv_shortcut.");
    }
  }

 private:
  int64_t in_dim_, out_dim_;
  const int64_t CACHE_T = 2;

 public:
  torch::nn::Functional nonlinearity_{nullptr};
  WanRMSNorm norm1_{nullptr}, norm2_{nullptr};
  WanCausalConv3D conv1_{nullptr}, conv2_{nullptr}, conv_shortcut_{nullptr};
  torch::nn::Dropout dropout_layer_{nullptr};
};
TORCH_MODULE(WanResidualBlock);

class WanAttentionBlockImpl : public torch::nn::Module {
 public:
  WanAttentionBlockImpl(int64_t dim) : dim_(dim) {
    norm_ = register_module("norm", WanRMSNorm(dim, true, true, false));
    to_qkv_ = register_module(
        "to_qkv", torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim * 3, 1)));
    proj_ = register_module(
        "proj", torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 1)));
  }

  torch::Tensor forward(torch::Tensor x, VaeSpatialParallel& ctx) {
    torch::Tensor identity = x;
    auto sizes = x.sizes();
    int64_t batch_size = sizes[0];
    int64_t channels = sizes[1];
    int64_t time = sizes[2];
    int64_t height = sizes[3];
    int64_t width = sizes[4];

    x = x.permute({0, 2, 1, 3, 4})
            .reshape({batch_size * time, channels, height, width});
    x = norm_->forward(x);

    auto qkv = to_qkv_->forward(x);

    torch::Tensor q, k, v;
    if (ctx.is_parallel()) {
      // Gather K/V in spatial domain before BNSD reshape
      auto spatial_chunks = qkv.chunk(3, 1);
      auto k_sp = ctx.merge(spatial_chunks[1].unsqueeze(2)).squeeze(2);
      auto v_sp = ctx.merge(spatial_chunks[2].unsqueeze(2)).squeeze(2);
      int64_t W_global = k_sp.size(-1);
      q = spatial_chunks[0]
              .reshape({batch_size * time, 1, channels, height * width})
              .permute({0, 1, 3, 2})
              .contiguous();
      k = k_sp.reshape({batch_size * time, 1, channels, height * W_global})
              .permute({0, 1, 3, 2})
              .contiguous();
      v = v_sp.reshape({batch_size * time, 1, channels, height * W_global})
              .permute({0, 1, 3, 2})
              .contiguous();
    } else {
      qkv = qkv.reshape({batch_size * time, 1, channels * 3, height * width});
      qkv = qkv.permute({0, 1, 3, 2}).contiguous();
      auto chunks = qkv.chunk(3, -1);
      q = chunks[0];
      k = chunks[1];
      v = chunks[2];
    }

    auto results = at_npu::native::custom_ops::npu_fusion_attention(
        q,
        k,
        v,
        /*head_num=*/1,
        /*input_layout=*/"BNSD",
        /*pse*/ torch::nullopt,
        /*padding_mask=*/torch::nullopt,
        /*atten_mask=*/torch::nullopt,
        /*scale=*/pow(channels, -0.5),
        /*keep_prob=*/1.0,
        /*pre_tockens=*/65535,
        /*next_tockens=*/65535);
    auto attn_output = std::get<0>(results);

    auto attn_out = attn_output.squeeze(1).permute({0, 2, 1}).reshape(
        {batch_size * time, channels, height, width});
    attn_out = proj_->forward(attn_out);

    attn_out = attn_out.view({batch_size, time, channels, height, width})
                   .permute({0, 2, 1, 3, 4});
    return attn_out + identity;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));

    weight::load_weight(
        state_dict, "to_qkv.weight", to_qkv_->weight, is_qkv_weight_loaded_);
    weight::load_weight(
        state_dict, "to_qkv.bias", to_qkv_->bias, is_qkv_bias_loaded_);
    weight::load_weight(
        state_dict, "proj.weight", proj_->weight, is_proj_weight_loaded_);
    weight::load_weight(
        state_dict, "proj.bias", proj_->bias, is_proj_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    norm_->verify_loaded_weights("norm.");

    CHECK(is_qkv_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(is_qkv_bias_loaded_)
        << "weight is not loaded for " << prefix + "bias";
    CHECK(is_proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(is_proj_bias_loaded_)
        << "weight is not loaded for " << prefix + "bias";
  }

 private:
  int64_t dim_;
  bool is_qkv_weight_loaded_{false};
  bool is_qkv_bias_loaded_{false};
  bool is_proj_weight_loaded_{false};
  bool is_proj_bias_loaded_{false};
  WanRMSNorm norm_{nullptr};
  torch::nn::Conv2d to_qkv_{nullptr};
  torch::nn::Conv2d proj_{nullptr};
};
TORCH_MODULE(WanAttentionBlock);

class WanMidBlockImpl : public torch::nn::Module {
 public:
  WanMidBlockImpl(int64_t dim, float dropout = 0.0f, int64_t num_layers = 1)
      : dim_(dim) {
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    attentions_ = register_module("attentions", torch::nn::ModuleList());
    resnets_->push_back(WanResidualBlock(dim, dim, dropout));
    for (int64_t i = 0; i < num_layers; ++i) {
      attentions_->push_back(WanAttentionBlock(dim));
      resnets_->push_back(WanResidualBlock(dim, dim, dropout));
    }
  }

  torch::Tensor forward(torch::Tensor x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        VaeSpatialParallel& ctx) {
    if (!feat_idx) feat_idx = std::make_shared<std::vector<int64_t>>(1, 0);

    x = resnets_[0]->as<WanResidualBlock>()->forward(
        x, feat_cache, feat_idx, ctx);
    for (size_t i = 0; i < attentions_->size(); ++i) {
      auto attn = attentions_[i]->as<WanAttentionBlock>();
      if (attn) {
        x = attn->forward(x, ctx);
      }
      auto resnet = resnets_[i + 1]->as<WanResidualBlock>();
      x = resnet->forward(x, feat_cache, feat_idx, ctx);
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); i++) {
      auto prefix = "resnets." + std::to_string(i) + ".";
      resnets_[i]->as<WanResidualBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix(prefix));
    }

    for (size_t i = 0; i < attentions_->size(); i++) {
      auto prefix = "attentions." + std::to_string(i) + ".";
      attentions_[i]->as<WanAttentionBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix(prefix));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < resnets_->size(); i++) {
      auto prefix = "resnets." + std::to_string(i) + ".";
      resnets_[i]->as<WanResidualBlock>()->verify_loaded_weights(prefix);
    }

    for (size_t i = 0; i < attentions_->size(); i++) {
      auto prefix = "attentions." + std::to_string(i) + ".";
      attentions_[i]->as<WanAttentionBlock>()->verify_loaded_weights(prefix);
    }
  }

 private:
  int64_t dim_;

 public:
  torch::nn::ModuleList resnets_{nullptr};
  torch::nn::ModuleList attentions_{nullptr};
};
TORCH_MODULE(WanMidBlock);

class WanResidualDownBlockImpl : public torch::nn::Module {
 public:
  WanResidualDownBlockImpl(int64_t in_dim,
                           int64_t out_dim,
                           float dropout,
                           int64_t num_res_blocks,
                           bool temperal_downsample = false,
                           bool down_flag = false)
      : in_dim_(in_dim),
        out_dim_(out_dim),
        dropout_(dropout),
        num_res_blocks_(num_res_blocks),
        temperal_downsample_(temperal_downsample),
        down_flag_(down_flag) {
    int64_t factor_t = temperal_downsample ? 2 : 1;
    int64_t factor_s = down_flag ? 2 : 1;
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    int64_t cur_in_dim = in_dim;
    for (int64_t i = 0; i < num_res_blocks; ++i) {
      resnets_->push_back(WanResidualBlock(cur_in_dim, out_dim, dropout));
      cur_in_dim = out_dim;
    }
    if (down_flag) {
      std::string mode = temperal_downsample ? "downsample3d" : "downsample2d";
      downsampler_ =
          register_module("downsampler", WanResample(out_dim, mode, -1));
    } else {
      downsampler_ = nullptr;
    }
  }

  torch::Tensor forward(torch::Tensor x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        VaeSpatialParallel& ctx) {
    if (!feat_idx) feat_idx = std::make_shared<std::vector<int64_t>>(1, 0);

    torch::Tensor x_copy = x.clone();
    for (size_t i = 0; i < resnets_->size(); ++i) {
      x = resnets_[i]->as<WanResidualBlock>()->forward(
          x, feat_cache, feat_idx, ctx);
    }
    if (downsampler_) {
      x = downsampler_->forward(x, feat_cache, feat_idx, ctx);
    }

    if (avg_shortcut_) {
      return x + avg_shortcut_->forward(x_copy);
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_[i]->as<WanResidualBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
    if (downsampler_) {
      downsampler_->as<WanResample>()->load_state_dict(
          state_dict.get_dict_with_prefix("downsampler."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < resnets_->size(); ++i)
      resnets_[i]->as<WanResidualBlock>()->verify_loaded_weights(
          "resnets." + std::to_string(i) + ".");
    if (downsampler_)
      downsampler_->as<WanResample>()->verify_loaded_weights("downsampler.");
  }

 private:
  int64_t in_dim_, out_dim_;
  float dropout_;
  int64_t num_res_blocks_;
  bool temperal_downsample_, down_flag_;
  AvgDown3D avg_shortcut_{nullptr};
  torch::nn::ModuleList resnets_;
  WanResample downsampler_{nullptr};
};
TORCH_MODULE(WanResidualDownBlock);

class WanVAEEncoder3DImpl : public torch::nn::Module {
 public:
  WanVAEEncoder3DImpl(int64_t in_channels = 3,
                      int64_t dim = 128,
                      int64_t z_dim = 4,
                      std::vector<int64_t> dim_mult = {1, 2, 4, 4},
                      int64_t num_res_blocks = 2,
                      std::vector<double> attn_scales = {},
                      std::vector<bool> temperal_downsample = {true,
                                                               true,
                                                               false},
                      float dropout = 0.0f,
                      bool is_residual = false) {
    nonlinearity_ = torch::nn::Functional(torch::silu);
    std::vector<int64_t> dims;
    dims.push_back(dim);
    for (auto u : dim_mult) dims.push_back(dim * u);
    double scale = 1.0;
    conv_in_ = register_module("conv_in",
                               WanCausalConv3D(in_channels,
                                               dims[0],
                                               std::vector<int64_t>{3, 3, 3},
                                               std::vector<int64_t>{1, 1, 1},
                                               std::vector<int64_t>{1, 1, 1}));
    down_blocks_ = register_module("down_blocks", torch::nn::ModuleList());
    for (size_t i = 0; i < dims.size() - 1; ++i) {
      int64_t in_dim = dims[i];
      int64_t out_dim = dims[i + 1];
      if (is_residual) {
        down_blocks_->push_back(WanResidualDownBlock(
            in_dim,
            out_dim,
            dropout,
            num_res_blocks,
            (i != dim_mult.size() - 1) ? temperal_downsample[i] : false,
            (i != dim_mult.size() - 1)));
      } else {
        int64_t current_dim = in_dim;
        for (int64_t j = 0; j < num_res_blocks; ++j) {
          down_blocks_->push_back(
              WanResidualBlock(current_dim, out_dim, dropout));
          if (std::find(attn_scales.begin(), attn_scales.end(), scale) !=
              attn_scales.end()) {
            down_blocks_->push_back(WanAttentionBlock(out_dim));
          }
          current_dim = out_dim;
        }
        if (i != dim_mult.size() - 1) {
          std::string mode =
              temperal_downsample[i] ? "downsample3d" : "downsample2d";
          down_blocks_->push_back(WanResample(out_dim, mode, -1));
          scale /= 2.0;
        }
      }
    }
    mid_block_ =
        register_module("mid_block", WanMidBlock(dims.back(), dropout, 1));
    norm_out_ = register_module("norm_out",
                                WanRMSNorm(dims.back(), true, false, false));
    conv_out_ = register_module("conv_out",
                                WanCausalConv3D(dims.back(),
                                                z_dim,
                                                std::vector<int64_t>{3, 3, 3},
                                                std::vector<int64_t>{1, 1, 1},
                                                std::vector<int64_t>{1, 1, 1}));
  }

  torch::Tensor forward(torch::Tensor x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        VaeSpatialParallel& ctx) {
    if (!feat_idx) feat_idx = std::make_shared<std::vector<int64_t>>(1, 0);
    if (feat_cache) {
      int64_t idx = (*feat_idx)[0];
      auto cache_x =
          x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(-CACHE_T, torch::indexing::None),
                   torch::indexing::Slice(),
                   torch::indexing::Slice()})
              .clone();
      if (cache_x.size(2) < 2 && (*feat_cache)[idx].numel() > 0) {
        cache_x = torch::cat({(*feat_cache)[idx]
                                  .index({torch::indexing::Slice(),
                                          torch::indexing::Slice(),
                                          -1,
                                          torch::indexing::Slice(),
                                          torch::indexing::Slice()})
                                  .unsqueeze(2)
                                  .to(cache_x.device()),
                              cache_x},
                             2);
      }
      x = conv_in_->forward(x, ctx, (*feat_cache)[idx]);
      (*feat_cache)[idx] = cache_x;
      (*feat_idx)[0] += 1;
    } else {
      x = conv_in_->forward(x, ctx);
    }

    // Type-safe forward call for down_blocks_
    for (size_t i = 0; i < down_blocks_->size(); ++i) {
      if (auto res_down = down_blocks_[i]->as<WanResidualDownBlock>()) {
        x = res_down->forward(x, feat_cache, feat_idx, ctx);
      } else if (auto down = down_blocks_[i]->as<WanResidualBlock>()) {
        x = feat_cache ? down->forward(x, feat_cache, feat_idx, ctx)
                       : down->forward(x, nullptr, nullptr, ctx);
      } else if (auto attn = down_blocks_[i]->as<WanAttentionBlock>()) {
        x = attn->forward(x, ctx);
      } else if (auto resample = down_blocks_[i]->as<WanResample>()) {
        x = feat_cache ? resample->forward(x, feat_cache, feat_idx, ctx)
                       : resample->forward(x, nullptr, nullptr, ctx);
      }
    }

    x = mid_block_->forward(x, feat_cache, feat_idx, ctx);
    x = norm_out_->forward(x);
    x = nonlinearity_(x);
    if (feat_cache) {
      int64_t idx = (*feat_idx)[0];
      auto cache_x =
          x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(-CACHE_T, torch::indexing::None),
                   torch::indexing::Slice(),
                   torch::indexing::Slice()})
              .clone();
      if (cache_x.size(2) < 2 && (*feat_cache)[idx].numel() > 0) {
        cache_x = torch::cat({(*feat_cache)[idx]
                                  .index({torch::indexing::Slice(),
                                          torch::indexing::Slice(),
                                          -1,
                                          torch::indexing::Slice(),
                                          torch::indexing::Slice()})
                                  .unsqueeze(2)
                                  .to(cache_x.device()),
                              cache_x},
                             2);
      }
      x = conv_out_->forward(x, ctx, (*feat_cache)[idx]);
      (*feat_cache)[idx] = cache_x;
      (*feat_idx)[0] += 1;
    } else {
      x = conv_out_->forward(x, ctx);
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    conv_in_->load_state_dict(state_dict.get_dict_with_prefix("conv_in."));

    // Safely load weights of down_blocks :
    for (size_t i = 0; i < down_blocks_->size(); ++i) {
      std::string prefix = "down_blocks." + std::to_string(i) + ".";

      if (down_blocks_[i]->as<WanResidualDownBlock>()) {
        down_blocks_[i]->as<WanResidualDownBlock>()->load_state_dict(
            state_dict.get_dict_with_prefix(prefix));
      } else if (down_blocks_[i]->as<WanResidualBlock>()) {
        down_blocks_[i]->as<WanResidualBlock>()->load_state_dict(
            state_dict.get_dict_with_prefix(prefix));
      } else if (down_blocks_[i]->as<WanAttentionBlock>()) {
        down_blocks_[i]->as<WanAttentionBlock>()->load_state_dict(
            state_dict.get_dict_with_prefix(prefix));
      } else if (down_blocks_[i]->as<WanResample>()) {
        down_blocks_[i]->as<WanResample>()->load_state_dict(
            state_dict.get_dict_with_prefix(prefix));
      }
    }

    mid_block_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
    norm_out_->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    conv_out_->load_state_dict(state_dict.get_dict_with_prefix("conv_out."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    conv_in_->verify_loaded_weights("conv_in.");
    for (size_t i = 0; i < down_blocks_->size(); ++i) {
      std::string p = "down_blocks." + std::to_string(i) + ".";
      if (down_blocks_[i]->as<WanResidualDownBlock>())
        down_blocks_[i]->as<WanResidualDownBlock>()->verify_loaded_weights(p);
      else if (down_blocks_[i]->as<WanResidualBlock>())
        down_blocks_[i]->as<WanResidualBlock>()->verify_loaded_weights(p);
      else if (down_blocks_[i]->as<WanAttentionBlock>())
        down_blocks_[i]->as<WanAttentionBlock>()->verify_loaded_weights(p);
      else if (down_blocks_[i]->as<WanResample>())
        down_blocks_[i]->as<WanResample>()->verify_loaded_weights(p);
    }
    mid_block_->verify_loaded_weights("mid_block.");
    norm_out_->verify_loaded_weights("norm_out.");
    conv_out_->verify_loaded_weights("conv_out.");
  }

 private:
  torch::nn::Functional nonlinearity_{nullptr};
  WanCausalConv3D conv_in_{nullptr};
  torch::nn::ModuleList down_blocks_{nullptr};
  WanMidBlock mid_block_{nullptr};
  WanRMSNorm norm_out_{nullptr};
  WanCausalConv3D conv_out_{nullptr};
  const int64_t CACHE_T = 2;
};
TORCH_MODULE(WanVAEEncoder3D);

class WanResidualUpBlockImpl : public torch::nn::Module {
 public:
  WanResidualUpBlockImpl(int64_t in_dim,
                         int64_t out_dim,
                         int64_t num_res_blocks,
                         float dropout = 0.0f,
                         bool temperal_upsample = false,
                         bool up_flag = false)
      : in_dim_(in_dim), out_dim_(out_dim), num_res_blocks_(num_res_blocks) {
    if (up_flag) {
      int64_t factor_t = temperal_upsample ? 2 : 1;
      int64_t factor_s = 2;
      avg_shortcut_ = register_module(
          "avg_shortcut", DupUp3D(in_dim, out_dim, factor_t, factor_s));
    } else {
      avg_shortcut_ = nullptr;
    }
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    int64_t current_dim = in_dim;
    for (int64_t i = 0; i < num_res_blocks + 1; ++i) {
      resnets_->push_back(WanResidualBlock(current_dim, out_dim, dropout));
      current_dim = out_dim;
    }
    if (up_flag) {
      std::string upsample_mode =
          temperal_upsample ? "upsample3d" : "upsample2d";
      upsampler_ = register_module(
          "upsampler", WanResample(out_dim, upsample_mode, out_dim));
    } else {
      upsampler_ = nullptr;
    }
  }

  torch::Tensor forward(torch::Tensor x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        bool first_chunk,
                        VaeSpatialParallel& ctx) {
    if (!feat_idx) feat_idx = std::make_shared<std::vector<int64_t>>(1, 0);

    torch::Tensor x_copy = x.clone();
    for (size_t i = 0; i < resnets_->size(); ++i) {
      if (feat_cache) {
        x = resnets_[i]->as<WanResidualBlock>()->forward(
            x, feat_cache, feat_idx, ctx);
      } else {
        x = resnets_[i]->as<WanResidualBlock>()->forward(
            x, nullptr, nullptr, ctx);
      }
    }
    if (upsampler_) {
      if (feat_cache) {
        x = upsampler_->as<WanResample>()->forward(
            x, feat_cache, feat_idx, ctx);
      } else {
        x = upsampler_->as<WanResample>()->forward(x, nullptr, nullptr, ctx);
      }
    }
    if (avg_shortcut_) {
      x = x + avg_shortcut_->as<DupUp3D>()->forward(x_copy, first_chunk);
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_[i]->as<WanResidualBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
    if (upsampler_) {
      upsampler_->load_state_dict(
          state_dict.get_dict_with_prefix("upsampler."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < resnets_->size(); i++) {
      auto prefix = "resnets." + std::to_string(i) + ".";
      resnets_[i]->as<WanResidualBlock>()->verify_loaded_weights(prefix);
    }

    if (upsampler_) {
      upsampler_->as<WanResample>()->verify_loaded_weights("upsampler.");
    }
  }

 private:
  int64_t in_dim_, out_dim_;
  int64_t num_res_blocks_;
  DupUp3D avg_shortcut_{nullptr};
  torch::nn::ModuleList resnets_{nullptr};
  WanResample upsampler_{nullptr};
};
TORCH_MODULE(WanResidualUpBlock);

class WanUpBlockImpl : public torch::nn::Module {
 public:
  WanUpBlockImpl(int64_t in_dim,
                 int64_t out_dim,
                 int64_t num_res_blocks,
                 float dropout = 0.0f,
                 const std::optional<std::string>& upsample_mode = std::nullopt)
      : in_dim_(in_dim), out_dim_(out_dim), num_res_blocks_(num_res_blocks) {
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    int64_t current_dim = in_dim;
    for (int64_t i = 0; i < num_res_blocks + 1; ++i) {
      resnets_->push_back(WanResidualBlock(current_dim, out_dim, dropout));
      current_dim = out_dim;
    }
    if (upsample_mode.has_value()) {
      upsamplers_ = register_module("upsamplers", torch::nn::ModuleList());
      upsamplers_->push_back(WanResample(out_dim, upsample_mode.value()));
    }
  }

  torch::Tensor forward(torch::Tensor x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        VaeSpatialParallel& ctx) {
    if (!feat_idx) feat_idx = std::make_shared<std::vector<int64_t>>(1, 0);

    torch::Tensor h = x;
    for (size_t i = 0; i < resnets_->size(); ++i) {
      auto resnet = resnets_[i]->as<WanResidualBlock>();
      if (feat_cache) {
        h = resnet->forward(h, feat_cache, feat_idx, ctx);
      } else {
        h = resnet->forward(h, nullptr, nullptr, ctx);
      }
    }
    if (upsamplers_ && upsamplers_->size() > 0) {
      auto upsampler = upsamplers_[0]->as<WanResample>();
      if (feat_cache) {
        h = upsampler->forward(h, feat_cache, feat_idx, ctx);
      } else {
        h = upsampler->forward(h, nullptr, nullptr, ctx);
      }
    }
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_[i]->as<WanResidualBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
    if (upsamplers_) {
      for (size_t i = 0; i < upsamplers_->size(); ++i) {
        upsamplers_[i]->as<WanResample>()->load_state_dict(
            state_dict.get_dict_with_prefix("upsamplers." + std::to_string(i) +
                                            "."));
      }
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < resnets_->size(); i++) {
      auto prefix = "resnets." + std::to_string(i) + ".";
      resnets_[i]->as<WanResidualBlock>()->verify_loaded_weights(prefix);
    }

    if (upsamplers_) {
      for (size_t i = 0; i < upsamplers_->size(); i++) {
        auto prefix = "upsamplers." + std::to_string(i) + ".";
        upsamplers_[i]->as<WanResample>()->verify_loaded_weights(prefix);
      }
    }
  }

 private:
  int64_t in_dim_;
  int64_t out_dim_;
  int64_t num_res_blocks_;
  torch::nn::ModuleList resnets_{nullptr};
  torch::nn::ModuleList upsamplers_{nullptr};
};
TORCH_MODULE(WanUpBlock);

class WanVAEDecoder3DImpl : public torch::nn::Module {
 public:
  WanVAEDecoder3DImpl(int64_t dim = 128,
                      int64_t z_dim = 4,
                      const std::vector<int64_t>& dim_mult = {1, 2, 4, 4},
                      int64_t num_res_blocks = 2,
                      const std::vector<double>& attn_scales = {},
                      const std::vector<bool>& temperal_upsample = {false,
                                                                    true,
                                                                    true},
                      float dropout = 0.0f,
                      int64_t out_channels = 3,
                      bool is_residual = false) {
    std::vector<int64_t> dims;
    dims.push_back(dim * dim_mult.back());
    for (auto it = dim_mult.rbegin(); it != dim_mult.rend(); ++it) {
      dims.push_back(dim * (*it));
    }
    conv_in_ = register_module("conv_in",
                               WanCausalConv3D(z_dim,
                                               dims[0],
                                               std::vector<int64_t>{3, 3, 3},
                                               std::vector<int64_t>{1, 1, 1},
                                               std::vector<int64_t>{1, 1, 1}));
    mid_block_ = register_module("mid_block", WanMidBlock(dims[0], dropout, 1));
    up_blocks_ = register_module("up_blocks", torch::nn::ModuleList());
    for (size_t i = 0; i < dims.size() - 1; ++i) {
      int64_t in_dim = dims[i];
      int64_t out_dim = dims[i + 1];
      if (i > 0 && !is_residual) {
        in_dim = in_dim / 2;
      }
      bool up_flag = (i != dim_mult.size() - 1);
      std::string upsample_mode;
      if (up_flag && temperal_upsample[i]) {
        upsample_mode = "upsample3d";
      } else if (up_flag) {
        upsample_mode = "upsample2d";
      }
      if (is_residual) {
        up_blocks_->push_back(
            WanResidualUpBlock(in_dim,
                               out_dim,
                               num_res_blocks,
                               dropout,
                               (up_flag ? temperal_upsample[i] : false),
                               up_flag));
      } else {
        up_blocks_->push_back(
            WanUpBlock(in_dim,
                       out_dim,
                       num_res_blocks,
                       dropout,
                       up_flag ? std::optional<std::string>(upsample_mode)
                               : std::nullopt));
      }
    }
    nonlinearity_ = torch::nn::Functional(torch::silu);
    norm_out_ = register_module("norm_out",
                                WanRMSNorm(dims.back(), true, false, false));
    conv_out_ = register_module("conv_out",
                                WanCausalConv3D(dims.back(),
                                                out_channels,
                                                std::vector<int64_t>{3, 3, 3},
                                                std::vector<int64_t>{1, 1, 1},
                                                std::vector<int64_t>{1, 1, 1}));
  }

  torch::Tensor forward(torch::Tensor x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        bool first_chunk,
                        VaeSpatialParallel& ctx) {
    if (!feat_idx) feat_idx = std::make_shared<std::vector<int64_t>>(1, 0);

    // conv_in
    if (feat_cache) {
      int64_t idx = (*feat_idx)[0];
      torch::Tensor cache_x =
          x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(-CACHE_T, torch::indexing::None),
                   torch::indexing::Slice(),
                   torch::indexing::Slice()})
              .clone();
      if (cache_x.size(2) < 2 && (*feat_cache)[idx].defined()) {
        cache_x = torch::cat({(*feat_cache)[idx]
                                  .index({torch::indexing::Slice(),
                                          torch::indexing::Slice(),
                                          -1,
                                          torch::indexing::Slice(),
                                          torch::indexing::Slice()})
                                  .unsqueeze(2)
                                  .to(cache_x.device()),
                              cache_x},
                             2);
      }
      x = conv_in_->forward(x, ctx, (*feat_cache)[idx]);
      (*feat_cache)[idx] = cache_x;
      (*feat_idx)[0] += 1;
    } else {
      x = conv_in_->forward(x, ctx);
    }

    // mid_block
    x = mid_block_->forward(x, feat_cache, feat_idx, ctx);

    // up_blocks : pass 'first_chunk' and ctx to WanResidualUpBlock
    for (size_t i = 0; i < up_blocks_->size(); ++i) {
      if (auto res_up = up_blocks_[i]->as<WanResidualUpBlock>()) {
        x = res_up->forward(x, feat_cache, feat_idx, first_chunk, ctx);
      } else if (auto up = up_blocks_[i]->as<WanUpBlock>()) {
        x = up->forward(x, feat_cache, feat_idx, ctx);
      }
    }

    x = norm_out_->forward(x);
    x = nonlinearity_(x);

    // conv_out
    if (feat_cache) {
      int64_t idx = (*feat_idx)[0];
      torch::Tensor cache_x =
          x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(-CACHE_T, torch::indexing::None),
                   torch::indexing::Slice(),
                   torch::indexing::Slice()})
              .clone();
      if (cache_x.size(2) < 2 && (*feat_cache)[idx].defined()) {
        cache_x = torch::cat(
            {(*feat_cache)[idx]
                 .index({torch::indexing::Slice(),
                         torch::indexing::Slice(),
                         torch::indexing::Slice(-1, torch::indexing::None),
                         torch::indexing::Slice(),
                         torch::indexing::Slice()})
                 .unsqueeze(2)
                 .to(cache_x.device()),
             cache_x},
            2);
      }
      x = conv_out_->forward(x, ctx, (*feat_cache)[idx]);
      (*feat_cache)[idx] = cache_x;
      (*feat_idx)[0] += 1;
    } else {
      x = conv_out_->forward(x, ctx);
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    conv_in_->load_state_dict(state_dict.get_dict_with_prefix("conv_in."));
    mid_block_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));

    for (size_t i = 0; i < up_blocks_->size(); ++i) {
      std::string prefix = "up_blocks." + std::to_string(i) + ".";

      if (up_blocks_[i]->as<WanResidualUpBlock>()) {
        up_blocks_[i]->as<WanResidualUpBlock>()->load_state_dict(
            state_dict.get_dict_with_prefix(prefix));
      } else if (up_blocks_[i]->as<WanUpBlock>()) {
        up_blocks_[i]->as<WanUpBlock>()->load_state_dict(
            state_dict.get_dict_with_prefix(prefix));
      }
    }
    norm_out_->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    conv_out_->load_state_dict(state_dict.get_dict_with_prefix("conv_out."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    conv_in_->verify_loaded_weights("conv_in.");
    mid_block_->verify_loaded_weights("mid_block.");

    for (size_t i = 0; i < up_blocks_->size(); ++i) {
      std::string p = "up_blocks." + std::to_string(i) + ".";
      if (up_blocks_[i]->as<WanResidualUpBlock>())
        up_blocks_[i]->as<WanResidualUpBlock>()->verify_loaded_weights(p);
      else if (up_blocks_[i]->as<WanUpBlock>())
        up_blocks_[i]->as<WanUpBlock>()->verify_loaded_weights(p);
    }

    norm_out_->verify_loaded_weights("norm_out.");
    conv_out_->verify_loaded_weights("conv_out.");
  }

 private:
  WanCausalConv3D conv_in_{nullptr};
  WanMidBlock mid_block_{nullptr};
  torch::nn::ModuleList up_blocks_{nullptr};
  WanRMSNorm norm_out_{nullptr};
  WanCausalConv3D conv_out_{nullptr};
  torch::nn::Functional nonlinearity_{nullptr};
  const int64_t CACHE_T = 2;
};
TORCH_MODULE(WanVAEDecoder3D);

class AutoencoderKLWanImpl : public torch::nn::Module {
 public:
  AutoencoderKLWanImpl(const ModelContext& context)
      : args_(context.get_model_args()),
        device_(context.get_tensor_options().device()),
        dtype_(context.get_tensor_options().dtype().toScalarType()),
        // VAE spatial parallel context (always constructed, even w_split=1).
        // VaeSpatialParallel's own constructor CHECKs that a process group is
        // provided whenever vae_size > 1.
        parallel_ctx_(context.get_parallel_args().vae_size(),
                      context.get_parallel_args().dit_vae_group_,
                      context.get_tensor_options().device()) {
    if (parallel_ctx_.is_parallel()) {
      LOG(INFO) << "VAE spatial parallel enabled: w_split="
                << parallel_ctx_.w_split();
    }
    encoder_ = register_module("encoder",
                               WanVAEEncoder3D(args_.in_channels(),
                                               args_.base_dim(),
                                               args_.z_dim() * 2,
                                               args_.dim_mult(),
                                               args_.num_res_blocks(),
                                               args_.attn_scales(),
                                               args_.temperal_downsample(),
                                               args_.dropout(),
                                               args_.vae_is_residual()));

    auto decoder_temporal = args_.temperal_downsample();
    std::reverse(decoder_temporal.begin(), decoder_temporal.end());

    decoder_ = register_module("decoder",
                               WanVAEDecoder3D(args_.base_dim(),
                                               args_.z_dim(),
                                               args_.dim_mult(),
                                               args_.num_res_blocks(),
                                               args_.attn_scales(),
                                               decoder_temporal,
                                               args_.dropout(),
                                               args_.out_channels(),
                                               args_.vae_is_residual()));

    quant_conv_ =
        register_module("quant_conv",
                        WanCausalConv3D(2 * args_.z_dim(),
                                        2 * args_.z_dim(),
                                        std::vector<int64_t>{1, 1, 1}));

    post_quant_conv_ = register_module(
        "post_quant_conv",
        WanCausalConv3D(
            args_.z_dim(), args_.z_dim(), std::vector<int64_t>{1, 1, 1}));
    init_cached_conv_count();
  }

  void clear_cache() {
    conv_num_ = cached_conv_count_["decoder"];
    conv_idx_ = std::make_shared<std::vector<int64_t>>(std::vector<int64_t>{0});
    feat_map_ = std::make_shared<std::vector<torch::Tensor>>(
        std::vector<torch::Tensor>(conv_num_));

    enc_conv_num_ = cached_conv_count_["encoder"];
    enc_conv_idx_ =
        std::make_shared<std::vector<int64_t>>(std::vector<int64_t>{0});
    enc_feat_map_ = std::make_shared<std::vector<torch::Tensor>>(
        std::vector<torch::Tensor>(enc_conv_num_));
  }

  torch::Tensor encode_(const torch::Tensor& videos) {
    auto orig_dtype = videos.dtype();
    auto x = videos.to(torch::kFloat32);
    if (parallel_ctx_.is_parallel()) {
      x = parallel_ctx_.split(x);
    }
    int64_t num_frame = x.size(2);
    int64_t height = x.size(3);
    int64_t width = x.size(4);
    int64_t iter_ = 1 + (num_frame - 1) / 4;
    clear_cache();
    feat_map_ = std::make_shared<std::vector<torch::Tensor>>(
        std::vector<torch::Tensor>(conv_num_));

    std::vector<torch::Tensor> enc_outputs;
    enc_outputs.reserve(iter_);
    for (int64_t i = 0; i < iter_; ++i) {
      enc_conv_idx_ = {0};
      if (i == 0) {
        auto x_slice = x.index({torch::indexing::Slice(),
                                torch::indexing::Slice(),
                                torch::indexing::Slice(0, 1),
                                torch::indexing::Slice(),
                                torch::indexing::Slice()});
        enc_outputs.push_back(
            encoder_(x_slice, enc_feat_map_, enc_conv_idx_, parallel_ctx_));
      } else {
        int64_t start = 1 + 4 * (i - 1);
        int64_t end = std::min(1 + 4 * i, num_frame);
        auto x_slice = x.index({torch::indexing::Slice(),
                                torch::indexing::Slice(),
                                torch::indexing::Slice(start, end),
                                torch::indexing::Slice(),
                                torch::indexing::Slice()});
        enc_outputs.push_back(
            encoder_(x_slice, enc_feat_map_, enc_conv_idx_, parallel_ctx_));
      }
    }
    torch::Tensor out = torch::cat(enc_outputs, 2);
    out = quant_conv_->forward(out, parallel_ctx_);
    // Merge latent back to global W before passing to DiT transformer
    if (parallel_ctx_.is_parallel()) {
      out = parallel_ctx_.merge(out);
    }
    clear_cache();
    return out.to(orig_dtype);
  }

  AutoencoderKLOutput encode(const torch::Tensor& videos) {
    auto hidden_states = encode_(videos);
    auto posterior = DiagonalGaussianDistribution(hidden_states);
    return AutoencoderKLOutput(posterior);
  }

  DecoderOutput decode_(const torch::Tensor& latents) {
    auto orig_dtype = latents.dtype();
    torch::Tensor processed_latents = latents.to(torch::kFloat32);
    if (parallel_ctx_.is_parallel()) {
      processed_latents = parallel_ctx_.split(processed_latents);
    }
    int64_t num_frame = processed_latents.size(2);
    int64_t height = processed_latents.size(3);
    int64_t width = processed_latents.size(4);
    clear_cache();
    processed_latents =
        post_quant_conv_->forward(processed_latents, parallel_ctx_);
    std::vector<torch::Tensor> dec_outputs;
    dec_outputs.reserve(num_frame);
    for (int64_t i = 0; i < num_frame; ++i) {
      conv_idx_ = {0};
      auto x_slice = processed_latents.slice(/*dim=*/2, i, i + 1);
      dec_outputs.push_back(decoder_(x_slice,
                                     feat_map_,
                                     conv_idx_,
                                     /*first_chunk=*/(i == 0),
                                     parallel_ctx_));
    }
    torch::Tensor out = torch::cat(dec_outputs, 2);
    if (parallel_ctx_.is_parallel()) {
      out = parallel_ctx_.merge(out);
    }
    auto dec = torch::clamp(out, -1.0f, 1.0f);

    clear_cache();
    return DecoderOutput(dec.to(orig_dtype));
  }

  DecoderOutput decode(
      const torch::Tensor& latents,
      const std::optional<torch::Generator>& generator = std::nullopt) {
    auto videos = decode_(latents).sample;
    return DecoderOutput(videos);
  }

  DecoderOutput forward_(torch::Tensor sample, bool sample_posterior = false) {
    torch::Tensor x = sample;
    DiagonalGaussianDistribution posterior = encode(x).latent_dist;

    if (sample_posterior) {
      x = posterior.sample(42);
    } else {
      x = posterior.mode();
    }

    return decode(x);
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    encoder_->to(torch::kFloat32);
    decoder_->to(torch::kFloat32);
    quant_conv_->to(torch::kFloat32);
    post_quant_conv_->to(torch::kFloat32);

    for (const auto& state_dict : loader->get_state_dicts()) {
      encoder_->load_state_dict(state_dict->get_dict_with_prefix("encoder."));
      decoder_->load_state_dict(state_dict->get_dict_with_prefix("decoder."));
      quant_conv_->load_state_dict(
          state_dict->get_dict_with_prefix("quant_conv."));
      post_quant_conv_->load_state_dict(
          state_dict->get_dict_with_prefix("post_quant_conv."));
    }
    verify_loaded_weights("");
  }

  void verify_loaded_weights(const std::string& prefix) {
    encoder_->verify_loaded_weights("encoder.");
    decoder_->verify_loaded_weights("decoder.");
    quant_conv_->verify_loaded_weights("quant_conv.");
    post_quant_conv_->verify_loaded_weights("post_quant_conv.");
  }

 private:
  bool is_quant_conv_loaded_ = false;
  bool is_post_quant_conv_loaded_ = false;
  WanVAEEncoder3D encoder_{nullptr};
  WanVAEDecoder3D decoder_{nullptr};
  WanCausalConv3D quant_conv_{nullptr};
  WanCausalConv3D post_quant_conv_{nullptr};
  ModelArgs args_;
  torch::Device device_;
  torch::ScalarType dtype_;
  VaeSpatialParallel parallel_ctx_;
  int64_t tile_sample_min_height_ = 256;
  int64_t tile_sample_min_width_ = 256;
  int64_t tile_sample_stride_height_ = 192;
  int64_t tile_sample_stride_width_ = 192;
  std::unordered_map<std::string, int64_t> cached_conv_count_;
  int64_t conv_num_ = 0;
  std::shared_ptr<std::vector<int64_t>> conv_idx_;
  std::shared_ptr<std::vector<torch::Tensor>> feat_map_;
  int64_t enc_conv_num_ = 0;
  std::shared_ptr<std::vector<int64_t>> enc_conv_idx_;
  std::shared_ptr<std::vector<torch::Tensor>> enc_feat_map_;

  void init_cached_conv_count() {
    int64_t decoder_count = 0;
    int64_t encoder_count = 0;
    if (decoder_) {
      for (const auto& m : decoder_->modules(/*include_self=*/false)) {
        if (dynamic_cast<WanCausalConv3DImpl*>(m.get()) != nullptr) {
          ++decoder_count;
        }
      }
    }
    if (encoder_) {
      for (const auto& m : encoder_->modules(/*include_self=*/false)) {
        if (dynamic_cast<WanCausalConv3DImpl*>(m.get()) != nullptr) {
          ++encoder_count;
        }
      }
    }
    cached_conv_count_["decoder"] = decoder_count;
    cached_conv_count_["encoder"] = encoder_count;
  }
};
TORCH_MODULE(AutoencoderKLWan);

REGISTER_MODEL_ARGS(AutoencoderKLWan, [&] {
  LOAD_ARG_OR(z_dim, "z_dim", 16);
  LOAD_ARG_OR(base_dim, "base_dim", 96);
  LOAD_ARG_OR(num_res_blocks, "num_res_blocks", 2);
  LOAD_ARG_OR(temperal_downsample,
              "temperal_downsample",
              (std::vector<bool>{true, true, false}));
  LOAD_ARG_OR(attn_scales, "attn_scales", (std::vector<double>{}));
  LOAD_ARG_OR(dim_mult, "dim_mult", (std::vector<int64_t>{1, 2, 4, 4}));
  LOAD_ARG_OR(dropout, "dropout", 0.0f);
  LOAD_ARG_OR(in_channels, "in_channels", 3);
  LOAD_ARG_OR(out_channels, "out_channels", 3);
  LOAD_ARG_OR(vae_is_residual, "is_residual", false);
  LOAD_ARG_OR(vae_scale_factor_temporal, "scale_factor_temporal", 4);
  LOAD_ARG_OR(vae_scale_factor_spatial, "scale_factor_spatial", 8);
  LOAD_ARG_OR(latents_mean,
              "latents_mean",
              (std::vector<double>{-0.7571,
                                   -0.7089,
                                   -0.9113,
                                   0.1075,
                                   -0.1745,
                                   0.9653,
                                   -0.1517,
                                   1.5508,
                                   0.4134,
                                   -0.0715,
                                   0.5517,
                                   -0.3632,
                                   -0.1922,
                                   -0.9497,
                                   0.2503,
                                   -0.2921}));
  LOAD_ARG_OR(latents_std,
              "latents_std",
              (std::vector<double>{2.8184,
                                   1.4541,
                                   2.3275,
                                   2.6558,
                                   1.2196,
                                   1.7708,
                                   2.6052,
                                   2.0743,
                                   3.2687,
                                   2.1526,
                                   2.8652,
                                   1.5579,
                                   1.6382,
                                   1.1253,
                                   2.8251,
                                   1.916}));
});

}  // namespace xllm
