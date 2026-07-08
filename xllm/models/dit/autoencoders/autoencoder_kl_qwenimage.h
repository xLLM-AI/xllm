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
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/libs/init_npu.h>

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
#include "models/dit/autoencoders/autoencoder_kl.h"
#include "models/dit/utils/diagonal_gaussian_distribution.h"
#include "models/dit/utils/util.h"
#include "models/dit/utils/vae_spatial_parallel.h"
#include "models/model_registry.h"

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

// VAE model compatible with huggingface weights
// ref to:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py

namespace xllm {

using xllm::dit::VaeSpatialParallel;

class QwenImageBaseModule : public torch::nn::Module {
 public:
  virtual torch::Tensor forward(
      const torch::Tensor& x,
      std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
      std::shared_ptr<std::vector<int64_t>> feat_idx,
      VaeSpatialParallel& ctx) = 0;
  virtual ~QwenImageBaseModule() = default;
};

const int64_t CACHE_T = 2;

class QwenImageCausalConv3dImpl : public torch::nn::Module {
 public:
  QwenImageCausalConv3dImpl(const ModelContext& context,
                            int64_t in_channels,
                            int64_t out_channels,
                            torch::IntArrayRef kernel_size,
                            torch::IntArrayRef stride = 1,
                            torch::IntArrayRef padding = 0) {
    conv_ = register_module(
        "conv",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(0)
                .bias(true)));

    auto p = padding.size() == 1
                 ? std::vector<int64_t>{padding[0], padding[0], padding[0]}
                 : std::vector<int64_t>(padding.begin(), padding.end());

    padding_ = {p[2], p[2], p[1], p[1], 2 * p[0], 0};
    stride_ = stride.vec();
  }

  torch::Tensor forward(const torch::Tensor& x,
                        VaeSpatialParallel& ctx,
                        const torch::Tensor& cache_x = torch::Tensor()) {
    auto padding_vec = padding_;
    auto result_x = x;

    // Halo exchange is only needed for 3×3 spatial convolutions (kernel=3,
    // pad=1). For 1×1 convolutions (e.g. quant_conv) the padding is {0,0,0}
    // so this condition is false and no exchange occurs.
    // padding_ is the 6-element F::pad format:
    //   {W_left, W_right, H_top, H_bottom, T_front, T_back}
    // padding_[1] is W_right padding, padding_[2] is H_top padding.
    // Both == 1 means spatial kernel is 3×3 with padding=1.
    bool use_halo = (ctx.is_parallel() && padding_[1] == 1 && padding_[2] == 1);

    // Temporal padding (causal) — must come BEFORE halo exchange so cache
    // and input have matching spatial dims.
    if (cache_x.defined() && padding_vec[4] > 0) {
      auto device_x = result_x.device();
      auto cache_device = cache_x.to(device_x);
      result_x = torch::cat({cache_device, result_x}, 2);
      padding_vec[4] -= cache_x.size(2);
    }

    // Spatial parallel: halo exchange provides W neighbors.
    // Zero W padding from F::pad (exchange already handles it).
    // H padding stays in F::pad since conv3d has padding=0.
    if (use_halo) {
      result_x = ctx.exchange(result_x, /*pad=*/true);
      padding_vec[0] = 0;  // W left  — handled by exchange
      padding_vec[1] = 0;  // W right — handled by exchange
    }

    result_x = torch::nn::functional::pad(
        result_x, torch::nn::functional::PadFuncOptions(padding_vec));

    return conv_(result_x);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight", conv_->weight, is_weight_loaded_);
    weight::load_weight(state_dict, "bias", conv_->bias, is_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(is_bias_loaded_) << "weight is not loaded for " << prefix + "bias";
  }

 private:
  bool is_weight_loaded_{false};
  bool is_bias_loaded_{false};
  torch::nn::Conv3d conv_ = nullptr;
  std::vector<int64_t> padding_;
  std::vector<int64_t> stride_;
};

TORCH_MODULE(QwenImageCausalConv3d);

class QwenImageRMS_normImpl : public torch::nn::Module {
 public:
  QwenImageRMS_normImpl(const ModelContext& context,
                        int64_t dim,
                        bool channel_first = true,
                        bool images = true,
                        bool is_bias = false,
                        bool fused = false)
      : channel_first_(channel_first), fused_(fused), is_bias_(is_bias) {
    auto broadcastable_dims =
        images ? std::vector<int64_t>{1, 1} : std::vector<int64_t>{1, 1, 1};
    auto shape = std::vector<int64_t>{dim};
    if (channel_first) {
      shape.insert(
          shape.end(), broadcastable_dims.begin(), broadcastable_dims.end());
    }

    scale_ = std::sqrt(dim);
    weight_ = register_parameter("gamma", torch::ones(shape));

    if (is_bias_) {
      bias_ = register_parameter("bias", torch::zeros(shape));
    }
  }

  torch::Tensor forward(const torch::Tensor& x) {
    if (fused_) {
      auto [output, rstd] =
          at_npu::native::custom_ops::npu_rms_norm(x, weight_, 0);

      if (is_bias_ && bias_.defined()) {
        output = output + bias_.to(output.device());
      }
      return output;
    } else {
      torch::Tensor normalized =
          torch::nn::functional::normalize(
              x.to(torch::kFloat),
              torch::nn::functional::NormalizeFuncOptions().dim(
                  channel_first_ ? 1 : -1))
              .to(x.dtype());

      if (is_bias_) {
        return normalized * scale_ * weight_ + bias_;
      } else {
        return normalized * scale_ * weight_;
      }
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "gamma", weight_, is_weight_loaded_);
    if (is_bias_) {
      weight::load_weight(state_dict, "bias", bias_, is_bias_loaded_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(!is_bias_ || is_bias_loaded_)
        << "bias is not loaded for " << prefix + "bias";
  }

 private:
  bool channel_first_;
  double scale_;
  bool is_bias_;
  bool fused_;
  bool is_weight_loaded_{false};
  bool is_bias_loaded_{false};
  torch::Tensor weight_;
  torch::Tensor bias_;
  torch::TensorOptions options_;
};

TORCH_MODULE(QwenImageRMS_norm);

class QwenImageUpsampleImpl : public torch::nn::Module {
 public:
  QwenImageUpsampleImpl(
      const ModelContext& context,
      const torch::nn::functional::InterpolateFuncOptions options)
      : options_(options) {}

  torch::Tensor forward(const torch::Tensor& x) {
    // auto result = upsample_(x.to(torch::kFloat));
    auto result =
        torch::nn::functional::interpolate(x.to(torch::kFloat), options_);
    return result.to(x.dtype());
  }

 private:
  torch::nn::functional::InterpolateFuncOptions options_;
  torch::nn::Upsample upsample_ = nullptr;
};

TORCH_MODULE(QwenImageUpsample);

class QwenImageResampleImpl : public QwenImageBaseModule {
 public:
  QwenImageResampleImpl(const ModelContext& context,
                        int64_t dim,
                        const std::string& mode)
      : dim_(dim), mode_(mode) {
    if (mode_ == "upsample2d") {
      resample_ = register_module(
          "resample",
          torch::nn::Sequential(
              QwenImageUpsample(context,
                                torch::nn::functional::InterpolateFuncOptions()
                                    .scale_factor(std::vector<double>{2.0, 2.0})
                                    .mode(torch::kNearestExact)),
              torch::nn::Conv2d(
                  torch::nn::Conv2dOptions(/*in_channels=*/dim,
                                           /*out_channels=*/dim / 2,
                                           /*kernel_size=*/3)
                      .padding(1))));
    } else if (mode_ == "upsample3d") {
      resample_ = register_module(
          "resample",
          torch::nn::Sequential(
              QwenImageUpsample(context,
                                torch::nn::functional::InterpolateFuncOptions()
                                    .scale_factor(std::vector<double>{2.0, 2.0})
                                    .mode(torch::kNearestExact)),
              torch::nn::Conv2d(
                  torch::nn::Conv2dOptions(/*in_channels=*/dim,
                                           /*out_channels=*/dim / 2,
                                           /*kernel_size=*/3)
                      .padding(1))));

      time_conv_ = register_module(
          "time_conv",
          QwenImageCausalConv3d(context,
                                /*in_channels=*/dim,
                                /*out_channels=*/dim * 2,
                                /*kernel_size=*/torch::IntArrayRef{3, 1, 1},
                                /*stride=*/torch::IntArrayRef{1, 1, 1},
                                /*padding=*/torch::IntArrayRef{1, 0, 0}));

    } else if (mode_ == "downsample2d") {
      resample_ = register_module(
          "resample",
          torch::nn::Sequential(
              torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions({/*left=*/0,
                                                                /*right=*/1,
                                                                /*top=*/0,
                                                                /*bottom=*/1})),
              torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/dim,
                                                         /*out_channels=*/dim,
                                                         /*kernel_size=*/3)
                                    .stride(2))));
    } else if (mode_ == "downsample3d") {
      resample_ = register_module(
          "resample",
          torch::nn::Sequential(
              torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions({/*left=*/0,
                                                                /*right=*/1,
                                                                /*top=*/0,
                                                                /*bottom=*/1})),
              torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/dim,
                                                         /*out_channels=*/dim,
                                                         /*kernel_size=*/3)
                                    .stride(2))));
      time_conv_ = register_module(
          "time_conv",
          QwenImageCausalConv3d(context,
                                /*in_channels=*/dim,
                                /*out_channels=*/dim,
                                /*kernel_size=*/torch::IntArrayRef{3, 1, 1},
                                /*stride=*/torch::IntArrayRef{2, 1, 1},
                                /*padding=*/torch::IntArrayRef{0, 0, 0}));
    } else {
      resample_ = register_module("resample",
                                  torch::nn::Sequential(torch::nn::Identity()));
    }

    rep_tensor_ = register_parameter("rep_tensor", torch::tensor({-999.0}));
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        VaeSpatialParallel& ctx) override {
    if (feat_idx == nullptr) {
      feat_idx =
          std::make_shared<std::vector<int64_t>>(std::vector<int64_t>{0});
    }
    auto sizes = x.sizes();
    auto b = sizes[0], c = sizes[1], t = sizes[2], h = sizes[3], w = sizes[4];
    auto result_x = x;

    if (mode_ == "upsample3d" && feat_cache && feat_idx) {
      auto idx = (*feat_idx)[0];

      if (idx < feat_cache->size() && feat_cache->at(idx).defined()) {
        auto cache_x = result_x
                           .index({torch::indexing::Slice(),
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice(
                                       -CACHE_T, torch::indexing::None)})
                           .clone();

        if (cache_x.size(2) < 2 && feat_cache->at(idx).defined() &&
            !torch::equal(rep_tensor_, feat_cache->at(idx))) {
          auto last_frame =
              feat_cache->at(idx)
                  .index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(-1, torch::indexing::None)})
                  .unsqueeze(2)
                  .to(cache_x.device());
          cache_x = torch::cat({last_frame, cache_x}, 2);
        }
        if (cache_x.size(2) < 2 && feat_cache->at(idx).defined() &&
            torch::equal(rep_tensor_, feat_cache->at(idx))) {
          cache_x = torch::cat(
              {torch::zeros_like(cache_x).to(cache_x.device()), cache_x}, 2);
        }
        if (torch::equal(rep_tensor_, feat_cache->at(idx))) {
          result_x = time_conv_->forward(result_x, ctx);
        } else {
          result_x = time_conv_->forward(result_x, ctx, feat_cache->at(idx));
        }
        feat_cache->at(idx) = cache_x;
        (*feat_idx)[0]++;

        result_x = result_x.reshape({b, 2, c, t, h, w});
        result_x = torch::stack({result_x.index({torch::indexing::Slice(), 0}),
                                 result_x.index({torch::indexing::Slice(), 1})},
                                3);
        result_x = result_x.reshape({b, c, t * 2, h, w});
      } else {
        feat_cache->at(idx) = rep_tensor_;
        (*feat_idx)[0]++;
      }
    }

    t = result_x.size(2);

    bool is_upsample = (mode_ == "upsample2d" || mode_ == "upsample3d");
    bool is_downsample = (mode_ == "downsample2d" || mode_ == "downsample3d");

    // Upsample: exchange halo columns before spatial upsampling
    if (ctx.is_parallel() && is_upsample) {
      result_x = ctx.exchange(result_x, /*pad=*/false);
      w = result_x.size(-1);
    }
    // Downsample: merge spatial slices before spatial downsampling
    if (ctx.is_parallel() && is_downsample) {
      result_x = ctx.merge(result_x);
      w = result_x.size(-1);
      h = result_x.size(-2);
    }

    result_x = result_x.permute({0, 2, 1, 3, 4}).reshape({b * t, c, h, w});
    result_x = resample_->forward(result_x);
    result_x =
        result_x
            .view({b, t, result_x.size(1), result_x.size(2), result_x.size(3)})
            .permute({0, 2, 1, 3, 4});

    // Upsample: trim excess columns introduced by halo exchange.
    //
    // Before upsample: exchange(pad=false) adds 1 neighbor column per side.
    // The interpolate 2× doubles each column, and the following Conv2d(k=3,
    // pad=1) preserves size. So each halo column produces 2 excess columns.
    //
    // Trim 2 from each side where a neighbor contributed halo:
    //   - middle rank (has_left && has_right): trim 2 from each side
    //   - first rank  (!has_left):           trim 2 from right only
    //   - last rank   (!has_right):          trim 2 from left only
    if (ctx.is_parallel() && is_upsample) {
      auto cur_w = result_x.size(-1);
      if (ctx.has_left() && ctx.has_right()) {
        result_x = result_x.slice(/*dim=*/-1, 2, cur_w - 2);
      } else if (!ctx.has_left()) {
        result_x = result_x.slice(/*dim=*/-1, 0, cur_w - 2);
      } else {
        result_x = result_x.slice(/*dim=*/-1, 2, cur_w);
      }
    }
    // Downsample: split full result back to local slices
    if (ctx.is_parallel() && is_downsample) {
      result_x = ctx.split(result_x);
    }

    if (mode_ == "downsample3d" && feat_cache && feat_idx) {
      auto idx = (*feat_idx)[0];

      if (idx < feat_cache->size() && feat_cache->at(idx).defined()) {
        auto cache_x =
            result_x
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(-1, torch::indexing::None)})
                .clone();

        auto concat_x = torch::cat(
            {feat_cache->at(idx).index(
                 {torch::indexing::Slice(),
                  torch::indexing::Slice(),
                  torch::indexing::Slice(-1, torch::indexing::None)}),
             result_x},
            2);

        result_x = time_conv_->forward(concat_x, ctx);
        feat_cache->at(idx) = cache_x;
        (*feat_idx)[0]++;
      } else {
        feat_cache->at(idx) = result_x.clone();
        (*feat_idx)[0]++;
      }
    }

    return result_x;
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
  std::string mode_;
  bool is_weight_loaded_{false};
  bool is_bias_loaded_{false};
  torch::Tensor rep_tensor_;
  torch::nn::Sequential resample_{nullptr};
  QwenImageCausalConv3d time_conv_{nullptr};
};

TORCH_MODULE(QwenImageResample);

class QwenImageResidualBlockImpl : public QwenImageBaseModule {
 public:
  QwenImageResidualBlockImpl(const ModelContext& context,
                             int64_t in_dim,
                             int64_t out_dim,
                             double dropout = 0.0,
                             const std::string& non_linearity = "silu")
      : in_dim_(in_dim), out_dim_(out_dim) {
    activation_ = register_module("silu", torch::nn::SiLU());

    norm1_ = register_module("norm1",
                             QwenImageRMS_norm(context,
                                               in_dim,
                                               /*channel_first=*/true,
                                               /*images=*/false,
                                               /*is_bias=*/false,
                                               /*fused=*/false));
    conv1_ = register_module(
        "conv1",
        QwenImageCausalConv3d(context,
                              in_dim,
                              out_dim,
                              /*kernel_size=*/torch::IntArrayRef{3, 3, 3},
                              /*stride=*/torch::IntArrayRef{1, 1, 1},
                              /*padding=*/torch::IntArrayRef{1, 1, 1}));
    norm2_ = register_module("norm2",
                             QwenImageRMS_norm(context,
                                               out_dim,
                                               /*channel_first=*/true,
                                               /*images=*/false,
                                               /*is_bias=*/false,
                                               /*fused=*/false));
    dropout_layer_ = register_module("dropout", torch::nn::Dropout(dropout));
    conv2_ = register_module(
        "conv2",
        QwenImageCausalConv3d(context,
                              out_dim,
                              out_dim,
                              /*kernel_size=*/torch::IntArrayRef{3, 3, 3},
                              /*stride=*/torch::IntArrayRef{1, 1, 1},
                              /*padding=*/torch::IntArrayRef{1, 1, 1}));

    if (in_dim != out_dim) {
      conv_shortcut_ = register_module(
          "conv_shortcut",
          QwenImageCausalConv3d(context,
                                in_dim,
                                out_dim,
                                /*kernel_size=*/torch::IntArrayRef{1, 1, 1},
                                /*stride=*/torch::IntArrayRef{1, 1, 1},
                                /*padding=*/torch::IntArrayRef{0, 0, 0}));
    } else {
      identity_ = register_module("conv_shortcut", torch::nn::Identity());
    }
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        VaeSpatialParallel& ctx) override {
    if (feat_idx == nullptr) {
      feat_idx =
          std::make_shared<std::vector<int64_t>>(std::vector<int64_t>{0});
    }
    torch::Tensor h = torch::empty({0});
    if (conv_shortcut_) {
      h = conv_shortcut_->forward(x, ctx);
    } else {
      h = identity_->forward(x);
    }
    auto result_x = x;

    result_x = norm1_->forward(result_x);
    result_x = activation_->forward(result_x);

    if (feat_cache && feat_idx) {
      auto idx = (*feat_idx)[0];
      auto cache_x =
          result_x
              .index({torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(-CACHE_T, torch::indexing::None)})
              .clone();
      if (cache_x.size(2) < 2 && feat_cache->at(idx).defined()) {
        auto last_frame =
            feat_cache->at(idx)
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(-1, torch::indexing::None)})
                .unsqueeze(2)
                .to(cache_x.device());
        cache_x = torch::cat({last_frame, cache_x}, 2);
      }

      result_x = conv1_->forward(result_x, ctx, feat_cache->at(idx));
      feat_cache->at(idx) = cache_x;
      (*feat_idx)[0]++;
    } else {
      result_x = conv1_->forward(result_x, ctx);
    }
    result_x = norm2_->forward(result_x);
    result_x = activation_->forward(result_x);
    result_x = dropout_layer_->forward(result_x);

    if (feat_cache && feat_idx) {
      auto idx = (*feat_idx)[0];
      auto cache_x =
          result_x
              .index({torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(-CACHE_T, torch::indexing::None)})
              .clone();

      if (cache_x.size(2) < 2 && feat_cache->at(idx).defined()) {
        auto last_frame =
            feat_cache->at(idx)
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(-1, torch::indexing::None)})
                .unsqueeze(2)
                .to(cache_x.device());
        cache_x = torch::cat({last_frame, cache_x}, 2);
      }
      result_x = conv2_->forward(result_x, ctx, feat_cache->at(idx));
      feat_cache->at(idx) = cache_x;
      (*feat_idx)[0]++;
    } else {
      result_x = conv2_->forward(result_x, ctx);
    }

    return result_x + h;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));

    conv1_->load_state_dict(state_dict.get_dict_with_prefix("conv1."));

    conv2_->load_state_dict(state_dict.get_dict_with_prefix("conv2."));

    if (conv_shortcut_) {
      conv_shortcut_->load_state_dict(
          state_dict.get_dict_with_prefix("conv_shortcut."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    norm1_->verify_loaded_weights("norm1.");
    norm2_->verify_loaded_weights("norm2.");
    conv1_->verify_loaded_weights("conv1.");
    conv2_->verify_loaded_weights("conv2.");
    if (conv_shortcut_) {
      conv_shortcut_->verify_loaded_weights("conv_shortcut.");
    }
  }

 private:
  int64_t in_dim_, out_dim_;
  QwenImageRMS_norm norm1_{nullptr}, norm2_{nullptr};
  QwenImageCausalConv3d conv1_{nullptr}, conv2_{nullptr};
  QwenImageCausalConv3d conv_shortcut_{nullptr};
  torch::nn::Dropout dropout_layer_{nullptr};
  torch::nn::SiLU activation_{nullptr};
  torch::nn::Identity identity_{nullptr};
};

TORCH_MODULE(QwenImageResidualBlock);

class QwenImageAttentionBlockImpl : public QwenImageBaseModule {
 public:
  QwenImageAttentionBlockImpl(const ModelContext& context, int64_t dim)
      : dim_(dim) {
    norm_ = register_module("norm",
                            QwenImageRMS_norm(context,
                                              dim,
                                              /*channel_first=*/true,
                                              /*images=*/true,
                                              /*is_bias=*/false,
                                              /*fused=*/false));
    to_qkv_ = register_module(
        "to_qkv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/dim,
                                                   /*out_channels=*/dim * 3,
                                                   /*kernel_size=*/1)));
    proj_ = register_module(
        "proj",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/dim,
                                                   /*out_channels=*/dim,
                                                   /*kernel_size=*/1)));
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        VaeSpatialParallel& ctx) override {
    (void)ctx;  // attention block has no conv3d sub-calls
    if (feat_idx == nullptr) {
      feat_idx =
          std::make_shared<std::vector<int64_t>>(std::vector<int64_t>{0});
    }
    auto identity = x;
    auto sizes = x.sizes();
    auto b = sizes[0], c = sizes[1], t = sizes[2], h = sizes[3], w = sizes[4];

    auto reshaped_x = x.permute({0, 2, 1, 3, 4}).reshape({b * t, c, h, w});
    reshaped_x = norm_->forward(reshaped_x);

    auto qkv = to_qkv_->forward(reshaped_x);

    torch::Tensor q, k, v;
    if (ctx.is_parallel()) {
      // Gather K/V in spatial domain before BNSD reshape
      auto spatial_chunks = qkv.chunk(3, 1);
      auto k_sp = ctx.merge(spatial_chunks[1].unsqueeze(2)).squeeze(2);
      auto v_sp = ctx.merge(spatial_chunks[2].unsqueeze(2)).squeeze(2);
      int64_t W_global = k_sp.size(-1);
      q = spatial_chunks[0]
              .reshape({b * t, 1, c, h * w})
              .permute({0, 1, 3, 2})
              .contiguous();
      k = k_sp.reshape({b * t, 1, c, h * W_global})
              .permute({0, 1, 3, 2})
              .contiguous();
      v = v_sp.reshape({b * t, 1, c, h * W_global})
              .permute({0, 1, 3, 2})
              .contiguous();
    } else {
      qkv = qkv.reshape({b * t, 1, c * 3, h * w});
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
        /*scale=*/pow(c, -0.5),
        /*keep_prob=*/1.0,
        /*pre_tockens=*/65535,
        /*next_tockens=*/65535);
    auto attn_output = std::get<0>(results);
    attn_output =
        attn_output.squeeze(1).permute({0, 2, 1}).reshape({b * t, c, h, w});

    auto output = proj_->forward(attn_output);

    output = output.view({b, t, c, h, w}).permute({0, 2, 1, 3, 4});

    return output + identity;
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
  QwenImageRMS_norm norm_{nullptr};
  torch::nn::Conv2d to_qkv_{nullptr};
  torch::nn::Conv2d proj_{nullptr};
  bool is_qkv_weight_loaded_{false};
  bool is_qkv_bias_loaded_{false};
  bool is_proj_weight_loaded_{false};
  bool is_proj_bias_loaded_{false};
};

TORCH_MODULE(QwenImageAttentionBlock);

class QwenImageMidBlockImpl : public torch::nn::Module {
 public:
  QwenImageMidBlockImpl(const ModelContext& context,
                        int64_t dim,
                        double dropout = 0.0,
                        const std::string& non_linearity = "silu",
                        int64_t num_layers = 1)
      : dim_(dim) {
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    attentions_ = register_module("attentions", torch::nn::ModuleList());

    auto resnet_0 =
        QwenImageResidualBlock(context, dim, dim, dropout, non_linearity);
    resnets_->push_back(resnet_0);

    for (int64_t i = 0; i < num_layers; i++) {
      auto attention = QwenImageAttentionBlock(context, dim);
      attentions_->push_back(attention);

      auto resnet =
          QwenImageResidualBlock(context, dim, dim, dropout, non_linearity);
      resnets_->push_back(resnet);
    }
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        VaeSpatialParallel& ctx) {
    if (feat_idx == nullptr) {
      feat_idx =
          std::make_shared<std::vector<int64_t>>(std::vector<int64_t>{0});
    }
    auto result_x = x;

    result_x = resnets_[0]->as<QwenImageResidualBlock>()->forward(
        result_x, feat_cache, feat_idx, ctx);

    for (size_t i = 0; i < attentions_->size(); i++) {
      result_x = attentions_[i]->as<QwenImageAttentionBlock>()->forward(
          result_x, nullptr, nullptr, ctx);
      result_x = resnets_[i + 1]->as<QwenImageResidualBlock>()->forward(
          result_x, feat_cache, feat_idx, ctx);
    }

    return result_x;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); i++) {
      auto prefix = "resnets." + std::to_string(i) + ".";
      resnets_[i]->as<QwenImageResidualBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix(prefix));
    }

    for (size_t i = 0; i < attentions_->size(); i++) {
      auto prefix = "attentions." + std::to_string(i) + ".";
      attentions_[i]->as<QwenImageAttentionBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix(prefix));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < resnets_->size(); i++) {
      auto prefix = "resnets." + std::to_string(i) + ".";
      resnets_[i]->as<QwenImageResidualBlock>()->verify_loaded_weights(prefix);
    }

    for (size_t i = 0; i < attentions_->size(); i++) {
      auto prefix = "attentions." + std::to_string(i) + ".";
      attentions_[i]->as<QwenImageAttentionBlock>()->verify_loaded_weights(
          prefix);
    }
  }

 private:
  int64_t dim_;
  torch::nn::ModuleList resnets_;
  torch::nn::ModuleList attentions_;
};

TORCH_MODULE(QwenImageMidBlock);

class QwenImageEncoder3dImpl : public torch::nn::Module {
 public:
  QwenImageEncoder3dImpl(const ModelContext& context,
                         int64_t dim = 128,
                         int64_t z_dim = 4,
                         std::vector<int64_t> dim_mult = {1, 2, 4, 4},
                         int64_t num_res_blocks = 2,
                         std::vector<double> attn_scales = {},
                         std::vector<bool> temperal_downsample = {true,
                                                                  true,
                                                                  false},
                         double dropout = 0.0,
                         int64_t input_channels = 3,
                         const std::string& non_linearity = "silu")
      : dim_(dim),
        z_dim_(z_dim),
        dim_mult_(dim_mult),
        num_res_blocks_(num_res_blocks),
        attn_scales_(attn_scales),
        temperal_downsample_(temperal_downsample) {
    nonlinearity_ = register_module("silu", torch::nn::SiLU());

    std::vector<int64_t> dims = {dim * 1};
    for (auto u : dim_mult_) {
      dims.push_back(dim * u);
    }

    double scale = 1.0;

    conv_in_ = register_module(
        "conv_in",
        QwenImageCausalConv3d(context,
                              input_channels,
                              /*out_channels=*/dims[0],
                              /*kernel_size=*/torch::IntArrayRef{3, 3, 3},
                              /*stride=*/torch::IntArrayRef{1, 1, 1},
                              /*padding=*/torch::IntArrayRef{1, 1, 1}));

    down_blocks_ = register_module("down_blocks", torch::nn::ModuleList());

    size_t counter = 0;
    for (size_t i = 0; i < dims.size() - 1; i++) {
      int64_t in_dim = dims[i];
      int64_t out_dim = dims[i + 1];

      for (int64_t j = 0; j < num_res_blocks_; j++) {
        auto res_block = QwenImageResidualBlock(
            context, in_dim, out_dim, dropout, non_linearity);
        down_blocks_->push_back(res_block);
        resnet_blocks_idx_.push_back(counter);
        counter += 1;

        if (std::find(attn_scales_.begin(), attn_scales_.end(), scale) !=
            attn_scales_.end()) {
          auto attn_block = QwenImageAttentionBlock(context, out_dim);
          down_blocks_->push_back(attn_block);
          attention_blocks_idx_.push_back(counter);
          counter += 1;
        }
        in_dim = out_dim;
      }

      if (i != dim_mult_.size() - 1) {
        std::string mode =
            temperal_downsample_[i] ? "downsample3d" : "downsample2d";
        auto downsample = QwenImageResample(context, out_dim, mode);
        down_blocks_->push_back(downsample);
        resample_blocks_idx_.push_back(counter);
        counter += 1;
        scale /= 2.0;
      }
    }

    mid_block_ = register_module(
        "mid_block",
        QwenImageMidBlock(
            context, dims.back(), dropout, non_linearity, /*num_layers=*/1));

    norm_out_ = register_module("norm_out",
                                QwenImageRMS_norm(context,
                                                  dims.back(),
                                                  /*channel_first=*/true,
                                                  /*images=*/false,
                                                  /*is_bias=*/false,
                                                  /*fused=*/false));
    conv_out_ = register_module(
        "conv_out",
        QwenImageCausalConv3d(context,
                              /*in_channels=*/dims.back(),
                              /*out_channels=*/z_dim,
                              /*kernel_size=*/torch::IntArrayRef{3, 3, 3},
                              /*stride=*/torch::IntArrayRef{1, 1, 1},
                              /*padding=*/torch::IntArrayRef{1, 1, 1}));
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        VaeSpatialParallel& ctx) {
    if (feat_idx == nullptr) {
      feat_idx =
          std::make_shared<std::vector<int64_t>>(std::vector<int64_t>{0});
    }
    torch::Tensor result_x;

    if (feat_cache && feat_idx) {
      auto idx = (*feat_idx)[0];
      auto cache_x =
          x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(-CACHE_T, torch::indexing::None)})
              .clone();

      if (cache_x.size(2) < 2 && feat_cache->at(idx).defined()) {
        auto last_frame =
            feat_cache->at(idx)
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(-1, torch::indexing::None)})
                .unsqueeze(2)
                .to(cache_x.device());
        cache_x = torch::cat({last_frame, cache_x}, 2);
      }
      result_x = conv_in_->forward(x, ctx, feat_cache->at(idx));
      feat_cache->at(idx) = cache_x;
      (*feat_idx)[0]++;
    } else {
      result_x = conv_in_->forward(x, ctx);
    }

    int64_t counter = 0;
    for (auto& layer : *down_blocks_) {
      if (feat_cache) {
        counter = counter + 1;
        result_x =
            std::dynamic_pointer_cast<QwenImageBaseModule>(layer)->forward(
                result_x, feat_cache, feat_idx, ctx);
      } else {
        result_x =
            std::dynamic_pointer_cast<QwenImageBaseModule>(layer)->forward(
                result_x,
                nullptr,
                std::make_shared<std::vector<int64_t>>(std::vector<int64_t>{0}),
                ctx);
      }
    }

    result_x = mid_block_->forward(result_x, feat_cache, feat_idx, ctx);

    result_x = norm_out_->forward(result_x);
    result_x = nonlinearity_->forward(result_x);

    if (feat_cache && feat_idx) {
      auto idx = (*feat_idx)[0];
      auto cache_x =
          result_x
              .index({torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(-CACHE_T, torch::indexing::None)})
              .clone();

      if (cache_x.size(2) < 2 && idx < feat_cache->size() &&
          feat_cache->at(idx).defined()) {
        auto last_frame =
            feat_cache->at(idx)
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(-1, torch::indexing::None)})
                .unsqueeze(2)
                .to(cache_x.device());
        cache_x = torch::cat({last_frame, cache_x}, 2);
      }

      result_x = conv_out_->forward(result_x, ctx, feat_cache->at(idx));
      feat_cache->at(idx) = cache_x;
      (*feat_idx)[0]++;
    } else {
      result_x = conv_out_->forward(result_x, ctx);
    }

    return result_x;
  }

  void load_state_dict(const StateDict& state_dict) {
    conv_in_->load_state_dict(state_dict.get_dict_with_prefix("conv_in."));

    for (size_t resnet_idx : resnet_blocks_idx_) {
      down_blocks_[resnet_idx]->as<QwenImageResidualBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("down_blocks." +
                                          std::to_string(resnet_idx) + "."));
    }

    for (size_t attention_idx : attention_blocks_idx_) {
      down_blocks_[attention_idx]
          ->as<QwenImageAttentionBlock>()
          ->load_state_dict(state_dict.get_dict_with_prefix(
              "down_blocks." + std::to_string(attention_idx) + "."));
    }

    for (size_t resample_idx : resample_blocks_idx_) {
      down_blocks_[resample_idx]->as<QwenImageResample>()->load_state_dict(
          state_dict.get_dict_with_prefix("down_blocks." +
                                          std::to_string(resample_idx) + "."));
    }

    mid_block_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
    norm_out_->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    conv_out_->load_state_dict(state_dict.get_dict_with_prefix("conv_out."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    conv_in_->verify_loaded_weights("conv_in.");
    for (size_t resnet_idx : resnet_blocks_idx_) {
      down_blocks_[resnet_idx]
          ->as<QwenImageResidualBlock>()
          ->verify_loaded_weights(std::to_string(resnet_idx) + ".");
    }

    for (size_t attention_idx : attention_blocks_idx_) {
      down_blocks_[attention_idx]
          ->as<QwenImageAttentionBlock>()
          ->verify_loaded_weights(std::to_string(attention_idx) + ".");
    }

    for (size_t resample_idx : resample_blocks_idx_) {
      down_blocks_[resample_idx]
          ->as<QwenImageResample>()
          ->verify_loaded_weights(std::to_string(resample_idx) + ".");
    }
    mid_block_->verify_loaded_weights("mid_block.");
    norm_out_->verify_loaded_weights("norm_out.");
    conv_out_->verify_loaded_weights("conv_out.");
  }

 private:
  int64_t dim_, z_dim_;
  std::vector<int64_t> dim_mult_;
  std::vector<size_t> resnet_blocks_idx_;
  std::vector<size_t> attention_blocks_idx_;
  std::vector<size_t> resample_blocks_idx_;
  int64_t num_res_blocks_;
  std::vector<double> attn_scales_;
  std::vector<bool> temperal_downsample_;

  torch::nn::SiLU nonlinearity_{nullptr};
  QwenImageCausalConv3d conv_in_{nullptr};
  torch::nn::ModuleList down_blocks_{nullptr};
  QwenImageMidBlock mid_block_{nullptr};
  QwenImageRMS_norm norm_out_{nullptr};
  QwenImageCausalConv3d conv_out_{nullptr};
};

TORCH_MODULE(QwenImageEncoder3d);

class QwenImageUpBlockImpl : public torch::nn::Module {
 public:
  QwenImageUpBlockImpl(const ModelContext& context,
                       int64_t in_dim,
                       int64_t out_dim,
                       int64_t num_res_blocks,
                       double dropout = 0.0,
                       const std::string& upsample_mode = "",
                       const std::string& non_linearity = "silu")
      : in_dim_(in_dim), out_dim_(out_dim) {
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    int64_t current_dim = in_dim;

    for (int64_t i = 0; i < num_res_blocks + 1; i++) {
      auto resnet = QwenImageResidualBlock(
          context, current_dim, out_dim, dropout, non_linearity);
      resnets_->push_back(resnet);
      current_dim = out_dim;
    }

    if (!upsample_mode.empty()) {
      upsamplers_ = register_module("upsamplers", torch::nn::ModuleList());
      auto upsample = QwenImageResample(context, out_dim, upsample_mode);
      upsamplers_->push_back(upsample);
    }
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        VaeSpatialParallel& ctx) {
    if (feat_idx == nullptr) {
      feat_idx =
          std::make_shared<std::vector<int64_t>>(std::vector<int64_t>{0});
    }

    auto result_x = x;

    for (auto& resnet : *resnets_) {
      if (feat_cache && feat_idx) {
        result_x =
            std::dynamic_pointer_cast<QwenImageBaseModule>(resnet)->forward(
                result_x, feat_cache, feat_idx, ctx);
      } else {
        result_x =
            std::dynamic_pointer_cast<QwenImageBaseModule>(resnet)->forward(
                result_x,
                nullptr,
                std::make_shared<std::vector<int64_t>>(std::vector<int64_t>{0}),
                ctx);
      }
    }

    if (upsamplers_) {
      if (feat_cache && feat_idx) {
        result_x =
            std::dynamic_pointer_cast<QwenImageBaseModule>(upsamplers_[0])
                ->forward(result_x, feat_cache, feat_idx, ctx);
      } else {
        result_x =
            std::dynamic_pointer_cast<QwenImageBaseModule>(upsamplers_[0])
                ->forward(result_x,
                          nullptr,
                          std::make_shared<std::vector<int64_t>>(
                              std::vector<int64_t>{0}),
                          ctx);
      }
    }

    return result_x;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); i++) {
      auto prefix = "resnets." + std::to_string(i) + ".";
      resnets_[i]->as<QwenImageResidualBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix(prefix));
    }

    if (upsamplers_) {
      upsamplers_[0]->as<QwenImageResample>()->load_state_dict(
          state_dict.get_dict_with_prefix("upsamplers.0."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < resnets_->size(); i++) {
      auto prefix = "resnets." + std::to_string(i) + ".";
      resnets_[i]->as<QwenImageResidualBlock>()->verify_loaded_weights(prefix);
    }

    if (upsamplers_) {
      upsamplers_[0]->as<QwenImageResample>()->verify_loaded_weights(
          "upsamplers.0.");
    }
  }

 private:
  int64_t in_dim_, out_dim_;
  torch::nn::ModuleList resnets_{nullptr};
  torch::nn::ModuleList upsamplers_{nullptr};
};

TORCH_MODULE(QwenImageUpBlock);

class QwenImageDecoder3dImpl : public torch::nn::Module {
 public:
  QwenImageDecoder3dImpl(const ModelContext& context,
                         int64_t dim = 128,
                         int64_t z_dim = 4,
                         std::vector<int64_t> dim_mult = {1, 2, 4, 4},
                         int64_t num_res_blocks = 2,
                         std::vector<double> attn_scales = {},
                         std::vector<bool> temperal_upsample = {false,
                                                                true,
                                                                true},
                         double dropout = 0.0,
                         int64_t input_channels = 3,
                         const std::string& non_linearity = "silu")
      : dim_(dim),
        z_dim_(z_dim),
        dim_mult_(dim_mult),
        num_res_blocks_(num_res_blocks),
        attn_scales_(attn_scales),
        temperal_upsample_(temperal_upsample) {
    nonlinearity_ = register_module("silu", torch::nn::SiLU());

    std::vector<int64_t> dims = {dim * dim_mult.back()};
    for (int64_t i = dim_mult.size() - 1; i >= 0; i--) {
      dims.push_back(dim * dim_mult.at(i));
    }

    double scale = 1.0 / std::pow(2, dim_mult.size() - 2);

    conv_in_ =
        register_module("conv_in",
                        QwenImageCausalConv3d(context,
                                              z_dim,
                                              dims[0],
                                              torch::IntArrayRef{3, 3, 3},
                                              torch::IntArrayRef{1, 1, 1},
                                              torch::IntArrayRef{1, 1, 1}));

    mid_block_ = register_module(
        "mid_block",
        QwenImageMidBlock(
            context, dims[0], dropout, non_linearity, /*num_layers=*/1));

    up_blocks_ = register_module("up_blocks", torch::nn::ModuleList());
    for (size_t i = 0; i < dims.size() - 1; i++) {
      int64_t in_dim = dims[i];
      int64_t out_dim = dims[i + 1];

      if (i > 0) {
        in_dim = in_dim / 2;
      }

      std::string upsample_mode;
      if (i != dim_mult.size() - 1) {
        upsample_mode = temperal_upsample[i] ? "upsample3d" : "upsample2d";
      }

      auto up_block = QwenImageUpBlock(context,
                                       in_dim,
                                       out_dim,
                                       num_res_blocks,
                                       dropout,
                                       upsample_mode,
                                       non_linearity);
      up_blocks_->push_back(up_block);

      if (!upsample_mode.empty()) {
        scale *= 2.0;
      }
    }

    norm_out_ = register_module(
        "norm_out",
        QwenImageRMS_norm(context, dims.back(), true, false, false, false));
    conv_out_ = register_module(
        "conv_out",
        QwenImageCausalConv3d(context,
                              /*in_channels=*/dims.back(),
                              /*out_channels=*/input_channels,
                              /*kernel_size=*/torch::IntArrayRef{3, 3, 3},
                              /*stride=*/torch::IntArrayRef{1, 1, 1},
                              /*padding=*/torch::IntArrayRef{1, 1, 1}));
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::shared_ptr<std::vector<torch::Tensor>> feat_cache,
                        std::shared_ptr<std::vector<int64_t>> feat_idx,
                        VaeSpatialParallel& ctx) {
    if (feat_idx == nullptr) {
      feat_idx =
          std::make_shared<std::vector<int64_t>>(std::vector<int64_t>{0});
    }
    auto result_x = x;

    if (feat_cache) {
      auto idx = (*feat_idx)[0];
      auto cache_x =
          result_x
              .index({torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(-CACHE_T, torch::indexing::None)})
              .clone();

      if (cache_x.size(2) < 2 && feat_cache->at(idx).defined()) {
        auto last_frame =
            feat_cache->at(idx)
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(-1, torch::indexing::None)})
                .unsqueeze(2)
                .to(cache_x.device());
        cache_x = torch::cat({last_frame, cache_x}, 2);
      }

      result_x = conv_in_->forward(result_x, ctx, feat_cache->at(idx));
      feat_cache->at(idx) = cache_x;
      (*feat_idx)[0]++;
    } else {
      result_x = conv_in_->forward(result_x, ctx);
    }

    result_x = mid_block_->forward(result_x, feat_cache, feat_idx, ctx);

    for (auto& up_block : *up_blocks_) {
      result_x = up_block->as<QwenImageUpBlock>()->forward(
          result_x, feat_cache, feat_idx, ctx);
    }

    result_x = norm_out_->forward(result_x);
    result_x = nonlinearity_->forward(result_x);

    if (feat_cache) {
      auto idx = (*feat_idx)[0];
      auto cache_x =
          result_x
              .index({torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(-CACHE_T, torch::indexing::None)})
              .clone();

      if (cache_x.size(2) < 2 && idx < feat_cache->size() &&
          feat_cache->at(idx).defined()) {
        auto last_frame =
            feat_cache->at(idx)
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(-1, torch::indexing::None)})
                .unsqueeze(2)
                .to(cache_x.device());
        cache_x = torch::cat({last_frame, cache_x}, 2);
      }

      result_x = conv_out_->forward(result_x, ctx, feat_cache->at(idx));
      feat_cache->at(idx) = cache_x;
      (*feat_idx)[0]++;
    } else {
      result_x = conv_out_->forward(result_x, ctx);
    }
    return result_x;
  }

  void load_state_dict(const StateDict& state_dict) {
    conv_in_->load_state_dict(state_dict.get_dict_with_prefix("conv_in."));
    mid_block_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));

    for (size_t i = 0; i < up_blocks_->size(); i++) {
      auto prefix = "up_blocks." + std::to_string(i) + ".";
      up_blocks_[i]->as<QwenImageUpBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix(prefix));
    }

    norm_out_->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    conv_out_->load_state_dict(state_dict.get_dict_with_prefix("conv_out."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    conv_in_->verify_loaded_weights("conv_in.");

    mid_block_->verify_loaded_weights("mid_block.");
    for (size_t i = 0; i < up_blocks_->size(); i++) {
      auto prefix = "up_blocks." + std::to_string(i) + ".";
      up_blocks_[i]->as<QwenImageUpBlock>()->verify_loaded_weights(prefix);
    }

    norm_out_->verify_loaded_weights("norm_out.");
    conv_out_->verify_loaded_weights("conv_out.");
  }

  std::vector<std::shared_ptr<Module>> get_modules() const {
    std::vector<std::shared_ptr<Module>> module = modules();
    return module;
  }

 private:
  int64_t dim_, z_dim_;
  std::vector<int64_t> dim_mult_;
  int64_t num_res_blocks_;
  std::vector<double> attn_scales_;
  std::vector<bool> temperal_upsample_;

  torch::nn::SiLU nonlinearity_{nullptr};
  QwenImageCausalConv3d conv_in_{nullptr};
  QwenImageMidBlock mid_block_{nullptr};
  torch::nn::ModuleList up_blocks_{nullptr};
  QwenImageRMS_norm norm_out_{nullptr};
  QwenImageCausalConv3d conv_out_{nullptr};
};

TORCH_MODULE(QwenImageDecoder3d);

class AutoencoderKLQwenImageImpl : public torch::nn::Module {
 public:
  AutoencoderKLQwenImageImpl(const ModelContext& context)
      : args_(context.get_model_args()),
        z_dim_(context.get_model_args().z_dim()),
        temperal_downsample_(context.get_model_args().temperal_downsample()),
        base_dim_(context.get_model_args().base_dim()),
        dim_mult_(context.get_model_args().dim_mult()),
        num_res_blocks_(context.get_model_args().num_res_blocks()),
        attn_scales_(context.get_model_args().attn_scales()),
        dropout_(context.get_model_args().dropout()),
        // VAE spatial parallel context (always constructed, even w_split=1).
        // VaeSpatialParallel's own constructor CHECKs that a process group is
        // provided whenever vae_size > 1.
        parallel_ctx_(context.get_parallel_args().vae_size(),
                      context.get_parallel_args().dit_vae_group_,
                      context.get_tensor_options().device()) {
    temperal_upsample_ = std::vector<bool>(temperal_downsample_.rbegin(),
                                           temperal_downsample_.rend());

    int64_t input_channels = context.get_model_args().in_channels();
    encoder_ = register_module("encoder",
                               QwenImageEncoder3d(context,
                                                  base_dim_,
                                                  z_dim_ * 2,
                                                  dim_mult_,
                                                  num_res_blocks_,
                                                  attn_scales_,
                                                  temperal_downsample_,
                                                  dropout_,
                                                  input_channels));

    quant_conv_ =
        register_module("quant_conv",
                        QwenImageCausalConv3d(context,
                                              z_dim_ * 2,
                                              z_dim_ * 2,
                                              torch::IntArrayRef{1, 1, 1},
                                              torch::IntArrayRef{1, 1, 1},
                                              torch::IntArrayRef{0, 0, 0}));

    post_quant_conv_ =
        register_module("post_quant_conv",
                        QwenImageCausalConv3d(context,
                                              z_dim_,
                                              z_dim_,
                                              torch::IntArrayRef{1, 1, 1},
                                              torch::IntArrayRef{1, 1, 1},
                                              torch::IntArrayRef{0, 0, 0}));

    decoder_ = register_module("decoder",
                               QwenImageDecoder3d(context,
                                                  base_dim_,
                                                  z_dim_,
                                                  dim_mult_,
                                                  num_res_blocks_,
                                                  attn_scales_,
                                                  temperal_upsample_,
                                                  dropout_,
                                                  input_channels));

    spatial_compression_ratio_ =
        static_cast<int64_t>(std::pow(2, temperal_downsample_.size()));

    use_slicing_ = false;
    use_tiling_ = false;
    tile_sample_min_height_ = 256;
    tile_sample_min_width_ = 256;
    tile_sample_stride_height_ = 192;
    tile_sample_stride_width_ = 192;

    cached_conv_counts_ = {{"decoder", count_conv3d_modules(*decoder_)},
                           {"encoder", count_conv3d_modules(*encoder_)}};

    if (parallel_ctx_.is_parallel()) {
      LOG(INFO) << "VAE spatial parallel enabled: w_split="
                << parallel_ctx_.w_split();
    }
  }

  void enable_tiling(int64_t tile_sample_min_height = -1,
                     int64_t tile_sample_min_width = -1,
                     int64_t tile_sample_stride_height = -1,
                     int64_t tile_sample_stride_width = -1) {
    use_tiling_ = true;
    if (tile_sample_min_height > 0)
      tile_sample_min_height_ = tile_sample_min_height;
    if (tile_sample_min_width > 0)
      tile_sample_min_width_ = tile_sample_min_width;
    if (tile_sample_stride_height > 0)
      tile_sample_stride_height_ = tile_sample_stride_height;
    if (tile_sample_stride_width > 0)
      tile_sample_stride_width_ = tile_sample_stride_width;
  }

  void clear_cache() {
    conv_num_ = count_conv3d_modules(*decoder_);
    conv_idx_ = std::make_shared<std::vector<int64_t>>(std::vector<int64_t>{0});
    feat_map_ = std::make_shared<std::vector<torch::Tensor>>(
        std::vector<torch::Tensor>(conv_num_));

    enc_conv_num_ = count_conv3d_modules(*encoder_);
    enc_conv_idx_ =
        std::make_shared<std::vector<int64_t>>(std::vector<int64_t>{0});
    enc_feat_map_ = std::make_shared<std::vector<torch::Tensor>>(
        std::vector<torch::Tensor>(enc_conv_num_));
  }

  torch::Tensor _encode(const torch::Tensor& x_input) {
    auto x = x_input;
    auto sizes = x.sizes();
    auto b = sizes[0], c = sizes[1], num_frame = sizes[2], height = sizes[3],
         width = sizes[4];

    if (parallel_ctx_.is_parallel()) {
      x = parallel_ctx_.split(x);
      width = x.size(-1);
    }

    if (use_tiling_ &&
        (width > tile_sample_min_width_ || height > tile_sample_min_height_)) {
      return tiled_encode(x);
    }

    clear_cache();
    auto iter = 1 + (num_frame - 1) / 4;

    std::vector<torch::Tensor> enc_outputs;
    enc_outputs.reserve(iter);
    for (int64_t i = 0; i < iter; i++) {
      enc_conv_idx_->at(0) = 0;
      torch::Tensor tile;

      if (i == 0) {
        tile = x.index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(0, 1),
                        torch::indexing::Slice(),
                        torch::indexing::Slice()});
      } else {
        tile = x.index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(1 + 4 * (i - 1), 1 + 4 * i),
                        torch::indexing::Slice(),
                        torch::indexing::Slice()});
      }

      enc_outputs.push_back(
          encoder_->forward(tile, enc_feat_map_, enc_conv_idx_, parallel_ctx_));
    }
    torch::Tensor out = torch::cat(enc_outputs, 2);

    auto enc = quant_conv_->forward(out, parallel_ctx_);
    // Merge latent back to global W before passing to DiT transformer
    if (parallel_ctx_.is_parallel()) {
      enc = parallel_ctx_.merge(enc);
    }
    clear_cache();
    return enc;
  }

  AutoencoderKLOutput encode(const torch::Tensor& x, bool return_dict = true) {
    torch::Tensor h;

    if (use_slicing_ && x.size(0) > 1) {
      std::vector<torch::Tensor> encoded_slices;
      auto slices = x.split(1);
      for (auto& slice : slices) {
        encoded_slices.push_back(_encode(slice));
      }
      h = torch::cat(encoded_slices);
    } else {
      h = _encode(x);
    }

    auto posterior = DiagonalGaussianDistribution(h);

    if (!return_dict) {
      return {posterior};
    }

    AutoencoderKLOutput output(posterior);
    return output;
  }

  DecoderOutput _decode(const torch::Tensor& z_input, bool return_dict = true) {
    auto z = z_input;
    auto sizes = z.sizes();
    auto b = sizes[0], c = sizes[1], num_frame = sizes[2], height = sizes[3],
         width = sizes[4];

    if (parallel_ctx_.is_parallel()) {
      z = parallel_ctx_.split(z);
      width = z.size(-1);
    }

    auto tile_latent_min_height =
        tile_sample_min_height_ / spatial_compression_ratio_;
    auto tile_latent_min_width =
        tile_sample_min_width_ / spatial_compression_ratio_;

    if (use_tiling_ &&
        (width > tile_latent_min_width || height > tile_latent_min_height)) {
      return tiled_decode(z, return_dict);
    }

    clear_cache();
    auto x = post_quant_conv_->forward(z, parallel_ctx_);

    std::vector<torch::Tensor> dec_outputs;
    dec_outputs.reserve(num_frame);
    for (int64_t i = 0; i < num_frame; i++) {
      conv_idx_->at(0) = 0;
      auto frame = x.slice(/*dim=*/2, i, i + 1);
      dec_outputs.push_back(
          decoder_->forward(frame, feat_map_, conv_idx_, parallel_ctx_));
    }
    torch::Tensor out = torch::cat(dec_outputs, 2);

    if (parallel_ctx_.is_parallel()) {
      out = parallel_ctx_.merge(out);
    }

    out = torch::clamp(out, -1.0, 1.0);
    clear_cache();

    if (!return_dict) {
      return {out};
    }
    DecoderOutput output(out);

    return output;
  }

  DecoderOutput decode(const torch::Tensor& z, bool return_dict = true) {
    torch::Tensor decoded;

    if (use_slicing_ && z.size(0) > 1) {
      std::vector<torch::Tensor> decoded_slices;
      auto slices = z.split(1);
      for (auto& slice : slices) {
        auto output = _decode(slice, true);
        decoded_slices.push_back(output.sample);
      }
      decoded = torch::cat(decoded_slices);
    } else {
      auto output = _decode(z, true);
      decoded = output.sample;
    }

    if (!return_dict) {
      return {decoded};
    }
    DecoderOutput output(decoded);

    return output;
  }

  torch::Tensor blend_v(const torch::Tensor& a,
                        const torch::Tensor& b,
                        int64_t blend_extent) {
    auto result_b = b.clone();
    blend_extent = std::min({a.size(3), b.size(3), blend_extent});

    for (int64_t y = 0; y < blend_extent; y++) {
      auto weight_a = 1.0 - static_cast<double>(y) / blend_extent;
      auto weight_b = static_cast<double>(y) / blend_extent;

      auto a_slice = a.index({torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              -blend_extent + y,
                              torch::indexing::Slice()});

      auto b_slice = result_b.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     y,
                                     torch::indexing::Slice()});

      auto blended = a_slice * weight_a + b_slice * weight_b;
      result_b.index_put_({torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           y,
                           torch::indexing::Slice()},
                          blended);
    }

    return result_b;
  }

  torch::Tensor blend_h(const torch::Tensor& a,
                        const torch::Tensor& b,
                        int64_t blend_extent) {
    auto result_b = b.clone();
    blend_extent = std::min({a.size(4), b.size(4), blend_extent});

    for (int64_t x = 0; x < blend_extent; x++) {
      auto weight_a = 1.0 - static_cast<double>(x) / blend_extent;
      auto weight_b = static_cast<double>(x) / blend_extent;

      auto a_slice = a.index({torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              -blend_extent + x});

      auto b_slice = result_b.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     x});

      auto blended = a_slice * weight_a + b_slice * weight_b;
      result_b.index_put_({torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           x},
                          blended);
    }

    return result_b;
  }

  torch::Tensor tiled_encode(const torch::Tensor& x) {
    auto sizes = x.sizes();
    auto b = sizes[0], c = sizes[1], num_frames = sizes[2], height = sizes[3],
         width = sizes[4];

    auto latent_height = height / spatial_compression_ratio_;
    auto latent_width = width / spatial_compression_ratio_;

    auto tile_latent_min_height =
        tile_sample_min_height_ / spatial_compression_ratio_;
    auto tile_latent_min_width =
        tile_sample_min_width_ / spatial_compression_ratio_;
    auto tile_latent_stride_height =
        tile_sample_stride_height_ / spatial_compression_ratio_;
    auto tile_latent_stride_width =
        tile_sample_stride_width_ / spatial_compression_ratio_;

    auto blend_height = tile_latent_min_height - tile_latent_stride_height;
    auto blend_width = tile_latent_min_width - tile_latent_stride_width;

    std::vector<std::vector<torch::Tensor>> rows;

    for (int64_t i = 0; i < height; i += tile_sample_stride_height_) {
      std::vector<torch::Tensor> row;

      for (int64_t j = 0; j < width; j += tile_sample_stride_width_) {
        clear_cache();
        std::vector<torch::Tensor> time_frames;
        auto frame_range = 1 + (num_frames - 1) / 4;

        for (int64_t k = 0; k < frame_range; k++) {
          enc_conv_idx_->at(0) = 0;
          torch::Tensor tile;

          if (k == 0) {
            tile = x.index(
                {torch::indexing::Slice(),
                 torch::indexing::Slice(),
                 torch::indexing::Slice(0, 1),
                 torch::indexing::Slice(i, i + tile_sample_min_height_),
                 torch::indexing::Slice(j, j + tile_sample_min_width_)});
          } else {
            tile = x.index(
                {torch::indexing::Slice(),
                 torch::indexing::Slice(),
                 torch::indexing::Slice(1 + 4 * (k - 1), 1 + 4 * k),
                 torch::indexing::Slice(i, i + tile_sample_min_height_),
                 torch::indexing::Slice(j, j + tile_sample_min_width_)});
          }

          auto encoded_tile = encoder_->forward(
              tile, enc_feat_map_, enc_conv_idx_, parallel_ctx_);
          auto quantized_tile =
              quant_conv_->forward(encoded_tile, parallel_ctx_);
          time_frames.push_back(quantized_tile);
        }

        row.push_back(torch::cat(time_frames, 2));
      }
      rows.push_back(row);
    }
    clear_cache();

    std::vector<torch::Tensor> result_rows;

    for (int64_t i = 0; i < static_cast<int64_t>(rows.size()); i++) {
      std::vector<torch::Tensor> result_row;

      for (int64_t j = 0; j < static_cast<int64_t>(rows[i].size()); j++) {
        auto tile = rows[i][j];

        if (i > 0) {
          tile = blend_v(rows[i - 1][j], tile, blend_height);
        }
        if (j > 0) {
          tile = blend_h(rows[i][j - 1], tile, blend_width);
        }

        result_row.push_back(
            tile.index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(0, tile_latent_stride_height),
                        torch::indexing::Slice(0, tile_latent_stride_width)}));
      }

      result_rows.push_back(torch::cat(result_row, -1));
    }

    auto enc = torch::cat(result_rows, 3)
                   .index({torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(0, latent_height),
                           torch::indexing::Slice(0, latent_width)});

    return enc;
  }

  DecoderOutput tiled_decode(const torch::Tensor& z, bool return_dict = true) {
    auto sizes = z.sizes();
    auto b = sizes[0], c = sizes[1], num_frames = sizes[2], height = sizes[3],
         width = sizes[4];

    auto sample_height = height * spatial_compression_ratio_;
    auto sample_width = width * spatial_compression_ratio_;

    auto tile_latent_min_height =
        tile_sample_min_height_ / spatial_compression_ratio_;
    auto tile_latent_min_width =
        tile_sample_min_width_ / spatial_compression_ratio_;
    auto tile_latent_stride_height =
        tile_sample_stride_height_ / spatial_compression_ratio_;
    auto tile_latent_stride_width =
        tile_sample_stride_width_ / spatial_compression_ratio_;

    auto blend_height = tile_sample_min_height_ - tile_sample_stride_height_;
    auto blend_width = tile_sample_min_width_ - tile_sample_stride_width_;

    std::vector<std::vector<torch::Tensor>> rows;

    for (int64_t i = 0; i < height; i += tile_latent_stride_height) {
      std::vector<torch::Tensor> row;

      for (int64_t j = 0; j < width; j += tile_latent_stride_width) {
        clear_cache();
        std::vector<torch::Tensor> time_frames;

        for (int64_t k = 0; k < num_frames; k++) {
          conv_idx_->at(0) = 0;
          auto tile =
              z.index({torch::indexing::Slice(),
                       torch::indexing::Slice(),
                       torch::indexing::Slice(k, k + 1),
                       torch::indexing::Slice(i, i + tile_latent_min_height),
                       torch::indexing::Slice(j, j + tile_latent_min_width)});

          auto post_quant_tile = post_quant_conv_->forward(tile, parallel_ctx_);
          auto decoded_tile = decoder_->forward(
              post_quant_tile, feat_map_, conv_idx_, parallel_ctx_);
          time_frames.push_back(decoded_tile);
        }

        row.push_back(torch::cat(time_frames, 2));
      }
      rows.push_back(row);
    }
    clear_cache();

    std::vector<torch::Tensor> result_rows;

    for (int64_t i = 0; i < static_cast<int64_t>(rows.size()); i++) {
      std::vector<torch::Tensor> result_row;

      for (int64_t j = 0; j < static_cast<int64_t>(rows[i].size()); j++) {
        auto tile = rows[i][j];

        if (i > 0) {
          tile = blend_v(rows[i - 1][j], tile, blend_height);
        }
        if (j > 0) {
          tile = blend_h(rows[i][j - 1], tile, blend_width);
        }

        result_row.push_back(
            tile.index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(0, tile_sample_stride_height_),
                        torch::indexing::Slice(0, tile_sample_stride_width_)}));
      }

      result_rows.push_back(torch::cat(result_row, -1));
    }

    auto dec = torch::cat(result_rows, 3)
                   .index({torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(0, sample_height),
                           torch::indexing::Slice(0, sample_width)});

    if (!return_dict) {
      return {dec};
    }
    DecoderOutput output(dec);
    return output;
  }

  DecoderOutput forward(const torch::Tensor& sample,
                        bool sample_posterior = false,
                        bool return_dict = true,
                        int64_t seed = 42) {
    auto x = sample;

    auto encode_output = encode(x, true);
    auto posterior = encode_output.latent_dist;

    torch::Tensor z;
    if (sample_posterior) {
      z = posterior.sample(seed);
    } else {
      z = posterior.mode();
    }

    auto dec = decode(z, return_dict);
    return dec;
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
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
  template <typename ModuleType>
  int64_t count_conv3d_modules(const ModuleType& module) {
    int64_t count = 0;
    for (const auto& m : module.named_modules()) {
      if (auto conv =
              dynamic_cast<QwenImageCausalConv3dImpl*>(m.value().get())) {
        count++;
      }
    }
    return count;
  }

  int64_t base_dim_;
  int64_t z_dim_;
  std::vector<int64_t> dim_mult_;
  int64_t num_res_blocks_;
  std::vector<double> attn_scales_;
  std::vector<bool> temperal_downsample_;
  std::vector<bool> temperal_upsample_;
  double dropout_;

  int64_t spatial_compression_ratio_;
  bool use_slicing_, use_tiling_;
  int64_t tile_sample_min_height_;
  int64_t tile_sample_min_width_;
  int64_t tile_sample_stride_height_;
  int64_t tile_sample_stride_width_;

  std::unordered_map<std::string, int64_t> cached_conv_counts_;

  int64_t conv_num_;
  int64_t enc_conv_num_;
  std::shared_ptr<std::vector<int64_t>> conv_idx_;
  std::shared_ptr<std::vector<int64_t>> enc_conv_idx_;
  std::shared_ptr<std::vector<torch::Tensor>> feat_map_;
  std::shared_ptr<std::vector<torch::Tensor>> enc_feat_map_;

  QwenImageEncoder3d encoder_{nullptr};
  QwenImageCausalConv3d quant_conv_{nullptr};
  QwenImageCausalConv3d post_quant_conv_{nullptr};
  QwenImageDecoder3d decoder_{nullptr};

  ModelArgs args_;
  VaeSpatialParallel parallel_ctx_;
};

TORCH_MODULE(AutoencoderKLQwenImage);

REGISTER_MODEL_ARGS(AutoencoderKLQwenImage, [&] {
  LOAD_ARG_OR(base_dim, "base_dim", 96);
  LOAD_ARG_OR(z_dim, "z_dim", 16);
  LOAD_ARG_OR(in_channels, "in_channels", 3);
  LOAD_ARG_OR(dim_mult, "dim_mult", (std::vector<int64_t>{1, 2, 4, 4}));
  LOAD_ARG_OR(attn_scales, "attn_scales", (std::vector<double>{}));
  LOAD_ARG_OR(temperal_downsample,
              "temperal_downsample",
              (std::vector<bool>{false, true, true}));
  LOAD_ARG_OR(num_res_blocks, "num_res_blocks", 2);
  LOAD_ARG_OR(dropout, "dropout", 0);
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
