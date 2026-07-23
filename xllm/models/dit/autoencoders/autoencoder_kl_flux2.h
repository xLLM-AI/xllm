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
#include "models/dit/autoencoders/autoencoder_kl.h"

namespace xllm {

// VAE implementation for Flux2, including encoder and decoder with BatchNorm2d
class AutoencoderKLFlux2Impl final : public torch::nn::Module {
 public:
  explicit AutoencoderKLFlux2Impl(const ModelContext& context)
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

    int64_t patch_size_prod = 1;
    for (int64_t ps : args_.ae_patch_size()) {
      patch_size_prod *= ps;
    }
    int64_t bn_num_features = patch_size_prod * args_.latent_channels();

    bn_ = register_module(
        "bn",
        torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(bn_num_features)
                                   .eps(args_.batch_norm_eps())
                                   .momentum(args_.batch_norm_momentum())
                                   .affine(false)
                                   .track_running_stats(true)));
    bn_->to(dtype);
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

      weight::load_weight(state_dict->get_dict_with_prefix("bn."),
                          "running_mean",
                          bn_->running_mean,
                          is_bn_running_mean_);
      weight::load_weight(state_dict->get_dict_with_prefix("bn."),
                          "running_var",
                          bn_->running_var,
                          is_bn_running_var_);
      /*weight::load_weight(state_dict->get_dict_with_prefix("bn."),
                          "num_batches_tracked",
                          bn_->num_batches_tracked,
                          is_bn_num_batches_tracked_);*/
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
    CHECK(is_bn_running_mean_)
        << "running_mean is not loaded for " << prefix + "bn.running_mean";
    CHECK(is_bn_running_var_)
        << "running_var is not loaded for " << prefix + "bn.running_var";
    /* CHECK(is_bn_num_batches_tracked_)
         << "num_batches_tracked is not loaded for "
         << prefix + "bn.num_batches_tracked";*/
  }

  torch::Tensor get_bn_running_mean() const { return bn_->running_mean; }

  torch::Tensor get_bn_running_var() const { return bn_->running_var; }

  float get_batch_norm_eps() const { return args_.batch_norm_eps(); }

 private:
  VAEEncoder encoder_ = nullptr;
  VAEDecoder decoder_ = nullptr;
  torch::nn::Conv2d quant_conv_ = nullptr;
  torch::nn::Conv2d post_quant_conv_ = nullptr;
  torch::nn::BatchNorm2d bn_ = nullptr;
  bool use_post_quant_conv_ = false;

  bool is_quant_conv_weight_ = false;
  bool is_quant_conv_bias_ = false;
  bool is_post_quant_conv_weight_ = false;
  bool is_post_quant_conv_bias_ = false;
  bool is_bn_running_mean_ = false;
  bool is_bn_running_var_ = false;
  /*bool is_bn_num_batches_tracked_ = false;*/
  ModelArgs args_;
};
TORCH_MODULE(AutoencoderKLFlux2);

// register VAE model with the model registry
REGISTER_MODEL_ARGS(AutoencoderKLFlux2, [&] {
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
  LOAD_ARG_OR(latent_channels, "latent_channels", 32);
  LOAD_ARG_OR(norm_num_groups, "norm_num_groups", 32);
  LOAD_ARG_OR(sample_size, "sample_size", 1024);
  LOAD_ARG_OR(mid_block_add_attention, "mid_block_add_attention", true);
  LOAD_ARG_OR(force_upcast, "force_upcast", true);
  LOAD_ARG_OR(use_quant_conv, "use_quant_conv", false);
  LOAD_ARG_OR(use_post_quant_conv, "use_post_quant_conv", false);
  LOAD_ARG_OR(batch_norm_eps, "batch_norm_eps", 1e-04f);
  LOAD_ARG_OR(act_fn, "act_fn", "silu");
  LOAD_ARG_OR(batch_norm_momentum, "batch_norm_momentum", 0.1f);
  LOAD_ARG_OR(ae_patch_size, "patch_size", (std::vector<int64_t>{2, 2}));
});
}  // namespace xllm
