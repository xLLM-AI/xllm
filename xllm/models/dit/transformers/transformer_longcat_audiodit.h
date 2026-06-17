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

// AudioDiT transformer model for LongCat-AudioDiT.
// Ref:
// https://github.com/meituan-longcat/LongCat-AudioDiT/blob/main/audiodit/modeling_audiodit.py
// Components:
//   - AudioDiTVae: WAV-VAE encoder/decoder with SnakeBeta activations and
//     weight_norm convolutions.
//   - AudioDiTTransformer: DiT backbone (24 blocks) with:
//       timestep embedding, input/text embedders, ConvNeXtV2 text processing,
//       rotary position embedding, self-attn + cross-attn + FFN per block,
//       global AdaLN, long-skip connection.
//   - APG guidance helpers.
//   - UMT5TextEncoder: text encoder wrapper (google/umt5-base).
//   - Weight loading utilities (checkpoint_key_to_cpp_key,
//     load_module_from_state_dicts).

#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/request/dit_request_state.h"
#include "models/dit/autoencoders/autoencoder_kl.h"  // randn_tensor
#include "models/dit/encoders/umt5_encoder.h"

namespace xllm {

// ============================================================================
// VAE helpers
// ============================================================================

// SnakeBeta activation: x + (1/beta) * sin(x * alpha)^2
// alpha and beta are learnable per-channel parameters (log-scale by default).
class AudioSnakeBetaImpl : public torch::nn::Module {
 public:
  explicit AudioSnakeBetaImpl(int64_t in_features, bool alpha_logscale = true)
      : alpha_logscale_(alpha_logscale) {
    alpha_ = register_parameter("alpha", torch::zeros({in_features}));
    beta_ = register_parameter("beta", torch::zeros({in_features}));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    // x: (B, C, T)
    torch::Tensor alpha = alpha_.unsqueeze(0).unsqueeze(-1);  // (1, C, 1)
    torch::Tensor beta = beta_.unsqueeze(0).unsqueeze(-1);    // (1, C, 1)
    if (alpha_logscale_) {
      alpha = torch::exp(alpha);
      beta = torch::exp(beta);
    }
    return x + (1.0 / (beta + 1e-9f)) * torch::sin(x * alpha).pow(2);
  }

 private:
  torch::Tensor alpha_;
  torch::Tensor beta_;
  bool alpha_logscale_;
};
TORCH_MODULE(AudioSnakeBeta);

// Pixel-unshuffle for 1-D signals: (B, C, W) -> (B, C*factor, W/factor)
inline torch::Tensor pixel_unshuffle_1d(const torch::Tensor& x,
                                        int64_t factor) {
  int64_t b = x.size(0), c = x.size(1), w = x.size(2);
  return x.view({b, c, w / factor, factor})
      .permute({0, 1, 3, 2})
      .contiguous()
      .view({b, c * factor, w / factor});
}

// Pixel-shuffle for 1-D signals: (B, C*factor, W) -> (B, C, W*factor)
inline torch::Tensor pixel_shuffle_1d(const torch::Tensor& x, int64_t factor) {
  int64_t b = x.size(0), c_full = x.size(1), w = x.size(2);
  int64_t c = c_full / factor;
  return x.view({b, c, factor, w})
      .permute({0, 1, 3, 2})
      .contiguous()
      .view({b, c, w * factor});
}

// Downsampling shortcut: pixel-unshuffle then average over groups
class VaeDownsampleShortcutImpl : public torch::nn::Module {
 public:
  VaeDownsampleShortcutImpl(int64_t in_channels,
                            int64_t out_channels,
                            int64_t factor)
      : factor_(factor),
        group_size_(in_channels * factor / out_channels),
        out_channels_(out_channels) {}

  torch::Tensor forward(const torch::Tensor& x) {
    torch::Tensor y = pixel_unshuffle_1d(x, factor_);
    int64_t b = y.size(0), n = y.size(2);
    return y.view({b, out_channels_, group_size_, n}).mean(2);
  }

 private:
  int64_t factor_, group_size_, out_channels_;
};
TORCH_MODULE(VaeDownsampleShortcut);

// Upsampling shortcut: repeat-interleave then pixel-shuffle
class VaeUpsampleShortcutImpl : public torch::nn::Module {
 public:
  VaeUpsampleShortcutImpl(int64_t in_channels,
                          int64_t out_channels,
                          int64_t factor)
      : factor_(factor), repeats_(out_channels * factor / in_channels) {}

  torch::Tensor forward(const torch::Tensor& x) {
    return pixel_shuffle_1d(x.repeat_interleave(repeats_, 1), factor_);
  }

 private:
  int64_t factor_, repeats_;
};
TORCH_MODULE(VaeUpsampleShortcut);

// Single residual unit: act -> wn_conv(dilation) -> act -> wn_conv(1x1)
class VaeResidualUnitImpl : public torch::nn::Module {
 public:
  VaeResidualUnitImpl(int64_t in_channels,
                      int64_t out_channels,
                      int64_t dilation,
                      bool use_snake = false,
                      int64_t kernel_size = 7) {
    int64_t padding = (dilation * (kernel_size - 1)) / 2;
    if (use_snake) {
      act0_ = register_module("layers_0", AudioSnakeBeta(out_channels));
      act1_ = register_module("layers_2", AudioSnakeBeta(out_channels));
      use_snake_ = true;
    } else {
      elu0_ = register_module("layers_0", torch::nn::ELU());
      elu1_ = register_module("layers_2", torch::nn::ELU());
      use_snake_ = false;
    }
    // Weight-normed convolutions
    conv0_ = register_module(
        "layers_1",
        torch::nn::Conv1d(
            torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                .dilation(dilation)
                .padding(padding)));
    conv1_ = register_module("layers_3",
                             torch::nn::Conv1d(torch::nn::Conv1dOptions(
                                 out_channels, out_channels, 1)));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    torch::Tensor h;
    if (use_snake_) {
      h = act0_->forward(x);
    } else {
      h = elu0_->forward(x);
    }
    h = conv0_->forward(h);
    if (use_snake_) {
      h = act1_->forward(h);
    } else {
      h = elu1_->forward(h);
    }
    h = conv1_->forward(h);
    return x + h;
  }

 private:
  bool use_snake_ = false;
  AudioSnakeBeta act0_{nullptr}, act1_{nullptr};
  torch::nn::ELU elu0_{nullptr}, elu1_{nullptr};
  torch::nn::Conv1d conv0_{nullptr}, conv1_{nullptr};
};
TORCH_MODULE(VaeResidualUnit);

// Encoder block: 3x residual units + act + strided conv
class VaeEncoderBlockImpl : public torch::nn::Module {
 public:
  VaeEncoderBlockImpl(int64_t in_ch,
                      int64_t out_ch,
                      int64_t stride,
                      bool use_snake = false,
                      bool use_downsample_shortcut = false) {
    // Residual units: Python layers.0, layers.1, layers.2
    int64_t layer_idx = 0;
    for (int64_t d : {1LL, 3LL, 9LL}) {
      enc_res_.push_back(
          register_module("layers_" + std::to_string(layer_idx),
                          VaeResidualUnit(in_ch, in_ch, d, use_snake)));
      ++layer_idx;
    }
    // Activation: Python layers.3
    if (use_snake) {
      act_ = register_module("layers_" + std::to_string(layer_idx),
                             AudioSnakeBeta(in_ch));
      use_snake_ = true;
    } else {
      elu_ = register_module("layers_" + std::to_string(layer_idx),
                             torch::nn::ELU());
      use_snake_ = false;
    }
    ++layer_idx;
    // Strided conv: Python layers.4
    int64_t pad = static_cast<int64_t>(std::ceil(stride / 2.0));
    down_conv_ = register_module(
        "layers_" + std::to_string(layer_idx),
        torch::nn::Conv1d(torch::nn::Conv1dOptions(in_ch, out_ch, 2 * stride)
                              .stride(stride)
                              .padding(pad)));
    // Shortcut: Python res
    if (use_downsample_shortcut) {
      shortcut_ =
          register_module("res", VaeDownsampleShortcut(in_ch, out_ch, stride));
      has_shortcut_ = true;
    }
  }

  torch::Tensor forward(const torch::Tensor& x) {
    torch::Tensor h = x;
    for (auto& r : enc_res_) {
      h = r->forward(h);
    }
    if (use_snake_) {
      h = act_->forward(h);
    } else {
      h = elu_->forward(h);
    }
    torch::Tensor out = down_conv_->forward(h);
    if (has_shortcut_) {
      out = out + shortcut_->forward(x);
    }
    return out;
  }

 private:
  std::vector<VaeResidualUnit> enc_res_;
  bool use_snake_ = false, has_shortcut_ = false;
  AudioSnakeBeta act_{nullptr};
  torch::nn::ELU elu_{nullptr};
  torch::nn::Conv1d down_conv_{nullptr};
  VaeDownsampleShortcut shortcut_{nullptr};
};
TORCH_MODULE(VaeEncoderBlock);

// Decoder block: act + strided transpose conv + 3x residual units
class VaeDecoderBlockImpl : public torch::nn::Module {
 public:
  VaeDecoderBlockImpl(int64_t in_ch,
                      int64_t out_ch,
                      int64_t stride,
                      bool use_snake = false,
                      bool use_upsample_shortcut = false) {
    // Activation: Python layers.0
    if (use_snake) {
      act_ = register_module("layers_0", AudioSnakeBeta(in_ch));
      use_snake_ = true;
    } else {
      elu_ = register_module("layers_0", torch::nn::ELU());
      use_snake_ = false;
    }
    // Transposed conv: Python layers.1
    int64_t pad = static_cast<int64_t>(std::ceil(stride / 2.0));
    up_conv_ = register_module(
        "layers_1",
        torch::nn::ConvTranspose1d(
            torch::nn::ConvTranspose1dOptions(in_ch, out_ch, 2 * stride)
                .stride(stride)
                .padding(pad)));
    // Residual units: Python layers.2, layers.3, layers.4
    int64_t res_layer_idx = 2;
    for (int64_t d : {1LL, 3LL, 9LL}) {
      dec_res_.push_back(
          register_module("layers_" + std::to_string(res_layer_idx),
                          VaeResidualUnit(out_ch, out_ch, d, use_snake)));
      ++res_layer_idx;
    }
    // Shortcut: Python res
    if (use_upsample_shortcut) {
      shortcut_ =
          register_module("res", VaeUpsampleShortcut(in_ch, out_ch, stride));
      has_shortcut_ = true;
    }
  }

  torch::Tensor forward(const torch::Tensor& x) {
    // Official Python: layers = Sequential(act, conv_transpose, res1, res2,
    // res3)
    //   forward: layers(x) + res(x)   -- shortcut added to full block output
    torch::Tensor h;
    if (use_snake_) {
      h = act_->forward(x);
    } else {
      h = elu_->forward(x);
    }
    torch::Tensor out = up_conv_->forward(h);
    for (auto& r : dec_res_) {
      out = r->forward(out);
    }
    if (has_shortcut_) {
      out = out + shortcut_->forward(x);
    }
    return out;
  }

 private:
  std::vector<VaeResidualUnit> dec_res_;
  bool use_snake_ = false, has_shortcut_ = false;
  AudioSnakeBeta act_{nullptr};
  torch::nn::ELU elu_{nullptr};
  torch::nn::ConvTranspose1d up_conv_{nullptr};
  VaeUpsampleShortcut shortcut_{nullptr};
};
TORCH_MODULE(VaeDecoderBlock);

// VAE Encoder: audio -> latent (batch, encoder_latent_dim,
// T/downsampling_ratio)
struct VaeEncoderConfig {
  int64_t in_channels = 1;
  int64_t channels = 128;
  std::vector<int64_t> c_mults = {1, 2, 4, 8, 16};
  std::vector<int64_t> strides = {2, 4, 4, 8, 8};
  int64_t encoder_latent_dim = 128;  // 2 * latent_dim for VAE bottleneck
  bool use_snake = true;
  bool use_downsample_shortcut = false;  // "averaging" shortcut
};

class VaeEncoderImpl : public torch::nn::Module {
 public:
  explicit VaeEncoderImpl(const VaeEncoderConfig& cfg) {
    std::vector<int64_t> c_mults_with_1 = {1};
    c_mults_with_1.insert(
        c_mults_with_1.end(), cfg.c_mults.begin(), cfg.c_mults.end());

    // Initial conv (Python: layers.0)
    int64_t ch0 = c_mults_with_1[0] * cfg.channels;
    in_conv_ = register_module(
        "layers_0",
        torch::nn::Conv1d(
            torch::nn::Conv1dOptions(cfg.in_channels, ch0, 7).padding(3)));

    // Encoder blocks (Python: layers.1 ... layers.N-1)
    for (int64_t i = 0; i < static_cast<int64_t>(c_mults_with_1.size()) - 1;
         ++i) {
      int64_t in_ch = c_mults_with_1[i] * cfg.channels;
      int64_t out_ch = c_mults_with_1[i + 1] * cfg.channels;
      int64_t s = cfg.strides[i];
      enc_blocks_.push_back(register_module(
          "layers_" + std::to_string(i + 1),
          VaeEncoderBlock(
              in_ch, out_ch, s, cfg.use_snake, cfg.use_downsample_shortcut)));
    }

    // Output conv (Python: layers.N)
    int64_t ch_last = c_mults_with_1.back() * cfg.channels;
    int64_t out_layer_idx =
        static_cast<int64_t>(c_mults_with_1.size());  // = num_blocks + 1
    out_conv_ = register_module(
        "layers_" + std::to_string(out_layer_idx),
        torch::nn::Conv1d(
            torch::nn::Conv1dOptions(ch_last, cfg.encoder_latent_dim, 3)
                .padding(1)));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    torch::Tensor h = in_conv_->forward(x);
    for (auto& blk : enc_blocks_) {
      h = blk->forward(h);
    }
    return out_conv_->forward(h);
  }

 private:
  torch::nn::Conv1d in_conv_{nullptr};
  std::vector<VaeEncoderBlock> enc_blocks_;
  torch::nn::Conv1d out_conv_{nullptr};
};
TORCH_MODULE(VaeEncoder);

// VAE Decoder: latent -> audio
struct VaeDecoderConfig {
  int64_t in_channels = 1;
  int64_t channels = 128;
  std::vector<int64_t> c_mults = {1, 2, 4, 8, 16};
  std::vector<int64_t> strides = {2, 4, 4, 8, 8};
  int64_t latent_dim = 64;
  bool use_snake = true;
  bool use_upsample_shortcut = false;  // per-block "duplicating" shortcut
  bool use_in_shortcut = false;        // top-level "duplicating" in_shortcut
  bool final_tanh = false;
};

class VaeDecoderImpl : public torch::nn::Module {
 public:
  explicit VaeDecoderImpl(const VaeDecoderConfig& cfg) {
    std::vector<int64_t> c_mults_with_1 = {1};
    c_mults_with_1.insert(
        c_mults_with_1.end(), cfg.c_mults.begin(), cfg.c_mults.end());

    // Top-level in_shortcut (Python: self.shortcut, stride=1 upsample of
    // latent)
    int64_t ch_last = c_mults_with_1.back() * cfg.channels;
    if (cfg.use_in_shortcut) {
      in_shortcut_ = register_module(
          "shortcut", VaeUpsampleShortcut(cfg.latent_dim, ch_last, 1));
      has_in_shortcut_ = true;
    }

    // Initial conv (Python: layers.0)
    in_conv_ = register_module(
        "layers_0",
        torch::nn::Conv1d(
            torch::nn::Conv1dOptions(cfg.latent_dim, ch_last, 7).padding(3)));

    // Decoder blocks (reverse order; Python: layers.1 ... layers.N-1)
    int64_t dec_layer_idx = 1;
    for (int64_t i = static_cast<int64_t>(c_mults_with_1.size()) - 1; i > 0;
         --i) {
      int64_t in_ch = c_mults_with_1[i] * cfg.channels;
      int64_t out_ch = c_mults_with_1[i - 1] * cfg.channels;
      int64_t s = cfg.strides[i - 1];
      dec_blocks_.push_back(register_module(
          "layers_" + std::to_string(dec_layer_idx),
          VaeDecoderBlock(
              in_ch, out_ch, s, cfg.use_snake, cfg.use_upsample_shortcut)));
      ++dec_layer_idx;
    }

    // Output activation (Python: layers.N) + conv (Python: layers.N+1)
    int64_t ch0 = c_mults_with_1[0] * cfg.channels;
    if (cfg.use_snake) {
      out_act_snake_ = register_module(
          "layers_" + std::to_string(dec_layer_idx), AudioSnakeBeta(ch0));
      use_snake_ = true;
    } else {
      out_act_elu_ = register_module("layers_" + std::to_string(dec_layer_idx),
                                     torch::nn::ELU());
      use_snake_ = false;
    }
    ++dec_layer_idx;
    out_conv_ = register_module(
        "layers_" + std::to_string(dec_layer_idx),
        torch::nn::Conv1d(torch::nn::Conv1dOptions(ch0, cfg.in_channels, 7)
                              .padding(3)
                              .bias(false)));
    if (cfg.final_tanh) {
      use_tanh_ = true;
    }
  }

  torch::Tensor forward(const torch::Tensor& z) {
    // Python: if shortcut: x = shortcut(z) + layers[0](z); x = layers[1:](x)
    //         else:        return layers(z)
    torch::Tensor h = in_conv_->forward(z);
    if (has_in_shortcut_) {
      h = h + in_shortcut_->forward(z);
    }
    for (auto& blk : dec_blocks_) {
      h = blk->forward(h);
    }
    if (use_snake_) {
      h = out_act_snake_->forward(h);
    } else {
      h = out_act_elu_->forward(h);
    }
    h = out_conv_->forward(h);
    if (use_tanh_) {
      h = torch::tanh(h);
    }
    return h;
  }

 private:
  torch::nn::Conv1d in_conv_{nullptr};
  VaeUpsampleShortcut in_shortcut_{nullptr};
  bool has_in_shortcut_ = false;
  std::vector<VaeDecoderBlock> dec_blocks_;
  bool use_snake_ = false, use_tanh_ = false;
  AudioSnakeBeta out_act_snake_{nullptr};
  torch::nn::ELU out_act_elu_{nullptr};
  torch::nn::Conv1d out_conv_{nullptr};
};
TORCH_MODULE(VaeDecoder);

// Top-level WAV-VAE: encoder + VAE bottleneck + decoder
// Encoder runs in float16 (matching original model_half=True).
struct AudioDiTVaeConfig {
  VaeEncoderConfig encoder_cfg;
  VaeDecoderConfig decoder_cfg;
  float scale = 0.71f;
  int64_t downsampling_ratio = 2048;  // VAE temporal downsampling
  int64_t latent_dim = 64;
};

class AudioDiTVaeImpl : public torch::nn::Module {
 public:
  explicit AudioDiTVaeImpl(const AudioDiTVaeConfig& cfg)
      : scale_(cfg.scale),
        downsampling_ratio_(cfg.downsampling_ratio),
        latent_dim_(cfg.latent_dim) {
    encoder_ = register_module("encoder", VaeEncoder(cfg.encoder_cfg));
    decoder_ = register_module("decoder", VaeDecoder(cfg.decoder_cfg));
  }

  // Encode audio (batch,1,T) -> latent (batch,latent_dim,T/downsampling_ratio)
  // VAE bottleneck: split encoder output into mean+logscale, sample.
  torch::Tensor encode(const torch::Tensor& audio) {
    // Check if encoder is in float16 mode
    bool is_half = false;
    for (const auto& kv : encoder_->named_parameters()) {
      is_half = (kv.value().dtype() == torch::kFloat16);
      break;
    }
    torch::Tensor x = is_half ? audio.to(torch::kFloat16) : audio;
    torch::Tensor enc_out = encoder_->forward(x);  // (B, 2*latent_dim, T')

    // VAE bottleneck (runs in encoder dtype)
    std::vector<torch::Tensor> chunks = enc_out.chunk(2, 1);
    torch::Tensor mean = chunks[0];
    torch::Tensor scale_param = chunks[1];
    torch::Tensor stdev =
        torch::nn::functional::softplus(
            scale_param,
            torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(
                20.0)) +
        1e-4f;
    torch::Tensor noise = torch::randn_like(mean);
    torch::Tensor latents = noise * stdev + mean;

    if (is_half) {
      latents = latents.to(torch::kFloat32);
    }
    return latents / scale_;
  }

  // Decode latent (batch,latent_dim,T') -> audio (batch,1,T)
  torch::Tensor decode(const torch::Tensor& latents) {
    torch::Tensor z = latents * scale_;
    bool is_half = false;
    for (const auto& kv : decoder_->named_parameters()) {
      is_half = (kv.value().dtype() == torch::kFloat16);
      break;
    }
    if (is_half) {
      z = z.to(torch::kFloat16);
    }
    torch::Tensor decoded = decoder_->forward(z);
    if (is_half) {
      decoded = decoded.to(torch::kFloat32);
    }
    return decoded;
  }

  // Convert encoder/decoder to float16 (matching original model_half=True)
  void to_half() {
    encoder_->to(torch::kFloat16);
    decoder_->to(torch::kFloat16);
  }

  VaeEncoder encoder_{nullptr};
  VaeDecoder decoder_{nullptr};
  float scale_;
  int64_t downsampling_ratio_, latent_dim_;
};
TORCH_MODULE(AudioDiTVae);

// ============================================================================
// Transformer components
// ============================================================================

// RMS normalization
class AudioRMSNormImpl : public torch::nn::Module {
 public:
  explicit AudioRMSNormImpl(int64_t dim, float eps = 1e-6f) : eps_(eps) {
    weight_ = register_parameter("weight", torch::ones({dim}));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    // Compute in float32 for numerical stability, then cast back
    torch::Tensor x_f = x.to(torch::kFloat32);
    torch::Tensor variance = x_f.pow(2).mean(-1, /*keepdim=*/true);
    x_f = x_f * torch::rsqrt(variance + eps_);
    return x_f.to(x.dtype()) * weight_;
  }

 private:
  float eps_;
  torch::Tensor weight_;
};
TORCH_MODULE(AudioRMSNorm);

// Sinusoidal position embedding for timesteps
class AudioSinusPositionEmbeddingImpl : public torch::nn::Module {
 public:
  explicit AudioSinusPositionEmbeddingImpl(int64_t dim) : dim_(dim) {}

  torch::Tensor forward(const torch::Tensor& x, float scale = 1000.0f) {
    // x: (B,) scalar timesteps
    torch::Device dev = x.device();
    int64_t half_dim = dim_ / 2;
    double log_val = std::log(10000.0);
    torch::Tensor emb =
        torch::arange(half_dim, torch::dtype(torch::kFloat32).device(dev))
            .mul(-log_val / static_cast<double>(half_dim - 1))
            .exp();
    emb = scale * x.to(torch::kFloat32).unsqueeze(1) * emb.unsqueeze(0);
    return torch::cat({emb.sin(), emb.cos()}, -1);
  }

 private:
  int64_t dim_;
};
TORCH_MODULE(AudioSinusPositionEmbedding);

// Timestep embedding: sinusoidal -> MLP(SiLU)
class AudioTimestepEmbeddingImpl : public torch::nn::Module {
 public:
  explicit AudioTimestepEmbeddingImpl(int64_t dim,
                                      int64_t freq_embed_dim = 256) {
    time_embed_ = register_module("time_embed",
                                  AudioSinusPositionEmbedding(freq_embed_dim));
    time_mlp_ = register_module(
        "time_mlp",
        torch::nn::Sequential(torch::nn::Linear(freq_embed_dim, dim),
                              torch::nn::SiLU(),
                              torch::nn::Linear(dim, dim)));
  }

  torch::Tensor forward(const torch::Tensor& timestep) {
    torch::Tensor h = time_embed_->forward(timestep);
    h = h.to(timestep.dtype());
    return time_mlp_->forward(h);
  }

 private:
  AudioSinusPositionEmbedding time_embed_{nullptr};
  torch::nn::Sequential time_mlp_{nullptr};
};
TORCH_MODULE(AudioTimestepEmbedding);

// Rotary position embedding (Qwen2-style, lazy build)
class AudioRotaryEmbeddingImpl : public torch::nn::Module {
 public:
  explicit AudioRotaryEmbeddingImpl(int64_t dim,
                                    int64_t max_position_embeddings = 2048,
                                    float base = 100000.0f)
      : dim_(dim),
        max_position_embeddings_(max_position_embeddings),
        base_(base),
        cached_len_(0) {}

  // Returns (cos, sin) tensors each of shape (seq_len, dim)
  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x,
                                                  int64_t seq_len = -1) {
    if (seq_len < 0) {
      seq_len = x.size(1);
    }
    if (!cos_.defined() || seq_len > cached_len_ ||
        x.device() != cached_device_) {
      build(std::max(seq_len, max_position_embeddings_),
            x.device(),
            x.scalar_type());
    }
    return {cos_.slice(0, 0, seq_len).to(x.dtype()),
            sin_.slice(0, 0, seq_len).to(x.dtype())};
  }

 private:
  void build(int64_t seq_len, torch::Device device, torch::ScalarType dtype) {
    auto arange_opts = torch::dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor inv_freq =
        1.0f /
        torch::pow(static_cast<float>(base_),
                   torch::arange(0, dim_, 2, arange_opts).to(torch::kFloat32) /
                       static_cast<float>(dim_));
    torch::Tensor t = torch::arange(seq_len, arange_opts).to(torch::kFloat32);
    torch::Tensor freqs = torch::outer(t, inv_freq);
    torch::Tensor emb = torch::cat({freqs, freqs}, -1);
    cos_ = emb.cos().to(dtype).to(device);
    sin_ = emb.sin().to(dtype).to(device);
    cached_len_ = seq_len;
    cached_device_ = device;
  }

  int64_t dim_, max_position_embeddings_, cached_len_;
  float base_;
  torch::Device cached_device_{torch::kCPU};
  torch::Tensor cos_, sin_;
};
TORCH_MODULE(AudioRotaryEmbedding);

// Apply rotary embedding to (B, H, S, D) tensor
inline torch::Tensor rotate_half(const torch::Tensor& x) {
  std::vector<torch::Tensor> chunks = x.chunk(2, -1);
  return torch::cat({-chunks[1], chunks[0]}, -1);
}

inline torch::Tensor apply_rotary_emb(const torch::Tensor& x,
                                      const torch::Tensor& cos,
                                      const torch::Tensor& sin) {
  // cos/sin: (S, D); x: (B, H, S, D)
  torch::Tensor c = cos.unsqueeze(0).unsqueeze(0).to(x.device());
  torch::Tensor s = sin.unsqueeze(0).unsqueeze(0).to(x.device());
  return (x.to(torch::kFloat32) * c + rotate_half(x).to(torch::kFloat32) * s)
      .to(x.dtype());
}

// Global Response Normalization (GRN) for ConvNeXtV2
class AudioGRNImpl : public torch::nn::Module {
 public:
  explicit AudioGRNImpl(int64_t dim) {
    gamma_ = register_parameter("gamma", torch::zeros({1, 1, dim}));
    beta_ = register_parameter("beta", torch::zeros({1, 1, dim}));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    // x: (B, S, D)
    torch::Tensor gx = torch::norm(x, 2, 1, /*keepdim=*/true);
    torch::Tensor nx = gx / (gx.mean(-1, /*keepdim=*/true) + 1e-6f);
    return gamma_ * (x * nx) + beta_ + x;
  }

 private:
  torch::Tensor gamma_, beta_;
};
TORCH_MODULE(AudioGRN);

// ConvNeXtV2 block for 1-D sequence text conditioning
class AudioConvNeXtV2BlockImpl : public torch::nn::Module {
 public:
  AudioConvNeXtV2BlockImpl(int64_t dim,
                           int64_t intermediate_dim,
                           int64_t dilation = 1,
                           int64_t kernel_size = 7,
                           float eps = 1e-6f) {
    int64_t padding = (dilation * (kernel_size - 1)) / 2;
    dwconv_ = register_module(
        "dwconv",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(dim, dim, kernel_size)
                              .padding(padding)
                              .groups(dim)
                              .dilation(dilation)));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim}).eps(eps)));
    pwconv1_ =
        register_module("pwconv1", torch::nn::Linear(dim, intermediate_dim));
    act_ = register_module("act", torch::nn::SiLU());
    grn_ = register_module("grn", AudioGRN(intermediate_dim));
    pwconv2_ =
        register_module("pwconv2", torch::nn::Linear(intermediate_dim, dim));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    // x: (B, S, D)
    torch::Tensor h = x.transpose(1, 2);  // -> (B, D, S)
    h = dwconv_->forward(h);
    h = h.transpose(1, 2);  // -> (B, S, D)
    h = norm_->forward(h);
    h = pwconv1_->forward(h);
    h = act_->forward(h);
    h = grn_->forward(h);
    h = pwconv2_->forward(h);
    return x + h;
  }

 private:
  torch::nn::Conv1d dwconv_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  torch::nn::Linear pwconv1_{nullptr};
  torch::nn::SiLU act_{nullptr};
  AudioGRN grn_{nullptr};
  torch::nn::Linear pwconv2_{nullptr};
};
TORCH_MODULE(AudioConvNeXtV2Block);

// Embedder: linear projection with optional mask
class AudioEmbedderImpl : public torch::nn::Module {
 public:
  AudioEmbedderImpl(int64_t in_dim, int64_t out_dim) {
    proj_ = register_module(
        "proj",
        torch::nn::Sequential(torch::nn::Linear(in_dim, out_dim),
                              torch::nn::SiLU(),
                              torch::nn::Linear(out_dim, out_dim)));
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::optional<torch::Tensor> mask = std::nullopt) {
    torch::Tensor h = x;
    if (mask.has_value()) {
      h = h.masked_fill(mask->logical_not().unsqueeze(-1), 0.0f);
    }
    h = proj_->forward(h);
    if (mask.has_value()) {
      h = h.masked_fill(mask->logical_not().unsqueeze(-1), 0.0f);
    }
    return h;
  }

 private:
  torch::nn::Sequential proj_{nullptr};
};
TORCH_MODULE(AudioEmbedder);

// AdaLN MLP: SiLU -> Linear
class AudioAdaLNMLPImpl : public torch::nn::Module {
 public:
  AudioAdaLNMLPImpl(int64_t in_dim, int64_t out_dim) {
    mlp_ = register_module(
        "mlp",
        torch::nn::Sequential(torch::nn::SiLU(),
                              torch::nn::Linear(in_dim, out_dim)));
  }

  torch::Tensor forward(const torch::Tensor& x) { return mlp_->forward(x); }

 private:
  torch::nn::Sequential mlp_{nullptr};
};
TORCH_MODULE(AudioAdaLNMLP);

// AdaLayerNormZeroFinal: used for output normalization
class AudioAdaLayerNormZeroFinalImpl : public torch::nn::Module {
 public:
  explicit AudioAdaLayerNormZeroFinalImpl(int64_t dim, float eps = 1e-6f) {
    linear_ = register_module("linear", torch::nn::Linear(dim, dim * 2));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));
  }

  torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& emb) {
    torch::Tensor e = linear_->forward(torch::silu(emb));
    std::vector<torch::Tensor> chunks = e.chunk(2, -1);
    torch::Tensor scale = chunks[0], shift = chunks[1];
    torch::Tensor h = norm_->forward(x.to(torch::kFloat32)).to(x.dtype());
    if (scale.dim() == 2) {
      h = h * (1.0f + scale.unsqueeze(1)) + shift.unsqueeze(1);
    } else {
      h = h * (1.0f + scale) + shift;
    }
    return h;
  }

 private:
  torch::nn::Linear linear_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
};
TORCH_MODULE(AudioAdaLayerNormZeroFinal);

// LayerNorm + modulate helper: normalize then apply (scale, shift)
inline torch::Tensor audio_modulate(const torch::Tensor& x,
                                    const torch::Tensor& scale,
                                    const torch::Tensor& shift,
                                    float eps = 1e-6f) {
  torch::Tensor h = torch::layer_norm(x.to(torch::kFloat32),
                                      {x.size(-1)},
                                      /*weight=*/{},
                                      /*bias=*/{},
                                      eps)
                        .to(x.dtype());
  if (scale.dim() == 2) {
    return h * (1.0f + scale.unsqueeze(1)) + shift.unsqueeze(1);
  }
  return h * (1.0f + scale) + shift;
}

// Self-attention with optional RoPE
class AudioSelfAttentionImpl : public torch::nn::Module {
 public:
  AudioSelfAttentionImpl(int64_t dim,
                         int64_t heads,
                         int64_t dim_head,
                         bool use_qk_norm = false,
                         float eps = 1e-6f)
      : heads_(heads), inner_dim_(dim_head * heads) {
    to_q_ = register_module("to_q", torch::nn::Linear(dim, inner_dim_));
    to_k_ = register_module("to_k", torch::nn::Linear(dim, inner_dim_));
    to_v_ = register_module("to_v", torch::nn::Linear(dim, inner_dim_));
    if (use_qk_norm) {
      q_norm_ = register_module("q_norm", AudioRMSNorm(inner_dim_, eps));
      k_norm_ = register_module("k_norm", AudioRMSNorm(inner_dim_, eps));
      use_qk_norm_ = true;
    }
    // Python: to_out is a ModuleList [Linear, Dropout]; we only need to_out.0
    out_proj_ = register_module("to_out_0", torch::nn::Linear(inner_dim_, dim));
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::optional<torch::Tensor> mask = std::nullopt,
                        std::optional<std::pair<torch::Tensor, torch::Tensor>>
                            rope = std::nullopt) {
    int64_t B = x.size(0);
    int64_t S = x.size(1);
    int64_t head_dim = inner_dim_ / heads_;

    torch::Tensor q = to_q_->forward(x);
    torch::Tensor k = to_k_->forward(x);
    torch::Tensor v = to_v_->forward(x);

    if (use_qk_norm_) {
      q = q_norm_->forward(q);
      k = k_norm_->forward(k);
    }

    // Reshape to (B, H, S, head_dim)
    q = q.view({B, S, heads_, head_dim}).transpose(1, 2);
    k = k.view({B, S, heads_, head_dim}).transpose(1, 2);
    v = v.view({B, S, heads_, head_dim}).transpose(1, 2);

    if (rope.has_value()) {
      q = apply_rotary_emb(q, rope->first, rope->second);
      k = apply_rotary_emb(k, rope->first, rope->second);
    }

    // Attention mask: convert bool (True=attend) to additive float mask.
    // cuDNN/Flash backends require a float additive mask, not a bool mask.
    std::optional<torch::Tensor> attn_mask;
    if (mask.has_value()) {
      // mask: (B, S) -> (B, 1, 1, S) -> broadcast to (B, H, S, S)
      torch::Tensor bool_mask = mask->unsqueeze(1)
                                    .unsqueeze(1)
                                    .expand({B, heads_, S, S})
                                    .contiguous();
      attn_mask = torch::zeros_like(bool_mask, q.options())
                      .masked_fill_(~bool_mask,
                                    -std::numeric_limits<float>::infinity());
    }

    // Manual attention to avoid torch::scaled_dot_product_attention dispatch,
    // which may select the cuDNN backend (unsupported for AudioDiT head_dim).
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    torch::Tensor scores =
        torch::matmul(q, k.transpose(-2, -1)) * scale;  // (B,H,S,S)
    if (attn_mask.has_value()) {
      scores = scores + attn_mask.value();
    }
    torch::Tensor weights =
        torch::softmax(scores.to(torch::kFloat32), -1).to(q.dtype());
    torch::Tensor out = torch::matmul(weights, v);  // (B,H,S,head_dim)

    out =
        out.transpose(1, 2).contiguous().view({B, S, inner_dim_}).to(q.dtype());
    return out_proj_->forward(out);
  }

 private:
  int64_t heads_, inner_dim_;
  bool use_qk_norm_ = false;
  torch::nn::Linear to_q_{nullptr}, to_k_{nullptr}, to_v_{nullptr};
  torch::nn::Linear out_proj_{nullptr};
  AudioRMSNorm q_norm_{nullptr}, k_norm_{nullptr};
};
TORCH_MODULE(AudioSelfAttention);

// Cross-attention
class AudioCrossAttentionImpl : public torch::nn::Module {
 public:
  AudioCrossAttentionImpl(int64_t q_dim,
                          int64_t kv_dim,
                          int64_t heads,
                          int64_t dim_head,
                          bool use_qk_norm = false,
                          float eps = 1e-6f)
      : heads_(heads), inner_dim_(dim_head * heads) {
    to_q_ = register_module("to_q", torch::nn::Linear(q_dim, inner_dim_));
    to_k_ = register_module("to_k", torch::nn::Linear(kv_dim, inner_dim_));
    to_v_ = register_module("to_v", torch::nn::Linear(kv_dim, inner_dim_));
    if (use_qk_norm) {
      q_norm_ = register_module("q_norm", AudioRMSNorm(inner_dim_, eps));
      k_norm_ = register_module("k_norm", AudioRMSNorm(inner_dim_, eps));
      use_qk_norm_ = true;
    }
    // Python: to_out is a ModuleList [Linear, Dropout]; we only need to_out.0
    out_proj_ =
        register_module("to_out_0", torch::nn::Linear(inner_dim_, q_dim));
  }

  torch::Tensor forward(
      const torch::Tensor& x,
      const torch::Tensor& cond,
      std::optional<torch::Tensor> mask = std::nullopt,
      std::optional<torch::Tensor> cond_mask = std::nullopt,
      std::optional<std::pair<torch::Tensor, torch::Tensor>> rope =
          std::nullopt,
      std::optional<std::pair<torch::Tensor, torch::Tensor>> cond_rope =
          std::nullopt) {
    int64_t B = x.size(0);
    int64_t S = x.size(1);
    int64_t Sc = cond.size(1);
    int64_t head_dim = inner_dim_ / heads_;

    torch::Tensor q = to_q_->forward(x);
    torch::Tensor k = to_k_->forward(cond);
    torch::Tensor v = to_v_->forward(cond);

    if (use_qk_norm_) {
      q = q_norm_->forward(q);
      k = k_norm_->forward(k);
    }

    q = q.view({B, S, heads_, head_dim}).transpose(1, 2);
    k = k.view({B, Sc, heads_, head_dim}).transpose(1, 2);
    v = v.view({B, Sc, heads_, head_dim}).transpose(1, 2);

    if (rope.has_value()) {
      q = apply_rotary_emb(q, rope->first, rope->second);
    }
    if (cond_rope.has_value()) {
      k = apply_rotary_emb(k, cond_rope->first, cond_rope->second);
    }

    // Build cross-attn mask from cond_mask: (B, H, S, Sc)
    // cond_mask is True for valid (attend) tokens.
    // Convert bool to additive float mask for cuDNN/Flash backend
    // compatibility.
    std::optional<torch::Tensor> attn_mask;
    if (cond_mask.has_value()) {
      // cond_mask: (B, Sc) -> (B, 1, 1, Sc) -> broadcast to (B, H, S, Sc)
      torch::Tensor bool_mask = cond_mask->unsqueeze(1)
                                    .unsqueeze(1)
                                    .expand({B, heads_, S, Sc})
                                    .contiguous();
      attn_mask = torch::zeros_like(bool_mask, q.options())
                      .masked_fill_(~bool_mask,
                                    -std::numeric_limits<float>::infinity());
    }

    // Manual attention to avoid torch::scaled_dot_product_attention dispatch,
    // which may select the cuDNN backend (unsupported for AudioDiT head_dim).
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    torch::Tensor scores =
        torch::matmul(q, k.transpose(-2, -1)) * scale;  // (B,H,S,Sc)
    if (attn_mask.has_value()) {
      scores = scores + attn_mask.value();
    }
    // nan_to_num(0): when all keys are masked (all-false cond_mask), every
    // score is -inf and softmax produces NaN (0/0).  Replace NaN with 0 so the
    // unconditional branch returns zeros from cross-attention instead of NaN.
    torch::Tensor weights =
        torch::nan_to_num(torch::softmax(scores.to(torch::kFloat32), -1), 0.0)
            .to(q.dtype());
    torch::Tensor out = torch::matmul(weights, v);  // (B,H,S,head_dim)

    out =
        out.transpose(1, 2).contiguous().view({B, S, inner_dim_}).to(q.dtype());
    return out_proj_->forward(out);
  }

 private:
  int64_t heads_, inner_dim_;
  bool use_qk_norm_ = false;
  torch::nn::Linear to_q_{nullptr}, to_k_{nullptr}, to_v_{nullptr};
  torch::nn::Linear out_proj_{nullptr};
  AudioRMSNorm q_norm_{nullptr}, k_norm_{nullptr};
};
TORCH_MODULE(AudioCrossAttention);

// FeedForward: Linear -> GELU(tanh) -> Dropout -> Linear
// Official Python: self.ff = nn.Sequential(Linear, GELU, Dropout, Linear)
// Checkpoint keys: "ff.0.weight" (first Linear), "ff.3.weight" (second Linear).
// load_module_from_state_dicts applies checkpoint_key_to_cpp_key to both sides
// so "ff.0" -> "ff_0" and "ff.3" -> "ff_3" match correctly.
class AudioFeedForwardImpl : public torch::nn::Module {
 public:
  explicit AudioFeedForwardImpl(int64_t dim,
                                float mult = 4.0f,
                                float dropout = 0.0f) {
    int64_t inner = static_cast<int64_t>(dim * mult);
    ff_ = register_module(
        "ff",
        torch::nn::Sequential(
            torch::nn::Linear(dim, inner),
            torch::nn::GELU(torch::nn::GELUOptions().approximate("tanh")),
            torch::nn::Dropout(torch::nn::DropoutOptions(dropout)),
            torch::nn::Linear(inner, dim)));
  }

  torch::Tensor forward(const torch::Tensor& x) { return ff_->forward(x); }

 private:
  torch::nn::Sequential ff_{nullptr};
};
TORCH_MODULE(AudioFeedForward);

// ============================================================================
// AudioDiT Transformer Block
// ============================================================================

struct AudioDiTBlockConfig {
  int64_t dim = 1536;
  int64_t heads = 24;
  float ff_mult = 4.0f;
  bool use_cross_attn = true;
  bool use_cross_attn_norm = false;  // dit_cross_attn_norm=False by default
  bool use_qk_norm = true;           // dit_qk_norm=True by default
  // "global" uses a single global AdaLN MLP; "local" uses per-block MLP
  std::string adaln_type = "global";
  float eps = 1e-6f;
};

class AudioDiTBlockImpl : public torch::nn::Module {
 public:
  explicit AudioDiTBlockImpl(const AudioDiTBlockConfig& cfg)
      : use_cross_attn_(cfg.use_cross_attn),
        use_cross_attn_norm_(cfg.use_cross_attn_norm),
        adaln_type_(cfg.adaln_type) {
    int64_t dim = cfg.dim;
    int64_t heads = cfg.heads;
    int64_t dim_head = dim / heads;

    // AdaLN gating params
    if (cfg.adaln_type == "local") {
      adaln_mlp_ = register_module("adaln_mlp", AudioAdaLNMLP(dim, dim * 6));
    } else {
      // Global: per-block learnable scale_shift offset
      adaln_scale_shift_ = register_parameter(
          "adaln_scale_shift",
          torch::randn({dim * 6}) / std::sqrt(static_cast<float>(dim)));
    }

    self_attn_ = register_module(
        "self_attn",
        AudioSelfAttention(dim, heads, dim_head, cfg.use_qk_norm, cfg.eps));

    if (cfg.use_cross_attn) {
      cross_attn_ = register_module(
          "cross_attn",
          AudioCrossAttention(
              dim, dim, heads, dim_head, cfg.use_qk_norm, cfg.eps));
      if (cfg.use_cross_attn_norm) {
        cross_attn_norm_x_ = register_module(
            "cross_attn_norm",
            torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({dim}).elementwise_affine(true).eps(
                    cfg.eps)));
        cross_attn_norm_cond_ = register_module(
            "cross_attn_norm_c",
            torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({dim}).elementwise_affine(true).eps(
                    cfg.eps)));
      }
    }

    ffn_ = register_module("ffn", AudioFeedForward(dim, cfg.ff_mult));
  }

  // forward: returns updated x
  // adaln_global_out: pre-computed global AdaLN output if adaln_type=="global"
  torch::Tensor forward(
      const torch::Tensor& x,
      const torch::Tensor& t,
      const torch::Tensor& cond,
      std::optional<torch::Tensor> mask = std::nullopt,
      std::optional<torch::Tensor> cond_mask = std::nullopt,
      std::optional<std::pair<torch::Tensor, torch::Tensor>> rope =
          std::nullopt,
      std::optional<std::pair<torch::Tensor, torch::Tensor>> cond_rope =
          std::nullopt,
      std::optional<torch::Tensor> adaln_global_out = std::nullopt) {
    torch::Tensor adaln_out;
    if (adaln_type_ == "local") {
      adaln_out = adaln_mlp_->forward(t);
    } else {
      // global: add per-block learnable offset to pre-computed global out
      adaln_out = adaln_global_out.value() +
                  adaln_scale_shift_.unsqueeze(0);  // (1, dim*6) broadcast
    }

    // Split into 6 chunks: gate_sa, scale_sa, shift_sa, gate_ffn, scale_ffn,
    // shift_ffn
    std::vector<torch::Tensor> chunks = adaln_out.chunk(6, -1);
    torch::Tensor gate_sa = chunks[0];
    torch::Tensor scale_sa = chunks[1];
    torch::Tensor shift_sa = chunks[2];
    torch::Tensor gate_ffn = chunks[3];
    torch::Tensor scale_ffn = chunks[4];
    torch::Tensor shift_ffn = chunks[5];

    torch::Tensor h = x;

    // Self-attention with AdaLN modulation
    torch::Tensor norm_x = audio_modulate(h, scale_sa, shift_sa);
    torch::Tensor sa_out = self_attn_->forward(norm_x, mask, rope);
    // gate: (B, dim) -> unsqueeze to (B, 1, dim) to broadcast over seq
    h = h + gate_sa.unsqueeze(1) * sa_out;

    // Cross-attention (no gating — plain residual add)
    torch::Tensor ca_out;
    if (use_cross_attn_) {
      torch::Tensor xn = h, cn = cond;
      if (use_cross_attn_norm_) {
        xn = cross_attn_norm_x_->forward(h);
        cn = cross_attn_norm_cond_->forward(cond);
      }
      ca_out = cross_attn_->forward(xn, cn, mask, cond_mask, rope, cond_rope);
      h = h + ca_out;
    }

    // FFN with AdaLN modulation
    torch::Tensor norm_h = audio_modulate(h, scale_ffn, shift_ffn);
    torch::Tensor ff_out = ffn_->forward(norm_h);
    h = h + gate_ffn.unsqueeze(1) * ff_out;

    return h;
  }

 private:
  bool use_cross_attn_, use_cross_attn_norm_;
  std::string adaln_type_;
  AudioAdaLNMLP adaln_mlp_{nullptr};
  torch::Tensor adaln_scale_shift_;  // (dim*6,) parameter for global AdaLN
  AudioSelfAttention self_attn_{nullptr};
  AudioCrossAttention cross_attn_{nullptr};
  torch::nn::LayerNorm cross_attn_norm_x_{nullptr};
  torch::nn::LayerNorm cross_attn_norm_cond_{nullptr};
  AudioFeedForward ffn_{nullptr};
};
TORCH_MODULE(AudioDiTBlock);

// ============================================================================
// AudioDiT Transformer (backbone)
// ============================================================================

struct AudioDiTTransformerConfig {
  int64_t dim = 1536;
  int64_t depth = 24;
  int64_t heads = 24;
  float ff_mult = 4.0f;
  int64_t latent_dim = 64;
  int64_t text_dim = 768;  // UMT5-base output dim
  bool long_skip = true;
  bool text_conv = true;  // 4 ConvNeXtV2 blocks on text
  bool use_latent_condition = true;
  std::string adaln_type = "global";
  float eps = 1e-6f;
  // Official Python: repa_dit_layer (default 8). After block (repa_layer-1),
  // apply long_skip early: x = x + x_clone. Then apply again after all blocks.
  int64_t repa_layer = 8;
};

class AudioDiTTransformerImpl final : public torch::nn::Module {
 public:
  explicit AudioDiTTransformerImpl(const AudioDiTTransformerConfig& cfg)
      : dim_(cfg.dim),
        depth_(cfg.depth),
        long_skip_(cfg.long_skip),
        text_conv_(cfg.text_conv),
        use_latent_condition_(cfg.use_latent_condition),
        adaln_type_(cfg.adaln_type),
        repa_layer_(cfg.repa_layer) {
    int64_t dim_head = cfg.dim / cfg.heads;

    time_embed_ =
        register_module("time_embed", AudioTimestepEmbedding(cfg.dim));
    input_embed_ =
        register_module("input_embed", AudioEmbedder(cfg.latent_dim, cfg.dim));
    text_embed_ =
        register_module("text_embed", AudioEmbedder(cfg.text_dim, cfg.dim));
    rotary_embed_ = register_module(
        "rotary_embed", AudioRotaryEmbedding(dim_head, 2048, 100000.0f));

    // Transformer blocks
    AudioDiTBlockConfig blk_cfg;
    blk_cfg.dim = cfg.dim;
    blk_cfg.heads = cfg.heads;
    blk_cfg.ff_mult = cfg.ff_mult;
    blk_cfg.use_cross_attn = true;
    blk_cfg.use_cross_attn_norm = false;  // dit_cross_attn_norm=False
    blk_cfg.use_qk_norm = true;           // dit_qk_norm=True
    blk_cfg.adaln_type = cfg.adaln_type;
    blk_cfg.eps = cfg.eps;
    for (int64_t i = 0; i < cfg.depth; ++i) {
      dit_blocks_.push_back(register_module("blocks_" + std::to_string(i),
                                            AudioDiTBlock(blk_cfg)));
    }

    // Output
    norm_out_ = register_module("norm_out",
                                AudioAdaLayerNormZeroFinal(cfg.dim, cfg.eps));
    proj_out_ =
        register_module("proj_out", torch::nn::Linear(cfg.dim, cfg.latent_dim));

    // Global AdaLN MLP
    if (cfg.adaln_type == "global") {
      adaln_global_mlp_ = register_module("adaln_global_mlp",
                                          AudioAdaLNMLP(cfg.dim, cfg.dim * 6));
    }

    // Text ConvNeXtV2 stack
    if (cfg.text_conv) {
      for (int64_t i = 0; i < 4; ++i) {
        text_conv_blocks_.push_back(
            register_module("text_conv_layer_" + std::to_string(i),
                            AudioConvNeXtV2Block(cfg.dim, cfg.dim * 2)));
      }
    }

    // Latent conditioning
    if (cfg.use_latent_condition) {
      latent_embed_ = register_module("latent_embed",
                                      AudioEmbedder(cfg.latent_dim, cfg.dim));
      latent_cond_embedder_ = register_module(
          "latent_cond_embedder", AudioEmbedder(cfg.dim * 2, cfg.dim));
    }
  }

  // Forward pass
  // x:    (B, S, latent_dim)   — noised latent
  // text: (B, Sc, text_dim)    — text embeddings
  // text_len: (B,)             — number of valid text tokens
  // time: (B,) or scalar       — timestep in [0,1]
  // mask: (B, S) bool          — valid latent frames
  // cond_mask: (B, Sc) bool    — valid text tokens
  // latent_cond: (B, S, latent_dim) — prompt conditioning
  torch::Tensor forward(
      const torch::Tensor& x_in,
      const torch::Tensor& text_in,
      const torch::Tensor& text_len,
      const torch::Tensor& time,
      std::optional<torch::Tensor> mask = std::nullopt,
      std::optional<torch::Tensor> cond_mask = std::nullopt,
      std::optional<torch::Tensor> latent_cond = std::nullopt) {
    // Cast to model dtype (use proj_out_ weight as reference)
    auto dtype = proj_out_->weight.scalar_type();
    torch::Tensor x = x_in.to(dtype);
    // text_in is already truncated to actual token length at the pipeline level
    // (no padding zeros), matching official Python which never pads text.
    torch::Tensor text = text_in.to(dtype);
    torch::Tensor t_in = time.to(dtype);

    int64_t B = x.size(0);
    if (t_in.dim() == 0) {
      t_in = t_in.repeat(B);
    }

    // Timestep embedding
    torch::Tensor t = time_embed_->forward(t_in);  // (B, dim)

    // Text embedding + optional ConvNeXtV2 processing
    text = text_embed_->forward(text, cond_mask);
    if (text_conv_) {
      for (size_t ci = 0; ci < text_conv_blocks_.size(); ++ci) {
        text = text_conv_blocks_[ci]->forward(text);
      }
      if (cond_mask.has_value()) {
        text = text.masked_fill(cond_mask->logical_not().unsqueeze(-1), 0.0f);
      }
    }

    // Input embedding + optional latent conditioning
    x = input_embed_->forward(x, mask);
    if (use_latent_condition_ && latent_cond.has_value()) {
      torch::Tensor lc =
          latent_embed_->forward(latent_cond.value().to(dtype), mask);
      // Do NOT pass mask here: input_embed and latent_embed have already
      // zeroed out padding positions; passing mask again would corrupt the
      // valid positions after the concat+proj. (Matches official Python code.)
      x = latent_cond_embedder_->forward(torch::cat({x, lc}, -1));
    }

    // Long skip clone
    torch::Tensor x_clone;
    if (long_skip_) {
      x_clone = x.clone();
    }

    // Rotary embeddings
    int64_t seq_len = x.size(1);
    int64_t text_seq_len = text.size(1);
    auto [cos_x, sin_x] = rotary_embed_->forward(x, seq_len);
    auto [cos_t, sin_t] = rotary_embed_->forward(text, text_seq_len);
    auto rope = std::make_optional(std::make_pair(cos_x, sin_x));
    auto cond_rope = std::make_optional(std::make_pair(cos_t, sin_t));

    // Global AdaLN
    std::optional<torch::Tensor> adaln_mlp_out;
    torch::Tensor norm_cond;
    if (adaln_type_ == "global") {
      // Use text mean for conditioning.
      // Compute in float32 to avoid fp16 underflow: 1e-9 < fp16 min (6e-5).
      if (cond_mask.has_value()) {
        torch::Tensor text_len_f =
            text_len.unsqueeze(1).to(torch::kFloat32) + 1e-9f;
        torch::Tensor text_mean =
            (text.to(torch::kFloat32).sum(1) / text_len_f).to(text.dtype());
        norm_cond = t + text_mean;
      } else {
        norm_cond = t;
      }
      adaln_mlp_out = adaln_global_mlp_->forward(norm_cond);
    }

    // Run blocks
    for (int64_t i = 0; i < depth_; ++i) {
      x = dit_blocks_[i]->forward(
          x, t, text, mask, cond_mask, rope, cond_rope, adaln_mlp_out);
      // Official Python: after block (repa_layer-1), apply long_skip early.
      // repa_dit_layer=8 → after block index 7, x = x + x_clone.
      // The long_skip is then applied again after all blocks (total 2
      // additions).
      if (long_skip_ && repa_layer_ > 0 && i == repa_layer_ - 1) {
        x = x + x_clone;
      }
    }

    // Long skip connection (second application, after all blocks)
    if (long_skip_) {
      x = x + x_clone;
    }

    // Output normalization and projection
    torch::Tensor cond_for_out =
        (adaln_type_ == "global" && norm_cond.defined()) ? norm_cond : t;
    x = norm_out_->forward(x, cond_for_out);
    x = proj_out_->forward(x);  // (B, S, latent_dim)
    return x;
  }

 private:
  int64_t dim_, depth_, repa_layer_;
  bool long_skip_, text_conv_, use_latent_condition_;
  std::string adaln_type_;

  AudioTimestepEmbedding time_embed_{nullptr};
  AudioEmbedder input_embed_{nullptr};
  AudioEmbedder text_embed_{nullptr};
  AudioRotaryEmbedding rotary_embed_{nullptr};
  std::vector<AudioDiTBlock> dit_blocks_;
  AudioAdaLayerNormZeroFinal norm_out_{nullptr};
  torch::nn::Linear proj_out_{nullptr};
  AudioAdaLNMLP adaln_global_mlp_{nullptr};
  std::vector<AudioConvNeXtV2Block> text_conv_blocks_;
  AudioEmbedder latent_embed_{nullptr};
  AudioEmbedder latent_cond_embedder_{nullptr};
};
TORCH_MODULE(AudioDiTTransformer);

// ============================================================================
// APG (Adaptive Projected Guidance) helpers
// ============================================================================

struct MomentumBuffer {
  float momentum = -0.3f;
  torch::Tensor running_average;
};

inline void momentum_buffer_update(MomentumBuffer& buf,
                                   const torch::Tensor& update_value) {
  if (!buf.running_average.defined()) {
    buf.running_average = update_value.clone();
  } else {
    buf.running_average = update_value + buf.momentum * buf.running_average;
  }
}

// Project v0 onto the direction of v1 (and orthogonal complement)
inline std::pair<torch::Tensor, torch::Tensor> apg_project(
    const torch::Tensor& v0,
    const torch::Tensor& v1) {
  // Work in double precision for numerical stability
  torch::Tensor v0d = v0.to(torch::kDouble);
  torch::Tensor v1d = v1.to(torch::kDouble);
  // Normalize v1 over last two dims
  torch::Tensor v1_norm =
      v1d.norm(2, {-1, -2}, /*keepdim=*/true).clamp_min(1e-12);
  torch::Tensor v1n = v1d / v1_norm;
  torch::Tensor v0_parallel = (v0d * v1n).sum({-1, -2}, /*keepdim=*/true) * v1n;
  torch::Tensor v0_orthogonal = v0d - v0_parallel;
  return {v0_parallel.to(v0.dtype()), v0_orthogonal.to(v0.dtype())};
}

// APG guidance: combine cond/uncond predictions with momentum + projection
inline torch::Tensor apg_forward(const torch::Tensor& pred_cond,
                                 const torch::Tensor& pred_uncond,
                                 float guidance_scale,
                                 MomentumBuffer* buffer,
                                 float eta = 0.5f,
                                 float norm_threshold = 0.0f) {
  torch::Tensor diff = pred_cond - pred_uncond;
  if (buffer != nullptr) {
    momentum_buffer_update(*buffer, diff);
    diff = buffer->running_average;
  }
  if (norm_threshold > 0.0f) {
    torch::Tensor ones = torch::ones_like(diff);
    torch::Tensor diff_norm = diff.norm(2, {-1, -2}, /*keepdim=*/true);
    torch::Tensor scale_factor = torch::minimum(
        ones, torch::tensor(norm_threshold).to(diff.device()) / diff_norm);
    diff = diff * scale_factor;
  }
  auto [diff_parallel, diff_orthogonal] = apg_project(diff, pred_cond);
  torch::Tensor normalized_update = diff_orthogonal + eta * diff_parallel;
  return pred_cond + guidance_scale * normalized_update;
}

// ============================================================================
// UMT5 Text Encoder wrapper
// ============================================================================

// Wraps UMT5EncoderModel with UMT5-specific post-processing matching the
// official Python AudioDiTModel.encode_text():
//   emb = F.layer_norm(last_hidden_state, (d_model,), eps=1e-6)
//   first_hidden = F.layer_norm(embed_tokens(input_ids), (d_model,), eps=1e-6)
//   return (emb + first_hidden).float()
class UMT5TextEncoderImpl : public torch::nn::Module {
 public:
  explicit UMT5TextEncoderImpl(const ModelContext& context)
      : context_(context) {
    umt5_ = register_module("umt5", UMT5EncoderModel(context));
    d_model_ = context.get_model_args().d_model();
  }

  // Encode text tokens to embeddings.
  // input_ids: (B, S) int64
  // Returns: (B, S, d_model) float32
  torch::Tensor forward(const torch::Tensor& input_ids) {
    // UMT5EncoderModel applies final_layer_norm internally.
    // Apply F.layer_norm (no learnable params) to match official Python.
    torch::Tensor umt5_out = umt5_->forward(input_ids).to(torch::kFloat32);
    torch::Tensor last_hidden =
        torch::layer_norm(umt5_out, {d_model_}, {}, {}, 1e-6f);

    torch::Tensor embed_out =
        umt5_->get_input_embeddings()->forward(input_ids).to(torch::kFloat32);
    torch::Tensor first_hidden =
        torch::layer_norm(embed_out, {d_model_}, {}, {}, 1e-6f);

    return last_hidden + first_hidden;
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    umt5_->load_model(std::move(loader));
  }

  void load_model_from_state_dicts(
      std::vector<std::unique_ptr<StateDict>>& state_dicts,
      const std::string& key_prefix = "text_encoder.") {
    umt5_->load_from_state_dicts(state_dicts, key_prefix);
  }

 private:
  ModelContext context_;
  UMT5EncoderModel umt5_{nullptr};
  int64_t d_model_ = 768;
};
TORCH_MODULE(UMT5TextEncoder);

// ============================================================================
// Weight loading helper
// ============================================================================

// Load all weights from a DiTFolderLoader into a torch::nn::Module by matching
// named parameters/buffers. This is a bulk copy approach that works for any
// standard torch::nn module hierarchy without requiring hand-written
// load_state_dict methods.
//
// Weight names in the checkpoint are expected to match the C++ module's
// named_parameters() keys exactly (as produced by TORCH_MODULE registration).
// Load weights from state dicts into a module.
// If key_prefix is non-empty, only keys starting with that prefix are loaded,
// with the prefix stripped before matching against the module's parameters.
// This supports both per-component loaders (prefix="") and flat loaders where
// all components share one safetensors file (prefix="transformer.", "vae.",
// etc.) Translate a checkpoint key to the C++ module key. libtorch
// register_module forbids dots in names, so we register with underscores (e.g.
// "layers_0", "blocks_3", "to_out_0") while checkpoints use dots (e.g.
// "layers.0", "blocks.3", "to_out.0"). This function replaces ".<digit>" with
// "_<digit>" throughout the key.
inline std::string checkpoint_key_to_cpp_key(const std::string& key) {
  std::string result;
  result.reserve(key.size());
  for (size_t i = 0; i < key.size(); ++i) {
    if (key[i] == '.' && i + 1 < key.size() && std::isdigit(key[i + 1])) {
      result += '_';
    } else {
      result += key[i];
    }
  }
  return result;
}

inline void load_module_from_state_dicts(DiTFolderLoader& folder_loader,
                                         torch::nn::Module* module,
                                         const std::string& key_prefix = "") {
  // Keep named_parameters/named_buffers alive while we hold pointers into them.
  auto params = module->named_parameters(/*recurse=*/true);
  auto buffers = module->named_buffers(/*recurse=*/true);

  // Build param_map keyed by checkpoint_key_to_cpp_key(named_param_key) so that
  // lookup against the converted checkpoint key always succeeds.
  // Background: torch::nn::Sequential names children "0","1","2",... so
  // named_parameters() returns e.g. "mlp.1.weight".
  // checkpoint_key_to_cpp_key converts ".1" -> "_1", giving "mlp_1.weight".
  // If we store by raw key "mlp.1.weight" the lookup for "mlp_1.weight" fails.
  // Solution: store by the converted key so both sides use the same format.
  std::unordered_map<std::string, torch::Tensor*> param_map;
  for (auto& kv : params) {
    std::string mapped = checkpoint_key_to_cpp_key(kv.key());
    param_map[mapped] = &kv.value();
  }
  for (auto& kv : buffers) {
    param_map[checkpoint_key_to_cpp_key(kv.key())] = &kv.value();
  }

  // First pass: collect weight_g / weight_v pairs for weight_norm
  // reconstruction. PyTorch weight_norm stores a Conv's weight as two params:
  // weight_g (magnitude) and weight_v (direction). The effective weight = g * v
  // / ||v||. checkpoint key pattern: "some.module.weight_g" /
  // "some.module.weight_v" C++ param key pattern:  "some.module.weight"
  auto strip_prefix_fn = [&](const std::string& raw) -> std::string {
    if (key_prefix.empty()) return raw;
    if (raw.substr(0, key_prefix.size()) != key_prefix) return "";
    return raw.substr(key_prefix.size());
  };

  // key = stripped key (still dot-separated, no digit->underscore yet)
  std::unordered_map<std::string, torch::Tensor> wn_g, wn_v;
  static const std::string kSuffixG = ".weight_g";
  static const std::string kSuffixV = ".weight_v";
  for (const auto& state_dict_ptr : folder_loader.get_state_dicts()) {
    for (const auto& kv : *state_dict_ptr) {
      std::string key = strip_prefix_fn(kv.first);
      if (key.empty()) continue;
      if (key.size() > kSuffixG.size() &&
          key.substr(key.size() - kSuffixG.size()) == kSuffixG) {
        wn_g[key.substr(0, key.size() - kSuffixG.size())] = kv.second;
      } else if (key.size() > kSuffixV.size() &&
                 key.substr(key.size() - kSuffixV.size()) == kSuffixV) {
        wn_v[key.substr(0, key.size() - kSuffixV.size())] = kv.second;
      }
    }
  }

  int64_t loaded = 0;
  for (const auto& state_dict_ptr : folder_loader.get_state_dicts()) {
    for (const auto& kv : *state_dict_ptr) {
      const std::string& raw_key = kv.first;
      const torch::Tensor& src_tensor = kv.second;

      // Strip prefix if specified
      std::string key = raw_key;
      if (!key_prefix.empty()) {
        if (raw_key.substr(0, key_prefix.size()) != key_prefix) {
          continue;
        }
        key = raw_key.substr(key_prefix.size());
      }

      // Skip weight_norm components; they are reconstructed separately below.
      if ((key.size() > kSuffixG.size() &&
           key.substr(key.size() - kSuffixG.size()) == kSuffixG) ||
          (key.size() > kSuffixV.size() &&
           key.substr(key.size() - kSuffixV.size()) == kSuffixV)) {
        continue;
      }

      // Translate checkpoint key (dots before digits) to C++ key (underscores)
      std::string cpp_key = checkpoint_key_to_cpp_key(key);

      auto it = param_map.find(cpp_key);
      if (it == param_map.end()) {
        if (key_prefix.empty()) {
          LOG(WARNING) << "[AudioDiT load] Unknown key in checkpoint: " << key;
        }
        continue;
      }
      torch::Tensor& dst = *(it->second);
      if (!dst.defined()) {
        LOG(WARNING) << "[AudioDiT load] Skipping key with undefined dst: "
                     << key;
        continue;
      }
      torch::NoGradGuard no_grad;
      dst.copy_(src_tensor.to(dst.dtype()).to(dst.device()));
      ++loaded;
    }
  }

  // Second pass: reconstruct weight_norm weights.
  // effective_weight = weight_g * weight_v / ||weight_v||_per_output_channel
  for (const auto& gkv : wn_g) {
    const std::string& base = gkv.first;  // e.g. "encoder.in_conv"
    auto vit = wn_v.find(base);
    if (vit == wn_v.end()) continue;

    // The plain weight key in C++ is base + ".weight" (with digit->_
    // translation)
    std::string weight_cpp_key = checkpoint_key_to_cpp_key(base + ".weight");
    auto it = param_map.find(weight_cpp_key);
    if (it == param_map.end()) {
      LOG(WARNING) << "[AudioDiT load] weight_norm base not found in module: "
                   << weight_cpp_key;
      continue;
    }

    torch::Tensor g = gkv.second.to(torch::kFloat32);
    torch::Tensor v = vit->second.to(torch::kFloat32);
    // Norm over all dims except output channel (dim 0): reshape to (C_out, -1)
    int64_t c_out = v.size(0);
    torch::Tensor v_norm = v.view({c_out, -1}).norm(2, 1, /*keepdim=*/true);
    // Reshape v_norm back to (C_out, 1, 1, ...) for broadcasting
    std::vector<int64_t> norm_shape(v.dim(), 1);
    norm_shape[0] = c_out;
    v_norm = v_norm.view(norm_shape);
    torch::Tensor w = g * v / (v_norm + 1e-12f);

    torch::Tensor& dst = *(it->second);
    torch::NoGradGuard no_grad;
    dst.copy_(w.to(dst.dtype()).to(dst.device()));
    ++loaded;
  }
}

}  // namespace xllm
