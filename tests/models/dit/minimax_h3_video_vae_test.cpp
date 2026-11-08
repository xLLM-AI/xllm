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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>

#include "models/dit/autoencoders/autoencoder_kl_minimax_h3_video.h"

namespace xllm {
namespace {

AutoencoderKLMiniMaxH3 make_small_video_vae() {
  JsonReader reader;
  CHECK(reader.parse_text(R"({
    "in_channels": 1,
    "out_channels": 1,
    "latent_channels": 1,
    "block_out_channels": [1, 1, 1],
    "layers_per_block": 1,
    "spatial_downsample_factors": [1, 1, 1],
    "temporal_downsample_factors": [1, 2, 2],
    "norm_num_groups": 1,
    "spatial_padding_mode": "constant",
    "decoder_num_layers": 1,
    "decoder_num_attention_heads": 1,
    "decoder_attention_head_dim": 64,
    "decoder_num_register_tokens": 1,
    "decoder_ffn_mult": 1,
    "latents_mean": [0.0],
    "latents_std": [1.0]
  })"));
  ModelArgs args;
  const auto loader =
      ModelRegistry::get_model_args_loader("AutoencoderKLMiniMaxH3");
  CHECK(loader(reader, &args));
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::TensorOptions options =
      torch::TensorOptions().device(device).dtype(torch::kFloat32);
  const ModelContext context(
      ParallelArgs(1, 1, nullptr), args, QuantArgs(), options);
  AutoencoderKLMiniMaxH3 vae(context);
  torch::manual_seed(42);
  for (torch::Tensor parameter : vae->parameters()) {
    parameter.uniform_(-0.25, 0.25);
  }
  vae->to(device);
  vae->eval();
  return vae;
}

TEST(MiniMaxH3VideoVaeTest, FinalFramesRetainCompleteTemporalWindowContext) {
  torch::NoGradGuard no_grad;
  auto vae = make_small_video_vae();
  const auto children = vae->named_children();
  auto* post_quant =
      children["post_quant_conv"]->as<minimax_h3_video::Conv3d1x1Impl>();
  auto* decoder = children["decoder"]->as<minimax_h3_video::ViTDecoder3dImpl>();
  ASSERT_NE(post_quant, nullptr);
  ASSERT_NE(decoder, nullptr);
  const torch::TensorOptions options = vae->parameters().front().options();
  for (const int64_t tokens : {7, 12, 37}) {
    SCOPED_TRACE(tokens);
    const torch::Tensor latent =
        torch::arange(tokens, options).view({1, 1, tokens, 1, 1});
    const torch::Tensor decoded = vae->decode(latent);
    const torch::Tensor final_window = decoder->forward(
        post_quant->forward(latent.slice(/*dim=*/2, tokens - 7, tokens)));
    const torch::Tensor expected_tail = final_window.slice(/*dim=*/2, 23, 28);
    const torch::Tensor actual_tail =
        decoded.slice(/*dim=*/2, decoded.size(2) - 5);
    EXPECT_TRUE(torch::allclose(actual_tail, expected_tail, 1e-5, 1e-6))
        << "maximum tail error: "
        << (actual_tail - expected_tail).abs().max().item<float>();
  }
}

TEST(MiniMaxH3VideoVaeTest, TemporalPaddingTrimsOnlyRepeatedFinalFrames) {
  torch::NoGradGuard no_grad;
  auto vae = make_small_video_vae();
  const torch::TensorOptions options = vae->parameters().front().options();
  const torch::Tensor latent =
      torch::arange(36, options).view({1, 1, 36, 1, 1});
  const torch::Tensor padded =
      torch::cat({latent, latent.slice(/*dim=*/2, 35)}, /*dim=*/2);
  const torch::Tensor decoded = vae->decode(latent);
  const torch::Tensor expected = vae->decode(padded).slice(/*dim=*/2, 0, 120);
  ASSERT_EQ(decoded.size(2), 120);
  EXPECT_TRUE(torch::allclose(decoded, expected, 1e-5, 1e-6));
}

TEST(MiniMaxH3VideoVaeTest, ShortLatentsRetainSingleWindowDecode) {
  torch::NoGradGuard no_grad;
  auto vae = make_small_video_vae();
  const torch::TensorOptions options = vae->parameters().front().options();
  for (const int64_t tokens : {1, 2}) {
    SCOPED_TRACE(tokens);
    const torch::Tensor decoded =
        vae->decode(torch::ones({1, 1, tokens, 1, 1}, options));
    EXPECT_EQ(decoded.size(2), tokens == 1 ? 1 : 5);
    EXPECT_TRUE(torch::isfinite(decoded).all().item<bool>());
  }
}

}  // namespace
}  // namespace xllm
