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

#include <cmath>

namespace xllm::dit {

// Sinusoidal timestep embedding shared by the DiT transformers. Callers select
// the checkpoint-specific layout via flip_sin_to_cos and downscale_freq_shift;
// scale and max_period match the reference implementations (1.0 / 10000).
inline torch::Tensor get_timestep_embedding(const torch::Tensor& timesteps,
                                            int64_t embedding_dim,
                                            bool flip_sin_to_cos = false,
                                            double downscale_freq_shift = 1.0,
                                            double scale = 1.0,
                                            int64_t max_period = 10000) {
  CHECK_EQ(timesteps.dim(), 1) << "Timesteps should be a 1d-array";
  const int64_t half_dim = embedding_dim / 2;
  torch::Tensor exponent = -std::log(static_cast<double>(max_period)) *
                           torch::arange(0,
                                         half_dim,
                                         torch::TensorOptions()
                                             .dtype(torch::kFloat32)
                                             .device(timesteps.device()));
  exponent = exponent / (half_dim - downscale_freq_shift);

  torch::Tensor emb = torch::exp(exponent);
  emb = timesteps.unsqueeze(1).to(torch::kFloat32) * emb.unsqueeze(0);
  emb = scale * emb;
  emb = torch::cat({torch::sin(emb), torch::cos(emb)}, -1);

  if (flip_sin_to_cos) {
    emb = torch::cat({emb.slice(-1, half_dim), emb.slice(-1, 0, half_dim)}, -1);
  }
  if (embedding_dim % 2 == 1) {
    emb = torch::nn::functional::pad(
        emb, torch::nn::functional::PadFuncOptions({0, 1, 0, 0}));
  }
  return emb;
}

// Timestep projection module shared by the DiT transformers: holds the
// checkpoint-specific embedding layout and forwards to get_timestep_embedding.
class TimestepsImpl : public torch::nn::Module {
 public:
  explicit TimestepsImpl(int64_t num_channels,
                         bool flip_sin_to_cos = true,
                         float downscale_freq_shift = 0.0)
      : num_channels_(num_channels),
        flip_sin_to_cos_(flip_sin_to_cos),
        downscale_freq_shift_(downscale_freq_shift) {}

  torch::Tensor forward(const torch::Tensor& timesteps) {
    return get_timestep_embedding(
        timesteps, num_channels_, flip_sin_to_cos_, downscale_freq_shift_);
  }

 private:
  int64_t num_channels_;
  bool flip_sin_to_cos_;
  float downscale_freq_shift_;
};
TORCH_MODULE(Timesteps);

}  // namespace xllm::dit
