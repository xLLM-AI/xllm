/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/framework/state_dict/state_dict.h"
#include "lanczos_resample.h"
#include "models/dit/schedulers/flowmatch_euler_discrete_scheduler.h"

namespace xllm::dit {

float calculate_shift(int64_t image_seq_len,
                      int64_t base_seq_len = 256,
                      int64_t max_seq_len = 4096,
                      float base_shift = 0.5f,
                      float max_shift = 1.15f) {
  float m =
      (max_shift - base_shift) / static_cast<float>(max_seq_len - base_seq_len);
  float b = base_shift - m * static_cast<float>(base_seq_len);
  float mu = static_cast<float>(image_seq_len) * m + b;
  return mu;
}

std::pair<torch::Tensor, int64_t> retrieve_timesteps(
    xllm::FlowMatchEulerDiscreteScheduler scheduler,
    int64_t num_inference_steps = 0,
    torch::Device device = torch::kCPU,
    std::optional<std::vector<float>> sigmas = std::nullopt,
    std::optional<float> mu = std::nullopt) {
  torch::Tensor scheduler_timesteps;
  int64_t steps;
  if (sigmas.has_value()) {
    steps = sigmas->size();
    scheduler->set_timesteps(
        static_cast<int64_t>(steps), device, *sigmas, mu, std::nullopt);

    scheduler_timesteps = scheduler->timesteps();
  } else {
    steps = num_inference_steps;
    scheduler->set_timesteps(
        static_cast<int64_t>(steps), device, std::nullopt, mu, std::nullopt);
    scheduler_timesteps = scheduler->timesteps();
  }
  if (scheduler_timesteps.device() != device) {
    scheduler_timesteps = scheduler_timesteps.to(device);
  }
  return {scheduler_timesteps, steps};
}

std::pair<int64_t, int64_t> calculate_dimensions(double target_area,
                                                 double ratio) {
  double width = std::sqrt(target_area * ratio);
  double height = width / ratio;

  width = std::round(width / 32) * 32;
  height = std::round(height / 32) * 32;

  return {static_cast<int64_t>(width), static_cast<int64_t>(height)};
}

// Same as the seed-based overload, but draws from a caller-owned generator so
// consecutive calls stay deterministic as a sequence rather than per call.
inline torch::Tensor randn_tensor(
    const std::vector<int64_t>& shape,
    torch::Generator& generator,
    const torch::TensorOptions& options,
    torch::ScalarType target_type = torch::kFloat32) {
  if (shape.empty()) {
    LOG(FATAL) << "Shape must not be empty.";
  }
  return torch::randn(
             shape, generator, options.device(torch::kCPU).dtype(target_type))
      .to(options.device());
}

torch::Tensor randn_tensor(const std::vector<int64_t>& shape,
                           int64_t seed,
                           torch::TensorOptions& options,
                           torch::ScalarType target_type = torch::kBFloat16) {
  torch::Generator generator =
      torch::make_generator<torch::CPUGeneratorImpl>().clone();
  generator.set_current_seed(seed);
  return randn_tensor(shape, generator, options, target_type);
}

// Convert module parameters & buffers to BF16 on the target device,
// but leave Int8 (quantized) tensors unchanged — to(bf16) would corrupt them.
inline void to_bf16_preserve_quant(torch::nn::Module& module,
                                   torch::Device device) {
  for (auto& p : module.named_parameters()) {
    if (p.value().scalar_type() == torch::kInt8) {
      p.value().set_data(p.value().to(device, torch::kInt8));
    } else {
      p.value().set_data(p.value().to(device, torch::kBFloat16));
    }
  }
  for (auto& b : module.named_buffers()) {
    if (b.value().scalar_type() == torch::kInt8) {
      b.value().set_data(b.value().to(device, torch::kInt8));
    } else {
      b.value().set_data(b.value().to(device, torch::kBFloat16));
    }
  }
}

// Product of a VAE downsample-factor list (e.g. temporal_downsample_factors).
inline int64_t product_checked(const std::vector<int64_t>& factors) {
  int64_t product = 1;
  for (int64_t factor : factors) {
    CHECK_GT(factor, 0);
    CHECK_LE(product, std::numeric_limits<int64_t>::max() / factor)
        << "factor product overflows int64_t";
    product *= factor;
  }
  return product;
}

inline torch::Tensor normalize_latents(const torch::Tensor& latents,
                                       const torch::Tensor& mean,
                                       const torch::Tensor& stddev) {
  return ((latents.to(torch::kFloat32) - mean.to(latents.device())) /
          stddev.to(latents.device()))
      .contiguous();
}

inline torch::Tensor denormalize_latents(const torch::Tensor& latents,
                                         const torch::Tensor& mean,
                                         const torch::Tensor& stddev) {
  return (latents.to(torch::kFloat32) * stddev.to(latents.device()) +
          mean.to(latents.device()))
      .contiguous();
}

// FP32 RMS normalization over the last dimension, optionally scaled by
// `weight` (skipped when undefined). Shared by the MiniMax-H3 transformer
// qk-norm modules and its video VAE attention.
inline torch::Tensor rms_norm(const torch::Tensor& input,
                              const torch::Tensor& weight,
                              double eps) {
  torch::Tensor x = input.to(torch::kFloat32);
  torch::Tensor variance = x.pow(2).mean(-1, true);
  torch::Tensor output = x * torch::rsqrt(variance + eps);
  if (weight.defined()) {
    output = output * weight.to(output.device(), output.scalar_type());
  }
  return output.to(input.scalar_type());
}

// Applies rotary position embedding over the leading `cos.size(-1)` features
// of `hidden_states` (the trailing features pass through unchanged). `cos` and
// `sin` must already be broadcastable against `hidden_states`, e.g. [B,S,D]
// for a [B,S,D] input or [B,S,1,D] for a [B,S,H,D] input. Shared by the
// MiniMax-H3 transformer attention and its video VAE attention.
inline torch::Tensor apply_rotary(const torch::Tensor& hidden_states,
                                  const torch::Tensor& cos,
                                  const torch::Tensor& sin) {
  const int64_t rotary_dim = cos.size(-1);
  torch::Tensor rotary = hidden_states.slice(-1, 0, rotary_dim);
  torch::Tensor pass = hidden_states.slice(-1, rotary_dim);
  std::vector<torch::Tensor> chunks = rotary.chunk(2, -1);
  torch::Tensor rotated = torch::cat({-chunks[1], chunks[0]}, -1);
  rotary = rotary * cos.to(hidden_states.scalar_type()) +
           rotated * sin.to(hidden_states.scalar_type());
  return torch::cat({rotary, pass}, -1).contiguous();
}

// Slices the TP rank-local shard out of a fused SwiGLU projection weight.
//
// Checkpoints store the fused gate/up projection as two stacked halves, so a
// naive chunk(tp_size) over the fused rows hands each rank a slice that mixes
// rows from the two halves. Instead, split the halves first, take the rank-th
// chunk of each half, and stack them back in the original order so each rank
// keeps matching gate/up pairs.
inline torch::Tensor shard_swiglu_weight(const torch::Tensor& weight,
                                         int64_t tp_size,
                                         int64_t tp_rank) {
  CHECK_EQ(weight.size(0) % (2 * tp_size), 0)
      << "SwiGLU dimension must be divisible by TP size";
  const int64_t ffn_dim = weight.size(0) / 2;
  const torch::Tensor& first = weight.slice(0, 0, ffn_dim);
  const torch::Tensor& second = weight.slice(0, ffn_dim);
  return torch::cat(
      {first.chunk(tp_size, 0)[tp_rank], second.chunk(tp_size, 0)[tp_rank]}, 0);
}

// Loads/verifies every block of a ModuleList with the "prefix<i>." scheme.
template <typename Block>
void load_module_list(torch::nn::ModuleListImpl& list,
                      const StateDict& state_dict,
                      const std::string& prefix,
                      size_t index_offset = 0) {
  for (size_t i = 0; i < list.size(); ++i) {
    list[i]->as<Block>()->load_state_dict(state_dict.get_dict_with_prefix(
        prefix + std::to_string(i + index_offset) + "."));
  }
}

template <typename Block>
void verify_module_list(torch::nn::ModuleListImpl& list,
                        const std::string& prefix,
                        size_t index_offset = 0) {
  for (size_t i = 0; i < list.size(); ++i) {
    list[i]->as<Block>()->verify_loaded_weights(
        prefix + std::to_string(i + index_offset) + ".");
  }
}

}  // namespace xllm::dit
