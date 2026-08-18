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

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <tuple>

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

struct DeepseekV4HCPreOutput {
  torch::Tensor output;
  torch::Tensor post;
  torch::Tensor comb;
};

class DeepseekV4HCPreImpl final : public torch::nn::Module {
 public:
  DeepseekV4HCPreImpl() = default;

  DeepseekV4HCPreImpl(
      int64_t hc_mult,
      int64_t dim,
      int64_t sinkhorn_iters,
      double hc_eps,
      double norm_eps,
      const torch::TensorOptions& options =
          torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));

  DeepseekV4HCPreOutput forward(
      const torch::Tensor& x,
      const std::optional<torch::Tensor>& rsqrt = std::nullopt);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  fused_post_pre_norm(const torch::Tensor& x,
                      const torch::Tensor& residual,
                      const torch::Tensor& post,
                      const torch::Tensor& comb,
                      const torch::Tensor& gamma);

  bool supports_fused_mhc() const { return hc_mult_ == 4 && dim_ == 4096; }

  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t hc_mult_ = 0;
  int64_t dim_ = 0;
  int64_t sinkhorn_iters_ = 20;
  double hc_eps_ = 1e-6;
  double norm_eps_ = 1e-6;

  DEFINE_WEIGHT(hc_fn);
  DEFINE_WEIGHT(hc_base);
  DEFINE_WEIGHT(hc_scale);
};

class DeepseekV4HCPostImpl final : public torch::nn::Module {
 public:
  DeepseekV4HCPostImpl() = default;

  explicit DeepseekV4HCPostImpl(double norm_eps);

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& residual,
      const torch::Tensor& post,
      const torch::Tensor& comb,
      bool compute_rms = false);

 private:
  double norm_eps_ = 1e-6;
};

class DeepseekV4HCHeadImpl final : public torch::nn::Module {
 public:
  DeepseekV4HCHeadImpl() = default;

  DeepseekV4HCHeadImpl(
      int64_t hc_mult,
      int64_t dim,
      double hc_eps,
      double norm_eps,
      const torch::TensorOptions& options =
          torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));

  torch::Tensor forward(const torch::Tensor& x);

  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t hc_mult_ = 0;
  int64_t dim_ = 0;
  double hc_eps_ = 1e-6;
  double norm_eps_ = 1e-6;

  DEFINE_WEIGHT(hc_head_fn);
  DEFINE_WEIGHT(hc_head_base);
  DEFINE_WEIGHT(hc_head_scale);
};

TORCH_MODULE(DeepseekV4HCPre);
TORCH_MODULE(DeepseekV4HCPost);
TORCH_MODULE(DeepseekV4HCHead);

}  // namespace layer
}  // namespace xllm
