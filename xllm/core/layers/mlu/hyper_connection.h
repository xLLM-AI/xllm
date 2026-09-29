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

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <tuple>

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

struct PendingMHC final {
  torch::Tensor x;
  torch::Tensor residual;
  torch::Tensor post;
  torch::Tensor comb;
};

struct MHCFusionContext final {
  bool optimization_enabled = true;
  bool is_prefill = false;
  bool is_chunked_prefill = false;
  bool supports_fused_mhc = false;
  bool has_pending_storage = false;
  bool has_pending = false;
  bool is_last_layer = true;
};

struct MHCFusionPlan final {
  bool use_fused_mhc = false;
  bool consume_pending = false;
  bool defer_post = false;
};

MHCFusionPlan resolve_mhc_fusion(const MHCFusionContext& context);

struct MHCPreOutput final {
  torch::Tensor output;
  torch::Tensor post;
  torch::Tensor comb;
};

class MHCPreImpl final : public torch::nn::Module {
 public:
  MHCPreImpl() = default;

  MHCPreImpl(int64_t hc_mult,
             int64_t dim,
             int64_t sinkhorn_iters,
             double hc_eps,
             double norm_eps,
             const torch::TensorOptions& options = torch::TensorOptions()
                                                       .dtype(torch::kBFloat16)
                                                       .device(torch::kCPU));

  MHCPreOutput forward(
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
  void verify_loaded_weights(const std::string& prefix) const;

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

class MHCPostImpl final : public torch::nn::Module {
 public:
  MHCPostImpl() = default;

  explicit MHCPostImpl(double norm_eps);

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& residual,
      const torch::Tensor& post,
      const torch::Tensor& comb,
      bool compute_rms = false);

 private:
  double norm_eps_ = 1e-6;
};

class MHCHeadImpl final : public torch::nn::Module {
 public:
  MHCHeadImpl() = default;

  MHCHeadImpl(int64_t hc_mult,
              int64_t dim,
              double hc_eps,
              double norm_eps,
              const torch::TensorOptions& options = torch::TensorOptions()
                                                        .dtype(torch::kBFloat16)
                                                        .device(torch::kCPU));

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

TORCH_MODULE(MHCPre);
TORCH_MODULE(MHCPost);
TORCH_MODULE(MHCHead);

}  // namespace layer
}  // namespace xllm
