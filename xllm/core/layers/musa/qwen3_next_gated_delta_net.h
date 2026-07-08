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

#include <string>
#include <utility>

#include "layers/musa/qwen3_gated_delta_net_base.h"

namespace xllm {
namespace layer {

class Qwen3NextGatedDeltaNetImpl : public Qwen3GatedDeltaNetBaseImpl {
 public:
  Qwen3NextGatedDeltaNetImpl() = default;
  Qwen3NextGatedDeltaNetImpl(const ModelArgs& args,
                             const QuantArgs& quant_args,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights(const std::string& prefix) const override;

 protected:
  Qwen3NextGatedDeltaNetImpl(const ModelArgs& args,
                             const QuantArgs& quant_args,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options,
                             bool init_projections);

  std::pair<torch::Tensor, torch::Tensor> project_decode_inputs(
      const torch::Tensor& hidden_states) override;
  std::pair<torch::Tensor, torch::Tensor> project_flat_inputs(
      const torch::Tensor& hidden_states) override;

  virtual void load_projection_state_dict(const StateDict& state_dict);
  virtual void verify_projection_weights(const std::string& prefix) const;

  void init_next_projections(const ModelArgs& args,
                             const QuantArgs& quant_args,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options);

 private:
  ColumnParallelLinear qkvz_proj_{nullptr};
  ColumnParallelLinear ba_proj_{nullptr};
};
TORCH_MODULE(Qwen3NextGatedDeltaNet);

class Qwen3_5GatedDeltaNetImpl final : public Qwen3NextGatedDeltaNetImpl {
 public:
  Qwen3_5GatedDeltaNetImpl() = default;
  Qwen3_5GatedDeltaNetImpl(const ModelArgs& args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options);

 protected:
  std::pair<torch::Tensor, torch::Tensor> project_decode_inputs(
      const torch::Tensor& hidden_states) override;
  std::pair<torch::Tensor, torch::Tensor> project_flat_inputs(
      const torch::Tensor& hidden_states) override;
  bool use_fla_ssm_state_layout() const override { return true; }

  void load_projection_state_dict(const StateDict& state_dict) override;
  void verify_projection_weights(const std::string& prefix) const override;

 private:
  torch::Tensor merge_qkvz_from_split_activations(const torch::Tensor& qkv,
                                                  const torch::Tensor& z) const;
  torch::Tensor merge_ba_from_split_activations(const torch::Tensor& b,
                                                const torch::Tensor& a) const;

  ColumnParallelLinear in_proj_qkv_{nullptr};
  ColumnParallelLinear in_proj_z_{nullptr};
  ColumnParallelLinear in_proj_b_{nullptr};
  ColumnParallelLinear in_proj_a_{nullptr};

#if defined(USE_CUDA) || defined(USE_MUSA)
  // Persistent buffers that replace the two `torch::cat` calls in
  // merge_qkvz_from_split_activations / merge_ba_from_split_activations.
  // `torch::cat` allocates a fresh output via at::empty -> EmptyMUSA which
  // is forbidden during MUSA graph capture; the equivalent of the cat is a
  // sequence of strided `copy_` writes into pre-allocated rows, so we keep
  // one persistent buffer per merge call and reuse it across replays.
  //
  // Sized lazily on the first (warmup) forward and grow-only thereafter,
  // matching the contract used by ColumnParallelLinearImpl::output_buf_ and
  // friends. Same pattern is used by sglang's MergedColumnParallelLinear
  // emitting the qkvz/ba tensors directly; here we keep the existing four
  // split projections (matching the on-disk checkpoint layout) and just
  // make the final concat capture-safe.
  mutable torch::Tensor qkvz_merge_buf_;
  mutable torch::Tensor ba_merge_buf_;
#endif
};
TORCH_MODULE(Qwen3_5GatedDeltaNet);

}  // namespace layer
}  // namespace xllm
