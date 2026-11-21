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

#include <utility>

#include "core/framework/sampling/sampling_params.h"
#include "runtime/dflash_worker_impl.h"

namespace xllm {

class DSparkWorkerImpl final : public DFlashWorkerImpl {
 public:
  using DFlashWorkerImpl::DFlashWorkerImpl;

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

 protected:
  void apply_draft_model_overrides(ModelArgs& args) override;
  DecodeExpansionOptions prepare_for_draft() const override;
  std::pair<torch::Tensor, torch::Tensor> run_decode_draft(
      const LlmForwardInput& input,
      LlmForwardInput& validate_input,
      LlmForwardInput& draft_forward_input) override;

 private:
  std::pair<torch::Tensor, torch::Tensor> sample_markov_block(
      const torch::Tensor& base_logits,
      const torch::Tensor& anchors,
      const SamplingParameters& sampling);
};

}  // namespace xllm
