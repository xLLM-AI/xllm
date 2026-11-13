/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/speculative/draft_proposal.h"
#include "runtime/dflash_worker_impl.h"

namespace xllm {

class DFlash2WorkerImpl final : public DFlashWorkerImpl {
 public:
  DFlash2WorkerImpl(const ParallelArgs& parallel_args,
                    const torch::Device& device,
                    const runtime::Options& options);

  ~DFlash2WorkerImpl() override = default;

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

 protected:
  DraftProposal run_decode_draft(const LlmForwardInput& input,
                                 LlmForwardInput& validate_input) override;
};

}  // namespace xllm
