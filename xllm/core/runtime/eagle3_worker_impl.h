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

#include "forward_params.h"
#include "runtime/mtp_worker_impl.h"

namespace xllm {

template <typename TargetInput>
class Eagle3WorkerImpl final : public MTPWorkerImpl<TargetInput> {
 public:
  Eagle3WorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options,
                   WorkerType worker_type);

  ~Eagle3WorkerImpl() override;

  // EAGLE-3 draft input_embedding is num_aux_layers * target_hidden_size
  int64_t get_embedding_placeholder_size() const override;

 protected:
  using MTPWorkerImpl<TargetInput>::context_;
  using MTPWorkerImpl<TargetInput>::draft_impl_;

  void check_draft_input_embedding(const torch::Tensor& embedding,
                                   const std::string& phase) const override;
};

}  // namespace xllm
