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

#include "forward_params.h"
#include "runtime/mtp_worker_impl.h"

namespace xllm {

class Eagle3WorkerImpl : public MTPWorkerImpl {
 public:
  Eagle3WorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options,
                   WorkerType worker_type);

  ~Eagle3WorkerImpl() override = default;

  // Override init_model to load hot_token_id_ for EAGLE-3
  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

  // EAGLE-3 draft input_embedding is 3 * target_hidden_size
  int64_t get_embedding_placeholder_size() override;

  // Get hot_token_id for draft-to-target token mapping
  torch::Tensor get_hot_token_id() const { return hot_token_id_; }

 protected:
  // EAGLE-3 specific draft output post-processing during decode:
  // selected prob extraction + draft->target token id mapping.
  void process_draft_sample_output(SampleOutput& sample_output) override;

  void check_draft_input_embedding(
      const torch::Tensor& embedding,
      const std::string& phase) const override;

  // EAGLE-3 specific: hot_token_id for draft-to-target token mapping
  // hot_token_id = d2t + arange(d2t.size(0))
  torch::Tensor hot_token_id_;
};

}  // namespace xllm
