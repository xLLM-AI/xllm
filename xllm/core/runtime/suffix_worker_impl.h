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

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "runtime/speculative_worker_impl.h"
#include "util/suffix_decoding_cache.h"

namespace xllm {

// Suffix-based speculative decoding worker.
// Uses a suffix tree cache to generate draft tokens from previously seen
// patterns, without requiring a separate draft model.
class SuffixWorkerImpl : public SpeculativeWorkerImpl {
 public:
  SuffixWorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options,
                   WorkerType worker_type);

  ~SuffixWorkerImpl() override = default;

 protected:
  std::optional<ForwardOutput> step_prefill(const ForwardInput& input) override;
  std::optional<ForwardOutput> step_decode(const ForwardInput& inputs) override;
  std::optional<ForwardOutput> step_empty(const ForwardInput& inputs) override;

 private:
  SampleOutput validate(const SamplingParameters& sampling_params,
                        const torch::Tensor& draft_token_ids,
                        const torch::Tensor& draft_probs,
                        const ForwardOutput& target_output);

 private:
  std::unique_ptr<SuffixDecodingCache> suffix_cache_;
  std::unordered_map<std::string, std::vector<int32_t>> suffix_recent_tokens_;
  std::unordered_set<std::string> suffix_active_decode_req_ids_;
};
}  // namespace xllm
