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

#include <folly/Try.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/runtime/forward_params.h"

namespace xllm {

class EplbOptions;
struct ModelArgs;

// Owns the EPLB lifecycle at the engine boundary. The controller keeps
// topology, input annotation, worker-result decoding, and overlap bookkeeping
// out of distributed_runtime so LLMEngine only deals with batch-level facts.
class EplbController final {
 public:
  // Returns nullptr when EPLB is disabled in the process configuration.
  static std::unique_ptr<EplbController> create(const ModelArgs& model_args,
                                                int32_t worker_num,
                                                int32_t ep_size);

  ~EplbController();

  EplbController(const EplbController&) = delete;
  EplbController& operator=(const EplbController&) = delete;

  // EPLB requires every worker result because each worker owns a different
  // physical expert shard.
  bool requires_all_worker_results() const { return true; }

  // Adds the current EPLB command and one global decode mask to every DP
  // input. The caller supplies only batch facts; all EPLB representation
  // details stay inside the controller.
  void annotate_inputs(std::vector<LlmForwardInput>& inputs,
                       const std::vector<int32_t>& dp_token_counts,
                       bool allow_eplb_command);

  // Records the activation token carried by a dispatched step. The token is
  // consumed when the corresponding worker results are submitted.
  void on_step_dispatched(const std::vector<LlmForwardInput>& inputs,
                          bool is_graph_warmup);

  // Validates and submits the worker EPLB samples to the manager.
  void on_step_completed(
      const std::vector<folly::Try<std::optional<RawForwardOutput>>>& results,
      bool is_graph_warmup);

 private:
  EplbController(int32_t num_layers,
                 int32_t num_experts,
                 int32_t worker_num,
                 int32_t device_num,
                 EplbOptions options);

  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xllm
