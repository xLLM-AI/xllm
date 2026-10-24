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

#include <cstdint>
#include <memory>
#include <vector>

#include "core/runtime/vlm_forward_params.h"

namespace xllm {

class BatchGroup;
class ThreadPool;
struct ModelArgs;

struct VlmForwardInputFactoryOptions {
  uint32_t dp_size = 1;
  bool enable_dp_global_json_object_active = false;
};

// The owning engine serializes calls and reuses preparation threads.
class VlmForwardInputFactory final {
 public:
  explicit VlmForwardInputFactory(VlmForwardInputFactoryOptions options);
  ~VlmForwardInputFactory();

  void create_inputs(BatchGroup& batches,
                     const ModelArgs& model_args,
                     std::vector<VlmForwardInput>& inputs);

 private:
  struct PreparationState {
    std::vector<VlmForwardInput> inputs;
    std::vector<int32_t> dp_token_counts;
    std::vector<int32_t> dp_sequence_counts;
    std::vector<int32_t> dp_kv_max_seq_lens;
    std::vector<int32_t> dp_global_json_object_active;
    std::vector<int32_t> dp_is_decode;
    BatchForwardType batch_forward_type;
  };

  void prepare_rank_inputs(BatchGroup& batches,
                           const ModelArgs& model_args,
                           PreparationState& state);
  void finalize_inputs(PreparationState& state);

  VlmForwardInputFactoryOptions options_;
  std::unique_ptr<ThreadPool> threadpool_;
};

}  // namespace xllm
