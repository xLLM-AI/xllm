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
#include <mutex>
#include <vector>

#include "core/runtime/rec_forward_params.h"

namespace xllm {

class MPMCThreadPool;
class RecBatch;
class RecBatchGroup;
class ThreadPool;
struct ModelArgs;

struct RecForwardInputFactoryOptions {
  uint32_t dp_size = 1;
  uint32_t num_decoding_tokens = 1;
  uint32_t min_decoding_batch_size = 0;
};

// The owning engine serializes calls and reuses the selected builder's pool.
class RecForwardInputFactory final {
 public:
  explicit RecForwardInputFactory(RecForwardInputFactoryOptions options);
  ~RecForwardInputFactory();

  void create_inputs(RecBatchGroup& batches,
                     const ModelArgs& model_args,
                     std::vector<RecForwardInput>& inputs);
  RecForwardInput create_input(RecBatch& batch, const ModelArgs& model_args);

 private:
  struct PreparationState {
    std::vector<RecForwardInput> inputs;
    std::vector<int32_t> dp_token_counts;
    std::vector<int32_t> dp_sequence_counts;
    std::vector<int32_t> dp_is_decode;
    BatchForwardType batch_forward_type;
  };

  void prepare_rank_inputs(RecBatchGroup& batches,
                           const ModelArgs& model_args,
                           PreparationState& state);
  void finalize_inputs(PreparationState& state);

  RecForwardInputFactoryOptions options_;
  std::once_flag threadpool_once_;
  std::unique_ptr<ThreadPool> threadpool_;
  std::once_flag input_builder_threadpool_once_;
  std::unique_ptr<MPMCThreadPool> input_builder_threadpool_;
};

}  // namespace xllm
