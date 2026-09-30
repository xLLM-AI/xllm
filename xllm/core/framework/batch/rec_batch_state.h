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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/framework/batch/batch_state.h"

namespace xllm {

// Rec-specific input contract and sequence view around shared batch state.
class RecBatchState final {
 public:
  explicit RecBatchState(BatchInputType input_type);

  BatchInputType input_type() const { return input_type_; }
  BatchState& sequence_state() { return sequence_state_; }
  const BatchState& sequence_state() const { return sequence_state_; }

  bool uses_group_input() const;
  size_t size() const;
  Sequence* sequence(size_t index) const;
  std::vector<Sequence*> get_sequences() const;
  void refresh_sequences_from_groups();

  ForwardInput prepare_forward_input(uint32_t num_decoding_tokens,
                                     uint32_t min_decoding_batch_size,
                                     const ModelArgs& args,
                                     int32_t cp_size);
  ForwardInput prepare_forward_input(const ModelArgs& args,
                                     ThreadPool* thread_pool,
                                     int32_t cp_size);
  ForwardInput prepare_rec_forward_input(uint32_t num_decoding_tokens,
                                         uint32_t min_decoding_batch_size,
                                         const ModelArgs& args,
                                         MPMCThreadPool* thread_pool);

 private:
  BatchState sequence_state_;
  BatchInputType input_type_;
};

}  // namespace xllm
