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

#include <cstdint>
#include <memory>

#include "core/framework/batch/batch_input_data.h"
#include "core/framework/model/model_args.h"
#include "core/runtime/rec_forward_params.h"
#include "core/util/threadpool.h"

namespace xllm {

class RecForwardInputBuilder {
 public:
  virtual ~RecForwardInputBuilder() = default;

  virtual RecForwardInput build_rec_forward_input(
      uint32_t num_decoding_tokens,
      uint32_t min_decoding_batch_size) = 0;

  static std::unique_ptr<RecForwardInputBuilder> create(
      BatchInputType input_type,
      const BatchInputData& data,
      const ModelArgs* args,
      MPMCThreadPool* thread_pool = nullptr);
};

}  // namespace xllm
