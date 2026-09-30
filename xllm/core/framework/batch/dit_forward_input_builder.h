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

#include <memory>
#include <vector>

#include "core/framework/request/dit_request.h"
#include "core/runtime/dit_forward_params.h"

namespace xllm {

class DiTForwardInputBuilder final {
 public:
  // The request container is owned by DiTBatch and outlives this builder.
  explicit DiTForwardInputBuilder(
      const std::vector<std::shared_ptr<DiTRequest>>& requests)
      : requests_(requests) {}

  DiTForwardInput build_forward_input() const;

 private:
  const std::vector<std::shared_ptr<DiTRequest>>& requests_;
};

}  // namespace xllm
