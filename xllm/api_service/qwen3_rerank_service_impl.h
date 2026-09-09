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

#include "api_service/rerank_service_impl.h"

namespace xllm {

class Qwen3RerankServiceImpl final : public RerankServiceImpl {
 public:
  Qwen3RerankServiceImpl(LLMMaster* master,
                         const std::vector<std::string>& models);

  void process_async_impl(std::shared_ptr<RerankCall> call) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(Qwen3RerankServiceImpl);
};

}  // namespace xllm
