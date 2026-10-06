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

#include "api_service/openai_responses_call.h"
#include "api_service/openai_responses_request.h"
#include "core/distributed_runtime/master_manager.h"

namespace xllm {

class ResponsesServiceImpl final {
 public:
  explicit ResponsesServiceImpl(std::shared_ptr<MasterManager> master_manager);

  void process_async(std::shared_ptr<ResponsesCall> call,
                     api_service::ResponsesRequest request);

 private:
  std::shared_ptr<MasterManager> master_manager_;
};

}  // namespace xllm
