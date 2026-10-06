/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "api_service_impl.h"
#include "completion.pb.h"
#include "stream_call.h"

namespace xllm {

class MasterManager;

using CompletionCall =
    StreamCall<proto::CompletionRequest, proto::CompletionResponse>;

// a class to handle completion requests
class CompletionServiceImpl final : public APIServiceImpl<CompletionCall> {
 public:
  CompletionServiceImpl(LLMMaster* master,
                        const std::vector<std::string>& models,
                        std::shared_ptr<MasterManager> master_manager);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<CompletionCall> call);

  void process_async_rpc_impl(const proto::CompletionRequest* request);

 private:
  std::shared_ptr<LLMMaster> get_model_master(const std::string& model) const;
  DISALLOW_COPY_AND_ASSIGN(CompletionServiceImpl);
  std::shared_ptr<MasterManager> master_manager_;
};

}  // namespace xllm
