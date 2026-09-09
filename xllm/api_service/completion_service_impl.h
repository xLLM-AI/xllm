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

#include <shared_mutex>
#include <unordered_map>

#include "api_service_impl.h"
#include "completion.pb.h"
#include "stream_call.h"

namespace xllm {

using CompletionCall =
    StreamCall<proto::CompletionRequest, proto::CompletionResponse>;

// a class to handle completion requests
class CompletionServiceImpl final : public APIServiceImpl<CompletionCall> {
 public:
  CompletionServiceImpl(LLMMaster* master,
                        const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<CompletionCall> call);

  void process_async_rpc_impl(const proto::CompletionRequest* request);

  void add_model_master(const std::string& model, LLMMaster* master);

 private:
  LLMMaster* get_model_master(const std::string& model) const;
  DISALLOW_COPY_AND_ASSIGN(CompletionServiceImpl);
  LLMMaster* master_ = nullptr;
  mutable std::shared_mutex llm_model_to_master_mutex_;
  std::unordered_map<std::string, LLMMaster*> llm_model_to_master_;
};

}  // namespace xllm
