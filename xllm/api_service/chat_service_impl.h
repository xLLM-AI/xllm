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
#include <string>

#include "api_service/api_service_impl.h"
#include "api_service/stream_call.h"
#include "chat.pb.h"
#include "multimodal.pb.h"

namespace xllm {

class MasterManager;

using ChatCall = StreamCall<proto::ChatRequest, proto::ChatResponse>;

// a class to handle completion requests
class ChatServiceImpl final : public APIServiceImpl<ChatCall> {
 public:
  // Constructor for LLM backend
  ChatServiceImpl(LLMMaster* master,
                  const std::vector<std::string>& models,
                  std::shared_ptr<MasterManager> master_manager);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<ChatCall> call) override;

  void process_async_rpc_impl(const proto::ChatRequest* request) override;

 private:
  // Service-owned requests return output through XServiceClient.
  void process_request_from_service(const proto::ChatRequest& request,
                                    const std::string& x_request_id = "",
                                    const std::string& x_request_time = "");
  std::shared_ptr<LLMMaster> get_model_master(const std::string& model) const;

  DISALLOW_COPY_AND_ASSIGN(ChatServiceImpl);

  std::shared_ptr<MasterManager> master_manager_;
  const std::string tool_call_parser_format_;
  const std::string reasoning_parser_format_;
};

class VLMMaster;
using MMChatCall = StreamCall<proto::MMChatRequest, proto::ChatResponse>;

// a class to handle mm chat completion requests
class MMChatServiceImpl : public APIServiceImpl<MMChatCall> {
 public:
  MMChatServiceImpl(VLMMaster* master, const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<MMChatCall> call);

 private:
  DISALLOW_COPY_AND_ASSIGN(MMChatServiceImpl);

  VLMMaster* master_ = nullptr;
  const std::string tool_call_parser_format_;
  const std::string reasoning_parser_format_;
};

}  // namespace xllm
