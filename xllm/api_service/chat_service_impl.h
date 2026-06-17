/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <shared_mutex>
#include <unordered_map>

#include "api_service/api_service_impl.h"
#include "api_service/stream_call.h"
#include "chat.pb.h"
#include "multimodal.pb.h"

namespace xllm {

class RecMaster;

using ChatCall = StreamCall<proto::ChatRequest, proto::ChatResponse>;

// a class to handle completion requests
class ChatServiceImpl final : public APIServiceImpl<ChatCall> {
 public:
  // Constructor for LLM backend
  ChatServiceImpl(LLMMaster* master, const std::vector<std::string>& models);

  // Constructor for Rec backend (LlmRec only, e.g., Qwen3)
  ChatServiceImpl(RecMaster* master, const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<ChatCall> call);

  void process_async_rpc_impl(const proto::ChatRequest* request);

  void add_model_master(const std::string& model, LLMMaster* master);

 private:
  void process_rec_chat_request(std::shared_ptr<ChatCall> call);
  LLMMaster* get_model_master(const std::string& model) const;

  DISALLOW_COPY_AND_ASSIGN(ChatServiceImpl);

  LLMMaster* master_ = nullptr;
  RecMaster* rec_master_ = nullptr;
  mutable std::shared_mutex llm_model_to_master_mutex_;
  std::unordered_map<std::string, LLMMaster*> llm_model_to_master_;
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
