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
#include <absl/container/flat_hash_set.h>

#include "api_service/api_service_impl.h"
#include "api_service/call.h"
#include "api_service/non_stream_call.h"
#include "core/distributed_runtime/vlm_master.h"
#include "embedding.pb.h"

namespace xllm {

using EmbeddingCall =
    NonStreamCall<proto::EmbeddingRequest, proto::EmbeddingResponse>;

// a class to handle completion requests
class EmbeddingServiceImpl final : public APIServiceImpl<EmbeddingCall> {
 public:
  EmbeddingServiceImpl(LLMMaster* master,
                       const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<EmbeddingCall> call);

 private:
  DISALLOW_COPY_AND_ASSIGN(EmbeddingServiceImpl);
  LLMMaster* master_ = nullptr;
};

using MMEmbeddingCall =
    NonStreamCall<proto::MMEmbeddingRequest, proto::EmbeddingResponse>;
class MMEmbeddingServiceImpl : public APIServiceImpl<MMEmbeddingCall> {
 public:
  MMEmbeddingServiceImpl(VLMMaster* master,
                         const std::vector<std::string>& models);
  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<MMEmbeddingCall> call);

 private:
  DISALLOW_COPY_AND_ASSIGN(MMEmbeddingServiceImpl);
  VLMMaster* master_ = nullptr;
};

}  // namespace xllm
