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

#include "api_service_impl.h"
#include "completion.pb.h"
#include "core/distributed_runtime/rec_master.h"
#include "rec.pb.h"
#include "stream_call.h"

namespace xllm {

using CompletionCall =
    StreamCall<proto::CompletionRequest, proto::CompletionResponse>;

// a class to handle completion requests
class RecCompletionServiceImpl final : public APIServiceImpl<CompletionCall> {
 public:
  RecCompletionServiceImpl(RecMaster* master,
                           const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<CompletionCall> call);

 private:
  DISALLOW_COPY_AND_ASSIGN(RecCompletionServiceImpl);
  RecMaster* master_ = nullptr;
};

}  // namespace xllm
