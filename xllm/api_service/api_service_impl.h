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

#include <memory>

#include "call.h"
#include "chat.pb.h"
#include "completion.pb.h"
#include "core/common/macros.h"
#include "core/distributed_runtime/llm_master.h"

namespace xllm {

template <typename CallType>
class APIServiceImpl {
 public:
  using RequestType = typename CallType::ReqType;
  using ResponseType = typename CallType::ResType;
  APIServiceImpl(const std::vector<std::string>& models)
      : models_(models.begin(), models.end()) {
    CHECK(!models_.empty());
  }
  virtual ~APIServiceImpl() = default;

  void process_async(std::shared_ptr<Call> call) {
    std::shared_ptr<CallType> call_cast =
        std::dynamic_pointer_cast<CallType>(call);
    process_async_impl(call_cast);
  }

  virtual void process_async_impl(std::shared_ptr<CallType> call) = 0;

  virtual void process_async_rpc_impl(const RequestType* request) {
    NOT_IMPLEMENTED();
  }

 protected:
  absl::flat_hash_set<std::string> models_;
};

}  // namespace xllm
