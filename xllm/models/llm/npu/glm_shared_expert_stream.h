/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <glog/logging.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string_view>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "core/framework/model_context.h"
#include "core/platform/device.h"

namespace xllm::npu::model {

class GlmSharedExpertStreamOwner {
 public:
  explicit GlmSharedExpertStreamOwner(const ModelContext& context) {
    const ParallelArgs& parallel_args = context.get_parallel_args();
    const ModelArgs& model_args = context.get_model_args();
    const int32_t expert_parallel_degree =
        EPLBConfig::get_instance().expert_parallel_degree();
    if (parallel_args.ep_size() <= 1 || expert_parallel_degree != 2 ||
        model_args.n_shared_experts() <= 0) {
      return;
    }

    constexpr const char* kAllocatorEnv = "ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE";
    const char* allocator = std::getenv(kAllocatorEnv);
    if (allocator == nullptr || std::string_view(allocator) != "3") {
      LOG(WARNING) << "GLM EPLv2 is enabled, but " << kAllocatorEnv << " is "
                   << (allocator == nullptr ? "not set" : allocator)
                   << ". Set 'export " << kAllocatorEnv
                   << "=3' before starting xLLM; otherwise shared expert "
                      "overlap may produce incorrect results.";
    }

    Device npu_device(context.get_tensor_options().device());
    shared_expert_stream_ = npu_device.get_stream_from_pool();
    atb::Context* atb_context =
        const_cast<atb::Context*>(context.get_atb_context());
    CHECK(atb_context != nullptr)
        << "ATB context is null while registering GLM shared expert stream";
    const std::vector<aclrtStream> execute_streams =
        atb_context->GetExecuteStreams();
    CHECK_LE(execute_streams.size(), 1)
        << "GLM shared expert overlap requires an unused execute stream 1";
    const aclrtStream main_stream = execute_streams.empty()
                                        ? atb_context->GetExecuteStream()
                                        : execute_streams.at(0);
    const aclrtStream shared_stream =
        shared_expert_stream_->get_stream()->stream();
    CHECK(main_stream != nullptr);
    CHECK(shared_stream != nullptr);
    CHECK_NE(main_stream, shared_stream);
    CHECK_EQ(atb_context->SetExecuteStreams({main_stream, shared_stream}),
             atb::NO_ERROR)
        << "Failed to register GLM shared expert execute stream";
  }

 private:
  std::unique_ptr<Stream> shared_expert_stream_;
};

}  // namespace xllm::npu::model
