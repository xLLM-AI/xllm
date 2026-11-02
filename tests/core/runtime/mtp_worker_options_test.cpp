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

#include <gtest/gtest.h>

#include <initializer_list>

#include "core/framework/parallel_state/parallel_args.h"
#include "core/platform/platform.h"
#include "core/runtime/llm_worker_impl.h"
#include "core/runtime/mtp_worker_impl.h"
#include "core/runtime/options.h"

namespace xllm {
namespace {

class VlmMtpOptionsWorker final : public MTPWorkerImpl<VlmForwardInput> {
 public:
  VlmMtpOptionsWorker(const torch::Device& device,
                      const runtime::Options& options)
      : MTPWorkerImpl<VlmForwardInput>(ParallelArgs(/*rank=*/0,
                                                    /*world_size=*/1,
                                                    /*dp_size=*/1,
                                                    /*process_group=*/nullptr),
                                       device,
                                       options,
                                       WorkerType::VLM) {}

  const runtime::Options& target_options() const {
    return impl_.get()->*&VlmMtpOptionsWorker::options_;
  }

  const runtime::Options& draft_options() const {
    return draft_impl_.get()->*&VlmMtpOptionsWorker::options_;
  }
};

TEST(MtpWorkerOptionsTest, VlmTargetKeepsVisionWhileDraftUsesText) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator is required for MTP worker construction.";
  }
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  for (const bool task_pipeline : {false, true}) {
    SCOPED_TRACE(task_pipeline);
    runtime::Options options;
    options.backend("vlm")
        .enable_graph(true)
        .enable_task_pipeline(task_pipeline)
        .num_speculative_tokens(4);
    VlmMtpOptionsWorker worker(device, options);

    EXPECT_EQ(worker.target_options().backend(), "vlm");
    EXPECT_EQ(worker.draft_options().backend(), "llm");
    EXPECT_TRUE(worker.target_options().enable_graph());
    EXPECT_TRUE(worker.draft_options().enable_graph());
  }
}

}  // namespace
}  // namespace xllm
