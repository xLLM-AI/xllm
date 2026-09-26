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

#include "core/distributed_runtime/worker_service.h"

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <unistd.h>

#include <memory>
#include <string>

#include "core/runtime/options.h"

namespace xllm {

namespace {

TEST(WorkerServiceShutdownTest, JoinsIdleShmPollingThreadForEveryBackend) {
  for (const std::string& backend : {"llm", "vlm", "rec", "dit"}) {
    SCOPED_TRACE(backend);
    runtime::Options options;
    options.backend(backend);
    auto service =
        std::make_unique<WorkerService>(options, torch::Device("npu:0"));
    const std::string prefix =
        "worker_shutdown_" + std::to_string(getpid()) + "_" + backend;
    bool is_creator = false;
    auto input =
        std::make_unique<ForwardSharedMemoryManager>(prefix + "_input",
                                                     /*size=*/1 << 20,
                                                     is_creator,
                                                     ForwardType::RAW_INPUT);
    auto output =
        std::make_unique<ForwardSharedMemoryManager>(prefix + "_output",
                                                     /*size=*/1 << 20,
                                                     is_creator,
                                                     ForwardType::RAW_OUTPUT);
    service->create_polling_shm_thread(std::move(input), std::move(output));
    // No input is ever published: destruction must cancel the read and join,
    // including when shutdown races with the polling thread's first read.
    service.reset();
  }
}

TEST(WorkerServiceShutdownTest, StopsWithoutShmPollingThread) {
  runtime::Options options;
  WorkerService service(options, torch::Device("npu:0"));
}

}  // namespace
}  // namespace xllm
