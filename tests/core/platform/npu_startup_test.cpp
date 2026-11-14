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
#include <pybind11/embed.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <thread>

#include "core/platform/npu/npu_runtime.h"
#include "core/platform/npu/startup.h"

namespace xllm::npu {
namespace {

TEST(NpuStartupTest, InitializesExplicitDeviceBeforeWorkerThread) {
  pybind11::scoped_interpreter interpreter(/*init_signal_handlers=*/false);
  ASSERT_TRUE(initialize_npu_startup(/*device_index=*/0));
  {
    pybind11::gil_scoped_release release;
    torch::Tensor result;
    std::thread worker([&]() {
      init_npu(/*device_index=*/0);
      auto input = torch::ones({4}, torch::TensorOptions().device("npu:0"));
      result = (input + 2).cpu();
      c10_npu::npuSynchronizeDevice();
    });
    worker.join();
    EXPECT_TRUE(torch::equal(result, torch::full({4}, 3.0F)));
  }
}

}  // namespace
}  // namespace xllm::npu
