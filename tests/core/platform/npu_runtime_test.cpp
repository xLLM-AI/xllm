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

#include "core/platform/npu/npu_runtime.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <array>
#include <cstdint>
#include <string>
#include <thread>

#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::npu {
namespace {

TEST(NpuRuntimeTest, RepeatedInitializationPreservesTensors) {
  init_npu(/*device_index=*/0);
  const auto options =
      torch::TensorOptions().device("npu:0").dtype(torch::kFloat32);
  auto tensor = torch::arange(16, options);
  init_npu(/*device_index=*/0);
  c10_npu::npuSynchronizeDevice();
  EXPECT_EQ(c10_npu::current_device(), 0);
  EXPECT_TRUE(torch::equal(tensor.cpu(), torch::arange(16, torch::kFloat32)));
}

TEST(NpuRuntimeTest, InitializesWorkerThreadDevice) {
  torch::Tensor result;
  int32_t selected_device = -1;
  std::thread worker([&]() {
    init_npu(/*device_index=*/0);
    selected_device = c10_npu::current_device();
    const auto options =
        torch::TensorOptions().device("npu:0").dtype(torch::kFloat32);
    auto input = torch::ones({4, 4}, options);
    result = torch::matmul(input, input).cpu();
    c10_npu::npuSynchronizeDevice();
  });
  worker.join();
  EXPECT_EQ(selected_device, 0);
  EXPECT_TRUE(torch::equal(result, torch::full({4, 4}, 4.0F)));
}

class InternalFormatGuard final {
 public:
  InternalFormatGuard() {
    pybind11::gil_scoped_acquire gil;
    auto module = pybind11::module_::import("torch_npu");
    auto previous =
        module.attr("_C").attr("_npu_getOption")("ALLOW_INTERNAL_FORMAT");
    previous_ = !previous.is_none() && previous.cast<std::string>() == "enable";
    auto config = module.attr("npu").attr("config");
    config.attr("allow_internal_format") = true;
  }

  ~InternalFormatGuard() {
    pybind11::gil_scoped_acquire gil;
    pybind11::module_::import("torch_npu")
        .attr("npu")
        .attr("config")
        .attr("allow_internal_format") = previous_;
  }

 private:
  bool previous_ = false;
};

TEST(NpuRuntimeTest, AllocatesNdAndNzIntegerTensors) {
  InternalFormatGuard format_guard;
  init_npu(/*device_index=*/0);
  const auto options =
      torch::TensorOptions().device("npu:0").dtype(torch::kInt32);
  const auto expected = torch::full({32, 8}, 7, torch::kInt32);

  for (const int64_t format : std::array<int64_t, 2>{2, 29}) {
    auto tensor = at_npu::native::empty_with_format(
        {32, 8}, options, format, /*keep_format=*/true);
    EXPECT_EQ(tensor.scalar_type(), torch::kInt32);
    EXPECT_EQ(at_npu::native::get_npu_format(tensor), format);
    tensor.fill_(7);
    EXPECT_TRUE(torch::equal(tensor.cpu(), expected));
  }
}

TEST(NpuRuntimeTest, QuantizesNdTensorToPackedInt4) {
  init_npu(/*device_index=*/0);
  const auto options =
      torch::TensorOptions().device("npu:0").dtype(torch::kFloat16);
  auto input = torch::ones({32, 64}, options);
  auto scales = torch::ones({1}, options);
  auto quantized = kernel::npu::quantize_per_tensor(
      input, scales, torch::Tensor(), torch::ScalarType::QUInt4x2, /*axis=*/-1);
  EXPECT_EQ(quantized.scalar_type(), torch::kInt32);
  const auto expected = torch::full({32, 8}, 0x11111111, torch::kInt32);
  EXPECT_TRUE(torch::equal(quantized.cpu(), expected));
}

}  // namespace
}  // namespace xllm::npu
