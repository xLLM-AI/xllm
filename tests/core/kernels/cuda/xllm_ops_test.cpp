/* Copyright 2025-2026 The xLLM Authors.

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

// M0 acceptance test for the xllm_ops torch-op library.
//
// Verifies two things end to end:
//   1. The TORCH_LIBRARY(xllm_ops) registrations survive linking (via the
//      ensure_xllm_ops_registered anchor) and are callable through the torch
//      dispatcher, producing results that match a plain-torch reference.
//   2. An *embedded* CPython interpreter running in this same process (sharing
//      this binary's libtorch) can see and call torch.ops.xllm_ops.* — this is
//      exactly the path PyCausalLM uses at runtime.

#include <ATen/core/dispatch/Dispatcher.h>
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <torch/cuda.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "xllm_ops_library.h"

namespace py = pybind11;

namespace xllm {
namespace {

torch::Tensor rms_norm_reference(const torch::Tensor& input,
                                 const torch::Tensor& weight,
                                 double eps) {
  auto x = input.to(torch::kFloat32);
  auto var = x.pow(2).mean(-1, /*keepdim=*/true);
  auto normed = x * torch::rsqrt(var + eps);
  return (normed * weight.to(torch::kFloat32)).to(input.scalar_type());
}

torch::Tensor silu_and_mul_reference(const torch::Tensor& input) {
  const int64_t d = input.size(-1) / 2;
  auto a = input.slice(-1, 0, d);
  auto b = input.slice(-1, d, 2 * d);
  return (a * torch::sigmoid(a)) * b;
}

class XllmOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Force-link the xllm_ops registration TU.
    xllm::ensure_xllm_ops_registered();
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available; skipping xllm_ops CUDA test.";
    }
  }
};

// (1) Call through the torch dispatcher from C++.
TEST_F(XllmOpsTest, DispatcherRmsNormMatchesReference) {
  auto opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
  auto input = torch::randn({8, 128}, opts);
  auto weight = torch::randn({128}, opts);
  const double eps = 1e-6;

  auto op =
      c10::Dispatcher::singleton().findSchemaOrThrow("xllm_ops::rms_norm", "");
  auto out = op.typed<torch::Tensor(const torch::Tensor&,
                                    const torch::Tensor&,
                                    double)>()
                 .call(input, weight, eps);

  auto ref = rms_norm_reference(input, weight, eps);
  EXPECT_TRUE(torch::allclose(out, ref, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max abs diff = "
      << (out.to(torch::kFloat32) - ref.to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

// (2) Call through an embedded interpreter's torch.ops.xllm_ops.* — proves the
// Python-side graph will see the ops with no extra registration.
TEST_F(XllmOpsTest, EmbeddedInterpreterSeesOps) {
  if (!Py_IsInitialized()) {
    Py_InitializeEx(0);
  }
  py::gil_scoped_acquire gil;

  auto opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
  auto gate_up = torch::randn({8, 256}, opts);

  py::module_ torch_mod = py::module_::import("torch");
  py::object xllm_ops = torch_mod.attr("ops").attr("xllm_ops");
  py::object out_obj = xllm_ops.attr("silu_and_mul")(gate_up);
  auto out = out_obj.cast<torch::Tensor>();

  auto ref = silu_and_mul_reference(gate_up);
  ASSERT_EQ(out.size(-1), 128);
  EXPECT_TRUE(torch::allclose(out, ref, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max abs diff = "
      << (out.to(torch::kFloat32) - ref.to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

}  // namespace
}  // namespace xllm
