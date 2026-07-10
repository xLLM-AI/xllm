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

#include <filesystem>

#include "core/kernels/cuda/cuda_ops_library.h"

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

void prepend_python_model_path() {
  std::filesystem::path repo_root(__FILE__);
  for (int i = 0; i < 5; ++i) {
    repo_root = repo_root.parent_path();
  }
  const std::string python_model_path = (repo_root / "xllm").string();
  py::list sys_path = py::module_::import("sys").attr("path");
  sys_path.attr("insert")(0, python_model_path);
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
  auto out = op.typed<torch::Tensor(
      const torch::Tensor&, const torch::Tensor&, double)>()
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

TEST_F(XllmOpsTest, EmbeddedPythonCollectivesUseTorchDistributed) {
  if (!Py_IsInitialized()) {
    Py_InitializeEx(0);
  }
  py::gil_scoped_acquire gil;
  prepend_python_model_path();

  py::module_ collectives = py::module_::import("python.ops.collectives");
  py::object group =
      collectives.attr("init_tp_group")("127.0.0.1", 0, 0, 1, "cuda:0");

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto input = torch::ones({2, 3}, options);

  auto reduced = collectives.attr("all_reduce")(input).cast<torch::Tensor>();
  EXPECT_TRUE(torch::equal(reduced, input));
  EXPECT_TRUE(torch::equal(input, torch::ones_like(input)));

  auto gathered =
      collectives.attr("all_gather")(input, -1, 1).cast<torch::Tensor>();
  group.attr("shutdown")();

  ASSERT_EQ(gathered.sizes(), torch::IntArrayRef({2, 3}));
  EXPECT_TRUE(torch::equal(gathered, input));
}

TEST_F(XllmOpsTest, DecodeGraphPaddingReservesKvIndexCapacity) {
  if (!Py_IsInitialized()) {
    Py_InitializeEx(0);
  }
  py::gil_scoped_acquire gil;
  prepend_python_model_path();

  py::exec(R"PY(
import torch

from python.attn_metadata import AttentionMetadata
from python.model_runner.graph_runner import DecodeFullGraphRunner

runner = DecodeFullGraphRunner(torch.nn.Identity(), max_batch=4)
input_ids = torch.zeros(1, dtype=torch.int64)
positions = torch.zeros(1, dtype=torch.int64)
meta = AttentionMetadata({
    "slot_mapping": torch.zeros(1, dtype=torch.int32),
    "paged_kv_indptr": torch.tensor([0, 4], dtype=torch.int32),
    "paged_kv_indices": torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    "paged_kv_last_page_len": torch.ones(1, dtype=torch.int32),
    "is_prefill": False,
    "is_chunked_prefill": False,
    "enable_cuda_graph": False,
    "use_tensor_core": False,
})
kv_cache = torch.empty((4, 1, 1, 1))
entry = runner._alloc_entry(
    4, input_ids, positions, meta, [(kv_cache, kv_cache.clone())]
)
runner._fill_buffers(entry, input_ids, positions, meta, batch=1)

assert entry.static_paged_kv_indices.numel() == 8
assert entry.static_paged_kv_indptr.tolist() == [0, 4, 5, 6, 7]
assert entry.static_paged_kv_indices[:4].tolist() == [0, 1, 2, 3]
assert entry.static_paged_kv_indices[4:7].tolist() == [0, 0, 0]
)PY");
}

}  // namespace
}  // namespace xllm
