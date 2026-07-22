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
  // sys.path must contain the directory holding the 'xllm' package so the
  // 'xllm.python' subpackage resolves (same contract as --python_model_path).
  const std::string python_model_path = repo_root.string();
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

  py::module_ collectives = py::module_::import("xllm.python.ops.collectives");
  py::object group =
      collectives.attr("init_tp_group")("127.0.0.1", 0, 0, 1, "cuda:0");

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto input = torch::ones({2, 3}, options);

  py::object all_reduce = collectives.attr("all_reduce_");
  const std::string all_reduce_schema =
      py::str(all_reduce.attr("_opoverload").attr("_schema"))
          .cast<std::string>();
  EXPECT_NE(all_reduce_schema.find("Tensor(a0!) x"), std::string::npos);
  EXPECT_TRUE(all_reduce(input).is_none());
  EXPECT_TRUE(torch::equal(input, torch::ones_like(input)));

  auto gathered =
      collectives.attr("all_gather")(input, -1, 1).cast<torch::Tensor>();
  group.attr("shutdown")();

  ASSERT_EQ(gathered.sizes(), torch::IntArrayRef({2, 3}));
  EXPECT_TRUE(torch::equal(gathered, input));
}

TEST_F(XllmOpsTest, DecodeGraphSharesMaximumBlockTableBufferAcrossBuckets) {
  if (!Py_IsInitialized()) {
    Py_InitializeEx(0);
  }
  py::gil_scoped_acquire gil;
  prepend_python_model_path();

  py::exec(R"PY(
from types import SimpleNamespace

import torch

from xllm.python.model_executor.runners.decode_cuda_graph import (
    DecodeCudaGraphRunner,
)

device = torch.device("cuda")
backend = SimpleNamespace(page_size=4, num_kv_blocks=6)
runner = DecodeCudaGraphRunner(
    torch.nn.Identity(), backend, device, max_batch=4, max_model_len=24
)
metadata = SimpleNamespace(
    slot_mapping=torch.zeros(2, dtype=torch.int32, device=device),
    kv_cu_seq_lens=torch.tensor([0, 24, 48], dtype=torch.int32, device=device),
    kv_seq_lens_host=torch.tensor([0, 24, 48], dtype=torch.int32),
    paged_kv_indptr=torch.tensor([0, 6, 12], dtype=torch.int32, device=device),
    paged_kv_indices=torch.tensor(
        list(range(6)) * 2, dtype=torch.int32, device=device
    ),
    paged_kv_last_page_len=torch.ones(2, dtype=torch.int32, device=device),
)
input_ids = torch.zeros(2, dtype=torch.int32, device=device)
positions = torch.zeros(2, dtype=torch.int64, device=device)

small_entry = runner._allocate_entry(2, input_ids, positions, metadata)
runner._fill_entry(small_entry, input_ids, positions, metadata, batch_size=2)
assert small_entry.static_positions.dtype == torch.int32
large_entry = runner._allocate_entry(4, input_ids, positions, metadata)

# Four sequences with at most six model blocks each.
expected_capacity = 4 * 6
assert small_entry.static_metadata.paged_kv_indices.numel() == expected_capacity
torch.testing.assert_close(
    small_entry.static_metadata.paged_kv_indices[:12], metadata.paged_kv_indices
)
assert (
    small_entry.static_metadata.paged_kv_indices.data_ptr()
    == large_entry.static_metadata.paged_kv_indices.data_ptr()
)
)PY");
}

TEST_F(XllmOpsTest, ModelExecutorUsesExplicitRuntimeBatchLimit) {
  if (!Py_IsInitialized()) {
    Py_InitializeEx(0);
  }
  py::gil_scoped_acquire gil;
  prepend_python_model_path();

  py::exec(R"PY(
import torch
from unittest.mock import patch

from xllm.python.layers.attention import Attention
from xllm.python.model_executor import executor as executor_module


class FakeBackend:
    def __init__(self, **kwargs):
        pass

    def bind_kv_caches(self, kv_caches):
        pass

    def prepare(self, metadata, *, graph_mode=False):
        pass

    def execute(self, q, k, v, layer):
        return q

    @property
    def num_kv_blocks(self):
        return 0

    @property
    def page_size(self):
        return 1


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, device="cuda"))
        self.attention = Attention(1, 1, 8, 1.0, 0, 0)
        self.model = torch.nn.Identity()


with patch.object(executor_module, "_create_attention_backend", return_value=FakeBackend()):
    model_executor = executor_module.ModelExecutor(
        FakeModel(),
        {
            "python_graph_backend": "cudagraphs",
            "max_position_embeddings": 64,
            "max_seqs_per_batch": 1024,
        },
        max_seqs_per_batch=3,
    )
    assert model_executor.decode_cuda_graph_runner.max_batch == 3
)PY");
}

TEST_F(XllmOpsTest, DecodeGraphMetadataKernelMatchesReferenceAcrossShapes) {
  if (!Py_IsInitialized()) {
    Py_InitializeEx(0);
  }
  py::gil_scoped_acquire gil;
  prepend_python_model_path();

  py::exec(R"PY(
import torch

device = "cuda"
shapes = [
    (1, 1), (2, 2), (3, 4), (5, 8), (7, 8), (8, 8),
    (13, 16), (16, 16), (31, 32), (47, 48), (100, 112),
]

for actual_batch, padded_batch in shapes:
    page_counts = torch.arange(1, actual_batch + 1, dtype=torch.int32) % 7 + 1
    if actual_batch == 31:
        page_counts[0] = 5000  # Exercise the kernel's grid-stride loop.
    indptr_cpu = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(page_counts, dim=0, dtype=torch.int32),
        ]
    )
    num_indices = int(indptr_cpu[-1])
    padding = padded_batch - actual_batch

    src = {
        "tokens": torch.arange(
            11, 11 + actual_batch, dtype=torch.int32, device=device
        ),
        "positions": torch.arange(
            1000,
            1000 + actual_batch,
            dtype=torch.int32,
            device=device,
        ),
        "slots": torch.arange(
            101, 101 + actual_batch, dtype=torch.int32, device=device
        ),
        "kv_lens": indptr_cpu.to(device),
        "indptr": indptr_cpu.to(device),
        "indices": torch.arange(
            10000, 10000 + num_indices, dtype=torch.int32, device=device
        ),
        "last_page": torch.ones(
            actual_batch, dtype=torch.int32, device=device
        ),
    }
    dst = {
        "tokens": torch.empty(padded_batch, dtype=torch.int32, device=device),
        "positions": torch.empty(
            padded_batch, dtype=torch.int32, device=device
        ),
        "slots": torch.empty(padded_batch, dtype=torch.int32, device=device),
        "kv_lens": torch.empty(
            padded_batch + 1, dtype=torch.int32, device=device
        ),
        "kv_lens_delta": torch.empty(
            padded_batch, dtype=torch.int32, device=device
        ),
        "indptr": torch.empty(
            padded_batch + 1, dtype=torch.int32, device=device
        ),
        "indices": torch.empty(
            num_indices, dtype=torch.int32, device=device
        ),
        "last_page": torch.empty(
            padded_batch, dtype=torch.int32, device=device
        ),
    }

    out = torch.ops.xllm_ops.update_decode_graph_metadata(
        src["tokens"], src["positions"], src["slots"], src["kv_lens"],
        src["indptr"], src["indices"], src["last_page"], dst["tokens"],
        dst["positions"], dst["slots"], dst["kv_lens"],
        dst["kv_lens_delta"], dst["indptr"], dst["indices"],
        dst["last_page"], padded_batch,
    )

    repeated_cu = torch.cat([
        indptr_cpu,
        torch.full((padding,), num_indices, dtype=torch.int32),
    ])
    expected = {
        "tokens": torch.cat([
            src["tokens"].cpu(), torch.zeros(padding, dtype=torch.int32)
        ]),
        "positions": torch.cat([
            src["positions"].cpu(), torch.zeros(padding, dtype=torch.int32)
        ]),
        "slots": torch.cat([
            src["slots"].cpu(), -torch.ones(padding, dtype=torch.int32)
        ]),
        "kv_lens": repeated_cu,
        "kv_lens_delta": torch.diff(repeated_cu),
        "indptr": repeated_cu,
        "indices": src["indices"].cpu(),
        "last_page": torch.ones(padded_batch, dtype=torch.int32),
    }

    assert out.data_ptr() == dst["tokens"].data_ptr()
    for name, expected_tensor in expected.items():
        torch.testing.assert_close(dst[name].cpu(), expected_tensor, rtol=0, atol=0)
)PY");
}

}  // namespace
}  // namespace xllm
