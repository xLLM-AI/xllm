/* Copyright 2026 The xLLM Authors.

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

// NPU acceptance test for the xllm_ops torch-op library.
//
// Mirrors the CUDA xllm_ops_test: verifies TORCH_LIBRARY registrations survive
// linking on NPU (PrivateUse1), ops are callable via the dispatcher, and the
// embedded Python interpreter sees torch.ops.xllm_ops.*.

#include <acl/acl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <filesystem>
#include <limits>
#include <optional>
#include <string>

#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/xllm_torch_ops.h"

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
  const std::string python_model_path = repo_root.string();
  py::list sys_path = py::module_::import("sys").attr("path");
  sys_path.attr("insert")(0, python_model_path);
}

bool is_npu_available() {
  return c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
             ->deviceCount() > 0;
}

bool is_ascend950_device() {
  const char* soc_name = aclrtGetSocName();
  return soc_name != nullptr &&
         std::string(soc_name).find("Ascend950") != std::string::npos;
}

torch::Tensor expand_kv_heads_reference(const torch::Tensor& tensor,
                                        int64_t num_heads) {
  const int64_t num_kv_heads = tensor.size(1);
  EXPECT_EQ(num_heads % num_kv_heads, 0);
  const int64_t expansion_factor = num_heads / num_kv_heads;
  return tensor.unsqueeze(2)
      .expand({tensor.size(0), num_kv_heads, expansion_factor, tensor.size(2)})
      .reshape({tensor.size(0), num_heads, tensor.size(2)});
}

torch::Tensor packed_causal_attention_reference(const torch::Tensor& query,
                                                const torch::Tensor& key,
                                                const torch::Tensor& value,
                                                double scale) {
  const auto query_float = query.to(torch::kFloat32).permute({1, 0, 2});
  const auto key_float =
      expand_kv_heads_reference(key.to(torch::kFloat32), query.size(1));
  const auto value_float =
      expand_kv_heads_reference(value.to(torch::kFloat32), query.size(1));
  auto scores =
      torch::matmul(query_float, key_float.permute({1, 2, 0})) * scale;
  const auto causal_mask =
      torch::ones({query.size(0), key.size(0)}, torch::kBool).triu(1);
  scores.masked_fill_(causal_mask, -std::numeric_limits<float>::infinity());
  return torch::matmul(torch::softmax(scores, -1),
                       value_float.permute({1, 0, 2}))
      .permute({1, 0, 2});
}

torch::Tensor decode_attention_reference(const torch::Tensor& query,
                                         const torch::Tensor& key,
                                         const torch::Tensor& value,
                                         double scale) {
  const auto query_float = query.to(torch::kFloat32).squeeze(0);
  const auto key_float =
      expand_kv_heads_reference(key.to(torch::kFloat32), query.size(1));
  const auto value_float =
      expand_kv_heads_reference(value.to(torch::kFloat32), query.size(1));
  const auto scores =
      torch::matmul(query_float.unsqueeze(1), key_float.permute({1, 2, 0})) *
      scale;
  return torch::matmul(torch::softmax(scores, -1),
                       value_float.permute({1, 0, 2}))
      .squeeze(1)
      .unsqueeze(0);
}

class NpuXllmOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    xllm::ensure_xllm_torch_ops_registered();
    if (!is_npu_available()) {
      GTEST_SKIP() << "NPU not available; skipping xllm_ops NPU test.";
    }
    if (!Py_IsInitialized()) {
      setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);
      Py_InitializeEx(0);
    }
    py::gil_scoped_acquire gil;
    prepend_python_model_path();
    py::module_::import("xllm.python._npu_bootstrap");
    py::module_::import("xllm.python");
  }
};

TEST_F(NpuXllmOpsTest, DispatcherRmsNormMatchesReference) {
  py::gil_scoped_acquire gil;
  auto opts =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kPrivateUse1);
  auto input = torch::randn({8, 128}, opts);
  auto weight = torch::randn({128}, opts);
  const double eps = 1e-6;

  auto op =
      c10::Dispatcher::singleton().findSchemaOrThrow("xllm_ops::rms_norm", "");
  auto out = op.typed<torch::Tensor(
      const torch::Tensor&, const torch::Tensor&, double)>()
                 .call(input, weight, eps);

  auto ref = rms_norm_reference(input, weight, eps);
  EXPECT_TRUE(
      torch::allclose(out.cpu(), ref.cpu(), /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max abs diff = "
      << (out.cpu().to(torch::kFloat32) - ref.cpu().to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, DispatcherSiluAndMulMatchesReference) {
  py::gil_scoped_acquire gil;
  auto opts =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kPrivateUse1);
  auto gate_up = torch::randn({8, 256}, opts);

  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "xllm_ops::silu_and_mul", "");
  auto out = op.typed<torch::Tensor(const torch::Tensor&)>().call(gate_up);

  auto ref = silu_and_mul_reference(gate_up);
  ASSERT_EQ(out.size(-1), 128);
  EXPECT_TRUE(
      torch::allclose(out.cpu(), ref.cpu(), /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max abs diff = "
      << (out.cpu().to(torch::kFloat32) - ref.cpu().to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, EmbeddedInterpreterSeesOps) {
  py::gil_scoped_acquire gil;

  auto opts =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kPrivateUse1);
  auto gate_up = torch::randn({8, 256}, opts);

  py::module_ torch_mod = py::module_::import("torch");
  py::object xllm_ops = torch_mod.attr("ops").attr("xllm_ops");
  py::object out_obj = xllm_ops.attr("silu_and_mul")(gate_up);
  auto out = out_obj.cast<torch::Tensor>();

  auto ref = silu_and_mul_reference(gate_up);
  ASSERT_EQ(out.size(-1), 128);
  EXPECT_TRUE(
      torch::allclose(out.cpu(), ref.cpu(), /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max abs diff = "
      << (out.cpu().to(torch::kFloat32) - ref.cpu().to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, Qwen35_27B_TP4_FullAttentionMatchesReference) {
  py::gil_scoped_acquire gil;
  if (!is_ascend950_device()) {
    GTEST_SKIP() << "Ascend950 is required for the A5 attention path.";
  }

  constexpr int64_t kSequenceLength = 129;
  constexpr int64_t kQueryHeads = 6;
  constexpr int64_t kKvHeads = 1;
  constexpr int64_t kHeadDim = 256;
  constexpr double kScale = 1.0 / 16.0;
  torch::manual_seed(20260729);

  const auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  const auto query_cpu =
      (0.25 * torch::randn({kSequenceLength, kQueryHeads, kHeadDim}, cpu_float))
          .to(torch::kBFloat16);
  const auto key_cpu =
      (0.25 * torch::randn({kSequenceLength, kKvHeads, kHeadDim}, cpu_float))
          .to(torch::kBFloat16);
  const auto value_cpu =
      torch::randn({kSequenceLength, kKvHeads, kHeadDim}, cpu_float)
          .to(torch::kBFloat16);
  const auto query = query_cpu.to(torch::kPrivateUse1);
  const auto key = key_cpu.to(torch::kPrivateUse1);
  const auto value = value_cpu.to(torch::kPrivateUse1);

  const auto [actual, softmax_lse] =
      xllm::kernel::npu::npu_fused_infer_attention(query,
                                                   key,
                                                   value,
                                                   std::nullopt,
                                                   std::nullopt,
                                                   {kSequenceLength},
                                                   {kSequenceLength},
                                                   kQueryHeads,
                                                   kKvHeads,
                                                   kScale,
                                                   /*block_size=*/128,
                                                   /*sparse_mode=*/0,
                                                   /*input_layout=*/"TND",
                                                   /*softmax_lse_flag=*/false);
  const auto expected =
      packed_causal_attention_reference(query_cpu, key_cpu, value_cpu, kScale);

  EXPECT_EQ(actual.sizes(), query.sizes());
  EXPECT_EQ(softmax_lse.numel(), 0);
  EXPECT_TRUE(torch::allclose(actual.cpu().to(torch::kFloat32),
                              expected,
                              /*rtol=*/5e-2,
                              /*atol=*/5e-2))
      << "max abs diff = "
      << (actual.cpu().to(torch::kFloat32) - expected)
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, Qwen35_27B_TP4_KvCacheCrosses128TokenBoundary) {
  py::gil_scoped_acquire gil;
  if (!is_ascend950_device()) {
    GTEST_SKIP() << "Ascend950 is required for the A5 paged-cache path.";
  }

  constexpr int64_t kSequenceLength = 130;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kNumPhysicalBlocks = 3;
  constexpr int64_t kQueryHeads = 6;
  constexpr int64_t kKvHeads = 1;
  constexpr int64_t kHeadDim = 256;
  constexpr double kScale = 1.0 / 16.0;
  torch::manual_seed(20260730);

  const auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  const auto key_cpu =
      torch::randn({kSequenceLength, kKvHeads, kHeadDim}, cpu_float)
          .to(torch::kBFloat16);
  const auto value_cpu =
      torch::randn({kSequenceLength, kKvHeads, kHeadDim}, cpu_float)
          .to(torch::kBFloat16);
  const auto query_cpu =
      (0.25 * torch::randn({1, kQueryHeads, kHeadDim}, cpu_float))
          .to(torch::kBFloat16);
  auto key = key_cpu.to(torch::kPrivateUse1);
  auto value_tensor = value_cpu.to(torch::kPrivateUse1);
  std::optional<torch::Tensor> value = value_tensor;

  const auto npu_bfloat = torch::TensorOptions()
                              .dtype(torch::kBFloat16)
                              .device(torch::kPrivateUse1);
  auto key_cache = torch::zeros(
      {kNumPhysicalBlocks, kBlockSize, kKvHeads, kHeadDim}, npu_bfloat);
  auto value_cache_tensor = torch::zeros_like(key_cache);
  std::optional<torch::Tensor> value_cache = value_cache_tensor;

  const auto first_block_slots =
      torch::arange(2 * kBlockSize,
                    3 * kBlockSize,
                    torch::TensorOptions().dtype(torch::kInt32));
  const auto second_block_slots =
      torch::arange(0, 2, torch::TensorOptions().dtype(torch::kInt32));
  const auto slot_mapping = torch::cat({first_block_slots, second_block_slots})
                                .to(torch::kPrivateUse1);
  xllm::kernel::npu::reshape_paged_cache(
      key, value, key_cache, value_cache, slot_mapping);

  auto expected_key_cache =
      torch::zeros({kNumPhysicalBlocks, kBlockSize, kKvHeads, kHeadDim},
                   torch::TensorOptions().dtype(torch::kBFloat16));
  auto expected_value_cache = torch::zeros_like(expected_key_cache);
  expected_key_cache[2].copy_(key_cpu.narrow(0, 0, kBlockSize));
  expected_value_cache[2].copy_(value_cpu.narrow(0, 0, kBlockSize));
  expected_key_cache[0].narrow(0, 0, 2).copy_(key_cpu.narrow(0, kBlockSize, 2));
  expected_value_cache[0].narrow(0, 0, 2).copy_(
      value_cpu.narrow(0, kBlockSize, 2));

  EXPECT_TRUE(torch::equal(key_cache.cpu(), expected_key_cache));
  EXPECT_TRUE(torch::equal(value_cache.value().cpu(), expected_value_cache));

  const auto query = query_cpu.to(torch::kPrivateUse1);
  const auto block_table =
      torch::tensor({{2, 0}}, torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kPrivateUse1);
  const auto seq_lens =
      torch::tensor({kSequenceLength},
                    torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kPrivateUse1);
  auto actual = torch::empty_like(query);
  xllm::kernel::npu::batch_decode(query,
                                  key_cache,
                                  value_cache.value(),
                                  kScale,
                                  block_table,
                                  seq_lens,
                                  actual);
  const auto expected =
      decode_attention_reference(query_cpu, key_cpu, value_cpu, kScale);

  EXPECT_TRUE(torch::allclose(actual.cpu().to(torch::kFloat32),
                              expected,
                              /*rtol=*/5e-2,
                              /*atol=*/5e-2))
      << "max abs diff = "
      << (actual.cpu().to(torch::kFloat32) - expected)
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, ModelExecutorUsesExplicitRuntimeBatchLimit) {
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
        self.weight = torch.nn.Parameter(
            torch.zeros(1, device="privateuseone:0")
        )
        self.attention = Attention(1, 1, 8, 1.0, 0, 0)
        self.model = torch.nn.Identity()


with patch.object(
    executor_module, "_create_attention_backend", return_value=FakeBackend()
):
    model_executor = executor_module.ModelExecutor(
        FakeModel(),
        {"python_graph_backend": "off"},
        max_seqs_per_batch=3,
    )
    assert model_executor._num_attention_layers == 1
    assert model_executor.decode_graph_runner is None
    assert model_executor.inductor_runner is None
)PY");
}

}  // namespace
}  // namespace xllm
