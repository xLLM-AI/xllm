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

#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
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

struct MegaGdnInputs {
  torch::Tensor qkv;
  torch::Tensor z;
  torch::Tensor b;
  torch::Tensor a;
  torch::Tensor conv_weight;
  torch::Tensor conv_state;
  torch::Tensor a_log;
  torch::Tensor dt_bias;
  torch::Tensor ssm_state;
  torch::Tensor read_indices;
  torch::Tensor write_indices;
  torch::Tensor norm_weight;
};

MegaGdnInputs make_mega_gdn_inputs(int64_t sequence_length, bool same_slot) {
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kKeyHeads = 1;
  constexpr int64_t kValueHeads = 1;
  constexpr int64_t kConvDim = (2 * kKeyHeads + kValueHeads) * kHeadDim;
  constexpr int64_t kSlots = 2;
  torch::manual_seed(20260831 + sequence_length);
  const auto cpu_bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  const auto npu = torch::Device(torch::kPrivateUse1);
  const int64_t write_slot = same_slot ? 0 : 1;

  MegaGdnInputs inputs;
  inputs.qkv = (torch::randn({1, sequence_length, kConvDim}, cpu_float) * 0.1)
                   .to(torch::kBFloat16)
                   .to(npu);
  inputs.z =
      (torch::randn({1, sequence_length, kValueHeads, kHeadDim}, cpu_float) *
       0.1)
          .to(torch::kBFloat16)
          .to(npu);
  inputs.b = (torch::randn({1, sequence_length, kValueHeads}, cpu_float) * 0.1)
                 .to(torch::kBFloat16)
                 .to(npu);
  inputs.a = (torch::randn({1, sequence_length, kValueHeads}, cpu_float) * 0.1)
                 .to(torch::kBFloat16)
                 .to(npu);
  inputs.conv_weight = (torch::randn({4, kConvDim}, cpu_float) * 0.05)
                           .to(torch::kBFloat16)
                           .to(npu);
  inputs.conv_state =
      (torch::randn({kSlots, sequence_length + 2, kConvDim}, cpu_float) * 0.1)
          .to(torch::kBFloat16)
          .to(npu);
  inputs.a_log = torch::full({kValueHeads}, -1.0, cpu_float).to(npu);
  inputs.dt_bias = torch::zeros({kValueHeads}, cpu_float).to(npu);
  inputs.ssm_state =
      (torch::randn({kSlots * sequence_length, kValueHeads, kHeadDim, kHeadDim},
                    cpu_float) *
       0.02)
          .to(npu);
  inputs.read_indices =
      torch::tensor({0}, torch::TensorOptions().dtype(torch::kInt32)).to(npu);
  inputs.write_indices =
      torch::tensor({write_slot}, torch::TensorOptions().dtype(torch::kInt32))
          .to(npu);
  inputs.norm_weight = torch::ones({kHeadDim}, cpu_bf16).to(npu);
  return inputs;
}

void expect_same_storage(const torch::Tensor& output,
                         const torch::Tensor& input) {
  EXPECT_EQ(output.const_data_ptr(), input.const_data_ptr());
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
    py::module_::import("xllm.python").attr("initialize_runtime")();
  }
};

TEST(NpuXllmOpsCapabilityTest, MlaPreprocessV2MatchesRuntimeSymbols) {
  xllm::ensure_xllm_torch_ops_registered();
  const bool expected =
      kernel::npu::aclnn::detail::get_op_api_func_addr(
          "aclnnMlaPreprocessV2GetWorkspaceSize") != nullptr &&
      kernel::npu::aclnn::detail::get_op_api_func_addr(
          "aclnnMlaPreprocessV2") != nullptr;
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "xllm_ops::has_mla_preprocess_v2", "");
  const bool actual = op.typed<bool()>().call();

  EXPECT_EQ(actual, expected);
}

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

TEST_F(NpuXllmOpsTest, DispatcherAtbMatmulEinSumMatchesReference) {
  py::gil_scoped_acquire gil;
  torch::manual_seed(20260923);
  const auto input_cpu =
      torch::randn({4, 4, 512}, torch::TensorOptions().dtype(torch::kBFloat16));
  const auto weight_cpu =
      (torch::randn({4, 512, 256}, torch::kFloat32) / std::sqrt(512.0))
          .to(torch::kBFloat16);
  const auto input = input_cpu.to(torch::kPrivateUse1);
  const auto weight = weight_cpu.to(torch::kPrivateUse1);
  const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "xllm_ops::atb_matmul_ein_sum", "");
  const auto output =
      op.typed<torch::Tensor(const torch::Tensor&, const torch::Tensor&)>()
          .call(input, weight);
  const auto actual = output.cpu().to(torch::kFloat32);
  const auto reference = torch::einsum(
      "thd,hdo->tho",
      {input_cpu.to(torch::kFloat32), weight_cpu.to(torch::kFloat32)});

  ASSERT_EQ(output.sizes(), reference.sizes());
  EXPECT_EQ(output.scalar_type(), input.scalar_type());
  EXPECT_EQ(output.device(), input.device());
  EXPECT_TRUE(output.is_contiguous());
  EXPECT_TRUE(torch::allclose(actual, reference, /*rtol=*/1e-2, /*atol=*/2e-2))
      << "max abs diff = " << (actual - reference).abs().max().item<float>();
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

TEST_F(NpuXllmOpsTest, DispatcherQuantizeMatchesStaticW8A8Reference) {
  py::gil_scoped_acquire gil;
  const auto input_opts = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto scale_opts = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto zero_point_opts = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto input_cpu = torch::tensor(
      std::vector<float>{-40.0F, -1.0F, -0.25F, 0.0F, 0.25F, 1.0F, 40.0F},
      input_opts);
  const auto scale_cpu = torch::full({1}, 0.25, scale_opts);
  const auto zero_point_cpu = torch::full({1}, 2, zero_point_opts);
  const auto input = input_cpu.to(torch::kPrivateUse1);
  const auto scale = scale_cpu.to(torch::kPrivateUse1);
  const auto zero_point = zero_point_cpu.to(torch::kPrivateUse1);

  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "xllm_ops::quantize_per_tensor", "");
  auto actual = op.typed<torch::Tensor(const torch::Tensor&,
                                       const torch::Tensor&,
                                       const torch::Tensor&,
                                       at::ScalarType,
                                       int64_t)>()
                    .call(input, scale, zero_point, at::ScalarType::QInt8, -1);

  const auto expected =
      torch::clamp(torch::round(input_cpu.to(torch::kFloat32) /
                                    scale_cpu.to(torch::kFloat32) +
                                zero_point_cpu.to(torch::kFloat32)),
                   -128,
                   127)
          .to(torch::kInt8);
  EXPECT_EQ(actual.scalar_type(), torch::kInt8);
  EXPECT_TRUE(torch::equal(actual.cpu(), expected));
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

TEST_F(NpuXllmOpsTest, MlaIndexCacheWriteReplaysWithChangingPadding) {
  py::gil_scoped_acquire gil;
  py::exec(R"PY(
import torch
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend

device = torch.device("npu:0")
for cache_dtype in (torch.bfloat16, torch.float16, torch.int8):
    for slot_dtype in (torch.int32, torch.int64):
        cache = torch.full((2, 4, 1, 128), -3, dtype=cache_dtype, device=device)
        values = (torch.arange(6 * 128).reshape(6, 128) % 97).to(
            device=device, dtype=cache_dtype
        )
        slots = torch.tensor([0, -1, 3, -2, 7, -1], dtype=slot_dtype, device=device)
        scale_cache = None
        scales = None
        if cache_dtype == torch.int8:
            scale_cache = torch.full((2, 4, 1, 1), -3, dtype=torch.float16, device=device)
            scales = torch.arange(1, 7, dtype=torch.float16, device=device).view(6, 1)

        def _write() -> None:
            NpuPagedAttentionBackend._update_mla_index_cache(
                cache, scale_cache, slots, values, scales
            )

        def _check() -> None:
            expected = torch.full((8, 128), -3, dtype=cache_dtype)
            expected_scale = torch.full((8, 1), -3, dtype=torch.float16)
            for row, slot in enumerate(slots.cpu().tolist()):
                if slot >= 0:
                    expected[slot] = values[row].cpu()
                    if scales is not None:
                        expected_scale[slot] = scales[row].cpu()
            torch.testing.assert_close(cache.cpu().view(8, 128), expected, rtol=0, atol=0)
            if scale_cache is not None:
                torch.testing.assert_close(
                    scale_cache.cpu().view(8, 1), expected_scale, rtol=0, atol=0
                )

        _write()
        _check()
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            _write()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            _write()
        for replay_slots in ([7, -1, 0, -1, 3, -1], [-1] * 6):
            cache.fill_(-3)
            values.add_(1)
            slots.copy_(torch.tensor(replay_slots, dtype=slot_dtype, device=device))
            if scale_cache is not None:
                scale_cache.fill_(-3)
                scales.add_(1)
            graph.replay()
            torch.npu.synchronize()
            _check()
        NpuPagedAttentionBackend._update_mla_index_cache(
            cache, scale_cache, slots[:0], values[:0],
            None if scales is None else scales[:0]
        )
        _check()
)PY");
}

TEST_F(NpuXllmOpsTest, Dsv4OpsUseNpuDispatchKeys) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch

device_ops = (
    "moe_gating_top_k_hash",
    "dequant_swiglu_quant",
    "hc_pre",
    "hc_post",
    "compressor",
    "mega_moe",
    "sparse_attn_sharedkv",
    "quant_lightning_indexer",
)
for op_name in device_ops:
    qualname = f"xllm_ops::{op_name}"
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        qualname, "PrivateUse1"
    ), qualname
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(
        qualname, "CompositeExplicitAutograd"
    ), qualname

for op_name in (
    "sparse_attn_sharedkv_metadata",
    "quant_lightning_indexer_metadata",
):
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        f"xllm_ops::{op_name}", "CompositeExplicitAutograd"
    ), op_name
)PY");
}

TEST_F(NpuXllmOpsTest, Dsv4GroupGemmMatchesInt32Reference) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch
from xllm.python.kernels_npu.moe import _group_gemm

device = torch.device("privateuseone:0")
torch.manual_seed(20260814)
tokens, experts, input_dim, output_dim = 8, 2, 128, 256
x_cpu = torch.randint(-4, 5, (tokens, input_dim), dtype=torch.int8)
w_cpu = torch.randint(-4, 5, (experts, input_dim, output_dim), dtype=torch.int8)
group_list_cpu = torch.tensor([4, 4], dtype=torch.int64)

x = x_cpu.to(device)
w = w_cpu.to(device)
group_list = group_list_cpu.to(device)
out = _group_gemm(
    x=x,
    weight=w,
    scale=None,
    per_token_scale=None,
    group_list=group_list,
    split_item=2,
    group_type=0,
    group_list_type=1,
    output_dtype=torch.int32,
)
torch.npu.synchronize()

expected = torch.cat((
    x_cpu[:4].to(torch.int32) @ w_cpu[0].to(torch.int32),
    x_cpu[4:].to(torch.int32) @ w_cpu[1].to(torch.int32),
), dim=0)
assert out.shape == (tokens, output_dim)
assert out.dtype == torch.int32
torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
)PY");
}

TEST_F(NpuXllmOpsTest, Dsv4GroupGemmAcceptsScaleAndPerTokenScale) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch
from xllm.python.kernels_npu.moe import _group_gemm

device = torch.device("privateuseone:0")
tokens, experts, input_dim, output_dim = 8, 2, 128, 128
x = torch.randint(-4, 5, (tokens, input_dim), dtype=torch.int8, device=device)
w = torch.randint(-4, 5, (experts, input_dim, output_dim), dtype=torch.int8, device=device)
scale = torch.ones((experts, output_dim), dtype=torch.bfloat16, device=device)
per_token_scale = torch.ones((tokens,), dtype=torch.float32, device=device)
group_list = torch.tensor([4, 4], dtype=torch.int64, device=device)

out = _group_gemm(
    x=x,
    weight=w,
    scale=scale,
    per_token_scale=per_token_scale,
    group_list=group_list,
    split_item=2,
    group_type=0,
    group_list_type=1,
    output_dtype=torch.bfloat16,
)
torch.npu.synchronize()
assert out.shape == (tokens, output_dim)
assert out.dtype == torch.bfloat16
)PY");
}

TEST_F(NpuXllmOpsTest, Dsv4PartialRotaryPythonWrapperRunsOnNpu) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch
from xllm.python.kernels_npu.rotary_embedding import (
    npu_inplace_partial_rotary_mul,
)

torch.manual_seed(2026)
x_cpu = torch.randn((8, 2, 128), dtype=torch.float32).to(torch.bfloat16)
cos_cpu = torch.randn((8, 64), dtype=torch.float32).to(torch.bfloat16)
sin_cpu = torch.randn((8, 64), dtype=torch.float32).to(torch.bfloat16)

expected = x_cpu.float().clone()
segment = x_cpu[..., 64:128].float()
swapped = torch.empty_like(segment)
swapped[..., 0::2] = segment[..., 1::2]
swapped[..., 1::2] = segment[..., 0::2]
sign = torch.ones_like(cos_cpu.float())
sign[..., 0::2] = -1
expected[..., 64:128] = (
    segment * cos_cpu.float().unsqueeze(1)
    + swapped * sin_cpu.float().unsqueeze(1) * sign.unsqueeze(1)
)
expected = expected.to(torch.bfloat16).float()

x = x_cpu.to("privateuseone:0")
cos = cos_cpu.to(x.device)
sin = sin_cpu.to(x.device)
result = npu_inplace_partial_rotary_mul(x, cos, sin, 64, 64)
torch.npu.synchronize()

assert result.data_ptr() == x.data_ptr()
torch.testing.assert_close(
    x.cpu().float(), expected, atol=2e-2, rtol=2e-2
)
)PY");
}

TEST_F(NpuXllmOpsTest, Dsv4CompressorPythonWrapperRunsOnNpu) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch
from xllm.python.kernels_npu.dsa import compressor

device = torch.device("privateuseone:0")
torch.manual_seed(2025)
batch, tokens, hidden = 1, 128, 1024
ratio, head_dim, coff, rope_dim = 128, 512, 1, 64
compressed_tokens = tokens // ratio

x_cpu = (torch.randn(batch, tokens, hidden) * 0.1).to(torch.float16)
wkv_cpu = (torch.randn(coff * head_dim, hidden) * 0.05).to(torch.float16)
wgate_cpu = (torch.randn(coff * head_dim, hidden) * 0.05).to(torch.float16)
ape_cpu = (torch.randn(ratio, coff * head_dim) * 0.1).float()
norm_cpu = (torch.randn(head_dim) * 0.1 + 1).to(torch.float16)
rope_cos_cpu = (
    torch.randn(batch, compressed_tokens, rope_dim) * 0.1
).to(torch.float16)
rope_sin_cpu = (
    torch.randn(batch, compressed_tokens, rope_dim) * 0.1
).to(torch.float16)

projected_kv = x_cpu.float()[0] @ wkv_cpu.float().T
scores = x_cpu.float()[0] @ wgate_cpu.float().T + ape_cpu
pooled = (torch.softmax(scores, dim=0) * projected_kv).sum(0, keepdim=True)
variance = pooled.square().mean(-1, keepdim=True)
expected = pooled * torch.rsqrt(variance + 1e-6) * norm_cpu.float()
rope_segment = expected[:, -rope_dim:].clone()
half = rope_dim // 2
rotated = torch.cat((-rope_segment[:, half:], rope_segment[:, :half]), dim=-1)
expected[:, -rope_dim:] = (
    rope_segment * rope_cos_cpu.float()[0]
    + rotated * rope_sin_cpu.float()[0]
)
expected = expected.view(batch, compressed_tokens, head_dim).half().float()

x = x_cpu.to(device)
wkv = wkv_cpu.to(device)
wgate = wgate_cpu.to(device)
ape = ape_cpu.to(device)
norm_weight = norm_cpu.to(device)
rope_sin = rope_sin_cpu.to(device)
rope_cos = rope_cos_cpu.to(device)
kv_state = torch.zeros((1, 128, head_dim), dtype=torch.float32, device=device)
score_state = torch.zeros_like(kv_state)
kv_block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
score_block_table = torch.tensor([[0]], dtype=torch.int32, device=device)

out, wkv_proj, softmax_res, norm_x, norm_rstd = compressor(
    x,
    wkv,
    wgate,
    kv_state,
    score_state,
    ape,
    norm_weight,
    rope_sin,
    rope_cos,
    kv_block_table,
    score_block_table,
    None,
    None,
    None,
    rope_dim,
    ratio,
    coff,
    1e-6,
    1,
    False,
)
torch.npu.synchronize()

assert out.shape == (batch, compressed_tokens, head_dim)
assert out.dtype == torch.float16
assert wkv_proj.numel() == 0
assert softmax_res.numel() == 0
assert norm_x.numel() == 0
assert norm_rstd.numel() == 0
torch.testing.assert_close(
    out.cpu().float(), expected, atol=2e-2, rtol=2e-2
)
)PY");
}

TEST_F(NpuXllmOpsTest,
       FusedInferAttentionDecodeOutMatchesEagerAcrossBlockBoundary) {
  py::gil_scoped_acquire gil;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kQueryHeads = 16;
  constexpr int64_t kKvHeads = 4;
  constexpr int64_t kHeadDim = 256;
  constexpr int64_t kNumPhysicalBlocks = 4;
  constexpr double kScale = 1.0 / 16.0;
  const std::vector<int64_t> actual_seq_lengths = {1, 2, 3};
  const std::vector<int64_t> actual_seq_lengths_kv = {127, 128, 129};

  torch::manual_seed(20260811);
  const torch::TensorOptions cpu_float =
      torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor query = torch::randn({3, kQueryHeads, kHeadDim}, cpu_float)
                            .to(torch::kBFloat16)
                            .to(torch::kPrivateUse1)
                            .contiguous();
  torch::Tensor key =
      torch::randn({kNumPhysicalBlocks, kBlockSize, kKvHeads * kHeadDim},
                   cpu_float)
          .to(torch::kBFloat16)
          .to(torch::kPrivateUse1)
          .contiguous();
  torch::Tensor value =
      torch::randn({kNumPhysicalBlocks, kBlockSize, kKvHeads * kHeadDim},
                   cpu_float)
          .to(torch::kBFloat16)
          .to(torch::kPrivateUse1)
          .contiguous();
  torch::Tensor block_table =
      torch::tensor({{0, 0}, {1, 0}, {2, 3}},
                    torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kPrivateUse1);

  auto eager_result = xllm::kernel::npu::npu_fused_infer_attention(
      query,
      key,
      value,
      /*atten_mask=*/std::nullopt,
      std::make_optional(block_table),
      actual_seq_lengths,
      actual_seq_lengths_kv,
      kQueryHeads,
      kKvHeads,
      kScale,
      kBlockSize,
      /*sparse_mode=*/0,
      /*input_layout=*/"TND");
  torch::Tensor eager_output = std::get<0>(eager_result);

  torch::Tensor workspace =
      xllm::kernel::npu::npu_fused_infer_attention_decode_get_max_workspace(
          query,
          key,
          value,
          block_table,
          actual_seq_lengths,
          actual_seq_lengths_kv,
          kQueryHeads,
          kKvHeads,
          kScale,
          kBlockSize);
  ASSERT_TRUE(workspace.defined());
  EXPECT_EQ(workspace.device(), query.device());

  torch::Tensor out = torch::zeros_like(eager_output);
  torch::Tensor softmax_lse = torch::empty({0}, query.options());
  const void* out_data = out.const_data_ptr();
  xllm::kernel::npu::npu_fused_infer_attention_decode_out(query,
                                                          key,
                                                          value,
                                                          block_table,
                                                          actual_seq_lengths,
                                                          actual_seq_lengths_kv,
                                                          kQueryHeads,
                                                          kKvHeads,
                                                          kScale,
                                                          kBlockSize,
                                                          workspace,
                                                          out,
                                                          softmax_lse);

  EXPECT_EQ(out.const_data_ptr(), out_data);
  EXPECT_EQ(out.sizes(), eager_output.sizes());
  EXPECT_EQ(out.scalar_type(), torch::kBFloat16);
  EXPECT_EQ(softmax_lse.numel(), 0);
  const torch::Tensor actual = out.cpu().to(torch::kFloat32);
  const torch::Tensor expected = eager_output.cpu().to(torch::kFloat32);
  EXPECT_TRUE(torch::allclose(actual,
                              expected,
                              /*rtol=*/1e-3,
                              /*atol=*/2e-3))
      << "max abs diff = " << (actual - expected).abs().max().item<float>();
}

TEST_F(NpuXllmOpsTest,
       FusedInferAttentionDecodeCachedOutMatchesEagerAcrossDynamicShapes) {
  py::gil_scoped_acquire gil;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kQueryHeads = 16;
  constexpr int64_t kKvHeads = 4;
  constexpr int64_t kHeadDim = 256;
  constexpr int64_t kNumPhysicalBlocks = 64;
  constexpr double kScale = 1.0 / 16.0;

  torch::manual_seed(20260826);
  const torch::TensorOptions cpu_float =
      torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor key =
      torch::randn({kNumPhysicalBlocks, kBlockSize, kKvHeads * kHeadDim},
                   cpu_float)
          .to(torch::kBFloat16)
          .to(torch::kPrivateUse1)
          .contiguous();
  torch::Tensor value =
      torch::randn({kNumPhysicalBlocks, kBlockSize, kKvHeads * kHeadDim},
                   cpu_float)
          .to(torch::kBFloat16)
          .to(torch::kPrivateUse1)
          .contiguous();

  auto run_case = [&](const std::vector<int64_t>& actual_seq_lengths_kv,
                      const std::vector<int32_t>& block_ids,
                      int64_t block_table_width) {
    const int64_t num_tokens =
        static_cast<int64_t>(actual_seq_lengths_kv.size());
    std::vector<int64_t> actual_seq_lengths;
    actual_seq_lengths.reserve(static_cast<size_t>(num_tokens));
    for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
      actual_seq_lengths.emplace_back(token_idx + 1);
    }

    torch::Tensor query =
        torch::randn({num_tokens, kQueryHeads, kHeadDim}, cpu_float)
            .to(torch::kBFloat16)
            .to(torch::kPrivateUse1)
            .contiguous();
    torch::Tensor block_table =
        torch::tensor(block_ids, torch::TensorOptions().dtype(torch::kInt32))
            .view({num_tokens, block_table_width})
            .to(torch::kPrivateUse1);
    auto eager_result = xllm::kernel::npu::npu_fused_infer_attention(
        query,
        key,
        value,
        /*atten_mask=*/std::nullopt,
        std::make_optional(block_table),
        actual_seq_lengths,
        actual_seq_lengths_kv,
        kQueryHeads,
        kKvHeads,
        kScale,
        kBlockSize,
        /*sparse_mode=*/0,
        /*input_layout=*/"TND");
    torch::Tensor expected = std::get<0>(eager_result);
    torch::Tensor output = torch::zeros_like(expected);
    const void* output_data = output.const_data_ptr();

    xllm::kernel::npu::npu_fused_infer_attention_decode_out_cached(
        query,
        key,
        value,
        block_table,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        kQueryHeads,
        kKvHeads,
        kScale,
        kBlockSize,
        output);

    EXPECT_EQ(output.const_data_ptr(), output_data);
    const torch::Tensor actual_cpu = output.cpu();
    const torch::Tensor expected_cpu = expected.cpu();
    EXPECT_TRUE(torch::equal(actual_cpu, expected_cpu))
        << "cached FIA output must be bitwise identical, max abs diff = "
        << (actual_cpu.to(torch::kFloat32) - expected_cpu.to(torch::kFloat32))
               .abs()
               .max()
               .item<float>();
  };

  run_case(/*actual_seq_lengths_kv=*/{127},
           /*block_ids=*/{0},
           /*block_table_width=*/1);
  run_case(/*actual_seq_lengths_kv=*/{127, 128, 129},
           /*block_ids=*/{0, 0, 1, 0, 2, 3},
           /*block_table_width=*/2);
  std::vector<int32_t> long_context_block_ids;
  long_context_block_ids.reserve(kNumPhysicalBlocks);
  for (int32_t block_id = 0;
       block_id < static_cast<int32_t>(kNumPhysicalBlocks);
       ++block_id) {
    long_context_block_ids.emplace_back(block_id);
  }
  run_case(/*actual_seq_lengths_kv=*/{kNumPhysicalBlocks * kBlockSize},
           long_context_block_ids,
           /*block_table_width=*/kNumPhysicalBlocks);
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
    supports_prepared_metadata = False

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

TEST_F(NpuXllmOpsTest, LightningIndexerOutKeepsBuffersAcrossGraphReplay) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch

query = torch.randn(
    (1, 64, 128), dtype=torch.bfloat16, device="privateuseone:0"
)
key = torch.randn(
    (1, 16, 1, 128), dtype=torch.bfloat16, device="privateuseone:0"
)
weights = torch.randn(
    (1, 64), dtype=torch.bfloat16, device="privateuseone:0"
)
query_seq_lengths = torch.tensor(
    [1], dtype=torch.int32, device="privateuseone:0"
)
key_seq_lengths = torch.tensor(
    [16], dtype=torch.int32, device="privateuseone:0"
)
block_table = torch.tensor(
    [[0]], dtype=torch.int32, device="privateuseone:0"
)
sparse_indices = torch.empty(
    (1, 1, 4), dtype=torch.int32, device="privateuseone:0"
)
sparse_values = torch.empty(
    (1, 1, 4), dtype=torch.bfloat16, device="privateuseone:0"
)
indices_address = sparse_indices.data_ptr()
values_address = sparse_values.data_ptr()


def run_indexer():
    return torch.ops.xllm_ops.lightning_indexer_out(
        query,
        key,
        weights,
        query_seq_lengths,
        key_seq_lengths,
        block_table,
        "TND",
        "PA_BSND",
        4,
        3,
        2**63 - 1,
        2**63 - 1,
        False,
        sparse_indices,
        sparse_values,
    )


eager_result = run_indexer()
assert eager_result.shape == (1, 1, 4)
assert eager_result.dtype == torch.int32
assert eager_result.data_ptr() == indices_address
assert sparse_values.shape == (1, 1, 4)
assert sparse_values.dtype == torch.bfloat16
assert sparse_values.data_ptr() == values_address

stream = torch.npu.Stream()
graph = torch.npu.NPUGraph()
with torch.npu.stream(stream):
    run_indexer()
torch.npu.synchronize()
with torch.npu.stream(stream):
    with torch.npu.graph(graph, stream=stream):
        graph_result = run_indexer()
torch.npu.synchronize()

with torch.npu.stream(stream):
    query.add_(0.25)
    graph.replay()
    query.sub_(0.5)
    graph.replay()
torch.npu.synchronize()

assert graph_result.data_ptr() == indices_address
assert sparse_indices.data_ptr() == indices_address
assert sparse_values.data_ptr() == values_address
)PY");
}

TEST_F(NpuXllmOpsTest, MegaGdnPrefillColdAndSameSlotKeepD2DState) {
  py::gil_scoped_acquire gil;
  constexpr int64_t kSequenceLength = 128;
  constexpr int64_t kCheckpointStride = 2;
  for (const bool cold_start : {true, false}) {
    SCOPED_TRACE(::testing::Message() << "cold_start=" << cold_start);
    MegaGdnInputs inputs = make_mega_gdn_inputs(kSequenceLength,
                                                /*same_slot=*/true);
    inputs.conv_state =
        inputs.conv_state.slice(/*dim=*/1, 0, kCheckpointStride + 2)
            .contiguous();
    inputs.ssm_state =
        inputs.ssm_state.slice(/*dim=*/0, 0, 2 * kCheckpointStride)
            .contiguous();
    const auto index_options = torch::TensorOptions().dtype(torch::kInt32);
    auto conv_read_indices =
        torch::tensor({cold_start ? int32_t{-1} : int32_t{1}}, index_options)
            .to(torch::kPrivateUse1);
    auto conv_write_indices =
        torch::tensor({1}, index_options).to(torch::kPrivateUse1);
    auto ssm_read_indices =
        torch::tensor({cold_start ? int32_t{-1}
                                  : static_cast<int32_t>(kCheckpointStride)},
                      index_options)
            .to(torch::kPrivateUse1);
    auto ssm_write_indices =
        torch::tensor({static_cast<int32_t>(kCheckpointStride)}, index_options)
            .to(torch::kPrivateUse1);
    auto cu_seqlens =
        torch::tensor({int32_t{0}, static_cast<int32_t>(kSequenceLength)},
                      torch::TensorOptions().dtype(torch::kInt32))
            .to(torch::kPrivateUse1);

    const torch::Tensor expected_conv_tail =
        inputs.qkv.squeeze(0).slice(/*dim=*/0, kSequenceLength - 3).clone();
    torch::Tensor packed_qkv = inputs.qkv.squeeze(0);
    torch::Tensor packed_b = inputs.b.squeeze(0);
    torch::Tensor packed_a = inputs.a.squeeze(0);
    torch::Tensor packed_z = inputs.z.squeeze(0);
    const auto outputs =
        xllm::kernel::npu::npu_mega_gdn_prefill(packed_qkv,
                                                packed_b,
                                                packed_a,
                                                packed_z,
                                                inputs.conv_weight,
                                                inputs.conv_state,
                                                inputs.a_log,
                                                inputs.dt_bias,
                                                conv_read_indices,
                                                conv_write_indices,
                                                ssm_read_indices,
                                                ssm_write_indices,
                                                inputs.ssm_state,
                                                cu_seqlens,
                                                inputs.norm_weight,
                                                /*num_matrices=*/1);

    EXPECT_EQ(std::get<0>(outputs).sizes(), packed_z.sizes());
    expect_same_storage(std::get<1>(outputs), inputs.conv_state);
    expect_same_storage(std::get<2>(outputs), inputs.ssm_state);
    EXPECT_TRUE(torch::equal(inputs.conv_state[1].slice(/*dim=*/0, 0, 3).cpu(),
                             expected_conv_tail.cpu()));
    EXPECT_TRUE(torch::isfinite(std::get<0>(outputs)).all().item<bool>());
  }
}

TEST_F(NpuXllmOpsTest, MegaGdnPrefillDirectReadPreservesSourceSlot) {
  py::gil_scoped_acquire gil;
  constexpr int64_t kSequenceLength = 128;
  constexpr int64_t kCheckpointStride = 2;
  MegaGdnInputs inputs = make_mega_gdn_inputs(kSequenceLength,
                                              /*same_slot=*/false);
  inputs.conv_state =
      inputs.conv_state.slice(/*dim=*/1, 0, kCheckpointStride + 2).contiguous();
  inputs.ssm_state =
      inputs.ssm_state.slice(/*dim=*/0, 0, 2 * kCheckpointStride).contiguous();
  const torch::Tensor source_conv = inputs.conv_state[0].clone();
  const torch::Tensor source_ssm =
      inputs.ssm_state.narrow(0, 0, kCheckpointStride).clone();
  const auto index_options = torch::TensorOptions().dtype(torch::kInt32);
  auto conv_read_indices =
      torch::tensor({int32_t{0}}, index_options).to(torch::kPrivateUse1);
  auto conv_write_indices =
      torch::tensor({int32_t{1}}, index_options).to(torch::kPrivateUse1);
  auto ssm_read_indices =
      torch::tensor({int32_t{0}}, index_options).to(torch::kPrivateUse1);
  auto ssm_write_indices =
      torch::tensor({static_cast<int32_t>(kCheckpointStride)}, index_options)
          .to(torch::kPrivateUse1);
  auto cu_seqlens =
      torch::tensor({int32_t{0}, static_cast<int32_t>(kSequenceLength)},
                    index_options)
          .to(torch::kPrivateUse1);
  torch::Tensor packed_qkv = inputs.qkv.squeeze(0);
  torch::Tensor packed_b = inputs.b.squeeze(0);
  torch::Tensor packed_a = inputs.a.squeeze(0);
  torch::Tensor packed_z = inputs.z.squeeze(0);

  const auto outputs =
      xllm::kernel::npu::npu_mega_gdn_prefill(packed_qkv,
                                              packed_b,
                                              packed_a,
                                              packed_z,
                                              inputs.conv_weight,
                                              inputs.conv_state,
                                              inputs.a_log,
                                              inputs.dt_bias,
                                              conv_read_indices,
                                              conv_write_indices,
                                              ssm_read_indices,
                                              ssm_write_indices,
                                              inputs.ssm_state,
                                              cu_seqlens,
                                              inputs.norm_weight,
                                              /*num_matrices=*/1);

  EXPECT_TRUE(torch::equal(inputs.conv_state[0].cpu(), source_conv.cpu()));
  EXPECT_TRUE(
      torch::equal(inputs.ssm_state.narrow(0, 0, kCheckpointStride).cpu(),
                   source_ssm.cpu()));
  EXPECT_FALSE(torch::equal(inputs.conv_state[1].cpu(), source_conv.cpu()));
  EXPECT_FALSE(torch::equal(
      inputs.ssm_state.narrow(0, kCheckpointStride, kCheckpointStride).cpu(),
      source_ssm.cpu()));
  EXPECT_TRUE(torch::isfinite(std::get<0>(outputs)).all().item<bool>());
}

TEST_F(NpuXllmOpsTest, MegaGdnDecodeSameSlotKeepsD2DState) {
  py::gil_scoped_acquire gil;
  MegaGdnInputs inputs = make_mega_gdn_inputs(/*sequence_length=*/1,
                                              /*same_slot=*/true);

  const torch::Tensor initial_conv_state = inputs.conv_state.clone();
  const auto outputs =
      xllm::kernel::npu::npu_mega_gdn_decode(inputs.qkv.squeeze(1),
                                             inputs.z.squeeze(1),
                                             inputs.b.squeeze(1),
                                             inputs.a.squeeze(1),
                                             inputs.conv_weight,
                                             inputs.conv_state,
                                             inputs.a_log,
                                             inputs.dt_bias,
                                             inputs.ssm_state,
                                             inputs.read_indices,
                                             inputs.write_indices,
                                             inputs.norm_weight,
                                             /*fla_ssm_state_layout=*/true);

  EXPECT_EQ(std::get<0>(outputs).sizes(), inputs.qkv.squeeze(1).sizes());
  EXPECT_EQ(std::get<3>(outputs).sizes(), inputs.z.squeeze(1).sizes());
  expect_same_storage(std::get<1>(outputs), inputs.conv_state);
  expect_same_storage(std::get<2>(outputs), inputs.ssm_state);
  EXPECT_TRUE(torch::equal(inputs.conv_state[0][0].cpu(),
                           initial_conv_state[0][1].cpu()));
  EXPECT_TRUE(torch::equal(inputs.conv_state[0][1].cpu(),
                           initial_conv_state[0][2].cpu()));
  EXPECT_TRUE(
      torch::equal(inputs.conv_state[0][2].cpu(), inputs.qkv[0][0].cpu()));
  EXPECT_TRUE(torch::isfinite(std::get<3>(outputs)).all().item<bool>());
}

TEST_F(NpuXllmOpsTest, MegaGdnMtpSameSlotAcceptedBoundariesKeepD2DState) {
  py::gil_scoped_acquire gil;
  constexpr int64_t kSequenceLength = 5;
  for (const int32_t accepted : {1, static_cast<int32_t>(kSequenceLength)}) {
    SCOPED_TRACE(::testing::Message() << "accepted=" << accepted);
    MegaGdnInputs inputs = make_mega_gdn_inputs(kSequenceLength,
                                                /*same_slot=*/true);
    const auto accepted_tokens =
        torch::tensor({accepted}, torch::TensorOptions().dtype(torch::kInt32))
            .to(torch::kPrivateUse1);

    const torch::Tensor initial_conv_state = inputs.conv_state.clone();
    const auto outputs = xllm::kernel::npu::npu_mega_gdn_mtp_decode(
        inputs.qkv,
        inputs.z,
        inputs.b,
        inputs.a,
        inputs.conv_weight,
        inputs.conv_state,
        inputs.a_log,
        inputs.dt_bias,
        inputs.ssm_state,
        inputs.read_indices,
        inputs.write_indices,
        accepted_tokens,
        inputs.norm_weight,
        /*fla_ssm_state_layout=*/true);

    EXPECT_EQ(std::get<0>(outputs).sizes(), inputs.qkv.sizes());
    EXPECT_EQ(std::get<3>(outputs).sizes(), inputs.z.sizes());
    expect_same_storage(std::get<1>(outputs), inputs.conv_state);
    expect_same_storage(std::get<2>(outputs), inputs.ssm_state);
    EXPECT_TRUE(torch::equal(
        inputs.conv_state[0].slice(/*dim=*/0, 0, 2).cpu(),
        initial_conv_state[0].slice(/*dim=*/0, accepted, accepted + 2).cpu()));
    EXPECT_TRUE(torch::equal(
        inputs.conv_state[0].slice(/*dim=*/0, 2, kSequenceLength + 2).cpu(),
        inputs.qkv[0].cpu()));
    EXPECT_TRUE(torch::isfinite(std::get<3>(outputs)).all().item<bool>());
  }
}

}  // namespace
}  // namespace xllm
