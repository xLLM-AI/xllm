/* Copyright 2025-2026 The xLLM Authors.

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

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_FUSED_GDN_GATING_REGISTRY_INC
#error "XLLM_TL_FUSED_GDN_GATING_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kCompileMaxBatch = 262144;
constexpr int64_t kCompileMaxHeads = 128;
constexpr int32_t kBatchSpecializationMin = 2;
constexpr int32_t kBatchSpecializationStep = 2;

#include XLLM_TL_FUSED_GDN_GATING_REGISTRY_INC

int32_t max_compiled_batch_size(int32_t num_heads, TilelangDType dtype) {
  int32_t max_batch_size = 0;
  for (const auto& entry : kFusedGdnGatingRegistry) {
    const auto& spec = entry.spec;
    if (spec.num_heads == num_heads && spec.dtype == dtype) {
      max_batch_size = std::max(max_batch_size, spec.batch_size);
    }
  }
  return max_batch_size;
}

int32_t select_launch_batch_size(int64_t num_batches,
                                 int32_t num_heads,
                                 TilelangDType dtype) {
  CHECK_GT(num_batches, 0)
      << "TileLang fused_gdn_gating: num_batches must be > 0";
  const int32_t max_batch_size = max_compiled_batch_size(num_heads, dtype);
  CHECK_GT(max_batch_size, 0)
      << "TileLang fused_gdn_gating: no compiled batch_size variant for "
      << "num_heads=" << num_heads << ", dtype=" << static_cast<int>(dtype);
  CHECK_GE(max_batch_size, kBatchSpecializationMin)
      << "TileLang fused_gdn_gating: compiled batch_size variants must be >= "
      << kBatchSpecializationMin;

  const int64_t capped = std::min<int64_t>(num_batches, max_batch_size);
  int64_t rounded_up_even =
      ((capped + kBatchSpecializationStep - 1) / kBatchSpecializationStep) *
      kBatchSpecializationStep;
  rounded_up_even = std::max<int64_t>(rounded_up_even, kBatchSpecializationMin);
  rounded_up_even = std::min<int64_t>(rounded_up_even, max_batch_size);
  if ((rounded_up_even % kBatchSpecializationStep) != 0) {
    rounded_up_even -= 1;
  }
  return static_cast<int32_t>(rounded_up_even);
}

void check_supported(const torch::Tensor& A_log,
                     const torch::Tensor& a,
                     const torch::Tensor& b,
                     const torch::Tensor& dt_bias) {
  CHECK(A_log.defined()) << "TileLang fused_gdn_gating: A_log must be defined";
  CHECK(a.defined()) << "TileLang fused_gdn_gating: a must be defined";
  CHECK(b.defined()) << "TileLang fused_gdn_gating: b must be defined";
  CHECK(dt_bias.defined())
      << "TileLang fused_gdn_gating: dt_bias must be defined";

  CHECK(A_log.device().type() == c10::DeviceType::PrivateUse1 &&
        a.device().type() == c10::DeviceType::PrivateUse1 &&
        b.device().type() == c10::DeviceType::PrivateUse1 &&
        dt_bias.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang fused_gdn_gating: all tensors must be on NPU";

  CHECK_EQ(A_log.dim(), 1)
      << "TileLang fused_gdn_gating: A_log must be 1D [num_heads]";
  CHECK_EQ(dt_bias.dim(), 1)
      << "TileLang fused_gdn_gating: dt_bias must be 1D [num_heads]";
  CHECK_EQ(a.dim(), 2) << "TileLang fused_gdn_gating: a must be 2D [B, H]";
  CHECK_EQ(b.dim(), 2) << "TileLang fused_gdn_gating: b must be 2D [B, H]";
  CHECK_EQ(a.sizes(), b.sizes())
      << "TileLang fused_gdn_gating: a/b shape mismatch";
  CHECK_EQ(A_log.size(0), a.size(1))
      << "TileLang fused_gdn_gating: A_log head size mismatch";
  CHECK_EQ(dt_bias.size(0), a.size(1))
      << "TileLang fused_gdn_gating: dt_bias head size mismatch";
  CHECK_GT(a.size(1), 0) << "TileLang fused_gdn_gating: num_heads must be > 0";
  CHECK_LE(a.size(1), kCompileMaxHeads)
      << "TileLang fused_gdn_gating: num_heads must be <= " << kCompileMaxHeads
      << ", got " << a.size(1);

  CHECK_EQ(A_log.dtype(), torch::kFloat32)
      << "TileLang fused_gdn_gating: A_log must be float32";
  CHECK_EQ(dt_bias.dtype(), torch::kFloat32)
      << "TileLang fused_gdn_gating: dt_bias must be float32";
  CHECK_EQ(a.dtype(), b.dtype())
      << "TileLang fused_gdn_gating: a/b dtype mismatch";
  CHECK_EQ(a.dtype(), torch::kBFloat16)
      << "TileLang fused_gdn_gating: only bf16 inputs are supported";

  CHECK(A_log.is_contiguous())
      << "TileLang fused_gdn_gating: A_log must be contiguous";
  CHECK(dt_bias.is_contiguous())
      << "TileLang fused_gdn_gating: dt_bias must be contiguous";
  CHECK_EQ(a.stride(1), 1)
      << "TileLang fused_gdn_gating: a last-dim stride must be 1";
  CHECK_EQ(b.stride(1), 1)
      << "TileLang fused_gdn_gating: b last-dim stride must be 1";
  CHECK_EQ(a.stride(0), a.size(1))
      << "TileLang fused_gdn_gating: a must be row-contiguous";
  CHECK_EQ(b.stride(0), b.size(1))
      << "TileLang fused_gdn_gating: b must be row-contiguous";
}

FusedGdnGatingSpecialization build_runtime_specialization(
    const torch::Tensor& a) {
  CHECK_EQ(a.dim(), 2) << "TileLang fused_gdn_gating: a must be 2D";
  const TilelangDType dtype = to_tilelang_dtype(a.scalar_type());
  const int32_t num_heads = static_cast<int32_t>(a.size(1));
  const int32_t batch_size =
      select_launch_batch_size(a.size(0), num_heads, dtype);
  return make_fused_gdn_gating_specialization(
      FusedGdnGatingBatchSize{batch_size},
      FusedGdnGatingNumHeads{num_heads},
      FusedGdnGatingDType{dtype});
}

void run_tilelang_fused_gdn_gating_chunk(const torch::Tensor& A_log,
                                         const torch::Tensor& a,
                                         const torch::Tensor& b,
                                         const torch::Tensor& dt_bias,
                                         torch::Tensor& g_out,
                                         torch::Tensor& beta_out,
                                         float softplus_beta,
                                         float softplus_threshold) {
  CHECK_EQ(a.dim(), 2) << "TileLang fused_gdn_gating: a must be 2D";
  CHECK_EQ(b.dim(), 2) << "TileLang fused_gdn_gating: b must be 2D";
  CHECK_EQ(g_out.dim(), 3)
      << "TileLang fused_gdn_gating: g_out must be 3D [1, B, H]";
  CHECK_EQ(beta_out.dim(), 3)
      << "TileLang fused_gdn_gating: beta_out must be 3D [1, B, H]";
  CHECK_EQ(g_out.size(0), 1)
      << "TileLang fused_gdn_gating: g_out first dim must be 1";
  CHECK_EQ(beta_out.size(0), 1)
      << "TileLang fused_gdn_gating: beta_out first dim must be 1";
  CHECK_EQ(g_out.size(1), a.size(0))
      << "TileLang fused_gdn_gating: g_out batch mismatch";
  CHECK_EQ(beta_out.size(1), a.size(0))
      << "TileLang fused_gdn_gating: beta_out batch mismatch";
  CHECK_EQ(g_out.size(2), a.size(1))
      << "TileLang fused_gdn_gating: g_out head mismatch";
  CHECK_EQ(beta_out.size(2), a.size(1))
      << "TileLang fused_gdn_gating: beta_out head mismatch";
  CHECK_EQ(g_out.dtype(), torch::kFloat32)
      << "TileLang fused_gdn_gating: g_out must be float32";
  CHECK_EQ(beta_out.dtype(), a.dtype())
      << "TileLang fused_gdn_gating: beta_out dtype mismatch";
  CHECK_EQ(g_out.stride(2), 1)
      << "TileLang fused_gdn_gating: g_out last-dim stride must be 1";
  CHECK_EQ(beta_out.stride(2), 1)
      << "TileLang fused_gdn_gating: beta_out last-dim stride must be 1";
  CHECK_GT(g_out.stride(1), 0)
      << "TileLang fused_gdn_gating: g_out row stride must be > 0";
  CHECK_GT(beta_out.stride(1), 0)
      << "TileLang fused_gdn_gating: beta_out row stride must be > 0";
  CHECK_LE(a.size(0), kCompileMaxBatch)
      << "TileLang fused_gdn_gating: chunk batch exceeds compile limit "
      << kCompileMaxBatch;
  CHECK_GT(softplus_beta, 0.0F)
      << "TileLang fused_gdn_gating: softplus_beta must be > 0";

  auto specialization = build_runtime_specialization(a);
  const auto* entry = find_fused_gdn_gating_kernel_entry(specialization);
  // Expected fast path: compiled batch_size variants are dense [2, 4, ..., 48].
  // If a value is missing, fall back to the nearest smaller batch_size.
  if (entry == nullptr) {
    int32_t fallback_batch_size =
        specialization.batch_size - kBatchSpecializationStep;
    while (fallback_batch_size >= kBatchSpecializationMin && entry == nullptr) {
      specialization = make_fused_gdn_gating_specialization(
          FusedGdnGatingBatchSize{fallback_batch_size},
          FusedGdnGatingNumHeads{specialization.num_heads},
          FusedGdnGatingDType{specialization.dtype});
      entry = find_fused_gdn_gating_kernel_entry(specialization);
      fallback_batch_size -= kBatchSpecializationStep;
    }
  }
  CHECK(entry != nullptr)
      << "TileLang fused_gdn_gating: no compiled variant. Available variants: "
      << available_fused_gdn_gating_variant_keys();

  const int64_t num_batches = a.size(0);

  const int32_t device_id = a.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  auto g_rows = g_out.squeeze(0);
  auto beta_rows = beta_out.squeeze(0);
  entry->fn(static_cast<uint8_t*>(A_log.data_ptr()),
            static_cast<uint8_t*>(a.data_ptr()),
            static_cast<uint8_t*>(b.data_ptr()),
            static_cast<uint8_t*>(dt_bias.data_ptr()),
            static_cast<uint8_t*>(g_rows.data_ptr()),
            static_cast<uint8_t*>(beta_rows.data_ptr()),
            static_cast<int32_t>(num_batches),
            softplus_beta,
            softplus_threshold,
            stream);
}

}  // namespace

std::pair<torch::Tensor, torch::Tensor> fused_gdn_gating(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& dt_bias,
    float softplus_beta,
    float softplus_threshold) {
  check_supported(A_log, a, b, dt_bias);

  const auto num_batches = a.size(0);
  const auto num_heads = a.size(1);
  auto g_out = torch::empty({1, num_batches, num_heads},
                            a.options().dtype(torch::kFloat32));
  auto beta_out = torch::empty({1, num_batches, num_heads}, a.options());

  for (int64_t start = 0; start < num_batches; start += kCompileMaxBatch) {
    const int64_t chunk_batches =
        std::min(kCompileMaxBatch, num_batches - start);
    auto a_chunk = a.narrow(0, start, chunk_batches);
    auto b_chunk = b.narrow(0, start, chunk_batches);
    auto g_chunk = g_out.narrow(1, start, chunk_batches);
    auto beta_chunk = beta_out.narrow(1, start, chunk_batches);
    run_tilelang_fused_gdn_gating_chunk(A_log,
                                        a_chunk,
                                        b_chunk,
                                        dt_bias,
                                        g_chunk,
                                        beta_chunk,
                                        softplus_beta,
                                        softplus_threshold);
  }

  return {g_out, beta_out};
}

}  // namespace xllm::kernel::npu::tilelang
