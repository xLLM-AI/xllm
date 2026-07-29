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

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_SPLIT_QKV_RMSNORM_MROPE_REGISTRY_INC
#error "XLLM_TL_SPLIT_QKV_RMSNORM_MROPE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int32_t kTokenSpecializationMin = 2;
constexpr int32_t kTokenSpecializationStep = 2;

#include XLLM_TL_SPLIT_QKV_RMSNORM_MROPE_REGISTRY_INC

int32_t max_compiled_num_tokens(int32_t head_size,
                                int32_t rope_dim,
                                int32_t num_q_heads,
                                int32_t num_kv_heads,
                                TilelangDType dtype) {
  int32_t max_num_tokens = 0;
  for (const auto& entry : kSplitQkvRmsnormMropeRegistry) {
    const auto& spec = entry.spec;
    if (spec.head_size == head_size && spec.rope_dim == rope_dim &&
        spec.num_q_heads == num_q_heads && spec.num_kv_heads == num_kv_heads &&
        spec.dtype == dtype) {
      max_num_tokens = std::max(max_num_tokens, spec.num_tokens);
    }
  }
  return max_num_tokens;
}

int32_t select_launch_num_tokens(int64_t num_tokens,
                                 int32_t head_size,
                                 int32_t rope_dim,
                                 int32_t num_q_heads,
                                 int32_t num_kv_heads,
                                 TilelangDType dtype) {
  CHECK_GT(num_tokens, 0)
      << "TileLang split_qkv_rmsnorm_mrope: num_tokens must be > 0";
  const int32_t max_num_tokens = max_compiled_num_tokens(
      head_size, rope_dim, num_q_heads, num_kv_heads, dtype);
  CHECK_GT(max_num_tokens, 0)
      << "TileLang split_qkv_rmsnorm_mrope: no compiled num_tokens variant for "
      << "head_size=" << head_size << ", rope_dim=" << rope_dim
      << ", num_q_heads=" << num_q_heads << ", num_kv_heads=" << num_kv_heads
      << ", dtype=" << static_cast<int>(dtype);
  CHECK_GE(max_num_tokens, kTokenSpecializationMin)
      << "TileLang split_qkv_rmsnorm_mrope: compiled num_tokens variants must "
      << "be >= " << kTokenSpecializationMin;

  const int64_t capped = std::min<int64_t>(num_tokens, max_num_tokens);
  int64_t rounded_up =
      ((capped + kTokenSpecializationStep - 1) / kTokenSpecializationStep) *
      kTokenSpecializationStep;
  rounded_up = std::max<int64_t>(rounded_up, kTokenSpecializationMin);
  rounded_up = std::min<int64_t>(rounded_up, max_num_tokens);
  if ((rounded_up % kTokenSpecializationStep) != 0) {
    rounded_up -= 1;
  }
  return static_cast<int32_t>(rounded_up);
}

void check_supported(const torch::Tensor& qkvg,
                     const torch::Tensor& q_weight,
                     const torch::Tensor& k_weight,
                     const torch::Tensor& cos_sin,
                     const torch::Tensor& gather_pattern,
                     float eps,
                     int64_t num_q_heads,
                     int64_t num_kv_heads,
                     int64_t head_size) {
  CHECK(qkvg.defined())
      << "TileLang split_qkv_rmsnorm_mrope: qkvg must be defined";
  CHECK(q_weight.defined())
      << "TileLang split_qkv_rmsnorm_mrope: q_weight must be defined";
  CHECK(k_weight.defined())
      << "TileLang split_qkv_rmsnorm_mrope: k_weight must be defined";
  CHECK(cos_sin.defined())
      << "TileLang split_qkv_rmsnorm_mrope: cos_sin must be defined";
  CHECK(gather_pattern.defined())
      << "TileLang split_qkv_rmsnorm_mrope: gather_pattern must be defined";

  CHECK(qkvg.device().type() == c10::DeviceType::PrivateUse1 &&
        q_weight.device().type() == c10::DeviceType::PrivateUse1 &&
        k_weight.device().type() == c10::DeviceType::PrivateUse1 &&
        cos_sin.device().type() == c10::DeviceType::PrivateUse1 &&
        gather_pattern.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang split_qkv_rmsnorm_mrope: all tensors must be on NPU";

  CHECK_EQ(qkvg.dim(), 2)
      << "TileLang split_qkv_rmsnorm_mrope: qkvg must be 2D [T, Q|G|K|V]";
  CHECK_EQ(q_weight.dim(), 1)
      << "TileLang split_qkv_rmsnorm_mrope: q_weight must be 1D [head_size]";
  CHECK_EQ(k_weight.dim(), 1)
      << "TileLang split_qkv_rmsnorm_mrope: k_weight must be 1D [head_size]";
  CHECK_EQ(cos_sin.dim(), 2)
      << "TileLang split_qkv_rmsnorm_mrope: cos_sin must be 2D "
      << "[T, 3 * rope_dim]";
  CHECK_EQ(gather_pattern.dim(), 1)
      << "TileLang split_qkv_rmsnorm_mrope: gather_pattern must be 1D";

  CHECK_EQ(qkvg.dtype(), torch::kBFloat16)
      << "TileLang split_qkv_rmsnorm_mrope: only bf16 qkvg is supported";
  CHECK_EQ(q_weight.dtype(), qkvg.dtype())
      << "TileLang split_qkv_rmsnorm_mrope: q_weight dtype mismatch";
  CHECK_EQ(k_weight.dtype(), qkvg.dtype())
      << "TileLang split_qkv_rmsnorm_mrope: k_weight dtype mismatch";
  CHECK_EQ(cos_sin.dtype(), qkvg.dtype())
      << "TileLang split_qkv_rmsnorm_mrope: cos_sin dtype mismatch";
  CHECK_EQ(gather_pattern.dtype(), torch::kUInt32)
      << "TileLang split_qkv_rmsnorm_mrope: gather_pattern must be uint32";
  [[maybe_unused]] TilelangDType dtype = to_tilelang_dtype(qkvg.scalar_type());

  CHECK_EQ(qkvg.stride(1), 1)
      << "TileLang split_qkv_rmsnorm_mrope: qkvg last-dim stride must be 1";
  CHECK_EQ(cos_sin.stride(1), 1)
      << "TileLang split_qkv_rmsnorm_mrope: cos_sin last-dim stride must be 1";
  CHECK_EQ(gather_pattern.stride(0), 1)
      << "TileLang split_qkv_rmsnorm_mrope: gather_pattern stride must be 1";

  CHECK_EQ(static_cast<int64_t>(q_weight.numel()), head_size)
      << "TileLang split_qkv_rmsnorm_mrope: q_weight size must match "
      << "head_size";
  CHECK_EQ(static_cast<int64_t>(k_weight.numel()), head_size)
      << "TileLang split_qkv_rmsnorm_mrope: k_weight size must match "
      << "head_size";
  CHECK_GT(eps, 0.0F) << "TileLang split_qkv_rmsnorm_mrope: eps must be > 0";
  CHECK_GT(num_q_heads, 0)
      << "TileLang split_qkv_rmsnorm_mrope: num_q_heads must be > 0";
  CHECK_GT(num_kv_heads, 0)
      << "TileLang split_qkv_rmsnorm_mrope: num_kv_heads must be > 0";
  CHECK_EQ(head_size, 256)
      << "TileLang split_qkv_rmsnorm_mrope: only head_size=256 is supported";
  CHECK_EQ(cos_sin.size(1), 192)
      << "TileLang split_qkv_rmsnorm_mrope: only rope_dim=64 is supported";
  CHECK_EQ(cos_sin.size(0), qkvg.size(0))
      << "TileLang split_qkv_rmsnorm_mrope: cos_sin token dim mismatch";

  const int64_t q_size = num_q_heads * head_size;
  const int64_t kv_size = num_kv_heads * head_size;
  CHECK_EQ(qkvg.size(1), q_size * 2 + kv_size * 2)
      << "TileLang split_qkv_rmsnorm_mrope: qkvg width mismatch for "
      << "[Q|G|K|V] layout";
}

SplitQkvRmsnormMropeSpecialization build_runtime_specialization(
    int64_t num_tokens,
    int64_t head_size,
    int64_t rope_dim,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    c10::ScalarType dtype) {
  CHECK_LE(head_size, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang split_qkv_rmsnorm_mrope: head_size exceeds int32";
  CHECK_LE(rope_dim, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang split_qkv_rmsnorm_mrope: rope_dim exceeds int32";
  CHECK_LE(num_q_heads,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang split_qkv_rmsnorm_mrope: num_q_heads exceeds int32";
  CHECK_LE(num_kv_heads,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang split_qkv_rmsnorm_mrope: num_kv_heads exceeds int32";
  const TilelangDType tilelang_dtype = to_tilelang_dtype(dtype);
  const int32_t selected_num_tokens =
      select_launch_num_tokens(num_tokens,
                               static_cast<int32_t>(head_size),
                               static_cast<int32_t>(rope_dim),
                               static_cast<int32_t>(num_q_heads),
                               static_cast<int32_t>(num_kv_heads),
                               tilelang_dtype);

  return make_split_qkv_rmsnorm_mrope_specialization(
      SplitQkvRmsnormMropeHeadSize{static_cast<int32_t>(head_size)},
      SplitQkvRmsnormMropeRopeDim{static_cast<int32_t>(rope_dim)},
      SplitQkvRmsnormMropeNumTokens{selected_num_tokens},
      SplitQkvRmsnormMropeNumQHeads{static_cast<int32_t>(num_q_heads)},
      SplitQkvRmsnormMropeNumKvHeads{static_cast<int32_t>(num_kv_heads)},
      SplitQkvRmsnormMropeDType{tilelang_dtype});
}

torch::Tensor build_mrope_gather_pattern_merged(
    int64_t rope_dim,
    const std::vector<int64_t>& mrope_section,
    bool is_interleaved,
    const torch::Device& device) {
  CHECK_EQ(rope_dim % 2, 0)
      << "TileLang split_qkv_rmsnorm_mrope: rope_dim must be even";
  const int64_t half_rope_dim = rope_dim / 2;
  const int64_t gather_pad_dim = ((3 * rope_dim + 127) / 128) * 128;
  const auto cpu_options =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
  torch::Tensor pattern = torch::zeros({gather_pad_dim}, cpu_options);
  int32_t* pattern_ptr = pattern.data_ptr<int32_t>();

  std::vector<int32_t> axis_id(half_rope_dim, 0);
  const int64_t t_len = mrope_section[0];
  const int64_t h_len = mrope_section[1];
  const int64_t w_len = mrope_section[2];
  if (is_interleaved) {
    for (int64_t i = 0; i < half_rope_dim; ++i) {
      if ((i % 3) == 1 && i <= 3 * h_len) {
        axis_id[i] = 1;
      } else if ((i % 3) == 2 && i <= 3 * w_len) {
        axis_id[i] = 2;
      }
    }
  } else {
    const int64_t t_end = t_len;
    const int64_t h_end = t_end + h_len;
    for (int64_t i = 0; i < half_rope_dim; ++i) {
      if (i >= h_end) {
        axis_id[i] = 2;
      } else if (i >= t_end) {
        axis_id[i] = 1;
      }
    }
  }

  for (int64_t i = 0; i < half_rope_dim; ++i) {
    pattern_ptr[i] = static_cast<int32_t>(axis_id[i] * rope_dim + i);
    pattern_ptr[half_rope_dim + i] =
        static_cast<int32_t>(axis_id[i] * rope_dim + half_rope_dim + i);
  }

  torch::Tensor device_pattern =
      pattern.to(torch::TensorOptions().device(device).dtype(torch::kInt32));
  return device_pattern.view(torch::kUInt32);
}

const auto* find_kernel_entry_with_fallback(
    SplitQkvRmsnormMropeSpecialization specialization) {
  const auto* entry = find_split_qkv_rmsnorm_mrope_kernel_entry(specialization);
  if (entry != nullptr) {
    return entry;
  }
  int32_t fallback_num_tokens =
      specialization.num_tokens - kTokenSpecializationStep;
  while (fallback_num_tokens >= kTokenSpecializationMin && entry == nullptr) {
    specialization = make_split_qkv_rmsnorm_mrope_specialization(
        SplitQkvRmsnormMropeHeadSize{specialization.head_size},
        SplitQkvRmsnormMropeRopeDim{specialization.rope_dim},
        SplitQkvRmsnormMropeNumTokens{fallback_num_tokens},
        SplitQkvRmsnormMropeNumQHeads{specialization.num_q_heads},
        SplitQkvRmsnormMropeNumKvHeads{specialization.num_kv_heads},
        SplitQkvRmsnormMropeDType{specialization.dtype});
    entry = find_split_qkv_rmsnorm_mrope_kernel_entry(specialization);
    fallback_num_tokens -= kTokenSpecializationStep;
  }
  return entry;
}

}  // namespace

bool has_split_qkv_rmsnorm_mrope_specialization(int64_t num_q_heads,
                                                int64_t num_kv_heads,
                                                int64_t head_size) {
  return max_compiled_num_tokens(static_cast<int32_t>(head_size),
                                 /*rope_dim=*/64,
                                 static_cast<int32_t>(num_q_heads),
                                 static_cast<int32_t>(num_kv_heads),
                                 TilelangDType::kBF16) > 0;
}

torch::Tensor build_split_qkv_rmsnorm_mrope_gather_pattern(
    int64_t rope_dim,
    const std::vector<int64_t>& mrope_section,
    bool is_interleaved,
    const torch::Device& device) {
  return build_mrope_gather_pattern_merged(
      rope_dim, mrope_section, is_interleaved, device);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
split_qkv_rmsnorm_mrope(const torch::Tensor& qkvg,
                        const torch::Tensor& q_weight,
                        const torch::Tensor& k_weight,
                        const torch::Tensor& cos_sin,
                        const torch::Tensor& gather_pattern,
                        float eps,
                        int64_t num_q_heads,
                        int64_t num_kv_heads,
                        int64_t head_size) {
  check_supported(qkvg,
                  q_weight,
                  k_weight,
                  cos_sin,
                  gather_pattern,
                  eps,
                  num_q_heads,
                  num_kv_heads,
                  head_size);

  const int64_t num_tokens = qkvg.size(0);
  const int64_t q_width = num_q_heads * head_size;
  const int64_t kv_width = num_kv_heads * head_size;
  const int64_t rope_dim = cos_sin.size(1) / 3;

  torch::Tensor q_out_flat =
      torch::empty({num_tokens, q_width}, qkvg.options());
  torch::Tensor k_out_flat =
      torch::empty({num_tokens, kv_width}, qkvg.options());
  torch::Tensor v_out_flat =
      torch::empty({num_tokens, kv_width}, qkvg.options());
  torch::Tensor gate_out_flat =
      torch::empty({num_tokens, q_width}, qkvg.options());

  auto specialization = build_runtime_specialization(num_tokens,
                                                     head_size,
                                                     rope_dim,
                                                     num_q_heads,
                                                     num_kv_heads,
                                                     qkvg.scalar_type());
  const auto* entry = find_kernel_entry_with_fallback(specialization);
  CHECK(entry != nullptr)
      << "TileLang split_qkv_rmsnorm_mrope: no compiled variant. Available "
      << "variants: " << available_split_qkv_rmsnorm_mrope_variant_keys();
  CHECK_LE(num_tokens,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang split_qkv_rmsnorm_mrope: num_tokens exceeds int32";

  const int32_t device_id = qkvg.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  entry->fn(static_cast<uint8_t*>(qkvg.data_ptr()),
            static_cast<uint8_t*>(q_weight.data_ptr()),
            static_cast<uint8_t*>(k_weight.data_ptr()),
            static_cast<uint8_t*>(cos_sin.data_ptr()),
            reinterpret_cast<uint8_t*>(gather_pattern.data_ptr()),
            static_cast<uint8_t*>(q_out_flat.data_ptr()),
            static_cast<uint8_t*>(k_out_flat.data_ptr()),
            static_cast<uint8_t*>(v_out_flat.data_ptr()),
            static_cast<uint8_t*>(gate_out_flat.data_ptr()),
            static_cast<int32_t>(num_tokens),
            eps,
            stream);

  return {q_out_flat.view({num_tokens, num_q_heads, head_size}),
          k_out_flat.view({num_tokens, num_kv_heads, head_size}),
          v_out_flat.view({num_tokens, num_kv_heads, head_size}),
          gate_out_flat.view({num_tokens, num_q_heads, head_size})};
}

}  // namespace xllm::kernel::npu::tilelang
