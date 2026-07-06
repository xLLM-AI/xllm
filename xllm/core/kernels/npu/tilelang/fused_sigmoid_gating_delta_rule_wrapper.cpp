/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <tuple>
#include <utility>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_FUSED_SIGMOID_GATING_DELTA_RULE_REGISTRY_INC
#error "XLLM_TL_FUSED_SIGMOID_GATING_DELTA_RULE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int32_t kVEC_NUM = 2;
constexpr int32_t kNSpecializationMin = 1;
constexpr int32_t kNSpecializationStep = 1;
constexpr int32_t kUseQkL2norm = 1;

#include XLLM_TL_FUSED_SIGMOID_GATING_DELTA_RULE_REGISTRY_INC

int32_t auto_block_v(int32_t dv) {
  int32_t block_v = std::min(dv, 128);
  if (block_v >= 32) {
    block_v = (block_v / 32) * 32;
  }
  if (block_v < kVEC_NUM) {
    block_v = (dv >= kVEC_NUM) ? kVEC_NUM : dv;
  }
  if (block_v % kVEC_NUM != 0) {
    block_v = (block_v / kVEC_NUM) * kVEC_NUM;
  }
  while (block_v > 0 && dv % block_v != 0) {
    block_v -= kVEC_NUM;
  }
  if (block_v <= 0) {
    block_v = std::min(dv, kVEC_NUM);
    if (block_v % kVEC_NUM != 0) {
      block_v = (block_v / kVEC_NUM) * kVEC_NUM;
    }
  }
  CHECK_GT(block_v, 0) << "TileLang fused_sigmoid_gating_delta_rule: "
                       << "could not auto-derive block_v for dv=" << dv;
  CHECK_LE(block_v, dv);
  CHECK_EQ(block_v % kVEC_NUM, 0);
  CHECK_EQ(dv % block_v, 0);
  return block_v;
}

int32_t max_compiled_num_seqs(int32_t nk,
                              int32_t nv,
                              int32_t dk,
                              int32_t dv,
                              int32_t block_v,
                              TilelangDType dtype) {
  int32_t max_n = 0;
  for (const auto& entry : kFusedSigmoidGatingDeltaRuleRegistry) {
    const auto& spec = entry.spec;
    if (spec.nk == nk && spec.nv == nv && spec.dk == dk && spec.dv == dv &&
        spec.block_v == block_v && spec.use_qk_l2norm == kUseQkL2norm &&
        spec.dtype == dtype) {
      max_n = std::max(max_n, spec.max_num_seqs);
    }
  }
  return max_n;
}

int32_t select_launch_num_seqs(int64_t actual_n,
                               int32_t nk,
                               int32_t nv,
                               int32_t dk,
                               int32_t dv,
                               int32_t block_v,
                               TilelangDType dtype) {
  CHECK_GT(actual_n, 0)
      << "TileLang fused_sigmoid_gating_delta_rule: actual_n must be > 0";
  const int32_t max_n = max_compiled_num_seqs(nk, nv, dk, dv, block_v, dtype);
  CHECK_GT(max_n, 0)
      << "TileLang fused_sigmoid_gating_delta_rule: no compiled num_seqs "
      << "variant for nk=" << nk << ", nv=" << nv << ", dk=" << dk
      << ", dv=" << dv << ", dtype=" << static_cast<int>(dtype);
  CHECK_GE(max_n, kNSpecializationMin)
      << "TileLang fused_sigmoid_gating_delta_rule: compiled num_seqs "
      << "variants must be >= " << kNSpecializationMin;
  CHECK_LE(actual_n, static_cast<int64_t>(max_n))
      << "TileLang fused_sigmoid_gating_delta_rule: actual_n=" << actual_n
      << " exceeds max compiled num_seqs=" << max_n;

  int64_t rounded_up =
      ((actual_n + kNSpecializationStep - 1) / kNSpecializationStep) *
      kNSpecializationStep;
  rounded_up = std::max<int64_t>(rounded_up, kNSpecializationMin);
  rounded_up = std::min<int64_t>(rounded_up, max_n);
  return static_cast<int32_t>(rounded_up);
}

void check_supported(const torch::Tensor& A_log,
                     const torch::Tensor& a,
                     const torch::Tensor& dt_bias,
                     const torch::Tensor& query,
                     const torch::Tensor& key,
                     const torch::Tensor& value,
                     const torch::Tensor& beta,
                     const torch::Tensor& init_state,
                     const torch::Tensor& ssm_state_indices,
                     const torch::Tensor& cu_seqlens) {
  CHECK(A_log.defined())
      << "TileLang fused_sigmoid_gating_delta_rule: A_log must be defined";
  CHECK(a.defined())
      << "TileLang fused_sigmoid_gating_delta_rule: a must be defined";
  CHECK(dt_bias.defined())
      << "TileLang fused_sigmoid_gating_delta_rule: dt_bias must be defined";
  CHECK(query.defined())
      << "TileLang fused_sigmoid_gating_delta_rule: query must be defined";
  CHECK(key.defined())
      << "TileLang fused_sigmoid_gating_delta_rule: key must be defined";
  CHECK(value.defined())
      << "TileLang fused_sigmoid_gating_delta_rule: value must be defined";
  CHECK(beta.defined())
      << "TileLang fused_sigmoid_gating_delta_rule: beta must be defined";
  CHECK(init_state.defined())
      << "TileLang fused_sigmoid_gating_delta_rule: init_state must be defined";
  CHECK(ssm_state_indices.defined())
      << "TileLang fused_sigmoid_gating_delta_rule: ssm_state_indices must be "
         "defined";
  CHECK(cu_seqlens.defined())
      << "TileLang fused_sigmoid_gating_delta_rule: cu_seqlens must be defined";

  CHECK(A_log.device().type() == c10::DeviceType::PrivateUse1 &&
        a.device().type() == c10::DeviceType::PrivateUse1 &&
        dt_bias.device().type() == c10::DeviceType::PrivateUse1 &&
        query.device().type() == c10::DeviceType::PrivateUse1 &&
        key.device().type() == c10::DeviceType::PrivateUse1 &&
        value.device().type() == c10::DeviceType::PrivateUse1 &&
        beta.device().type() == c10::DeviceType::PrivateUse1 &&
        init_state.device().type() == c10::DeviceType::PrivateUse1 &&
        ssm_state_indices.device().type() == c10::DeviceType::PrivateUse1 &&
        cu_seqlens.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang fused_sigmoid_gating_delta_rule: all tensors must be on NPU";

  CHECK_EQ(A_log.dim(), 1)
      << "TileLang fused_sigmoid_gating_delta_rule: A_log must be 1D [nv]";
  CHECK_EQ(dt_bias.dim(), 1)
      << "TileLang fused_sigmoid_gating_delta_rule: dt_bias must be 1D [nv]";
  CHECK_EQ(a.dim(), 2)
      << "TileLang fused_sigmoid_gating_delta_rule: a must be 2D [T, nv]";
  CHECK_EQ(beta.dim(), 2)
      << "TileLang fused_sigmoid_gating_delta_rule: beta must be 2D [T, nv]";
  CHECK_EQ(a.sizes(), beta.sizes())
      << "TileLang fused_sigmoid_gating_delta_rule: a/beta shape mismatch";

  CHECK_EQ(query.dim(), 3)
      << "TileLang fused_sigmoid_gating_delta_rule: query must be 3D [T, nk, "
         "dk]";
  CHECK_EQ(key.dim(), 3)
      << "TileLang fused_sigmoid_gating_delta_rule: key must be 3D [T, nk, dk]";
  CHECK_EQ(query.sizes(), key.sizes())
      << "TileLang fused_sigmoid_gating_delta_rule: query/key shape mismatch";

  CHECK_EQ(value.dim(), 3)
      << "TileLang fused_sigmoid_gating_delta_rule: value must be 3D [T, nv, "
         "dv]";
  CHECK_EQ(value.size(0), query.size(0))
      << "TileLang fused_sigmoid_gating_delta_rule: token dim mismatch between "
         "value and query";

  const int64_t nv = value.size(1);
  const int64_t nk = query.size(1);
  const int64_t dk = query.size(2);
  const int64_t dv = value.size(2);

  CHECK_EQ(A_log.size(0), nv)
      << "TileLang fused_sigmoid_gating_delta_rule: A_log head size mismatch";
  CHECK_EQ(dt_bias.size(0), nv)
      << "TileLang fused_sigmoid_gating_delta_rule: dt_bias head size mismatch";
  CHECK_EQ(a.size(1), nv)
      << "TileLang fused_sigmoid_gating_delta_rule: a head size mismatch";
  CHECK_EQ(beta.size(1), nv)
      << "TileLang fused_sigmoid_gating_delta_rule: beta head size mismatch";
  CHECK_EQ(nv % nk, 0)
      << "TileLang fused_sigmoid_gating_delta_rule: nv must be divisible by nk";
  CHECK_GT(nk, 0);
  CHECK_GT(nv, 0);
  CHECK_GT(dk, 0);
  CHECK_GT(dv, 0);

  CHECK_EQ(init_state.dim(), 4)
      << "TileLang fused_sigmoid_gating_delta_rule: init_state must be 4D [C, "
         "nv, dk, dv]";
  CHECK_EQ(init_state.size(1), nv);
  CHECK_EQ(init_state.size(2), dk);
  CHECK_EQ(init_state.size(3), dv);

  CHECK_EQ(ssm_state_indices.dim(), 1)
      << "TileLang fused_sigmoid_gating_delta_rule: ssm_state_indices must be "
         "1D";
  CHECK_EQ(cu_seqlens.dim(), 1)
      << "TileLang fused_sigmoid_gating_delta_rule: cu_seqlens must be 1D";
  CHECK_EQ(ssm_state_indices.size(0), cu_seqlens.size(0) - 1)
      << "TileLang fused_sigmoid_gating_delta_rule: ssm_state_indices / "
         "cu_seqlens length mismatch";

  CHECK_EQ(ssm_state_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(cu_seqlens.scalar_type(), torch::kInt32);

  CHECK_EQ(a.scalar_type(), torch::kBFloat16)
      << "TileLang fused_sigmoid_gating_delta_rule: a/q/k/v/beta must be bf16";
  CHECK_EQ(A_log.dtype(), torch::kFloat32)
      << "TileLang fused_sigmoid_gating_delta_rule: A_log must be float32";
  CHECK_EQ(dt_bias.dtype(), torch::kFloat32)
      << "TileLang fused_sigmoid_gating_delta_rule: dt_bias must be float32";
  CHECK(init_state.scalar_type() == torch::kFloat32 ||
        init_state.scalar_type() == torch::kBFloat16)
      << "TileLang fused_sigmoid_gating_delta_rule: init_state must be float32 "
         "or bf16";
  CHECK_EQ(a.dtype(), query.dtype());
  CHECK_EQ(a.dtype(), key.dtype());
  CHECK_EQ(a.dtype(), value.dtype());
  CHECK_EQ(a.dtype(), beta.dtype());

  CHECK(A_log.is_contiguous());
  CHECK(dt_bias.is_contiguous());
  CHECK(ssm_state_indices.is_contiguous());
  CHECK(cu_seqlens.is_contiguous());

  CHECK_EQ(a.stride(1), 1);
  CHECK_EQ(a.stride(0), a.size(1));
  CHECK_EQ(beta.stride(1), 1);
  CHECK_EQ(beta.stride(0), beta.size(1));

  CHECK_EQ(query.stride(2), 1);
  CHECK_EQ(query.stride(1), dk);
  CHECK_EQ(key.stride(2), 1);
  CHECK_EQ(key.stride(1), dk);
  CHECK_EQ(value.stride(2), 1);
  CHECK_EQ(value.stride(1), dv);

  CHECK_EQ(init_state.stride(3), 1);
  CHECK_EQ(init_state.stride(2), dv);
  CHECK_EQ(init_state.stride(1), dk * dv);
  CHECK_EQ(a.size(0), query.size(0))
      << "TileLang fused_sigmoid_gating_delta_rule: a/query token dim mismatch";
  CHECK_EQ(beta.size(0), query.size(0))
      << "TileLang fused_sigmoid_gating_delta_rule: beta/query token dim "
         "mismatch";
}

FusedSigmoidGatingDeltaRuleSpecialization build_runtime_specialization(
    int32_t compiled_num_seqs,
    const torch::Tensor& query,
    const torch::Tensor& value) {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(value.dim(), 3);
  const int32_t nk = static_cast<int32_t>(query.size(1));
  const int32_t nv = static_cast<int32_t>(value.size(1));
  const int32_t dk = static_cast<int32_t>(query.size(2));
  const int32_t dv = static_cast<int32_t>(value.size(2));
  const int32_t block_v = auto_block_v(dv);
  const TilelangDType dtype = to_tilelang_dtype(query.scalar_type());

  return make_fused_sigmoid_gating_delta_rule_specialization(
      FusedSigmoidGatingDeltaRuleMaxNumSeqs{compiled_num_seqs},
      FusedSigmoidGatingDeltaRuleNk{nk},
      FusedSigmoidGatingDeltaRuleNv{nv},
      FusedSigmoidGatingDeltaRuleDk{dk},
      FusedSigmoidGatingDeltaRuleDv{dv},
      FusedSigmoidGatingDeltaRuleBlockV{block_v},
      FusedSigmoidGatingDeltaRuleUseQkL2norm{kUseQkL2norm},
      FusedSigmoidGatingDeltaRuleDType{dtype});
}

const auto* find_entry_with_fallback(
    FusedSigmoidGatingDeltaRuleSpecialization specialization) {
  const auto* entry =
      find_fused_sigmoid_gating_delta_rule_kernel_entry(specialization);
  if (entry != nullptr) {
    return entry;
  }
  int32_t fallback_n = specialization.max_num_seqs - kNSpecializationStep;
  while (fallback_n >= kNSpecializationMin && entry == nullptr) {
    specialization = make_fused_sigmoid_gating_delta_rule_specialization(
        FusedSigmoidGatingDeltaRuleMaxNumSeqs{fallback_n},
        FusedSigmoidGatingDeltaRuleNk{specialization.nk},
        FusedSigmoidGatingDeltaRuleNv{specialization.nv},
        FusedSigmoidGatingDeltaRuleDk{specialization.dk},
        FusedSigmoidGatingDeltaRuleDv{specialization.dv},
        FusedSigmoidGatingDeltaRuleBlockV{specialization.block_v},
        FusedSigmoidGatingDeltaRuleUseQkL2norm{specialization.use_qk_l2norm},
        FusedSigmoidGatingDeltaRuleDType{specialization.dtype});
    entry = find_fused_sigmoid_gating_delta_rule_kernel_entry(specialization);
    fallback_n -= kNSpecializationStep;
  }
  return entry;
}

int32_t checked_i32_stride(int64_t stride,
                           int64_t max_stride,
                           const char* name) {
  CHECK_GT(stride, 0) << "TileLang fused_sigmoid_gating_delta_rule: " << name
                      << " must be positive";
  CHECK_LE(stride, max_stride)
      << "TileLang fused_sigmoid_gating_delta_rule: " << name
      << " exceeds supported qkv token stride";
  CHECK_LE(stride, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang fused_sigmoid_gating_delta_rule: " << name
      << " exceeds int32";
  return static_cast<int32_t>(stride);
}

void launch_tilelang_fused_sigmoid_gating_delta_rule(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& dt_bias,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& init_state_accum,
    const torch::Tensor& ssm_state_indices,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& out,
    const torch::Tensor& final_state,
    int32_t query_stride_t,
    int32_t key_stride_t,
    int32_t value_stride_t,
    float softplus_beta,
    float scale,
    bool use_qk_l2norm,
    float softplus_threshold) {
  const int64_t actual_n = ssm_state_indices.size(0);
  CHECK_GT(actual_n, 0)
      << "TileLang fused_sigmoid_gating_delta_rule: actual_n must be > 0";
  CHECK_EQ(cu_seqlens.size(0), actual_n + 1)
      << "TileLang fused_sigmoid_gating_delta_rule: chunked cu_seqlens length "
      << "mismatch";

  const int32_t nk = static_cast<int32_t>(query.size(1));
  const int32_t nv = static_cast<int32_t>(value.size(1));
  const int32_t dk = static_cast<int32_t>(query.size(2));
  const int32_t dv = static_cast<int32_t>(value.size(2));
  const int32_t block_v = auto_block_v(dv);
  const TilelangDType dtype = to_tilelang_dtype(query.scalar_type());
  const int32_t compiled_n =
      select_launch_num_seqs(actual_n, nk, nv, dk, dv, block_v, dtype);

  // Pad per-launch metadata to the compiled num_seqs. The cu_seqlens values are
  // absolute token offsets into the full q/k/v/out tensors, so chunks do not
  // need q/k/v copies or token-offset rebasing.
  torch::Tensor indices_prepared;
  if (static_cast<int64_t>(compiled_n) > actual_n) {
    indices_prepared = torch::empty({compiled_n}, ssm_state_indices.options());
    indices_prepared.narrow(0, 0, actual_n).copy_(ssm_state_indices);
    indices_prepared.narrow(0, actual_n, compiled_n - actual_n).fill_(-1);
  } else {
    indices_prepared = ssm_state_indices;
  }

  torch::Tensor cu_prepared;
  if (static_cast<int64_t>(compiled_n) > actual_n) {
    cu_prepared = torch::empty({compiled_n + 1}, cu_seqlens.options());
    cu_prepared.narrow(0, 0, actual_n + 1).copy_(cu_seqlens);
    const int64_t num_dummy = compiled_n - actual_n;
    cu_prepared.narrow(0, actual_n + 1, num_dummy)
        .copy_(cu_seqlens.narrow(0, actual_n, 1).expand({num_dummy}));
  } else {
    cu_prepared = cu_seqlens;
  }

  auto specialization = build_runtime_specialization(compiled_n, query, value);
  const auto* entry = find_entry_with_fallback(specialization);
  CHECK(entry != nullptr)
      << "TileLang fused_sigmoid_gating_delta_rule: no compiled variant. "
      << "Available variants: "
      << available_fused_sigmoid_gating_delta_rule_variant_keys();
  CHECK_GE(entry->spec.max_num_seqs, actual_n)
      << "TileLang fused_sigmoid_gating_delta_rule: selected compiled "
      << "num_seqs=" << entry->spec.max_num_seqs
      << " is smaller than actual_n=" << actual_n;

  const int32_t device_id = query.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

  entry->fn(static_cast<uint8_t*>(A_log.data_ptr()),
            static_cast<uint8_t*>(a.data_ptr()),
            static_cast<uint8_t*>(dt_bias.data_ptr()),
            static_cast<uint8_t*>(query.data_ptr()),
            static_cast<uint8_t*>(key.data_ptr()),
            static_cast<uint8_t*>(value.data_ptr()),
            static_cast<uint8_t*>(beta.data_ptr()),
            static_cast<uint8_t*>(init_state_accum.data_ptr()),
            static_cast<uint8_t*>(indices_prepared.data_ptr()),
            static_cast<uint8_t*>(cu_prepared.data_ptr()),
            static_cast<uint8_t*>(out.data_ptr()),
            static_cast<uint8_t*>(final_state.data_ptr()),
            softplus_beta,
            scale,
            static_cast<int32_t>(use_qk_l2norm ? 1 : 0),
            softplus_threshold,
            query_stride_t,
            key_stride_t,
            value_stride_t,
            static_cast<int32_t>(query.size(0)),
            static_cast<int32_t>(init_state_accum.size(0)),
            stream);
}

std::tuple<torch::Tensor, torch::Tensor>
run_tilelang_fused_sigmoid_gating_delta_rule(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& dt_bias,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& init_state,
    const torch::Tensor& ssm_state_indices,
    const torch::Tensor& cu_seqlens,
    float softplus_beta,
    float scale,
    bool use_qk_l2norm,
    float softplus_threshold) {
  const int64_t actual_n = ssm_state_indices.size(0);
  CHECK_GT(actual_n, 0)
      << "TileLang fused_sigmoid_gating_delta_rule: actual_n must be > 0";

  const int32_t nk = static_cast<int32_t>(query.size(1));
  const int32_t nv = static_cast<int32_t>(value.size(1));
  const int32_t dk = static_cast<int32_t>(query.size(2));
  const int32_t dv = static_cast<int32_t>(value.size(2));
  const int32_t block_v = auto_block_v(dv);
  const TilelangDType dtype = to_tilelang_dtype(query.scalar_type());
  const int32_t max_compiled_n =
      max_compiled_num_seqs(nk, nv, dk, dv, block_v, dtype);
  CHECK_GT(max_compiled_n, 0)
      << "TileLang fused_sigmoid_gating_delta_rule: no compiled num_seqs "
      << "variant for nk=" << nk << ", nv=" << nv << ", dk=" << dk
      << ", dv=" << dv << ", dtype=" << static_cast<int>(dtype);
  const int64_t max_qkv_token_stride =
      static_cast<int64_t>(2) * nk * dk + static_cast<int64_t>(nv) * dv;
  const int32_t query_stride_t = checked_i32_stride(
      query.stride(0), max_qkv_token_stride, "query stride(0)");
  const int32_t key_stride_t =
      checked_i32_stride(key.stride(0), max_qkv_token_stride, "key stride(0)");
  const int32_t value_stride_t = checked_i32_stride(
      value.stride(0), max_qkv_token_stride, "value stride(0)");

  const auto options = query.options();
  const auto init_state_accum = init_state.scalar_type() == torch::kFloat32
                                    ? init_state
                                    : init_state.to(torch::kFloat32);

  auto out = torch::empty({query.size(0), nv, dv}, options);
  auto final_state = init_state_accum;
  CHECK_EQ(out.stride(2), 1);
  CHECK_EQ(out.stride(1), dv);
  CHECK_EQ(final_state.stride(3), 1);
  CHECK_EQ(final_state.stride(2), dv);
  CHECK_EQ(final_state.stride(1), dk * dv);

  for (int64_t seq_start = 0; seq_start < actual_n;
       seq_start += max_compiled_n) {
    const int64_t chunk_n =
        std::min<int64_t>(max_compiled_n, actual_n - seq_start);
    auto indices_chunk =
        ssm_state_indices.narrow(0, seq_start, chunk_n).contiguous();
    auto cu_chunk = cu_seqlens.narrow(0, seq_start, chunk_n + 1).contiguous();
    launch_tilelang_fused_sigmoid_gating_delta_rule(A_log,
                                                    a,
                                                    dt_bias,
                                                    query,
                                                    key,
                                                    value,
                                                    beta,
                                                    init_state_accum,
                                                    indices_chunk,
                                                    cu_chunk,
                                                    out,
                                                    final_state,
                                                    query_stride_t,
                                                    key_stride_t,
                                                    value_stride_t,
                                                    softplus_beta,
                                                    scale,
                                                    use_qk_l2norm,
                                                    softplus_threshold);
  }

  if (init_state.scalar_type() != torch::kFloat32) {
    // The update API only returns out, so a fp32 accumulation copy must be
    // written back to the original bf16 cache before returning.
    auto valid_indices_i32 =
        ssm_state_indices.masked_select(ssm_state_indices.ge(0));
    if (valid_indices_i32.numel() > 0) {
      auto valid_indices = valid_indices_i32.to(torch::kLong);
      auto updated_state = final_state.index_select(0, valid_indices)
                               .to(init_state.scalar_type());
      init_state.index_put_({valid_indices}, updated_state);
    }
  }

  auto final_state_sliced = final_state.narrow(0, 0, actual_n);
  return {out, final_state_sliced};
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> fused_sigmoid_gating_delta_rule(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& dt_bias,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& init_state,
    const torch::Tensor& ssm_state_indices,
    const torch::Tensor& cu_seqlens,
    std::optional<float> scale,
    bool use_qk_l2norm_in_kernel,
    float softplus_beta,
    float softplus_threshold) {
  check_supported(A_log,
                  a,
                  dt_bias,
                  query,
                  key,
                  value,
                  beta,
                  init_state,
                  ssm_state_indices,
                  cu_seqlens);

  const float runtime_scale =
      scale.has_value() ? scale.value()
                        : 1.0f / std::sqrt(static_cast<float>(query.size(2)));

  return run_tilelang_fused_sigmoid_gating_delta_rule(A_log,
                                                      a,
                                                      dt_bias,
                                                      query,
                                                      key,
                                                      value,
                                                      beta,
                                                      init_state,
                                                      ssm_state_indices,
                                                      cu_seqlens,
                                                      softplus_beta,
                                                      runtime_scale,
                                                      use_qk_l2norm_in_kernel,
                                                      softplus_threshold);
}

}  // namespace xllm::kernel::npu::tilelang
