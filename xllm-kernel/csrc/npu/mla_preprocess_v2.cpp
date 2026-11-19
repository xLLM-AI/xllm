// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <torch/library.h>
#include <torch/types.h>

#include <mutex>
#include <tuple>

#include "csrc/npu/ascendc_api.h"
#include "op_plugin/utils/op_api_common_base.h"

namespace xllm_kernel {
namespace {

const AscendcApi& get_api() {
  static const auto api =
      find_ascendc_api("aclnnXllmKernelMlaPreprocessV2",
                       "aclnnXllmKernelMlaPreprocessV2GetWorkspaceSize");
  return api;
}

bool has_mla_preprocess_v2() {
  const auto& api = get_api();
  return api.launch && api.workspace;
}

void GetApiFunc(const char* /*api_name*/,
                const char* /*workspace_name*/,
                void*& launch,
                void*& workspace) {
  static std::once_flag initialize;
  std::call_once(initialize, [&] {
    const auto& api = get_api();
    CHECK(api.launch && api.workspace) << "MlaPreprocessV2 is unavailable";
    launch = api.launch;
    workspace = api.workspace;
  });
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mla_preprocess_v2(
    const torch::Tensor& hidden,
    const torch::Tensor& input_norm_weight,
    const torch::Tensor& input_norm_bias,
    const torch::Tensor& qkv_input_scale,
    const torch::Tensor& qkv_input_offset,
    const torch::Tensor& qkv_weight,
    const torch::Tensor& qkv_deq_scale,
    const torch::Tensor& qkv_quant_bias,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& q_norm_bias,
    const torch::Tensor& q_b_input_scale,
    const torch::Tensor& q_b_input_offset,
    const torch::Tensor& q_b_weight,
    const torch::Tensor& q_b_deq_scale,
    const torch::Tensor& q_b_quant_bias,
    const torch::Tensor& kv_norm_weight,
    const torch::Tensor& rope_cos,
    const torch::Tensor& rope_sin,
    const torch::Tensor& w_uk,
    torch::Tensor& kv_cache,
    torch::Tensor& rope_cache,
    const torch::Tensor& slot_mapping,
    int64_t q_lora_rank,
    int64_t rope_dim,
    double eps,
    int64_t cache_mode) {
  constexpr int64_t kKvLoraRank = 512;
  constexpr int64_t kNdCache = 1;
  constexpr int64_t kNzCache = 3;
  CHECK(cache_mode == kNdCache || cache_mode == kNzCache);
  const int64_t tokens = hidden.size(0);
  const int64_t heads = w_uk.size(0);
  auto q_latent =
      torch::empty({tokens, heads, kKvLoraRank}, kv_cache.options());
  auto q_pe = torch::empty({tokens, heads, rope_dim}, hidden.options());
  auto q_down = torch::empty({tokens, q_norm_weight.numel()}, hidden.options());
  const auto cos = rope_cos.view({tokens, -1});
  const auto sin = rope_sin.view({tokens, -1});

  const int64_t rotary_coeff = 2;
  const bool transpose_weight = true;
  const int64_t per_tensor_asymmetric = 0;
  const bool apply_input_rms_norm = false;
  const int64_t wdkv_split_count = 1;
  const bool return_q_down = true;
  const torch::DeviceGuard guard(hidden.device());
  EXEC_NPU_CMD_EXT(aclnnXllmKernelMlaPreprocessV2,
                   hidden,
                   input_norm_weight,
                   input_norm_bias,
                   qkv_input_scale,
                   qkv_input_offset,
                   qkv_weight,
                   qkv_deq_scale,
                   qkv_quant_bias,
                   q_norm_weight,
                   q_norm_bias,
                   q_b_input_scale,
                   q_b_input_offset,
                   q_b_weight,
                   q_b_deq_scale,
                   q_b_quant_bias,
                   kv_norm_weight,
                   cos,
                   sin,
                   w_uk,
                   kv_cache,
                   rope_cache,
                   slot_mapping,
                   /*ctkv_scale=*/qkv_input_scale,
                   /*q_nope_scale=*/q_b_input_scale,
                   q_lora_rank,
                   /*q_rope_dim=*/rope_dim,
                   /*k_rope_dim=*/rope_dim,
                   eps,
                   /*q_rotary_coeff=*/rotary_coeff,
                   /*k_rotary_coeff=*/rotary_coeff,
                   /*transpose_wdq=*/transpose_weight,
                   /*transpose_wuq=*/transpose_weight,
                   /*transpose_wuk=*/transpose_weight,
                   cache_mode,
                   per_tensor_asymmetric,
                   apply_input_rms_norm,
                   wdkv_split_count,
                   return_q_down,
                   q_latent,
                   kv_cache,
                   q_pe,
                   rope_cache,
                   q_down);
  return {q_down, q_latent, q_pe};
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("has_mla_preprocess_v2() -> bool", TORCH_FN(has_mla_preprocess_v2));
  m.def(
      "mla_preprocess_v2(Tensor hidden, Tensor input_norm_weight, Tensor "
      "input_norm_bias, Tensor qkv_input_scale, Tensor qkv_input_offset, "
      "Tensor qkv_weight, Tensor qkv_deq_scale, Tensor qkv_quant_bias, "
      "Tensor q_norm_weight, Tensor q_norm_bias, Tensor q_b_input_scale, "
      "Tensor q_b_input_offset, Tensor q_b_weight, Tensor q_b_deq_scale, "
      "Tensor q_b_quant_bias, Tensor kv_norm_weight, Tensor rope_cos, "
      "Tensor rope_sin, Tensor w_uk, Tensor(a!) kv_cache, "
      "Tensor(b!) rope_cache, Tensor slot_mapping, int q_lora_rank, "
      "int rope_dim, float eps, int cache_mode) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("mla_preprocess_v2", TORCH_FN(mla_preprocess_v2));
}

}  // namespace
}  // namespace xllm_kernel
