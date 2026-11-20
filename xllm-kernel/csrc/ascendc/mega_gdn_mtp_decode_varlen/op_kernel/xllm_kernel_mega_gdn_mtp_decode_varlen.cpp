/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mega_gdn_mtp_decode_pto_kernel.h"

struct MegaGdnMtpDecodeVarlenTilingData {
  int64_t batch_size;
  int64_t sequence_length;
  int64_t num_k_heads;
  int64_t num_v_heads;
  int64_t state_index_width;
};

template <bool FlaSsmStateLayout>
AICORE PTO_INLINE void run_varlen(
    GM_ADDR qkv,
    GM_ADDR z,
    GM_ADDR b,
    GM_ADDR a,
    GM_ADDR conv_weight,
    GM_ADDR conv_state,
    GM_ADDR a_log,
    GM_ADDR dt_bias,
    GM_ADDR ssm_state,
    GM_ADDR state_indices,
    GM_ADDR accepted,
    GM_ADDR norm_weight,
    GM_ADDR cu_seqlens,
    GM_ADDR conv_out,
    GM_ADDR conv_state_out,
    GM_ADDR ssm_state_out,
    GM_ADDR out,
    const MegaGdnMtpDecodeVarlenTilingData& data) {
  mega_gdn_mtp_decode_pto::Run<0, false, false, FlaSsmStateLayout, true>(
      reinterpret_cast<__gm__ bfloat16_t*>(qkv),
      reinterpret_cast<__gm__ bfloat16_t*>(z),
      reinterpret_cast<__gm__ bfloat16_t*>(b),
      reinterpret_cast<__gm__ bfloat16_t*>(a),
      reinterpret_cast<__gm__ bfloat16_t*>(conv_weight),
      reinterpret_cast<__gm__ bfloat16_t*>(conv_state),
      reinterpret_cast<__gm__ float*>(a_log),
      reinterpret_cast<__gm__ float*>(dt_bias),
      reinterpret_cast<__gm__ float*>(ssm_state),
      reinterpret_cast<__gm__ int*>(state_indices),
      reinterpret_cast<__gm__ int*>(state_indices),
      reinterpret_cast<__gm__ int*>(accepted),
      reinterpret_cast<__gm__ bfloat16_t*>(norm_weight),
      reinterpret_cast<__gm__ bfloat16_t*>(conv_out),
      reinterpret_cast<__gm__ bfloat16_t*>(conv_state_out),
      reinterpret_cast<__gm__ float*>(ssm_state_out),
      reinterpret_cast<__gm__ bfloat16_t*>(out),
      static_cast<int32_t>(data.num_k_heads),
      static_cast<int32_t>(data.num_v_heads),
      static_cast<int32_t>(data.batch_size),
      static_cast<int32_t>(data.sequence_length),
      reinterpret_cast<__gm__ int32_t*>(cu_seqlens),
      static_cast<int32_t>(data.state_index_width));
}

extern "C" __global__ __aicore__ void xllm_kernel_mega_gdn_mtp_decode_varlen(
    GM_ADDR qkv,
    GM_ADDR z,
    GM_ADDR b,
    GM_ADDR a,
    GM_ADDR conv_weight,
    GM_ADDR conv_state,
    GM_ADDR a_log,
    GM_ADDR dt_bias,
    GM_ADDR ssm_state,
    GM_ADDR state_indices,
    GM_ADDR accepted,
    GM_ADDR norm_weight,
    GM_ADDR cu_seqlens,
    GM_ADDR conv_out,
    GM_ADDR conv_state_out,
    GM_ADDR ssm_state_out,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  REGISTER_TILING_DEFAULT(MegaGdnMtpDecodeVarlenTilingData);
  GET_TILING_DATA_WITH_STRUCT(MegaGdnMtpDecodeVarlenTilingData, data, tiling);
  (void)workspace;
#define RUN_VARLEN(LAYOUT)           \
  run_varlen<LAYOUT>(qkv,            \
                     z,              \
                     b,              \
                     a,              \
                     conv_weight,    \
                     conv_state,     \
                     a_log,          \
                     dt_bias,        \
                     ssm_state,      \
                     state_indices,  \
                     accepted,       \
                     norm_weight,    \
                     cu_seqlens,     \
                     conv_out,       \
                     conv_state_out, \
                     ssm_state_out,  \
                     out,            \
                     data)
  if (TILING_KEY_IS(100)) {
    RUN_VARLEN(true);
  } else if (TILING_KEY_IS(1100)) {
    RUN_VARLEN(false);
  }
#undef RUN_VARLEN
}

#include "lib/matmul_intf.h"
