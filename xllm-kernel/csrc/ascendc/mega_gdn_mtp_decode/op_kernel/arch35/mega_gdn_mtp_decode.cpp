/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "mega_gdn_mtp_decode/op_kernel/mega_gdn_mtp_decode_pto_kernel.h"

// The included PTO implementation owns its explicit MTE2/V/MTE3 event chain.
struct MegaGdnMtpDecodeTilingData {
  int64_t batch_size;
  int64_t sequence_length;
  int64_t num_k_heads;
  int64_t num_v_heads;
  int64_t state_index_width;
};

#if defined(PTO_NPU_ARCH_A5)
template <int32_t SpeculativeTokens,
          bool UseQkGroupCache,
          bool UseDeferredNorm,
          bool UseTwoOwnerQkGroups = false,
          bool FlaSsmStateLayout = true,
          bool UseRegBase = false>
#else
template <int32_t SpeculativeTokens,
          bool UseQkGroupCache,
          bool UseDeferredNorm,
          bool FlaSsmStateLayout = true,
          bool UseRegBase = false>
#endif
AICORE PTO_INLINE void RunMegaGdnMtpDecode(GM_ADDR qkv,
                                           GM_ADDR z,
                                           GM_ADDR b,
                                           GM_ADDR a,
                                           GM_ADDR conv_weight,
                                           GM_ADDR conv_state,
                                           GM_ADDR a_log,
                                           GM_ADDR dt_bias,
                                           GM_ADDR ssm_state,
                                           GM_ADDR read_state_indices,
                                           GM_ADDR write_state_indices,
                                           GM_ADDR num_accepted_tokens,
                                           GM_ADDR norm_weight,
                                           GM_ADDR conv_out,
                                           GM_ADDR conv_state_out,
                                           GM_ADDR ssm_state_out,
                                           GM_ADDR out,
                                           int32_t num_k_heads,
                                           int32_t num_v_heads,
                                           int32_t batch_size,
                                           int32_t runtime_sequence_length,
                                           int32_t state_index_width) {
#if defined(PTO_NPU_ARCH_A5)
  mega_gdn_mtp_decode_pto::Run<SpeculativeTokens,
                               UseQkGroupCache,
                               UseDeferredNorm,
                               UseTwoOwnerQkGroups,
                               FlaSsmStateLayout,
                               UseRegBase>(
#else
  mega_gdn_mtp_decode_pto::Run<SpeculativeTokens,
                               UseQkGroupCache,
                               UseDeferredNorm,
                               FlaSsmStateLayout,
                               UseRegBase>(
#endif
      reinterpret_cast<__gm__ bfloat16_t*>(qkv),
      reinterpret_cast<__gm__ bfloat16_t*>(z),
      reinterpret_cast<__gm__ bfloat16_t*>(b),
      reinterpret_cast<__gm__ bfloat16_t*>(a),
      reinterpret_cast<__gm__ bfloat16_t*>(conv_weight),
      reinterpret_cast<__gm__ bfloat16_t*>(conv_state),
      reinterpret_cast<__gm__ float*>(a_log),
      reinterpret_cast<__gm__ float*>(dt_bias),
      reinterpret_cast<__gm__ float*>(ssm_state),
      reinterpret_cast<__gm__ int*>(read_state_indices),
      reinterpret_cast<__gm__ int*>(write_state_indices),
      reinterpret_cast<__gm__ int*>(num_accepted_tokens),
      reinterpret_cast<__gm__ bfloat16_t*>(norm_weight),
      reinterpret_cast<__gm__ bfloat16_t*>(conv_out),
      reinterpret_cast<__gm__ bfloat16_t*>(conv_state_out),
      reinterpret_cast<__gm__ float*>(ssm_state_out),
      reinterpret_cast<__gm__ bfloat16_t*>(out),
      num_k_heads,
      num_v_heads,
      batch_size,
      runtime_sequence_length,
      state_index_width);
}

extern "C" __global__ __aicore__ void xllm_kernel_mega_gdn_mtp_decode(
    GM_ADDR qkv,
    GM_ADDR z,
    GM_ADDR b,
    GM_ADDR a,
    GM_ADDR conv_weight,
    GM_ADDR conv_state,
    GM_ADDR a_log,
    GM_ADDR dt_bias,
    GM_ADDR ssm_state,
    GM_ADDR read_state_indices,
    GM_ADDR write_state_indices,
    GM_ADDR num_accepted_tokens,
    GM_ADDR norm_weight,
    GM_ADDR conv_out,
    GM_ADDR conv_state_out,
    GM_ADDR ssm_state_out,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling) {
#if defined(PTO_NPU_ARCH_A5)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#else
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
  REGISTER_TILING_DEFAULT(MegaGdnMtpDecodeTilingData);
  GET_TILING_DATA_WITH_STRUCT(MegaGdnMtpDecodeTilingData, tiling_data, tiling);
  (void)workspace;

#if defined(PTO_NPU_ARCH_A5)
#define RUN_MTP_IMPL(K,                        \
                     USE_QK_GROUP_CACHE,       \
                     USE_DEFERRED_NORM,        \
                     USE_TWO_OWNER_QK_GROUPS,  \
                     FLA_SSM_STATE_LAYOUT,     \
                     USE_REGBASE)              \
  RunMegaGdnMtpDecode<K,                       \
                      USE_QK_GROUP_CACHE,      \
                      USE_DEFERRED_NORM,       \
                      USE_TWO_OWNER_QK_GROUPS, \
                      FLA_SSM_STATE_LAYOUT,    \
                      USE_REGBASE>
#else
#define RUN_MTP_IMPL(K,                       \
                     USE_QK_GROUP_CACHE,      \
                     USE_DEFERRED_NORM,       \
                     USE_TWO_OWNER_QK_GROUPS, \
                     FLA_SSM_STATE_LAYOUT,    \
                     USE_REGBASE)             \
  RunMegaGdnMtpDecode<K,                      \
                      USE_QK_GROUP_CACHE,     \
                      USE_DEFERRED_NORM,      \
                      FLA_SSM_STATE_LAYOUT>
#endif

#define RUN_MTP(K,                              \
                USE_QK_GROUP_CACHE,             \
                USE_DEFERRED_NORM,              \
                USE_TWO_OWNER_QK_GROUPS,        \
                FLA_SSM_STATE_LAYOUT)           \
  RUN_MTP_WITH_REGBASE(K,                       \
                       USE_QK_GROUP_CACHE,      \
                       USE_DEFERRED_NORM,       \
                       USE_TWO_OWNER_QK_GROUPS, \
                       FLA_SSM_STATE_LAYOUT,    \
                       false)

#define RUN_MTP_WITH_REGBASE(K,                          \
                             USE_QK_GROUP_CACHE,         \
                             USE_DEFERRED_NORM,          \
                             USE_TWO_OWNER_QK_GROUPS,    \
                             FLA_SSM_STATE_LAYOUT,       \
                             USE_REGBASE)                \
  RUN_MTP_IMPL(K,                                        \
               USE_QK_GROUP_CACHE,                       \
               USE_DEFERRED_NORM,                        \
               USE_TWO_OWNER_QK_GROUPS,                  \
               FLA_SSM_STATE_LAYOUT,                     \
               USE_REGBASE)(                             \
      qkv,                                               \
      z,                                                 \
      b,                                                 \
      a,                                                 \
      conv_weight,                                       \
      conv_state,                                        \
      a_log,                                             \
      dt_bias,                                           \
      ssm_state,                                         \
      read_state_indices,                                \
      write_state_indices,                               \
      num_accepted_tokens,                               \
      norm_weight,                                       \
      conv_out,                                          \
      conv_state_out,                                    \
      ssm_state_out,                                     \
      out,                                               \
      static_cast<int32_t>(tiling_data.num_k_heads),     \
      static_cast<int32_t>(tiling_data.num_v_heads),     \
      static_cast<int32_t>(tiling_data.batch_size),      \
      static_cast<int32_t>(tiling_data.sequence_length), \
      static_cast<int32_t>(tiling_data.state_index_width))

  if constexpr (TILING_KEY_IS(101)) {
    RUN_MTP(1, false, false, false, true);
  } else if constexpr (TILING_KEY_IS(102)) {
    RUN_MTP(2, false, false, false, true);
  } else if constexpr (TILING_KEY_IS(103)) {
    RUN_MTP(3, false, false, false, true);
  } else if constexpr (TILING_KEY_IS(104)) {
    RUN_MTP(4, false, false, false, true);
  } else if constexpr (TILING_KEY_IS(105)) {
    RUN_MTP(5, false, false, false, true);
  } else if constexpr (TILING_KEY_IS(108)) {
    RUN_MTP(8, false, false, false, true);
  } else if constexpr (TILING_KEY_IS(208)) {
    RUN_MTP(8, true, true, false, true);
#if defined(PTO_NPU_ARCH_A5)
  } else if constexpr (TILING_KEY_IS(308)) {
    RUN_MTP(8, true, true, true, true);
  } else if constexpr (TILING_KEY_IS(303)) {
    RUN_MTP(3, true, true, true, true);
  } else if constexpr (TILING_KEY_IS(203)) {
    RUN_MTP(3, false, true, false, true);
  } else if constexpr (TILING_KEY_IS(403)) {
    RUN_MTP_WITH_REGBASE(3, false, true, false, true, true);
  } else if constexpr (TILING_KEY_IS(503)) {
    RUN_MTP_WITH_REGBASE(3, true, true, true, true, true);
#endif
  } else if constexpr (TILING_KEY_IS(210)) {
    RUN_MTP(10, false, true, false, true);
  } else if constexpr (TILING_KEY_IS(211)) {
    RUN_MTP(11, false, true, false, true);
  } else if constexpr (TILING_KEY_IS(212)) {
    RUN_MTP(12, false, true, false, true);
  } else if constexpr (TILING_KEY_IS(213)) {
    RUN_MTP(13, false, true, false, true);
  } else if constexpr (TILING_KEY_IS(214)) {
    RUN_MTP(14, false, true, false, true);
  } else if constexpr (TILING_KEY_IS(215)) {
    RUN_MTP(15, false, true, false, true);
  } else if constexpr (TILING_KEY_IS(216)) {
    RUN_MTP(16, false, true, false, true);
  } else if constexpr (TILING_KEY_IS(100)) {
    RUN_MTP(0, false, false, false, true);
  } else if constexpr (TILING_KEY_IS(1101)) {
    RUN_MTP(1, false, false, false, false);
  } else if constexpr (TILING_KEY_IS(1102)) {
    RUN_MTP(2, false, false, false, false);
  } else if constexpr (TILING_KEY_IS(1103)) {
    RUN_MTP(3, false, false, false, false);
  } else if constexpr (TILING_KEY_IS(1104)) {
    RUN_MTP(4, false, false, false, false);
  } else if constexpr (TILING_KEY_IS(1105)) {
    RUN_MTP(5, false, false, false, false);
  } else if constexpr (TILING_KEY_IS(1108)) {
    RUN_MTP(8, false, false, false, false);
  } else if constexpr (TILING_KEY_IS(1100)) {
    RUN_MTP(0, false, false, false, false);
  }
#undef RUN_MTP
#undef RUN_MTP_WITH_REGBASE
#undef RUN_MTP_IMPL
}

// Keep this include last; generated mixed-kernel wrappers call
// matmul::clearWorkspace.
#include "lib/matmul_intf.h"
