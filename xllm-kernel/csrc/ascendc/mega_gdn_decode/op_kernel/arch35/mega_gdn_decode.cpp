/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "mega_gdn_decode/op_kernel/mega_gdn_decode_pto_kernel.h"

struct MegaGdnDecodeTilingData {
  int64_t batch_size;
  int64_t num_k_heads;
  int64_t num_v_heads;
  int64_t state_index_width;
};

template <bool IsBatchOne,
          bool FlaSsmStateLayout,
          bool UseA5B4DeferredNorm = false,
          bool UseA5B4RegBase = false,
          bool SplitStateIndices = false>
AICORE PTO_INLINE void RunMegaGdnDecode(GM_ADDR qkv,
                                        GM_ADDR z,
                                        GM_ADDR b,
                                        GM_ADDR a,
                                        GM_ADDR convWeight,
                                        GM_ADDR convState,
                                        GM_ADDR a_log,
                                        GM_ADDR dtBias,
                                        GM_ADDR ssmState,
                                        GM_ADDR readStateIndices,
                                        GM_ADDR writeStateIndices,
                                        GM_ADDR normWeight,
                                        GM_ADDR convOut,
                                        GM_ADDR convStateOut,
                                        GM_ADDR ssmStateOut,
                                        GM_ADDR out,
                                        int32_t numKHeads,
                                        int32_t numVHeads,
                                        int32_t batchSize) {
  mega_gdn_decode_pto::Run<IsBatchOne,
                           FlaSsmStateLayout,
                           UseA5B4DeferredNorm,
                           UseA5B4RegBase,
                           SplitStateIndices>(
      reinterpret_cast<__gm__ bfloat16_t*>(qkv),
      reinterpret_cast<__gm__ bfloat16_t*>(z),
      reinterpret_cast<__gm__ bfloat16_t*>(b),
      reinterpret_cast<__gm__ bfloat16_t*>(a),
      reinterpret_cast<__gm__ bfloat16_t*>(convWeight),
      reinterpret_cast<__gm__ bfloat16_t*>(convState),
      reinterpret_cast<__gm__ float*>(a_log),
      reinterpret_cast<__gm__ float*>(dtBias),
      reinterpret_cast<__gm__ float*>(ssmState),
      reinterpret_cast<__gm__ int*>(readStateIndices),
      reinterpret_cast<__gm__ int*>(writeStateIndices),
      reinterpret_cast<__gm__ bfloat16_t*>(normWeight),
      reinterpret_cast<__gm__ bfloat16_t*>(convOut),
      reinterpret_cast<__gm__ bfloat16_t*>(convStateOut),
      reinterpret_cast<__gm__ float*>(ssmStateOut),
      reinterpret_cast<__gm__ bfloat16_t*>(out),
      numKHeads,
      numVHeads,
      batchSize);
}

template <bool FlaSsmStateLayout>
AICORE PTO_INLINE void RunSplitStateDecode(
    GM_ADDR qkv,
    GM_ADDR z,
    GM_ADDR b,
    GM_ADDR a,
    GM_ADDR conv_weight,
    GM_ADDR conv_state,
    GM_ADDR a_log,
    GM_ADDR dt_bias,
    GM_ADDR ssm_state,
    GM_ADDR read_indices,
    GM_ADDR write_indices,
    GM_ADDR norm_weight,
    GM_ADDR conv_out,
    GM_ADDR conv_state_out,
    GM_ADDR ssm_state_out,
    GM_ADDR out,
    const MegaGdnDecodeTilingData& data) {
  // Only row addressing differs from ordinary Decode. Keep its FP32 gate
  // and recurrent update rather than the MTP kernel's BF16 gate contract.
  if (data.batch_size == 1) {
    RunMegaGdnDecode<true, FlaSsmStateLayout, false, false, true>(
        qkv,
        z,
        b,
        a,
        conv_weight,
        conv_state,
        a_log,
        dt_bias,
        ssm_state,
        read_indices,
        write_indices,
        norm_weight,
        conv_out,
        conv_state_out,
        ssm_state_out,
        out,
        static_cast<int32_t>(data.num_k_heads),
        static_cast<int32_t>(data.num_v_heads),
        static_cast<int32_t>(data.batch_size));
  } else {
    RunMegaGdnDecode<false, FlaSsmStateLayout, false, false, true>(
        qkv,
        z,
        b,
        a,
        conv_weight,
        conv_state,
        a_log,
        dt_bias,
        ssm_state,
        read_indices,
        write_indices,
        norm_weight,
        conv_out,
        conv_state_out,
        ssm_state_out,
        out,
        static_cast<int32_t>(data.num_k_heads),
        static_cast<int32_t>(data.num_v_heads),
        static_cast<int32_t>(data.batch_size));
  }
}

extern "C" __global__ __aicore__ void xllm_kernel_mega_gdn_decode(
    GM_ADDR qkv,
    GM_ADDR z,
    GM_ADDR b,
    GM_ADDR a,
    GM_ADDR convWeight,
    GM_ADDR convState,
    GM_ADDR a_log,
    GM_ADDR dtBias,
    GM_ADDR ssmState,
    GM_ADDR readStateIndices,
    GM_ADDR writeStateIndices,
    GM_ADDR normWeight,
    GM_ADDR convOut,
    GM_ADDR convStateOut,
    GM_ADDR ssmStateOut,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling) {
#if defined(PTO_NPU_ARCH_A5)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#else
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
  REGISTER_TILING_DEFAULT(MegaGdnDecodeTilingData);
  GET_TILING_DATA_WITH_STRUCT(MegaGdnDecodeTilingData, tilingData, tiling);
  (void)workspace;

  if constexpr (TILING_KEY_IS(2001)) {
    RunSplitStateDecode<true>(qkv,
                              z,
                              b,
                              a,
                              convWeight,
                              convState,
                              a_log,
                              dtBias,
                              ssmState,
                              readStateIndices,
                              writeStateIndices,
                              normWeight,
                              convOut,
                              convStateOut,
                              ssmStateOut,
                              out,
                              tilingData);
  } else if constexpr (TILING_KEY_IS(2011)) {
    RunSplitStateDecode<false>(qkv,
                               z,
                               b,
                               a,
                               convWeight,
                               convState,
                               a_log,
                               dtBias,
                               ssmState,
                               readStateIndices,
                               writeStateIndices,
                               normWeight,
                               convOut,
                               convStateOut,
                               ssmStateOut,
                               out,
                               tilingData);
  } else if constexpr (TILING_KEY_IS(2)) {
    RunMegaGdnDecode<true, true>(qkv,
                                 z,
                                 b,
                                 a,
                                 convWeight,
                                 convState,
                                 a_log,
                                 dtBias,
                                 ssmState,
                                 readStateIndices,
                                 writeStateIndices,
                                 normWeight,
                                 convOut,
                                 convStateOut,
                                 ssmStateOut,
                                 out,
                                 static_cast<int32_t>(tilingData.num_k_heads),
                                 static_cast<int32_t>(tilingData.num_v_heads),
                                 1);
  } else if constexpr (TILING_KEY_IS(1)) {
    const int32_t batchSize = static_cast<int32_t>(tilingData.batch_size);
    RunMegaGdnDecode<false, true>(qkv,
                                  z,
                                  b,
                                  a,
                                  convWeight,
                                  convState,
                                  a_log,
                                  dtBias,
                                  ssmState,
                                  readStateIndices,
                                  writeStateIndices,
                                  normWeight,
                                  convOut,
                                  convStateOut,
                                  ssmStateOut,
                                  out,
                                  static_cast<int32_t>(tilingData.num_k_heads),
                                  static_cast<int32_t>(tilingData.num_v_heads),
                                  batchSize);
#if defined(PTO_NPU_ARCH_A5)
  } else if constexpr (TILING_KEY_IS(101)) {
    const int32_t batchSize = static_cast<int32_t>(tilingData.batch_size);
    RunMegaGdnDecode<false, true, true>(
        qkv,
        z,
        b,
        a,
        convWeight,
        convState,
        a_log,
        dtBias,
        ssmState,
        readStateIndices,
        writeStateIndices,
        normWeight,
        convOut,
        convStateOut,
        ssmStateOut,
        out,
        static_cast<int32_t>(tilingData.num_k_heads),
        static_cast<int32_t>(tilingData.num_v_heads),
        batchSize);
  } else if constexpr (TILING_KEY_IS(102)) {
    const int32_t batchSize = static_cast<int32_t>(tilingData.batch_size);
    RunMegaGdnDecode<false, true, true, true>(
        qkv,
        z,
        b,
        a,
        convWeight,
        convState,
        a_log,
        dtBias,
        ssmState,
        readStateIndices,
        writeStateIndices,
        normWeight,
        convOut,
        convStateOut,
        ssmStateOut,
        out,
        static_cast<int32_t>(tilingData.num_k_heads),
        static_cast<int32_t>(tilingData.num_v_heads),
        batchSize);
  } else if constexpr (TILING_KEY_IS(103)) {
    const int32_t batchSize = static_cast<int32_t>(tilingData.batch_size);
    RunMegaGdnDecode<false, true, true, true>(
        qkv,
        z,
        b,
        a,
        convWeight,
        convState,
        a_log,
        dtBias,
        ssmState,
        readStateIndices,
        writeStateIndices,
        normWeight,
        convOut,
        convStateOut,
        ssmStateOut,
        out,
        static_cast<int32_t>(tilingData.num_k_heads),
        static_cast<int32_t>(tilingData.num_v_heads),
        batchSize);
#endif
  } else if constexpr (TILING_KEY_IS(12)) {
    RunMegaGdnDecode<true, false>(qkv,
                                  z,
                                  b,
                                  a,
                                  convWeight,
                                  convState,
                                  a_log,
                                  dtBias,
                                  ssmState,
                                  readStateIndices,
                                  writeStateIndices,
                                  normWeight,
                                  convOut,
                                  convStateOut,
                                  ssmStateOut,
                                  out,
                                  static_cast<int32_t>(tilingData.num_k_heads),
                                  static_cast<int32_t>(tilingData.num_v_heads),
                                  1);
  } else if constexpr (TILING_KEY_IS(11)) {
    const int32_t batchSize = static_cast<int32_t>(tilingData.batch_size);
    RunMegaGdnDecode<false, false>(qkv,
                                   z,
                                   b,
                                   a,
                                   convWeight,
                                   convState,
                                   a_log,
                                   dtBias,
                                   ssmState,
                                   readStateIndices,
                                   writeStateIndices,
                                   normWeight,
                                   convOut,
                                   convStateOut,
                                   ssmStateOut,
                                   out,
                                   static_cast<int32_t>(tilingData.num_k_heads),
                                   static_cast<int32_t>(tilingData.num_v_heads),
                                   batchSize);
  }
}

// The generated mixed-kernel wrapper calls matmul::clearWorkspace. Keep this
// include after PTO code to avoid the CANN DYNAMIC/pto::DYNAMIC name collision.
#include "lib/matmul_intf.h"
