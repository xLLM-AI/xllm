// Preserve upstream main for non-A5 targets.
#if defined(GDN_PREFILL_TARGET_A5) ||                  \
    (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || \
    (defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
#include "arch35/mega_gdn_mtp_decode_pto_kernel.h"
#else
/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

// Reuse the verified PTO tile aliases and vector primitives from the
// single-token kernel. The MTP schedule and state contract remain independent.
#include "mega_gdn_decode_pto_kernel.h"

namespace mega_gdn_mtp_decode_pto {

using namespace pto;
using mega_gdn_decode_pto::TileUbDataDN;
using mega_gdn_decode_pto::TileUbDataND;

constexpr int32_t kHeadDim = 128;
constexpr uint16_t kStatePublishAivFlag = 13;
constexpr int32_t kSsmHeadElements = kHeadDim * kHeadDim;
constexpr int32_t kMaxBatchSize = 32;
constexpr int32_t kMaxNumKHeads = 16;
constexpr int32_t kMaxNumVHeads = 64;
constexpr int32_t kMaxConvDim = (2 * kMaxNumKHeads + kMaxNumVHeads) * kHeadDim;
constexpr int32_t kQkGroupCacheSequenceLength = 9;

#if defined(PTO_NPU_ARCH_A5)
// Kept for mega_gdn_draft_decode, which shares this header and still uses its
// existing AIV-only software barrier. mega_gdn_mtp_decode does not call it.
constexpr uint32_t kSoftSyncUbAddress = 188 * 1024;

#if defined(__DAV_VEC__) || defined(__DAV_C220_VEC__) || \
    defined(__DAV_C310_VEC__)
AICORE PTO_INLINE void SyncAllAivSoft(GM_ADDR sync_workspace,
                                      int32_t used_aiv_cores,
                                      int32_t aiv_index) {
  pipe_barrier(PIPE_ALL);
  pto::SYNCALL_SOFT_AIV_BARRIER(
      reinterpret_cast<__gm__ int32_t*>(sync_workspace),
      reinterpret_cast<__ubuf__ int32_t*>(kSoftSyncUbAddress),
      used_aiv_cores,
      aiv_index);
  pipe_barrier(PIPE_ALL);
}
#endif

AICORE PTO_INLINE void SyncAllMixA5() {
  // Match the validated PTO MegaGDN A5 protocol: each AIC gathers its two
  // paired AIVs, all AICs enter the hardware barrier, then release both AIVs.
  pto::SYNCALL<pto::SyncCoreType::Mix>();
}
#endif

#if !defined(PTO_NPU_ARCH_A5) && \
    (defined(__DAV_VEC__) || defined(__DAV_C220_VEC__))
AICORE PTO_INLINE void SyncAllAivStatePublish() {
  pipe_barrier(PIPE_ALL);
  ffts_cross_core_sync(
      PIPE_MTE3, mega_gdn_decode_pto::GetFftsMessage(0, kStatePublishAivFlag));
  wait_flag_dev(kStatePublishAivFlag);
}
#endif

namespace ub_layout {

// All values are byte offsets in UB. The regions are intentionally reused
// only after the named lifetime ends:
//
// Conv phase:
// - kUbConvWeightHalf0..3: BF16[128] each, one channel tile.
// - kUbConvWeight0..3: FP32[128] each, one channel tile.
// - kUbConvHistoryHalf0..2: BF16[128] each, one batch row.
// - kUbConvHistory0..2: FP32[128] each, one batch row.
// - kUbConvInput{Half,HalfPong}: BF16[128], one token.
// - kUbConvInput/kUbConvAcc/kUbConvTmp/kUbConvOutput: FP32[128], one token.
// - kUbConvOutput{Half,HalfPong}: BF16[128], one token GM hand-off.
//
// Recurrent phase:
// - kUbQHalf/kUbKHalf/kUbVHalf: BF16[128], one token.
// - kUbAHalf/kUbBHalf: BF16[16] scalar lanes, one token.
// - kUbQ/kUbK/kUbV: FP32[128], one normalized/current token.
// - kUbScalar/kUbScalar2: FP32[8], a/b scalars for one token.
// - kUbScalarTmp/kUbScalarWork: FP32[8], g/beta scratch for one step.
// - kUbExpA: BF16[16], g/beta rounding scratch for one step.
// - kUbState: FP32[128,128], persistent for one head across all tokens.
// - kUbStateCompute: FP32[128,128], one recurrent projection/update.
// - kUbPrediction/kUbDelta: FP32[128], one recurrent step.
// - kUbColumnSumTmp: FP32[32,128], one reduction.
//
// Norm phase:
// - kUbOutputHalf/kUbNormHalf/kUbZHalf: BF16[128], one token.
// - kUbNormWeightHalf: BF16[128], one head.
// - kUbNorm/kUbZ/kUbSquare/kUbGate: FP32[128], one token.
// - kUbNormWeight: FP32[128], one head.
// - kUbRms: FP32[8], one RMS scalar; kUbFinalHalf: BF16[128] output.
// - kUbVectorScratch: FP32[128], one Conv/Norm vector operation.
// - kUbNormSquare: FP32[128], one Q/K normalization.
// - kUbNormValue: FP32[8], one Q/K norm scalar.
// - kUbReduceTmp: UINT8[128,64], one Q/K or RMS reduction.
// - kUbALogStatic/kUbDtBiasStatic: FP32[8], one head.
//
// Specialized tails:
// - kUbQkBatch{Q,K}Half: BF16[9,128], reuse Conv weights for one Q/K group.
// - kUbQkNormSquare: FP32[9,128]; kUbQkNormReduceTmp: FP32[9,64].
// - kUbQkNorm/kUbQkNormSqrt: FP32[16] each. These Q/K Norm buffers reuse
//   State..V before the initial SSM State is loaded.
// - kUbQkCacheTail/kUbQkCacheK: FP32[9,128], one K8 Q/K group.
// - kUbDeferredReadoutHalf/kUbDeferredZHalf: BF16[9,128], survive all
//   recurrent steps until the deferred Norm phase.
constexpr int32_t kUbConvWeightHalf0 = 0;
constexpr int32_t kUbConvWeightHalf1 = 256;
constexpr int32_t kUbConvWeightHalf2 = 512;
constexpr int32_t kUbConvWeightHalf3 = 768;
constexpr int32_t kUbConvWeight0 = 1024;
constexpr int32_t kUbConvWeight1 = 1536;
constexpr int32_t kUbConvWeight2 = 2048;
constexpr int32_t kUbConvWeight3 = 2560;
constexpr int32_t kUbConvHistoryHalf0 = 3072;
constexpr int32_t kUbConvHistoryHalf1 = 3328;
constexpr int32_t kUbConvHistoryHalf2 = 3584;
constexpr int32_t kUbConvInputHalf = 3840;
constexpr int32_t kUbConvHistory0 = 4096;
constexpr int32_t kUbConvHistory1 = 4608;
constexpr int32_t kUbConvHistory2 = 5120;
constexpr int32_t kUbConvInput = 5632;
constexpr int32_t kUbConvAcc = 6144;
constexpr int32_t kUbConvTmp = 6656;
constexpr int32_t kUbConvOutput = 7168;
constexpr int32_t kUbConvOutputHalf = 7680;
constexpr int32_t kUbConvInputHalfPong = 7936;
constexpr int32_t kUbConvOutputHalfPong = 8192;
constexpr int32_t kUbQHalf = 8704;
constexpr int32_t kUbKHalf = 8960;
constexpr int32_t kUbAHalf = 9216;
constexpr int32_t kUbBHalf = 9248;
constexpr int32_t kUbQ = 9280;
constexpr int32_t kUbK = 9792;
constexpr int32_t kUbScalar = 10304;
constexpr int32_t kUbScalar2 = 10336;
constexpr int32_t kUbNormSquare = 10368;
constexpr int32_t kUbNormValue = 10880;
constexpr int32_t kUbReduceTmp = 10912;
constexpr int32_t kUbScalarTmp = 19104;
constexpr int32_t kUbExpA = 19136;
constexpr int32_t kUbScalarWork = 19168;
constexpr int32_t kUbVHalf = 19200;
constexpr int32_t kUbState = 19456;
constexpr int32_t kUbV = 84992;
constexpr int32_t kUbStateCompute = 85504;
constexpr int32_t kUbPrediction = 151040;
constexpr int32_t kUbDelta = 151552;
constexpr int32_t kUbOutputHalf = 152064;
constexpr int32_t kUbNormHalf = 152320;
constexpr int32_t kUbZHalf = 152576;
constexpr int32_t kUbNormWeightHalf = 152832;
constexpr int32_t kUbNorm = 153088;
constexpr int32_t kUbZ = 153600;
constexpr int32_t kUbNormWeight = 154112;
constexpr int32_t kUbSquare = 154624;
constexpr int32_t kUbRms = 155136;
constexpr int32_t kUbGate = 155168;
constexpr int32_t kUbFinalHalf = 155680;
constexpr int32_t kUbVectorScratch = 155936;
constexpr int32_t kUbColumnSumTmp = 156672;
constexpr int32_t kUbALogStatic = 173824;
constexpr int32_t kUbDtBiasStatic = 174080;
constexpr int32_t kUbQkCacheTail = 174112;
constexpr int32_t kQkCacheBatchBytes = kQkGroupCacheSequenceLength * kHeadDim *
                                       static_cast<int32_t>(sizeof(float));
constexpr int32_t kUbQkCacheK = kUbQkCacheTail + kQkCacheBatchBytes;
constexpr int32_t kUbQkCacheEnd = kUbQkCacheK + kQkCacheBatchBytes;
constexpr int32_t kDeferredNormRows = kQkGroupCacheSequenceLength;
constexpr int32_t kUbDeferredReadoutHalf = kUbQkCacheEnd;
constexpr int32_t kUbDeferredZHalf =
    kUbDeferredReadoutHalf +
    kQkGroupCacheSequenceLength * kHeadDim * sizeof(bfloat16_t);
constexpr int32_t kUbDeferredRowsEnd =
    kUbDeferredZHalf +
    kQkGroupCacheSequenceLength * kHeadDim * sizeof(bfloat16_t);
constexpr int32_t kUbQkBatchQHalf = kUbConvWeightHalf0;
constexpr int32_t kUbQkBatchKHalf =
    kUbQkBatchQHalf +
    kQkGroupCacheSequenceLength * kHeadDim * sizeof(bfloat16_t);
constexpr int32_t kUbQkNormSquare = kUbState;
constexpr int32_t kUbQkNormReduceTmp =
    kUbQkNormSquare + kQkGroupCacheSequenceLength * kHeadDim * sizeof(float);
constexpr int32_t kUbQkNorm =
    kUbQkNormReduceTmp + kQkGroupCacheSequenceLength * 64 * sizeof(float);
constexpr int32_t kUbQkNormSqrt = kUbQkNorm + 16 * sizeof(float);

static_assert(kUbQHalf % 32 == 0);
static_assert(kUbKHalf == kUbQHalf + kHeadDim * sizeof(bfloat16_t));
static_assert(kUbAHalf >= kUbQHalf + 2 * kHeadDim * sizeof(bfloat16_t));
static_assert(kUbQkCacheTail == kUbDtBiasStatic + 8 * sizeof(float));
static_assert(kUbQkCacheEnd == 183328);
static_assert(kUbDeferredRowsEnd == 187936);
static_assert(kUbDeferredRowsEnd <= 192 * 1024);
static_assert(kUbQkNormSqrt + 16 * sizeof(float) <= kUbV);

}  // namespace ub_layout

using namespace ub_layout;

#if !defined(PTO_NPU_ARCH_A5)
template <typename T, int32_t Rows, int32_t Cols>
AICORE PTO_INLINE void SiluNoCopy(TileUbDataND<T, Rows, Cols>& dst,
                                  TileUbDataND<T, Rows, Cols>& src,
                                  TileUbDataND<T, Rows, Cols>& tmp) {
  TMULS(tmp, src, -1);
  mega_gdn_decode_pto::VectorBarrier();
  TEXP(tmp, tmp);
  mega_gdn_decode_pto::VectorBarrier();
  TADDS(tmp, tmp, 1);
  mega_gdn_decode_pto::VectorBarrier();
  TRECIP(dst, tmp);
  mega_gdn_decode_pto::VectorBarrier();
  TMUL(dst, src, dst);
}
#endif

AICORE PTO_INLINE void LoadBf16Row(__gm__ bfloat16_t* handle,
                                   int32_t ub_address) {
  mega_gdn_decode_pto::CopyGmToUb<bfloat16_t,
                                  bfloat16_t,
                                  1,
                                  1,
                                  1,
                                  1,
                                  128,
                                  1,
                                  1,
                                  1,
                                  128,
                                  1,
                                  1,
                                  128,
                                  pto::PadValue::Zero>(
      handle, ub_address, 0, 1, 128);
}

AICORE PTO_INLINE void LoadQkBf16Rows(__gm__ bfloat16_t* handle,
                                      int32_t ub_address,
                                      int32_t row_stride) {
  using QkShape = pto::Shape<1, 1, 1, 2, kHeadDim>;
  using QkStride = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;
  QkShape shape;
  QkStride stride(row_stride);
  pto::GlobalTensor<bfloat16_t, QkShape, QkStride> tensor(
      handle, shape, stride);
  TileUbDataND<bfloat16_t, 2, kHeadDim> tile;
  TASSIGN(tile, ub_address);
  TLOAD(tile, tensor);
}

template <int32_t Rows>
AICORE PTO_INLINE void LoadStridedBf16Rows(__gm__ bfloat16_t* handle,
                                           int32_t ub_address,
                                           int32_t row_stride) {
  using RowShape = pto::Shape<1, 1, 1, Rows, kHeadDim>;
  using RowStride = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;
  RowShape shape;
  RowStride stride(row_stride);
  pto::GlobalTensor<bfloat16_t, RowShape, RowStride> tensor(
      handle, shape, stride);
  TileUbDataND<bfloat16_t, Rows, kHeadDim> tile;
  TASSIGN(tile, ub_address);
  TLOAD(tile, tensor);
}

AICORE PTO_INLINE void StoreBf16Row(__gm__ bfloat16_t* handle,
                                    int32_t ub_address) {
  mega_gdn_decode_pto::CopyUbToGm<bfloat16_t,
                                  bfloat16_t,
                                  1,
                                  1,
                                  1,
                                  1,
                                  128,
                                  1,
                                  1,
                                  1,
                                  128,
                                  1,
                                  1,
                                  128>(handle, ub_address, 0, 1, 128);
}

template <int32_t Rows>
AICORE PTO_INLINE void StoreStridedBf16Rows(__gm__ bfloat16_t* handle,
                                            int32_t ub_address,
                                            int32_t row_stride) {
  using RowShape = pto::Shape<1, 1, 1, Rows, kHeadDim>;
  using RowStride = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;
  RowShape shape;
  RowStride stride(row_stride);
  pto::GlobalTensor<bfloat16_t, RowShape, RowStride> tensor(
      handle, shape, stride);
  TileUbDataND<bfloat16_t, Rows, kHeadDim> tile;
  TASSIGN(tile, ub_address);
  TSTORE(tensor, tile);
}

AICORE PTO_INLINE void LoadBf16Scalar(__gm__ bfloat16_t* handle,
                                      int32_t ub_address) {
  mega_gdn_decode_pto::CopyGmToUb<bfloat16_t,
                                  bfloat16_t,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  16,
                                  pto::PadValue::Null>(
      handle, ub_address, 0, 1, 1);
}

AICORE PTO_INLINE void LoadFloatScalar(__gm__ float* handle,
                                       int32_t ub_address) {
  mega_gdn_decode_pto::CopyGmToUb<float,
                                  float,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  8,
                                  pto::PadValue::Null>(
      handle, ub_address, 0, 1, 1);
}

AICORE PTO_INLINE void LoadState(__gm__ float* handle, int32_t ub_address) {
  mega_gdn_decode_pto::CopyGmToUb<float,
                                  float,
                                  1,
                                  1,
                                  1,
                                  128,
                                  128,
                                  1,
                                  1,
                                  1,
                                  128,
                                  1,
                                  128,
                                  128,
                                  pto::PadValue::Zero>(
      handle, ub_address, 0, 128, 128);
}

AICORE PTO_INLINE void StoreState(__gm__ float* handle, int32_t ub_address) {
  mega_gdn_decode_pto::
      CopyUbToGm<float, float, 1, 1, 1, 128, 128, 1, 1, 1, 128, 1, 128, 128>(
          handle, ub_address, 0, 128, 128);
}

template <bool ApplyQScale, int32_t Rows>
AICORE PTO_INLINE void NormalizeQkRows(int32_t rows_address,
                                       int32_t square_address,
                                       int32_t reduce_tmp_address,
                                       int32_t norm_address,
                                       int32_t norm_sqrt_address) {
  constexpr int32_t kNormCapacity = 16;
  TileUbDataND<float, Rows, kHeadDim> rows;
  TASSIGN(rows, rows_address);
  TileUbDataND<float, Rows, kHeadDim> square;
  TASSIGN(square, square_address);
  TileUbDataND<float, Rows, 64> reduce_tmp;
  TASSIGN(reduce_tmp, reduce_tmp_address);
  TileUbDataDN<float, kNormCapacity, 1, Rows, 1> norm_dn;
  TASSIGN(norm_dn, norm_address);
  TileUbDataND<float, 1, kNormCapacity, 1, Rows> norm;
  TASSIGN(norm, norm_address);
  TileUbDataND<float, 1, kNormCapacity, 1, Rows> norm_sqrt;
  TASSIGN(norm_sqrt, norm_sqrt_address);

  TMUL(square, rows, rows);
  mega_gdn_decode_pto::VectorBarrier();
  TROWSUM(norm_dn, square, reduce_tmp);
  mega_gdn_decode_pto::VectorBarrier();
  TADDS(norm, norm, 1.0e-6f);
  mega_gdn_decode_pto::VectorBarrier();
  TSQRT(norm_sqrt, norm);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  for (int32_t row = 0; row < Rows; ++row) {
    TileUbDataND<float, 1, kHeadDim> row_tile;
    TASSIGN(row_tile, rows_address + row * kHeadDim * sizeof(float));
    const float row_norm = norm_sqrt.GetValue(row);
#if defined(PTO_NPU_ARCH_A5)
    TDIVS(row_tile, row_tile, row_norm);
#else
    // The in-place 910B TDIVS path becomes reciprocal+vmuls. Reuse this
    // row's dead square tile as a non-alias destination to match TileLang's
    // vector divf rounding.
    TileUbDataND<float, 1, kHeadDim> square_row_tile;
    TASSIGN(square_row_tile, square_address + row * kHeadDim * sizeof(float));
    TDIVS(square_row_tile, row_tile, row_norm);
    mega_gdn_decode_pto::VectorBarrier();
    TMOV(row_tile, square_row_tile);
#endif
  }
  mega_gdn_decode_pto::VectorBarrier();
  if constexpr (ApplyQScale) {
#if defined(PTO_NPU_ARCH_A5)
    // Exact FP32 rounding of 1 / sqrt(128), matching Triton's scale.
    TMULS(rows, rows, 0.0883883461356163f);
#else
    TMULS(rows, rows, 8.838835e-02f);
#endif
    mega_gdn_decode_pto::VectorBarrier();
  }
}

template <int32_t SpeculativeTokens, bool PackedVarlen = false>
AICORE PTO_INLINE void RunConvPhase(__gm__ bfloat16_t* qkv_handle,
                                    __gm__ bfloat16_t* conv_weight_handle,
                                    __gm__ bfloat16_t* conv_state_handle,
                                    __gm__ int* read_state_indices_handle,
                                    __gm__ int* write_state_indices_handle,
                                    __gm__ int* num_accepted_tokens_handle,
                                    __gm__ bfloat16_t* conv_out_handle,
                                    __gm__ bfloat16_t* conv_state_out_handle,
                                    int32_t sequence_length,
                                    int32_t batch_size,
                                    int32_t conv_dim,
                                    int32_t conv_tile_count,
                                    int32_t conv_state_stride,
                                    int32_t vector_core_idx,
                                    int32_t vector_core_count,
                                    __gm__ int32_t* cu_seqlens_handle = nullptr,
                                    int32_t state_index_width = 1) {
  TileUbDataND<bfloat16_t, 1, 128> w_half0;
  TASSIGN(w_half0, kUbConvWeightHalf0);
  TileUbDataND<bfloat16_t, 1, 128> w_half1;
  TASSIGN(w_half1, kUbConvWeightHalf1);
  TileUbDataND<bfloat16_t, 1, 128> w_half2;
  TASSIGN(w_half2, kUbConvWeightHalf2);
  TileUbDataND<bfloat16_t, 1, 128> w_half3;
  TASSIGN(w_half3, kUbConvWeightHalf3);
  TileUbDataND<float, 1, 128> w0;
  TASSIGN(w0, kUbConvWeight0);
  TileUbDataND<float, 1, 128> w1;
  TASSIGN(w1, kUbConvWeight1);
  TileUbDataND<float, 1, 128> w2;
  TASSIGN(w2, kUbConvWeight2);
  TileUbDataND<float, 1, 128> w3;
  TASSIGN(w3, kUbConvWeight3);
  TileUbDataND<bfloat16_t, 1, 128> hist_half0;
  TASSIGN(hist_half0, kUbConvHistoryHalf0);
  TileUbDataND<bfloat16_t, 1, 128> hist_half1;
  TASSIGN(hist_half1, kUbConvHistoryHalf1);
  TileUbDataND<bfloat16_t, 1, 128> hist_half2;
  TASSIGN(hist_half2, kUbConvHistoryHalf2);
  TileUbDataND<bfloat16_t, 1, 128> x_half;
  TASSIGN(x_half, kUbConvInputHalf);
  TileUbDataND<float, 1, 128> hist0;
  TASSIGN(hist0, kUbConvHistory0);
  TileUbDataND<float, 1, 128> hist1;
  TASSIGN(hist1, kUbConvHistory1);
  TileUbDataND<float, 1, 128> hist2;
  TASSIGN(hist2, kUbConvHistory2);
  TileUbDataND<float, 1, 128> x_fp32;
  TASSIGN(x_fp32, kUbConvInput);
  TileUbDataND<float, 1, 128> conv_acc;
  TASSIGN(conv_acc, kUbConvAcc);
  TileUbDataND<float, 1, 128> conv_tmp;
  TASSIGN(conv_tmp, kUbConvTmp);
  TileUbDataND<float, 1, 128> conv_y;
  TASSIGN(conv_y, kUbConvOutput);
  TileUbDataND<bfloat16_t, 1, 128> conv_y_half;
  TASSIGN(conv_y_half, kUbConvOutputHalf);

  for (int32_t conv_tile = vector_core_idx; conv_tile < conv_tile_count;
       conv_tile += vector_core_count) {
    const int32_t channel_offset = conv_tile * kHeadDim;
    mega_gdn_decode_pto::CopyConvWeightsGmToUb(
        conv_weight_handle + channel_offset, kUbConvWeightHalf0, conv_dim);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(w0, w_half0, RoundMode::CAST_NONE);
    TCVT(w1, w_half1, RoundMode::CAST_NONE);
    TCVT(w2, w_half2, RoundMode::CAST_NONE);
    TCVT(w3, w_half3, RoundMode::CAST_NONE);
    mega_gdn_decode_pto::VectorBarrier();

    int8_t conv_pingpong_flag;
    if constexpr (SpeculativeTokens == 8) {
      conv_pingpong_flag = 0;
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    }

    for (int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      const int32_t token_start = PackedVarlen ? cu_seqlens_handle[batch_idx]
                                               : batch_idx * sequence_length;
      const int32_t token_count =
          PackedVarlen ? cu_seqlens_handle[batch_idx + 1] - token_start
                       : sequence_length;
      if (token_count == 0) {
        continue;
      }
      const int32_t read_slot =
          *(read_state_indices_handle + batch_idx * state_index_width);
      const int32_t write_slot =
          *(write_state_indices_handle + batch_idx * state_index_width);
      if (read_slot < 0 || write_slot < 0) {
        continue;
      }
      const int32_t accepted =
          sequence_length == 1 ? 1 : *(num_accepted_tokens_handle + batch_idx);
      const int64_t read_history_offset =
          static_cast<int64_t>(read_slot) * conv_state_stride +
          (accepted - 1) * conv_dim + channel_offset;
      mega_gdn_decode_pto::CopyConvHistoryGmToUb(
          conv_state_handle + read_history_offset,
          kUbConvHistoryHalf0,
          conv_dim);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TCVT(hist0, hist_half0, RoundMode::CAST_NONE);
      TCVT(hist1, hist_half1, RoundMode::CAST_NONE);
      TCVT(hist2, hist_half2, RoundMode::CAST_NONE);
      mega_gdn_decode_pto::VectorBarrier();

      // Snapshot accepted rows before a same-slot write can overwrite them.
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      StoreBf16Row(conv_state_out_handle +
                       static_cast<int64_t>(write_slot) * conv_state_stride +
                       channel_offset,
                   kUbConvHistoryHalf1);
      StoreBf16Row(conv_state_out_handle +
                       static_cast<int64_t>(write_slot) * conv_state_stride +
                       conv_dim + channel_offset,
                   kUbConvHistoryHalf2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

      for (int32_t token_idx = 0; token_idx < token_count; ++token_idx) {
        const int32_t token_offset =
            (token_start + token_idx) * conv_dim + channel_offset;
        int32_t conv_input_half_address;
        int32_t conv_output_half_address;
        event_t conv_pingpong_event;
        if constexpr (SpeculativeTokens == 8) {
          conv_input_half_address =
              conv_pingpong_flag == 0 ? kUbConvInputHalf : kUbConvInputHalfPong;
          conv_output_half_address = conv_pingpong_flag == 0
                                         ? kUbConvOutputHalf
                                         : kUbConvOutputHalfPong;
          conv_pingpong_event = static_cast<event_t>(conv_pingpong_flag + 2);
          wait_flag(PIPE_MTE3, PIPE_MTE2, conv_pingpong_event);
          LoadBf16Row(qkv_handle + token_offset, conv_input_half_address);
          set_flag(PIPE_MTE2, PIPE_V, conv_pingpong_event);
          wait_flag(PIPE_MTE2, PIPE_V, conv_pingpong_event);
          TileUbDataND<bfloat16_t, 1, 128> conv_input_half;
          TASSIGN(conv_input_half, conv_input_half_address);
          TCVT(x_fp32, conv_input_half, RoundMode::CAST_NONE);
        } else {
          LoadBf16Row(qkv_handle + token_offset, kUbConvInputHalf);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          TCVT(x_fp32, x_half, RoundMode::CAST_NONE);
        }
        mega_gdn_decode_pto::VectorBarrier();

        TMUL(conv_acc, w0, hist0);
#if defined(PTO_NPU_ARCH_A5)
        mega_gdn_decode_pto::VectorBarrier();
        mega_gdn_decode_pto::MulAddDst<float, 1, 128>(
            conv_acc, hist1, w1, conv_tmp);
        mega_gdn_decode_pto::VectorBarrier();
        mega_gdn_decode_pto::MulAddDst<float, 1, 128>(
            conv_acc, hist2, w2, conv_tmp);
        mega_gdn_decode_pto::VectorBarrier();
#else
        TMUL(conv_tmp, w1, hist1);
        mega_gdn_decode_pto::VectorBarrier();
        TADD(conv_acc, conv_acc, conv_tmp);
        mega_gdn_decode_pto::VectorBarrier();
        TMUL(conv_tmp, w2, hist2);
        mega_gdn_decode_pto::VectorBarrier();
        TADD(conv_acc, conv_acc, conv_tmp);
        mega_gdn_decode_pto::VectorBarrier();
#endif
        TileUbDataND<float, 1, 128> mul_add_tmp;
        TASSIGN(mul_add_tmp, kUbVectorScratch);
        mega_gdn_decode_pto::MulAddDst<float, 1, 128>(
            conv_acc, x_fp32, w3, mul_add_tmp);
        mega_gdn_decode_pto::VectorBarrier();
        TileUbDataND<float, 1, 128> silu_tmp;
        TASSIGN(silu_tmp, kUbVectorScratch);
        mega_gdn_decode_pto::CausalConvSilu<float, 1, 128>(
            conv_y, conv_acc, silu_tmp);
        mega_gdn_decode_pto::VectorBarrier();
        // Numerical contract: Conv output is materialized as BF16 before
        // Q/K/V split, matching the unfused CausalConv hand-off.
        if constexpr (SpeculativeTokens == 8) {
          wait_flag(PIPE_MTE3, PIPE_V, conv_pingpong_event);
          TileUbDataND<bfloat16_t, 1, 128> conv_output_half;
          TASSIGN(conv_output_half, conv_output_half_address);
          TCVT(conv_output_half, conv_y, RoundMode::CAST_RINT);
        } else {
          TCVT(conv_y_half, conv_y, RoundMode::CAST_RINT);
        }
        mega_gdn_decode_pto::VectorBarrier();
        if constexpr (SpeculativeTokens == 8) {
          set_flag(PIPE_V, PIPE_MTE3, conv_pingpong_event);
          wait_flag(PIPE_V, PIPE_MTE3, conv_pingpong_event);
          StoreBf16Row(conv_out_handle + token_offset,
                       conv_output_half_address);
          StoreBf16Row(
              conv_state_out_handle +
                  static_cast<int64_t>(write_slot) * conv_state_stride +
                  (token_idx + 2) * conv_dim + channel_offset,
              conv_input_half_address);
          set_flag(PIPE_MTE3, PIPE_MTE2, conv_pingpong_event);
          set_flag(PIPE_MTE3, PIPE_V, conv_pingpong_event);
        } else {
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
          StoreBf16Row(conv_out_handle + token_offset, kUbConvOutputHalf);
          StoreBf16Row(
              conv_state_out_handle +
                  static_cast<int64_t>(write_slot) * conv_state_stride +
                  (token_idx + 2) * conv_dim + channel_offset,
              kUbConvInputHalf);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        }

        TMOV(hist0, hist1);
        TMOV(hist1, hist2);
        TMOV(hist2, x_fp32);
        mega_gdn_decode_pto::VectorBarrier();
        if constexpr (SpeculativeTokens == 8) {
          conv_pingpong_flag = conv_pingpong_flag == 0 ? 1 : 0;
        }
      }
    }
    if constexpr (SpeculativeTokens == 8) {
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    }
  }
}

AICORE PTO_INLINE void LoadInitialState(__gm__ float* ssm_state_handle,
                                        __gm__ bfloat16_t* norm_weight_handle,
                                        __gm__ float* a_log_handle,
                                        __gm__ float* dt_bias_handle,
                                        int32_t read_slot,
                                        int32_t accepted,
                                        int32_t sequence_length,
                                        int32_t ssm_checkpoint_stride,
                                        int32_t head_idx,
                                        bool load_norm_weight) {
  const int32_t read_checkpoint = read_slot * sequence_length + accepted - 1;
  const int64_t read_state_offset =
      static_cast<int64_t>(read_checkpoint) * ssm_checkpoint_stride +
      head_idx * kSsmHeadElements;

  TileUbDataND<bfloat16_t, 1, 128> norm_weight_half;
  TASSIGN(norm_weight_half, kUbNormWeightHalf);
  TileUbDataND<float, 1, 128> norm_weight_fp32;
  TASSIGN(norm_weight_fp32, kUbNormWeight);
  TileUbDataND<float, 1, 8, 1, 1> a_log_static;
  TASSIGN(a_log_static, kUbALogStatic);

  LoadState(ssm_state_handle + read_state_offset, kUbState);
  if (load_norm_weight) {
    LoadBf16Row(norm_weight_handle, kUbNormWeightHalf);
  }
  LoadFloatScalar(a_log_handle + head_idx, kUbALogStatic);
  LoadFloatScalar(dt_bias_handle + head_idx, kUbDtBiasStatic);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  if (load_norm_weight) {
    TCVT(norm_weight_fp32, norm_weight_half, RoundMode::CAST_NONE);
  }
  TEXP(a_log_static, a_log_static);
  mega_gdn_decode_pto::VectorBarrier();
}

template <bool UseDeferredNorm, bool FlaSsmStateLayout>
AICORE PTO_INLINE void RunRecurrentStep(int32_t q_token_address,
                                        int32_t k_token_address,
                                        int32_t deferred_readout_address) {
  TileUbDataND<float, 1, 128> q_token;
  TASSIGN(q_token, q_token_address);
  TileUbDataND<float, 1, 128> k_token;
  TASSIGN(k_token, k_token_address);
  TileUbDataND<float, 1, 8, 1, 1> scalar;
  TASSIGN(scalar, kUbScalar);
  TileUbDataND<float, 1, 8, 1, 1> scalar2;
  TASSIGN(scalar2, kUbScalar2);
  TileUbDataND<float, 1, 8, 1, 1> scalar_tmp;
  TASSIGN(scalar_tmp, kUbScalarTmp);
  TileUbDataND<bfloat16_t, 1, 16, 1, 1> rounded_gate_half;
  TASSIGN(rounded_gate_half, kUbExpA);
  TileUbDataND<float, 1, 8, 1, 1> scalar_work;
  TASSIGN(scalar_work, kUbScalarWork);
  TileUbDataND<float, 1, 8, 1, 1> a_log_static;
  TASSIGN(a_log_static, kUbALogStatic);
  TileUbDataND<float, 1, 8, 1, 1> dt_bias_static;
  TASSIGN(dt_bias_static, kUbDtBiasStatic);
  TileUbDataND<float, 128, 128> state;
  TASSIGN(state, kUbState);
  TileUbDataND<float, 1, 128> v_fp32;
  TASSIGN(v_fp32, kUbV);
  TileUbDataND<float, 128, 128> compute;
  TASSIGN(compute, kUbStateCompute);
  TileUbDataND<float, 1, 128> prediction;
  TASSIGN(prediction, kUbPrediction);
  TileUbDataND<float, 1, 128> delta;
  TASSIGN(delta, kUbDelta);
  TileUbDataND<float, 32, 128> colsum_tmp;
  TASSIGN(colsum_tmp, kUbColumnSumTmp);
  TileUbDataND<bfloat16_t, 1, 128> readout_half;
  TASSIGN(readout_half, kUbOutputHalf);
  TileUbDataND<bfloat16_t, 1, 128> norm_half;
  TASSIGN(norm_half, kUbNormHalf);

  // decay = exp(-exp(a_log) * softplus(a + dt_bias)).
  TADD(scalar_tmp, scalar, dt_bias_static);
  mega_gdn_decode_pto::VectorBarrier();
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  const float gate_input = scalar_tmp.GetValue(0);
  if (gate_input > 20.0f) {
    TMOV(scalar_work, scalar_tmp);
  } else {
#if defined(PTO_NPU_ARCH_A5)
    // Compensated log1p: log(1 + y) alone loses the low bits of y when
    // y = exp(x) is small. Recover the addition residual before applying
    // the BF16 gate rounding used by the unfused path.
    TEXP(scalar_work, scalar_tmp);
    mega_gdn_decode_pto::VectorBarrier();
    TADDS(scalar_tmp, scalar_work, 1.0f);
    mega_gdn_decode_pto::VectorBarrier();
    TADDS(scalar, scalar_tmp, -1.0f);
    mega_gdn_decode_pto::VectorBarrier();
    TSUB(scalar, scalar_work, scalar);
    mega_gdn_decode_pto::VectorBarrier();
    TDIV(scalar, scalar, scalar_tmp);
    mega_gdn_decode_pto::VectorBarrier();
    TLOG(scalar_work, scalar_tmp);
    mega_gdn_decode_pto::VectorBarrier();
    TADD(scalar_work, scalar_work, scalar);
#else
    TEXP(scalar_work, scalar_tmp);
    mega_gdn_decode_pto::VectorBarrier();
    TADDS(scalar_work, scalar_work, 1.0f);
    mega_gdn_decode_pto::VectorBarrier();
    TLOG(scalar_work, scalar_work);
#endif
  }
  mega_gdn_decode_pto::VectorBarrier();
  TMUL(scalar_tmp, a_log_static, scalar_work);
  mega_gdn_decode_pto::VectorBarrier();
  TMULS(scalar_tmp, scalar_tmp, -1.0f);
  mega_gdn_decode_pto::VectorBarrier();
  // Numerical contract: the unfused gate path materializes g as BF16
  // before exp(g); removing this RINT changes accepted-token numerics.
  TCVT(rounded_gate_half, scalar_tmp, RoundMode::CAST_RINT);
  mega_gdn_decode_pto::VectorBarrier();
  TCVT(scalar_tmp, rounded_gate_half, RoundMode::CAST_NONE);
  mega_gdn_decode_pto::VectorBarrier();
  TEXP(scalar_work, scalar_tmp);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  const float decay = scalar_work.GetValue(0);

  // beta = sigmoid(b).
  TMULS(scalar_tmp, scalar2, -1.0f);
  mega_gdn_decode_pto::VectorBarrier();
  TEXP(scalar_work, scalar_tmp);
  mega_gdn_decode_pto::VectorBarrier();
  TADDS(scalar_tmp, scalar_work, 1.0f);
  mega_gdn_decode_pto::VectorBarrier();
  TRECIP(scalar_work, scalar_tmp);
  mega_gdn_decode_pto::VectorBarrier();
  // Numerical contract: torch::sigmoid(BF16) returns BF16 in the unfused
  // path, so beta must RINT before the FP32 recurrent update.
  TCVT(rounded_gate_half, scalar_work, RoundMode::CAST_RINT);
  mega_gdn_decode_pto::VectorBarrier();
  TCVT(scalar_work, rounded_gate_half, RoundMode::CAST_NONE);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  const float beta = scalar_work.GetValue(0);

  TMULS(state, state, decay);
  mega_gdn_decode_pto::VectorBarrier();
  mega_gdn_decode_pto::StateVectorProduct128<FlaSsmStateLayout>(
      prediction, state, k_token, compute, colsum_tmp);
  TSUB(delta, v_fp32, prediction);
  mega_gdn_decode_pto::VectorBarrier();
  TMULS(delta, delta, beta);
  mega_gdn_decode_pto::VectorBarrier();
  mega_gdn_decode_pto::StateRankOneUpdate128<FlaSsmStateLayout>(
      state, k_token, delta, compute);
  mega_gdn_decode_pto::StateVectorProduct128<FlaSsmStateLayout>(
      prediction, state, q_token, compute, colsum_tmp);
  // Numerical contract: recurrent readout is BF16 before RMSNorm, matching
  // the unfused RecurrentGatedDeltaRule -> Norm tensor boundary.
  if constexpr (UseDeferredNorm) {
    TileUbDataND<bfloat16_t, 1, kHeadDim> cached_readout_half;
    TASSIGN(cached_readout_half, deferred_readout_address);
    TCVT(cached_readout_half, prediction, RoundMode::CAST_RINT);
  } else {
    TCVT(readout_half, prediction, RoundMode::CAST_RINT);
  }
  mega_gdn_decode_pto::VectorBarrier();
  if constexpr (!UseDeferredNorm) {
    TMOV(norm_half, readout_half);
  }
}

AICORE PTO_INLINE void WriteCheckpoint(__gm__ float* ssm_state_out_handle,
                                       int32_t write_slot,
                                       int32_t sequence_length,
                                       int32_t ssm_checkpoint_stride,
                                       int32_t head_idx,
                                       int32_t token_idx) {
  const int32_t write_checkpoint = write_slot * sequence_length + token_idx;
  const int64_t write_state_offset =
      static_cast<int64_t>(write_checkpoint) * ssm_checkpoint_stride +
      head_idx * kSsmHeadElements;
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  StoreState(ssm_state_out_handle + write_state_offset, kUbState);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
}

template <int32_t SpeculativeTokens>
AICORE PTO_INLINE void RunNormStep(__gm__ bfloat16_t* out_handle,
                                   int32_t batch_idx,
                                   int32_t token_idx,
                                   int32_t head_idx,
                                   int32_t sequence_length,
                                   int32_t v_width) {
  TileUbDataND<bfloat16_t, 1, 128> norm_half;
  TASSIGN(norm_half, kUbNormHalf);
  TileUbDataND<float, 1, 128> norm_fp32;
  TASSIGN(norm_fp32, kUbNorm);
  TileUbDataND<float, 1, 128> z_fp32;
  TASSIGN(z_fp32, kUbZ);
  TileUbDataND<float, 1, 128> norm_weight_fp32;
  TASSIGN(norm_weight_fp32, kUbNormWeight);
  TileUbDataND<float, 1, 128> square_fp32;
  TASSIGN(square_fp32, kUbSquare);
  TileUbDataND<float, 1, 8, 1, 1> rms;
  TASSIGN(rms, kUbRms);
#if defined(PTO_NPU_ARCH_A5)
  TileUbDataDN<float, 8, 1, 1, 1> rstd_dn;
  TASSIGN(rstd_dn, kUbRms);
#endif
  TileUbDataND<float, 1, 8, 1, 1> scalar_tmp;
  TASSIGN(scalar_tmp, kUbScalarTmp);
#if !defined(PTO_NPU_ARCH_A5)
  TileUbDataND<float, 1, 128> gate_fp32;
  TASSIGN(gate_fp32, kUbGate);
#endif
  TileUbDataND<bfloat16_t, 1, 128> final_half;
  TASSIGN(final_half, kUbFinalHalf);

  TCVT(norm_fp32, norm_half, RoundMode::CAST_NONE);
  mega_gdn_decode_pto::VectorBarrier();
  TMUL(square_fp32, norm_fp32, norm_fp32);
  mega_gdn_decode_pto::VectorBarrier();
  TileUbDataDN<float, 8, 1, 1, 1> rms_dn;
  TASSIGN(rms_dn, kUbRms);
  TileUbDataND<float, 1, 64, 1, 64> rms_reduce_tmp;
  TASSIGN(rms_reduce_tmp, kUbReduceTmp);
#if defined(PTO_NPU_ARCH_A5)
  TileUbDataND<float, 1, 64> square_low;
  TASSIGN(square_low, kUbSquare);
  TileUbDataND<float, 1, 64> square_high;
  TASSIGN(square_high, kUbSquare + 64 * sizeof(float));
  // Match layer_norm_fwd's N=128 reduction tree: add the two 64-wide
  // halves elementwise, then reduce the resulting 64 values.
  TADD(rms_reduce_tmp, square_low, square_high);
  mega_gdn_decode_pto::VectorBarrier();
  TROWSUM(rms_dn, rms_reduce_tmp, square_fp32);
#else
  TROWSUM(rms_dn, square_fp32, rms_reduce_tmp);
#endif
  mega_gdn_decode_pto::VectorBarrier();
  TMULS(rms, rms, 1.0f / 128.0f);
  mega_gdn_decode_pto::VectorBarrier();
  TADDS(rms, rms, 1.0e-6f);
  mega_gdn_decode_pto::VectorBarrier();
  TSQRT(scalar_tmp, rms);
#if defined(PTO_NPU_ARCH_A5)
  mega_gdn_decode_pto::VectorBarrier();
  TMULS(rms, rms, 0.0f);
  mega_gdn_decode_pto::VectorBarrier();
  TADDS(rms, rms, 1.0f);
  mega_gdn_decode_pto::VectorBarrier();
  // layer_norm_fwd obtains rstd with vector Div rather than scalar division.
  TDIV(rms, rms, scalar_tmp);
  mega_gdn_decode_pto::VectorBarrier();
  TROWEXPANDMUL(norm_fp32, norm_fp32, rstd_dn);
  mega_gdn_decode_pto::VectorBarrier();
  TMUL(norm_fp32, norm_fp32, norm_weight_fp32);
  // Match layer_norm_fwd's A5 hot path exactly: multiply by z before the
  // division so FP32 rounding is identical at BF16 midpoint boundaries.
  TMUL(norm_fp32, norm_fp32, z_fp32);
  mega_gdn_decode_pto::VectorBarrier();
  TMULS(z_fp32, z_fp32, -1.0f);
  mega_gdn_decode_pto::VectorBarrier();
  TEXP(z_fp32, z_fp32);
  mega_gdn_decode_pto::VectorBarrier();
  TADDS(z_fp32, z_fp32, 1.0f);
  mega_gdn_decode_pto::VectorBarrier();
  TDIV(norm_fp32, norm_fp32, z_fp32);
  mega_gdn_decode_pto::VectorBarrier();
  // Triton's A5 f32-to-bf16 conversion rounds midpoint ties away from zero.
  TCVT(final_half, norm_fp32, RoundMode::CAST_ROUND);
#else
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  const float rms_value = scalar_tmp.GetValue(0);
  TMULS(norm_fp32, norm_fp32, 1.0f / rms_value);
  mega_gdn_decode_pto::VectorBarrier();
  TMUL(norm_fp32, norm_fp32, norm_weight_fp32);
  TileUbDataND<float, 1, 128> silu_tmp;
  TASSIGN(silu_tmp, kUbVectorScratch);
  mega_gdn_decode_pto::Silu<float, 1, 128>(gate_fp32, z_fp32, silu_tmp);
  mega_gdn_decode_pto::VectorBarrier();
  TMUL(norm_fp32, norm_fp32, gate_fp32);
  mega_gdn_decode_pto::VectorBarrier();
  TCVT(final_half, norm_fp32, RoundMode::CAST_RINT);
#endif
  mega_gdn_decode_pto::VectorBarrier();
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  StoreBf16Row(out_handle +
                   (batch_idx * sequence_length + token_idx) * v_width +
                   head_idx * kHeadDim,
               kUbFinalHalf);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  if constexpr (SpeculativeTokens != 8) {
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  }
}

template <int32_t Rows, bool UseQkGroupCache>
class DeferredNormUbLayout final {
 public:
  // BF16 cached rows live from recurrent readout until deferred Norm.
  static constexpr int32_t kReadoutHalf =
      UseQkGroupCache ? kUbDeferredReadoutHalf : kUbQkCacheTail;
  static constexpr int32_t kZHalf =
      kReadoutHalf + Rows * kHeadDim * sizeof(bfloat16_t);
  static constexpr int32_t kCachedRowsEnd =
      kZHalf + Rows * kHeadDim * sizeof(bfloat16_t);

  // The StateCompute..Prediction range is dead after the recurrent loop and
  // is reused by batched Norm: BF16 input/Z, FP32 input/Z/square/gate,
  // BF16 output, FP32 SiLU/reduction scratch, and FP32 RMS scalars.
  static constexpr int32_t kInputHalfBatch = kUbStateCompute;
  static constexpr int32_t kZHalfBatch =
      kInputHalfBatch + Rows * kHeadDim * sizeof(bfloat16_t);
  static constexpr int32_t kInputBatch =
      kZHalfBatch + Rows * kHeadDim * sizeof(bfloat16_t);
  static constexpr int32_t kZBatch =
      kInputBatch + Rows * kHeadDim * sizeof(float);
  static constexpr int32_t kSquareBatch =
      kZBatch + Rows * kHeadDim * sizeof(float);
  static constexpr int32_t kGateBatch =
      kSquareBatch + Rows * kHeadDim * sizeof(float);
  static constexpr int32_t kFinalHalfBatch =
      kGateBatch + Rows * kHeadDim * sizeof(float);
  static constexpr int32_t kSiluTmpBatch =
      kFinalHalfBatch + Rows * kHeadDim * sizeof(bfloat16_t);
  static constexpr int32_t kReduceTmpBatch =
      kSiluTmpBatch + Rows * kHeadDim * sizeof(float);
  static constexpr int32_t kRmsBatch =
      kReduceTmpBatch + Rows * 64 * sizeof(float);
  static constexpr int32_t kRmsBatchCapacity = Rows <= 16 ? 16 : 32;
  static constexpr int32_t kRmsSqrtBatch =
      kRmsBatch + kRmsBatchCapacity * sizeof(float);
  static constexpr int32_t kWorkEnd =
      kRmsSqrtBatch + kRmsBatchCapacity * sizeof(float);

  static_assert(kCachedRowsEnd <= 192 * 1024);
  static_assert(kWorkEnd <= kUbPrediction);
};

template <int32_t Rows, bool UseQkGroupCache>
AICORE PTO_INLINE void RunDeferredNorm(__gm__ bfloat16_t* out_handle,
                                       int32_t batch_idx,
                                       int32_t head_idx,
                                       int32_t sequence_length,
                                       int32_t v_width) {
  using Layout = DeferredNormUbLayout<Rows, UseQkGroupCache>;
  pipe_barrier(PIPE_ALL);

  // Full-span aliases keep the static UB footprint parser aware of every
  // dynamically addressed row below.
  TileUbDataND<bfloat16_t, Rows, kHeadDim> cached_readout_half_batch;
  TASSIGN(cached_readout_half_batch, Layout::kReadoutHalf);
  TileUbDataND<bfloat16_t, Rows, kHeadDim> cached_z_half_batch;
  TASSIGN(cached_z_half_batch, Layout::kZHalf);
  TileUbDataND<bfloat16_t, Rows, kHeadDim> readout_half_batch;
  TASSIGN(readout_half_batch, Layout::kInputHalfBatch);
  TileUbDataND<bfloat16_t, Rows, kHeadDim> z_half_batch;
  TASSIGN(z_half_batch, Layout::kZHalfBatch);
  TileUbDataND<float, Rows, kHeadDim> input_batch;
  TASSIGN(input_batch, Layout::kInputBatch);
  TileUbDataND<float, Rows, kHeadDim> z_batch;
  TASSIGN(z_batch, Layout::kZBatch);
  TileUbDataND<float, Rows, kHeadDim> square_batch;
  TASSIGN(square_batch, Layout::kSquareBatch);
#if !defined(PTO_NPU_ARCH_A5)
  TileUbDataND<float, Rows, kHeadDim> gate_batch;
  TASSIGN(gate_batch, Layout::kGateBatch);
#endif
  TileUbDataND<bfloat16_t, Rows, kHeadDim> final_half_batch;
  TASSIGN(final_half_batch, Layout::kFinalHalfBatch);
#if !defined(PTO_NPU_ARCH_A5)
  TileUbDataND<float, Rows, kHeadDim> silu_tmp_batch;
  TASSIGN(silu_tmp_batch, Layout::kSiluTmpBatch);
#endif
  TileUbDataDN<float, Layout::kRmsBatchCapacity, 1, Rows, 1> rms_batch_dn;
  TASSIGN(rms_batch_dn, Layout::kRmsBatch);
  TileUbDataND<float, 1, Layout::kRmsBatchCapacity, 1, Rows> rms_batch;
  TASSIGN(rms_batch, Layout::kRmsBatch);
  TileUbDataND<float, 1, Layout::kRmsBatchCapacity, 1, Rows> rms_sqrt_batch;
  TASSIGN(rms_sqrt_batch, Layout::kRmsSqrtBatch);
  TileUbDataND<float, Rows, 64> reduce_tmp_batch;
  TASSIGN(reduce_tmp_batch, Layout::kReduceTmpBatch);
  TileUbDataND<float, 1, 128> norm_weight_fp32;
  TASSIGN(norm_weight_fp32, kUbNormWeight);

  for (int32_t row = 0; row < Rows; ++row) {
    TileUbDataND<bfloat16_t, 1, kHeadDim> cached_readout_row;
    TASSIGN(cached_readout_row,
            Layout::kReadoutHalf + row * kHeadDim * sizeof(bfloat16_t));
    TileUbDataND<bfloat16_t, 1, kHeadDim> cached_z_row;
    TASSIGN(cached_z_row, Layout::kZHalf + row * kHeadDim * sizeof(bfloat16_t));
    TileUbDataND<float, 1, kHeadDim> input_row;
    TASSIGN(input_row, Layout::kInputBatch + row * kHeadDim * sizeof(float));
    TileUbDataND<float, 1, kHeadDim> z_row;
    TASSIGN(z_row, Layout::kZBatch + row * kHeadDim * sizeof(float));
    TCVT(input_row, cached_readout_row, RoundMode::CAST_NONE);
    TCVT(z_row, cached_z_row, RoundMode::CAST_NONE);
  }
  mega_gdn_decode_pto::VectorBarrier();
  TMUL(square_batch, input_batch, input_batch);
  mega_gdn_decode_pto::VectorBarrier();
#if defined(PTO_NPU_ARCH_A5)
  for (int32_t row = 0; row < Rows; ++row) {
    TileUbDataND<float, 1, 64> square_low;
    TASSIGN(square_low, Layout::kSquareBatch + row * kHeadDim * sizeof(float));
    TileUbDataND<float, 1, 64> square_high;
    TASSIGN(square_high,
            Layout::kSquareBatch + (row * kHeadDim + 64) * sizeof(float));
    TileUbDataND<float, 1, 64> reduced_row;
    TASSIGN(reduced_row, Layout::kReduceTmpBatch + row * 64 * sizeof(float));
    TADD(reduced_row, square_low, square_high);
  }
  mega_gdn_decode_pto::VectorBarrier();
  TROWSUM(rms_batch_dn, reduce_tmp_batch, square_batch);
#else
  TROWSUM(rms_batch_dn, square_batch, reduce_tmp_batch);
#endif
  mega_gdn_decode_pto::VectorBarrier();
  TMULS(rms_batch, rms_batch, 1.0f / 128.0f);
  mega_gdn_decode_pto::VectorBarrier();
  TADDS(rms_batch, rms_batch, 1.0e-6f);
  mega_gdn_decode_pto::VectorBarrier();
  TSQRT(rms_sqrt_batch, rms_batch);
#if defined(PTO_NPU_ARCH_A5)
  mega_gdn_decode_pto::VectorBarrier();
  TMULS(rms_batch, rms_batch, 0.0f);
  mega_gdn_decode_pto::VectorBarrier();
  TADDS(rms_batch, rms_batch, 1.0f);
  mega_gdn_decode_pto::VectorBarrier();
  TDIV(rms_batch, rms_batch, rms_sqrt_batch);
  mega_gdn_decode_pto::VectorBarrier();
  TROWEXPANDMUL(input_batch, input_batch, rms_batch_dn);
#else
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  for (int32_t row = 0; row < Rows; ++row) {
    TileUbDataND<float, 1, kHeadDim> input_row;
    TASSIGN(input_row, Layout::kInputBatch + row * kHeadDim * sizeof(float));
    const float rms_value = rms_sqrt_batch.GetValue(row);
    TMULS(input_row, input_row, 1.0f / rms_value);
  }
#endif
  mega_gdn_decode_pto::VectorBarrier();
  for (int32_t row = 0; row < Rows; ++row) {
    TileUbDataND<float, 1, kHeadDim> input_row;
    TASSIGN(input_row, Layout::kInputBatch + row * kHeadDim * sizeof(float));
    TMUL(input_row, input_row, norm_weight_fp32);
  }
  mega_gdn_decode_pto::VectorBarrier();
#if defined(PTO_NPU_ARCH_A5)
  // Preserve layer_norm_fwd's (value * z) / (1 + exp(-z)) operation order.
  TMUL(input_batch, input_batch, z_batch);
  mega_gdn_decode_pto::VectorBarrier();
  TMULS(z_batch, z_batch, -1.0f);
  mega_gdn_decode_pto::VectorBarrier();
  TEXP(z_batch, z_batch);
  mega_gdn_decode_pto::VectorBarrier();
  TADDS(z_batch, z_batch, 1.0f);
  mega_gdn_decode_pto::VectorBarrier();
  TDIV(input_batch, input_batch, z_batch);
  mega_gdn_decode_pto::VectorBarrier();
  // Match the A5 Triton backend's midpoint-tie behavior at the norm boundary.
  TCVT(final_half_batch, input_batch, RoundMode::CAST_ROUND);
#else
  SiluNoCopy<float, Rows, kHeadDim>(gate_batch, z_batch, silu_tmp_batch);
  mega_gdn_decode_pto::VectorBarrier();
  TMUL(input_batch, input_batch, gate_batch);
  mega_gdn_decode_pto::VectorBarrier();
  TCVT(final_half_batch, input_batch, RoundMode::CAST_RINT);
#endif
  mega_gdn_decode_pto::VectorBarrier();

  const int64_t output_base_offset =
      static_cast<int64_t>(batch_idx) * sequence_length * v_width +
      head_idx * kHeadDim;
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  StoreStridedBf16Rows<Rows>(
      out_handle + output_base_offset, Layout::kFinalHalfBatch, v_width);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
}

#if defined(PTO_NPU_ARCH_A5)
template <int32_t SpeculativeTokens,
          bool UseQkGroupCache,
          bool UseDeferredNorm = UseQkGroupCache,
          bool UseTwoOwnerQkGroups = false,
          bool FlaSsmStateLayout = true,
          bool PackedVarlen = false>
#else
template <int32_t SpeculativeTokens,
          bool UseQkGroupCache,
          bool UseDeferredNorm = UseQkGroupCache,
          bool FlaSsmStateLayout = true,
          bool PackedVarlen = false>
#endif
AICORE PTO_INLINE void Run(__gm__ bfloat16_t* qkv_handle,
                           __gm__ bfloat16_t* z_handle,
                           __gm__ bfloat16_t* b_handle,
                           __gm__ bfloat16_t* a_handle,
                           __gm__ bfloat16_t* conv_weight_handle,
                           __gm__ bfloat16_t* conv_state_handle,
                           __gm__ float* a_log_handle,
                           __gm__ float* dt_bias_handle,
                           __gm__ float* ssm_state_handle,
                           __gm__ int* read_state_indices_handle,
                           __gm__ int* write_state_indices_handle,
                           __gm__ int* num_accepted_tokens_handle,
                           __gm__ bfloat16_t* norm_weight_handle,
                           __gm__ bfloat16_t* conv_out_handle,
                           __gm__ bfloat16_t* conv_state_out_handle,
                           __gm__ float* ssm_state_out_handle,
                           __gm__ bfloat16_t* out_handle,
                           int32_t num_k_heads,
                           int32_t num_v_heads,
                           int32_t batch_size,
                           int32_t runtime_sequence_length,
                           __gm__ int32_t* cu_seqlens_handle = nullptr,
                           int32_t state_index_width = 1) {
  constexpr bool kIsDynamic = SpeculativeTokens == 0;
  static_assert(!PackedVarlen || (kIsDynamic && !UseQkGroupCache));
  static_assert(!UseDeferredNorm || !kIsDynamic);
#if defined(PTO_NPU_ARCH_A5)
  static_assert(!UseTwoOwnerQkGroups || UseQkGroupCache);
  static_assert(!UseTwoOwnerQkGroups || SpeculativeTokens == 8);
#endif
  constexpr int32_t kRunDeferredNormRows =
      UseDeferredNorm ? SpeculativeTokens + 1 : 1;
  using RunDeferredLayout =
      DeferredNormUbLayout<kRunDeferredNormRows, UseQkGroupCache>;
  constexpr int32_t kRunDeferredReadoutHalf = RunDeferredLayout::kReadoutHalf;
  constexpr int32_t kRunDeferredZHalf = RunDeferredLayout::kZHalf;
  const int32_t sequence_length =
      kIsDynamic ? runtime_sequence_length : SpeculativeTokens + 1;
  const int32_t conv_state_length = sequence_length + 2;
  const int32_t conv_dim = (2 * num_k_heads + num_v_heads) * kHeadDim;
  const int32_t conv_tile_count = conv_dim / kHeadDim;
  const int32_t conv_state_stride = conv_state_length * conv_dim;
  const int32_t v_heads_per_k = num_v_heads / num_k_heads;
  const int32_t v_width = num_v_heads * kHeadDim;
  const int32_t ssm_checkpoint_stride = num_v_heads * kSsmHeadElements;

  TileUbDataND<bfloat16_t, 1, 128> q_half;
  TASSIGN(q_half, kUbQHalf);
  TileUbDataND<bfloat16_t, 1, 128> k_half;
  TASSIGN(k_half, kUbKHalf);
  TileUbDataND<bfloat16_t, 1, 16, 1, 1> a_half;
  TASSIGN(a_half, kUbAHalf);
  TileUbDataND<bfloat16_t, 1, 16, 1, 1> b_half;
  TASSIGN(b_half, kUbBHalf);
  TileUbDataND<float, 1, 128> q_fp32;
  TASSIGN(q_fp32, kUbQ);
  TileUbDataND<float, 1, 128> k_fp32;
  TASSIGN(k_fp32, kUbK);
  TileUbDataND<float, 1, 8, 1, 1> scalar;
  TASSIGN(scalar, kUbScalar);
  TileUbDataND<float, 1, 8, 1, 1> scalar2;
  TASSIGN(scalar2, kUbScalar2);
  TileUbDataND<float, 1, 128> norm_square;
  TASSIGN(norm_square, kUbNormSquare);
  TileUbDataND<float, 1, 8, 1, 1> norm_value;
  TASSIGN(norm_value, kUbNormValue);
  TileUbDataND<uint8_t, 128, 64> reduce_tmp;
  TASSIGN(reduce_tmp, kUbReduceTmp);
  TileUbDataND<float, 1, 8, 1, 1> scalar_tmp;
  TASSIGN(scalar_tmp, kUbScalarTmp);
  TileUbDataND<bfloat16_t, 1, 128> v_half;
  TASSIGN(v_half, kUbVHalf);
  TileUbDataND<float, 1, 128> v_fp32;
  TASSIGN(v_fp32, kUbV);
  TileUbDataND<bfloat16_t, 1, 128> z_half;
  TASSIGN(z_half, kUbZHalf);
  TileUbDataND<float, 1, 128> z_fp32;
  TASSIGN(z_fp32, kUbZ);
  // Full-span alias keeps the static UB footprint parser aware of the
  // dynamically addressed all-token Q/K sub-tiles.
  TileUbDataND<float, 2 * kQkGroupCacheSequenceLength, 128> qk_cache_tail;
  TASSIGN(qk_cache_tail, kUbQkCacheTail);
  TileUbDataND<bfloat16_t, kRunDeferredNormRows, kHeadDim>
      deferred_readout_half;
  TASSIGN(deferred_readout_half, kRunDeferredReadoutHalf);
  TileUbDataND<bfloat16_t, kRunDeferredNormRows, kHeadDim> deferred_z_half;
  TASSIGN(deferred_z_half, kRunDeferredZHalf);

#if defined(__DAV_VEC__) || defined(__DAV_C220_VEC__)
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5)
  const auto cid = get_block_idx();
  const auto vid = get_subblockid();
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int32_t vector_core_idx = cid * get_subblockdim() + vid;
  const int32_t vector_core_count = get_block_num() * get_subblockdim();
#else
  const int32_t vector_core_idx = get_block_idx();
  const int32_t vector_core_count = get_block_num();
#endif

  // Phase 1: select accepted Conv history and write the private window.
  RunConvPhase<SpeculativeTokens, PackedVarlen>(qkv_handle,
                                                conv_weight_handle,
                                                conv_state_handle,
                                                read_state_indices_handle,
                                                write_state_indices_handle,
                                                num_accepted_tokens_handle,
                                                conv_out_handle,
                                                conv_state_out_handle,
                                                sequence_length,
                                                batch_size,
                                                conv_dim,
                                                conv_tile_count,
                                                conv_state_stride,
                                                vector_core_idx,
                                                vector_core_count,
                                                cu_seqlens_handle,
                                                state_index_width);
#endif

  // Conv output is a GM hand-off between different channel/head owners.
#if defined(PTO_NPU_ARCH_A5)
  SyncAllMixA5();
#elif defined(__DAV_VEC__) || defined(__DAV_C220_VEC__)
  mega_gdn_decode_pto::SyncAllAiv();
#endif

#if defined(__DAV_VEC__) || defined(__DAV_C220_VEC__)
#if defined(PTO_NPU_ARCH_A5)
  // Phase 2: each owner keeps one complete FP32 state in UB for all S steps.
  // Key 208 assigns all value heads in one Q/K group to one owner. A5 key 308
  // assigns heads 0/1 to the group owner and head 2 to a singleton owner.
  // With the A5 1:2 MIX geometry, 28 AICs launch 56 AIVs, so eight singleton
  // owners process a second group.
  static_assert(!UseQkGroupCache || SpeculativeTokens == 8);
  const int32_t total_heads = batch_size * num_v_heads;
  const int32_t qk_group_count = batch_size * num_k_heads;
  const bool use_two_owner_schedule = UseTwoOwnerQkGroups &&
                                      v_heads_per_k == 3 &&
                                      vector_core_count > qk_group_count;
  int32_t owner_task_count = 0;
  if constexpr (UseTwoOwnerQkGroups) {
    if (use_two_owner_schedule) {
      if (vector_core_idx < qk_group_count) {
        owner_task_count = v_heads_per_k - 1;
      } else {
        const int32_t singleton_owner_count =
            vector_core_count - qk_group_count;
        const int32_t first_singleton_group = vector_core_idx - qk_group_count;
        if (first_singleton_group < qk_group_count) {
          owner_task_count = (qk_group_count - first_singleton_group +
                              singleton_owner_count - 1) /
                             singleton_owner_count;
        }
      }
    } else if (vector_core_idx < qk_group_count) {
      owner_task_count = v_heads_per_k;
    }
  } else if constexpr (UseQkGroupCache) {
    owner_task_count = vector_core_idx < qk_group_count ? v_heads_per_k : 0;
  } else if (vector_core_idx < total_heads) {
    owner_task_count = (total_heads - vector_core_idx + vector_core_count - 1) /
                       vector_core_count;
  }

  int32_t cached_qk_group = -1;
  for (int32_t owner_task_idx = 0; owner_task_idx < owner_task_count;
       ++owner_task_idx) {
    int32_t qk_group = 0;
    int32_t local_v_head_idx = 0;
    if constexpr (UseTwoOwnerQkGroups) {
      if (use_two_owner_schedule) {
        if (vector_core_idx < qk_group_count) {
          qk_group = vector_core_idx;
          local_v_head_idx = owner_task_idx;
        } else {
          const int32_t singleton_owner_count =
              vector_core_count - qk_group_count;
          qk_group = vector_core_idx - qk_group_count +
                     owner_task_idx * singleton_owner_count;
          local_v_head_idx = v_heads_per_k - 1;
        }
      } else {
        qk_group = vector_core_idx;
        local_v_head_idx = owner_task_idx;
      }
    } else if constexpr (UseQkGroupCache) {
      qk_group = vector_core_idx;
      local_v_head_idx = owner_task_idx;
    } else {
      const int32_t head_index =
          vector_core_idx + owner_task_idx * vector_core_count;
      const int32_t batch_head_idx = head_index % num_v_heads;
      qk_group = (head_index / num_v_heads) * num_k_heads +
                 batch_head_idx / v_heads_per_k;
      local_v_head_idx = batch_head_idx % v_heads_per_k;
    }
    const int32_t batch_idx = qk_group / num_k_heads;
    const int32_t qk_head_idx = qk_group % num_k_heads;
    const int32_t head_idx = qk_head_idx * v_heads_per_k + local_v_head_idx;
    const bool load_norm_weight = !UseQkGroupCache || owner_task_idx == 0;

    if constexpr (UseQkGroupCache) {
      if (qk_group != cached_qk_group) {
        const int32_t batch_conv_offset =
            batch_idx * sequence_length * conv_dim;
        LoadStridedBf16Rows<kDeferredNormRows>(
            conv_out_handle + batch_conv_offset + qk_head_idx * kHeadDim,
            kUbQkBatchQHalf,
            conv_dim);
        LoadStridedBf16Rows<kDeferredNormRows>(
            conv_out_handle + batch_conv_offset + num_k_heads * kHeadDim +
                qk_head_idx * kHeadDim,
            kUbQkBatchKHalf,
            conv_dim);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TileUbDataND<bfloat16_t, kDeferredNormRows, kHeadDim> q_half_batch;
        TASSIGN(q_half_batch, kUbQkBatchQHalf);
        TileUbDataND<bfloat16_t, kDeferredNormRows, kHeadDim> k_half_batch;
        TASSIGN(k_half_batch, kUbQkBatchKHalf);
        TileUbDataND<float, kDeferredNormRows, kHeadDim> q_batch;
        TASSIGN(q_batch, kUbQkCacheTail);
        TileUbDataND<float, kDeferredNormRows, kHeadDim> k_batch;
        TASSIGN(k_batch, kUbQkCacheK);
        TCVT(q_batch, q_half_batch, RoundMode::CAST_NONE);
        TCVT(k_batch, k_half_batch, RoundMode::CAST_NONE);
        mega_gdn_decode_pto::VectorBarrier();

        NormalizeQkRows<true, kDeferredNormRows>(kUbQkCacheTail,
                                                 kUbQkNormSquare,
                                                 kUbQkNormReduceTmp,
                                                 kUbQkNorm,
                                                 kUbQkNormSqrt);
        NormalizeQkRows<false, kDeferredNormRows>(kUbQkCacheK,
                                                  kUbQkNormSquare,
                                                  kUbQkNormReduceTmp,
                                                  kUbQkNorm,
                                                  kUbQkNormSqrt);
        cached_qk_group = qk_group;
      }
    }
#else
  // Phase 2: each owner keeps one complete FP32 state in UB for all S steps.
  // The K8/B4/NK8/NV24 bucket maps one batch x Q/K group to each of the
  // first 32 AIVs, caches normalized Q/K for all nine tokens, and then
  // processes the group's three value heads without recomputing Q/K.
  static_assert(!UseQkGroupCache || SpeculativeTokens == 8);
  const int32_t total_heads = batch_size * num_v_heads;
  const int32_t recurrent_loop_end =
      UseQkGroupCache ? vector_core_count * v_heads_per_k : total_heads;
  for (int32_t head_index = vector_core_idx; head_index < recurrent_loop_end;
       head_index += vector_core_count) {
    if constexpr (UseQkGroupCache) {
      if (vector_core_idx >= batch_size * num_k_heads) {
        break;
      }
    }
    const int32_t task_head_idx =
        UseQkGroupCache ? (head_index - vector_core_idx) / vector_core_count
                        : 0;
    const int32_t batch_idx = UseQkGroupCache ? vector_core_idx / num_k_heads
                                              : head_index / num_v_heads;
    const int32_t qk_head_idx =
        UseQkGroupCache ? vector_core_idx % num_k_heads
                        : (head_index % num_v_heads) / v_heads_per_k;
    const int32_t head_idx = UseQkGroupCache
                                 ? qk_head_idx * v_heads_per_k + task_head_idx
                                 : head_index % num_v_heads;
    const bool load_norm_weight = !UseQkGroupCache || task_head_idx == 0;

    if constexpr (UseQkGroupCache) {
      if (task_head_idx == 0) {
        const int32_t batch_conv_offset =
            batch_idx * sequence_length * conv_dim;
        LoadStridedBf16Rows<kDeferredNormRows>(
            conv_out_handle + batch_conv_offset + qk_head_idx * kHeadDim,
            kUbQkBatchQHalf,
            conv_dim);
        LoadStridedBf16Rows<kDeferredNormRows>(
            conv_out_handle + batch_conv_offset + num_k_heads * kHeadDim +
                qk_head_idx * kHeadDim,
            kUbQkBatchKHalf,
            conv_dim);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TileUbDataND<bfloat16_t, kDeferredNormRows, kHeadDim> q_half_batch;
        TASSIGN(q_half_batch, kUbQkBatchQHalf);
        TileUbDataND<bfloat16_t, kDeferredNormRows, kHeadDim> k_half_batch;
        TASSIGN(k_half_batch, kUbQkBatchKHalf);
        TileUbDataND<float, kDeferredNormRows, kHeadDim> q_batch;
        TASSIGN(q_batch, kUbQkCacheTail);
        TileUbDataND<float, kDeferredNormRows, kHeadDim> k_batch;
        TASSIGN(k_batch, kUbQkCacheK);
        TCVT(q_batch, q_half_batch, RoundMode::CAST_NONE);
        TCVT(k_batch, k_half_batch, RoundMode::CAST_NONE);
        mega_gdn_decode_pto::VectorBarrier();

        NormalizeQkRows<true, kDeferredNormRows>(kUbQkCacheTail,
                                                 kUbQkNormSquare,
                                                 kUbQkNormReduceTmp,
                                                 kUbQkNorm,
                                                 kUbQkNormSqrt);
        NormalizeQkRows<false, kDeferredNormRows>(kUbQkCacheK,
                                                  kUbQkNormSquare,
                                                  kUbQkNormReduceTmp,
                                                  kUbQkNorm,
                                                  kUbQkNormSqrt);
      }
    }
#endif
    // Cache strides stay fixed even when this round verifies a shorter prefix.
    const int32_t token_start = PackedVarlen ? cu_seqlens_handle[batch_idx]
                                             : batch_idx * sequence_length;
    const int32_t token_count =
        PackedVarlen ? cu_seqlens_handle[batch_idx + 1] - token_start
                     : sequence_length;
    if (token_count == 0) {
      continue;
    }
    const int32_t read_slot =
        *(read_state_indices_handle + batch_idx * state_index_width +
          state_index_width - 1);
    const int32_t write_slot =
        *(write_state_indices_handle + batch_idx * state_index_width +
          state_index_width - 1);
    if (read_slot < 0 || write_slot < 0) {
      TileUbDataND<bfloat16_t, 1, 128> zero_output;
      TASSIGN(zero_output, kUbOutputHalf);
      TEXPANDS(zero_output, static_cast<bfloat16_t>(0));
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      for (int32_t token_idx = 0; token_idx < token_count; ++token_idx) {
        StoreBf16Row(out_handle + (token_start + token_idx) * v_width +
                         head_idx * kHeadDim,
                     kUbOutputHalf);
      }
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      continue;
    }

    const int32_t accepted =
        sequence_length == 1 ? 1 : *(num_accepted_tokens_handle + batch_idx);
    LoadInitialState(ssm_state_handle,
                     norm_weight_handle,
                     a_log_handle,
                     dt_bias_handle,
                     read_slot,
                     accepted,
                     state_index_width == 2 ? 1 : sequence_length,
                     ssm_checkpoint_stride,
                     head_idx,
                     load_norm_weight);

    for (int32_t token_idx = 0; token_idx < token_count; ++token_idx) {
      if constexpr (SpeculativeTokens == 8) {
        if (token_idx == 0) {
          const int32_t conv_token_offset =
              batch_idx * sequence_length * conv_dim;
          if constexpr (!UseQkGroupCache) {
            LoadQkBf16Rows(
                conv_out_handle + conv_token_offset + qk_head_idx * kHeadDim,
                kUbQHalf,
                num_k_heads * kHeadDim);
          }
          LoadBf16Row(conv_out_handle + conv_token_offset +
                          2 * num_k_heads * kHeadDim + head_idx * kHeadDim,
                      kUbVHalf);
          const int32_t scalar_offset =
              batch_idx * sequence_length * num_v_heads + head_idx;
          LoadBf16Scalar(a_handle + scalar_offset, kUbAHalf);
          LoadBf16Scalar(b_handle + scalar_offset, kUbBHalf);
          LoadBf16Row(z_handle + batch_idx * sequence_length * v_width +
                          head_idx * kHeadDim,
                      kUbZHalf);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        }
      } else {
        const int32_t conv_token_offset = (token_start + token_idx) * conv_dim;
        LoadBf16Row(
            conv_out_handle + conv_token_offset + qk_head_idx * kHeadDim,
            kUbQHalf);
        LoadBf16Row(conv_out_handle + conv_token_offset +
                        num_k_heads * kHeadDim + qk_head_idx * kHeadDim,
                    kUbKHalf);
        LoadBf16Row(conv_out_handle + conv_token_offset +
                        2 * num_k_heads * kHeadDim + head_idx * kHeadDim,
                    kUbVHalf);
        const int32_t scalar_offset =
            (token_start + token_idx) * num_v_heads + head_idx;
        LoadBf16Scalar(a_handle + scalar_offset, kUbAHalf);
        LoadBf16Scalar(b_handle + scalar_offset, kUbBHalf);
        LoadBf16Row(z_handle + (token_start + token_idx) * v_width +
                        head_idx * kHeadDim,
                    kUbZHalf);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      }
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

      if constexpr (!UseQkGroupCache) {
        TCVT(q_fp32, q_half, RoundMode::CAST_NONE);
        TCVT(k_fp32, k_half, RoundMode::CAST_NONE);
      }
      TCVT(v_fp32, v_half, RoundMode::CAST_NONE);
      if constexpr (UseDeferredNorm) {
        TileUbDataND<bfloat16_t, 1, kHeadDim> cached_z_half;
        TASSIGN(cached_z_half,
                kRunDeferredZHalf + token_idx * kHeadDim * sizeof(bfloat16_t));
        TMOV(cached_z_half, z_half);
      } else {
        TCVT(z_fp32, z_half, RoundMode::CAST_NONE);
      }
      TCVT(scalar, a_half, RoundMode::CAST_NONE);
      TCVT(scalar2, b_half, RoundMode::CAST_NONE);
      mega_gdn_decode_pto::VectorBarrier();
      if constexpr (SpeculativeTokens == 8) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        if (token_idx + 1 < sequence_length) {
          const int32_t next_token_idx = token_idx + 1;
          const int32_t next_conv_token_offset =
              (batch_idx * sequence_length + next_token_idx) * conv_dim;
          if constexpr (!UseQkGroupCache) {
            LoadQkBf16Rows(conv_out_handle + next_conv_token_offset +
                               qk_head_idx * kHeadDim,
                           kUbQHalf,
                           num_k_heads * kHeadDim);
          }
          LoadBf16Row(conv_out_handle + next_conv_token_offset +
                          2 * num_k_heads * kHeadDim + head_idx * kHeadDim,
                      kUbVHalf);
          const int32_t next_scalar_offset =
              (batch_idx * sequence_length + next_token_idx) * num_v_heads +
              head_idx;
          LoadBf16Scalar(a_handle + next_scalar_offset, kUbAHalf);
          LoadBf16Scalar(b_handle + next_scalar_offset, kUbBHalf);
          LoadBf16Row(
              z_handle +
                  (batch_idx * sequence_length + next_token_idx) * v_width +
                  head_idx * kHeadDim,
              kUbZHalf);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        }
      }
      if constexpr (!UseQkGroupCache) {
        mega_gdn_decode_pto::NormalizeQk128<true>(
            q_fp32, norm_square, norm_value, reduce_tmp, scalar_tmp);
        mega_gdn_decode_pto::NormalizeQk128<false>(
            k_fp32, norm_square, norm_value, reduce_tmp, scalar_tmp);
      }

      int32_t q_token_address = kUbQ;
      int32_t k_token_address = kUbK;
      if constexpr (UseQkGroupCache) {
        q_token_address = kUbQkCacheTail + token_idx * kHeadDim * sizeof(float);
        k_token_address = kUbQkCacheK + token_idx * kHeadDim * sizeof(float);
      }
      const int32_t deferred_readout_address =
          kRunDeferredReadoutHalf + token_idx * kHeadDim * sizeof(bfloat16_t);
      RunRecurrentStep<UseDeferredNorm, FlaSsmStateLayout>(
          q_token_address, k_token_address, deferred_readout_address);
      WriteCheckpoint(ssm_state_out_handle,
                      write_slot,
                      state_index_width == 2 ? 1 : sequence_length,
                      ssm_checkpoint_stride,
                      head_idx,
                      token_idx);
      if constexpr (!UseDeferredNorm) {
        RunNormStep<SpeculativeTokens>(
            PackedVarlen ? out_handle + token_start * v_width : out_handle,
            PackedVarlen ? 0 : batch_idx,
            token_idx,
            head_idx,
            sequence_length,
            v_width);
      }
    }

    if constexpr (UseDeferredNorm) {
      RunDeferredNorm<kRunDeferredNormRows, UseQkGroupCache>(
          out_handle, batch_idx, head_idx, sequence_length, v_width);
    }
  }
#endif

  // Keep persistent-state publication separate from the Conv-to-GDN hand-off
  // barrier. Reusing one device flag for both phases can mix barrier
  // generations under graph replay.
#if defined(PTO_NPU_ARCH_A5)
  SyncAllMixA5();
#elif defined(__DAV_VEC__) || defined(__DAV_C220_VEC__)
  SyncAllAivStatePublish();
#endif
}

}  // namespace mega_gdn_mtp_decode_pto

#endif
