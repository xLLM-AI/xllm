#ifndef XLLM_OPS_ARCH35_MEGA_GDN_DECODE_PTO_KERNEL_H_
#define XLLM_OPS_ARCH35_MEGA_GDN_DECODE_PTO_KERNEL_H_

/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <type_traits>

#include "kernel_operator.h"

namespace mega_gdn_decode_pto {

using namespace pto;

AICORE PTO_INLINE void VectorBarrier() {
#if defined(PTO_NPU_ARCH_A5)
  pipe_barrier(PIPE_ALL);
#else
  pipe_barrier(PIPE_V);
#endif
}

template <typename T,
          int Rows,
          int Cols,
          int RowValid = Rows,
          int ColValid = Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataND = pto::Tile<pto::TileType::Vec,
                               T,
                               Rows,
                               Cols,
                               pto::BLayout::RowMajor,
                               RowValid,
                               ColValid,
                               pto::SLayout::NoneBox,
                               512,
                               PadVal>;

template <typename T,
          int Rows,
          int Cols,
          int RowValid = Rows,
          int ColValid = Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataDN = pto::Tile<pto::TileType::Vec,
                               T,
                               Rows,
                               Cols,
                               pto::BLayout::ColMajor,
                               RowValid,
                               ColValid,
                               pto::SLayout::NoneBox,
                               512,
                               PadVal>;

template <typename T1,
          typename T2,
          int32_t Shape1,
          int32_t Shape2,
          int32_t Shape3,
          int32_t Shape4,
          int32_t Shape5,
          int32_t Stride1,
          int32_t Stride2,
          int32_t Stride3,
          int32_t Stride4,
          int32_t Stride5,
          uint32_t UbRows,
          uint32_t UbCols,
          pto::PadValue PadVal = pto::PadValue::Null>
AICORE PTO_INLINE void CopyGmToUb(__gm__ T1* handle,
                                  int32_t ubAddress,
                                  int32_t ubOffset,
                                  int32_t validRows,
                                  int32_t validCols) {
  static_assert(std::is_same_v<T1, T2>);
  pto::Shape<Shape1, Shape2, Shape3, pto::DYNAMIC, pto::DYNAMIC> shape;
  shape.shape[3] = validRows;
  shape.shape[4] = validCols;
  pto::GlobalTensor<
      T1,
      pto::Shape<Shape1, Shape2, Shape3, pto::DYNAMIC, pto::DYNAMIC>,
      pto::Stride<Stride1, Stride2, Stride3, Stride4, Stride5>>
      tensor(handle, shape);

  TileUbDataND<T2, UbRows, UbCols, pto::DYNAMIC, pto::DYNAMIC, PadVal> tile(
      validRows, validCols);
  pto::TASSIGN(tile, ubAddress + ubOffset * sizeof(T2));
  pto::TLOAD(tile, tensor);

  if constexpr (PadVal != pto::PadValue::Null) {
    if (validRows != static_cast<int32_t>(UbRows) ||
        validCols != static_cast<int32_t>(UbCols)) {
      TileUbDataND<T2, UbRows, UbCols, UbRows, UbCols, PadVal> padded;
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pto::TASSIGN(padded, ubAddress + ubOffset * sizeof(T2));
      pto::TFILLPAD_INPLACE(padded, tile);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      VectorBarrier();
    }
  }
}

AICORE PTO_INLINE void CopyConvHistoryGmToUb(__gm__ bfloat16_t* handle,
                                             int32_t ubAddress,
                                             int32_t convDim) {
  using HistoryShape = pto::Shape<1, 1, 1, 3, 128>;
  using HistoryStride = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;
  HistoryShape shape;
  HistoryStride stride(convDim);
  pto::GlobalTensor<bfloat16_t, HistoryShape, HistoryStride> tensor(
      handle, shape, stride);
  TileUbDataND<bfloat16_t, 3, 128> tile;
  pto::TASSIGN(tile, ubAddress);
  pto::TLOAD(tile, tensor);
}

AICORE PTO_INLINE void CopyConvHistoryUbToGm(__gm__ bfloat16_t* handle,
                                             int32_t ubAddress,
                                             int32_t convDim) {
  using HistoryShape = pto::Shape<1, 1, 1, 3, 128>;
  using HistoryStride = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;
  HistoryShape shape;
  HistoryStride stride(convDim);
  pto::GlobalTensor<bfloat16_t, HistoryShape, HistoryStride> tensor(
      handle, shape, stride);
  TileUbDataND<bfloat16_t, 3, 128> tile;
  pto::TASSIGN(tile, ubAddress);
  pto::TSTORE(tensor, tile);
}

AICORE PTO_INLINE void CopyConvWeightsGmToUb(__gm__ bfloat16_t* handle,
                                             int32_t ubAddress,
                                             int32_t convDim) {
  using WeightShape = pto::Shape<1, 1, 1, 4, 128>;
  using WeightStride = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;
  WeightShape shape;
  WeightStride stride(convDim);
  pto::GlobalTensor<bfloat16_t, WeightShape, WeightStride> tensor(
      handle, shape, stride);
  TileUbDataND<bfloat16_t, 4, 128> tile;
  pto::TASSIGN(tile, ubAddress);
  pto::TLOAD(tile, tensor);
}

template <typename T1,
          typename T2,
          int32_t Shape1,
          int32_t Shape2,
          int32_t Shape3,
          int32_t Shape4,
          int32_t Shape5,
          int32_t Stride1,
          int32_t Stride2,
          int32_t Stride3,
          int32_t Stride4,
          int32_t Stride5,
          uint32_t UbRows,
          uint32_t UbCols>
AICORE PTO_INLINE void CopyUbToGm(__gm__ T1* handle,
                                  int32_t ubAddress,
                                  int32_t ubOffset,
                                  int32_t validRows,
                                  int32_t validCols) {
  static_assert(std::is_same_v<T1, T2>);
  pto::Shape<Shape1, Shape2, Shape3, pto::DYNAMIC, pto::DYNAMIC> shape;
  shape.shape[3] = validRows;
  shape.shape[4] = validCols;
  pto::GlobalTensor<
      T1,
      pto::Shape<Shape1, Shape2, Shape3, pto::DYNAMIC, pto::DYNAMIC>,
      pto::Stride<Stride1, Stride2, Stride3, Stride4, Stride5>>
      tensor(handle, shape);

  constexpr bool kUseNd = static_cast<uint64_t>(UbCols) * sizeof(T2) >= 32;
  if constexpr (kUseNd) {
    TileUbDataND<T2, UbRows, UbCols, pto::DYNAMIC, pto::DYNAMIC> tile(
        validRows, validCols);
    pto::TASSIGN(tile, ubAddress + ubOffset * sizeof(T2));
    pto::TSTORE(tensor, tile);
  } else {
    TileUbDataDN<T2, UbRows, UbCols, pto::DYNAMIC, pto::DYNAMIC> tile(
        validRows, validCols);
    pto::TASSIGN(tile, ubAddress + ubOffset * sizeof(T2));
    pto::TSTORE(tensor, tile);
  }
}

template <typename T, int32_t Rows, int32_t Cols>
AICORE PTO_INLINE void Sigmoid(TileUbDataND<T, Rows, Cols>& dst,
                               TileUbDataND<T, Rows, Cols>& src) {
  TMULS(src, src, -1);
  VectorBarrier();
  TEXP(src, src);
  VectorBarrier();
  TADDS(src, src, 1);
  VectorBarrier();
  TRECIP(dst, src);
}

template <typename T, int32_t Rows, int32_t Cols>
AICORE PTO_INLINE void Silu(TileUbDataND<T, Rows, Cols>& dst,
                            TileUbDataND<T, Rows, Cols>& src,
                            TileUbDataND<T, Rows, Cols>& tmp) {
  TMOV(tmp, src);
  VectorBarrier();
#if defined(PTO_NPU_ARCH_A5)
  TMULS(src, src, -1);
  VectorBarrier();
  TEXP(src, src);
  VectorBarrier();
  TADDS(src, src, 1);
  VectorBarrier();
  // Match AscendC::Silu exactly. Multiplying by a separately rounded
  // reciprocal can cross a BF16 rounding boundary after the fused conv.
  TDIV(dst, tmp, src);
#else
  Sigmoid(dst, src);
  VectorBarrier();
  TMUL(dst, tmp, dst);
#endif
}

template <typename T, int32_t Rows, int32_t Cols>
AICORE PTO_INLINE void CausalConvSilu(TileUbDataND<T, Rows, Cols>& dst,
                                      TileUbDataND<T, Rows, Cols>& src,
                                      TileUbDataND<T, Rows, Cols>& tmp) {
  TMOV(tmp, src);
  VectorBarrier();
  TMULS(src, src, -1);
  VectorBarrier();
  TEXP(src, src);
  VectorBarrier();
  TADDS(src, src, 1);
  VectorBarrier();
  // Numerical contract: match AscendC::Silu at the CausalConv BF16
  // hand-off; reciprocal followed by multiply can cross a BF16 midpoint.
  TDIV(dst, tmp, src);
}

template <typename T, int32_t Rows, int32_t Cols>
AICORE PTO_INLINE void MulAddDst(TileUbDataND<T, Rows, Cols>& dst,
                                 TileUbDataND<T, Rows, Cols>& src0,
                                 TileUbDataND<T, Rows, Cols>& src1,
                                 TileUbDataND<T, Rows, Cols>& tmp) {
#if defined(PTO_NPU_ARCH_A5)
  (void)tmp;
  TMULADDDST(dst, src0, src1);
#else
  TMUL(tmp, src0, src1);
  VectorBarrier();
  TADD(dst, dst, tmp);
#endif
}

#if !defined(PTO_NPU_ARCH_A5)
template <typename TileDst, typename TileRow, typename TileCol>
__tf__ PTO_INTERNAL void OuterProductAdd128Impl(
    typename TileDst::TileDType __in__ __out__ dst,
    typename TileRow::TileDType __in__ row,
    typename TileCol::TileDType __in__ col) {
  __ubuf__ float* dstPtr =
      reinterpret_cast<__ubuf__ float*>(__cce_get_tile_ptr(dst));
  __ubuf__ float* rowPtr =
      reinterpret_cast<__ubuf__ float*>(__cce_get_tile_ptr(row));
  __ubuf__ uint32_t* colPtr =
      reinterpret_cast<__ubuf__ uint32_t*>(__cce_get_tile_ptr(col));
  __ubuf__ uint32_t* broadcast =
      reinterpret_cast<__ubuf__ uint32_t*>(TMP_UB_OFFSET);
  __ubuf__ float* colValues = reinterpret_cast<__ubuf__ float*>(broadcast);

  vbrcb(broadcast, colPtr, 1, 8, 16);
  VectorBarrier();
  vmla(dstPtr, rowPtr, colValues, 128, 1, 1, 0, 16, 0, 1);
  vmla(dstPtr + 64, rowPtr + 64, colValues, 128, 1, 1, 0, 16, 0, 1);
}

template <typename TileDst, typename TileRow, typename TileCol>
AICORE PTO_INLINE void OuterProductAdd128(TileDst& dst,
                                          TileRow& row,
                                          TileCol& col) {
  OuterProductAdd128Impl<TileDst, TileRow, TileCol>(
      dst.data(), row.data(), col.data());
}
#endif

#if !defined(PTO_NPU_ARCH_A5)
template <typename TileDst, typename TileSrc, typename TileTmp>
__tf__ PTO_INTERNAL void ColSum128Impl(typename TileDst::TileDType __out__ dst,
                                       typename TileSrc::TileDType __in__ src,
                                       typename TileTmp::TileDType __out__
                                           tmp) {
  __ubuf__ float* dstPtr =
      reinterpret_cast<__ubuf__ float*>(__cce_get_tile_ptr(dst));
  __ubuf__ float* srcPtr =
      reinterpret_cast<__ubuf__ float*>(__cce_get_tile_ptr(src));
  (void)tmp;

  set_mask_norm();
  set_vector_mask(-1, -1);
#define MEGA_GDN_COLSUM_BUTTERFLY_STAGE(rows)                                \
  vadd(srcPtr, srcPtr, srcPtr + (rows) * 128, (rows) * 2, 1, 1, 1, 8, 8, 8); \
  VectorBarrier()
  MEGA_GDN_COLSUM_BUTTERFLY_STAGE(64);
  MEGA_GDN_COLSUM_BUTTERFLY_STAGE(32);
  MEGA_GDN_COLSUM_BUTTERFLY_STAGE(16);
  MEGA_GDN_COLSUM_BUTTERFLY_STAGE(8);
  MEGA_GDN_COLSUM_BUTTERFLY_STAGE(4);
  MEGA_GDN_COLSUM_BUTTERFLY_STAGE(2);
  MEGA_GDN_COLSUM_BUTTERFLY_STAGE(1);
#undef MEGA_GDN_COLSUM_BUTTERFLY_STAGE
  copy_ubuf_to_ubuf(dstPtr, srcPtr, 0, 1, 16, 0, 0);
  VectorBarrier();
}
#endif

template <typename TileDst, typename TileSrc, typename TileTmp>
AICORE PTO_INLINE void ColSum128(TileDst& dst, TileSrc& src, TileTmp& tmp) {
#if defined(PTO_NPU_ARCH_A5)
  (void)tmp;
  // Contract: TileLang lowers tl.sum(axis=0) for [128, 64] to a butterfly
  // tree with descending row strides on both 910B and A5. The product tile
  // is dead after reduction, so compact partial sums into its lower rows.
  const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(src.data());
  constexpr std::uintptr_t rowBytes = 128 * sizeof(float);
  TileUbDataND<float, 64, 128> low64;
  TileUbDataND<float, 64, 128> high64;
  TASSIGN(low64, base);
  TASSIGN(high64, base + 64 * rowBytes);
  TADD(low64, low64, high64);
  VectorBarrier();

  TileUbDataND<float, 32, 128> low32;
  TileUbDataND<float, 32, 128> high32;
  TASSIGN(low32, base);
  TASSIGN(high32, base + 32 * rowBytes);
  TADD(low32, low32, high32);
  VectorBarrier();

  TileUbDataND<float, 16, 128> low16;
  TileUbDataND<float, 16, 128> high16;
  TASSIGN(low16, base);
  TASSIGN(high16, base + 16 * rowBytes);
  TADD(low16, low16, high16);
  VectorBarrier();

  TileUbDataND<float, 8, 128> low8;
  TileUbDataND<float, 8, 128> high8;
  TASSIGN(low8, base);
  TASSIGN(high8, base + 8 * rowBytes);
  TADD(low8, low8, high8);
  VectorBarrier();

  TileUbDataND<float, 4, 128> low4;
  TileUbDataND<float, 4, 128> high4;
  TASSIGN(low4, base);
  TASSIGN(high4, base + 4 * rowBytes);
  TADD(low4, low4, high4);
  VectorBarrier();

  TileUbDataND<float, 2, 128> low2;
  TileUbDataND<float, 2, 128> high2;
  TASSIGN(low2, base);
  TASSIGN(high2, base + 2 * rowBytes);
  TADD(low2, low2, high2);
  VectorBarrier();

  TileUbDataND<float, 1, 128> low1;
  TileUbDataND<float, 1, 128> high1;
  TASSIGN(low1, base);
  TASSIGN(high1, base + rowBytes);
  TADD(low1, low1, high1);
  VectorBarrier();
  TMOV(dst, low1);
#else
  ColSum128Impl<TileDst, TileSrc, TileTmp>(dst.data(), src.data(), tmp.data());
#endif
}

template <typename TileDst, typename TileSrc, typename TileTmp>
AICORE PTO_INLINE void RowSum128(TileDst& dst, TileSrc& src, TileTmp& tmp) {
  const std::uintptr_t srcBase = reinterpret_cast<std::uintptr_t>(src.data());
  const std::uintptr_t dstBase = reinterpret_cast<std::uintptr_t>(dst.data());
  constexpr std::uintptr_t kHalfStateBytes = 64 * 128 * sizeof(float);
  constexpr std::uintptr_t kHalfOutputBytes = 64 * sizeof(float);

  TileUbDataND<float, 64, 128> srcLow;
  TASSIGN(srcLow, srcBase);
  TileUbDataDN<float, 64, 1> dstLow;
  TASSIGN(dstLow, dstBase);
  TROWSUM(dstLow, srcLow, tmp);
  VectorBarrier();

  TileUbDataND<float, 64, 128> srcHigh;
  TASSIGN(srcHigh, srcBase + kHalfStateBytes);
  TileUbDataDN<float, 64, 1> dstHigh;
  TASSIGN(dstHigh, dstBase + kHalfOutputBytes);
  TROWSUM(dstHigh, srcHigh, tmp);
  VectorBarrier();
}

template <bool FlaSsmStateLayout>
AICORE PTO_INLINE void StateVectorProduct128(
    TileUbDataND<float, 1, 128>& dst,
    TileUbDataND<float, 128, 128>& state,
    TileUbDataND<float, 1, 128>& vector,
    TileUbDataND<float, 128, 128>& compute,
    TileUbDataND<float, 32, 128>& reduceTmp) {
  if constexpr (FlaSsmStateLayout) {
    TileUbDataDN<float, 128, 1> vectorDn;
    TASSIGN(vectorDn, reinterpret_cast<std::uintptr_t>(vector.data()));
    TROWEXPANDMUL(compute, state, vectorDn);
    VectorBarrier();
    ColSum128(dst, compute, reduceTmp);
  } else {
    TCOLEXPANDMUL(compute, state, vector);
    VectorBarrier();
    RowSum128(dst, compute, reduceTmp);
  }
}

template <bool FlaSsmStateLayout>
AICORE PTO_INLINE void StateRankOneUpdate128(
    TileUbDataND<float, 128, 128>& state,
    TileUbDataND<float, 1, 128>& key,
    TileUbDataND<float, 1, 128>& delta,
    TileUbDataND<float, 128, 128>& compute) {
  if constexpr (FlaSsmStateLayout) {
    TileUbDataDN<float, 128, 1> keyDn;
    TASSIGN(keyDn, reinterpret_cast<std::uintptr_t>(key.data()));
#if defined(PTO_NPU_ARCH_A5)
    TROWEXPAND(compute, keyDn);
    VectorBarrier();
    TCOLEXPANDMUL(compute, compute, delta);
    VectorBarrier();
    TADD(state, state, compute);
#else
    OuterProductAdd128(state, delta, keyDn);
#endif
  } else {
    TileUbDataDN<float, 128, 1> deltaDn;
    TASSIGN(deltaDn, reinterpret_cast<std::uintptr_t>(delta.data()));
#if defined(PTO_NPU_ARCH_A5)
    TROWEXPAND(compute, deltaDn);
    VectorBarrier();
    TCOLEXPANDMUL(compute, compute, key);
    VectorBarrier();
    TADD(state, state, compute);
#else
    OuterProductAdd128(state, key, deltaDn);
#endif
  }
  VectorBarrier();
}

template <bool ApplyQScale>
AICORE PTO_INLINE void NormalizeQk128(
    TileUbDataND<float, 1, 128>& row,
    TileUbDataND<float, 1, 128>& norm_sq,
    TileUbDataND<float, 1, 8, 1, 1>& norm_val,
    TileUbDataND<uint8_t, 128, 64>& tmp_ub,
    TileUbDataND<float, 1, 8, 1, 1>& scalar_tmp) {
  TMUL(norm_sq, row, row);
  VectorBarrier();
  TileUbDataDN<float, 8, 1, 1, 1> norm_val_dn;
  TASSIGN(norm_val_dn, reinterpret_cast<std::uintptr_t>(norm_val.data()));
  TileUbDataND<float, 1, 64, 1, 64> reduce_tmp;
  TASSIGN(reduce_tmp, reinterpret_cast<std::uintptr_t>(tmp_ub.data()));
  TROWSUM(norm_val_dn, norm_sq, reduce_tmp);
  VectorBarrier();
  TADDS(norm_val, norm_val, 1.000000e-06f);
  VectorBarrier();
  TSQRT(scalar_tmp, norm_val);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  const float norm = scalar_tmp.GetValue(0);
  // Contract: TileLang lowers q / sqrt(sum(q * q) + eps) as a direct
  // vector division. On 910B, in-place TDIVS takes a reciprocal+vmuls
  // shortcut, so use the dead square buffer as a non-alias destination to
  // preserve vdiv rounding.
#if defined(PTO_NPU_ARCH_A5)
  TDIVS(row, row, norm);
#else
  TDIVS(norm_sq, row, norm);
  VectorBarrier();
  TMOV(row, norm_sq);
#endif
  if constexpr (ApplyQScale) {
    VectorBarrier();
    // Exact FP32 rounding of 1 / sqrt(128), matching TileLang's scale.
    TMULS(row, row, 0.0883883461356163f);
  }
}

constexpr uint16_t kSyncAivOnlyFlag = 14;
constexpr uint16_t kStatePublishAivFlag = 13;
constexpr uint16_t kSyncModeShift = 4;
constexpr uint16_t kSyncFlagShift = 8;

AICORE PTO_INLINE uint16_t GetFftsMessage(uint16_t mode, uint16_t flag) {
  return 0x1 + ((mode & 0x3) << kSyncModeShift) +
         ((flag & 0xf) << kSyncFlagShift);
}

AICORE PTO_INLINE void SyncAllAiv() {
  pipe_barrier(PIPE_ALL);
#if defined(PTO_NPU_ARCH_A5)
  SYNCALL<pto::SyncCoreType::AIVOnly>();
#else
  ffts_cross_core_sync(PIPE_MTE3, GetFftsMessage(0, kSyncAivOnlyFlag));
  wait_flag_dev(kSyncAivOnlyFlag);
#endif
}

AICORE PTO_INLINE void SyncAllAivStatePublish() {
  pipe_barrier(PIPE_ALL);
#if defined(PTO_NPU_ARCH_A5)
  SYNCALL<pto::SyncCoreType::AIVOnly>();
#else
  ffts_cross_core_sync(PIPE_MTE3, GetFftsMessage(0, kStatePublishAivFlag));
  wait_flag_dev(kStatePublishAivFlag);
#endif
}

constexpr int32_t kCompiledNumCacheSlots = 1024;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kConvStateLen = 3;
constexpr int32_t kSsmHeadElements = kHeadDim * kHeadDim;
constexpr int32_t kMaxNumKHeads = 16;
constexpr int32_t kMaxNumVHeads = 64;
constexpr int32_t kMaxBatchSize = 32;
constexpr int32_t kMaxConvDim = (2 * kMaxNumKHeads + kMaxNumVHeads) * kHeadDim;
constexpr int32_t kMaxConvWeightElements = 4 * kMaxConvDim;
constexpr int32_t kMaxConvStateElements =
    kCompiledNumCacheSlots * kConvStateLen * kMaxConvDim;
constexpr int32_t kMaxSsmStateElements =
    kCompiledNumCacheSlots * kMaxNumVHeads * kSsmHeadElements;

constexpr int32_t kConvBuffer0 = 3072;
constexpr int32_t kConvBuffer1 = 8704;
constexpr int32_t kConvHistHalf0 = 0;
constexpr int32_t kConvHistHalf1 = 256;
constexpr int32_t kConvHistHalf2 = 512;
constexpr int32_t kConvInputHalf = 768;
constexpr int32_t kConvHist0 = 1024;
constexpr int32_t kConvHist1 = 1536;
constexpr int32_t kConvHist2 = 2048;
constexpr int32_t kConvInput = 2560;
constexpr int32_t kConvAcc = 3072;
constexpr int32_t kConvTmp = 3584;
constexpr int32_t kConvOutput = 4096;
constexpr int32_t kConvOutputHalf = 4608;
constexpr int32_t kConvSaveHalf0 = 4864;
constexpr int32_t kConvSaveHalf1 = 5120;
constexpr int32_t kConvSaveHalf2 = 5376;
constexpr int32_t kConvVectorScratch = 155936;

template <int32_t BufferBase, int32_t LoadEvent, int32_t ReuseEvent>
AICORE PTO_INLINE void PrefetchConvBatch(__gm__ bfloat16_t* qkv_handle,
                                         __gm__ bfloat16_t* conv_state_handle,
                                         int32_t batch_idx,
                                         int32_t state_idx,
                                         int32_t conv_dim,
                                         int32_t conv_state_stride,
                                         int32_t channel_offset) {
  wait_flag(PIPE_MTE3, PIPE_MTE2, ReuseEvent);
  CopyConvHistoryGmToUb(
      conv_state_handle + state_idx * conv_state_stride + channel_offset,
      BufferBase + kConvHistHalf0,
      conv_dim);
  CopyGmToUb<bfloat16_t,
             bfloat16_t,
             1,
             1,
             1,
             1,
             128,
             1,
             1,
             1,
             kMaxBatchSize * kMaxConvDim,
             1,
             1,
             128,
             pto::PadValue::Zero>(
      qkv_handle + batch_idx * conv_dim + channel_offset,
      BufferBase + kConvInputHalf,
      0,
      1,
      128);
  set_flag(PIPE_MTE2, PIPE_V, LoadEvent);
}

template <int32_t BufferBase,
          int32_t LoadEvent,
          int32_t StoreEvent,
          int32_t ReuseEvent>
AICORE PTO_INLINE void ComputeAndStoreConvBatch(
    __gm__ bfloat16_t* conv_out_handle,
    __gm__ bfloat16_t* conv_state_out_handle,
    TileUbDataND<float, 1, 128, 1, 128>& w0,
    TileUbDataND<float, 1, 128, 1, 128>& w1,
    TileUbDataND<float, 1, 128, 1, 128>& w2,
    TileUbDataND<float, 1, 128, 1, 128>& w3,
    int32_t batch_idx,
    int32_t state_idx,
    int32_t conv_dim,
    int32_t conv_state_stride,
    int32_t channel_offset) {
  TileUbDataND<bfloat16_t, 1, 128, 1, 128> hist_half0;
  TASSIGN(hist_half0, BufferBase + kConvHistHalf0);
  TileUbDataND<bfloat16_t, 1, 128, 1, 128> hist_half1;
  TASSIGN(hist_half1, BufferBase + kConvHistHalf1);
  TileUbDataND<bfloat16_t, 1, 128, 1, 128> hist_half2;
  TASSIGN(hist_half2, BufferBase + kConvHistHalf2);
  TileUbDataND<bfloat16_t, 1, 128, 1, 128> x_half;
  TASSIGN(x_half, BufferBase + kConvInputHalf);
  TileUbDataND<float, 1, 128, 1, 128> hist0;
  TASSIGN(hist0, BufferBase + kConvHist0);
  TileUbDataND<float, 1, 128, 1, 128> hist1;
  TASSIGN(hist1, BufferBase + kConvHist1);
  TileUbDataND<float, 1, 128, 1, 128> hist2;
  TASSIGN(hist2, BufferBase + kConvHist2);
  TileUbDataND<float, 1, 128, 1, 128> x_fp32;
  TASSIGN(x_fp32, BufferBase + kConvInput);
  TileUbDataND<float, 1, 128, 1, 128> conv_acc;
  TASSIGN(conv_acc, BufferBase + kConvAcc);
  TileUbDataND<float, 1, 128, 1, 128> conv_tmp;
  TASSIGN(conv_tmp, BufferBase + kConvTmp);
  TileUbDataND<float, 1, 128, 1, 128> conv_y;
  TASSIGN(conv_y, BufferBase + kConvOutput);
  TileUbDataND<bfloat16_t, 1, 128, 1, 128> y_half;
  TASSIGN(y_half, BufferBase + kConvOutputHalf);
  TileUbDataND<bfloat16_t, 1, 128, 1, 128> save_half0;
  TASSIGN(save_half0, BufferBase + kConvSaveHalf0);
  TileUbDataND<bfloat16_t, 1, 128, 1, 128> save_half1;
  TASSIGN(save_half1, BufferBase + kConvSaveHalf1);
  TileUbDataND<bfloat16_t, 1, 128, 1, 128> save_half2;
  TASSIGN(save_half2, BufferBase + kConvSaveHalf2);

  wait_flag(PIPE_MTE2, PIPE_V, LoadEvent);
  TCVT(hist0, hist_half0, RoundMode::CAST_NONE);
  TCVT(hist1, hist_half1, RoundMode::CAST_NONE);
  TCVT(hist2, hist_half2, RoundMode::CAST_NONE);
  TCVT(x_fp32, x_half, RoundMode::CAST_NONE);
  VectorBarrier();
  TMUL(conv_acc, w0, hist0);
#if defined(PTO_NPU_ARCH_A5)
  VectorBarrier();
  MulAddDst<float, 1, 128>(conv_acc, hist1, w1, conv_tmp);
  VectorBarrier();
  MulAddDst<float, 1, 128>(conv_acc, hist2, w2, conv_tmp);
  VectorBarrier();
#else
  TMUL(conv_tmp, w1, hist1);
  VectorBarrier();
  TADD(conv_acc, conv_acc, conv_tmp);
  VectorBarrier();
  TMUL(conv_tmp, w2, hist2);
  VectorBarrier();
  TADD(conv_acc, conv_acc, conv_tmp);
  VectorBarrier();
#endif
  TileUbDataND<float, 1, 128> muladd_tmp;
  TASSIGN(muladd_tmp, kConvVectorScratch);
  MulAddDst<float, 1, 128>(conv_acc, x_fp32, w3, muladd_tmp);
  VectorBarrier();
  TileUbDataND<float, 1, 128> silu_tmp;
  TASSIGN(silu_tmp, kConvVectorScratch);
  CausalConvSilu<float, 1, 128>(conv_y, conv_acc, silu_tmp);
  VectorBarrier();
  TCVT(y_half, conv_y, RoundMode::CAST_RINT);
  TCVT(save_half0, hist1, RoundMode::CAST_RINT);
  TCVT(save_half1, hist2, RoundMode::CAST_RINT);
  TCVT(save_half2, x_fp32, RoundMode::CAST_RINT);
  set_flag(PIPE_V, PIPE_MTE3, StoreEvent);
  wait_flag(PIPE_V, PIPE_MTE3, StoreEvent);
  CopyUbToGm<bfloat16_t,
             bfloat16_t,
             1,
             1,
             1,
             1,
             128,
             1,
             1,
             1,
             kMaxBatchSize * kMaxConvDim,
             1,
             1,
             128>(conv_out_handle + batch_idx * conv_dim + channel_offset,
                  BufferBase + kConvOutputHalf,
                  0,
                  1,
                  128);
  CopyConvHistoryUbToGm(
      conv_state_out_handle + state_idx * conv_state_stride + channel_offset,
      BufferBase + kConvSaveHalf0,
      conv_dim);
  set_flag(PIPE_MTE3, PIPE_MTE2, ReuseEvent);
}

#if defined(PTO_NPU_ARCH_A5)
namespace a5_b4_deferred {

AICORE PTO_INLINE void NormVectorBarrier() {
  // Deferred Norm only has vector-to-vector dependencies until its final
  // output store.  Ascend950 does not accept a single-pipe PIPE_V barrier,
  // so fence the vector queue through the scalar pipe instead.  This keeps
  // MTE3 running while the last FP32 state store overlaps this tail; the
  // final V->MTE3 event and state-publish barrier still order and drain both
  // stores.
  PtoSetWaitFlag<PIPE_V, PIPE_S>(EVENT_ID0, EVENT_ID0);
  PtoSetWaitFlag<PIPE_S, PIPE_V>(EVENT_ID0, EVENT_ID0);
}

// Reuse the ordinary decode UB layout.  The two 64-KiB state regions alternate
// between current state and recurrent scratch.  Cached BF16 readout/Z rows live
// above the scalar caches and survive until the batched Norm tail.
constexpr int32_t kUbQHalf = 8704;
constexpr int32_t kUbKHalf = 8960;
constexpr int32_t kUbQ = 9280;
constexpr int32_t kUbK = 9792;
constexpr int32_t kUbNormSquare = 10368;
constexpr int32_t kUbNormValue = 10880;
constexpr int32_t kUbReduceTmp = 10912;
constexpr int32_t kUbScalarTmp = 19104;
constexpr int32_t kUbVHalf = 19200;
constexpr int32_t kUbState0 = 19456;
constexpr int32_t kUbV = 84992;
constexpr int32_t kUbState1 = 85504;
constexpr int32_t kUbPrediction = 151040;
constexpr int32_t kUbDelta = 151552;
constexpr int32_t kUbNormWeightHalf = 152832;
constexpr int32_t kUbNormWeight = 154112;
constexpr int32_t kUbColumnSumTmp = 156672;
constexpr int32_t kUbACache = 173312;
constexpr int32_t kUbBCache = 173568;
constexpr int32_t kUbCachedReadoutHalf = 174592;
constexpr int32_t kUbCachedZHalf =
    kUbCachedReadoutHalf + 4 * kHeadDim * sizeof(bfloat16_t);
constexpr int32_t kUbCachedRowsEnd =
    kUbCachedZHalf + 4 * kHeadDim * sizeof(bfloat16_t);

static_assert(kUbState0 + kSsmHeadElements * sizeof(float) == kUbV);
static_assert(kUbState1 + kSsmHeadElements * sizeof(float) == kUbPrediction);
static_assert(kUbCachedRowsEnd <= static_cast<int32_t>(pto::TMP_UB_OFFSET));

template <int32_t Rows>
AICORE PTO_INLINE void LoadBf16Rows(__gm__ bfloat16_t* handle,
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

template <int32_t Rows>
AICORE PTO_INLINE void StoreBf16Rows(__gm__ bfloat16_t* handle,
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

template <int32_t Rows, int32_t WorkBase>
class NormLayout final {
 public:
  static constexpr int32_t kInput = WorkBase;
  static constexpr int32_t kZ = kInput + Rows * kHeadDim * sizeof(float);
  static constexpr int32_t kSquare = kZ + Rows * kHeadDim * sizeof(float);
  static constexpr int32_t kReduceTmp =
      kSquare + Rows * kHeadDim * sizeof(float);
  static constexpr int32_t kRms = kReduceTmp + Rows * 64 * sizeof(float);
  static constexpr int32_t kRmsSqrt = kRms + 16 * sizeof(float);
  static constexpr int32_t kFinalHalf = kRmsSqrt + 16 * sizeof(float);
  static constexpr int32_t kWorkEnd =
      kFinalHalf + Rows * kHeadDim * sizeof(bfloat16_t);

  static_assert(Rows >= 1 && Rows <= 4);
  static_assert(kWorkEnd <= WorkBase + kSsmHeadElements * sizeof(float));
};

// Ascend950-only row-streaming recurrent primitives.  Each vector register
// owns one 64-value half of the V dimension while rows stream through the
// registers in K order.  The state layout is [K, V].  Separate Mul/Add keeps
// the FP32 update rounding aligned with the existing PTO implementation.
AICORE PTO_INLINE void StateVectorProductRegBase128(
    TileUbDataND<float, 1, 128>& dst,
    TileUbDataND<float, 128, 128>& state,
    TileUbDataND<float, 1, 128>& vector) {
  __ubuf__ float* dst_addr = reinterpret_cast<__ubuf__ float*>(dst.data());
  __ubuf__ float* state_addr = reinterpret_cast<__ubuf__ float*>(state.data());
  __ubuf__ float* vector_addr =
      reinterpret_cast<__ubuf__ float*>(vector.data());

  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<float> accum_low;
    AscendC::MicroAPI::RegTensor<float> accum_high;
    AscendC::MicroAPI::RegTensor<float> state_low;
    AscendC::MicroAPI::RegTensor<float> state_high;
    AscendC::MicroAPI::RegTensor<float> vector_broadcast;
    AscendC::MicroAPI::RegTensor<float> product_low;
    AscendC::MicroAPI::RegTensor<float> product_high;
    AscendC::MicroAPI::MaskReg full_mask =
        AscendC::MicroAPI::CreateMask<float,
                                      AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::Duplicate(accum_low, 0.0f, full_mask);
    AscendC::MicroAPI::Duplicate(accum_high, 0.0f, full_mask);
    for (uint16_t row = 0; row < 128; ++row) {
      AscendC::MicroAPI::DataCopy(state_low, state_addr + row * 128);
      AscendC::MicroAPI::DataCopy(state_high, state_addr + row * 128 + 64);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
          vector_broadcast, vector_addr + row);
      AscendC::MicroAPI::Mul(
          product_low, state_low, vector_broadcast, full_mask);
      AscendC::MicroAPI::Mul(
          product_high, state_high, vector_broadcast, full_mask);
      AscendC::MicroAPI::Add(accum_low, accum_low, product_low, full_mask);
      AscendC::MicroAPI::Add(accum_high, accum_high, product_high, full_mask);
    }
    AscendC::MicroAPI::DataCopy(dst_addr, accum_low, full_mask);
    AscendC::MicroAPI::DataCopy(dst_addr + 64, accum_high, full_mask);
  }
}

AICORE PTO_INLINE void StateRankOneUpdateAndProductRegBase128(
    TileUbDataND<float, 1, 128>& dst,
    TileUbDataND<float, 128, 128>& state,
    TileUbDataND<float, 1, 128>& key,
    TileUbDataND<float, 1, 128>& delta,
    TileUbDataND<float, 1, 128>& query) {
  __ubuf__ float* dst_addr = reinterpret_cast<__ubuf__ float*>(dst.data());
  __ubuf__ float* state_addr = reinterpret_cast<__ubuf__ float*>(state.data());
  __ubuf__ float* key_addr = reinterpret_cast<__ubuf__ float*>(key.data());
  __ubuf__ float* delta_addr = reinterpret_cast<__ubuf__ float*>(delta.data());
  __ubuf__ float* query_addr = reinterpret_cast<__ubuf__ float*>(query.data());

  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<float> delta_low;
    AscendC::MicroAPI::RegTensor<float> delta_high;
    AscendC::MicroAPI::RegTensor<float> accum_low;
    AscendC::MicroAPI::RegTensor<float> accum_high;
    AscendC::MicroAPI::RegTensor<float> state_low;
    AscendC::MicroAPI::RegTensor<float> state_high;
    AscendC::MicroAPI::RegTensor<float> key_broadcast;
    AscendC::MicroAPI::RegTensor<float> query_broadcast;
    AscendC::MicroAPI::RegTensor<float> product_low;
    AscendC::MicroAPI::RegTensor<float> product_high;
    AscendC::MicroAPI::MaskReg full_mask =
        AscendC::MicroAPI::CreateMask<float,
                                      AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::DataCopy(delta_low, delta_addr);
    AscendC::MicroAPI::DataCopy(delta_high, delta_addr + 64);
    AscendC::MicroAPI::Duplicate(accum_low, 0.0f, full_mask);
    AscendC::MicroAPI::Duplicate(accum_high, 0.0f, full_mask);
    for (uint16_t row = 0; row < 128; ++row) {
      AscendC::MicroAPI::DataCopy(state_low, state_addr + row * 128);
      AscendC::MicroAPI::DataCopy(state_high, state_addr + row * 128 + 64);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
          key_broadcast, key_addr + row);
      AscendC::MicroAPI::Mul(product_low, delta_low, key_broadcast, full_mask);
      AscendC::MicroAPI::Mul(
          product_high, delta_high, key_broadcast, full_mask);
      AscendC::MicroAPI::Add(state_low, state_low, product_low, full_mask);
      AscendC::MicroAPI::Add(state_high, state_high, product_high, full_mask);
      AscendC::MicroAPI::DataCopy(state_addr + row * 128, state_low, full_mask);
      AscendC::MicroAPI::DataCopy(
          state_addr + row * 128 + 64, state_high, full_mask);

      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
          query_broadcast, query_addr + row);
      AscendC::MicroAPI::Mul(
          product_low, state_low, query_broadcast, full_mask);
      AscendC::MicroAPI::Mul(
          product_high, state_high, query_broadcast, full_mask);
      AscendC::MicroAPI::Add(accum_low, accum_low, product_low, full_mask);
      AscendC::MicroAPI::Add(accum_high, accum_high, product_high, full_mask);
    }
    AscendC::MicroAPI::DataCopy(dst_addr, accum_low, full_mask);
    AscendC::MicroAPI::DataCopy(dst_addr + 64, accum_high, full_mask);
  }
}

template <bool FlaSsmStateLayout, bool UseRegBase = false>
AICORE PTO_INLINE void RunRecurrentHead(
    TileUbDataND<float, 128, 128>& state,
    TileUbDataND<float, 128, 128>& compute,
    TileUbDataND<float, 1, 128>& q,
    TileUbDataND<float, 1, 128>& k,
    TileUbDataND<float, 1, 128>& v,
    TileUbDataND<float, 1, 128>& prediction,
    TileUbDataND<float, 1, 128>& delta,
    TileUbDataND<float, 32, 128>& colsum_tmp,
    float decay,
    float beta_gate,
    int32_t cached_readout_address) {
  TMULS(state, state, decay);
  if constexpr (UseRegBase) {
    static_assert(FlaSsmStateLayout);
    NormVectorBarrier();
    StateVectorProductRegBase128(prediction, state, k);
    NormVectorBarrier();
  } else {
    VectorBarrier();
    StateVectorProduct128<FlaSsmStateLayout>(
        prediction, state, k, compute, colsum_tmp);
  }
  TSUB(delta, v, prediction);
  if constexpr (UseRegBase) {
    NormVectorBarrier();
  } else {
    VectorBarrier();
  }
  TMULS(delta, delta, beta_gate);
  if constexpr (UseRegBase) {
    NormVectorBarrier();
    StateRankOneUpdateAndProductRegBase128(prediction, state, k, delta, q);
    NormVectorBarrier();
  } else {
    VectorBarrier();
    StateRankOneUpdate128<FlaSsmStateLayout>(state, k, delta, compute);
    StateVectorProduct128<FlaSsmStateLayout>(
        prediction, state, q, compute, colsum_tmp);
  }
  TileUbDataND<bfloat16_t, 1, kHeadDim> cached_readout;
  TASSIGN(cached_readout, cached_readout_address);
  // Preserve the public BF16 recurrent-to-Norm hand-off exactly.
  TCVT(cached_readout, prediction, RoundMode::CAST_RINT);
  VectorBarrier();
}

template <int32_t Rows, int32_t WorkBase>
AICORE PTO_INLINE void RunDeferredNorm(__gm__ bfloat16_t* out_handle,
                                       int32_t first_head_index) {
  using Layout = NormLayout<Rows, WorkBase>;
  TileUbDataND<bfloat16_t, Rows, kHeadDim> cached_readout_batch;
  TASSIGN(cached_readout_batch, kUbCachedReadoutHalf);
  TileUbDataND<bfloat16_t, Rows, kHeadDim> cached_z_batch;
  TASSIGN(cached_z_batch, kUbCachedZHalf);
  TileUbDataND<float, Rows, kHeadDim> input_batch;
  TASSIGN(input_batch, Layout::kInput);
  TileUbDataND<float, Rows, kHeadDim> z_batch;
  TASSIGN(z_batch, Layout::kZ);
  TileUbDataND<float, Rows, kHeadDim> square_batch;
  TASSIGN(square_batch, Layout::kSquare);
  TileUbDataND<float, Rows, 64> reduce_tmp_batch;
  TASSIGN(reduce_tmp_batch, Layout::kReduceTmp);
  TileUbDataDN<float, 16, 1, Rows, 1> rms_batch_dn;
  TASSIGN(rms_batch_dn, Layout::kRms);
  TileUbDataND<float, 1, 16, 1, Rows> rms_batch;
  TASSIGN(rms_batch, Layout::kRms);
  TileUbDataND<float, 1, 16, 1, Rows> rms_sqrt_batch;
  TASSIGN(rms_sqrt_batch, Layout::kRmsSqrt);
  TileUbDataND<bfloat16_t, Rows, kHeadDim> final_half_batch;
  TASSIGN(final_half_batch, Layout::kFinalHalf);
  TileUbDataND<float, 1, kHeadDim> norm_weight;
  TASSIGN(norm_weight, kUbNormWeight);

  for (int32_t row = 0; row < Rows; ++row) {
    TileUbDataND<bfloat16_t, 1, kHeadDim> cached_readout;
    TASSIGN(cached_readout,
            kUbCachedReadoutHalf + row * kHeadDim * sizeof(bfloat16_t));
    TileUbDataND<bfloat16_t, 1, kHeadDim> cached_z;
    TASSIGN(cached_z, kUbCachedZHalf + row * kHeadDim * sizeof(bfloat16_t));
    TileUbDataND<float, 1, kHeadDim> input;
    TASSIGN(input, Layout::kInput + row * kHeadDim * sizeof(float));
    TileUbDataND<float, 1, kHeadDim> z;
    TASSIGN(z, Layout::kZ + row * kHeadDim * sizeof(float));
    TCVT(input, cached_readout, RoundMode::CAST_NONE);
    TCVT(z, cached_z, RoundMode::CAST_NONE);
  }
  NormVectorBarrier();
  TMUL(square_batch, input_batch, input_batch);
  NormVectorBarrier();
  for (int32_t row = 0; row < Rows; ++row) {
    TileUbDataND<float, 1, 64> square_low;
    TASSIGN(square_low, Layout::kSquare + row * kHeadDim * sizeof(float));
    TileUbDataND<float, 1, 64> square_high;
    TASSIGN(square_high,
            Layout::kSquare + (row * kHeadDim + 64) * sizeof(float));
    TileUbDataND<float, 1, 64> reduced;
    TASSIGN(reduced, Layout::kReduceTmp + row * 64 * sizeof(float));
    TADD(reduced, square_low, square_high);
  }
  NormVectorBarrier();
  TROWSUM(rms_batch_dn, reduce_tmp_batch, square_batch);
  NormVectorBarrier();
  TMULS(rms_batch, rms_batch, 1.0f / 128.0f);
  NormVectorBarrier();
  TADDS(rms_batch, rms_batch, 1.0e-6f);
  NormVectorBarrier();
  TSQRT(rms_sqrt_batch, rms_batch);
  NormVectorBarrier();
  TMULS(rms_batch, rms_batch, 0.0f);
  NormVectorBarrier();
  TADDS(rms_batch, rms_batch, 1.0f);
  NormVectorBarrier();
  TDIV(rms_batch, rms_batch, rms_sqrt_batch);
  NormVectorBarrier();
  TROWEXPANDMUL(input_batch, input_batch, rms_batch_dn);
  NormVectorBarrier();
  for (int32_t row = 0; row < Rows; ++row) {
    TileUbDataND<float, 1, kHeadDim> input;
    TASSIGN(input, Layout::kInput + row * kHeadDim * sizeof(float));
    TMUL(input, input, norm_weight);
  }
  NormVectorBarrier();
  // Preserve layer_norm_fwd's (value * z) / (1 + exp(-z)) order.
  TMUL(input_batch, input_batch, z_batch);
  NormVectorBarrier();
  TMULS(z_batch, z_batch, -1.0f);
  NormVectorBarrier();
  TEXP(z_batch, z_batch);
  NormVectorBarrier();
  TADDS(z_batch, z_batch, 1.0f);
  NormVectorBarrier();
  TDIV(input_batch, input_batch, z_batch);
  NormVectorBarrier();
  TCVT(final_half_batch, input_batch, RoundMode::CAST_ROUND);
  NormVectorBarrier();
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
  StoreBf16Rows<Rows>(
      out_handle + static_cast<int64_t>(first_head_index) * kHeadDim,
      Layout::kFinalHalf,
      kHeadDim);
}

template <int32_t Rows, bool FlaSsmStateLayout, bool UseRegBase = false>
AICORE PTO_INLINE void RunHeads(__gm__ bfloat16_t* z_handle,
                                __gm__ bfloat16_t* conv_out_handle,
                                __gm__ float* ssm_state_handle,
                                __gm__ int* read_state_indices_handle,
                                __gm__ int* write_state_indices_handle,
                                __gm__ bfloat16_t* norm_weight_handle,
                                __gm__ float* ssm_state_out_handle,
                                __gm__ bfloat16_t* out_handle,
                                int32_t num_k_heads,
                                int32_t num_v_heads,
                                int32_t first_head_index) {
  static_assert(Rows >= 1 && Rows <= 4);
  const int32_t conv_dim = (2 * num_k_heads + num_v_heads) * kHeadDim;
  const int32_t v_heads_per_k = num_v_heads / num_k_heads;
  const int32_t ssm_state_stride = num_v_heads * kSsmHeadElements;

  TileUbDataND<bfloat16_t, 1, kHeadDim> q_half;
  TASSIGN(q_half, kUbQHalf);
  TileUbDataND<bfloat16_t, 1, kHeadDim> k_half;
  TASSIGN(k_half, kUbKHalf);
  TileUbDataND<float, 1, kHeadDim> q;
  TASSIGN(q, kUbQ);
  TileUbDataND<float, 1, kHeadDim> k;
  TASSIGN(k, kUbK);
  TileUbDataND<float, 1, kHeadDim> norm_square;
  TASSIGN(norm_square, kUbNormSquare);
  TileUbDataND<float, 1, 8, 1, 1> norm_value;
  TASSIGN(norm_value, kUbNormValue);
  TileUbDataND<uint8_t, 128, 64> reduce_tmp;
  TASSIGN(reduce_tmp, kUbReduceTmp);
  TileUbDataND<float, 1, 8, 1, 1> scalar_tmp;
  TASSIGN(scalar_tmp, kUbScalarTmp);
  TileUbDataND<bfloat16_t, 1, kHeadDim> v_half;
  TASSIGN(v_half, kUbVHalf);
  TileUbDataND<float, 1, kHeadDim> v;
  TASSIGN(v, kUbV);
  TileUbDataND<float, 128, 128> state0;
  TASSIGN(state0, kUbState0);
  TileUbDataND<float, 128, 128> state1;
  TASSIGN(state1, kUbState1);
  TileUbDataND<float, 1, kHeadDim> prediction;
  TASSIGN(prediction, kUbPrediction);
  TileUbDataND<float, 1, kHeadDim> delta;
  TASSIGN(delta, kUbDelta);
  TileUbDataND<float, 32, kHeadDim> colsum_tmp;
  TASSIGN(colsum_tmp, kUbColumnSumTmp);
  TileUbDataND<float, 1, 64> a_cache;
  TASSIGN(a_cache, kUbACache);
  TileUbDataND<float, 1, 64> b_cache;
  TASSIGN(b_cache, kUbBCache);
  TileUbDataND<bfloat16_t, 1, kHeadDim> norm_weight_half;
  TASSIGN(norm_weight_half, kUbNormWeightHalf);
  TileUbDataND<float, 1, kHeadDim> norm_weight;
  TASSIGN(norm_weight, kUbNormWeight);

  pipe_barrier(PIPE_ALL);
  const int32_t first_batch_idx = first_head_index / num_v_heads;
  const int32_t first_head_idx = first_head_index % num_v_heads;
  const int32_t first_read_state_idx =
      *(read_state_indices_handle + first_batch_idx);
  CopyGmToUb<float,
             float,
             1,
             1,
             1,
             128,
             128,
             kMaxSsmStateElements,
             kMaxNumVHeads * kSsmHeadElements,
             kSsmHeadElements,
             128,
             1,
             128,
             128,
             pto::PadValue::Zero>(ssm_state_handle +
                                      first_read_state_idx * ssm_state_stride +
                                      first_head_idx * kSsmHeadElements,
                                  kUbState0,
                                  0,
                                  128,
                                  128);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);

  int32_t cached_qk_group = -1;
  for (int32_t task_idx = 0; task_idx < Rows; ++task_idx) {
    const int32_t head_index = first_head_index + task_idx;
    const int32_t batch_idx = head_index / num_v_heads;
    const int32_t head_idx = head_index % num_v_heads;
    const int32_t qk_head_idx = head_idx / v_heads_per_k;
    const int32_t qk_group = batch_idx * num_k_heads + qk_head_idx;
    const bool load_qk = qk_group != cached_qk_group;
    if (load_qk) {
      CopyGmToUb<bfloat16_t,
                 bfloat16_t,
                 1,
                 1,
                 1,
                 1,
                 128,
                 1,
                 1,
                 1,
                 kMaxBatchSize * kMaxConvDim,
                 1,
                 1,
                 128,
                 pto::PadValue::Zero>(
          conv_out_handle + batch_idx * conv_dim + qk_head_idx * kHeadDim,
          kUbQHalf,
          0,
          1,
          128);
      CopyGmToUb<bfloat16_t,
                 bfloat16_t,
                 1,
                 1,
                 1,
                 1,
                 128,
                 1,
                 1,
                 1,
                 kMaxBatchSize * kMaxConvDim,
                 1,
                 1,
                 128,
                 pto::PadValue::Zero>(conv_out_handle + batch_idx * conv_dim +
                                          num_k_heads * kHeadDim +
                                          qk_head_idx * kHeadDim,
                                      kUbKHalf,
                                      0,
                                      1,
                                      128);
    }
    CopyGmToUb<bfloat16_t,
               bfloat16_t,
               1,
               1,
               1,
               1,
               128,
               1,
               1,
               1,
               kMaxBatchSize * kMaxConvDim,
               1,
               1,
               128,
               pto::PadValue::Zero>(conv_out_handle + batch_idx * conv_dim +
                                        2 * num_k_heads * kHeadDim +
                                        head_idx * kHeadDim,
                                    kUbVHalf,
                                    0,
                                    1,
                                    128);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    if (load_qk) {
      TCVT(q, q_half, RoundMode::CAST_NONE);
      TCVT(k, k_half, RoundMode::CAST_NONE);
    }
    TCVT(v, v_half, RoundMode::CAST_NONE);
    VectorBarrier();
    if (load_qk) {
      NormalizeQk128<true>(q, norm_square, norm_value, reduce_tmp, scalar_tmp);
      NormalizeQk128<false>(k, norm_square, norm_value, reduce_tmp, scalar_tmp);
      cached_qk_group = qk_group;
    }

    if (task_idx == 0) {
      LoadBf16Rows<Rows>(
          z_handle + static_cast<int64_t>(first_head_index) * kHeadDim,
          kUbCachedZHalf,
          kHeadDim);
      CopyGmToUb<bfloat16_t,
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
          norm_weight_handle, kUbNormWeightHalf, 0, 1, 128);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    }

    const float decay = a_cache.GetValue(task_idx);
    const float beta_gate = b_cache.GetValue(task_idx);
    const int32_t cached_readout_address =
        kUbCachedReadoutHalf + task_idx * kHeadDim * sizeof(bfloat16_t);
    if ((task_idx & 1) == 0) {
      RunRecurrentHead<FlaSsmStateLayout, UseRegBase>(state0,
                                                      state1,
                                                      q,
                                                      k,
                                                      v,
                                                      prediction,
                                                      delta,
                                                      colsum_tmp,
                                                      decay,
                                                      beta_gate,
                                                      cached_readout_address);
    } else {
      RunRecurrentHead<FlaSsmStateLayout, UseRegBase>(state1,
                                                      state0,
                                                      q,
                                                      k,
                                                      v,
                                                      prediction,
                                                      delta,
                                                      colsum_tmp,
                                                      decay,
                                                      beta_gate,
                                                      cached_readout_address);
    }

    if (task_idx == 0) {
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TCVT(norm_weight, norm_weight_half, RoundMode::CAST_NONE);
      VectorBarrier();
    }

    const int32_t write_state_idx = *(write_state_indices_handle + batch_idx);
    const int32_t state_address = (task_idx & 1) == 0 ? kUbState0 : kUbState1;
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    CopyUbToGm<float,
               float,
               1,
               1,
               1,
               128,
               128,
               kMaxSsmStateElements,
               kMaxNumVHeads * kSsmHeadElements,
               kSsmHeadElements,
               128,
               1,
               128,
               128>(ssm_state_out_handle + write_state_idx * ssm_state_stride +
                        head_idx * kSsmHeadElements,
                    state_address,
                    0,
                    128,
                    128);

    if (task_idx + 1 < Rows) {
      const int32_t next_head_index = head_index + 1;
      const int32_t next_batch_idx = next_head_index / num_v_heads;
      const int32_t next_head_idx = next_head_index % num_v_heads;
      const int32_t next_read_state_idx =
          *(read_state_indices_handle + next_batch_idx);
      const int32_t next_state_address =
          (task_idx & 1) == 0 ? kUbState1 : kUbState0;
      CopyGmToUb<float,
                 float,
                 1,
                 1,
                 1,
                 128,
                 128,
                 kMaxSsmStateElements,
                 kMaxNumVHeads * kSsmHeadElements,
                 kSsmHeadElements,
                 128,
                 1,
                 128,
                 128,
                 pto::PadValue::Zero>(
          ssm_state_handle + next_read_state_idx * ssm_state_stride +
              next_head_idx * kSsmHeadElements,
          next_state_address,
          0,
          128,
          128);
      // The current MTE3 state store and next MTE2 state load target
      // disjoint UB/GM ranges and therefore run concurrently.  Drain
      // both before swapping their state/scratch roles.
      pipe_barrier(PIPE_ALL);
    }
  }

  if constexpr ((Rows & 1) == 0) {
    RunDeferredNorm<Rows, kUbState0>(out_handle, first_head_index);
  } else {
    RunDeferredNorm<Rows, kUbState1>(out_handle, first_head_index);
  }
}

}  // namespace a5_b4_deferred
#endif

// The generated PTO kernel body follows. The tile shape stays fixed while
// model-dependent tensor strides and loop bounds come from host tiling data.

template <bool IsBatchOne,
          bool FlaSsmStateLayout,
          bool UseA5B4DeferredNorm = false,
          bool UseA5B4RegBase = false,
          bool SplitStateIndices = false>
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
                           __gm__ bfloat16_t* norm_weight_handle,
                           __gm__ bfloat16_t* conv_out_handle,
                           __gm__ bfloat16_t* conv_state_out_handle,
                           __gm__ float* ssm_state_out_handle,
                           __gm__ bfloat16_t* out_handle,
                           int32_t num_k_heads,
                           int32_t num_v_heads,
                           int32_t runtime_batch_size) {
  // Static UB layout. Regions with multiple names are intentionally reused
  // after the previous value's last consumer.
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
  constexpr int32_t kUbConvSaveHalf0 = 7936;
  constexpr int32_t kUbConvSaveHalf1 = 8192;
  constexpr int32_t kUbConvSaveHalf2 = 8448;
  constexpr int32_t kUbQHalf = 8704;
  constexpr int32_t kUbKHalf = 8960;
  constexpr int32_t kUbAHalf = 9216;
  constexpr int32_t kUbBHalf = 9248;
  constexpr int32_t kUbQOrGatherIndices = 9280;
  constexpr int32_t kUbGatherWork = 9536;
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
  constexpr int32_t kUbACacheHalfOrScratch = 173056;
  constexpr int32_t kUbBCacheHalf = 173184;
  constexpr int32_t kUbACache = 173312;
  constexpr int32_t kUbBCache = 173568;
  constexpr int32_t kUbALogCache = 173824;
  constexpr int32_t kUbDtBiasCache = 174080;
#if defined(PTO_NPU_ARCH_A5)
  constexpr int32_t kUbGateInputCache = 174336;
#endif
  constexpr int32_t kStateIndexWidth = SplitStateIndices ? 2 : 1;
  const int32_t batch_size = IsBatchOne ? 1 : runtime_batch_size;
  const int32_t conv_dim = (2 * num_k_heads + num_v_heads) * kHeadDim;
  const int32_t conv_tile_count = conv_dim / kHeadDim;
  const int32_t conv_state_stride = kConvStateLen * conv_dim;
  const int32_t v_heads_per_k = num_v_heads / num_k_heads;
  const int32_t v_width = num_v_heads * kHeadDim;
  const int32_t ssm_state_stride = num_v_heads * kSsmHeadElements;
  auto cid = get_block_idx();

  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> w_half0;
  TASSIGN(w_half0, kUbConvWeightHalf0);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> w_half1;
  TASSIGN(w_half1, kUbConvWeightHalf1);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> w_half2;
  TASSIGN(w_half2, kUbConvWeightHalf2);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> w_half3;
  TASSIGN(w_half3, kUbConvWeightHalf3);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> w0;
  TASSIGN(w0, kUbConvWeight0);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> w1;
  TASSIGN(w1, kUbConvWeight1);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> w2;
  TASSIGN(w2, kUbConvWeight2);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> w3;
  TASSIGN(w3, kUbConvWeight3);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> hist_half0;
  TASSIGN(hist_half0, kUbConvHistoryHalf0);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> hist_half1;
  TASSIGN(hist_half1, kUbConvHistoryHalf1);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> hist_half2;
  TASSIGN(hist_half2, kUbConvHistoryHalf2);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> x_half;
  TASSIGN(x_half, kUbConvInputHalf);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> hist0;
  TASSIGN(hist0, kUbConvHistory0);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> hist1;
  TASSIGN(hist1, kUbConvHistory1);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> hist2;
  TASSIGN(hist2, kUbConvHistory2);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> x_fp32;
  TASSIGN(x_fp32, kUbConvInput);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> conv_acc;
  TASSIGN(conv_acc, kUbConvAcc);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> conv_tmp;
  TASSIGN(conv_tmp, kUbConvTmp);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> conv_y;
  TASSIGN(conv_y, kUbConvOutput);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> y_half;
  TASSIGN(y_half, kUbConvOutputHalf);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> save_half0;
  TASSIGN(save_half0, kUbConvSaveHalf0);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> save_half1;
  TASSIGN(save_half1, kUbConvSaveHalf1);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> save_half2;
  TASSIGN(save_half2, kUbConvSaveHalf2);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> q_half;
  TASSIGN(q_half, kUbQHalf);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> k_half;
  TASSIGN(k_half, kUbKHalf);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 16, 1, 1> a_half;
  TASSIGN(a_half, kUbAHalf);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 16, 1, 1> b_half;
  TASSIGN(b_half, kUbBHalf);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> q_fp32;
  TASSIGN(q_fp32, kUbQOrGatherIndices);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> k_fp32;
  TASSIGN(k_fp32, kUbK);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar;
  TASSIGN(scalar, kUbScalar);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar2;
  TASSIGN(scalar2, kUbScalar2);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> norm_sq;
  TASSIGN(norm_sq, kUbNormSquare);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> norm_val;
  TASSIGN(norm_val, kUbNormValue);
  mega_gdn_decode_pto::TileUbDataND<uint8_t, 128, 64, 128, 64> tmp_ub;
  TASSIGN(tmp_ub, kUbReduceTmp);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp;
  TASSIGN(scalar_tmp, kUbScalarTmp);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> exp_a_buf;
  TASSIGN(exp_a_buf, kUbExpA);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_work;
  TASSIGN(scalar_work, kUbScalarWork);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> norm_half;
  TASSIGN(norm_half, kUbNormHalf);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> z_half;
  TASSIGN(z_half, kUbZHalf);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> weight_half;
  TASSIGN(weight_half, kUbNormWeightHalf);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> norm_fp32;
  TASSIGN(norm_fp32, kUbNorm);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> z_fp32;
  TASSIGN(z_fp32, kUbZ);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> weight_fp32;
  TASSIGN(weight_fp32, kUbNormWeight);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> square_fp32;
  TASSIGN(square_fp32, kUbSquare);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> rms;
  TASSIGN(rms, kUbRms);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> gate_fp32;
  TASSIGN(gate_fp32, kUbGate);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> final_half;
  TASSIGN(final_half, kUbFinalHalf);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> v_half;
  TASSIGN(v_half, kUbVHalf);
  mega_gdn_decode_pto::TileUbDataND<float, 128, 128, 128, 128> h_vec;
  TASSIGN(h_vec, kUbState);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> v_fp32;
  TASSIGN(v_fp32, kUbV);
  mega_gdn_decode_pto::TileUbDataND<float, 128, 128, 128, 128> compute_buf;
  TASSIGN(compute_buf, kUbStateCompute);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> pred;
  TASSIGN(pred, kUbPrediction);
  // The reduction scratch ends before PTO's fixed TMP_UB_OFFSET.
  mega_gdn_decode_pto::TileUbDataND<float, 32, 128, 32, 128> colsum_tmp;
  TASSIGN(colsum_tmp, kUbColumnSumTmp);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 64> a_cache_half;
  TASSIGN(a_cache_half, kUbACacheHalfOrScratch);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 64> b_cache_half;
  TASSIGN(b_cache_half, kUbBCacheHalf);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 64> a_cache;
  TASSIGN(a_cache, kUbACache);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 64> b_cache;
  TASSIGN(b_cache, kUbBCache);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 64> a_log_cache;
  TASSIGN(a_log_cache, kUbALogCache);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 64> dt_bias_cache;
  TASSIGN(dt_bias_cache, kUbDtBiasCache);
  mega_gdn_decode_pto::TileUbDataND<float, 1, 128, 1, 128> delta;
  TASSIGN(delta, kUbDelta);
  mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> out_half;
  TASSIGN(out_half, kUbOutputHalf);
#if defined(__DAV_VEC__) || defined(__DAV_C220_VEC__)
#if defined(PTO_NPU_ARCH_A2A3)
  const auto vid = get_subblockid();
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int32_t vector_core_idx = cid * get_subblockdim() + vid;
  const int32_t vector_core_count = get_block_num() * get_subblockdim();
#else
  const int32_t vector_core_idx = cid;
  const int32_t vector_core_count = get_block_num();
#endif
  const int32_t batch_one_read_state_idx =
      IsBatchOne ? *read_state_indices_handle : 0;
  const int32_t batch_one_write_state_idx =
      IsBatchOne ? *write_state_indices_handle : 0;

  // Phase 1: causal convolution and convolution-state update.
  for (int32_t conv_tile = vector_core_idx; conv_tile < conv_tile_count;
       conv_tile += vector_core_count) {
    const int32_t channel_offset = conv_tile * kHeadDim;
    if constexpr (IsBatchOne) {
      mega_gdn_decode_pto::CopyConvWeightsGmToUb(
          conv_weight_handle + channel_offset, kUbConvWeightHalf0, conv_dim);
    } else {
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
                                      kMaxConvWeightElements,
                                      1,
                                      1,
                                      128,
                                      pto::PadValue::Zero>(
          conv_weight_handle + channel_offset, kUbConvWeightHalf0, 0, 1, 128);
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
                                      kMaxConvWeightElements,
                                      1,
                                      1,
                                      128,
                                      pto::PadValue::Zero>(
          conv_weight_handle + channel_offset + conv_dim,
          kUbConvWeightHalf1,
          0,
          1,
          128);
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
                                      kMaxConvWeightElements,
                                      1,
                                      1,
                                      128,
                                      pto::PadValue::Zero>(
          conv_weight_handle + channel_offset + 2 * conv_dim,
          kUbConvWeightHalf2,
          0,
          1,
          128);
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
                                      kMaxConvWeightElements,
                                      1,
                                      1,
                                      128,
                                      pto::PadValue::Zero>(
          conv_weight_handle + channel_offset + 3 * conv_dim,
          kUbConvWeightHalf3,
          0,
          1,
          128);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TCVT(w0, w_half0, RoundMode::CAST_NONE);
    TCVT(w1, w_half1, RoundMode::CAST_NONE);
    TCVT(w2, w_half2, RoundMode::CAST_NONE);
    TCVT(w3, w_half3, RoundMode::CAST_NONE);

    if constexpr (!IsBatchOne && !SplitStateIndices) {
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

      const int32_t buffer0_read_state_idx = *read_state_indices_handle;
      int32_t buffer0_write_state_idx = *write_state_indices_handle;
      mega_gdn_decode_pto::PrefetchConvBatch<kConvBuffer0,
                                             EVENT_ID2,
                                             EVENT_ID0>(qkv_handle,
                                                        conv_state_handle,
                                                        0,
                                                        buffer0_read_state_idx,
                                                        conv_dim,
                                                        conv_state_stride,
                                                        channel_offset);

      for (int32_t batch_pair = 0; batch_pair < batch_size; batch_pair += 2) {
        const int32_t buffer1_batch_idx = batch_pair + 1;
        int32_t buffer1_read_state_idx = 0;
        int32_t buffer1_write_state_idx = 0;
        if (buffer1_batch_idx < batch_size) {
          buffer1_read_state_idx =
              *(read_state_indices_handle + buffer1_batch_idx);
          buffer1_write_state_idx =
              *(write_state_indices_handle + buffer1_batch_idx);
          mega_gdn_decode_pto::
              PrefetchConvBatch<kConvBuffer1, EVENT_ID3, EVENT_ID1>(
                  qkv_handle,
                  conv_state_handle,
                  buffer1_batch_idx,
                  buffer1_read_state_idx,
                  conv_dim,
                  conv_state_stride,
                  channel_offset);
        }

        mega_gdn_decode_pto::ComputeAndStoreConvBatch<kConvBuffer0,
                                                      EVENT_ID2,
                                                      EVENT_ID4,
                                                      EVENT_ID0>(
            conv_out_handle,
            conv_state_out_handle,
            w0,
            w1,
            w2,
            w3,
            batch_pair,
            buffer0_write_state_idx,
            conv_dim,
            conv_state_stride,
            channel_offset);

        const int32_t next_buffer0_batch_idx = batch_pair + 2;
        int32_t next_buffer0_read_state_idx = 0;
        int32_t next_buffer0_write_state_idx = 0;
        if (next_buffer0_batch_idx < batch_size) {
          next_buffer0_read_state_idx =
              *(read_state_indices_handle + next_buffer0_batch_idx);
          next_buffer0_write_state_idx =
              *(write_state_indices_handle + next_buffer0_batch_idx);
          mega_gdn_decode_pto::
              PrefetchConvBatch<kConvBuffer0, EVENT_ID2, EVENT_ID0>(
                  qkv_handle,
                  conv_state_handle,
                  next_buffer0_batch_idx,
                  next_buffer0_read_state_idx,
                  conv_dim,
                  conv_state_stride,
                  channel_offset);
        }

        if (buffer1_batch_idx < batch_size) {
          mega_gdn_decode_pto::ComputeAndStoreConvBatch<kConvBuffer1,
                                                        EVENT_ID3,
                                                        EVENT_ID5,
                                                        EVENT_ID1>(
              conv_out_handle,
              conv_state_out_handle,
              w0,
              w1,
              w2,
              w3,
              buffer1_batch_idx,
              buffer1_write_state_idx,
              conv_dim,
              conv_state_stride,
              channel_offset);
        }
        buffer0_write_state_idx = next_buffer0_write_state_idx;
      }
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    } else {
      for (int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        if constexpr (!IsBatchOne) {
          pipe_barrier(PIPE_ALL);
#if !defined(PTO_NPU_ARCH_A5)
          pipe_barrier(PIPE_ALL);
#endif
        }
        const int32_t read_state_idx =
            IsBatchOne
                ? batch_one_read_state_idx
                : *(read_state_indices_handle + batch_idx * kStateIndexWidth);
        const int32_t write_state_idx =
            IsBatchOne
                ? batch_one_write_state_idx
                : *(write_state_indices_handle + batch_idx * kStateIndexWidth);
        if constexpr (SplitStateIndices) {
          if (read_state_idx < 0 || write_state_idx < 0) {
            continue;
          }
        }
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
                                        kMaxConvStateElements,
                                        1,
                                        1,
                                        128,
                                        pto::PadValue::Zero>(
            conv_state_handle +
                static_cast<int64_t>(read_state_idx) * conv_state_stride +
                channel_offset,
            kUbConvHistoryHalf0,
            0,
            1,
            128);
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
                                        kMaxConvStateElements,
                                        1,
                                        1,
                                        128,
                                        pto::PadValue::Zero>(
            conv_state_handle +
                static_cast<int64_t>(read_state_idx) * conv_state_stride +
                channel_offset + conv_dim,
            kUbConvHistoryHalf1,
            0,
            1,
            128);
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
                                        kMaxConvStateElements,
                                        1,
                                        1,
                                        128,
                                        pto::PadValue::Zero>(
            conv_state_handle +
                static_cast<int64_t>(read_state_idx) * conv_state_stride +
                channel_offset + 2 * conv_dim,
            kUbConvHistoryHalf2,
            0,
            1,
            128);
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
                                        kMaxBatchSize * kMaxConvDim,
                                        1,
                                        1,
                                        128,
                                        pto::PadValue::Zero>(
            qkv_handle + batch_idx * conv_dim + channel_offset,
            kUbConvInputHalf,
            0,
            1,
            128);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        TCVT(hist0, hist_half0, RoundMode::CAST_NONE);
        TCVT(hist1, hist_half1, RoundMode::CAST_NONE);
        TCVT(hist2, hist_half2, RoundMode::CAST_NONE);
        TCVT(x_fp32, x_half, RoundMode::CAST_NONE);
        VectorBarrier();
        TMUL(conv_acc, w0, hist0);
#if defined(PTO_NPU_ARCH_A5)
        VectorBarrier();
        mega_gdn_decode_pto::MulAddDst<float, 1, 128>(
            conv_acc, hist1, w1, conv_tmp);
        VectorBarrier();
        mega_gdn_decode_pto::MulAddDst<float, 1, 128>(
            conv_acc, hist2, w2, conv_tmp);
        VectorBarrier();
#else
        TMUL(conv_tmp, w1, hist1);
        VectorBarrier();
        TADD(conv_acc, conv_acc, conv_tmp);
        VectorBarrier();
        TMUL(conv_tmp, w2, hist2);
        VectorBarrier();
        TADD(conv_acc, conv_acc, conv_tmp);
        VectorBarrier();
#endif
        mega_gdn_decode_pto::TileUbDataND<float, 1, 128>
            conv_acc_temp_0_muladddst_tmp;
        TASSIGN(conv_acc_temp_0_muladddst_tmp, kUbVectorScratch);
        mega_gdn_decode_pto::MulAddDst<float, 1, 128>(
            conv_acc, x_fp32, w3, conv_acc_temp_0_muladddst_tmp);
        VectorBarrier();
        mega_gdn_decode_pto::TileUbDataND<float, 1, 128> conv_y_temp_0_silu_tmp;
        TASSIGN(conv_y_temp_0_silu_tmp, kUbVectorScratch);
        mega_gdn_decode_pto::CausalConvSilu<float, 1, 128>(
            conv_y, conv_acc, conv_y_temp_0_silu_tmp);
        VectorBarrier();
        TCVT(y_half, conv_y, RoundMode::CAST_RINT);
        TCVT(save_half0, hist1, RoundMode::CAST_RINT);
        TCVT(save_half1, hist2, RoundMode::CAST_RINT);
        TCVT(save_half2, x_fp32, RoundMode::CAST_RINT);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
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
                                        kMaxBatchSize * kMaxConvDim,
                                        1,
                                        1,
                                        128>(
            conv_out_handle + batch_idx * conv_dim + channel_offset,
            kUbConvOutputHalf,
            0,
            1,
            128);
        mega_gdn_decode_pto::CopyConvHistoryUbToGm(
            conv_state_out_handle +
                static_cast<int64_t>(write_state_idx) * conv_state_stride +
                channel_offset,
            kUbConvSaveHalf0,
            conv_dim);
        if constexpr (!IsBatchOne) {
          pipe_barrier(PIPE_ALL);
#if !defined(PTO_NPU_ARCH_A5)
          pipe_barrier(PIPE_ALL);
#endif
        }
      }
    }
    if constexpr (IsBatchOne) {
      if (conv_tile + vector_core_count < conv_tile_count) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      }
    }
  }
  // Phase 2: make convolution output/state visible to every vector core.
  mega_gdn_decode_pto::SyncAllAiv();

  const int32_t total_ssm_heads = batch_size * num_v_heads;
  const bool reuse_qk = !IsBatchOne && v_heads_per_k > 1;
  // Deferred-Norm owners require contiguous rows even when ratio=1.  That
  // makes their cached rows and alternating state buffers describe the same
  // heads that RunHeads processes; Q/K is still reloaded whenever the group
  // changes, so ratio=1 never reuses a different head's Q/K.
  const bool use_contiguous_head_schedule = reuse_qk || UseA5B4DeferredNorm;
  const int32_t heads_per_core = total_ssm_heads / vector_core_count;
  const int32_t extra_head_cores = total_ssm_heads % vector_core_count;
  const int32_t contiguous_head_start =
      vector_core_idx * heads_per_core +
      (vector_core_idx < extra_head_cores ? vector_core_idx : extra_head_cores);
  const int32_t contiguous_head_count =
      heads_per_core + (vector_core_idx < extra_head_cores ? 1 : 0);
  const int32_t first_head_index =
      use_contiguous_head_schedule ? contiguous_head_start : vector_core_idx;
  const int32_t head_index_end =
      use_contiguous_head_schedule
          ? contiguous_head_start + contiguous_head_count
          : total_ssm_heads;
  const int32_t head_index_step =
      use_contiguous_head_schedule ? 1 : vector_core_count;
  int32_t cached_qk_group = -1;
  const bool cache_head_scalars =
      use_contiguous_head_schedule && contiguous_head_count > 0;

  // Phase 3: precompute per-head recurrent scalars for contiguous owners.
  if (cache_head_scalars) {
    const int32_t head_phase = contiguous_head_start % num_v_heads;
    mega_gdn_decode_pto::CopyGmToUb<bfloat16_t,
                                    bfloat16_t,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    kMaxBatchSize * kMaxNumVHeads,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    64,
                                    pto::PadValue::Null>(
        a_handle + contiguous_head_start,
        kUbACacheHalfOrScratch,
        0,
        1,
        contiguous_head_count);
    mega_gdn_decode_pto::CopyGmToUb<bfloat16_t,
                                    bfloat16_t,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    kMaxBatchSize * kMaxNumVHeads,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    64,
                                    pto::PadValue::Null>(
        b_handle + contiguous_head_start,
        kUbBCacheHalf,
        0,
        1,
        contiguous_head_count);
    mega_gdn_decode_pto::CopyGmToUb<float,
                                    float,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    kMaxNumVHeads,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    64,
                                    pto::PadValue::Null>(
        a_log_handle, kUbALogCache, 0, 1, num_v_heads);
    mega_gdn_decode_pto::CopyGmToUb<float,
                                    float,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    kMaxNumVHeads,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    64,
                                    pto::PadValue::Null>(
        dt_bias_handle, kUbDtBiasCache, 0, 1, num_v_heads);
    mega_gdn_decode_pto::
        TileUbDataND<int32_t, 1, 64, pto::DYNAMIC, pto::DYNAMIC>
            gather_indices(1, contiguous_head_count);
    TASSIGN(gather_indices, kUbQOrGatherIndices);
    mega_gdn_decode_pto::
        TileUbDataND<int32_t, 1, 64, pto::DYNAMIC, pto::DYNAMIC>
            gather_work(1, contiguous_head_count);
    TASSIGN(gather_work, kUbGatherWork);
    for (int32_t i = 0; i < contiguous_head_count; ++i) {
      int32_t source_head = head_phase + i;
      if (source_head >= num_v_heads) {
        source_head -= num_v_heads;
      }
      gather_indices.SetValue(i, source_head);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    mega_gdn_decode_pto::
        TileUbDataND<bfloat16_t, 1, 64, pto::DYNAMIC, pto::DYNAMIC>
            a_cache_half_valid(1, contiguous_head_count);
    TASSIGN(a_cache_half_valid, kUbACacheHalfOrScratch);
    mega_gdn_decode_pto::
        TileUbDataND<bfloat16_t, 1, 64, pto::DYNAMIC, pto::DYNAMIC>
            b_cache_half_valid(1, contiguous_head_count);
    TASSIGN(b_cache_half_valid, kUbBCacheHalf);
    mega_gdn_decode_pto::TileUbDataND<float, 1, 64, pto::DYNAMIC, pto::DYNAMIC>
        a_cache_valid(1, contiguous_head_count);
    TASSIGN(a_cache_valid, kUbACache);
    mega_gdn_decode_pto::TileUbDataND<float, 1, 64, pto::DYNAMIC, pto::DYNAMIC>
        b_cache_valid(1, contiguous_head_count);
    TASSIGN(b_cache_valid, kUbBCache);
    mega_gdn_decode_pto::TileUbDataND<float, 1, 64, pto::DYNAMIC, pto::DYNAMIC>
        a_log_cache_valid(1, contiguous_head_count);
    TASSIGN(a_log_cache_valid, kUbALogCache);
    mega_gdn_decode_pto::TileUbDataND<float, 1, 64, pto::DYNAMIC, pto::DYNAMIC>
        dt_bias_cache_valid(1, contiguous_head_count);
    TASSIGN(dt_bias_cache_valid, kUbDtBiasCache);
#if defined(PTO_NPU_ARCH_A5)
    mega_gdn_decode_pto::TileUbDataND<float, 1, 64, pto::DYNAMIC, pto::DYNAMIC>
        gate_input_cache(1, contiguous_head_count);
    TASSIGN(gate_input_cache, kUbGateInputCache);
#endif
    TCVT(a_cache_valid, a_cache_half_valid, RoundMode::CAST_NONE);
    TCVT(b_cache_valid, b_cache_half_valid, RoundMode::CAST_NONE);
    VectorBarrier();
    // The converted BF16 cache is dead here, so reuse it as vector scratch.
    mega_gdn_decode_pto::TileUbDataND<float, 1, 64, pto::DYNAMIC, pto::DYNAMIC>
        gate_tmp(1, contiguous_head_count);
    TASSIGN(gate_tmp, kUbACacheHalfOrScratch);
    TGATHER(gate_tmp, a_log_cache, gather_indices, gather_work);
    VectorBarrier();
    TMOV(a_log_cache_valid, gate_tmp);
    VectorBarrier();
    TGATHER(gate_tmp, dt_bias_cache, gather_indices, gather_work);
    VectorBarrier();
    TMOV(dt_bias_cache_valid, gate_tmp);
    VectorBarrier();
    TEXP(a_log_cache_valid, a_log_cache_valid);
    VectorBarrier();
    TMULS(gate_tmp, b_cache_valid, -1.000000e+00f);
    VectorBarrier();
    TEXP(gate_tmp, gate_tmp);
    VectorBarrier();
    TADDS(gate_tmp, gate_tmp, 1.000000e+00f);
    VectorBarrier();
    TRECIP(b_cache_valid, gate_tmp);
    VectorBarrier();
    TADD(a_cache_valid, a_cache_valid, dt_bias_cache_valid);
    VectorBarrier();
#if defined(PTO_NPU_ARCH_A5)
    TMOV(gate_input_cache, a_cache_valid);
    VectorBarrier();
    TMINS(gate_tmp, a_cache_valid, 2.000000e+01f);
    VectorBarrier();
    TEXP(gate_tmp, gate_tmp);
    VectorBarrier();
    TADDS(dt_bias_cache_valid, gate_tmp, 1.000000e+00f);
    VectorBarrier();
    TADDS(a_cache_valid, dt_bias_cache_valid, -1.000000e+00f);
    VectorBarrier();
    TSUB(a_cache_valid, gate_tmp, a_cache_valid);
    VectorBarrier();
    TDIV(a_cache_valid, a_cache_valid, dt_bias_cache_valid);
    VectorBarrier();
    TLOG(dt_bias_cache_valid, dt_bias_cache_valid);
    VectorBarrier();
    TADD(gate_tmp, dt_bias_cache_valid, a_cache_valid);
    VectorBarrier();
    TMAX(dt_bias_cache_valid, gate_input_cache, gate_tmp);
#else
    TMINS(gate_tmp, a_cache_valid, 2.000000e+01f);
    VectorBarrier();
    TEXP(gate_tmp, gate_tmp);
    VectorBarrier();
    TADDS(gate_tmp, gate_tmp, 1.000000e+00f);
    VectorBarrier();
    TLOG(gate_tmp, gate_tmp);
    VectorBarrier();
    TMAX(dt_bias_cache_valid, a_cache_valid, gate_tmp);
#endif
    VectorBarrier();
    TMUL(a_cache_valid, a_log_cache_valid, dt_bias_cache_valid);
    VectorBarrier();
    TMULS(a_cache_valid, a_cache_valid, -1.000000e+00f);
    VectorBarrier();
    TEXP(a_cache_valid, a_cache_valid);
    set_flag(PIPE_V, PIPE_S, EVENT_ID6);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID6);
  }

#if defined(PTO_NPU_ARCH_A5)
  if constexpr (UseA5B4DeferredNorm) {
    static_assert(!IsBatchOne && FlaSsmStateLayout);
    static_assert(!UseA5B4RegBase || UseA5B4DeferredNorm);
    if (contiguous_head_count == 4) {
      a5_b4_deferred::RunHeads<4, FlaSsmStateLayout, UseA5B4RegBase>(
          z_handle,
          conv_out_handle,
          ssm_state_handle,
          read_state_indices_handle,
          write_state_indices_handle,
          norm_weight_handle,
          ssm_state_out_handle,
          out_handle,
          num_k_heads,
          num_v_heads,
          contiguous_head_start);
    } else if (contiguous_head_count == 3) {
      a5_b4_deferred::RunHeads<3, FlaSsmStateLayout, UseA5B4RegBase>(
          z_handle,
          conv_out_handle,
          ssm_state_handle,
          read_state_indices_handle,
          write_state_indices_handle,
          norm_weight_handle,
          ssm_state_out_handle,
          out_handle,
          num_k_heads,
          num_v_heads,
          contiguous_head_start);
    } else if (contiguous_head_count == 2) {
      a5_b4_deferred::RunHeads<2, FlaSsmStateLayout, UseA5B4RegBase>(
          z_handle,
          conv_out_handle,
          ssm_state_handle,
          read_state_indices_handle,
          write_state_indices_handle,
          norm_weight_handle,
          ssm_state_out_handle,
          out_handle,
          num_k_heads,
          num_v_heads,
          contiguous_head_start);
    } else if (contiguous_head_count == 1) {
      a5_b4_deferred::RunHeads<1, FlaSsmStateLayout, UseA5B4RegBase>(
          z_handle,
          conv_out_handle,
          ssm_state_handle,
          read_state_indices_handle,
          write_state_indices_handle,
          norm_weight_handle,
          ssm_state_out_handle,
          out_handle,
          num_k_heads,
          num_v_heads,
          contiguous_head_start);
    }
  } else {
#endif
    // Phase 4: recurrent GDN update, RMSNorm, gate, and final output.
    bool norm_weight_loaded = false;
    for (int32_t head_index = first_head_index; head_index < head_index_end;
         head_index += head_index_step) {
      if constexpr (!IsBatchOne) {
        pipe_barrier(PIPE_ALL);
#if !defined(PTO_NPU_ARCH_A5)
        pipe_barrier(PIPE_ALL);
#endif
      }
      const int32_t batch_idx = IsBatchOne ? 0 : head_index / num_v_heads;
      const int32_t head_idx =
          IsBatchOne ? head_index : head_index % num_v_heads;
      const int32_t qk_head_idx = head_idx / v_heads_per_k;
      const int32_t qk_group = batch_idx * num_k_heads + qk_head_idx;
      const bool load_qk = !reuse_qk || qk_group != cached_qk_group;
#if defined(PTO_NPU_ARCH_A5)
      // norm_weight is shared by every batch item and value head.  Its UB
      // regions are not reused by the recurrent loop, so one load/conversion
      // per owner is sufficient even when Q/K itself cannot be reused (NV ==
      // NK).
      const bool load_norm_weight =
          (SplitStateIndices ? !norm_weight_loaded
                             : head_index == first_head_index);
#else
    const bool load_norm_weight =
        !reuse_qk || (SplitStateIndices ? !norm_weight_loaded
                                        : head_index == first_head_index);
#endif
      const int32_t read_state_idx =
          *(read_state_indices_handle + batch_idx * kStateIndexWidth +
            kStateIndexWidth - 1);
      const int32_t write_state_idx =
          *(write_state_indices_handle + batch_idx * kStateIndexWidth +
            kStateIndexWidth - 1);
      if constexpr (SplitStateIndices) {
        if (read_state_idx < 0 || write_state_idx < 0) {
          TEXPANDS(final_half, static_cast<bfloat16_t>(0));
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
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
                                          kMaxBatchSize * kMaxNumVHeads *
                                              kHeadDim,
                                          1,
                                          1,
                                          128>(
              out_handle + batch_idx * v_width + head_idx * kHeadDim,
              kUbFinalHalf,
              0,
              1,
              128);
          pipe_barrier(PIPE_ALL);
          continue;
        }
      }
      norm_weight_loaded = true;
      if (load_qk) {
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
                                        kMaxBatchSize * kMaxConvDim,
                                        1,
                                        1,
                                        128,
                                        pto::PadValue::Zero>(
            conv_out_handle + batch_idx * conv_dim + qk_head_idx * kHeadDim,
            kUbQHalf,
            0,
            1,
            128);
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
                                        kMaxBatchSize * kMaxConvDim,
                                        1,
                                        1,
                                        128,
                                        pto::PadValue::Zero>(
            conv_out_handle + batch_idx * conv_dim + num_k_heads * kHeadDim +
                qk_head_idx * kHeadDim,
            kUbKHalf,
            0,
            1,
            128);
      }
      if (!cache_head_scalars) {
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
                                        kMaxBatchSize * kMaxNumVHeads,
                                        1,
                                        1,
                                        16,
                                        pto::PadValue::Null>(
            a_handle + head_index, kUbAHalf, 0, 1, 1);
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
                                        kMaxBatchSize * kMaxNumVHeads,
                                        1,
                                        1,
                                        16,
                                        pto::PadValue::Null>(
            b_handle + head_index, kUbBHalf, 0, 1, 1);
      }
      if (load_qk || !cache_head_scalars) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      }
      if (load_qk) {
        TCVT(q_fp32, q_half, RoundMode::CAST_NONE);
        TCVT(k_fp32, k_half, RoundMode::CAST_NONE);
      }
      if (!cache_head_scalars) {
        mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 16, 1, 1>
            a_half_temp_0;
        TASSIGN(a_half_temp_0, kUbAHalf + 0 * 2);
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_temp_0;
        TASSIGN(scalar_temp_0, kUbScalar + 0 * 4);
        TCVT(scalar_temp_0, a_half_temp_0, RoundMode::CAST_NONE);
        mega_gdn_decode_pto::TileUbDataND<bfloat16_t, 1, 16, 1, 1>
            b_half_temp_0;
        TASSIGN(b_half_temp_0, kUbBHalf + 0 * 2);
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar2_temp_0;
        TASSIGN(scalar2_temp_0, kUbScalar2 + 0 * 4);
        TCVT(scalar2_temp_0, b_half_temp_0, RoundMode::CAST_NONE);
      }
      VectorBarrier();
      if (load_qk) {
        mega_gdn_decode_pto::NormalizeQk128<true>(
            q_fp32, norm_sq, norm_val, tmp_ub, scalar_tmp);
        mega_gdn_decode_pto::NormalizeQk128<false>(
            k_fp32, norm_sq, norm_val, tmp_ub, scalar_tmp);
        cached_qk_group = qk_group;
      }
      float decay;
      float beta_gate;
      if (cache_head_scalars) {
        const int32_t local_head_index = head_index - contiguous_head_start;
        decay = a_cache.GetValue(local_head_index);
        beta_gate = b_cache.GetValue(local_head_index);
      } else {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
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
                                        kMaxNumVHeads,
                                        1,
                                        1,
                                        8,
                                        pto::PadValue::Null>(
            a_log_handle + head_idx, kUbScalarTmp, 0, 1, 1);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_2;
        TASSIGN(scalar_tmp_temp_2, kUbScalarTmp + 0 * 4);
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> exp_a_buf_temp_0;
        TASSIGN(exp_a_buf_temp_0, kUbExpA + 0 * 4);
        TEXP(exp_a_buf_temp_0, scalar_tmp_temp_2);
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
                                        kMaxNumVHeads,
                                        1,
                                        1,
                                        8,
                                        pto::PadValue::Null>(
            dt_bias_handle + head_idx, kUbNormValue, 0, 1, 1);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        VectorBarrier();
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_temp_1;
        TASSIGN(scalar_temp_1, kUbScalar + 0 * 4);
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> norm_val_temp_4;
        TASSIGN(norm_val_temp_4, kUbNormValue + 0 * 4);
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_3;
        TASSIGN(scalar_tmp_temp_3, kUbScalarTmp + 0 * 4);
        TADD(scalar_tmp_temp_3, scalar_temp_1, norm_val_temp_4);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        const float x_gate = scalar_tmp.GetValue(0);
        if (2.000000e+01f < x_gate) {
          TADDS(scalar_work, scalar_tmp, 0.000000e+00f);
        } else {
          mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1>
              scalar_tmp_temp_4;
          TASSIGN(scalar_tmp_temp_4, kUbScalarTmp + 0 * 4);
          mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1>
              scalar_work_temp_0;
          TASSIGN(scalar_work_temp_0, kUbScalarWork + 0 * 4);
          TEXP(scalar_work_temp_0, scalar_tmp_temp_4);
          VectorBarrier();
          TADDS(norm_val, scalar_work, 1.000000e+00f);
          VectorBarrier();
          mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> norm_val_temp_5;
          TASSIGN(norm_val_temp_5, kUbNormValue + 0 * 4);
          mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1>
              scalar_work_temp_1;
          TASSIGN(scalar_work_temp_1, kUbScalarWork + 0 * 4);
#if defined(PTO_NPU_ARCH_A5)
          TADDS(scalar, norm_val_temp_5, -1.000000e+00f);
          VectorBarrier();
          TSUB(scalar, scalar_work_temp_0, scalar);
          VectorBarrier();
          TDIV(scalar, scalar, norm_val_temp_5);
          VectorBarrier();
          TLOG(scalar_work_temp_1, norm_val_temp_5);
          VectorBarrier();
          TADD(scalar_work_temp_1, scalar_work_temp_1, scalar);
#else
        TLOG(scalar_work_temp_1, norm_val_temp_5);
#endif
        }
        pipe_barrier(PIPE_ALL);
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> exp_a_buf_temp_1;
        TASSIGN(exp_a_buf_temp_1, kUbExpA + 0 * 4);
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_work_temp_2;
        TASSIGN(scalar_work_temp_2, kUbScalarWork + 0 * 4);
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_5;
        TASSIGN(scalar_tmp_temp_5, kUbScalarTmp + 0 * 4);
        TMUL(scalar_tmp_temp_5, exp_a_buf_temp_1, scalar_work_temp_2);
        VectorBarrier();
        TMULS(scalar_tmp, scalar_tmp, -1.000000e+00f);
        VectorBarrier();
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_6;
        TASSIGN(scalar_tmp_temp_6, kUbScalarTmp + 0 * 4);
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_work_temp_3;
        TASSIGN(scalar_work_temp_3, kUbScalarWork + 0 * 4);
        TEXP(scalar_work_temp_3, scalar_tmp_temp_6);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        decay = scalar_work.GetValue(0);
        VectorBarrier();
        TMULS(scalar_tmp, scalar2, -1.000000e+00f);
        VectorBarrier();
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_7;
        TASSIGN(scalar_tmp_temp_7, kUbScalarTmp + 0 * 4);
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_work_temp_4;
        TASSIGN(scalar_work_temp_4, kUbScalarWork + 0 * 4);
        TEXP(scalar_work_temp_4, scalar_tmp_temp_7);
        VectorBarrier();
        TADDS(scalar_tmp, scalar_work, 1.000000e+00f);
        VectorBarrier();
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_8;
        TASSIGN(scalar_tmp_temp_8, kUbScalarTmp + 0 * 4);
        mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_work_temp_5;
        TASSIGN(scalar_work_temp_5, kUbScalarWork + 0 * 4);
        TRECIP(scalar_work_temp_5, scalar_tmp_temp_8);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        beta_gate = scalar_work.GetValue(0);
      }
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
                                      kMaxBatchSize * kMaxConvDim,
                                      1,
                                      1,
                                      128,
                                      pto::PadValue::Zero>(
          conv_out_handle + batch_idx * conv_dim + 2 * num_k_heads * kHeadDim +
              head_idx * kHeadDim,
          kUbVHalf,
          0,
          1,
          128);
      mega_gdn_decode_pto::CopyGmToUb<float,
                                      float,
                                      1,
                                      1,
                                      1,
                                      128,
                                      128,
                                      kMaxSsmStateElements,
                                      kMaxNumVHeads * kSsmHeadElements,
                                      kSsmHeadElements,
                                      128,
                                      1,
                                      128,
                                      128,
                                      pto::PadValue::Zero>(
          ssm_state_handle +
              static_cast<int64_t>(read_state_idx) * ssm_state_stride +
              head_idx * kSsmHeadElements,
          kUbState,
          0,
          128,
          128);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TCVT(v_fp32, v_half, RoundMode::CAST_NONE);
      TMULS(h_vec, h_vec, decay);
      VectorBarrier();
      mega_gdn_decode_pto::StateVectorProduct128<FlaSsmStateLayout>(
          pred, h_vec, k_fp32, compute_buf, colsum_tmp);
      TSUB(delta, v_fp32, pred);
      VectorBarrier();
      TMULS(delta, delta, beta_gate);
      VectorBarrier();
      mega_gdn_decode_pto::StateRankOneUpdate128<FlaSsmStateLayout>(
          h_vec, k_fp32, delta, compute_buf);
      mega_gdn_decode_pto::StateVectorProduct128<FlaSsmStateLayout>(
          pred, h_vec, q_fp32, compute_buf, colsum_tmp);
      TCVT(out_half, pred, RoundMode::CAST_RINT);
      VectorBarrier();
      TMOV(norm_half, out_half);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      mega_gdn_decode_pto::CopyUbToGm<float,
                                      float,
                                      1,
                                      1,
                                      1,
                                      128,
                                      128,
                                      kMaxSsmStateElements,
                                      kMaxNumVHeads * kSsmHeadElements,
                                      kSsmHeadElements,
                                      128,
                                      1,
                                      128,
                                      128>(
          ssm_state_out_handle +
              static_cast<int64_t>(write_state_idx) * ssm_state_stride +
              head_idx * kSsmHeadElements,
          kUbState,
          0,
          128,
          128);
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
                                      kMaxBatchSize * kMaxNumVHeads * kHeadDim,
                                      1,
                                      1,
                                      128,
                                      pto::PadValue::Zero>(
          z_handle + batch_idx * v_width + head_idx * kHeadDim,
          kUbZHalf,
          0,
          1,
          128);
      if (load_norm_weight) {
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
            norm_weight_handle, kUbNormWeightHalf, 0, 1, 128);
      }
      VectorBarrier();
      TCVT(norm_fp32, norm_half, RoundMode::CAST_NONE);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TCVT(z_fp32, z_half, RoundMode::CAST_NONE);
      if (load_norm_weight) {
        TCVT(weight_fp32, weight_half, RoundMode::CAST_NONE);
      }
      VectorBarrier();
      TMUL(square_fp32, norm_fp32, norm_fp32);
      VectorBarrier();
      mega_gdn_decode_pto::TileUbDataDN<float, 8, 1, 1, 1> rms_temp_0;
      TASSIGN(rms_temp_0, kUbRms + 0 * 4);
      mega_gdn_decode_pto::TileUbDataND<float, 1, 64, 1, 64> tmp_ub_temp_4;
      TASSIGN(tmp_ub_temp_4, kUbReduceTmp + 0 * 4);
#if defined(PTO_NPU_ARCH_A5)
      mega_gdn_decode_pto::TileUbDataND<float, 1, 64> square_low;
      TASSIGN(square_low, kUbSquare);
      mega_gdn_decode_pto::TileUbDataND<float, 1, 64> square_high;
      TASSIGN(square_high, kUbSquare + 64 * sizeof(float));
      // Match layer_norm_fwd's N=128 reduction tree.
      TADD(tmp_ub_temp_4, square_low, square_high);
      VectorBarrier();
      TROWSUM(rms_temp_0, tmp_ub_temp_4, square_fp32);
#else
    TROWSUM(rms_temp_0, square_fp32, tmp_ub_temp_4);
#endif
      VectorBarrier();
      TMULS(rms, rms, 1.0f / 1.280000e+02f);
      VectorBarrier();
      TADDS(rms, rms, 1.000000e-06f);
      VectorBarrier();
      mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> rms_temp_1;
      TASSIGN(rms_temp_1, kUbRms + 0 * 4);
      mega_gdn_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_9;
      TASSIGN(scalar_tmp_temp_9, kUbScalarTmp + 0 * 4);
      TSQRT(scalar_tmp_temp_9, rms_temp_1);
      VectorBarrier();
#if defined(PTO_NPU_ARCH_A5)
      TMULS(rms_temp_1, rms_temp_1, 0.0f);
      VectorBarrier();
      TADDS(rms_temp_1, rms_temp_1, 1.0f);
      VectorBarrier();
      TDIV(rms_temp_1, rms_temp_1, scalar_tmp_temp_9);
      VectorBarrier();
      TROWEXPANDMUL(norm_fp32, norm_fp32, rms_temp_0);
      VectorBarrier();
      TMUL(norm_fp32, norm_fp32, weight_fp32);
      // Preserve layer_norm_fwd's (value * z) / (1 + exp(-z)) order.
      TMUL(norm_fp32, norm_fp32, z_fp32);
      VectorBarrier();
      TMULS(z_fp32, z_fp32, -1.0f);
      VectorBarrier();
      TEXP(z_fp32, z_fp32);
      VectorBarrier();
      TADDS(z_fp32, z_fp32, 1.0f);
      VectorBarrier();
      TDIV(norm_fp32, norm_fp32, z_fp32);
      VectorBarrier();
      TCVT(final_half, norm_fp32, RoundMode::CAST_ROUND);
#else
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    auto scalar_tmp_scalar_temp_0 = scalar_tmp.GetValue(0);
    TMULS(norm_fp32, norm_fp32, 1.0f / scalar_tmp_scalar_temp_0);
    VectorBarrier();
    TMUL(norm_fp32, norm_fp32, weight_fp32);
    mega_gdn_decode_pto::TileUbDataND<float, 1, 128> gate_fp32_temp_0_silu_tmp;
    TASSIGN(gate_fp32_temp_0_silu_tmp, kUbVectorScratch);
    mega_gdn_decode_pto::Silu<float, 1, 128>(
        gate_fp32, z_fp32, gate_fp32_temp_0_silu_tmp);
    VectorBarrier();
    TMUL(norm_fp32, norm_fp32, gate_fp32);
    VectorBarrier();
    TCVT(final_half, norm_fp32, RoundMode::CAST_RINT);
#endif
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
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
                                      kMaxBatchSize * kMaxNumVHeads * kHeadDim,
                                      1,
                                      1,
                                      128>(
          out_handle + batch_idx * v_width + head_idx * kHeadDim,
          kUbFinalHalf,
          0,
          1,
          128);
#if defined(PTO_NPU_ARCH_A5)
      // B>1 already enters the next head through the PIPE_ALL barrier at the
      // top of this loop.  The final iteration is drained by
      // SyncAllAivStatePublish(), which also starts with PIPE_ALL.  Keep the
      // tail barrier for B=1 because a core can still own a second head when
      // NV exceeds the physical AIV count, and that path has no head-entry
      // barrier.
      if constexpr (IsBatchOne) {
        pipe_barrier(PIPE_ALL);
      }
#else
    pipe_barrier(PIPE_ALL);
    if constexpr (!IsBatchOne) {
      pipe_barrier(PIPE_ALL);
    }
#endif
    }
#if defined(PTO_NPU_ARCH_A5)
  }
#endif
  // Publish all output and persistent-state stores before the next decode
  // invocation can consume the in-place Conv/SSM caches.
  mega_gdn_decode_pto::SyncAllAivStatePublish();
#endif
}

}  // namespace mega_gdn_decode_pto

#endif  // XLLM_OPS_ARCH35_MEGA_GDN_DECODE_PTO_KERNEL_H_
