/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

/*
 * Generated AscendC kernel for layer_norm_fwd.
 *
 * This file is intentionally kept independent from the PyTorch extension
 * wrapper. Build it as a custom OPP kernel that exports
 * aclnnXllmKernelLayerNormFwd.
 */

#include "kernel_operator.h"

using namespace AscendC;

#ifndef DTYPE_X
#define DTYPE_X half
#endif

#ifndef DTYPE_WEIGHT
#define DTYPE_WEIGHT DTYPE_X
#endif

namespace layer_norm_fwd_impl {

constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kFp32BlockElems = 8;
constexpr uint32_t kFp32RepeatElems = 64;
constexpr uint32_t kQwenGroupSize = 128;
constexpr uint32_t kN128ReduceBatchRows = 32;
constexpr float kInvQwenGroupSize = 0.0078125f;
constexpr float kDefaultEps = 1e-6f;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
constexpr bool kNeedsExplicitScalarSync = true;
#else
constexpr bool kNeedsExplicitScalarSync = false;
#endif

enum KernelMode : uint32_t {
  kFullRow = 0,
  kStreamingTwoPass = 1,
};

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) {
  return ((value + align - 1) / align) * align;
}

__aicore__ inline float U32ToFloat(uint32_t value) {
  return static_cast<float>(static_cast<int32_t>(value));
}

__aicore__ inline float NormalizeEps(float eps) {
  return eps > 0.0f ? eps : kDefaultEps;
}

template <HardEvent event>
__aicore__ inline event_t ResolveEventId(event_t fixed_event_id) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  return fixed_event_id;
#else
  return static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
#endif
}

__aicore__ inline void WaitMte2ToV() {
  const event_t event_id =
      ResolveEventId<HardEvent::MTE2_V>(static_cast<event_t>(EVENT_ID0));
  SetFlag<HardEvent::MTE2_V>(event_id);
  WaitFlag<HardEvent::MTE2_V>(event_id);
}

__aicore__ inline void WaitSToMte3() {
  const event_t event_id =
      ResolveEventId<HardEvent::S_MTE3>(static_cast<event_t>(EVENT_ID1));
  SetFlag<HardEvent::S_MTE3>(event_id);
  WaitFlag<HardEvent::S_MTE3>(event_id);
}

__aicore__ inline void WaitVToS() {
  const event_t event_id =
      ResolveEventId<HardEvent::V_S>(static_cast<event_t>(EVENT_ID7));
  SetFlag<HardEvent::V_S>(event_id);
  WaitFlag<HardEvent::V_S>(event_id);
}

__aicore__ inline void WaitSToV() {
  const event_t event_id =
      ResolveEventId<HardEvent::S_V>(static_cast<event_t>(EVENT_ID6));
  SetFlag<HardEvent::S_V>(event_id);
  WaitFlag<HardEvent::S_V>(event_id);
}

__aicore__ inline void WaitVToMte3() {
  const event_t event_id =
      ResolveEventId<HardEvent::V_MTE3>(static_cast<event_t>(EVENT_ID2));
  SetFlag<HardEvent::V_MTE3>(event_id);
  WaitFlag<HardEvent::V_MTE3>(event_id);
}

__aicore__ inline void WaitVToMte2() {
  const event_t event_id =
      ResolveEventId<HardEvent::V_MTE2>(static_cast<event_t>(EVENT_ID3));
  SetFlag<HardEvent::V_MTE2>(event_id);
  WaitFlag<HardEvent::V_MTE2>(event_id);
}

__aicore__ inline void WaitMte3ToV() {
  const event_t event_id =
      ResolveEventId<HardEvent::MTE3_V>(static_cast<event_t>(EVENT_ID4));
  SetFlag<HardEvent::MTE3_V>(event_id);
  WaitFlag<HardEvent::MTE3_V>(event_id);
}

template <typename T>
__aicore__ inline void CopyInPad(LocalTensor<T> dst,
                                 GlobalTensor<T> src,
                                 uint32_t block_count,
                                 uint32_t block_len,
                                 uint32_t src_stride_bytes,
                                 uint32_t dst_stride_blocks,
                                 uint32_t right_pad) {
  DataCopyExtParams params;
  params.blockCount = static_cast<uint16_t>(block_count);
  params.blockLen = block_len;
  params.srcStride = src_stride_bytes;
  params.dstStride = static_cast<uint16_t>(dst_stride_blocks);
  DataCopyPadExtParams<T> pad{
      right_pad != 0, 0, static_cast<uint8_t>(right_pad), static_cast<T>(0)};
  DataCopyPad(dst, src, params, pad);
}

template <typename T>
__aicore__ inline void CopyOutPad(GlobalTensor<T> dst,
                                  LocalTensor<T> src,
                                  uint32_t block_count,
                                  uint32_t block_len,
                                  uint32_t src_stride_blocks,
                                  uint32_t dst_stride_bytes) {
  DataCopyExtParams params;
  params.blockCount = static_cast<uint16_t>(block_count);
  params.blockLen = block_len;
  params.srcStride = static_cast<uint16_t>(src_stride_blocks);
  params.dstStride = dst_stride_bytes;
  DataCopyPad(dst, src, params);
}

template <typename T>
__aicore__ inline void CastToFloat(LocalTensor<float> dst,
                                   LocalTensor<T> src,
                                   uint32_t count) {
  if constexpr (sizeof(T) == sizeof(float)) {
    DataCopy(dst, src.template ReinterpretCast<float>(), count);
  } else {
    Cast(dst, src, RoundMode::CAST_NONE, count);
  }
}

template <typename T>
__aicore__ inline void CastFromFloat(LocalTensor<T> dst,
                                     LocalTensor<float> src,
                                     uint32_t count) {
  if constexpr (sizeof(T) == sizeof(float)) {
    DataCopy(dst.template ReinterpretCast<float>(), src, count);
  } else {
    Cast(dst, src, RoundMode::CAST_ROUND, count);
  }
}

__aicore__ inline void Silu(LocalTensor<float> dst,
                            LocalTensor<float> src,
                            LocalTensor<float> backup,
                            uint32_t count) {
  // silu(x) = x / (1 + exp(-x))
  // Use Div instead of Reciprocal+Mul, matching the group_norm_silu pattern.
  Muls(backup, src, -1.0f, count);
  PipeBarrier<PIPE_V>();
  Exp(backup, backup, count);
  PipeBarrier<PIPE_V>();
  Adds(backup, backup, 1.0f, count);
  PipeBarrier<PIPE_V>();
  Div(dst, src, backup, count);
  PipeBarrier<PIPE_V>();
}

__aicore__ inline float ReduceSumValue(LocalTensor<float> scalar,
                                       LocalTensor<float> src,
                                       LocalTensor<float> backup,
                                       LocalTensor<float> reduce_tmp,
                                       uint32_t count) {
  Adds(backup, src, 0.0f, count);
  PipeBarrier<PIPE_V>();
  ReduceSum(scalar, backup, reduce_tmp, count);
  PipeBarrier<PIPE_V>();
  if constexpr (kNeedsExplicitScalarSync) {
    WaitVToS();
  }
  return scalar.GetValue(0);
}

__aicore__ inline float ReduceSumValueNoCopy(LocalTensor<float> scalar,
                                             LocalTensor<float> src,
                                             LocalTensor<float> reduce_tmp,
                                             uint32_t count) {
  if constexpr (kNeedsExplicitScalarSync) {
    WaitSToV();
  }
  ReduceSum(scalar, src, reduce_tmp, count);
  PipeBarrier<PIPE_V>();
  if constexpr (kNeedsExplicitScalarSync) {
    WaitVToS();
    const float value = scalar.GetValue(0);
    WaitSToV();
    return value;
  }
  return scalar.GetValue(0);
}

__aicore__ inline float SqrtValue(LocalTensor<float> scalar, float value) {
  if constexpr (kNeedsExplicitScalarSync) {
    WaitVToS();
  }
  scalar.SetValue(0, value);
  if constexpr (kNeedsExplicitScalarSync) {
    WaitSToV();
  }
  Sqrt(scalar, scalar, 1);
  if constexpr (kNeedsExplicitScalarSync) {
    WaitVToS();
  } else {
    PipeBarrier<PIPE_V>();
  }
  return scalar.GetValue(0);
}

__aicore__ inline void StoreFloatValues(GlobalTensor<float> dst,
                                        LocalTensor<float> src,
                                        uint32_t count,
                                        uint32_t use_a2_scalar_store_fix) {
  if (use_a2_scalar_store_fix != 0 && count == kFp32BlockElems) {
    WaitVToS();
    for (uint32_t i = 0; i < count; ++i) {
      dst.SetValue(i, src.GetValue(i));
    }
    DataCacheCleanAndInvalid<float,
                             CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(dst);
    return;
  }
  if (count >= kFp32BlockElems && count % kFp32BlockElems == 0) {
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = count * sizeof(float);
    params.srcStride = 0;
    params.dstStride = 0;
    WaitVToMte3();
    WaitSToMte3();
    DataCopyPad(dst, src, params);
    WaitMte3ToV();
    return;
  }

  WaitVToS();
  for (uint32_t i = 0; i < count; ++i) {
    dst.SetValue(i, src.GetValue(i));
  }
}

template <typename T>
__aicore__ inline float RstdFromVariance(LocalTensor<float> scalar,
                                         float variance,
                                         float eps) {
  const float denom = SqrtValue(scalar, variance + eps);
  return 1.0f / denom;
}

template <typename T>
__aicore__ inline float RstdFromSquareSum(LocalTensor<float> scalar,
                                          LocalTensor<float> row_x,
                                          LocalTensor<float> square_buf,
                                          LocalTensor<float> reduce_tmp,
                                          uint32_t count,
                                          float eps) {
  // This is the dominant scalar chain in profiler:
  // square -> ReduceSum -> rstd scalar -> GetValue for the scalar Muls below.
  Mul(square_buf, row_x, row_x, count);
  PipeBarrier<PIPE_V>();
  const float square_sum =
      ReduceSumValue(scalar, square_buf, square_buf, reduce_tmp, count);
  const float variance = square_sum / U32ToFloat(count);
  return RstdFromVariance<T>(scalar, variance, eps);
}

__aicore__ inline void ReduceSumRowsN128(LocalTensor<float> dst,
                                         LocalTensor<float> src,
                                         LocalTensor<float> work,
                                         uint32_t rows) {
  BinaryRepeatParams repeat_params;
  repeat_params.dstBlkStride = 1;
  repeat_params.src0BlkStride = 1;
  repeat_params.src1BlkStride = 1;
  repeat_params.dstRepStride = kFp32RepeatElems / kFp32BlockElems;
  repeat_params.src0RepStride = kQwenGroupSize / kFp32BlockElems;
  repeat_params.src1RepStride = kFp32RepeatElems / kFp32BlockElems;

  Duplicate(work, 0.0f, rows * kFp32RepeatElems);
  PipeBarrier<PIPE_V>();
  Add(work, src, work, kFp32RepeatElems, rows, repeat_params);
  PipeBarrier<PIPE_V>();
  Add(work, src[kFp32RepeatElems], work, kFp32RepeatElems, rows, repeat_params);
  PipeBarrier<PIPE_V>();

  AscendCUtils::SetMask<float>(kFp32RepeatElems);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
  if (g_coreType == AIV) {
    WholeReduceSum<float, false>(dst,
                                 work,
                                 MASK_PLACEHOLDER,
                                 rows,
                                 1,
                                 1,
                                 kFp32RepeatElems / kFp32BlockElems);
  }
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3510)
  WholeReduceSum(dst,
                 work,
                 kFp32RepeatElems,
                 rows,
                 1,
                 1,
                 kFp32RepeatElems / kFp32BlockElems);
#else
  WholeReduceSum<float, false>(dst,
                               work,
                               MASK_PLACEHOLDER,
                               rows,
                               1,
                               1,
                               kFp32RepeatElems / kFp32BlockElems);
#endif
  PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void RstdRowsN128(LocalTensor<float> rstd,
                                    LocalTensor<float> square_buf,
                                    LocalTensor<float> reduce_work,
                                    uint32_t rows,
                                    float eps) {
  uint32_t row_offset = 0;
  while (row_offset < rows) {
    const uint32_t batch_rows = MinU32(kN128ReduceBatchRows, rows - row_offset);
    ReduceSumRowsN128(rstd[row_offset],
                      square_buf[row_offset * kQwenGroupSize],
                      reduce_work,
                      batch_rows);
    row_offset += batch_rows;
  }

  const uint32_t aligned_rows = AlignUp(rows, kFp32BlockElems);
  Muls(rstd, rstd, kInvQwenGroupSize, aligned_rows);
  PipeBarrier<PIPE_V>();
  Adds(rstd, rstd, eps, aligned_rows);
  PipeBarrier<PIPE_V>();
  Sqrt(rstd, rstd, aligned_rows);
  PipeBarrier<PIPE_V>();
  Duplicate(reduce_work, 1.0f, aligned_rows);
  PipeBarrier<PIPE_V>();
  Div(rstd, reduce_work, rstd, aligned_rows);
  PipeBarrier<PIPE_V>();
}

__aicore__ inline void MulRowsN128ByScalars(LocalTensor<float> dst,
                                            LocalTensor<float> src,
                                            LocalTensor<float> row_scalars,
                                            LocalTensor<float> broadcast,
                                            uint32_t rows) {
  const uint32_t brcb_repeats = CeilDiv(rows, kFp32BlockElems);
  Brcb(broadcast, row_scalars, static_cast<uint8_t>(brcb_repeats), {1, 8});
  PipeBarrier<PIPE_V>();

  BinaryRepeatParams repeat_params;
  repeat_params.dstBlkStride = 1;
  repeat_params.src0BlkStride = 1;
  repeat_params.src1BlkStride = 0;
  repeat_params.dstRepStride = kQwenGroupSize / kFp32BlockElems;
  repeat_params.src0RepStride = kQwenGroupSize / kFp32BlockElems;
  repeat_params.src1RepStride = 1;
  Mul(dst, src, broadcast, kFp32RepeatElems, rows, repeat_params);
  Mul(dst[kFp32RepeatElems],
      src[kFp32RepeatElems],
      broadcast,
      kFp32RepeatElems,
      rows,
      repeat_params);
  PipeBarrier<PIPE_V>();
}

__aicore__ inline void ApplySiluGate(LocalTensor<float> value,
                                     LocalTensor<float> gate,
                                     uint32_t count) {
  // value * silu(gate) = (value * gate) / (1 + exp(-gate)).
  // This form keeps the whole tile vectorized without another tile-sized UB.
  Mul(value, value, gate, count);
  PipeBarrier<PIPE_V>();
  Muls(gate, gate, -1.0f, count);
  PipeBarrier<PIPE_V>();
  Exp(gate, gate, count);
  PipeBarrier<PIPE_V>();
  Adds(gate, gate, 1.0f, count);
  PipeBarrier<PIPE_V>();
  Div(value, value, gate, count);
  PipeBarrier<PIPE_V>();
}

__aicore__ inline void MulRowsN128ByWeight(LocalTensor<float> dst,
                                           LocalTensor<float> src,
                                           LocalTensor<float> weight,
                                           uint32_t rows) {
  BinaryRepeatParams repeat_params;
  repeat_params.dstBlkStride = 1;
  repeat_params.src0BlkStride = 1;
  repeat_params.src1BlkStride = 1;
  repeat_params.dstRepStride = kQwenGroupSize / kFp32BlockElems;
  repeat_params.src0RepStride = kQwenGroupSize / kFp32BlockElems;
  repeat_params.src1RepStride = 0;
  Mul(dst, src, weight, kFp32RepeatElems, rows, repeat_params);
  Mul(dst[kFp32RepeatElems],
      src[kFp32RepeatElems],
      weight[kFp32RepeatElems],
      kFp32RepeatElems,
      rows,
      repeat_params);
  PipeBarrier<PIPE_V>();
}

template <typename T, typename TW>
class KernelFullRow {
 public:
  __aicore__ inline KernelFullRow() {}

  __aicore__ inline void Init(GM_ADDR x,
                              GM_ADDR weight,
                              GM_ADDR bias,
                              GM_ADDR z,
                              GM_ADDR y,
                              GM_ADDR mean,
                              GM_ADDR rstd,
                              uint32_t m,
                              uint32_t full_n,
                              uint32_t group_size,
                              uint32_t ngroups,
                              uint32_t stride_x,
                              uint32_t stride_y,
                              uint32_t stride_z,
                              uint32_t group_align,
                              uint32_t tile_rows,
                              float eps,
                              uint32_t has_bias,
                              uint32_t has_z,
                              uint32_t norm_before_gate,
                              uint32_t is_rms_norm,
                              uint32_t use_a2_scalar_store_fix) {
    m_ = m;
    fullN_ = full_n;
    groupSize_ = group_size;
    ngroups_ = ngroups;
    strideX_ = stride_x;
    strideY_ = stride_y;
    strideZ_ = stride_z;
    groupAlign_ = group_align;
    tileRows_ = tile_rows == 0 ? 1 : tile_rows;
    eps_ = NormalizeEps(eps);
    hasBias_ = has_bias;
    hasZ_ = has_z;
    normBeforeGate_ = norm_before_gate;
    isRmsNorm_ = is_rms_norm;
    useA2ScalarStoreFix_ = use_a2_scalar_store_fix;
    blockDim_ = GetBlockNum();
    coreId_ = GetBlockIdx();

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    wGm_.SetGlobalBuffer((__gm__ TW*)weight);
    if (hasBias_ != 0) {
      bGm_.SetGlobalBuffer((__gm__ TW*)bias);
    }
    if (hasZ_ != 0) {
      zGm_.SetGlobalBuffer((__gm__ T*)z);
    }
    if (isRmsNorm_ == 0) {
      meanGm_.SetGlobalBuffer((__gm__ float*)mean);
    }
    rstdGm_.SetGlobalBuffer((__gm__ float*)rstd);

    const uint32_t tile_elems = tileRows_ * groupAlign_;
    pipe_.InitBuffer(xQueue_, kBufferNum, tile_elems * sizeof(T));
    pipe_.InitBuffer(yQueue_, kBufferNum, tile_elems * sizeof(T));
    if (hasZ_ != 0) {
      pipe_.InitBuffer(zQueue_, kBufferNum, tile_elems * sizeof(T));
    }
    pipe_.InitBuffer(xFp32Buf_, tile_elems * sizeof(float));
    pipe_.InitBuffer(tmpFp32Buf_, tile_elems * sizeof(float));
    // Rows are processed serially, so they can share one reduction scratch.
    pipe_.InitBuffer(reduceTmpBuf_,
                     AlignUp(groupAlign_, kFp32BlockElems) * sizeof(float));
    if (IsRmsGateN128() || IsRmsN128NoZ()) {
      pipe_.InitBuffer(rowReduceTmpBuf_,
                       kN128ReduceBatchRows * kFp32RepeatElems * sizeof(float));
    }
    pipe_.InitBuffer(scalarBuf_, kFp32BlockElems * sizeof(float));
    pipe_.InitBuffer(weightRawBuf_, groupAlign_ * sizeof(TW));
    pipe_.InitBuffer(weightFp32Buf_, groupAlign_ * sizeof(float));
    if (hasBias_ != 0) {
      pipe_.InitBuffer(biasRawBuf_, groupAlign_ * sizeof(TW));
      pipe_.InitBuffer(biasFp32Buf_, groupAlign_ * sizeof(float));
    }
    pipe_.InitBuffer(statBuf_,
                     AlignUp(tileRows_, kFp32BlockElems) * 2 * sizeof(float));
  }

  __aicore__ inline void Process() {
    if (IsRmsGateN128()) {
      ProcessRmsGateN128();
      return;
    }
    if (IsRmsN128NoZ()) {
      ProcessRmsN128NoZ();
      return;
    }
    if (tileRows_ == 1) {
      ProcessLinearRows();
    } else {
      ProcessRowBlocks();
    }
  }

 private:
  __aicore__ inline bool IsRmsGateN128() const {
    // Qwen3.5 hot path: one 128-wide RMSNorm group followed by SiLU gate.
    // It avoids generic padding/stride handling and keeps weight cached per
    // core.
    return groupSize_ == kQwenGroupSize && groupAlign_ == kQwenGroupSize &&
           fullN_ == kQwenGroupSize && ngroups_ == 1 &&
           strideX_ == kQwenGroupSize && strideY_ == kQwenGroupSize &&
           strideZ_ == kQwenGroupSize && hasBias_ == 0 && hasZ_ != 0 &&
           normBeforeGate_ != 0 && isRmsNorm_ != 0;
  }

  __aicore__ inline bool IsRmsN128NoZ() const {
    return groupSize_ == kQwenGroupSize && groupAlign_ == kQwenGroupSize &&
           fullN_ == kQwenGroupSize && ngroups_ == 1 &&
           strideX_ == kQwenGroupSize && strideY_ == kQwenGroupSize &&
           hasBias_ == 0 && hasZ_ == 0 && isRmsNorm_ != 0;
  }

  __aicore__ inline void ProcessRmsGateN128() {
    LoadWeightN128();
    const uint32_t work_rows = StatSafeWorkRows();
    const uint32_t tile_count = CeilDiv(m_, work_rows);
    for (uint32_t tile = coreId_; tile < tile_count; tile += blockDim_) {
      const uint32_t row_start = tile * work_rows;
      const uint32_t rows = MinU32(work_rows, m_ - row_start);
      ProcessTilesRmsGateN128(row_start, rows);
    }
  }

  __aicore__ inline void ProcessRmsN128NoZ() {
    LoadWeightN128();
    const uint32_t work_rows = StatSafeWorkRows();
    const uint32_t tile_count = CeilDiv(m_, work_rows);
    for (uint32_t tile = coreId_; tile < tile_count; tile += blockDim_) {
      const uint32_t row_start = tile * work_rows;
      const uint32_t rows = MinU32(work_rows, m_ - row_start);
      ProcessTilesRmsN128NoZ(row_start, rows);
    }
  }

  __aicore__ inline void LoadWeightN128() {
    LocalTensor<TW> w_raw = weightRawBuf_.Get<TW>();
    LocalTensor<float> w_fp32 = weightFp32Buf_.Get<float>();
    DataCopy(w_raw, wGm_, kQwenGroupSize);
    WaitMte2ToV();
    CastToFloat<TW>(w_fp32, w_raw, kQwenGroupSize);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void PrefetchTileRmsGateN128(uint32_t row_start,
                                                 uint32_t rows) {
    const uint32_t elem_count = rows * kQwenGroupSize;
    LocalTensor<T> x_local = xQueue_.AllocTensor<T>();
    DataCopy(x_local, xGm_[row_start * kQwenGroupSize], elem_count);
    xQueue_.EnQue(x_local);

    LocalTensor<T> z_local = zQueue_.AllocTensor<T>();
    DataCopy(z_local, zGm_[row_start * kQwenGroupSize], elem_count);
    zQueue_.EnQue(z_local);
  }

  __aicore__ inline void WaitTileRmsGateN128(LocalTensor<T>& x_local,
                                             LocalTensor<T>& z_local) {
    x_local = xQueue_.DeQue<T>();
    WaitMte2ToV();
    z_local = zQueue_.DeQue<T>();
    WaitMte2ToV();
  }

  __aicore__ inline void ComputeTileRmsGateN128(uint32_t row_start,
                                                uint32_t rows,
                                                LocalTensor<T> x_local,
                                                LocalTensor<T> z_local) {
    const uint32_t elem_count = rows * kQwenGroupSize;
    LocalTensor<T> y_local = yQueue_.AllocTensor<T>();
    LocalTensor<float> x_fp32 = xFp32Buf_.Get<float>();
    LocalTensor<float> tmp_fp32 = tmpFp32Buf_.Get<float>();
    LocalTensor<float> row_reduce_tmp = rowReduceTmpBuf_.Get<float>();
    LocalTensor<float> rstd_stat = statBuf_.Get<float>();

    CastToFloat<T>(x_fp32, x_local, elem_count);
    PipeBarrier<PIPE_V>();
    Mul(tmp_fp32, x_fp32, x_fp32, elem_count);
    PipeBarrier<PIPE_V>();
    RstdRowsN128<T>(rstd_stat, tmp_fp32, row_reduce_tmp, rows, eps_);

    MulRowsN128ByScalars(x_fp32, x_fp32, rstd_stat, row_reduce_tmp, rows);
    MulRowsN128ByWeight(x_fp32, x_fp32, weightFp32Buf_.Get<float>(), rows);

    CastToFloat<T>(tmp_fp32, z_local, elem_count);
    PipeBarrier<PIPE_V>();
    ApplySiluGate(x_fp32, tmp_fp32, elem_count);

    StoreFloatValues(rstdGm_[row_start], rstd_stat, rows, useA2ScalarStoreFix_);

    CastFromFloat<T>(y_local, x_fp32, elem_count);
    PipeBarrier<PIPE_V>();
    yQueue_.EnQue(y_local);
    xQueue_.FreeTensor(x_local);
    zQueue_.FreeTensor(z_local);

    y_local = yQueue_.DeQue<T>();
    WaitVToMte3();
    DataCopy(yGm_[row_start * kQwenGroupSize], y_local, elem_count);
    yQueue_.FreeTensor(y_local);
  }

  __aicore__ inline void ProcessTileRmsGateN128(uint32_t row_start,
                                                uint32_t rows) {
    LocalTensor<T> x_local;
    LocalTensor<T> z_local;
    PrefetchTileRmsGateN128(row_start, rows);
    WaitTileRmsGateN128(x_local, z_local);
    ComputeTileRmsGateN128(row_start, rows, x_local, z_local);
  }

  __aicore__ inline void ProcessTilesRmsGateN128(uint32_t row_start,
                                                 uint32_t rows_left) {
    uint32_t current_row = row_start;
    uint32_t current_rows = MinU32(tileRows_, rows_left);
    PrefetchTileRmsGateN128(current_row, current_rows);
    rows_left -= current_rows;
    uint32_t next_row = current_row + current_rows;

    while (true) {
      LocalTensor<T> x_local;
      LocalTensor<T> z_local;
      WaitTileRmsGateN128(x_local, z_local);

      const bool has_next = rows_left > 0;
      uint32_t prefetched_row = next_row;
      uint32_t prefetched_rows = 0;
      if (has_next) {
        prefetched_rows = MinU32(tileRows_, rows_left);
        PrefetchTileRmsGateN128(prefetched_row, prefetched_rows);
      }

      ComputeTileRmsGateN128(current_row, current_rows, x_local, z_local);
      if (!has_next) {
        break;
      }

      current_row = prefetched_row;
      current_rows = prefetched_rows;
      next_row = prefetched_row + prefetched_rows;
      rows_left -= prefetched_rows;
    }
  }

  __aicore__ inline void PrefetchTileRmsN128NoZ(uint32_t row_start,
                                                uint32_t rows) {
    const uint32_t elem_count = rows * kQwenGroupSize;
    LocalTensor<T> x_local = xQueue_.AllocTensor<T>();
    DataCopy(x_local, xGm_[row_start * kQwenGroupSize], elem_count);
    xQueue_.EnQue(x_local);
  }

  __aicore__ inline void WaitTileRmsN128NoZ(LocalTensor<T>& x_local) {
    x_local = xQueue_.DeQue<T>();
    WaitMte2ToV();
  }

  __aicore__ inline void ComputeTileRmsN128NoZ(uint32_t row_start,
                                               uint32_t rows,
                                               LocalTensor<T> x_local) {
    const uint32_t elem_count = rows * kQwenGroupSize;
    LocalTensor<T> y_local = yQueue_.AllocTensor<T>();
    LocalTensor<float> x_fp32 = xFp32Buf_.Get<float>();
    LocalTensor<float> tmp_fp32 = tmpFp32Buf_.Get<float>();
    LocalTensor<float> row_reduce_tmp = rowReduceTmpBuf_.Get<float>();
    LocalTensor<float> rstd_stat = statBuf_.Get<float>();

    CastToFloat<T>(x_fp32, x_local, elem_count);
    PipeBarrier<PIPE_V>();
    Mul(tmp_fp32, x_fp32, x_fp32, elem_count);
    PipeBarrier<PIPE_V>();
    RstdRowsN128<T>(rstd_stat, tmp_fp32, row_reduce_tmp, rows, eps_);
    MulRowsN128ByScalars(x_fp32, x_fp32, rstd_stat, row_reduce_tmp, rows);
    MulRowsN128ByWeight(x_fp32, x_fp32, weightFp32Buf_.Get<float>(), rows);

    StoreFloatValues(rstdGm_[row_start], rstd_stat, rows, useA2ScalarStoreFix_);

    CastFromFloat<T>(y_local, x_fp32, elem_count);
    PipeBarrier<PIPE_V>();
    yQueue_.EnQue(y_local);
    xQueue_.FreeTensor(x_local);

    y_local = yQueue_.DeQue<T>();
    WaitVToMte3();
    DataCopy(yGm_[row_start * kQwenGroupSize], y_local, elem_count);
    yQueue_.FreeTensor(y_local);
  }

  __aicore__ inline void ProcessTileRmsN128NoZ(uint32_t row_start,
                                               uint32_t rows) {
    LocalTensor<T> x_local;
    PrefetchTileRmsN128NoZ(row_start, rows);
    WaitTileRmsN128NoZ(x_local);
    ComputeTileRmsN128NoZ(row_start, rows, x_local);
  }

  __aicore__ inline void ProcessTilesRmsN128NoZ(uint32_t row_start,
                                                uint32_t rows_left) {
    uint32_t cur = row_start;
    while (rows_left > 0) {
      const uint32_t cur_rows = MinU32(tileRows_, rows_left);
      ProcessTileRmsN128NoZ(cur, cur_rows);
      cur += cur_rows;
      rows_left -= cur_rows;
    }
  }

  __aicore__ inline void ProcessLinearRows() { ProcessGroupsByStatBlocks(); }

  __aicore__ inline void ProcessRowBlocks() { ProcessGroupsByStatBlocks(); }

  __aicore__ inline uint32_t StatSafeWorkRows() const {
    if (tileRows_ >= kFp32BlockElems) {
      return (tileRows_ / kFp32BlockElems) * kFp32BlockElems;
    }
    return kFp32BlockElems;
  }

  __aicore__ inline void ProcessGroupsByStatBlocks() {
    const uint32_t work_rows = StatSafeWorkRows();
    const uint32_t blocks_per_group = CeilDiv(m_, work_rows);
    for (uint32_t group = 0; group < ngroups_; ++group) {
      LoadParams(group);
      for (uint32_t block = coreId_; block < blocks_per_group;
           block += blockDim_) {
        const uint32_t row_start = block * work_rows;
        const uint32_t rows = MinU32(work_rows, m_ - row_start);
        ProcessTiles(group, row_start, rows);
      }
    }
  }

  __aicore__ inline void LoadParams(uint32_t group) {
    LocalTensor<TW> w_raw = weightRawBuf_.Get<TW>();
    LocalTensor<float> w_fp32 = weightFp32Buf_.Get<float>();
    const uint32_t right_pad = groupAlign_ - groupSize_;
    CopyInPad<TW>(w_raw,
                  wGm_[group * groupSize_],
                  1,
                  groupSize_ * sizeof(TW),
                  0,
                  0,
                  right_pad);
    WaitMte2ToV();
    CastToFloat<TW>(w_fp32, w_raw, groupAlign_);
    PipeBarrier<PIPE_V>();

    if (hasBias_ != 0) {
      LocalTensor<TW> b_raw = biasRawBuf_.Get<TW>();
      LocalTensor<float> b_fp32 = biasFp32Buf_.Get<float>();
      CopyInPad<TW>(b_raw,
                    bGm_[group * groupSize_],
                    1,
                    groupSize_ * sizeof(TW),
                    0,
                    0,
                    right_pad);
      WaitMte2ToV();
      CastToFloat<TW>(b_fp32, b_raw, groupAlign_);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void PrefetchTile(uint32_t group,
                                      uint32_t row_start,
                                      uint32_t rows) {
    LocalTensor<T> x_local = xQueue_.AllocTensor<T>();
    const uint32_t right_pad = groupAlign_ - groupSize_;
    const uint32_t dst_gap = right_pad * sizeof(T) / 32;
    const uint32_t x_src_gap = (strideX_ - groupSize_) * sizeof(T);
    CopyInPad<T>(x_local,
                 xGm_[row_start * strideX_ + group * groupSize_],
                 rows,
                 groupSize_ * sizeof(T),
                 x_src_gap,
                 dst_gap,
                 right_pad);
    xQueue_.EnQue(x_local);

    LocalTensor<T> z_local;
    if (hasZ_ != 0) {
      z_local = zQueue_.AllocTensor<T>();
      const uint32_t z_src_gap = (strideZ_ - groupSize_) * sizeof(T);
      CopyInPad<T>(z_local,
                   zGm_[row_start * strideZ_ + group * groupSize_],
                   rows,
                   groupSize_ * sizeof(T),
                   z_src_gap,
                   dst_gap,
                   right_pad);
      zQueue_.EnQue(z_local);
    }
  }

  __aicore__ inline void WaitTile(LocalTensor<T>& x_local,
                                  LocalTensor<T>& z_local) {
    x_local = xQueue_.DeQue<T>();
    WaitMte2ToV();
    if (hasZ_ != 0) {
      z_local = zQueue_.DeQue<T>();
      WaitMte2ToV();
    }
  }

  __aicore__ inline void ComputeTile(uint32_t group,
                                     uint32_t row_start,
                                     uint32_t rows,
                                     LocalTensor<T> x_local,
                                     LocalTensor<T> z_local,
                                     uint32_t stat_offset = 0,
                                     uint32_t store_stats = 1) {
    const uint32_t right_pad = groupAlign_ - groupSize_;
    LocalTensor<T> y_local = yQueue_.AllocTensor<T>();
    LocalTensor<float> x_fp32 = xFp32Buf_.Get<float>();
    LocalTensor<float> tmp_fp32 = tmpFp32Buf_.Get<float>();
    LocalTensor<float> reduce_tmp = reduceTmpBuf_.Get<float>();
    LocalTensor<float> scalar = scalarBuf_.Get<float>();
    LocalTensor<float> stats = statBuf_.Get<float>();
    LocalTensor<float> mean_stat = stats;
    LocalTensor<float> rstd_stat = stats[AlignUp(tileRows_, kFp32BlockElems)];

    CastToFloat<T>(x_fp32, x_local, rows * groupAlign_);
    PipeBarrier<PIPE_V>();

    for (uint32_t r = 0; r < rows; ++r) {
      LocalTensor<float> row_x = x_fp32[r * groupAlign_];
      LocalTensor<float> row_tmp = tmp_fp32[r * groupAlign_];
      LocalTensor<float> row_reduce = reduce_tmp;

      if (hasZ_ != 0 && normBeforeGate_ == 0) {
        CastToFloat<T>(row_tmp, z_local[r * groupAlign_], groupAlign_);
        PipeBarrier<PIPE_V>();
        Silu(row_tmp, row_tmp, row_reduce, groupSize_);
        Mul(row_x, row_x, row_tmp, groupSize_);
        PipeBarrier<PIPE_V>();
      }

      float mean_val = 0.0f;
      if (isRmsNorm_ == 0) {
        const float sum_val =
            ReduceSumValue(scalar, row_x, row_tmp, row_reduce, groupSize_);
        mean_val = sum_val / U32ToFloat(groupSize_);
        mean_stat.SetValue(stat_offset + r, mean_val);
        Adds(row_x, row_x, -mean_val, groupSize_);
        PipeBarrier<PIPE_V>();
      }

      const float rstd_val = RstdFromSquareSum<T>(
          scalar, row_x, row_tmp, row_reduce, groupSize_, eps_);
      rstd_stat.SetValue(stat_offset + r, rstd_val);

      Muls(row_x, row_x, rstd_val, groupSize_);
      PipeBarrier<PIPE_V>();
      Mul(row_x, row_x, weightFp32Buf_.Get<float>(), groupSize_);
      PipeBarrier<PIPE_V>();
      if (hasBias_ != 0) {
        Add(row_x, row_x, biasFp32Buf_.Get<float>(), groupSize_);
        PipeBarrier<PIPE_V>();
      }

      if (hasZ_ != 0 && normBeforeGate_ != 0) {
        CastToFloat<T>(row_tmp, z_local[r * groupAlign_], groupAlign_);
        PipeBarrier<PIPE_V>();
        Silu(row_tmp, row_tmp, row_reduce, groupSize_);
        Mul(row_x, row_x, row_tmp, groupSize_);
        PipeBarrier<PIPE_V>();
      }
    }

    if (store_stats != 0 && isRmsNorm_ == 0) {
      StoreFloatValues(meanGm_[group * m_ + row_start],
                       mean_stat[stat_offset],
                       rows,
                       useA2ScalarStoreFix_);
    }
    if (store_stats != 0) {
      StoreFloatValues(rstdGm_[group * m_ + row_start],
                       rstd_stat[stat_offset],
                       rows,
                       useA2ScalarStoreFix_);
    }

    CastFromFloat<T>(y_local, x_fp32, rows * groupAlign_);
    PipeBarrier<PIPE_V>();
    yQueue_.EnQue(y_local);
    xQueue_.FreeTensor(x_local);
    if (hasZ_ != 0) {
      zQueue_.FreeTensor(z_local);
    }

    y_local = yQueue_.DeQue<T>();
    WaitVToMte3();
    const uint32_t y_src_gap = right_pad * sizeof(T) / 32;
    const uint32_t y_dst_gap = (strideY_ - groupSize_) * sizeof(T);
    CopyOutPad<T>(yGm_[row_start * strideY_ + group * groupSize_],
                  y_local,
                  rows,
                  groupSize_ * sizeof(T),
                  y_src_gap,
                  y_dst_gap);
    yQueue_.FreeTensor(y_local);
  }

  __aicore__ inline void ProcessTile(uint32_t group,
                                     uint32_t row_start,
                                     uint32_t rows,
                                     bool params_loaded = false) {
    if (!params_loaded) {
      LoadParams(group);
    }

    LocalTensor<T> x_local;
    LocalTensor<T> z_local;
    PrefetchTile(group, row_start, rows);
    WaitTile(x_local, z_local);
    ComputeTile(group, row_start, rows, x_local, z_local);
  }

  __aicore__ inline void ProcessTiles(uint32_t group,
                                      uint32_t row_start,
                                      uint32_t rows_left) {
    if (tileRows_ < kFp32BlockElems && rows_left <= kFp32BlockElems) {
      ProcessTilesStatBlock(group, row_start, rows_left);
      return;
    }

    uint32_t current_row = row_start;
    uint32_t current_rows = MinU32(tileRows_, rows_left);
    PrefetchTile(group, current_row, current_rows);
    rows_left -= current_rows;
    uint32_t next_row = current_row + current_rows;

    while (true) {
      LocalTensor<T> x_local;
      LocalTensor<T> z_local;
      WaitTile(x_local, z_local);

      const bool has_next = rows_left > 0;
      uint32_t prefetched_row = next_row;
      uint32_t prefetched_rows = 0;
      if (has_next) {
        prefetched_rows = MinU32(tileRows_, rows_left);
        PrefetchTile(group, prefetched_row, prefetched_rows);
      }

      ComputeTile(group, current_row, current_rows, x_local, z_local);
      if (!has_next) {
        break;
      }

      current_row = prefetched_row;
      current_rows = prefetched_rows;
      next_row = prefetched_row + prefetched_rows;
      rows_left -= prefetched_rows;
    }
  }

  __aicore__ inline void ProcessTilesStatBlock(uint32_t group,
                                               uint32_t row_start,
                                               uint32_t rows_left) {
    LocalTensor<float> stats = statBuf_.Get<float>();
    const uint32_t stat_stride = AlignUp(tileRows_, kFp32BlockElems);
    if (isRmsNorm_ == 0) {
      Duplicate(stats, 0.0f, stat_stride * 2);
    } else {
      Duplicate(stats[stat_stride], 0.0f, stat_stride);
    }
    PipeBarrier<PIPE_V>();

    const uint32_t total_rows = rows_left;
    uint32_t current_row = row_start;
    uint32_t current_rows = MinU32(tileRows_, rows_left);
    uint32_t stat_offset = 0;
    PrefetchTile(group, current_row, current_rows);
    rows_left -= current_rows;
    uint32_t next_row = current_row + current_rows;

    while (true) {
      LocalTensor<T> x_local;
      LocalTensor<T> z_local;
      WaitTile(x_local, z_local);

      const bool has_next = rows_left > 0;
      uint32_t prefetched_row = next_row;
      uint32_t prefetched_rows = 0;
      if (has_next) {
        prefetched_rows = MinU32(tileRows_, rows_left);
        PrefetchTile(group, prefetched_row, prefetched_rows);
      }

      ComputeTile(
          group, current_row, current_rows, x_local, z_local, stat_offset, 0);
      stat_offset += current_rows;
      if (!has_next) {
        break;
      }

      current_row = prefetched_row;
      current_rows = prefetched_rows;
      next_row = prefetched_row + prefetched_rows;
      rows_left -= prefetched_rows;
    }

    if (isRmsNorm_ == 0) {
      StoreFloatValues(meanGm_[group * m_ + row_start],
                       stats,
                       total_rows,
                       useA2ScalarStoreFix_);
    }
    StoreFloatValues(rstdGm_[group * m_ + row_start],
                     stats[stat_stride],
                     total_rows,
                     useA2ScalarStoreFix_);
  }

 private:
  TPipe pipe_;
  TQue<QuePosition::VECIN, kBufferNum> xQueue_;
  TQue<QuePosition::VECIN, kBufferNum> zQueue_;
  TQue<QuePosition::VECOUT, kBufferNum> yQueue_;
  TBuf<TPosition::VECCALC> xFp32Buf_;
  TBuf<TPosition::VECCALC> tmpFp32Buf_;
  TBuf<TPosition::VECCALC> reduceTmpBuf_;
  TBuf<TPosition::VECCALC> rowReduceTmpBuf_;
  TBuf<TPosition::VECCALC> scalarBuf_;
  TBuf<TPosition::VECCALC> statBuf_;
  TBuf<TPosition::VECCALC> weightRawBuf_;
  TBuf<TPosition::VECCALC> weightFp32Buf_;
  TBuf<TPosition::VECCALC> biasRawBuf_;
  TBuf<TPosition::VECCALC> biasFp32Buf_;

  GlobalTensor<T> xGm_;
  GlobalTensor<T> yGm_;
  GlobalTensor<T> zGm_;
  GlobalTensor<TW> wGm_;
  GlobalTensor<TW> bGm_;
  GlobalTensor<float> meanGm_;
  GlobalTensor<float> rstdGm_;

  uint32_t m_ = 0;
  uint32_t fullN_ = 0;
  uint32_t groupSize_ = 0;
  uint32_t ngroups_ = 0;
  uint32_t strideX_ = 0;
  uint32_t strideY_ = 0;
  uint32_t strideZ_ = 0;
  uint32_t groupAlign_ = 0;
  uint32_t tileRows_ = 1;
  uint32_t blockDim_ = 1;
  uint32_t coreId_ = 0;
  uint32_t hasBias_ = 0;
  uint32_t hasZ_ = 0;
  uint32_t normBeforeGate_ = 1;
  uint32_t isRmsNorm_ = 0;
  uint32_t useA2ScalarStoreFix_ = 0;
  float eps_ = 1e-6f;
};

template <typename T, typename TW>
class KernelStreamingTwoPass {
 public:
  __aicore__ inline KernelStreamingTwoPass() {}

  __aicore__ inline void Init(GM_ADDR x,
                              GM_ADDR weight,
                              GM_ADDR bias,
                              GM_ADDR z,
                              GM_ADDR y,
                              GM_ADDR mean,
                              GM_ADDR rstd,
                              uint32_t m,
                              uint32_t full_n,
                              uint32_t group_size,
                              uint32_t ngroups,
                              uint32_t stride_x,
                              uint32_t stride_y,
                              uint32_t stride_z,
                              uint32_t chunk_size,
                              uint32_t chunk_align,
                              float eps,
                              uint32_t has_bias,
                              uint32_t has_z,
                              uint32_t norm_before_gate,
                              uint32_t is_rms_norm,
                              uint32_t use_a2_scalar_store_fix) {
    m_ = m;
    fullN_ = full_n;
    groupSize_ = group_size;
    ngroups_ = ngroups;
    strideX_ = stride_x;
    strideY_ = stride_y;
    strideZ_ = stride_z;
    chunkSize_ = chunk_size;
    chunkAlign_ = chunk_align;
    eps_ = NormalizeEps(eps);
    hasBias_ = has_bias;
    hasZ_ = has_z;
    normBeforeGate_ = norm_before_gate;
    isRmsNorm_ = is_rms_norm;
    useA2ScalarStoreFix_ = use_a2_scalar_store_fix;
    blockDim_ = GetBlockNum();
    coreId_ = GetBlockIdx();

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    wGm_.SetGlobalBuffer((__gm__ TW*)weight);
    if (hasBias_ != 0) {
      bGm_.SetGlobalBuffer((__gm__ TW*)bias);
    }
    if (hasZ_ != 0) {
      zGm_.SetGlobalBuffer((__gm__ T*)z);
    }
    if (isRmsNorm_ == 0) {
      meanGm_.SetGlobalBuffer((__gm__ float*)mean);
    }
    rstdGm_.SetGlobalBuffer((__gm__ float*)rstd);

    pipe_.InitBuffer(xQueue_, kBufferNum, chunkAlign_ * sizeof(T));
    pipe_.InitBuffer(yQueue_, kBufferNum, chunkAlign_ * sizeof(T));
    if (hasZ_ != 0) {
      pipe_.InitBuffer(zQueue_, kBufferNum, chunkAlign_ * sizeof(T));
    }
    pipe_.InitBuffer(xFp32Buf_, chunkAlign_ * sizeof(float));
    pipe_.InitBuffer(tmpFp32Buf_, chunkAlign_ * sizeof(float));
    pipe_.InitBuffer(reduceTmpBuf_,
                     AlignUp(chunkAlign_, kFp32BlockElems) * sizeof(float));
    pipe_.InitBuffer(scalarBuf_, kFp32BlockElems * sizeof(float));
    pipe_.InitBuffer(statBuf_, kFp32BlockElems * 2 * sizeof(float));
    pipe_.InitBuffer(weightQueue_, kBufferNum, chunkAlign_ * sizeof(TW));
    pipe_.InitBuffer(weightFp32Buf_, chunkAlign_ * sizeof(float));
    if (hasBias_ != 0) {
      pipe_.InitBuffer(biasQueue_, kBufferNum, chunkAlign_ * sizeof(TW));
      pipe_.InitBuffer(biasFp32Buf_, chunkAlign_ * sizeof(float));
    }
  }

  __aicore__ inline void Process() {
    const uint32_t rows_per_stat_block = kFp32BlockElems;
    const uint32_t blocks_per_group = CeilDiv(m_, rows_per_stat_block);
    const uint32_t total_blocks = blocks_per_group * ngroups_;
    for (uint32_t linear = coreId_; linear < total_blocks;
         linear += blockDim_) {
      const uint32_t group = linear / blocks_per_group;
      const uint32_t block = linear - group * blocks_per_group;
      const uint32_t row_start = block * rows_per_stat_block;
      const uint32_t rows = MinU32(rows_per_stat_block, m_ - row_start);
      LocalTensor<float> stats = statBuf_.Get<float>();
      LocalTensor<float> mean_stat = stats;
      LocalTensor<float> rstd_stat = stats[kFp32BlockElems];
      Duplicate(stats, 0.0f, kFp32BlockElems * 2);
      PipeBarrier<PIPE_V>();
      for (uint32_t r = 0; r < rows; ++r) {
        ProcessLogicalRow(group, row_start + r, mean_stat, rstd_stat, r, 0);
      }
      if (isRmsNorm_ == 0) {
        StoreFloatValues(meanGm_[group * m_ + row_start],
                         mean_stat,
                         rows,
                         useA2ScalarStoreFix_);
      }
      StoreFloatValues(rstdGm_[group * m_ + row_start],
                       rstd_stat,
                       rows,
                       useA2ScalarStoreFix_);
    }
  }

 private:
  __aicore__ inline void CopyChunkX(LocalTensor<T> x_local,
                                    uint32_t group,
                                    uint32_t row,
                                    uint32_t col,
                                    uint32_t valid) {
    const uint32_t right_pad = chunkAlign_ - valid;
    CopyInPad<T>(x_local,
                 xGm_[row * strideX_ + group * groupSize_ + col],
                 1,
                 valid * sizeof(T),
                 0,
                 0,
                 right_pad);
  }

  __aicore__ inline void CopyChunkZ(LocalTensor<T> z_local,
                                    uint32_t group,
                                    uint32_t row,
                                    uint32_t col,
                                    uint32_t valid) {
    const uint32_t right_pad = chunkAlign_ - valid;
    CopyInPad<T>(z_local,
                 zGm_[row * strideZ_ + group * groupSize_ + col],
                 1,
                 valid * sizeof(T),
                 0,
                 0,
                 right_pad);
  }

  __aicore__ inline void PrefetchInputChunk(uint32_t group,
                                            uint32_t row,
                                            uint32_t col,
                                            uint32_t valid,
                                            uint32_t need_z) {
    LocalTensor<T> x_local = xQueue_.AllocTensor<T>();
    CopyChunkX(x_local, group, row, col, valid);
    xQueue_.EnQue(x_local);

    if (need_z != 0) {
      LocalTensor<T> z_local = zQueue_.AllocTensor<T>();
      CopyChunkZ(z_local, group, row, col, valid);
      zQueue_.EnQue(z_local);
    }
  }

  __aicore__ inline void WaitInputChunk(LocalTensor<T>& x_local,
                                        LocalTensor<T>& z_local,
                                        uint32_t need_z) {
    x_local = xQueue_.DeQue<T>();
    WaitMte2ToV();
    if (need_z != 0) {
      z_local = zQueue_.DeQue<T>();
      WaitMte2ToV();
    }
  }

  __aicore__ inline void PrefetchParamChunk(uint32_t group,
                                            uint32_t col,
                                            uint32_t valid) {
    const uint32_t right_pad = chunkAlign_ - valid;
    LocalTensor<TW> w_raw = weightQueue_.AllocTensor<TW>();
    CopyInPad<TW>(w_raw,
                  wGm_[group * groupSize_ + col],
                  1,
                  valid * sizeof(TW),
                  0,
                  0,
                  right_pad);
    weightQueue_.EnQue(w_raw);

    if (hasBias_ != 0) {
      LocalTensor<TW> b_raw = biasQueue_.AllocTensor<TW>();
      CopyInPad<TW>(b_raw,
                    bGm_[group * groupSize_ + col],
                    1,
                    valid * sizeof(TW),
                    0,
                    0,
                    right_pad);
      biasQueue_.EnQue(b_raw);
    }
  }

  __aicore__ inline void WaitParamChunk(LocalTensor<TW>& w_raw,
                                        LocalTensor<TW>& b_raw) {
    w_raw = weightQueue_.DeQue<TW>();
    WaitMte2ToV();
    if (hasBias_ != 0) {
      b_raw = biasQueue_.DeQue<TW>();
      WaitMte2ToV();
    }
  }

  __aicore__ inline void CastParamChunk(LocalTensor<TW> w_raw,
                                        LocalTensor<TW> b_raw) {
    LocalTensor<float> w_fp32 = weightFp32Buf_.Get<float>();
    CastToFloat<TW>(w_fp32, w_raw, chunkAlign_);
    PipeBarrier<PIPE_V>();

    if (hasBias_ != 0) {
      LocalTensor<float> b_fp32 = biasFp32Buf_.Get<float>();
      CastToFloat<TW>(b_fp32, b_raw, chunkAlign_);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void FreeParamChunk(LocalTensor<TW> w_raw,
                                        LocalTensor<TW> b_raw) {
    weightQueue_.FreeTensor(w_raw);
    if (hasBias_ != 0) {
      biasQueue_.FreeTensor(b_raw);
    }
  }

  __aicore__ inline void ProcessLogicalRow(uint32_t group,
                                           uint32_t row,
                                           LocalTensor<float> mean_stat,
                                           LocalTensor<float> rstd_stat,
                                           uint32_t stat_offset,
                                           uint32_t store_stats = 1) {
    float sum_val = 0.0f;
    float square_sum_val = 0.0f;
    LocalTensor<float> x_fp32 = xFp32Buf_.Get<float>();
    LocalTensor<float> tmp_fp32 = tmpFp32Buf_.Get<float>();
    LocalTensor<float> reduce_tmp = reduceTmpBuf_.Get<float>();
    LocalTensor<float> scalar = scalarBuf_.Get<float>();

    uint32_t current_col = 0;
    uint32_t current_valid = MinU32(chunkSize_, groupSize_);
    const uint32_t pass1_need_z =
        (hasZ_ != 0 && normBeforeGate_ == 0) ? 1U : 0U;
    PrefetchInputChunk(group, row, current_col, current_valid, pass1_need_z);
    while (true) {
      LocalTensor<T> x_local;
      LocalTensor<T> z_local;
      WaitInputChunk(x_local, z_local, pass1_need_z);

      const uint32_t next_col = current_col + current_valid;
      const bool has_next = next_col < groupSize_;
      uint32_t next_valid = 0;
      if (has_next) {
        next_valid = MinU32(chunkSize_, groupSize_ - next_col);
        PrefetchInputChunk(group, row, next_col, next_valid, pass1_need_z);
      }

      CastToFloat<T>(x_fp32, x_local, chunkAlign_);
      PipeBarrier<PIPE_V>();

      if (pass1_need_z != 0) {
        CastToFloat<T>(tmp_fp32, z_local, chunkAlign_);
        PipeBarrier<PIPE_V>();
        Silu(tmp_fp32, tmp_fp32, reduce_tmp, current_valid);
        Mul(x_fp32, x_fp32, tmp_fp32, current_valid);
        PipeBarrier<PIPE_V>();
        zQueue_.FreeTensor(z_local);
      }

      // LayerNorm needs E[x] and E[x^2]. RMSNorm only needs E[x^2], so avoid
      // an otherwise unused ReduceSum/GetValue chain in the streaming path.
      if (isRmsNorm_ == 0) {
        sum_val +=
            ReduceSumValue(scalar, x_fp32, tmp_fp32, reduce_tmp, current_valid);
      }
      Mul(tmp_fp32, x_fp32, x_fp32, current_valid);
      PipeBarrier<PIPE_V>();
      square_sum_val +=
          ReduceSumValueNoCopy(scalar, tmp_fp32, reduce_tmp, current_valid);
      xQueue_.FreeTensor(x_local);

      if (!has_next) {
        break;
      }
      current_col = next_col;
      current_valid = next_valid;
    }

    const float mean_val =
        isRmsNorm_ == 0 ? sum_val / U32ToFloat(groupSize_) : 0.0f;
    const float ex2 = square_sum_val / U32ToFloat(groupSize_);
    float variance = ex2;
    if (isRmsNorm_ == 0) {
      variance = ex2 - mean_val * mean_val;
      variance = variance > 0.0f ? variance : 0.0f;
      mean_stat.SetValue(stat_offset, mean_val);
      if (store_stats != 0) {
        StoreFloatValues(meanGm_[group * m_ + row],
                         mean_stat[stat_offset],
                         1,
                         useA2ScalarStoreFix_);
      }
    }
    const float rstd_val = RstdFromVariance<T>(scalar, variance, eps_);
    rstd_stat.SetValue(stat_offset, rstd_val);
    if (store_stats != 0) {
      StoreFloatValues(rstdGm_[group * m_ + row],
                       rstd_stat[stat_offset],
                       1,
                       useA2ScalarStoreFix_);
    }

    current_col = 0;
    current_valid = MinU32(chunkSize_, groupSize_);
    const uint32_t pass2_need_z = hasZ_ != 0 ? 1U : 0U;
    PrefetchInputChunk(group, row, current_col, current_valid, pass2_need_z);
    PrefetchParamChunk(group, current_col, current_valid);
    while (true) {
      LocalTensor<T> x_local;
      LocalTensor<T> z_local;
      LocalTensor<TW> w_raw;
      LocalTensor<TW> b_raw;
      WaitInputChunk(x_local, z_local, pass2_need_z);
      WaitParamChunk(w_raw, b_raw);

      const uint32_t next_col = current_col + current_valid;
      const bool has_next = next_col < groupSize_;
      uint32_t next_valid = 0;
      if (has_next) {
        next_valid = MinU32(chunkSize_, groupSize_ - next_col);
        PrefetchInputChunk(group, row, next_col, next_valid, pass2_need_z);
        PrefetchParamChunk(group, next_col, next_valid);
      }

      LocalTensor<T> y_local = yQueue_.AllocTensor<T>();
      CastToFloat<T>(x_fp32, x_local, chunkAlign_);
      PipeBarrier<PIPE_V>();

      if (pass2_need_z != 0) {
        if (normBeforeGate_ == 0) {
          CastToFloat<T>(tmp_fp32, z_local, chunkAlign_);
          PipeBarrier<PIPE_V>();
          Silu(tmp_fp32, tmp_fp32, reduce_tmp, current_valid);
          Mul(x_fp32, x_fp32, tmp_fp32, current_valid);
          PipeBarrier<PIPE_V>();
        }
      }

      if (isRmsNorm_ == 0) {
        Adds(x_fp32, x_fp32, -mean_val, current_valid);
        PipeBarrier<PIPE_V>();
      }
      Muls(x_fp32, x_fp32, rstd_val, current_valid);
      PipeBarrier<PIPE_V>();
      CastParamChunk(w_raw, b_raw);
      Mul(x_fp32, x_fp32, weightFp32Buf_.Get<float>(), current_valid);
      PipeBarrier<PIPE_V>();
      if (hasBias_ != 0) {
        Add(x_fp32, x_fp32, biasFp32Buf_.Get<float>(), current_valid);
        PipeBarrier<PIPE_V>();
      }
      if (hasZ_ != 0 && normBeforeGate_ != 0) {
        CastToFloat<T>(tmp_fp32, z_local, chunkAlign_);
        PipeBarrier<PIPE_V>();
        Silu(tmp_fp32, tmp_fp32, reduce_tmp, current_valid);
        Mul(x_fp32, x_fp32, tmp_fp32, current_valid);
        PipeBarrier<PIPE_V>();
      }
      if (hasZ_ != 0) {
        zQueue_.FreeTensor(z_local);
      }
      FreeParamChunk(w_raw, b_raw);

      CastFromFloat<T>(y_local, x_fp32, chunkAlign_);
      PipeBarrier<PIPE_V>();
      yQueue_.EnQue(y_local);
      xQueue_.FreeTensor(x_local);
      y_local = yQueue_.DeQue<T>();
      WaitVToMte3();
      CopyOutPad<T>(yGm_[row * strideY_ + group * groupSize_ + current_col],
                    y_local,
                    1,
                    current_valid * sizeof(T),
                    0,
                    0);
      // The next chunk reloads x/weight into reused UB buffers. Without this
      // fence, V may still read the previous weight while MTE2 overwrites it.
      WaitVToMte2();
      yQueue_.FreeTensor(y_local);

      if (!has_next) {
        break;
      }
      current_col = next_col;
      current_valid = next_valid;
    }
  }

 private:
  TPipe pipe_;
  TQue<QuePosition::VECIN, kBufferNum> xQueue_;
  TQue<QuePosition::VECIN, kBufferNum> zQueue_;
  TQue<QuePosition::VECIN, kBufferNum> weightQueue_;
  TQue<QuePosition::VECIN, kBufferNum> biasQueue_;
  TQue<QuePosition::VECOUT, kBufferNum> yQueue_;
  TBuf<TPosition::VECCALC> xFp32Buf_;
  TBuf<TPosition::VECCALC> tmpFp32Buf_;
  TBuf<TPosition::VECCALC> reduceTmpBuf_;
  TBuf<TPosition::VECCALC> scalarBuf_;
  TBuf<TPosition::VECCALC> statBuf_;
  TBuf<TPosition::VECCALC> weightFp32Buf_;
  TBuf<TPosition::VECCALC> biasFp32Buf_;

  GlobalTensor<T> xGm_;
  GlobalTensor<T> yGm_;
  GlobalTensor<T> zGm_;
  GlobalTensor<TW> wGm_;
  GlobalTensor<TW> bGm_;
  GlobalTensor<float> meanGm_;
  GlobalTensor<float> rstdGm_;

  uint32_t m_ = 0;
  uint32_t fullN_ = 0;
  uint32_t groupSize_ = 0;
  uint32_t ngroups_ = 0;
  uint32_t strideX_ = 0;
  uint32_t strideY_ = 0;
  uint32_t strideZ_ = 0;
  uint32_t chunkSize_ = 0;
  uint32_t chunkAlign_ = 0;
  uint32_t blockDim_ = 1;
  uint32_t coreId_ = 0;
  uint32_t hasBias_ = 0;
  uint32_t hasZ_ = 0;
  uint32_t normBeforeGate_ = 1;
  uint32_t isRmsNorm_ = 0;
  uint32_t useA2ScalarStoreFix_ = 0;
  float eps_ = 1e-6f;
};

}  // namespace layer_norm_fwd_impl

extern "C" __global__ __aicore__ void xllm_kernel_layer_norm_fwd(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR z,
    GM_ADDR y,
    GM_ADDR mean,
    GM_ADDR rstd,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(tiling_data, tiling);
  if (tiling_data.kernel_mode == layer_norm_fwd_impl::kStreamingTwoPass) {
    layer_norm_fwd_impl::KernelStreamingTwoPass<DTYPE_X, DTYPE_WEIGHT> op;
    op.Init(x,
            weight,
            bias,
            z,
            y,
            mean,
            rstd,
            tiling_data.m,
            tiling_data.full_n,
            tiling_data.group_size,
            tiling_data.ngroups,
            tiling_data.stride_x,
            tiling_data.stride_y,
            tiling_data.stride_z,
            tiling_data.chunk_size,
            tiling_data.group_align,
            tiling_data.eps,
            tiling_data.has_bias,
            tiling_data.has_z,
            tiling_data.norm_before_gate,
            tiling_data.is_rms_norm,
            tiling_data.use_a2_scalar_store_fix);
    op.Process();
    return;
  }

  layer_norm_fwd_impl::KernelFullRow<DTYPE_X, DTYPE_WEIGHT> op;
  op.Init(x,
          weight,
          bias,
          z,
          y,
          mean,
          rstd,
          tiling_data.m,
          tiling_data.full_n,
          tiling_data.group_size,
          tiling_data.ngroups,
          tiling_data.stride_x,
          tiling_data.stride_y,
          tiling_data.stride_z,
          tiling_data.group_align,
          tiling_data.tile_rows,
          tiling_data.eps,
          tiling_data.has_bias,
          tiling_data.has_z,
          tiling_data.norm_before_gate,
          tiling_data.is_rms_norm,
          tiling_data.use_a2_scalar_store_fix);
  op.Process();
}
