// Reusable prefill GDN frontend helpers.
//
// RunConv performs causal convolution, cache update, SiLU, and packed
// normalized Q/K/V preparation. PrepareGate produces the FP32 decay and
// FP16/BF16 gate tensors consumed by the GDN pipeline.

#pragma once

#ifndef GDN_D
#define GDN_D 128
#endif
#ifndef GDN_C
#define GDN_C 128
#endif
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif

#include "acl/acl.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#define CAUSAL_CONV1D_SKIP_TPL_REGISTRATION
#if defined(GDN_PREFILL_ARCH_A5)
#define CAUSAL_CONV1D_DISABLE_LEGACY_PACKED_QKV_MAPPING
#endif
#ifndef GDN_PREFILL_PACKED_QKV_DTYPE
#define GDN_PREFILL_PACKED_QKV_DTYPE bfloat16_t
#endif
#define DTYPE_Y GDN_PREFILL_PACKED_QKV_DTYPE
#include "causal_conv1d/op_kernel/causal_conv1d_fn.h"
#undef DTYPE_Y
#if defined(GDN_PREFILL_ARCH_A5)
#undef CAUSAL_CONV1D_DISABLE_LEGACY_PACKED_QKV_MAPPING
#endif
#undef CAUSAL_CONV1D_SKIP_TPL_REGISTRATION

struct GdnPrefillFrontendTilingData {
  uint32_t num_heads;
  uint32_t num_key_heads;
  uint32_t token_block_size;
  uint32_t token_block_count;
  uint32_t base_dim;
  uint32_t base_dim_count;
  uint32_t conv_dim;
  uint32_t conv_state_slots;
  int64_t total_tokens;
};

#ifdef __CCE_AICORE__
namespace gdn_prefill_frontend {

template <typename T, uint32_t dstN, uint32_t dstM = 1>
__aicore__ inline void CopyGmToUb(AscendC::LocalTensor<T> dst,
                                  AscendC::GlobalTensor<T> src,
                                  uint32_t realSrcN = 1,
                                  uint32_t maskShapeM = dstM,
                                  uint32_t maskShapeN = dstN,
                                  T padValue = T(0)) {
  if (maskShapeM != dstM || maskShapeN != dstN) {
    AscendC::Duplicate(dst, padValue, dstM * dstN);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(7);
  }
  const bool isPad = maskShapeN != dstN;
  AscendC::DataCopyExtParams params(maskShapeM,
                                    maskShapeN * sizeof(T),
                                    (realSrcN - maskShapeN) * sizeof(T),
                                    (dstN - maskShapeN) * sizeof(T) / 32,
                                    0);
  AscendC::DataCopyPadExtParams<T> padParams(isPad, 0, 0, padValue);
  AscendC::DataCopyPad(dst, src, params, padParams);
}

template <typename T, uint32_t srcN, uint32_t srcM = 1>
__aicore__ inline void CopyUbToGm(AscendC::GlobalTensor<T> dst,
                                  AscendC::LocalTensor<T> src,
                                  uint32_t realDstN = 1,
                                  uint32_t maskShapeM = srcM,
                                  uint32_t maskShapeN = srcN) {
  AscendC::DataCopyExtParams params(maskShapeM,
                                    maskShapeN * sizeof(T),
                                    (srcN - maskShapeN) * sizeof(T) / 32,
                                    (realDstN - maskShapeN) * sizeof(T),
                                    0);
  AscendC::DataCopyPad(dst, src, params);
}

template <typename T, uint32_t widthKey, uint32_t fnPlanKey>
class PrefillConv : public NsCausalConv1d::XllmKernelCausalConv1d<
                        T,
                        CAUSAL_CONV1D_TPL_RUN_MODE_FN,
                        widthKey,
                        fnPlanKey> {
 public:
  __aicore__ inline void Init(GM_ADDR x,
                              GM_ADDR weight,
                              GM_ADDR convStateIn,
                              GM_ADDR convStateOut,
                              GM_ADDR packedQkv,
                              const CausalConv1dTilingData* tilingData) {
    this->ResetRuntimeState(tilingData);
    this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    this->weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight));
    this->convStatesGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(convStateIn));
    this->convStatesOutGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(convStateOut));
    convStateOutRaw_ = reinterpret_cast<__gm__ T*>(convStateOut);
    this->packedQkvYGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ NsCausalConv1d::PackedQkvT*>(packedQkv));
    this->InitSharedBuffersAndEvents();
  }

  __aicore__ inline void CleanConvStateRows(int32_t writeCacheIndex,
                                            int32_t stateRowBegin,
                                            int32_t stateRowEnd,
                                            int32_t channelStart,
                                            int32_t baseDim) {
#if defined(GDN_PREFILL_ARCH_A5)
    if (writeCacheIndex < 0 ||
        writeCacheIndex >=
            static_cast<int32_t>(this->tilingData_->numCacheLines) ||
        stateRowBegin >= stateRowEnd) {
      return;
    }
    // DCCI cleans the issuing AIV's private cache, so publish each line on
    // the same token/channel task that issued its MTE3 state write.
    pipe_barrier(PIPE_ALL);
    constexpr int32_t kElementsPerCacheLine = 64 / sizeof(T);
    const int32_t dim = static_cast<int32_t>(this->tilingData_->dim);
    for (int32_t stateRow = stateRowBegin; stateRow < stateRowEnd; ++stateRow) {
      const int64_t rowOffset =
          (static_cast<int64_t>(writeCacheIndex) * this->tilingData_->stateLen +
           stateRow) *
              dim +
          channelStart;
      for (int32_t column = 0; column < baseDim;
           column += kElementsPerCacheLine) {
        dcci(static_cast<__gm__ void*>(convStateOutRaw_ + rowOffset + column),
             SINGLE_CACHE_LINE);
      }
    }
    dsb(DSB_ALL);
#else
    (void)writeCacheIndex;
    (void)stateRowBegin;
    (void)stateRowEnd;
    (void)channelStart;
    (void)baseDim;
#endif
  }

  __aicore__ inline void CopyCheckpointTail(int32_t readCacheIndex,
                                            int32_t writeCacheIndex,
                                            int32_t channelStart,
                                            int32_t baseDim,
                                            bool separateSource) {
    if (readCacheIndex < 0 ||
        (!separateSource && readCacheIndex == writeCacheIndex)) {
      return;
    }

    const int32_t stateLen = static_cast<int32_t>(this->tilingData_->stateLen);
    const int32_t width = static_cast<int32_t>(this->tilingData_->width);
    const int32_t dim = static_cast<int32_t>(this->tilingData_->dim);
    auto tmp = this->inBuf.template Get<T>();
    for (int32_t pos = width - 1; pos < stateLen; ++pos) {
      const int64_t srcOffset =
          (static_cast<int64_t>(readCacheIndex) * stateLen + pos) * dim +
          channelStart;
      const int64_t dstOffset =
          (static_cast<int64_t>(writeCacheIndex) * stateLen + pos) * dim +
          channelStart;
      AscendC::DataCopy(tmp, this->convStatesGm[srcOffset], baseDim);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
          this->stateShiftMte2ToMte3Event_);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
          this->stateShiftMte2ToMte3Event_);
      AscendC::DataCopy(this->convStatesOutGm[dstOffset], tmp, baseDim);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
          this->stateShiftMte3ToMte2Event_);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
          this->stateShiftMte3ToMte2Event_);
    }
  }

#if defined(GDN_PREFILL_ARCH_A5)
  __aicore__ inline void ProcessSingle(int32_t readCacheIndex,
                                       int32_t writeCacheIndex,
                                       bool separateSource = false)
#else
  __aicore__ inline void ProcessSingle(int32_t readCacheIndex,
                                       int32_t writeCacheIndex)
#endif
  {
#if defined(GDN_PREFILL_ARCH_A5)
    const int32_t processReadCacheIndex =
        separateSource && readCacheIndex >= 0 ? 0 : readCacheIndex;
#endif
    const int32_t blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
    const int32_t tokenCount =
        static_cast<int32_t>(this->tilingData_->cuSeqlen);
    const auto task = NsCausalConv1d::ResolveFnPackedQkvBlockTask(
        blockIdx,
        static_cast<int32_t>(this->tilingData_->tokenBlockCnt),
        static_cast<int32_t>(this->tilingData_->tokenBlockSize),
        tokenCount,
        static_cast<int32_t>(this->tilingData_->baseDimCnt),
        static_cast<int32_t>(this->tilingData_->baseDim),
        static_cast<int32_t>(this->tilingData_->dim),
        static_cast<int32_t>(this->tilingData_->packedQDim),
        static_cast<int32_t>(this->tilingData_->packedKDim),
        static_cast<int32_t>(this->tilingData_->packedVDim));
    if (task.valid) {
      this->ProcessFnChunk(
#if defined(GDN_PREFILL_ARCH_A5)
          processReadCacheIndex,
          writeCacheIndex,
          readCacheIndex >= 0,
          0,
#else
          readCacheIndex,
          writeCacheIndex,
          readCacheIndex >= 0,
          0,
#endif
          tokenCount,
          task.tokenStart,
          task.tokenEnd - task.tokenStart,
          task.channelStart,
          task.baseDimSize,
          static_cast<int32_t>(this->tilingData_->dim));
      if (task.tokenEnd == tokenCount) {
        CleanConvStateRows(writeCacheIndex,
                           0,
                           static_cast<int32_t>(this->tilingData_->width) - 1,
                           task.channelStart,
                           task.baseDimSize);
      }
      if (task.tokenTileId == 0) {
#if defined(GDN_PREFILL_ARCH_A5)
        CopyCheckpointTail(processReadCacheIndex,
                           writeCacheIndex,
                           task.channelStart,
                           task.baseDimSize,
                           separateSource);
        if (readCacheIndex >= 0 &&
            (separateSource || readCacheIndex != writeCacheIndex)) {
#else
        CopyCheckpointTail(readCacheIndex,
                           writeCacheIndex,
                           task.channelStart,
                           task.baseDimSize,
                           false);
        if (readCacheIndex >= 0 && readCacheIndex != writeCacheIndex) {
#endif
          CleanConvStateRows(writeCacheIndex,
                             static_cast<int32_t>(this->tilingData_->width) - 1,
                             static_cast<int32_t>(this->tilingData_->stateLen),
                             task.channelStart,
                             task.baseDimSize);
        }
      }
    }
    this->ReleaseEvents();
    this->pipe.Destroy();
  }

  __aicore__ inline void Process(__gm__ int32_t* readStateIndices,
                                 __gm__ int32_t* writeStateIndices,
                                 __gm__ int32_t* cuSeqlens,
                                 int32_t batch,
                                 bool useCompactSnapshot) {
    const int32_t blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
    const int32_t tokenCount =
        static_cast<int32_t>(this->tilingData_->cuSeqlen);
    const auto task = NsCausalConv1d::ResolveFnPackedQkvBlockTask(
        blockIdx,
        static_cast<int32_t>(this->tilingData_->tokenBlockCnt),
        static_cast<int32_t>(this->tilingData_->tokenBlockSize),
        tokenCount,
        static_cast<int32_t>(this->tilingData_->baseDimCnt),
        static_cast<int32_t>(this->tilingData_->baseDim),
        static_cast<int32_t>(this->tilingData_->dim),
        static_cast<int32_t>(this->tilingData_->packedQDim),
        static_cast<int32_t>(this->tilingData_->packedKDim),
        static_cast<int32_t>(this->tilingData_->packedVDim));
    if (task.valid && batch > 0) {
      int32_t left = 0;
      int32_t right = batch;
      while (left < right) {
        const int32_t mid = left + ((right - left) >> 1);
        if (task.tokenStart < cuSeqlens[mid + 1]) {
          right = mid;
        } else {
          left = mid + 1;
        }
      }

      int32_t seq = left;
      int32_t cursor = task.tokenStart;
      while (cursor < task.tokenEnd && seq < batch) {
        const int32_t seqStart = cuSeqlens[seq];
        const int32_t seqEnd = cuSeqlens[seq + 1];
        if (cursor < seqStart) {
          cursor = seqStart;
        }
        if (cursor >= seqEnd) {
          ++seq;
          continue;
        }
        const int32_t tileEnd = task.tokenEnd < seqEnd ? task.tokenEnd : seqEnd;
        const int32_t tileLen = tileEnd - cursor;
        const int32_t readCacheIndex = readStateIndices[seq];
        const int32_t writeCacheIndex = writeStateIndices[seq];
        if (tileLen > 0 &&
            readCacheIndex <
                static_cast<int32_t>(this->tilingData_->numCacheLines) &&
            writeCacheIndex >= 0 &&
            writeCacheIndex <
                static_cast<int32_t>(this->tilingData_->numCacheLines)) {
          const int32_t processReadCacheIndex =
              useCompactSnapshot && readCacheIndex >= 0 ? seq : readCacheIndex;
          this->ProcessFnChunk(processReadCacheIndex,
                               writeCacheIndex,
                               readCacheIndex >= 0,
                               seqStart,
                               seqEnd - seqStart,
                               cursor,
                               tileLen,
                               task.channelStart,
                               task.baseDimSize,
                               static_cast<int32_t>(this->tilingData_->dim));
          if (tileEnd == seqEnd) {
            CleanConvStateRows(
                writeCacheIndex,
                0,
                static_cast<int32_t>(this->tilingData_->width) - 1,
                task.channelStart,
                task.baseDimSize);
          }
          if (cursor == seqStart) {
            CopyCheckpointTail(processReadCacheIndex,
                               writeCacheIndex,
                               task.channelStart,
                               task.baseDimSize,
                               useCompactSnapshot);
            if (processReadCacheIndex >= 0 &&
                (useCompactSnapshot ||
                 processReadCacheIndex != writeCacheIndex)) {
              CleanConvStateRows(
                  writeCacheIndex,
                  static_cast<int32_t>(this->tilingData_->width) - 1,
                  static_cast<int32_t>(this->tilingData_->stateLen),
                  task.channelStart,
                  task.baseDimSize);
            }
          }
        }
        cursor = tileEnd;
        ++seq;
      }
    }
    this->ReleaseEvents();
    this->pipe.Destroy();
  }

  __aicore__ inline bool NeedsCompactSnapshot(__gm__ int32_t* readStateIndices,
                                              __gm__ int32_t* writeStateIndices,
                                              int32_t batch) const {
    if (batch <= 1) {
      return false;
    }
    for (int32_t readSeq = 0; readSeq < batch; ++readSeq) {
      const int32_t readCacheIndex = readStateIndices[readSeq];
      if (readCacheIndex < 0) {
        continue;
      }
      for (int32_t writeSeq = 0; writeSeq < batch; ++writeSeq) {
        if (readCacheIndex == writeStateIndices[writeSeq]) {
          return true;
        }
      }
    }
    return false;
  }

  __aicore__ inline void SnapshotInitialStates(GM_ADDR compactSnapshot,
                                               __gm__ int32_t* readStateIndices,
                                               int32_t batch) {
    const int32_t blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
    const int32_t tokenCount =
        static_cast<int32_t>(this->tilingData_->cuSeqlen);
    const auto task = NsCausalConv1d::ResolveFnPackedQkvBlockTask(
        blockIdx,
        static_cast<int32_t>(this->tilingData_->tokenBlockCnt),
        static_cast<int32_t>(this->tilingData_->tokenBlockSize),
        tokenCount,
        static_cast<int32_t>(this->tilingData_->baseDimCnt),
        static_cast<int32_t>(this->tilingData_->baseDim),
        static_cast<int32_t>(this->tilingData_->dim),
        static_cast<int32_t>(this->tilingData_->packedQDim),
        static_cast<int32_t>(this->tilingData_->packedKDim),
        static_cast<int32_t>(this->tilingData_->packedVDim));
    if (!NsCausalConv1d::IsFnInitStateSnapshotOwnerBlock(task)) {
      return;
    }

    const int32_t stateLen = static_cast<int32_t>(this->tilingData_->stateLen);
    const int32_t dim = static_cast<int32_t>(this->tilingData_->dim);
    const int32_t numCacheLines =
        static_cast<int32_t>(this->tilingData_->numCacheLines);
    auto tmp = this->inBuf.template Get<T>();
    AscendC::GlobalTensor<T> snapshotGm;
    snapshotGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(compactSnapshot));
    for (int32_t seq = 0; seq < batch; ++seq) {
      const int32_t readCacheIndex = readStateIndices[seq];
      if (readCacheIndex < 0 || readCacheIndex >= numCacheLines) {
        continue;
      }
      for (int32_t pos = 0; pos < stateLen; ++pos) {
        const int64_t srcOffset =
            (static_cast<int64_t>(readCacheIndex) * stateLen + pos) * dim +
            task.channelStart;
        const int64_t dstOffset =
            (static_cast<int64_t>(seq) * stateLen + pos) * dim +
            task.channelStart;
        AscendC::DataCopy(tmp, this->convStatesGm[srcOffset], task.baseDimSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
            this->stateShiftMte2ToMte3Event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
            this->stateShiftMte2ToMte3Event_);
        AscendC::DataCopy(snapshotGm[dstOffset], tmp, task.baseDimSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
            this->stateShiftMte3ToMte2Event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
            this->stateShiftMte3ToMte2Event_);
      }
    }
  }

  __aicore__ inline void UseCompactSnapshot(GM_ADDR compactSnapshot) {
    this->convStatesGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(compactSnapshot));
  }

 private:
  __gm__ T* convStateOutRaw_ = nullptr;
};

__aicore__ inline void RunConv(
    GM_ADDR mixedQkv,
    GM_ADDR convWeight,
    GM_ADDR convStateIn,
    GM_ADDR convStateOut,
    GM_ADDR readStateIndices,
    GM_ADDR writeStateIndices,
    GM_ADDR cuSeqlens,
    GM_ADDR packedQkv,
    GM_ADDR compactConvStateSnapshot,
    const GdnPrefillFrontendTilingData& frontendTiling,
    uint32_t batchSize,
    uint32_t convStateLen) {
#if defined(__DAV_C220_VEC__) || defined(__DAV_VEC__)
  CausalConv1dTilingData tiling{};
  tiling.dim = frontendTiling.conv_dim;
  tiling.cuSeqlen = frontendTiling.total_tokens;
  tiling.seqLen = frontendTiling.total_tokens;
  tiling.inputMode = 0;
  tiling.width = 4;
  tiling.stateLen = convStateLen;
  tiling.numCacheLines = frontendTiling.conv_state_slots;
  tiling.batch = batchSize;
  tiling.activationMode = CAUSAL_CONV1D_ACTIVATION_SILU_PACKED_QKV;
  tiling.padSlotId = -1;
  tiling.hasBias = 0;
  tiling.packedQDim = frontendTiling.num_key_heads * GDN_D;
  tiling.packedKDim = frontendTiling.num_key_heads * GDN_D;
  tiling.packedVDim = frontendTiling.num_heads * GDN_D;
  tiling.packedHeadDim = GDN_D;
  tiling.baseDim = frontendTiling.base_dim;
  tiling.baseDimCnt = frontendTiling.base_dim_count;
  tiling.hasNumAcceptedTokens = 0;
  tiling.hasCacheIndices = 0;
  tiling.hasInitialStateMode = 0;
  tiling.hasInitStateWorkspace = 0;
  tiling.tokenBlockSize = frontendTiling.token_block_size;
  tiling.tokenBlockCnt = frontendTiling.token_block_count;
  tiling.hasExplicitTokenSeqRanges = 0;

  PrefillConv<bfloat16_t,
              CAUSAL_CONV1D_TPL_WIDTH_4,
              CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD>
      op;
  op.Init(mixedQkv, convWeight, convStateIn, convStateOut, packedQkv, &tiling);
  auto readIndices = reinterpret_cast<__gm__ int32_t*>(readStateIndices);
  auto writeIndices = reinterpret_cast<__gm__ int32_t*>(writeStateIndices);
  if (batchSize == 1) {
#if defined(GDN_PREFILL_ARCH_A5)
    const int32_t readCacheIndex = readIndices[0];
    const int32_t writeCacheIndex = writeIndices[0];
    const bool useCompactSnapshot =
        compactConvStateSnapshot != nullptr && readCacheIndex >= 0 &&
        readCacheIndex <
            static_cast<int32_t>(frontendTiling.conv_state_slots) &&
        readCacheIndex == writeCacheIndex;
    if (useCompactSnapshot) {
      op.SnapshotInitialStates(compactConvStateSnapshot, readIndices, 1);
      // No token tile may read or overwrite the aliased cache slot until
      // every channel slice of sequence 0 has reached the snapshot.
      pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
      op.UseCompactSnapshot(compactConvStateSnapshot);
    }
    op.ProcessSingle(readCacheIndex, writeCacheIndex, useCompactSnapshot);
#else
    op.ProcessSingle(readIndices[0], writeIndices[0]);
#endif
  } else {
    const bool useCompactSnapshot =
        compactConvStateSnapshot != nullptr &&
        op.NeedsCompactSnapshot(
            readIndices, writeIndices, static_cast<int32_t>(batchSize));
    if (useCompactSnapshot) {
      op.SnapshotInitialStates(compactConvStateSnapshot,
                               readIndices,
                               static_cast<int32_t>(batchSize));
#if defined(GDN_PREFILL_ARCH_A5)
      pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
#else
      AscendC::SyncAll();
#endif
      op.UseCompactSnapshot(compactConvStateSnapshot);
    }
    op.Process(readIndices,
               writeIndices,
               reinterpret_cast<__gm__ int32_t*>(cuSeqlens),
               static_cast<int32_t>(batchSize),
               useCompactSnapshot);
  }
#endif
}

__aicore__ inline int32_t VectorTaskId() {
#if defined(__DAV_C220_VEC__) || defined(__DAV_VEC__)
  return static_cast<int32_t>(AscendC::GetBlockIdx() / 2 * 2 +
                              AscendC::GetSubBlockIdx());
#else
  return 0;
#endif
}

template <typename BetaT = bfloat16_t>
__aicore__ inline void PrepareGate(GM_ADDR aPtr,
                                   GM_ADDR bPtr,
                                   GM_ADDR aLogPtr,
                                   GM_ADDR dtBiasPtr,
                                   GM_ADDR gPtr,
                                   GM_ADDR betaPtr,
                                   int64_t totalTokens,
                                   int32_t numHeads,
                                   int32_t vectorTaskCount,
                                   bool roundGToBf16) {
#if defined(__DAV_C220_VEC__) || defined(__DAV_VEC__)
  static_assert(AscendC::IsSameType<BetaT, half>::value ||
                    AscendC::IsSameType<BetaT, bfloat16_t>::value,
                "Gate beta output supports FP16 or BF16.");
  if (numHeads <= 0 || numHeads > 64 || vectorTaskCount <= 0) {
    return;
  }
  constexpr int32_t kVectorWidth = 64;
  constexpr int32_t kRowsPerTile = 16;
  constexpr int32_t kTileElements = kRowsPerTile * kVectorWidth;

  const int32_t taskId = VectorTaskId();
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
  pipe.InitBuffer(ub, 98304);

  auto a_log = ub.GetWithOffset<float>(kVectorWidth, 0);
  auto dtBias = ub.GetWithOffset<float>(kVectorWidth, 256);
  auto negExpARows = ub.GetWithOffset<float>(kTileElements, 512);
  auto dtBiasRows = ub.GetWithOffset<float>(kTileElements, 4608);
  auto aBf16 = ub.GetWithOffset<bfloat16_t>(kTileElements, 8704);
  auto bBf16 = ub.GetWithOffset<bfloat16_t>(kTileElements, 10752);
  auto x = ub.GetWithOffset<float>(kTileElements, 12800);
  auto betaX = ub.GetWithOffset<float>(kTileElements, 16896);
  auto absX = ub.GetWithOffset<float>(kTileElements, 20992);
  auto tmp = ub.GetWithOffset<float>(kTileElements, 25088);
  auto betaFp32 = ub.GetWithOffset<float>(kTileElements, 29184);
  auto betaRounded = ub.GetWithOffset<bfloat16_t>(kTileElements, 33280);
  auto cmpMask = ub.GetWithOffset<uint8_t>(kTileElements, 35328);
  auto sigmoidTmp = ub.GetWithOffset<uint8_t>(32768, 36352);

  AscendC::GlobalTensor<float> aLogGm;
  AscendC::GlobalTensor<float> dtBiasGm;
  AscendC::GlobalTensor<bfloat16_t> aGm;
  AscendC::GlobalTensor<bfloat16_t> bGm;
  AscendC::GlobalTensor<float> gGm;
  AscendC::GlobalTensor<BetaT> betaGm;
  aLogGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aLogPtr));
  dtBiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dtBiasPtr));
  aGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(aPtr));
  bGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(bPtr));
  gGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gPtr));
  betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT*>(betaPtr));

  AscendC::Duplicate(a_log, 0.0f, kVectorWidth);
  AscendC::Duplicate(dtBias, 0.0f, kVectorWidth);
  AscendC::PipeBarrier<PIPE_V>();
#if defined(GDN_PREFILL_ARCH_A5)
  // For a full 64-head row CopyGmToUb does not enter its padding path, so
  // it does not emit the V->MTE2 dependency used by partial rows.  Order
  // these loads after the initialization writes explicitly; otherwise the
  // MTE2 load can race the Vector duplicate only at numHeads == 64.
  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(7);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(7);
#endif
  CopyGmToUb<float, kVectorWidth>(
      a_log[0], aLogGm[0], numHeads, 1, numHeads, 0.0f);
  CopyGmToUb<float, kVectorWidth>(
      dtBias[0], dtBiasGm[0], numHeads, 1, numHeads, 0.0f);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
  AscendC::Exp(a_log[0], a_log[0], kVectorWidth);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Muls(a_log[0], a_log[0], -1.0f, kVectorWidth);
  AscendC::PipeBarrier<PIPE_V>();
  for (int32_t row = 0; row < kRowsPerTile; ++row) {
    AscendC::DataCopy(negExpARows[row * kVectorWidth], a_log[0], kVectorWidth);
    AscendC::DataCopy(dtBiasRows[row * kVectorWidth], dtBias[0], kVectorWidth);
  }
  AscendC::PipeBarrier<PIPE_V>();

  const int64_t rowsPerTask =
      (totalTokens + vectorTaskCount - 1) / vectorTaskCount;
  const int64_t taskStart = static_cast<int64_t>(taskId) * rowsPerTask;
  const int64_t taskEnd = (taskStart + rowsPerTask < totalTokens)
                              ? taskStart + rowsPerTask
                              : totalTokens;
  for (int64_t tileStart = taskStart; tileStart < taskEnd;
       tileStart += kRowsPerTile) {
    const int32_t validRows = static_cast<int32_t>(
        (tileStart + kRowsPerTile < taskEnd) ? kRowsPerTile
                                             : taskEnd - tileStart);
    const uint32_t elements = static_cast<uint32_t>(validRows * kVectorWidth);
    CopyGmToUb<bfloat16_t, kVectorWidth, kRowsPerTile>(
        aBf16[0],
        aGm[tileStart * numHeads],
        numHeads,
        validRows,
        numHeads,
        bfloat16_t(0.0f));
    CopyGmToUb<bfloat16_t, kVectorWidth, kRowsPerTile>(
        bBf16[0],
        bGm[tileStart * numHeads],
        numHeads,
        validRows,
        numHeads,
        bfloat16_t(0.0f));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);

    AscendC::Cast(x[0], aBf16[0], AscendC::RoundMode::CAST_NONE, elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(x[0], x[0], dtBiasRows[0], elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Abs(absX[0], x[0], elements);
    AscendC::Add(betaX[0], x[0], absX[0], elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(betaX[0], betaX[0], 0.5f, elements);
    AscendC::Muls(tmp[0], absX[0], -1.0f, elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(absX[0], tmp[0], elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(betaFp32[0], absX[0], 1.0f, elements);
    AscendC::PipeBarrier<PIPE_V>();
    // Compensated log1p(y) = log(1 + y) + (y - ((1 + y) - 1)) / (1 + y).
    // The correction preserves the low bits used by the unfused softplus
    // before its BF16 gate rounding.
    AscendC::Adds(tmp[0], betaFp32[0], -1.0f, elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(tmp[0], absX[0], tmp[0], elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(tmp[0], tmp[0], betaFp32[0], elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Ln(absX[0], betaFp32[0], elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(tmp[0], absX[0], tmp[0], elements);
    AscendC::CompareScalar(
        cmpMask[0], x[0], 20.0f, AscendC::CMPMODE::GT, elements);
    AscendC::Add(betaX[0], betaX[0], tmp[0], elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Select(betaX[0],
                    cmpMask[0],
                    x[0],
                    betaX[0],
                    AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                    elements);

    AscendC::Cast(x[0], bBf16[0], AscendC::RoundMode::CAST_NONE, elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sigmoid(betaFp32[0], x[0], sigmoidTmp[0], elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(x[0], negExpARows[0], betaX[0], elements);
    AscendC::Cast(
        betaRounded[0], betaFp32[0], AscendC::RoundMode::CAST_RINT, elements);
    if constexpr (AscendC::IsSameType<BetaT, half>::value) {
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(
          betaFp32[0], betaRounded[0], AscendC::RoundMode::CAST_NONE, elements);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(betaRounded[0].template ReinterpretCast<half>(),
                    betaFp32[0],
                    AscendC::RoundMode::CAST_RINT,
                    elements);
    }
    if (roundGToBf16) {
      // Match paths that materialize g in the BF16 dtype of a before
      // MegaChunkGdn converts it back to FP32.
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(aBf16[0], x[0], AscendC::RoundMode::CAST_RINT, elements);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(x[0], aBf16[0], AscendC::RoundMode::CAST_NONE, elements);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::PipeBarrier<PIPE_MTE3>();
    CopyUbToGm<float, kVectorWidth, kRowsPerTile>(
        gGm[tileStart * numHeads], x[0], numHeads, validRows, numHeads);
    AscendC::PipeBarrier<PIPE_MTE3>();
    CopyUbToGm<BetaT, kVectorWidth, kRowsPerTile>(
        betaGm[tileStart * numHeads],
        betaRounded[0].template ReinterpretCast<BetaT>(),
        numHeads,
        validRows,
        numHeads);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
    AscendC::PipeBarrier<PIPE_ALL>();
  }
  AscendC::PipeBarrier<PIPE_ALL>();
  pipe.Destroy();
#endif
}

}  // namespace gdn_prefill_frontend
#endif
