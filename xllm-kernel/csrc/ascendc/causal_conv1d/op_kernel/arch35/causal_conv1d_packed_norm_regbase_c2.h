/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef CAUSAL_CONV1D_PACKED_NORM_REGBASE_C2_H
#define CAUSAL_CONV1D_PACKED_NORM_REGBASE_C2_H

namespace NsCausalConv1d {

inline constexpr uint32_t C2_PACKED_HEAD_DIM = 128;
inline constexpr uint32_t C2_FP32_VECTOR_LENGTH =
    AscendC::VECTOR_REG_WIDTH / sizeof(float);
inline constexpr float C2_PACKED_NORM_EPSILON = 1.0e-6f;

static_assert(
    C2_FP32_VECTOR_LENGTH == C2_PACKED_HEAD_DIM / 2,
    "C2 packed normalization requires two FP32 vector registers per head.");

inline constexpr AscendC::MicroAPI::CastTrait C2_CAST_B16_TO_B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

inline constexpr AscendC::MicroAPI::CastTrait C2_CAST_B32_TO_B16_RINT = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

// Preserve the existing packed-Q/K numeric boundary exactly:
// BF16 input -> FP32 normalize -> BF16 RINT -> FP32 -> FP16 RINT.
static __simd_vf__ inline void NormalizeAndPackQkHeadsC2(
    __ubuf__ bfloat16_t* srcAddr,
    __ubuf__ half* dstAddr,
    uint16_t headCount) {
  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<bfloat16_t> srcLo;
    AscendC::MicroAPI::RegTensor<bfloat16_t> srcHi;
    AscendC::MicroAPI::RegTensor<float> xLo;
    AscendC::MicroAPI::RegTensor<float> xHi;
    AscendC::MicroAPI::RegTensor<float> squareLo;
    AscendC::MicroAPI::RegTensor<float> squareHi;
    AscendC::MicroAPI::RegTensor<float> normScalar;
    AscendC::MicroAPI::RegTensor<half> packedLo;
    AscendC::MicroAPI::RegTensor<half> packedHi;
    AscendC::MicroAPI::MaskReg fullMask =
        AscendC::MicroAPI::CreateMask<float,
                                      AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg scalarMask =
        AscendC::MicroAPI::CreateMask<float,
                                      AscendC::MicroAPI::MaskPattern::VL1>();

    for (uint16_t head = 0; head < headCount; ++head) {
      const uint32_t offset = static_cast<uint32_t>(head) * C2_PACKED_HEAD_DIM;
      // Both halves must be loaded before the in-place FP16 stores below.
      AscendC::MicroAPI::DataCopy<bfloat16_t,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          srcLo, srcAddr + offset);
      AscendC::MicroAPI::DataCopy<bfloat16_t,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          srcHi, srcAddr + offset + C2_FP32_VECTOR_LENGTH);
      AscendC::MicroAPI::Cast<float, bfloat16_t, C2_CAST_B16_TO_B32>(
          xLo, srcLo, fullMask);
      AscendC::MicroAPI::Cast<float, bfloat16_t, C2_CAST_B16_TO_B32>(
          xHi, srcHi, fullMask);

      // Match the legacy reduction structure: sum(lo^2 + hi^2).
      AscendC::MicroAPI::Mul(squareLo, xLo, xLo, fullMask);
      AscendC::MicroAPI::Mul(squareHi, xHi, xHi, fullMask);
      AscendC::MicroAPI::Add(squareLo, squareLo, squareHi, fullMask);
      AscendC::MicroAPI::ReduceSum(normScalar, squareLo, fullMask);
      AscendC::MicroAPI::Adds(
          normScalar, normScalar, C2_PACKED_NORM_EPSILON, scalarMask);
      AscendC::MicroAPI::Sqrt(normScalar, normScalar, scalarMask);
      AscendC::MicroAPI::Duplicate(squareHi, normScalar, fullMask);
      AscendC::MicroAPI::Div(xLo, xLo, squareHi, fullMask);
      AscendC::MicroAPI::Div(xHi, xHi, squareHi, fullMask);

      // Reuse the BF16 input registers for the mandatory post-norm round.
      AscendC::MicroAPI::Cast<bfloat16_t, float, C2_CAST_B32_TO_B16_RINT>(
          srcLo, xLo, fullMask);
      AscendC::MicroAPI::Cast<bfloat16_t, float, C2_CAST_B32_TO_B16_RINT>(
          srcHi, xHi, fullMask);
      AscendC::MicroAPI::Cast<float, bfloat16_t, C2_CAST_B16_TO_B32>(
          xLo, srcLo, fullMask);
      AscendC::MicroAPI::Cast<float, bfloat16_t, C2_CAST_B16_TO_B32>(
          xHi, srcHi, fullMask);
      AscendC::MicroAPI::Cast<half, float, C2_CAST_B32_TO_B16_RINT>(
          packedLo, xLo, fullMask);
      AscendC::MicroAPI::Cast<half, float, C2_CAST_B32_TO_B16_RINT>(
          packedHi, xHi, fullMask);
      AscendC::MicroAPI::DataCopy<half,
                                  AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
          dstAddr + offset, packedLo, fullMask);
      AscendC::MicroAPI::DataCopy<half,
                                  AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
          dstAddr + offset + C2_FP32_VECTOR_LENGTH, packedHi, fullMask);
    }
  }
}

static __simd_vf__ inline void CastAndPackVRangeC2(__ubuf__ bfloat16_t* srcAddr,
                                                   __ubuf__ half* dstAddr,
                                                   uint32_t dataCount,
                                                   uint16_t loopCount) {
  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<bfloat16_t> src;
    AscendC::MicroAPI::RegTensor<float> expanded;
    AscendC::MicroAPI::RegTensor<half> packed;
    AscendC::MicroAPI::MaskReg loopMask;
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
      loopMask = AscendC::MicroAPI::UpdateMask<float>(dataCount);
      const uint32_t offset =
          static_cast<uint32_t>(loop) * C2_FP32_VECTOR_LENGTH;
      AscendC::MicroAPI::DataCopy<bfloat16_t,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          src, srcAddr + offset);
      AscendC::MicroAPI::Cast<float, bfloat16_t, C2_CAST_B16_TO_B32>(
          expanded, src, loopMask);
      AscendC::MicroAPI::Cast<half, float, C2_CAST_B32_TO_B16_RINT>(
          packed, expanded, loopMask);
      AscendC::MicroAPI::DataCopy<half,
                                  AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
          dstAddr + offset, packed, loopMask);
    }
  }
}

__aicore__ inline void PreparePackedQkvOutputRegbaseC2(
    const AscendC::LocalTensor<bfloat16_t>& outSlot,
    int32_t channelStart,
    int32_t baseDim,
    int32_t qkDim,
    int32_t totalDim) {
  const int32_t channelEnd = channelStart + baseDim;
  const int32_t qkStart = channelStart < qkDim ? channelStart : qkDim;
  const int32_t qkEnd = channelEnd < qkDim ? channelEnd : qkDim;
  __ubuf__ bfloat16_t* srcAddr =
      reinterpret_cast<__ubuf__ bfloat16_t*>(outSlot.GetPhyAddr());
  __ubuf__ half* dstAddr =
      reinterpret_cast<__ubuf__ half*>(outSlot.GetPhyAddr());

  if (qkStart < qkEnd) {
    const int32_t localOffset = qkStart - channelStart;
    const uint16_t headCount = static_cast<uint16_t>(
        (qkEnd - qkStart) / static_cast<int32_t>(C2_PACKED_HEAD_DIM));
    AscendC::VF_CALL<NormalizeAndPackQkHeadsC2>(
        srcAddr + localOffset, dstAddr + localOffset, headCount);
  }

  const int32_t vStart = channelStart > qkDim ? channelStart : qkDim;
  const int32_t vEnd = channelEnd < totalDim ? channelEnd : totalDim;
  if (vStart < vEnd) {
    const int32_t localOffset = vStart - channelStart;
    const uint32_t dataCount = static_cast<uint32_t>(vEnd - vStart);
    const uint16_t loopCount =
        static_cast<uint16_t>(AscendC::Ceil(dataCount, C2_FP32_VECTOR_LENGTH));
    AscendC::VF_CALL<CastAndPackVRangeC2>(
        srcAddr + localOffset, dstAddr + localOffset, dataCount, loopCount);
  }
}

}  // namespace NsCausalConv1d

#endif  // CAUSAL_CONV1D_PACKED_NORM_REGBASE_C2_H
