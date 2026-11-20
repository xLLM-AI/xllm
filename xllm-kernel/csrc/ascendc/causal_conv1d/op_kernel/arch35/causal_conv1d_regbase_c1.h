/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef CAUSAL_CONV1D_REGBASE_C1_H
#define CAUSAL_CONV1D_REGBASE_C1_H

namespace NsCausalConv1d {

constexpr uint16_t C1_VECTOR_LENGTH = AscendC::VECTOR_REG_WIDTH / sizeof(float);
constexpr AscendC::MicroAPI::CastTrait C1_CAST_B16_TO_B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

#if defined(MEGA_GDN_PREFILL_A5_CONV_FUSED_ROLL)
constexpr AscendC::MicroAPI::CastTrait C1_CAST_B32_TO_B16_RINT = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <typename OutputT>
static __simd_vf__ inline void RunPackedBf16Width4FusedRollImpl(
    __ubuf__ bfloat16_t* ringAddr,
    __ubuf__ float* weightAddr,
    __ubuf__ float* state0Addr,
    __ubuf__ float* state1Addr,
    __ubuf__ float* state2Addr,
    __ubuf__ OutputT* outAddr,
    uint32_t dataCount,
    uint16_t colLoopTimes,
    uint32_t weightStep) {
  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<bfloat16_t> ringReg;
    AscendC::MicroAPI::RegTensor<float> currFReg;
    AscendC::MicroAPI::RegTensor<float> old1FReg;
    AscendC::MicroAPI::RegTensor<float> old2FReg;
    AscendC::MicroAPI::RegTensor<float> weight0FReg;
    AscendC::MicroAPI::RegTensor<float> weight1FReg;
    AscendC::MicroAPI::RegTensor<float> weight2FReg;
    AscendC::MicroAPI::RegTensor<float> weight3FReg;
    AscendC::MicroAPI::RegTensor<float> outputFReg;
    AscendC::MicroAPI::RegTensor<float> siluTmpReg;
    AscendC::MicroAPI::RegTensor<float> next0FReg;
    AscendC::MicroAPI::RegTensor<float> next1FReg;
    AscendC::MicroAPI::RegTensor<float> next2FReg;
    AscendC::MicroAPI::RegTensor<bfloat16_t> outputBf16Reg;
    AscendC::MicroAPI::MaskReg mask;
    for (uint16_t j = 0; j < colLoopTimes; ++j) {
      mask = AscendC::MicroAPI::UpdateMask<float>(dataCount);
      const uint32_t offset = static_cast<uint32_t>(j) * C1_VECTOR_LENGTH;
      AscendC::MicroAPI::DataCopy<bfloat16_t,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          ringReg, ringAddr + offset);
      AscendC::MicroAPI::DataCopy(outputFReg, state0Addr + offset);
      AscendC::MicroAPI::DataCopy(old1FReg, state1Addr + offset);
      AscendC::MicroAPI::DataCopy(old2FReg, state2Addr + offset);
      AscendC::MicroAPI::Cast<float, bfloat16_t, C1_CAST_B16_TO_B32>(
          currFReg, ringReg, mask);
      AscendC::MicroAPI::DataCopy(weight3FReg,
                                  weightAddr + 3 * weightStep + offset);
      AscendC::MicroAPI::MulAddDst(outputFReg, currFReg, weight3FReg, mask);
      AscendC::MicroAPI::Muls(siluTmpReg, outputFReg, -1.0f, mask);
      AscendC::MicroAPI::Exp(siluTmpReg, siluTmpReg, mask);
      AscendC::MicroAPI::Adds(siluTmpReg, siluTmpReg, 1.0f, mask);
      AscendC::MicroAPI::Div(outputFReg, outputFReg, siluTmpReg, mask);
      if constexpr (std::is_same_v<OutputT, bfloat16_t>) {
        AscendC::MicroAPI::Cast<bfloat16_t, float, C1_CAST_B32_TO_B16_RINT>(
            outputBf16Reg, outputFReg, mask);
        AscendC::MicroAPI::
            DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                outAddr + offset, outputBf16Reg, mask);
      } else {
        AscendC::MicroAPI::DataCopy(outAddr + offset, outputFReg, mask);
      }

      AscendC::MicroAPI::DataCopy(weight2FReg,
                                  weightAddr + 2 * weightStep + offset);
      AscendC::MicroAPI::Mul(next0FReg, currFReg, weight2FReg, mask);
      AscendC::MicroAPI::Add(next0FReg, next0FReg, old1FReg, mask);
      AscendC::MicroAPI::DataCopy(weight1FReg,
                                  weightAddr + weightStep + offset);
      AscendC::MicroAPI::Mul(next1FReg, currFReg, weight1FReg, mask);
      AscendC::MicroAPI::Add(next1FReg, next1FReg, old2FReg, mask);
      AscendC::MicroAPI::DataCopy(weight0FReg, weightAddr + offset);
      AscendC::MicroAPI::Mul(next2FReg, currFReg, weight0FReg, mask);
      AscendC::MicroAPI::DataCopy(state0Addr + offset, next0FReg, mask);
      AscendC::MicroAPI::DataCopy(state1Addr + offset, next1FReg, mask);
      AscendC::MicroAPI::DataCopy(state2Addr + offset, next2FReg, mask);
    }
  }
}

template <typename OutputT>
__aicore__ inline void RunPackedBf16Width4FusedRoll(
    AscendC::LocalTensor<bfloat16_t> ring,
    AscendC::LocalTensor<float> weightF,
    AscendC::LocalTensor<float> state0F,
    AscendC::LocalTensor<float> state1F,
    AscendC::LocalTensor<float> state2F,
    AscendC::LocalTensor<OutputT> out,
    uint32_t dataCount,
    uint32_t weightStep) {
  const uint16_t colLoopTimes =
      static_cast<uint16_t>(AscendC::Ceil(dataCount, C1_VECTOR_LENGTH));
  AscendC::VF_CALL<RunPackedBf16Width4FusedRollImpl<OutputT>>(
      (__ubuf__ bfloat16_t*)ring.GetPhyAddr(),
      (__ubuf__ float*)weightF.GetPhyAddr(),
      (__ubuf__ float*)state0F.GetPhyAddr(),
      (__ubuf__ float*)state1F.GetPhyAddr(),
      (__ubuf__ float*)state2F.GetPhyAddr(),
      (__ubuf__ OutputT*)out.GetPhyAddr(),
      dataCount,
      colLoopTimes,
      weightStep);
}
#endif

template <typename T>
static __simd_vf__ inline void AdvanceFnLocalPartialsRegbaseC1Impl(
    __ubuf__ T* ringAddr,
    __ubuf__ float* weight0FAddr,
    __ubuf__ float* weight1FAddr,
    __ubuf__ float* weight2FAddr,
    __ubuf__ float* state0FAddr,
    __ubuf__ float* state1FAddr,
    __ubuf__ float* state2FAddr,
    uint32_t dataCount,
    uint16_t colLoopTimes) {
  AscendC::MicroAPI::RegTensor<T> ringReg;
  AscendC::MicroAPI::RegTensor<float> currFReg;
  AscendC::MicroAPI::RegTensor<float> weight0FReg;
  AscendC::MicroAPI::RegTensor<float> weight1FReg;
  AscendC::MicroAPI::RegTensor<float> weight2FReg;
  AscendC::MicroAPI::RegTensor<float> state0FReg;
  AscendC::MicroAPI::RegTensor<float> state1FReg;
  AscendC::MicroAPI::RegTensor<float> state2FReg;
  AscendC::MicroAPI::MaskReg pregLoop;
  for (uint16_t j = 0; j < colLoopTimes; ++j) {
    pregLoop = AscendC::MicroAPI::UpdateMask<float>(dataCount);
    AscendC::MicroAPI::DataCopy<T,
                                AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
        ringReg, ringAddr + j * C1_VECTOR_LENGTH);
    AscendC::MicroAPI::DataCopy(state1FReg, state1FAddr + j * C1_VECTOR_LENGTH);
    AscendC::MicroAPI::DataCopy(state2FReg, state2FAddr + j * C1_VECTOR_LENGTH);
    AscendC::MicroAPI::Cast<float, T, C1_CAST_B16_TO_B32>(
        currFReg, ringReg, pregLoop);
    AscendC::MicroAPI::DataCopy(weight2FReg,
                                weight2FAddr + j * C1_VECTOR_LENGTH);
    AscendC::MicroAPI::Mul(state0FReg, currFReg, weight2FReg, pregLoop);
    AscendC::MicroAPI::DataCopy(weight1FReg,
                                weight1FAddr + j * C1_VECTOR_LENGTH);
    AscendC::MicroAPI::Add(state0FReg, state0FReg, state1FReg, pregLoop);
    AscendC::MicroAPI::Mul(state1FReg, currFReg, weight1FReg, pregLoop);
    AscendC::MicroAPI::DataCopy(weight0FReg,
                                weight0FAddr + j * C1_VECTOR_LENGTH);
    AscendC::MicroAPI::Add(state1FReg, state1FReg, state2FReg, pregLoop);
    AscendC::MicroAPI::Mul(state2FReg, currFReg, weight0FReg, pregLoop);
    AscendC::MicroAPI::DataCopy(
        state0FAddr + j * C1_VECTOR_LENGTH, state0FReg, pregLoop);
    AscendC::MicroAPI::DataCopy(
        state1FAddr + j * C1_VECTOR_LENGTH, state1FReg, pregLoop);
    AscendC::MicroAPI::DataCopy(
        state2FAddr + j * C1_VECTOR_LENGTH, state2FReg, pregLoop);
  }
}

template <typename T>
__aicore__ inline void AdvanceFnLocalPartialsRegbaseC1(
    AscendC::LocalTensor<T> ring,
    AscendC::LocalTensor<float> weightF,
    AscendC::LocalTensor<float> state0F,
    AscendC::LocalTensor<float> state1F,
    AscendC::LocalTensor<float> state2F,
    uint32_t dataCount,
    uint32_t weightStep) {
  uint16_t colLoopTimes =
      static_cast<uint16_t>(AscendC::Ceil(dataCount, C1_VECTOR_LENGTH));
  __ubuf__ T* ringAddr = (__ubuf__ T*)ring.GetPhyAddr();
  __ubuf__ float* weight0FAddr = (__ubuf__ float*)weightF.GetPhyAddr();
  __ubuf__ float* weight1FAddr = weight0FAddr + weightStep;
  __ubuf__ float* weight2FAddr = weight1FAddr + weightStep;
  __ubuf__ float* state0FAddr = (__ubuf__ float*)state0F.GetPhyAddr();
  __ubuf__ float* state1FAddr = (__ubuf__ float*)state1F.GetPhyAddr();
  __ubuf__ float* state2FAddr = (__ubuf__ float*)state2F.GetPhyAddr();
  AscendC::VF_CALL<AdvanceFnLocalPartialsRegbaseC1Impl<T>>(ringAddr,
                                                           weight0FAddr,
                                                           weight1FAddr,
                                                           weight2FAddr,
                                                           state0FAddr,
                                                           state1FAddr,
                                                           state2FAddr,
                                                           dataCount,
                                                           colLoopTimes);
}

}  // namespace NsCausalConv1d

#endif  // CAUSAL_CONV1D_REGBASE_C1_H
