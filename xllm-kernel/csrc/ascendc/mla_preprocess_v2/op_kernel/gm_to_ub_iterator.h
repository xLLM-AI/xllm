/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
the terms and conditions of  * CANN Open Software License Agreement Version 2.0
(the "License").  * Please refer to the License for details. You may not use
this file except in compliance with the License.  * THIS SOFTWARE IS PROVIDED ON
AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,  *
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE.  * See LICENSE in the root of the software repository for
the full text of the License.  */
#ifndef MLA_PREPROCESS_GM_TO_UB_ITERATOR_H
#define MLA_PREPROCESS_GM_TO_UB_ITERATOR_H

template <Architecture ArchTag, typename DType>
struct gm_to_ub {
  __aicore__ inline gm_to_ub(AscendC::LocalTensor<DType> dstTensor,
                             AscendC::GlobalTensor<DType> srcTensor,
                             uint8_t sid,
                             uint16_t nBurst,
                             uint16_t lenBurst,
                             uint16_t srcStride,
                             uint16_t dstStride) {
    AscendC::DataCopy(
        dstTensor,
        srcTensor,
        AscendC::DataCopyParams(nBurst, lenBurst, srcStride, dstStride));
  };
};

template <Architecture ArchTag, typename DType>
struct gm_to_ub_align {
  __aicore__ inline gm_to_ub_align(AscendC::LocalTensor<DType> dstTensor,
                                   AscendC::GlobalTensor<DType> srcTensor,
                                   uint8_t sid,
                                   uint16_t nBurst,
                                   uint32_t lenBurst,
                                   uint8_t leftPaddingNum,
                                   uint8_t rightPaddingNum,
                                   uint32_t srcGap,
                                   uint32_t dstGap) {
    AscendC::DataCopyPad(
        dstTensor,
        srcTensor,
        AscendC::DataCopyExtParams(nBurst, lenBurst, srcGap, dstGap, 0),
        AscendC::DataCopyPadExtParams<DType>(
            false, leftPaddingNum, rightPaddingNum, 0));
  };
};

template <Architecture ArchTag, typename DType>
struct ub_to_ub {
  __aicore__ inline ub_to_ub(AscendC::LocalTensor<DType> dstTensor,
                             AscendC::LocalTensor<DType> srcTensor,
                             uint8_t sid,
                             uint16_t nBurst,
                             uint16_t lenBurst,
                             uint16_t srcStride,
                             uint16_t dstStride) {
    AscendC::DataCopy(
        dstTensor,
        srcTensor,
        AscendC::DataCopyParams(nBurst, lenBurst, srcStride, dstStride));
  };
};

template <Architecture ArchTag,
          typename DataType,
          DataFormat InDataFormat = DataFormat::ND,
          DataFormat OutDataFormat = DataFormat::ND>
struct ub_to_gm {
  __aicore__ inline ub_to_gm(AscendC::GlobalTensor<DataType> dstTensor,
                             AscendC::LocalTensor<DataType> srcTensor,
                             uint8_t sid,
                             uint16_t nBurst,
                             uint16_t lenBurst,
                             uint16_t srcStride,
                             uint16_t dstStride) {
    AscendC::DataCopy(
        dstTensor,
        srcTensor,
        AscendC::DataCopyParams(nBurst, lenBurst, srcStride, dstStride));
  };
};

template <Architecture ArchTag, typename DataType>
struct ub_to_gm<ArchTag, DataType, DataFormat::NZ, DataFormat::NZ> {
  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t BLOCK_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(DataType);

  __aicore__ ub_to_gm(AscendC::GlobalTensor<DataType> gmTensor,
                      AscendC::LocalTensor<DataType> ubTensor,
                      uint32_t nTileActual,
                      uint32_t nTileCeil,
                      uint32_t nVal,
                      uint32_t dTileActual,
                      uint32_t dTileCeil,
                      uint32_t dVal) {
    constexpr uint32_t STRIDE_LIMIT = 65536;
    uint64_t dstStride = nVal - nTileCeil;
    if (dstStride < STRIDE_LIMIT) {
      AscendC::DataCopy(
          gmTensor,                                        // dst
          ubTensor,                                        // src
          AscendC::DataCopyParams(dTileCeil / BLOCK_SIZE,  // nBurst
                                  nTileCeil,               // lenBurst
                                  0,                       // srcGap
                                  dstStride));             // dstGap
    } else {
      for (uint64_t i = 0; i < dTileCeil / BLOCK_SIZE; ++i) {
        uint64_t dstOffset = i * nVal * BLOCK_SIZE;
        uint64_t srcOffset = i * nTileCeil * BLOCK_SIZE;
        AscendC::DataCopy(gmTensor[dstOffset],                // dst
                          ubTensor[srcOffset],                // src
                          AscendC::DataCopyParams(1,          // nBurst
                                                  nTileCeil,  // lenBurst
                                                  0,          // srcGap
                                                  0));        // dstGap
      }
    }
  };
};

template <Architecture ArchTag, typename DType>
struct ub_to_gm_align {
  __aicore__ inline ub_to_gm_align(AscendC::GlobalTensor<DType> dstTensor,
                                   AscendC::LocalTensor<DType> srcTensor,
                                   uint8_t sid,
                                   uint16_t nBurst,
                                   uint32_t lenBurst,
                                   uint8_t leftPaddingNum,
                                   uint8_t rightPaddingNum,
                                   uint32_t srcGap,
                                   uint32_t dstGap) {
    AscendC::DataCopyPad(
        dstTensor,
        srcTensor,
        AscendC::DataCopyExtParams(nBurst, lenBurst, srcGap, dstGap, 0));
  };
};

#endif  // XLLM_OPS_MLA_PREPROCESS_OP_KERNEL_GM_TO_UB_ITERATOR_H_