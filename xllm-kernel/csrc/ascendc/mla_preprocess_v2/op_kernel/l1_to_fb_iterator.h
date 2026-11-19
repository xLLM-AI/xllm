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
#ifndef XLLM_OPS_MLA_PREPROCESS_OP_KERNEL_L1_TO_FB_ITERATOR_H_
#define XLLM_OPS_MLA_PREPROCESS_OP_KERNEL_L1_TO_FB_ITERATOR_H_

/////////////////////////////////////////////////////
// l1_to_fb
/////////////////////////////////////////////////////

// Partial specialization for V220
template <typename DataType>
struct l1_to_fb<Architecture::ASCEND_V220, DataType> {
  __aicore__ l1_to_fb(AscendC::LocalTensor<DataType>& dst,
                      AscendC::LocalTensor<DataType>& src,
                      uint16_t burstNum,
                      uint16_t burstLen,
                      uint16_t srcGap,
                      uint16_t dstGap) {
    dst.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2PIPE2GM);
    AscendC::DataCopy(dst,
                      src,
                      AscendC::DataCopyParams(burstNum,  // nBurst
                                              burstLen,  // lenBurst
                                              srcGap,    // srcGap
                                              dstGap));  // dstGap);
  }
};

#endif  // XLLM_OPS_MLA_PREPROCESS_OP_KERNEL_L1_TO_FB_ITERATOR_H_