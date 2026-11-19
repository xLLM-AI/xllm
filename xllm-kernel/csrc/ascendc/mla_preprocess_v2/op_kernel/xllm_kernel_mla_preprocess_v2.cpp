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

/*!
 * \file mal_preprocess.cpp
 * \brief
 */
#include "kernel_operator.h"

#if ORIG_DTYPE_INPUT == DT_FLOAT16

#include "mla_preprocess_fp16.h"

#elif ORIG_DTYPE_INPUT == DT_BF16

#include "mla_preprocess_bf16.h"

#endif

using namespace MlaPreprocess;

extern "C" __global__ __aicore__ void xllm_kernel_mla_preprocess_v2(
    GM_ADDR hiddenStateGm,
    GM_ADDR gamma1Gm,
    GM_ADDR beta1Gm,
    GM_ADDR quantScale1Gm,
    GM_ADDR quantOffset1Gm,
    GM_ADDR wdqkvGm,
    GM_ADDR descale1Gm,
    GM_ADDR bias1Gm,
    GM_ADDR gamma2Gm,
    GM_ADDR beta2Gm,
    GM_ADDR quantScale2Gm,
    GM_ADDR quantOffset2Gm,
    GM_ADDR wuqGm,
    GM_ADDR descale2Gm,
    GM_ADDR bias2Gm,
    GM_ADDR gamma3Gm,
    GM_ADDR cos1Gm,
    GM_ADDR sin1Gm,
    GM_ADDR wukGm,
    GM_ADDR keycacheGm,
    GM_ADDR keycacheRopeGm,
    GM_ADDR slotMappingGm,
    GM_ADDR gmCtkvScale,
    GM_ADDR gmQnopeScale,
    GM_ADDR qGm,
    GM_ADDR keycacheOutGm,
    GM_ADDR qGm2,
    GM_ADDR keycacheOutGm2,
    GM_ADDR qDownGm,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  SetAtomicnone();
  SetMasknorm();
#ifdef __DAV_C220_CUBE__
  SetPadding<uint64_t>((uint64_t)0);
  SetNdpara(1, 0, 0);
#endif
  GET_TILING_DATA(tilingData, tiling);
  auto s1Gm = AscendC::GetUserWorkspace(workspace);
  auto s2Gm = s1Gm + tilingData.maxWorkspaceSize;
  auto s3Gm = s2Gm + tilingData.maxWorkspaceSize;

#if ORIG_DTYPE_INPUT == DT_FLOAT16
  if (TILING_KEY_IS(56)) {
    MLAOperation<CACHE_MODE_KROPE_CTKV,
                 DataFormat::NZ,
                 DataFormat::NZ,
                 DataFormat::ND>
        op(tilingData);
    op.Init(hiddenStateGm,
            gamma1Gm,
            beta1Gm,
            quantScale1Gm,
            quantOffset1Gm,
            wdqkvGm,
            bias1Gm,
            gamma2Gm,
            beta2Gm,
            quantScale2Gm,
            quantOffset2Gm,
            gamma3Gm,
            sin1Gm,
            cos1Gm,
            sin1Gm,
            cos1Gm,
            keycacheGm,
            slotMappingGm,
            wuqGm,
            bias2Gm,
            wukGm,
            descale1Gm,
            descale2Gm,
            gmCtkvScale,
            gmQnopeScale,
            qGm,
            keycacheOutGm,
            qGm2,
            keycacheOutGm2,
            s1Gm,
            s2Gm,
            s3Gm,
            qDownGm);
    op.ProcessCube();
    op.ProcessVector();
  } else if (TILING_KEY_IS(60)) {
    MLAOperation<CACHE_MODE_KROPE_CTKV,
                 DataFormat::NZ,
                 DataFormat::NZ,
                 DataFormat::NZ>
        op(tilingData);
    op.Init(hiddenStateGm,
            gamma1Gm,
            beta1Gm,
            quantScale1Gm,
            quantOffset1Gm,
            wdqkvGm,
            bias1Gm,
            gamma2Gm,
            beta2Gm,
            quantScale2Gm,
            quantOffset2Gm,
            gamma3Gm,
            sin1Gm,
            cos1Gm,
            sin1Gm,
            cos1Gm,
            keycacheGm,
            slotMappingGm,
            wuqGm,
            bias2Gm,
            wukGm,
            descale1Gm,
            descale2Gm,
            gmCtkvScale,
            gmQnopeScale,
            qGm,
            keycacheOutGm,
            qGm2,
            keycacheOutGm2,
            s1Gm,
            s2Gm,
            s3Gm,
            qDownGm);
    op.ProcessCube();
    op.ProcessVector();
  } else if (TILING_KEY_IS(120)) {
    MLAOperation<CACHE_MODE_NZCACHE,
                 DataFormat::NZ,
                 DataFormat::NZ,
                 DataFormat::ND>
        op(tilingData);
    op.Init(hiddenStateGm,
            gamma1Gm,
            beta1Gm,
            quantScale1Gm,
            quantOffset1Gm,
            wdqkvGm,
            bias1Gm,
            gamma2Gm,
            beta2Gm,
            quantScale2Gm,
            quantOffset2Gm,
            gamma3Gm,
            sin1Gm,
            cos1Gm,
            sin1Gm,
            cos1Gm,
            keycacheGm,
            slotMappingGm,
            wuqGm,
            bias2Gm,
            wukGm,
            descale1Gm,
            descale2Gm,
            gmCtkvScale,
            gmQnopeScale,
            qGm,
            keycacheOutGm,
            qGm2,
            keycacheOutGm2,
            s1Gm,
            s2Gm,
            s3Gm,
            qDownGm);
    op.ProcessCube();
    op.ProcessVector();
  } else if (TILING_KEY_IS(124)) {
    MLAOperation<CACHE_MODE_NZCACHE,
                 DataFormat::NZ,
                 DataFormat::NZ,
                 DataFormat::NZ>
        op(tilingData);
    op.Init(hiddenStateGm,
            gamma1Gm,
            beta1Gm,
            quantScale1Gm,
            quantOffset1Gm,
            wdqkvGm,
            bias1Gm,
            gamma2Gm,
            beta2Gm,
            quantScale2Gm,
            quantOffset2Gm,
            gamma3Gm,
            sin1Gm,
            cos1Gm,
            sin1Gm,
            cos1Gm,
            keycacheGm,
            slotMappingGm,
            wuqGm,
            bias2Gm,
            wukGm,
            descale1Gm,
            descale2Gm,
            gmCtkvScale,
            gmQnopeScale,
            qGm,
            keycacheOutGm,
            qGm2,
            keycacheOutGm2,
            s1Gm,
            s2Gm,
            s3Gm,
            qDownGm);
    op.ProcessCube();
    op.ProcessVector();
  }

#elif ORIG_DTYPE_INPUT == DT_BF16
  PRELOAD(2);
  auto s4Gm = s3Gm + tilingData.maxWorkspaceSize;
  auto s5Gm = s4Gm + tilingData.maxWorkspaceSize;
  // InDtype[1] cacheMode[2] weightFormat1[1] weightFormat2[1] weightFormat3[1]
  // quantMode[2]
  if (TILING_KEY_IS(184)) {
    MLAOperation<bfloat16_t,
                 1,
                 DataFormat::NZ,
                 DataFormat::NZ,
                 DataFormat::ND,
                 QuantMode::PER_TENSOR_ASYMM_QUANT>
        op(tilingData);
    op.Init(hiddenStateGm,
            gamma1Gm,
            beta1Gm,
            quantScale1Gm,
            quantOffset1Gm,
            wdqkvGm,
            bias1Gm,
            gamma2Gm,
            beta2Gm,
            quantScale2Gm,
            quantOffset2Gm,
            gamma3Gm,
            sin1Gm,
            cos1Gm,
            sin1Gm,
            cos1Gm,
            keycacheGm,
            slotMappingGm,
            wuqGm,
            bias2Gm,
            wukGm,
            descale1Gm,
            descale2Gm,
            gmCtkvScale,
            gmQnopeScale,
            qGm,
            keycacheOutGm,
            qGm2,
            keycacheOutGm2,
            s1Gm,
            s2Gm,
            s3Gm,
            s4Gm,
            s5Gm,
            qDownGm);
    op.ProcessCube();
    op.ProcessVector();
  } else if (TILING_KEY_IS(188)) {
    MLAOperation<bfloat16_t,
                 1,
                 DataFormat::NZ,
                 DataFormat::NZ,
                 DataFormat::NZ,
                 QuantMode::PER_TENSOR_ASYMM_QUANT>
        op(tilingData);
    op.Init(hiddenStateGm,
            gamma1Gm,
            beta1Gm,
            quantScale1Gm,
            quantOffset1Gm,
            wdqkvGm,
            bias1Gm,
            gamma2Gm,
            beta2Gm,
            quantScale2Gm,
            quantOffset2Gm,
            gamma3Gm,
            sin1Gm,
            cos1Gm,
            sin1Gm,
            cos1Gm,
            keycacheGm,
            slotMappingGm,
            wuqGm,
            bias2Gm,
            wukGm,
            descale1Gm,
            descale2Gm,
            gmCtkvScale,
            gmQnopeScale,
            qGm,
            keycacheOutGm,
            qGm2,
            keycacheOutGm2,
            s1Gm,
            s2Gm,
            s3Gm,
            s4Gm,
            s5Gm,
            qDownGm);
    op.ProcessCube();
    op.ProcessVector();
  } else if (TILING_KEY_IS(248)) {
    MLAOperation<bfloat16_t,
                 3,
                 DataFormat::NZ,
                 DataFormat::NZ,
                 DataFormat::ND,
                 QuantMode::PER_TENSOR_ASYMM_QUANT>
        op(tilingData);
    op.Init(hiddenStateGm,
            gamma1Gm,
            beta1Gm,
            quantScale1Gm,
            quantOffset1Gm,
            wdqkvGm,
            bias1Gm,
            gamma2Gm,
            beta2Gm,
            quantScale2Gm,
            quantOffset2Gm,
            gamma3Gm,
            sin1Gm,
            cos1Gm,
            sin1Gm,
            cos1Gm,
            keycacheGm,
            slotMappingGm,
            wuqGm,
            bias2Gm,
            wukGm,
            descale1Gm,
            descale2Gm,
            gmCtkvScale,
            gmQnopeScale,
            qGm,
            keycacheOutGm,
            qGm2,
            keycacheOutGm2,
            s1Gm,
            s2Gm,
            s3Gm,
            s4Gm,
            s5Gm,
            qDownGm);
    op.ProcessCube();
    op.ProcessVector();
  } else if (TILING_KEY_IS(252)) {
    MLAOperation<bfloat16_t,
                 3,
                 DataFormat::NZ,
                 DataFormat::NZ,
                 DataFormat::NZ,
                 QuantMode::PER_TENSOR_ASYMM_QUANT>
        op(tilingData);
    op.Init(hiddenStateGm,
            gamma1Gm,
            beta1Gm,
            quantScale1Gm,
            quantOffset1Gm,
            wdqkvGm,
            bias1Gm,
            gamma2Gm,
            beta2Gm,
            quantScale2Gm,
            quantOffset2Gm,
            gamma3Gm,
            sin1Gm,
            cos1Gm,
            sin1Gm,
            cos1Gm,
            keycacheGm,
            slotMappingGm,
            wuqGm,
            bias2Gm,
            wukGm,
            descale1Gm,
            descale2Gm,
            gmCtkvScale,
            gmQnopeScale,
            qGm,
            keycacheOutGm,
            qGm2,
            keycacheOutGm2,
            s1Gm,
            s2Gm,
            s3Gm,
            s4Gm,
            s5Gm,
            qDownGm);
    op.ProcessCube();
    op.ProcessVector();
  }
#endif
}
