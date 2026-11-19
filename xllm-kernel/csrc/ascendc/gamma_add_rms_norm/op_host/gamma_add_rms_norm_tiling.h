/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Copyright 2026 The xLLM Authors. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_GAMMA_ADD_RMS_NORM_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_GAMMA_ADD_RMS_NORM_H_
#include "gamma_add_rms_norm_error_log.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GammaAddRMSNormTilingData)
TILING_DATA_FIELD_DEF(uint32_t, num_row);
TILING_DATA_FIELD_DEF(uint32_t, num_col);
TILING_DATA_FIELD_DEF(uint32_t, block_factor);
TILING_DATA_FIELD_DEF(uint32_t, row_factor);
TILING_DATA_FIELD_DEF(uint32_t, ub_factor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avg_factor);
TILING_DATA_FIELD_DEF(uint32_t, num_col_align);
TILING_DATA_FIELD_DEF(uint32_t, last_block_factor);
TILING_DATA_FIELD_DEF(uint32_t, row_loop);
TILING_DATA_FIELD_DEF(uint32_t, last_block_row_loop);
TILING_DATA_FIELD_DEF(uint32_t, row_tail);
TILING_DATA_FIELD_DEF(uint32_t, last_block_row_tail);
TILING_DATA_FIELD_DEF(uint32_t, mul_loop_fp32);
TILING_DATA_FIELD_DEF(uint32_t, mul_tail_fp32);
TILING_DATA_FIELD_DEF(uint32_t, dst_rep_stride_fp32);
TILING_DATA_FIELD_DEF(uint32_t, mul_loop_fp16);
TILING_DATA_FIELD_DEF(uint32_t, mul_tail_fp16);
TILING_DATA_FIELD_DEF(uint32_t, dst_rep_stride_fp16);
TILING_DATA_FIELD_DEF(uint32_t, is_performance);
TILING_DATA_FIELD_DEF(uint32_t, add_gamma_offset);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(GammaAddRMSNormRegbaseTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numRow);
TILING_DATA_FIELD_DEF(uint32_t, numCol);
TILING_DATA_FIELD_DEF(uint32_t, numColAlign);
TILING_DATA_FIELD_DEF(uint32_t, blockFactor);
TILING_DATA_FIELD_DEF(uint32_t, rowFactor);
TILING_DATA_FIELD_DEF(uint32_t, ubFactor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
TILING_DATA_FIELD_DEF(uint32_t, ubLoop);
TILING_DATA_FIELD_DEF(uint32_t, colBuferLength);
TILING_DATA_FIELD_DEF(uint32_t, multiNNum);
TILING_DATA_FIELD_DEF(uint32_t, isNddma);
TILING_DATA_FIELD_DEF(uint32_t, addGammaOffset);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(GammaAddRMSNormRegbaseRFullLoadTilingData)
TILING_DATA_FIELD_DEF(uint64_t, numRow);
TILING_DATA_FIELD_DEF(uint64_t, numCol);
TILING_DATA_FIELD_DEF(uint64_t, numColAlign);
TILING_DATA_FIELD_DEF(uint64_t, blockFactor);
TILING_DATA_FIELD_DEF(uint64_t, rowFactor);
TILING_DATA_FIELD_DEF(uint64_t, binAddQuotient);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
TILING_DATA_FIELD_DEF(uint32_t, addGammaOffset);
END_TILING_DATA_DEF;

struct GammaAddRmsNormCompileInfo {
  uint32_t totalCoreNum = 0;
  uint64_t totalUbSize = 0;
  platform_ascendc::SocVersion socVersion =
      platform_ascendc::SocVersion::ASCEND910B;
};

namespace gammaAddRmsNormRegbase {
ge::graphStatus TilingGammaAddRmsNormRegbase(gert::TilingContext* context);
}

REGISTER_TILING_DATA_CLASS(XllmKernelGammaAddRmsNorm, GammaAddRMSNormTilingData)
REGISTER_TILING_DATA_CLASS(XllmKernelGammaAddRmsNorm_1000,
                           GammaAddRMSNormRegbaseRFullLoadTilingData)
REGISTER_TILING_DATA_CLASS(XllmKernelGammaAddRmsNorm_2000,
                           GammaAddRMSNormRegbaseTilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_GAMMA_ADD_RMS_NORM_H_
