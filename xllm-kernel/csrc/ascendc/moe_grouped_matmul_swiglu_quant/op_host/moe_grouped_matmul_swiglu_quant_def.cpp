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
 * \file grouped_matmul_swiglu_quant_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"
namespace ops {
class XllmKernelMoeGroupedMatmulSwigluQuant : public OpDef {
 public:
  explicit XllmKernelMoeGroupedMatmulSwigluQuant(const char* name)
      : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("weight")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
        .Format({ge::FORMAT_FRACTAL_NZ,
                 ge::FORMAT_FRACTAL_NZ,
                 ge::FORMAT_FRACTAL_NZ});
    this->Input("weight_scale")
        .ParamType(REQUIRED)
        .DataTypeList({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("x_scale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("group_list")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y_scale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("clamp_limit").AttrType(OPTIONAL).Float(0.0F);
    this->Attr("use_split_arithmetic").AttrType(OPTIONAL).Bool(false);
    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true);

    this->AICore().AddConfig("ascend910b", aicore_config);
    this->AICore().AddConfig("ascend910_93", aicore_config);
    this->AICore().AddConfig("ascend950", aicore_config);
  }
};

OP_ADD(XllmKernelMoeGroupedMatmulSwigluQuant);
}  // namespace ops
