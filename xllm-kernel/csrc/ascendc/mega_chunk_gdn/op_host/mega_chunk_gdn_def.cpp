/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <vector>

#include "register/op_def_registry.h"

namespace ops {
class XllmKernelMegaChunkGdn : public OpDef {
 public:
  explicit XllmKernelMegaChunkGdn(const char* name) : OpDef(name) {
    const std::vector<ge::DataType> computeTypes = {ge::DT_FLOAT16,
                                                    ge::DT_BF16};
    const std::vector<ge::DataType> floatTypes = {ge::DT_FLOAT, ge::DT_FLOAT};
    const std::vector<ge::DataType> int32Types = {ge::DT_INT32, ge::DT_INT32};
    const std::vector<ge::Format> formats = {ge::FORMAT_ND, ge::FORMAT_ND};

    this->Input("q")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats)
        .AutoContiguous();
    this->Input("k")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats)
        .AutoContiguous();
    this->Input("v")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats)
        .AutoContiguous();
    this->Input("g")
        .ParamType(REQUIRED)
        .DataType(floatTypes)
        .Format(formats)
        .AutoContiguous();
    this->Input("beta")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats)
        .AutoContiguous();
    this->Input("mask_lower")
        .ParamType(REQUIRED)
        .DataType(floatTypes)
        .Format(formats);
    this->Input("mask_full")
        .ParamType(REQUIRED)
        .DataType(floatTypes)
        .Format(formats);
    this->Input("minus_identity")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats);
    this->Input("cu_seqlens")
        .ParamType(REQUIRED)
        .DataType(int32Types)
        .Format(formats)
        .AutoContiguous();
    this->Input("initial_state")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats)
        .AutoContiguous();

    this->Output("out")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats);
    this->Output("g_sum")
        .ParamType(REQUIRED)
        .DataType(floatTypes)
        .Format(formats);
    this->Output("g_t")
        .ParamType(REQUIRED)
        .DataType(floatTypes)
        .Format(formats);
    this->Output("beta_t")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats);
    this->Output("a")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats);
    this->Output("a_inv_f32")
        .ParamType(REQUIRED)
        .DataType(floatTypes)
        .Format(formats);
    this->Output("a_inv")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats);
    this->Output("w")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats);
    this->Output("u")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats);
    this->Output("h")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats);
    this->Output("v_new")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats);
    this->Output("final_state")
        .ParamType(REQUIRED)
        .DataType(computeTypes)
        .Format(formats);

    this->Attr("num_matrices").AttrType(OPTIONAL).Int(0);
    this->Attr("has_initial_state").AttrType(OPTIONAL).Bool(false);
    this->Attr("ffts_addr").AttrType(OPTIONAL).Int(0);

    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
    this->AICore().AddConfig("ascend950");
  }
};

OP_ADD(XllmKernelMegaChunkGdn);
}  // namespace ops
