/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "register/op_def_registry.h"

namespace ops {

class XllmKernelMegaGdnMtpDecodeVarlen final : public OpDef {
 public:
  explicit XllmKernelMegaGdnMtpDecodeVarlen(const char* name) : OpDef(name) {
    this->Input("qkv")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("z")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("b")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("a")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("convWeight")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("convState")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("a_log")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("dtBias")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("ssmState")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("stateIndices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("numAcceptedTokens")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("normWeight")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("cuSeqLens")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Attr("flaSsmStateLayout").AttrType(OPTIONAL).Bool(true);
    this->Output("convOut")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("convStateOut")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("ssmStateOut")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();

    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(false)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(false)
        .ExtendCfgInfo("coreType.value", "AiCore");
    this->AICore().AddConfig("ascend910b", config);
    this->AICore().AddConfig("ascend910_93", config);
  }
};

OP_ADD(XllmKernelMegaGdnMtpDecodeVarlen);

}  // namespace ops
