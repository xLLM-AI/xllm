/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "register/op_def_registry.h"

namespace ops {

class XllmKernelMegaGdnMtpDecode : public OpDef {
 public:
  explicit XllmKernelMegaGdnMtpDecode(const char* name) : OpDef(name) {
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
    this->Input("readStateIndices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("writeStateIndices")
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
    this->AICore().AddConfig("ascend950", config);
  }
};

OP_ADD(XllmKernelMegaGdnMtpDecode);

}  // namespace ops
