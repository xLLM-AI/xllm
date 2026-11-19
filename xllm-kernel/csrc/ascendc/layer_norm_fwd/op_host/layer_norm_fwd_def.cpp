/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include <vector>

#include "register/op_def_registry.h"

namespace ops {
namespace {
const std::vector<ge::DataType> kNormDtypes = {ge::DT_FLOAT16,
                                               ge::DT_BF16,
                                               ge::DT_FLOAT};
const std::vector<ge::DataType> kFloatDtypes = {ge::DT_FLOAT,
                                                ge::DT_FLOAT,
                                                ge::DT_FLOAT};
const std::vector<ge::Format> kNdFormats = {ge::FORMAT_ND,
                                            ge::FORMAT_ND,
                                            ge::FORMAT_ND};
}  // namespace

class XllmKernelLayerNormFwd : public OpDef {
 public:
  explicit XllmKernelLayerNormFwd(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType(kNormDtypes)
        .Format(kNdFormats)
        .UnknownShapeFormat(kNdFormats)
        .AutoContiguous();
    this->Input("weight")
        .ParamType(REQUIRED)
        .DataType(kNormDtypes)
        .Format(kNdFormats)
        .UnknownShapeFormat(kNdFormats)
        .AutoContiguous();
    this->Input("bias")
        .ParamType(OPTIONAL)
        .DataType(kNormDtypes)
        .Format(kNdFormats)
        .UnknownShapeFormat(kNdFormats)
        .AutoContiguous();
    this->Input("z")
        .ParamType(OPTIONAL)
        .DataType(kNormDtypes)
        .Format(kNdFormats)
        .UnknownShapeFormat(kNdFormats)
        .AutoContiguous();

    this->Output("y")
        .ParamType(REQUIRED)
        .DataType(kNormDtypes)
        .Format(kNdFormats)
        .UnknownShapeFormat(kNdFormats)
        .AutoContiguous();
    this->Output("mean")
        .ParamType(REQUIRED)
        .DataType(kFloatDtypes)
        .Format(kNdFormats)
        .UnknownShapeFormat(kNdFormats)
        .AutoContiguous();
    this->Output("rstd")
        .ParamType(REQUIRED)
        .DataType(kFloatDtypes)
        .Format(kNdFormats)
        .UnknownShapeFormat(kNdFormats)
        .AutoContiguous();

    this->Attr("eps").AttrType(REQUIRED).Float(1e-6f);
    this->Attr("group_size").AttrType(REQUIRED).Int(-1);
    this->Attr("norm_before_gate").AttrType(REQUIRED).Bool(true);
    this->Attr("is_rms_norm").AttrType(REQUIRED).Bool(false);

    OpAICoreConfig aicoreConfig;
    aicoreConfig.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("coreType.value", "AiCore");
    this->AICore().AddConfig("ascend910b", aicoreConfig);
    this->AICore().AddConfig("ascend910_93", aicoreConfig);
    this->AICore().AddConfig("ascend950", aicoreConfig);
  }
};

OP_ADD(XllmKernelLayerNormFwd);

}  // namespace ops
