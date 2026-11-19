#include <cstdint>

#include "register/op_def_registry.h"

namespace ops {

class XllmKernelCausalConv1dQkv : public OpDef {
 public:
  explicit XllmKernelCausalConv1dQkv(const char* name) : OpDef(name) {
    constexpr int64_t kPackedQkvActivation = 2;
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("weight")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("bias")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_BF16, ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("convStates")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("queryStartLoc")
        .ParamType(OPTIONAL)
        .DataTypeList({ge::DT_INT64})
        .FormatList({ge::FORMAT_ND})
        .ValueDepend(OPTIONAL)
        .AutoContiguous();
    this->Input("cacheIndices")
        .ParamType(OPTIONAL)
        .DataTypeList({ge::DT_INT64})
        .FormatList({ge::FORMAT_ND})
        .ValueDepend(OPTIONAL)
        .AutoContiguous();
    this->Input("initialStateMode")
        .ParamType(OPTIONAL)
        .DataTypeList({ge::DT_INT64})
        .FormatList({ge::FORMAT_ND})
        .ValueDepend(OPTIONAL)
        .AutoContiguous();
    this->Input("numAcceptedTokens")
        .ParamType(OPTIONAL)
        .DataTypeList({ge::DT_INT64})
        .FormatList({ge::FORMAT_ND})
        .ValueDepend(OPTIONAL)
        .AutoContiguous();
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();

    this->Attr("activationMode").AttrType(OPTIONAL).Int(kPackedQkvActivation);
    this->Attr("padSlotId").AttrType(OPTIONAL).Int(-1);
    this->Attr("runMode").AttrType(OPTIONAL).Int(0);
    this->Attr("qDim").AttrType(OPTIONAL).Int(1024);
    this->Attr("kDim").AttrType(OPTIONAL).Int(1024);
    this->Attr("vDim").AttrType(OPTIONAL).Int(3072);
    this->Attr("headDim").AttrType(OPTIONAL).Int(128);

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
OP_ADD(XllmKernelCausalConv1dQkv);

}  // namespace ops
