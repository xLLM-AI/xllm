/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include <vector>

#include "register/op_def_registry.h"

namespace ops {
class XllmKernelMegaGdnPrefillOp : public OpDef {
 public:
  explicit XllmKernelMegaGdnPrefillOp(const char* name) : OpDef(name) {
    const std::vector<ge::DataType> bf16 = {ge::DT_BF16};
    const std::vector<ge::DataType> fp32 = {ge::DT_FLOAT};
    const std::vector<ge::DataType> int32 = {ge::DT_INT32};
    const std::vector<ge::Format> nd = {ge::FORMAT_ND};

    this->Input("mixed_qkv")
        .ParamType(REQUIRED)
        .DataType(bf16)
        .Format(nd)
        .AutoContiguous();
    this->Input("b")
        .ParamType(REQUIRED)
        .DataType(bf16)
        .Format(nd)
        .AutoContiguous();
    this->Input("a")
        .ParamType(REQUIRED)
        .DataType(bf16)
        .Format(nd)
        .AutoContiguous();
    this->Input("z")
        .ParamType(REQUIRED)
        .DataType(bf16)
        .Format(nd)
        .AutoContiguous();
    this->Input("conv_weight")
        .ParamType(REQUIRED)
        .DataType(bf16)
        .Format(nd)
        .AutoContiguous();
    this->Input("conv_state")
        .ParamType(REQUIRED)
        .DataType(bf16)
        .Format(nd)
        .AutoContiguous();
    this->Input("a_log")
        .ParamType(REQUIRED)
        .DataType(fp32)
        .Format(nd)
        .AutoContiguous();
    this->Input("dt_bias")
        .ParamType(REQUIRED)
        .DataType(fp32)
        .Format(nd)
        .AutoContiguous();
    this->Input("conv_state_read_indices")
        .ParamType(REQUIRED)
        .DataType(int32)
        .Format(nd)
        .AutoContiguous();
    this->Input("conv_state_write_indices")
        .ParamType(REQUIRED)
        .DataType(int32)
        .Format(nd)
        .AutoContiguous();
    this->Input("ssm_state_read_indices")
        .ParamType(REQUIRED)
        .DataType(int32)
        .Format(nd)
        .AutoContiguous();
    this->Input("ssm_state_write_indices")
        .ParamType(REQUIRED)
        .DataType(int32)
        .Format(nd)
        .AutoContiguous();
    this->Input("ssm_cache")
        .ParamType(REQUIRED)
        .DataType(fp32)
        .Format(nd)
        .AutoContiguous();
    this->Input("mask_lower").ParamType(REQUIRED).DataType(fp32).Format(nd);
    this->Input("mask_full").ParamType(REQUIRED).DataType(fp32).Format(nd);
    this->Input("minus_identity").ParamType(REQUIRED).DataType(bf16).Format(nd);
    this->Input("cu_seqlens")
        .ParamType(REQUIRED)
        .DataType(int32)
        .Format(nd)
        .AutoContiguous();
    this->Input("norm_weight")
        .ParamType(REQUIRED)
        .DataType(bf16)
        .Format(nd)
        .AutoContiguous();

    this->Output("norm_output").ParamType(REQUIRED).DataType(bf16).Format(nd);
    this->Output("conv_state_out")
        .ParamType(REQUIRED)
        .DataType(bf16)
        .Format(nd);
    this->Output("ssm_cache_out").ParamType(REQUIRED).DataType(fp32).Format(nd);
    this->Attr("ffts_addr").AttrType(REQUIRED).Int();
    this->Attr("num_matrices").AttrType(REQUIRED).Int();

    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
    this->AICore().AddConfig("ascend950");
  }
};

OP_ADD(XllmKernelMegaGdnPrefillOp);
}  // namespace ops
