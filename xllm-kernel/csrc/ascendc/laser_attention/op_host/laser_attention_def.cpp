/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 */

#include "register/op_def_registry.h"

namespace {
constexpr size_t MAX_TOKEN = 2147483647;
}

namespace ops {
class XllmKernelLaserAttention : public OpDef {
 public:
  explicit XllmKernelLaserAttention(const char* name) : OpDef(name) {
    this->Input("query")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("key")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("value")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("atten_mask")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("alibi_mask")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("drop_mask")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_UINT8, ge::DT_UINT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    // （qseqlen，1）
    this->Output("softmax_log_max_sum")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("attention_out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("scale_value").AttrType(REQUIRED).Float();
    this->Attr("head_num").AttrType(REQUIRED).Int();
    this->Attr("input_layout").AttrType(REQUIRED).String();
    this->Attr("keep_prob").AttrType(OPTIONAL).Float(1.0);
    this->Attr("pre_tokens").AttrType(OPTIONAL).Int(MAX_TOKEN);
    this->Attr("next_tokens").AttrType(OPTIONAL).Int(1);
    this->Attr("is_highPrecision").AttrType(OPTIONAL).Bool(true);

    this->AICore().AddConfig("ascend910");
    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
    this->AICore().AddConfig("ascend950");
  }
};

OP_ADD(XllmKernelLaserAttention);
}  // namespace ops
