/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "register/op_def_registry.h"

namespace ops {
class XllmKernelReshapeAndCacheA5 : public OpDef {
 public:
  explicit XllmKernelReshapeAndCacheA5(const char* name) : OpDef(name) {
    const std::initializer_list<ge::DataType> cache_types = {ge::DT_FLOAT16,
                                                             ge::DT_BF16};
    const std::initializer_list<ge::Format> cache_formats = {ge::FORMAT_ND,
                                                             ge::FORMAT_ND};

    this->Input("key")
        .ParamType(REQUIRED)
        .DataType(cache_types)
        .Format(cache_formats)
        .UnknownShapeFormat(cache_formats)
        .AutoContiguous();
    this->Input("value")
        .ParamType(REQUIRED)
        .DataType(cache_types)
        .Format(cache_formats)
        .UnknownShapeFormat(cache_formats)
        .AutoContiguous();
    this->Input("keyCache")
        .ParamType(REQUIRED)
        .DataType(cache_types)
        .Format(cache_formats)
        .UnknownShapeFormat(cache_formats);
    this->Input("valueCache")
        .ParamType(REQUIRED)
        .DataType(cache_types)
        .Format(cache_formats)
        .UnknownShapeFormat(cache_formats);
    this->Input("slotMapping")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32})
        .Format(cache_formats)
        .UnknownShapeFormat(cache_formats)
        .AutoContiguous();
    this->Output("keyCacheOut")
        .ParamType(REQUIRED)
        .DataType(cache_types)
        .Format(cache_formats)
        .UnknownShapeFormat(cache_formats);
    this->Output("valueCacheOut")
        .ParamType(REQUIRED)
        .DataType(cache_types)
        .Format(cache_formats)
        .UnknownShapeFormat(cache_formats);

    OpAICoreConfig a5_config;
    a5_config.DynamicCompileStaticFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .ExtendCfgInfo("opFile.value", "reshape_and_cache_a5")
        .ExtendCfgInfo("opInterface.value", "reshape_and_cache_a5");
    this->AICore().AddConfig("ascend950", a5_config);
  }
};

OP_ADD(XllmKernelReshapeAndCacheA5);
}  // namespace ops
