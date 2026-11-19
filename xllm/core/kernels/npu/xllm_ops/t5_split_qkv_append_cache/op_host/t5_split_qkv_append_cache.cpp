/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "common/t5_plan.h"
#include "op_host/t5_split_qkv_append_cache_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
template <class Shape>
bool shapes_ok(const Shape& q,
               const Shape& k,
               const Shape& v,
               int64_t h_attr,
               int64_t d_attr) {
  if (q.GetDimNum() != 3 || k.GetDimNum() != 4 || v.GetDimNum() != 4) {
    return false;
  }
  for (size_t i = 0; i < 4; ++i) {
    if (k.GetDim(i) != v.GetDim(i)) {
      return false;
    }
  }
  const int64_t b = k.GetDim(0), h = k.GetDim(1), l = k.GetDim(2),
                d = k.GetDim(3);
  return t5qkv::valid_shape(b, h, l, d) && h == h_attr && d == d_attr &&
         q.GetDim(0) == b && q.GetDim(1) == 1 && q.GetDim(2) == 3 * h * d;
}
}  // namespace
namespace optiling {
namespace {
ge::graphStatus tiling(gert::TilingContext* ctx) {
  const auto* qs = ctx->GetInputShape(0);
  const auto* ks = ctx->GetInputShape(1);
  const auto* vs = ctx->GetInputShape(2);
  const auto* attrs = ctx->GetAttrs();
  if (!qs || !ks || !vs || !attrs) {
    return ge::GRAPH_FAILED;
  }
  const auto* h = attrs->GetAttrPointer<int64_t>(0);
  const auto* d = attrs->GetAttrPointer<int64_t>(1);
  if (!h || !d ||
      !shapes_ok(qs->GetStorageShape(),
                 ks->GetStorageShape(),
                 vs->GetStorageShape(),
                 *h,
                 *d)) {
    return ge::GRAPH_FAILED;
  }
  ge::DataType dtype = ctx->GetInputDesc(0)->GetDataType();
  if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) {
    return ge::GRAPH_FAILED;
  }
  for (int32_t i = 0; i < 3; ++i) {
    if (ctx->GetInputDesc(i)->GetDataType() != dtype ||
        ctx->GetInputDesc(i)->GetStorageFormat() != ge::FORMAT_ND ||
        ctx->GetOutputDesc(i)->GetDataType() != dtype ||
        ctx->GetOutputDesc(i)->GetStorageFormat() != ge::FORMAT_ND) {
      return ge::GRAPH_FAILED;
    }
  }
  const auto& k = ks->GetStorageShape();
  const int64_t b = k.GetDim(0), l = k.GetDim(2);
  for (int32_t i = 0; i < 3; ++i) {
    const auto* out = ctx->GetOutputShape(i);
    if (!out) {
      return ge::GRAPH_FAILED;
    }
    const auto& s = out->GetStorageShape();
    if (s.GetDimNum() != 4 || s.GetDim(0) != b || s.GetDim(1) != *h ||
        s.GetDim(2) != (i == 0 ? 1 : l + 1) || s.GetDim(3) != *d) {
      return ge::GRAPH_FAILED;
    }
  }
  auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
  uint64_t ub_bytes = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bytes);
  if (ub_bytes < 2 * t5qkv::kTileElements * sizeof(uint16_t)) {
    return ge::GRAPH_FAILED;
  }
  uint32_t cores = platform.GetCoreNumAiv();
  if (!cores) {
    return ge::GRAPH_FAILED;
  }
  if (cores > t5qkv::kMaxVectorCores) {
    cores = t5qkv::kMaxVectorCores;
  }
  auto p = t5qkv::make_plan(b, *h, l, *d, cores);
  T5SplitQkvAppendCacheTilingData t;
  t.set_batch(p.batch_);
  t.set_heads(p.heads_);
  t.set_length(p.length_);
  t.set_dim(p.dim_);
  t.set_tile_elements(p.tile_elements_);
  t.set_query_tiles(p.query_tiles_);
  t.set_cache_tiles(p.cache_tiles_);
  t.set_total_tiles(p.total_tiles_);
  ctx->SetBlockDim(p.blocks_);
  ctx->SetTilingKey(0);
  t.SaveToBuffer(ctx->GetRawTilingData()->GetData(),
                 ctx->GetRawTilingData()->GetCapacity());
  ctx->GetRawTilingData()->SetDataSize(t.GetDataSize());
  ctx->GetWorkspaceSizes(1)[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}  // namespace
}  // namespace optiling
namespace ge {
namespace {
graphStatus infer_shape(gert::InferShapeContext* ctx) {
  const auto* q = ctx->GetInputShape(0);
  const auto* k = ctx->GetInputShape(1);
  const auto* v = ctx->GetInputShape(2);
  const auto* attrs = ctx->GetAttrs();
  if (!q || !k || !v || !attrs) {
    return GRAPH_FAILED;
  }
  const auto* h = attrs->GetAttrPointer<int64_t>(0);
  const auto* d = attrs->GetAttrPointer<int64_t>(1);
  if (!h || !d || !shapes_ok(*q, *k, *v, *h, *d)) {
    return GRAPH_FAILED;
  }
  for (int32_t i = 0; i < 3; ++i) {
    auto* out = ctx->GetOutputShape(i);
    if (!out) {
      return GRAPH_FAILED;
    }
    *out = *k;
    out->SetDim(2, i == 0 ? 1 : k->GetDim(2) + 1);
  }
  return GRAPH_SUCCESS;
}
graphStatus infer_type(gert::InferDataTypeContext* ctx) {
  const DataType type = ctx->GetInputDataType(0);
  if ((type != DT_FLOAT16 && type != DT_BF16) ||
      ctx->GetInputDataType(1) != type || ctx->GetInputDataType(2) != type) {
    return GRAPH_FAILED;
  }
  for (int32_t i = 0; i < 3; ++i) {
    ctx->SetOutputDataType(i, type);
  }
  return GRAPH_SUCCESS;
}
}  // namespace
}  // namespace ge
namespace ops {
class T5SplitQkvAppendCache final : public OpDef {
 public:
  explicit T5SplitQkvAppendCache(const char* name) : OpDef(name) {
    for (const char* input : {"qkv", "past_key", "past_value"}) {
      this->Input(input)
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_BF16})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    }
    for (const char* output : {"query", "key", "value"}) {
      this->Output(output)
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_BF16})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    }
    this->Attr("head_num").AttrType(REQUIRED).Int();
    this->Attr("head_dim").AttrType(REQUIRED).Int();
    this->SetInferShape(ge::infer_shape).SetInferDataType(ge::infer_type);
    this->AICore().SetTiling(optiling::tiling).AddConfig("ascend910_93");
  }
};
OP_ADD(T5SplitQkvAppendCache);
}  // namespace ops
