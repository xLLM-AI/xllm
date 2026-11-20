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

#include "mega_gdn_mtp_decode_varlen_tiling.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

bool has_shape(const gert::Shape& shape, std::initializer_list<int64_t> dims) {
  if (shape.GetDimNum() != dims.size()) {
    return false;
  }
  size_t index = 0;
  for (const int64_t dim : dims) {
    if (shape.GetDim(index++) != dim) {
      return false;
    }
  }
  return true;
}

ge::graphStatus tile_varlen(gert::TilingContext* context) {
  constexpr std::array<ge::DataType, 13> kTypes = {ge::DT_BF16,
                                                   ge::DT_BF16,
                                                   ge::DT_BF16,
                                                   ge::DT_BF16,
                                                   ge::DT_BF16,
                                                   ge::DT_BF16,
                                                   ge::DT_FLOAT,
                                                   ge::DT_FLOAT,
                                                   ge::DT_FLOAT,
                                                   ge::DT_INT32,
                                                   ge::DT_INT32,
                                                   ge::DT_BF16,
                                                   ge::DT_INT32};
  for (size_t i = 0; i < kTypes.size(); ++i) {
    if (context->GetInputShape(i) == nullptr ||
        context->GetInputDesc(i) == nullptr ||
        context->GetInputDesc(i)->GetDataType() != kTypes[i]) {
      return ge::GRAPH_FAILED;
    }
  }
  const gert::Shape& qkv = context->GetInputShape(0)->GetStorageShape();
  const gert::Shape& z = context->GetInputShape(1)->GetStorageShape();
  const gert::Shape& conv = context->GetInputShape(5)->GetStorageShape();
  const gert::Shape& state_indices =
      context->GetInputShape(9)->GetStorageShape();
  if (qkv.GetDimNum() != 2 || z.GetDimNum() != 3 || conv.GetDimNum() != 3 ||
      (state_indices.GetDimNum() != 1 && state_indices.GetDimNum() != 2)) {
    return ge::GRAPH_FAILED;
  }
  const int64_t tokens = qkv.GetDim(0);
  const int64_t channels = qkv.GetDim(1);
  const int64_t batch = state_indices.GetDim(0);
  const bool split_indices = state_indices.GetDimNum() == 2;
  if (split_indices && state_indices.GetDim(1) != 2) {
    return ge::GRAPH_FAILED;
  }
  const int64_t slots = conv.GetDim(0);
  const int64_t stride = conv.GetDim(1) - 2;
  const int64_t nv = z.GetDim(1);
  const int64_t qk_width = channels - nv * 128;
  if (tokens < 1 || batch < 1 || batch > 128 || slots < 1 ||
      (!split_indices && slots > 1024) || stride < 2 || stride > 17 || nv < 1 ||
      qk_width < 256 || qk_width % 256 != 0) {
    return ge::GRAPH_FAILED;
  }
  const int64_t nk = qk_width / 256;
  if (tokens > std::numeric_limits<int32_t>::max() / channels) {
    return ge::GRAPH_FAILED;
  }
  if (nk > 16 || (nk & (nk - 1)) != 0 || nv % nk != 0 || nv / nk > 4) {
    return ge::GRAPH_FAILED;
  }
  if (!has_shape(z, {tokens, nv, 128}) ||
      !has_shape(context->GetInputShape(2)->GetStorageShape(), {tokens, nv}) ||
      !has_shape(context->GetInputShape(3)->GetStorageShape(), {tokens, nv}) ||
      !has_shape(context->GetInputShape(4)->GetStorageShape(), {4, channels}) ||
      !has_shape(conv, {slots, stride + 2, channels}) ||
      !has_shape(context->GetInputShape(6)->GetStorageShape(), {nv}) ||
      !has_shape(context->GetInputShape(7)->GetStorageShape(), {nv}) ||
      !has_shape(context->GetInputShape(8)->GetStorageShape(),
                 {split_indices
                      ? context->GetInputShape(8)->GetStorageShape().GetDim(0)
                      : slots * stride,
                  nv,
                  128,
                  128}) ||
      !has_shape(context->GetInputShape(10)->GetStorageShape(), {batch}) ||
      !has_shape(context->GetInputShape(11)->GetStorageShape(), {128}) ||
      !has_shape(context->GetInputShape(12)->GetStorageShape(), {batch + 1})) {
    return ge::GRAPH_FAILED;
  }
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ub_bytes = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bytes);
  if (ub_bytes < 175360) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t tasks =
      static_cast<uint32_t>(std::max(channels / 128, batch * nv));
  const uint32_t aic = platform.GetCoreNumAic();
  const uint32_t aiv = platform.GetCoreNumAiv();
  const uint32_t blocks =
      platform.CalcTschBlockDim(std::min(tasks, aiv), aic, aiv);
  const auto* attrs = context->GetAttrs();
  const bool* fla_layout =
      attrs == nullptr ? nullptr : attrs->GetAttrPointer<bool>(0);
  if (blocks == 0 || fla_layout == nullptr) {
    return ge::GRAPH_FAILED;
  }
  MegaGdnMtpDecodeVarlenTilingData data;
  data.set_batch_size(batch);
  data.set_state_index_width(split_indices ? 2 : 1);
  data.set_sequence_length(stride);
  data.set_num_k_heads(nk);
  data.set_num_v_heads(nv);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  context->SetTilingKey(*fla_layout ? 100 : 1100);
  context->SetBlockDim(blocks);
  context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
  return context->SetScheduleMode(1);
}

}  // namespace

IMPL_OP_OPTILING(XllmKernelMegaGdnMtpDecodeVarlen).Tiling(tile_varlen);

}  // namespace optiling
