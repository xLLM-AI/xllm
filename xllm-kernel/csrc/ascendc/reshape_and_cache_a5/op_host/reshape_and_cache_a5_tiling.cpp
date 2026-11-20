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

#include "reshape_and_cache_a5_tiling.h"

#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t kBlockBytes = 32;
constexpr uint32_t kReservedUbBytes = 8 * 1024;
constexpr int32_t kKeyInput = 0;
constexpr int32_t kValueInput = 1;
constexpr int32_t kKeyCacheInput = 2;
constexpr int32_t kValueCacheInput = 3;
constexpr int32_t kSlotMappingInput = 4;

bool ShapesMatch(const gert::Shape& lhs, const gert::Shape& rhs) {
  if (lhs.GetDimNum() != rhs.GetDimNum()) {
    return false;
  }
  for (size_t dim = 0; dim < lhs.GetDimNum(); ++dim) {
    if (lhs.GetDim(dim) != rhs.GetDim(dim)) {
      return false;
    }
  }
  return true;
}

ge::graphStatus TilingFunc(gert::TilingContext* context) {
  if (context == nullptr || context->GetInputShape(kKeyInput) == nullptr ||
      context->GetInputShape(kValueInput) == nullptr ||
      context->GetInputShape(kKeyCacheInput) == nullptr ||
      context->GetInputShape(kValueCacheInput) == nullptr ||
      context->GetInputShape(kSlotMappingInput) == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const gert::Shape key_shape =
      context->GetInputShape(kKeyInput)->GetStorageShape();
  const gert::Shape value_shape =
      context->GetInputShape(kValueInput)->GetStorageShape();
  const gert::Shape cache_shape =
      context->GetInputShape(kKeyCacheInput)->GetStorageShape();
  const gert::Shape value_cache_shape =
      context->GetInputShape(kValueCacheInput)->GetStorageShape();
  const gert::Shape slot_shape =
      context->GetInputShape(kSlotMappingInput)->GetStorageShape();
  if (key_shape.GetDimNum() != 3 || cache_shape.GetDimNum() != 4 ||
      slot_shape.GetDimNum() != 1 || !ShapesMatch(key_shape, value_shape) ||
      !ShapesMatch(cache_shape, value_cache_shape) ||
      key_shape.GetDim(0) != slot_shape.GetDim(0) ||
      key_shape.GetDim(1) != cache_shape.GetDim(2) ||
      key_shape.GetDim(2) != cache_shape.GetDim(3) ||
      key_shape.GetDim(1) <= 0 || key_shape.GetDim(2) <= 0 ||
      cache_shape.GetDim(0) <= 0 || cache_shape.GetDim(1) <= 0) {
    return ge::GRAPH_FAILED;
  }

  const uint32_t num_tokens = static_cast<uint32_t>(key_shape.GetDim(0));
  const uint32_t row_elements =
      static_cast<uint32_t>(key_shape.GetDim(1) * key_shape.GetDim(2));
  const uint32_t total_slots =
      static_cast<uint32_t>(cache_shape.GetDim(0) * cache_shape.GetDim(1));

  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  uint64_t ub_bytes = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bytes);
  const uint32_t dtype_bytes = static_cast<uint32_t>(
      ge::GetSizeByDataType(context->GetInputDesc(kKeyInput)->GetDataType()));
  const uint64_t usable_ub_bytes =
      ub_bytes > kReservedUbBytes ? ub_bytes - kReservedUbBytes : ub_bytes;
  if (dtype_bytes == 0 || usable_ub_bytes < kBlockBytes) {
    return ge::GRAPH_FAILED;
  }
  uint32_t tile_elements = static_cast<uint32_t>(usable_ub_bytes / dtype_bytes);
  const uint32_t elements_per_block = kBlockBytes / dtype_bytes;
  tile_elements =
      std::max(elements_per_block,
               tile_elements / elements_per_block * elements_per_block);
  tile_elements = std::min(row_elements, tile_elements);

  ReshapeAndCacheA5TilingData tiling;
  tiling.set_num_tokens(num_tokens);
  tiling.set_row_elements(row_elements);
  tiling.set_total_slots(total_slots);
  tiling.set_tile_elements(tile_elements);

  const uint32_t aiv_num = platform.GetCoreNumAiv();
  context->SetBlockDim(std::max(1U, std::min(num_tokens, aiv_num)));
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(XllmKernelReshapeAndCacheA5).Tiling(TilingFunc);
}  // namespace optiling
