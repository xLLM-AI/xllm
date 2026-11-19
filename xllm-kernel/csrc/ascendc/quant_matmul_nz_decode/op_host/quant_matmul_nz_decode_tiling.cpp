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

#include "quant_matmul_nz_decode_tiling.h"

#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t kTileM = 16;
constexpr uint32_t kMaxOptimizedM = 16;
constexpr uint32_t kWorkspaceStages = 2;
constexpr uint32_t kGateUpSmallMTileN = 320;
constexpr uint32_t kGateUpLargeMTileN = 160;
constexpr uint32_t kQkvTileN = 128;
constexpr uint32_t kDownM1TileN = 64;
constexpr uint32_t kDownM2To4TileN = 128;
constexpr uint32_t kDownLargeMTileN = 256;
constexpr uint64_t kGateUpM1TilingKey = 6;
constexpr uint64_t kGateUpM2TilingKey = 7;
constexpr uint64_t kGateUpM4TilingKey = 8;
constexpr uint64_t kGateUpM8TilingKey = 9;
constexpr uint64_t kGateUpM16TilingKey = 10;
constexpr uint64_t kQkvM1TilingKey = 11;
constexpr uint64_t kQkvM2TilingKey = 12;
constexpr uint64_t kQkvM4TilingKey = 13;
constexpr uint64_t kQkvM8TilingKey = 14;
constexpr uint64_t kQkvM16TilingKey = 15;

struct TilingConfig {
  uint64_t tiling_key;
  uint32_t tile_n;
};

uint32_t ceil_div(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

TilingConfig select_tiling_config(uint32_t m,
                                  bool is_down_shape,
                                  bool is_qkv_shape) {
  if (is_down_shape) {
    if (m == 1) {
      return {1, kDownM1TileN};
    }
    if (m <= 4) {
      return {3, kDownM2To4TileN};
    }
    return {2, kDownLargeMTileN};
  }

  if (is_qkv_shape) {
    if (m == 1) {
      return {kQkvM1TilingKey, kQkvTileN};
    }
    if (m == 2) {
      return {kQkvM2TilingKey, kQkvTileN};
    }
    if (m == 4) {
      return {kQkvM4TilingKey, kQkvTileN};
    }
    if (m == 8) {
      return {kQkvM8TilingKey, kQkvTileN};
    }
    if (m == 16) {
      return {kQkvM16TilingKey, kQkvTileN};
    }
    return {4, kQkvTileN};
  }

  if (m == 1) {
    return {kGateUpM1TilingKey, kGateUpSmallMTileN};
  }
  if (m == 2) {
    return {kGateUpM2TilingKey, kGateUpSmallMTileN};
  }
  if (m == 4) {
    return {kGateUpM4TilingKey, kGateUpLargeMTileN};
  }
  if (m == 8) {
    return {kGateUpM8TilingKey, kGateUpLargeMTileN};
  }
  if (m == 16) {
    return {kGateUpM16TilingKey, kGateUpLargeMTileN};
  }
  return {m > 2 ? 5UL : 0UL, m > 2 ? kGateUpLargeMTileN : kGateUpSmallMTileN};
}

ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const auto x_shape = context->GetInputShape(0)->GetOriginShape();
  const auto weight_shape = context->GetInputShape(1)->GetOriginShape();
  const auto scale_shape = context->GetInputShape(2)->GetOriginShape();
  const auto bias_shape = context->GetInputShape(3)->GetOriginShape();
  const size_t x_dim = x_shape.GetDimNum();
  const size_t weight_dim = weight_shape.GetDimNum();
  if (x_dim != 2 || weight_dim != 2 || scale_shape.GetDimNum() != 1 ||
      bias_shape.GetDimNum() != 1) {
    return ge::GRAPH_FAILED;
  }

  const int64_t m_dim = x_shape.GetDim(0);
  const int64_t k_dim = x_shape.GetDim(1);
  const int64_t n_dim = weight_shape.GetDim(1);

  const bool is_gate_up_shape = k_dim == 5120 && n_dim == 6400;
  const bool is_down_shape = k_dim == 3200 && n_dim == 5120;
  const bool is_qkv_shape = k_dim == 5120 && n_dim == 1280;
  if (m_dim <= 0 || m_dim > kMaxOptimizedM ||
      (!is_gate_up_shape && !is_down_shape && !is_qkv_shape) ||
      weight_shape.GetDim(0) != k_dim || scale_shape.GetDim(0) != n_dim ||
      bias_shape.GetDim(0) != n_dim) {
    return ge::GRAPH_FAILED;
  }

  const uint32_t m = static_cast<uint32_t>(m_dim);
  const uint32_t k = static_cast<uint32_t>(k_dim);
  const uint32_t n = static_cast<uint32_t>(n_dim);
  QuantMatmulNzDecodeTilingData tiling;
  tiling.set_m(m);
  tiling.set_k(k);
  tiling.set_n(n);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  const TilingConfig tiling_config =
      select_tiling_config(m, is_down_shape, is_qkv_shape);
  context->SetTilingKey(tiling_config.tiling_key);
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const uint32_t aic_core_count = platform.GetCoreNumAic();
  if (aic_core_count == 0) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t task_count =
      ceil_div(m, kTileM) * ceil_div(n, tiling_config.tile_n);
  const uint32_t block_dim = std::min(aic_core_count, task_count);
  context->SetBlockDim(block_dim);

  size_t* workspace_size = context->GetWorkspaceSizes(1);
  workspace_size[0] = platform.GetLibApiWorkSpaceSize() +
                      static_cast<size_t>(kTileM) * tiling_config.tile_n *
                          block_dim * kWorkspaceStages * sizeof(int32_t);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(XllmKernelQuantMatmulNzDecode).Tiling(TilingFunc);
}  // namespace optiling
