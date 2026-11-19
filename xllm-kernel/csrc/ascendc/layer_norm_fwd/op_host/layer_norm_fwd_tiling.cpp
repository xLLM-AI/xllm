/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "layer_norm_fwd_tiling.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t kInputX = 0;
constexpr size_t kInputWeight = 1;
constexpr size_t kInputBias = 2;
constexpr size_t kInputZ = 3;
constexpr size_t kAttrEps = 0;
constexpr size_t kAttrGroupSize = 1;
constexpr size_t kAttrNormBeforeGate = 2;
constexpr size_t kAttrIsRmsNorm = 3;

enum KernelMode : uint32_t {
  kFullRow = 0,
  kStreamingTwoPass = 1,
};

int64_t AlignUp(int64_t value, int64_t align) {
  return ((value + align - 1) / align) * align;
}

int64_t AlignDown(int64_t value, int64_t align) {
  return (value / align) * align;
}

int64_t LimitTailPadding(int64_t group_size, int64_t chunk, int64_t align) {
  chunk = std::min(chunk, group_size);
  chunk = AlignDown(chunk, align);
  chunk = std::max(chunk, align);
  for (int64_t candidate = chunk; candidate >= align; candidate -= align) {
    if (group_size % candidate == 0) {
      return candidate;
    }
  }
  for (int64_t candidate = chunk; candidate >= align; candidate -= align) {
    const int64_t tail = group_size % candidate;
    if (tail == 0 || candidate - tail <= 255) {
      return candidate;
    }
  }
  return align;
}

int64_t GetEnvInt64(const char* name, int64_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }

  char* end = nullptr;
  const int64_t parsed = std::strtoll(value, &end, 10);
  if (end == value || *end != '\0') {
    return default_value;
  }
  return parsed;
}

int64_t GetN128RowsPerCore() {
  static const int64_t rows_per_core =
      GetEnvInt64("XLLM_LAYER_NORM_FWD_N128_ROWS_PER_CORE", 6);
  return rows_per_core;
}

int64_t GetN128SmallRowsPerCore() {
  static const int64_t rows_per_core = std::max<int64_t>(
      1, GetEnvInt64("XLLM_LAYER_NORM_FWD_N128_SMALL_ROWS_PER_CORE", 2));
  return rows_per_core;
}

int64_t GetTypeSize(ge::DataType dtype) {
  switch (dtype) {
    case ge::DT_FLOAT:
      return 4;
    case ge::DT_FLOAT16:
    case ge::DT_BF16:
      return 2;
    default:
      return 2;
  }
}

}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  LayerNormFwdTilingData tiling;

  const gert::StorageShape* x_shape_info = context->GetInputShape(kInputX);
  if (x_shape_info == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto x_shape = x_shape_info->GetStorageShape();
  const int64_t dim_num = x_shape.GetDimNum();
  if (dim_num <= 0) {
    return ge::GRAPH_FAILED;
  }
  int64_t full_n = x_shape.GetDim(dim_num - 1);
  if (full_n <= 0) {
    return ge::GRAPH_FAILED;
  }
  int64_t m = 1;
  for (int64_t i = 0; i < dim_num - 1; ++i) {
    m *= x_shape.GetDim(i);
  }

  auto attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const float eps = *(attrs->GetAttrPointer<float>(kAttrEps));
  int64_t group_size = *(attrs->GetAttrPointer<int64_t>(kAttrGroupSize));
  if (group_size <= 0) {
    group_size = full_n;
  }
  if (group_size <= 0 || full_n % group_size != 0) {
    return ge::GRAPH_FAILED;
  }
  const bool norm_before_gate =
      *(attrs->GetAttrPointer<bool>(kAttrNormBeforeGate));
  const bool is_rms_norm = *(attrs->GetAttrPointer<bool>(kAttrIsRmsNorm));
  const int64_t ngroups = full_n / group_size;

  const bool has_bias = context->GetOptionalInputDesc(kInputBias) != nullptr;
  const bool has_z = context->GetOptionalInputDesc(kInputZ) != nullptr;
  const int64_t x_dtype_size =
      GetTypeSize(context->GetInputDesc(kInputX)->GetDataType());
  const int64_t weight_dtype_size =
      GetTypeSize(context->GetInputDesc(kInputWeight)->GetDataType());

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const bool use_a2_scalar_store_fix =
      platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910B;
  const int64_t core_num = platform.GetCoreNumAiv();
  uint64_t ub_size = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

  const int64_t align_elems = std::max<int64_t>(1, 32 / x_dtype_size);
  const int64_t group_align = AlignUp(group_size, align_elems);

  constexpr int64_t kBufferNum = 2;
  constexpr int64_t kN128ReduceBatchRows = 32;
  constexpr int64_t kFp32RepeatElems = 64;
  const int64_t scalar_bytes = 32;
  const int64_t ub_reserve_bytes = 8 * 1024;
  const bool n128_fast_path = full_n == 128 && group_size == 128 && !has_bias;
  const int64_t n128_reduce_work_bytes =
      n128_fast_path ? kN128ReduceBatchRows * kFp32RepeatElems * 4 : 0;
  const int64_t reduce_tmp_bytes = std::max<int64_t>(group_align, 64) * 4;
  const int64_t param_bytes =
      group_align *
      (weight_dtype_size + 4 + (has_bias ? (weight_dtype_size + 4) : 0));
  const int64_t queue_bytes =
      kBufferNum * group_align * (2 + (has_z ? 1 : 0)) * x_dtype_size;
  const int64_t compute_bytes = group_align * 2 * 4;
  const int64_t stat_bytes = 2 * 4;
  const int64_t per_row_bytes = queue_bytes + compute_bytes + stat_bytes;
  const int64_t full_row_fixed =
      scalar_bytes + reduce_tmp_bytes + param_bytes + n128_reduce_work_bytes;

  int64_t tile_rows = 0;
  if (static_cast<int64_t>(ub_size) > full_row_fixed + ub_reserve_bytes &&
      per_row_bytes > 0) {
    tile_rows =
        (static_cast<int64_t>(ub_size) - full_row_fixed - ub_reserve_bytes) /
        per_row_bytes;
  }

  const int64_t logical_rows = m * ngroups;
  int64_t used_cores = std::max<int64_t>(1, std::min(core_num, logical_rows));
  const int64_t n128_small_rows_per_core = GetN128SmallRowsPerCore();
  if (full_n == 128 && group_size == 128 && has_z && !has_bias &&
      logical_rows > 1 && logical_rows <= 32 && n128_small_rows_per_core > 1) {
    used_cores = std::max<int64_t>(
        1,
        std::min<int64_t>(used_cores,
                          (logical_rows + n128_small_rows_per_core - 1) /
                              n128_small_rows_per_core));
  }
  const int64_t n128_rows_per_core = GetN128RowsPerCore();
  if (full_n == 128 && group_size == 128 && logical_rows > 32 &&
      n128_rows_per_core > 1) {
    used_cores = std::max<int64_t>(
        1,
        std::min<int64_t>(
            used_cores,
            (logical_rows + n128_rows_per_core - 1) / n128_rows_per_core));
  }

  uint32_t kernel_mode = kFullRow;
  uint32_t tile_rows_out = 1;
  uint32_t chunk_size = static_cast<uint32_t>(group_align);
  uint32_t group_align_out = static_cast<uint32_t>(group_align);

  if (tile_rows >= 1) {
    const int64_t max_tile_rows = tile_rows;
    if (m >= used_cores * 2) {
      const int64_t rows_for_occupancy =
          std::max<int64_t>(1, (m + used_cores - 1) / used_cores);
      tile_rows = std::min<int64_t>(tile_rows, rows_for_occupancy);
      tile_rows = std::min<int64_t>(tile_rows, 255);
    } else {
      tile_rows = 1;
    }
    if (m >= 8 && max_tile_rows >= 8) {
      tile_rows = std::max<int64_t>(tile_rows, 8);
      tile_rows = std::min<int64_t>(tile_rows, max_tile_rows);
      tile_rows = AlignDown(tile_rows, 8);
    }
    tile_rows_out = static_cast<uint32_t>(std::max<int64_t>(1, tile_rows));
  } else {
    const int64_t stream_fixed = scalar_bytes;
    const int64_t stream_queue_coeff =
        kBufferNum * (2 + (has_z ? 1 : 0)) * x_dtype_size;
    const int64_t stream_compute_coeff = 3 * 4;
    const int64_t stream_param_coeff =
        kBufferNum * weight_dtype_size + 4 +
        (has_bias ? (kBufferNum * weight_dtype_size + 4) : 0);
    const int64_t stream_coeff =
        stream_queue_coeff + stream_compute_coeff + stream_param_coeff;
    int64_t chunk = 0;
    if (static_cast<int64_t>(ub_size) > stream_fixed + ub_reserve_bytes &&
        stream_coeff > 0) {
      chunk =
          (static_cast<int64_t>(ub_size) - stream_fixed - ub_reserve_bytes) /
          stream_coeff;
    }
    chunk = std::max<int64_t>(align_elems, AlignDown(chunk, align_elems));
    chunk = std::min<int64_t>(chunk, group_size);
    chunk = std::min<int64_t>(chunk, 1024);
    chunk = LimitTailPadding(group_size, chunk, align_elems);

    kernel_mode = kStreamingTwoPass;
    tile_rows_out = 1;
    chunk_size = static_cast<uint32_t>(chunk);
    group_align_out = static_cast<uint32_t>(AlignUp(chunk, align_elems));
  }

  tiling.set_m(static_cast<uint32_t>(m));
  tiling.set_full_n(static_cast<uint32_t>(full_n));
  tiling.set_group_size(static_cast<uint32_t>(group_size));
  tiling.set_ngroups(static_cast<uint32_t>(ngroups));
  tiling.set_stride_x(static_cast<uint32_t>(full_n));
  tiling.set_stride_y(static_cast<uint32_t>(full_n));
  tiling.set_stride_z(static_cast<uint32_t>(full_n));
  tiling.set_group_align(group_align_out);
  tiling.set_tile_rows(tile_rows_out);
  tiling.set_chunk_size(chunk_size);
  tiling.set_eps(eps);
  tiling.set_has_bias(has_bias ? 1U : 0U);
  tiling.set_has_z(has_z ? 1U : 0U);
  tiling.set_norm_before_gate(norm_before_gate ? 1U : 0U);
  tiling.set_is_rms_norm(is_rms_norm ? 1U : 0U);
  tiling.set_use_a2_scalar_store_fix(use_a2_scalar_store_fix ? 1U : 0U);
  tiling.set_kernel_mode(kernel_mode);

  const int64_t block_dim =
      use_a2_scalar_store_fix && !is_rms_norm ? 1 : used_cores;
  context->SetBlockDim(static_cast<uint32_t>(block_dim));
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  size_t* workspace = context->GetWorkspaceSizes(1);
  workspace[0] = platform.GetLibApiWorkSpaceSize();
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(XllmKernelLayerNormFwd).Tiling(TilingFunc);
}  // namespace optiling
