// A5-only host implementation; included after the upstream headers.
namespace gdn_a5_backend {
using namespace ::optiling;

namespace {
constexpr uint64_t kAlignBytes = 512;
constexpr uint64_t kHeadDim = 128;
constexpr uint64_t kChunkSize = 128;
constexpr uint64_t kDtypeBytes = 2;
constexpr uint64_t kFloatBytes = 4;
// Reserve the largest per-core H-workspace skew used by the compiled kernel.
constexpr uint64_t kHWorkspacePadBytes = 8192;
constexpr uint64_t kHWorkspaceAlignmentBytes = 16 * 1024 * 1024;
constexpr uint64_t kHWorkspaceMaxPhaseBytes = 8 * 1024 * 1024;
constexpr uint64_t kSolveWyReadyStride = 16;
constexpr uint64_t kWyHFreeStride = 16;
constexpr uint64_t kHoReadyStride = 16;
constexpr uint32_t kSolveWyMinChunks = 8;
// KKT(2) + H(8) + primary O mailboxes(3) + independent O ping
// mailboxes(4). WY contributes two dynamically sized slot groups below.
constexpr uint64_t kFixedWorkspaceTileCount = 17;
constexpr uint32_t kPreferredBaseDim = 3072;
constexpr uint32_t kPackedHeadDim = 128;

enum class GdnTargetArch : uint32_t {
  A2A3 = 0,
  A5 = 1,
};

struct GdnArchPolicy {
  GdnTargetArch target;
};

bool ResolveArchPolicy(platform_ascendc::SocVersion soc_version,
                       GdnArchPolicy* policy) {
  if (policy == nullptr) {
    return false;
  }
  switch (soc_version) {
    case platform_ascendc::SocVersion::ASCEND910B:
    case platform_ascendc::SocVersion::ASCEND910_93:
      *policy = {GdnTargetArch::A2A3};
      return true;
    case platform_ascendc::SocVersion::ASCEND950:
      *policy = {GdnTargetArch::A5};
      return true;
    default:
      return false;
  }
}

enum InputIndex {
  MIXED_QKV_INDEX = 0,
  B_INDEX,
  A_INDEX,
  Z_INDEX,
  CONV_WEIGHT_INDEX,
  CONV_STATE_INDEX,
  A_LOG_INDEX,
  DT_BIAS_INDEX,
  CONV_STATE_READ_INDICES_INDEX,
  CONV_STATE_WRITE_INDICES_INDEX,
  SSM_STATE_READ_INDICES_INDEX,
  SSM_STATE_WRITE_INDICES_INDEX,
  SSM_CACHE_INDEX,
  MASK_LOWER_INDEX,
  MASK_FULL_INDEX,
  MINUS_IDENTITY_INDEX,
  CU_SEQLENS_INDEX,
  NORM_WEIGHT_INDEX,
};

uint32_t CeilDiv(uint32_t value, uint32_t divisor) {
  return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

struct ConvChannelTiling {
  uint32_t base_dim;
  uint32_t base_dim_count;
};

ConvChannelTiling ResolveConvChannelTiling(uint32_t conv_dim,
                                           const GdnArchPolicy& policy,
                                           uint32_t vector_task_count) {
  if (policy.target != GdnTargetArch::A5) {
    const uint32_t base_dim = std::min<uint32_t>(conv_dim, kPreferredBaseDim);
    return {base_dim, CeilDiv(conv_dim, base_dim)};
  }

  // Keep the A5 channel grid complete for every physical-core SKU while
  // preserving packed-head boundaries and the causal-conv UB limit.
  for (uint32_t channel_count = 1; channel_count <= vector_task_count;
       ++channel_count) {
    if (vector_task_count % channel_count != 0) {
      continue;
    }
    uint32_t base_dim = CeilDiv(conv_dim, channel_count);
    base_dim = CeilDiv(base_dim, kPackedHeadDim) * kPackedHeadDim;
    if (base_dim > kPreferredBaseDim ||
        CeilDiv(conv_dim, base_dim) != channel_count) {
      continue;
    }
    return {base_dim, channel_count};
  }

  const uint32_t base_dim = std::min<uint32_t>(conv_dim, kPreferredBaseDim);
  return {base_dim, CeilDiv(conv_dim, base_dim)};
}

uint64_t AlignWorkspace(uint64_t bytes) {
  return (bytes + kAlignBytes - 1) / kAlignBytes * kAlignBytes;
}

uint64_t CalcSolveWyWorkspaceSlots(uint32_t block_dim,
                                   uint32_t matrices,
                                   uint32_t batch_size,
                                   uint32_t tokens,
                                   uint32_t heads,
                                   uint32_t key_heads,
                                   GdnTargetArch target) {
  if (target != GdnTargetArch::A2A3 || block_dim == 0 || batch_size != 1 ||
      tokens < kSolveWyMinChunks * kChunkSize || tokens % kChunkSize != 0 ||
      key_heads == 0 || heads % key_heads != 0 ||
      matrices != tokens / kChunkSize * heads) {
    return 1;
  }
  const uint64_t group_size = heads / key_heads;
  if (group_size < 1 || group_size > 4) {
    return 1;
  }
  const uint64_t group_count = matrices / group_size;
  const uint64_t producer_waves = (group_count + block_dim - 1) / block_dim;
  return std::max<uint64_t>(producer_waves * group_size, 1);
}

bool IsSupportedHeadCount(uint32_t heads) {
  switch (heads) {
    case 1:
    case 2:
    case 3:
    case 4:
    case 6:
    case 8:
    case 12:
    case 16:
    case 24:
    case 32:
    case 48:
    case 64:
      return true;
    default:
      return false;
  }
}

bool HasShape(const gert::StorageShape* storage_shape,
              std::initializer_list<int64_t> dims) {
  if (storage_shape == nullptr) {
    return false;
  }
  const auto& shape = storage_shape->GetOriginShape();
  if (shape.GetDimNum() != dims.size()) {
    return false;
  }
  size_t index = 0;
  for (const int64_t dim : dims) {
    if (dim >= 0 && shape.GetDim(index) != dim) {
      return false;
    }
    ++index;
  }
  return true;
}

uint64_t CalcUserWorkspaceBytes(uint32_t block_dim,
                                uint32_t matrices,
                                uint32_t batch_size,
                                uint32_t tokens,
                                uint32_t heads,
                                uint32_t key_heads,
                                uint32_t conv_dim,
                                uint32_t conv_state_len,
                                GdnTargetArch target) {
  uint64_t bytes = 0;
  bytes += AlignWorkspace(static_cast<uint64_t>(batch_size) * conv_state_len *
                          conv_dim * kDtypeBytes);
  bytes +=
      AlignWorkspace(static_cast<uint64_t>(tokens) * conv_dim * kDtypeBytes);
  bytes += AlignWorkspace(static_cast<uint64_t>(tokens) * heads * kFloatBytes);
  bytes += AlignWorkspace(static_cast<uint64_t>(tokens) * heads * kDtypeBytes);
  bytes += AlignWorkspace(static_cast<uint64_t>(tokens) * heads * kDtypeBytes);
  bytes += AlignWorkspace(3 * kChunkSize * kChunkSize * kDtypeBytes);
  bytes += AlignWorkspace(static_cast<uint64_t>(tokens) * heads * kFloatBytes);
  bytes += AlignWorkspace(static_cast<uint64_t>(tokens) * heads * kFloatBytes);
  bytes += AlignWorkspace(static_cast<uint64_t>(tokens) * heads * kDtypeBytes);
  bytes += 2 * AlignWorkspace(static_cast<uint64_t>(tokens) * heads *
                              kChunkSize * kDtypeBytes);
  bytes += 2 * AlignWorkspace(static_cast<uint64_t>(tokens) * heads * kHeadDim *
                              kDtypeBytes);
  bytes += AlignWorkspace(static_cast<uint64_t>(matrices) * kHeadDim *
                          kHeadDim * kDtypeBytes);
  bytes += AlignWorkspace(static_cast<uint64_t>(tokens) * heads * kHeadDim *
                          kDtypeBytes);
  const uint64_t tile_bytes = kChunkSize * kChunkSize * kDtypeBytes;
  const uint64_t solve_wy_workspace_slots = CalcSolveWyWorkspaceSlots(
      block_dim, matrices, batch_size, tokens, heads, key_heads, target);
  bytes +=
      static_cast<uint64_t>(block_dim) *
      ((kFixedWorkspaceTileCount + 2 * solve_wy_workspace_slots) * tile_bytes +
       8 * kHWorkspacePadBytes);
  bytes += AlignWorkspace(static_cast<uint64_t>(block_dim) * 2 *
                          kSolveWyReadyStride * sizeof(int32_t));
  bytes += AlignWorkspace(static_cast<uint64_t>(block_dim) * kWyHFreeStride *
                          sizeof(int32_t));
  bytes += AlignWorkspace(static_cast<uint64_t>(batch_size) * heads *
                          kHoReadyStride * sizeof(int32_t));
  bytes += kHWorkspaceAlignmentBytes + kHWorkspaceMaxPhaseBytes;
  return bytes;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  GdnArchPolicy arch_policy{};
  if (!ResolveArchPolicy(platform.GetSocVersion(), &arch_policy)) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t aic_core_count = platform.GetCoreNumAic();
  const uint32_t aiv_core_count = platform.GetCoreNumAiv();
  if (aic_core_count == 0 || aiv_core_count == 0) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t block_dim = aic_core_count;
  // Both A5 vector siblings participate in the isolated Ascend950 backend.
  // Match its flattened task IDs; the A2/A3 task count remains unchanged.
  const uint32_t vector_task_count = aiv_core_count;
  if (vector_task_count == 0) {
    return ge::GRAPH_FAILED;
  }

  const auto* mixed_storage = context->GetInputShape(MIXED_QKV_INDEX);
  const auto* b_storage = context->GetInputShape(B_INDEX);
  if (mixed_storage == nullptr || b_storage == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto& mixed = mixed_storage->GetOriginShape();
  const auto& b = b_storage->GetOriginShape();
  if (mixed.GetDimNum() != 2 || b.GetDimNum() != 2) {
    return ge::GRAPH_FAILED;
  }

  const int64_t tokens64 = mixed.GetDim(0);
  const int64_t conv_dim64 = mixed.GetDim(1);
  const int64_t heads64 = b.GetDim(1);
  if (tokens64 <= 0 || tokens64 > std::numeric_limits<uint32_t>::max() ||
      conv_dim64 <= 0 || conv_dim64 > std::numeric_limits<uint32_t>::max() ||
      heads64 <= 0 || heads64 > 64 || b.GetDim(0) != tokens64 ||
      conv_dim64 % kHeadDim != 0) {
    return ge::GRAPH_FAILED;
  }

  const int64_t qk_head_axes = conv_dim64 / kHeadDim - heads64;
  if (qk_head_axes <= 0 || qk_head_axes % 2 != 0) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t tokens = static_cast<uint32_t>(tokens64);
  const uint32_t conv_dim = static_cast<uint32_t>(conv_dim64);
  const uint32_t heads = static_cast<uint32_t>(heads64);
  const uint32_t key_heads = static_cast<uint32_t>(qk_head_axes / 2);
  if (!IsSupportedHeadCount(heads) || key_heads == 0 ||
      heads % key_heads != 0) {
    return ge::GRAPH_FAILED;
  }

  // A5's MIX SYNCALL contract requires the complete physical MIX block set.
  // Stage kernels already use grid-stride ownership, so blocks without a
  // head task still participate in every barrier and safely remain idle.

  const auto* cu_seqlens_storage = context->GetInputShape(CU_SEQLENS_INDEX);
  if (cu_seqlens_storage == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto& cu_seqlens = cu_seqlens_storage->GetOriginShape();
  if (cu_seqlens.GetDimNum() != 1 || cu_seqlens.GetDim(0) < 2 ||
      cu_seqlens.GetDim(0) - 1 >
          static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return ge::GRAPH_FAILED;
  }
  const int64_t batch_size64 = cu_seqlens.GetDim(0) - 1;
  const uint32_t batch_size = static_cast<uint32_t>(batch_size64);

  const auto* conv_state_storage = context->GetInputShape(CONV_STATE_INDEX);
  const auto* ssm_cache_storage = context->GetInputShape(SSM_CACHE_INDEX);
  if (conv_state_storage == nullptr || ssm_cache_storage == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto& conv_state = conv_state_storage->GetOriginShape();
  const auto& ssm_cache = ssm_cache_storage->GetOriginShape();
  if (conv_state.GetDimNum() != 3 || conv_state.GetDim(0) <= 0 ||
      conv_state.GetDim(0) > std::numeric_limits<int32_t>::max() ||
      conv_state.GetDim(1) < 3 ||
      conv_state.GetDim(1) > std::numeric_limits<uint32_t>::max() ||
      conv_state.GetDim(2) != conv_dim64 || ssm_cache.GetDimNum() != 4 ||
      ssm_cache.GetDim(0) <= 0 ||
      ssm_cache.GetDim(0) > std::numeric_limits<int32_t>::max()) {
    return ge::GRAPH_FAILED;
  }
  const int64_t checkpoint_stride64 = conv_state.GetDim(1) - 2;
  const int64_t expected_conv_state_len = checkpoint_stride64 + 2;
  if (checkpoint_stride64 <= 0 ||
      checkpoint_stride64 > std::numeric_limits<uint32_t>::max() ||
      conv_state.GetDim(1) != expected_conv_state_len) {
    return ge::GRAPH_FAILED;
  }
  if (!HasShape(context->GetInputShape(A_INDEX), {tokens64, heads64}) ||
      !HasShape(context->GetInputShape(Z_INDEX),
                {tokens64, heads64, static_cast<int64_t>(kHeadDim)}) ||
      !HasShape(context->GetInputShape(CONV_WEIGHT_INDEX), {4, conv_dim64}) ||
      !HasShape(context->GetInputShape(A_LOG_INDEX), {heads64}) ||
      !HasShape(context->GetInputShape(DT_BIAS_INDEX), {heads64}) ||
      !HasShape(context->GetInputShape(CONV_STATE_READ_INDICES_INDEX),
                {batch_size64}) ||
      !HasShape(context->GetInputShape(CONV_STATE_WRITE_INDICES_INDEX),
                {batch_size64}) ||
      !HasShape(context->GetInputShape(SSM_STATE_READ_INDICES_INDEX),
                {batch_size64}) ||
      !HasShape(context->GetInputShape(SSM_STATE_WRITE_INDICES_INDEX),
                {batch_size64}) ||
      !HasShape(ssm_cache_storage,
                {-1,
                 heads64,
                 static_cast<int64_t>(kHeadDim),
                 static_cast<int64_t>(kHeadDim)}) ||
      !HasShape(context->GetInputShape(MASK_LOWER_INDEX),
                {static_cast<int64_t>(kChunkSize),
                 static_cast<int64_t>(kChunkSize)}) ||
      !HasShape(context->GetInputShape(MASK_FULL_INDEX),
                {static_cast<int64_t>(kChunkSize),
                 static_cast<int64_t>(kChunkSize)}) ||
      !HasShape(context->GetInputShape(MINUS_IDENTITY_INDEX),
                {static_cast<int64_t>(kChunkSize),
                 static_cast<int64_t>(kChunkSize)}) ||
      !HasShape(cu_seqlens_storage, {batch_size64 + 1}) ||
      !HasShape(context->GetInputShape(NORM_WEIGHT_INDEX),
                {static_cast<int64_t>(kHeadDim)})) {
    return ge::GRAPH_FAILED;
  }

  const ConvChannelTiling conv_channel_tiling =
      ResolveConvChannelTiling(conv_dim, arch_policy, vector_task_count);
  const uint32_t base_dim = conv_channel_tiling.base_dim;
  const uint32_t base_dim_count = conv_channel_tiling.base_dim_count;
  const uint32_t token_core_budget =
      std::max<uint32_t>(vector_task_count / base_dim_count, 1);
  const uint32_t token_block_size = CeilDiv(tokens, token_core_budget);
  const uint32_t token_block_count = CeilDiv(tokens, token_block_size);
  uint64_t ffts_addr = 0;
  const auto* attrs = context->GetAttrs();
  if (attrs == nullptr || attrs->GetAttrPointer<int64_t>(0) == nullptr ||
      attrs->GetAttrPointer<int64_t>(1) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ffts_addr = static_cast<uint64_t>(*attrs->GetAttrPointer<int64_t>(0));
  const int64_t num_matrices64 = *attrs->GetAttrPointer<int64_t>(1);
  if (num_matrices64 <= 0 ||
      num_matrices64 >
          static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      num_matrices64 % heads64 != 0 || num_matrices64 > tokens64 * heads64) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t num_matrices = static_cast<uint32_t>(num_matrices64);

  MegaGdnPrefillOpTilingData tiling;
  tiling.set_block_dim(block_dim);
  tiling.set_vector_task_count(vector_task_count);
  tiling.set_target_arch(static_cast<uint32_t>(arch_policy.target));
  tiling.set_num_matrices(num_matrices);
  tiling.set_batch_size(batch_size);
  tiling.set_num_heads(heads);
  tiling.set_num_key_heads(key_heads);
  tiling.set_token_block_size(token_block_size);
  tiling.set_token_block_count(token_block_count);
  tiling.set_base_dim(base_dim);
  tiling.set_base_dim_count(base_dim_count);
  tiling.set_conv_dim(conv_dim);
  tiling.set_conv_state_slots(static_cast<uint32_t>(conv_state.GetDim(0)));
  tiling.set_conv_state_len(static_cast<uint32_t>(conv_state.GetDim(1)));
  tiling.set_ssm_state_slots(static_cast<uint32_t>(ssm_cache.GetDim(0)));
  tiling.set_checkpoint_stride(static_cast<uint32_t>(checkpoint_stride64));
  tiling.set_total_tokens(tokens);
  tiling.set_ffts_addr(ffts_addr);

  context->SetBlockDim(block_dim);
  if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  size_t* workspace_sizes = context->GetWorkspaceSizes(1);
  if (workspace_sizes == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const uint64_t user_workspace_bytes =
      CalcUserWorkspaceBytes(block_dim,
                             num_matrices,
                             batch_size,
                             tokens,
                             heads,
                             key_heads,
                             conv_dim,
                             static_cast<uint32_t>(conv_state.GetDim(1)),
                             arch_policy.target);
  workspace_sizes[0] = user_workspace_bytes + platform.GetLibApiWorkSpaceSize();
  return ge::GRAPH_SUCCESS;
}

struct MegaGdnPrefillOpCompileInfo {};

static ge::graphStatus TilingParse(gert::TilingParseContext* context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

}  // namespace gdn_a5_backend
