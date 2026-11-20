// A5-only host implementation; included after the upstream headers.
namespace gdn_a5_backend {
using namespace ::optiling;

namespace optiling {
namespace {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kMinSequenceLength = 2;
constexpr int64_t kMaxSequenceLength = 17;
constexpr int64_t kMaxBatchSize = 32;
constexpr int64_t kMaxNumCacheSlots = 1024;
constexpr uint64_t kRequiredUbBytes = 175360;
constexpr uint64_t kQkGroupCacheRequiredUbBytes = 187936;
constexpr uint64_t kQkGroupCacheTilingKey = 208;
constexpr uint64_t kA5TwoOwnerQkGroupTilingKeyBase = 300;
constexpr uint64_t kA5RegBaseDeferredTilingKey = 403;
constexpr uint64_t kA5RegBaseTwoOwnerTilingKey = 503;
constexpr int64_t kMinDeferredNormSpeculativeTokens = 10;
constexpr uint64_t kDeferredNormRequiredUbBytes = 182816;
constexpr uint64_t kDeferredNormTilingKeyBase = 200;
constexpr int64_t kA5DeferredNormSpeculativeTokens = 3;
constexpr uint32_t kQkGroupCacheRequiredAivCores = 32;
constexpr uint32_t kAivPerA2A3MixedBlock = 2;
constexpr uint32_t kAivPerA5MixedBlock = 2;
constexpr size_t kFlaSsmStateLayoutAttr = 0;
constexpr uint64_t kNonFlaTilingKeyOffset = 1000;

enum InputIndex : size_t {
  kQkv = 0,
  kZ = 1,
  kB = 2,
  kA = 3,
  kConvWeight = 4,
  kConvState = 5,
  kALog = 6,
  kDtBias = 7,
  kSsmState = 8,
  kReadStateIndices = 9,
  kWriteStateIndices = 10,
  kNumAcceptedTokens = 11,
  kNormWeight = 12,
};

struct ShapeInfo {
  int64_t batch_size = 0;
  int64_t speculative_tokens = 0;
  int64_t sequence_length = 0;
  int64_t conv_state_length = 0;
  int64_t num_k_heads = 0;
  int64_t num_v_heads = 0;
  int64_t conv_tile_count = 0;
  int64_t state_index_width = 1;
};

bool IsSupportedHeadGeometry(int64_t num_k_heads, int64_t num_v_heads) {
  if (num_k_heads < 1 || num_k_heads > 16 ||
      (num_k_heads & (num_k_heads - 1)) != 0 ||
      num_v_heads % num_k_heads != 0) {
    return false;
  }
  const int64_t v_heads_per_k = num_v_heads / num_k_heads;
  return v_heads_per_k >= 1 && v_heads_per_k <= 4;
}

bool HasShape(const gert::Shape& shape, std::initializer_list<int64_t> dims) {
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

ge::graphStatus GetShapeInfo(gert::TilingContext* context, ShapeInfo& info) {
  constexpr std::array<size_t, 13> kRequiredInputs = {kQkv,
                                                      kZ,
                                                      kB,
                                                      kA,
                                                      kConvWeight,
                                                      kConvState,
                                                      kALog,
                                                      kDtBias,
                                                      kSsmState,
                                                      kReadStateIndices,
                                                      kWriteStateIndices,
                                                      kNumAcceptedTokens,
                                                      kNormWeight};
  for (const size_t index : kRequiredInputs) {
    if (context->GetInputShape(index) == nullptr ||
        context->GetInputDesc(index) == nullptr) {
      return ge::GRAPH_FAILED;
    }
  }

  const gert::Shape& qkv = context->GetInputShape(kQkv)->GetStorageShape();
  const gert::Shape& z = context->GetInputShape(kZ)->GetStorageShape();
  const gert::Shape& b = context->GetInputShape(kB)->GetStorageShape();
  const gert::Shape& a = context->GetInputShape(kA)->GetStorageShape();
  const gert::Shape& conv_weight =
      context->GetInputShape(kConvWeight)->GetStorageShape();
  const gert::Shape& conv_state =
      context->GetInputShape(kConvState)->GetStorageShape();
  const gert::Shape& a_log = context->GetInputShape(kALog)->GetStorageShape();
  const gert::Shape& dt_bias =
      context->GetInputShape(kDtBias)->GetStorageShape();
  const gert::Shape& ssm_state =
      context->GetInputShape(kSsmState)->GetStorageShape();
  const gert::Shape& read_indices =
      context->GetInputShape(kReadStateIndices)->GetStorageShape();
  const gert::Shape& write_indices =
      context->GetInputShape(kWriteStateIndices)->GetStorageShape();
  const gert::Shape& accepted =
      context->GetInputShape(kNumAcceptedTokens)->GetStorageShape();
  const gert::Shape& norm_weight =
      context->GetInputShape(kNormWeight)->GetStorageShape();

  if (qkv.GetDimNum() != 3 || z.GetDimNum() != 4 || b.GetDimNum() != 3 ||
      a.GetDimNum() != 3 || conv_weight.GetDimNum() != 2 ||
      conv_state.GetDimNum() != 3 || a_log.GetDimNum() != 1 ||
      dt_bias.GetDimNum() != 1 || ssm_state.GetDimNum() != 4 ||
      accepted.GetDimNum() != 1 || norm_weight.GetDimNum() != 1) {
    return ge::GRAPH_FAILED;
  }

  info.batch_size = qkv.GetDim(0);
  info.state_index_width = read_indices.GetDimNum() == 2 ? 2 : 1;
  const bool split_indices = info.state_index_width == 2;
  const bool valid_indices =
      split_indices ? HasShape(read_indices, {info.batch_size, 2}) &&
                          HasShape(write_indices, {info.batch_size, 2})
                    : HasShape(read_indices, {info.batch_size}) &&
                          HasShape(write_indices, {info.batch_size});
  info.sequence_length = qkv.GetDim(1);
  const int64_t conv_dim = qkv.GetDim(2);
  info.speculative_tokens = info.sequence_length - 1;
  info.conv_state_length = info.sequence_length + 2;
  info.num_v_heads = z.GetDim(2);
  const int64_t qk_width = conv_dim - info.num_v_heads * kHeadDim;

  if (info.batch_size < 1 || info.batch_size > kMaxBatchSize ||
      info.sequence_length < kMinSequenceLength ||
      info.sequence_length > kMaxSequenceLength || conv_dim <= 0 ||
      conv_dim % kHeadDim != 0 || qk_width <= 0 ||
      qk_width % (2 * kHeadDim) != 0 ||
      !HasShape(z,
                {info.batch_size,
                 info.sequence_length,
                 info.num_v_heads,
                 kHeadDim}) ||
      !HasShape(b, {info.batch_size, info.sequence_length, info.num_v_heads}) ||
      !HasShape(a, {info.batch_size, info.sequence_length, info.num_v_heads}) ||
      !HasShape(conv_weight, {4, conv_dim}) || conv_state.GetDim(0) < 1 ||
      (!split_indices && conv_state.GetDim(0) > kMaxNumCacheSlots) ||
      !HasShape(conv_state,
                {conv_state.GetDim(0), info.conv_state_length, conv_dim}) ||
      !HasShape(a_log, {info.num_v_heads}) ||
      !HasShape(dt_bias, {info.num_v_heads}) ||
      !HasShape(ssm_state,
                {split_indices ? ssm_state.GetDim(0)
                               : conv_state.GetDim(0) * info.sequence_length,
                 info.num_v_heads,
                 kHeadDim,
                 kHeadDim}) ||
      !valid_indices || ssm_state.GetDim(0) < 1 ||
      !HasShape(accepted, {info.batch_size}) ||
      !HasShape(norm_weight, {kHeadDim})) {
    return ge::GRAPH_FAILED;
  }

  info.num_k_heads = qk_width / (2 * kHeadDim);
  if (!IsSupportedHeadGeometry(info.num_k_heads, info.num_v_heads)) {
    return ge::GRAPH_FAILED;
  }
  info.conv_tile_count = conv_dim / kHeadDim;
  return ge::GRAPH_SUCCESS;
}

uint64_t GetTilingKey(int64_t speculative_tokens) {
  switch (speculative_tokens) {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 8:
      return static_cast<uint64_t>(100 + speculative_tokens);
    default:
      return 100;
  }
}

}  // namespace

static ge::graphStatus MegaGdnMtpDecodeTiling(gert::TilingContext* context) {
  ShapeInfo info;
  if (GetShapeInfo(context, info) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ub_size = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  if (ub_size < kRequiredUbBytes) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t aic_core_count = platform.GetCoreNumAic();
  const uint32_t aiv_core_count = platform.GetCoreNumAiv();
  const auto* attrs = context->GetAttrs();
  const bool* fla_ssm_state_layout =
      attrs == nullptr ? nullptr
                       : attrs->GetAttrPointer<bool>(kFlaSsmStateLayoutAttr);
  if (fla_ssm_state_layout == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t recurrent_tasks =
      static_cast<uint32_t>(info.batch_size * info.num_v_heads);
  const uint32_t conv_tasks = static_cast<uint32_t>(info.conv_tile_count);
  const bool is_ascend950 =
      platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
  const bool is_qk_group_cache_shape =
      info.speculative_tokens == 8 && info.batch_size == 4 &&
      info.num_k_heads == 8 && info.num_v_heads == 24;
  const bool is_a5_two_owner_qk_group_shape =
      (info.speculative_tokens == 3 || info.speculative_tokens == 8) &&
      info.batch_size == 4 && info.num_k_heads == 8 && info.num_v_heads == 24;
  const uint32_t qk_group_count =
      static_cast<uint32_t>(info.batch_size * info.num_k_heads);
  const uint32_t a5_mixed_aiv_core_count =
      std::min(aiv_core_count, aic_core_count * kAivPerA5MixedBlock);
  const uint32_t schedulable_aiv_core_count =
      is_ascend950 ? a5_mixed_aiv_core_count : aiv_core_count;
  const uint32_t min_two_owner_aiv_cores = (recurrent_tasks + 1) / 2;
  const uint32_t max_useful_two_owner_aiv_cores = 2 * qk_group_count;
  const bool use_a5_two_owner_qk_group =
      *fla_ssm_state_layout && is_ascend950 && is_a5_two_owner_qk_group_shape &&
      schedulable_aiv_core_count >= min_two_owner_aiv_cores &&
      ub_size >= kQkGroupCacheRequiredUbBytes;
  const uint32_t recurrent_dispatch_tasks =
      use_a5_two_owner_qk_group
          ? std::min(recurrent_tasks, max_useful_two_owner_aiv_cores)
          : recurrent_tasks;
  const uint32_t task_count = std::max(conv_tasks, recurrent_dispatch_tasks);
  const uint32_t used_aiv_cores =
      std::min(task_count, schedulable_aiv_core_count);
  const uint32_t block_dim =
      is_ascend950
          ? (used_aiv_cores + kAivPerA5MixedBlock - 1) / kAivPerA5MixedBlock
          : platform.CalcTschBlockDim(
                used_aiv_cores, aic_core_count, aiv_core_count);
  if (block_dim == 0) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t launched_aiv_cores = is_ascend950
                                          ? block_dim * kAivPerA5MixedBlock
                                          : block_dim * kAivPerA2A3MixedBlock;
  const bool use_qk_group_cache =
      *fla_ssm_state_layout && is_qk_group_cache_shape &&
      launched_aiv_cores >= kQkGroupCacheRequiredAivCores &&
      ub_size >= kQkGroupCacheRequiredUbBytes;
  const bool use_deferred_norm =
      *fla_ssm_state_layout &&
      (info.speculative_tokens >= kMinDeferredNormSpeculativeTokens ||
       (is_ascend950 &&
        info.speculative_tokens == kA5DeferredNormSpeculativeTokens)) &&
      ub_size >= kDeferredNormRequiredUbBytes;
  // Keep the row-streaming recurrent experiment isolated from the proven
  // MTP keys.  These are precisely the A5 B4/MTP3 model geometries exercised
  // by Qwen3.5-2B TP1 and Qwen3.5-27B TP1/TP2.  A2/A3, non-FLA layout, other
  // batches/token counts and all other head geometries retain their old key.
  const bool use_a5_regbase_two_owner =
      use_a5_two_owner_qk_group && info.speculative_tokens == 3;
  const bool use_a5_regbase_deferred =
      *fla_ssm_state_layout && is_ascend950 && use_deferred_norm &&
      info.speculative_tokens == 3 && info.batch_size == 4 &&
      info.num_k_heads == 16 &&
      (info.num_v_heads == 16 || info.num_v_heads == 48);
  uint64_t tiling_key = GetTilingKey(info.speculative_tokens);
  if (use_a5_regbase_two_owner) {
    tiling_key = kA5RegBaseTwoOwnerTilingKey;
  } else if (use_a5_regbase_deferred) {
    tiling_key = kA5RegBaseDeferredTilingKey;
  } else if (use_a5_two_owner_qk_group) {
    tiling_key = kA5TwoOwnerQkGroupTilingKeyBase +
                 static_cast<uint64_t>(info.speculative_tokens);
  } else if (use_qk_group_cache) {
    tiling_key = kQkGroupCacheTilingKey;
  } else if (use_deferred_norm) {
    tiling_key = kDeferredNormTilingKeyBase +
                 static_cast<uint64_t>(info.speculative_tokens);
  }
  if (!*fla_ssm_state_layout) {
    tiling_key += kNonFlaTilingKeyOffset;
  }

  MegaGdnMtpDecodeTilingData tiling;
  tiling.set_batch_size(info.batch_size);
  tiling.set_state_index_width(info.state_index_width);
  tiling.set_sequence_length(info.sequence_length);
  tiling.set_num_k_heads(info.num_k_heads);
  tiling.set_num_v_heads(info.num_v_heads);
  const size_t lib_workspace_size = platform.GetLibApiWorkSpaceSize();
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetTilingKey(tiling_key);

  context->SetBlockDim(block_dim);
  if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  size_t* workspace_sizes = context->GetWorkspaceSizes(1);
  workspace_sizes[0] = lib_workspace_size;
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

}  // namespace gdn_a5_backend
