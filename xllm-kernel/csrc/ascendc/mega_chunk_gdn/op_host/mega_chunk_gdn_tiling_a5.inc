// A5-only host implementation; included after the upstream headers.
namespace gdn_a5_backend {
using namespace ::optiling;

namespace {
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kChunkSize = 128;
constexpr uint32_t kHalfBytes = 2;
constexpr uint64_t kHWorkspacePadBytes = 512;
constexpr uint64_t kHWorkspaceAlignmentBytes = 16 * 1024 * 1024;
constexpr uint64_t kHWorkspaceMaxPhaseBytes = 8 * 1024 * 1024;

enum InputIndex {
  Q_INDEX = 0,
  K_INDEX,
  V_INDEX,
  G_INDEX,
  BETA_INDEX,
  MASK_LOWER_INDEX,
  MASK_FULL_INDEX,
  MINUS_IDENTITY_INDEX,
  CU_SEQLENS_INDEX,
  INITIAL_STATE_INDEX,
};

enum AttrIndex {
  NUM_MATRICES_ATTR = 0,
  HAS_INITIAL_STATE_ATTR,
  FFTS_ADDR_ATTR,
};

bool IsSupportedHeadPair(int64_t valueHeads, int64_t keyHeads) {
  constexpr int64_t kMaxHeads = 64;
  return keyHeads > 0 && valueHeads >= 1 && valueHeads <= kMaxHeads &&
         valueHeads % keyHeads == 0;
}

uint32_t CeilDiv(uint32_t value, uint32_t divisor) {
  return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint64_t CalcUserWorkspaceBytes(uint32_t blockDim) {
  const uint64_t tileBytes =
      static_cast<uint64_t>(kChunkSize) * kChunkSize * kHalfBytes;
  // KKT/WY(4) + H(4) + primary O mailboxes(3) + independent O ping
  // mailboxes(4). Keeping O separate preserves async producer lifetimes.
  constexpr uint64_t kWorkspaceTileCount = 15;
  return static_cast<uint64_t>(blockDim) *
             (kWorkspaceTileCount * tileBytes + 4 * kHWorkspacePadBytes) +
         kHWorkspaceAlignmentBytes + kHWorkspaceMaxPhaseBytes;
}

std::string ShapeToString(const gert::Shape& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.GetDimNum(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << shape.GetDim(i);
  }
  oss << "]";
  return oss.str();
}

ge::graphStatus FailTiling(const std::string& reason) {
  std::cerr << "[MegaChunkGdnTiling] " << reason << std::endl;
  return ge::GRAPH_FAILED;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  auto platformInfo =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const bool isAscend950 =
      platformInfo.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
  uint32_t blockDim = platformInfo.GetCoreNumAic();
  if (blockDim == 0) {
    blockDim = platformInfo.GetCoreNumAiv();
  }
  blockDim = std::max<uint32_t>(blockDim, 1);

  const gert::StorageShape* qShape = context->GetInputShape(Q_INDEX);
  const gert::StorageShape* vShape = context->GetInputShape(V_INDEX);
  const gert::StorageShape* cuShape = context->GetInputShape(CU_SEQLENS_INDEX);
  if (qShape == nullptr || vShape == nullptr || cuShape == nullptr) {
    return FailTiling("missing q/v/cu_seqlens shape");
  }

  const auto& qOriginShape = qShape->GetOriginShape();
  const auto& vOriginShape = vShape->GetOriginShape();
  const auto& cuOriginShape = cuShape->GetOriginShape();
  if (qOriginShape.GetDimNum() != 4 || vOriginShape.GetDimNum() != 4 ||
      qOriginShape.GetDim(3) != kHeadDim ||
      vOriginShape.GetDim(3) != kHeadDim) {
    return FailTiling("invalid q/v shape q=" + ShapeToString(qOriginShape) +
                      " v=" + ShapeToString(vOriginShape) +
                      " cu=" + ShapeToString(cuOriginShape));
  }
  const uint32_t totalTokens = static_cast<uint32_t>(qOriginShape.GetDim(1));
  const uint32_t numKeyHeads = static_cast<uint32_t>(qOriginShape.GetDim(2));
  const uint32_t numHeads = static_cast<uint32_t>(vOriginShape.GetDim(2));
  if (!IsSupportedHeadPair(numHeads, numKeyHeads)) {
    return FailTiling("unsupported head pair H=" + std::to_string(numHeads) +
                      " Hg=" + std::to_string(numKeyHeads) +
                      " q=" + ShapeToString(qOriginShape) +
                      " v=" + ShapeToString(vOriginShape));
  }
  const int64_t batchSize = cuOriginShape.GetDim(0) - 1;

  int64_t numMatricesAttr = 0;
  auto attrs = context->GetAttrs();
  if (attrs != nullptr &&
      attrs->GetAttrPointer<int64_t>(NUM_MATRICES_ATTR) != nullptr) {
    numMatricesAttr = *attrs->GetAttrPointer<int64_t>(NUM_MATRICES_ATTR);
  }
  int64_t hasInitialState = 0;
  if (attrs != nullptr &&
      attrs->GetAttrPointer<bool>(HAS_INITIAL_STATE_ATTR) != nullptr) {
    hasInitialState =
        *attrs->GetAttrPointer<bool>(HAS_INITIAL_STATE_ATTR) ? 1 : 0;
  }
  uint64_t fftsAddr = 0;
  if (attrs != nullptr &&
      attrs->GetAttrPointer<int64_t>(FFTS_ADDR_ATTR) != nullptr) {
    fftsAddr =
        static_cast<uint64_t>(*attrs->GetAttrPointer<int64_t>(FFTS_ADDR_ATTR));
  }

  uint32_t numMatrices = 0;
  if (numMatricesAttr > 0 &&
      numMatricesAttr <= std::numeric_limits<uint32_t>::max()) {
    numMatrices = static_cast<uint32_t>(numMatricesAttr);
  }
  if (numMatrices == 0) {
    numMatrices = CeilDiv(totalTokens, kChunkSize) * numHeads;
  }
  if (isAscend950) {
    // Keep every chunk/sequence of a head on the same MIX block.  The A5
    // Cube/UB tiled stages use local intra-block events and per-block
    // workspaces, so stable head ownership avoids a software all-core
    // barrier between fused stages.  Since work indices advance in
    // multiples of numHeads, any divisor of numHeads preserves ownership.
    const uint32_t maxBlocks =
        std::min<uint32_t>(blockDim, std::max<uint32_t>(numMatrices, 1));
    blockDim = std::min<uint32_t>(std::max<uint32_t>(numHeads, 1), maxBlocks);
    while (blockDim > 1 && (numHeads % blockDim) != 0) {
      --blockDim;
    }
  }

  MegaChunkGdnTilingData tiling;
  tiling.set_block_dim(blockDim);
  tiling.set_num_matrices(numMatrices);
  tiling.set_num_heads(numHeads);
  tiling.set_num_key_heads(numKeyHeads);
  tiling.set_has_initial_state(hasInitialState);
  tiling.set_batch_size(batchSize);
  tiling.set_seq_len(totalTokens);
  tiling.set_total_tokens(totalTokens);
  tiling.set_ffts_addr(fftsAddr);

  context->SetBlockDim(blockDim);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  size_t* workspaceSizes = context->GetWorkspaceSizes(1);
  if (workspaceSizes == nullptr) {
    return FailTiling(
        "workspace size pointer is null H=" + std::to_string(numHeads) +
        " Hg=" + std::to_string(numKeyHeads) +
        " T=" + std::to_string(totalTokens) +
        " num_matrices=" + std::to_string(numMatrices));
  }
  workspaceSizes[0] =
      CalcUserWorkspaceBytes(blockDim) + platformInfo.GetLibApiWorkSpaceSize();
  return ge::GRAPH_SUCCESS;
}

struct MegaChunkGdnCompileInfo {};

static ge::graphStatus TilingParseForMegaChunkGdn(
    gert::TilingParseContext* context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

}  // namespace gdn_a5_backend
