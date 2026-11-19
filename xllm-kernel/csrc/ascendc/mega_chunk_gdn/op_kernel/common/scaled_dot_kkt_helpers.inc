// Shared verbatim by the A2/A3 and A5 KKT implementations.
// Intentionally no include guard: this fragment follows the caller's namespace.
// ── Compile-time constants (set by the JIT compiler from Python) ──────
// D/C stay compile-time because tile shapes depend on them. H/Hg are runtime.
#ifndef GDN_D
#define GDN_D 128  // D = hidden dimension per head
#endif

#ifndef GDN_C
#define GDN_C 128  // C = chunk size (tokens processed per chunk)
#endif

// ── PTO type aliases (device-only, guarded by __CCE_AICORE__) ───────────────
// These are only compiled for the NPU device compiler (__CCE_AICORE__ is
// defined when compiling for AI Core hardware, similar to __CUDA_ARCH__ in
// CUDA).
#ifdef __CCE_AICORE__
// UbND = UB tile in row-major (ND) layout for Vec engine.
//   Think of it as: torch.empty((R, C), dtype=T) in on-chip SRAM.
//   RV, CV = valid region (for dynamic shapes, like a[:valid_rows,
//   :valid_cols]) The Vec engine (SIMD unit) reads/writes these tiles for
//   element-wise ops.
template <typename T,
          int R,
          int C,
          int RV = R,
          int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec,
                       T,
                       R,
                       C,
                       pto::BLayout::RowMajor,
                       RV,
                       CV,
                       pto::SLayout::NoneBox,
                       512,
                       P>;

// UbDN = UB tile in column-major (DN) layout — needed for TROWEXPAND source.
//   TROWEXPAND requires its source vector in column-major (transposed) format.
//   Same physical memory (UB SRAM), just different indexing convention.
template <typename T, int R, int C, int RV = R, int CV = C>
using UbDN = pto::Tile<pto::TileType::Vec,
                       T,
                       R,
                       C,
                       pto::BLayout::ColMajor,
                       RV,
                       CV,
                       pto::SLayout::NoneBox,
                       512>;

// L1Mat = L1 cache tile in NZ fractal format (col-major blocks, row-major
// within).
//   This is the standard input format for the Cube matrix engine.
//   Think of it as a matrix in L1 cache ready for GEMM.
//   NZ = "Normal-Z": the default fractal layout that Cube expects for
//   left/right operands.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1Mat = pto::Tile<pto::TileType::Mat,
                        T,
                        R,
                        C,
                        pto::BLayout::ColMajor,
                        RV,
                        CV,
                        pto::SLayout::RowMajor,
                        512,
                        pto::PadValue::Zero>;

// L1MatZN = L1 tile in ZN fractal format (row-major blocks, col-major within).
//   Used when you need to transpose a matrix before GEMM:
//   TRESHAPE(l1_zn, l1_nz) reinterprets NZ→ZN layout = logical transpose.
//   This is FREE (no data movement) — it just changes how the Cube reads the
//   bits.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1MatZN = pto::Tile<pto::TileType::Mat,
                          T,
                          R,
                          C,
                          pto::BLayout::RowMajor,
                          RV,
                          CV,
                          pto::SLayout::ColMajor,
                          512,
                          pto::PadValue::Zero>;

using GmShape2D = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
using GmStride2D = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;

template <typename T>
using GmTensor2D = pto::GlobalTensor<T, GmShape2D, GmStride2D>;

#endif

template <int32_t ChunkSize>
AICORE inline bool CanReuseGroupKk(int64_t batch_size,
                                   int64_t total_tokens,
                                   uint32_t num_heads,
                                   uint32_t num_key_heads,
                                   uint32_t num_matrices,
                                   __gm__ int32_t* cu_seqlens) {
#ifdef MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_KK
  if (batch_size < 1 || total_tokens <= 0 || cu_seqlens == nullptr ||
#else
  if (batch_size != 1 || total_tokens <= 0 || cu_seqlens == nullptr ||
#endif
      num_key_heads == 0 || num_heads <= num_key_heads ||
      (num_heads % num_key_heads) != 0) {
    return false;
  }

  uint32_t chunk_count = 0;
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t seq_start = static_cast<int64_t>(cu_seqlens[seq_idx]);
    const int64_t seq_end = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
    const int64_t seq_tokens = seq_end - seq_start;
    if (seq_tokens <= 0) {
      return false;
    }
    chunk_count +=
        static_cast<uint32_t>((seq_tokens + ChunkSize - 1) / ChunkSize);
  }

  const uint64_t cache_elements = static_cast<uint64_t>(chunk_count) *
                                  num_key_heads * ChunkSize * ChunkSize;
  const uint64_t workspace_elements =
      static_cast<uint64_t>(total_tokens) * num_heads * ChunkSize;
#ifndef MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_KK
  constexpr uint32_t MinGroupKkChunks = 16;
  if (chunk_count < MinGroupKkChunks) {
    return false;
  }
#endif
  return num_matrices == chunk_count * num_heads &&
         cache_elements <= workspace_elements;
}

template <int32_t ChunkSize>
AICORE inline bool ResolveGroupKkChunk(int64_t global_chunk_idx,
                                       int64_t batch_size,
                                       __gm__ int32_t* cu_seqlens,
                                       int64_t& bos,
                                       int64_t& slen,
                                       int64_t& local_chunk_idx) {
  int64_t accumulated_chunks = 0;
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t seq_start = static_cast<int64_t>(cu_seqlens[seq_idx]);
    const int64_t seq_end = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
    const int64_t seq_tokens = seq_end - seq_start;
    const int64_t seq_chunks = (seq_tokens + ChunkSize - 1) / ChunkSize;
    if (global_chunk_idx < accumulated_chunks + seq_chunks) {
      bos = seq_start;
      slen = seq_tokens;
      local_chunk_idx = global_chunk_idx - accumulated_chunks;
      return true;
    }
    accumulated_chunks += seq_chunks;
  }
  return false;
}

template <int32_t HiddenSize, int32_t ChunkSize>
AICORE inline void BuildGroupKkCache(__gm__ ComputeT* k_handle,
                                     __gm__ ComputeT* kk_cache_handle,
                                     __gm__ int32_t* cu_seqlens,
                                     int64_t batch_size,
                                     uint32_t num_matrices,
                                     uint32_t num_heads,
                                     uint32_t num_key_heads) {
#if defined(__DAV_C220_CUBE__)
  const int32_t h = static_cast<int32_t>(num_heads);
  const int32_t hg = static_cast<int32_t>(num_key_heads);
  if (h <= hg || hg <= 0 || (h % hg) != 0 || batch_size < 1 ||
      cu_seqlens == nullptr || num_matrices % num_heads != 0) {
    return;
  }

  const int64_t chunk_count = num_matrices / num_heads;
  const int64_t total_work = chunk_count * hg;
  const int32_t qk_stride = hg * HiddenSize;
  const int64_t core_id = static_cast<int64_t>(get_block_idx());
  const int64_t core_count = static_cast<int64_t>(get_block_num());

  L1Mat<ComputeT, ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 0);
  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> kk_l0;
  TASSIGN(kk_l0, 0);

  for (int64_t work_idx = core_id; work_idx < total_work;
       work_idx += core_count) {
    const int32_t key_head = static_cast<int32_t>(work_idx % hg);
    const int64_t global_chunk_idx = work_idx / hg;
    int64_t bos = 0;
    int64_t slen = 0;
    int64_t local_chunk_idx = 0;
    if (!ResolveGroupKkChunk<ChunkSize>(global_chunk_idx,
                                        batch_size,
                                        cu_seqlens,
                                        bos,
                                        slen,
                                        local_chunk_idx)) {
      continue;
    }
    const int64_t chunk_start = local_chunk_idx * ChunkSize;
    const int64_t remaining = slen - chunk_start;
    const int32_t valid_rows =
        static_cast<int32_t>(remaining < ChunkSize ? remaining : ChunkSize);
    const int64_t k_offset =
        ((bos + chunk_start) * static_cast<int64_t>(hg) + key_head) *
        HiddenSize;

    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    {
      L1Mat<ComputeT, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> k_load(
          valid_rows, HiddenSize);
      TASSIGN(k_load, 0);
      GmShape2D shape(valid_rows, HiddenSize);
      GmStride2D stride(qk_stride);
      GmTensor2D<ComputeT> k_global(k_handle + k_offset, shape, stride);
      TLOAD(k_load, k_global);
      if (valid_rows != ChunkSize) {
        TFILLPAD(k_load, k_load);
      }
    }

    TileLeft<ComputeT, ChunkSize, HiddenSize, ChunkSize, HiddenSize> l0a;
    TileRight<ComputeT, HiddenSize, ChunkSize, HiddenSize, ChunkSize> l0b;
    TASSIGN(l0a, 0);
    TASSIGN(l0b, 0);
    auto event = EVENT_ID1;
    set_flag(PIPE_MTE2, PIPE_MTE1, event);
    wait_flag(PIPE_MTE2, PIPE_MTE1, event);
    set_flag(PIPE_M, PIPE_MTE1, event);
    wait_flag(PIPE_M, PIPE_MTE1, event);
    TEXTRACT(l0a, k_l1, 0, 0);
    L1MatZN<ComputeT, HiddenSize, ChunkSize> k_zn;
    TRESHAPE(k_zn, k_l1);
    TEXTRACT(l0b, k_zn, 0, 0);
    set_flag(PIPE_MTE1, PIPE_M, event);
    wait_flag(PIPE_MTE1, PIPE_M, event);
    TMATMUL(kk_l0, l0a, l0b);
    set_flag(PIPE_MTE1, PIPE_MTE2, event);
    wait_flag(PIPE_MTE1, PIPE_MTE2, event);
    set_flag(PIPE_M, PIPE_FIX, event);
    wait_flag(PIPE_M, PIPE_FIX, event);

    TileAcc<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> kk_store(ChunkSize,
                                                                    ChunkSize);
    TASSIGN(kk_store, 0);
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
    shape.shape[3] = ChunkSize;
    shape.shape[4] = ChunkSize;
    GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, ChunkSize, 1>>
        kk_global(kk_cache_handle + work_idx * ChunkSize * ChunkSize, shape);
    TSTORE(kk_global, kk_store);
  }
#endif
}
