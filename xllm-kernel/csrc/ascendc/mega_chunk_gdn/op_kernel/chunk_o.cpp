// ============================================================================
// chunk_o_kernel.cpp — Output computation for GatedDeltaNet (chunk-wise)
//
// Mathematical operation (per chunk of C tokens, per head h):
//
//   O = (QK_gated @ V) + exp(g) * (Q @ S)
//     = intra_chunk_attention + inter_chunk_state_contribution
//
// where:
//   Q, K, V ∈ ℝ^{C×D}    — query/key/value projections for this chunk
//   S ∈ ℝ^{D×D}           — accumulated hidden state entering this chunk
//   G ∈ ℝ^{C}             — cumulative gate values (pre-transposed [H,T])
//   Msk ∈ ℝ^{C×C}         — lower-triangular causal mask
//
// Cube phase (3 GEMMs per chunk):
//   1. QK   = Q @ K^T         — intra-chunk attention scores
//   2. QS   = Q @ S           — query applied to accumulated state
//   3. QKV  = QK_gated @ V    — gated attention applied to values
//
// Vec phase (two sub-blocks process upper/lower C/2 rows):
//   a. Load G → compute gating coefficients:
//        coeff[i,j] = exp(min(g[i] - g[j], 0)) * mask[i,j]
//   b. Apply gating to QK: QK_gated = QK * coeff
//   c. Scale QS by exp(g): QS_gated = QS * exp(g_row)
//   d. Combine: O = QS_gated + QKV
//   e. Store O to GM in BSND layout
//
// Cross-core sync protocol (Cube ↔ Vec via FFTS):
//   flag 0: Cube→Vec  — QK and QS results ready in workspace
//   flag 1: Vec→Cube  — QK_gated written back, Cube can proceed to GEMM 3
//   flag 2: Cube→Vec  — QKV result ready in workspace
//   flag 3: Vec→Cube  — Vec done with this chunk, Cube can reuse workspace
//
// NPU memory hierarchy used:
//   GM → L1 (Cube-accessible) → L0A/L0B (matrix engines) → L0C (accumulator)
//   GM → UB (Vec-accessible, on-chip SRAM)
//
// ── PTO / NPU Primer ──────────────────────────────────────────────────
// This kernel combines matrix multiplication (Cube) with element-wise gating
// (Vec) in a tightly coordinated 3-GEMM + gating pipeline per chunk.
//
// Execution timeline for one chunk:
//   Cube: GEMM1(Q@K^T) → GEMM2(Q@S) → store QK,QS → signal Vec ──────┐
//   Vec:  (meanwhile) load G, compute gating coefficients                │
//   Vec:  ←── wait for Cube signal ──── apply gating to QK → QK_gated  │
//   Vec:  store QK_gated → signal Cube ────────────────────────────────┐│
//   Cube: ←── wait for Vec signal ──── GEMM3(QK_gated@V) → store QKV ─┘│
//   Vec:  ←── wait for Cube signal ──── scale QS, combine O=QKV+QS_g   │
//   Vec:  store O → signal Cube "done" ─────────────────────────────────┘
//
// numpy pseudocode for the entire chunk computation:
//   QK = Q @ K.T                                          # GEMM 1
//   QS = Q @ S                                            # GEMM 2
//   coeff = exp(min(g_row - g_col, 0)) * mask             # gating (dynamic
//   PTO)
//   (``static_baseline/run_chunk_o_static.py`` uses exp(g_row-g_col) without
//   min.) QK_gated = QK * coeff                                 # apply gating
//   QKV = QK_gated @ V                                    # GEMM 3
//   O = QKV + QS * np.exp(g_row).reshape(-1, 1)           # final output
//
// Key PTO APIs (with numpy/torch equivalents):
//   TLOAD(dst, gm)          — dst = gm_data      (DMA: GM→UB/L1, async)
//   TSTORE(gm, src)         — gm = src            (DMA: UB/L0C→GM, async)
//   TASSIGN(tile, addr)     — bind tile descriptor to buffer address
//   TCVT(dst, src, mode)    — converts between float and ComputeT
//   TMOV(dst, src)          — copy: dst = src.clone()
//   TADD(d, a, b)           — d = a + b
//   TSUB(d, a, b)           — d = a - b
//   TMUL(d, a, b)           — d = a * b
//   TMINS(d, s, val)        — d = torch.clamp(s, max=val)
//   TEXP(d, s)              — d = torch.exp(s)
//   TROWEXPAND(2d, col)     — 2d[i,j] = col[i] (broadcast column→rows)
//   TCOLEXPAND(2d, row)     — 2d[i,j] = row[j] (broadcast row→columns)
//   TEXTRACT(l0, l1, r, c)  — copy L1 sub-tile → L0A/L0B (Cube input regs)
//   TRESHAPE(zn, nz)        — reinterpret L1 fractal layout (transpose, free)
//   TMATMUL(C, A, B)        — C = A @ B (Cube engine, ComputeT→FP32 accum)
//   set_flag / wait_flag    — synchronize pipes within same AI core
//   ffts_cross_core_sync    — signal across Cube↔Vec cores
//   wait_flag_dev(flag)     — wait for cross-core signal
// ============================================================================

#include <pto/pto-inst.hpp>

#include "acl/acl.h"
#include "gdn_sync.h"
using namespace pto;

// ── Compile-time configuration (overridable at build time via -D flags) ──
// D/C stay compile-time because tile shapes depend on them. H/Hg are runtime.
#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

#ifndef GDN_MAX_HEADS
#define GDN_MAX_HEADS 64
#endif

// ── PTO type aliases (device-only, guarded for host pass safety) ────────────
// The bisheng compiler performs 3 passes: vec core, cube core (__CCE_AICORE__
// defined), and host (__CCE_AICORE__ NOT defined). Type aliases using PTO
// tile types must be guarded so the host pass never sees them.
#ifdef __CCE_AICORE__

// UbND = Unified Buffer tile, row-major (ND) layout, for Vec SIMD ops.
//   Like torch.empty((R, C), dtype=T) in fast on-chip SRAM (~256KB).
//   RV, CV = valid region (handles dynamic shapes, partial chunks).
//   PadValue::Zero = fill with 0 outside valid region during TLOAD.
// T=dtype, R×C=static shape, RV×CV=valid region, P=pad fill for TLOAD.
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

// UbDN = UB tile in column-major (DN) layout.
//   Needed as source for TROWEXPAND which requires column-format input.
//   TROWEXPAND takes a column vector and broadcasts it across all columns
//   of a destination ND tile: dst[i,j] = col[i] for all j.
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

// L1Mat = L1 cache tile in NZ fractal format — standard Cube GEMM input.
//   Data is loaded here from GM via TLOAD, then fed to L0A/L0B via TEXTRACT.
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

// L1MatZN = ZN fractal format — used for transposed GEMM operands.
//   TRESHAPE(l1_zn, l1_nz) converts NZ→ZN = logical matrix transpose (free, no
//   data movement).
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

#endif  // __CCE_AICORE__

#if defined(__DAV_C220_CUBE__)
#define GDN_CHUNK_O_KERNEL chunk_o_kernel_aic
#elif defined(__DAV_C220_VEC__)
#define GDN_CHUNK_O_KERNEL chunk_o_kernel_aiv
#else
#define GDN_CHUNK_O_KERNEL chunk_o_kernel
#endif

template <int32_t ChunkSize>
AICORE inline bool CanReuseGroupQk(int64_t batch_size,
                                   int64_t total_tokens,
                                   uint32_t num_heads,
                                   uint32_t num_key_heads,
                                   uint32_t num_matrices,
                                   __gm__ int32_t* cu_seqlens) {
#ifdef MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_QK
  if (batch_size < 1 || total_tokens <= 0 || cu_seqlens == nullptr ||
#else
  if (batch_size != 1 || total_tokens <= 0 || cu_seqlens == nullptr ||
#endif
      num_key_heads == 0 || num_heads <= num_key_heads ||
      (num_heads % num_key_heads) != 0) {
    return false;
  }
#ifdef MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_QK
  const uint32_t group = num_heads / num_key_heads;
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE
  // Variant17 initially validates only the Qwen3.5 value/key-head ratio.
  if (group != 3) {
    return false;
  }
#else
  if (group < 2 || group > 3) {
    return false;
  }
#endif
  uint32_t chunk_count = 0;
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t seq_start = static_cast<int64_t>(cu_seqlens[seq_idx]);
    const int64_t seq_end = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
    const int64_t seq_tokens = seq_end - seq_start;
    if (seq_tokens <= 0) {
      return false;
    }
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE
    const uint32_t sequence_chunk_count =
        static_cast<uint32_t>((seq_tokens + ChunkSize - 1) / ChunkSize);
    // A5's precomputed-QS storage supports 64 chunks per sequence. Multiple
    // sequences may each consume that full range in the same launch.
    if (sequence_chunk_count > 64) {
      return false;
    }
    chunk_count += sequence_chunk_count;
#else
    chunk_count +=
        static_cast<uint32_t>((seq_tokens + ChunkSize - 1) / ChunkSize);
#endif
  }
#else
  // The group-mailbox schedule is tuned for the Qwen3.5-27B TP2 mapping.
  // Other valid head layouts use the per-head H/O pipeline below.
  if (num_heads != 24 || num_key_heads != 8) {
    return false;
  }
  const uint32_t chunk_count =
      static_cast<uint32_t>((total_tokens + ChunkSize - 1) / ChunkSize);
#endif
  const uint64_t group_work_count =
      static_cast<uint64_t>(chunk_count) * num_key_heads;
  return num_matrices == chunk_count * num_heads &&
         group_work_count >= get_block_num();
}

#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
// Variant18 keeps group-QK synchronization on the producing/consuming DMA
// pipes. Unlike gdn_sync::Signal/Wait, these helpers intentionally do not
// inject PIPE_ALL barriers. They are instantiated only by the group branch.
template <pipe_t ProducerPipe>
AICORE inline void GroupDirectedSignal(uint16_t config) {
  const uint16_t event_id =
      (config >> gdn_sync::kEventIdOffset) & gdn_sync::kEventIdMask;
#if defined(__DAV_CUBE__)
  set_intra_block(ProducerPipe, event_id);
  set_intra_block(ProducerPipe, event_id + gdn_sync::kVecCoreIdOffset);
#elif defined(__DAV_VEC__)
  set_intra_block(ProducerPipe, event_id);
#endif
}

template <pipe_t ConsumerPipe>
AICORE inline void GroupDirectedWait(uint16_t event_id) {
  wait_intra_block(ConsumerPipe, event_id);
#if defined(__DAV_CUBE__)
  wait_intra_block(ConsumerPipe, event_id + gdn_sync::kVecCoreIdOffset);
#endif
}

template <bool DirectedSync, pipe_t ProducerPipe>
AICORE inline void GroupSignal(uint16_t config) {
  if constexpr (DirectedSync) {
    GroupDirectedSignal<ProducerPipe>(config);
  } else {
    gdn_sync::Signal<ProducerPipe>(config);
  }
}

template <bool DirectedSync, pipe_t ConsumerPipe>
AICORE inline void GroupWait(uint16_t event_id) {
  if constexpr (DirectedSync) {
    GroupDirectedWait<ConsumerPipe>(event_id);
  } else {
    gdn_sync::Wait<ConsumerPipe>(event_id);
  }
}
#endif

template <int32_t ChunkSize>
AICORE inline bool ResolveHoGlobalChunk(int64_t global_chunk_idx,
                                        int64_t batch_size,
                                        __gm__ int32_t* cu_seqlens,
                                        int64_t& seq_idx,
                                        int64_t& bos,
                                        int64_t& slen,
                                        int64_t& local_chunk_idx) {
  int64_t accumulated_chunks = 0;
  for (int64_t current_seq = 0; current_seq < batch_size; ++current_seq) {
    const int64_t seq_start = static_cast<int64_t>(cu_seqlens[current_seq]);
    const int64_t seq_end = static_cast<int64_t>(cu_seqlens[current_seq + 1]);
    const int64_t seq_tokens = seq_end - seq_start;
    const int64_t seq_chunks = (seq_tokens + ChunkSize - 1) / ChunkSize;
    if (global_chunk_idx < accumulated_chunks + seq_chunks) {
      seq_idx = current_seq;
      bos = seq_start;
      slen = seq_tokens;
      local_chunk_idx = global_chunk_idx - accumulated_chunks;
      return true;
    }
    accumulated_chunks += seq_chunks;
  }
  return false;
}

#if defined(PTO_NPU_ARCH_A2A3)
template <int32_t Rows, typename TileDataOut, typename TileDataIn>
__tf__ PTO_INTERNAL void ReduceRowsFp32Normal(TileDataOut& dst,
                                              TileDataIn& src) {
  __ubuf__ float* dst_ptr =
      reinterpret_cast<__ubuf__ float*>(__cce_get_tile_ptr(dst.data()));
  __ubuf__ float* src_ptr =
      reinterpret_cast<__ubuf__ float*>(__cce_get_tile_ptr(src.data()));
  vcadd(dst_ptr, src_ptr, Rows, 1, 1, 8, false);
}
#endif

template <int32_t HiddenSize,
          int32_t Rows,
          int32_t OutputAddr,
          int32_t SquareAddr,
          int32_t ReduceTmpAddr,
          int32_t RowSumAddr,
          int32_t RowScaleAddr>
AICORE inline void NormalizeRmsRows() {
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
  UbND<float, Rows, HiddenSize> output_fp32;
  TASSIGN(output_fp32, OutputAddr);
  UbND<float, Rows, HiddenSize> square;
  TASSIGN(square, SquareAddr);
  UbND<float, Rows, HiddenSize> reduce_tmp;
  TASSIGN(reduce_tmp, ReduceTmpAddr);
  UbDN<float, Rows, 1> row_sum_col;
  TASSIGN(row_sum_col, RowSumAddr);
  UbND<float, 1, Rows> row_sum_vec;
  TASSIGN(row_sum_vec, RowSumAddr);
  UbND<float, Rows, 8> row_scale_blocks;
  TASSIGN(row_scale_blocks, RowScaleAddr);

  TMUL(square, output_fp32, output_fp32);
  pipe_barrier(PIPE_V);
  if constexpr (HiddenSize == 128) {
    // Match LayerNormFwd's N=128 reduction tree: add the two 64-wide
    // halves elementwise, then reduce the resulting 64 values.  The source
    // tiles retain a physical row stride of 128 while exposing 64 valid
    // columns, so PTO emits one strided vector add for all rows.
    UbND<float, Rows, HiddenSize, Rows, HiddenSize / 2> square_low;
    TASSIGN(square_low, SquareAddr);
    UbND<float, Rows, HiddenSize, Rows, HiddenSize / 2> square_high;
    TASSIGN(square_high, SquareAddr + HiddenSize / 2 * sizeof(float));
    UbND<float, Rows, HiddenSize / 2> reduced_halves;
    TASSIGN(reduced_halves, ReduceTmpAddr);
    TADD(reduced_halves, square_low, square_high);
    pipe_barrier(PIPE_V);
#if defined(PTO_NPU_ARCH_A2A3)
    // PTO's 64-wide TROWSUM selects count mode on A2/A3. LayerNormFwd uses
    // normal mode with one explicit repeat per row, so issue the same vcadd
    // form to preserve its reduction tree bit for bit.
    set_mask_norm();
    set_vector_mask(-1, -1);
    ReduceRowsFp32Normal<Rows>(row_sum_col, reduced_halves);
#else
    TROWSUM(row_sum_col, reduced_halves, square);
#endif
  } else {
    TROWSUM(row_sum_col, square, reduce_tmp);
  }
  pipe_barrier(PIPE_V);
  TRESHAPE(row_sum_vec, row_sum_col);
  TMULS(row_sum_vec, row_sum_vec, 1.0f / static_cast<float>(HiddenSize));
  pipe_barrier(PIPE_V);
  TADDS(row_sum_vec, row_sum_vec, 1.0e-6f);
  pipe_barrier(PIPE_V);
  TSQRT(row_sum_vec, row_sum_vec);
  pipe_barrier(PIPE_V);
  UbND<float, 1, Rows> reciprocal_numerator;
  TASSIGN(reciprocal_numerator, ReduceTmpAddr);
  TMULS(reciprocal_numerator, row_sum_vec, 0.0f);
  pipe_barrier(PIPE_V);
  TADDS(reciprocal_numerator, reciprocal_numerator, 1.0f);
  pipe_barrier(PIPE_V);
  TDIV(row_sum_vec, reciprocal_numerator, row_sum_vec);
  pipe_barrier(PIPE_V);
  TRESHAPE(row_sum_col, row_sum_vec);
  TROWEXPAND(row_scale_blocks, row_sum_col);
  pipe_barrier(PIPE_V);
  TROWEXPANDMUL(output_fp32, output_fp32, row_scale_blocks);
  pipe_barrier(PIPE_V);
#endif
}

template <int32_t HiddenSize, int32_t ChunkSize, bool FuseGatedRmsNorm>
AICORE inline void StoreChunkOutput(__gm__ GDN_PUBLIC_DTYPE* output_handle,
                                    __gm__ GDN_PUBLIC_DTYPE* z_handle,
                                    int64_t output_offset,
                                    int32_t row_stride,
                                    int32_t local_rows) {
#if defined(__DAV_C220_VEC__)
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t OutputFp32Addr = 33280;
  // The optimized 64x128 TROWSUM path needs a 512-byte-aligned result
  // reservation even though the logical 64x1 FP32 tile is only 256 bytes.
  constexpr int32_t RowSumAddr = 182272;
  constexpr int32_t RowRstdAddr = 182784;
  constexpr int32_t ReduceTmpAddr = 66304;
  constexpr int32_t ZBf16Addr = 99072;
  constexpr int32_t SquareAddr = 131840;
  constexpr int32_t OutputComputeAddr = 115456;
  constexpr int32_t OutputPublicAddr =
      std::is_same_v<ComputeT, GDN_PUBLIC_DTYPE> ? OutputComputeAddr : 164608;
  constexpr int32_t NormWeightFp32Addr = 181504;
  // torch FP16 tensor-scalar Mul converts 1/sqrt(128) to FP16 first.
  constexpr float OutputScale = 0.08837890625f;

  UbND<float, HalfChunk, HiddenSize> output_fp32;
  TASSIGN(output_fp32, OutputFp32Addr);
  UbND<ComputeT, HalfChunk, HiddenSize, HalfChunk, HiddenSize, PadValue::Zero>
      output_compute;
  TASSIGN(output_compute, OutputComputeAddr);
  UbND<GDN_PUBLIC_DTYPE,
       HalfChunk,
       HiddenSize,
       HalfChunk,
       HiddenSize,
       PadValue::Zero>
      output_public;
  TASSIGN(output_public, OutputPublicAddr);

  // Match the standalone XllmKernelMegaChunkGdn compute boundary before
  // scaling.
#if defined(PTO_NPU_ARCH_A5)
  // The caller has just produced output_fp32 with TADD.  C220's compiler
  // inserts the required Vector dependency, whereas C310 needs the explicit
  // PR #41-style barrier before the conversion can consume that tile.
  pipe_barrier(PIPE_V);
#endif
  TCVT(output_compute, output_fp32, pto::RoundMode::CAST_NONE);

  if constexpr (FuseGatedRmsNorm) {
    {
      Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
      shape.shape[3] = local_rows;
      shape.shape[4] = HiddenSize;
      GmStride2D stride(row_stride);
      GmTensor2D<GDN_PUBLIC_DTYPE> z_global(
          z_handle + output_offset, shape, stride);
      UbND<GDN_PUBLIC_DTYPE,
           HalfChunk,
           HiddenSize,
           DYNAMIC,
           DYNAMIC,
           PadValue::Zero>
          z_load(local_rows, HiddenSize);
      TASSIGN(z_load, ZBf16Addr);
      TLOAD(z_load, z_global);
      if (local_rows != HalfChunk) {
        UbND<GDN_PUBLIC_DTYPE,
             HalfChunk,
             HiddenSize,
             HalfChunk,
             HiddenSize,
             PadValue::Zero>
            z_padded;
        TASSIGN(z_padded, ZBf16Addr);
        TFILLPAD_INPLACE(z_padded, z_load);
      }
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    pipe_barrier(PIPE_V);
    TCVT(output_fp32, output_compute, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    TMULS(output_fp32, output_fp32, OutputScale);
    pipe_barrier(PIPE_V);
    TCVT(output_compute, output_fp32, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    TCVT(output_fp32, output_compute, pto::RoundMode::CAST_NONE);
    if constexpr (!std::is_same_v<ComputeT, GDN_PUBLIC_DTYPE>) {
      // The standalone path returns the scaled FP16 result as BF16 before
      // the model converts it back to FP32 for gated RMSNorm.
      pipe_barrier(PIPE_V);
      TCVT(output_public, output_fp32, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TCVT(output_fp32, output_public, pto::RoundMode::CAST_NONE);
    }

    NormalizeRmsRows<HiddenSize,
                     HalfChunk,
                     OutputFp32Addr,
                     SquareAddr,
                     ReduceTmpAddr,
                     RowSumAddr,
                     RowRstdAddr>();

    UbND<float, 1, HiddenSize> norm_weight_fp32;
    TASSIGN(norm_weight_fp32, NormWeightFp32Addr);
    TCOLEXPANDMUL(output_fp32, output_fp32, norm_weight_fp32);

    UbND<float, HalfChunk, HiddenSize> scratch;
    TASSIGN(scratch, SquareAddr);
    UbND<float, HalfChunk, HiddenSize> reduce_tmp;
    TASSIGN(reduce_tmp, ReduceTmpAddr);

    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    UbND<GDN_PUBLIC_DTYPE,
         HalfChunk,
         HiddenSize,
         HalfChunk,
         HiddenSize,
         PadValue::Zero>
        z_public;
    TASSIGN(z_public, ZBf16Addr);
    TCVT(scratch, z_public, pto::RoundMode::CAST_NONE);
#if defined(PTO_NPU_ARCH_A5)
    pipe_barrier(PIPE_V);
#endif
    TNEG(reduce_tmp, scratch);
#if defined(PTO_NPU_ARCH_A5)
    pipe_barrier(PIPE_V);
#endif
    TEXP(reduce_tmp, reduce_tmp);
#if defined(PTO_NPU_ARCH_A5)
    pipe_barrier(PIPE_V);
#endif
    TADDS(reduce_tmp, reduce_tmp, 1.0f);
    pipe_barrier(PIPE_V);

    TMUL(output_fp32, output_fp32, scratch);
#if defined(PTO_NPU_ARCH_A5)
    pipe_barrier(PIPE_V);
#endif
    TDIV(output_fp32, output_fp32, reduce_tmp);
#if defined(PTO_NPU_ARCH_A5)
    pipe_barrier(PIPE_V);
#endif
    TCVT(output_public, output_fp32, pto::RoundMode::CAST_ROUND);
  } else if constexpr (!std::is_same_v<ComputeT, GDN_PUBLIC_DTYPE>) {
    pipe_barrier(PIPE_V);
    TCVT(output_fp32, output_compute, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    TCVT(output_public, output_fp32, pto::RoundMode::CAST_NONE);
  }

  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  {
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
    shape.shape[3] = local_rows;
    shape.shape[4] = HiddenSize;
    GmStride2D stride(row_stride);
    GmTensor2D<GDN_PUBLIC_DTYPE> output_global(
        output_handle + output_offset, shape, stride);
    UbND<GDN_PUBLIC_DTYPE, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC>
        output_store(local_rows, HiddenSize);
    TASSIGN(output_store, OutputPublicAddr);
    TSTORE(output_global, output_store);
  }
#endif
}

AICORE inline int64_t GetHoConsumerOwner(int64_t global_item,
                                         int64_t block_num,
                                         bool rebalance_consumers,
                                         int64_t balanced_span,
                                         int64_t heavy_core_count,
                                         int64_t light_core_count) {
  int64_t owner = global_item % block_num;
  if (rebalance_consumers && global_item >= balanced_span) {
    owner = heavy_core_count + (global_item - balanced_span) % light_core_count;
  }
  return owner;
}

AICORE inline int64_t GetHoOwnedItem(int64_t owned_idx,
                                     int64_t total_items,
                                     int64_t core_id,
                                     int64_t block_num,
                                     bool rebalance_consumers,
                                     int64_t balanced_span,
                                     int64_t heavy_core_count,
                                     int64_t light_core_count,
                                     int64_t prefix_count,
                                     int64_t tail_owner_shift) {
  if (!rebalance_consumers || owned_idx < prefix_count) {
    const int64_t item = core_id + owned_idx * block_num;
    return item < total_items ? item : total_items;
  }
  if (core_id < heavy_core_count || light_core_count <= 0) {
    return total_items;
  }
  const int64_t tail_round = owned_idx - prefix_count;
  const int64_t tail_round_start =
      balanced_span + tail_round * light_core_count;
  int64_t tail_rank = core_id - heavy_core_count - tail_owner_shift;
  if (tail_rank < 0) {
    tail_rank += light_core_count;
  }
  const int64_t item =
      balanced_span + tail_rank + tail_round * light_core_count;
  return item < total_items ? item : total_items;
}

template <int32_t ChunkSize>
AICORE inline void WaitHoChunkReady(__gm__ int32_t* ready_handle,
                                    int32_t head_idx,
                                    int64_t chunk_idx) {
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
  constexpr int64_t ReadyStride = 16;
  const int64_t ready_offset = static_cast<int64_t>(head_idx) * ReadyStride;
  __gm__ int64_t* ready_cacheline =
      reinterpret_cast<__gm__ int64_t*>(ready_handle + ready_offset);
  __gm__ volatile int32_t* ready_ptr = ready_handle + ready_offset;
  const int32_t required_count = static_cast<int32_t>(chunk_idx + 1);
  while (true) {
    dcci(ready_cacheline,
         cache_line_t::SINGLE_CACHE_LINE,
         dcci_dst_t::CACHELINE_OUT);
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
    // Acquire the launch-local doorbell value before testing it. The matching
    // producer release orders QS and both V_new stripes before this count.
    dsb(DSB_DDR);
#endif
    const int32_t ready_count = *ready_ptr;
    if (ready_count >= required_count) {
      break;
    }
  }
#endif
}

template <int32_t HiddenSize,
          int32_t ChunkSize
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
          ,
          bool DirectedSync = false
#endif
          >
AICORE inline void PublishQKTile(__gm__ ComputeT* q_handle,
                                 __gm__ ComputeT* k_handle,
                                 __gm__ ComputeT* qk_mailbox,
                                 int64_t core_id,
                                 int64_t qk_offset,
                                 int32_t qk_stride,
                                 int32_t valid_rows,
                                 int32_t qk_ready_flag,
                                 uint32_t strong_publish) {
#if defined(__DAV_C220_CUBE__)
  L1Mat<ComputeT, ChunkSize, HiddenSize> q_l1;
  TASSIGN(q_l1, 0);
  L1Mat<ComputeT, ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 32768);
  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> qk_l0;
  TASSIGN(qk_l0, 0);

  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

  {
    L1Mat<ComputeT, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> q_load(valid_rows,
                                                                    HiddenSize);
    TASSIGN(q_load, 0);
    GmShape2D shape(valid_rows, HiddenSize);
    GmStride2D stride(qk_stride);
    GmTensor2D<ComputeT> q_global(q_handle + qk_offset, shape, stride);
    TLOAD(q_load, q_global);
    if (valid_rows != ChunkSize) {
      TFILLPAD(q_load, q_load);
    }
  }
  {
    L1Mat<ComputeT, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> k_load(valid_rows,
                                                                    HiddenSize);
    TASSIGN(k_load, 32768);
    GmShape2D shape(valid_rows, HiddenSize);
    GmStride2D stride(qk_stride);
    GmTensor2D<ComputeT> k_global(k_handle + qk_offset, shape, stride);
    TLOAD(k_load, k_global);
    if (valid_rows != ChunkSize) {
      TFILLPAD(k_load, k_load);
    }
  }

  {
    TileLeft<ComputeT, ChunkSize, HiddenSize, ChunkSize, HiddenSize> l0a;
    TileRight<ComputeT, HiddenSize, ChunkSize, HiddenSize, ChunkSize> l0b;
    TASSIGN(l0a, 0x0);
    TASSIGN(l0b, 0x0);
    auto event = EVENT_ID1;
    set_flag(PIPE_MTE2, PIPE_MTE1, event);
    wait_flag(PIPE_MTE2, PIPE_MTE1, event);
    set_flag(PIPE_M, PIPE_MTE1, event);
    wait_flag(PIPE_M, PIPE_MTE1, event);
    TEXTRACT(l0a, q_l1, 0, 0);
    L1MatZN<ComputeT, HiddenSize, ChunkSize> k_zn;
    TRESHAPE(k_zn, k_l1);
    TEXTRACT(l0b, k_zn, 0, 0);
    set_flag(PIPE_MTE1, PIPE_M, event);
    wait_flag(PIPE_MTE1, PIPE_M, event);
    TMATMUL(qk_l0, l0a, l0b);
    set_flag(PIPE_MTE1, PIPE_MTE2, event);
    wait_flag(PIPE_MTE1, PIPE_MTE2, event);
    set_flag(PIPE_M, PIPE_FIX, event);
    wait_flag(PIPE_M, PIPE_FIX, event);
  }

  {
    TileAcc<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> qk_store(ChunkSize,
                                                                    ChunkSize);
    TASSIGN(qk_store, 0);
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
    shape.shape[3] = ChunkSize;
    shape.shape[4] = ChunkSize;
    GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, ChunkSize, 1>>
        qk_global(
            qk_mailbox + core_id * static_cast<int64_t>(ChunkSize) * ChunkSize,
            shape);
    TSTORE(qk_global, qk_store);
  }
  if (strong_publish != 0) {
    pipe_barrier(PIPE_ALL);
  }
  if (qk_ready_flag >= 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
    GroupSignal<DirectedSync, PIPE_FIX>(1 | (2 << 4) | (qk_ready_flag << 8));
#else
    ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (qk_ready_flag << 8));
#endif
  }
#endif
}

template <int32_t HiddenSize,
          int32_t ChunkSize
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
          ,
          bool DirectedSync = false
#endif
          >
AICORE inline void PublishQKVTile(__gm__ ComputeT* v_handle,
                                  __gm__ ComputeT* qk_gated_mailbox,
                                  __gm__ ComputeT* qkv_mailbox,
                                  int64_t core_id,
                                  int64_t v_offset,
                                  int32_t v_stride,
                                  int32_t valid_rows,
                                  int32_t qk_gated_ready_flag,
                                  int32_t qkv_ready_flag,
                                  int32_t qk_gated_bottom_ready_flag = -1,
                                  int32_t qkv_bottom_ready_flag = -1,
                                  uint32_t wait_before_v_load = 0u,
                                  int32_t qk_gated_bottom_tail_ready_flag = -1,
                                  int32_t qkv_bottom_tail_ready_flag = -1) {
#if defined(__DAV_C220_CUBE__)
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t QuarterChunk = HalfChunk / 2;
  const bool split_rows =
      qk_gated_bottom_ready_flag >= 0 && qkv_bottom_ready_flag >= 0;
  const bool split_bottom_quarters = split_rows &&
                                     qk_gated_bottom_tail_ready_flag >= 0 &&
                                     qkv_bottom_tail_ready_flag >= 0;
  L1Mat<ComputeT, HalfChunk, ChunkSize> qk_gated_l1;
  TASSIGN(qk_gated_l1, 98304);
  L1Mat<ComputeT, ChunkSize, HiddenSize> v_l1;
  TASSIGN(v_l1, 131072);
  TileAcc<float, HalfChunk, HiddenSize, HalfChunk, HiddenSize> qkv_l0;
  TASSIGN(qkv_l0, 0);
  TileRight<ComputeT, ChunkSize, HiddenSize, ChunkSize, HiddenSize> l0b;
  TASSIGN(l0b, 0x0);

#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
  // The caller observed H-ready before entering this helper. Drop any clean
  // lines retained by this AIC from an earlier work item or graph replay.
  constexpr int32_t DcciCacheLineElems =
      64 / static_cast<int32_t>(sizeof(ComputeT));
  for (int32_t row = 0; row < valid_rows; ++row) {
    for (int32_t r = 0; r < HiddenSize; r += DcciCacheLineElems) {
      dcci(static_cast<__gm__ void*>(v_handle + v_offset + row * v_stride + r),
           SINGLE_CACHE_LINE);
    }
  }
  dsb(DSB_DDR);
#endif
  if (wait_before_v_load != 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
    GroupWait<DirectedSync, PIPE_MTE2>(qk_gated_ready_flag);
#elif defined(PTO_NPU_ARCH_A5)
    gdn_sync::Wait<PIPE_MTE2>(qk_gated_ready_flag);
#else
    wait_flag_dev(qk_gated_ready_flag);
#endif
  }

  // V is independent of QK gating. Start its GM->L1 transfer before the
  // cross-core wait so MTE2 can run while Vec produces QK_gated.
  {
    L1Mat<ComputeT, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> v_load(valid_rows,
                                                                    HiddenSize);
    TASSIGN(v_load, 131072);
    GmShape2D shape(valid_rows, HiddenSize);
    GmStride2D stride(v_stride);
    GmTensor2D<ComputeT> v_global(v_handle + v_offset, shape, stride);
    TLOAD(v_load, v_global);
    if (valid_rows != ChunkSize) {
      TFILLPAD(v_load, v_load);
    }
  }

  if (wait_before_v_load == 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
    GroupWait<DirectedSync, PIPE_MTE2>(qk_gated_ready_flag);
#elif defined(PTO_NPU_ARCH_A5)
    gdn_sync::Wait<PIPE_MTE2>(qk_gated_ready_flag);
#else
    wait_flag_dev(qk_gated_ready_flag);
#endif
  }

  for (int32_t half_idx = 0; half_idx < 2; ++half_idx) {
    if (half_idx != 0 && split_bottom_quarters) {
      for (int32_t quarter_idx = 0; quarter_idx < 2; ++quarter_idx) {
        const int32_t gated_ready_flag = quarter_idx == 0
                                             ? qk_gated_bottom_ready_flag
                                             : qk_gated_bottom_tail_ready_flag;
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
        GroupWait<DirectedSync, PIPE_MTE2>(gated_ready_flag);
#elif defined(PTO_NPU_ARCH_A5)
        gdn_sync::Wait<PIPE_MTE2>(gated_ready_flag);
#else
        wait_flag_dev(gated_ready_flag);
#endif

        const int32_t row_start = HalfChunk + quarter_idx * QuarterChunk;
        int32_t quarter_valid_rows = valid_rows - row_start;
        if (quarter_valid_rows < 0) {
          quarter_valid_rows = 0;
        }
        if (quarter_valid_rows > QuarterChunk) {
          quarter_valid_rows = QuarterChunk;
        }

        L1Mat<ComputeT, QuarterChunk, ChunkSize> qk_quarter_l1;
        TASSIGN(qk_quarter_l1, 98304);
        if (quarter_valid_rows > 0) {
          L1Mat<ComputeT, QuarterChunk, ChunkSize, DYNAMIC, DYNAMIC> gated_load(
              quarter_valid_rows, ChunkSize);
          TASSIGN(gated_load, 98304);
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
          shape.shape[3] = quarter_valid_rows;
          shape.shape[4] = ChunkSize;
          GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, ChunkSize, 1>>
              gated_global(
                  qk_gated_mailbox +
                      core_id * static_cast<int64_t>(ChunkSize) * ChunkSize +
                      static_cast<int64_t>(row_start) * ChunkSize,
                  shape);
          TLOAD(gated_load, gated_global);
          if (quarter_valid_rows != QuarterChunk) {
            TFILLPAD(gated_load, gated_load);
          }
        }

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

        TileLeft<ComputeT, QuarterChunk, ChunkSize, QuarterChunk, ChunkSize>
            l0a;
        TASSIGN(l0a, 0x0);
        TileAcc<float, QuarterChunk, HiddenSize, QuarterChunk, HiddenSize>
            qkv_quarter_l0;
        TASSIGN(qkv_quarter_l0, 0);
        auto event = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, event);
        wait_flag(PIPE_MTE2, PIPE_MTE1, event);
        set_flag(PIPE_M, PIPE_MTE1, event);
        wait_flag(PIPE_M, PIPE_MTE1, event);
        TEXTRACT(l0a, qk_quarter_l1, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, event);
        wait_flag(PIPE_MTE1, PIPE_M, event);
        TMATMUL(qkv_quarter_l0, l0a, l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, event);
        wait_flag(PIPE_MTE1, PIPE_MTE2, event);
        set_flag(PIPE_M, PIPE_FIX, event);
        wait_flag(PIPE_M, PIPE_FIX, event);

        if (quarter_valid_rows > 0) {
          TileAcc<float, QuarterChunk, HiddenSize, DYNAMIC, DYNAMIC> qkv_store(
              quarter_valid_rows, HiddenSize);
          TASSIGN(qkv_store, 0);
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
          shape.shape[3] = quarter_valid_rows;
          shape.shape[4] = HiddenSize;
          GlobalTensor<ComputeT,
                       decltype(shape),
                       Stride<1, 1, 1, HiddenSize, 1>>
              qkv_global(
                  qkv_mailbox +
                      core_id * static_cast<int64_t>(ChunkSize) * HiddenSize +
                      static_cast<int64_t>(row_start) * HiddenSize,
                  shape);
          TSTORE(qkv_global, qkv_store);
        }

        const int32_t ready_flag = quarter_idx == 0
                                       ? qkv_bottom_ready_flag
                                       : qkv_bottom_tail_ready_flag;
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
        GroupSignal<DirectedSync, PIPE_FIX>(1 | (2 << 4) | (ready_flag << 8));
#else
        ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (ready_flag << 8));
#endif
      }
      continue;
    }
    if (half_idx != 0 && split_rows) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
      GroupWait<DirectedSync, PIPE_MTE2>(qk_gated_bottom_ready_flag);
#elif defined(PTO_NPU_ARCH_A5)
      gdn_sync::Wait<PIPE_MTE2>(qk_gated_bottom_ready_flag);
#else
      wait_flag_dev(qk_gated_bottom_ready_flag);
#endif
    }
    const int32_t row_start = half_idx * HalfChunk;
    int32_t half_valid_rows = valid_rows - row_start;
    if (half_valid_rows < 0) {
      half_valid_rows = 0;
    }
    if (half_valid_rows > HalfChunk) {
      half_valid_rows = HalfChunk;
    }

    if (half_valid_rows > 0) {
      L1Mat<ComputeT, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> gated_load(
          half_valid_rows, ChunkSize);
      TASSIGN(gated_load, 98304);
      Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
      shape.shape[3] = half_valid_rows;
      shape.shape[4] = ChunkSize;
      GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, ChunkSize, 1>>
          gated_global(
              qk_gated_mailbox +
                  core_id * static_cast<int64_t>(ChunkSize) * ChunkSize +
                  static_cast<int64_t>(row_start) * ChunkSize,
              shape);
      TLOAD(gated_load, gated_global);
      if (half_valid_rows != HalfChunk) {
        TFILLPAD(gated_load, gated_load);
      }
    }

    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

    TileLeft<ComputeT, HalfChunk, ChunkSize, HalfChunk, ChunkSize> l0a;
    TASSIGN(l0a, 0x0);
    auto event = EVENT_ID1;
    set_flag(PIPE_MTE2, PIPE_MTE1, event);
    wait_flag(PIPE_MTE2, PIPE_MTE1, event);
    set_flag(PIPE_M, PIPE_MTE1, event);
    wait_flag(PIPE_M, PIPE_MTE1, event);
    TEXTRACT(l0a, qk_gated_l1, 0, 0);
    if (half_idx == 0) {
      TEXTRACT(l0b, v_l1, 0, 0);
    }
    set_flag(PIPE_MTE1, PIPE_M, event);
    wait_flag(PIPE_MTE1, PIPE_M, event);
    TMATMUL(qkv_l0, l0a, l0b);
    set_flag(PIPE_MTE1, PIPE_MTE2, event);
    wait_flag(PIPE_MTE1, PIPE_MTE2, event);
    set_flag(PIPE_M, PIPE_FIX, event);
    wait_flag(PIPE_M, PIPE_FIX, event);

    if (half_valid_rows > 0) {
      TileAcc<float, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC> qkv_store(
          half_valid_rows, HiddenSize);
      TASSIGN(qkv_store, 0);
      Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
      shape.shape[3] = half_valid_rows;
      shape.shape[4] = HiddenSize;
      GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, HiddenSize, 1>>
          qkv_global(
              qkv_mailbox +
                  core_id * static_cast<int64_t>(ChunkSize) * HiddenSize +
                  static_cast<int64_t>(row_start) * HiddenSize,
              shape);
      TSTORE(qkv_global, qkv_store);
    }

    if (split_rows || half_idx == 1) {
      const int32_t ready_flag =
          half_idx == 0 ? qkv_ready_flag
                        : (split_rows ? qkv_bottom_ready_flag : qkv_ready_flag);
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
      GroupSignal<DirectedSync, PIPE_FIX>(1 | (2 << 4) | (ready_flag << 8));
#else
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (ready_flag << 8));
#endif
    }
  }
#endif
}

template <int32_t HiddenSize,
          int32_t ChunkSize
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
          ,
          bool DirectedSync = false
#endif
          >
AICORE inline void PublishGatedQKTile(
    __gm__ float* g_handle,
    __gm__ ComputeT* qk_mailbox,
    __gm__ ComputeT* qk_gated_mailbox,
    int64_t core_id,
    int64_t total_tokens,
    int64_t chunk_token_start,
    int32_t head_idx,
    int32_t valid_rows,
    int32_t local_rows,
    int32_t vec_id,
    int32_t row_gate_addr,
    uint32_t wait_for_qk_ready,
    uint32_t release_qk_mailbox,
    int32_t qk_ready_flag,
    int32_t qk_mailbox_free_flag,
    int32_t qk_gated_ready_flag,
    int32_t qk_gated_bottom_ready_flag = -1,
    int32_t qk_gated_bottom_tail_ready_flag = -1) {
#if defined(__DAV_C220_VEC__)
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t QuarterChunk = HalfChunk / 2;
  constexpr int32_t PrefetchRows = HalfChunk / 4;
  constexpr int32_t RemainingRows = HalfChunk - PrefetchRows;
  constexpr int32_t GAllUbAddr = 115712;
  constexpr int32_t MskUbAddr = 512;
  constexpr int32_t QKUbAddr = 33280;
  constexpr int32_t CoeffUbAddr = 66304;
  constexpr int32_t QKHalfUbAddr = 99072;
  constexpr int32_t QSUbAddr = 131840;
  constexpr int32_t QKPrefetchUbAddr = 184320;
  // Rejected QK prefetch experiments remain compiled out while O-stage
  // scheduling is evaluated against the v73 behavior.
  constexpr bool pipeline_qk = false;
  const bool pipeline_bottom_quarters = qk_gated_bottom_tail_ready_flag >= 0 &&
                                        local_rows == HalfChunk && vec_id != 0;

  if (local_rows > 0) {
    {
      Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
      shape.shape[3] = 1;
      shape.shape[4] = valid_rows;
      GlobalTensor<float, decltype(shape), Stride<1, 1, 1, 1, 1>> g_global(
          g_handle + static_cast<int64_t>(head_idx) * total_tokens +
              chunk_token_start,
          shape);
      UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> g_load(
          1, valid_rows);
      TASSIGN(g_load, GAllUbAddr);
      TLOAD(g_load, g_global);
      if (valid_rows != ChunkSize) {
        UbND<float, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> g_padded;
        TASSIGN(g_padded, GAllUbAddr);
        TFILLPAD_INPLACE(g_padded, g_load);
      }
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if (pipeline_qk) {
      Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
      shape.shape[3] = PrefetchRows;
      shape.shape[4] = ChunkSize;
      GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, ChunkSize, 1>>
          qk_global(qk_mailbox +
                        core_id * static_cast<int64_t>(ChunkSize) * ChunkSize +
                        static_cast<int64_t>(vec_id) * HalfChunk * ChunkSize,
                    shape);
      UbND<ComputeT,
           PrefetchRows,
           ChunkSize,
           PrefetchRows,
           ChunkSize,
           PadValue::Zero>
          qk_prefetch;
      TASSIGN(qk_prefetch, QKPrefetchUbAddr);
      TLOAD(qk_prefetch, qk_global);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    }

    UbND<float, 1, ChunkSize> g_all;
    TASSIGN(g_all, GAllUbAddr);
    UbND<float, 1, HalfChunk> g_rows;
    TASSIGN(g_rows, row_gate_addr);
    UbND<float, 1, HalfChunk> g_rows_source;
    TASSIGN(
        g_rows_source,
        GAllUbAddr + vec_id * HalfChunk * static_cast<int32_t>(sizeof(float)));
    TMOV(g_rows, g_rows_source);

    if (pipeline_bottom_quarters) {
      UbND<float, QuarterChunk, ChunkSize> row_gates;
      TASSIGN(row_gates, QKUbAddr);
      UbDN<float, QuarterChunk, 1> row_gates_col;
      TASSIGN(row_gates_col, row_gate_addr);
      UbND<float, QuarterChunk, ChunkSize> coefficients;
      TASSIGN(coefficients, CoeffUbAddr);
      UbND<float, QuarterChunk, ChunkSize> causal_mask;
      TASSIGN(causal_mask, MskUbAddr);
      TROWEXPAND(row_gates, row_gates_col);
      TCOLEXPAND(coefficients, g_all);
      TSUB(coefficients, row_gates, coefficients);
      pipe_barrier(PIPE_V);
      TMINS(coefficients, coefficients, 0.0f);
      pipe_barrier(PIPE_V);
      TEXP(coefficients, coefficients);
      pipe_barrier(PIPE_V);
      TMUL(coefficients, coefficients, causal_mask);
      pipe_barrier(PIPE_V);
    } else {
      UbND<float, HalfChunk, ChunkSize> row_gates;
      TASSIGN(row_gates, QKUbAddr);
      UbDN<float, HalfChunk, 1> row_gates_col;
      TASSIGN(row_gates_col, row_gate_addr);
      UbND<float, HalfChunk, ChunkSize> coefficients;
      TASSIGN(coefficients, CoeffUbAddr);
      UbND<float, HalfChunk, ChunkSize> causal_mask;
      TASSIGN(causal_mask, MskUbAddr);
      TROWEXPAND(row_gates, row_gates_col);
      TCOLEXPAND(coefficients, g_all);
      TSUB(coefficients, row_gates, coefficients);
      pipe_barrier(PIPE_V);
      TMINS(coefficients, coefficients, 0.0f);
      pipe_barrier(PIPE_V);
      TEXP(coefficients, coefficients);
      pipe_barrier(PIPE_V);
      TMUL(coefficients, coefficients, causal_mask);
      pipe_barrier(PIPE_V);
    }
    TEXP(g_rows, g_rows);
  }

  if (wait_for_qk_ready != 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
    GroupWait<DirectedSync, PIPE_MTE2>(qk_ready_flag);
#else
    wait_flag_dev(qk_ready_flag);
#endif
  }
  const bool split_rows = qk_gated_bottom_ready_flag >= 0;
  if (split_rows) {
    const int32_t peer_ready_flag =
        vec_id == 0 ? qk_gated_bottom_ready_flag : qk_gated_ready_flag;
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
    GroupSignal<DirectedSync, PIPE_MTE3>(1 | (2 << 4) | (peer_ready_flag << 8));
#else
    ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (peer_ready_flag << 8));
#endif
    if (qk_gated_bottom_tail_ready_flag >= 0 && vec_id == 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
      GroupSignal<DirectedSync, PIPE_MTE3>(
          1 | (2 << 4) | (qk_gated_bottom_tail_ready_flag << 8));
#else
      ffts_cross_core_sync(
          PIPE_MTE3, 1 | (2 << 4) | (qk_gated_bottom_tail_ready_flag << 8));
#endif
    }
  }
  if (local_rows > 0) {
    if (pipeline_bottom_quarters) {
      UbND<ComputeT,
           QuarterChunk,
           ChunkSize,
           QuarterChunk,
           ChunkSize,
           PadValue::Zero>
          qk_first_bf16;
      TASSIGN(qk_first_bf16, QKHalfUbAddr);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
        shape.shape[3] = QuarterChunk;
        shape.shape[4] = ChunkSize;
        GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, ChunkSize, 1>>
            qk_global(
                qk_mailbox +
                    core_id * static_cast<int64_t>(ChunkSize) * ChunkSize +
                    static_cast<int64_t>(vec_id) * HalfChunk * ChunkSize,
                shape);
        TLOAD(qk_first_bf16, qk_global);
      }
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      UbND<float, QuarterChunk, ChunkSize> qk_fp32;
      TASSIGN(qk_fp32, QKUbAddr);
      UbND<float, QuarterChunk, ChunkSize> coefficients;
      TASSIGN(coefficients, CoeffUbAddr);
      TCVT(qk_fp32, qk_first_bf16, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TMUL(qk_fp32, qk_fp32, coefficients);
      pipe_barrier(PIPE_V);
      TCVT(qk_first_bf16, qk_fp32, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
        shape.shape[3] = QuarterChunk;
        shape.shape[4] = ChunkSize;
        GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, ChunkSize, 1>>
            gated_global(
                qk_gated_mailbox +
                    core_id * static_cast<int64_t>(ChunkSize) * ChunkSize +
                    static_cast<int64_t>(vec_id) * HalfChunk * ChunkSize,
                shape);
        TSTORE(gated_global, qk_first_bf16);
      }

      UbND<ComputeT,
           QuarterChunk,
           ChunkSize,
           QuarterChunk,
           ChunkSize,
           PadValue::Zero>
          qk_second_bf16;
      TASSIGN(qk_second_bf16,
              QKHalfUbAddr + QuarterChunk * ChunkSize *
                                 static_cast<int32_t>(sizeof(ComputeT)));
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
        shape.shape[3] = QuarterChunk;
        shape.shape[4] = ChunkSize;
        GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, ChunkSize, 1>>
            qk_global(
                qk_mailbox +
                    core_id * static_cast<int64_t>(ChunkSize) * ChunkSize +
                    (static_cast<int64_t>(vec_id) * HalfChunk + QuarterChunk) *
                        ChunkSize,
                shape);
        TLOAD(qk_second_bf16, qk_global);
      }
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
      GroupSignal<DirectedSync, PIPE_MTE3>(1 | (2 << 4) |
                                           (qk_gated_bottom_ready_flag << 8));
#else
      ffts_cross_core_sync(PIPE_MTE3,
                           1 | (2 << 4) | (qk_gated_bottom_ready_flag << 8));
#endif

      UbND<float, 1, ChunkSize> g_all;
      TASSIGN(g_all, GAllUbAddr);
      UbND<float, QuarterChunk, ChunkSize> row_gates;
      TASSIGN(row_gates, QKUbAddr);
      UbDN<float, QuarterChunk, 1> row_gates_col;
      TASSIGN(row_gates_col,
              GAllUbAddr + (vec_id * HalfChunk + QuarterChunk) *
                               static_cast<int32_t>(sizeof(float)));
      UbND<float, QuarterChunk, ChunkSize> second_coefficients;
      TASSIGN(second_coefficients, CoeffUbAddr);
      UbND<float, QuarterChunk, ChunkSize> causal_mask;
      TASSIGN(causal_mask,
              MskUbAddr + QuarterChunk * ChunkSize *
                              static_cast<int32_t>(sizeof(float)));
      TROWEXPAND(row_gates, row_gates_col);
      TCOLEXPAND(second_coefficients, g_all);
      TSUB(second_coefficients, row_gates, second_coefficients);
      pipe_barrier(PIPE_V);
      TMINS(second_coefficients, second_coefficients, 0.0f);
      pipe_barrier(PIPE_V);
      TEXP(second_coefficients, second_coefficients);
      pipe_barrier(PIPE_V);
      TMUL(second_coefficients, second_coefficients, causal_mask);
      pipe_barrier(PIPE_V);
      UbND<float, 1, HalfChunk> g_rows;
      TASSIGN(g_rows, row_gate_addr);

      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      if (release_qk_mailbox != 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
        GroupSignal<DirectedSync, PIPE_MTE2>(1 | (2 << 4) |
                                             (qk_mailbox_free_flag << 8));
#else
        ffts_cross_core_sync(PIPE_MTE2,
                             1 | (2 << 4) | (qk_mailbox_free_flag << 8));
#endif
      }
      TCVT(qk_fp32, qk_second_bf16, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TMUL(qk_fp32, qk_fp32, second_coefficients);
      pipe_barrier(PIPE_V);
      TCVT(qk_second_bf16, qk_fp32, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
        shape.shape[3] = QuarterChunk;
        shape.shape[4] = ChunkSize;
        GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, ChunkSize, 1>>
            gated_global(
                qk_gated_mailbox +
                    core_id * static_cast<int64_t>(ChunkSize) * ChunkSize +
                    (static_cast<int64_t>(vec_id) * HalfChunk + QuarterChunk) *
                        ChunkSize,
                shape);
        TSTORE(gated_global, qk_second_bf16);
      }
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
      GroupSignal<DirectedSync, PIPE_MTE3>(
          1 | (2 << 4) | (qk_gated_bottom_tail_ready_flag << 8));
#else
      ffts_cross_core_sync(
          PIPE_MTE3, 1 | (2 << 4) | (qk_gated_bottom_tail_ready_flag << 8));
#endif
    } else if (pipeline_qk) {
      UbND<ComputeT,
           PrefetchRows,
           ChunkSize,
           PrefetchRows,
           ChunkSize,
           PadValue::Zero>
          qk_prefetch;
      TASSIGN(qk_prefetch, QKPrefetchUbAddr);
      UbND<ComputeT,
           RemainingRows,
           ChunkSize,
           RemainingRows,
           ChunkSize,
           PadValue::Zero>
          qk_second;
      TASSIGN(qk_second,
              QKHalfUbAddr + PrefetchRows * ChunkSize *
                                 static_cast<int32_t>(sizeof(ComputeT)));
      UbND<ComputeT,
           PrefetchRows,
           ChunkSize,
           PrefetchRows,
           ChunkSize,
           PadValue::Zero>
          qk_first;
      TASSIGN(qk_first, QKHalfUbAddr);
      UbND<ComputeT, HalfChunk, ChunkSize, HalfChunk, ChunkSize, PadValue::Zero>
          qk_bf16;
      TASSIGN(qk_bf16, QKHalfUbAddr);

      Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
      shape.shape[3] = RemainingRows;
      shape.shape[4] = ChunkSize;
      GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, ChunkSize, 1>>
          qk_global(
              qk_mailbox +
                  core_id * static_cast<int64_t>(ChunkSize) * ChunkSize +
                  (static_cast<int64_t>(vec_id) * HalfChunk + PrefetchRows) *
                      ChunkSize,
              shape);
      TLOAD(qk_second, qk_global);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);

      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TMOV(qk_first, qk_prefetch);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      if (release_qk_mailbox != 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
        GroupSignal<DirectedSync, PIPE_MTE2>(1 | (2 << 4) |
                                             (qk_mailbox_free_flag << 8));
#else
        ffts_cross_core_sync(PIPE_MTE2,
                             1 | (2 << 4) | (qk_mailbox_free_flag << 8));
#endif
      }

      UbND<float, HalfChunk, ChunkSize> qk_fp32;
      TASSIGN(qk_fp32, QKUbAddr);
      UbND<float, HalfChunk, ChunkSize> coefficients;
      TASSIGN(coefficients, CoeffUbAddr);
      TCVT(qk_fp32, qk_bf16, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TMUL(qk_fp32, qk_fp32, coefficients);
      pipe_barrier(PIPE_V);
      TCVT(qk_bf16, qk_fp32, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> store_shape;
        store_shape.shape[3] = local_rows;
        store_shape.shape[4] = ChunkSize;
        GlobalTensor<ComputeT,
                     decltype(store_shape),
                     Stride<1, 1, 1, ChunkSize, 1>>
            gated_global(
                qk_gated_mailbox +
                    core_id * static_cast<int64_t>(ChunkSize) * ChunkSize +
                    static_cast<int64_t>(vec_id) * HalfChunk * ChunkSize,
                store_shape);
        TSTORE(gated_global, qk_bf16);
      }
    } else {
      UbND<ComputeT, HalfChunk, ChunkSize, HalfChunk, ChunkSize, PadValue::Zero>
          qk_bf16;
      TASSIGN(qk_bf16, QKHalfUbAddr);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
        shape.shape[3] = local_rows;
        shape.shape[4] = ChunkSize;
        GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, ChunkSize, 1>>
            qk_global(
                qk_mailbox +
                    core_id * static_cast<int64_t>(ChunkSize) * ChunkSize +
                    static_cast<int64_t>(vec_id) * HalfChunk * ChunkSize,
                shape);
        UbND<ComputeT, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero>
            qk_load(local_rows, ChunkSize);
        TASSIGN(qk_load, QKHalfUbAddr);
        TLOAD(qk_load, qk_global);
        if (local_rows != HalfChunk) {
          TFILLPAD_INPLACE(qk_bf16, qk_load);
        }
      }
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      if (release_qk_mailbox != 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
        GroupSignal<DirectedSync, PIPE_MTE2>(1 | (2 << 4) |
                                             (qk_mailbox_free_flag << 8));
#else
        ffts_cross_core_sync(PIPE_MTE2,
                             1 | (2 << 4) | (qk_mailbox_free_flag << 8));
#endif
      }

      UbND<float, HalfChunk, ChunkSize> qk_fp32;
      TASSIGN(qk_fp32, QKUbAddr);
      UbND<float, HalfChunk, ChunkSize> coefficients;
      TASSIGN(coefficients, CoeffUbAddr);
      TCVT(qk_fp32, qk_bf16, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TMUL(qk_fp32, qk_fp32, coefficients);
      pipe_barrier(PIPE_V);
      TCVT(qk_bf16, qk_fp32, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
        shape.shape[3] = local_rows;
        shape.shape[4] = ChunkSize;
        GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, ChunkSize, 1>>
            gated_global(
                qk_gated_mailbox +
                    core_id * static_cast<int64_t>(ChunkSize) * ChunkSize +
                    static_cast<int64_t>(vec_id) * HalfChunk * ChunkSize,
                shape);
        UbND<ComputeT, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> gated_store(
            local_rows, ChunkSize);
        TASSIGN(gated_store, QKHalfUbAddr);
        TSTORE(gated_global, gated_store);
      }
    }
  } else if (release_qk_mailbox != 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
    GroupSignal<DirectedSync, PIPE_MTE2>(1 | (2 << 4) |
                                         (qk_mailbox_free_flag << 8));
#else
    ffts_cross_core_sync(PIPE_MTE2, 1 | (2 << 4) | (qk_mailbox_free_flag << 8));
#endif
  }
  if (!pipeline_bottom_quarters) {
    const int32_t own_ready_flag = split_rows && vec_id != 0
                                       ? qk_gated_bottom_ready_flag
                                       : qk_gated_ready_flag;
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
    GroupSignal<DirectedSync, PIPE_MTE3>(1 | (2 << 4) | (own_ready_flag << 8));
#else
    ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (own_ready_flag << 8));
#endif
  }
#endif
}

template <int32_t HiddenSize,
          int32_t ChunkSize,
          bool FuseGatedRmsNorm
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
          ,
          bool DirectedSync = false
#endif
          >
AICORE inline void ConsumeQKVTile(__gm__ ComputeT* precomputed_qs_handle,
                                  __gm__ ComputeT* qkv_mailbox,
                                  __gm__ GDN_PUBLIC_DTYPE* output_handle,
                                  __gm__ GDN_PUBLIC_DTYPE* z_handle,
                                  int64_t core_id,
                                  int64_t chunk_idx,
                                  int64_t chunk_token_start,
                                  int32_t head_idx,
                                  int32_t num_heads,
                                  int32_t valid_rows,
                                  int32_t local_rows,
                                  int32_t vec_id,
                                  int32_t row_gate_addr,
                                  int32_t qkv_ready_flag,
                                  int32_t qkv_mailbox_free_flag,
                                  int32_t qkv_bottom_ready_flag = -1,
                                  int32_t qkv_bottom_tail_ready_flag = -1) {
#if defined(__DAV_C220_VEC__)
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t QuarterChunk = HalfChunk / 2;
  constexpr int32_t QKUbAddr = 33280;
  constexpr int32_t CoeffUbAddr = 66304;
  constexpr int32_t QSHalfUbAddr = 115456;
  constexpr int32_t QSUbAddr = 131840;
  constexpr int32_t OHalfUbAddr = 164608;
  constexpr int32_t OUbAddr = QKUbAddr;
  const int32_t output_stride = num_heads * HiddenSize;
  const bool split_rows = qkv_bottom_ready_flag >= 0;
  const bool split_bottom_quarters = split_rows &&
                                     qkv_bottom_tail_ready_flag >= 0 &&
                                     local_rows == HalfChunk && vec_id != 0;

  if (local_rows > 0) {
    UbND<float, 1, HalfChunk> g_rows;
    TASSIGN(g_rows, row_gate_addr);

    UbND<ComputeT, HalfChunk, HiddenSize, HalfChunk, HiddenSize, PadValue::Zero>
        qs_bf16;
    TASSIGN(qs_bf16, QSHalfUbAddr);
    {
      Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
      shape.shape[3] = local_rows;
      shape.shape[4] = HiddenSize;
      __gm__ ComputeT* qs_source =
          precomputed_qs_handle + (chunk_idx * num_heads + head_idx) *
                                      static_cast<int64_t>(HiddenSize) *
                                      HiddenSize;
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
      // QK-ready is emitted only after the AIC acquired H-ready in variant21,
      // so it is the control predecessor for this AIV-side QS acquire.
      constexpr int32_t DcciCacheLineElems =
          64 / static_cast<int32_t>(sizeof(ComputeT));
      for (int32_t row = 0; row < local_rows; ++row) {
        for (int32_t r = 0; r < HiddenSize; r += DcciCacheLineElems) {
          dcci(static_cast<__gm__ void*>(
                   qs_source +
                   (static_cast<int64_t>(vec_id) * HalfChunk + row) *
                       HiddenSize +
                   r),
               SINGLE_CACHE_LINE);
        }
      }
      dsb(DSB_DDR);
#endif
      GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, HiddenSize, 1>>
          qs_global(
              qs_source + static_cast<int64_t>(vec_id) * HalfChunk * HiddenSize,
              shape);
      UbND<ComputeT, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC, PadValue::Zero>
          qs_load(local_rows, HiddenSize);
      TASSIGN(qs_load, QSHalfUbAddr);
      TLOAD(qs_load, qs_global);
      if (local_rows != HalfChunk) {
        TFILLPAD_INPLACE(qs_bf16, qs_load);
      }
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    UbND<float, HalfChunk, HiddenSize> qs_fp32;
    TASSIGN(qs_fp32, QSUbAddr);
    TCVT(qs_fp32, qs_bf16, pto::RoundMode::CAST_NONE);
    UbND<float, HalfChunk, HiddenSize> expanded_gates;
    TASSIGN(expanded_gates, CoeffUbAddr);
    UbDN<float, HalfChunk, 1> row_gates_col;
    TASSIGN(row_gates_col, row_gate_addr);
    TROWEXPAND(expanded_gates, row_gates_col);
    pipe_barrier(PIPE_V);
    TMUL(qs_fp32, qs_fp32, expanded_gates);

#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
    GroupWait<DirectedSync, PIPE_MTE2>(qkv_ready_flag);
#else
    wait_flag_dev(qkv_ready_flag);
#endif
    if (split_rows && vec_id != 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
      GroupWait<DirectedSync, PIPE_MTE2>(qkv_bottom_ready_flag);
#else
      wait_flag_dev(qkv_bottom_ready_flag);
#endif
    }

    UbND<ComputeT, HalfChunk, HiddenSize, HalfChunk, HiddenSize, PadValue::Zero>
        qkv_bf16;
    TASSIGN(qkv_bf16, OHalfUbAddr);
    if (split_bottom_quarters) {
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
        shape.shape[3] = QuarterChunk;
        shape.shape[4] = HiddenSize;
        GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, HiddenSize, 1>>
            qkv_global(
                qkv_mailbox +
                    core_id * static_cast<int64_t>(ChunkSize) * HiddenSize +
                    static_cast<int64_t>(vec_id) * HalfChunk * HiddenSize,
                shape);
        UbND<ComputeT,
             QuarterChunk,
             HiddenSize,
             QuarterChunk,
             HiddenSize,
             PadValue::Zero>
            qkv_load;
        TASSIGN(qkv_load, OHalfUbAddr);
        TLOAD(qkv_load, qkv_global);
      }
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
      GroupWait<DirectedSync, PIPE_MTE2>(qkv_bottom_tail_ready_flag);
#else
      wait_flag_dev(qkv_bottom_tail_ready_flag);
#endif
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
        shape.shape[3] = QuarterChunk;
        shape.shape[4] = HiddenSize;
        GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, HiddenSize, 1>>
            qkv_global(
                qkv_mailbox +
                    core_id * static_cast<int64_t>(ChunkSize) * HiddenSize +
                    (static_cast<int64_t>(vec_id) * HalfChunk + QuarterChunk) *
                        HiddenSize,
                shape);
        UbND<ComputeT,
             QuarterChunk,
             HiddenSize,
             QuarterChunk,
             HiddenSize,
             PadValue::Zero>
            qkv_load;
        TASSIGN(qkv_load,
                OHalfUbAddr + QuarterChunk * HiddenSize *
                                  static_cast<int32_t>(sizeof(ComputeT)));
        TLOAD(qkv_load, qkv_global);
      }
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      if (qkv_mailbox_free_flag >= 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
        GroupSignal<DirectedSync, PIPE_MTE2>(1 | (2 << 4) |
                                             (qkv_mailbox_free_flag << 8));
#else
        ffts_cross_core_sync(PIPE_MTE2,
                             1 | (2 << 4) | (qkv_mailbox_free_flag << 8));
#endif
      }
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    } else {
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> shape;
        shape.shape[3] = local_rows;
        shape.shape[4] = HiddenSize;
        GlobalTensor<ComputeT, decltype(shape), Stride<1, 1, 1, HiddenSize, 1>>
            qkv_global(
                qkv_mailbox +
                    core_id * static_cast<int64_t>(ChunkSize) * HiddenSize +
                    static_cast<int64_t>(vec_id) * HalfChunk * HiddenSize,
                shape);
        UbND<ComputeT, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC, PadValue::Zero>
            qkv_load(local_rows, HiddenSize);
        TASSIGN(qkv_load, OHalfUbAddr);
        TLOAD(qkv_load, qkv_global);
        if (local_rows != HalfChunk) {
          TFILLPAD_INPLACE(qkv_bf16, qkv_load);
        }
      }
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      if (qkv_mailbox_free_flag >= 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
        GroupSignal<DirectedSync, PIPE_MTE2>(1 | (2 << 4) |
                                             (qkv_mailbox_free_flag << 8));
#else
        ffts_cross_core_sync(PIPE_MTE2,
                             1 | (2 << 4) | (qkv_mailbox_free_flag << 8));
#endif
      }
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    UbND<float, HalfChunk, HiddenSize> output_fp32;
    TASSIGN(output_fp32, OUbAddr);
    TCVT(output_fp32, qkv_bf16, pto::RoundMode::CAST_NONE);
    TADD(output_fp32, qs_fp32, output_fp32);

    const int64_t output_offset =
        (chunk_token_start * static_cast<int64_t>(num_heads) + head_idx) *
            static_cast<int64_t>(HiddenSize) +
        static_cast<int64_t>(vec_id) * HalfChunk * output_stride;
    StoreChunkOutput<HiddenSize, ChunkSize, FuseGatedRmsNorm>(
        output_handle, z_handle, output_offset, output_stride, local_rows);
    if (split_rows && vec_id == 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
      GroupWait<DirectedSync, PIPE_MTE2>(qkv_bottom_ready_flag);
#else
      wait_flag_dev(qkv_bottom_ready_flag);
#endif
      if (qkv_bottom_tail_ready_flag >= 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
        GroupWait<DirectedSync, PIPE_MTE2>(qkv_bottom_tail_ready_flag);
#else
        wait_flag_dev(qkv_bottom_tail_ready_flag);
#endif
      }
    }
  } else {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
    GroupWait<DirectedSync, PIPE_MTE2>(qkv_ready_flag);
#else
    wait_flag_dev(qkv_ready_flag);
#endif
    if (split_rows) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
      GroupWait<DirectedSync, PIPE_MTE2>(qkv_bottom_ready_flag);
#else
      wait_flag_dev(qkv_bottom_ready_flag);
#endif
      if (qkv_bottom_tail_ready_flag >= 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
        GroupWait<DirectedSync, PIPE_MTE2>(qkv_bottom_tail_ready_flag);
#else
        wait_flag_dev(qkv_bottom_tail_ready_flag);
#endif
      }
    }
    if (qkv_mailbox_free_flag >= 0) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
      GroupSignal<DirectedSync, PIPE_MTE3>(1 | (2 << 4) |
                                           (qkv_mailbox_free_flag << 8));
#else
      ffts_cross_core_sync(PIPE_MTE3,
                           1 | (2 << 4) | (qkv_mailbox_free_flag << 8));
#endif
    }
  }
#endif
}

template <int32_t HiddenSize, int32_t ChunkSize, bool FuseGatedRmsNorm = false>
AICORE void GDN_CHUNK_O_KERNEL(__gm__ ComputeT* Q_handle,
                               __gm__ ComputeT* K_handle,
                               __gm__ ComputeT* V_handle,
                               __gm__ ComputeT* S_handle,
                               __gm__ float* G_handle,
                               __gm__ float* Msk_handle,
                               __gm__ ComputeT* workspace_qk_handle,
                               __gm__ ComputeT* workspace_qs_qkv_handle,
                               __gm__ ComputeT* workspace_qk_gated_handle,
                               __gm__ ComputeT* workspace_ping_qk_handle,
                               __gm__ ComputeT* workspace_ping_qs_qkv_handle,
                               __gm__ ComputeT* workspace_ping_qk_gated_handle,
                               uint32_t reuse_group_qk,
                               __gm__ GDN_PUBLIC_DTYPE* O_handle,
                               __gm__ int32_t* cu_seqlens,
                               int64_t batch_size,
                               int64_t seq_len,
                               int64_t total_tokens,
                               uint32_t num_heads,
                               uint32_t num_key_heads,
                               uint32_t precompute_qs,
                               __gm__ int32_t* h_o_ready_handle,
                               uint64_t ffts_addr,
                               __gm__ GDN_PUBLIC_DTYPE* z_handle,
                               __gm__ GDN_PUBLIC_DTYPE* norm_weight_handle) {
  // Half the chunk — each Vec sub-block handles C/2 rows independently.
  constexpr int32_t HalfChunk = ChunkSize / 2;
  // KTail / CTail: the number of valid elements in the last 128-element tile
  // when D or C isn't a multiple of 128. Used internally by PTO for partial
  // tiles.
  constexpr uint32_t KTail = (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);
  constexpr uint32_t CTail = (ChunkSize % 128 == 0) ? 128 : (ChunkSize % 128);

  const int32_t H = static_cast<int32_t>(num_heads);
  const int32_t Hg = static_cast<int32_t>(num_key_heads);
  if (H <= 0 || Hg <= 0 || (H % Hg) != 0) return;
  const int32_t GROUP = H / Hg;
  const int32_t BSND_V_STRIDE = H * HiddenSize;
  const int32_t BSND_QK_STRIDE = Hg * HiddenSize;

  // Workspace sizes (in elements) shared between Cube and Vec via GM
  constexpr int32_t WsQKSize = ChunkSize * ChunkSize;
  constexpr int32_t WsQSSize = ChunkSize * HiddenSize;
  constexpr int32_t WsGatedSize = ChunkSize * ChunkSize;
  constexpr int32_t SecondL0CAddr = ChunkSize * HiddenSize * sizeof(float);

  // ── UB memory map (byte addresses within Unified Buffer) ─────────────
  constexpr int32_t GUbAddr = 0;
  constexpr int32_t MskUbAddr = 512;
  constexpr int32_t QKUbAddr = 33280;
  constexpr int32_t GvUbAddr = 66048;
  constexpr int32_t CoeffUbAddr = 66304;
  constexpr int32_t QKHalfUbAddr = 99072;
  constexpr int32_t QSHalfUbAddr = 115456;
  constexpr int32_t QSUbAddr = 131840;
  constexpr int32_t OHalfUbAddr = 164608;
  constexpr int32_t OUbAddr = QKUbAddr;

  // Initialize the cross-core FFTS signaling base address for this AI core.
  gdn_sync::InitAddress(ffts_addr);
  // cid = which AI core am I? (0..block_num-1). Used to partition work items.
  auto cid = get_block_idx();
  // block_num = total number of AI cores running this kernel in parallel.
  auto block_num = get_block_num();
  int64_t h_o_chunk_count = 0;
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
  bool h_o_multibatch_ragged = false;
  int64_t h_o_first_sequence_tokens = -1;
#endif
  if (cu_seqlens != nullptr) {
    for (int64_t si = 0; si < batch_size; ++si) {
      const int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      const int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
      const int64_t sequence_tokens = eos - bos;
      h_o_chunk_count += (sequence_tokens + ChunkSize - 1) / ChunkSize;
      if (si == 0) {
        h_o_first_sequence_tokens = sequence_tokens;
      } else if (sequence_tokens != h_o_first_sequence_tokens) {
        h_o_multibatch_ragged = true;
      }
#else
      h_o_chunk_count += (eos - bos + ChunkSize - 1) / ChunkSize;
#endif
    }
  } else {
    h_o_chunk_count = batch_size * ((seq_len + ChunkSize - 1) / ChunkSize);
  }
#if defined(MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE) || \
    defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
  const bool use_precomputed_qs =
      precompute_qs != 0 && ChunkSize == HiddenSize && H >= 8 &&
      batch_size >= 1 && cu_seqlens != nullptr && h_o_chunk_count >= 4 &&
      h_o_chunk_count <= batch_size * static_cast<int64_t>(64);
#else
  const bool use_precomputed_qs = precompute_qs != 0 &&
                                  ChunkSize == HiddenSize && H >= 8 &&
                                  batch_size >= 1 && cu_seqlens != nullptr &&
                                  h_o_chunk_count >= 4 && h_o_chunk_count <= 64;
#endif
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
  // QS must not be fetched by either AIV before the corresponding ready
  // counter is acquired. The first safe candidate keeps QK behind ready.
  constexpr bool decouple_qk_h_ready = false;
#else
  const bool decouple_qk_h_ready =
      use_precomputed_qs && H > static_cast<int32_t>(block_num);
#endif
  constexpr int64_t H_O_READY_STRIDE = 16;
  AscendC::GlobalTensor<int32_t> h_o_ready_gm;
  h_o_ready_gm.SetGlobalBuffer(h_o_ready_handle);

#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
  const int64_t h_o_heavy_core_count =
      (batch_size * static_cast<int64_t>(H)) % static_cast<int64_t>(block_num);
#else
  const int64_t h_o_heavy_core_count = H % static_cast<int64_t>(block_num);
#endif
  const int64_t h_o_light_core_count =
      static_cast<int64_t>(block_num) - h_o_heavy_core_count;
  const int64_t h_o_total_items = h_o_chunk_count * H;
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
  // Ragged sequences give each H producer a different chunk weight. Keep the
  // overlap protocol, but use the exact round-robin consumer map instead of
  // applying the uniform-batch heavy/light rebalance heuristic.
  const bool rebalance_h_o_consumers =
      use_precomputed_qs && !h_o_multibatch_ragged &&
      h_o_heavy_core_count > 0 &&
      h_o_total_items >= static_cast<int64_t>(block_num);
#else
  const bool rebalance_h_o_consumers =
      use_precomputed_qs && h_o_heavy_core_count > 0 &&
      h_o_total_items >= static_cast<int64_t>(block_num);
#endif
  const int64_t h_o_delta_o_items = (13 * h_o_chunk_count + 8) / 17;
  const int64_t h_o_shifted_items =
      h_o_total_items - h_o_delta_o_items * h_o_light_core_count;
  const int64_t h_o_heavy_quota_base =
      h_o_shifted_items > static_cast<int64_t>(block_num)
          ? h_o_shifted_items / static_cast<int64_t>(block_num)
          : 1;
  // The heavy prefix is derived from the H producer assignment (variant21
  // flattens batch_size * H; earlier variants use H). Once QK no longer waits
  // for H-ready, move one O item from each heavy core to the light-core pool.
  const int64_t h_o_heavy_quota =
      decouple_qk_h_ready && h_o_heavy_quota_base > 1 ? h_o_heavy_quota_base - 1
                                                      : h_o_heavy_quota_base;
  const int64_t h_o_balanced_span =
      h_o_heavy_quota * static_cast<int64_t>(block_num);
  const int64_t h_o_prefix_limit =
      h_o_balanced_span < h_o_total_items ? h_o_balanced_span : h_o_total_items;
  const int64_t h_o_owned_prefix_count =
      static_cast<int64_t>(cid) < h_o_prefix_limit
          ? 1 + (h_o_prefix_limit - 1 - static_cast<int64_t>(cid)) /
                    static_cast<int64_t>(block_num)
          : 0;
  const int64_t h_o_tail_owner_shift =
      decouple_qk_h_ready && h_o_light_core_count > 0
          ? h_o_heavy_core_count % h_o_light_core_count
          : 0;
  // vid = Vec sub-block ID (0 or 1). Each Vec core has 2 sub-blocks that
  // process the upper (vid=0) and lower (vid=1) halves of C/2 rows.
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;

  // ── L1 tiles for Cube GEMM operands ──────────────────────────────────
  // L1 holds matrices in NZ (col-major fractal) format for the matrix engine.
  // Each tile is assigned a fixed L1 byte address to avoid runtime allocation.
  //
  // ── L1 tile layout for Cube GEMMs ────────────────────────────────────
  // L1 cache (~1MB) is manually partitioned for the 3 GEMMs:
  //   q_l1   at 0:      Q [C×D]       — shared by GEMM 1 and GEMM 2
  //   k_l1   at 32768:  K [C×D]       — used in GEMM 1 (transposed via
  //   TRESHAPE) s_l1   at 65536:  S [D×D]       — accumulated state, used in
  //   GEMM 2 qk_gated at 98304: QK_gated [C×C] — from Vec, used in GEMM 3 v_l1
  //   at 131072: V [C×D]       — values, used in GEMM 3
  L1Mat<ComputeT, ChunkSize, HiddenSize> q_l1;
  TASSIGN(q_l1, 0);
  L1Mat<ComputeT, ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 32768);
  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> qk_l0;
  TASSIGN(qk_l0, 0);
  L1Mat<ComputeT, HiddenSize, HiddenSize> s_l1;
  TASSIGN(s_l1, 65536);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> qs_l0;
  TASSIGN(qs_l0, SecondL0CAddr);
  L1Mat<ComputeT, ChunkSize, ChunkSize> qk_gated_l1;
  TASSIGN(qk_gated_l1, 98304);
  L1Mat<ComputeT, ChunkSize, HiddenSize> v_l1;
  TASSIGN(v_l1, 131072);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> qkv_l0;
  TASSIGN(qkv_l0, 0);

  // ── UB tiles for Vec element-wise operations ─────────────────────────
  // UB (Unified Buffer) is on-chip SRAM accessible by the Vec engine.
  // Tiles here are row-major (ND) for standard element-wise ops.
  //
  // ── UB tile layout for Vec element-wise ops ──────────────────────────
  // Each Vec sub-block (vid=0 or vid=1) processes C/2 rows of the C×C or C×D
  // matrices. The UB layout (byte addresses) is designed so all needed tiles
  // fit simultaneously in the ~256KB UB without overlapping:
  //   g_ub:       gate values [1, C] float            @ 0
  //   msk_ub:     causal mask [C/2, C] float          @ 512     (loaded once,
  //   reused) qk_ub:      QK scores in float [C/2, C]         @ 33280   (after
  //   cast from ComputeT) g_v_ub:     this sub-block's gate slice [1, C/2] @
  //   66048 coeff_ub:   gating coefficients [C/2, C] float  @ 66304 qk_ub_half:
  //   QK in ComputeT [C/2, C]                @ 99072 qs_ub_half: QS in ComputeT
  //   [C/2, D]                @ 115456 qs_ub:      QS in float [C/2, D] @
  //   131840 o_ub_half:  output O in ComputeT [C/2, D]           @ 164608 o_ub:
  //   output O in float [C/2, D]          @ QKUbAddr (reuses qk_ub space)
  UbND<float, 1, ChunkSize> g_ub;
  TASSIGN(g_ub, GUbAddr);
  UbND<float, HalfChunk, ChunkSize> msk_ub;
  TASSIGN(msk_ub, MskUbAddr);
  UbND<float, HalfChunk, ChunkSize> qk_ub;
  TASSIGN(qk_ub, QKUbAddr);
  UbND<float, 1, HalfChunk> g_v_ub;
  TASSIGN(g_v_ub, GvUbAddr);
  UbND<float, HalfChunk, ChunkSize> coeff_ub;
  TASSIGN(coeff_ub, CoeffUbAddr);
  UbND<ComputeT, HalfChunk, ChunkSize, HalfChunk, ChunkSize, PadValue::Zero>
      qk_ub_half;
  TASSIGN(qk_ub_half, QKHalfUbAddr);
  UbND<ComputeT, HalfChunk, HiddenSize, HalfChunk, HiddenSize, PadValue::Zero>
      qs_ub_half;
  TASSIGN(qs_ub_half, QSHalfUbAddr);
  UbND<float, HalfChunk, HiddenSize> qs_ub;
  TASSIGN(qs_ub, QSUbAddr);
  UbND<ComputeT, HalfChunk, HiddenSize, HalfChunk, HiddenSize, PadValue::Zero>
      o_ub_half;
  TASSIGN(o_ub_half, OHalfUbAddr);
  UbND<float, HalfChunk, HiddenSize> o_ub;
  TASSIGN(o_ub, OUbAddr);

  // Total work items = (batches * chunks_per_sequence * heads).
  // Each AI core (cid) picks every block_num-th work item (round-robin).
  int64_t total_work = 0;
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    total_work = num_seqs * chunks_per_seq * H;
  }

// =====================================================================
// CUBE CORE — Three GEMMs per chunk: QK, QS, QKV
// Each AI core processes a different (chunk, head) pair. The Cube engine
// performs the heavy matmuls, then writes results to GM workspace for
// the Vec engine to apply gating and produce the final output.
// =====================================================================
#if defined(__DAV_C220_CUBE__)
  if (cu_seqlens == nullptr) {
    // ── Fixed-length sequence path ──────────────────────────────────────
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    int64_t global_chunk_base = 0;
    bool first_cube_iter = true;

    for (int64_t work_idx = static_cast<int64_t>(cid); work_idx < total_work;
         work_idx += static_cast<int64_t>(block_num)) {
      // Wait for Vec to finish with previous chunk's workspace (flag 3)
      if (!first_cube_iter) {
#if defined(PTO_NPU_ARCH_A5)
        gdn_sync::Wait<PIPE_FIX>(3);
#else
        wait_flag_dev(3);
#endif
      }
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      int32_t head_idx = static_cast<int32_t>(work_idx % H);
      int32_t head_g = head_idx / GROUP;
      int64_t chunk_head_idx = work_idx / H;
      int64_t seq_idx = chunk_head_idx / chunks_per_seq;
      int64_t ci = chunk_head_idx % chunks_per_seq;

      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows =
          static_cast<int32_t>(remaining < ChunkSize ? remaining : ChunkSize);
      int64_t chunk_token_start = bos + chunk_start;
      int32_t row_offset = static_cast<int32_t>(vid) * HalfChunk;
      int32_t local_rows = valid_rows - row_offset;
      if (local_rows < 0) local_rows = 0;
      if (local_rows > HalfChunk) local_rows = HalfChunk;

      int64_t qk_off = (chunk_token_start * static_cast<int64_t>(Hg) +
                        static_cast<int64_t>(head_g)) *
                       static_cast<int64_t>(HiddenSize);
#ifdef MEGA_CHUNK_GDN_A5_PACKED_WUV
      int64_t v_off =
          (static_cast<int64_t>(head_idx) * total_tokens + chunk_token_start) *
          static_cast<int64_t>(HiddenSize);
#else
      int64_t v_off = (chunk_token_start * static_cast<int64_t>(H) +
                       static_cast<int64_t>(head_idx)) *
                      static_cast<int64_t>(HiddenSize);
#endif

      int64_t chunk_global_idx = seq_idx * chunks_per_seq + ci;
      int64_t s_offset = (chunk_global_idx * H + head_idx) *
                         static_cast<int64_t>(HiddenSize) *
                         static_cast<int64_t>(HiddenSize);

      // ── Load Q [valid_rows × D] from GM → L1 ────────────────────────
      // GlobalTensor describes the GM layout with BSND strides.
      // TLOAD performs DMA (MTE2 pipe). TFILLPAD zero-pads tail rows so
      // downstream GEMMs see a clean C×D matrix.
      {
        L1Mat<ComputeT, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(
            valid_rows, HiddenSize);
        TASSIGN(_l1, 0);
        GmShape2D _gs(valid_rows, HiddenSize);
        GmStride2D _stride(BSND_QK_STRIDE);
        GmTensor2D<ComputeT> _gm(Q_handle + qk_off, _gs, _stride);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }
      // ── Load K [valid_rows × D] from GM → L1 ────────────────────────
      {
        L1Mat<ComputeT, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(
            valid_rows, HiddenSize);
        TASSIGN(_l1, 32768);
        GmShape2D _gs(valid_rows, HiddenSize);
        GmStride2D _stride(BSND_QK_STRIDE);
        GmTensor2D<ComputeT> _gm(K_handle + qk_off, _gs, _stride);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }

      // ── GEMM 1: QK = Q @ K^T  (intra-chunk attention scores) ────────
      // ── GEMM 1: QK = Q @ K^T ─────────────────────────────────────────
      // numpy: QK = Q @ K.T  →  [C×D] @ [D×C] = [C×C]
      //
      // How transpose works on NPU:
      //   K is loaded into L1 in NZ (col-major fractal) format.
      //   TRESHAPE(l1_zn, k_l1) reinterprets it as ZN (row-major fractal) =
      //   K^T. This is a ZERO-COST operation — no data movement, just metadata
      //   change. TEXTRACT then loads the transposed view into L0B.
      //
      // Cube GEMM pipeline:
      //   TEXTRACT(l0a, q_l1, 0, 0)  — Q → L0A (left operand)
      //   TEXTRACT(l0b, k_zn, 0, 0)  — K^T → L0B (right operand)
      //   TMATMUL(qk_l0, l0a, l0b)   — QK = L0A × L0B → L0C accumulator
      //
      // transpose_B: TRESHAPE converts k_l1 from NZ → ZN fractal layout,
      // effectively transposing K before TEXTRACT loads it into L0B.
      {
        TileLeft<ComputeT, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0a;
        TileRight<ComputeT, HiddenSize, ChunkSize, HiddenSize, ChunkSize> _l0b;
        TASSIGN(_l0a, 0x0);
        TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we);
        wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we);
        wait_flag(PIPE_M, PIPE_MTE1, _we);
        TEXTRACT(_l0a, q_l1, 0, 0);
        L1MatZN<ComputeT, HiddenSize, ChunkSize> _bzn;
        TRESHAPE(_bzn, k_l1);
        TEXTRACT(_l0b, _bzn, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we);
        wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(qk_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we);
        wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we);
        wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // ── Load S [D × D] from GM → L1  (accumulated hidden state) ─────
      {
        L1Mat<ComputeT, HiddenSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(
            HiddenSize, HiddenSize);
        TASSIGN(_l1, 65536);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = HiddenSize;
        _gs.shape[4] = HiddenSize;
        GlobalTensor<ComputeT,
                     decltype(_gs),
                     pto::Stride<1, 1, 1, HiddenSize, 1>>
            _gm(S_handle + s_offset, _gs);
        TLOAD(_l1, _gm);
      }

      // ── GEMM 2: QS = Q @ S  (query applied to accumulated state) ────
      {
        TileLeft<ComputeT, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0a;
        TileRight<ComputeT, HiddenSize, HiddenSize, HiddenSize, HiddenSize>
            _l0b;
        TASSIGN(_l0a, 0x0);
        TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we);
        wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we);
        wait_flag(PIPE_M, PIPE_MTE1, _we);
        TEXTRACT(_l0a, q_l1, 0, 0);
        TEXTRACT(_l0b, s_l1, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we);
        wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(qs_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we);
        wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we);
        wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // ── Store QK [C × C] from L0C → GM workspace (FP32→ComputeT cast) ───
      // TSTORE on TileAcc triggers MTE3 DMA with implicit type conversion.
      {
        TileAcc<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l0(ChunkSize,
                                                                   ChunkSize);
        TASSIGN(_l0, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize;
        _gs.shape[4] = ChunkSize;
        GlobalTensor<ComputeT,
                     decltype(_gs),
                     pto::Stride<1, 1, 1, ChunkSize, 1>>
            _gm(workspace_qk_handle + static_cast<int64_t>(cid) * WsQKSize,
                _gs);
        TSTORE(_gm, _l0);
      }

      // ── Store QS [C × D] from L0C → GM workspace ────────────────────
      {
        TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(ChunkSize,
                                                                    HiddenSize);
        TASSIGN(_l0, SecondL0CAddr);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize;
        _gs.shape[4] = HiddenSize;
        GlobalTensor<ComputeT,
                     decltype(_gs),
                     pto::Stride<1, 1, 1, HiddenSize, 1>>
            _gm(workspace_qs_qkv_handle + static_cast<int64_t>(cid) * WsQSSize,
                _gs);
        TSTORE(_gm, _l0);
      }

      // Signal Vec: QK and QS are ready (flag 0, Cube→Vec)
      // ── Cross-core sync protocol ──────────────────────────────────────
      // Cube and Vec are SEPARATE physical cores. They exchange data through GM
      // and coordinate via FFTS flags. Think of it as two processes
      // communicating through shared memory with semaphores.
      //
      // ffts_cross_core_sync(PIPE_FIX, config):
      //   config = 1 | (mode << 4) | (flag_id << 8)
      //   mode=2: broadcast signal to all cores in this block
      //   flag_id: identifies which signal (0, 1, 2, 3)
      //
      // Protocol for this kernel:
      //   flag 0: Cube→Vec "QK and QS are ready in workspace"
      //   flag 1: Vec→Cube "QK_gated is ready for GEMM 3"
      //   flag 2: Cube→Vec "QKV (GEMM 3 result) is ready"
      //   flag 3: Vec→Cube "I'm done with this chunk, you can reuse workspace"
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));

      // Wait for Vec to write QK_gated back (flag 1, Vec→Cube)
#if defined(PTO_NPU_ARCH_A5)
      gdn_sync::Wait<PIPE_MTE2>(1);
#else
      wait_flag_dev(1);
#endif

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      // ── Load QK_gated [C × C] from GM workspace → L1 ────────────────
      {
        L1Mat<ComputeT, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(ChunkSize,
                                                                    ChunkSize);
        TASSIGN(_l1, 98304);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize;
        _gs.shape[4] = ChunkSize;
        GlobalTensor<ComputeT,
                     decltype(_gs),
                     pto::Stride<1, 1, 1, ChunkSize, 1>>
            _gm(workspace_qk_gated_handle +
                    static_cast<int64_t>(cid) * WsGatedSize,
                _gs);
        TLOAD(_l1, _gm);
      }
      // ── Load V [valid_rows × D] from GM → L1 ────────────────────────
      {
        L1Mat<ComputeT, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(
            valid_rows, HiddenSize);
        TASSIGN(_l1, 131072);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows;
        _gs.shape[4] = HiddenSize;
#ifdef MEGA_CHUNK_GDN_A5_PACKED_WUV
        GmStride2D _stride(HiddenSize);
#else
        GmStride2D _stride(BSND_V_STRIDE);
#endif
        GmTensor2D<ComputeT> _gm(V_handle + v_off, _gs, _stride);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }

      // ── GEMM 3: QKV = QK_gated @ V  (gated attention → values) ──────
      {
        TileLeft<ComputeT, ChunkSize, ChunkSize, ChunkSize, ChunkSize> _l0a;
        TileRight<ComputeT, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0b;
        TASSIGN(_l0a, 0x0);
        TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we);
        wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we);
        wait_flag(PIPE_M, PIPE_MTE1, _we);
        TEXTRACT(_l0a, qk_gated_l1, 0, 0);
        TEXTRACT(_l0b, v_l1, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we);
        wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(qkv_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we);
        wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we);
        wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // ── Store QKV [C × D] from L0C → GM workspace ───────────────────
      // ── Workspace buffer reuse ────────────────────────────────────────
      // workspace_qs_qkv_handle is shared between QS (GEMM 2 output) and QKV
      // (GEMM 3 output). This is safe because:
      //   1. Vec reads QS BEFORE Cube writes QKV to the same buffer
      //   2. The cross-core flags ensure proper ordering:
      //      - flag 0: QS ready (Vec reads QS)
      //      - flag 1: QK_gated ready (Vec done reading QS, Cube can write QKV)
      //      - flag 2: QKV ready (Vec reads QKV from same buffer)
      {
        TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(ChunkSize,
                                                                    HiddenSize);
        TASSIGN(_l0, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize;
        _gs.shape[4] = HiddenSize;
        GlobalTensor<ComputeT,
                     decltype(_gs),
                     pto::Stride<1, 1, 1, HiddenSize, 1>>
            _gm(workspace_qs_qkv_handle + static_cast<int64_t>(cid) * WsQSSize,
                _gs);
        TSTORE(_gm, _l0);
      }

      // Signal Vec: QKV is ready (flag 2, Cube→Vec)
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
      first_cube_iter = false;
    }
  } else if (use_precomputed_qs && reuse_group_qk != 0) {
    const int64_t group_work_count = h_o_chunk_count * Hg;
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE
    // A5 exposes intra-block event IDs 0..10. H and O are separated by a full
    // MIX barrier; reserve event 3 exclusively for the outer O-completion
    // handshake and drain the eight group events inside this branch.
    constexpr int32_t GroupFlowFlag = 0;
    constexpr int32_t GatedSlot0ReadyFlag = 1;
    constexpr int32_t QkvSlot0ReadyFlag = 2;
    constexpr int32_t GatedSlot1ReadyFlag = 4;
    constexpr int32_t QkvSlot1ReadyFlag = 5;
    constexpr int32_t GatedBottomReadyFlag = 6;
    constexpr int32_t QkvBottomReadyFlag = 7;
    constexpr int32_t QkvSlot2ReadyFlag = 8;
    static_assert(QkvSlot2ReadyFlag <= 10,
                  "A5 group-QK events must fit IDs 0..10");
#else
    constexpr int32_t GroupFlowFlag = 8;
    constexpr int32_t GatedSlot0ReadyFlag = 9;
    constexpr int32_t QkvSlot0ReadyFlag = 10;
    constexpr int32_t GatedSlot1ReadyFlag = 11;
    constexpr int32_t QkvSlot1ReadyFlag = 12;
    constexpr int32_t GatedBottomReadyFlag = 13;
    constexpr int32_t QkvBottomReadyFlag = 14;
    constexpr int32_t QkvSlot2ReadyFlag = 15;
#endif
    constexpr int32_t QkMailboxFreeFlag = QkvSlot2ReadyFlag;
    const bool use_inplace_lane2 = GROUP == 3;
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DOUBLE_MAILBOX
    // Variant20 keeps the KKT workspace's second per-core tile for the
    // lookahead QK. Variant19 no longer publishes H/O ready counters, so the
    // first KKT tile is free after the full H-to-O MIX barrier and can hold
    // lane2's in-place gated/QKV data without expanding workspace.
    __gm__ ComputeT* lane2_mailbox = workspace_ping_qk_handle;
#else
    __gm__ ComputeT* lane2_mailbox =
        workspace_ping_qk_handle + static_cast<int64_t>(block_num) * WsQKSize;
#endif
    const int64_t regular_group_rounds =
        group_work_count / static_cast<int64_t>(block_num);
    const bool rebalance_group_owners = h_o_heavy_core_count > 0 &&
                                        h_o_light_core_count > 0 &&
                                        regular_group_rounds > GROUP;
    const int64_t group_prefix_rounds = rebalance_group_owners
                                            ? regular_group_rounds - GROUP
                                            : regular_group_rounds;
    const int64_t group_balanced_span =
        group_prefix_rounds * static_cast<int64_t>(block_num);
    const int64_t group_tail_owner_shift =
        h_o_heavy_core_count % h_o_light_core_count;
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DOUBLE_MAILBOX
    bool current_qk_published = false;
#else
    bool first_group = true;
#endif
    int64_t group_owned_idx = 0;
    int64_t group_work = GetHoOwnedItem(group_owned_idx,
                                        group_work_count,
                                        static_cast<int64_t>(cid),
                                        static_cast<int64_t>(block_num),
                                        rebalance_group_owners,
                                        group_balanced_span,
                                        h_o_heavy_core_count,
                                        h_o_light_core_count,
                                        group_prefix_rounds,
                                        group_tail_owner_shift);

    while (group_work < group_work_count) {
#ifndef MEGA_CHUNK_GDN_A5_GROUP_QK_DOUBLE_MAILBOX
      if (!first_group) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
        // QK is overwritten by PIPE_FIX. Wait there for both AIV siblings to
        // finish their PIPE_MTE2 mailbox reads without stalling unrelated DMA.
        GroupDirectedWait<PIPE_FIX>(QkMailboxFreeFlag);
#else
        wait_flag_dev(QkMailboxFreeFlag);
#endif
      }
      __gm__ ComputeT* current_qk_mailbox = workspace_qk_handle;
#else
      const int32_t current_qk_slot = static_cast<int32_t>(group_owned_idx & 1);
      __gm__ ComputeT* current_qk_mailbox =
          current_qk_slot == 0 ? workspace_qk_handle
                               : workspace_ping_qk_handle +
                                     static_cast<int64_t>(block_num) * WsQKSize;
#endif

      const int64_t global_chunk_idx = group_work / Hg;
      const int32_t head_group =
          static_cast<int32_t>(group_work - global_chunk_idx * Hg);
      int64_t seq_idx = 0;
      int64_t bos = 0;
      int64_t slen = 0;
      int64_t local_chunk_idx = 0;
      if (!ResolveHoGlobalChunk<ChunkSize>(global_chunk_idx,
                                           batch_size,
                                           cu_seqlens,
                                           seq_idx,
                                           bos,
                                           slen,
                                           local_chunk_idx)) {
        break;
      }
      const int64_t chunk_start = local_chunk_idx * ChunkSize;
      const int64_t remaining = slen - chunk_start;
      const int32_t valid_rows =
          static_cast<int32_t>(remaining < ChunkSize ? remaining : ChunkSize);
      const int64_t chunk_token_start = bos + chunk_start;
      const int64_t qk_offset =
          (chunk_token_start * static_cast<int64_t>(Hg) + head_group) *
          static_cast<int64_t>(HiddenSize);
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DOUBLE_MAILBOX
      // Only the first round computes its current QK here. Later rounds use
      // the QK written by the preceding round's one-item lookahead.
      if (!current_qk_published) {
        if (group_owned_idx >= 2) {
          GroupDirectedWait<PIPE_FIX>(QkMailboxFreeFlag);
        }
        PublishQKTile<HiddenSize, ChunkSize, true>(Q_handle,
                                                   K_handle,
                                                   current_qk_mailbox,
                                                   static_cast<int64_t>(cid),
                                                   qk_offset,
                                                   BSND_QK_STRIDE,
                                                   valid_rows,
                                                   -1,
                                                   0u);
      }

      const int64_t next_owned_idx = group_owned_idx + 1;
      const int64_t next_group_work =
          GetHoOwnedItem(next_owned_idx,
                         group_work_count,
                         static_cast<int64_t>(cid),
                         static_cast<int64_t>(block_num),
                         rebalance_group_owners,
                         group_balanced_span,
                         h_o_heavy_core_count,
                         h_o_light_core_count,
                         group_prefix_rounds,
                         group_tail_owner_shift);
      bool next_qk_published = false;
      int64_t next_qk_offset = 0;
      int32_t next_valid_rows = 0;
      __gm__ ComputeT* next_qk_mailbox = workspace_qk_handle;
      if (next_group_work < group_work_count) {
        const int64_t next_global_chunk_idx = next_group_work / Hg;
        const int32_t next_head_group =
            static_cast<int32_t>(next_group_work - next_global_chunk_idx * Hg);
        int64_t next_seq_idx = 0;
        int64_t next_bos = 0;
        int64_t next_slen = 0;
        int64_t next_local_chunk_idx = 0;
        if (ResolveHoGlobalChunk<ChunkSize>(next_global_chunk_idx,
                                            batch_size,
                                            cu_seqlens,
                                            next_seq_idx,
                                            next_bos,
                                            next_slen,
                                            next_local_chunk_idx)) {
          const int64_t next_chunk_start = next_local_chunk_idx * ChunkSize;
          const int64_t next_remaining = next_slen - next_chunk_start;
          next_valid_rows = static_cast<int32_t>(
              next_remaining < ChunkSize ? next_remaining : ChunkSize);
          const int64_t next_chunk_token_start = next_bos + next_chunk_start;
          next_qk_offset = (next_chunk_token_start * static_cast<int64_t>(Hg) +
                            next_head_group) *
                           static_cast<int64_t>(HiddenSize);
          const int32_t next_qk_slot = static_cast<int32_t>(next_owned_idx & 1);
          next_qk_mailbox =
              next_qk_slot == 0
                  ? workspace_qk_handle
                  : workspace_ping_qk_handle +
                        static_cast<int64_t>(block_num) * WsQKSize;
          next_qk_published = true;
        }
      }

      // event8 is deliberately a single, slot-agnostic token. Consume the
      // free token for QK[i-1] before event0 admits the AIVs to QK[i]; this
      // prevents free(i-1) and free(i) from ever being simultaneously live.
      if (group_owned_idx > 0) {
        GroupDirectedWait<PIPE_FIX>(QkMailboxFreeFlag);
      }
      GroupDirectedSignal<PIPE_FIX>(1 | (2 << 4) | (GroupFlowFlag << 8));

      // Prewrite exactly one QK without publishing its ready token. The next
      // round's PIPE_FIX event0 signal orders this TSTORE before either AIV's
      // MTE2 read, while current gated/QKV processing remains strictly serial.
      if (next_qk_published) {
        PublishQKTile<HiddenSize, ChunkSize, true>(Q_handle,
                                                   K_handle,
                                                   next_qk_mailbox,
                                                   static_cast<int64_t>(cid),
                                                   next_qk_offset,
                                                   BSND_QK_STRIDE,
                                                   next_valid_rows,
                                                   -1,
                                                   0u);
      }
#else
      PublishQKTile<HiddenSize,
                    ChunkSize
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
                    ,
                    true
#endif
                    >(Q_handle,
                      K_handle,
                      current_qk_mailbox,
                      static_cast<int64_t>(cid),
                      qk_offset,
                      BSND_QK_STRIDE,
                      valid_rows,
                      -1,
                      0u);

#ifndef MEGA_CHUNK_GDN_A5_GROUP_QK_SKIP_HO_READY
      for (int32_t group_lane = 0; group_lane < GROUP; ++group_lane) {
        const int32_t head_idx = head_group * GROUP + group_lane;
        WaitHoChunkReady<ChunkSize>(
            h_o_ready_handle,
            static_cast<int32_t>(seq_idx * H + head_idx),
            local_chunk_idx);
      }
#endif
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
      GroupDirectedSignal<PIPE_FIX>(1 | (2 << 4) | (GroupFlowFlag << 8));
#else
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (GroupFlowFlag << 8));
#endif
#endif

      for (int32_t group_lane = 0; group_lane < GROUP; ++group_lane) {
        const int32_t head_idx = head_group * GROUP + group_lane;
#ifdef MEGA_CHUNK_GDN_A5_PACKED_WUV
        constexpr int32_t v_row_stride = HiddenSize;
        const int64_t v_offset =
            (static_cast<int64_t>(head_idx) * total_tokens +
             chunk_token_start) *
            static_cast<int64_t>(HiddenSize);
#else
        const bool use_packed_v = batch_size == 1 && (slen % ChunkSize) == 0;
        const int32_t v_row_stride = use_packed_v ? HiddenSize : BSND_V_STRIDE;
        const int64_t v_offset =
            use_packed_v
                ? (global_chunk_idx * H + head_idx) *
                      static_cast<int64_t>(ChunkSize) * HiddenSize
                : (chunk_token_start * static_cast<int64_t>(H) + head_idx) *
                      static_cast<int64_t>(HiddenSize);
#endif
        const int32_t slot = group_lane & 1;
        const bool inplace_lane2 = use_inplace_lane2 && group_lane == 2;
        __gm__ ComputeT* gated_mailbox =
            inplace_lane2 ? lane2_mailbox
                          : (slot == 0 ? workspace_qk_gated_handle
                                       : workspace_ping_qk_gated_handle);
        __gm__ ComputeT* qkv_mailbox =
            inplace_lane2 ? lane2_mailbox
                          : (slot == 0 ? workspace_qs_qkv_handle
                                       : workspace_ping_qs_qkv_handle);
        if (group_lane >= 2 && !inplace_lane2) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
          GroupDirectedWait<PIPE_MTE2>(GroupFlowFlag);
#else
          wait_flag_dev(GroupFlowFlag);
#endif
        }
        const int32_t gated_ready_flag =
            inplace_lane2
                ? GroupFlowFlag
                : (slot == 0 ? GatedSlot0ReadyFlag : GatedSlot1ReadyFlag);
        const int32_t qkv_ready_flag =
            inplace_lane2 ? QkvSlot2ReadyFlag
                          : (slot == 0 ? QkvSlot0ReadyFlag : QkvSlot1ReadyFlag);
        PublishQKVTile<HiddenSize,
                       ChunkSize
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
                       ,
                       true
#endif
                       >(V_handle,
                         gated_mailbox,
                         qkv_mailbox,
                         static_cast<int64_t>(cid),
                         v_offset,
                         v_row_stride,
                         valid_rows,
                         gated_ready_flag,
                         qkv_ready_flag,
                         GatedBottomReadyFlag,
                         QkvBottomReadyFlag);
      }
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DOUBLE_MAILBOX
      current_qk_published = next_qk_published;
      ++group_owned_idx;
      group_work = next_group_work;
#else
      first_group = false;
      ++group_owned_idx;
      group_work = GetHoOwnedItem(group_owned_idx,
                                  group_work_count,
                                  static_cast<int64_t>(cid),
                                  static_cast<int64_t>(block_num),
                                  rebalance_group_owners,
                                  group_balanced_span,
                                  h_o_heavy_core_count,
                                  h_o_light_core_count,
                                  group_prefix_rounds,
                                  group_tail_owner_shift);
#endif
    }

  } else if (use_precomputed_qs && batch_size == 1) {
    // Keep the next QK item in the other mailbox slot while the current
    // item waits for gating and runs QKV. This overlaps Vec gating for
    // item N+1 with Cube QKV for item N.
    const int64_t bos = static_cast<int64_t>(cu_seqlens[0]);
    const int64_t eos = static_cast<int64_t>(cu_seqlens[1]);
    const int64_t slen = eos - bos;
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
    constexpr int32_t v_row_stride = HiddenSize;
#else
    const bool use_packed_v = (slen % ChunkSize) == 0;
    const int32_t v_row_stride = use_packed_v ? HiddenSize : BSND_V_STRIDE;
    const int64_t v_sequence_base =
        use_packed_v ? 0 : bos * static_cast<int64_t>(H) * HiddenSize;
    const int64_t v_head_stride =
        use_packed_v ? static_cast<int64_t>(ChunkSize) * HiddenSize
                     : HiddenSize;
    const int64_t v_chunk_stride =
        static_cast<int64_t>(H) * ChunkSize * HiddenSize;
#endif
    int64_t owned_idx = 0;
    int64_t current_item = GetHoOwnedItem(0,
                                          h_o_total_items,
                                          static_cast<int64_t>(cid),
                                          static_cast<int64_t>(block_num),
                                          rebalance_h_o_consumers,
                                          h_o_balanced_span,
                                          h_o_heavy_core_count,
                                          h_o_light_core_count,
                                          h_o_owned_prefix_count,
                                          h_o_tail_owner_shift);
    bool current_qk_published = false;

    while (current_item < h_o_total_items) {
      const int64_t ci = current_item / H;
      const int32_t head_idx = static_cast<int32_t>(current_item - ci * H);
      const int64_t chunk_start = ci * ChunkSize;
      const int64_t remaining = slen - chunk_start;
      const int32_t valid_rows =
          static_cast<int32_t>(remaining < ChunkSize ? remaining : ChunkSize);
      const int64_t chunk_token_start = bos + chunk_start;
      const int32_t head_group = head_idx / GROUP;
      const int64_t qk_offset =
          (chunk_token_start * static_cast<int64_t>(Hg) + head_group) *
          static_cast<int64_t>(HiddenSize);
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
      const int64_t v_offset =
          (static_cast<int64_t>(head_idx) * total_tokens + chunk_token_start) *
          static_cast<int64_t>(HiddenSize);
#else
      const int64_t v_offset = v_sequence_base + ci * v_chunk_stride +
                               static_cast<int64_t>(head_idx) * v_head_stride;
#endif
      const int32_t slot = static_cast<int32_t>(owned_idx & 1);
      __gm__ ComputeT* qk_mailbox =
          slot == 0 ? workspace_qk_handle
                    : workspace_ping_qk_handle +
                          static_cast<int64_t>(block_num) * WsQKSize;
      __gm__ ComputeT* qkv_mailbox =
          slot == 0 ? workspace_qs_qkv_handle : workspace_ping_qs_qkv_handle;
      __gm__ ComputeT* gated_mailbox = slot == 0
                                           ? workspace_qk_gated_handle
                                           : workspace_ping_qk_gated_handle;

      if (!current_qk_published && reuse_group_qk == 0) {
        if (!decouple_qk_h_ready) {
          WaitHoChunkReady<ChunkSize>(h_o_ready_handle, head_idx, ci);
        }
        PublishQKTile<HiddenSize, ChunkSize>(Q_handle,
                                             K_handle,
                                             qk_mailbox,
                                             static_cast<int64_t>(cid),
                                             qk_offset,
                                             BSND_QK_STRIDE,
                                             valid_rows,
                                             slot,
                                             0u);
      }

      const int64_t next_item = GetHoOwnedItem(owned_idx + 1,
                                               h_o_total_items,
                                               static_cast<int64_t>(cid),
                                               static_cast<int64_t>(block_num),
                                               rebalance_h_o_consumers,
                                               h_o_balanced_span,
                                               h_o_heavy_core_count,
                                               h_o_light_core_count,
                                               h_o_owned_prefix_count,
                                               h_o_tail_owner_shift);
      if (next_item < h_o_total_items && reuse_group_qk == 0) {
        const int64_t next_owned_idx = owned_idx + 1;
        const int32_t next_slot = static_cast<int32_t>(next_owned_idx & 1);
        if (next_owned_idx >= 2) {
          wait_flag_dev(6 + next_slot);
        }

        const int64_t next_ci = next_item / H;
        const int32_t next_head_idx =
            static_cast<int32_t>(next_item - next_ci * H);
        const int64_t next_chunk_start = next_ci * ChunkSize;
        const int64_t next_remaining = slen - next_chunk_start;
        const int32_t next_valid_rows = static_cast<int32_t>(
            next_remaining < ChunkSize ? next_remaining : ChunkSize);
        const int64_t next_chunk_token_start = bos + next_chunk_start;
        const int32_t next_head_group = next_head_idx / GROUP;
        const int64_t next_qk_offset =
            (next_chunk_token_start * static_cast<int64_t>(Hg) +
             next_head_group) *
            static_cast<int64_t>(HiddenSize);
        __gm__ ComputeT* next_qk_mailbox =
            next_slot == 0 ? workspace_qk_handle
                           : workspace_ping_qk_handle +
                                 static_cast<int64_t>(block_num) * WsQKSize;

        if (!decouple_qk_h_ready) {
          WaitHoChunkReady<ChunkSize>(h_o_ready_handle, next_head_idx, next_ci);
        }
        PublishQKTile<HiddenSize, ChunkSize>(Q_handle,
                                             K_handle,
                                             next_qk_mailbox,
                                             static_cast<int64_t>(cid),
                                             next_qk_offset,
                                             BSND_QK_STRIDE,
                                             next_valid_rows,
                                             next_slot,
                                             0u);
      }

      // For head-rich shapes, Q/K are independent of the H-stage output:
      // publish QK early and wait only when V/precomputed QS are consumed.
      // Small-head shapes retain the original ordering because their short
      // owner queues cannot safely absorb this lookahead.
      if (decouple_qk_h_ready) {
        WaitHoChunkReady<ChunkSize>(h_o_ready_handle, head_idx, ci);
      }
      PublishQKVTile<HiddenSize, ChunkSize>(V_handle,
                                            gated_mailbox,
                                            qkv_mailbox,
                                            static_cast<int64_t>(cid),
                                            v_offset,
                                            v_row_stride,
                                            valid_rows,
                                            2 + slot,
                                            4 + slot);

      current_item = next_item;
      current_qk_published = reuse_group_qk != 0 || next_item < h_o_total_items;
      ++owned_idx;
    }
  } else {
    // ── Variable-length sequence path (cu_seqlens != nullptr) ──────────
    int64_t gi = 0;
    int64_t chunk_global_idx = 0;
    bool first_cube_iter_v = true;
    int64_t cube_owned_idx_v = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < H; ++h) {
          int64_t consumer_owner = gi % static_cast<int64_t>(block_num);
          if (rebalance_h_o_consumers && gi >= h_o_balanced_span) {
            consumer_owner = h_o_heavy_core_count +
                             (gi - h_o_balanced_span) % h_o_light_core_count;
          }
          if (consumer_owner == static_cast<int64_t>(cid)) {
            const int32_t mailbox_slot =
                use_precomputed_qs ? static_cast<int32_t>(cube_owned_idx_v & 1)
                                   : 0;
            if (use_precomputed_qs) {
              if (cube_owned_idx_v >= 2) {
                wait_flag_dev(6 + mailbox_slot);
              }
            } else if (!first_cube_iter_v) {
#if defined(PTO_NPU_ARCH_A5)
              gdn_sync::Wait<PIPE_FIX>(3);
#else
              wait_flag_dev(3);
#endif
            }
            __gm__ ComputeT* qk_mailbox =
                mailbox_slot == 0
                    ? workspace_qk_handle
                    : workspace_ping_qk_handle +
                          static_cast<int64_t>(block_num) * WsQKSize;
            __gm__ ComputeT* qs_qkv_mailbox =
                mailbox_slot == 0 ? workspace_qs_qkv_handle
                                  : workspace_ping_qs_qkv_handle;
            __gm__ ComputeT* qk_gated_mailbox =
                mailbox_slot == 0 ? workspace_qk_gated_handle
                                  : workspace_ping_qk_gated_handle;
            const int32_t qk_ready_flag = use_precomputed_qs ? mailbox_slot : 0;
            const int32_t qk_gated_ready_flag =
                use_precomputed_qs ? 2 + mailbox_slot : 1;
            const int32_t qkv_ready_flag =
                use_precomputed_qs ? 4 + mailbox_slot : 2;
            if (use_precomputed_qs) {
              const int64_t ready_offset =
                  (si * static_cast<int64_t>(H) + h) * H_O_READY_STRIDE;
              const int32_t required_count = static_cast<int32_t>(ci + 1);
              while (true) {
                __asm__ __volatile__("");
                AscendC::DataCacheCleanAndInvalid<
                    int32_t,
                    AscendC::CacheLine::SINGLE_CACHE_LINE,
                    AscendC::DcciDst::CACHELINE_OUT>(
                    h_o_ready_gm[ready_offset]);
                __asm__ __volatile__("");
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
                dsb(DSB_DDR);
#endif
                const int32_t ready_count = h_o_ready_gm.GetValue(ready_offset);
                if (ready_count >= required_count) {
                  break;
                }
              }
            }
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t head_idx = h;
            int32_t head_g = head_idx / GROUP;

            int64_t qk_off = (chunk_token_start * static_cast<int64_t>(Hg) +
                              static_cast<int64_t>(head_g)) *
                             static_cast<int64_t>(HiddenSize);
#ifdef MEGA_CHUNK_GDN_A5_PACKED_WUV
            int64_t v_off = (static_cast<int64_t>(head_idx) * total_tokens +
                             chunk_token_start) *
                            static_cast<int64_t>(HiddenSize);
#else
            int64_t v_off = (chunk_token_start * static_cast<int64_t>(H) +
                             static_cast<int64_t>(head_idx)) *
                            static_cast<int64_t>(HiddenSize);
#endif
            int64_t s_offset = (chunk_global_idx * H + head_idx) *
                               static_cast<int64_t>(HiddenSize) *
                               static_cast<int64_t>(HiddenSize);

            // Load Q
            {
              L1Mat<ComputeT, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(
                  valid_rows, HiddenSize);
              TASSIGN(_l1, 0);
              GmShape2D _gs(valid_rows, HiddenSize);
              GmStride2D _stride(BSND_QK_STRIDE);
              GmTensor2D<ComputeT> _gm(Q_handle + qk_off, _gs, _stride);
              TLOAD(_l1, _gm);
              if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
            }
            // Load K
            {
              L1Mat<ComputeT, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(
                  valid_rows, HiddenSize);
              TASSIGN(_l1, 32768);
              GmShape2D _gs(valid_rows, HiddenSize);
              GmStride2D _stride(BSND_QK_STRIDE);
              GmTensor2D<ComputeT> _gm(K_handle + qk_off, _gs, _stride);
              TLOAD(_l1, _gm);
              if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
            }

            // GEMM 1: QK = Q @ K^T (transpose_B via TRESHAPE NZ→ZN)
            {
              TileLeft<ComputeT, ChunkSize, HiddenSize, ChunkSize, HiddenSize>
                  _l0a;
              TileRight<ComputeT, HiddenSize, ChunkSize, HiddenSize, ChunkSize>
                  _l0b;
              TASSIGN(_l0a, 0x0);
              TASSIGN(_l0b, 0x0);
              auto _we = EVENT_ID1;
              set_flag(PIPE_MTE2, PIPE_MTE1, _we);
              wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
              set_flag(PIPE_M, PIPE_MTE1, _we);
              wait_flag(PIPE_M, PIPE_MTE1, _we);
              TEXTRACT(_l0a, q_l1, 0, 0);
              L1MatZN<ComputeT, HiddenSize, ChunkSize> _bzn;
              TRESHAPE(_bzn, k_l1);
              TEXTRACT(_l0b, _bzn, 0, 0);
              set_flag(PIPE_MTE1, PIPE_M, _we);
              wait_flag(PIPE_MTE1, PIPE_M, _we);
              TMATMUL(qk_l0, _l0a, _l0b);
              set_flag(PIPE_MTE1, PIPE_MTE2, _we);
              wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
              set_flag(PIPE_M, PIPE_FIX, _we);
              wait_flag(PIPE_M, PIPE_FIX, _we);
            }

            // H has already consumed S and replaced this slot with Q @ S.
            if (!use_precomputed_qs) {
              L1Mat<ComputeT, HiddenSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(
                  HiddenSize, HiddenSize);
              TASSIGN(_l1, 65536);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = HiddenSize;
              _gs.shape[4] = HiddenSize;
              GlobalTensor<ComputeT,
                           decltype(_gs),
                           pto::Stride<1, 1, 1, HiddenSize, 1>>
                  _gm(S_handle + s_offset, _gs);
              TLOAD(_l1, _gm);
            }

            // GEMM 2: QS = Q @ S
            if (!use_precomputed_qs) {
              TileLeft<ComputeT, ChunkSize, HiddenSize, ChunkSize, HiddenSize>
                  _l0a;
              TileRight<ComputeT,
                        HiddenSize,
                        HiddenSize,
                        HiddenSize,
                        HiddenSize>
                  _l0b;
              TASSIGN(_l0a, 0x0);
              TASSIGN(_l0b, 0x0);
              auto _we = EVENT_ID1;
              set_flag(PIPE_MTE2, PIPE_MTE1, _we);
              wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
              set_flag(PIPE_M, PIPE_MTE1, _we);
              wait_flag(PIPE_M, PIPE_MTE1, _we);
              TEXTRACT(_l0a, q_l1, 0, 0);
              TEXTRACT(_l0b, s_l1, 0, 0);
              set_flag(PIPE_MTE1, PIPE_M, _we);
              wait_flag(PIPE_MTE1, PIPE_M, _we);
              TMATMUL(qs_l0, _l0a, _l0b);
              set_flag(PIPE_MTE1, PIPE_MTE2, _we);
              wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
              set_flag(PIPE_M, PIPE_FIX, _we);
              wait_flag(PIPE_M, PIPE_FIX, _we);
            }

            // Store QK → workspace
            {
              TileAcc<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l0(
                  ChunkSize, ChunkSize);
              TASSIGN(_l0, 0);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize;
              _gs.shape[4] = ChunkSize;
              GlobalTensor<ComputeT,
                           decltype(_gs),
                           pto::Stride<1, 1, 1, ChunkSize, 1>>
                  _gm(qk_mailbox + static_cast<int64_t>(cid) * WsQKSize, _gs);
              TSTORE(_gm, _l0);
            }

            // Store QS on the legacy path. The pipelined path reads the
            // precomputed value directly from S_handle.
            if (!use_precomputed_qs) {
              TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(
                  ChunkSize, HiddenSize);
              TASSIGN(_l0, SecondL0CAddr);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize;
              _gs.shape[4] = HiddenSize;
              GlobalTensor<ComputeT,
                           decltype(_gs),
                           pto::Stride<1, 1, 1, HiddenSize, 1>>
                  _gm(qs_qkv_mailbox + static_cast<int64_t>(cid) * WsQSSize,
                      _gs);
              TSTORE(_gm, _l0);
            }

            // Cube→Vec: QK & QS ready (flag 0)
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (qk_ready_flag << 8));

            // Wait Vec→Cube: QK_gated ready (flag 1)
            if (use_precomputed_qs) {
#if defined(PTO_NPU_ARCH_A5)
              gdn_sync::Wait<PIPE_MTE2>(qk_gated_ready_flag);
#else
              wait_flag_dev(qk_gated_ready_flag);
#endif
            } else {
#if defined(PTO_NPU_ARCH_A5)
              gdn_sync::Wait<PIPE_MTE2>(1);
#else
              wait_flag_dev(1);
#endif
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

            // Load QK_gated
            {
              L1Mat<ComputeT, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(
                  ChunkSize, ChunkSize);
              TASSIGN(_l1, 98304);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize;
              _gs.shape[4] = ChunkSize;
              GlobalTensor<ComputeT,
                           decltype(_gs),
                           pto::Stride<1, 1, 1, ChunkSize, 1>>
                  _gm(qk_gated_mailbox +
                          static_cast<int64_t>(cid) * WsGatedSize,
                      _gs);
              TLOAD(_l1, _gm);
            }
            // Load V
            {
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
              if (use_precomputed_qs) {
                constexpr int32_t DcciCacheLineElems =
                    64 / static_cast<int32_t>(sizeof(ComputeT));
                for (int32_t row = 0; row < valid_rows; ++row) {
                  for (int32_t r = 0; r < HiddenSize; r += DcciCacheLineElems) {
                    dcci(static_cast<__gm__ void*>(V_handle + v_off +
                                                   row * HiddenSize + r),
                         SINGLE_CACHE_LINE);
                  }
                }
                dsb(DSB_DDR);
              }
#endif
              L1Mat<ComputeT, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(
                  valid_rows, HiddenSize);
              TASSIGN(_l1, 131072);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows;
              _gs.shape[4] = HiddenSize;
#ifdef MEGA_CHUNK_GDN_A5_PACKED_WUV
              GmStride2D _stride(HiddenSize);
#else
              GmStride2D _stride(BSND_V_STRIDE);
#endif
              GmTensor2D<ComputeT> _gm(V_handle + v_off, _gs, _stride);
              TLOAD(_l1, _gm);
              if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
            }

            // GEMM 3: QKV = QK_gated @ V
            {
              TileLeft<ComputeT, ChunkSize, ChunkSize, ChunkSize, ChunkSize>
                  _l0a;
              TileRight<ComputeT, ChunkSize, HiddenSize, ChunkSize, HiddenSize>
                  _l0b;
              TASSIGN(_l0a, 0x0);
              TASSIGN(_l0b, 0x0);
              auto _we = EVENT_ID1;
              set_flag(PIPE_MTE2, PIPE_MTE1, _we);
              wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
              set_flag(PIPE_M, PIPE_MTE1, _we);
              wait_flag(PIPE_M, PIPE_MTE1, _we);
              TEXTRACT(_l0a, qk_gated_l1, 0, 0);
              TEXTRACT(_l0b, v_l1, 0, 0);
              set_flag(PIPE_MTE1, PIPE_M, _we);
              wait_flag(PIPE_MTE1, PIPE_M, _we);
              TMATMUL(qkv_l0, _l0a, _l0b);
              set_flag(PIPE_MTE1, PIPE_MTE2, _we);
              wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
              set_flag(PIPE_M, PIPE_FIX, _we);
              wait_flag(PIPE_M, PIPE_FIX, _we);
            }

            {
              TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(
                  ChunkSize, HiddenSize);
              TASSIGN(_l0, 0);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize;
              _gs.shape[4] = HiddenSize;
              GlobalTensor<ComputeT,
                           decltype(_gs),
                           pto::Stride<1, 1, 1, HiddenSize, 1>>
                  _gm(qs_qkv_mailbox + static_cast<int64_t>(cid) * WsQSSize,
                      _gs);
              TSTORE(_gm, _l0);
            }

            ffts_cross_core_sync(PIPE_FIX,
                                 1 | (2 << 4) | (qkv_ready_flag << 8));
            first_cube_iter_v = false;
            ++cube_owned_idx_v;
          }
          gi++;
        }
        chunk_global_idx++;
      }
    }
  }
#endif

// =====================================================================
// VEC CORE — Gating, element-wise ops, output assembly
// Two Vec sub-blocks (vid=0,1) process upper/lower C/2 rows in parallel.
// Each sub-block independently:
//   1. Computes gating coefficients from G and the causal mask
//   2. Applies gating to the Cube's QK result → QK_gated
//   3. Scales the Cube's QS result by exp(g)
//   4. Combines QKV + scaled QS → final output O
// =====================================================================
#if defined(__DAV_C220_VEC__)
  // Vec engine initialization: set_mask_norm selects "normal" masking mode,
  // and set_vector_mask(-1, -1) enables ALL SIMD lanes (no masking).
  set_mask_norm();
  set_vector_mask(-1, -1);

  constexpr int32_t NormWeightBf16Addr = 180992;
  constexpr int32_t NormWeightFp32Addr = 181504;
  if constexpr (FuseGatedRmsNorm) {
    Shape<1, 1, 1, 1, HiddenSize> shape;
    GlobalTensor<GDN_PUBLIC_DTYPE,
                 decltype(shape),
                 Stride<1, 1, 1, HiddenSize, 1>>
        weight_global(norm_weight_handle, shape);
    UbND<GDN_PUBLIC_DTYPE, 1, HiddenSize> weight_load;
    TASSIGN(weight_load, NormWeightBf16Addr);
    TLOAD(weight_load, weight_global);
  }

  // ── Load causal mask once (reused across all chunks) ─────────────────
  // ── Causal mask (loaded once, reused) ─────────────────────────────────
  // The causal mask is a C×C lower-triangular matrix of 0s and 1s:
  //   mask[i,j] = 1 if i >= j else 0
  // Each sub-block loads its C/2 rows. Applied via TMUL to zero out
  // non-causal (future) attention scores.
  //
  // Each sub-block (vid=0,1) loads its C/2 rows of the C×C lower-tri mask.
  {
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = HalfChunk;
    _gs.shape[4] = ChunkSize;
    GlobalTensor<float, decltype(_gs), pto::Stride<1, 1, 1, ChunkSize, 1>> _gm(
        Msk_handle + static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
    UbND<float, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(
        HalfChunk, ChunkSize);
    TASSIGN(_ld, MskUbAddr);
    TLOAD(_ld, _gm);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  if constexpr (FuseGatedRmsNorm) {
    UbND<GDN_PUBLIC_DTYPE, 1, HiddenSize> norm_weight_public;
    TASSIGN(norm_weight_public, NormWeightBf16Addr);
    UbND<float, 1, HiddenSize> norm_weight_fp32;
    TASSIGN(norm_weight_fp32, NormWeightFp32Addr);
    TCVT(norm_weight_fp32, norm_weight_public, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
  }

  if (cu_seqlens == nullptr) {
    // ── Fixed-length sequence path ──────────────────────────────────────
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;

    for (int64_t work_idx = static_cast<int64_t>(cid); work_idx < total_work;
         work_idx += static_cast<int64_t>(block_num)) {
      int32_t head_idx = static_cast<int32_t>(work_idx % H);
      int64_t chunk_head_idx = work_idx / H;
      int64_t seq_idx = chunk_head_idx / chunks_per_seq;
      int64_t ci = chunk_head_idx % chunks_per_seq;

      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows =
          static_cast<int32_t>(remaining < ChunkSize ? remaining : ChunkSize);
      int64_t chunk_token_start = bos + chunk_start;
      int32_t row_offset = static_cast<int32_t>(vid) * HalfChunk;
      int32_t local_rows = valid_rows - row_offset;
      if (local_rows < 0) local_rows = 0;
      if (local_rows > HalfChunk) local_rows = HalfChunk;

      if (local_rows > 0) {
        // ── Load G [1 × valid_rows] — gate values for this chunk ────────
        // G is pre-transposed to [H, total_tokens], contiguous per head.
        {
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = 1;
          _gs.shape[4] = valid_rows;
          GlobalTensor<float, decltype(_gs), pto::Stride<1, 1, 1, 1, 1>> _gm(
              G_handle + static_cast<int64_t>(head_idx) * total_tokens +
                  chunk_token_start,
              _gs);
          UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(
              1, valid_rows);
          TASSIGN(_ld, GUbAddr);
          TLOAD(_ld, _gm);
          if (valid_rows != ChunkSize) {
            UbND<float, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
            TASSIGN(_pd, GUbAddr);
            TFILLPAD_INPLACE(_pd, _ld);
          }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // ── Compute gating coefficients ──────────────────────────────────
        // ── Gating coefficient computation (numpy pseudocode) ─────────────
        // For this sub-block's rows (vid=0: rows 0..C/2-1, vid=1: rows
        // C/2..C-1):
        //
        //   g_row = g[my_start:my_start+C/2]    # my gates (shape [C/2])
        //   g_col = g[0:C]                       # full chunk gates (shape [C])
        //
        //   # Broadcast to 2D matrices:
        //   g_r_2d = g_row[:, None] * np.ones((1, C))    # TROWEXPAND: [C/2, C]
        //   g_c_2d = np.ones((C/2, 1)) * g_col[None, :]  # TCOLEXPAND: [C/2, C]
        //   coeff = exp(min(g_r_2d - g_c_2d, 0)) * mask
        //
        //   # Also compute exp(g_row) for QS scaling:
        //   exp_g_row = np.exp(g_row)                     # TEXP
        UbND<float, 1, HalfChunk> g_ub_temp_0;
        TASSIGN(g_ub_temp_0,
                GUbAddr + static_cast<int32_t>(vid) * HalfChunk *
                              static_cast<int32_t>(sizeof(float)));
        TMOV(g_v_ub, g_ub_temp_0);

        // Broadcast g_row into [C/2 × C] and g_col into [C/2 × C]
        UbND<float, HalfChunk, ChunkSize> g_r_2d;
        TASSIGN(g_r_2d, QSUbAddr);
        UbDN<float, HalfChunk, 1> g_v_col;
        TASSIGN(g_v_col, GvUbAddr);
        TROWEXPAND(g_r_2d, g_v_col);       // g_r_2d[i,j] = g_row[i]
        TCOLEXPAND(coeff_ub, g_ub);        // coeff[i,j] = g_col[j]
        TSUB(coeff_ub, g_r_2d, coeff_ub);  // d = g_row - g_col
        pipe_barrier(PIPE_V);
        TMINS(coeff_ub, coeff_ub, 0.0f);
        pipe_barrier(PIPE_V);
        TEXP(coeff_ub, coeff_ub);
        pipe_barrier(PIPE_V);
        TMUL(coeff_ub, coeff_ub, msk_ub);
        pipe_barrier(PIPE_V);
        TEXP(g_v_ub, g_v_ub);  // exp(g_row) for QS scaling
      }

      // ── Wait for Cube→Vec flag 0: QK & QS ready ─────────────────────
      wait_flag_dev(0);
      if (local_rows == 0) {
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
        wait_flag_dev(2);
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
        continue;
      }

      // ── Load QK [C/2 × C] from workspace → UB ───────────────────────
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = local_rows;
        _gs.shape[4] = ChunkSize;
        GlobalTensor<ComputeT,
                     decltype(_gs),
                     pto::Stride<1, 1, 1, ChunkSize, 1>>
            _gm(workspace_qk_handle + static_cast<int64_t>(cid) * WsQKSize +
                    static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                _gs);
        UbND<ComputeT, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero>
            _ld(local_rows, ChunkSize);
        TASSIGN(_ld, QKHalfUbAddr);
        TLOAD(_ld, _gm);
        if (local_rows != HalfChunk) {
          TFILLPAD_INPLACE(qk_ub_half, _ld);
        }
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(qk_ub, qk_ub_half, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

      // ── Load QS [C/2 × D] from workspace → UB ───────────────────────
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = local_rows;
        _gs.shape[4] = HiddenSize;
        GlobalTensor<ComputeT,
                     decltype(_gs),
                     pto::Stride<1, 1, 1, HiddenSize, 1>>
            _gm(workspace_qs_qkv_handle + static_cast<int64_t>(cid) * WsQSSize +
                    static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                _gs);
        UbND<ComputeT, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC, PadValue::Zero>
            _ld(local_rows, HiddenSize);
        TASSIGN(_ld, QSHalfUbAddr);
        TLOAD(_ld, _gm);
        if (local_rows != HalfChunk) {
          TFILLPAD_INPLACE(qs_ub_half, _ld);
        }
      }

      // ── Apply gating: QK_gated = QK * exp(d*mask)*mask
      TMUL(qk_ub, qk_ub, coeff_ub);
      TCVT(qk_ub_half, qk_ub, pto::RoundMode::CAST_NONE);

      // ── Store QK_gated [C/2 × C] → workspace for Cube's GEMM 3 ─────
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = local_rows;
        _gs.shape[4] = ChunkSize;
        GlobalTensor<ComputeT,
                     decltype(_gs),
                     pto::Stride<1, 1, 1, ChunkSize, 1>>
            _gm(workspace_qk_gated_handle +
                    static_cast<int64_t>(cid) * WsGatedSize +
                    static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                _gs);
        UbND<ComputeT, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(local_rows,
                                                                   ChunkSize);
        TASSIGN(_st, QKHalfUbAddr);
        TSTORE(_gm, _st);
      }
      // Vec→Cube: QK_gated ready (flag 1)
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));

      // ── Scale QS by exp(g): QS_gated = QS * exp(g_row) ──────────────
      // ── Scale QS by exp(g): inter-chunk state contribution ────────────
      // numpy: QS_scaled = QS * np.exp(g_row)[:, None]   (broadcast across D
      // columns) TROWEXPAND broadcasts the scalar exp(g[i]) for each row i
      // across all D columns, then TMUL applies it element-wise. This gates how
      // much the accumulated state contributes to each token's output.
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(qs_ub, qs_ub_half, pto::RoundMode::CAST_NONE);
      UbND<float, HalfChunk, HiddenSize> g_exp_2d;
      TASSIGN(g_exp_2d, CoeffUbAddr);
      UbDN<float, HalfChunk, 1> g_v_col2;
      TASSIGN(g_v_col2, GvUbAddr);
      TROWEXPAND(g_exp_2d, g_v_col2);  // broadcast exp(g_row) across columns
      pipe_barrier(PIPE_V);
      TMUL(qs_ub, qs_ub, g_exp_2d);  // QS_gated = QS * exp(g_row)

      // ── Wait for Cube→Vec flag 2: QKV ready ─────────────────────────
      wait_flag_dev(2);

      // ── Load QKV [C/2 × D] from workspace → UB ──────────────────────
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = local_rows;
        _gs.shape[4] = HiddenSize;
        GlobalTensor<ComputeT,
                     decltype(_gs),
                     pto::Stride<1, 1, 1, HiddenSize, 1>>
            _gm(workspace_qs_qkv_handle + static_cast<int64_t>(cid) * WsQSSize +
                    static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                _gs);
        UbND<ComputeT, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC, PadValue::Zero>
            _ld(local_rows, HiddenSize);
        TASSIGN(_ld, OHalfUbAddr);
        TLOAD(_ld, _gm);
        if (local_rows != HalfChunk) {
          TFILLPAD_INPLACE(o_ub_half, _ld);
        }
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      // ── Combine: O = QS_gated + QKV ─────────────────────────────────
      // ── Final output: O = QKV + QS_scaled ─────────────────────────────
      // numpy: O = (QK_gated @ V) + (Q @ S) * exp(g)[:, None]
      //       = intra_chunk_attention + inter_chunk_state_contribution
      // TCVT ComputeT→float for QKV, then add the state contribution.
      TCVT(o_ub, o_ub_half, pto::RoundMode::CAST_NONE);
      TADD(o_ub, qs_ub, o_ub);

      int64_t o_offset = (chunk_token_start * static_cast<int64_t>(H) +
                          static_cast<int64_t>(head_idx)) *
                             static_cast<int64_t>(HiddenSize) +
                         static_cast<int64_t>(vid) * HalfChunk *
                             static_cast<int64_t>(BSND_V_STRIDE);
      StoreChunkOutput<HiddenSize, ChunkSize, FuseGatedRmsNorm>(
          O_handle, z_handle, o_offset, BSND_V_STRIDE, local_rows);

      // Vec→Cube: done with this chunk (flag 3)
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
    }
  } else if (use_precomputed_qs && reuse_group_qk != 0) {
    const int64_t group_work_count = h_o_chunk_count * Hg;
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE
    constexpr int32_t GroupFlowFlag = 0;
    constexpr int32_t GatedSlot0ReadyFlag = 1;
    constexpr int32_t QkvSlot0ReadyFlag = 2;
    constexpr int32_t GatedSlot1ReadyFlag = 4;
    constexpr int32_t QkvSlot1ReadyFlag = 5;
    constexpr int32_t GatedBottomReadyFlag = 6;
    constexpr int32_t QkvBottomReadyFlag = 7;
    constexpr int32_t QkvSlot2ReadyFlag = 8;
    static_assert(QkvSlot2ReadyFlag <= 10,
                  "A5 group-QK events must fit IDs 0..10");
#else
    constexpr int32_t GroupFlowFlag = 8;
    constexpr int32_t GatedSlot0ReadyFlag = 9;
    constexpr int32_t QkvSlot0ReadyFlag = 10;
    constexpr int32_t GatedSlot1ReadyFlag = 11;
    constexpr int32_t QkvSlot1ReadyFlag = 12;
    constexpr int32_t GatedBottomReadyFlag = 13;
    constexpr int32_t QkvBottomReadyFlag = 14;
    constexpr int32_t QkvSlot2ReadyFlag = 15;
#endif
    constexpr int32_t QkMailboxFreeFlag = QkvSlot2ReadyFlag;
    const bool use_inplace_lane2 = GROUP == 3;
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DOUBLE_MAILBOX
    __gm__ ComputeT* lane2_mailbox = workspace_ping_qk_handle;
#else
    __gm__ ComputeT* lane2_mailbox =
        workspace_ping_qk_handle + static_cast<int64_t>(block_num) * WsQKSize;
#endif
    const int64_t regular_group_rounds =
        group_work_count / static_cast<int64_t>(block_num);
    const bool rebalance_group_owners = h_o_heavy_core_count > 0 &&
                                        h_o_light_core_count > 0 &&
                                        regular_group_rounds > GROUP;
    const int64_t group_prefix_rounds = rebalance_group_owners
                                            ? regular_group_rounds - GROUP
                                            : regular_group_rounds;
    const int64_t group_balanced_span =
        group_prefix_rounds * static_cast<int64_t>(block_num);
    const int64_t group_tail_owner_shift =
        h_o_heavy_core_count % h_o_light_core_count;
    bool produced_output = false;
    constexpr int32_t GateSlot0Addr = GUbAddr;
    constexpr int32_t GateSlot1Addr = GvUbAddr;
    int64_t group_owned_idx = 0;
    int64_t group_work = GetHoOwnedItem(group_owned_idx,
                                        group_work_count,
                                        static_cast<int64_t>(cid),
                                        static_cast<int64_t>(block_num),
                                        rebalance_group_owners,
                                        group_balanced_span,
                                        h_o_heavy_core_count,
                                        h_o_light_core_count,
                                        group_prefix_rounds,
                                        group_tail_owner_shift);

    while (group_work < group_work_count) {
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DOUBLE_MAILBOX
      const int32_t current_qk_slot = static_cast<int32_t>(group_owned_idx & 1);
      __gm__ ComputeT* current_qk_mailbox =
          current_qk_slot == 0 ? workspace_qk_handle
                               : workspace_ping_qk_handle +
                                     static_cast<int64_t>(block_num) * WsQKSize;
#else
      __gm__ ComputeT* current_qk_mailbox = workspace_qk_handle;
#endif
      const int64_t global_chunk_idx = group_work / Hg;
      const int32_t head_group =
          static_cast<int32_t>(group_work - global_chunk_idx * Hg);
      int64_t seq_idx = 0;
      int64_t bos = 0;
      int64_t slen = 0;
      int64_t local_chunk_idx = 0;
      if (!ResolveHoGlobalChunk<ChunkSize>(global_chunk_idx,
                                           batch_size,
                                           cu_seqlens,
                                           seq_idx,
                                           bos,
                                           slen,
                                           local_chunk_idx)) {
        break;
      }
      const int64_t chunk_start = local_chunk_idx * ChunkSize;
      const int64_t remaining = slen - chunk_start;
      const int32_t valid_rows =
          static_cast<int32_t>(remaining < ChunkSize ? remaining : ChunkSize);
      const int64_t chunk_token_start = bos + chunk_start;
      const int32_t row_offset = static_cast<int32_t>(vid) * HalfChunk;
      int32_t local_rows = valid_rows - row_offset;
      if (local_rows < 0) {
        local_rows = 0;
      }
      if (local_rows > HalfChunk) {
        local_rows = HalfChunk;
      }
      for (int32_t group_lane = 0; group_lane < GROUP; ++group_lane) {
        const int32_t head_idx = head_group * GROUP + group_lane;
        const int32_t slot = group_lane & 1;
        const bool inplace_lane2 = use_inplace_lane2 && group_lane == 2;
        __gm__ ComputeT* qkv_mailbox =
            inplace_lane2 ? lane2_mailbox
                          : (slot == 0 ? workspace_qs_qkv_handle
                                       : workspace_ping_qs_qkv_handle);
        const int32_t row_gate_addr = slot == 0 ? GateSlot0Addr : GateSlot1Addr;

        if (group_lane == 0) {
          PublishGatedQKTile<HiddenSize,
                             ChunkSize
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
                             ,
                             true
#endif
                             >(G_handle,
                               current_qk_mailbox,
                               workspace_qk_gated_handle,
                               static_cast<int64_t>(cid),
                               total_tokens,
                               chunk_token_start,
                               head_idx,
                               valid_rows,
                               local_rows,
                               static_cast<int32_t>(vid),
                               row_gate_addr,
                               1u,
                               0u,
                               GroupFlowFlag,
                               GroupFlowFlag,
                               GatedSlot0ReadyFlag,
                               GatedBottomReadyFlag);
        }

        const int32_t next_lane = group_lane + 1;
        if (next_lane < GROUP) {
          const int32_t next_head_idx = head_group * GROUP + next_lane;
          const int32_t next_slot = next_lane & 1;
          const bool next_inplace_lane2 = use_inplace_lane2 && next_lane == 2;
          __gm__ ComputeT* next_gated_mailbox =
              next_inplace_lane2
                  ? lane2_mailbox
                  : (next_slot == 0 ? workspace_qk_gated_handle
                                    : workspace_ping_qk_gated_handle);
          const int32_t next_row_gate_addr =
              next_slot == 0 ? GateSlot0Addr : GateSlot1Addr;
          const bool next_releases_qk = next_lane + 1 == GROUP;
          PublishGatedQKTile<HiddenSize,
                             ChunkSize
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
                             ,
                             true
#endif
                             >(
              G_handle,
              current_qk_mailbox,
              next_gated_mailbox,
              static_cast<int64_t>(cid),
              total_tokens,
              chunk_token_start,
              next_head_idx,
              valid_rows,
              local_rows,
              static_cast<int32_t>(vid),
              next_row_gate_addr,
              0u,
              (next_inplace_lane2 || next_releases_qk) ? 1u : 0u,
              GroupFlowFlag,
              QkMailboxFreeFlag,
              next_inplace_lane2 ? GroupFlowFlag
                                 : (next_slot == 0 ? GatedSlot0ReadyFlag
                                                   : GatedSlot1ReadyFlag),
              GatedBottomReadyFlag);
        }

        const bool release_group_flow =
            !use_inplace_lane2 &&
            (group_lane + 2 < GROUP || group_lane + 1 == GROUP);
        ConsumeQKVTile<HiddenSize,
                       ChunkSize,
                       FuseGatedRmsNorm
#ifdef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
                       ,
                       true
#endif
                       >(
            S_handle,
            qkv_mailbox,
            O_handle,
            z_handle,
            static_cast<int64_t>(cid),
            global_chunk_idx,
            chunk_token_start,
            head_idx,
            H,
            valid_rows,
            local_rows,
            static_cast<int32_t>(vid),
            row_gate_addr,
            inplace_lane2 ? QkvSlot2ReadyFlag
                          : (slot == 0 ? QkvSlot0ReadyFlag : QkvSlot1ReadyFlag),
            release_group_flow ? GroupFlowFlag : -1,
            QkvBottomReadyFlag);
        produced_output = true;
      }
      ++group_owned_idx;
      group_work = GetHoOwnedItem(group_owned_idx,
                                  group_work_count,
                                  static_cast<int64_t>(cid),
                                  static_cast<int64_t>(block_num),
                                  rebalance_group_owners,
                                  group_balanced_span,
                                  h_o_heavy_core_count,
                                  h_o_light_core_count,
                                  group_prefix_rounds,
                                  group_tail_owner_shift);
    }

    if (produced_output) {
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
    }
  } else if (use_precomputed_qs && batch_size == 1) {
    // Keep one row-gate vector per mailbox slot. G-all uses the otherwise
    // idle QS input buffer during gating, so lookahead no longer needs to
    // save and restore the current gate around item N+1.
    constexpr int32_t GateSlot0Addr = GUbAddr;
    constexpr int32_t GateSlot1Addr = GvUbAddr;

    const int64_t bos = static_cast<int64_t>(cu_seqlens[0]);
    const int64_t eos = static_cast<int64_t>(cu_seqlens[1]);
    const int64_t slen = eos - bos;
    int64_t owned_idx = 0;
    int64_t current_item = GetHoOwnedItem(0,
                                          h_o_total_items,
                                          static_cast<int64_t>(cid),
                                          static_cast<int64_t>(block_num),
                                          rebalance_h_o_consumers,
                                          h_o_balanced_span,
                                          h_o_heavy_core_count,
                                          h_o_light_core_count,
                                          h_o_owned_prefix_count,
                                          h_o_tail_owner_shift);
    bool current_gating_published = false;

    while (current_item < h_o_total_items) {
      const int64_t ci = current_item / H;
      const int32_t head_idx = static_cast<int32_t>(current_item - ci * H);
      const int64_t chunk_start = ci * ChunkSize;
      const int64_t remaining = slen - chunk_start;
      const int32_t valid_rows =
          static_cast<int32_t>(remaining < ChunkSize ? remaining : ChunkSize);
      const int64_t chunk_token_start = bos + chunk_start;
      const int32_t row_offset = static_cast<int32_t>(vid) * HalfChunk;
      int32_t local_rows = valid_rows - row_offset;
      if (local_rows < 0) {
        local_rows = 0;
      }
      if (local_rows > HalfChunk) {
        local_rows = HalfChunk;
      }

      const int32_t slot = static_cast<int32_t>(owned_idx & 1);
      __gm__ ComputeT* qk_mailbox =
          slot == 0 ? workspace_qk_handle
                    : workspace_ping_qk_handle +
                          static_cast<int64_t>(block_num) * WsQKSize;
      __gm__ ComputeT* qkv_mailbox =
          slot == 0 ? workspace_qs_qkv_handle : workspace_ping_qs_qkv_handle;
      __gm__ ComputeT* gated_mailbox = slot == 0
                                           ? workspace_qk_gated_handle
                                           : workspace_ping_qk_gated_handle;
      const int32_t row_gate_addr = slot == 0 ? GateSlot0Addr : GateSlot1Addr;

      if (!current_gating_published) {
        PublishGatedQKTile<HiddenSize, ChunkSize>(G_handle,
                                                  qk_mailbox,
                                                  gated_mailbox,
                                                  static_cast<int64_t>(cid),
                                                  total_tokens,
                                                  chunk_token_start,
                                                  head_idx,
                                                  valid_rows,
                                                  local_rows,
                                                  static_cast<int32_t>(vid),
                                                  row_gate_addr,
                                                  1u,
                                                  1u,
                                                  slot,
                                                  6 + slot,
                                                  2 + slot);
      }

      const int64_t next_item = GetHoOwnedItem(owned_idx + 1,
                                               h_o_total_items,
                                               static_cast<int64_t>(cid),
                                               static_cast<int64_t>(block_num),
                                               rebalance_h_o_consumers,
                                               h_o_balanced_span,
                                               h_o_heavy_core_count,
                                               h_o_light_core_count,
                                               h_o_owned_prefix_count,
                                               h_o_tail_owner_shift);
      if (next_item < h_o_total_items) {
        const int64_t next_ci = next_item / H;
        const int32_t next_head_idx =
            static_cast<int32_t>(next_item - next_ci * H);
        const int64_t next_chunk_start = next_ci * ChunkSize;
        const int64_t next_remaining = slen - next_chunk_start;
        const int32_t next_valid_rows = static_cast<int32_t>(
            next_remaining < ChunkSize ? next_remaining : ChunkSize);
        const int64_t next_chunk_token_start = bos + next_chunk_start;
        int32_t next_local_rows = next_valid_rows - row_offset;
        if (next_local_rows < 0) {
          next_local_rows = 0;
        }
        if (next_local_rows > HalfChunk) {
          next_local_rows = HalfChunk;
        }
        const int32_t next_slot = static_cast<int32_t>((owned_idx + 1) & 1);
        __gm__ ComputeT* next_qk_mailbox =
            next_slot == 0 ? workspace_qk_handle
                           : workspace_ping_qk_handle +
                                 static_cast<int64_t>(block_num) * WsQKSize;
        __gm__ ComputeT* next_gated_mailbox =
            next_slot == 0 ? workspace_qk_gated_handle
                           : workspace_ping_qk_gated_handle;
        const int32_t next_row_gate_addr =
            next_slot == 0 ? GateSlot0Addr : GateSlot1Addr;

        PublishGatedQKTile<HiddenSize, ChunkSize>(G_handle,
                                                  next_qk_mailbox,
                                                  next_gated_mailbox,
                                                  static_cast<int64_t>(cid),
                                                  total_tokens,
                                                  next_chunk_token_start,
                                                  next_head_idx,
                                                  next_valid_rows,
                                                  next_local_rows,
                                                  static_cast<int32_t>(vid),
                                                  next_row_gate_addr,
                                                  1u,
                                                  1u,
                                                  next_slot,
                                                  6 + next_slot,
                                                  2 + next_slot);
      }

      ConsumeQKVTile<HiddenSize, ChunkSize, FuseGatedRmsNorm>(
          S_handle,
          qkv_mailbox,
          O_handle,
          z_handle,
          static_cast<int64_t>(cid),
          ci,
          chunk_token_start,
          head_idx,
          H,
          valid_rows,
          local_rows,
          static_cast<int32_t>(vid),
          row_gate_addr,
          4 + slot,
          -1);

      current_item = next_item;
      current_gating_published = next_item < h_o_total_items;
      ++owned_idx;
    }

    if (owned_idx > 0) {
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
    }
  } else {
    // ── Variable-length sequence path (cu_seqlens != nullptr) ──────────
    int64_t gi = 0;
    int64_t chunk_global_idx = 0;
    int64_t vec_owned_idx_v = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < H; ++h) {
          int64_t consumer_owner = gi % static_cast<int64_t>(block_num);
          if (rebalance_h_o_consumers && gi >= h_o_balanced_span) {
            consumer_owner = h_o_heavy_core_count +
                             (gi - h_o_balanced_span) % h_o_light_core_count;
          }
          if (consumer_owner == static_cast<int64_t>(cid)) {
            const int32_t mailbox_slot =
                use_precomputed_qs ? static_cast<int32_t>(vec_owned_idx_v & 1)
                                   : 0;
            __gm__ ComputeT* qk_mailbox =
                mailbox_slot == 0
                    ? workspace_qk_handle
                    : workspace_ping_qk_handle +
                          static_cast<int64_t>(block_num) * WsQKSize;
            __gm__ ComputeT* qs_qkv_mailbox =
                mailbox_slot == 0 ? workspace_qs_qkv_handle
                                  : workspace_ping_qs_qkv_handle;
            __gm__ ComputeT* qk_gated_mailbox =
                mailbox_slot == 0 ? workspace_qk_gated_handle
                                  : workspace_ping_qk_gated_handle;
            const int32_t qk_ready_flag = use_precomputed_qs ? mailbox_slot : 0;
            const int32_t qk_gated_ready_flag =
                use_precomputed_qs ? 2 + mailbox_slot : 1;
            const int32_t qkv_ready_flag =
                use_precomputed_qs ? 4 + mailbox_slot : 2;
            const int32_t mailbox_free_flag =
                use_precomputed_qs ? 6 + mailbox_slot : 3;
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t head_idx = h;
            int32_t row_offset = static_cast<int32_t>(vid) * HalfChunk;
            int32_t local_rows = valid_rows - row_offset;
            if (local_rows < 0) local_rows = 0;
            if (local_rows > HalfChunk) local_rows = HalfChunk;

            if (local_rows > 0) {
              // Load G
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = 1;
                _gs.shape[4] = valid_rows;
                GlobalTensor<float, decltype(_gs), pto::Stride<1, 1, 1, 1, 1>>
                    _gm(G_handle +
                            static_cast<int64_t>(head_idx) * total_tokens +
                            chunk_token_start,
                        _gs);
                UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(
                    1, valid_rows);
                TASSIGN(_ld, GUbAddr);
                TLOAD(_ld, _gm);
                if (valid_rows != ChunkSize) {
                  UbND<float, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
                  TASSIGN(_pd, GUbAddr);
                  TFILLPAD_INPLACE(_pd, _ld);
                }
              }
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

              // Compute gating coefficients (same math as fixed-length path —
              // see detailed pseudocode above)
              UbND<float, 1, HalfChunk> g_ub_temp_v;
              TASSIGN(g_ub_temp_v,
                      GUbAddr + static_cast<int32_t>(vid) * HalfChunk *
                                    static_cast<int32_t>(sizeof(float)));
              TMOV(g_v_ub, g_ub_temp_v);

              UbND<float, HalfChunk, ChunkSize> g_r_2d_v;
              TASSIGN(g_r_2d_v, QSUbAddr);
              UbDN<float, HalfChunk, 1> g_v_col_v;
              TASSIGN(g_v_col_v, GvUbAddr);
              TROWEXPAND(g_r_2d_v, g_v_col_v);
              TCOLEXPAND(coeff_ub, g_ub);
              TSUB(coeff_ub, g_r_2d_v, coeff_ub);  // d = g_row - g_col
              pipe_barrier(PIPE_V);
              TMINS(coeff_ub, coeff_ub, 0.0f);
              pipe_barrier(PIPE_V);
              TEXP(coeff_ub, coeff_ub);
              pipe_barrier(PIPE_V);
              TMUL(coeff_ub, coeff_ub, msk_ub);
              pipe_barrier(PIPE_V);
              TEXP(g_v_ub, g_v_ub);
            }

            wait_flag_dev(qk_ready_flag);
            if (local_rows == 0) {
              ffts_cross_core_sync(PIPE_MTE3,
                                   1 | (2 << 4) | (qk_gated_ready_flag << 8));
              wait_flag_dev(qkv_ready_flag);
              ffts_cross_core_sync(PIPE_MTE3,
                                   1 | (2 << 4) | (mailbox_free_flag << 8));
            } else {
              // Load QK from workspace
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = local_rows;
                _gs.shape[4] = ChunkSize;
                GlobalTensor<ComputeT,
                             decltype(_gs),
                             pto::Stride<1, 1, 1, ChunkSize, 1>>
                    _gm(qk_mailbox + static_cast<int64_t>(cid) * WsQKSize +
                            static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                        _gs);
                UbND<ComputeT,
                     HalfChunk,
                     ChunkSize,
                     DYNAMIC,
                     DYNAMIC,
                     PadValue::Zero>
                    _ld(local_rows, ChunkSize);
                TASSIGN(_ld, QKHalfUbAddr);
                TLOAD(_ld, _gm);
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(qk_ub_half, _ld);
                }
              }

              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              TCVT(qk_ub, qk_ub_half, pto::RoundMode::CAST_NONE);

              set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

              // Load QS from workspace
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = local_rows;
                _gs.shape[4] = HiddenSize;
                __gm__ ComputeT* qs_source =
                    qs_qkv_mailbox + static_cast<int64_t>(cid) * WsQSSize;
                if (use_precomputed_qs) {
                  qs_source = S_handle + (chunk_global_idx * H + head_idx) *
                                             static_cast<int64_t>(HiddenSize) *
                                             HiddenSize;
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
                  constexpr int32_t DcciCacheLineElems =
                      64 / static_cast<int32_t>(sizeof(ComputeT));
                  for (int32_t row = 0; row < local_rows; ++row) {
                    for (int32_t r = 0; r < HiddenSize;
                         r += DcciCacheLineElems) {
                      dcci(static_cast<__gm__ void*>(
                               qs_source +
                               (static_cast<int64_t>(vid) * HalfChunk + row) *
                                   HiddenSize +
                               r),
                           SINGLE_CACHE_LINE);
                    }
                  }
                  dsb(DSB_DDR);
#endif
                }
                GlobalTensor<ComputeT,
                             decltype(_gs),
                             pto::Stride<1, 1, 1, HiddenSize, 1>>
                    _gm(qs_source +
                            static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                        _gs);
                UbND<ComputeT,
                     HalfChunk,
                     HiddenSize,
                     DYNAMIC,
                     DYNAMIC,
                     PadValue::Zero>
                    _ld(local_rows, HiddenSize);
                TASSIGN(_ld, QSHalfUbAddr);
                TLOAD(_ld, _gm);
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(qs_ub_half, _ld);
                }
              }

              TMUL(qk_ub, qk_ub, coeff_ub);
              TCVT(qk_ub_half,
                   qk_ub,
                   pto::RoundMode::CAST_NONE);  // float→ComputeT for GM store

              // Store QK_gated → workspace
              set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = local_rows;
                _gs.shape[4] = ChunkSize;
                GlobalTensor<ComputeT,
                             decltype(_gs),
                             pto::Stride<1, 1, 1, ChunkSize, 1>>
                    _gm(qk_gated_mailbox +
                            static_cast<int64_t>(cid) * WsGatedSize +
                            static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                        _gs);
                UbND<ComputeT, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(
                    local_rows, ChunkSize);
                TASSIGN(_st, QKHalfUbAddr);
                TSTORE(_gm, _st);
              }
              // Vec→Cube: QK_gated ready (flag 1)
              ffts_cross_core_sync(PIPE_MTE3,
                                   1 | (2 << 4) | (qk_gated_ready_flag << 8));

              // Scale QS by exp(g): QS_scaled = QS * exp(g_row)[:, None]
              // (same inter-chunk state scaling as fixed-length path)
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              TCVT(qs_ub,
                   qs_ub_half,
                   pto::RoundMode::CAST_NONE);  // ComputeT→float for Vec math

              UbND<float, HalfChunk, HiddenSize> g_exp_2d_v;
              TASSIGN(g_exp_2d_v, CoeffUbAddr);
              UbDN<float, HalfChunk, 1> g_v_col2_v;
              TASSIGN(g_v_col2_v, GvUbAddr);
              TROWEXPAND(g_exp_2d_v, g_v_col2_v);
              pipe_barrier(PIPE_V);
              TMUL(qs_ub, qs_ub, g_exp_2d_v);

              wait_flag_dev(qkv_ready_flag);

              // Load QKV from workspace
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = local_rows;
                _gs.shape[4] = HiddenSize;
                GlobalTensor<ComputeT,
                             decltype(_gs),
                             pto::Stride<1, 1, 1, HiddenSize, 1>>
                    _gm(qs_qkv_mailbox + static_cast<int64_t>(cid) * WsQSSize +
                            static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                        _gs);
                UbND<ComputeT,
                     HalfChunk,
                     HiddenSize,
                     DYNAMIC,
                     DYNAMIC,
                     PadValue::Zero>
                    _ld(local_rows, HiddenSize);
                TASSIGN(_ld, OHalfUbAddr);
                TLOAD(_ld, _gm);
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(o_ub_half, _ld);
                }
              }

              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

              // O = QS_gated + QKV  (final output: intra-chunk attention +
              // inter-chunk state)
              TCVT(o_ub, o_ub_half, pto::RoundMode::CAST_NONE);
              TADD(o_ub, qs_ub, o_ub);

              int64_t o_offset = (chunk_token_start * static_cast<int64_t>(H) +
                                  static_cast<int64_t>(head_idx)) *
                                     static_cast<int64_t>(HiddenSize) +
                                 static_cast<int64_t>(vid) * HalfChunk *
                                     static_cast<int64_t>(BSND_V_STRIDE);
              StoreChunkOutput<HiddenSize, ChunkSize, FuseGatedRmsNorm>(
                  O_handle, z_handle, o_offset, BSND_V_STRIDE, local_rows);

              // Vec→Cube: done with this chunk (flag 3)
              ffts_cross_core_sync(PIPE_MTE3,
                                   1 | (2 << 4) | (mailbox_free_flag << 8));
            }
            ++vec_owned_idx_v;
          }
          gi++;
        }
        ++chunk_global_idx;
      }
    }
    if (use_precomputed_qs && vec_owned_idx_v > 0) {
      // Keep the outer XllmKernelMegaChunkGdn drain protocol unchanged.
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
    }
  }
#endif
}

#undef GDN_CHUNK_O_KERNEL

// ── Device kernel entry point ─────────────────────────────────────────
// extern "C" __global__ AICORE: NPU kernel function.
// Runs on each AI core independently. Args are uint8_t* (type-erased)
// because the NPU launch ABI passes all pointers as raw bytes; we
// reinterpret_cast them to the correct types before calling the template.
