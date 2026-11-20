// ============================================================================
// wy_fast_kernel.cpp — WY representation for GatedDeltaNet chunk recurrence
//
// Computes the WY update matrices U and W for each chunk of C tokens:
//   U = A2 @ V     where A2 = A * beta_2d        (beta-scaled attention)
//   W = A1 @ K     where A1 = A * (exp(g)*beta)_2d (gate+beta-scaled attention)
//
// beta is the decay factor, g is the gate value, A is the triangular attention
// matrix (from the kkt kernel).  The column-broadcast notation x_2d means
// expanding a 1xC vector into a C/2 x C matrix by replicating across rows.
//
// Architecture: Vec+Cube cooperative kernel using cross-core synchronization.
//
//  Vec core (two sub-blocks for upper/lower C/2 rows):
//    For each chunk:
//      1. Load beta [H,T] and A [B,S,H,C], compute A2 = A * beta_2d -> ws
//      2. Load G [H,T], compute A1 = A * (exp(g)*beta)_2d -> ws
//      3. Signal Cube via cross-core flags when workspaces are ready
//
//  Cube core (waits for Vec signals):
//    For each chunk:
//      1. Load K, V from BSND layout into L1
//      2. Load A2 from workspace -> GEMM: U = A2 @ V
//      3. Load A1 from workspace -> GEMM: W = A1 @ K
//      4. Store U, W back to BSND layout
//
// NPU memory hierarchy used:
//   GM -> UB (Vec), GM -> L1 -> L0A/L0B -> L0C -> GM (Cube)
//
// ── PTO / NPU Primer ──────────────────────────────────────────────────
// This kernel uses BOTH the Cube engine (matrix multiply) and Vec engine
// (SIMD element-wise ops), running on SEPARATE physical cores that
// communicate via Global Memory (GM) + cross-core flags (FFTS).
//
// Execution flow:
//   Vec core:  load A,beta,G → compute A2,A1 → store to GM workspace
//   Cube core: wait for workspace → load A2/A1 + K/V → GEMM → store U,W
//
// Key PTO APIs (with numpy/torch equivalents):
//   TLOAD(ub_tile, gm)      — ub_tile = gm[...]          (DMA: GM→UB, async
//   MTE2) TSTORE(gm, ub_tile)     — gm[...] = ub_tile          (DMA: UB→GM,
//   async MTE3) TCVT(dst, src, mode)    — converts between float and ComputeT
//   TMOV(dst, src)          — dst = src.clone()
//   TMUL(d, a, b)           — d = a * b                   (element-wise)
//   TEXP(d, s)              — d = torch.exp(s)
//   TCOLEXPAND(2d, row)     — 2d[i,j] = row[j]  (broadcast row across all rows)
//   TEXTRACT(l0, l1, r, c)  — L1 sub-block → L0A/L0B     (MTE1 for Cube GEMM)
//   TMATMUL(C, A, B)        — C = A @ B in Cube engine (ComputeT→FP32
//   accumulate) set_flag / wait_flag    — sync between pipes on SAME core
//   ffts_cross_core_sync    — signal ACROSS Cube↔Vec cores
//   wait_flag_dev(flag)     — wait for cross-core signal
// ============================================================================

#include <pto/pto-inst.hpp>
#include <type_traits>

#include "acl/acl.h"
#include "mega_chunk_gdn/op_kernel/gdn_sync.h"
using namespace pto;

#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

#ifdef __CCE_AICORE__

namespace {

template <typename T,
          int Rows,
          int Cols,
          int RowValid = Rows,
          int ColValid = Cols>
using TileMatL1 = pto::Tile<pto::TileType::Mat,
                            T,
                            Rows,
                            Cols,
                            pto::BLayout::ColMajor,
                            RowValid,
                            ColValid,
                            pto::SLayout::RowMajor,
                            512,
                            pto::PadValue::Zero>;

template <typename T,
          int Rows,
          int Cols,
          int RowValid = Rows,
          int ColValid = Cols>
using TileMatL1ZN = pto::Tile<pto::TileType::Mat,
                              T,
                              Rows,
                              Cols,
                              pto::BLayout::RowMajor,
                              RowValid,
                              ColValid,
                              pto::SLayout::ColMajor,
                              512,
                              pto::PadValue::Zero>;

template <typename T,
          int Rows,
          int Cols,
          int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0A = pto::TileLeft<T, Rows, Cols, RowValid, ColValid>;

template <typename T,
          int Rows,
          int Cols,
          int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0B = pto::TileRight<T, Rows, Cols, RowValid, ColValid>;

template <typename T,
          int Rows,
          int Cols,
          int RowValid = Rows,
          int ColValid = Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataND = pto::Tile<pto::TileType::Vec,
                               T,
                               Rows,
                               Cols,
                               pto::BLayout::RowMajor,
                               RowValid,
                               ColValid,
                               pto::SLayout::NoneBox,
                               512,
                               PadVal>;

template <typename T,
          int Rows,
          int Cols,
          int RowValid = Rows,
          int ColValid = Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataDN = pto::Tile<pto::TileType::Vec,
                               T,
                               Rows,
                               Cols,
                               pto::BLayout::ColMajor,
                               RowValid,
                               ColValid,
                               pto::SLayout::NoneBox,
                               512,
                               PadVal>;

using GmShape2D = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
using GmStride2D = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;

template <typename T>
using GmTensor2D = pto::GlobalTensor<T, GmShape2D, GmStride2D>;

template <typename T, int32_t Rows, int32_t Cols>
using DynMatL1 = pto::Tile<pto::TileType::Mat,
                           T,
                           Rows,
                           Cols,
                           pto::BLayout::ColMajor,
                           pto::DYNAMIC,
                           pto::DYNAMIC,
                           pto::SLayout::RowMajor,
                           512,
                           pto::PadValue::Zero>;

template <typename T,
          int32_t Rows,
          int32_t Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using DynVecTile = pto::Tile<pto::TileType::Vec,
                             T,
                             Rows,
                             Cols,
                             pto::BLayout::RowMajor,
                             pto::DYNAMIC,
                             pto::DYNAMIC,
                             pto::SLayout::NoneBox,
                             512,
                             PadVal>;

template <typename T, int32_t Rows, int32_t Cols>
using DynAccTile = pto::TileAcc<T, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;

// PTO cheat sheet for readers coming from PyTorch / NumPy:
//   - `GlobalTensor<T>` is a GM tensor view with explicit shape/stride
//   metadata.
//   - `Tile<..., Mat, ...>` is an on-chip matrix tile used by Cube kernels.
//   - `Tile<..., Vec, ...>` is an on-chip UB tile used by SIMD vector kernels.
//   - `TileAcc<T, ...>` is the matmul accumulator tile.
//   - `TLOAD` / `TSTORE` are DMA copies between GM and local memory.
//   - `TCOLEXPAND` is broadcast like `x[None, :].expand(rows, -1)`.
//   - `TMUL`, `TEXP`, `TCVT` are vector ops on UB tiles.

template <typename T1,
          typename T2,
          uint32_t M,
          uint32_t N,
          uint32_t K,
          uint32_t validM = M,
          uint32_t validN = N,
          uint32_t validK = K,
          uint32_t K_tail,
          bool transpose_A = false,
          bool transpose_B = false>
AICORE PTO_INLINE void gemm_v0(
    std::conditional_t<transpose_A,
                       TileMatL1<T1, K, M, validK, validM>,
                       TileMatL1<T1, M, K, validM, validK>>& A,
    std::conditional_t<transpose_B,
                       TileMatL1<T1, N, K, validN, validK>,
                       TileMatL1<T1, K, N, validK, validN>>& B,
    pto::TileAcc<T2, M, N, validM, validN>& C,
    bool clear) {
  // Local K-sliced matmul helper:
  //   C = A @ B
  // PTO exposes the L1 -> L0 -> Cube movement explicitly, so keeping this tiny
  // helper local lets readers see the schedule without hiding it in a repo-wide
  // wrapper layer.
  //
  // PyTorch mental model:
  //   C = 0
  //   for k0 in range(0, K, kL0Size):
  //       C += A[:, k0:k1] @ B[k0:k1, :]
  constexpr uint32_t kL0Size = 128;
  const uint32_t kL0split = (K + kL0Size - 1) / kL0Size;

  auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);
  set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);

  for (uint32_t kL0Idx = 0; kL0Idx < kL0split; ++kL0Idx) {
    const bool initflag = clear && (kL0Idx == 0);
    const bool is_tail_block = (kL0Idx == kL0split - 1);

    if (is_tail_block) {
      TileMatL0A<T1, M, K_tail, M, K_tail> l0a;
      TileMatL0B<T1, K_tail, N, K_tail, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      if constexpr (!transpose_A) {
        pto::TEXTRACT(l0a, A, 0, kL0Idx * K_tail);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        pto::TEXTRACT(l0a, A_t, 0, kL0Idx * K_tail);
      }

      if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, kL0Idx * K_tail, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, kL0Idx * K_tail, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL(C, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
      }
    } else {
      TileMatL0A<T1, M, kL0Size, M, kL0Size> l0a;
      TileMatL0B<T1, kL0Size, N, kL0Size, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      set_flag(PIPE_FIX, PIPE_M, war_event_id);
      wait_flag(PIPE_FIX, PIPE_M, war_event_id);

      if constexpr (!transpose_A) {
        pto::TEXTRACT(l0a, A, 0, kL0Idx * kL0Size);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        pto::TEXTRACT(l0a, A_t, 0, kL0Idx * kL0Size);
      }

      if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, kL0Idx * kL0Size, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, kL0Idx * kL0Size, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL(C, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
      }

      set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
    }
  }

  set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
  wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);

  set_flag(PIPE_M, PIPE_FIX, war_event_id);
  wait_flag(PIPE_M, PIPE_FIX, war_event_id);
}

}  // namespace

#endif

#if defined(__DAV_C220_CUBE__)
#define GDN_WY_FAST_KERNEL wy_fast_kernel_aic
#elif defined(__DAV_C220_VEC__)
#define GDN_WY_FAST_KERNEL wy_fast_kernel_aiv
#else
#define GDN_WY_FAST_KERNEL wy_fast_kernel
#endif

template <int32_t HiddenSize, int32_t ChunkSize>
AICORE void GDN_WY_FAST_KERNEL(__gm__ ComputeT* K_handle,
                               __gm__ ComputeT* V_handle,
                               __gm__ ComputeT* Beta_handle,
                               __gm__ float* G_handle,
                               __gm__ ComputeT* A_handle,
                               __gm__ ComputeT* workspace_a1_handle,
                               __gm__ ComputeT* workspace_a2_handle,
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
                               __gm__ ComputeT* workspace_reuse_handle,
#endif
                               __gm__ ComputeT* W_handle,
                               __gm__ ComputeT* U_handle,
                               __gm__ int32_t* cu_seqlens,
                               int64_t batch_size,
                               int64_t seq_len,
                               int64_t total_tokens,
                               uint32_t num_heads,
                               uint32_t num_key_heads,
                               uint64_t ffts_addr,
                               bool a_head_major = false) {
  // WY recompute materializes two diagonal reweightings of the same A tile:
  //   A2[:, j] = A[:, j] * beta_j
  //   A1[:, j] = A[:, j] * exp(g_j) * beta_j
  // and then forms the two branch outputs
  //   U = A2 @ V,   W = A1 @ K.
  //
  // Shapes for one (sequence, head, chunk):
  //   A_chunk : [valid, valid]
  //   beta    : [valid]
  //   g       : [valid]
  //   K, V    : [valid, D]
  //
  // PyTorch / NumPy sketch:
  //   A2 = A_chunk * beta[None, :]
  //   A1 = A_chunk * (exp(g) * beta)[None, :]
  //   U  = A2 @ V_chunk
  //   W  = A1 @ K_chunk
  //
  // PTO split:
  //   Vec builds the two reweighted A tiles in workspace.
  //   Cube later consumes those workspaces in two GEMMs.
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr uint32_t KTail = (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);

  const int32_t H = static_cast<int32_t>(num_heads);
  const int32_t Hg = static_cast<int32_t>(num_key_heads);
  if (H <= 0 || Hg <= 0 || (H % Hg) != 0) return;
  const int32_t GROUP = H / Hg;
  const int32_t BSND_V_STRIDE = H * HiddenSize;
  const int32_t BSND_QK_STRIDE = Hg * HiddenSize;

  constexpr int32_t BetaHalfUbAddr = 0;
  constexpr int32_t A1HalfUbAddr = 256;
  constexpr int32_t BetaUbAddr = 16640;
  constexpr int32_t BetaRUbAddr = 17152;
  constexpr int32_t Beta2dUbAddr = 17664;
  constexpr int32_t TmpUbAddr = 50432;
  constexpr int32_t A1UbAddr = 75008;
  constexpr int32_t A2UbAddr = 107776;
  constexpr int32_t A2HalfUbAddr = 140544;
  constexpr int32_t GUbAddr = 156928;
  constexpr int32_t GRUbAddr = 157440;
  constexpr int32_t G2dUbAddr = 157952;

  constexpr int32_t GBlockUbAddr = TmpUbAddr;
  constexpr int32_t BetaBlockUbAddr = TmpUbAddr;

  constexpr int32_t WsA1Size = ChunkSize * ChunkSize;
  constexpr int32_t WsA2Size = ChunkSize * ChunkSize;

  gdn_sync::InitAddress(ffts_addr);
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;

  TileUbDataND<ComputeT, 1, ChunkSize, 1, ChunkSize, pto::PadValue::Zero>
      beta_ub_half;
  TASSIGN(beta_ub_half, BetaHalfUbAddr);
  TileUbDataND<ComputeT,
               HalfChunk,
               ChunkSize,
               HalfChunk,
               ChunkSize,
               pto::PadValue::Zero>
      a1_ub_half;
  TASSIGN(a1_ub_half, A1HalfUbAddr);
  TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> beta_ub;
  TASSIGN(beta_ub, BetaUbAddr);
  TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> beta_r_ub;
  TASSIGN(beta_r_ub, BetaRUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize, HalfChunk, ChunkSize> beta_2d_ub;
  TASSIGN(beta_2d_ub, Beta2dUbAddr);
  TileUbDataND<uint8_t, 1, 24576, 1, 24576> tmp_ub;
  TASSIGN(tmp_ub, TmpUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize, HalfChunk, ChunkSize> a1_ub;
  TASSIGN(a1_ub, A1UbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize, HalfChunk, ChunkSize> a2_ub;
  TASSIGN(a2_ub, A2UbAddr);
  TileUbDataND<ComputeT, HalfChunk, ChunkSize, HalfChunk, ChunkSize> a2_ub_half;
  TASSIGN(a2_ub_half, A2HalfUbAddr);
  TileUbDataND<float, 1, ChunkSize, 1, ChunkSize, pto::PadValue::Zero> g_ub;
  TASSIGN(g_ub, GUbAddr);
  TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> g_r_ub;
  TASSIGN(g_r_ub, GRUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize, HalfChunk, ChunkSize> g_2d_ub;
  TASSIGN(g_2d_ub, G2dUbAddr);

  TileMatL1<ComputeT, ChunkSize, HiddenSize, ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 0);
  TileMatL1<ComputeT, ChunkSize, HiddenSize, ChunkSize, HiddenSize> v_l1;
  TASSIGN(v_l1, 32768);
  TileMatL1<ComputeT, ChunkSize, ChunkSize, ChunkSize, ChunkSize> a2_l1;
  TASSIGN(a2_l1, 65536);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> u_l0;
  // Keep u_l0 in the primary L0C slot: on A5 the secondary slot drops tail
  // rows for this 128x128 fp32 accumulator, and the A5 packed path reuses
  // u_l0 for both GEMMs (U is drained before W starts).
  TASSIGN(u_l0, 0);
  TileMatL1<ComputeT, ChunkSize, ChunkSize, ChunkSize, ChunkSize> a1_l1;
  TASSIGN(a1_l1, 98304);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> w_l0;
  TASSIGN(w_l0, 65536);

  int64_t total_work = 0;
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    total_work = num_seqs * chunks_per_seq * H;
  }
#ifdef MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
  // A group is one (sequence, chunk, key-head) item. Assign all three value
  // heads of a Qwen group to the same MIX block only when there are enough
  // independent groups to occupy every block; short-prefill parallelism then
  // remains identical to the per-head schedule.
  int64_t a5_group_work_count = 0;
  if (GROUP == 3) {
    for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      const int64_t bos = cu_seqlens == nullptr
                              ? seq_idx * seq_len
                              : static_cast<int64_t>(cu_seqlens[seq_idx]);
      const int64_t eos = cu_seqlens == nullptr
                              ? bos + seq_len
                              : static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
      const int64_t slen = eos - bos;
      a5_group_work_count +=
          ((slen + ChunkSize - 1) / ChunkSize) * static_cast<int64_t>(Hg);
    }
  }
  const bool a5_reuse_group_k =
      GROUP == 3 && a5_group_work_count >= static_cast<int64_t>(block_num);
#ifndef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
#error "A5 WY group-K reuse requires the single-A-read workspace layout"
#endif
#endif

#if defined(__DAV_C310_VEC__)
  // A5: gather strided BSND operands and apply column weights in 16x128 UB
  // tiles. The optional dual-AIV schedule gives each sibling alternating row
  // bands and rendezvouses both before Cube consumes a packed workspace.
  static_assert(HiddenSize == ChunkSize,
                "A5 WY packed path expects D == chunk size");
  constexpr int32_t A5Rows = 16;
  using A5TileH = TileUbDataND<ComputeT,
                               A5Rows,
                               ChunkSize,
                               A5Rows,
                               ChunkSize,
                               pto::PadValue::Zero>;
  using A5TileF = TileUbDataND<float, A5Rows, ChunkSize, A5Rows, ChunkSize>;
  using A5DynamicH =
      DynVecTile<ComputeT, A5Rows, ChunkSize, pto::PadValue::Zero>;
  using A5PackedShape = Shape<1, 1, 1, A5Rows, ChunkSize>;
  using A5PackedStride = pto::Stride<1, 1, 1, ChunkSize, 1>;
  using A5PackedGlobal = GlobalTensor<ComputeT, A5PackedShape, A5PackedStride>;

  A5TileH a5_half;
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
  // One AIV owns four alternating 16-row bands. Keep its raw A rows in the
  // existing 64x128 half UB region until the second WY weighting pass, and
  // use the other half UB region for transient A2/A1/V/K data.
  A5TileH a5_io_half;
#endif
  A5TileF a5_float;
  A5TileF a5_weight_2d;
#ifdef MEGA_CHUNK_GDN_A5_WY_FUSED_A1_A2_PREP
  // The generic 64-row beta_2d_ub allocation leaves three 16-row bands
  // unused by the A5 path.  Keep the A1 weight in the second band and use
  // tmp_ub for a second FP32 tile so raw A remains live for both products.
  constexpr int32_t A5A1WeightAddr =
      Beta2dUbAddr + A5Rows * ChunkSize * sizeof(float);
  A5TileF a5_a1_weight_2d;
  A5TileF a5_weighted_float;
#endif
  TASSIGN(a5_half, A1HalfUbAddr);
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
  TASSIGN(a5_io_half, A2HalfUbAddr);
#endif
  TASSIGN(a5_float, A1UbAddr);
  TASSIGN(a5_weight_2d, Beta2dUbAddr);
#ifdef MEGA_CHUNK_GDN_A5_WY_FUSED_A1_A2_PREP
  TASSIGN(a5_a1_weight_2d, A5A1WeightAddr);
  TASSIGN(a5_weighted_float, TmpUbAddr);
#endif

#ifdef MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
  constexpr bool A5VectorOwner = true;
  const int32_t a5_first_row = static_cast<int32_t>(vid) * A5Rows;
  constexpr int32_t A5RowStride = 2 * A5Rows;
#else
  constexpr bool A5VectorOwner = false;
  constexpr int32_t a5_first_row = 0;
  constexpr int32_t A5RowStride = A5Rows;
#endif
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
  static_assert(A5RowStride == 2 * A5Rows,
                "A5 single-read WY requires the dual-AIV row split");
#ifndef MEGA_CHUNK_GDN_A5_PACKED_WUV
#error "A5 single-read WY requires packed W/U so cached A stays resident"
#endif
#endif
#ifdef MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
#ifndef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
#error "A5 WY sliding pipeline requires four independent mailbox slots"
#endif
#ifndef MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
#error "A5 WY sliding pipeline requires the dual-AIV row split"
#endif
#ifndef MEGA_CHUNK_GDN_A5_PACKED_WUV
#error "A5 WY sliding pipeline requires direct packed W/U stores"
#endif
#endif
#ifdef MEGA_CHUNK_GDN_A5_WY_STAGGERED_MAILBOX
#ifndef MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
#error "A5 staggered WY mailboxes require the sliding pipeline"
#endif
#ifndef MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
#error "A5 staggered WY mailboxes require paired AIV producers"
#endif
#endif
#ifdef MEGA_CHUNK_GDN_A5_WY_FUSED_A1_A2_PREP
#ifndef MEGA_CHUNK_GDN_A5_WY_STAGGERED_MAILBOX
#error "A5 fused A1/A2 preparation requires staggered WY mailboxes"
#endif
#ifndef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
#error "A5 fused A1/A2 preparation requires the single-A-read layout"
#endif
#endif
#ifdef MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_K_L1
#ifndef MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
#error "A5 direct-input K-to-L1 requires the group-K schedule"
#endif
#ifndef MEGA_CHUNK_GDN_A5_WY_FUSED_A1_A2_PREP
#error "A5 direct-input K-to-L1 requires fused A1/A2 preparation"
#endif
#ifndef MEGA_CHUNK_GDN_A5_PUBLISH_INPUTS_BEFORE_WY
#error "A5 direct-input K-to-L1 requires producer cache publication"
#endif
#endif
#ifdef MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_V_L1
#ifndef MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_K_L1
#error "A5 direct-input V-to-L1 requires the published direct-K schedule"
#endif
#endif
  int64_t a5_work = 0;
#ifdef MEGA_CHUNK_GDN_A5_WY_STAGGERED_MAILBOX
  // Once Cube has consumed A2/V into L1, both AIV siblings may refill those
  // two mailboxes for the next owned head while the current U GEMM runs.  A1/K
  // remain protected by flag 3 until Cube has loaded them for the current W.
  bool a5_has_pending_a1k = false;
#endif
  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int64_t bos = cu_seqlens == nullptr
                            ? seq_idx * seq_len
                            : static_cast<int64_t>(cu_seqlens[seq_idx]);
    const int64_t eos = cu_seqlens == nullptr
                            ? bos + seq_len
                            : static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
    const int64_t slen = eos - bos;
    const int64_t num_chunks = (slen + ChunkSize - 1) / ChunkSize;
    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      const int64_t chunk_start = chunk_idx * ChunkSize;
      const int32_t valid_rows = static_cast<int32_t>(
          min(static_cast<int64_t>(ChunkSize), slen - chunk_start));
      const int64_t token_start = bos + chunk_start;
      for (int32_t head_idx = 0; head_idx < H; ++head_idx, ++a5_work) {
#ifdef MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
        const int64_t a5_owner_work =
            a5_reuse_group_k ? a5_work / GROUP : a5_work;
#else
        const int64_t a5_owner_work = a5_work;
#endif
        if (a5_owner_work % static_cast<int64_t>(block_num) !=
            static_cast<int64_t>(cid)) {
          continue;
        }
        __gm__ ComputeT* packed_rhs =
            workspace_a1_handle + static_cast<int64_t>(cid) * WsA1Size;
        __gm__ ComputeT* packed_weighted =
            workspace_a2_handle + static_cast<int64_t>(cid) * WsA2Size;
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
        // Solve no longer needs its two per-block KKT tiles. Reuse them for
        // A1 and K so A2/V can remain live while Cube executes the U GEMM.
        __gm__ ComputeT* packed_a1 =
            workspace_reuse_handle + static_cast<int64_t>(cid) * 2 * WsA1Size;
        __gm__ ComputeT* packed_k = packed_a1 + WsA1Size;
#endif

        if (vid == 0 || A5VectorOwner) {
          // Contiguous beta vector for A2 = A * beta[None, :].
          {
            GmShape2D beta_shape(1, valid_rows);
            GmStride2D beta_stride(1);
            GmTensor2D<ComputeT> beta_global(
                Beta_handle + static_cast<int64_t>(head_idx) * total_tokens +
                    token_start,
                beta_shape,
                beta_stride);
            DynVecTile<ComputeT, 1, ChunkSize, pto::PadValue::Zero> beta_load(
                1, valid_rows);
            TASSIGN(beta_load, BetaHalfUbAddr);
            TLOAD(beta_load, beta_global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            if (valid_rows != ChunkSize) {
              TFILLPAD_INPLACE(beta_ub_half, beta_load);
            }
          }
          TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
          pipe_barrier(PIPE_V);
          TCOLEXPAND(a5_weight_2d, beta_ub);
          pipe_barrier(PIPE_V);

#ifdef MEGA_CHUNK_GDN_A5_WY_FUSED_A1_A2_PREP
          // Build the second weighting vector before reading A.  Each A row
          // can then form A2 and A1 from the same FP32 conversion while
          // preserving the validated operation order A * (exp(g) * beta).
          {
            GmShape2D g_shape(1, valid_rows);
            GmStride2D g_stride(1);
            GmTensor2D<float> g_global(
                G_handle + static_cast<int64_t>(head_idx) * total_tokens +
                    token_start,
                g_shape,
                g_stride);
            DynVecTile<float, 1, ChunkSize, pto::PadValue::Zero> g_load(
                1, valid_rows);
            TASSIGN(g_load, GUbAddr);
            TLOAD(g_load, g_global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            if (valid_rows != ChunkSize) {
              TFILLPAD_INPLACE(g_ub, g_load);
            }
          }
          TEXP(g_ub, g_ub);
          pipe_barrier(PIPE_V);
          TMUL(g_ub, g_ub, beta_ub);
          pipe_barrier(PIPE_V);
          TCOLEXPAND(a5_a1_weight_2d, g_ub);
          pipe_barrier(PIPE_V);
#endif

          for (int32_t tile_row = a5_first_row; tile_row < ChunkSize;
               tile_row += A5RowStride) {
            const int32_t live_rows =
                valid_rows > tile_row ? min(valid_rows - tile_row, A5Rows) : 0;
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
            const int32_t cached_tile_row = (tile_row / A5RowStride) * A5Rows;
            const int32_t cached_a_addr =
                A1HalfUbAddr + cached_tile_row * ChunkSize *
                                   static_cast<int32_t>(sizeof(ComputeT));
            A5TileH a5_cached_a;
            TASSIGN(a5_cached_a, cached_a_addr);
#endif
            if (live_rows > 0) {
              const int64_t a_offset =
                  ((token_start + tile_row) * static_cast<int64_t>(H) +
                   head_idx) *
                  ChunkSize;
              constexpr int32_t a_row_stride = 0;
              (void)a_head_major;
              GmShape2D a_shape(live_rows, ChunkSize);
              GmStride2D a_stride(H * ChunkSize);
              GmTensor2D<ComputeT> a_global(
                  A_handle + a_offset, a_shape, a_stride);
              A5DynamicH a_load(live_rows, ChunkSize);
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
              TASSIGN(a_load, cached_a_addr);
#else
              TASSIGN(a_load, A1HalfUbAddr);
#endif
              TLOAD(a_load, a_global);
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              if (live_rows != A5Rows) {
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
                TFILLPAD_INPLACE(a5_cached_a, a_load);
#else
                TFILLPAD_INPLACE(a5_half, a_load);
#endif
              }
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
              TCVT(a5_float, a5_cached_a, pto::RoundMode::CAST_NONE);
#else
              TCVT(a5_float, a5_half, pto::RoundMode::CAST_NONE);
#endif
              pipe_barrier(PIPE_V);
#ifdef MEGA_CHUNK_GDN_A5_WY_FUSED_A1_A2_PREP
              TMUL(a5_weighted_float, a5_float, a5_weight_2d);
              pipe_barrier(PIPE_V);
              TCVT(a5_io_half, a5_weighted_float, pto::RoundMode::CAST_NONE);
#else
              TMUL(a5_float, a5_float, a5_weight_2d);
              pipe_barrier(PIPE_V);
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
              TCVT(a5_io_half, a5_float, pto::RoundMode::CAST_NONE);
#else
              TCVT(a5_half, a5_float, pto::RoundMode::CAST_NONE);
#endif
#endif
            } else {
              TEXPANDS(a5_float, 0.0f);
              pipe_barrier(PIPE_V);
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
              TCVT(a5_io_half, a5_float, pto::RoundMode::CAST_NONE);
#else
              TCVT(a5_half, a5_float, pto::RoundMode::CAST_NONE);
#endif
            }
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            A5PackedGlobal a2_dst(packed_weighted + tile_row * ChunkSize);
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
            TSTORE(a2_dst, a5_io_half);
#else
            TSTORE(a2_dst, a5_half);
#endif

            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

#ifdef MEGA_CHUNK_GDN_A5_WY_FUSED_A1_A2_PREP
            // Raw A is still live in a5_float.  Materialize A1 with the same
            // FP32 expression as the former second pass, then cache the half
            // result over the no-longer-needed raw-A half tile.
            TMUL(a5_weighted_float, a5_float, a5_a1_weight_2d);
            pipe_barrier(PIPE_V);
            TCVT(a5_cached_a, a5_weighted_float, pto::RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);
#endif

            // Pack V for the right-hand GEMM operand.  Published full chunks
            // can be consumed directly by Cube; ragged tails retain this
            // padded mailbox path.
#ifdef MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_V_L1
            if (valid_rows != ChunkSize) {
#endif
              if (live_rows > 0) {
                const int64_t v_offset =
                    ((token_start + tile_row) * static_cast<int64_t>(H) +
                     head_idx) *
                    HiddenSize;
                GmShape2D v_shape(live_rows, HiddenSize);
                GmStride2D v_stride(BSND_V_STRIDE);
                GmTensor2D<ComputeT> v_global(
                    V_handle + v_offset, v_shape, v_stride);
                A5DynamicH v_load(live_rows, HiddenSize);
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
                TASSIGN(v_load, A2HalfUbAddr);
#else
              TASSIGN(v_load, A1HalfUbAddr);
#endif
                TLOAD(v_load, v_global);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                if (live_rows != A5Rows) {
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
                  TFILLPAD_INPLACE(a5_io_half, v_load);
#else
                TFILLPAD_INPLACE(a5_half, v_load);
#endif
                }
              } else {
                TEXPANDS(a5_float, 0.0f);
                pipe_barrier(PIPE_V);
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
                TCVT(a5_io_half, a5_float, pto::RoundMode::CAST_NONE);
#else
              TCVT(a5_half, a5_float, pto::RoundMode::CAST_NONE);
#endif
              }
              set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              A5PackedGlobal v_dst(packed_rhs + tile_row * HiddenSize);
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
              TSTORE(v_dst, a5_io_half);
#else
            TSTORE(v_dst, a5_half);
#endif
              set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
              wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
#ifdef MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_V_L1
            }
#endif
          }
        }
        if (vid == 0 || A5VectorOwner) {
#ifdef MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
          set_intra_block(PIPE_MTE3, 0 + vid * SYNC_FLAG_ID_MAX);
#else
          set_intra_block(PIPE_MTE3, 0);
#endif
        }
#ifndef MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
        wait_intra_block(PIPE_MTE2, 1);
#endif
#ifdef MEGA_CHUNK_GDN_A5_WY_STAGGERED_MAILBOX
        if (a5_has_pending_a1k) {
          wait_intra_block(PIPE_MTE2, 3);
        }
#endif

        if (vid == 0 || A5VectorOwner) {
#ifndef MEGA_CHUNK_GDN_A5_PACKED_WUV
          // Scatter contiguous U back to BSND.
          for (int32_t tile_row = a5_first_row; tile_row < valid_rows;
               tile_row += A5RowStride) {
            const int32_t live_rows = min(valid_rows - tile_row, A5Rows);
            A5PackedGlobal u_src(packed_weighted + tile_row * HiddenSize);
            TLOAD(a5_half, u_src);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            const int64_t u_offset =
                ((token_start + tile_row) * static_cast<int64_t>(H) +
                 head_idx) *
                HiddenSize;
            GmShape2D u_shape(live_rows, HiddenSize);
            GmStride2D u_stride(BSND_V_STRIDE);
            GmTensor2D<ComputeT> u_global(
                U_handle + u_offset, u_shape, u_stride);
            A5DynamicH u_store(live_rows, HiddenSize);
            TASSIGN(u_store, A1HalfUbAddr);
            TSTORE(u_global, u_store);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
          }
#endif

          // Build exp(g) * beta and A1 = A * weight[None, :].
#ifndef MEGA_CHUNK_GDN_A5_WY_FUSED_A1_A2_PREP
          {
            GmShape2D g_shape(1, valid_rows);
            GmStride2D g_stride(1);
            GmTensor2D<float> g_global(
                G_handle + static_cast<int64_t>(head_idx) * total_tokens +
                    token_start,
                g_shape,
                g_stride);
            DynVecTile<float, 1, ChunkSize, pto::PadValue::Zero> g_load(
                1, valid_rows);
            TASSIGN(g_load, GUbAddr);
            TLOAD(g_load, g_global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            if (valid_rows != ChunkSize) {
              TFILLPAD_INPLACE(g_ub, g_load);
            }
          }
          TEXP(g_ub, g_ub);
          pipe_barrier(PIPE_V);
          TMUL(g_ub, g_ub, beta_ub);
          pipe_barrier(PIPE_V);
          TCOLEXPAND(a5_weight_2d, g_ub);
          pipe_barrier(PIPE_V);
#endif

          const int32_t key_head = head_idx / GROUP;
#ifdef MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
          const bool a5_pack_group_k =
#ifdef MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_K_L1
              valid_rows != ChunkSize &&
#endif
              (!a5_reuse_group_k || (head_idx % GROUP) == 0);
#endif
          for (int32_t tile_row = a5_first_row; tile_row < ChunkSize;
               tile_row += A5RowStride) {
            const int32_t live_rows =
                valid_rows > tile_row ? min(valid_rows - tile_row, A5Rows) : 0;
#ifdef MEGA_CHUNK_GDN_A5_WY_FUSED_A1_A2_PREP
            const int32_t cached_tile_row = (tile_row / A5RowStride) * A5Rows;
            const int32_t cached_a_addr =
                A1HalfUbAddr + cached_tile_row * ChunkSize *
                                   static_cast<int32_t>(sizeof(ComputeT));
            A5TileH a5_cached_a;
            TASSIGN(a5_cached_a, cached_a_addr);
#else
            if (live_rows > 0) {
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
              const int32_t cached_tile_row = (tile_row / A5RowStride) * A5Rows;
              const int32_t cached_a_addr =
                  A1HalfUbAddr + cached_tile_row * ChunkSize *
                                     static_cast<int32_t>(sizeof(ComputeT));
              A5TileH a5_cached_a;
              TASSIGN(a5_cached_a, cached_a_addr);
              TCVT(a5_float, a5_cached_a, pto::RoundMode::CAST_NONE);
#else
              const int64_t a_offset =
                  ((token_start + tile_row) * static_cast<int64_t>(H) +
                   head_idx) *
                  ChunkSize;
              GmShape2D a_shape(live_rows, ChunkSize);
              GmStride2D a_stride(H * ChunkSize);
              GmTensor2D<ComputeT> a_global(
                  A_handle + a_offset, a_shape, a_stride);
              A5DynamicH a_load(live_rows, ChunkSize);
              TASSIGN(a_load, A1HalfUbAddr);
              TLOAD(a_load, a_global);
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              if (live_rows != A5Rows) {
                TFILLPAD_INPLACE(a5_half, a_load);
              }
              TCVT(a5_float, a5_half, pto::RoundMode::CAST_NONE);
#endif
              pipe_barrier(PIPE_V);
              TMUL(a5_float, a5_float, a5_weight_2d);
              pipe_barrier(PIPE_V);
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
              TCVT(a5_io_half, a5_float, pto::RoundMode::CAST_NONE);
#else
              TCVT(a5_half, a5_float, pto::RoundMode::CAST_NONE);
#endif
            } else {
              TEXPANDS(a5_float, 0.0f);
              pipe_barrier(PIPE_V);
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
              TCVT(a5_io_half, a5_float, pto::RoundMode::CAST_NONE);
#else
              TCVT(a5_half, a5_float, pto::RoundMode::CAST_NONE);
#endif
            }
#endif
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            A5PackedGlobal a1_dst(
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
                packed_a1 + tile_row * ChunkSize);
#ifdef MEGA_CHUNK_GDN_A5_WY_FUSED_A1_A2_PREP
            TSTORE(a1_dst, a5_cached_a);
#else
            TSTORE(a1_dst, a5_io_half);
#endif
#else
                packed_weighted + tile_row * ChunkSize);
            TSTORE(a1_dst, a5_half);
#endif
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

#ifdef MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
            if (a5_pack_group_k) {
#endif
              if (live_rows > 0) {
                const int64_t k_offset =
                    ((token_start + tile_row) * static_cast<int64_t>(Hg) +
                     key_head) *
                    HiddenSize;
                GmShape2D k_shape(live_rows, HiddenSize);
                GmStride2D k_stride(BSND_QK_STRIDE);
                GmTensor2D<ComputeT> k_global(
                    K_handle + k_offset, k_shape, k_stride);
                A5DynamicH k_load(live_rows, HiddenSize);
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
                TASSIGN(k_load, A2HalfUbAddr);
#else
              TASSIGN(k_load, A1HalfUbAddr);
#endif
                TLOAD(k_load, k_global);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                if (live_rows != A5Rows) {
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
                  TFILLPAD_INPLACE(a5_io_half, k_load);
#else
                TFILLPAD_INPLACE(a5_half, k_load);
#endif
                }
              } else {
                TEXPANDS(a5_float, 0.0f);
                pipe_barrier(PIPE_V);
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
                TCVT(a5_io_half, a5_float, pto::RoundMode::CAST_NONE);
#else
              TCVT(a5_half, a5_float, pto::RoundMode::CAST_NONE);
#endif
              }
              set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              A5PackedGlobal k_dst(
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
                  packed_k + tile_row * HiddenSize);
              TSTORE(k_dst, a5_io_half);
#else
                packed_rhs + tile_row * HiddenSize);
            TSTORE(k_dst, a5_half);
#endif
              set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
              wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
#ifdef MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
            }
#endif
          }
        }
        if (vid == 0 || A5VectorOwner) {
#ifdef MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
          set_intra_block(PIPE_MTE3, 2 + vid * SYNC_FLAG_ID_MAX);
#else
          set_intra_block(PIPE_MTE3, 2);
#endif
        }
#ifdef MEGA_CHUNK_GDN_A5_WY_STAGGERED_MAILBOX
        a5_has_pending_a1k = true;
        // A2/V for this head are already resident in Cube L1.  Wait only for
        // their early release before starting the next head's first phase;
        // A1/K are released separately immediately before their next write.
        wait_intra_block(PIPE_MTE2, 1);
#else
        wait_intra_block(PIPE_MTE2, 3);
#endif

#ifndef MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
        if (vid == 0 || A5VectorOwner) {
#ifndef MEGA_CHUNK_GDN_A5_PACKED_WUV
          for (int32_t tile_row = a5_first_row; tile_row < valid_rows;
               tile_row += A5RowStride) {
            const int32_t live_rows = min(valid_rows - tile_row, A5Rows);
            A5PackedGlobal w_src(packed_weighted + tile_row * HiddenSize);
            TLOAD(a5_half, w_src);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            const int64_t w_offset =
                ((token_start + tile_row) * static_cast<int64_t>(H) +
                 head_idx) *
                HiddenSize;
            GmShape2D w_shape(live_rows, HiddenSize);
            GmStride2D w_stride(BSND_V_STRIDE);
            GmTensor2D<ComputeT> w_global(
                W_handle + w_offset, w_shape, w_stride);
            A5DynamicH w_store(live_rows, HiddenSize);
            TASSIGN(w_store, A1HalfUbAddr);
            TSTORE(w_global, w_store);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
          }
#endif
        }
        if (vid == 0 || A5VectorOwner) {
#ifdef MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
          set_intra_block(PIPE_MTE3, 4 + vid * SYNC_FLAG_ID_MAX);
#else
          set_intra_block(PIPE_MTE3, 4);
#endif
        }
        wait_intra_block(PIPE_MTE2, 5);
#endif
      }
    }
  }
#ifdef MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
  // Scratch-free (flag 3) is intentionally early: it is sourced by Cube MTE2
  // after A1/K have reached L1, so AIV may prepare the next head while Cube
  // computes W.  Do not leave WY on that early flag, however; H consumes W/U.
  // The once-per-block flag 5 below is sourced by the final FIX store and is
  // also emitted for blocks with no assigned head task.
  wait_intra_block(PIPE_MTE2, 5 + static_cast<int32_t>(vid) * SYNC_FLAG_ID_MAX);
#endif
  return;

#elif defined(__DAV_C310_CUBE__)
  // A5 Cube side of the packed schedule above. Both operands and both results
  // are contiguous, so no architecture-dependent BSND ND/NZ conversion is
  // involved in the two 128x128 GEMMs.
  int64_t a5_work = 0;
  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int64_t bos = cu_seqlens == nullptr
                            ? seq_idx * seq_len
                            : static_cast<int64_t>(cu_seqlens[seq_idx]);
    const int64_t eos = cu_seqlens == nullptr
                            ? bos + seq_len
                            : static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
    const int64_t slen = eos - bos;
    const int64_t num_chunks = (slen + ChunkSize - 1) / ChunkSize;
    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      for (int32_t head_idx = 0; head_idx < H; ++head_idx, ++a5_work) {
#ifdef MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
        const int64_t a5_owner_work =
            a5_reuse_group_k ? a5_work / GROUP : a5_work;
        const bool a5_load_group_k =
            !a5_reuse_group_k || (head_idx % GROUP) == 0;
#else
        const int64_t a5_owner_work = a5_work;
#endif
        if (a5_owner_work % static_cast<int64_t>(block_num) !=
            static_cast<int64_t>(cid)) {
          continue;
        }
        __gm__ ComputeT* packed_rhs =
            workspace_a1_handle + static_cast<int64_t>(cid) * WsA1Size;
        __gm__ ComputeT* packed_weighted =
            workspace_a2_handle + static_cast<int64_t>(cid) * WsA2Size;
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
        __gm__ ComputeT* packed_a1 =
            workspace_reuse_handle + static_cast<int64_t>(cid) * 2 * WsA1Size;
        __gm__ ComputeT* packed_k = packed_a1 + WsA1Size;
#endif
        GmShape2D packed_shape(ChunkSize, HiddenSize);
        GmStride2D packed_stride(HiddenSize);
#if defined(MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_K_L1) || \
    defined(MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_V_L1)
        const int64_t token_start = bos + chunk_idx * ChunkSize;
        const int32_t valid_rows = static_cast<int32_t>(
            min(static_cast<int64_t>(ChunkSize), slen - chunk_idx * ChunkSize));
#endif

        wait_intra_block(PIPE_S, 0);
#ifdef MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
        wait_intra_block(PIPE_S, 0 + SYNC_FLAG_ID_MAX);
#endif
        GmTensor2D<ComputeT> a2_global(
            packed_weighted, packed_shape, packed_stride);
        GmTensor2D<ComputeT> v_global(packed_rhs, packed_shape, packed_stride);
        TLOAD(a2_l1, a2_global);
#ifdef MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_V_L1
        if (valid_rows == ChunkSize) {
          GmShape2D v_input_shape(ChunkSize, HiddenSize);
          GmStride2D v_input_stride(BSND_V_STRIDE);
          GmTensor2D<ComputeT> v_input_global(
              V_handle + (token_start * static_cast<int64_t>(H) + head_idx) *
                             HiddenSize,
              v_input_shape,
              v_input_stride);
          TLOAD(v_l1, v_input_global);
        } else {
          TLOAD(v_l1, v_global);
        }
#else
        TLOAD(v_l1, v_global);
#endif
#ifdef MEGA_CHUNK_GDN_A5_WY_STAGGERED_MAILBOX
        // Both first-phase mailboxes are now resident in L1.  Release them
        // from MTE2 so Vec can prepare A2/V for the next owned head while U
        // computes; A1/K keep their independent flag-3 lifetime below.
        set_intra_block(PIPE_MTE2, 1);
        set_intra_block(PIPE_MTE2, 1 + SYNC_FLAG_ID_MAX);
#endif
#ifdef MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
        // The previous head's W result can still be draining through FIX while
        // these independent L1 loads run.  Order that drain before reusing the
        // single active L0C accumulator for this U GEMM.
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
#endif
        gemm_v0<ComputeT,
                float,
                ChunkSize,
                HiddenSize,
                ChunkSize,
                ChunkSize,
                HiddenSize,
                ChunkSize,
                KTail,
                false,
                false>(a2_l1, v_l1, u_l0, true);
#ifdef MEGA_CHUNK_GDN_A5_PACKED_WUV
        // W/U are private intermediates.  Persist them as [H,T,D] so the
        // following H stage can issue contiguous loads instead of gathering
        // the BSND rows which the AIV path used to scatter here.
        {
          const int64_t token_start = bos + chunk_idx * ChunkSize;
          const int32_t valid_rows = static_cast<int32_t>(min(
              static_cast<int64_t>(ChunkSize), slen - chunk_idx * ChunkSize));
          GmShape2D u_shape(valid_rows, HiddenSize);
          GmStride2D u_stride(HiddenSize);
          GmTensor2D<ComputeT> u_global(
              U_handle + (static_cast<int64_t>(head_idx) * total_tokens +
                          token_start) *
                             HiddenSize,
              u_shape,
              u_stride);
          DynAccTile<float, ChunkSize, HiddenSize> u_store(valid_rows,
                                                           HiddenSize);
          TASSIGN(u_store, 0);
          TSTORE(u_global, u_store);
        }
#else
        GmTensor2D<ComputeT> u_global(
            packed_weighted, packed_shape, packed_stride);
        TSTORE(u_global, u_l0);
#endif
#ifndef MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
        set_intra_block(PIPE_FIX, 1);
        set_intra_block(PIPE_FIX, 1 + SYNC_FLAG_ID_MAX);
#endif

        wait_intra_block(PIPE_S, 2);
#ifdef MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
        wait_intra_block(PIPE_S, 2 + SYNC_FLAG_ID_MAX);
#endif
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
        GmTensor2D<ComputeT> a1_global(packed_a1, packed_shape, packed_stride);
        GmTensor2D<ComputeT> k_global(packed_k, packed_shape, packed_stride);
        TLOAD(a2_l1, a1_global);
#ifdef MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
        if (a5_load_group_k) {
#ifdef MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_K_L1
          if (valid_rows == ChunkSize) {
            const int32_t key_head = head_idx / GROUP;
            GmShape2D k_input_shape(ChunkSize, HiddenSize);
            GmStride2D k_input_stride(BSND_QK_STRIDE);
            GmTensor2D<ComputeT> k_input_global(
                K_handle + (token_start * static_cast<int64_t>(Hg) + key_head) *
                               HiddenSize,
                k_input_shape,
                k_input_stride);
            TLOAD(k_l1, k_input_global);
          } else {
            TLOAD(k_l1, k_global);
          }
#else
          TLOAD(k_l1, k_global);
#endif
        }
#else
        TLOAD(v_l1, k_global);
#endif
#else
        TLOAD(a2_l1, a2_global);
        TLOAD(v_l1, v_global);
#endif
#ifdef MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
        // All four GM mailboxes are now dead for this head: U has consumed
        // A2/V, and A1/K are resident in L1.  Publish that fact directly from
        // MTE2, before W's Cube work, to overlap W(n) with Vec preparation of
        // head n+1.  Both sibling AIVs receive their own flag generation.
        set_intra_block(PIPE_MTE2, 3);
        set_intra_block(PIPE_MTE2, 3 + SYNC_FLAG_ID_MAX);
#endif
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        gemm_v0<ComputeT,
                float,
                ChunkSize,
                HiddenSize,
                ChunkSize,
                ChunkSize,
                HiddenSize,
                ChunkSize,
                KTail,
                false,
                false>(
#ifdef MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
            a2_l1, k_l1, u_l0, true);
#else
            a2_l1, v_l1, u_l0, true);
#endif
#ifdef MEGA_CHUNK_GDN_A5_PACKED_WUV
        {
          const int64_t token_start = bos + chunk_idx * ChunkSize;
          const int32_t valid_rows = static_cast<int32_t>(min(
              static_cast<int64_t>(ChunkSize), slen - chunk_idx * ChunkSize));
          GmShape2D w_shape(valid_rows, HiddenSize);
          GmStride2D w_stride(HiddenSize);
          GmTensor2D<ComputeT> w_global(
              W_handle + (static_cast<int64_t>(head_idx) * total_tokens +
                          token_start) *
                             HiddenSize,
              w_shape,
              w_stride);
          DynAccTile<float, ChunkSize, HiddenSize> w_store(valid_rows,
                                                           HiddenSize);
          TASSIGN(w_store, 0);
          TSTORE(w_global, w_store);
        }
#else
        TSTORE(u_global, u_l0);
#endif
#ifndef MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
        set_intra_block(PIPE_FIX, 3);
        set_intra_block(PIPE_FIX, 3 + SYNC_FLAG_ID_MAX);

        wait_intra_block(PIPE_S, 4);
#ifdef MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
        wait_intra_block(PIPE_S, 4 + SYNC_FLAG_ID_MAX);
#endif
        set_intra_block(PIPE_S, 5);
        set_intra_block(PIPE_S, 5 + SYNC_FLAG_ID_MAX);
#endif
      }
    }
  }
#ifdef MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
  // Flag 3 permits mailbox reuse, not stage exit.  Publish the final W store
  // exactly once so neither AIV can enter the downstream H stage early.  A
  // PIPE_FIX source also closes the L0C/FIX lifetime before this AIC returns.
  set_intra_block(PIPE_FIX, 5);
  set_intra_block(PIPE_FIX, 5 + SYNC_FLAG_ID_MAX);
#endif
  return;
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  // Vec prepares the two reweighted A workspaces (`A2` and `A1`) that the
  // Cube phase consumes later.
  if (cu_seqlens == nullptr) {
    bool first_iter = true;
    int64_t gi = 0;
    for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t head_idx = 0; head_idx < H; ++head_idx) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            // Each Vec sub-block owns one HalfChunk-row stripe of the chunk.
            // For a tail chunk, the upper stripe (vid=0) may hold fewer than
            // 64 rows, and the lower stripe (vid=1) may hold only a suffix or
            // no rows at all.  `local_rows` is the exact number of live rows in
            // THIS sub-block's stripe.
            int32_t local_rows =
                valid_rows - static_cast<int32_t>(vid) * HalfChunk;
            if (local_rows < 0) local_rows = 0;
            if (local_rows > HalfChunk) local_rows = HalfChunk;
            if (local_rows == 0) {
#if defined(PTO_NPU_ARCH_A5)
              if (!first_iter) gdn_sync::AllocateVecGm(3);
              gdn_sync::RecordVecGm(1 | (2 << 4) | (2 << 8));
#else
              if (!first_iter) wait_flag_dev(3);
              ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));
#endif

#if defined(PTO_NPU_ARCH_A5)
              if (!first_iter) gdn_sync::AllocateVecGm(4);
              gdn_sync::RecordVecGm(1 | (2 << 4) | (1 << 8));
#else
              if (!first_iter) wait_flag_dev(4);
              ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
#endif
              first_iter = false;
              gi++;
              continue;
            }

            // Beta is pre-transposed to [H, total_tokens] for contiguous loads.
            {
              GmShape2D beta_shape(1, valid_rows);
              GmStride2D beta_stride(1);
              GmTensor2D<ComputeT> beta_global(
                  Beta_handle + static_cast<int64_t>(head_idx) * total_tokens +
                      chunk_token_start,
                  beta_shape,
                  beta_stride);
              DynVecTile<ComputeT, 1, ChunkSize, pto::PadValue::Zero> beta_load(
                  1, valid_rows);
              TASSIGN(beta_load, BetaHalfUbAddr);
              TLOAD(beta_load, beta_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD_INPLACE(beta_ub_half, beta_load);
              }
            }

            // Load only the live rows for this sub-block, then zero-pad the
            // remainder of the HalfChunk tile.  The Cube phase always consumes
            // a full [HalfChunk, ChunkSize] workspace tile, so stale rows here
            // would leak garbage into ragged tails and cross-sequence
            // boundaries.
            if (local_rows > 0) {
              int64_t a_gm_offset =
                  ((chunk_token_start + static_cast<int64_t>(vid) * HalfChunk) *
                       H +
                   head_idx) *
                  static_cast<int64_t>(ChunkSize);
              GmShape2D a_shape(local_rows, valid_rows);
              GmStride2D a_stride(H * ChunkSize);
              GmTensor2D<ComputeT> a_global(
                  A_handle + a_gm_offset, a_shape, a_stride);
              DynVecTile<ComputeT, HalfChunk, ChunkSize, pto::PadValue::Zero>
                  a_load(local_rows, valid_rows);
              TASSIGN(a_load, A1HalfUbAddr);
              TLOAD(a_load, a_global);
              if (local_rows != HalfChunk || valid_rows != ChunkSize) {
                TFILLPAD_INPLACE(a1_ub_half, a_load);
              }
            } else {
              // Fully empty lower-half tail: materialize an all-zero tile so
              // the workspace still looks like a correctly padded HalfChunk
              // block.
              TEXPANDS(a1_ub, 0.0f);
              pipe_barrier(PIPE_V);
              TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);
            TMOV(beta_r_ub, beta_ub);
            pipe_barrier(PIPE_V);
            // Replicate beta_j across rows so every column j of A gets the same
            // beta. PyTorch-like:
            //   beta_2d = beta[None, :].expand(HalfChunk, ChunkSize)
            TCOLEXPAND(beta_2d_ub, beta_r_ub);

            TCVT(a1_ub, a1_ub_half, pto::RoundMode::CAST_NONE);
            // Form the beta-scaled tile that the later U = A2 * V matmul
            // consumes.
            //   a2_ub = a1_ub * beta_2d_ub
            TMUL(a2_ub, a1_ub, beta_2d_ub);
            TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);

#if defined(PTO_NPU_ARCH_A5)
            if (!first_iter) gdn_sync::AllocateVecGm(3);
#else
            if (!first_iter) wait_flag_dev(3);
#endif
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D a2_shape(HalfChunk, ChunkSize);
              GmStride2D a2_stride(ChunkSize);
              GmTensor2D<ComputeT> workspace_a2_global(
                  workspace_a2_handle + static_cast<int64_t>(cid) * WsA2Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                  a2_shape,
                  a2_stride);
              TSTORE(workspace_a2_global, a2_ub_half);
            }
            pipe_barrier(PIPE_ALL);
#if defined(PTO_NPU_ARCH_A5)
            gdn_sync::RecordVecGm(1 | (2 << 4) | (2 << 8));
#else
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));
#endif

            // G is pre-transposed to [H, total_tokens] for contiguous loads.
            {
              GmShape2D g_shape(1, valid_rows);
              GmStride2D g_stride(1);
              GmTensor2D<float> g_global(
                  G_handle + static_cast<int64_t>(head_idx) * total_tokens +
                      chunk_token_start,
                  g_shape,
                  g_stride);
              DynVecTile<float, 1, ChunkSize, pto::PadValue::Zero> g_load(
                  1, valid_rows);
              TASSIGN(g_load, GUbAddr);
              TLOAD(g_load, g_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD_INPLACE(g_ub, g_load);
              }
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            // Build the g-based column weights before forming the W = A1 * K
            // branch. Torch-like:
            //   g_weight = exp(g) * beta
            TEXP(g_ub, g_ub);
            pipe_barrier(PIPE_V);
            TMUL(g_ub, g_ub, beta_ub);
            pipe_barrier(PIPE_V);
            TMOV(g_r_ub, g_ub);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(g_2d_ub, g_r_ub);
            // A1 keeps the same A columns but multiplies each one by exp(g_j) *
            // beta_j.
            //   a1_ub = a1_ub * g_weight[None, :]
            TMUL(a1_ub, a1_ub, g_2d_ub);
            TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);

#if defined(PTO_NPU_ARCH_A5)
            if (!first_iter) gdn_sync::AllocateVecGm(4);
#else
            if (!first_iter) wait_flag_dev(4);
#endif
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D a1_shape(HalfChunk, ChunkSize);
              GmStride2D a1_stride(ChunkSize);
              GmTensor2D<ComputeT> workspace_a1_global(
                  workspace_a1_handle + static_cast<int64_t>(cid) * WsA1Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                  a1_shape,
                  a1_stride);
              TSTORE(workspace_a1_global, a1_ub_half);
            }
            pipe_barrier(PIPE_ALL);
#if defined(PTO_NPU_ARCH_A5)
            gdn_sync::RecordVecGm(1 | (2 << 4) | (1 << 8));
#else
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
#endif
            first_iter = false;
          }
          gi++;
        }
      }
    }
  } else {
    // Same WY math as above; only the work enumeration changes for varlen
    // input.
    int64_t gi = 0;
    bool first_iter_v = true;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < H; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            // Same HalfChunk ownership rule as the fixed-length path above:
            // each Vec sub-block handles one 64-row stripe, and ragged varlen
            // tails may leave that stripe partially full or fully empty.
            int32_t local_rows =
                valid_rows - static_cast<int32_t>(vid) * HalfChunk;
            if (local_rows < 0) local_rows = 0;
            if (local_rows > HalfChunk) local_rows = HalfChunk;
            int32_t head_idx = h;
            if (local_rows == 0) {
#if defined(PTO_NPU_ARCH_A5)
              if (!first_iter_v) gdn_sync::AllocateVecGm(3);
              gdn_sync::RecordVecGm(1 | (2 << 4) | (2 << 8));
#else
              if (!first_iter_v) wait_flag_dev(3);
              ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));
#endif

#if defined(PTO_NPU_ARCH_A5)
              if (!first_iter_v) gdn_sync::AllocateVecGm(4);
              gdn_sync::RecordVecGm(1 | (2 << 4) | (1 << 8));
#else
              if (!first_iter_v) wait_flag_dev(4);
              ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
#endif
              first_iter_v = false;
              gi++;
              continue;
            }

            // Beta is pre-transposed to [H, total_tokens] for contiguous loads.
            {
              GmShape2D beta_shape(1, valid_rows);
              GmStride2D beta_stride(1);
              GmTensor2D<ComputeT> beta_global(
                  Beta_handle + static_cast<int64_t>(head_idx) * total_tokens +
                      chunk_token_start,
                  beta_shape,
                  beta_stride);
              DynVecTile<ComputeT, 1, ChunkSize, pto::PadValue::Zero> beta_load(
                  1, valid_rows);
              TASSIGN(beta_load, BetaHalfUbAddr);
              TLOAD(beta_load, beta_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD_INPLACE(beta_ub_half, beta_load);
              }
            }

            // Tail-safe A loading is especially important in varlen mode
            // because the final chunk of one sequence may be immediately
            // followed by the first chunk of the next sequence in packed
            // storage.
            if (local_rows > 0) {
              int64_t a_gm_offset =
                  ((chunk_token_start + static_cast<int64_t>(vid) * HalfChunk) *
                       H +
                   head_idx) *
                  static_cast<int64_t>(ChunkSize);
              GmShape2D a_shape(local_rows, valid_rows);
              GmStride2D a_stride(H * ChunkSize);
              GmTensor2D<ComputeT> a_global(
                  A_handle + a_gm_offset, a_shape, a_stride);
              DynVecTile<ComputeT, HalfChunk, ChunkSize, pto::PadValue::Zero>
                  a_load(local_rows, valid_rows);
              TASSIGN(a_load, A1HalfUbAddr);
              TLOAD(a_load, a_global);
              if (local_rows != HalfChunk || valid_rows != ChunkSize) {
                TFILLPAD_INPLACE(a1_ub_half, a_load);
              }
            } else {
              // Empty stripe for this sub-block: write zeros so the downstream
              // full-tile Cube GEMM sees valid padding rather than old
              // workspace.
              TEXPANDS(a1_ub, 0.0f);
              pipe_barrier(PIPE_V);
              TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);
            TMOV(beta_r_ub, beta_ub);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(beta_2d_ub, beta_r_ub);

            TCVT(a1_ub, a1_ub_half, pto::RoundMode::CAST_NONE);
            // Form the beta-scaled tile that the later U = A2 * V matmul
            // consumes.
            TMUL(a2_ub, a1_ub, beta_2d_ub);
            TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);

#if defined(PTO_NPU_ARCH_A5)
            if (!first_iter_v) gdn_sync::AllocateVecGm(3);
#else
            if (!first_iter_v) wait_flag_dev(3);
#endif
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D a2_shape(HalfChunk, ChunkSize);
              GmStride2D a2_stride(ChunkSize);
              GmTensor2D<ComputeT> workspace_a2_global(
                  workspace_a2_handle + static_cast<int64_t>(cid) * WsA2Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                  a2_shape,
                  a2_stride);
              TSTORE(workspace_a2_global, a2_ub_half);
            }
            pipe_barrier(PIPE_ALL);
#if defined(PTO_NPU_ARCH_A5)
            gdn_sync::RecordVecGm(1 | (2 << 4) | (2 << 8));
#else
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));
#endif

            // G is pre-transposed to [H, total_tokens] for contiguous loads.
            {
              GmShape2D g_shape(1, valid_rows);
              GmStride2D g_stride(1);
              GmTensor2D<float> g_global(
                  G_handle + static_cast<int64_t>(head_idx) * total_tokens +
                      chunk_token_start,
                  g_shape,
                  g_stride);
              DynVecTile<float, 1, ChunkSize, pto::PadValue::Zero> g_load(
                  1, valid_rows);
              TASSIGN(g_load, GUbAddr);
              TLOAD(g_load, g_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD_INPLACE(g_ub, g_load);
              }
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            // Build the g-based column weights before forming the W = A1 * K
            // branch.
            TEXP(g_ub, g_ub);
            pipe_barrier(PIPE_V);
            TMUL(g_ub, g_ub, beta_ub);
            pipe_barrier(PIPE_V);
            TMOV(g_r_ub, g_ub);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(g_2d_ub, g_r_ub);
            TMUL(a1_ub, a1_ub, g_2d_ub);
            TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);

#if defined(PTO_NPU_ARCH_A5)
            if (!first_iter_v) gdn_sync::AllocateVecGm(4);
#else
            if (!first_iter_v) wait_flag_dev(4);
#endif
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D a1_shape(HalfChunk, ChunkSize);
              GmStride2D a1_stride(ChunkSize);
              GmTensor2D<ComputeT> workspace_a1_global(
                  workspace_a1_handle + static_cast<int64_t>(cid) * WsA1Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                  a1_shape,
                  a1_stride);
              TSTORE(workspace_a1_global, a1_ub_half);
            }
            pipe_barrier(PIPE_ALL);
#if defined(PTO_NPU_ARCH_A5)
            gdn_sync::RecordVecGm(1 | (2 << 4) | (1 << 8));
#else
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
#endif
            first_iter_v = false;
          }
          gi++;
        }
      }
    }
  }
#endif

#if defined(__DAV_C220_CUBE__)
  // Cube consumes the two Vec-generated workspaces and turns them into the
  // branch outputs U and W.
  if (cu_seqlens == nullptr) {
    int64_t gi = 0;
    for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t head_idx = 0; head_idx < H; ++head_idx) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;

            int32_t head_g = head_idx / GROUP;
            int64_t k_off = (chunk_token_start * static_cast<int64_t>(Hg) +
                             static_cast<int64_t>(head_g)) *
                            static_cast<int64_t>(HiddenSize);
            int64_t v_off = (chunk_token_start * static_cast<int64_t>(H) +
                             static_cast<int64_t>(head_idx)) *
                            static_cast<int64_t>(HiddenSize);

            {
              GmShape2D k_shape(valid_rows, HiddenSize);
              GmStride2D k_stride(BSND_QK_STRIDE);
              GmTensor2D<ComputeT> k_global(
                  K_handle + k_off, k_shape, k_stride);
              DynMatL1<ComputeT, ChunkSize, HiddenSize> k_l1_load(valid_rows,
                                                                  HiddenSize);
              TASSIGN(k_l1_load, 0);
              TLOAD(k_l1_load, k_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD(k_l1_load, k_l1_load);
              }
            }
            {
              GmShape2D v_shape(valid_rows, HiddenSize);
              GmStride2D v_stride(BSND_V_STRIDE);
              GmTensor2D<ComputeT> v_global(
                  V_handle + v_off, v_shape, v_stride);
              DynMatL1<ComputeT, ChunkSize, HiddenSize> v_l1_load(valid_rows,
                                                                  HiddenSize);
              TASSIGN(v_l1_load, 32768);
              TLOAD(v_l1_load, v_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD(v_l1_load, v_l1_load);
              }
            }

#if defined(PTO_NPU_ARCH_A5)
            gdn_sync::WaitVecGm(2);
#else
            wait_flag_dev(2);
#endif
            {
              GmShape2D a2_shape(ChunkSize, ChunkSize);
              GmStride2D a2_stride(ChunkSize);
              GmTensor2D<ComputeT> workspace_a2_global(
                  workspace_a2_handle + static_cast<int64_t>(cid) * WsA2Size,
                  a2_shape,
                  a2_stride);
              // Load the Vec-prepared A2 tile:
              //   A2 = A * beta[None, :]
              TLOAD(a2_l1, workspace_a2_global);
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            // U = A2 * V keeps the beta-scaled path separate from the K-side
            // update.
            gemm_v0<ComputeT,
                    float,
                    ChunkSize,
                    HiddenSize,
                    ChunkSize,
                    ChunkSize,
                    HiddenSize,
                    ChunkSize,
                    KTail,
                    false,
                    false>(a2_l1, v_l1, u_l0, true);

            {
              GmShape2D u_shape(valid_rows, HiddenSize);
              GmStride2D u_stride(BSND_V_STRIDE);
              GmTensor2D<ComputeT> u_global(
                  U_handle + v_off, u_shape, u_stride);
              DynAccTile<float, ChunkSize, HiddenSize> u_store(valid_rows,
                                                               HiddenSize);
              TASSIGN(u_store, 0);
              // Store only the valid token rows even though the accumulator
              // tile is physically ChunkSize x HiddenSize.
              TSTORE(u_global, u_store);
            }
#if defined(PTO_NPU_ARCH_A5)
            gdn_sync::FreeVecGm(1 | (2 << 4) | (3 << 8));
#else
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (3 << 8));
#endif

#if defined(PTO_NPU_ARCH_A5)
            gdn_sync::WaitVecGm(1);
#else
            wait_flag_dev(1);
#endif
            {
              GmShape2D a1_shape(ChunkSize, ChunkSize);
              GmStride2D a1_stride(ChunkSize);
              GmTensor2D<ComputeT> workspace_a1_global(
                  workspace_a1_handle + static_cast<int64_t>(cid) * WsA1Size,
                  a1_shape,
                  a1_stride);
              // Load the Vec-prepared A1 tile:
              //   A1 = A * (exp(g) * beta)[None, :]
              TLOAD(a1_l1, workspace_a1_global);
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            // W = A1 * K uses the g-reweighted path for the complementary WY
            // factor.
            gemm_v0<ComputeT,
                    float,
                    ChunkSize,
                    HiddenSize,
                    ChunkSize,
                    ChunkSize,
                    HiddenSize,
                    ChunkSize,
                    KTail,
                    false,
                    false>(a1_l1, k_l1, w_l0, true);

            {
              GmShape2D w_shape(valid_rows, HiddenSize);
              GmStride2D w_stride(BSND_V_STRIDE);
              GmTensor2D<ComputeT> w_global(
                  W_handle + v_off, w_shape, w_stride);
              DynAccTile<float, ChunkSize, HiddenSize> w_store(valid_rows,
                                                               HiddenSize);
              TASSIGN(w_store, 65536);
              TSTORE(w_global, w_store);
            }
#if defined(PTO_NPU_ARCH_A5)
            gdn_sync::FreeVecGm(1 | (2 << 4) | (4 << 8));
#else
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (4 << 8));
#endif
          }
          gi++;
        }
      }
    }
  } else {
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < H; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t head_idx = h;

            int32_t head_g = head_idx / GROUP;
            int64_t k_off = (chunk_token_start * static_cast<int64_t>(Hg) +
                             static_cast<int64_t>(head_g)) *
                            static_cast<int64_t>(HiddenSize);
            int64_t v_off = (chunk_token_start * static_cast<int64_t>(H) +
                             static_cast<int64_t>(head_idx)) *
                            static_cast<int64_t>(HiddenSize);

            {
              GmShape2D k_shape(valid_rows, HiddenSize);
              GmStride2D k_stride(BSND_QK_STRIDE);
              GmTensor2D<ComputeT> k_global(
                  K_handle + k_off, k_shape, k_stride);
              DynMatL1<ComputeT, ChunkSize, HiddenSize> k_l1_load(valid_rows,
                                                                  HiddenSize);
              TASSIGN(k_l1_load, 0);
              TLOAD(k_l1_load, k_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD(k_l1_load, k_l1_load);
              }
            }
            {
              GmShape2D v_shape(valid_rows, HiddenSize);
              GmStride2D v_stride(BSND_V_STRIDE);
              GmTensor2D<ComputeT> v_global(
                  V_handle + v_off, v_shape, v_stride);
              DynMatL1<ComputeT, ChunkSize, HiddenSize> v_l1_load(valid_rows,
                                                                  HiddenSize);
              TASSIGN(v_l1_load, 32768);
              TLOAD(v_l1_load, v_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD(v_l1_load, v_l1_load);
              }
            }

#if defined(PTO_NPU_ARCH_A5)
            gdn_sync::WaitVecGm(2);
#else
            wait_flag_dev(2);
#endif
            {
              GmShape2D a2_shape(ChunkSize, ChunkSize);
              GmStride2D a2_stride(ChunkSize);
              GmTensor2D<ComputeT> workspace_a2_global(
                  workspace_a2_handle + static_cast<int64_t>(cid) * WsA2Size,
                  a2_shape,
                  a2_stride);
              TLOAD(a2_l1, workspace_a2_global);
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            // U = A2 * V keeps the beta-scaled path separate from the K-side
            // update.
            gemm_v0<ComputeT,
                    float,
                    ChunkSize,
                    HiddenSize,
                    ChunkSize,
                    ChunkSize,
                    HiddenSize,
                    ChunkSize,
                    KTail,
                    false,
                    false>(a2_l1, v_l1, u_l0, true);

            {
              GmShape2D u_shape(valid_rows, HiddenSize);
              GmStride2D u_stride(BSND_V_STRIDE);
              GmTensor2D<ComputeT> u_global(
                  U_handle + v_off, u_shape, u_stride);
              DynAccTile<float, ChunkSize, HiddenSize> u_store(valid_rows,
                                                               HiddenSize);
              TASSIGN(u_store, 0);
              TSTORE(u_global, u_store);
            }
#if defined(PTO_NPU_ARCH_A5)
            gdn_sync::FreeVecGm(1 | (2 << 4) | (3 << 8));
#else
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (3 << 8));
#endif

#if defined(PTO_NPU_ARCH_A5)
            gdn_sync::WaitVecGm(1);
#else
            wait_flag_dev(1);
#endif
            {
              GmShape2D a1_shape(ChunkSize, ChunkSize);
              GmStride2D a1_stride(ChunkSize);
              GmTensor2D<ComputeT> workspace_a1_global(
                  workspace_a1_handle + static_cast<int64_t>(cid) * WsA1Size,
                  a1_shape,
                  a1_stride);
              TLOAD(a1_l1, workspace_a1_global);
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            // W = A1 * K uses the g-reweighted path for the complementary WY
            // factor.
            gemm_v0<ComputeT,
                    float,
                    ChunkSize,
                    HiddenSize,
                    ChunkSize,
                    ChunkSize,
                    HiddenSize,
                    ChunkSize,
                    KTail,
                    false,
                    false>(a1_l1, k_l1, w_l0, true);

            {
              GmShape2D w_shape(valid_rows, HiddenSize);
              GmStride2D w_stride(BSND_V_STRIDE);
              GmTensor2D<ComputeT> w_global(
                  W_handle + v_off, w_shape, w_stride);
              DynAccTile<float, ChunkSize, HiddenSize> w_store(valid_rows,
                                                               HiddenSize);
              TASSIGN(w_store, 65536);
              TSTORE(w_global, w_store);
            }
#if defined(PTO_NPU_ARCH_A5)
            gdn_sync::FreeVecGm(1 | (2 << 4) | (4 << 8));
#else
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (4 << 8));
#endif
          }
          gi++;
        }
      }
    }
  }
#endif
}

#undef GDN_WY_FAST_KERNEL
