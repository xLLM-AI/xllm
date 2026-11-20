// ============================================================================
// chunk_h_kernel.cpp — Recurrent hidden state update for GatedDeltaNet
//
// Mathematical recurrence per chunk c:
//   S_{c+1} = exp(g_last) * S_c  +  K^T @ V
//
// where g_last = exp(g[valid-1]) is the chunk's final gate value, S is the
// D×D hidden state, K ∈ ℝ^{C×D}, V ∈ ℝ^{C×D}, and g ∈ ℝ^C is the per-token
// gate.
//
// ── Cube phase (two GEMMs per chunk, sequentially): ──────────────────────
//   1. WS = W @ S       project current state through W (wy_fast output)
//      W ∈ ℝ^{C×D}, S ∈ ℝ^{D×D}  →  WS ∈ ℝ^{C×D}
//   2. KV = K^T @ V     outer product of keys and values (transpose_A!)
//      K stored as D×C, V ∈ ℝ^{C×D}  →  KV ∈ ℝ^{D×D}
//
// ── Vec phase (two sub-blocks handle upper/lower C/2 rows): ─────────────
//   For each chunk:
//     1. Load K, G (pre-transposed), U (from wy_fast)
//     2. Compute coeff[i] = exp(g[i] - g[valid-1])  — time-decay scaling
//        Uses TROWEXPAND to broadcast coefficients across D columns
//     3. Scale K: K_scaled[i,:] = K[i,:] * coeff[i]
//     4. Load WS from Cube workspace, compute V_new = U - WS (residual)
//     5. Store V_new and K_scaled to workspace for Cube's next iteration
//     6. Update state: S = exp(g_last) * S + KV (from Cube workspace)
//     7. Store final state FS after last chunk
//
// Cross-core sync: Cube→Vec flags for WS/KV ready, Vec→Cube flags for
// K/S ready.
//
// Inputs:
//   K  [total_tokens, Hg, D] ComputeT   — keys (BSND layout; GQA/MQA group
//   heads) W  [total_tokens, H, D]  ComputeT   — wy_fast output (BSND layout)
//   U  [total_tokens, H, D]  ComputeT   — values pre-residual (BSND layout)
//   G  [H, total_tokens]     float  — pre-transposed cumulative gates
//   S  [total_chunks, H, D, D] ComputeT — per-chunk state snapshots (output)
//   V  [total_tokens, H, D]  ComputeT   — residual-corrected values (output)
//   FS [batch, H, D, D]      ComputeT   — final state per sequence (output)
//   H0 [batch, H, D, D]      ComputeT   — optional initial state per sequence
//   workspace [per-core scratch]     — Cube↔Vec communication buffer
//
// NPU memory hierarchy:
//   GM → L1 (Cube-accessible) → L0A/L0B/L0C (Cube GEMM registers)
//   GM → UB (Vec-accessible, on-chip SRAM)
//   Cross-core sync via FFTS (Fast Fine-grained Task Synchronization)
//
// ── PTO / NPU Primer ──────────────────────────────────────────────────
// This is the most complex kernel in the GDN suite. It implements the
// recurrent state update, requiring sequential chunk processing (chunks
// within a sequence CANNOT be parallelized — each depends on the previous).
//
// Key PTO APIs (numpy/torch equivalents):
//   TLOAD(dst, gm)          — dst = gm_data        (DMA: GM→L1 or GM→UB)
//   TSTORE(gm, src)         — gm_data = src        (DMA: UB/L0C→GM)
//   TASSIGN(tile, addr)     — tile = memory[addr]   (bind tile to buffer
//   address) TCVT(dst, src, mode)    — converts between float and ComputeT
//   TMOV(dst, src)          — dst = src.clone()
//   TADD(d, a, b)           — d = a + b
//   TSUB(d, a, b)           — d = a - b
//   TMUL(d, a, b)           — d = a * b
//   TMULS(d, s, scalar)     — d = s * scalar       (scalar multiply)
//   TADDS(d, s, scalar)     — d = s + scalar       (scalar add)
//   TEXP(d, s)              — d = torch.exp(s)
//   TEXPANDS(tile, scalar)  — tile[:] = scalar     (fill with constant)
//   TROWEXPAND(2d, col)     — 2d[i,j] = col[i]    (broadcast col across row
//   dim) TFILLPAD(dst, src)      — zero-fill L1 tile padding (for tail chunks)
//   TEXTRACT(l0, l1, r, c)  — L1 sub-tile → L0A/L0B
//   TRESHAPE(zn, nz)        — reinterpret layout NZ↔ZN (logical transpose,
//   free) TMATMUL(C, A, B)        — C = A @ B (Cube GEMM, ComputeT inputs →
//   FP32 accum) set_flag/wait_flag      — pipe sync within same core
//   ffts_cross_core_sync    — cross-core signal Cube↔Vec
//   wait_flag_dev(flag)     — wait for cross-core signal
//   GetValue(idx)           — read a single scalar from a UB tile (slow, use
//   sparingly)
//
// ── Workspace memory layout (shared between Cube and Vec via GM) ──────
// The workspace is field-major so adjacent cores do not all hit GM with the
// same 128 KiB stride:
//   WS_WS [C×D]:  Cube writes WS = W @ S here → Vec reads it
//   WS_K  [D×C]:  Vec writes K_scaled here → Cube reads it for KV = K^T @ V
//   WS_S  [D×D]:  Vec writes current state S here → Cube reads it for GEMM 1
//   WS_KV [D×D]:  Cube writes KV = K^T @ V here → Vec reads it to update S
//
// Data flow per chunk (think of it as a ping-pong between Cube and Vec):
//   Vec: write S₀ to WS_S → signal Cube (flag 3)
//   Cube: read S from WS_S, load W → compute WS = W@S → write WS_WS → signal
//   Vec (flag 0) Vec: read WS, compute V_new = U - WS, compute K_scaled → write
//   WS_K → signal Cube (flag 1) Cube: read K from WS_K, load V → compute KV =
//   K^T@V → write WS_KV → signal Vec (flag 2) Vec: read KV, update S =
//   exp(g_last)*S + KV → write S to WS_S → signal Cube (flag 3)
//   ... repeat for next chunk ...
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

#ifndef GDN_PAIR_DEBUG_STAGE
#define GDN_PAIR_DEBUG_STAGE 0
#endif

#ifdef __CCE_AICORE__

namespace {

#if defined(GDN_A5_VECTOR_KERNEL) && defined(MEGA_CHUNK_GDN_A5_H_STATE_REGBASE)
inline constexpr AscendC::MicroAPI::CastTrait H_STATE_CAST_B16_TO_B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

inline constexpr AscendC::MicroAPI::CastTrait H_STATE_CAST_B32_TO_B16_RINT = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

// Keep the recurrent state in FP32 while producing the ComputeT snapshot
// consumed by the next Cube iteration.  The old A5 path issued one PTO op per
// row to avoid the stale-row behavior of a 64x128 aggregate tile.  RegBase
// accesses each vector register explicitly, preserving the same FP32 mul/add
// order without hundreds of per-row launches.
template <typename T>
static __simd_vf__ inline void UpdateHStateRegbase(__ubuf__ float* state,
                                                   __ubuf__ T* kv,
                                                   __ubuf__ T* state_snapshot,
                                                   float decay,
                                                   uint32_t count) {
  __VEC_SCOPE__ {
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>,
                  "H RegBase state update supports FP16/BF16 snapshots.");
    constexpr uint32_t ElementsPerRegister =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::MicroAPI::RegTensor<float> state0;
    AscendC::MicroAPI::RegTensor<float> state1;
    AscendC::MicroAPI::RegTensor<float> kv0;
    AscendC::MicroAPI::RegTensor<float> kv1;
    AscendC::MicroAPI::RegTensor<T> kv_raw0;
    AscendC::MicroAPI::RegTensor<T> kv_raw1;
    AscendC::MicroAPI::RegTensor<T> snapshot0;
    AscendC::MicroAPI::RegTensor<T> snapshot1;
    AscendC::MicroAPI::MaskReg mask0;
    AscendC::MicroAPI::MaskReg mask1;

    uint32_t remaining = count;
    uint32_t offset = 0;
    while (remaining > ElementsPerRegister) {
      mask0 = AscendC::MicroAPI::UpdateMask<float>(remaining);
      mask1 = AscendC::MicroAPI::UpdateMask<float>(remaining);
      AscendC::MicroAPI::DataCopy<T,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          kv_raw0, kv + offset);
      AscendC::MicroAPI::DataCopy<T,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          kv_raw1, kv + offset + ElementsPerRegister);
      AscendC::MicroAPI::Cast<float, T, H_STATE_CAST_B16_TO_B32>(
          kv0, kv_raw0, mask0);
      AscendC::MicroAPI::Cast<float, T, H_STATE_CAST_B16_TO_B32>(
          kv1, kv_raw1, mask1);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::LoadDist::DIST_NORM>(
          state0, state + offset);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::LoadDist::DIST_NORM>(
          state1, state + offset + ElementsPerRegister);
      AscendC::MicroAPI::Muls(state0, state0, decay, mask0);
      AscendC::MicroAPI::Muls(state1, state1, decay, mask1);
      AscendC::MicroAPI::Add(state0, state0, kv0, mask0);
      AscendC::MicroAPI::Add(state1, state1, kv1, mask1);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
          state + offset, state0, mask0);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
          state + offset + ElementsPerRegister, state1, mask1);
      AscendC::MicroAPI::Cast<T, float, H_STATE_CAST_B32_TO_B16_RINT>(
          snapshot0, state0, mask0);
      AscendC::MicroAPI::Cast<T, float, H_STATE_CAST_B32_TO_B16_RINT>(
          snapshot1, state1, mask1);
      AscendC::MicroAPI::DataCopy<T,
                                  AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
          state_snapshot + offset, snapshot0, mask0);
      AscendC::MicroAPI::DataCopy<T,
                                  AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
          state_snapshot + offset + ElementsPerRegister, snapshot1, mask1);
      offset += 2 * ElementsPerRegister;
    }
    if (remaining > 0) {
      mask0 = AscendC::MicroAPI::UpdateMask<float>(remaining);
      AscendC::MicroAPI::DataCopy<T,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          kv_raw0, kv + offset);
      AscendC::MicroAPI::Cast<float, T, H_STATE_CAST_B16_TO_B32>(
          kv0, kv_raw0, mask0);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::LoadDist::DIST_NORM>(
          state0, state + offset);
      AscendC::MicroAPI::Muls(state0, state0, decay, mask0);
      AscendC::MicroAPI::Add(state0, state0, kv0, mask0);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
          state + offset, state0, mask0);
      AscendC::MicroAPI::Cast<T, float, H_STATE_CAST_B32_TO_B16_RINT>(
          snapshot0, state0, mask0);
      AscendC::MicroAPI::DataCopy<T,
                                  AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
          state_snapshot + offset, snapshot0, mask0);
    }
  }
}

#if defined(MEGA_CHUNK_GDN_A5_H_STATE_REGBASE4)
// The A5 production shape always updates 64x128 FP32 state elements.  Process
// four full vector registers per loop so the fixed all-lane mask and loop
// control are amortized without changing the arithmetic or snapshot rounding.
template <typename T, uint32_t Count>
static __simd_vf__ inline void UpdateHStateRegbase4(__ubuf__ float* state,
                                                    __ubuf__ T* kv,
                                                    __ubuf__ T* state_snapshot,
                                                    float decay) {
  __VEC_SCOPE__ {
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>,
                  "H RegBase state update supports FP16/BF16 snapshots.");
    constexpr uint32_t ElementsPerRegister =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint32_t ElementsPerIteration = 4 * ElementsPerRegister;
    static_assert(Count % ElementsPerIteration == 0,
                  "four-register H update requires full vector groups");
    AscendC::MicroAPI::RegTensor<float> state0;
    AscendC::MicroAPI::RegTensor<float> state1;
    AscendC::MicroAPI::RegTensor<float> state2;
    AscendC::MicroAPI::RegTensor<float> state3;
    AscendC::MicroAPI::RegTensor<float> kv0;
    AscendC::MicroAPI::RegTensor<float> kv1;
    AscendC::MicroAPI::RegTensor<float> kv2;
    AscendC::MicroAPI::RegTensor<float> kv3;
    AscendC::MicroAPI::RegTensor<T> kv_raw0;
    AscendC::MicroAPI::RegTensor<T> kv_raw1;
    AscendC::MicroAPI::RegTensor<T> kv_raw2;
    AscendC::MicroAPI::RegTensor<T> kv_raw3;
    AscendC::MicroAPI::RegTensor<T> snapshot0;
    AscendC::MicroAPI::RegTensor<T> snapshot1;
    AscendC::MicroAPI::RegTensor<T> snapshot2;
    AscendC::MicroAPI::RegTensor<T> snapshot3;
    AscendC::MicroAPI::MaskReg full_mask =
        AscendC::MicroAPI::CreateMask<float,
                                      AscendC::MicroAPI::MaskPattern::ALL>();

    for (uint16_t offset = 0; offset < Count;
         offset = static_cast<uint16_t>(offset + ElementsPerIteration)) {
      AscendC::MicroAPI::DataCopy<T,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          kv_raw0, kv + offset);
      AscendC::MicroAPI::DataCopy<T,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          kv_raw1, kv + offset + ElementsPerRegister);
      AscendC::MicroAPI::DataCopy<T,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          kv_raw2, kv + offset + 2 * ElementsPerRegister);
      AscendC::MicroAPI::DataCopy<T,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          kv_raw3, kv + offset + 3 * ElementsPerRegister);
      AscendC::MicroAPI::Cast<float, T, H_STATE_CAST_B16_TO_B32>(
          kv0, kv_raw0, full_mask);
      AscendC::MicroAPI::Cast<float, T, H_STATE_CAST_B16_TO_B32>(
          kv1, kv_raw1, full_mask);
      AscendC::MicroAPI::Cast<float, T, H_STATE_CAST_B16_TO_B32>(
          kv2, kv_raw2, full_mask);
      AscendC::MicroAPI::Cast<float, T, H_STATE_CAST_B16_TO_B32>(
          kv3, kv_raw3, full_mask);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::LoadDist::DIST_NORM>(
          state0, state + offset);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::LoadDist::DIST_NORM>(
          state1, state + offset + ElementsPerRegister);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::LoadDist::DIST_NORM>(
          state2, state + offset + 2 * ElementsPerRegister);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::LoadDist::DIST_NORM>(
          state3, state + offset + 3 * ElementsPerRegister);
      AscendC::MicroAPI::Muls(state0, state0, decay, full_mask);
      AscendC::MicroAPI::Muls(state1, state1, decay, full_mask);
      AscendC::MicroAPI::Muls(state2, state2, decay, full_mask);
      AscendC::MicroAPI::Muls(state3, state3, decay, full_mask);
      AscendC::MicroAPI::Add(state0, state0, kv0, full_mask);
      AscendC::MicroAPI::Add(state1, state1, kv1, full_mask);
      AscendC::MicroAPI::Add(state2, state2, kv2, full_mask);
      AscendC::MicroAPI::Add(state3, state3, kv3, full_mask);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
          state + offset, state0, full_mask);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
          state + offset + ElementsPerRegister, state1, full_mask);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
          state + offset + 2 * ElementsPerRegister, state2, full_mask);
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
          state + offset + 3 * ElementsPerRegister, state3, full_mask);
      AscendC::MicroAPI::Cast<T, float, H_STATE_CAST_B32_TO_B16_RINT>(
          snapshot0, state0, full_mask);
      AscendC::MicroAPI::Cast<T, float, H_STATE_CAST_B32_TO_B16_RINT>(
          snapshot1, state1, full_mask);
      AscendC::MicroAPI::Cast<T, float, H_STATE_CAST_B32_TO_B16_RINT>(
          snapshot2, state2, full_mask);
      AscendC::MicroAPI::Cast<T, float, H_STATE_CAST_B32_TO_B16_RINT>(
          snapshot3, state3, full_mask);
      AscendC::MicroAPI::DataCopy<T,
                                  AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
          state_snapshot + offset, snapshot0, full_mask);
      AscendC::MicroAPI::DataCopy<T,
                                  AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
          state_snapshot + offset + ElementsPerRegister, snapshot1, full_mask);
      AscendC::MicroAPI::DataCopy<T,
                                  AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
          state_snapshot + offset + 2 * ElementsPerRegister,
          snapshot2,
          full_mask);
      AscendC::MicroAPI::DataCopy<T,
                                  AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
          state_snapshot + offset + 3 * ElementsPerRegister,
          snapshot3,
          full_mask);
    }
  }
}

#endif
#endif

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

template <typename T,
          int32_t Rows,
          int32_t Cols,
          int32_t RowValid = Rows,
          int32_t ColValid = Cols>
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
          int32_t Rows,
          int32_t Cols,
          int32_t RowValid = Rows,
          int32_t ColValid = Cols>
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
          int32_t Rows,
          int32_t Cols,
          int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL0A = pto::TileLeft<T, Rows, Cols, RowValid, ColValid>;

template <typename T,
          int32_t Rows,
          int32_t Cols,
          int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL0B = pto::TileRight<T, Rows, Cols, RowValid, ColValid>;

template <typename T,
          int32_t Rows,
          int32_t Cols,
          int32_t RowValid = Rows,
          int32_t ColValid = Cols,
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
          int32_t Rows,
          int32_t Cols,
          int32_t RowValid = Rows,
          int32_t ColValid = Cols,
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

// PTO cheat sheet for the recurrent kernel:
//   - `GlobalTensor<T>` is a GM tensor view with explicit runtime shape/stride.
//   - `Tile<..., Mat, ...>` lives in L1 and feeds Cube matmul instructions.
//   - `Tile<..., Vec, ...>` lives in UB for elementwise vector work.
//   - `TileAcc<T, ...>` is a Cube accumulator tile.
//   - `TLOAD` / `TSTORE` are DMA copies between GM and on-chip memory.
//   - `TROWEXPAND` broadcasts a column vector across the feature dimension.
//   - `TFILLPAD(_INPLACE)` zero-pads tail rows so full-tile code can still run.

template <typename T1,
          typename T2,
          uint32_t M,
          uint32_t N,
          uint32_t K,
          uint32_t validM = M,
          uint32_t validN = N,
          uint32_t validK = K,
          uint32_t K_tail = K,
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
  // PTO exposes the L1/L0 staging explicitly, so this stays as a tiny file-
  // local helper instead of a shared wrapper.
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

#if defined(__DAV_C220_CUBE__)
template <int32_t D,
          int32_t C,
          int32_t FlagBase,
          bool WaitStateReady = true,
          bool PublishReady = true>
AICORE PTO_INLINE void paired_cube_ws(__gm__ ComputeT* W_handle,
                                      __gm__ ComputeT* workspace_handle,
                                      int64_t head,
                                      int64_t chunk_start,
                                      int32_t valid,
                                      int32_t H,
                                      int64_t ws_ws_base,
                                      int64_t ws_s_base,
                                      TileMatL1<ComputeT, D, D, D, D>& s_l1,
                                      TileMatL1<ComputeT, C, D, C, D>& w_l1,
                                      TileAcc<float, C, D, C, D>& ws_l0) {
  if constexpr (WaitStateReady) {
    wait_flag_dev(FlagBase + 3);
  }

  const int64_t w_offset = (chunk_start * H + head) * D;
  {
    GmShape2D w_shape(valid, D);
    GmStride2D w_stride(H * D);
    GmTensor2D<ComputeT> w_global(W_handle + w_offset, w_shape, w_stride);
    TLOAD(w_l1, w_global);
    if (valid != C) {
      TFILLPAD(w_l1, w_l1);
    }
  }
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
  {
    GmShape2D s_shape(D, D);
    GmStride2D s_stride(D);
    GmTensor2D<ComputeT> s_global(
        workspace_handle + ws_s_base, s_shape, s_stride);
    TLOAD(s_l1, s_global);
  }
#ifdef MEGA_STOP_AFTER_H
  if constexpr (GDN_PAIR_DEBUG_STAGE == 8) {
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (FlagBase << 8));
    return;
  }
#endif

#ifdef MEGA_STOP_AFTER_H
  if constexpr (GDN_PAIR_DEBUG_STAGE == 14 || GDN_PAIR_DEBUG_STAGE == 15) {
    if constexpr (GDN_PAIR_DEBUG_STAGE == 14) {
      ffts_cross_core_sync(PIPE_MTE2, 1 | (2 << 4) | (FlagBase << 8));
      return;
    }
    TileMatL0A<ComputeT, C, D, C, D> w_l0a;
    TileMatL0B<ComputeT, D, D, D, D> s_l0b;
    TASSIGN(w_l0a, 0);
    TASSIGN(s_l0b, 0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TEXTRACT(w_l0a, w_l1, 0, 0);
    TEXTRACT(s_l0b, s_l1, 0, 0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    TMATMUL(ws_l0, w_l0a, s_l0b);
    ffts_cross_core_sync(PIPE_M, 1 | (2 << 4) | (FlagBase << 8));
    return;
  }
#endif

  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  gemm_v0<ComputeT, float, C, D, D, C, D, D, D, false, false>(
      w_l1, s_l1, ws_l0, true);
#ifdef MEGA_STOP_AFTER_H
  if constexpr (GDN_PAIR_DEBUG_STAGE == 9) {
    ffts_cross_core_sync(PIPE_M, 1 | (2 << 4) | (FlagBase << 8));
    return;
  }
#endif

  {
    GmShape2D ws_shape(C, D);
    GmStride2D ws_stride(D);
    GmTensor2D<ComputeT> ws_global(
        workspace_handle + ws_ws_base, ws_shape, ws_stride);
    DynAccTile<float, C, D> ws_store(C, D);
    TASSIGN(ws_store, 0);
    TSTORE(ws_global, ws_store);
  }
  if constexpr (PublishReady) {
    ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (FlagBase << 8));
  }
}

#include "mega_chunk_gdn/op_kernel/common/chunk_h_paired_cube.inc"
#endif

#if defined(__DAV_C220_VEC__)
template <int32_t D, int32_t C, int32_t FlagBase, bool PublishReady = true>
AICORE PTO_INLINE void paired_vec_init_zero(
    __gm__ ComputeT* S_handle,
    __gm__ ComputeT* workspace_handle,
    int64_t head,
    int64_t chunk_offset,
    int32_t H,
    int32_t vid,
    int64_t ws_s_base,
    TileUbDataND<float, C / 2, D, C / 2, D>& state_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D>& state_half) {
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t DD = D * D;
  TEXPANDS(state_ub, 0.0f);
  TCVT(state_half, state_ub, pto::RoundMode::CAST_NONE);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  {
    GmShape2D s_shape(HalfC, D);
    GmStride2D s_stride(D);
    GmTensor2D<ComputeT> s_global(
        workspace_handle + ws_s_base + vid * HalfC * D, s_shape, s_stride);
    TSTORE(s_global, state_half);
  }
  {
    const int64_t s_out_offset = (chunk_offset * H + head) * DD;
    GmShape2D s_out_shape(HalfC, D);
    GmStride2D s_out_stride(D);
    GmTensor2D<ComputeT> s_out_global(
        S_handle + s_out_offset + vid * HalfC * D, s_out_shape, s_out_stride);
    TSTORE(s_out_global, state_half);
  }
  if constexpr (PublishReady) {
    ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | ((FlagBase + 3) << 8));
  } else {
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  }
}

template <int32_t D,
          int32_t C,
          int32_t FlagBase,
          int32_t GUbAddr,
          int32_t CoeffUbAddr,
          int32_t UUbAddr>
AICORE PTO_INLINE void paired_vec_front(
    __gm__ ComputeT* K_handle,
    __gm__ ComputeT* U_handle,
    __gm__ float* G_handle,
    __gm__ ComputeT* V_handle,
    __gm__ ComputeT* workspace_handle,
    int64_t head,
    int64_t head_g,
    int64_t chunk_start,
    int64_t valid,
    int32_t ci,
    int32_t H,
    int32_t Hg,
    int64_t total_tokens,
    int32_t vid,
    int32_t v_stride,
    int64_t v_head_base,
    int64_t v_chunk_stride,
    int64_t ws_ws_base,
    int64_t ws_k_base,
    TileUbDataND<float, C / 2, D, C / 2, D>& state_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D, pto::PadValue::Zero>& k_ub_half,
    TileUbDataND<float, 1, C, 1, C, pto::PadValue::Zero>& g_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D, pto::PadValue::Zero>& u_ub_half,
    TileUbDataND<float, C / 2, D, C / 2, D>& k_ub,
    TileUbDataND<float, 1, 8, 1, 8>& g_last_tail_ub,
    TileUbDataND<float, 1, 64, 1, 64>& coeff_ub,
    TileUbDataND<float, C / 2, D, C / 2, D>& u_ub,
    TileUbDataND<float, C / 2, D, C / 2, D>& ws_ub) {
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t ExpTailElements = 8;
  const int32_t qkv_stride = H * D;
  const int32_t k_stride = Hg * D;
  int32_t valid_rows =
      static_cast<int32_t>(valid - static_cast<int64_t>(vid) * HalfC);
  if (valid_rows < 0) valid_rows = 0;
  if (valid_rows > HalfC) valid_rows = HalfC;

  const int64_t u_offset =
      (chunk_start * H + head) * D + vid * HalfC * qkv_stride;
  if (valid_rows > 0) {
    GmShape2D u_shape(valid_rows, D);
    GmStride2D u_stride(qkv_stride);
    GmTensor2D<ComputeT> u_global(U_handle + u_offset, u_shape, u_stride);
    TLOAD(u_ub_half, u_global);
  } else {
    TEXPANDS(u_ub, 0.0f);
    TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);
  }

  const int64_t k_offset =
      (chunk_start * Hg + head_g) * D + vid * HalfC * k_stride;
  if (valid_rows > 0) {
    GmShape2D k_shape(valid_rows, D);
    GmStride2D k_gm_stride(k_stride);
    GmTensor2D<ComputeT> k_global(K_handle + k_offset, k_shape, k_gm_stride);
    TLOAD(k_ub_half, k_global);
  } else {
    TEXPANDS(k_ub, 0.0f);
    TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);
  }

  {
    GmShape2D g_shape(1, static_cast<int32_t>(valid));
    GmStride2D g_stride(1);
    GmTensor2D<float> g_global(
        G_handle + head * total_tokens + chunk_start, g_shape, g_stride);
    TLOAD(g_ub, g_global);
  }

  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(k_ub, k_ub_half, pto::RoundMode::CAST_NONE);

  TileUbDataND<float, 1, 64, 1, 64> g_ub_temp;
  TASSIGN(g_ub_temp, GUbAddr + vid * 64 * sizeof(float));
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  const float g_last = g_ub.GetValue(static_cast<int32_t>(valid) - 1);
  TADDS(coeff_ub, g_ub_temp, -g_last);
  pipe_barrier(PIPE_V);
  TNEG(coeff_ub, coeff_ub);
  pipe_barrier(PIPE_V);
  TEXP(coeff_ub, coeff_ub);
  if (valid == C) {
    TEXP(g_last_tail_ub, g_last_tail_ub);
  } else {
    TEXP(g_ub, g_ub);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  TileUbDataDN<float, HalfC, 1, HalfC, 1> coeff_col_ub;
  TASSIGN(coeff_col_ub, CoeffUbAddr);
  TileUbDataND<float, HalfC, D, HalfC, D> coeff_2d_ub;
  TASSIGN(coeff_2d_ub, UUbAddr);
  TROWEXPAND(coeff_2d_ub, coeff_col_ub);
  pipe_barrier(PIPE_V);
  TMUL(k_ub, k_ub, coeff_2d_ub);
  pipe_barrier(PIPE_V);
  TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);
  TCVT(u_ub, u_ub_half, pto::RoundMode::CAST_NONE);

  wait_flag_dev(FlagBase);
  {
    GmShape2D ws_shape(HalfC, D);
    GmStride2D ws_stride(D);
    GmTensor2D<ComputeT> ws_global(
        workspace_handle + ws_ws_base + vid * HalfC * D, ws_shape, ws_stride);
    TLOAD(u_ub_half, ws_global);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(ws_ub, u_ub_half, pto::RoundMode::CAST_NONE);
  TSUB(u_ub, u_ub, ws_ub);
  TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);

  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  const int64_t v_offset = v_head_base +
                           static_cast<int64_t>(ci) * v_chunk_stride +
                           static_cast<int64_t>(vid) * HalfC * v_stride;
  if (valid_rows > 0) {
    GmShape2D v_shape(valid_rows, D);
    GmStride2D v_gm_stride(v_stride);
    GmTensor2D<ComputeT> v_global(V_handle + v_offset, v_shape, v_gm_stride);
    TSTORE(v_global, u_ub_half);
  }
  {
    GmShape2D k_shape(HalfC, D);
    GmStride2D k_ws_stride(D);
    GmTensor2D<ComputeT> k_global(
        workspace_handle + ws_k_base + vid * HalfC * D, k_shape, k_ws_stride);
    TSTORE(k_global, k_ub_half);
  }
  ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | ((FlagBase + 1) << 8));

  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  const float exp_g_last = valid == C
                               ? g_last_tail_ub.GetValue(ExpTailElements - 1)
                               : g_ub.GetValue(static_cast<int32_t>(valid) - 1);
  TMULS(state_ub, state_ub, exp_g_last);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
}

template <int32_t D, int32_t C, int32_t FlagBase>
AICORE PTO_INLINE void paired_vec_finish(
    __gm__ ComputeT* S_handle,
    __gm__ ComputeT* workspace_handle,
    __gm__ int32_t* h_o_ready_handle,
    int64_t head,
    int64_t chunk_offset,
    int32_t ci,
    int32_t num_chunks,
    int32_t H,
    int32_t vid,
    int64_t ws_s_base,
    int64_t ws_kv_base,
    bool emit_precomputed_qs,
    TileUbDataND<float, C / 2, D, C / 2, D>& state_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D>& state_half,
    TileUbDataND<float, C / 2, D, C / 2, D>& kv_ub,
    TileUbDataND<int32_t, 1, 8, 1, 8>& h_o_ready_ub) {
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t DD = D * D;
  constexpr int64_t H_O_READY_STRIDE = 16;
  wait_flag_dev(FlagBase + 2);
  if (emit_precomputed_qs && vid == 0) {
    const int64_t ready_offset = head * H_O_READY_STRIDE;
    h_o_ready_ub.SetValue(0, ci + 1);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
    GmShape2D ready_shape(1, 8);
    GmStride2D ready_stride(8);
    GmTensor2D<int32_t> ready_global(
        h_o_ready_handle + ready_offset, ready_shape, ready_stride);
    TSTORE(ready_global, h_o_ready_ub);
  }
  {
    GmShape2D kv_shape(HalfC, D);
    GmStride2D kv_stride(D);
    GmTensor2D<ComputeT> kv_global(
        workspace_handle + ws_kv_base + vid * HalfC * D, kv_shape, kv_stride);
    TLOAD(state_half, kv_global);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(kv_ub, state_half, pto::RoundMode::CAST_NONE);
  pipe_barrier(PIPE_ALL);
  TADD(state_ub, state_ub, kv_ub);
  TCVT(state_half, state_ub, pto::RoundMode::CAST_NONE);

  if (ci + 1 < num_chunks) {
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    {
      GmShape2D s_shape(HalfC, D);
      GmStride2D s_stride(D);
      GmTensor2D<ComputeT> s_global(
          workspace_handle + ws_s_base + vid * HalfC * D, s_shape, s_stride);
      TSTORE(s_global, state_half);
    }
    const int64_t s_out_offset = ((chunk_offset + ci + 1) * H + head) * DD;
    {
      GmShape2D s_out_shape(HalfC, D);
      GmStride2D s_out_stride(D);
      GmTensor2D<ComputeT> s_out_global(
          S_handle + s_out_offset + vid * HalfC * D, s_out_shape, s_out_stride);
      TSTORE(s_out_global, state_half);
    }
    ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | ((FlagBase + 3) << 8));
  }
}

template <int32_t D, int32_t C, bool StoreFinalStateCache>
AICORE PTO_INLINE void paired_vec_store_final(
    __gm__ ComputeT* FS_handle,
    __gm__ float* final_state_cache,
    __gm__ int32_t* state_indices,
    int64_t head,
    int32_t H,
    int32_t vid,
    int32_t state_index_stride,
    int64_t state_cache_slots,
    int64_t output_final_state,
    TileUbDataND<float, C / 2, D, C / 2, D>& state_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D>& state_half) {
  if (output_final_state == 0) return;
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t DD = D * D;
  TCVT(state_half, state_ub, pto::RoundMode::CAST_NONE);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  if constexpr (StoreFinalStateCache) {
    TCVT(state_ub, state_half, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    const int32_t state_index = state_indices[0] * state_index_stride;
    if (state_index >= 0 && state_index < state_cache_slots) {
      const int64_t cache_offset =
          (static_cast<int64_t>(state_index) * H + head) * DD + vid * HalfC * D;
      GmShape2D cache_shape(HalfC, D);
      GmStride2D cache_stride(D);
      GmTensor2D<float> cache_global(
          final_state_cache + cache_offset, cache_shape, cache_stride);
      TSTORE(cache_global, state_ub);
    }
  } else {
    const int64_t fs_offset = head * DD + vid * HalfC * D;
    GmShape2D fs_shape(HalfC, D);
    GmStride2D fs_stride(D);
    GmTensor2D<ComputeT> fs_global(FS_handle + fs_offset, fs_shape, fs_stride);
    TSTORE(fs_global, state_half);
  }
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}
#endif

}  // namespace

#endif

template <int32_t HiddenSize,
          int32_t ChunkSize,
          bool StoreFinalStateCache = false,
          bool LoadInitialStateCache = false>
AICORE void chunk_h_kernel(__gm__ ComputeT* Q_handle,
                           __gm__ ComputeT* K_handle,
                           __gm__ ComputeT* W_handle,
                           __gm__ ComputeT* U_handle,
                           __gm__ float* G_handle,
                           __gm__ ComputeT* S_handle,
                           __gm__ ComputeT* V_handle,
                           __gm__ ComputeT* FS_handle,
                           __gm__ ComputeT* H0_handle,
                           int64_t has_initial_state,
                           int64_t output_final_state,
                           __gm__ ComputeT* workspace_handle,
                           __gm__ int32_t* cu_seqlens,
                           int64_t batch_size,
                           int64_t seq_len,
                           int64_t total_tokens,
                           uint32_t num_heads,
                           uint32_t num_key_heads,
                           uint32_t precompute_qs,
                           __gm__ int32_t* h_o_ready_handle,
                           uint64_t ffts_addr,
                           __gm__ float* initial_state_cache,
                           __gm__ int32_t* initial_state_indices,
                           __gm__ float* final_state_cache,
                           __gm__ int32_t* state_indices,
                           int32_t state_index_stride,
                           int64_t state_cache_slots) {
  // chunk_h advances the recurrent hidden state chunk by chunk:
  //   ws_i      = W_i @ S_i
  //   v_i_new   = U_i - ws_i
  //   k_i_tilde = exp(g_last - g_i) * K_i
  //   S_{i+1}   = exp(g_last) * S_i + k_i_tilde^T @ v_i_new.
  //
  // Shapes for one (sequence, head, chunk):
  //   W_i, U_i, K_i, V_i_new : [valid, D]
  //   S_i, S_{i+1}           : [D, D]
  //
  // PyTorch / NumPy sketch:
  //   ws = W_i @ S_i
  //   v_new = U_i - ws
  //   decay = exp(g_last - g_i)[:, None]
  //   k_tilde = decay * K_i
  //   kv = k_tilde.T @ v_new
  //   S = exp(g_last) * S + kv
  //
  // PTO split:
  //   Cube forms the two matmuls (`W_i @ S_i` and `K_i^T @ V_i_new`).
  //   Vec does the elementwise gating/decay and carries the running state.
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  gdn_sync::InitAddress(ffts_addr);

  constexpr int32_t D = HiddenSize;
  constexpr int32_t C = ChunkSize;
  const int32_t H = static_cast<int32_t>(num_heads);
  const int32_t Hg = static_cast<int32_t>(num_key_heads);
  if (H <= 0 || Hg <= 0 || (H % Hg) != 0) return;
  const int32_t GROUP = H / Hg;
  constexpr int32_t HalfC = C / 2;
  const int32_t BSND_QKV_STRIDE = H * D;
  const int32_t BSND_K_STRIDE = Hg * D;
  constexpr int32_t DD = D * D;
#if defined(GDN_A5_CUBE_KERNEL)
  // C310 exposes 128 KiB of L0C; 0x20000 is the first byte past it.  The two
  // GEMMs are separated by a Cube->Vector->Cube handshake, so reuse slot 0.
  constexpr int32_t SecondL0CAddr = 0;
#else
  constexpr int32_t SecondL0CAddr = C * D * sizeof(float);
#endif

  static_assert(GDN_H_WORKSPACE_PAD_BYTES % sizeof(ComputeT) == 0);
  constexpr int32_t WS_FIELD_STRIDE =
      DD + GDN_H_WORKSPACE_PAD_BYTES / sizeof(ComputeT);

  TileMatL1<ComputeT, D, D, D, D> s_l1;
  TASSIGN(s_l1, 0);
  TileMatL1<ComputeT, C, D, C, D> w_l1;
  TASSIGN(w_l1, D * D * sizeof(ComputeT));
  TileMatL1<ComputeT, C, D, C, D> q_l1;
  TASSIGN(q_l1, D * D * sizeof(ComputeT));
  TileAcc<float, C, D, C, D> ws_l0;
  TASSIGN(ws_l0, 0);
  TileMatL1<ComputeT, D, C, D, C> k_l1;
  TASSIGN(k_l1, (DD + C * D) * sizeof(ComputeT));
  TileMatL1<ComputeT, C, D, C, D> v_l1;
  TASSIGN(v_l1, (DD + C * D + D * C) * sizeof(ComputeT));
  TileAcc<float, D, D, D, D> kv_l0;
  TASSIGN(kv_l0, SecondL0CAddr);

  constexpr int32_t G_BLOCK_UB = 0;
  // Leading UB scratch: legacy kernels used ``C * H * sizeof(float)``, which
  // overflows UB when Keep the same slack as the historical H=16 build (8192
  // bytes).
  constexpr int32_t ZERO_UB =
      ChunkSize * 16 * static_cast<int32_t>(sizeof(float));
  constexpr int32_t S_UB = ZERO_UB + 64 * sizeof(float);
  constexpr int32_t K_UB_HALF = S_UB + HalfC * D * sizeof(float);
  constexpr int32_t G_UB = K_UB_HALF + HalfC * D * sizeof(ComputeT);
  constexpr int32_t U_UB_HALF = G_UB + C * sizeof(float);
  constexpr int32_t K_UB = U_UB_HALF + HalfC * D * sizeof(ComputeT);
  constexpr int32_t G_V_UB = K_UB + HalfC * D * sizeof(float);
  constexpr int32_t COEFF_UB = G_V_UB + 64 * sizeof(float);
  constexpr int32_t U_UB = COEFF_UB + 64 * sizeof(float);
  // k_ub is dead after k_tilde is rounded to BF16. Reuse that 32 KiB region
  // for W@S instead of keeping a second full FP32 tile live.
  constexpr int32_t WS_UB = K_UB;
  constexpr int32_t KV_UB = U_UB_HALF;
  constexpr int32_t S_UB_HALF = U_UB + HalfC * D * sizeof(float);
  constexpr int32_t S_ALT_UB = S_UB_HALF + HalfC * D * sizeof(ComputeT);
#if defined(MEGA_CHUNK_GDN_A5_H_WS_DIRECT) || \
    defined(MEGA_CHUNK_GDN_A5_H_KV_DIRECT)
  static_assert(C == 128 && D == 128,
                "H WS direct experiment requires C128/D128");
  constexpr int32_t DirectWsUbAddr = 224 * 1024;
  static_assert(S_ALT_UB + HalfC * D * sizeof(float) <= DirectWsUbAddr,
                "H WS direct slot overlaps live state scratch");
  static_assert(DirectWsUbAddr + HalfC * D * sizeof(float) <= 256 * 1024,
                "H WS direct slot exceeds A5 UB");
#endif

  TileUbDataND<float, HalfC, D, HalfC, D> s_ub;
  TASSIGN(s_ub, S_UB);
  TileUbDataND<ComputeT, HalfC, D, HalfC, D, pto::PadValue::Zero> k_ub_half;
  TASSIGN(k_ub_half, K_UB_HALF);
  TileUbDataND<float, 1, C, 1, C, pto::PadValue::Zero> g_ub;
  TASSIGN(g_ub, G_UB);
  TileUbDataND<ComputeT, HalfC, D, HalfC, D> s_ub_half;
  TASSIGN(s_ub_half, S_UB_HALF);
  TileUbDataND<ComputeT, HalfC, D, HalfC, D, pto::PadValue::Zero> u_ub_half;
  TASSIGN(u_ub_half, U_UB_HALF);
  TileUbDataND<float, HalfC, D, HalfC, D> k_ub;
  TASSIGN(k_ub, K_UB);
  constexpr int32_t ExpTailElements = 8;
  TileUbDataND<float, 1, ExpTailElements, 1, ExpTailElements> g_last_tail_ub;
  TASSIGN(g_last_tail_ub,
          G_UB + (C - ExpTailElements) * static_cast<int32_t>(sizeof(float)));
  TileUbDataND<float, 1, 64, 1, 64> coeff_ub;
  TASSIGN(coeff_ub, COEFF_UB);
  TileUbDataND<float, HalfC, D, HalfC, D> u_ub;
  TASSIGN(u_ub, U_UB);
  TileUbDataND<float, HalfC, D, HalfC, D> ws_ub;
  TASSIGN(ws_ub, WS_UB);
  TileUbDataND<float, HalfC, D, HalfC, D> kv_ub;
  TASSIGN(kv_ub, KV_UB);
  TileUbDataND<float, HalfC, D, HalfC, D> s_alt_ub;
  TASSIGN(s_alt_ub, S_ALT_UB);

  auto vid = get_subblockid();
  constexpr int64_t H_O_READY_STRIDE = 16;
  TileUbDataND<int32_t, 1, 8, 1, 8> h_o_ready_ub;
  TASSIGN(h_o_ready_ub, ZERO_UB);
  AscendC::GlobalTensor<int32_t> h_o_ready_gm;
  h_o_ready_gm.SetGlobalBuffer(h_o_ready_handle);

#if defined(GDN_A5_VECTOR_KERNEL) && \
    defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
  if constexpr (LoadInitialStateCache) {
    if (has_initial_state != 0) {
      // Acquire both the slot metadata and FP32 payload once per AIV before
      // any work item can dereference an index cached by an earlier launch.
      dcci((__gm__ void*)0, ENTIRE_DATA_CACHE);
      dsb(DSB_DDR);
    }
  }
#endif

  int64_t num_seqs = batch_size;
  int64_t total_work = num_seqs * H;
  const int64_t paired_core_count = total_work - block_num;
  const int64_t dense_chunk_count = total_tokens / C;
  constexpr bool use_paired_head_pipeline = false;

#if defined(__DAV_C220_CUBE__)
  if (use_paired_head_pipeline && cid < paired_core_count) {
    const int64_t head0 = cid;
    const int64_t head1 = block_num + cid;
    const int64_t head_g0 = head0 / GROUP;
    const int64_t head_g1 = head1 / GROUP;
    const int64_t bos = static_cast<int64_t>(cu_seqlens[0]);
    const int64_t slen = static_cast<int64_t>(cu_seqlens[1]) - bos;
    const int32_t num_chunks = static_cast<int32_t>(slen / C);
    const int64_t ws_core_offset = static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
    const int64_t ws_field_span =
        static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
    const int64_t ws_slot_span = 4 * ws_field_span;
    const int64_t ws_ws_base0 = ws_core_offset;
    const int64_t ws_k_base0 = ws_field_span + ws_core_offset;
    const int64_t ws_s_base0 = 2 * ws_field_span + ws_core_offset;
    const int64_t ws_kv_base0 = 3 * ws_field_span + ws_core_offset;
    const int64_t ws_ws_base1 = ws_slot_span + ws_ws_base0;
    const int64_t ws_k_base1 = ws_slot_span + ws_k_base0;
    const int64_t ws_s_base1 = ws_slot_span + ws_s_base0;
    const int64_t ws_kv_base1 = ws_slot_span + ws_kv_base0;
    const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;
    const int64_t v_head_base0 = head0 * static_cast<int64_t>(C) * D;
    const int64_t v_head_base1 = head1 * static_cast<int64_t>(C) * D;

    wait_flag_dev(3);
#ifdef MEGA_STOP_AFTER_H
    if constexpr (GDN_PAIR_DEBUG_STAGE == 1) return;
    if constexpr (GDN_PAIR_DEBUG_STAGE == 10) {
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));
      return;
    }
    if constexpr (GDN_PAIR_DEBUG_STAGE == 12) {
      const int64_t w_offset = (bos * H + head0) * D;
      GmShape2D w_shape(C, D);
      GmStride2D w_stride(H * D);
      GmTensor2D<ComputeT> w_global(W_handle + w_offset, w_shape, w_stride);
      DynMatL1<ComputeT, C, D> w_l1_load(C, D);
      TASSIGN(w_l1_load, D * D * static_cast<int32_t>(sizeof(ComputeT)));
      TLOAD(w_l1_load, w_global);
      ffts_cross_core_sync(PIPE_MTE2, 1 | (2 << 4) | (0 << 8));
      return;
    }
    if constexpr (GDN_PAIR_DEBUG_STAGE == 13) {
      const int64_t w_offset = (bos * H + head0) * D;
      GmShape2D w_shape(C, D);
      GmStride2D w_stride(H * D);
      GmTensor2D<ComputeT> w_global(W_handle + w_offset, w_shape, w_stride);
      DynMatL1<ComputeT, C, D> w_l1_load(C, D);
      TASSIGN(w_l1_load, D * D * static_cast<int32_t>(sizeof(ComputeT)));
      TLOAD(w_l1_load, w_global);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);

      GmShape2D s_shape(D, D);
      GmStride2D s_stride(D);
      GmTensor2D<ComputeT> s_global(
          workspace_handle + ws_s_base0, s_shape, s_stride);
      DynMatL1<ComputeT, D, D> s_l1_load(D, D);
      TASSIGN(s_l1_load, 0);
      TLOAD(s_l1_load, s_global);
      ffts_cross_core_sync(PIPE_MTE2, 1 | (2 << 4) | (0 << 8));
      return;
    }
    if constexpr (GDN_PAIR_DEBUG_STAGE == 15) {
      const int64_t w_offset = (bos * H + head0) * D;
      GmShape2D w_shape(C, D);
      GmStride2D w_stride(H * D);
      GmTensor2D<ComputeT> w_global(W_handle + w_offset, w_shape, w_stride);
      DynMatL1<ComputeT, C, D> w_l1_load(C, D);
      TASSIGN(w_l1_load, D * D * static_cast<int32_t>(sizeof(ComputeT)));
      TLOAD(w_l1_load, w_global);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);

      GmShape2D s_shape(D, D);
      GmStride2D s_stride(D);
      GmTensor2D<ComputeT> s_global(
          workspace_handle + ws_s_base0, s_shape, s_stride);
      DynMatL1<ComputeT, D, D> s_l1_load(D, D);
      TASSIGN(s_l1_load, 0);
      TLOAD(s_l1_load, s_global);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);

      TileMatL0A<ComputeT, C, D, C, D> w_l0a;
      TileMatL0B<ComputeT, D, D, D, D> s_l0b;
      TASSIGN(w_l0a, 0);
      TASSIGN(s_l0b, 0);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      ffts_cross_core_sync(PIPE_MTE2, 1 | (2 << 4) | (0 << 8));
      return;
    }
#endif
    for (int32_t ci = 0; ci < num_chunks; ++ci) {
      const int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      paired_cube_ws<D, C, 0, false, true>(W_handle,
                                           workspace_handle,
                                           head0,
                                           chunk_start,
                                           C,
                                           H,
                                           ws_ws_base0,
                                           ws_s_base0,
                                           s_l1,
                                           w_l1,
                                           ws_l0);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 2 || GDN_PAIR_DEBUG_STAGE == 8 ||
                    GDN_PAIR_DEBUG_STAGE == 9 || GDN_PAIR_DEBUG_STAGE == 11 ||
                    GDN_PAIR_DEBUG_STAGE == 14 || GDN_PAIR_DEBUG_STAGE == 15)
        return;
#endif
      paired_cube_qs<D, C>(Q_handle,
                           S_handle,
                           head0,
                           head_g0,
                           chunk_start,
                           0,
                           ci,
                           C,
                           H,
                           Hg,
                           s_l1,
                           q_l1,
                           ws_l0);

      paired_cube_ws<D, C, 4, false, false>(W_handle,
                                            workspace_handle,
                                            head1,
                                            chunk_start,
                                            C,
                                            H,
                                            ws_ws_base1,
                                            ws_s_base1,
                                            s_l1,
                                            w_l1,
                                            ws_l0);
      wait_flag_dev(1);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 3) return;
#endif
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (4 << 8));
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 4) return;
#endif
      paired_cube_qs<D, C>(Q_handle,
                           S_handle,
                           head1,
                           head_g1,
                           chunk_start,
                           0,
                           ci,
                           C,
                           H,
                           Hg,
                           s_l1,
                           q_l1,
                           ws_l0);

      paired_cube_kv<D, C, 0, false, false>(
          V_handle,
          workspace_handle,
          C,
          D,
          v_head_base0 + static_cast<int64_t>(ci) * v_chunk_stride,
          ws_k_base0,
          ws_kv_base0,
          k_l1,
          v_l1,
          kv_l0);
      wait_flag_dev(5);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 5) return;
#endif
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 6) return;
#endif

      paired_cube_kv<D, C, 4, false, false>(
          V_handle,
          workspace_handle,
          C,
          D,
          v_head_base1 + static_cast<int64_t>(ci) * v_chunk_stride,
          ws_k_base1,
          ws_kv_base1,
          k_l1,
          v_l1,
          kv_l0);
      if (ci + 1 < num_chunks) {
        wait_flag_dev(3);
      }
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (6 << 8));
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 7) return;
#endif
      if (ci + 1 < num_chunks) {
        wait_flag_dev(7);
      }
    }
#ifndef MEGA_STOP_AFTER_H
    wait_flag_dev(7);
#endif
  } else {
    for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
      int64_t pid = wi * block_num + cid;
      if (pid >= total_work) break;

      int64_t head = pid % H;
      int64_t head_g = head / GROUP;
      int64_t seq_idx = pid / H;

      int64_t bos, slen;
      int64_t chunk_offset = 0;
      if (cu_seqlens != nullptr) {
        bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
        int64_t eos = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
        slen = eos - bos;
        for (int64_t si = 0; si < seq_idx; ++si) {
          int64_t sb = static_cast<int64_t>(cu_seqlens[si]);
          int64_t se = static_cast<int64_t>(cu_seqlens[si + 1]);
          chunk_offset += (se - sb + C - 1) / C;
        }
      } else {
        bos = seq_idx * seq_len;
        slen = seq_len;
        chunk_offset = seq_idx * ((seq_len + C - 1) / C);
      }
      int64_t num_chunks = (slen + C - 1) / C;
      const bool emit_precomputed_qs =
          precompute_qs != 0 && C == D && H >= 8 && cu_seqlens != nullptr;
#ifdef MEGA_CHUNK_GDN_A5_PACKED_WUV
      // Variant16 keeps the private V_new tensor in [H,T,D].
      constexpr int32_t v_stride = D;
      const int64_t v_head_base =
          (head * total_tokens + bos) * static_cast<int64_t>(D);
      constexpr int64_t v_chunk_stride = static_cast<int64_t>(C) * D;
#else
      const bool use_packed_v =
          emit_precomputed_qs && batch_size == 1 && (slen % C) == 0;
      const int32_t v_stride = use_packed_v ? D : BSND_QKV_STRIDE;
      const int64_t v_head_base =
          use_packed_v ? (chunk_offset * H + head) * static_cast<int64_t>(C) * D
                       : (bos * H + head) * static_cast<int64_t>(D);
      const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;
#endif
      const int64_t ws_core_offset =
          static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
      const int64_t ws_field_span =
          static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
      const int64_t ws_ws_base = ws_core_offset;
      const int64_t ws_k_base = ws_field_span + ws_core_offset;
      const int64_t ws_s_base = 2 * ws_field_span + ws_core_offset;
      const int64_t ws_kv_base = 3 * ws_field_span + ws_core_offset;
      // One per-core scratch region stores:
      //   WS_WS : ws = W_i @ S_i
      //   WS_K  : k_tilde
      //   WS_S  : running state S_i
      //   WS_KV : k_tilde^T @ v_i_new

      for (int32_t ci = 0; ci < num_chunks; ++ci) {
#if defined(PTO_NPU_ARCH_A5)
        gdn_sync::Wait<PIPE_MTE2>(3);
#else
        wait_flag_dev(3);
#endif
        int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
        int64_t valid = slen - static_cast<int64_t>(ci) * C;
        if (valid > C) valid = C;

#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
        // S and W are both producer-published before this chunk begins. One
        // full clean+invalidate replaces thousands of scalar cacheline DCCIs.
        dcci((__gm__ void*)0, ENTIRE_DATA_CACHE);
#elif defined(GDN_A5_KERNEL)
        // Consumer-side invalidate for the Vec-published state: A5 Cube MTE2
        // loads can return clean stale lines left by a previous launch.
        for (int32_t r = 0; r < D * D; r += 16) {
          dcci(static_cast<__gm__ void*>(workspace_handle + ws_s_base + r),
               SINGLE_CACHE_LINE);
        }
#endif
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        {
          const int64_t s_state_offset = ((chunk_offset + ci) * H + head) * DD;
          GmShape2D s_shape(D, D);
          GmStride2D s_stride(D);
          GmTensor2D<ComputeT> s_global(
              S_handle + s_state_offset, s_shape, s_stride);
          DynMatL1<ComputeT, D, D> s_l1_load(D, D);
          TASSIGN(s_l1_load, 0);
          // Load S_i before this slot is overwritten with precomputed Q_i @
          // S_i.
          TLOAD(s_l1_load, s_global);
        }

        int64_t w_offset;
#ifdef MEGA_CHUNK_GDN_A5_PACKED_WUV
        w_offset =
            (head * total_tokens + chunk_start) * static_cast<int64_t>(D);
#else
        w_offset = ((chunk_start)*H + head) * D;
#endif
#if defined(GDN_A5_KERNEL) && !defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
        // W is published by the WY stage Vector cores; same stale-line hazard.
        // Rows are BSND-strided, so invalidate row by row.
        for (int32_t row = 0; row < valid; ++row) {
          for (int32_t r = 0; r < D; r += 16) {
            dcci(static_cast<__gm__ void*>(W_handle + w_offset +
                                           row * BSND_QKV_STRIDE + r),
                 SINGLE_CACHE_LINE);
          }
        }
#endif
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        {
          GmShape2D w_shape(static_cast<int32_t>(valid), D);
#ifdef MEGA_CHUNK_GDN_A5_PACKED_WUV
          GmStride2D w_stride(D);
#else
          GmStride2D w_stride(BSND_QKV_STRIDE);
#endif
          GmTensor2D<ComputeT> w_global(W_handle + w_offset, w_shape, w_stride);
          DynMatL1<ComputeT, C, D> w_l1_load(static_cast<int32_t>(valid), D);
          TASSIGN(w_l1_load, D * D * static_cast<int32_t>(sizeof(ComputeT)));
          TLOAD(w_l1_load, w_global);
          if (valid != C) {
            TFILLPAD(w_l1_load, w_l1_load);
          }
        }

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        // Apply the carried recurrent state to every token in this chunk.
        gemm_v0<ComputeT, float, C, D, D, C, D, D, D, false, false>(
            w_l1, s_l1, ws_l0, (bool)1);

        if (emit_precomputed_qs) {
          const int64_t q_offset = (chunk_start * Hg + head_g) * D;
          GmShape2D q_shape(static_cast<int32_t>(valid), D);
          GmStride2D q_stride(BSND_K_STRIDE);
          GmTensor2D<ComputeT> q_global(Q_handle + q_offset, q_shape, q_stride);
          DynMatL1<ComputeT, C, D> q_l1_load(static_cast<int32_t>(valid), D);
          TASSIGN(q_l1_load, D * D * static_cast<int32_t>(sizeof(ComputeT)));
          TLOAD(q_l1_load, q_global);
          if (valid != C) {
            TFILLPAD(q_l1_load, q_l1_load);
          }
        }

#if defined(MEGA_CHUNK_GDN_A5_H_WS_DIRECT)
        if (valid == C) {
          TileUbDataND<float, C / 2, D, C / 2, D> ws_direct_ub;
          TASSIGN(ws_direct_ub, DirectWsUbAddr);
          // DualModeSplitM publishes FP32 to both AIVs.  The Vec side below
          // preserves the old FP32->ComputeT->FP32 rounding boundary.
          TMOV_IMPL<decltype(ws_direct_ub),
                    decltype(ws_l0),
                    AccToVecMode::DualModeSplitM>(ws_direct_ub, ws_l0);
        } else {
#endif
          {
            GmShape2D ws_shape(C, D);
            GmStride2D ws_stride(D);
            GmTensor2D<ComputeT> ws_global(
                workspace_handle + ws_ws_base, ws_shape, ws_stride);
            DynAccTile<float, C, D> ws_store(C, D);
            TASSIGN(ws_store, 0);
            // Save ws_i so the Vec phase can do `v_new = U_i - ws_i`.
            TSTORE(ws_global, ws_store);
          }
#if defined(MEGA_CHUNK_GDN_A5_H_WS_DIRECT)
        }
#endif
#if defined(MEGA_CHUNK_GDN_A5_H_DIRECT_SYNC) && \
    defined(MEGA_CHUNK_GDN_A5_H_WS_DIRECT)
        if (valid == C) {
          set_intra_block(PIPE_FIX, 0);
          set_intra_block(PIPE_FIX, 0 + SYNC_FLAG_ID_MAX);
        } else {
          ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));
        }
#else
        ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));
#endif

        if (emit_precomputed_qs) {
          set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
          wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
          gemm_v0<ComputeT, float, C, D, D, C, D, D, D, false, false>(
              q_l1, s_l1, ws_l0, (bool)1);

          const int64_t qs_offset = ((chunk_offset + ci) * H + head) * DD;
          GmShape2D qs_shape(C, D);
          GmStride2D qs_stride(D);
          GmTensor2D<ComputeT> qs_global(
              S_handle + qs_offset, qs_shape, qs_stride);
          DynAccTile<float, C, D> qs_store(C, D);
          TASSIGN(qs_store, 0);
          TSTORE(qs_global, qs_store);
        }

#if defined(PTO_NPU_ARCH_A5)
        gdn_sync::Wait<PIPE_MTE2>(1);
#else
        wait_flag_dev(1);
#endif

#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
        // k_tilde and V_new become visible at the same producer boundary.
        dcci((__gm__ void*)0, ENTIRE_DATA_CACHE);
#elif defined(GDN_A5_KERNEL)
        // Consumer-side invalidate for the Vec-published k_tilde tile before
        // the KV GEMM; a stale line here corrupts the state update
        // (final_state).
        for (int32_t r = 0; r < D * C; r += 16) {
          dcci(static_cast<__gm__ void*>(workspace_handle + ws_k_base + r),
               SINGLE_CACHE_LINE);
        }
#endif
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        const int64_t v_offset =
            v_head_base + static_cast<int64_t>(ci) * v_chunk_stride;
        {
          GmShape2D k_shape(D, C);
          GmStride2D k_stride(C);
          GmTensor2D<ComputeT> k_global(
              workspace_handle + ws_k_base, k_shape, k_stride);
          DynMatL1<ComputeT, D, C> k_l1_load(D, C);
          TASSIGN(k_l1_load,
                  (DD + C * D) * static_cast<int32_t>(sizeof(ComputeT)));
          TLOAD(k_l1_load, k_global);
        }

#if defined(GDN_A5_KERNEL) && !defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
        // V_new is published by this stage's Vector cores earlier in the chunk
        // loop; the Cube MTE2 load can still hold a stale line from a previous
        // work item or launch, which corrupts the KV product (final_state).
        for (int32_t row = 0; row < valid; ++row) {
          for (int32_t r = 0; r < D; r += 16) {
            dcci(static_cast<__gm__ void*>(V_handle + v_offset +
                                           row * v_stride + r),
                 SINGLE_CACHE_LINE);
          }
        }
#endif
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        {
          GmShape2D v_shape(static_cast<int32_t>(valid), D);
          GmStride2D v_gm_stride(v_stride);
          GmTensor2D<ComputeT> v_global(
              V_handle + v_offset, v_shape, v_gm_stride);
          DynMatL1<ComputeT, C, D> v_l1_load(static_cast<int32_t>(valid), D);
          TASSIGN(
              v_l1_load,
              (DD + C * D + D * C) * static_cast<int32_t>(sizeof(ComputeT)));
          TLOAD(v_l1_load, v_global);
          if (valid != C) {
            TFILLPAD(v_l1_load, v_l1_load);
          }
        }

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        // This chunk contributes the additive update K_i^T V_i to the state
        // recurrence.
        gemm_v0<ComputeT, float, D, D, C, D, D, C, C, true, false>(
            k_l1, v_l1, kv_l0, (bool)1);

        {
          GmShape2D kv_shape(D, D);
          GmStride2D kv_stride(D);
          GmTensor2D<ComputeT> kv_global(
              workspace_handle + ws_kv_base, kv_shape, kv_stride);
          DynAccTile<float, D, D> kv_store(D, D);
          TASSIGN(kv_store, SecondL0CAddr);
          // Save kv = k_tilde^T @ v_i_new so Vec can finish the state update.
#if defined(MEGA_CHUNK_GDN_A5_H_KV_DIRECT)
          if (valid == C) {
            TileUbDataND<float, C / 2, D, C / 2, D> kv_direct_ub;
            TASSIGN(kv_direct_ub, DirectWsUbAddr);
            TMOV_IMPL<decltype(kv_direct_ub),
                      decltype(kv_l0),
                      AccToVecMode::DualModeSplitM>(kv_direct_ub, kv_l0);
          } else {
#endif
            TSTORE(kv_global, kv_store);
#if defined(MEGA_CHUNK_GDN_A5_H_KV_DIRECT)
          }
#endif
        }
#if defined(MEGA_CHUNK_GDN_A5_H_DIRECT_SYNC) && \
    defined(MEGA_CHUNK_GDN_A5_H_KV_DIRECT) &&   \
    !defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
        if (valid == C && !emit_precomputed_qs) {
          set_intra_block(PIPE_FIX, 2);
          set_intra_block(PIPE_FIX, 2 + SYNC_FLAG_ID_MAX);
        } else {
          ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
        }
#else
        ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
#endif
      }
    }
  }
#endif
#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (use_paired_head_pipeline && cid < paired_core_count) {
    const int64_t head0 = cid;
    const int64_t head1 = block_num + cid;
    const int64_t head_g0 = head0 / GROUP;
    const int64_t head_g1 = head1 / GROUP;
    const int64_t bos = static_cast<int64_t>(cu_seqlens[0]);
    const int64_t slen = static_cast<int64_t>(cu_seqlens[1]) - bos;
    const int32_t num_chunks = static_cast<int32_t>(slen / C);
    const int64_t ws_core_offset = static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
    const int64_t ws_field_span =
        static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
    const int64_t ws_slot_span = 4 * ws_field_span;
    const int64_t ws_ws_base0 = ws_core_offset;
    const int64_t ws_k_base0 = ws_field_span + ws_core_offset;
    const int64_t ws_s_base0 = 2 * ws_field_span + ws_core_offset;
    const int64_t ws_kv_base0 = 3 * ws_field_span + ws_core_offset;
    const int64_t ws_ws_base1 = ws_slot_span + ws_ws_base0;
    const int64_t ws_k_base1 = ws_slot_span + ws_k_base0;
    const int64_t ws_s_base1 = ws_slot_span + ws_s_base0;
    const int64_t ws_kv_base1 = ws_slot_span + ws_kv_base0;
    const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;
    const int64_t v_head_base0 = head0 * static_cast<int64_t>(C) * D;
    const int64_t v_head_base1 = head1 * static_cast<int64_t>(C) * D;

    paired_vec_init_zero<D, C, 0, true>(S_handle,
                                        workspace_handle,
                                        head0,
                                        0,
                                        H,
                                        vid,
                                        ws_s_base0,
                                        s_ub,
                                        s_ub_half);
    paired_vec_init_zero<D, C, 0, false>(S_handle,
                                         workspace_handle,
                                         head1,
                                         0,
                                         H,
                                         vid,
                                         ws_s_base1,
                                         s_alt_ub,
                                         s_ub_half);
#ifdef MEGA_STOP_AFTER_H
    if constexpr (GDN_PAIR_DEBUG_STAGE == 1) return;
    if constexpr (GDN_PAIR_DEBUG_STAGE == 14 || GDN_PAIR_DEBUG_STAGE == 15) {
      wait_flag_dev(0);
      return;
    }
    if constexpr (GDN_PAIR_DEBUG_STAGE == 10) {
      wait_flag_dev(0);
      return;
    }
    if constexpr (GDN_PAIR_DEBUG_STAGE == 12) {
      wait_flag_dev(0);
      return;
    }
    if constexpr (GDN_PAIR_DEBUG_STAGE == 13) {
      wait_flag_dev(0);
      return;
    }
    if constexpr (GDN_PAIR_DEBUG_STAGE == 2 || GDN_PAIR_DEBUG_STAGE == 8 ||
                  GDN_PAIR_DEBUG_STAGE == 9 || GDN_PAIR_DEBUG_STAGE == 11) {
      wait_flag_dev(0);
      return;
    }
#endif

    for (int32_t ci = 0; ci < num_chunks; ++ci) {
      const int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      paired_vec_front<D, C, 0, G_UB, COEFF_UB, U_UB>(K_handle,
                                                      U_handle,
                                                      G_handle,
                                                      V_handle,
                                                      workspace_handle,
                                                      head0,
                                                      head_g0,
                                                      chunk_start,
                                                      C,
                                                      ci,
                                                      H,
                                                      Hg,
                                                      total_tokens,
                                                      vid,
                                                      D,
                                                      v_head_base0,
                                                      v_chunk_stride,
                                                      ws_ws_base0,
                                                      ws_k_base0,
                                                      s_ub,
                                                      k_ub_half,
                                                      g_ub,
                                                      u_ub_half,
                                                      k_ub,
                                                      g_last_tail_ub,
                                                      coeff_ub,
                                                      u_ub,
                                                      ws_ub);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 3) return;
      if constexpr (GDN_PAIR_DEBUG_STAGE == 4) {
        wait_flag_dev(4);
        return;
      }
#endif
      paired_vec_front<D, C, 4, G_UB, COEFF_UB, U_UB>(K_handle,
                                                      U_handle,
                                                      G_handle,
                                                      V_handle,
                                                      workspace_handle,
                                                      head1,
                                                      head_g1,
                                                      chunk_start,
                                                      C,
                                                      ci,
                                                      H,
                                                      Hg,
                                                      total_tokens,
                                                      vid,
                                                      D,
                                                      v_head_base1,
                                                      v_chunk_stride,
                                                      ws_ws_base1,
                                                      ws_k_base1,
                                                      s_alt_ub,
                                                      k_ub_half,
                                                      g_ub,
                                                      u_ub_half,
                                                      k_ub,
                                                      g_last_tail_ub,
                                                      coeff_ub,
                                                      u_ub,
                                                      ws_ub);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 5) return;
#endif
      paired_vec_finish<D, C, 0>(S_handle,
                                 workspace_handle,
                                 h_o_ready_handle,
                                 head0,
                                 0,
                                 ci,
                                 num_chunks,
                                 H,
                                 vid,
                                 ws_s_base0,
                                 ws_kv_base0,
                                 true,
                                 s_ub,
                                 s_ub_half,
                                 kv_ub,
                                 h_o_ready_ub);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 6) return;
#endif
      paired_vec_finish<D, C, 4>(S_handle,
                                 workspace_handle,
                                 h_o_ready_handle,
                                 head1,
                                 0,
                                 ci,
                                 num_chunks,
                                 H,
                                 vid,
                                 ws_s_base1,
                                 ws_kv_base1,
                                 true,
                                 s_alt_ub,
                                 s_ub_half,
                                 kv_ub,
                                 h_o_ready_ub);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 7) return;
#endif
    }

    paired_vec_store_final<D, C, StoreFinalStateCache>(FS_handle,
                                                       final_state_cache,
                                                       state_indices,
                                                       head0,
                                                       H,
                                                       vid,
                                                       state_index_stride,
                                                       state_cache_slots,
                                                       output_final_state,
                                                       s_ub,
                                                       s_ub_half);
    paired_vec_store_final<D, C, StoreFinalStateCache>(FS_handle,
                                                       final_state_cache,
                                                       state_indices,
                                                       head1,
                                                       H,
                                                       vid,
                                                       state_index_stride,
                                                       state_cache_slots,
                                                       output_final_state,
                                                       s_alt_ub,
                                                       s_ub_half);
#ifndef MEGA_STOP_AFTER_H
    ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (7 << 8));
#endif
  } else {
    // Vec owns the running recurrent state S_i and updates it after every
    // chunk.
    for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
      int64_t pid = wi * block_num + cid;
      if (pid >= total_work) break;

      int64_t head = pid % H;
      int64_t head_g = head / GROUP;
      int64_t seq_idx = pid / H;

      int64_t bos, slen;
      int64_t chunk_offset = 0;
      if (cu_seqlens != nullptr) {
        bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
        int64_t eos = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
        slen = eos - bos;
        for (int64_t si = 0; si < seq_idx; ++si) {
          int64_t sb = static_cast<int64_t>(cu_seqlens[si]);
          int64_t se = static_cast<int64_t>(cu_seqlens[si + 1]);
          chunk_offset += (se - sb + C - 1) / C;
        }
      } else {
        bos = seq_idx * seq_len;
        slen = seq_len;
        chunk_offset = seq_idx * ((seq_len + C - 1) / C);
      }
      int64_t num_chunks = (slen + C - 1) / C;
      const bool emit_precomputed_qs =
          precompute_qs != 0 && C == D && H >= 8 && cu_seqlens != nullptr;
#ifdef MEGA_CHUNK_GDN_A5_PACKED_WUV
      constexpr int32_t v_stride = D;
      const int64_t v_head_base =
          (head * total_tokens + bos) * static_cast<int64_t>(D);
      constexpr int64_t v_chunk_stride = static_cast<int64_t>(C) * D;
#else
      const bool use_packed_v =
          emit_precomputed_qs && batch_size == 1 && (slen % C) == 0;
      const int32_t v_stride = use_packed_v ? D : BSND_QKV_STRIDE;
      const int64_t v_head_base =
          use_packed_v ? (chunk_offset * H + head) * static_cast<int64_t>(C) * D
                       : (bos * H + head) * static_cast<int64_t>(D);
      const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;
#endif
      const int64_t ws_core_offset =
          static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
      const int64_t ws_field_span =
          static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
      const int64_t ws_ws_base = ws_core_offset;
      const int64_t ws_k_base = ws_field_span + ws_core_offset;
      const int64_t ws_s_base = 2 * ws_field_span + ws_core_offset;
      const int64_t ws_kv_base = 3 * ws_field_span + ws_core_offset;

      if (has_initial_state != 0) {
        if constexpr (LoadInitialStateCache) {
#if defined(GDN_A5_VECTOR_KERNEL) && \
    !defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
          dcci(static_cast<__gm__ void*>(initial_state_indices + seq_idx),
               SINGLE_CACHE_LINE);
          dsb(DSB_DDR);
#endif
          const int32_t state_index = initial_state_indices[seq_idx];
          if (state_index >= 0 && state_index < state_cache_slots) {
            const int64_t cache_offset =
                (static_cast<int64_t>(state_index) * H + head) * DD +
                vid * HalfC * D;
#if defined(GDN_A5_KERNEL) && !defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
            for (int32_t r = 0; r < HalfC * D; r += 16) {
              dcci(static_cast<__gm__ void*>(initial_state_cache +
                                             cache_offset + r),
                   SINGLE_CACHE_LINE);
            }
            dsb(DSB_DDR);
#endif
#if defined(GDN_A5_VECTOR_KERNEL)
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
#endif
            GmShape2D cache_shape(HalfC, D);
            GmStride2D cache_stride(D);
            GmTensor2D<float> cache_global(
                initial_state_cache + cache_offset, cache_shape, cache_stride);
            DynVecTile<float, HalfC, D> cache_load(HalfC, D);
            TASSIGN(cache_load, S_UB);
            TLOAD(cache_load, cache_global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            // Match the standalone FP32 cache -> BF16 initial-state boundary.
            TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_RINT);
            TCVT(s_ub, s_ub_half, pto::RoundMode::CAST_NONE);
          } else {
            TEXPANDS(s_ub, 0.0f);
            TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);
          }
        } else {
          int64_t h0_offset = (seq_idx * H + head) * DD + vid * HalfC * D;
#if defined(GDN_A5_KERNEL)
          // H0 GM lines may be stale in this Vector core's cache when the
          // address was read by a previous work item or launch.
          for (int32_t r = 0; r < HalfC * D; r += 16) {
            dcci(static_cast<__gm__ void*>(H0_handle + h0_offset + r),
                 SINGLE_CACHE_LINE);
          }
#endif
          set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
          GmShape2D h0_shape(HalfC, D);
          GmStride2D h0_stride(D);
          GmTensor2D<ComputeT> h0_global(
              H0_handle + h0_offset, h0_shape, h0_stride);
          DynVecTile<ComputeT, HalfC, D> h0_load(HalfC, D);
          TASSIGN(h0_load, S_UB_HALF);
          TLOAD(h0_load, h0_global);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          TCVT(s_ub, s_ub_half, pto::RoundMode::CAST_NONE);
        }
      } else {
        // Start each sequence/head recurrence from S_0 = 0.
        TEXPANDS(s_ub, 0.0f);
        TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);
      }
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        int64_t s_out_offset = (chunk_offset * H + head) * DD;
        GmShape2D s_out_shape(HalfC, D);
        GmStride2D s_out_stride(D);
        GmTensor2D<ComputeT> s_out_global(
            S_handle + s_out_offset + vid * HalfC * D,
            s_out_shape,
            s_out_stride);
        DynVecTile<ComputeT, HalfC, D> s_out_store(HalfC, D);
        TASSIGN(s_out_store, S_UB_HALF);
        TSTORE(s_out_global, s_out_store);
      }
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));

      int64_t chunk_start_0 = bos;
      int64_t valid0 = slen;
      if (valid0 > C) valid0 = C;
      // Vec work is split by row stripe, not by individual token.  For the
      // first chunk we compute exactly how many live rows belong to this
      // sub-block's HalfC stripe so short tails do not overrun the packed BSND
      // input.
      int32_t valid_rows_0 =
          static_cast<int32_t>(valid0 - static_cast<int64_t>(vid) * HalfC);
      if (valid_rows_0 < 0) valid_rows_0 = 0;
      if (valid_rows_0 > HalfC) valid_rows_0 = HalfC;

      int64_t k_offset_0 =
          (chunk_start_0 * Hg + head_g) * D + vid * HalfC * BSND_K_STRIDE;
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
      // Initial K and G are consumed from the same completed frontend stage.
      dcci((__gm__ void*)0, ENTIRE_DATA_CACHE);
#endif
      if (valid_rows_0 > 0) {
#if defined(GDN_A5_KERNEL) && !defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
        // K only feeds the state update (final_state); a stale line here does
        // not show up in the main output, so it must be invalidated explicitly.
        for (int32_t row = 0; row < valid_rows_0; ++row) {
          for (int32_t r = 0; r < D; r += 16) {
            dcci(static_cast<__gm__ void*>(K_handle + k_offset_0 +
                                           row * BSND_K_STRIDE + r),
                 SINGLE_CACHE_LINE);
          }
        }
#endif
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        GmShape2D k_shape(valid_rows_0, D);
        GmStride2D k_stride(BSND_K_STRIDE);
        GmTensor2D<ComputeT> k_global(K_handle + k_offset_0, k_shape, k_stride);
        DynVecTile<ComputeT, HalfC, D, pto::PadValue::Zero> k_load(valid_rows_0,
                                                                   D);
        TASSIGN(k_load, K_UB_HALF);
        TLOAD(k_load, k_global);
        if (valid_rows_0 != HalfC) {
          TFILLPAD_INPLACE(k_ub_half, k_load);
        }
      } else {
        // Empty stripe (typically vid=1 on a very short tail chunk): synthesize
        // a zero tile so later full-width vector math and workspace stores
        // still observe proper padding semantics.
        TEXPANDS(k_ub, 0.0f);
        TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);
      }

      {
#if defined(GDN_A5_KERNEL) && !defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
        // g (log-sigmoid gate) only feeds the state recurrence; same
        // consumer-side stale-line hazard as K.
        for (int32_t r = 0; r < valid0 * 4; r += 32) {
          dcci(static_cast<__gm__ void*>(
                   reinterpret_cast<__gm__ char*>(
                       G_handle + head * total_tokens + chunk_start_0) +
                   r),
               SINGLE_CACHE_LINE);
        }
#endif
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        GmShape2D g_shape(1, static_cast<int32_t>(valid0));
        GmStride2D g_stride(1);
        GmTensor2D<float> g_global(
            G_handle + head * total_tokens + chunk_start_0, g_shape, g_stride);
        DynVecTile<float, 1, C, pto::PadValue::Zero> g_load(
            1, static_cast<int32_t>(valid0));
        TASSIGN(g_load, G_UB);
        TLOAD(g_load, g_global);
        if (valid0 != C) {
          TFILLPAD_INPLACE(g_ub, g_load);
        }
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      for (int32_t ci = 0; ci < static_cast<int32_t>(num_chunks); ++ci) {
        int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
        int64_t valid = slen - static_cast<int64_t>(ci) * C;
        if (valid > C) valid = C;
        int32_t valid_rows =
            static_cast<int32_t>(valid - static_cast<int64_t>(vid) * HalfC);
        if (valid_rows < 0) valid_rows = 0;
        if (valid_rows > HalfC) valid_rows = HalfC;
        // Each Vec subblock owns one contiguous HalfC-row stripe of the chunk.
        // For short tail chunks, `valid_rows` may be smaller or even zero. This
        // is the key fix that keeps ragged tails and dense varlen boundary
        // mixes from reading or writing beyond the live rows in this stripe.

        int64_t u_offset;
#ifdef MEGA_CHUNK_GDN_A5_PACKED_WUV
        u_offset = (head * total_tokens + chunk_start +
                    static_cast<int64_t>(vid) * HalfC) *
                   D;
#else
        u_offset = (chunk_start * H + head) * D + vid * HalfC * BSND_QKV_STRIDE;
#endif
        if (valid_rows > 0) {
          GmShape2D u_shape(valid_rows, D);
#ifdef MEGA_CHUNK_GDN_A5_PACKED_WUV
          GmStride2D u_stride(D);
#else
          GmStride2D u_stride(BSND_QKV_STRIDE);
#endif
          GmTensor2D<ComputeT> u_global(U_handle + u_offset, u_shape, u_stride);
          DynVecTile<ComputeT, HalfC, D, pto::PadValue::Zero> u_load(valid_rows,
                                                                     D);
          TASSIGN(u_load, U_UB_HALF);
          TLOAD(u_load, u_global);
          if (valid_rows != HalfC) {
            TFILLPAD_INPLACE(u_ub_half, u_load);
          }
        } else {
          // No live rows for this stripe in the current chunk; keep the tile
          // explicitly zero-padded so the remainder of the recurrence logic can
          // run in full-tile form without special-casing every later step.
          TEXPANDS(u_ub, 0.0f);
          TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);
        }

        TCVT(k_ub, k_ub_half, pto::RoundMode::CAST_NONE);

        TileUbDataND<float, 1, 64, 1, 64> g_ub_temp;
        TASSIGN(g_ub_temp, G_UB + vid * 64 * sizeof(float));

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float g_last = g_ub.GetValue(static_cast<int32_t>(valid) - 1);
        // Rebase the chunk gate around g_last so the intra-chunk decay stays
        // numerically local. Torch-like:
        //   coeff = exp(g_last - g_rows_owned_by_this_subblock)
        TADDS(coeff_ub, g_ub_temp, -g_last);
        pipe_barrier(PIPE_V);
        TNEG(coeff_ub, coeff_ub);
        pipe_barrier(PIPE_V);
        TEXP(coeff_ub, coeff_ub);

        if (valid == C) {
          // Only exp(g_last) is consumed below. Keep the tail path unchanged,
          // while avoiding two full-vector EXP repeats for dense chunks.
          TEXP(g_last_tail_ub, g_last_tail_ub);
        } else {
          TEXP(g_ub, g_ub);
        }

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TileUbDataDN<float, HalfC, 1, HalfC, 1> coeff_col_ub;
        TASSIGN(coeff_col_ub, COEFF_UB);
        TileUbDataND<float, HalfC, D, HalfC, D> coeff_2d_ub;
        TASSIGN(coeff_2d_ub, U_UB);
        // Broadcast one decay scalar per token row across the D feature
        // columns:
        //   coeff_2d[row, :] = coeff[row]
        TROWEXPAND(coeff_2d_ub, coeff_col_ub);
        pipe_barrier(PIPE_V);
        // `k_ub` now holds k_tilde = exp(g_last - g_i) * K_i.
        TMUL(k_ub, k_ub, coeff_2d_ub);
        pipe_barrier(PIPE_V);
        TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);
        TCVT(u_ub, u_ub_half, pto::RoundMode::CAST_NONE);

#if defined(MEGA_CHUNK_GDN_A5_H_DIRECT_SYNC) && \
    defined(MEGA_CHUNK_GDN_A5_H_WS_DIRECT)
        if (valid == C) {
          wait_intra_block(PIPE_V, 0);
        } else {
          wait_flag_dev(0);
        }
#else
        wait_flag_dev(0);
#endif
#if defined(MEGA_CHUNK_GDN_A5_H_WS_DIRECT)
        if (valid == C) {
          TileUbDataND<float, HalfC, D, HalfC, D> ws_direct_ub;
          TASSIGN(ws_direct_ub, DirectWsUbAddr);
          // Preserve the legacy GM TSTORE FP32->ComputeT boundary. The common
          // conversion below then produces the FP32 operand for TSUB.
          TCVT(u_ub_half, ws_direct_ub, pto::RoundMode::CAST_NONE);
          pipe_barrier(PIPE_V);
        } else {
#endif
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
          dcci((__gm__ void*)0, ENTIRE_DATA_CACHE);
#elif defined(GDN_A5_KERNEL)
        // A5 Vector MTE2 loads can return clean stale lines when the workspace
        // addresses were read by a previous launch.  Invalidate the Cube
        // product rows on the consumer immediately before loading them.
        for (int32_t r = 0; r < HalfC * D; r += 16) {
          dcci(static_cast<__gm__ void*>(workspace_handle + ws_ws_base +
                                         vid * HalfC * D + r),
               SINGLE_CACHE_LINE);
        }
#endif
          set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
          {
            GmShape2D ws_shape(HalfC, D);
            GmStride2D ws_stride(D);
            GmTensor2D<ComputeT> ws_global(
                workspace_handle + ws_ws_base + vid * HalfC * D,
                ws_shape,
                ws_stride);
            DynVecTile<ComputeT, HalfC, D, pto::PadValue::Zero> ws_load(HalfC,
                                                                        D);
            TASSIGN(ws_load, U_UB_HALF);
            TLOAD(ws_load, ws_global);
          }
#if defined(MEGA_CHUNK_GDN_A5_H_WS_DIRECT)
        }
#endif

#if defined(MEGA_CHUNK_GDN_A5_H_WS_DIRECT)
        if (valid != C) {
#endif
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#if defined(MEGA_CHUNK_GDN_A5_H_WS_DIRECT)
        }
#endif
        TCVT(ws_ub, u_ub_half, pto::RoundMode::CAST_NONE);
        // v_i_new = U_i - W_i @ S_i.
        // In PyTorch notation:
        //   u_ub = u_ub - ws_ub
        TSUB(u_ub, u_ub, ws_ub);
        pipe_barrier(PIPE_V);
        // W@S is formed as a full C-row Cube tile. On a ragged tail, rows
        // outside valid_rows are not part of the sequence and must not feed
        // the later K^T V state update, especially when S starts non-zero.
        if (valid_rows == 0) {
          TEXPANDS(u_ub, 0.0f);
        } else if (valid_rows != HalfC) {
          DynVecTile<float, HalfC, D, pto::PadValue::Zero> u_live(valid_rows,
                                                                  D);
          TASSIGN(u_live, U_UB);
          TileUbDataND<float, HalfC, D, HalfC, D, pto::PadValue::Zero>
              u_ub_padded;
          TASSIGN(u_ub_padded, U_UB);
          TFILLPAD_INPLACE(u_ub_padded, u_live);
        }
        pipe_barrier(PIPE_V);
        TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        const int64_t v_offset = v_head_base +
                                 static_cast<int64_t>(ci) * v_chunk_stride +
                                 static_cast<int64_t>(vid) * HalfC * v_stride;
        if (valid_rows > 0) {
          GmShape2D v_shape(valid_rows, D);
          GmStride2D v_gm_stride(v_stride);
          GmTensor2D<ComputeT> v_global(
              V_handle + v_offset, v_shape, v_gm_stride);
          DynVecTile<ComputeT, HalfC, D> v_store(valid_rows, D);
          TASSIGN(v_store, U_UB_HALF);
          TSTORE(v_global, v_store);
        }

        // Spill both V_i_new and k_i_tilde so the Cube stage can form
        // k_i_tilde^T @ V_i_new for this chunk.
        {
          GmShape2D k_shape(HalfC, D);
          GmStride2D k_stride(D);
          GmTensor2D<ComputeT> k_global(
              workspace_handle + ws_k_base + vid * HalfC * D,
              k_shape,
              k_stride);
          DynVecTile<ComputeT, HalfC, D> k_store(HalfC, D);
          TASSIGN(k_store, K_UB_HALF);
          TSTORE(k_global, k_store);
        }

        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));

        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        float exp_g_last = valid == C
                               ? g_last_tail_ub.GetValue(ExpTailElements - 1)
                               : g_ub.GetValue(static_cast<int32_t>(valid) - 1);
        // Carry the recurrence across chunks: S_{i+1} = exp(g_last) * S_i +
        // K_i^T V_i.
#if defined(GDN_A5_VECTOR_KERNEL) && defined(MEGA_CHUNK_GDN_A5_H_STATE_REGBASE)
        // Fold this multiply into the RegBase state update after KV arrives.
#elif defined(GDN_A5_KERNEL)
        for (int32_t row = 0; row < HalfC; ++row) {
          TileUbDataND<float, 1, D> s_row;
          TASSIGN(s_row, S_UB + row * D * static_cast<int32_t>(sizeof(float)));
          TMULS(s_row, s_row, exp_g_last);
        }
#else
        TMULS(s_ub, s_ub, exp_g_last);
#endif

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        if (ci + 1 < static_cast<int32_t>(num_chunks)) {
          int64_t next_start = bos + static_cast<int64_t>(ci + 1) * C;
          int64_t next_valid = slen - static_cast<int64_t>(ci + 1) * C;
          if (next_valid > C) next_valid = C;
          int32_t next_valid_rows = static_cast<int32_t>(
              next_valid - static_cast<int64_t>(vid) * HalfC);
          if (next_valid_rows < 0) next_valid_rows = 0;
          if (next_valid_rows > HalfC) next_valid_rows = HalfC;

          int64_t nk_off =
              (next_start * Hg + head_g) * D + vid * HalfC * BSND_K_STRIDE;
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
          dcci((__gm__ void*)0, ENTIRE_DATA_CACHE);
#endif
          if (next_valid_rows > 0) {
#if defined(GDN_A5_KERNEL) && !defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
            // Same stale-line invalidation for the prefetched K rows.
            for (int32_t row = 0; row < next_valid_rows; ++row) {
              for (int32_t r = 0; r < D; r += 16) {
                dcci(static_cast<__gm__ void*>(K_handle + nk_off +
                                               row * BSND_K_STRIDE + r),
                     SINGLE_CACHE_LINE);
              }
            }
#endif
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            GmShape2D k_shape(next_valid_rows, D);
            GmStride2D k_stride(BSND_K_STRIDE);
            GmTensor2D<ComputeT> k_global(K_handle + nk_off, k_shape, k_stride);
            DynVecTile<ComputeT, HalfC, D, pto::PadValue::Zero> k_load(
                next_valid_rows, D);
            TASSIGN(k_load, K_UB_HALF);
            TLOAD(k_load, k_global);
            if (next_valid_rows != HalfC) {
              TFILLPAD_INPLACE(k_ub_half, k_load);
            }
          } else {
            // Same tail-safe zero materialization for the prefetch path: the
            // next chunk may have no rows in this stripe even though the other
            // stripe is still active.
            TEXPANDS(k_ub, 0.0f);
            TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);
          }

          {
#if defined(GDN_A5_KERNEL) && !defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
            // Same stale-line invalidation for the prefetched gate row.
            for (int32_t r = 0; r < next_valid * 4; r += 32) {
              dcci(static_cast<__gm__ void*>(
                       reinterpret_cast<__gm__ char*>(
                           G_handle + head * total_tokens + next_start) +
                       r),
                   SINGLE_CACHE_LINE);
            }
#endif
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            GmShape2D g_shape(1, static_cast<int32_t>(next_valid));
            GmStride2D g_stride(1);
            GmTensor2D<float> g_global(
                G_handle + head * total_tokens + next_start, g_shape, g_stride);
            DynVecTile<float, 1, C, pto::PadValue::Zero> g_load(
                1, static_cast<int32_t>(next_valid));
            TASSIGN(g_load, G_UB);
            TLOAD(g_load, g_global);
            if (next_valid != C) {
              TFILLPAD_INPLACE(g_ub, g_load);
            }
          }
        }

#if defined(MEGA_CHUNK_GDN_A5_H_DIRECT_SYNC) && \
    defined(MEGA_CHUNK_GDN_A5_H_KV_DIRECT) &&   \
    !defined(MEGA_CHUNK_GDN_A5_HO_OVERLAP)
        if (valid == C && !emit_precomputed_qs) {
          wait_intra_block(PIPE_V, 2);
        } else {
          wait_flag_dev(2);
        }
#else
        wait_flag_dev(2);
#endif
        if (emit_precomputed_qs && vid == 0) {
          const int64_t ready_offset = (seq_idx * H + head) * H_O_READY_STRIDE;
          h_o_ready_gm.SetValue(ready_offset, static_cast<int32_t>(ci + 1));
          __asm__ __volatile__("");
          AscendC::DataCacheCleanAndInvalid<
              int32_t,
              AscendC::CacheLine::SINGLE_CACHE_LINE,
              AscendC::DcciDst::CACHELINE_ALL>(h_o_ready_gm[ready_offset]);
          __asm__ __volatile__("");
        }
#if defined(MEGA_CHUNK_GDN_A5_H_KV_DIRECT)
        if (valid == C) {
          TileUbDataND<float, HalfC, D, HalfC, D> kv_direct_ub;
          TASSIGN(kv_direct_ub, DirectWsUbAddr);
          // Preserve the legacy GM TSTORE FP32->ComputeT boundary before the
          // existing FP32 recurrent update (including RegBase).
          TCVT(s_ub_half, kv_direct_ub, pto::RoundMode::CAST_NONE);
          pipe_barrier(PIPE_V);
        } else {
#endif
#if defined(GDN_A5_KERNEL) && defined(MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI)
          dcci((__gm__ void*)0, ENTIRE_DATA_CACHE);
#elif defined(GDN_A5_KERNEL)
        // Same consumer-side invalidation for the k_tilde^T @ v_new product
        // published by the Cube stage; without it the recurrence can fold in a
        // stale line from a previous launch and corrupt the final state.
        for (int32_t r = 0; r < HalfC * D; r += 16) {
          dcci(static_cast<__gm__ void*>(workspace_handle + ws_kv_base +
                                         vid * HalfC * D + r),
               SINGLE_CACHE_LINE);
        }
#endif
          set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
          {
            GmShape2D kv_shape(HalfC, D);
            GmStride2D kv_stride(D);
            GmTensor2D<ComputeT> kv_global(
                workspace_handle + ws_kv_base + vid * HalfC * D,
                kv_shape,
                kv_stride);
            DynVecTile<ComputeT, HalfC, D, pto::PadValue::Zero> kv_load(HalfC,
                                                                        D);
            TASSIGN(kv_load, S_UB_HALF);
            TLOAD(kv_load, kv_global);
          }
#if defined(MEGA_CHUNK_GDN_A5_H_KV_DIRECT)
        }
#endif
#if defined(MEGA_CHUNK_GDN_A5_H_KV_DIRECT)
        if (valid != C) {
#endif
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#if defined(MEGA_CHUNK_GDN_A5_H_KV_DIRECT)
        }
#endif
#if defined(GDN_A5_VECTOR_KERNEL) && defined(MEGA_CHUNK_GDN_A5_H_STATE_REGBASE)
        // Conversion is fused below with the FP32 recurrent update.
#elif defined(GDN_A5_KERNEL)
        // A5 hardware quirk (also noted in the non-tiled A5 correctness path):
        // vector ops on 64x128 UB tiles can leave stale rows on a physical
        // Vector subblock.  Do the kv conversion and the state update row by
        // row so no tile row can go stale.
        for (int32_t row = 0; row < HalfC; ++row) {
          TileUbDataND<float, 1, D> kv_row_f;
          TileUbDataND<ComputeT, 1, D> kv_row_h;
          TASSIGN(kv_row_f,
                  KV_UB + row * D * static_cast<int32_t>(sizeof(float)));
          TASSIGN(kv_row_h,
                  S_UB_HALF + row * D * static_cast<int32_t>(sizeof(ComputeT)));
          TCVT(kv_row_f, kv_row_h, pto::RoundMode::CAST_NONE);
        }
#else
        TCVT(kv_ub, s_ub_half, pto::RoundMode::CAST_NONE);
#endif
        pipe_barrier(PIPE_ALL);
        // Finish S_{i+1} = exp(g_last) * S_i + k_i_tilde^T @ v_i_new.
        // Torch-like:
        //   s_ub = s_ub + kv_ub
#if defined(GDN_A5_VECTOR_KERNEL) && defined(MEGA_CHUNK_GDN_A5_H_STATE_REGBASE)
#if defined(MEGA_CHUNK_GDN_A5_H_STATE_REGBASE4)
        AscendC::VF_CALL<UpdateHStateRegbase4<ComputeT, HalfC * D>>(
            reinterpret_cast<__ubuf__ float*>(s_ub.data()),
            reinterpret_cast<__ubuf__ ComputeT*>(s_ub_half.data()),
            reinterpret_cast<__ubuf__ ComputeT*>(s_ub_half.data()),
            exp_g_last);
#else
        AscendC::VF_CALL<UpdateHStateRegbase<ComputeT>>(
            reinterpret_cast<__ubuf__ float*>(s_ub.data()),
            reinterpret_cast<__ubuf__ ComputeT*>(s_ub_half.data()),
            reinterpret_cast<__ubuf__ ComputeT*>(s_ub_half.data()),
            exp_g_last,
            static_cast<uint32_t>(HalfC * D));
#endif
        pipe_barrier(PIPE_V);
#elif defined(GDN_A5_KERNEL)
        for (int32_t row = 0; row < HalfC; ++row) {
          TileUbDataND<float, 1, D> s_row;
          TileUbDataND<float, 1, D> kv_row;
          TileUbDataND<ComputeT, 1, D> out_row;
          TASSIGN(s_row, S_UB + row * D * static_cast<int32_t>(sizeof(float)));
          TASSIGN(kv_row,
                  KV_UB + row * D * static_cast<int32_t>(sizeof(float)));
          TASSIGN(out_row,
                  S_UB_HALF + row * D * static_cast<int32_t>(sizeof(ComputeT)));
          TADD(s_row, s_row, kv_row);
          TCVT(out_row, s_row, pto::RoundMode::CAST_NONE);
        }
#else
        TADD(s_ub, s_ub, kv_ub);
        TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);
#endif

        if (ci + 1 < static_cast<int32_t>(num_chunks)) {
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          // Expose the post-chunk state so the next chunk and output snapshot
          // can see S_{i+1}. Conceptually:
          //   S_handle[chunk_idx + 1, head] = S_{i+1}
          int64_t s_out_offset = ((chunk_offset + ci + 1) * H + head) * DD;
          {
            GmShape2D s_out_shape(HalfC, D);
            GmStride2D s_out_stride(D);
            GmTensor2D<ComputeT> s_out_global(
                S_handle + s_out_offset + vid * HalfC * D,
                s_out_shape,
                s_out_stride);
            DynVecTile<ComputeT, HalfC, D> s_out_store(HalfC, D);
            TASSIGN(s_out_store, S_UB_HALF);
            TSTORE(s_out_global, s_out_store);
          }
          ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
        }

        if (ci + 1 < static_cast<int32_t>(num_chunks)) {
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        }
      }

      if (output_final_state != 0) {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        int64_t fs_offset = (seq_idx * H + head) * DD;
        if constexpr (StoreFinalStateCache) {
          TCVT(s_ub, s_ub_half, pto::RoundMode::CAST_NONE);
          pipe_barrier(PIPE_V);
          const int32_t state_index =
              state_indices[seq_idx] * state_index_stride;
          if (state_index >= 0 && state_index < state_cache_slots) {
            const int64_t cache_offset =
                (static_cast<int64_t>(state_index) * H + head) * DD +
                vid * HalfC * D;
            GmShape2D cache_shape(HalfC, D);
            GmStride2D cache_stride(D);
            GmTensor2D<float> cache_global(
                final_state_cache + cache_offset, cache_shape, cache_stride);
            DynVecTile<float, HalfC, D> cache_store(HalfC, D);
            TASSIGN(cache_store, S_UB);
            TSTORE(cache_global, cache_store);
          }
        } else {
          GmShape2D fs_shape(HalfC, D);
          GmStride2D fs_stride(D);
          GmTensor2D<ComputeT> fs_global(
              FS_handle + fs_offset + vid * HalfC * D, fs_shape, fs_stride);
          DynVecTile<ComputeT, HalfC, D> fs_store(HalfC, D);
          TASSIGN(fs_store, S_UB_HALF);
          TSTORE(fs_global, fs_store);
        }
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
#if defined(GDN_A5_KERNEL)
        // The FS TSTORE reads S_UB_HALF asynchronously on the MTE3 pipe.  The
        // next work item on this block immediately reuses that UB region (H0
        // TLOAD, TCVT, TEXPANDS), which can overwrite rows the store has not
        // read yet and corrupt final_state with the next item's input data.
        // Order the following V-pipe writes after the store completes.
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        // A5 MIX Vector GM writes are cached per Vector core and are not
        // snooped by the next kernel.  Clean the address that was actually
        // written above.  Prefill stores directly into final_state_cache,
        // whereas the standalone operator stores into FS_handle; cleaning
        // FS_handle unconditionally leaves Prefill's cache slot stale.
        if constexpr (StoreFinalStateCache) {
          const int32_t state_index =
              state_indices[seq_idx] * state_index_stride;
          if (state_index >= 0 && state_index < state_cache_slots) {
            const int64_t cache_offset =
                (static_cast<int64_t>(state_index) * H + head) * DD +
                vid * HalfC * D;
            for (int32_t r = 0; r < HalfC * D; r += 16) {
              dcci(static_cast<__gm__ void*>(final_state_cache + cache_offset +
                                             r),
                   SINGLE_CACHE_LINE);
            }
          }
        } else {
          for (int32_t r = 0; r < HalfC * D; r += 16) {
            dcci(static_cast<__gm__ void*>(FS_handle + fs_offset +
                                           vid * HalfC * D + r),
                 SINGLE_CACHE_LINE);
          }
        }
        dsb(DSB_ALL);
#endif
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      }
    }
  }
#endif
}
