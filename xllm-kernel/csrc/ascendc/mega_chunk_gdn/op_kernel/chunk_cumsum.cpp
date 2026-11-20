// ============================================================================
// chunk_cumsum_kernel.cpp — Prefix sum of gate values G along time dimension
//
// Mathematical operation (per chunk of C tokens, independently per head h):
//   g_sum[t, h] = Σ_{i=0}^{t} g[i, h]    for t = 0 .. valid-1
//
// Input:  g     [total_tokens, H]  float, BSND layout  — raw gate values
// Output: g_sum [total_tokens, H]  float               — cumulative sums
//
// The prefix sum enables downstream kernels to compute exponential decay
// coefficients:  exp(g_sum[i] - g_sum[j])  gives the cumulative gate
// from token j to token i within a chunk.
//
// Architecture: Vec-only kernel (no Cube/GEMM). Single Vec sub-block.
// Pipeline: MTE2(load) → Vec(compute) → MTE3(store), serialized per chunk.
//
// NPU memory hierarchy used:
//   GM (Global Memory) → UB (Unified Buffer, on-chip SRAM, Vec-accessible)
//
// ─── PTO / NPU Primer for This Kernel ──────────────────────────────────────
//
// AI Core: The basic processing unit of an NPU, analogous to a Streaming
//   Multiprocessor (SM) on a GPU. A single chip has many AI cores, and each
//   core runs the same kernel code on different data (SPMD model).
//
// Memory hierarchy (outer → inner):
//   GM  (Global Memory) — Off-chip DRAM, like GPU HBM. Large (several GB)
//       but high latency. All AI cores share GM.
//   UB  (Unified Buffer) — On-chip SRAM, ~256 KB per AI core. Like GPU
//       shared memory. Very fast, but small. The Vec engine can only operate
//       on data that lives in UB, so every tensor must be DMA'd in first.
//
// Hardware pipes (execute in parallel, like independent GPU warps):
//   Vec   — SIMD vector processor. Performs element-wise math (add, mul, etc.)
//           on data already in UB. Think of it as a wide SIMD ALU.
//   MTE2  — DMA engine for loads: copies data from GM → UB.
//   MTE3  — DMA engine for stores: copies data from UB → GM.
//   Cube  — Matrix engine for GEMMs (not used in this kernel).
//
// Synchronization (set_flag / wait_flag):
//   Because Vec, MTE2, and MTE3 run in parallel on separate hardware, you
//   must explicitly synchronize them to ensure data is ready:
//     set_flag(SRC_PIPE, DST_PIPE, event): SRC signals that it is done.
//     wait_flag(SRC_PIPE, DST_PIPE, event): DST blocks until the signal.
//   Example: After MTE2 loads data into UB, Vec must wait_flag before reading
//   it. This is like a fine-grained torch.cuda.synchronize() between pipes.
//   Events (EVENT_ID0 .. EVENT_ID7) are semaphore indices.
//
// ============================================================================

#include <pto/pto-inst.hpp>

#include "acl/acl.h"
#include "gdn_sync.h"
using namespace pto;

// GDN_C: Compile-time chunk size injected by the build system.
// Using compile-time constants allows the compiler to optimize tile sizes,
// unroll loops, and compute UB addresses at compile time.
#ifndef GDN_C
#define GDN_C 128
#endif

// GDN_MAX_HEADS: compile-time ceiling on the value-head count. NumHeads is now
// a RUNTIME argument (one .so serves every head count), but the UB tiles must
// still be sized at compile time, so they are sized for the worst case
// HTC=align(MAX,8). Any NumHeads <= GDN_MAX_HEADS works; the unused columns are
// zero-padded and ride along the SIMD lanes harmlessly. Must match the
// host-side _MAX_HEADS guard.
#ifndef GDN_MAX_HEADS
#define GDN_MAX_HEADS 64
#endif

// ── Recurrent prefix-sum chain synchronization ──────────────────────────────
// The recurrent chain alternates TADD(acc,acc,g_i) (Vec) and TMOV(s_i,acc)
// (Vec) with TASSIGN address/descriptor setup on each row. TADD/TMOV lower to
// PIPE_V, but TASSIGN is a SCALAR op (PIPE_S, see opPipeList in
// pto/common/event.hpp), and on this core the scalar unit ISSUES the vec
// instructions. So the ordering edge that actually matters is Vec→Scalar:
// without it, the scalar issuer launches the next TADD/TMOV before the previous
// vec write to acc_ub has committed → a RAW race on acc_ub.
//
// pipe_barrier(PIPE_V) only fences Vec-vs-Vec and never enforces this V↔S edge,
// so it is insufficient under the fused kernel — the recurrence races and
// corrupts the last rows of each chunk non-deterministically (surfaced at H=64,
// whose wider tile lengthens the race window). It happens to be harmless
// standalone because the scheduler there doesn't reorder across the boundary
// the same way.
//
// The fix is an explicit Vec→Scalar sync: set_flag(PIPE_V, PIPE_S) + wait_flag,
// here via PtoSetWaitFlag<PIPE_V, PIPE_S>(). This stalls the scalar issuer
// until the vec pipe drains, serializing the recurrence correctly. EVENT_ID2 is
// used to avoid colliding with the EVENT_ID0 V↔MTE3 flags already live in this
// stage.
//
// MINIMAL placement (per-site bisection, H=64, 50-run determinism + e2e +
// partial chunks): the V↔S sync is needed at EXACTLY ONE site — immediately
// after the per-row TADD(acc,acc,g_i) write inside the main loop (in both the
// fixed-length and varlen paths). On A5's tiled path that site uses PIPE_ALL
// because C310 has no raw PIPE_V barrier; the non-tiled path uses the targeted
// V_S event. Every other site (the row-0 prologue, the per-row TMOV(s_i,acc)
// read, and the partial-chunk tail zero-fill) remains pipe_barrier(PIPE_V).
// Verified: bit2-only is correct AND deterministic for full and partial chunks;
// all-PIPE_V or prologue-only-V↔S races. (The post-read TMOV site is an
// equally-valid single choice; we sync after the write because it orders acc
// before every consumer.)
//
// Two things that do NOT work, for the record:
//   • pipe_barrier(PIPE_ALL) also fixes it, but only incidentally — it is a
//   superset
//     fence that happens to include the V↔S edge. It is heavier and obscures
//     the real cause, so we use the targeted V↔S sync instead.
//   • set_flag(PIPE_V, PIPE_MTE3) does NOT help: it targets the wrong pipe pair
//     (store-DMA, not the scalar issuer), leaving the real edge unenforced and
//     producing deterministic-but-WRONG output (~20% frob error) that the
//     run-to-run determinism test cannot detect — always run the cumsum
//     *correctness* test.

// ── PTO type aliases (device-only, guarded by __CCE_AICORE__) ───────────────
// UB tile in row-major (ND) layout, used by Vec engine.
// T=dtype, R×C=static shape, RV×CV=valid region, P=pad value for TLOAD.
//
// Think of UbND as: torch.empty((R, C), dtype=T) allocated in on-chip SRAM
// (UB).
//   - TileType::Vec  = this tile lives in UB, operated on by the Vec (SIMD)
//   engine
//   - BLayout::RowMajor = row-major storage, like C arrays or numpy default
//   - RV, CV = "valid" region within the R×C buffer (for handling partial/tail
//   chunks)
//   - PadValue = what to fill outside the valid region during TLOAD (Zero or
//   Null)
//   - 512 = alignment in bytes (hardware requirement for efficient DMA)
#ifdef __CCE_AICORE__
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

#if defined(GDN_A5_KERNEL)
template <int32_t Rows, int32_t Cols, int32_t SrcAddr, int32_t DstAddr>
AICORE inline void GdnA5ParallelPrefixScan() {
  using Full = UbND<float, Rows, Cols, Rows, Cols, PadValue::Zero>;
  using Dyn = UbND<float, Rows, Cols, DYNAMIC, DYNAMIC, PadValue::Zero>;
  constexpr int32_t RowBytes = Cols * static_cast<int32_t>(sizeof(float));

  for (int32_t step = 1, stage = 0; step < Rows; step <<= 1, ++stage) {
    if ((stage & 1) == 0) {
      Full src;
      Full dst;
      TASSIGN(src, SrcAddr);
      TASSIGN(dst, DstAddr);
      TMOV(dst, src);
      pipe_barrier(PIPE_V);

      Dyn dst_tail(Rows - step, Cols);
      Dyn src_tail(Rows - step, Cols);
      Dyn src_head(Rows - step, Cols);
      TASSIGN(dst_tail, DstAddr + step * RowBytes);
      TASSIGN(src_tail, SrcAddr + step * RowBytes);
      TASSIGN(src_head, SrcAddr);
      TADD(dst_tail, src_tail, src_head);
    } else {
      Full src;
      Full dst;
      TASSIGN(src, DstAddr);
      TASSIGN(dst, SrcAddr);
      TMOV(dst, src);
      pipe_barrier(PIPE_V);

      Dyn dst_tail(Rows - step, Cols);
      Dyn src_tail(Rows - step, Cols);
      Dyn src_head(Rows - step, Cols);
      TASSIGN(dst_tail, SrcAddr + step * RowBytes);
      TASSIGN(src_tail, DstAddr + step * RowBytes);
      TASSIGN(src_head, DstAddr);
      TADD(dst_tail, src_tail, src_head);
    }
    pipe_barrier(PIPE_V);
  }
}
#endif
#endif

template <int32_t ChunkSize>
AICORE void cumsum_kernel(__gm__ float* g_ptr,
                          __gm__ float* g_sum_ptr,
                          __gm__ int32_t* cu_seqlens,
                          int64_t batch_size,
                          int64_t seq_len,
                          int32_t NumHeads,
                          uint64_t ffts_addr) {
  // get_block_idx(): Returns this AI core's index (0..block_num-1).
  //   Like blockIdx.x in CUDA — identifies which core this code runs on.
  // get_block_num(): Total number of AI cores launched (like gridDim.x in
  // CUDA). get_subblockid(): Returns 0 or 1 — selects which Vec sub-block
  // within the core.
  //   Each AI core has 2 Vec sub-blocks that can run in parallel.
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();
  // set_ffts_base_addr(ffts_addr): Configure the base address for FFTS
  // (Fast Fine-grained Task Synchronization) — the cross-core signaling
  // mechanism. Required before any cross-core sync (ffts_cross_core_sync /
  // wait_flag_dev).
  gdn_sync::InitAddress(ffts_addr);

// #if defined(__DAV_C220_VEC__): This block only compiles for the Vec core
// pass. The bisheng compiler makes 3 passes over the same source file:
//   Pass 1: __DAV_C220_VEC__  defined → compiles Vec (SIMD) code
//   Pass 2: __DAV_C220_CUBE__ defined → compiles Cube (matrix) code
//   Pass 3: neither defined → compiles host (CPU) launcher code
// Using these guards lets us put Vec, Cube, and host code in one file.
#if defined(__DAV_C220_VEC__)
  // Keep the second A5 Vector subcore in the caller so it can participate in
  // the following MIX-core stage barrier. Returning here can terminate the
  // inlined caller on C310; guard the work instead.
  if (vid == 0) {
    // set_mask_norm(): Reset Vec mask to normal mode (all lanes active).
    // set_vector_mask(-1, -1): Enable all SIMD lanes (128 lanes for fp32).
    //   The -1 sets all 64 bits to 1 in each of the two 64-bit mask registers.
    //   This is like setting torch's computation to operate on all elements.
    set_mask_norm();
    set_vector_mask(-1, -1);

    // HeadTileCols: worst-case head count rounded up to 8-element alignment
    // (32B for float). NumHeads is a runtime value, so the UB tiles are sized
    // for the compile-time ceiling GDN_MAX_HEADS; the actual NumHeads columns
    // are used and the padding columns (NumHeads..HTC) are zero-filled, so the
    // prefix sum over them stays zero and only rides the SIMD lanes at no
    // correctness cost. Why 8-alignment? The Vec engine processes data in
    // 32-byte granularity; for float (4 bytes) that's 8 elements per SIMD
    // "word".
    constexpr int32_t HTC = ((GDN_MAX_HEADS + 7) / 8) * 8;
    constexpr int32_t BlockBytes =
        ChunkSize * HTC * static_cast<int32_t>(sizeof(float));
    constexpr int32_t RowBytes = HTC * static_cast<int32_t>(sizeof(float));

    // ── UB memory layout ──────────────────────────────────────────────────
    //  [0            .. BlockBytes)     = g input  (ChunkSize × HTC floats)
    //  [BlockBytes   .. 2*BlockBytes)   = g_sum output
    //  [2*BlockBytes .. 2*BlockBytes+RowBytes) = row accumulator (1 × HTC)
    constexpr int32_t GUbAddr = 0;
    constexpr int32_t SUbAddr = BlockBytes;
    constexpr int32_t AccUbAddr = BlockBytes * 2;

    // GlobalTensor types for g/g_sum in [total_tokens, NumHeads] layout.
    // 5D shape with last two dims dynamic; stride encodes row pitch.
    //
    // GlobalTensor is a "view" into GM (Global Memory), like
    // torch.as_strided().
    //   GlobalTensor<dtype, Shape, Stride>(base_ptr, shape)
    // Shape<1,1,1,DYNAMIC,DYNAMIC> = 5D shape where first 3 dims are 1
    // (unused),
    //   last 2 dims are set at runtime (valid rows × NumHeads).
    // pto::Stride<1,1,1,NumHeads,1> = stride between elements. The 4th stride =
    // NumHeads
    //   means consecutive rows in GM are NumHeads elements apart (BSND layout:
    //   token[t] at offset t*NumHeads, head[h] at offset h within that token).
    // This is equivalent to:
    //   g_gm = torch.as_strided(g_ptr, size=[valid, NumHeads],
    //   stride=[NumHeads, 1])
    using GmShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using GmStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using GmFloat = GlobalTensor<float, GmShape, GmStride>;
    // Runtime row pitch = NumHeads elements (BSND: token t at offset
    // t*NumHeads).
    GmStride g_stride(NumHeads);

    // Pre-assign row accumulator at fixed UB address
    // TASSIGN(tile, address): Binds a tile descriptor to a fixed byte address
    // in UB. Think of it as: tile = ub_memory[address:address+sizeof(tile)]
    // This does NOT allocate or move data — it just tells the hardware where
    // the tile lives. We manually manage UB memory layout (like a memory pool)
    // via compile-time addresses.
    UbND<float, 1, HTC> acc_ub;
    TASSIGN(acc_ub, AccUbAddr);

    int64_t num_seqs = batch_size;

    // ── Fixed-length sequence path (cu_seqlens == nullptr) ────────────────
    if (cu_seqlens == nullptr) {
      int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
      int64_t total_chunks = num_seqs * chunks_per_seq;

      // Work distribution: Each AI core processes chunks in a round-robin
      // pattern. Core `cid` handles chunks cid, cid+block_num, cid+2*block_num,
      // ... This is the NPU equivalent of CUDA's grid-stride loop:
      //   for (int i = blockIdx.x; i < total; i += gridDim.x)
      // Stage-level SyncAll in the fused caller makes each block's GM output
      // visible before transpose, so chunks can remain grid-stride partitioned.
      const int64_t chunk_begin = static_cast<int64_t>(cid);
      const int64_t chunk_stride = static_cast<int64_t>(block_num);
      for (int64_t gi = chunk_begin; gi < total_chunks; gi += chunk_stride) {
        int64_t seq_idx = gi / chunks_per_seq;
        int64_t local_chunk = gi % chunks_per_seq;
        int64_t bos = seq_idx * seq_len;
        int64_t chunk_start = bos + local_chunk * ChunkSize;
        int64_t remaining = seq_len - local_chunk * ChunkSize;
        int32_t valid =
            static_cast<int32_t>(remaining < ChunkSize ? remaining : ChunkSize);

        // ── DMA: load g[chunk_start .. +valid] from GM → UB (MTE2 pipe) ──
        // Constructs a GlobalTensor view over the g array, loads into UB,
        // then zero-pads the tail region (rows beyond `valid`, cols beyond
        // NumHeads up to the 8-aligned HTC) so downstream Vec ops see zeros.
        {
          GmShape gs;
          gs.shape[3] = valid;
          gs.shape[4] = NumHeads;
          GmFloat g_gm(g_ptr + chunk_start * NumHeads, gs, g_stride);
          UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC, PadValue::Zero> g_load(
              valid, NumHeads);
          TASSIGN(g_load, GUbAddr);
          // TLOAD(ub_tile, gm_tensor): DMA transfer from GM → UB.
          // Equivalent to: ub_tile[:valid, :NumHeads] = gm_tensor[:valid,
          // :NumHeads] This is an ASYNC operation on the MTE2 pipe — the
          // CPU/Vec engine can do other work while DMA is in progress. You must
          // call set_flag/wait_flag before reading the loaded data.
          TLOAD(g_load, g_gm);
          if (valid != ChunkSize || NumHeads != HTC) {
            UbND<float, ChunkSize, HTC, ChunkSize, HTC, PadValue::Zero> g_pad;
            TASSIGN(g_pad, GUbAddr);
            // TFILLPAD_INPLACE(full_tile, partial_tile): Zero-fills the region
            // outside the valid area of partial_tile. Equivalent to:
            //   full_tile[valid:ChunkSize, :] = 0  # zero rows beyond valid
            //   full_tile[:, NumHeads:HTC] = 0     # zero cols beyond NumHeads
            //   (alignment padding)
            // This ensures downstream Vec operations see clean zeros in padded
            // regions.
            TFILLPAD_INPLACE(g_pad, g_load);
          }
        }
        // ── Synchronization: MTE2 → Vec ────────────────────────────────────
        // set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0): Signal from MTE2 (DMA load
        //   engine) to Vec (SIMD engine) that the DMA transfer is complete.
        // wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0): Vec waits here until MTE2
        //   has set the flag. After this, UB data from TLOAD is safe to read.
        // Think of it as: torch.cuda.synchronize() but fine-grained per pipe.
        // EVENT_ID0 is a semaphore index (0-7 available).
        // MTE2 → Vec sync: wait for DMA load to finish before Vec reads UB
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // ── Vec compute: prefix sum over rows (all H heads in parallel) ───
        // Row 0: acc[h] = g[0,h];  g_sum[0,h] = acc[h]
        UbND<float, 1, HTC> g_row_0;
        TASSIGN(g_row_0, GUbAddr);
        // TMOV(dst, src): Element-wise copy, like dst = src.clone() in UB.
        TMOV(acc_ub, g_row_0);
        pipe_barrier(PIPE_V);

        UbND<float, 1, HTC> s_row_0;
        TASSIGN(s_row_0, SUbAddr);
        TMOV(s_row_0, acc_ub);
        pipe_barrier(PIPE_V);

        // Rows 1..valid-1:  acc[h] += g[i,h];  g_sum[i,h] = acc[h]
        for (int32_t i = 1; i < valid; ++i) {
          UbND<float, 1, HTC> g_row_i;
          TASSIGN(g_row_i, GUbAddr + i * RowBytes);
          // TADD(dst, a, b): Element-wise add, like dst = a + b. All in UB.
          // Operates on all HTC elements in parallel (SIMD).
          // *** The one load-bearing sync (see chain-sync note above):
          // Vec→Scalar after this per-row write to acc_ub, ordering it before
          // the scalar issuer launches the next row. Every other barrier in
          // this chain is plain PIPE_V.
          TADD(acc_ub, acc_ub, g_row_i);
#if defined(GDN_A5_KERNEL)
          pipe_barrier(PIPE_ALL);
#else
          AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
#endif

          UbND<float, 1, HTC> s_row_i;
          TASSIGN(s_row_i, SUbAddr + i * RowBytes);
          TMOV(s_row_i, acc_ub);
          pipe_barrier(PIPE_V);
        }

        // Zero-fill rows beyond valid (tail padding for downstream kernels)
        // TEXPANDS(tile, scalar): Fill entire tile with a scalar value.
        // Equivalent to: tile[:] = scalar  (like torch.full_like(tile, scalar))
        TEXPANDS(acc_ub, 0.0f);
        pipe_barrier(PIPE_V);
        for (int32_t i = valid; i < ChunkSize; ++i) {
          UbND<float, 1, HTC> s_row_i;
          TASSIGN(s_row_i, SUbAddr + i * RowBytes);
          TMOV(s_row_i, acc_ub);
          pipe_barrier(PIPE_V);
        }
        // ── DMA: store g_sum from UB → GM (MTE3 pipe) ────────────────────
        // ── Synchronization: Vec → MTE3 ───────────────────────────────────
        // Vec signals MTE3 that computation is done and UB data is ready to
        // store. MTE3 (DMA store engine) waits for this before reading UB for
        // TSTORE. Without this sync, MTE3 might read stale/partial data from
        // UB. Vec → MTE3 sync: ensure Vec writes to UB are visible before DMA
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        {
          GmShape ss;
          ss.shape[3] = valid;
          ss.shape[4] = NumHeads;
          GmFloat gs_gm(g_sum_ptr + chunk_start * NumHeads, ss, g_stride);
          UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC> s_store(valid,
                                                                NumHeads);
          TASSIGN(s_store, SUbAddr);
          // TSTORE(gm_tensor, ub_tile): DMA transfer from UB → GM.
          // Equivalent to: gm_tensor[:valid, :NumHeads] = ub_tile[:valid,
          // :NumHeads] Async on MTE3 pipe. Must sync (Vec→MTE3) before calling,
          // and sync (MTE3→Vec) after if reusing the same UB region.
          TSTORE(gs_gm, s_store);
        }
        // ── Synchronization: MTE3 → Vec ───────────────────────────────────
        // MTE3 signals Vec that the DMA store is complete and UB can be reused.
        // Vec waits before starting the next iteration's TLOAD into the same UB
        // region. Without this, the next TLOAD could overwrite data still being
        // stored. MTE3 → Vec sync: wait for DMA store before reusing UB next
        // iter
#if defined(GDN_A5_KERNEL)
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
#else
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
#endif
      }
    }
    // ── Variable-length sequence path (cu_seqlens != nullptr) ─────────────
    else {
      int64_t gi = 0;
      for (int64_t si = 0; si < num_seqs; ++si) {
        int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
        int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
        int64_t slen = eos - bos;
        int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

        for (int64_t c = 0; c < nc; ++c) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = bos + c * ChunkSize;
            int64_t remaining = slen - c * ChunkSize;
            int32_t valid = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);

            // Load g chunk from GM → UB, zero-padded
            {
              GmShape gs;
              gs.shape[3] = valid;
              gs.shape[4] = NumHeads;
              GmFloat g_gm(g_ptr + chunk_start * NumHeads, gs, g_stride);
              UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC, PadValue::Zero>
                  g_load(valid, NumHeads);
              TASSIGN(g_load, GUbAddr);
              TLOAD(g_load, g_gm);
              if (valid != ChunkSize || NumHeads != HTC) {
                UbND<float, ChunkSize, HTC, ChunkSize, HTC, PadValue::Zero>
                    g_pad;
                TASSIGN(g_pad, GUbAddr);
                TFILLPAD_INPLACE(g_pad, g_load);
              }
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            // Prefix sum: acc = g[0]; g_sum[0] = acc
            UbND<float, 1, HTC> g_row_0;
            TASSIGN(g_row_0, GUbAddr);
            TMOV(acc_ub, g_row_0);
            pipe_barrier(PIPE_V);

            UbND<float, 1, HTC> s_row_0;
            TASSIGN(s_row_0, SUbAddr);
            TMOV(s_row_0, acc_ub);
            pipe_barrier(PIPE_V);

            // acc += g[i]; g_sum[i] = acc
            for (int32_t i = 1; i < valid; ++i) {
              UbND<float, 1, HTC> g_row_i;
              TASSIGN(g_row_i, GUbAddr + i * RowBytes);
              // The one load-bearing sync (see chain-sync note above):
              // Vec→Scalar after this per-row write to acc_ub. All other
              // barriers here are PIPE_V.
              TADD(acc_ub, acc_ub, g_row_i);
#if defined(GDN_A5_KERNEL)
              pipe_barrier(PIPE_ALL);
#else
              AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
#endif

              UbND<float, 1, HTC> s_row_i;
              TASSIGN(s_row_i, SUbAddr + i * RowBytes);
              TMOV(s_row_i, acc_ub);
              pipe_barrier(PIPE_V);
            }

            // Zero-fill padding rows
            TEXPANDS(acc_ub, 0.0f);
            pipe_barrier(PIPE_V);
            for (int32_t i = valid; i < ChunkSize; ++i) {
              UbND<float, 1, HTC> s_row_i;
              TASSIGN(s_row_i, SUbAddr + i * RowBytes);
              TMOV(s_row_i, acc_ub);
              pipe_barrier(PIPE_V);
            }
            // Store g_sum to GM
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            {
              GmShape ss;
              ss.shape[3] = valid;
              ss.shape[4] = NumHeads;
              GmFloat gs_gm(g_sum_ptr + chunk_start * NumHeads, ss, g_stride);
              UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC> s_store(valid,
                                                                    NumHeads);
              TASSIGN(s_store, SUbAddr);
              TSTORE(gs_gm, s_store);
            }
#if defined(GDN_A5_KERNEL)
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
#else
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
#endif
          }
          gi++;
        }
      }
    }
  }
#endif
}

// ── Device-side kernel entry point ─────────────────────────────────
// extern "C" __global__ AICORE: marks this as an NPU kernel function
//   (like __global__ in CUDA). Each AI core runs one instance of this function.
// Parameters are passed as uint8_t* (raw bytes) and reinterpret_cast'd to
// typed pointers — this is the standard NPU kernel calling convention.
