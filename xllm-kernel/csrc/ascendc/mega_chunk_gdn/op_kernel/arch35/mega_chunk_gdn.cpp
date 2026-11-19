// mega_kernel.cpp — GDN Mega-Kernel (group-value / GQA): all PTO stages in one
// launch
//
// Same pipeline as pto_mega_kernel, but H/Hg are runtime values.
//
// Stages:
//   1. cumsum      (Vec)
//   2. transpose   (Vec)
//   3. kkt         (Cube+Vec)  — K has Hg heads; β,g,A use H value heads
//   4. solve_tril  (Cube)
//   5. wy_fast     (Vec+Cube)
//   6. chunk_h     (Cube+Vec)
//   7. chunk_o     (Cube+Vec)

#ifndef GDN_D
#define GDN_D 128
#endif
#ifndef GDN_C
#define GDN_C 128
#endif
#ifndef GDN_MAX_HEADS
#define GDN_MAX_HEADS 64
#endif
#ifndef GDN_H_WORKSPACE_PAD_BYTES
#define GDN_H_WORKSPACE_PAD_BYTES 512
#endif
#ifndef GDN_H_WORKSPACE_ALIGNMENT_BYTES
#define GDN_H_WORKSPACE_ALIGNMENT_BYTES (16 * 1024 * 1024)
#endif
#ifndef GDN_H_WORKSPACE_PHASE_BYTES
#define GDN_H_WORKSPACE_PHASE_BYTES 0
#endif
#ifndef MEMORY_BASE
#define MEMORY_BASE
#endif
#ifndef GDN_KERNEL_NAME
#define GDN_KERNEL_NAME launch_mega_kernel
#endif
// Note the codegen parser does not support arguments of form "type *name", only
// "type* name"
// clang-format off
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
// clang-format off

#include "acl/acl.h"
#include "kernel_operator.h"
#include <pto/pto-inst.hpp>
#include "mega_chunk_gdn/op_kernel/gdn_sync.h"
#include <type_traits>

#ifdef MEGA_CHUNK_GDN_HELPER_NAMESPACE
namespace MEGA_CHUNK_GDN_HELPER_NAMESPACE {
#endif

using namespace pto;

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#define GDN_A5_KERNEL
#endif

#if defined(__DAV_C310_CUBE__)
#define GDN_A5_CUBE_KERNEL
#elif defined(__DAV_C310_VEC__)
#define GDN_A5_VECTOR_KERNEL
#endif

#ifndef GDN_COMPUTE_DTYPE
#define GDN_COMPUTE_DTYPE DTYPE_Q
#define GDN_COMPUTE_DTYPE_DEFAULTED
#endif

using ComputeT = GDN_COMPUTE_DTYPE;

static_assert(std::is_same_v<ComputeT, half> || std::is_same_v<ComputeT, bfloat16_t>,
              "XllmKernelMegaChunkGdn supports FP16 or BF16 compute tensors.");

#ifndef GDN_PUBLIC_DTYPE
#define GDN_PUBLIC_DTYPE DTYPE_Q
#endif

static_assert(std::is_same_v<GDN_PUBLIC_DTYPE, half> ||
                  std::is_same_v<GDN_PUBLIC_DTYPE, bfloat16_t>,
              "XllmKernelMegaChunkGdn supports FP16 or BF16 public tensors.");

struct MegaChunkGdnKernelTilingData {
    uint32_t block_dim;
    uint32_t num_matrices;
    uint32_t num_heads;
    uint32_t num_key_heads;
    int64_t has_initial_state;
    int64_t batch_size;
    int64_t seq_len;
    int64_t total_tokens;
    uint64_t ffts_addr;
};

// ===================================================================
// Device-only helpers (shared with standard mega-kernel)
// ===================================================================
#ifdef __CCE_AICORE__

// Sync flag ids are provided by PTO. Keep this kernel on the shared sync
// contract so updates to the runtime synchronization framework stay aligned.
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

AICORE inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

template <bool isAIVOnly = true>
AICORE inline void SyncAllImpl()
{
#if defined(GDN_A5_KERNEL)
    static_assert(!isAIVOnly,
                  "A5 MegaGDN uses the complete 1:2 MIX launch for sync.");
    // Match the validated PTO MegaGDN A5 protocol. The primitive drains local
    // stores, gathers both AIV siblings at their AIC, synchronizes all AICs,
    // and releases both AIVs.
    pto::SYNCALL<pto::SyncCoreType::Mix>();
#else
    pipe_barrier(PIPE_ALL);
    if constexpr (isAIVOnly) {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
        return;
    }
#if defined(__DAV_C220_CUBE__)
    wait_flag_dev(SYNC_AIV_FLAG);
    ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
    wait_flag_dev(SYNC_AIC_FLAG);
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_C220_VEC__)
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
    wait_flag_dev(SYNC_AIC_AIV_FLAG);
#endif
#endif
}

#if defined(GDN_A5_KERNEL)
// Scalar conversions between a 2-byte compute type and float.  The C310
// backend has no scalar bf16<->fp32 convert instruction ("not support bf16
// type cast"), so BF16 goes through bit manipulation; FP16 keeps the native
// scalar cast.  bf16->fp32 is an exact 16-bit left shift; fp32->bf16 uses
// round-to-nearest-even like the hardware vector TCVT.
template <typename Q>
AICORE inline float GdnA5ToF32(Q value)
{
    if constexpr (std::is_same_v<Q, bfloat16_t>) {
        uint16_t bits;
        __builtin_memcpy(&bits, &value, sizeof(bits));
        const uint32_t fbits = static_cast<uint32_t>(bits) << 16;
        float result;
        __builtin_memcpy(&result, &fbits, sizeof(result));
        return result;
    } else {
        return static_cast<float>(value);
    }
}

template <typename Q>
AICORE inline Q GdnA5FromF32(float value)
{
    if constexpr (std::is_same_v<Q, bfloat16_t>) {
        uint32_t bits;
        __builtin_memcpy(&bits, &value, sizeof(bits));
        bits += 0x7FFFu + ((bits >> 16) & 1u);  // round-to-nearest-even
        const uint16_t upper = static_cast<uint16_t>(bits >> 16);
        Q result;
        __builtin_memcpy(&result, &upper, sizeof(result));
        return result;
    } else {
        return static_cast<Q>(value);
    }
}
#endif

template <typename T>
AICORE inline void mega_transpose_TH_to_HT(__gm__ T *src, __gm__ T *dst, int64_t T_len, int32_t H)
{
#if defined(__DAV_C220_VEC__) || defined(GDN_A5_VECTOR_KERNEL)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

#if defined(GDN_A5_VECTOR_KERNEL)
    const int64_t cid = static_cast<int64_t>(get_block_idx());
    const int64_t block_num = static_cast<int64_t>(get_block_num());
    constexpr int32_t BLOCK = 128;
    constexpr int32_t HEAD_TILE = 16;
    constexpr int32_t ES = static_cast<int32_t>(sizeof(T));
    constexpr int32_t SRC_UB = 0;
    constexpr int32_t DST_UB = SRC_UB + BLOCK * HEAD_TILE * ES;
    constexpr int32_t TMP_UB = DST_UB + HEAD_TILE * BLOCK * ES;

    using UBSrc =
        Tile<TileType::Vec, T, BLOCK, HEAD_TILE, BLayout::RowMajor, BLOCK, HEAD_TILE,
             SLayout::NoneBox, 512, PadValue::Zero>;
    using UBSrcDyn =
        Tile<TileType::Vec, T, BLOCK, HEAD_TILE, BLayout::RowMajor, DYNAMIC, DYNAMIC,
             SLayout::NoneBox, 512, PadValue::Zero>;
    using UBDst =
        Tile<TileType::Vec, T, HEAD_TILE, BLOCK, BLayout::RowMajor, HEAD_TILE, BLOCK,
             SLayout::NoneBox, 512>;
    using UBTmp =
        Tile<TileType::Vec, T, BLOCK, HEAD_TILE, BLayout::RowMajor, BLOCK, HEAD_TILE,
             SLayout::NoneBox, 512>;
    using UBRowDyn =
        Tile<TileType::Vec, T, 1, BLOCK, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using Gm2D = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmSrcS = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using GmS1 = pto::Stride<1, 1, 1, 1, 1>;
    GmSrcS src_stride(H);

    UBSrc ub_src;
    UBDst ub_dst;
    UBTmp ub_tmp;
    TASSIGN(ub_src, SRC_UB);
    TASSIGN(ub_dst, DST_UB);
    TASSIGN(ub_tmp, TMP_UB);

    // The caller performs SyncAll after both transposes, so each MIX block can
    // own a grid-stride subset of token tiles and publish all heads for its tiles.
    for (int64_t t0 = cid * BLOCK; t0 < T_len;
         t0 += block_num * BLOCK) {
        const int32_t valid =
            (t0 + BLOCK <= T_len) ? BLOCK : static_cast<int32_t>(T_len - t0);
        for (int32_t head_base = 0; head_base < H; head_base += HEAD_TILE) {
            const int32_t tile_heads =
                (head_base + HEAD_TILE <= H) ? HEAD_TILE : (H - head_base);
            Gm2D src_shape;
            src_shape.shape[3] = valid;
            src_shape.shape[4] = tile_heads;
            GlobalTensor<T, Gm2D, GmSrcS> src_gm(
                src + t0 * H + head_base, src_shape, src_stride);
            UBSrcDyn load(valid, tile_heads);
            TASSIGN(load, SRC_UB);
            TLOAD(load, src_gm);
            // A partial tile must not be padded on the Vector pipe until the
            // preceding MTE2 load has finished writing SRC_UB.  Without this
            // dependency, odd head counts can race in TFILLPAD_INPLACE and
            // make beta_t (and every downstream solve output) nondeterministic.
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            if (valid != BLOCK || tile_heads != HEAD_TILE) {
                TFILLPAD_INPLACE(ub_src, load);
            }

            TTRANS(ub_dst, ub_src, ub_tmp);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            for (int32_t h = head_base; h < head_base + tile_heads; ++h) {
                Gm1D dst_shape;
                dst_shape.shape[4] = valid;
                GlobalTensor<T, Gm1D, GmS1> dst_gm(
                    dst + static_cast<int64_t>(h) * T_len + t0, dst_shape);
                UBRowDyn store(1, valid);
                TASSIGN(store, DST_UB + (h - head_base) * BLOCK * ES);
                TSTORE(dst_gm, store);
            }
            // TSTORE still reads DST_UB while the next TTRANS would overwrite
            // it, so order MTE3 before the next Vector operation.
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            // TTRANS reads SRC_UB on the Vector pipe.  Keep the next TLOAD
            // (including the following beta transpose call) from overwriting
            // SRC_UB until that read is complete.
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
    }
    dsb(DSB_ALL);
#else
    auto cid = get_block_idx();
    auto block_num = get_block_num();

    constexpr int32_t BLOCK = 128;
    constexpr int32_t ES = static_cast<int32_t>(sizeof(T));
    constexpr int32_t MinTransposeCols = 16;
    constexpr int32_t AlignElems = ((32 / ES) > MinTransposeCols) ? (32 / ES) : MinTransposeCols;
    constexpr int32_t HP = ((GDN_MAX_HEADS + AlignElems - 1) / AlignElems) * AlignElems;
    constexpr int32_t SRC_UB = 0;
    constexpr int32_t DST_UB = SRC_UB + BLOCK * HP * ES;
    constexpr int32_t TMP_UB = DST_UB + HP * BLOCK * ES;

    using UBSrcFull =
        Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, BLOCK, HP, SLayout::NoneBox, 512, PadValue::Zero>;
    using UBSrcDyn =
        Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    using UBDst = Tile<TileType::Vec, T, HP, BLOCK, BLayout::RowMajor, HP, BLOCK, SLayout::NoneBox, 512>;
    using UBDstDyn = Tile<TileType::Vec, T, HP, BLOCK, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using UBTmp = Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, BLOCK, HP, SLayout::NoneBox, 512>;

    using UBRow = Tile<TileType::Vec, T, 1, BLOCK, BLayout::RowMajor, 1, BLOCK, SLayout::NoneBox, 512>;
    using UBRowDyn = Tile<TileType::Vec, T, 1, BLOCK, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;

    using Gm2D = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmSrcS = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using GmS1 = pto::Stride<1, 1, 1, 1, 1>;
    GmSrcS src_stride(H);

    UBSrcFull ub_src;
    TASSIGN(ub_src, SRC_UB);
    UBDst ub_dst;
    TASSIGN(ub_dst, DST_UB);
    UBTmp ub_tmp;
    TASSIGN(ub_tmp, TMP_UB);

    int64_t num_tok_blocks = (T_len + BLOCK - 1) / BLOCK;

#if defined(GDN_A5_KERNEL)
    // All blocks materialize the small transposed gate tensors. The following
    // stages keep head h on block h % block_num, so this removes the need for
    // a global barrier between a token-block transpose producer and a different
    // head owner. Concurrent stores are byte-identical.
    const int64_t block_begin = 0;
    const int64_t block_stride = 1;
#else
    const int64_t block_begin = static_cast<int64_t>(cid);
    const int64_t block_stride = static_cast<int64_t>(block_num);
#endif
    for (int64_t bi = block_begin; bi < num_tok_blocks; bi += block_stride) {
        int64_t t0 = bi * BLOCK;
        int32_t valid = (t0 + BLOCK <= T_len) ? BLOCK : static_cast<int32_t>(T_len - t0);

        {
            Gm2D gs;
            gs.shape[3] = valid;
            gs.shape[4] = H;
            GlobalTensor<T, Gm2D, GmSrcS> gm(src + t0 * H, gs, src_stride);
            UBSrcDyn ld(valid, H);
            TASSIGN(ld, SRC_UB);
            TLOAD(ld, gm);
            if (valid != BLOCK || H != HP) TFILLPAD_INPLACE(ub_src, ld);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TTRANS(ub_dst, ub_src, ub_tmp);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

#if defined(GDN_A5_KERNEL)
        const int32_t head_begin = static_cast<int32_t>(cid);
        const int32_t head_stride = static_cast<int32_t>(block_num);
#else
        const int32_t head_begin = 0;
        const int32_t head_stride = 1;
#endif
        for (int32_t h = head_begin; h < H; h += head_stride) {
            Gm1D gs;
            gs.shape[4] = valid;
            GlobalTensor<T, Gm1D, GmS1> gm(dst + h * T_len + t0, gs);
            UBRowDyn st(1, valid);
            TASSIGN(st, DST_UB + h * BLOCK * ES);
            TSTORE(gm, st);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
#endif
#endif
}

template <typename SrcT, typename DstT>
AICORE inline void mega_cast_elements(__gm__ SrcT *src, __gm__ DstT *dst,
                                      int64_t element_count)
{
#if defined(__DAV_C220_VEC__)
    static_assert(!std::is_same_v<SrcT, DstT>,
                  "mega_cast_elements requires distinct types.");
    // Keep one vector subblock per 1C2V core. The AscendC queues below own
    // their UB storage, while block_idx already identifies the physical core.
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    const int64_t cid = static_cast<int64_t>(get_block_idx());
    const int64_t block_num = static_cast<int64_t>(get_block_num());
    constexpr int32_t TILE_ELEMENTS = 4096;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_buffer;
    pipe.InitBuffer(input_queue, 1, TILE_ELEMENTS * sizeof(SrcT));
    pipe.InitBuffer(output_queue, 1, TILE_ELEMENTS * sizeof(DstT));
    pipe.InitBuffer(fp32_buffer, TILE_ELEMENTS * sizeof(float));

    AscendC::GlobalTensor<SrcT> src_gm;
    AscendC::GlobalTensor<DstT> dst_gm;
    src_gm.SetGlobalBuffer(src);
    dst_gm.SetGlobalBuffer(dst);

    const int64_t block_stride = block_num * TILE_ELEMENTS;
    for (int64_t offset = cid * TILE_ELEMENTS; offset < element_count;
         offset += block_stride) {
        const int32_t valid = static_cast<int32_t>(
            offset + TILE_ELEMENTS <= element_count
                ? TILE_ELEMENTS
                : element_count - offset);
        AscendC::DataCopyExtParams input_params(
            1, valid * sizeof(SrcT), 0, 0, 0);
        AscendC::DataCopyPadExtParams<SrcT> input_pad(false, 0, 0,
                                                       SrcT(0));
        auto src_local = input_queue.AllocTensor<SrcT>();
        AscendC::DataCopyPad(src_local, src_gm[offset], input_params,
                            input_pad);
        input_queue.EnQue(src_local);
        src_local = input_queue.DeQue<SrcT>();

        auto dst_local = output_queue.AllocTensor<DstT>();
        auto fp32_local = fp32_buffer.Get<float>();

        if constexpr (std::is_same_v<SrcT, float>) {
            AscendC::Cast(dst_local, src_local,
                          AscendC::RoundMode::CAST_RINT, valid);
        } else if constexpr (std::is_same_v<DstT, float>) {
            AscendC::Cast(dst_local, src_local,
                          AscendC::RoundMode::CAST_NONE, valid);
        } else {
            AscendC::Cast(fp32_local, src_local,
                          AscendC::RoundMode::CAST_NONE, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(dst_local, fp32_local,
                          AscendC::RoundMode::CAST_RINT, valid);
        }

        output_queue.EnQue(dst_local);
        dst_local = output_queue.DeQue<DstT>();
        AscendC::DataCopyExtParams output_params(
            1, valid * sizeof(DstT), 0, 0, 0);
        AscendC::DataCopyPad(dst_gm[offset], dst_local, output_params);
        output_queue.FreeTensor(dst_local);
        input_queue.FreeTensor(src_local);
    }
#else
    (void)src;
    (void)dst;
    (void)element_count;
#endif
}

template <typename SrcT, typename DstT>
AICORE inline void mega_prepare_solve_constants(
    __gm__ SrcT *src, __gm__ DstT *dst, int64_t element_count)
{
#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    static_assert(!std::is_same_v<SrcT, DstT>,
                  "mega_prepare_solve_constants requires distinct types.");
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    const int64_t cid = static_cast<int64_t>(get_block_idx());
    const int64_t block_num = static_cast<int64_t>(get_block_num());
    constexpr int32_t TILE_ELEMENTS = 4096;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_buffer;
    pipe.InitBuffer(input_queue, 1, TILE_ELEMENTS * sizeof(SrcT));
    pipe.InitBuffer(output_queue, 1, TILE_ELEMENTS * sizeof(DstT));
    pipe.InitBuffer(fp32_buffer, TILE_ELEMENTS * sizeof(float));

    AscendC::GlobalTensor<SrcT> src_gm;
    AscendC::GlobalTensor<DstT> dst_gm;
    src_gm.SetGlobalBuffer(src);
    dst_gm.SetGlobalBuffer(dst);

    const int64_t block_stride = block_num * TILE_ELEMENTS;
    for (int64_t offset = cid * TILE_ELEMENTS; offset < element_count;
         offset += block_stride) {
        const int32_t valid = static_cast<int32_t>(
            offset + TILE_ELEMENTS <= element_count
                ? TILE_ELEMENTS
                : element_count - offset);
        AscendC::DataCopyExtParams input_params(
            1, valid * sizeof(SrcT), 0, 0, 0);
        AscendC::DataCopyPadExtParams<SrcT> input_pad(false, 0, 0,
                                                       SrcT(0));
        auto src_local = input_queue.AllocTensor<SrcT>();
        AscendC::DataCopyPad(src_local, src_gm[offset], input_params,
                            input_pad);
        input_queue.EnQue(src_local);
        src_local = input_queue.DeQue<SrcT>();

        auto fp32_local = fp32_buffer.Get<float>();
        AscendC::DataCopyExtParams output_params(
            1, valid * sizeof(DstT), 0, 0, 0);

        auto negative_local = output_queue.AllocTensor<DstT>();
        if constexpr (std::is_same_v<SrcT, float>) {
            AscendC::Cast(negative_local, src_local,
                          AscendC::RoundMode::CAST_RINT, valid);
        } else if constexpr (std::is_same_v<DstT, float>) {
            AscendC::Cast(negative_local, src_local,
                          AscendC::RoundMode::CAST_NONE, valid);
        } else {
            AscendC::Cast(fp32_local, src_local,
                          AscendC::RoundMode::CAST_NONE, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(negative_local, fp32_local,
                          AscendC::RoundMode::CAST_RINT, valid);
        }
        output_queue.EnQue(negative_local);
        negative_local = output_queue.DeQue<DstT>();
        AscendC::DataCopyPad(dst_gm[offset], negative_local, output_params);
        output_queue.FreeTensor(negative_local);

        auto identity_local = output_queue.AllocTensor<DstT>();
        if constexpr (std::is_same_v<SrcT, float>) {
            AscendC::Cast(identity_local, src_local,
                          AscendC::RoundMode::CAST_RINT, valid);
        } else if constexpr (std::is_same_v<DstT, float>) {
            AscendC::Cast(identity_local, src_local,
                          AscendC::RoundMode::CAST_NONE, valid);
        } else {
            AscendC::Cast(identity_local, fp32_local,
                          AscendC::RoundMode::CAST_RINT, valid);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(identity_local, identity_local, DstT(-1), valid);
        output_queue.EnQue(identity_local);
        identity_local = output_queue.DeQue<DstT>();
        AscendC::DataCopyPad(dst_gm[element_count + offset], identity_local,
                            output_params);
        output_queue.FreeTensor(identity_local);

        auto zero_local = output_queue.AllocTensor<DstT>();
        AscendC::Duplicate(zero_local, DstT(0), valid);
        output_queue.EnQue(zero_local);
        zero_local = output_queue.DeQue<DstT>();
        AscendC::DataCopyPad(dst_gm[2 * element_count + offset], zero_local,
                            output_params);
        output_queue.FreeTensor(zero_local);
        input_queue.FreeTensor(src_local);
    }
#else
    (void)src;
    (void)dst;
    (void)element_count;
#endif
}

template <int32_t H, int32_t C>
AICORE inline void mega_cast_fp32_to_dtype_bsnd(__gm__ float *src, __gm__ ComputeT *dst, uint32_t num_matrices,
                                               int64_t total_tokens)
{
#if defined(__DAV_C220_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    auto cid = get_block_idx();
    auto block_num = get_block_num();

    constexpr int32_t F32_UB = 0;
    constexpr int32_t F16_UB = C * static_cast<int32_t>(sizeof(float));

    using SrcUB = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, 1, C, SLayout::NoneBox, 512, PadValue::Zero>;
    using DynSrcUB =
        Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    using DstUB = Tile<TileType::Vec, ComputeT, 1, C, BLayout::RowMajor, 1, C, SLayout::NoneBox, 512>;
    using DynDstUB = Tile<TileType::Vec, ComputeT, 1, C, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmS1 = Stride<1, 1, 1, 1, 1>;

    SrcUB src_ub;
    TASSIGN(src_ub, F32_UB);
    DstUB dst_ub;
    TASSIGN(dst_ub, F16_UB);

    for (uint32_t m = cid; m < num_matrices; m += block_num) {
        uint32_t h = m % static_cast<uint32_t>(H);
        uint32_t chunk_idx = m / static_cast<uint32_t>(H);

        for (int64_t t = 0; t < total_tokens; ++t) {
            int64_t off = t * static_cast<int64_t>(H * C) + static_cast<int64_t>(h * C);

            {
                Gm1D gs;
                gs.shape[4] = C;
                GlobalTensor<float, Gm1D, GmS1> gm(src + off, gs);
                SrcUB ld;
                TASSIGN(ld, F32_UB);
                TLOAD(ld, gm);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TCVT(dst_ub, src_ub, RoundMode::CAST_NONE);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
                Gm1D gs;
                gs.shape[4] = C;
                GlobalTensor<ComputeT, Gm1D, GmS1> gm(dst + off, gs);
                DstUB st;
                TASSIGN(st, F16_UB);
                TSTORE(gm, st);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
    }
#endif
}

#endif  // __CCE_AICORE__

// ===================================================================
// Include original kernel implementations in separate namespaces.
// ===================================================================

// The original GDN kernels use the C220 guards to split Cube and Vector
// compilation. C310 keeps the same split, but uses intra-block events instead
// of the legacy FFTS CV signals. Keep the compatibility aliases local to the
// included GDN implementation and remove them before including CANN headers.
#if defined(GDN_A5_CUBE_KERNEL)
#define __DAV_C220_CUBE__ 1
#define ffts_cross_core_sync(pipe, config) gdn_sync::Signal<pipe>(static_cast<uint16_t>(config))
#define wait_flag_dev(flag_id) gdn_sync::Wait<PIPE_MTE2>(static_cast<uint16_t>(flag_id))
#define pipe_barrier(pipe) gdn_sync::VectorBarrier()
#elif defined(GDN_A5_VECTOR_KERNEL)
#define __DAV_C220_VEC__ 1
#define ffts_cross_core_sync(pipe, config) gdn_sync::Signal<pipe>(static_cast<uint16_t>(config))
#define wait_flag_dev(flag_id) gdn_sync::Wait<PIPE_MTE2>(static_cast<uint16_t>(flag_id))
#define pipe_barrier(pipe) gdn_sync::VectorBarrier()
#endif

namespace mk_cumsum {
using pto::Stride;
#include "mega_chunk_gdn/op_kernel/chunk_cumsum.cpp"
}

namespace mk_kkt {
using pto::Stride;
#include "scaled_dot_kkt.cpp"
}

namespace mk_solve {
using pto::Stride;
#include "tri_inverse_impl.cpp"
}

namespace mk_wy {
using pto::Stride;
#include "wy_fast.cpp"
}

namespace mk_h {
using pto::Stride;
#include "chunk_h.cpp"
}

namespace mk_o {
using pto::Stride;
#include "chunk_o.cpp"
}

#if defined(__DAV_C220_CUBE__)
#define GDN_WY_FAST_CALL wy_fast_kernel_aic
#define GDN_CHUNK_O_CALL chunk_o_kernel_aic
#elif defined(__DAV_C220_VEC__)
#define GDN_WY_FAST_CALL wy_fast_kernel_aiv
#define GDN_CHUNK_O_CALL chunk_o_kernel_aiv
#else
#define GDN_WY_FAST_CALL wy_fast_kernel
#define GDN_CHUNK_O_CALL chunk_o_kernel
#endif

template <bool WaitForKktReady = false, uint32_t ReadyGroupSize = 3>
AICORE inline void mega_solve_tril(__gm__ ComputeT *out, __gm__ ComputeT *in, __gm__ ComputeT *minus_id, uint32_t matrix_size,
                                   uint32_t num_matrices, uint32_t num_bsnd_heads, __gm__ int32_t *cu_seqlens,
                                   uint32_t is_lower,
                                   __gm__ ComputeT *packed_workspace,
                                   bool use_precomputed_m_neg = false,
                                   bool packed_head_major_output = false)
{
#ifdef MEGA_CHUNK_GDN_PRECOMPUTED_SOLVE_AUX
    constexpr bool PrecomputedAuxiliary = true;
#else
    constexpr bool PrecomputedAuxiliary = false;
#endif
#if defined(GDN_A5_KERNEL)
    // A5 launches this fused kernel with one MIX block. Keep one matrix in
    // flight and retain the packed solve workspace required by C310.
    mk_solve::runKernelTriInvRecUnroll<ComputeT, float, GDN_C, 1, true,
                                       ComputeT>(
        out, in, minus_id, num_matrices, num_bsnd_heads, cu_seqlens,
        is_lower, packed_workspace, use_precomputed_m_neg,
        packed_head_major_output);
#else
    if constexpr (WaitForKktReady) {
        mk_solve::runKernelTriInvRecUnroll<ComputeT, float, GDN_C,
                                           ReadyGroupSize,
                                           true, ComputeT, true,
                                           PrecomputedAuxiliary>(
            out, in, minus_id, num_matrices, num_bsnd_heads, cu_seqlens,
            is_lower, packed_workspace, use_precomputed_m_neg);
        return;
    }
#ifdef MEGA_CHUNK_GDN_SOLVE_TILES_PER_ITER
    mk_solve::runKernelTriInvRecUnroll<
        ComputeT, float, GDN_C, MEGA_CHUNK_GDN_SOLVE_TILES_PER_ITER, true,
        ComputeT, false, PrecomputedAuxiliary>(out, in, minus_id,
                        num_matrices, num_bsnd_heads,
                        cu_seqlens, is_lower, packed_workspace,
                        use_precomputed_m_neg);
#else
    if (num_matrices <= get_block_num())
        mk_solve::runKernelTriInvRecUnroll<ComputeT, float, GDN_C, 1, true,
                                           ComputeT, false,
                                           PrecomputedAuxiliary>(
            out, in, minus_id, num_matrices, num_bsnd_heads, cu_seqlens,
            is_lower, packed_workspace, use_precomputed_m_neg);
    else if (num_matrices <= 2u * get_block_num())
        mk_solve::runKernelTriInvRecUnroll<ComputeT, float, GDN_C, 2, true,
                                           ComputeT, false,
                                           PrecomputedAuxiliary>(
            out, in, minus_id, num_matrices, num_bsnd_heads, cu_seqlens,
            is_lower, packed_workspace, use_precomputed_m_neg);
    else
        mk_solve::runKernelTriInvRecUnroll<ComputeT, float, GDN_C, 4, true,
                                           ComputeT, false,
                                           PrecomputedAuxiliary>(
            out, in, minus_id, num_matrices, num_bsnd_heads, cu_seqlens,
            is_lower, packed_workspace, use_precomputed_m_neg);
#endif
#endif
}

template <bool FuseGatedRmsNorm = false,
          bool StoreFinalStateCache = false,
          bool EnableHoPipeline = false,
          bool EnableHoOverlap = true,
          bool LoadInitialStateCache = false,
          bool EnableKktSolvePipeline = false>
AICORE inline void mega_kernel_impl(GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR g_in_ptr, GM_ADDR beta_ptr,
                                    GM_ADDR msk_lower_ptr, GM_ADDR msk_full_ptr, GM_ADDR minus_id_ptr,
                                    GM_ADDR cu_seqlens_ptr, GM_ADDR o_ptr, GM_ADDR g_sum_ptr, GM_ADDR g_t_ptr,
                                    GM_ADDR beta_t_ptr, GM_ADDR A_ptr, GM_ADDR A_inv_f32_ptr, GM_ADDR A_inv_ptr,
                                    GM_ADDR w_ptr, GM_ADDR u_ptr, GM_ADDR s_ptr, GM_ADDR v_new_ptr, GM_ADDR fs_ptr,
                                    GM_ADDR h0_ptr, int64_t has_initial_state, GM_ADDR kkt_ws_ptr,
                                    GM_ADDR wy_ws_a1_ptr, GM_ADDR wy_ws_a2_ptr, GM_ADDR h_ws_ptr,
                                    GM_ADDR o_ws_qk_ptr, GM_ADDR o_ws_qs_ptr, GM_ADDR o_ws_gated_ptr,
                                    int32_t H, uint32_t num_key_heads, int64_t batch_size, int64_t seq_len,
                                    int64_t total_tokens, uint32_t num_matrices, uint64_t ffts_addr,
                                    GM_ADDR z_ptr, GM_ADDR norm_weight_ptr,
                                    GM_ADDR final_state_cache_ptr,
                                    GM_ADDR state_indices_ptr,
                                    int32_t state_index_stride,
                                    GM_ADDR initial_state_cache_ptr,
                                    GM_ADDR initial_state_indices_ptr,
                                    int64_t state_cache_slots)
{
    constexpr int32_t D = GDN_D;
    constexpr int32_t C = GDN_C;

    if (num_key_heads == 0 || (static_cast<uint32_t>(H) % num_key_heads) != 0) {
        return;
    }

#define GDN_STAGE_SYNC() SyncAllImpl<false>()


    mk_cumsum::cumsum_kernel<C>(reinterpret_cast<__gm__ float *>(g_in_ptr),
                                reinterpret_cast<__gm__ float *>(g_sum_ptr),
                                reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, H,
                                ffts_addr);
#ifdef MEGA_STOP_AFTER_CUMSUM
    // Profile-only terminal: all MIX participants must retire the same
    // rendezvous before the kernel exits.
    GDN_STAGE_SYNC();
    return;
#endif
#ifdef MEGA_CHUNK_GDN_A5_PUBLISH_INPUTS_BEFORE_WY
#if defined(GDN_A5_VECTOR_KERNEL)
    // Frontend K/V are produced by the AIV side of this fused kernel. Publish
    // every AIV's private cache before the following full MIX rendezvous lets
    // Cube bypass selected AIV packing mailboxes and read inputs from GM.
    dcci((__gm__ void *)0, ENTIRE_DATA_CACHE);
    dsb(DSB_ALL);
#endif
#endif
    GDN_STAGE_SYNC();

    const bool reuse_group_kk =
        mk_kkt::CanReuseGroupKk<C>(
            batch_size, total_tokens, static_cast<uint32_t>(H),
            num_key_heads, num_matrices,
            reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr));
#ifdef MEGA_CHUNK_GDN_BLOCKED_SOLVE
    constexpr bool CanPipelineKktSolve = false;
#else
    constexpr bool CanPipelineKktSolve = true;
#endif
    const uint32_t kkt_solve_group_size =
        static_cast<uint32_t>(H) / num_key_heads;
#if defined(PTO_NPU_ARCH_A2A3)
    const uint32_t kkt_solve_group_count =
        num_matrices / kkt_solve_group_size;
    const uint32_t kkt_solve_producer_waves =
        (kkt_solve_group_count + get_block_num() - 1) / get_block_num();
    // A2/A3 use the four-slot FFTS ready/free protocol. Its eighth producer
    // wave can expose stale KKT tiles, so long sequences use the stage barrier.
    const bool kkt_solve_wave_count_supported =
        kkt_solve_producer_waves <= 7;
#else
    // A5 has a separate intra-block/software-sync contract and must never enter
    // the A2/A3 four-slot FFTS protocol.
    constexpr bool kkt_solve_wave_count_supported = false;
#endif
    const bool use_kkt_solve_pipeline =
        EnableKktSolvePipeline && CanPipelineKktSolve && reuse_group_kk &&
        (static_cast<uint32_t>(H) == 2u * num_key_heads ||
         static_cast<uint32_t>(H) == 3u * num_key_heads) &&
        (num_matrices % kkt_solve_group_size) == 0 &&
        kkt_solve_wave_count_supported;

    mega_transpose_TH_to_HT<float>(reinterpret_cast<__gm__ float *>(g_sum_ptr),
                                   reinterpret_cast<__gm__ float *>(g_t_ptr), total_tokens, H);
    mega_transpose_TH_to_HT<ComputeT>(reinterpret_cast<__gm__ ComputeT *>(beta_ptr),
                                  reinterpret_cast<__gm__ ComputeT *>(beta_t_ptr), total_tokens, H);
    if (reuse_group_kk) {
        mk_kkt::BuildGroupKkCache<D, C>(
            reinterpret_cast<__gm__ ComputeT *>(k_ptr),
            reinterpret_cast<__gm__ ComputeT *>(A_inv_ptr),
            reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr),
            batch_size, num_matrices, static_cast<uint32_t>(H),
            num_key_heads);
    }

    GDN_STAGE_SYNC();

#ifdef MEGA_STOP_AFTER_TRANSPOSE
    return;
#endif

#ifdef MEGA_CHUNK_GDN_A5_PRECOMPUTED_M_NEG
    constexpr bool use_precomputed_m_neg = true;
#elif defined(MEGA_CHUNK_GDN_PRECOMPUTED_M_NEG)
    const bool use_precomputed_m_neg = !reuse_group_kk;
#else
    constexpr bool use_precomputed_m_neg = false;
#endif

    mk_kkt::kkt_kernel<D, C>(
        reinterpret_cast<__gm__ ComputeT *>(k_ptr), reinterpret_cast<__gm__ ComputeT *>(beta_t_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr), reinterpret_cast<__gm__ float *>(msk_lower_ptr),
        reinterpret_cast<__gm__ ComputeT *>(kkt_ws_ptr), reinterpret_cast<__gm__ ComputeT *>(A_ptr),
        reinterpret_cast<__gm__ ComputeT *>(A_inv_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, total_tokens,
        static_cast<uint32_t>(H), num_key_heads, num_matrices, ffts_addr,
        reuse_group_kk ? 1u : 0u,
        use_kkt_solve_pipeline ? 1u : 0u,
        use_precomputed_m_neg ? 1u : 0u);

#if defined(__DAV_C220_CUBE__)
    if (!reuse_group_kk) {
        pipe_barrier(PIPE_ALL);
        wait_flag_dev(2);
        wait_flag_dev(3);
    }
#endif


    if (!use_kkt_solve_pipeline) {
        GDN_STAGE_SYNC();
    }

#ifdef MEGA_STOP_AFTER_KKT
    if (use_kkt_solve_pipeline) {
        GDN_STAGE_SYNC();
    }
    return;
#endif

#if defined(GDN_A5_KERNEL) && \
    defined(MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE)
    // Variant17 remaps the group schedule onto A5's intra-block event range.
    // Limit reuse to the chunk range accepted by chunk O's precomputed-QS
    // path; outside it variant16 remains the correctness fallback.
    const uint32_t group_qk_chunk_count =
        H > 0 && (num_matrices % static_cast<uint32_t>(H)) == 0
            ? num_matrices / static_cast<uint32_t>(H)
            : 0;
    const uint64_t max_group_qk_chunk_count =
        batch_size > 0 ? static_cast<uint64_t>(batch_size) * 64u : 0u;
    const bool reuse_group_qk =
        EnableHoPipeline && H >= 8 && D == C &&
        static_cast<uint32_t>(H) == 3u * num_key_heads &&
        group_qk_chunk_count >= 4 &&
        static_cast<uint64_t>(group_qk_chunk_count) <=
            max_group_qk_chunk_count &&
        mk_o::CanReuseGroupQk<C>(
            batch_size, total_tokens, static_cast<uint32_t>(H),
            num_key_heads, num_matrices,
            reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr));
#elif defined(GDN_A5_KERNEL)
    constexpr bool reuse_group_qk = false;
#else
    const bool reuse_group_qk =
        EnableHoPipeline && H >= 8 && D == C &&
        mk_o::CanReuseGroupQk<C>(
            batch_size, total_tokens, static_cast<uint32_t>(H),
            num_key_heads, num_matrices,
            reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr));
#endif

    __gm__ ComputeT *wy_a_input_ptr =
        reinterpret_cast<__gm__ ComputeT *>(A_inv_ptr);
    constexpr bool use_packed_solve_wy_handoff = false;
#ifdef MEGA_CHUNK_GDN_BLOCKED_SOLVE
    const bool use_blocked_solve =
        batch_size >= 1 && cu_seqlens_ptr != nullptr && H > 0 &&
        num_matrices >= static_cast<uint32_t>(H);
    if (use_blocked_solve) {
        mk_solve::runKernelTriInvBlocked64ResidentInplaceBSND<ComputeT>(
            reinterpret_cast<__gm__ ComputeT *>(A_ptr),
            reinterpret_cast<__gm__ ComputeT *>(minus_id_ptr), num_matrices,
            static_cast<uint32_t>(H),
            reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr));
        wy_a_input_ptr = reinterpret_cast<__gm__ ComputeT *>(A_ptr);
    } else {
#endif
        if constexpr (EnableKktSolvePipeline) {
            if (use_kkt_solve_pipeline) {
                const uint32_t group_size =
                    static_cast<uint32_t>(H) / num_key_heads;
                if (group_size == 2) {
                    mega_solve_tril<true, 2>(
                        reinterpret_cast<__gm__ ComputeT *>(A_inv_ptr),
                        reinterpret_cast<__gm__ ComputeT *>(A_ptr),
                        reinterpret_cast<__gm__ ComputeT *>(minus_id_ptr), C,
                        num_matrices, H,
                    reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr),
                    1, reinterpret_cast<__gm__ ComputeT *>(kkt_ws_ptr),
                    use_precomputed_m_neg, use_packed_solve_wy_handoff);
                } else {
                    mega_solve_tril<true, 3>(
                        reinterpret_cast<__gm__ ComputeT *>(A_inv_ptr),
                        reinterpret_cast<__gm__ ComputeT *>(A_ptr),
                        reinterpret_cast<__gm__ ComputeT *>(minus_id_ptr), C,
                        num_matrices, H,
                        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr),
                        1, reinterpret_cast<__gm__ ComputeT *>(kkt_ws_ptr),
                        use_precomputed_m_neg, use_packed_solve_wy_handoff);
                }
            } else {
                mega_solve_tril<false>(
                    reinterpret_cast<__gm__ ComputeT *>(A_inv_ptr),
                    reinterpret_cast<__gm__ ComputeT *>(A_ptr),
                    reinterpret_cast<__gm__ ComputeT *>(minus_id_ptr), C,
                    num_matrices, H,
                    reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), 1,
                    reinterpret_cast<__gm__ ComputeT *>(kkt_ws_ptr),
                    use_precomputed_m_neg, use_packed_solve_wy_handoff);
            }
        } else {
            mega_solve_tril<false>(
                reinterpret_cast<__gm__ ComputeT *>(A_inv_ptr),
                reinterpret_cast<__gm__ ComputeT *>(A_ptr),
                reinterpret_cast<__gm__ ComputeT *>(minus_id_ptr), C,
                num_matrices, H,
                reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), 1,
                reinterpret_cast<__gm__ ComputeT *>(kkt_ws_ptr),
                use_precomputed_m_neg, use_packed_solve_wy_handoff);
        }
#ifdef MEGA_CHUNK_GDN_BLOCKED_SOLVE
    }
#endif


    GDN_STAGE_SYNC();

#ifdef MEGA_STOP_AFTER_SOLVE
    return;
#endif

#ifdef MEGA_CHUNK_GDN_A5_DUMP_SOLVE_OUTPUT
#if defined(__DAV_C310_VEC__)
    mk_solve::TriInvA5DumpBsndBuffer<ComputeT, C>(
        reinterpret_cast<__gm__ ComputeT *>(o_ptr),
        reinterpret_cast<__gm__ ComputeT *>(A_inv_ptr), num_matrices,
        static_cast<uint32_t>(H),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr));
#endif
    GDN_STAGE_SYNC();
    return;
#endif

#ifndef MEGA_CHUNK_GDN_A5_SINGLE_POST_SOLVE_SYNC
    GDN_STAGE_SYNC();
#endif
#ifdef MEGA_STOP_AFTER_SYNC_BEFORE_WY
    return;
#endif

    mk_wy::GDN_WY_FAST_CALL<D, C>(
        reinterpret_cast<__gm__ ComputeT *>(k_ptr), reinterpret_cast<__gm__ ComputeT *>(v_ptr),
        reinterpret_cast<__gm__ ComputeT *>(beta_t_ptr), reinterpret_cast<__gm__ float *>(g_t_ptr),
        wy_a_input_ptr, reinterpret_cast<__gm__ ComputeT *>(wy_ws_a1_ptr),
        reinterpret_cast<__gm__ ComputeT *>(wy_ws_a2_ptr),
#ifdef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
        reinterpret_cast<__gm__ ComputeT *>(kkt_ws_ptr),
#endif
        reinterpret_cast<__gm__ ComputeT *>(w_ptr),
        reinterpret_cast<__gm__ ComputeT *>(u_ptr), reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len,
        total_tokens, static_cast<uint32_t>(H), num_key_heads, ffts_addr,
        use_packed_solve_wy_handoff);

#if defined(__DAV_C220_VEC__) && !defined(GDN_A5_KERNEL)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
        wait_flag_dev(3);
        wait_flag_dev(4);
    }
#endif

#ifdef MEGA_STOP_AFTER_WY
    // Match every other profile endpoint: close the MIX stage before exit.
    GDN_STAGE_SYNC();
    return;
#endif

    const uint32_t h_o_chunk_count =
        H > 0 && (num_matrices % static_cast<uint32_t>(H)) == 0
            ? num_matrices / static_cast<uint32_t>(H)
            : 0;
    const uint32_t expected_h_o_matrices =
        h_o_chunk_count * static_cast<uint32_t>(H);
#if defined(GDN_A5_KERNEL)
    // Keep the validated A5 full MIX barrier unless an A5-only candidate
    // explicitly enables the per-chunk H/O readiness protocol.
    constexpr bool use_h_o_pipeline = false;
#else
    const bool use_h_o_pipeline =
        EnableHoPipeline && H >= 8 && D == C && batch_size >= 1 &&
        cu_seqlens_ptr != nullptr && h_o_chunk_count >= 4 &&
        h_o_chunk_count <= 64 &&
        num_matrices == expected_h_o_matrices;
#endif
#if defined(GDN_A5_KERNEL) && \
    defined(MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE)
    const bool precompute_qs = use_h_o_pipeline || reuse_group_qk;
#endif
#if defined(__DAV_C220_VEC__) && \
    !defined(MEGA_CHUNK_GDN_A5_O_ORDERED_TWO_BANK)
    if (use_h_o_pipeline && get_subblockid() == 0) {
        constexpr int64_t H_O_READY_STRIDE = 16;
        AscendC::GlobalTensor<int32_t> h_o_ready_gm;
        h_o_ready_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(kkt_ws_ptr));
        const int64_t cid = static_cast<int64_t>(get_block_idx());
        const int64_t block_num = static_cast<int64_t>(get_block_num());
        const int64_t ready_count = batch_size * static_cast<int64_t>(H);
        for (int64_t ready_idx = cid; ready_idx < ready_count;
             ready_idx += block_num) {
            const int64_t ready_offset = ready_idx * H_O_READY_STRIDE;
            h_o_ready_gm.SetValue(ready_offset, 0);
            __asm__ __volatile__("");
            AscendC::DataCacheCleanAndInvalid<
                int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_ALL>(
                h_o_ready_gm[ready_offset]);
            __asm__ __volatile__("");
        }
    }
#endif
    GDN_STAGE_SYNC();

    mk_h::chunk_h_kernel<D, C, StoreFinalStateCache,
                         LoadInitialStateCache>(
        reinterpret_cast<__gm__ ComputeT *>(q_ptr),
        reinterpret_cast<__gm__ ComputeT *>(k_ptr),
        reinterpret_cast<__gm__ ComputeT *>(w_ptr),
        reinterpret_cast<__gm__ ComputeT *>(u_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ ComputeT *>(s_ptr),
        reinterpret_cast<__gm__ ComputeT *>(v_new_ptr),
        reinterpret_cast<__gm__ ComputeT *>(fs_ptr),
        reinterpret_cast<__gm__ ComputeT *>(h0_ptr), has_initial_state, 1,
        reinterpret_cast<__gm__ ComputeT *>(h_ws_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size,
        seq_len, total_tokens, static_cast<uint32_t>(H), num_key_heads,
#if defined(GDN_A5_KERNEL) && \
    defined(MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE)
        precompute_qs ? 1u : 0u,
#else
        use_h_o_pipeline ? 1u : 0u,
#endif
        reinterpret_cast<__gm__ int32_t *>(kkt_ws_ptr), ffts_addr,
        reinterpret_cast<__gm__ float *>(initial_state_cache_ptr),
        reinterpret_cast<__gm__ int32_t *>(initial_state_indices_ptr),
        reinterpret_cast<__gm__ float *>(final_state_cache_ptr),
        reinterpret_cast<__gm__ int32_t *>(state_indices_ptr),
        state_index_stride, state_cache_slots);

#ifdef MEGA_STOP_AFTER_H
    GDN_STAGE_SYNC();
    return;
#endif

    const bool use_h_o_overlap =
        use_h_o_pipeline && EnableHoOverlap
        ;
    if (use_h_o_overlap) {
        pipe_barrier(PIPE_ALL);
    } else {
#if defined(__DAV_C220_VEC__)
        // O consumes H's GM state/workspace. Cross-core rendezvous alone does
        // not acknowledge outstanding MTE3 stores, so drain them before any
        // consumer is released into chunk O.
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
#endif
        GDN_STAGE_SYNC();
    }

    mk_o::GDN_CHUNK_O_CALL<D, C, FuseGatedRmsNorm>(
        reinterpret_cast<__gm__ ComputeT *>(q_ptr), reinterpret_cast<__gm__ ComputeT *>(k_ptr),
        reinterpret_cast<__gm__ ComputeT *>(v_new_ptr), reinterpret_cast<__gm__ ComputeT *>(s_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ float *>(msk_full_ptr),
        reinterpret_cast<__gm__ ComputeT *>(o_ws_qk_ptr), reinterpret_cast<__gm__ ComputeT *>(o_ws_qs_ptr),
        reinterpret_cast<__gm__ ComputeT *>(o_ws_gated_ptr),
        reinterpret_cast<__gm__ ComputeT *>(kkt_ws_ptr),
        reinterpret_cast<__gm__ ComputeT *>(wy_ws_a1_ptr),
        reinterpret_cast<__gm__ ComputeT *>(wy_ws_a2_ptr),
        reuse_group_qk ? 1u : 0u,
        reinterpret_cast<__gm__ GDN_PUBLIC_DTYPE *>(o_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, total_tokens,
        static_cast<uint32_t>(H), num_key_heads,
#if defined(GDN_A5_KERNEL) && \
    defined(MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE)
        precompute_qs ? 1u : 0u,
#else
        use_h_o_pipeline ? 1u : 0u,
#endif
        reinterpret_cast<__gm__ int32_t *>(kkt_ws_ptr), ffts_addr,
        reinterpret_cast<__gm__ GDN_PUBLIC_DTYPE *>(z_ptr),
        reinterpret_cast<__gm__ GDN_PUBLIC_DTYPE *>(norm_weight_ptr));

#undef GDN_STAGE_SYNC

#if defined(__DAV_C220_CUBE__)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
        wait_flag_dev(3);
    }
#endif
}

#undef GDN_WY_FAST_CALL
#undef GDN_CHUNK_O_CALL

#if defined(GDN_A5_KERNEL)
#undef ffts_cross_core_sync
#undef wait_flag_dev
#undef pipe_barrier
#endif
#if defined(GDN_A5_CUBE_KERNEL)
#undef __DAV_C220_CUBE__
#elif defined(GDN_A5_VECTOR_KERNEL)
#undef __DAV_C220_VEC__
#endif

#ifdef MEGA_CHUNK_GDN_HELPER_NAMESPACE
}  // namespace MEGA_CHUNK_GDN_HELPER_NAMESPACE
#endif

#ifndef MEGA_CHUNK_GDN_HELPERS_ONLY
extern "C" __global__ __aicore__ void
GDN_KERNEL_NAME(GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR g_in_ptr, GM_ADDR beta_ptr, GM_ADDR msk_lower_ptr,
                GM_ADDR msk_full_ptr, GM_ADDR minus_id_ptr, GM_ADDR cu_seqlens_ptr, GM_ADDR initial_state_ptr,
                GM_ADDR o_ptr, GM_ADDR g_sum_ptr, GM_ADDR g_t_ptr, GM_ADDR beta_t_ptr, GM_ADDR A_ptr,
                GM_ADDR A_inv_f32_ptr, GM_ADDR A_inv_ptr, GM_ADDR w_ptr, GM_ADDR u_ptr, GM_ADDR s_ptr,
                GM_ADDR v_new_ptr, GM_ADDR fs_ptr, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MegaChunkGdnKernelTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaChunkGdnKernelTilingData, tiling_data, tiling);
    GM_ADDR user_ws = AscendC::GetUserWorkspace(workspace);
    const uint64_t tile_bytes = static_cast<uint64_t>(GDN_C) * GDN_C * sizeof(ComputeT);

    GM_ADDR kkt_ws_ptr = user_ws;
    GM_ADDR wy_ws_a1_ptr = kkt_ws_ptr + static_cast<uint64_t>(tiling_data.block_dim) * 2 * tile_bytes;
    GM_ADDR wy_ws_a2_ptr = wy_ws_a1_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;
    GM_ADDR h_ws_unaligned =
        wy_ws_a2_ptr + static_cast<uint64_t>(tiling_data.block_dim) *
                           tile_bytes;
    const uint64_t h_ws_address =
        (reinterpret_cast<uint64_t>(h_ws_unaligned) +
         GDN_H_WORKSPACE_ALIGNMENT_BYTES - 1) &
        ~(static_cast<uint64_t>(GDN_H_WORKSPACE_ALIGNMENT_BYTES) - 1);
    const uint64_t h_ws_phased_address =
        h_ws_address + GDN_H_WORKSPACE_PHASE_BYTES;
    GM_ADDR h_ws_ptr = reinterpret_cast<GM_ADDR>(h_ws_phased_address);
    GM_ADDR o_ws_qk_ptr =
        h_ws_ptr + static_cast<uint64_t>(tiling_data.block_dim) *
                       4 * (tile_bytes + GDN_H_WORKSPACE_PAD_BYTES);
    GM_ADDR o_ws_qs_ptr = o_ws_qk_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;
    GM_ADDR o_ws_gated_ptr = o_ws_qs_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;
    if (tiling_data.num_heads == 0 || tiling_data.num_heads > GDN_MAX_HEADS) {
        return;
    }

    mega_kernel_impl<false, false>(q_ptr, k_ptr, v_ptr, g_in_ptr, beta_ptr, msk_lower_ptr, msk_full_ptr, minus_id_ptr,
                     cu_seqlens_ptr, o_ptr, g_sum_ptr, g_t_ptr, beta_t_ptr, A_ptr, A_inv_f32_ptr, A_inv_ptr, w_ptr,
                     u_ptr, s_ptr, v_new_ptr, fs_ptr, initial_state_ptr, tiling_data.has_initial_state, kkt_ws_ptr,
                     wy_ws_a1_ptr, wy_ws_a2_ptr, h_ws_ptr, o_ws_qk_ptr, o_ws_qs_ptr, o_ws_gated_ptr,
                     static_cast<int32_t>(tiling_data.num_heads), tiling_data.num_key_heads, tiling_data.batch_size,
                     tiling_data.seq_len, tiling_data.total_tokens, tiling_data.num_matrices, tiling_data.ffts_addr,
                     nullptr, nullptr, nullptr, nullptr, 1,
                     nullptr, nullptr, 0);
}

// The CANN wrapper generated for mixed AIC/AIV kernels calls matmul::clearWorkspace
// after including this source. Keep this include after PTO code so CANN's DYNAMIC
// enum does not collide with pto::DYNAMIC in the kernel templates above.
#include "lib/matmul_intf.h"
#endif

#undef GDN_A5_CUBE_KERNEL
#undef GDN_A5_VECTOR_KERNEL
#undef GDN_A5_TILED_VECTOR_KERNEL
#undef GDN_A5_KERNEL

#ifdef GDN_COMPUTE_DTYPE_DEFAULTED
#undef GDN_COMPUTE_DTYPE_DEFAULTED
#undef GDN_COMPUTE_DTYPE
#endif
