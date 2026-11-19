/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#ifndef MEMORY_BASE
#define MEMORY_BASE
#endif
#include <pto/pto-inst.hpp>

using namespace pto;

AICORE inline uint32_t CeilDiv(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

#define BSND_OFFSET(tile_id, N, S, D) \
  (((tile_id) / (N)) * (S) * (N) * (D) + ((tile_id) % (N)) * (D))

/*
 * For aligned BSND, tile_id enumerates chunk-major then head-major and maps to
 * a fixed-stride address inside the dense BSND tensor.
 */
AICORE inline uint32_t GetBSNDFixedTileOffset(uint32_t tile_id,
                                              uint32_t num_bsnd_heads,
                                              uint32_t matrix_size) {
  return BSND_OFFSET(tile_id, num_bsnd_heads, matrix_size, matrix_size);
}

/**
 * @brief Struct containing starting address and size of a single tile
 */
struct BSNDVarlenTileInfo {
  uint32_t bsnd_offset; /**< Contains the starting index in the global tensor */
  uint32_t valid_size;  /**< This is the size (num_rows/cols) of the tile */
};

/*
 * For cu_seqlens-based varlen BSND, tile_id still enumerates chunk-major then
 * head-major. We recover the owning sequence by scanning cu_seqlens and
 * counting chunks per sequence.
 */
AICORE inline BSNDVarlenTileInfo GetBSNDVarlenTileInfoFromCuSeqlens(
    uint32_t tile_id,
    uint32_t num_bsnd_heads,
    uint32_t matrix_size,
    __gm__ int32_t* cu_seqlens) {
  const uint32_t head_idx = tile_id % num_bsnd_heads;
  const uint32_t chunk_idx = tile_id / num_bsnd_heads;

  uint32_t seq_start = static_cast<uint32_t>(cu_seqlens[0]);
  uint32_t accumulated_chunks = 0;
  for (uint32_t seq_idx = 0;; ++seq_idx) {
    const uint32_t seq_end = static_cast<uint32_t>(cu_seqlens[seq_idx + 1]);
    const uint32_t seq_len = seq_end - seq_start;
    const uint32_t seq_num_chunks = CeilDiv(seq_len, matrix_size);
    if (chunk_idx < accumulated_chunks + seq_num_chunks) {
      const uint32_t local_chunk_idx = chunk_idx - accumulated_chunks;
      const uint32_t row_start = seq_start + local_chunk_idx * matrix_size;
      const uint32_t valid_size =
          min(static_cast<uint32_t>(seq_end - row_start), matrix_size);
      return {row_start * num_bsnd_heads * matrix_size + head_idx * matrix_size,
              valid_size};
    }
    accumulated_chunks += seq_num_chunks;
    seq_start = seq_end;
  }
}

/*
 * @brief: Takes as input two matrices of size MatrixSize * MatrixSize each.
 * The src matrix lies in L1, while the dst matrix lies either in L0A or L0B.
 * This kernel copies only the diagonal blocks (fractals) of size FractalSize *
 * FractalSize from the src matrix to the dst matrix.
 *
 * @tparam InputT Input data type (fp16).
 * @tparam FractalSize Size of each fractal matrix (diagonal block).
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam SrcL1TileT The actual tile type of the src matrix.
 * @tparam DstL0TileT The actual tile type of the dst matrix.
 *
 * @param src Tile in L1 memory.
 * @param dst Tile in L0A or L0B memory.
 */
template <typename InputT,
          uint32_t FractalSize,
          uint32_t MatrixSize,
          typename SrcL1TileT,
          typename DstL0TileT>
AICORE inline void CopyDiagonalFractalsL1ToL0(SrcL1TileT src, DstL0TileT dst) {
  constexpr uint32_t NumFractals = MatrixSize / FractalSize;
  constexpr bool is_left =
      std::is_same_v<DstL0TileT, TileLeft<InputT, MatrixSize, MatrixSize>>;
#if defined(__DAV_C310__) || defined(__DAV_C310_CUBE__)
  constexpr TileType LeftOrRight = is_left ? TileType::Left : TileType::Right;
  constexpr SLayout InnerLayout =
      is_left ? SLayout::RowMajor : SLayout::ColMajor;
  constexpr BLayout OuterLayout =
      is_left ? BLayout::ColMajor : BLayout::RowMajor;
  using FractalTile = Tile<LeftOrRight,
                           InputT,
                           FractalSize,
                           FractalSize,
                           OuterLayout,
                           FractalSize,
                           FractalSize,
                           InnerLayout,
                           512>;
#else
  using FractalTile = typename std::conditional<
      is_left,
      TileLeft<InputT, FractalSize, FractalSize>,
      TileRight<InputT, FractalSize, FractalSize>>::type;
#endif
  FractalTile fractals[NumFractals];
  const std::uintptr_t starting_address =
      reinterpret_cast<std::uintptr_t>(dst.data());
  for (uint32_t i = 0; i < NumFractals; ++i) {
    TASSIGN(fractals[i],
            starting_address +
                i * FractalSize * (MatrixSize + FractalSize) * sizeof(InputT));
    TEXTRACT(fractals[i], src, i * FractalSize, i * FractalSize);
  }
}

/*
 * @brief: Takes as input two matrices of size MatrixSize * MatrixSize each,
 * and an integer block_size. The src matrix lies in L1, while the dst matrix
 * either in L0A or L0B. This method copies some of the diagonal blocks from the
 * input to the output as follows:
 * - If dst is in L0A (left): copy even diagonal blocks 0, 2, 4, ...
 * - If dst is in L0B (right): copy odd blocks 1, 3, 5, ...
 * Important note: the dst matrix should be initialized to all-zeros before
 * calling this method
 *
 * @tparam InputT Input data type (fp16).
 * @tparam FractalSize Size of each fractal matrix (diagonal block).
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam SrcL1TileT The actual tile type of the src matrix.
 * @tparam DstL0TileT The actual tile type of the dst matrix.
 *
 * @param src Tile in L1 memory.
 * @param dst Tile in L0A or L0B memory.
 * @param block_size Size of diagonal blocks. Needs: block_size >= FractalSize.
 */
template <typename InputT,
          uint32_t FractalSize,
          uint32_t MatrixSize,
          typename SrcL1TileT,
          typename DstL0TileT>
AICORE inline void CopyOddOrEvenBlocksL1ToL0(SrcL1TileT src,
                                             DstL0TileT dst,
                                             uint32_t block_size,
                                             bool swap_parity = false) {
  constexpr bool is_left =
      std::is_same_v<DstL0TileT, TileLeft<InputT, MatrixSize, MatrixSize>>;
  // Default: left→even(0), right→odd(1). swap_parity flips this.
  const uint32_t starting_block_index =
      (is_left ? 0u : 1u) ^ (swap_parity ? 1u : 0u);

  const uint32_t num_blocks = MatrixSize / block_size;
  const uint32_t num_fractals_per_block = block_size / FractalSize;

  // might need fewer fractals if block_size < FractalSize
#if defined(__DAV_C310__) || defined(__DAV_C310_CUBE__)
  constexpr TileType LeftOrRight = is_left ? TileType::Left : TileType::Right;
  constexpr SLayout InnerLayout =
      is_left ? SLayout::RowMajor : SLayout::ColMajor;
  constexpr BLayout OuterLayout =
      is_left ? BLayout::ColMajor : BLayout::RowMajor;
  using FractalTile = Tile<LeftOrRight,
                           InputT,
                           FractalSize,
                           FractalSize,
                           OuterLayout,
                           FractalSize,
                           FractalSize,
                           InnerLayout,
                           512>;
#else
  using FractalTile = typename std::conditional<
      is_left,
      TileLeft<InputT, FractalSize, FractalSize>,
      TileRight<InputT, FractalSize, FractalSize>>::type;
#endif
  FractalTile fractals[MatrixSize / FractalSize];

  const std::uintptr_t starting_address =
      reinterpret_cast<std::uintptr_t>(dst.data());
  for (uint32_t i = 0; i < num_fractals_per_block; ++i) {
    for (uint32_t j = 0; j < num_fractals_per_block; ++j) {
      for (uint32_t b = starting_block_index; b < num_blocks; b += 2) {
#if defined(__DAV_C310__) || defined(__DAV_C310_CUBE__)
        const uint32_t row_stride = is_left ? FractalSize : MatrixSize;
        const uint32_t col_stride = is_left ? MatrixSize : FractalSize;
        const uint32_t offset =
            b * (MatrixSize + FractalSize) * block_size /* block_offset */ +
            j * col_stride * FractalSize /* col_fractal_offset */ +
            i * row_stride * FractalSize /* row_fractal_offset */;
#else
        const uint32_t offset =
            b * (MatrixSize + FractalSize) * block_size /* block_offset */ +
            i * MatrixSize * FractalSize /* col_fractal_offset */ +
            j * FractalSize * FractalSize /* row_fractal_offset */;
#endif
        TASSIGN(fractals[b], starting_address + offset * sizeof(InputT));
        TEXTRACT(fractals[b],
                 src,
                 b * block_size + i * FractalSize,
                 b * block_size + j * FractalSize);
      }
    }
  }
}

/*
 * @brief: Prepares Identity and Zeros matrix.
 *
 * @tparam TileL1AB The type of the input tiles in L1.
 * @tparam TileL0A The type of the input tiles in L0A.
 * @tparam TileL0B The type of the input tiles in L0B.
 * @tparam TileL0C The type of the input tiles in L0C.
 *
 * @param I_neg_l1_tile Tile containing the -I (negative identity) matrix.
 * @param Zero_l1_tile Tile to store the all-zero matrix.
 * @param I_l1_tile Tile to store the identity matrix.
 * @param a_l0_tile Tile in L0A for matmuls.
 * @param b_l0_tile Tile in L0B for matmuls.
 * @param c_l0_tile Tile in L0C for matmuls.
 */
template <typename TileL1AB,
          typename TileL0A,
          typename TileL0B,
          typename TileL0C>
AICORE inline void PrepareAuxiliaryMatrices(TileL1AB I_neg_l1_tile,
                                            TileL1AB Zero_l1_tile,
                                            TileL1AB I_l1_tile,
                                            TileL0A a_l0_tile,
                                            TileL0B b_l0_tile,
                                            TileL0C c_l0_tile) {
  TMOV(a_l0_tile, I_neg_l1_tile);  // a_l0 initialized with I_neg
  TMOV(b_l0_tile, I_neg_l1_tile);  // b_l0 initialized with I_neg
  set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
  wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));

  TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0 contains I
  set_flag(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
  wait_flag(PIPE_M, PIPE_FIX, static_cast<event_t>(0));

  TMOV(I_l1_tile, c_l0_tile);  // I_l1 now contains I
  set_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(0));
  wait_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(0));

  TMOV(b_l0_tile, I_l1_tile);  // b_l0 contains I
  set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
  wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));

  TMATMUL_ACC(c_l0_tile,
              c_l0_tile,
              a_l0_tile,
              b_l0_tile);  // c_l0 contains zeros
  set_flag(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
  wait_flag(PIPE_M, PIPE_FIX, static_cast<event_t>(0));

  TMOV(Zero_l1_tile, c_l0_tile);  // Zeros_l1 now contains zeros
  set_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(0));
  wait_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(0));
}

/*
 * @brief: Inverts a single matrix / tile of the global tensor.
 * The first part of the algorithm inverts the FractalSize * FractalSize
 * diagonal blocks of the input matrix (inv_trick part). The second phase
 * assembles the partial inverses using the cube unig (recursive part).
 *
 * @tparam InputT The type of the input elements.
 * @tparam TileL1AB The type of the input tiles in L1.
 * @tparam TileL0A The type of the input tiles in L0A.
 * @tparam TileL0B The type of the input tiles in L0B.
 * @tparam TileL0C The type of the input tiles in L0C.
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam FractalSize Size of matrix fractals.
 * @tparam NumTilesPerCubeIter How many matrices to load and invert in a single
 * cube iteration.
 *
 * @param X_l1_tile Tile in L1 used for intermediate computations.
 * @param I_l1_tile Tile containing the identity matrix.
 * @param I_neg_l1_tile Tile containing the negative identity matrix.
 * @param M_neg_l1_tile Tile containing the negative input matrix.
 * @param Zero_l1_tile Tile containing the all-zero matrix.
 * @param Y_l1_tile Tile in L1 used for intermediate computations.
 * @param a_l0_tile* Array of two tiles in L0A (for double-buffering).
 * @param b_l0_tile* Array of two tiles in L0B (for double-buffering).
 * @param c_l0_tile* Tile in L0C for matmuls.
 * @param tile_id Index of the current tile (used for sync).
 */
template <typename InputT,
          typename TileL1AB,
          typename TileL0A,
          typename TileL0B,
          typename TileL0C,
          uint32_t MatrixSize,
          uint32_t FractalSize,
          uint32_t NumTilesPerCubeIter>
AICORE inline void InvertSingleTile(TileL1AB X_l1_tile,
                                    TileL1AB I_l1_tile,
                                    TileL1AB I_neg_l1_tile,
                                    TileL1AB M_neg_l1_tile,
                                    TileL1AB Zero_l1_tile,
                                    TileL1AB Y_l1_tile,
                                    TileL0A* a_l0_tile,
                                    TileL0B* b_l0_tile,
                                    TileL0C* c_l0_tile,
                                    const uint32_t tile_id,
                                    const bool swap_parity = false,
                                    const bool use_precomputed_m_neg = false) {
  const event_t event_0 = static_cast<event_t>(tile_id);
  const event_t event_1 = static_cast<event_t>(tile_id + NumTilesPerCubeIter);

  if (!use_precomputed_m_neg) {
    TMOV(b_l0_tile[0], Y_l1_tile);      // b_l0[0] contains M
    TMOV(a_l0_tile[0], I_neg_l1_tile);  // a_l0[0] contains I_neg
    set_flag(PIPE_MTE1, PIPE_M, event_0);
  }
  TMOV(a_l0_tile[1], Zero_l1_tile);
  TMOV(b_l0_tile[1], Zero_l1_tile);
  set_flag(PIPE_MTE1, PIPE_M, event_1);
  wait_flag(PIPE_MTE1, PIPE_M, event_1);
  set_flag(PIPE_M, PIPE_MTE1, event_1);
  wait_flag(PIPE_M, PIPE_MTE1, event_1);
  CopyDiagonalFractalsL1ToL0<InputT, FractalSize, MatrixSize>(
      Y_l1_tile, a_l0_tile[1]);  // a_l0[1] = diag_fractals(M)
  CopyDiagonalFractalsL1ToL0<InputT, FractalSize, MatrixSize>(
      Y_l1_tile, b_l0_tile[1]);  // b_l0[1] = diag_fractals(M)
  set_flag(PIPE_MTE1, PIPE_M, event_1);

  if (!use_precomputed_m_neg) {
    /* First Matmul: event_0 */
    wait_flag(PIPE_MTE1, PIPE_M, event_0);
    TMATMUL(c_l0_tile[0],
            a_l0_tile[0],
            b_l0_tile[0]);  // c_l0[0] contains M_neg
    set_flag(PIPE_M, PIPE_FIX, event_0);
    set_flag(PIPE_M, PIPE_MTE1, event_0);

    wait_flag(PIPE_M, PIPE_FIX, event_0);
    TMOV(M_neg_l1_tile, c_l0_tile[0]);
    set_flag(PIPE_FIX, PIPE_M, event_0);
  } else {
    set_flag(PIPE_M, PIPE_MTE1, event_0);
    set_flag(PIPE_FIX, PIPE_M, event_0);
  }

  /* Second Matmul: event_1 */
  wait_flag(PIPE_MTE1, PIPE_M, event_1);
  set_flag(PIPE_MTE1, PIPE_M, event_1);
  TMATMUL(c_l0_tile[1],
          a_l0_tile[1],
          b_l0_tile[1]);  // c_l0[1] contains diag_fractals(M)^2
  set_flag(PIPE_M, PIPE_FIX, event_1);
  wait_flag(PIPE_M, PIPE_FIX, event_1);
  TMOV(Y_l1_tile,
       c_l0_tile[1]);  // Y_l1 now contains diag_fractals(M)^2
  set_flag(PIPE_FIX, PIPE_M, event_1);
  wait_flag(PIPE_FIX, PIPE_M, event_1);

  /* Third Matmul: event_0*/
  wait_flag(PIPE_M, PIPE_MTE1, event_0);
  if (use_precomputed_m_neg) {
    TMOV(b_l0_tile[0], I_l1_tile);
    TMOV(a_l0_tile[0], I_l1_tile);
  } else {
    TMOV(b_l0_tile[0], I_neg_l1_tile);
    TMOV(a_l0_tile[0], I_neg_l1_tile);
  }
  set_flag(PIPE_MTE1, PIPE_M, event_0);

  wait_flag(PIPE_MTE1, PIPE_M, event_0);
  wait_flag(PIPE_FIX, PIPE_M, event_0);
  wait_flag(PIPE_MTE1, PIPE_M, event_1);
  TMATMUL(c_l0_tile[0],
          a_l0_tile[1],
          b_l0_tile[0]);  // c_l0[0] = diag_fractals(M_neg)
  set_flag(PIPE_M, PIPE_FIX, event_0);
  wait_flag(PIPE_M, PIPE_FIX, event_0);
  set_flag(PIPE_FIX, PIPE_M, event_0);
  wait_flag(PIPE_FIX, PIPE_M, event_0);

  TMATMUL_ACC(c_l0_tile[0],
              c_l0_tile[0],
              a_l0_tile[0],
              b_l0_tile[0]);  // c_l0[0] has I-diag_fractals(M)
  set_flag(PIPE_M, PIPE_FIX, event_1);
  wait_flag(PIPE_M, PIPE_FIX, event_1);
  TMOV(X_l1_tile, c_l0_tile[0]);  // X_l1 now contains I-diag_fractals(M)

  /*
   * Inv Trick part:
   * X = I - M
   * Y = M
   * block_size = 1
   * while block_size < FractalSize / 2:
   *     Y = Y @ Y
   *     X = X + X @ Y
   *     block_size *= 2
   */
  set_flag(PIPE_FIX, PIPE_M, event_0);   // store c
  set_flag(PIPE_M, PIPE_MTE1, event_0);  // load matrices for matmuls
  set_flag(PIPE_FIX, PIPE_MTE1, event_0);
  set_flag(PIPE_FIX, PIPE_M, event_1);     // only for update Y
  set_flag(PIPE_M, PIPE_MTE1, event_1);    // only for update Y
  set_flag(PIPE_FIX, PIPE_MTE1, event_1);  // only for update Y
  for (uint32_t block_size = 1; block_size < FractalSize / 2; block_size *= 2) {
    wait_flag(PIPE_M, PIPE_MTE1, event_0);
    TMOV(b_l0_tile[0], I_l1_tile);
    wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
    TMOV(a_l0_tile[0], X_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, event_0);

    wait_flag(PIPE_FIX, PIPE_MTE1, event_1);
    TMOV(b_l0_tile[1], Y_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, event_1);

    wait_flag(PIPE_FIX, PIPE_M, event_0);   // from previous iter
    wait_flag(PIPE_MTE1, PIPE_M, event_0);  // from loading a_l0[0], b_l0[0]
    TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);  // c_l0[0] contains X
    set_flag(PIPE_M, PIPE_FIX, event_0);
    wait_flag(PIPE_M, PIPE_FIX, event_0);
    set_flag(PIPE_FIX, PIPE_M, event_0);
    wait_flag(PIPE_FIX, PIPE_M, event_0);

    if (block_size < FractalSize / 4) {  // Update Y except in last iteration
      wait_flag(PIPE_M, PIPE_MTE1, event_1);  // from previous iter
      TMOV(a_l0_tile[1], Y_l1_tile);
      wait_flag(PIPE_MTE1, PIPE_M, event_1);
      set_flag(PIPE_MTE1, PIPE_M, event_1);

      wait_flag(PIPE_MTE1, PIPE_M, event_1);
      wait_flag(PIPE_FIX, PIPE_M, event_1);  // from previous iter
      TMATMUL(c_l0_tile[1], a_l0_tile[1], b_l0_tile[1]);
      set_flag(PIPE_M, PIPE_MTE1, event_1);  // for next iter
      set_flag(PIPE_M, PIPE_FIX, event_1);
      set_flag(PIPE_MTE1, PIPE_M, event_1);

      wait_flag(PIPE_M, PIPE_FIX, event_1);
      TMOV(Y_l1_tile, c_l0_tile[1]);
      set_flag(PIPE_FIX, PIPE_M, event_1);  // for next iter
    }
    set_flag(PIPE_FIX, PIPE_MTE1, event_1);  // for next iter

    wait_flag(PIPE_MTE1, PIPE_M, event_1);
    TMATMUL_ACC(c_l0_tile[0],
                c_l0_tile[0],
                a_l0_tile[0],
                b_l0_tile[1]);  // c_l0[0] has X + X @ Y
    set_flag(PIPE_M, PIPE_MTE1, event_0);
    set_flag(PIPE_M, PIPE_FIX, event_0);

    wait_flag(PIPE_M, PIPE_FIX, event_0);
    TMOV(X_l1_tile, c_l0_tile[0]);
    set_flag(PIPE_FIX, PIPE_M, event_0);     // for next iter
    set_flag(PIPE_FIX, PIPE_MTE1, event_0);  // for next iter
  }
  wait_flag(PIPE_FIX, PIPE_MTE1, event_1);  // only for update Y
  wait_flag(PIPE_M, PIPE_MTE1, event_1);    // only for update Y
  wait_flag(PIPE_FIX, PIPE_M, event_1);     // only for update Y
  wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
  wait_flag(PIPE_M, PIPE_MTE1, event_0);
  wait_flag(PIPE_FIX, PIPE_M, event_0);

  /*
   * Unrolled recursion part:
   * Upper-tri (swap_parity=false):
   *   LX = even_blocks(X), RX = odd_blocks(X)
   *   Y = LX @ (-M) + I, X = Y @ RX + LX
   * Lower-tri (swap_parity=true):
   *   RX = even→L0A(odd via swap), LX = odd→L0B(even via swap)
   *   Y = RX @ (-M) + I, X = Y @ LX + RX
   */
  TMOV(b_l0_tile[1], M_neg_l1_tile);  // b_l0[1] contains M_neg
  TMOV(a_l0_tile[0], I_l1_tile);      // a_l0[0] contains I

  if constexpr (MatrixSize > FractalSize) {
    set_flag(PIPE_FIX, PIPE_M, event_1);
  }
  set_flag(PIPE_M, PIPE_MTE1, event_1);
  set_flag(PIPE_M, PIPE_MTE1, event_0);
  set_flag(PIPE_FIX, PIPE_MTE1, event_1);
  set_flag(PIPE_FIX, PIPE_M, event_0);
  for (uint32_t block_size = FractalSize; block_size < MatrixSize;
       block_size *= 2) {
    wait_flag(PIPE_M, PIPE_MTE1, event_0);  // Wait for last iter a_l0[1]
    TMOV(a_l0_tile[1], Zero_l1_tile);

    wait_flag(PIPE_M, PIPE_MTE1, event_1);
    TMOV(b_l0_tile[0], I_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, event_0);

    wait_flag(PIPE_FIX, PIPE_MTE1, event_1);  // Wait to write last X
    CopyOddOrEvenBlocksL1ToL0<InputT, FractalSize, MatrixSize>(
        X_l1_tile,
        a_l0_tile[1],
        block_size,
        swap_parity);  // a_l0[1]: even(LX) or odd(RX)
    set_flag(PIPE_MTE1, PIPE_M, event_1);

    wait_flag(PIPE_MTE1, PIPE_M, event_0);
    wait_flag(PIPE_FIX, PIPE_M, event_0);  // Wait c_l0[0] from previous iter
    TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);  // c_l0[0] has I

    wait_flag(PIPE_MTE1, PIPE_M, event_1);
    wait_flag(PIPE_FIX, PIPE_M, event_1);  // Wait c_l0[1] from previous iter
    TMATMUL(c_l0_tile[1], a_l0_tile[1], b_l0_tile[0]);  // c_l0[1] contains LX
    set_flag(PIPE_M, PIPE_MTE1, event_1);  // allow to load RX on b_l0[0]

    TMATMUL_ACC(c_l0_tile[0],
                c_l0_tile[0],
                a_l0_tile[1],
                b_l0_tile[1]);  // c_l0[0] <- LX * M_neg + I
    set_flag(PIPE_M, PIPE_FIX, event_0);
    set_flag(PIPE_M, PIPE_MTE1, event_0);

    wait_flag(PIPE_M, PIPE_FIX, event_0);
    TMOV(Y_l1_tile, c_l0_tile[0]);  // Y_l1 contains LX * M_neg + I
    set_flag(PIPE_FIX, PIPE_MTE1, event_0);
    set_flag(PIPE_FIX, PIPE_M, event_0);

    /* Load complementary blocks of X in L0B */
    wait_flag(PIPE_M, PIPE_MTE1, event_1);
    TMOV(b_l0_tile[0], Zero_l1_tile);
    CopyOddOrEvenBlocksL1ToL0<InputT, FractalSize, MatrixSize>(
        X_l1_tile,
        b_l0_tile[0],
        block_size,
        swap_parity);  // b_l0[0]: odd(RX) or even(LX)

    wait_flag(PIPE_M, PIPE_MTE1, event_0);  // Wait for previous use of a_l0[1]
    wait_flag(PIPE_FIX, PIPE_MTE1, event_0);  // Wait for Y_l1
    TMOV(a_l0_tile[1], Y_l1_tile);            // a_l0[1] contains LX * M_neg + I
    set_flag(PIPE_MTE1, PIPE_M, event_0);

    wait_flag(PIPE_MTE1, PIPE_M, event_0);
    TMATMUL_ACC(c_l0_tile[1], c_l0_tile[1], a_l0_tile[1], b_l0_tile[0]);
    set_flag(PIPE_M, PIPE_MTE1, event_0);  // next iter can read on a_l0[1]
    set_flag(PIPE_M, PIPE_MTE1, event_1);  // next iter can read on b_l0[0]
    set_flag(PIPE_M, PIPE_FIX, event_0);
    wait_flag(PIPE_M, PIPE_FIX, event_0);

    if (block_size < MatrixSize / 2) {  // Update X_l1 except in last iteration
      TMOV(X_l1_tile, c_l0_tile[1]);
      set_flag(PIPE_FIX, PIPE_M, event_1);  // release c_l0[1] for next iter
    }
    set_flag(PIPE_FIX, PIPE_MTE1, event_1);
  }
  wait_flag(PIPE_M, PIPE_MTE1, event_0);
  wait_flag(PIPE_M, PIPE_MTE1, event_1);
  wait_flag(PIPE_FIX, PIPE_M, event_0);
  wait_flag(PIPE_FIX, PIPE_MTE1, event_1);  // Write c_l0[1] to X_l1
}

#if defined(MEGA_CHUNK_GDN_INTERLEAVED_SOLVE) || \
    defined(MEGA_CHUNK_GDN_STREAMED_PAIR_SOLVE)
#define GDN_PAIR_ACQUIRE_AB()              \
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0); \
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1)
#define GDN_PAIR_COMMIT_AB()              \
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0); \
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1)
#define GDN_PAIR_MATMUL_KEEP_C()                     \
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);           \
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);            \
  TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]); \
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);            \
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);           \
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);            \
  TMATMUL(c_l0_tile[1], a_l0_tile[1], b_l0_tile[1]); \
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1)
#define GDN_PAIR_MATMUL_FINAL()          \
  GDN_PAIR_MATMUL_KEEP_C();              \
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0); \
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1)
#define GDN_PAIR_MATMUL_ACC_FINAL()                                    \
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);                             \
  TMATMUL_ACC(c_l0_tile[0], c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]); \
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);                              \
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);                               \
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);                             \
  TMATMUL_ACC(c_l0_tile[1], c_l0_tile[1], a_l0_tile[1], b_l0_tile[1]); \
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);                              \
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1)
#define GDN_PAIR_STORE_L1(dst0, dst1)     \
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0); \
  TMOV((dst0), c_l0_tile[0]);             \
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);  \
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1); \
  TMOV((dst1), c_l0_tile[1]);             \
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1)
#define GDN_PAIR_WAIT_L1_STORES()           \
  set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0); \
  wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0)
/*
 * An even number of independent 128x128 inverses share the two L0 buffer
 * sets.  The arithmetic and every FP16 L1 rounding boundary match
 * InvertSingleTile;
 * only independent matrices are interleaved so MTE1, Cube and FixPipe can
 * overlap.  X and Y are intentionally swapped after initialization to avoid
 * keeping a fourth per-matrix L1 tile.
 */
template <typename InputT,
          typename OutputT,
          typename StoreT,
          typename TileL1AB,
          typename TileL0A,
          typename TileL0B,
          typename TileL0C,
          uint32_t MatrixSize,
          uint32_t FractalSize,
          uint32_t NumInterleavedTiles>
AICORE inline void InvertTilesInterleavedBSND(
    __gm__ StoreT* M_inv,
    uint32_t num_bsnd_heads,
    const uint32_t* bsnd_tile_offsets,
    const uint32_t* bsnd_tile_valid_sizes,
    TileL1AB I_l1_tile,
    TileL1AB I_neg_l1_tile,
    TileL1AB Zero_l1_tile,
    TileL1AB* M_neg_l1_tile,
    TileL1AB* X_l1_tile,
    TileL1AB* Y_l1_tile,
    TileL0A* a_l0_tile,
    TileL0B* b_l0_tile,
    TileL0C* c_l0_tile,
    const bool swap_parity) {
  static_assert(MatrixSize == 128 && FractalSize == 16,
                "Interleaved Solve is specialized for 128x128 matrices.");
  static_assert(NumInterleavedTiles >= 2 && NumInterleavedTiles % 2 == 0,
                "Interleaved Solve requires complete matrix pairs.");
  constexpr uint32_t TileLen = MatrixSize * MatrixSize;
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

  // M_neg = (-I) @ M.
#pragma unroll
  for (uint32_t pair = 0; pair < NumInterleavedTiles; pair += 2) {
    const uint32_t t0 = pair;
    const uint32_t t1 = pair + 1;
    GDN_PAIR_ACQUIRE_AB();
    TMOV(a_l0_tile[0], I_neg_l1_tile);
    TMOV(b_l0_tile[0], Y_l1_tile[t0]);
    TMOV(a_l0_tile[1], I_neg_l1_tile);
    TMOV(b_l0_tile[1], Y_l1_tile[t1]);
    GDN_PAIR_COMMIT_AB();
    GDN_PAIR_MATMUL_FINAL();
    GDN_PAIR_STORE_L1(M_neg_l1_tile[t0], M_neg_l1_tile[t1]);
  }

  // Temporarily keep diag(M)^2 in X_l1 while Y_l1 still owns M.
#pragma unroll
  for (uint32_t pair = 0; pair < NumInterleavedTiles; pair += 2) {
    const uint32_t t0 = pair;
    const uint32_t t1 = pair + 1;
    GDN_PAIR_ACQUIRE_AB();
    TMOV(a_l0_tile[0], Zero_l1_tile);
    TMOV(b_l0_tile[0], Zero_l1_tile);
    CopyDiagonalFractalsL1ToL0<InputT, FractalSize, MatrixSize>(Y_l1_tile[t0],
                                                                a_l0_tile[0]);
    CopyDiagonalFractalsL1ToL0<InputT, FractalSize, MatrixSize>(Y_l1_tile[t0],
                                                                b_l0_tile[0]);
    TMOV(a_l0_tile[1], Zero_l1_tile);
    TMOV(b_l0_tile[1], Zero_l1_tile);
    CopyDiagonalFractalsL1ToL0<InputT, FractalSize, MatrixSize>(Y_l1_tile[t1],
                                                                a_l0_tile[1]);
    CopyDiagonalFractalsL1ToL0<InputT, FractalSize, MatrixSize>(Y_l1_tile[t1],
                                                                b_l0_tile[1]);
    GDN_PAIR_COMMIT_AB();
    GDN_PAIR_MATMUL_FINAL();
    GDN_PAIR_STORE_L1(X_l1_tile[t0], X_l1_tile[t1]);
  }

  // Y_l1 becomes X = diag(M) @ (-I) + (-I) @ (-I).
#pragma unroll
  for (uint32_t pair = 0; pair < NumInterleavedTiles; pair += 2) {
    const uint32_t t0 = pair;
    const uint32_t t1 = pair + 1;
    GDN_PAIR_ACQUIRE_AB();
    TMOV(a_l0_tile[0], Zero_l1_tile);
    CopyDiagonalFractalsL1ToL0<InputT, FractalSize, MatrixSize>(Y_l1_tile[t0],
                                                                a_l0_tile[0]);
    TMOV(b_l0_tile[0], I_neg_l1_tile);
    TMOV(a_l0_tile[1], Zero_l1_tile);
    CopyDiagonalFractalsL1ToL0<InputT, FractalSize, MatrixSize>(Y_l1_tile[t1],
                                                                a_l0_tile[1]);
    TMOV(b_l0_tile[1], I_neg_l1_tile);
    GDN_PAIR_COMMIT_AB();
    GDN_PAIR_MATMUL_KEEP_C();
    GDN_PAIR_ACQUIRE_AB();
    TMOV(a_l0_tile[0], I_neg_l1_tile);
    TMOV(b_l0_tile[0], I_neg_l1_tile);
    TMOV(a_l0_tile[1], I_neg_l1_tile);
    TMOV(b_l0_tile[1], I_neg_l1_tile);
    GDN_PAIR_COMMIT_AB();
    GDN_PAIR_MATMUL_ACC_FINAL();
    GDN_PAIR_STORE_L1(Y_l1_tile[t0], Y_l1_tile[t1]);
  }
  GDN_PAIR_WAIT_L1_STORES();

  // Inverse trick.  Y_l1 is X and X_l1 is Y from this point onward.
  for (uint32_t block_size = 1; block_size < FractalSize / 2; block_size *= 2) {
#pragma unroll
    for (uint32_t pair = 0; pair < NumInterleavedTiles; pair += 2) {
      const uint32_t t0 = pair;
      const uint32_t t1 = pair + 1;
      GDN_PAIR_ACQUIRE_AB();
      TMOV(a_l0_tile[0], Y_l1_tile[t0]);
      TMOV(b_l0_tile[0], I_l1_tile);
      TMOV(a_l0_tile[1], Y_l1_tile[t1]);
      TMOV(b_l0_tile[1], I_l1_tile);
      GDN_PAIR_COMMIT_AB();
      GDN_PAIR_MATMUL_KEEP_C();
      GDN_PAIR_ACQUIRE_AB();
      TMOV(a_l0_tile[0], Y_l1_tile[t0]);
      TMOV(b_l0_tile[0], X_l1_tile[t0]);
      TMOV(a_l0_tile[1], Y_l1_tile[t1]);
      TMOV(b_l0_tile[1], X_l1_tile[t1]);
      GDN_PAIR_COMMIT_AB();
      GDN_PAIR_MATMUL_ACC_FINAL();
      GDN_PAIR_STORE_L1(Y_l1_tile[t0], Y_l1_tile[t1]);
    }
    GDN_PAIR_WAIT_L1_STORES();

    if (block_size < FractalSize / 4) {
#pragma unroll
      for (uint32_t pair = 0; pair < NumInterleavedTiles; pair += 2) {
        const uint32_t t0 = pair;
        const uint32_t t1 = pair + 1;
        GDN_PAIR_ACQUIRE_AB();
        TMOV(a_l0_tile[0], X_l1_tile[t0]);
        TMOV(b_l0_tile[0], X_l1_tile[t0]);
        TMOV(a_l0_tile[1], X_l1_tile[t1]);
        TMOV(b_l0_tile[1], X_l1_tile[t1]);
        GDN_PAIR_COMMIT_AB();
        GDN_PAIR_MATMUL_FINAL();
        GDN_PAIR_STORE_L1(X_l1_tile[t0], X_l1_tile[t1]);
      }
      GDN_PAIR_WAIT_L1_STORES();
    }
  }

  // Recursive assembly of the 16, 32 and 64-wide block inverses.
  for (uint32_t block_size = FractalSize; block_size < MatrixSize;
       block_size *= 2) {
    // X_l1 is the recursion temporary Y = I + LX @ M_neg.
#pragma unroll
    for (uint32_t pair = 0; pair < NumInterleavedTiles; pair += 2) {
      const uint32_t t0 = pair;
      const uint32_t t1 = pair + 1;
      GDN_PAIR_ACQUIRE_AB();
      TMOV(a_l0_tile[0], I_l1_tile);
      TMOV(b_l0_tile[0], I_l1_tile);
      TMOV(a_l0_tile[1], I_l1_tile);
      TMOV(b_l0_tile[1], I_l1_tile);
      GDN_PAIR_COMMIT_AB();
      GDN_PAIR_MATMUL_KEEP_C();
      GDN_PAIR_ACQUIRE_AB();
      TMOV(a_l0_tile[0], Zero_l1_tile);
      CopyOddOrEvenBlocksL1ToL0<InputT, FractalSize, MatrixSize>(
          Y_l1_tile[t0], a_l0_tile[0], block_size, swap_parity);
      TMOV(b_l0_tile[0], M_neg_l1_tile[t0]);
      TMOV(a_l0_tile[1], Zero_l1_tile);
      CopyOddOrEvenBlocksL1ToL0<InputT, FractalSize, MatrixSize>(
          Y_l1_tile[t1], a_l0_tile[1], block_size, swap_parity);
      TMOV(b_l0_tile[1], M_neg_l1_tile[t1]);
      GDN_PAIR_COMMIT_AB();
      GDN_PAIR_MATMUL_ACC_FINAL();
      GDN_PAIR_STORE_L1(X_l1_tile[t0], X_l1_tile[t1]);
    }
    GDN_PAIR_WAIT_L1_STORES();

#pragma unroll
    for (uint32_t pair = 0; pair < NumInterleavedTiles; pair += 2) {
      const uint32_t t0 = pair;
      const uint32_t t1 = pair + 1;
      GDN_PAIR_ACQUIRE_AB();
      TMOV(a_l0_tile[0], Zero_l1_tile);
      CopyOddOrEvenBlocksL1ToL0<InputT, FractalSize, MatrixSize>(
          Y_l1_tile[t0], a_l0_tile[0], block_size, swap_parity);
      TMOV(b_l0_tile[0], I_l1_tile);
      TMOV(a_l0_tile[1], Zero_l1_tile);
      CopyOddOrEvenBlocksL1ToL0<InputT, FractalSize, MatrixSize>(
          Y_l1_tile[t1], a_l0_tile[1], block_size, swap_parity);
      TMOV(b_l0_tile[1], I_l1_tile);
      GDN_PAIR_COMMIT_AB();
      GDN_PAIR_MATMUL_KEEP_C();
      GDN_PAIR_ACQUIRE_AB();
      TMOV(a_l0_tile[0], X_l1_tile[t0]);
      TMOV(b_l0_tile[0], Zero_l1_tile);
      CopyOddOrEvenBlocksL1ToL0<InputT, FractalSize, MatrixSize>(
          Y_l1_tile[t0], b_l0_tile[0], block_size, swap_parity);
      TMOV(a_l0_tile[1], X_l1_tile[t1]);
      TMOV(b_l0_tile[1], Zero_l1_tile);
      CopyOddOrEvenBlocksL1ToL0<InputT, FractalSize, MatrixSize>(
          Y_l1_tile[t1], b_l0_tile[1], block_size, swap_parity);
      GDN_PAIR_COMMIT_AB();
      GDN_PAIR_MATMUL_ACC_FINAL();

      if (block_size < MatrixSize / 2) {
        GDN_PAIR_STORE_L1(Y_l1_tile[t0], Y_l1_tile[t1]);
      } else {
        using GlobalShape =
            TileShape2D<StoreT, MatrixSize, MatrixSize, Layout::ND>;
        using GlobalStride = pto::Stride<1, 1, 1, -1, 1>;
        using GlobalOut =
            GlobalTensor<StoreT, GlobalShape, GlobalStride, Layout::ND>;
        using DynamicShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
        using DynamicStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
        using DynamicOut =
            GlobalTensor<StoreT, DynamicShape, DynamicStride, Layout::ND>;
        using DynamicAcc =
            TileAcc<OutputT, MatrixSize, MatrixSize, DYNAMIC, DYNAMIC>;
        const int row_stride = static_cast<int>(MatrixSize * num_bsnd_heads);
        const uint32_t valid0 = bsnd_tile_valid_sizes[t0];
        const uint32_t valid1 = bsnd_tile_valid_sizes[t1];
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        if (valid0 < MatrixSize) {
          DynamicAcc out0(valid0, valid0);
          TASSIGN(out0, 0);
          DynamicOut gm0(
              M_inv + bsnd_tile_offsets[t0],
              {1, 1, 1, static_cast<int>(valid0), static_cast<int>(valid0)},
              {1, 1, 1, row_stride, 1});
          TSTORE(gm0, out0);
        } else {
          GlobalOut gm0(M_inv + bsnd_tile_offsets[t0], {}, {row_stride});
          TSTORE(gm0, c_l0_tile[0]);
        }
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
        if (valid1 < MatrixSize) {
          DynamicAcc out1(valid1, valid1);
          TASSIGN(out1, TileLen * sizeof(OutputT));
          DynamicOut gm1(
              M_inv + bsnd_tile_offsets[t1],
              {1, 1, 1, static_cast<int>(valid1), static_cast<int>(valid1)},
              {1, 1, 1, row_stride, 1});
          TSTORE(gm1, out1);
        } else {
          GlobalOut gm1(M_inv + bsnd_tile_offsets[t1], {}, {row_stride});
          TSTORE(gm1, c_l0_tile[1]);
        }
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
      }
    }
    if (block_size < MatrixSize / 2) {
      GDN_PAIR_WAIT_L1_STORES();
    }
  }

  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
}
#undef GDN_PAIR_WAIT_L1_STORES
#undef GDN_PAIR_STORE_L1
#undef GDN_PAIR_MATMUL_ACC_FINAL
#undef GDN_PAIR_MATMUL_FINAL
#undef GDN_PAIR_MATMUL_KEEP_C
#undef GDN_PAIR_COMMIT_AB
#undef GDN_PAIR_ACQUIRE_AB
#endif

/*
 * @brief: Runs the main kernel (inverts all matrices in the tensor)
 *
 * @tparam InputT The type of the input elements.
 * @tparam OutputT The type of the output elements.
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam NumTilesPerCubeIter How many matrices to load and invert in a single
 * cube iteration.
 * @tparam IsBSND If IsBSND is false, then the last two dimensions represent a
 * 2D triangular matrix in row-major format, while the other dimensions are
 * batch dimensions. If IsBSND is true, then the dimensions represent in order:
 * B batch size, S sequence length (which is chunked in tiles of size D), N
 * number of heads (equivalent to a second batch dimension for this kernel), and
 * D chunk size. The inverse is over the dimensions S (chunked) and D, row-major
 * within each tile.
 *
 * @param M_inv pointer to the global memory to store the final inverse.
 * @param M Pointer to the global tensor matrix in global memory.
 * @param I_neg Pointer to global memory that contains the negative identity.
 * @param total_tiles The total number of matrices to invert.
 * @param num_bsnd_heads The number of heads, only for BSND format.
 */
template <typename InputT,
          typename OutputT,
          uint32_t MatrixSize,
          uint32_t NumTilesPerCubeIter,
          bool IsBSND,
          typename StoreT = OutputT,
          bool WaitForKktReady = false,
          bool PrecomputedAuxiliary = false>
AICORE inline void TriInvRecUnrollKernel(__gm__ StoreT* M_inv,
                                         __gm__ InputT* M,
                                         __gm__ InputT* I_neg,
                                         uint32_t total_tiles,
                                         uint32_t num_bsnd_heads = 0,
                                         __gm__ int32_t* cu_seqlens = nullptr,
                                         uint32_t is_lower = 0,
                                         bool use_precomputed_m_neg = false) {
  /* Initializations */
  constexpr uint32_t TileLen = MatrixSize * MatrixSize;
  constexpr uint32_t FractalSize = 16;  // fractal size for half
  constexpr uint32_t NumFractalsRowWise = MatrixSize / FractalSize;
  constexpr uint32_t NumL0Buffers = 2;

  const uint32_t core_id = get_block_idx();
  const uint32_t core_count = get_block_num();
  uint32_t producer_group_count = 0;
  uint32_t producer_tile_count = 0;
  if constexpr (WaitForKktReady) {
    static_assert(
        IsBSND && (NumTilesPerCubeIter == 2 || NumTilesPerCubeIter == 3),
        "KKT-Solve pipeline requires grouped BSND Solve.");
    const uint32_t total_groups = total_tiles / NumTilesPerCubeIter;
    if (core_id >= total_groups) {
      return;
    }
    producer_group_count = 1 + (total_groups - 1 - core_id) / core_count;
    producer_tile_count = producer_group_count * NumTilesPerCubeIter;
  } else if (core_id * NumTilesPerCubeIter >= total_tiles) {
    return;
  }

  using GlobalTileShapeIn =
      TileShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTileStridesIn = typename std::conditional<
      !IsBSND,
      BaseShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>,
      pto::Stride<1, 1, 1, -1, 1>>::type;
  using GlobalTileIn =
      GlobalTensor<InputT, GlobalTileShapeIn, GlobalTileStridesIn, Layout::ND>;
  using GlobalTileDynamicShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GlobalTileDynamicStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  using GlobalTileDynamicIn = GlobalTensor<InputT,
                                           GlobalTileDynamicShape,
                                           GlobalTileDynamicStride,
                                           Layout::ND>;
  using GlobalTileStridesINeg =
      BaseShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTileINeg = GlobalTensor<InputT,
                                      GlobalTileShapeIn,
                                      GlobalTileStridesINeg,
                                      Layout::ND>;

  using GlobalTileShapeOut =
      TileShape2D<StoreT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTileStridesOut = typename std::conditional<
      !IsBSND,
      BaseShape2D<StoreT, MatrixSize, MatrixSize, Layout::ND>,
      pto::Stride<1, 1, 1, -1, 1>>::type;
  using GlobalTileOut = GlobalTensor<StoreT,
                                     GlobalTileShapeOut,
                                     GlobalTileStridesOut,
                                     Layout::ND>;
  using GlobalTileDynamicOut = GlobalTensor<StoreT,
                                            GlobalTileDynamicShape,
                                            GlobalTileDynamicStride,
                                            Layout::ND>;
  using TileL1AB = Tile<TileType::Mat,
                        InputT,
                        MatrixSize,
                        MatrixSize,
                        BLayout::ColMajor,
                        MatrixSize,
                        MatrixSize,
                        SLayout::RowMajor,
                        512>;
  using TileL1ABDynamic = Tile<TileType::Mat,
                               InputT,
                               MatrixSize,
                               MatrixSize,
                               BLayout::ColMajor,
                               DYNAMIC,
                               DYNAMIC,
                               SLayout::RowMajor,
                               512,
                               PadValue::Zero>;

  // L0 Memory
  using TileL0A = TileLeft<InputT, MatrixSize, MatrixSize>;
  using TileL0B = TileRight<InputT, MatrixSize, MatrixSize>;
  using TileL0C = TileAcc<OutputT, MatrixSize, MatrixSize>;
  using TileL0CDynamic =
      TileAcc<OutputT, MatrixSize, MatrixSize, DYNAMIC, DYNAMIC>;

  GlobalTileINeg I_neg_global_in(I_neg);
  GlobalTileINeg I_global_in(I_neg + TileLen);
  GlobalTileINeg Zero_global_in(I_neg + 2 * TileLen);

  TileL1AB I_l1_tile;
  TileL1AB I_neg_l1_tile;
  TileL1AB Zero_l1_tile;
#ifdef MEGA_CHUNK_GDN_STREAMED_PAIR_SOLVE
  constexpr uint32_t NumSolveStateTiles =
      NumTilesPerCubeIter == 4 ? 2 : NumTilesPerCubeIter;
  TileL1AB M_neg_l1_tile[NumSolveStateTiles];
  TileL1AB X_l1_tile[NumSolveStateTiles];
#elif defined(MEGA_CHUNK_GDN_INTERLEAVED_SOLVE)
  TileL1AB M_neg_l1_tile[NumTilesPerCubeIter];
  TileL1AB X_l1_tile[NumTilesPerCubeIter];
#else
  TileL1AB M_neg_l1_tile;
  TileL1AB X_l1_tile;
#endif
#ifdef MEGA_CHUNK_GDN_STREAMED_PAIR_SOLVE
  TileL1AB Y_l1_tile[NumSolveStateTiles];
#else
  TileL1AB Y_l1_tile[NumTilesPerCubeIter];
#endif

  TileL0A a_l0_tile[NumL0Buffers];
  TileL0B b_l0_tile[NumL0Buffers];
  TileL0C c_l0_tile[NumL0Buffers];

  TASSIGN(I_l1_tile, 0x0);
  TASSIGN(I_neg_l1_tile, 0x0 + TileLen * sizeof(InputT));
  TASSIGN(Zero_l1_tile, 0x0 + 2 * TileLen * sizeof(InputT));
#ifdef MEGA_CHUNK_GDN_STREAMED_PAIR_SOLVE
  for (uint32_t tile_id = 0; tile_id < NumSolveStateTiles; ++tile_id) {
    TASSIGN(M_neg_l1_tile[tile_id],
            0x0 + (3 + tile_id) * TileLen * sizeof(InputT));
    TASSIGN(
        X_l1_tile[tile_id],
        0x0 + (3 + NumSolveStateTiles + tile_id) * TileLen * sizeof(InputT));
    TASSIGN(Y_l1_tile[tile_id],
            0x0 + (3 + 2 * NumSolveStateTiles + tile_id) * TileLen *
                      sizeof(InputT));
  }
#elif defined(MEGA_CHUNK_GDN_INTERLEAVED_SOLVE)
  for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
    TASSIGN(M_neg_l1_tile[tile_id],
            0x0 + (3 + tile_id) * TileLen * sizeof(InputT));
    TASSIGN(
        X_l1_tile[tile_id],
        0x0 + (3 + NumTilesPerCubeIter + tile_id) * TileLen * sizeof(InputT));
    TASSIGN(Y_l1_tile[tile_id],
            0x0 + (3 + 2 * NumTilesPerCubeIter + tile_id) * TileLen *
                      sizeof(InputT));
  }
#else
  TASSIGN(M_neg_l1_tile, 0x0 + 3 * TileLen * sizeof(InputT));
  TASSIGN(X_l1_tile, 0x0 + 4 * TileLen * sizeof(InputT));
  for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
    TASSIGN(Y_l1_tile[tile_id], 0x0 + (5 + tile_id) * TileLen * sizeof(InputT));
  }
#endif

  for (uint32_t buffer_num = 0; buffer_num < NumL0Buffers; ++buffer_num) {
    TASSIGN(a_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(InputT));
    TASSIGN(b_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(InputT));
    TASSIGN(c_l0_tile[buffer_num],
            0x0 + buffer_num * TileLen * sizeof(OutputT));
  }
  TLOAD(I_neg_l1_tile, I_neg_global_in);
  if constexpr (PrecomputedAuxiliary) {
    TLOAD(I_l1_tile, I_global_in);
    TLOAD(Zero_l1_tile, Zero_global_in);
  }
  set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
  wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));

  if constexpr (!PrecomputedAuxiliary) {
    PrepareAuxiliaryMatrices<TileL1AB, TileL0A, TileL0B, TileL0C>(I_neg_l1_tile,
                                                                  Zero_l1_tile,
                                                                  I_l1_tile,
                                                                  a_l0_tile[0],
                                                                  b_l0_tile[0],
                                                                  c_l0_tile[0]);
  }

  uint32_t max_iters_per_aic = 0;
  if constexpr (WaitForKktReady) {
    max_iters_per_aic = CeilDiv(producer_tile_count, NumTilesPerCubeIter);
  } else {
    max_iters_per_aic = CeilDiv(total_tiles, NumTilesPerCubeIter * core_count);
  }

  /* Main iteration - Compute all tiles */
  uint32_t bsnd_tile_offsets[NumTilesPerCubeIter] = {0};
  uint32_t bsnd_tile_valid_sizes[NumTilesPerCubeIter] = {0};
  uint32_t global_tile_ids[NumTilesPerCubeIter] = {0};
  uint32_t next_ready_wave = 0;
  uint32_t next_tile_id_that_waits_for_pipe_fix_pipe_m = 0;
  set_flag(PIPE_FIX,
           PIPE_M,
           static_cast<event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
  for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
    set_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
  }
  for (uint32_t cube_iter = 0; cube_iter < max_iters_per_aic; ++cube_iter) {
    uint32_t tiles_this_iter = 0;
    uint32_t global_index = 0;
    if constexpr (WaitForKktReady) {
      const uint32_t local_tile_base = cube_iter * NumTilesPerCubeIter;
      const uint32_t local_remaining = producer_tile_count - local_tile_base;
      tiles_this_iter = local_remaining < NumTilesPerCubeIter
                            ? local_remaining
                            : NumTilesPerCubeIter;
      const uint32_t required_wave =
          (local_tile_base + tiles_this_iter - 1) / NumTilesPerCubeIter;
      while (next_ready_wave <= required_wave) {
        const uint32_t solve_slot = next_ready_wave & 3;
#if defined(PTO_NPU_ARCH_A5)
        gdn_sync::Wait<PIPE_MTE2>(solve_slot);
#else
        wait_flag_dev(solve_slot);
#endif
        if (next_ready_wave + 4 < producer_group_count) {
#if defined(PTO_NPU_ARCH_A5)
          gdn_sync::Signal<PIPE_FIX>(1 | (2 << 4) | ((4 + solve_slot) << 8));
#else
          ffts_cross_core_sync(PIPE_FIX,
                               1 | (2 << 4) | ((4 + solve_slot) << 8));
#endif
        }
        ++next_ready_wave;
      }
      for (uint32_t tile_id = 0; tile_id < tiles_this_iter; ++tile_id) {
        const uint32_t local_tile_id = local_tile_base + tile_id;
        const uint32_t producer_wave = local_tile_id / NumTilesPerCubeIter;
        const uint32_t group_lane = local_tile_id % NumTilesPerCubeIter;
        const uint32_t group_id = producer_wave * core_count + core_id;
        global_tile_ids[tile_id] = group_id * NumTilesPerCubeIter + group_lane;
      }
    } else {
      global_index = (cube_iter * core_count + core_id) * NumTilesPerCubeIter;
      if (global_index >= total_tiles) {
        break;
      }
      const uint32_t global_remaining = total_tiles - global_index;
      tiles_this_iter = global_remaining < NumTilesPerCubeIter
                            ? global_remaining
                            : NumTilesPerCubeIter;
      for (uint32_t tile_id = 0; tile_id < tiles_this_iter; ++tile_id) {
        global_tile_ids[tile_id] = global_index + tile_id;
      }
    }
#ifdef MEGA_CHUNK_GDN_STREAMED_PAIR_SOLVE
    if constexpr (IsBSND && MatrixSize == 128 && NumTilesPerCubeIter == 4) {
      for (uint32_t tile_id = 0; tile_id < tiles_this_iter; ++tile_id) {
        const uint32_t global_tile_id = global_tile_ids[tile_id];
        if (cu_seqlens != nullptr) {
          const BSNDVarlenTileInfo tile_info =
              GetBSNDVarlenTileInfoFromCuSeqlens(
                  global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens);
          bsnd_tile_offsets[tile_id] = tile_info.bsnd_offset;
          bsnd_tile_valid_sizes[tile_id] = tile_info.valid_size;
        } else {
          bsnd_tile_offsets[tile_id] = GetBSNDFixedTileOffset(
              global_tile_id, num_bsnd_heads, MatrixSize);
          bsnd_tile_valid_sizes[tile_id] = MatrixSize;
        }
      }

      wait_flag(
          PIPE_FIX,
          PIPE_M,
          static_cast<event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
      uint32_t pair_base = 0;
      for (; pair_base + 1 < tiles_this_iter; pair_base += 2) {
#pragma unroll
        for (uint32_t lane = 0; lane < 2; ++lane) {
          const uint32_t tile_id = pair_base + lane;
          const uint32_t bsnd_offset = bsnd_tile_offsets[tile_id];
          const uint32_t valid_size = bsnd_tile_valid_sizes[tile_id];
          const int row_stride = static_cast<int>(MatrixSize * num_bsnd_heads);
          wait_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
          if (valid_size < MatrixSize) {
            TileL1ABDynamic Y_dyn_l1_tile(valid_size, valid_size);
            TASSIGN(Y_dyn_l1_tile,
                    0x0 + (3 + 2 * NumSolveStateTiles + lane) * TileLen *
                              sizeof(InputT));
            GlobalTileDynamicIn M_global_in_dyn(M + bsnd_offset,
                                                {1,
                                                 1,
                                                 1,
                                                 static_cast<int>(valid_size),
                                                 static_cast<int>(valid_size)},
                                                {1, 1, 1, row_stride, 1});
            TLOAD(Y_dyn_l1_tile, M_global_in_dyn);
            set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(lane));
            wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(lane));
            TFILLPAD(Y_dyn_l1_tile, Y_dyn_l1_tile);
          } else {
            GlobalTileIn M_global_in(M + bsnd_offset, {}, {row_stride});
            TLOAD(Y_l1_tile[lane], M_global_in);
          }
          set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(lane));
        }
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        InvertTilesInterleavedBSND<InputT,
                                   OutputT,
                                   StoreT,
                                   TileL1AB,
                                   TileL0A,
                                   TileL0B,
                                   TileL0C,
                                   MatrixSize,
                                   FractalSize,
                                   2>(M_inv,
                                      num_bsnd_heads,
                                      bsnd_tile_offsets + pair_base,
                                      bsnd_tile_valid_sizes + pair_base,
                                      I_l1_tile,
                                      I_neg_l1_tile,
                                      Zero_l1_tile,
                                      M_neg_l1_tile,
                                      X_l1_tile,
                                      Y_l1_tile,
                                      a_l0_tile,
                                      b_l0_tile,
                                      c_l0_tile,
                                      is_lower != 0);
        if (pair_base + 3 < tiles_this_iter) {
          pipe_barrier(PIPE_ALL);
        }
        set_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(pair_base));
        set_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(pair_base + 1));
      }

      if (pair_base < tiles_this_iter) {
        const uint32_t tile_id = pair_base;
        const uint32_t bsnd_offset = bsnd_tile_offsets[tile_id];
        const uint32_t valid_size = bsnd_tile_valid_sizes[tile_id];
        const int row_stride = static_cast<int>(MatrixSize * num_bsnd_heads);
        wait_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
        if (valid_size < MatrixSize) {
          TileL1ABDynamic Y_dyn_l1_tile(valid_size, valid_size);
          TASSIGN(
              Y_dyn_l1_tile,
              0x0 + (3 + 2 * NumSolveStateTiles) * TileLen * sizeof(InputT));
          GlobalTileDynamicIn M_global_in_dyn(M + bsnd_offset,
                                              {1,
                                               1,
                                               1,
                                               static_cast<int>(valid_size),
                                               static_cast<int>(valid_size)},
                                              {1, 1, 1, row_stride, 1});
          TLOAD(Y_dyn_l1_tile, M_global_in_dyn);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          TFILLPAD(Y_dyn_l1_tile, Y_dyn_l1_tile);
        } else {
          GlobalTileIn M_global_in(M + bsnd_offset, {}, {row_stride});
          TLOAD(Y_l1_tile[0], M_global_in);
        }
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        InvertSingleTile<InputT,
                         TileL1AB,
                         TileL0A,
                         TileL0B,
                         TileL0C,
                         MatrixSize,
                         FractalSize,
                         2>(X_l1_tile[0],
                            I_l1_tile,
                            I_neg_l1_tile,
                            M_neg_l1_tile[0],
                            Zero_l1_tile,
                            Y_l1_tile[0],
                            a_l0_tile,
                            b_l0_tile,
                            c_l0_tile,
                            0,
                            is_lower != 0);
        constexpr uint32_t final_c_buffer_index =
            MatrixSize > FractalSize ? 1 : 0;
        if (valid_size < MatrixSize) {
          TileL0CDynamic c_l0_tail_tile(valid_size, valid_size);
          TASSIGN(c_l0_tail_tile,
                  final_c_buffer_index * TileLen * sizeof(OutputT));
          GlobalTileDynamicOut M_inv_global_out_dyn(
              M_inv + bsnd_offset,
              {1,
               1,
               1,
               static_cast<int>(valid_size),
               static_cast<int>(valid_size)},
              {1, 1, 1, row_stride, 1});
          TSTORE(M_inv_global_out_dyn, c_l0_tail_tile);
        } else {
          GlobalTileOut M_inv_global_out(M_inv + bsnd_offset, {}, {row_stride});
          TSTORE(M_inv_global_out, c_l0_tile[final_c_buffer_index]);
        }
        set_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
      }
      set_flag(
          PIPE_FIX,
          PIPE_M,
          static_cast<event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
      continue;
    }
#endif
    for (uint32_t tile_id = 0; tile_id < tiles_this_iter; ++tile_id) {
      if constexpr (IsBSND) {
        const uint32_t global_tile_id = global_tile_ids[tile_id];
        if (cu_seqlens != nullptr) {
          const BSNDVarlenTileInfo tile_info =
              GetBSNDVarlenTileInfoFromCuSeqlens(
                  global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens);
          bsnd_tile_offsets[tile_id] = tile_info.bsnd_offset;
          bsnd_tile_valid_sizes[tile_id] = tile_info.valid_size;
        } else {
          bsnd_tile_offsets[tile_id] = GetBSNDFixedTileOffset(
              global_tile_id, num_bsnd_heads, MatrixSize);
          bsnd_tile_valid_sizes[tile_id] = MatrixSize;
        }
        const uint32_t bsnd_offset = bsnd_tile_offsets[tile_id];
        const uint32_t valid_size = bsnd_tile_valid_sizes[tile_id];
        const int row_stride = static_cast<int>(MatrixSize * num_bsnd_heads);
        wait_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
        if (valid_size < MatrixSize) {
          TileL1ABDynamic Y_dyn_l1_tile(valid_size, valid_size);
#ifdef MEGA_CHUNK_GDN_STREAMED_PAIR_SOLVE
          TASSIGN(Y_dyn_l1_tile,
                  0x0 + (3 + 2 * NumSolveStateTiles + tile_id) * TileLen *
                            sizeof(InputT));
#elif defined(MEGA_CHUNK_GDN_INTERLEAVED_SOLVE)
          TASSIGN(Y_dyn_l1_tile,
                  0x0 + (3 + 2 * NumTilesPerCubeIter + tile_id) * TileLen *
                            sizeof(InputT));
#else
          TASSIGN(Y_dyn_l1_tile,
                  0x0 + (5 + tile_id) * TileLen * sizeof(InputT));
#endif
          GlobalTileDynamicIn M_global_in_dyn(M + bsnd_offset,
                                              {1,
                                               1,
                                               1,
                                               static_cast<int>(valid_size),
                                               static_cast<int>(valid_size)},
                                              {1, 1, 1, row_stride, 1});
          TLOAD(Y_dyn_l1_tile, M_global_in_dyn);
          set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
          wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
          TFILLPAD(Y_dyn_l1_tile, Y_dyn_l1_tile);
        } else {
          GlobalTileIn M_global_in(M + bsnd_offset, {}, {row_stride});
          TLOAD(Y_l1_tile[tile_id], M_global_in);
        }
      } else {
        GlobalTileIn M_global_in(M + global_tile_ids[tile_id] * TileLen);
        wait_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
        TLOAD(Y_l1_tile[tile_id],
              M_global_in);  // Copies NumTilesPerCubeIter tiles at once
      }
      set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
    }

    constexpr uint32_t final_c_buffer_index = MatrixSize > FractalSize ? 1 : 0;
#ifdef MEGA_CHUNK_GDN_INTERLEAVED_SOLVE
    bool use_interleaved_solve = false;
    if constexpr (IsBSND && MatrixSize == 128 && NumTilesPerCubeIter >= 2 &&
                  NumTilesPerCubeIter % 2 == 0) {
      use_interleaved_solve = tiles_this_iter == NumTilesPerCubeIter;
    }
    if (use_interleaved_solve) {
      if constexpr (IsBSND && MatrixSize == 128 && NumTilesPerCubeIter >= 2 &&
                    NumTilesPerCubeIter % 2 == 0) {
        wait_flag(
            PIPE_FIX,
            PIPE_M,
            static_cast<event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
        for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
          wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
        }
#pragma unroll
        for (uint32_t pair = 0; pair < NumTilesPerCubeIter; pair += 2) {
          InvertTilesInterleavedBSND<InputT,
                                     OutputT,
                                     StoreT,
                                     TileL1AB,
                                     TileL0A,
                                     TileL0B,
                                     TileL0C,
                                     MatrixSize,
                                     FractalSize,
                                     2>(M_inv,
                                        num_bsnd_heads,
                                        bsnd_tile_offsets + pair,
                                        bsnd_tile_valid_sizes + pair,
                                        I_l1_tile,
                                        I_neg_l1_tile,
                                        Zero_l1_tile,
                                        M_neg_l1_tile + pair,
                                        X_l1_tile + pair,
                                        Y_l1_tile + pair,
                                        a_l0_tile,
                                        b_l0_tile,
                                        c_l0_tile,
                                        is_lower != 0);
        }
        for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
          set_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
        }
        set_flag(
            PIPE_FIX,
            PIPE_M,
            static_cast<event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
      }
    } else {
#endif
      for (uint32_t tile_id = 0; tile_id < tiles_this_iter; ++tile_id) {
        // Wait for previous cube iter to write result
        wait_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(tile_id));
        // Wait for loading new matrices from GM
        wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));

        if (use_precomputed_m_neg) {
          __gm__ InputT* M_neg_precomputed = M;
          if constexpr (IsBSND) {
            const uint32_t bsnd_offset = bsnd_tile_offsets[tile_id];
            const uint32_t valid_size = bsnd_tile_valid_sizes[tile_id];
            const int row_stride =
                static_cast<int>(MatrixSize * num_bsnd_heads);
            if (valid_size < MatrixSize) {
              TileL1ABDynamic M_neg_dyn_l1_tile(valid_size, valid_size);
#if defined(MEGA_CHUNK_GDN_INTERLEAVED_SOLVE) || \
    defined(MEGA_CHUNK_GDN_STREAMED_PAIR_SOLVE)
              TASSIGN(M_neg_dyn_l1_tile,
                      0x0 + (3 + tile_id) * TileLen * sizeof(InputT));
#else
            TASSIGN(M_neg_dyn_l1_tile, 0x0 + 3 * TileLen * sizeof(InputT));
#endif
              GlobalTileDynamicIn M_neg_global_in_dyn(
                  M_neg_precomputed + bsnd_offset,
                  {1,
                   1,
                   1,
                   static_cast<int>(valid_size),
                   static_cast<int>(valid_size)},
                  {1, 1, 1, row_stride, 1});
              TLOAD(M_neg_dyn_l1_tile, M_neg_global_in_dyn);
            } else {
              GlobalTileIn M_neg_global_in(
                  M_neg_precomputed + bsnd_offset, {}, {row_stride});
#if defined(MEGA_CHUNK_GDN_INTERLEAVED_SOLVE) || \
    defined(MEGA_CHUNK_GDN_STREAMED_PAIR_SOLVE)
              TLOAD(M_neg_l1_tile[tile_id], M_neg_global_in);
#else
            TLOAD(M_neg_l1_tile, M_neg_global_in);
#endif
            }
          } else {
            GlobalTileIn M_neg_global_in(M_neg_precomputed +
                                         global_tile_ids[tile_id] * TileLen);
#if defined(MEGA_CHUNK_GDN_INTERLEAVED_SOLVE) || \
    defined(MEGA_CHUNK_GDN_STREAMED_PAIR_SOLVE)
            TLOAD(M_neg_l1_tile[tile_id], M_neg_global_in);
#else
          TLOAD(M_neg_l1_tile, M_neg_global_in);
#endif
          }
          set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
          wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
        }

        InvertSingleTile<InputT,
                         TileL1AB,
                         TileL0A,
                         TileL0B,
                         TileL0C,
                         MatrixSize,
                         FractalSize,
                         NumTilesPerCubeIter>(
#if defined(MEGA_CHUNK_GDN_INTERLEAVED_SOLVE) || \
    defined(MEGA_CHUNK_GDN_STREAMED_PAIR_SOLVE)
            X_l1_tile[tile_id],
            I_l1_tile,
            I_neg_l1_tile,
            M_neg_l1_tile[tile_id],
            Zero_l1_tile,
#else
          X_l1_tile,
          I_l1_tile,
          I_neg_l1_tile,
          M_neg_l1_tile,
          Zero_l1_tile,
#endif
            Y_l1_tile[tile_id],
            a_l0_tile,
            b_l0_tile,
            c_l0_tile,
            tile_id,
            is_lower != 0,
            use_precomputed_m_neg);

        // Allow next cube_iter to proceed for this tile_id
        set_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));

        /* Store result */
        if constexpr (IsBSND) {
          const uint32_t bsnd_offset = bsnd_tile_offsets[tile_id];
          const uint32_t valid_size = bsnd_tile_valid_sizes[tile_id];
          const int row_stride = static_cast<int>(MatrixSize * num_bsnd_heads);
          if (valid_size < MatrixSize) {
            TileL0CDynamic c_l0_tail_tile(valid_size, valid_size);
            TASSIGN(c_l0_tail_tile,
                    0x0 + final_c_buffer_index * TileLen * sizeof(OutputT));
            GlobalTileDynamicOut M_inv_global_out_dyn(
                M_inv + bsnd_offset,
                {1,
                 1,
                 1,
                 static_cast<int>(valid_size),
                 static_cast<int>(valid_size)},
                {1, 1, 1, row_stride, 1});
            TSTORE(M_inv_global_out_dyn, c_l0_tail_tile);
          } else {
            GlobalTileOut M_inv_global_out(
                M_inv + bsnd_offset, {}, {row_stride});
            TSTORE(M_inv_global_out, c_l0_tile[final_c_buffer_index]);
          }
        } else {
          GlobalTileOut M_inv_global_out(M_inv +
                                         global_tile_ids[tile_id] * TileLen);
          TSTORE(M_inv_global_out, c_l0_tile[final_c_buffer_index]);
        }
        next_tile_id_that_waits_for_pipe_fix_pipe_m =
            (tile_id + 1) % NumTilesPerCubeIter;
        set_flag(
            PIPE_FIX,
            PIPE_M,
            static_cast<event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
      }
#ifdef MEGA_CHUNK_GDN_INTERLEAVED_SOLVE
    }
#endif
  }
  for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
    wait_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
  }
  wait_flag(PIPE_FIX,
            PIPE_M,
            static_cast<event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
}

#ifdef MEGA_CHUNK_GDN_A5_SPLIT64_SOLVE
/*
 * Variant 22 is deliberately an all-or-nothing invocation switch.  Both
 * Cube and Vector evaluate this same predicate before entering the split
 * protocol, so a ragged sequence can never leave the two subblocks on
 * different event paths.  total_tiles includes heads; dividing by the head
 * count gives the exact number of 128-token chunks represented by
 * cu_seqlens.
 */
template <uint32_t MatrixSize>
AICORE inline bool CanUseA5Split64Solve(uint32_t total_tiles,
                                        uint32_t num_bsnd_heads,
                                        __gm__ int32_t* cu_seqlens) {
  if constexpr (MatrixSize != 128) {
    return false;
  }
  if (cu_seqlens == nullptr || total_tiles == 0 ||
      total_tiles % num_bsnd_heads != 0) {
    return false;
  }

  const uint32_t expected_chunks = total_tiles / num_bsnd_heads;
  int32_t seq_start = cu_seqlens[0];
  if (seq_start < 0) {
    return false;
  }
  uint32_t accumulated_chunks = 0;
  // Every non-empty aligned sequence contributes at least one chunk, so
  // expected_chunks is also a safe finite upper bound on the scan.
  for (uint32_t seq_idx = 0;
       seq_idx < expected_chunks && accumulated_chunks < expected_chunks;
       ++seq_idx) {
    const int32_t seq_end = cu_seqlens[seq_idx + 1];
    if (seq_end <= seq_start) {
      return false;
    }
    const uint32_t seq_len = static_cast<uint32_t>(seq_end - seq_start);
    if (seq_len % MatrixSize != 0) {
      return false;
    }
    const uint32_t seq_chunks = seq_len / MatrixSize;
    if (seq_chunks > expected_chunks - accumulated_chunks) {
      return false;
    }
    accumulated_chunks += seq_chunks;
    seq_start = seq_end;
  }
  return accumulated_chunks == expected_chunks;
}
#endif

#if defined(__DAV_C310_CUBE__)
/*
 * A5-specific triangular inverse.
 *
 * The A2/A3 implementation above assembles the inverse by extracting 16x16
 * L0 fractals.  A5 uses a different L0A base layout, so those address-based
 * fractal aliases are not portable.  For the fixed 128x128 GDN chunk, use the
 * finite Neumann series instead:
 *
 *   (I + A)^-1 = I - A + A^2 - ... - A^127
 *
 * With P = -A and X = I + P, six doubling rounds
 * `P = P @ P; X = X + X @ P` produce all powers through P^127.  A is strict
 * lower triangular, hence P^128 is exactly zero.  This path only uses full
 * A5-supported L1/L0 matmuls and avoids architecture-dependent fractal
 * addressing.
 */
template <typename TileL1, typename TileL0A, typename TileL0B, typename TileL0C>
AICORE inline void A5Matmul(TileL0C c,
                            TileL0A a,
                            TileL0B b,
                            TileL1 left,
                            TileL1 right,
                            bool accumulate,
                            event_t event = EVENT_ID0) {
  if (accumulate) {
    set_flag(PIPE_M, PIPE_MTE1, event);
    wait_flag(PIPE_M, PIPE_MTE1, event);
  }
  TMOV(a, left);
  TMOV(b, right);
  set_flag(PIPE_MTE1, PIPE_M, event);
  wait_flag(PIPE_MTE1, PIPE_M, event);
  if (accumulate) {
    TMATMUL_ACC(c, c, a, b);
  } else {
    TMATMUL(c, a, b);
  }
}

template <typename TileL1, typename TileL0C>
AICORE inline void A5MovAccToL1(TileL1 dst,
                                TileL0C src,
                                event_t event = EVENT_ID0) {
  set_flag(PIPE_M, PIPE_FIX, event);
  wait_flag(PIPE_M, PIPE_FIX, event);
  TMOV(dst, src);
  set_flag(PIPE_FIX, PIPE_MTE1, event);
  wait_flag(PIPE_FIX, PIPE_MTE1, event);
}

#ifdef MEGA_CHUNK_GDN_A5_SPLIT64_SOLVE
/*
 * Full 128x128 lower-triangular matrices are split as
 *
 *     L = [ A  0 ]       L^-1 = [ A^-1                0 ]
 *         [ B  D ]              [ -D^-1 B A^-1     D^-1 ] .
 *
 * The per-MIX packed fp32 slot already occupies 64 KiB.  Variant 22 reuses
 * it without expansion as four consecutive 64x64 regions:
 *
 *   block 0: A^-1, block 1: D^-1, block 2: -D^-1 B A^-1,
 *   block 3: reserved (never read; AIV creates the upper-right zero in UB).
 *
 * Only the first three blocks are published.  The existing event 8/9/10
 * protocol still brackets the complete three-block publication and the two
 * disjoint AIV scatters.
 */
template <typename InputT, typename OutputT, typename StoreT>
AICORE inline void TriInvA5Split64RecursiveKernel(
    __gm__ StoreT* M_inv,
    __gm__ InputT* M,
    __gm__ InputT* I_neg,
    uint32_t total_tiles,
    uint32_t num_bsnd_heads,
    __gm__ int32_t* cu_seqlens,
    __gm__ InputT* packed_workspace,
    uint32_t is_lower) {
  (void)M_inv;
  (void)is_lower;
  static_assert(sizeof(InputT) == 2,
                "The split64 workspace aliases two fp16 slots as fp32.");
  constexpr uint32_t FullSize = 128;
  constexpr uint32_t HalfSize = 64;
  constexpr uint32_t FractalSize = 16;
  constexpr uint32_t FullLen = FullSize * FullSize;
  constexpr uint32_t HalfLen = HalfSize * HalfSize;
  constexpr uint32_t HalfBytes = HalfLen * sizeof(InputT);

  using HalfShape = TileShape2D<InputT, HalfSize, HalfSize, Layout::ND>;
  using ConstStride = pto::Stride<1, 1, 1, FullSize, 1>;
  using ConstGlobal = GlobalTensor<InputT, HalfShape, ConstStride, Layout::ND>;
  using BsndStride = pto::Stride<1, 1, 1, -1, 1>;
  using BsndGlobal = GlobalTensor<InputT, HalfShape, BsndStride, Layout::ND>;
  using PackedShape = TileShape2D<float, HalfSize, HalfSize, Layout::ND>;
  using PackedStride = BaseShape2D<float, HalfSize, HalfSize, Layout::ND>;
  using PackedGlobal =
      GlobalTensor<float, PackedShape, PackedStride, Layout::ND>;
  using TileL1AB = Tile<TileType::Mat,
                        InputT,
                        HalfSize,
                        HalfSize,
                        BLayout::ColMajor,
                        HalfSize,
                        HalfSize,
                        SLayout::RowMajor,
                        512,
                        PadValue::Zero>;
  using TileL0A = TileLeft<InputT, HalfSize, HalfSize>;
  using TileL0B = TileRight<InputT, HalfSize, HalfSize>;
  using TileL0C = TileAcc<OutputT, HalfSize, HalfSize>;

  TileL1AB i_l1_tile;
  TileL1AB i_neg_l1_tile;
  TileL1AB zero_l1_tile;
  TileL1AB m_neg_l1_tile;
  TileL1AB x_l1_tile;
  TileL1AB y_l1_tile;
  TileL1AB a_inv_l1_tile;
  TileL1AB d_inv_l1_tile;
  TileL1AB cross_l1_tile;
  TileL0A a_l0_tile[2];
  TileL0B b_l0_tile[2];
  TileL0C c_l0_tile[2];

  TASSIGN(i_l1_tile, 0);
  TASSIGN(i_neg_l1_tile, HalfBytes);
  TASSIGN(zero_l1_tile, 2 * HalfBytes);
  TASSIGN(m_neg_l1_tile, 3 * HalfBytes);
  TASSIGN(x_l1_tile, 4 * HalfBytes);
  TASSIGN(y_l1_tile, 5 * HalfBytes);
  TASSIGN(a_inv_l1_tile, 6 * HalfBytes);
  TASSIGN(d_inv_l1_tile, 7 * HalfBytes);
  TASSIGN(cross_l1_tile, 8 * HalfBytes);
  for (uint32_t buffer = 0; buffer < 2; ++buffer) {
    TASSIGN(a_l0_tile[buffer], buffer * HalfBytes);
    TASSIGN(b_l0_tile[buffer], buffer * HalfBytes);
    TASSIGN(c_l0_tile[buffer], buffer * HalfLen * sizeof(OutputT));
  }

  ConstGlobal global_i_neg(I_neg);
  TLOAD(i_neg_l1_tile, global_i_neg);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  PrepareAuxiliaryMatrices<TileL1AB, TileL0A, TileL0B, TileL0C>(i_neg_l1_tile,
                                                                zero_l1_tile,
                                                                i_l1_tile,
                                                                a_l0_tile[0],
                                                                b_l0_tile[0],
                                                                c_l0_tile[0]);

  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (uint32_t global_tile_id = get_block_idx(); global_tile_id < total_tiles;
       global_tile_id += get_block_num()) {
    const BSNDVarlenTileInfo tile_info = GetBSNDVarlenTileInfoFromCuSeqlens(
        global_tile_id, num_bsnd_heads, FullSize, cu_seqlens);
    const uint32_t row_stride = FullSize * num_bsnd_heads;
    __gm__ float* packed_out = reinterpret_cast<__gm__ float*>(
        packed_workspace + get_block_idx() * 2 * FullLen);

    // A^-1.  Alignment eligibility guarantees a complete 64x64 load.
    BsndGlobal global_a(
        M + tile_info.bsnd_offset, {}, {static_cast<int>(row_stride)});
    TLOAD(y_l1_tile, global_a);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    InvertSingleTile<InputT,
                     TileL1AB,
                     TileL0A,
                     TileL0B,
                     TileL0C,
                     HalfSize,
                     FractalSize,
                     1>(x_l1_tile,
                        i_l1_tile,
                        i_neg_l1_tile,
                        m_neg_l1_tile,
                        zero_l1_tile,
                        y_l1_tile,
                        a_l0_tile,
                        b_l0_tile,
                        c_l0_tile,
                        0,
                        true);
    constexpr uint32_t FinalBuffer = 1;
    TMOV(a_inv_l1_tile, c_l0_tile[FinalBuffer]);
    PackedGlobal global_a_inv(packed_out);
    TSTORE(global_a_inv, c_l0_tile[FinalBuffer]);
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

    // D^-1.
    BsndGlobal global_d(
        M + tile_info.bsnd_offset + HalfSize * row_stride + HalfSize,
        {},
        {static_cast<int>(row_stride)});
    TLOAD(y_l1_tile, global_d);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    InvertSingleTile<InputT,
                     TileL1AB,
                     TileL0A,
                     TileL0B,
                     TileL0C,
                     HalfSize,
                     FractalSize,
                     1>(x_l1_tile,
                        i_l1_tile,
                        i_neg_l1_tile,
                        m_neg_l1_tile,
                        zero_l1_tile,
                        y_l1_tile,
                        a_l0_tile,
                        b_l0_tile,
                        c_l0_tile,
                        0,
                        true);
    TMOV(d_inv_l1_tile, c_l0_tile[FinalBuffer]);
    PackedGlobal global_d_inv(packed_out + HalfLen);
    TSTORE(global_d_inv, c_l0_tile[FinalBuffer]);
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

    // X10 = -D^-1 B A^-1.  Each bounded intermediate is retained in L1;
    // the final accumulator stays fp32 through the packed publication.
    BsndGlobal global_b(M + tile_info.bsnd_offset + HalfSize * row_stride,
                        {},
                        {static_cast<int>(row_stride)});
    TLOAD(y_l1_tile, global_b);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    A5Matmul(c_l0_tile[0],
             a_l0_tile[0],
             b_l0_tile[0],
             i_neg_l1_tile,
             y_l1_tile,
             false);
    A5MovAccToL1(cross_l1_tile, c_l0_tile[0]);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

    A5Matmul(c_l0_tile[0],
             a_l0_tile[0],
             b_l0_tile[0],
             d_inv_l1_tile,
             cross_l1_tile,
             false);
    A5MovAccToL1(cross_l1_tile, c_l0_tile[0]);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

    A5Matmul(c_l0_tile[0],
             a_l0_tile[0],
             b_l0_tile[0],
             cross_l1_tile,
             a_inv_l1_tile,
             false);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    PackedGlobal global_x10(packed_out + 2 * HalfLen);
    TSTORE(global_x10, c_l0_tile[0]);

    set_intra_block(PIPE_FIX, 8);
    set_intra_block(PIPE_FIX, 8 + SYNC_FLAG_ID_MAX);
    wait_intra_block(PIPE_S, 9);
    wait_intra_block(PIPE_S, 9 + SYNC_FLAG_ID_MAX);
    set_intra_block(PIPE_S, 10);
    set_intra_block(PIPE_S, 10 + SYNC_FLAG_ID_MAX);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}
#endif

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void A5CopyCbufToUbuf(
    typename DstTile::TileDType __out__ dst,
    typename SrcTile::TileDType __in__ src,
    uint16_t vector,
    uint16_t block_len) {
  copy_cbuf_to_ubuf((__ubuf__ void*)__cce_get_tile_ptr(dst),
                    (__cbuf__ void*)__cce_get_tile_ptr(src),
                    vector,
                    1,
                    block_len,
                    0,
                    0);
}

template <typename DstTile, typename SrcT>
__tf__ PTO_INTERNAL void A5CopyGmToCbuf(typename DstTile::TileDType __out__ dst,
                                        __gm__ SrcT* src,
                                        uint32_t byte_len) {
  // The workspace is already in Cube-native NZ physical order.  A raw
  // aligned copy preserves all four 16x16 fractals; PTO TLOAD's NZ path on
  // C310 currently transfers only one M1 row for a 32x32 half tile.
  copy_gm_to_cbuf_align_v2((__cbuf__ uint8_t*)__cce_get_tile_ptr(dst),
                           (__gm__ uint8_t*)src,
                           0,
                           1,
                           byte_len,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0);
}

template <typename TileL1, typename TileL0A, typename TileL0B, typename TileL0C>
AICORE inline void A5MatmulFinalAcc(TileL0C c,
                                    TileL0A a,
                                    TileL0B b,
                                    TileL1 left,
                                    TileL1 right) {
  pipe_barrier(PIPE_ALL);
  TMOV(a, left);
  TMOV(b, right);
  pipe_barrier(PIPE_ALL);
  TMATMUL_ACC(c, c, a, b);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
}

template <typename InputT,
          typename OutputT,
          uint32_t MatrixSize,
          bool IsBSND,
          typename StoreT = OutputT>
AICORE inline void TriInvA5SeriesKernel(__gm__ StoreT* M_inv,
                                        __gm__ InputT* M,
                                        __gm__ InputT* I_neg,
                                        uint32_t total_tiles,
                                        uint32_t num_bsnd_heads,
                                        __gm__ int32_t* cu_seqlens,
                                        __gm__ InputT* packed_workspace) {
  static_assert(IsBSND, "The A5 GDN solver expects BSND matrices.");
  // The fp32 UB recurrence is stable for the short/partial blocks seen in
  // prefill, but an all-scalar 128x128 solve can exceed A5's execution
  // budget.  Keep full and large blocks on Cube while fixing the precision
  // sensitive short-block case without a GM fallback.
  constexpr uint32_t StableUbMaxSize = 64;
  constexpr uint32_t TileLen = MatrixSize * MatrixSize;
  constexpr uint32_t TileBytes = TileLen * sizeof(InputT);

  using GlobalShape = TileShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalStride = pto::Stride<1, 1, 1, -1, 1>;
  using GlobalIn = GlobalTensor<InputT, GlobalShape, GlobalStride, Layout::ND>;
  using GlobalOutShape =
      TileShape2D<StoreT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalOut =
      GlobalTensor<StoreT, GlobalOutShape, GlobalStride, Layout::ND>;
  using DynamicShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using DynamicStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  using DynamicIn =
      GlobalTensor<InputT, DynamicShape, DynamicStride, Layout::ND>;
  using DynamicOut =
      GlobalTensor<StoreT, DynamicShape, DynamicStride, Layout::ND>;
  using IdentityStride =
      BaseShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using IdentityGlobal =
      GlobalTensor<InputT, GlobalShape, IdentityStride, Layout::ND>;
  using PackedOutStride =
      BaseShape2D<StoreT, MatrixSize, MatrixSize, Layout::ND>;
  using PackedOut =
      GlobalTensor<StoreT, GlobalOutShape, PackedOutStride, Layout::ND>;

  using TileL1 = Tile<TileType::Mat,
                      InputT,
                      MatrixSize,
                      MatrixSize,
                      BLayout::ColMajor,
                      MatrixSize,
                      MatrixSize,
                      SLayout::RowMajor,
                      512,
                      PadValue::Zero>;
  using DynamicTileL1 = Tile<TileType::Mat,
                             InputT,
                             MatrixSize,
                             MatrixSize,
                             BLayout::ColMajor,
                             DYNAMIC,
                             DYNAMIC,
                             SLayout::RowMajor,
                             512,
                             PadValue::Zero>;
  using TileL0A = TileLeft<InputT, MatrixSize, MatrixSize>;
  using TileL0B = TileRight<InputT, MatrixSize, MatrixSize>;
  using TileL0C = TileAcc<OutputT, MatrixSize, MatrixSize>;
  using DynamicTileL0C =
      TileAcc<OutputT, MatrixSize, MatrixSize, DYNAMIC, DYNAMIC>;

  TileL1 i_neg_l1;
  TileL1 i_l1;
  TileL1 a_l1;
  TileL1 p_l1;
  TileL1 x_l1;
  TileL0A l0a;
  TileL0B l0b;
  TileL0C l0c;
  TASSIGN(i_neg_l1, 0);
  TASSIGN(i_l1, TileBytes);
  TASSIGN(a_l1, 2 * TileBytes);
  TASSIGN(p_l1, 3 * TileBytes);
  TASSIGN(x_l1, 4 * TileBytes);
  TASSIGN(l0a, 0);
  TASSIGN(l0b, 0);
  TASSIGN(l0c, 0);

  IdentityGlobal i_neg_global(I_neg);
  TLOAD(i_neg_l1, i_neg_global);
  pipe_barrier(PIPE_ALL);

  // I = (-I) @ (-I).
  A5Matmul(l0c, l0a, l0b, i_neg_l1, i_neg_l1, false);
  TMOV(i_l1, l0c);
  pipe_barrier(PIPE_ALL);

  for (uint32_t global_tile_id = get_block_idx(); global_tile_id < total_tiles;
       global_tile_id += get_block_num()) {
    uint32_t bsnd_offset;
    uint32_t valid_size;
    if (cu_seqlens != nullptr) {
      const BSNDVarlenTileInfo tile_info = GetBSNDVarlenTileInfoFromCuSeqlens(
          global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens);
      bsnd_offset = tile_info.bsnd_offset;
      valid_size = tile_info.valid_size;
    } else {
      bsnd_offset =
          GetBSNDFixedTileOffset(global_tile_id, num_bsnd_heads, MatrixSize);
      valid_size = MatrixSize;
    }
    const int row_stride = static_cast<int>(MatrixSize * num_bsnd_heads);

    // Vector0 gathers the BSND rows into a contiguous per-MIX workspace.
    // This avoids A5's broken 128x128 strided ND->NZ conversion.
    wait_intra_block(PIPE_MTE2, 7);
    if (valid_size <= StableUbMaxSize) {
      // The stable UB path has already scattered the inverse.  Relay
      // its completion to both Vector subblocks and skip Cube work.
      set_intra_block(PIPE_S, 8);
      set_intra_block(PIPE_S, 8 + SYNC_FLAG_ID_MAX);
      continue;
    }
    __gm__ InputT* packed_in = packed_workspace + get_block_idx() * 2 * TileLen;
    __gm__ StoreT* packed_out = reinterpret_cast<__gm__ StoreT*>(
        packed_workspace + (get_block_idx() * 2 + 1) * TileLen);

    IdentityGlobal global_a(packed_in);
    TLOAD(a_l1, global_a);
    pipe_barrier(PIPE_ALL);

    // P = -A.
    A5Matmul(l0c, l0a, l0b, i_neg_l1, a_l1, false);
    TMOV(p_l1, l0c);
    pipe_barrier(PIPE_ALL);

    // X = I + P.
    A5Matmul(l0c, l0a, l0b, i_l1, i_l1, false);
    A5Matmul(l0c, l0a, l0b, i_l1, p_l1, true);
    TMOV(x_l1, l0c);
    pipe_barrier(PIPE_ALL);

    for (uint32_t power = 2; power < MatrixSize; power *= 2) {
      // P now represents (-A)^power.
      A5Matmul(l0c, l0a, l0b, p_l1, p_l1, false);
      TMOV(p_l1, l0c);
      pipe_barrier(PIPE_ALL);

      // X <- X * (I + P) = X + X * P.
      A5Matmul(l0c, l0a, l0b, x_l1, i_l1, false);
      if (power == MatrixSize / 2) {
        A5MatmulFinalAcc(l0c, l0a, l0b, x_l1, p_l1);
      } else {
        A5Matmul(l0c, l0a, l0b, x_l1, p_l1, true);
        TMOV(x_l1, l0c);
        pipe_barrier(PIPE_ALL);
      }
    }

    PackedOut global_out(packed_out);
    TSTORE(global_out, l0c);
    set_intra_block(PIPE_FIX, 8);
    set_intra_block(PIPE_FIX, 8 + SYNC_FLAG_ID_MAX);

    // Keep both Vector subblocks at the stage boundary until Vector0 has
    // scattered the packed result back to BSND.
    wait_intra_block(PIPE_S, 9);
    set_intra_block(PIPE_S, 10);
    set_intra_block(PIPE_S, 10 + SYNC_FLAG_ID_MAX);
    // TSTORE reads L0C through FIX.  Complete that read before the next
    // matrix (or the following fused stage) reuses the accumulator.
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
}

/*
 * A5 port of megagdn-pto's recursive Cube inverse.  Vector0 converts the
 * strided BSND matrix to a contiguous per-MIX slot before this routine runs;
 * all recursive 16/32/64/128 block assembly then stays in Cube L1/L0, and
 * only the final inverse is written to the second packed slot for scatter.
 */
template <typename InputT,
          typename OutputT,
          uint32_t MatrixSize,
          bool IsBSND,
          typename StoreT = OutputT,
          typename PackedStoreT = StoreT,
          bool DirectBsndInput = false>
AICORE inline void TriInvA5PackedRecursiveKernel(
    __gm__ StoreT* M_inv,
    __gm__ InputT* M,
    __gm__ InputT* I_neg,
    uint32_t total_tiles,
    uint32_t num_bsnd_heads,
    __gm__ int32_t* cu_seqlens,
    __gm__ InputT* packed_workspace,
    uint32_t is_lower) {
  static_assert(IsBSND,
                "The A5 packed recursive solver expects BSND matrices.");
#ifdef MEGA_CHUNK_GDN_A5_SPLIT64_SOLVE
  if constexpr (MatrixSize == 128) {
    if (is_lower != 0 && CanUseA5Split64Solve<MatrixSize>(
                             total_tiles, num_bsnd_heads, cu_seqlens)) {
      TriInvA5Split64RecursiveKernel<InputT, OutputT, StoreT>(M_inv,
                                                              M,
                                                              I_neg,
                                                              total_tiles,
                                                              num_bsnd_heads,
                                                              cu_seqlens,
                                                              packed_workspace,
                                                              is_lower);
      return;
    }
  }
#endif
  (void)M_inv;
  (void)M;
  (void)num_bsnd_heads;
  (void)cu_seqlens;
  constexpr uint32_t TileLen = MatrixSize * MatrixSize;
  constexpr uint32_t TileBytes = TileLen * sizeof(InputT);
  constexpr uint32_t FractalSize = 16;

  using GlobalShape = TileShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalStride = BaseShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalIn = GlobalTensor<InputT, GlobalShape, GlobalStride, Layout::ND>;
  using BsndStride = pto::Stride<1, 1, 1, -1, 1>;
  using BsndGlobalIn =
      GlobalTensor<InputT, GlobalShape, BsndStride, Layout::ND>;
  using DynamicShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using DynamicStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  using DynamicGlobalIn =
      GlobalTensor<InputT, DynamicShape, DynamicStride, Layout::ND>;
  using GlobalOutShape =
      TileShape2D<PackedStoreT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalOutStride =
      BaseShape2D<PackedStoreT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalOut =
      GlobalTensor<PackedStoreT, GlobalOutShape, GlobalOutStride, Layout::ND>;
  using TileL1AB = Tile<TileType::Mat,
                        InputT,
                        MatrixSize,
                        MatrixSize,
                        BLayout::ColMajor,
                        MatrixSize,
                        MatrixSize,
                        SLayout::RowMajor,
                        512,
                        PadValue::Zero>;
  using DynamicTileL1AB = Tile<TileType::Mat,
                               InputT,
                               MatrixSize,
                               MatrixSize,
                               BLayout::ColMajor,
                               DYNAMIC,
                               DYNAMIC,
                               SLayout::RowMajor,
                               512,
                               PadValue::Zero>;
  using TileL0A = TileLeft<InputT, MatrixSize, MatrixSize>;
  using TileL0B = TileRight<InputT, MatrixSize, MatrixSize>;
  using TileL0C = TileAcc<OutputT, MatrixSize, MatrixSize>;

  TileL1AB i_l1_tile;
  TileL1AB i_neg_l1_tile;
  TileL1AB zero_l1_tile;
  TileL1AB m_neg_l1_tile;
  TileL1AB x_l1_tile;
  TileL1AB y_l1_tile;
  TileL0A a_l0_tile[2];
  TileL0B b_l0_tile[2];
  TileL0C c_l0_tile[2];

  TASSIGN(i_l1_tile, 0);
  TASSIGN(i_neg_l1_tile, TileBytes);
  TASSIGN(zero_l1_tile, 2 * TileBytes);
  TASSIGN(m_neg_l1_tile, 3 * TileBytes);
  TASSIGN(x_l1_tile, 4 * TileBytes);
  TASSIGN(y_l1_tile, 5 * TileBytes);
  for (uint32_t buffer = 0; buffer < 2; ++buffer) {
    TASSIGN(a_l0_tile[buffer], buffer * TileBytes);
    TASSIGN(b_l0_tile[buffer], buffer * TileBytes);
    TASSIGN(c_l0_tile[buffer], buffer * TileLen * sizeof(OutputT));
  }

  GlobalIn global_i_neg(I_neg);
  TLOAD(i_neg_l1_tile, global_i_neg);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  PrepareAuxiliaryMatrices<TileL1AB, TileL0A, TileL0B, TileL0C>(i_neg_l1_tile,
                                                                zero_l1_tile,
                                                                i_l1_tile,
                                                                a_l0_tile[0],
                                                                b_l0_tile[0],
                                                                c_l0_tile[0]);

  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (uint32_t global_tile_id = get_block_idx(); global_tile_id < total_tiles;
       global_tile_id += get_block_num()) {
    __gm__ InputT* packed_in = packed_workspace + get_block_idx() * 2 * TileLen;
    __gm__ InputT* workspace_base =
        packed_workspace + get_block_idx() * 2 * TileLen;
    __gm__ PackedStoreT* packed_out;
    if constexpr (std::is_same_v<PackedStoreT, float>) {
      packed_out = reinterpret_cast<__gm__ PackedStoreT*>(workspace_base);
    } else {
      packed_out = reinterpret_cast<__gm__ PackedStoreT*>(
          packed_workspace + (get_block_idx() * 2 + 1) * TileLen);
    }

    GlobalIn global_in(packed_in);
    if constexpr (DirectBsndInput) {
      const BSNDVarlenTileInfo tile_info =
          cu_seqlens != nullptr
              ? GetBSNDVarlenTileInfoFromCuSeqlens(
                    global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens)
              : BSNDVarlenTileInfo{
                    GetBSNDFixedTileOffset(
                        global_tile_id, num_bsnd_heads, MatrixSize),
                    MatrixSize};
      const uint32_t row_stride = MatrixSize * num_bsnd_heads;
      if (tile_info.valid_size < MatrixSize) {
        DynamicTileL1AB dynamic_input(tile_info.valid_size,
                                      tile_info.valid_size);
        TASSIGN(dynamic_input, 5 * TileBytes);
        DynamicGlobalIn source(M + tile_info.bsnd_offset,
                               {1,
                                1,
                                1,
                                static_cast<int>(tile_info.valid_size),
                                static_cast<int>(tile_info.valid_size)},
                               {1, 1, 1, static_cast<int>(row_stride), 1});
        TLOAD(dynamic_input, source);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        TFILLPAD(dynamic_input, dynamic_input);
      } else {
        BsndGlobalIn source(
            M + tile_info.bsnd_offset, {}, {static_cast<int>(row_stride)});
        TLOAD(y_l1_tile, source);
      }
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    } else {
      wait_intra_block(PIPE_MTE2, 7);
      TLOAD(y_l1_tile, global_in);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    }
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    InvertSingleTile<InputT,
                     TileL1AB,
                     TileL0A,
                     TileL0B,
                     TileL0C,
                     MatrixSize,
                     FractalSize,
                     1>(x_l1_tile,
                        i_l1_tile,
                        i_neg_l1_tile,
                        m_neg_l1_tile,
                        zero_l1_tile,
                        y_l1_tile,
                        a_l0_tile,
                        b_l0_tile,
                        c_l0_tile,
                        0,
                        is_lower != 0);

    constexpr uint32_t FinalBuffer = MatrixSize > FractalSize ? 1 : 0;
#ifdef MEGA_CHUNK_GDN_A5_PACKED_RECURSIVE_IDENTITY_PROBE
    pipe_barrier(PIPE_ALL);
    TLOAD(y_l1_tile, global_in);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    A5Matmul(c_l0_tile[FinalBuffer],
             a_l0_tile[0],
             b_l0_tile[0],
             y_l1_tile,
             y_l1_tile,
             false);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    GlobalOut global_out(packed_out);
    TSTORE(global_out, c_l0_tile[FinalBuffer]);
    set_intra_block(PIPE_FIX, 8);
    set_intra_block(PIPE_FIX, 8 + SYNC_FLAG_ID_MAX);

#ifdef MEGA_CHUNK_GDN_A5_DUAL_FP32_SCATTER
    // Do not release either AIV into WY until both disjoint row owners
    // have drained their final BSND stores.
    wait_intra_block(PIPE_S, 9);
    wait_intra_block(PIPE_S, 9 + SYNC_FLAG_ID_MAX);
#else
    wait_intra_block(PIPE_S, 9);
#endif
    set_intra_block(PIPE_S, 10);
    set_intra_block(PIPE_S, 10 + SYNC_FLAG_ID_MAX);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}

/*
 * Numerically stable blocked A5 triangular inverse for large blocks.
 *
 * The full-matrix Neumann series above is fast for benign inputs, but its
 * fp16 P/X writebacks can grow far beyond the final inverse and overflow.
 * Here the diagonal 16x16 blocks use the same short nilpotent series, while
 * off-diagonal blocks are obtained by block forward substitution:
 *
 *   X_ij = -X_ii * sum(L_ik * X_kj, k=j..i-1).
 *
 * Every block sum stays in fp32 L0C; only completed, bounded 16x16 blocks are
 * converted to fp16.  This retains Cube tiling and local MIX synchronization
 * without the unstable full 128x128 intermediates.
 */
template <typename InputT,
          typename OutputT,
          uint32_t MatrixSize,
          bool IsBSND,
          typename StoreT = OutputT>
AICORE inline void TriInvA5BlockedKernel(__gm__ StoreT* M_inv,
                                         __gm__ InputT* M,
                                         __gm__ InputT* I_neg,
                                         uint32_t total_tiles,
                                         uint32_t num_bsnd_heads,
                                         __gm__ int32_t* cu_seqlens,
                                         __gm__ InputT* packed_workspace) {
  static_assert(IsBSND, "The A5 GDN solver expects BSND matrices.");
  static_assert(MatrixSize % 16 == 0,
                "The A5 blocked solver requires 16-aligned matrices.");
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
  // C310's PTO Cube path does not produce a usable 16x16 result for this
  // layout.  A 32x32 block is the smallest reliable unit and also halves
  // the number of dependency boundaries in each matrix dimension.
  constexpr uint32_t BlockSize = 32;
#else
  constexpr uint32_t BlockSize = 16;
#endif
  constexpr uint32_t BlocksPerMatrix = MatrixSize / BlockSize;
  constexpr uint32_t BlockLen = BlockSize * BlockSize;
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
  constexpr uint32_t LowerBlocks = BlocksPerMatrix * (BlocksPerMatrix + 1) / 2;
#endif
  constexpr uint32_t TileLen = MatrixSize * MatrixSize;
  constexpr uint32_t BlockBytes = BlockSize * BlockSize * sizeof(InputT);
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
  // Full chunks use the blocked Cube solver.  Partial chunks retain the
  // fp32 UB recurrence, which naturally handles non-16-aligned tails.
  constexpr uint32_t StableUbMaxSize = MatrixSize - 1;
#ifdef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
  constexpr uint32_t FloatBlockBytes = BlockSize * BlockSize * sizeof(float);
#endif
#ifdef MEGA_CHUNK_GDN_A5_CUBE_DIAG_AIV_OFFDIAG
  // Match the AIV UB layout: keep the dense input resident at address zero
  // and publish Cube diagonals into the otherwise-unused output staging
  // window after the fp32 partial-inverse tile.
  constexpr uint32_t HybridFirstSubblockColumns = (MatrixSize * 27 + 127) / 128;
  constexpr uint32_t HybridMaxColumnsPerSubblock =
      MatrixSize - HybridFirstSubblockColumns;
  constexpr uint32_t HybridStorageColumnsPerSubblock =
      (HybridMaxColumnsPerSubblock + 15) / 16 * 16;
  constexpr uint32_t HybridPublishBaseBytes =
      TileLen * sizeof(InputT) +
      MatrixSize * HybridStorageColumnsPerSubblock * sizeof(float);
#endif
#else
  constexpr uint32_t StableUbMaxSize = MatrixSize;
#endif

  using BlockShape = Shape<1, 1, 1, BlockSize, BlockSize>;
  using IdentityBlockStride = pto::Stride<1, 1, 1, MatrixSize, 1>;
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
  constexpr uint32_t FractalSize = 16;
  constexpr uint32_t FractalsPerBlock = BlockSize / FractalSize;
  constexpr uint32_t FractalLen = FractalSize * FractalSize;
  // Vector0 packs each 32x32 block directly in NZ order
  // [N1, M1, M0=16, N0=16].  Declaring that physical layout here avoids
  // A5's unreliable ND->NZ conversion while keeping every GM transfer
  // contiguous.
  using BlockNzShape =
      Shape<1, FractalsPerBlock, FractalsPerBlock, FractalSize, FractalSize>;
  using BlockNzStride = pto::Stride<BlockLen,
                                    FractalsPerBlock * FractalLen,
                                    FractalLen,
                                    FractalSize,
                                    1>;
  using BlockStride = pto::Stride<1, 1, 1, BlockSize, 1>;
#else
  using BlockStride = pto::Stride<1, 1, 1, MatrixSize, 1>;
#endif
  using IdentityBlockIn =
      GlobalTensor<InputT, BlockShape, IdentityBlockStride, Layout::ND>;
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
  using InputBlock =
      GlobalTensor<InputT, BlockNzShape, BlockNzStride, Layout::NZ>;
#else
  using InputBlock = GlobalTensor<InputT, BlockShape, BlockStride, Layout::ND>;
#endif
  using BlockOut = GlobalTensor<StoreT, BlockShape, BlockStride, Layout::ND>;
#ifdef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
  using BlockFloatGlobal =
      GlobalTensor<float, BlockShape, BlockStride, Layout::ND>;
#endif
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
  using BlockFractalShape = Shape<1, 1, 1, FractalSize, FractalSize>;
  using BlockFractalStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  using BlockFractalIn =
      GlobalTensor<InputT, BlockFractalShape, BlockFractalStride, Layout::ND>;
#endif
  using BlockL1 = Tile<TileType::Mat,
                       InputT,
                       BlockSize,
                       BlockSize,
                       BLayout::ColMajor,
                       BlockSize,
                       BlockSize,
                       SLayout::RowMajor,
                       512,
                       PadValue::Zero>;
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
  using BlockFractalL1 = Tile<TileType::Mat,
                              InputT,
                              FractalSize,
                              FractalSize,
                              BLayout::ColMajor,
                              FractalSize,
                              FractalSize,
                              SLayout::RowMajor,
                              512,
                              PadValue::Zero>;
#endif
  using BlockL0A = TileLeft<InputT, BlockSize, BlockSize>;
  using BlockL0B = TileRight<InputT, BlockSize, BlockSize>;
  using BlockL0C = TileAcc<OutputT, BlockSize, BlockSize>;
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
  using BlockVecOut = Tile<TileType::Vec,
                           StoreT,
                           BlockSize,
                           BlockSize,
                           BLayout::RowMajor,
                           BlockSize,
                           BlockSize,
                           SLayout::NoneBox,
                           512,
                           PadValue::Zero>;
#ifdef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
  using BlockFloatL1 = Tile<TileType::Mat,
                            float,
                            BlockSize,
                            BlockSize,
                            BLayout::ColMajor,
                            BlockSize,
                            BlockSize,
                            SLayout::RowMajor,
                            512,
                            PadValue::Zero>;
  using BlockFloatL0A = TileLeft<float, BlockSize, BlockSize>;
  using BlockFloatL0B = TileRight<float, BlockSize, BlockSize>;
#endif
#endif

  (void)M_inv;
  (void)M;
  BlockL1 neg_identity_l1;
  BlockL1 identity_l1;
  BlockL1 a_l1;
  BlockL1 p_l1;
  BlockL1 x_l1;
  BlockL1 x_block_l1;
  BlockL1 neg_diag_l1;
  BlockL1 sum_l1;
  BlockL0A l0a;
  BlockL0B l0b;
  BlockL0C l0c;
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
  BlockL0A offdiag_l0a;
  BlockL0B offdiag_l0b;
  BlockL0C offdiag_l0c;
#ifdef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
  BlockFloatL1 identity_float_l1;
  BlockFloatL1 p_float_l1;
  BlockFloatL1 x_float_l1;
  BlockFloatL0A float_l0a;
  BlockFloatL0B float_l0b;
#endif
#endif
  TASSIGN(neg_identity_l1, 0);
  TASSIGN(identity_l1, BlockBytes);
  TASSIGN(a_l1, 2 * BlockBytes);
  TASSIGN(p_l1, 3 * BlockBytes);
  TASSIGN(x_l1, 4 * BlockBytes);
  TASSIGN(x_block_l1, 5 * BlockBytes);
  TASSIGN(neg_diag_l1, 6 * BlockBytes);
  TASSIGN(sum_l1, 7 * BlockBytes);
  TASSIGN(l0a, 0);
  TASSIGN(l0b, 0);
  TASSIGN(l0c, 0);
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
  TASSIGN(offdiag_l0a, BlockBytes);
  TASSIGN(offdiag_l0b, BlockBytes);
  TASSIGN(offdiag_l0c, BlockSize * BlockSize * sizeof(OutputT));
  BlockL1 inverse_l1[LowerBlocks];
  for (uint32_t block = 0; block < LowerBlocks; ++block) {
    TASSIGN(inverse_l1[block], (8 + block) * BlockBytes);
  }
#ifdef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
  constexpr uint32_t FloatL1BaseBytes = (8 + LowerBlocks) * BlockBytes;
  TASSIGN(identity_float_l1, FloatL1BaseBytes);
  TASSIGN(p_float_l1, FloatL1BaseBytes + FloatBlockBytes);
  TASSIGN(x_float_l1, FloatL1BaseBytes + 2 * FloatBlockBytes);
  TASSIGN(float_l0a, 0);
  TASSIGN(float_l0b, 0);
#endif
  BlockVecOut block_vec_out[LowerBlocks];
  for (uint32_t block = 0; block < LowerBlocks; ++block) {
#ifdef MEGA_CHUNK_GDN_A5_CUBE_DIAG_AIV_OFFDIAG
    TASSIGN(block_vec_out[block], HybridPublishBaseBytes + block * BlockBytes);
#else
    TASSIGN(block_vec_out[block], block * BlockBytes);
#endif
  }
#endif

#ifndef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
  IdentityBlockIn neg_identity_global(I_neg);
  TLOAD(neg_identity_l1, neg_identity_global);
  pipe_barrier(PIPE_ALL);
  A5Matmul(l0c, l0a, l0b, neg_identity_l1, neg_identity_l1, false);
  TMOV(identity_l1, l0c);
  pipe_barrier(PIPE_ALL);
#endif

#ifdef MEGA_CHUNK_GDN_A5_CUBE_DIAG_AIV_OFFDIAG
  // Proven C310 handoff sequence.  Keep all Cube work in one uninterrupted
  // phase and publish only after all diagonal solves plus one
  // off-diagonal drain product have completed.
  const uint32_t hybrid_cube_block_idx = get_block_idx();
  const uint32_t hybrid_cube_block_num = get_block_num();
  const uint32_t hybrid_wave_count =
      (total_tiles + hybrid_cube_block_num - 1) / hybrid_cube_block_num;
  for (uint32_t wave = 0; wave < hybrid_wave_count; ++wave) {
    const uint32_t global_tile_id =
        wave * hybrid_cube_block_num + hybrid_cube_block_idx;
    const bool active = global_tile_id < total_tiles;
    uint32_t valid_size = 0;
    if (active) {
      if (cu_seqlens != nullptr) {
        valid_size = GetBSNDVarlenTileInfoFromCuSeqlens(
                         global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens)
                         .valid_size;
      } else {
        valid_size = MatrixSize;
      }
    }

    pto::SYNCALL<pto::SyncCoreType::Mix>();
    if (active && valid_size > StableUbMaxSize) {
      __gm__ InputT* packed_in =
          packed_workspace + hybrid_cube_block_idx * 2 * TileLen;
      A5CopyGmToCbuf<BlockL1, InputT>(
          neg_identity_l1.data(), packed_in + BlockLen, BlockBytes);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      A5Matmul(l0c, l0a, l0b, neg_identity_l1, neg_identity_l1, false);
#ifdef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
      // Keep every recurrent Cube operand in fp32.  Converting only
      // the final accumulator is too late because P/X are fed back
      // through L1 after every doubling round.
      A5MovAccToL1(identity_float_l1, l0c);
      __gm__ float* fp32_handoff = reinterpret_cast<__gm__ float*>(
          packed_workspace + (hybrid_cube_block_idx * 2 + 1) * TileLen);
#else
      A5MovAccToL1(identity_l1, l0c);
#endif

#ifdef MEGA_CHUNK_GDN_A5_FP16_INTERMEDIATE
      // Storage-typed AIC-to-AIV UB data is not persistent across the
      // later C310 Cube instructions.  Publish the two proven diagonal
      // blocks and let AIV finish the remaining diagonals in fp32.
      constexpr uint32_t HybridDiagonalRows = 2;
#else
      constexpr uint32_t HybridDiagonalRows = BlocksPerMatrix;
#endif
      for (uint32_t block_i = 0; block_i < HybridDiagonalRows; ++block_i) {
        const uint32_t diag_offset =
            (block_i * BlocksPerMatrix + block_i) * BlockLen;
        A5CopyGmToCbuf<BlockL1, InputT>(
            a_l1.data(), packed_in + diag_offset, BlockBytes);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        A5Matmul(l0c, l0a, l0b, neg_identity_l1, a_l1, false);
#ifdef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
        A5MovAccToL1(p_float_l1, l0c);
        A5Matmul(l0c,
                 float_l0a,
                 float_l0b,
                 identity_float_l1,
                 identity_float_l1,
                 false);
        A5Matmul(
            l0c, float_l0a, float_l0b, identity_float_l1, p_float_l1, true);
        A5MovAccToL1(x_float_l1, l0c);
#else
        A5MovAccToL1(p_l1, l0c);
        A5Matmul(l0c, l0a, l0b, identity_l1, identity_l1, false);
        A5Matmul(l0c, l0a, l0b, identity_l1, p_l1, true);
        A5MovAccToL1(x_l1, l0c);
        const uint32_t diag_index = block_i * (block_i + 1) / 2 + block_i;
#endif
        for (uint32_t power = 2; power < BlockSize; power *= 2) {
#ifdef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
          A5Matmul(l0c, float_l0a, float_l0b, p_float_l1, p_float_l1, false);
          A5MovAccToL1(p_float_l1, l0c);
          A5Matmul(
              l0c, float_l0a, float_l0b, x_float_l1, identity_float_l1, false);
          A5Matmul(l0c, float_l0a, float_l0b, x_float_l1, p_float_l1, true);
          if (power == BlockSize / 2) {
            // The second half of the existing per-core KKT
            // scratch is unused by this hybrid path.  Publish the
            // dense fp32 accumulator there without changing the
            // operator workspace contract.
            BlockFloatGlobal diagonal_out(fp32_handoff + block_i * BlockLen);
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            TSTORE(diagonal_out, l0c);
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
          } else {
            A5MovAccToL1(x_float_l1, l0c);
          }
#else
          A5Matmul(l0c, l0a, l0b, p_l1, p_l1, false);
          A5MovAccToL1(p_l1, l0c);
          A5Matmul(l0c, l0a, l0b, x_l1, identity_l1, false);
          A5Matmul(l0c, l0a, l0b, x_l1, p_l1, true);
          if (power == BlockSize / 2) {
            A5MovAccToL1(inverse_l1[diag_index], l0c);
          } else {
            A5MovAccToL1(x_l1, l0c);
          }
#endif
        }

#ifdef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
        if (block_i == 1) {
          // Preserve the v64 pipeline drain point before the MIX
          // rendezvous, but keep this otherwise-unused product in
          // fp32 as well.
          constexpr uint32_t FirstOffdiagOffset = BlocksPerMatrix * BlockLen;
          A5CopyGmToCbuf<BlockL1, InputT>(
              a_l1.data(), packed_in + FirstOffdiagOffset, BlockBytes);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
          A5Matmul(offdiag_l0c,
                   offdiag_l0a,
                   offdiag_l0b,
                   neg_identity_l1,
                   a_l1,
                   false,
                   EVENT_ID1);
          A5MovAccToL1(p_float_l1, offdiag_l0c, EVENT_ID1);

          BlockFloatGlobal diagonal_zero(fp32_handoff);
          TLOAD(x_float_l1, diagonal_zero);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
          A5Matmul(offdiag_l0c,
                   float_l0a,
                   float_l0b,
                   p_float_l1,
                   x_float_l1,
                   false,
                   EVENT_ID1);
          A5MovAccToL1(x_float_l1, offdiag_l0c, EVENT_ID1);
        }
#else
        for (uint32_t block_j = 0; block_j < block_i; ++block_j) {
          bool accumulate = false;
          for (uint32_t block_k = block_j; block_k < block_i; ++block_k) {
            const uint32_t a_offset =
                (block_i * BlocksPerMatrix + block_k) * BlockLen;
            A5CopyGmToCbuf<BlockL1, InputT>(
                a_l1.data(), packed_in + a_offset, BlockBytes);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            const uint32_t x_index = block_k * (block_k + 1) / 2 + block_j;
            TASSIGN(x_block_l1, (8 + x_index) * BlockBytes);
            A5Matmul(offdiag_l0c,
                     offdiag_l0a,
                     offdiag_l0b,
                     a_l1,
                     x_block_l1,
                     accumulate,
                     EVENT_ID1);
            accumulate = true;
          }
          // This result is intentionally not consumed from L1.
          // Its role is to leave the Cube pipeline in the proven
          // state before the AIC-to-AIV UB publication.
          A5MovAccToL1(sum_l1, offdiag_l0c, EVENT_ID1);
        }
#endif
      }

#ifndef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
      for (uint32_t block_i = 0; block_i < HybridDiagonalRows; ++block_i) {
        const uint32_t diag_index = block_i * (block_i + 1) / 2 + block_i;
        constexpr uint16_t BlockLen32B = BlockBytes / 32;
        A5CopyCbufToUbuf<BlockVecOut, BlockL1>(block_vec_out[diag_index].data(),
                                               inverse_l1[diag_index].data(),
                                               0,
                                               BlockLen32B);
        A5CopyCbufToUbuf<BlockVecOut, BlockL1>(block_vec_out[diag_index].data(),
                                               inverse_l1[diag_index].data(),
                                               1,
                                               BlockLen32B);
      }
      set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
#endif
    }

    pto::SYNCALL<pto::SyncCoreType::Mix>();
    pto::SYNCALL<pto::SyncCoreType::Mix>();
  }
  return;
#endif

#if defined(MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE) && \
    !defined(MEGA_CHUNK_GDN_A5_RESIDENT_FULL_CUBE_SOLVE)
  // Fixed-wave protocol: every launched MIX block participates in exactly
  // three SYNCALL generations per wave, including blocks without a live
  // matrix.  Vector packs NZ input, Cube computes and publishes NZ output,
  // then Vector scatters to BSND before any buffer is reused.
  const uint32_t cube_block_idx = get_block_idx();
  const uint32_t cube_block_num = get_block_num();
  const uint32_t wave_count =
      (total_tiles + cube_block_num - 1) / cube_block_num;
  for (uint32_t wave = 0; wave < wave_count; ++wave) {
    const uint32_t global_tile_id = wave * cube_block_num + cube_block_idx;
    const bool active = global_tile_id < total_tiles;
    uint32_t valid_size = 0;
    if (active) {
      if (cu_seqlens != nullptr) {
        valid_size = GetBSNDVarlenTileInfoFromCuSeqlens(
                         global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens)
                         .valid_size;
      } else {
        valid_size = MatrixSize;
      }
    }

    // Full chunks arrive in block-major NZ workspace.  Partial chunks are
    // solved directly by both AIV siblings before this boundary.
    pto::SYNCALL<pto::SyncCoreType::Mix>();
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_DIAG0_PROBE
    constexpr uint32_t ActiveBlockRows = 1;
#elif defined(MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_ROW1_PROBE) || \
    defined(MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_NEG_DIAG_PROBE)
    constexpr uint32_t ActiveBlockRows = 2;
#else
    constexpr uint32_t ActiveBlockRows = BlocksPerMatrix;
#endif
    if (active && valid_size > StableUbMaxSize) {
      __gm__ InputT* packed_in =
          packed_workspace + cube_block_idx * 2 * TileLen;
      A5CopyGmToCbuf<BlockL1, InputT>(
          neg_identity_l1.data(), packed_in + BlockLen, BlockBytes);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

      // I = (-I) @ (-I).
      A5Matmul(l0c, l0a, l0b, neg_identity_l1, neg_identity_l1, false);
      A5MovAccToL1(identity_l1, l0c);

#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_COMPUTE_IDENTITY_PROBE
      constexpr uint16_t BlockLen32B = BlockBytes / 32;
      A5CopyCbufToUbuf<BlockVecOut, BlockL1>(
          block_vec_out[0].data(), identity_l1.data(), 0, BlockLen32B);
      A5CopyCbufToUbuf<BlockVecOut, BlockL1>(
          block_vec_out[0].data(), identity_l1.data(), 1, BlockLen32B);
#elif defined(MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_A_INPUT_PROBE)
      A5CopyGmToCbuf<BlockL1, InputT>(a_l1.data(), packed_in, BlockBytes);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      constexpr uint16_t BlockLen32B = BlockBytes / 32;
      A5CopyCbufToUbuf<BlockVecOut, BlockL1>(
          block_vec_out[0].data(), a_l1.data(), 0, BlockLen32B);
      A5CopyCbufToUbuf<BlockVecOut, BlockL1>(
          block_vec_out[0].data(), a_l1.data(), 1, BlockLen32B);
#elif defined(MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_X_INIT_PROBE)
      A5CopyGmToCbuf<BlockL1, InputT>(a_l1.data(), packed_in, BlockBytes);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      A5Matmul(l0c, l0a, l0b, neg_identity_l1, a_l1, false);
      A5MovAccToL1(p_l1, l0c);
      A5Matmul(l0c, l0a, l0b, identity_l1, identity_l1, false);
      A5Matmul(l0c, l0a, l0b, identity_l1, p_l1, true);
      A5MovAccToL1(x_l1, l0c);
      constexpr uint16_t BlockLen32B = BlockBytes / 32;
      A5CopyCbufToUbuf<BlockVecOut, BlockL1>(
          block_vec_out[0].data(), x_l1.data(), 0, BlockLen32B);
      A5CopyCbufToUbuf<BlockVecOut, BlockL1>(
          block_vec_out[0].data(), x_l1.data(), 1, BlockLen32B);
#else
#ifdef MEGA_CHUNK_GDN_A5_CUBE_DIAG_AIV_OFFDIAG
      constexpr uint32_t DiagonalSolveRows = 2;
#elif defined(MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_OFFDIAG_SUM_NO_HANDOFF_PROBE)
      constexpr uint32_t DiagonalSolveRows = 0;
#else
      constexpr uint32_t DiagonalSolveRows = ActiveBlockRows;
#endif
      for (uint32_t block_i = 0; block_i < DiagonalSolveRows; ++block_i) {
        const uint32_t diag_offset =
            (block_i * BlocksPerMatrix + block_i) * BlockLen;
        A5CopyGmToCbuf<BlockL1, InputT>(
            a_l1.data(), packed_in + diag_offset, BlockBytes);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        // Invert the unit-lower 32x32 diagonal block with a bounded
        // nilpotent doubling series.
        A5Matmul(l0c, l0a, l0b, neg_identity_l1, a_l1, false);
        A5MovAccToL1(p_l1, l0c);
        A5Matmul(l0c, l0a, l0b, identity_l1, identity_l1, false);
        A5Matmul(l0c, l0a, l0b, identity_l1, p_l1, true);
        A5MovAccToL1(x_l1, l0c);
        const uint32_t diag_index = block_i * (block_i + 1) / 2 + block_i;
        for (uint32_t power = 2; power < BlockSize; power *= 2) {
          A5Matmul(l0c, l0a, l0b, p_l1, p_l1, false);
          A5MovAccToL1(p_l1, l0c);
          A5Matmul(l0c, l0a, l0b, x_l1, identity_l1, false);
          A5Matmul(l0c, l0a, l0b, x_l1, p_l1, true);
          if (power == BlockSize / 2) {
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_OFFDIAG_SUM_EARLY_RETURN_PROBE
            A5MovAccToL1(x_l1, l0c);
#else
            A5MovAccToL1(inverse_l1[diag_index], l0c);
#endif
          } else {
            A5MovAccToL1(x_l1, l0c);
          }
        }
      }

#ifdef MEGA_CHUNK_GDN_A5_CUBE_DIAG_AIV_OFFDIAG
      // Keep the AIC pipeline at the same proven handoff point as the
      // v64 path: finish X00, X11, and the first off-diagonal block sum
      // A10 * X00 without interrupting Cube.  The sum drains the Cube
      // pipeline only; consuming its FIX-written L1 tile immediately
      // from MTE1 deadlocks on C310, so AIV recomputes that block.
      constexpr uint32_t FirstOffdiagOffset = BlocksPerMatrix * BlockLen;
      A5CopyGmToCbuf<BlockL1, InputT>(
          a_l1.data(), packed_in + FirstOffdiagOffset, BlockBytes);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      A5Matmul(offdiag_l0c,
               offdiag_l0a,
               offdiag_l0b,
               a_l1,
               inverse_l1[0],
               false,
               EVENT_ID1);
      A5MovAccToL1(sum_l1, offdiag_l0c, EVENT_ID1);
#endif

      // Publish diagonal blocks immediately.  A5's AIC-to-AIV UB
      // window is not persistent across later Cube instructions, so
      // the AIV side copies these blocks to GM at the next boundary.
#ifndef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_OFFDIAG_SUM_PROBE
#ifdef MEGA_CHUNK_GDN_A5_CUBE_DIAG_AIV_OFFDIAG
      for (uint32_t block_i = 0; block_i < DiagonalSolveRows; ++block_i) {
        const uint32_t diag_index = block_i * (block_i + 1) / 2 + block_i;
        constexpr uint16_t BlockLen32B = BlockBytes / 32;
        A5CopyCbufToUbuf<BlockVecOut, BlockL1>(block_vec_out[diag_index].data(),
                                               inverse_l1[diag_index].data(),
                                               0,
                                               BlockLen32B);
        A5CopyCbufToUbuf<BlockVecOut, BlockL1>(block_vec_out[diag_index].data(),
                                               inverse_l1[diag_index].data(),
                                               1,
                                               BlockLen32B);
      }
#else
      for (uint32_t block_i = 0; block_i < DiagonalSolveRows; ++block_i) {
        const uint32_t diag_index = block_i * (block_i + 1) / 2 + block_i;
        constexpr uint16_t BlockLen32B = BlockBytes / 32;
        A5CopyCbufToUbuf<BlockVecOut, BlockL1>(block_vec_out[diag_index].data(),
                                               inverse_l1[diag_index].data(),
                                               0,
                                               BlockLen32B);
        A5CopyCbufToUbuf<BlockVecOut, BlockL1>(block_vec_out[diag_index].data(),
                                               inverse_l1[diag_index].data(),
                                               1,
                                               BlockLen32B);
      }
#endif
#endif
#endif
#ifndef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_OFFDIAG_SUM_PROBE
      set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
#endif
    }

#ifdef MEGA_CHUNK_GDN_A5_CUBE_DIAG_AIV_OFFDIAG
    // All Cube work for this wave is complete.  Hand the first two
    // diagonal inverse blocks to both AIV siblings, then wait for their
    // independent contiguous-column solves and BSND stores.
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    continue;
#endif

    // AIV persists the diagonal UB blocks to GM before Cube resumes.
#ifndef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_OFFDIAG_SUM_PROBE
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    pto::SYNCALL<pto::SyncCoreType::Mix>();
#endif

#if !defined(MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_DIAG0_PROBE) && \
    !defined(MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_DIAG_ONLY_PROBE)
    for (uint32_t block_i = 1; block_i < ActiveBlockRows; ++block_i) {
      for (uint32_t block_j = 0; block_j < block_i; ++block_j) {
        if (active && valid_size > StableUbMaxSize) {
          __gm__ InputT* packed_in =
              packed_workspace + cube_block_idx * 2 * TileLen;
          bool accumulate = false;
          for (uint32_t block_k = block_j; block_k < block_i; ++block_k) {
            const uint32_t a_offset =
                (block_i * BlocksPerMatrix + block_k) * BlockLen;
            A5CopyGmToCbuf<BlockL1, InputT>(
                a_l1.data(), packed_in + a_offset, BlockBytes);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            const uint32_t x_index = block_k * (block_k + 1) / 2 + block_j;
            TASSIGN(x_block_l1, (8 + x_index) * BlockBytes);
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_OFFDIAG_SUM_EARLY_RETURN_PROBE
            A5Matmul(l0c, l0a, l0b, a_l1, identity_l1, accumulate, EVENT_ID0);
#else
            A5Matmul(l0c, l0a, l0b, a_l1, x_block_l1, accumulate, EVENT_ID0);
#endif
            accumulate = true;
          }

          // C310 cannot feed this accumulated L0C value directly
          // back to M or deliver fp32 Fixpipe output to GM/AIV.
          // Convert once to the storage type in L1, then use the
          // verified raw L1->both-AIV UB path.  The following AIV
          // product still accumulates entirely in fp32.
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_OFFDIAG_SUM_EARLY_RETURN_PROBE
          A5MovAccToL1(p_l1, l0c, EVENT_ID0);
          set_intra_block(PIPE_MTE1, 6);
          set_intra_block(PIPE_MTE1, 6 + SYNC_FLAG_ID_MAX);
          return;
#else
          A5MovAccToL1(sum_l1, l0c, EVENT_ID0);
#endif
#ifndef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_OFFDIAG_SUM_NO_HANDOFF_PROBE
          constexpr uint16_t BlockLen32B = BlockBytes / 32;
          A5CopyCbufToUbuf<BlockVecOut, BlockL1>(
              block_vec_out[0].data(), sum_l1.data(), 0, BlockLen32B);
          A5CopyCbufToUbuf<BlockVecOut, BlockL1>(
              block_vec_out[0].data(), sum_l1.data(), 1, BlockLen32B);
          set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID1);
          wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID1);
          set_intra_block(PIPE_MTE1, 6);
          set_intra_block(PIPE_MTE1, 6 + SYNC_FLAG_ID_MAX);
#endif
        }

        // AIV acknowledges the point-to-point L1->UB handoff and
        // meets Cube after publishing the completed NZ block.
        pto::SYNCALL<pto::SyncCoreType::Mix>();

        if (active && valid_size > StableUbMaxSize) {
          __gm__ InputT* packed_out =
              packed_workspace + (cube_block_idx * 2 + 1) * TileLen;
          const uint32_t out_index = block_i * (block_i + 1) / 2 + block_j;
          TASSIGN(x_block_l1, (8 + out_index) * BlockBytes);
          A5CopyGmToCbuf<BlockL1, InputT>(
              x_block_l1.data(), packed_out + out_index * BlockLen, BlockBytes);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        }
      }
    }
#endif

    // Do not reuse per-wave workspace until AIV has scattered all blocks.
    pto::SYNCALL<pto::SyncCoreType::Mix>();
  }
  return;
#endif

  for (uint32_t global_tile_id = get_block_idx(); global_tile_id < total_tiles;
       global_tile_id += get_block_num()) {
    uint32_t valid_size;
    if (cu_seqlens != nullptr) {
      const BSNDVarlenTileInfo tile_info = GetBSNDVarlenTileInfoFromCuSeqlens(
          global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens);
      valid_size = tile_info.valid_size;
    } else {
      valid_size = MatrixSize;
    }

    if (valid_size <= StableUbMaxSize) {
#ifdef MEGA_CHUNK_GDN_A5_DUAL_AIV_SOLVE
      wait_intra_block(PIPE_MTE2, 7);
      wait_intra_block(PIPE_MTE2, 7 + SYNC_FLAG_ID_MAX);
#else
      wait_intra_block(PIPE_MTE2, 7);
#endif
      set_intra_block(PIPE_S, 8);
      set_intra_block(PIPE_S, 8 + SYNC_FLAG_ID_MAX);
      continue;
    }
#ifdef MEGA_CHUNK_GDN_A5_RESIDENT_FULL_CUBE_SOLVE
    // Vector0 owns the strided BSND->NZ pack.  Consume its completion
    // token exactly once before the resident Cube phase reads workspace;
    // the same token id is reused only after Vector0 scatters the result.
    wait_intra_block(PIPE_MTE2, 9);
#endif
    __gm__ InputT* packed_in = packed_workspace + get_block_idx() * 2 * TileLen;
    __gm__ StoreT* packed_out = reinterpret_cast<__gm__ StoreT*>(
        packed_workspace + (get_block_idx() * 2 + 1) * TileLen);
    const uint32_t valid_blocks = (valid_size + BlockSize - 1) / BlockSize;
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
    // Vector0 publishes a block already packed in NZ physical order.
    // The full-MIX barrier below is the producer/consumer boundary; it
    // cannot be satisfied by a stale local event from an earlier stage.
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_TEXTRACT_PROBE
    pto::SYNCALL<pto::SyncCoreType::Mix>();
#endif
    InputBlock neg_identity_global(packed_in + BlockLen);
    TLOAD(neg_identity_l1, neg_identity_global);
    pipe_barrier(PIPE_ALL);
    A5Matmul(l0c, l0a, l0b, neg_identity_l1, neg_identity_l1, false);
    TMOV(identity_l1, l0c);
    pipe_barrier(PIPE_ALL);

#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_IDENTITY_PROBE
    for (uint32_t block_i = 0; block_i < BlocksPerMatrix; ++block_i) {
      const uint32_t diag_offset =
          (block_i * BlocksPerMatrix + block_i) * BlockLen;
      InputBlock diag_in(packed_in + diag_offset);
      TLOAD(a_l1, diag_in);
      pipe_barrier(PIPE_ALL);
      A5Matmul(l0c, l0a, l0b, a_l1, a_l1, false);
      BlockOut diag_out(packed_out + diag_offset);
      TSTORE(diag_out, l0c);
      set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
    }
    set_intra_block(PIPE_FIX, 8);
    set_intra_block(PIPE_FIX, 8 + SYNC_FLAG_ID_MAX);
    wait_intra_block(PIPE_S, 9);
    set_intra_block(PIPE_S, 10);
    set_intra_block(PIPE_S, 10 + SYNC_FLAG_ID_MAX);
    continue;
#endif
#endif

#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_TEXTRACT_PROBE
    // Diagnostic-only acc-to-vec handoff: duplicate one deterministic
    // identity result into every lower-block destination, bypassing all
    // block-recursive inverse work.
    // TMOV writes the Mat tile through FIX, while the raw L1-to-UB copy
    // below is issued by MTE1.  PIPE_ALL does not establish this producer
    // dependency on A5; use the same FIX->MTE1 handoff as PTO's
    // tmov_acc2mat reference kernel.
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    for (uint32_t block = 0; block < 1; ++block) {
      constexpr uint16_t BlockLen32B = BlockBytes / 32;
      A5CopyCbufToUbuf<BlockVecOut, BlockL1>(
          block_vec_out[block].data(), neg_identity_l1.data(), 0, BlockLen32B);
      A5CopyCbufToUbuf<BlockVecOut, BlockL1>(
          block_vec_out[block].data(), neg_identity_l1.data(), 1, BlockLen32B);
    }
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    // Full-MIX barriers are used by this probe to validate the final
    // producer/consumer protocol without relying on persistent local
    // event state from an earlier fused stage.
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    continue;
#endif

    for (uint32_t block_i = 0; block_i < valid_blocks; ++block_i) {
      const uint32_t diag_offset =
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
          (block_i * BlocksPerMatrix + block_i) * BlockLen;
#else
          block_i * BlockSize * MatrixSize + block_i * BlockSize;
#endif
      InputBlock diag_in(packed_in + diag_offset);
      TLOAD(a_l1, diag_in);
      pipe_barrier(PIPE_ALL);

      // Invert the unit-lower 16x16 diagonal block.  Its three
      // doubling rounds are short enough that fp16 L1 remains bounded.
      A5Matmul(l0c, l0a, l0b, neg_identity_l1, a_l1, false);
      TMOV(p_l1, l0c);
      pipe_barrier(PIPE_ALL);
      A5Matmul(l0c, l0a, l0b, identity_l1, identity_l1, false);
      A5Matmul(l0c, l0a, l0b, identity_l1, p_l1, true);
      TMOV(x_l1, l0c);
      pipe_barrier(PIPE_ALL);
      for (uint32_t power = 2; power < BlockSize; power *= 2) {
        A5Matmul(l0c, l0a, l0b, p_l1, p_l1, false);
        TMOV(p_l1, l0c);
        pipe_barrier(PIPE_ALL);
        A5Matmul(l0c, l0a, l0b, x_l1, identity_l1, false);
        A5Matmul(l0c, l0a, l0b, x_l1, p_l1, true);
        TMOV(x_l1, l0c);
        pipe_barrier(PIPE_ALL);
      }
      BlockOut diag_out(packed_out + diag_offset);
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
      const uint32_t diag_index = block_i * (block_i + 1) / 2 + block_i;
      TMOV(inverse_l1[diag_index], l0c);
      pipe_barrier(PIPE_ALL);
#else
      TSTORE(diag_out, l0c);
      // The next block row may immediately reload this diagonal block.
      // Make the FIX write globally visible to MTE2 before continuing.
      set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
#endif

      // Keep -X_ii in L1 for all off-diagonal blocks in this row.
      A5Matmul(l0c, l0a, l0b, neg_identity_l1, x_l1, false);
      TMOV(neg_diag_l1, l0c);
      pipe_barrier(PIPE_ALL);

      for (uint32_t block_j = 0; block_j < block_i; ++block_j) {
        bool accumulate = false;
        for (uint32_t block_k = block_j; block_k < block_i; ++block_k) {
          const uint32_t a_offset =
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
              (block_i * BlocksPerMatrix + block_k) * BlockLen;
#else
              block_i * BlockSize * MatrixSize + block_k * BlockSize;
#endif
          const uint32_t x_offset =
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
              (block_k * BlocksPerMatrix + block_j) * BlockLen;
#else
              block_k * BlockSize * MatrixSize + block_j * BlockSize;
#endif
          InputBlock a_block(packed_in + a_offset);
          TLOAD(a_l1, a_block);
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
          const uint32_t x_index = block_k * (block_k + 1) / 2 + block_j;
#else
          InputBlock x_block(reinterpret_cast<__gm__ InputT*>(packed_out) +
                             x_offset);
          TLOAD(x_block_l1, x_block);
#endif
          pipe_barrier(PIPE_ALL);
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
          A5Matmul(l0c, l0a, l0b, a_l1, inverse_l1[x_index], accumulate);
#else
          A5Matmul(l0c, l0a, l0b, a_l1, x_block_l1, accumulate);
#endif
          accumulate = true;
        }

        TMOV(sum_l1, l0c);
        pipe_barrier(PIPE_ALL);
        A5Matmul(l0c, l0a, l0b, neg_diag_l1, sum_l1, false);
        const uint32_t out_offset =
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
            (block_i * BlocksPerMatrix + block_j) * BlockLen;
#else
            block_i * BlockSize * MatrixSize + block_j * BlockSize;
#endif
        BlockOut block_out(packed_out + out_offset);
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
        const uint32_t out_index = block_i * (block_i + 1) / 2 + block_j;
        TMOV(inverse_l1[out_index], l0c);
        pipe_barrier(PIPE_ALL);
#else
        TSTORE(block_out, l0c);
        // Later block rows consume this result through MTE2.
        set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
#endif
      }
    }

#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
    // Move each already-rounded L1 block directly into both Vector UBs.
    // This avoids both broken L0C->GM row bands and unsupported quantized
    // dual-destination acc-to-vec transfers on A5.
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    for (uint32_t block_i = 0; block_i < valid_blocks; ++block_i) {
      for (uint32_t block_j = 0; block_j <= block_i; ++block_j) {
        const uint32_t block_index = block_i * (block_i + 1) / 2 + block_j;
        constexpr uint16_t BlockLen32B = BlockBytes / 32;
        A5CopyCbufToUbuf<BlockVecOut, BlockL1>(
            block_vec_out[block_index].data(),
            inverse_l1[block_index].data(),
            0,
            BlockLen32B);
        A5CopyCbufToUbuf<BlockVecOut, BlockL1>(
            block_vec_out[block_index].data(),
            inverse_l1[block_index].data(),
            1,
            BlockLen32B);
      }
    }
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_MTE1, 8);
    set_intra_block(PIPE_MTE1, 8 + SYNC_FLAG_ID_MAX);
    // Vector0 owns the scatter and acknowledges after MTE3 drains.
    wait_intra_block(PIPE_S, 9);
    // Release both Vector subblocks into the next fused stage only after
    // the real writer has finished consuming all TEXTRACT destinations.
    set_intra_block(PIPE_S, 10);
    set_intra_block(PIPE_S, 10 + SYNC_FLAG_ID_MAX);
#else
    set_intra_block(PIPE_FIX, 8);
    set_intra_block(PIPE_FIX, 8 + SYNC_FLAG_ID_MAX);
    wait_intra_block(PIPE_S, 9);
    set_intra_block(PIPE_S, 10);
    set_intra_block(PIPE_S, 10 + SYNC_FLAG_ID_MAX);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
#endif
  }
}

/*
 * The numerically stable A5 solve runs on Vector0 with the matrix resident in
 * UB.  The Cube side remains in the MIX launch as a local synchronization
 * relay, preventing Vector1 from entering the next fused stage before the
 * inverse has been scattered.  This uses no GM polling or all-core barrier.
 */
template <typename InputT,
          typename OutputT,
          uint32_t MatrixSize,
          bool IsBSND,
          typename StoreT = OutputT>
AICORE inline void TriInvA5SyncCubeKernel(__gm__ StoreT* M_inv,
                                          __gm__ InputT* M,
                                          __gm__ InputT* I_neg,
                                          uint32_t total_tiles,
                                          uint32_t num_bsnd_heads,
                                          __gm__ int32_t* cu_seqlens,
                                          __gm__ InputT* packed_workspace) {
  static_assert(IsBSND, "The A5 GDN solver expects BSND matrices.");
  (void)M_inv;
  (void)M;
  (void)I_neg;
  (void)num_bsnd_heads;
  (void)cu_seqlens;
  (void)packed_workspace;

  for (uint32_t global_tile_id = get_block_idx(); global_tile_id < total_tiles;
       global_tile_id += get_block_num()) {
    (void)global_tile_id;
    wait_intra_block(PIPE_S, 7);
    set_intra_block(PIPE_S, 8);
    set_intra_block(PIPE_S, 8 + SYNC_FLAG_ID_MAX);
  }
}
#endif

#if defined(__DAV_C310_VEC__)
template <typename InputT, uint32_t MatrixSize>
AICORE inline void TriInvA5DumpBsndBuffer(__gm__ InputT* dst,
                                          __gm__ InputT* src,
                                          uint32_t total_tiles,
                                          uint32_t num_bsnd_heads,
                                          __gm__ int32_t* cu_seqlens) {
  constexpr uint32_t RowsPerTile = 16;
  using StridedShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using StridedStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  using StridedGlobal =
      GlobalTensor<InputT, StridedShape, StridedStride, Layout::ND>;
  using DynamicTile = Tile<TileType::Vec,
                           InputT,
                           RowsPerTile,
                           MatrixSize,
                           BLayout::RowMajor,
                           DYNAMIC,
                           DYNAMIC,
                           SLayout::NoneBox,
                           512,
                           PadValue::Zero>;
  if (get_subblockid() != 0) {
    return;
  }
  const uint32_t row_stride = MatrixSize * num_bsnd_heads;
  for (uint32_t global_tile_id = get_block_idx(); global_tile_id < total_tiles;
       global_tile_id += get_block_num()) {
    const BSNDVarlenTileInfo tile_info =
        cu_seqlens != nullptr
            ? GetBSNDVarlenTileInfoFromCuSeqlens(
                  global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens)
            : BSNDVarlenTileInfo{
                  GetBSNDFixedTileOffset(
                      global_tile_id, num_bsnd_heads, MatrixSize),
                  MatrixSize};
    for (uint32_t tile_row = 0; tile_row < tile_info.valid_size;
         tile_row += RowsPerTile) {
      const uint32_t live_rows =
          min(tile_info.valid_size - tile_row, RowsPerTile);
      DynamicTile tile(live_rows, MatrixSize);
      TASSIGN(tile, 0);
      StridedGlobal source(
          src + tile_info.bsnd_offset + tile_row * row_stride,
          {1, 1, 1, static_cast<int>(live_rows), static_cast<int>(MatrixSize)},
          {1, 1, 1, static_cast<int>(row_stride), 1});
      StridedGlobal destination(
          dst + tile_info.bsnd_offset + tile_row * row_stride,
          {1, 1, 1, static_cast<int>(live_rows), static_cast<int>(MatrixSize)},
          {1, 1, 1, static_cast<int>(row_stride), 1});
      TLOAD(tile, source);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(destination, tile);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
  }
}

/*
 * Gather/scatter companion for the A5 Cube series.  A5's Cube TLOAD/TSTORE
 * path corrupts alternating row bands when the source/destination is a BSND
 * matrix with a non-unit row stride.  Vector0 converts that matrix to and from
 * a contiguous two-slot workspace in 16x128 UB tiles; all synchronization is
 * local to the MIX block.
 */
template <typename InputT, uint32_t MatrixSize, bool IsBSND, typename StoreT>
AICORE inline void TriInvA5PackedVectorKernel(__gm__ StoreT* M_inv,
                                              __gm__ InputT* M,
                                              uint32_t total_tiles,
                                              uint32_t num_bsnd_heads,
                                              __gm__ int32_t* cu_seqlens,
                                              __gm__ InputT* packed_workspace) {
  static_assert(IsBSND, "The A5 GDN solver expects BSND matrices.");
  static_assert(
      std::is_same_v<InputT, StoreT> &&
          (std::is_same_v<InputT, half> || std::is_same_v<InputT, bfloat16_t>),
      "The A5 packed solver supports fp16/bf16 storage.");
  constexpr uint32_t RowsPerTile = 16;
  constexpr uint32_t TileLen = MatrixSize * MatrixSize;

  using PackedShape = Shape<1, 1, 1, RowsPerTile, MatrixSize>;
  using PackedStride = pto::Stride<1, 1, 1, MatrixSize, 1>;
  using PackedGlobal =
      GlobalTensor<InputT, PackedShape, PackedStride, Layout::ND>;
  using StridedShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using StridedStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  using StridedGlobal =
      GlobalTensor<InputT, StridedShape, StridedStride, Layout::ND>;
  using PackedTile = Tile<TileType::Vec,
                          InputT,
                          RowsPerTile,
                          MatrixSize,
                          BLayout::RowMajor,
                          RowsPerTile,
                          MatrixSize,
                          SLayout::NoneBox,
                          512,
                          PadValue::Zero>;
  using DynamicPackedTile = Tile<TileType::Vec,
                                 InputT,
                                 RowsPerTile,
                                 MatrixSize,
                                 BLayout::RowMajor,
                                 DYNAMIC,
                                 DYNAMIC,
                                 SLayout::NoneBox,
                                 512,
                                 PadValue::Zero>;

  const uint32_t block_idx = get_block_idx();
  const uint32_t block_num = get_block_num();
  const uint32_t row_stride = MatrixSize * num_bsnd_heads;
  const uint32_t vid = get_subblockid();
  PackedTile ub_tile;
  TASSIGN(ub_tile, 0);

  for (uint32_t global_tile_id = block_idx; global_tile_id < total_tiles;
       global_tile_id += block_num) {
    uint32_t bsnd_offset;
    uint32_t valid_size;
    if (cu_seqlens != nullptr) {
      const BSNDVarlenTileInfo tile_info = GetBSNDVarlenTileInfoFromCuSeqlens(
          global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens);
      bsnd_offset = tile_info.bsnd_offset;
      valid_size = tile_info.valid_size;
    } else {
      bsnd_offset =
          GetBSNDFixedTileOffset(global_tile_id, num_bsnd_heads, MatrixSize);
      valid_size = MatrixSize;
    }

    __gm__ InputT* packed_in = packed_workspace + block_idx * 2 * TileLen;
    __gm__ InputT* packed_out =
        packed_workspace + (block_idx * 2 + 1) * TileLen;

    if (vid == 0) {
      for (uint32_t tile_row = 0; tile_row < MatrixSize;
           tile_row += RowsPerTile) {
        const uint32_t live_rows =
            valid_size > tile_row ? min(valid_size - tile_row, RowsPerTile) : 0;
        if (live_rows > 0) {
          DynamicPackedTile dynamic_tile(live_rows, MatrixSize);
          TASSIGN(dynamic_tile, 0);
          StridedGlobal source(M + bsnd_offset + tile_row * row_stride,
                               {1,
                                1,
                                1,
                                static_cast<int>(live_rows),
                                static_cast<int>(MatrixSize)},
                               {1, 1, 1, static_cast<int>(row_stride), 1});
          TLOAD(dynamic_tile, source);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          if (live_rows != RowsPerTile) {
            TFILLPAD_INPLACE(ub_tile, dynamic_tile);
          }
        } else {
          TEXPANDS(ub_tile, GdnA5FromF32<InputT>(0.0f));
        }
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        PackedGlobal packed_dst(packed_in + tile_row * MatrixSize);
        TSTORE(packed_dst, ub_tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      }
      set_intra_block(PIPE_MTE3, 7);
    }

    // Cube publishes the contiguous inverse to both Vector subblocks.
    wait_intra_block(PIPE_MTE2, 8);

    if (vid == 0) {
      for (uint32_t tile_row = 0; tile_row < valid_size;
           tile_row += RowsPerTile) {
        const uint32_t live_rows = min(valid_size - tile_row, RowsPerTile);
        PackedGlobal packed_src(packed_out + tile_row * MatrixSize);
        TLOAD(ub_tile, packed_src);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        StridedGlobal destination(M_inv + bsnd_offset + tile_row * row_stride,
                                  {1,
                                   1,
                                   1,
                                   static_cast<int>(live_rows),
                                   static_cast<int>(MatrixSize)},
                                  {1, 1, 1, static_cast<int>(row_stride), 1});
        DynamicPackedTile store_tile(live_rows, MatrixSize);
        TASSIGN(store_tile, 0);
        TSTORE(destination, store_tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      }
      set_intra_block(PIPE_MTE3, 9);
    }

    // Cube relays Vector0's scatter completion to Vector1 so neither
    // subblock enters WY while the inverse output is still in flight.
    wait_intra_block(PIPE_MTE2, 10);
  }
}

#if defined(MEGA_CHUNK_GDN_A5_SPLIT64_SOLVE) && defined(__DAV_C310_VEC__)
/* AIV assembly companion for the three-block split64 fp32 handoff. */
template <typename InputT, typename StoreT>
AICORE inline void TriInvA5Split64Fp32VectorKernel(
    __gm__ StoreT* M_inv,
    uint32_t total_tiles,
    uint32_t num_bsnd_heads,
    __gm__ int32_t* cu_seqlens,
    __gm__ InputT* packed_workspace) {
  static_assert(sizeof(InputT) == 2,
                "The split64 workspace aliases two fp16 slots as fp32.");
  constexpr uint32_t FullSize = 128;
  constexpr uint32_t HalfSize = 64;
  constexpr uint32_t RowsPerTile = 16;
  constexpr uint32_t FullLen = FullSize * FullSize;
  constexpr uint32_t HalfLen = HalfSize * HalfSize;

  using Fp32Shape = Shape<1, 1, 1, RowsPerTile, HalfSize>;
  using Fp32Stride = pto::Stride<1, 1, 1, HalfSize, 1>;
  using PackedFp32 = GlobalTensor<float, Fp32Shape, Fp32Stride, Layout::ND>;
  using StridedShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using StridedStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  using StridedOutput =
      GlobalTensor<StoreT, StridedShape, StridedStride, Layout::ND>;
  using Fp32Tile = Tile<TileType::Vec,
                        float,
                        RowsPerTile,
                        HalfSize,
                        BLayout::RowMajor,
                        RowsPerTile,
                        HalfSize,
                        SLayout::NoneBox,
                        512,
                        PadValue::Zero>;
  using StoreHalfTile = Tile<TileType::Vec,
                             StoreT,
                             RowsPerTile,
                             HalfSize,
                             BLayout::RowMajor,
                             RowsPerTile,
                             HalfSize,
                             SLayout::NoneBox,
                             512,
                             PadValue::Zero>;
  using DynamicStoreHalfTile = Tile<TileType::Vec,
                                    StoreT,
                                    RowsPerTile,
                                    HalfSize,
                                    BLayout::RowMajor,
                                    DYNAMIC,
                                    DYNAMIC,
                                    SLayout::NoneBox,
                                    512,
                                    PadValue::Zero>;

  const uint32_t block_idx = get_block_idx();
  const uint32_t block_num = get_block_num();
  const uint32_t row_stride = FullSize * num_bsnd_heads;
  const uint32_t vid = get_subblockid();
  Fp32Tile fp32_tile;
  StoreHalfTile store_tile;
  TASSIGN(fp32_tile, 0);
  constexpr uint32_t StoreUbAddress = RowsPerTile * HalfSize * sizeof(float);
  TASSIGN(store_tile, StoreUbAddress);

  for (uint32_t global_tile_id = block_idx; global_tile_id < total_tiles;
       global_tile_id += block_num) {
    const BSNDVarlenTileInfo tile_info = GetBSNDVarlenTileInfoFromCuSeqlens(
        global_tile_id, num_bsnd_heads, FullSize, cu_seqlens);
    __gm__ float* fp32_base = reinterpret_cast<__gm__ float*>(
        packed_workspace + block_idx * 2 * FullLen);

    wait_intra_block(PIPE_MTE2, 8);

    // AIV0 owns row bands 0,32,... and AIV1 owns 16,48,... .  For a
    // top band the two halves are [Ainv, zero]; for a bottom band they
    // are [X10, Dinv].  The reserved fourth GM block is never touched.
    const uint32_t first_tile_row = vid * RowsPerTile;
    constexpr uint32_t TileRowStride = 2 * RowsPerTile;
    for (uint32_t tile_row = first_tile_row; tile_row < FullSize;
         tile_row += TileRowStride) {
      constexpr uint32_t live_rows = RowsPerTile;
      const bool is_top = tile_row < HalfSize;
      const uint32_t block_row = tile_row % HalfSize;
      const uint32_t left_block = is_top ? 0 : 2;

      PackedFp32 packed_left(fp32_base + left_block * HalfLen +
                             block_row * HalfSize);
      TLOAD(fp32_tile, packed_left);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(store_tile, fp32_tile, RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      StridedOutput left_destination(
          M_inv + tile_info.bsnd_offset + tile_row * row_stride,
          {1, 1, 1, static_cast<int>(live_rows), static_cast<int>(HalfSize)},
          {1, 1, 1, static_cast<int>(row_stride), 1});
      DynamicStoreHalfTile left_band(live_rows, HalfSize);
      TASSIGN(left_band, StoreUbAddress);
      TSTORE(left_destination, left_band);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

      if (is_top) {
        // Explicit UB zeroing makes graph replay independent of the
        // stale contents of reserved packed block 3.
        TEXPANDS(store_tile, GdnA5FromF32<StoreT>(0.0f));
      } else {
        PackedFp32 packed_right(fp32_base + HalfLen + block_row * HalfSize);
        TLOAD(fp32_tile, packed_right);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TCVT(store_tile, fp32_tile, RoundMode::CAST_RINT);
      }
      pipe_barrier(PIPE_V);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      StridedOutput right_destination(
          M_inv + tile_info.bsnd_offset + tile_row * row_stride + HalfSize,
          {1, 1, 1, static_cast<int>(live_rows), static_cast<int>(HalfSize)},
          {1, 1, 1, static_cast<int>(row_stride), 1});
      DynamicStoreHalfTile right_band(live_rows, HalfSize);
      TASSIGN(right_band, StoreUbAddress);
      TSTORE(right_destination, right_band);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }

    set_intra_block(PIPE_MTE3, 9 + vid * SYNC_FLAG_ID_MAX);
    wait_intra_block(PIPE_MTE2, 10);
  }
}
#endif

/* Vector layout companion for a full fp32 final Cube handoff. */
template <typename InputT,
          uint32_t MatrixSize,
          bool IsBSND,
          typename StoreT,
          bool PackInput = true>
AICORE inline void TriInvA5PackedFp32VectorKernel(
    __gm__ StoreT* M_inv,
    __gm__ InputT* M,
    uint32_t total_tiles,
    uint32_t num_bsnd_heads,
    __gm__ int32_t* cu_seqlens,
    __gm__ InputT* packed_workspace,
    uint32_t is_lower = 1) {
  static_assert(IsBSND, "The A5 fp32 packed solver expects BSND matrices.");
#if defined(MEGA_CHUNK_GDN_A5_SPLIT64_SOLVE) && defined(__DAV_C310_VEC__)
  if constexpr (MatrixSize == 128) {
    if (is_lower != 0 && CanUseA5Split64Solve<MatrixSize>(
                             total_tiles, num_bsnd_heads, cu_seqlens)) {
      TriInvA5Split64Fp32VectorKernel<InputT, StoreT>(
          M_inv, total_tiles, num_bsnd_heads, cu_seqlens, packed_workspace);
      return;
    }
  }
#endif
  constexpr uint32_t RowsPerTile = 16;
  constexpr uint32_t TileLen = MatrixSize * MatrixSize;
  using InputShape = Shape<1, 1, 1, RowsPerTile, MatrixSize>;
  using InputStride = pto::Stride<1, 1, 1, MatrixSize, 1>;
  using PackedInput = GlobalTensor<InputT, InputShape, InputStride, Layout::ND>;
  using Fp32Shape = Shape<1, 1, 1, RowsPerTile, MatrixSize>;
  using Fp32Stride = pto::Stride<1, 1, 1, MatrixSize, 1>;
  using PackedFp32 = GlobalTensor<float, Fp32Shape, Fp32Stride, Layout::ND>;
  using StridedShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using StridedStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  using StridedInput =
      GlobalTensor<InputT, StridedShape, StridedStride, Layout::ND>;
  using StridedOutput =
      GlobalTensor<StoreT, StridedShape, StridedStride, Layout::ND>;
  using InputTile = Tile<TileType::Vec,
                         InputT,
                         RowsPerTile,
                         MatrixSize,
                         BLayout::RowMajor,
                         RowsPerTile,
                         MatrixSize,
                         SLayout::NoneBox,
                         512,
                         PadValue::Zero>;
  using DynamicInputTile = Tile<TileType::Vec,
                                InputT,
                                RowsPerTile,
                                MatrixSize,
                                BLayout::RowMajor,
                                DYNAMIC,
                                DYNAMIC,
                                SLayout::NoneBox,
                                512,
                                PadValue::Zero>;
  using Fp32Tile = Tile<TileType::Vec,
                        float,
                        RowsPerTile,
                        MatrixSize,
                        BLayout::RowMajor,
                        RowsPerTile,
                        MatrixSize,
                        SLayout::NoneBox,
                        512,
                        PadValue::Zero>;
  using StoreTile = Tile<TileType::Vec,
                         StoreT,
                         RowsPerTile,
                         MatrixSize,
                         BLayout::RowMajor,
                         RowsPerTile,
                         MatrixSize,
                         SLayout::NoneBox,
                         512,
                         PadValue::Zero>;
  using DynamicStoreTile = Tile<TileType::Vec,
                                StoreT,
                                RowsPerTile,
                                MatrixSize,
                                BLayout::RowMajor,
                                DYNAMIC,
                                DYNAMIC,
                                SLayout::NoneBox,
                                512,
                                PadValue::Zero>;

  const uint32_t block_idx = get_block_idx();
  const uint32_t block_num = get_block_num();
  const uint32_t row_stride = MatrixSize * num_bsnd_heads;
  const uint32_t vid = get_subblockid();
  InputTile input_tile;
  Fp32Tile fp32_tile;
  StoreTile store_tile;
  TASSIGN(input_tile, 0);
  TASSIGN(fp32_tile, 0);
  TASSIGN(store_tile, RowsPerTile * MatrixSize * sizeof(float));

  for (uint32_t global_tile_id = block_idx; global_tile_id < total_tiles;
       global_tile_id += block_num) {
    const BSNDVarlenTileInfo tile_info =
        cu_seqlens != nullptr
            ? GetBSNDVarlenTileInfoFromCuSeqlens(
                  global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens)
            : BSNDVarlenTileInfo{
                  GetBSNDFixedTileOffset(
                      global_tile_id, num_bsnd_heads, MatrixSize),
                  MatrixSize};
    const uint32_t bsnd_offset = tile_info.bsnd_offset;
    const uint32_t valid_size = tile_info.valid_size;
    __gm__ InputT* workspace_base = packed_workspace + block_idx * 2 * TileLen;

    if constexpr (PackInput) {
      if (vid == 0) {
        for (uint32_t tile_row = 0; tile_row < MatrixSize;
             tile_row += RowsPerTile) {
          const uint32_t live_rows =
              valid_size > tile_row ? min(valid_size - tile_row, RowsPerTile)
                                    : 0;
          if (live_rows > 0) {
            DynamicInputTile dynamic_tile(live_rows, MatrixSize);
            TASSIGN(dynamic_tile, 0);
            StridedInput source(M + bsnd_offset + tile_row * row_stride,
                                {1,
                                 1,
                                 1,
                                 static_cast<int>(live_rows),
                                 static_cast<int>(MatrixSize)},
                                {1, 1, 1, static_cast<int>(row_stride), 1});
            TLOAD(dynamic_tile, source);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            if (live_rows != RowsPerTile) {
              TFILLPAD_INPLACE(input_tile, dynamic_tile);
            }
          } else {
            TEXPANDS(input_tile, GdnA5FromF32<InputT>(0.0f));
          }
          pipe_barrier(PIPE_V);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          PackedInput packed_dst(workspace_base + tile_row * MatrixSize);
          TSTORE(packed_dst, input_tile);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        set_intra_block(PIPE_MTE3, 7);
      }
    }

    wait_intra_block(PIPE_MTE2, 8);
#ifdef MEGA_CHUNK_GDN_A5_DUAL_FP32_SCATTER
    // Cube publishes the same fp32 matrix to both AIV siblings. Each
    // sibling owns alternating 16-row bands, so their MTE2/Vector/MTE3
    // pipelines run concurrently without overlapping any BSND writes.
    constexpr bool IsScatterOwner = true;
#else
    constexpr bool IsScatterOwner = false;
#endif
    if (vid == 0 || IsScatterOwner) {
#ifdef MEGA_CHUNK_GDN_A5_PACKED_RECURSIVE_BSND_SCATTER_PROBE
      // Publication diagnostic: bypass the Cube fp32 workspace and
      // copy the known non-zero solve input through the exact same
      // BSND scatter destination.  A non-zero model result proves the
      // AIV scatter/downstream path and local MIX rendezvous are live.
      static_assert(std::is_same_v<InputT, StoreT>,
                    "The A5 publication probe requires one storage type.");
#ifdef MEGA_CHUNK_GDN_A5_DUAL_FP32_SCATTER
      const uint32_t first_tile_row = vid * RowsPerTile;
      constexpr uint32_t TileRowStride = 2 * RowsPerTile;
#else
      constexpr uint32_t first_tile_row = 0;
      constexpr uint32_t TileRowStride = RowsPerTile;
#endif
      for (uint32_t tile_row = first_tile_row; tile_row < valid_size;
           tile_row += TileRowStride) {
        const uint32_t live_rows = min(valid_size - tile_row, RowsPerTile);
        DynamicInputTile probe_tile(live_rows, MatrixSize);
        TASSIGN(probe_tile, 0);
        StridedInput source(M + bsnd_offset + tile_row * row_stride,
                            {1,
                             1,
                             1,
                             static_cast<int>(live_rows),
                             static_cast<int>(MatrixSize)},
                            {1, 1, 1, static_cast<int>(row_stride), 1});
        TLOAD(probe_tile, source);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        StridedOutput destination(M_inv + bsnd_offset + tile_row * row_stride,
                                  {1,
                                   1,
                                   1,
                                   static_cast<int>(live_rows),
                                   static_cast<int>(MatrixSize)},
                                  {1, 1, 1, static_cast<int>(row_stride), 1});
        DynamicStoreTile store_band(live_rows, MatrixSize);
        TASSIGN(store_band, 0);
        TSTORE(destination, store_band);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      }
#else
      __gm__ float* fp32_base = reinterpret_cast<__gm__ float*>(workspace_base);
#ifdef MEGA_CHUNK_GDN_A5_DUAL_FP32_SCATTER
      const uint32_t first_tile_row = vid * RowsPerTile;
      constexpr uint32_t TileRowStride = 2 * RowsPerTile;
#else
      constexpr uint32_t first_tile_row = 0;
      constexpr uint32_t TileRowStride = RowsPerTile;
#endif
      for (uint32_t tile_row = first_tile_row; tile_row < valid_size;
           tile_row += TileRowStride) {
        const uint32_t live_rows = min(valid_size - tile_row, RowsPerTile);
        PackedFp32 packed_src(fp32_base + tile_row * MatrixSize);
        TLOAD(fp32_tile, packed_src);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TCVT(store_tile, fp32_tile, RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        StridedOutput destination(M_inv + bsnd_offset + tile_row * row_stride,
                                  {1,
                                   1,
                                   1,
                                   static_cast<int>(live_rows),
                                   static_cast<int>(MatrixSize)},
                                  {1, 1, 1, static_cast<int>(row_stride), 1});
        DynamicStoreTile store_band(live_rows, MatrixSize);
        TASSIGN(store_band, RowsPerTile * MatrixSize * sizeof(float));
        TSTORE(destination, store_band);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      }
#endif
#ifdef MEGA_CHUNK_GDN_A5_DUAL_FP32_SCATTER
      set_intra_block(PIPE_MTE3, 9 + vid * SYNC_FLAG_ID_MAX);
#else
      set_intra_block(PIPE_MTE3, 9);
#endif
    }
    wait_intra_block(PIPE_MTE2, 10);
  }
}

#if defined(__DAV_C310_VEC__) && defined(MEGA_CHUNK_GDN_A5_VECTOR_OFFDIAG)
template <uint32_t DstRowStride, uint32_t SrcRowStride>
AICORE inline void A5VectorAxpy(__ubuf__ float* dst,
                                __ubuf__ float* src,
                                float scalar,
                                uint32_t valid_columns) {
  constexpr uint32_t ElementsPerRepeat = CCE_VL / sizeof(float);
  pto::AxpyInstrSame<float,
                     float,
                     ElementsPerRepeat,
                     DstRowStride,
                     SrcRowStride>(dst, src, scalar, 1, valid_columns);
}
#endif

#ifdef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
/*
 * Refine one dense 32x32 Cube handoff entirely in AIV fp32.  Keeping this
 * full-chunk-only scalar loop in a separate function preserves the C310 MIX
 * dispatcher's proven partial-chunk code shape.
 */
template <typename InputT, uint32_t MatrixSize>
AICORE inline __attribute__((noinline)) void A5RefineFp32DiagonalHandoff(
    __ubuf__ float* diag_full,
    __ubuf__ InputT* input_ptr,
    __ubuf__ float* residual_block,
    __ubuf__ float* refined_block,
    uint32_t block_i) {
  constexpr uint32_t BlockSize = 32;
  for (uint32_t row = 0; row < BlockSize; ++row) {
    for (uint32_t col = 0; col < BlockSize; ++col) {
      float residual = 0.0f;
      if (col <= row) {
        residual =
            (row == col ? 1.0f : 0.0f) - diag_full[row * BlockSize + col];
        for (uint32_t inner = col; inner < row; ++inner) {
          residual -=
              GdnA5ToF32(input_ptr[(block_i * BlockSize + row) * MatrixSize +
                                   block_i * BlockSize + inner]) *
              diag_full[inner * BlockSize + col];
        }
      }
      residual_block[row * BlockSize + col] = residual;
    }
  }
  for (uint32_t row = 0; row < BlockSize; ++row) {
    for (uint32_t col = 0; col < BlockSize; ++col) {
      float refined = 0.0f;
      if (col <= row) {
        refined = diag_full[row * BlockSize + col];
        for (uint32_t inner = col; inner <= row; ++inner) {
          refined += diag_full[row * BlockSize + inner] *
                     residual_block[inner * BlockSize + col];
        }
      }
      refined_block[row * BlockSize + col] = refined;
    }
  }
  for (uint32_t element = 0; element < BlockSize * BlockSize; ++element) {
    diag_full[element] = refined_block[element];
  }
}
#endif

/*
 * Stable A5 triangular inverse.
 *
 * The finite Neumann/doubling series above creates large cancelling P/X
 * intermediates.  Its fp16 L1 writebacks can overflow for real GDN
 * activations even when the final inverse is small and well-conditioned.
 * With MEGA_CHUNK_GDN_A5_DUAL_AIV_SOLVE, both Vector subblocks gather A and
 * independently own balanced inverse-column ranges in fp32 UB.  The default
 * standalone path retains Vector0 ownership.  Every row update reuses earlier
 * fp32 UB rows:
 *
 *   inverse[i, :] -= A[i, k] * inverse[k, :],  k < i
 *
 * This keeps one-head-per-MIX-block parallelism and removes scalar GM loops,
 * DCCI invalidations, DDR barriers, and iterative fp16 writebacks.  The
 * blocked full-chunk path batches each contiguous 16-column update into a
 * native A5 vector AXPY while keeping the recurrence in fp32 UB.
 */
template <typename InputT, uint32_t MatrixSize, bool IsBSND, typename StoreT>
AICORE inline void TriInvA5UbVectorKernel(__gm__ StoreT* M_inv,
                                          __gm__ InputT* M,
                                          __gm__ InputT* I_neg,
                                          uint32_t total_tiles,
                                          uint32_t num_bsnd_heads,
                                          __gm__ int32_t* cu_seqlens,
                                          __gm__ InputT* packed_workspace) {
  static_assert(IsBSND, "The A5 GDN solver expects BSND matrices.");
  static_assert(
      std::is_same_v<InputT, StoreT> &&
          (std::is_same_v<InputT, half> || std::is_same_v<InputT, bfloat16_t>),
      "The A5 UB solver supports fp16/bf16 storage.");
  constexpr uint32_t RowsPerTile = 16;
  constexpr uint32_t TileLen = MatrixSize * MatrixSize;
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
  constexpr uint32_t CubeBlockSize = 32;
  constexpr uint32_t VecTilesPerCubeBlock = CubeBlockSize / RowsPerTile;
  constexpr uint32_t BlocksPerMatrix = MatrixSize / CubeBlockSize;
  constexpr uint32_t BlockLen = CubeBlockSize * CubeBlockSize;
  constexpr uint32_t BlockBytes = BlockLen * sizeof(StoreT);
  constexpr uint32_t LowerBlocks = BlocksPerMatrix * (BlocksPerMatrix + 1) / 2;
  constexpr uint32_t StableUbMaxSize = MatrixSize - 1;
#else
  constexpr uint32_t StableUbMaxSize = MatrixSize;
#endif
#ifdef MEGA_CHUNK_GDN_A5_DUAL_AIV_SOLVE
  // A lower-triangular inverse has much more work in its early columns.  For
  // D=128, these ranges carry 50.9% and 49.1% of the recurrence.
  constexpr uint32_t FirstSubblockColumns = (MatrixSize * 27 + 127) / 128;
  constexpr uint32_t MaxColumnsPerSubblock = MatrixSize - FirstSubblockColumns;
  constexpr uint32_t StorageColumnsPerSubblock =
      (MaxColumnsPerSubblock + 15) / 16 * 16;
  constexpr uint32_t InputTileBytes = TileLen * sizeof(InputT);
  constexpr uint32_t PartialFloatTileBytes =
      MatrixSize * StorageColumnsPerSubblock * sizeof(float);
  static_assert(FirstSubblockColumns > 0 && FirstSubblockColumns < MatrixSize,
                "The A5 UB solver requires two non-empty column ranges.");
#else
  constexpr uint32_t HalfTileBytes = TileLen * sizeof(InputT);
  constexpr uint32_t FloatTileBytes = TileLen * sizeof(float);
#endif

  using PackedShape = Shape<1, 1, 1, RowsPerTile, MatrixSize>;
  using PackedStride = pto::Stride<1, 1, 1, MatrixSize, 1>;
  using PackedGlobal =
      GlobalTensor<InputT, PackedShape, PackedStride, Layout::ND>;
  using StridedShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using StridedStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  using StridedGlobal =
      GlobalTensor<InputT, StridedShape, StridedStride, Layout::ND>;
  using PackedTile = Tile<TileType::Vec,
                          InputT,
                          RowsPerTile,
                          MatrixSize,
                          BLayout::RowMajor,
                          RowsPerTile,
                          MatrixSize,
                          SLayout::NoneBox,
                          512,
                          PadValue::Zero>;
  using DynamicPackedTile = Tile<TileType::Vec,
                                 InputT,
                                 RowsPerTile,
                                 MatrixSize,
                                 BLayout::RowMajor,
                                 DYNAMIC,
                                 DYNAMIC,
                                 SLayout::NoneBox,
                                 512,
                                 PadValue::Zero>;
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
  constexpr uint32_t CubeFractalSize = 16;
  constexpr uint32_t CubeFractalsPerBlock = CubeBlockSize / CubeFractalSize;
  constexpr uint32_t CubeFractalLen = CubeFractalSize * CubeFractalSize;
  constexpr uint32_t CubeFractalBytes = CubeFractalLen * sizeof(StoreT);
  using PackedFractalShape = Shape<1, 1, 1, CubeFractalSize, CubeFractalSize>;
  using PackedFractalStride = pto::Stride<1, 1, 1, CubeFractalSize, 1>;
  using PackedFractalGlobal =
      GlobalTensor<InputT, PackedFractalShape, PackedFractalStride, Layout::ND>;
  using PackedFractalTile = Tile<TileType::Vec,
                                 InputT,
                                 CubeFractalSize,
                                 CubeFractalSize,
                                 BLayout::RowMajor,
                                 CubeFractalSize,
                                 CubeFractalSize,
                                 SLayout::NoneBox,
                                 512,
                                 PadValue::Zero>;
  using PackedFloatFractalGlobal =
      GlobalTensor<float, PackedFractalShape, PackedFractalStride, Layout::ND>;
  using PackedFloatFractalTile = Tile<TileType::Vec,
                                      float,
                                      CubeFractalSize,
                                      CubeFractalSize,
                                      BLayout::RowMajor,
                                      CubeFractalSize,
                                      CubeFractalSize,
                                      SLayout::NoneBox,
                                      512,
                                      PadValue::Zero>;
#ifdef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
  using PackedFloatBlockShape = Shape<1, 1, 1, CubeBlockSize, CubeBlockSize>;
  using PackedFloatBlockStride = pto::Stride<1, 1, 1, CubeBlockSize, 1>;
  using PackedFloatBlockGlobal = GlobalTensor<float,
                                              PackedFloatBlockShape,
                                              PackedFloatBlockStride,
                                              Layout::ND>;
  using PackedFloatBlockTile = Tile<TileType::Vec,
                                    float,
                                    CubeBlockSize,
                                    CubeBlockSize,
                                    BLayout::RowMajor,
                                    CubeBlockSize,
                                    CubeBlockSize,
                                    SLayout::NoneBox,
                                    512,
                                    PadValue::Zero>;
#endif
  using CubeFractalOutputTile = Tile<TileType::Vec,
                                     StoreT,
                                     CubeFractalSize,
                                     CubeFractalSize,
                                     BLayout::RowMajor,
                                     CubeFractalSize,
                                     CubeFractalSize,
                                     SLayout::NoneBox,
                                     512,
                                     PadValue::Zero>;
#ifdef MEGA_CHUNK_GDN_A5_CUBE_DIAG_AIV_OFFDIAG
  constexpr uint32_t HybridStorageColumns = CubeBlockSize / 2;
  constexpr uint32_t HybridOutputStorageColumns = CubeBlockSize;
  using HybridOutputBandTile = Tile<TileType::Vec,
                                    StoreT,
                                    CubeFractalSize,
                                    HybridOutputStorageColumns,
                                    BLayout::RowMajor,
                                    DYNAMIC,
                                    DYNAMIC,
                                    SLayout::NoneBox,
                                    512,
                                    PadValue::Zero>;
#endif
#endif
  using FullInputHalfTile = Tile<TileType::Vec,
                                 InputT,
                                 MatrixSize,
                                 MatrixSize,
                                 BLayout::RowMajor,
                                 MatrixSize,
                                 MatrixSize,
                                 SLayout::NoneBox,
                                 512,
                                 PadValue::Zero>;
#ifdef MEGA_CHUNK_GDN_A5_DUAL_AIV_SOLVE
  using PartialHalfTile = Tile<TileType::Vec,
                               InputT,
                               MatrixSize,
                               StorageColumnsPerSubblock,
                               BLayout::RowMajor,
                               MatrixSize,
                               StorageColumnsPerSubblock,
                               SLayout::NoneBox,
                               512,
                               PadValue::Zero>;
  using PartialFloatTile = Tile<TileType::Vec,
                                float,
                                MatrixSize,
                                StorageColumnsPerSubblock,
                                BLayout::RowMajor,
                                MatrixSize,
                                StorageColumnsPerSubblock,
                                SLayout::NoneBox,
                                512,
                                PadValue::Zero>;
  using DynamicOutputBandTile = Tile<TileType::Vec,
                                     StoreT,
                                     RowsPerTile,
                                     StorageColumnsPerSubblock,
                                     BLayout::RowMajor,
                                     DYNAMIC,
                                     DYNAMIC,
                                     SLayout::NoneBox,
                                     512,
                                     PadValue::Zero>;
#else
  using FullFloatTile = Tile<TileType::Vec,
                             float,
                             MatrixSize,
                             MatrixSize,
                             BLayout::RowMajor,
                             MatrixSize,
                             MatrixSize,
                             SLayout::NoneBox,
                             512,
                             PadValue::Zero>;
#endif
  const uint32_t block_idx = get_block_idx();
  const uint32_t block_num = get_block_num();
  const uint32_t row_stride = MatrixSize * num_bsnd_heads;
  const uint32_t vid = get_subblockid();

  FullInputHalfTile input_ub;
#ifdef MEGA_CHUNK_GDN_A5_DUAL_AIV_SOLVE
  PartialFloatTile inverse_ub;
  PartialHalfTile output_ub;
  TASSIGN(input_ub, 0);
  TASSIGN(inverse_ub, InputTileBytes);
  TASSIGN(output_ub, InputTileBytes + PartialFloatTileBytes);
#else
  FullFloatTile inverse_ub;
  FullInputHalfTile output_ub;
  TASSIGN(input_ub, 0);
  TASSIGN(inverse_ub, HalfTileBytes);
  TASSIGN(output_ub, HalfTileBytes + FloatTileBytes);
#endif

  const uint64_t input_ub_addr = reinterpret_cast<uint64_t>(input_ub.data());
  const uint64_t output_ub_addr = reinterpret_cast<uint64_t>(output_ub.data());
  __ubuf__ InputT* input_ptr =
      reinterpret_cast<__ubuf__ InputT*>(input_ub.data());
  __ubuf__ float* inverse_ptr =
      reinterpret_cast<__ubuf__ float*>(inverse_ub.data());
  __ubuf__ StoreT* output_ptr =
      reinterpret_cast<__ubuf__ StoreT*>(output_ub.data());
#ifdef MEGA_CHUNK_GDN_A5_DUAL_AIV_SOLVE
  const uint32_t column_begin = vid == 0 ? 0 : FirstSubblockColumns;
  const uint32_t active_columns =
      vid == 0 ? FirstSubblockColumns : MaxColumnsPerSubblock;
  const uint32_t column_end = column_begin + active_columns;
#endif

#if defined(MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE) && \
    !defined(MEGA_CHUNK_GDN_A5_RESIDENT_FULL_CUBE_SOLVE)
  const uint32_t wave_count = (total_tiles + block_num - 1) / block_num;
  for (uint32_t wave = 0; wave < wave_count; ++wave) {
    const uint32_t global_tile_id = wave * block_num + block_idx;
    const bool active = global_tile_id < total_tiles;
    uint32_t bsnd_offset = 0;
    uint32_t valid_size = 0;
    if (active) {
      if (cu_seqlens != nullptr) {
        const BSNDVarlenTileInfo tile_info = GetBSNDVarlenTileInfoFromCuSeqlens(
            global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens);
        bsnd_offset = tile_info.bsnd_offset;
        valid_size = tile_info.valid_size;
      } else {
        bsnd_offset =
            GetBSNDFixedTileOffset(global_tile_id, num_bsnd_heads, MatrixSize);
        valid_size = MatrixSize;
      }
    }

    if (active && valid_size > StableUbMaxSize && vid == 0) {
      __gm__ InputT* packed_in = packed_workspace + block_idx * 2 * TileLen;

      PackedTile zero_band;
      TASSIGN(zero_band, 0);
      TEXPANDS(zero_band, GdnA5FromF32<InputT>(0.0f));
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      for (uint32_t tile_row = 0; tile_row < MatrixSize;
           tile_row += RowsPerTile) {
        StridedGlobal zero_destination(
            M_inv + bsnd_offset + tile_row * row_stride,
            {1,
             1,
             1,
             static_cast<int>(RowsPerTile),
             static_cast<int>(MatrixSize)},
            {1, 1, 1, static_cast<int>(row_stride), 1});
        TSTORE(zero_destination, zero_band);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      }

      for (uint32_t fractal_col = 0; fractal_col < CubeFractalsPerBlock;
           ++fractal_col) {
        for (uint32_t fractal_row = 0; fractal_row < CubeFractalsPerBlock;
             ++fractal_row) {
          PackedFractalTile packed_neg_identity;
          TASSIGN(packed_neg_identity, 0);
          TEXPANDS(packed_neg_identity, GdnA5FromF32<InputT>(0.0f));
          pipe_barrier(PIPE_ALL);
          const uint32_t fractal_index =
              fractal_col * CubeFractalsPerBlock + fractal_row;
          if (fractal_row == fractal_col) {
            __ubuf__ InputT* identity_ptr =
                reinterpret_cast<__ubuf__ InputT*>(packed_neg_identity.data());
            for (uint32_t row = 0; row < CubeFractalSize; ++row) {
              identity_ptr[row * CubeFractalSize + row] =
                  GdnA5FromF32<InputT>(-1.0f);
            }
          }
          set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
          PackedFractalGlobal packed_dst(packed_in + BlockLen +
                                         fractal_index * CubeFractalLen);
          TSTORE(packed_dst, packed_neg_identity);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
      }

      // Pack required lower 32x32 blocks in Cube-native NZ order.
      // Each 16x16 fragment is contiguous in workspace even though its
      // BSND source rows have a large dynamic stride.
#ifdef MEGA_CHUNK_GDN_A5_CUBE_DIAG_AIV_OFFDIAG
#ifdef MEGA_CHUNK_GDN_A5_SKIP_DIAGONAL_REFINEMENT
      // Cube consumes all four diagonal blocks.
      constexpr uint32_t PackedBlockRows = BlocksPerMatrix;
#else
      // The hybrid AIC phase consumes only A00, A10, and A11.  The AIV
      // continuation keeps its own dense input resident in UB.
      constexpr uint32_t PackedBlockRows = 2;
#endif
#else
      constexpr uint32_t PackedBlockRows = BlocksPerMatrix;
#endif
      for (uint32_t block_row = 0; block_row < PackedBlockRows; ++block_row) {
        for (uint32_t block_col = 0; block_col <= block_row; ++block_col) {
#ifdef MEGA_CHUNK_GDN_A5_SKIP_DIAGONAL_REFINEMENT
          // Cube consumes all four diagonal blocks plus A10 for
          // the proven pipeline-drain product.  Remaining strict-
          // lower blocks stay in the dense AIV-resident input.
          if (block_row >= 2 && block_col != block_row) {
            continue;
          }
#endif
          for (uint32_t fractal_col = 0; fractal_col < CubeFractalsPerBlock;
               ++fractal_col) {
            for (uint32_t fractal_row = 0; fractal_row < CubeFractalsPerBlock;
                 ++fractal_row) {
              PackedFractalTile packed_fractal;
              TASSIGN(packed_fractal, 0);
              StridedGlobal source(M + bsnd_offset +
                                       (block_row * CubeBlockSize +
                                        fractal_row * CubeFractalSize) *
                                           row_stride +
                                       block_col * CubeBlockSize +
                                       fractal_col * CubeFractalSize,
                                   {1,
                                    1,
                                    1,
                                    static_cast<int>(CubeFractalSize),
                                    static_cast<int>(CubeFractalSize)},
                                   {1, 1, 1, static_cast<int>(row_stride), 1});
              TLOAD(packed_fractal, source);
              set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
              const uint32_t fractal_index =
                  fractal_col * CubeFractalsPerBlock + fractal_row;
              PackedFractalGlobal packed_dst(
                  packed_in +
                  (block_row * BlocksPerMatrix + block_col) * BlockLen +
                  fractal_index * CubeFractalLen);
              TSTORE(packed_dst, packed_fractal);
              set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
              wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
          }
        }
      }
    }

#ifdef MEGA_CHUNK_GDN_A5_CUBE_DIAG_AIV_OFFDIAG
    if (active && valid_size > StableUbMaxSize) {
      // Keep a dense copy in each AIV UB.  The blocked continuation
      // reuses every off-diagonal input several times; one coalesced
      // load is cheaper than repeatedly fetching NZ fragments from GM.
      for (uint32_t tile_row = 0; tile_row < MatrixSize;
           tile_row += RowsPerTile) {
        const uint64_t band_addr =
            input_ub_addr + tile_row * MatrixSize * sizeof(InputT);
        PackedTile input_band;
        TASSIGN(input_band, band_addr);
        StridedGlobal source(M + bsnd_offset + tile_row * row_stride,
                             {1,
                              1,
                              1,
                              static_cast<int>(RowsPerTile),
                              static_cast<int>(MatrixSize)},
                             {1, 1, 1, static_cast<int>(row_stride), 1});
        TLOAD(input_band, source);
      }
      pipe_barrier(PIPE_ALL);
    }
#endif

    if (active && valid_size <= StableUbMaxSize) {
#ifdef MEGA_CHUNK_GDN_A5_DUAL_AIV_SOLVE
      // Partial chunks remain on the verified dual-AIV fp32 recurrence;
      // its dynamic rows naturally cover non-16-aligned tails.
      for (uint32_t tile_row = 0; tile_row < valid_size;
           tile_row += RowsPerTile) {
        const uint32_t live_rows = min(valid_size - tile_row, RowsPerTile);
        const uint64_t band_addr =
            input_ub_addr + tile_row * MatrixSize * sizeof(InputT);
        PackedTile input_band;
        TASSIGN(input_band, band_addr);
        StridedGlobal source(M + bsnd_offset + tile_row * row_stride,
                             {1,
                              1,
                              1,
                              static_cast<int>(live_rows),
                              static_cast<int>(MatrixSize)},
                             {1, 1, 1, static_cast<int>(row_stride), 1});
        if (live_rows == RowsPerTile) {
          TLOAD(input_band, source);
        } else {
          DynamicPackedTile dynamic_band(live_rows, MatrixSize);
          TASSIGN(dynamic_band, band_addr);
          TLOAD(dynamic_band, source);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          TFILLPAD_INPLACE(input_band, dynamic_band);
        }
      }
      pipe_barrier(PIPE_ALL);

      for (uint32_t i = 0; i < valid_size; ++i) {
        const uint32_t row_offset = i * StorageColumnsPerSubblock;
        for (uint32_t j = 0; j < active_columns; ++j) {
          inverse_ptr[row_offset + j] = 0.0f;
        }
        if (i >= column_begin && i < column_end) {
          inverse_ptr[row_offset + i - column_begin] = 1.0f;
        }
        for (uint32_t k = 0; k < i; ++k) {
          const float coefficient = -GdnA5ToF32(input_ptr[i * MatrixSize + k]);
          const uint32_t source_offset = k * StorageColumnsPerSubblock;
          const uint32_t column_limit = min(k + 1, column_end);
          for (uint32_t j = column_begin; j < column_limit; ++j) {
            const uint32_t local_column = j - column_begin;
            inverse_ptr[row_offset + local_column] +=
                coefficient * inverse_ptr[source_offset + local_column];
          }
        }
        for (uint32_t j = 0; j < active_columns; ++j) {
          output_ptr[row_offset + j] =
              GdnA5FromF32<StoreT>(inverse_ptr[row_offset + j]);
        }
      }

      set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      for (uint32_t tile_row = 0; tile_row < valid_size;
           tile_row += RowsPerTile) {
        const uint32_t live_rows = min(valid_size - tile_row, RowsPerTile);
        StridedGlobal destination(
            M_inv + bsnd_offset + tile_row * row_stride + column_begin,
            {1,
             1,
             1,
             static_cast<int>(live_rows),
             static_cast<int>(active_columns)},
            {1, 1, 1, static_cast<int>(row_stride), 1});
        DynamicOutputBandTile store_band(live_rows, active_columns);
        TASSIGN(store_band,
                output_ub_addr +
                    tile_row * StorageColumnsPerSubblock * sizeof(StoreT));
        TSTORE(destination, store_band);
      }
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
#endif
    }
    // Input packing is complete; Cube may start the diagonal solves.
    pto::SYNCALL<pto::SyncCoreType::Mix>();

#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_DIAG0_PROBE
    constexpr uint32_t ActiveBlockRows = 1;
#elif defined(MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_ROW1_PROBE) || \
    defined(MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_NEG_DIAG_PROBE)
    constexpr uint32_t ActiveBlockRows = 2;
#else
    constexpr uint32_t ActiveBlockRows = BlocksPerMatrix;
#endif

#ifdef MEGA_CHUNK_GDN_A5_CUBE_DIAG_AIV_OFFDIAG
    // Cube has finished all diagonal blocks and copied them to
    // both sibling UB windows.  From this point each AIV owns a contiguous
    // column range of every 32x32 off-diagonal block, so there are no
    // cross-AIV data dependencies and Cube never has to resume M.
    pto::SYNCALL<pto::SyncCoreType::Mix>();
#ifdef MEGA_CHUNK_GDN_A5_CUBE_DIAG_HANDOFF_PROBE
    if (active && valid_size > StableUbMaxSize && vid == 0) {
      for (uint32_t block_row = 0; block_row < 2; ++block_row) {
        for (uint32_t block_col = 0; block_col <= block_row; ++block_col) {
          const uint32_t block_index =
              block_row * (block_row + 1) / 2 + block_col;
          for (uint32_t fractal_col = 0; fractal_col < CubeFractalsPerBlock;
               ++fractal_col) {
            for (uint32_t fractal_row = 0; fractal_row < CubeFractalsPerBlock;
                 ++fractal_row) {
              const uint32_t fractal_index =
                  fractal_col * CubeFractalsPerBlock + fractal_row;
              CubeFractalOutputTile output_fractal;
              TASSIGN(
                  output_fractal,
                  block_index * BlockBytes + fractal_index * CubeFractalBytes);
              StridedGlobal destination(
                  M_inv + bsnd_offset +
                      (block_row * CubeBlockSize +
                       fractal_row * CubeFractalSize) *
                          row_stride +
                      block_col * CubeBlockSize + fractal_col * CubeFractalSize,
                  {1,
                   1,
                   1,
                   static_cast<int>(CubeFractalSize),
                   static_cast<int>(CubeFractalSize)},
                  {1, 1, 1, static_cast<int>(row_stride), 1});
              TSTORE(destination, output_fractal);
            }
          }
        }
      }
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    continue;
#endif
    if (active && valid_size > StableUbMaxSize) {
#ifdef MEGA_CHUNK_GDN_A5_FP16_INTERMEDIATE
      constexpr uint32_t CubeDiagonalRows = 2;
#else
      constexpr uint32_t CubeDiagonalRows = BlocksPerMatrix;
#endif
      constexpr uint32_t FirstOwnedColumns = CubeBlockSize / 2;
      const uint32_t column_begin = vid == 0 ? 0 : FirstOwnedColumns;
      const uint32_t active_columns =
          vid == 0 ? FirstOwnedColumns : CubeBlockSize - FirstOwnedColumns;
      constexpr uint32_t OwnedBlockLen = CubeBlockSize * HybridStorageColumns;
      constexpr uint32_t FullDiagonalBase = LowerBlocks * OwnedBlockLen;
      constexpr uint32_t FullDiagonalCount = BlocksPerMatrix;
      constexpr uint32_t SumOffset =
          FullDiagonalBase + FullDiagonalCount * BlockLen;
#ifdef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
      constexpr uint32_t RefinementOffset = SumOffset + BlockLen;
      static_assert(
          RefinementOffset + BlockLen <= MatrixSize * StorageColumnsPerSubblock,
          "The fp32 handoff refinement must fit in AIV UB.");
#endif
#ifndef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
      __ubuf__ InputT* cube_diagonal_ptr =
          reinterpret_cast<__ubuf__ InputT*>(output_ub_addr);
#endif
      __ubuf__ float* sum_block = inverse_ptr + SumOffset;
      const uint64_t output_fragment_addr =
          output_ub_addr + LowerBlocks * BlockBytes;
      HybridOutputBandTile output_fragment(CubeFractalSize, active_columns);
      TASSIGN(output_fragment, output_fragment_addr);
      __ubuf__ InputT* output_fragment_ptr =
          reinterpret_cast<__ubuf__ InputT*>(output_fragment.data());

#ifdef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
      // The Cube writes all dense fp32 diagonal inverses into the
      // otherwise-unused second KKT scratch slot.  Invalidate only
      // those cache lines before MTE2 consumes them; the surrounding
      // MIX rendezvous supplies the cross-core producer boundary.
      __gm__ float* fp32_handoff = reinterpret_cast<__gm__ float*>(
          packed_workspace + (block_idx * 2 + 1) * TileLen);
      for (uint32_t block_i = 0; block_i < CubeDiagonalRows; ++block_i) {
        __gm__ float* diagonal_source = fp32_handoff + block_i * BlockLen;
        for (uint32_t element = 0; element < BlockLen; element += 16) {
          dcci(static_cast<__gm__ void*>(diagonal_source + element),
               SINGLE_CACHE_LINE);
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        PackedFloatBlockGlobal diagonal_global(diagonal_source);
        PackedFloatBlockTile diagonal_tile;
        TASSIGN(diagonal_tile,
                InputTileBytes +
                    (FullDiagonalBase + block_i * BlockLen) * sizeof(float));
        TLOAD(diagonal_tile, diagonal_global);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
      }

#ifndef MEGA_CHUNK_GDN_A5_SKIP_DIAGONAL_REFINEMENT
      __ubuf__ float* refined_block = inverse_ptr + RefinementOffset;
      for (uint32_t block_i = 0; block_i < CubeDiagonalRows; ++block_i) {
        __ubuf__ float* diag_full =
            inverse_ptr + FullDiagonalBase + block_i * BlockLen;
        A5RefineFp32DiagonalHandoff<InputT, MatrixSize>(
            diag_full, input_ptr, sum_block, refined_block, block_i);
      }
#endif
#else
      // Expand the Cube-native storage-typed/NZ diagonals once.
      // This compatibility path is intentionally excluded from the
      // fp32 handoff because widening here cannot restore lost bits.
      for (uint32_t block_i = 0; block_i < CubeDiagonalRows; ++block_i) {
        const uint32_t diag_index = block_i * (block_i + 1) / 2 + block_i;
        __ubuf__ float* diag_full =
            inverse_ptr + FullDiagonalBase + block_i * BlockLen;
        for (uint32_t row = 0; row < CubeBlockSize; ++row) {
          for (uint32_t col = 0; col < CubeBlockSize; ++col) {
            const uint32_t raw_index =
                ((col / CubeFractalSize) * CubeFractalsPerBlock +
                 row / CubeFractalSize) *
                    CubeFractalLen +
                (row % CubeFractalSize) * CubeFractalSize +
                col % CubeFractalSize;
            diag_full[row * CubeBlockSize + col] = GdnA5ToF32(
                cube_diagonal_ptr[diag_index * BlockLen + raw_index]);
          }
        }
      }
#endif

      // Solve any diagonal blocks not provided by Cube in dense fp32.
      for (uint32_t block_i = CubeDiagonalRows; block_i < BlocksPerMatrix;
           ++block_i) {
        __ubuf__ float* diag_full =
            inverse_ptr + FullDiagonalBase + block_i * BlockLen;
        for (uint32_t row = 0; row < CubeBlockSize; ++row) {
          for (uint32_t col = 0; col < CubeBlockSize; ++col) {
            diag_full[row * CubeBlockSize + col] = row == col ? 1.0f : 0.0f;
          }
          for (uint32_t inner = 0; inner < row; ++inner) {
            const float coefficient = -GdnA5ToF32(
                input_ptr[(block_i * CubeBlockSize + row) * MatrixSize +
                          block_i * CubeBlockSize + inner]);
            for (uint32_t col = 0; col <= inner; ++col) {
              diag_full[row * CubeBlockSize + col] +=
                  coefficient * diag_full[inner * CubeBlockSize + col];
            }
          }
        }
      }

      // Keep each sibling's columns contiguous, but give the heavier
      // low-column range fewer columns (14/18).  Both Cube-native and
      // AIV diagonals are staged into the same aligned output band.
      for (uint32_t block_i = 0; block_i < BlocksPerMatrix; ++block_i) {
        const uint32_t diag_index = block_i * (block_i + 1) / 2 + block_i;
        for (uint32_t fractal_row = 0; fractal_row < CubeFractalsPerBlock;
             ++fractal_row) {
          StridedGlobal destination(
              M_inv + bsnd_offset +
                  (block_i * CubeBlockSize + fractal_row * CubeFractalSize) *
                      row_stride +
                  block_i * CubeBlockSize + column_begin,
              {1,
               1,
               1,
               static_cast<int>(CubeFractalSize),
               static_cast<int>(active_columns)},
              {1, 1, 1, static_cast<int>(row_stride), 1});
          __ubuf__ float* diag_full =
              inverse_ptr + FullDiagonalBase + block_i * BlockLen;
          for (uint32_t row = 0; row < CubeFractalSize; ++row) {
            for (uint32_t col = 0; col < active_columns; ++col) {
              output_fragment_ptr[row * HybridOutputStorageColumns + col] =
                  GdnA5FromF32<InputT>(
                      diag_full[(fractal_row * CubeFractalSize + row) *
                                    CubeBlockSize +
                                column_begin + col]);
            }
          }
          set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
          TSTORE(destination, output_fragment);
          set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        }
      }
      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);

      // Block forward substitution.  Previous blocks are retained in
      // fp32 UB in the same contiguous-column ownership, so a completed
      // block is never rounded before it is consumed by a later row.
      for (uint32_t block_i = 1; block_i < BlocksPerMatrix; ++block_i) {
        __ubuf__ float* diag_i =
            inverse_ptr + FullDiagonalBase + block_i * BlockLen;
        for (uint32_t block_j = 0; block_j < block_i; ++block_j) {
          for (uint32_t row = 0; row < CubeBlockSize; ++row) {
            for (uint32_t col = 0; col < active_columns; ++col) {
              sum_block[row * HybridStorageColumns + col] = 0.0f;
            }
          }

#ifdef MEGA_CHUNK_GDN_A5_VECTOR_OFFDIAG
          set_flag(PIPE_S, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
          for (uint32_t block_k = block_j; block_k < block_i; ++block_k) {
            const uint32_t x_index = block_k * (block_k + 1) / 2 + block_j;
            if (block_k == block_j) {
              __ubuf__ float* diag_k =
                  inverse_ptr + FullDiagonalBase + block_k * BlockLen;
              for (uint32_t row = 0; row < CubeBlockSize; ++row) {
                __ubuf__ float* sum_row =
                    sum_block + row * HybridStorageColumns;
                for (uint32_t inner = column_begin; inner < CubeBlockSize;
                     ++inner) {
                  const uint32_t valid_columns =
                      min(inner - column_begin + 1, active_columns);
                  const uint32_t a_index =
                      (block_i * CubeBlockSize + row) * MatrixSize +
                      block_k * CubeBlockSize + inner;
                  A5VectorAxpy<HybridStorageColumns, CubeBlockSize>(
                      sum_row,
                      diag_k + inner * CubeBlockSize + column_begin,
                      GdnA5ToF32(input_ptr[a_index]),
                      valid_columns);
                }
              }
            } else {
              __ubuf__ float* x_block = inverse_ptr + x_index * OwnedBlockLen;
              for (uint32_t row = 0; row < CubeBlockSize; ++row) {
                __ubuf__ float* sum_row =
                    sum_block + row * HybridStorageColumns;
                for (uint32_t inner = 0; inner < CubeBlockSize; ++inner) {
                  const uint32_t a_index =
                      (block_i * CubeBlockSize + row) * MatrixSize +
                      block_k * CubeBlockSize + inner;
                  A5VectorAxpy<HybridStorageColumns, HybridStorageColumns>(
                      sum_row,
                      x_block + inner * HybridStorageColumns,
                      GdnA5ToF32(input_ptr[a_index]),
                      active_columns);
                }
              }
            }
          }
#else
          for (uint32_t block_k = block_j; block_k < block_i; ++block_k) {
            const uint32_t x_index = block_k * (block_k + 1) / 2 + block_j;
            if (block_k == block_j) {
              __ubuf__ float* diag_k =
                  inverse_ptr + FullDiagonalBase + block_k * BlockLen;
              for (uint32_t row = 0; row < CubeBlockSize; ++row) {
                for (uint32_t col = 0; col < active_columns; ++col) {
                  const uint32_t owned_col = column_begin + col;
                  float value = sum_block[row * HybridStorageColumns + col];
                  // X_kk is lower triangular, so entries
                  // above the output column are zero.
                  for (uint32_t inner = owned_col; inner < CubeBlockSize;
                       ++inner) {
                    const uint32_t a_index =
                        (block_i * CubeBlockSize + row) * MatrixSize +
                        block_k * CubeBlockSize + inner;
                    value += GdnA5ToF32(input_ptr[a_index]) *
                             diag_k[inner * CubeBlockSize + owned_col];
                  }
                  sum_block[row * HybridStorageColumns + col] = value;
                }
              }
            } else {
              __ubuf__ float* x_block = inverse_ptr + x_index * OwnedBlockLen;
              for (uint32_t row = 0; row < CubeBlockSize; ++row) {
                for (uint32_t col = 0; col < active_columns; ++col) {
                  float value = sum_block[row * HybridStorageColumns + col];
                  for (uint32_t inner = 0; inner < CubeBlockSize; ++inner) {
                    const uint32_t a_index =
                        (block_i * CubeBlockSize + row) * MatrixSize +
                        block_k * CubeBlockSize + inner;
                    value += GdnA5ToF32(input_ptr[a_index]) *
                             x_block[inner * HybridStorageColumns + col];
                  }
                  sum_block[row * HybridStorageColumns + col] = value;
                }
              }
            }
          }
#endif

          const uint32_t out_index = block_i * (block_i + 1) / 2 + block_j;
          __ubuf__ float* result = inverse_ptr + out_index * OwnedBlockLen;
#ifdef MEGA_CHUNK_GDN_A5_VECTOR_OFFDIAG
          for (uint32_t row = 0; row < CubeBlockSize; ++row) {
            for (uint32_t col = 0; col < active_columns; ++col) {
              result[row * HybridStorageColumns + col] = 0.0f;
            }
          }
          set_flag(PIPE_S, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
          for (uint32_t row = 0; row < CubeBlockSize; ++row) {
            __ubuf__ float* result_row = result + row * HybridStorageColumns;
            for (uint32_t inner = 0; inner <= row; ++inner) {
              A5VectorAxpy<HybridStorageColumns, HybridStorageColumns>(
                  result_row,
                  sum_block + inner * HybridStorageColumns,
                  -diag_i[row * CubeBlockSize + inner],
                  active_columns);
            }
          }
          set_flag(PIPE_V, PIPE_S, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
#else
          for (uint32_t row = 0; row < CubeBlockSize; ++row) {
            for (uint32_t col = 0; col < active_columns; ++col) {
              float value = 0.0f;
              for (uint32_t inner = 0; inner <= row; ++inner) {
                value += diag_i[row * CubeBlockSize + inner] *
                         sum_block[inner * HybridStorageColumns + col];
              }
              result[row * HybridStorageColumns + col] = -value;
            }
          }
#endif

          for (uint32_t fractal_row = 0; fractal_row < CubeFractalsPerBlock;
               ++fractal_row) {
            for (uint32_t row = 0; row < CubeFractalSize; ++row) {
              for (uint32_t col = 0; col < active_columns; ++col) {
                output_fragment_ptr[row * HybridOutputStorageColumns + col] =
                    GdnA5FromF32<InputT>(
                        result[(fractal_row * CubeFractalSize + row) *
                                   HybridStorageColumns +
                               col]);
              }
            }
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            StridedGlobal destination(
                M_inv + bsnd_offset +
                    (block_i * CubeBlockSize + fractal_row * CubeFractalSize) *
                        row_stride +
                    block_j * CubeBlockSize + column_begin,
                {1,
                 1,
                 1,
                 static_cast<int>(CubeFractalSize),
                 static_cast<int>(active_columns)},
                {1, 1, 1, static_cast<int>(row_stride), 1});
            TSTORE(destination, output_fragment);
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
          }
        }
      }
    }
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    continue;
#endif

    // Cube has copied all diagonal inverse blocks into both sibling UB
    // windows.  Persist each AIV's 16-row fragment to NZ GM so it remains
    // available while Cube computes subsequent block sums.
#ifndef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_OFFDIAG_SUM_PROBE
    pto::SYNCALL<pto::SyncCoreType::Mix>();
#endif
    __gm__ InputT* packed_out =
        packed_workspace + (block_idx * 2 + 1) * TileLen;
#ifndef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_OFFDIAG_SUM_PROBE
    if (active && valid_size > StableUbMaxSize) {
      for (uint32_t block_i = 0; block_i < ActiveBlockRows; ++block_i) {
        const uint32_t diag_index = block_i * (block_i + 1) / 2 + block_i;
        for (uint32_t fractal_col = 0; fractal_col < CubeFractalsPerBlock;
             ++fractal_col) {
          const uint32_t fractal_index =
              fractal_col * CubeFractalsPerBlock + vid;
          PackedFractalTile diag_fragment;
          TASSIGN(diag_fragment,
                  diag_index * BlockBytes + fractal_index * CubeFractalBytes);
          PackedFractalGlobal diag_destination(packed_out +
                                               diag_index * BlockLen +
                                               fractal_index * CubeFractalLen);
          TSTORE(diag_destination, diag_fragment);
        }
      }
      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }
    pto::SYNCALL<pto::SyncCoreType::Mix>();
#endif

#if !defined(MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_DIAG0_PROBE) && \
    !defined(MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_DIAG_ONLY_PROBE)
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_OFFDIAG_SUM_EARLY_RETURN_PROBE
    wait_intra_block(PIPE_MTE3, 6);
    return;
#endif
    constexpr uint32_t DiagHalfBytes =
        (CubeBlockSize / 2) * CubeBlockSize * sizeof(InputT);
    constexpr uint32_t SumFloatBytes =
        CubeBlockSize * CubeBlockSize * sizeof(float);
    constexpr uint32_t ResultHalfBytes = DiagHalfBytes;
    constexpr uint32_t HalfTempOffset =
        DiagHalfBytes + SumFloatBytes + ResultHalfBytes;
    constexpr uint32_t FloatTempOffset = HalfTempOffset + CubeFractalBytes;
    __ubuf__ InputT* diag_half =
        reinterpret_cast<__ubuf__ InputT*>(input_ub_addr);
    __ubuf__ float* sum_float =
        reinterpret_cast<__ubuf__ float*>(input_ub_addr + DiagHalfBytes);
    __ubuf__ InputT* result_half = reinterpret_cast<__ubuf__ InputT*>(
        input_ub_addr + DiagHalfBytes + SumFloatBytes);
    PackedFractalTile half_temp;
    PackedFloatFractalTile float_temp;
    TASSIGN(half_temp, HalfTempOffset);
    TASSIGN(float_temp, FloatTempOffset);
    __ubuf__ InputT* half_temp_ptr =
        reinterpret_cast<__ubuf__ InputT*>(half_temp.data());
    __ubuf__ float* float_temp_ptr =
        reinterpret_cast<__ubuf__ float*>(float_temp.data());

    for (uint32_t block_i = 1; block_i < ActiveBlockRows; ++block_i) {
      const uint32_t diag_index = block_i * (block_i + 1) / 2 + block_i;
      for (uint32_t block_j = 0; block_j < block_i; ++block_j) {
        // Cube has published the storage-typed block sum in both UB
        // windows through MTE1's dedicated intra-block channel.
        if (active && valid_size > StableUbMaxSize) {
#ifndef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_OFFDIAG_SUM_NO_HANDOFF_PROBE
          wait_intra_block(PIPE_MTE3, 6);
#endif
        }
#ifndef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_OFFDIAG_SUM_PROBE
        if (active && valid_size > StableUbMaxSize) {
          // Load the 16 diagonal rows owned by this AIV sibling.
          for (uint32_t fractal_col = 0; fractal_col < CubeFractalsPerBlock;
               ++fractal_col) {
            const uint32_t fractal_index =
                fractal_col * CubeFractalsPerBlock + vid;
            PackedFractalGlobal diag_source(packed_out + diag_index * BlockLen +
                                            fractal_index * CubeFractalLen);
            TLOAD(half_temp, diag_source);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            for (uint32_t row = 0; row < CubeFractalSize; ++row) {
              for (uint32_t col = 0; col < CubeFractalSize; ++col) {
                diag_half[row * CubeBlockSize + fractal_col * CubeFractalSize +
                          col] = half_temp_ptr[row * CubeFractalSize + col];
              }
            }
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
          }

          // Reassemble the four fp32 NZ sum fragments into a dense
          // UB block without any storage-type conversion.
          __gm__ float* sum_scratch = reinterpret_cast<__gm__ float*>(
              packed_workspace + block_idx * 2 * TileLen + BlockLen);
          for (uint32_t fractal_col = 0; fractal_col < CubeFractalsPerBlock;
               ++fractal_col) {
            for (uint32_t fractal_row = 0; fractal_row < CubeFractalsPerBlock;
                 ++fractal_row) {
              const uint32_t fractal_index =
                  fractal_col * CubeFractalsPerBlock + fractal_row;
              PackedFloatFractalGlobal sum_source(
                  sum_scratch + fractal_index * CubeFractalLen);
              TLOAD(float_temp, sum_source);
              set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
              for (uint32_t row = 0; row < CubeFractalSize; ++row) {
                for (uint32_t col = 0; col < CubeFractalSize; ++col) {
                  sum_float[(fractal_row * CubeFractalSize + row) *
                                CubeBlockSize +
                            fractal_col * CubeFractalSize + col] =
                      float_temp_ptr[row * CubeFractalSize + col];
                }
              }
              set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
              wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            }
          }

          for (uint32_t row = 0; row < CubeFractalSize; ++row) {
            for (uint32_t col = 0; col < CubeBlockSize; ++col) {
              float value = 0.0f;
              for (uint32_t k = 0; k < CubeBlockSize; ++k) {
                value += GdnA5ToF32(diag_half[row * CubeBlockSize + k]) *
                         sum_float[k * CubeBlockSize + col];
              }
              result_half[row * CubeBlockSize + col] =
                  GdnA5FromF32<InputT>(-value);
            }
          }

          const uint32_t out_index = block_i * (block_i + 1) / 2 + block_j;
          for (uint32_t fractal_col = 0; fractal_col < CubeFractalsPerBlock;
               ++fractal_col) {
            for (uint32_t row = 0; row < CubeFractalSize; ++row) {
              for (uint32_t col = 0; col < CubeFractalSize; ++col) {
                half_temp_ptr[row * CubeFractalSize + col] =
                    result_half[row * CubeBlockSize +
                                fractal_col * CubeFractalSize + col];
              }
            }
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            const uint32_t fractal_index =
                fractal_col * CubeFractalsPerBlock + vid;
            PackedFractalGlobal result_destination(
                packed_out + out_index * BlockLen +
                fractal_index * CubeFractalLen);
            TSTORE(result_destination, half_temp);
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
          }
        }
#endif
        // Completed block is now available for Cube's later sums.
        pto::SYNCALL<pto::SyncCoreType::Mix>();
      }
    }
#endif

    // Scatter the persistent NZ results to the original BSND layout.
    if (active && valid_size > StableUbMaxSize) {
      PackedFractalTile output_fragment;
      TASSIGN(output_fragment, 0);
      for (uint32_t block_row = 0; block_row < ActiveBlockRows; ++block_row) {
        for (uint32_t block_col = 0; block_col <= block_row; ++block_col) {
          const uint32_t block_index =
              block_row * (block_row + 1) / 2 + block_col;
          for (uint32_t fractal_col = 0; fractal_col < CubeFractalsPerBlock;
               ++fractal_col) {
            const uint32_t fractal_index =
                fractal_col * CubeFractalsPerBlock + vid;
            PackedFractalGlobal output_source(packed_out +
                                              block_index * BlockLen +
                                              fractal_index * CubeFractalLen);
            TLOAD(output_fragment, output_source);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            StridedGlobal destination(
                M_inv + bsnd_offset +
                    (block_row * CubeBlockSize + vid * CubeFractalSize) *
                        row_stride +
                    block_col * CubeBlockSize + fractal_col * CubeFractalSize,
                {1,
                 1,
                 1,
                 static_cast<int>(CubeFractalSize),
                 static_cast<int>(CubeFractalSize)},
                {1, 1, 1, static_cast<int>(row_stride), 1});
            TSTORE(destination, output_fragment);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
          }
        }
      }
    }
    pto::SYNCALL<pto::SyncCoreType::Mix>();
  }
  return;
#endif

  for (uint32_t global_tile_id = block_idx; global_tile_id < total_tiles;
       global_tile_id += block_num) {
    uint32_t bsnd_offset;
    uint32_t valid_size;
    if (cu_seqlens != nullptr) {
      const BSNDVarlenTileInfo tile_info = GetBSNDVarlenTileInfoFromCuSeqlens(
          global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens);
      bsnd_offset = tile_info.bsnd_offset;
      valid_size = tile_info.valid_size;
    } else {
      bsnd_offset =
          GetBSNDFixedTileOffset(global_tile_id, num_bsnd_heads, MatrixSize);
      valid_size = MatrixSize;
    }

    if (valid_size > StableUbMaxSize) {
      __gm__ InputT* packed_in = packed_workspace + block_idx * 2 * TileLen;
      __gm__ InputT* packed_out =
          packed_workspace + (block_idx * 2 + 1) * TileLen;

#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
      // Pack only the lower-triangular 32x32 blocks.  Each workspace
      // block is physically contiguous, so Cube never performs a
      // strided ND->NZ conversion on A5.
      if (vid == 0) {
        // The Cube only transfers completed lower blocks.  Clear the
        // full BSND tile once so the untouched upper triangle is
        // deterministic without consuming output workspace.
        for (uint32_t tile_row = 0; tile_row < MatrixSize;
             tile_row += RowsPerTile) {
          PackedTile zero_band;
          TASSIGN(zero_band, 0);
          TEXPANDS(zero_band, GdnA5FromF32<InputT>(0.0f));
          pipe_barrier(PIPE_V);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          StridedGlobal zero_destination(
              M_inv + bsnd_offset + tile_row * row_stride,
              {1,
               1,
               1,
               static_cast<int>(RowsPerTile),
               static_cast<int>(MatrixSize)},
              {1, 1, 1, static_cast<int>(row_stride), 1});
          TSTORE(zero_destination, zero_band);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }

        // Pack -I directly in NZ physical order
        // [fractal_col, fractal_row, row16, col16].  The slot is an
        // otherwise unused upper-triangular workspace block.
        for (uint32_t fractal_col = 0; fractal_col < CubeFractalsPerBlock;
             ++fractal_col) {
          for (uint32_t fractal_row = 0; fractal_row < CubeFractalsPerBlock;
               ++fractal_row) {
            PackedFractalTile packed_neg_identity;
            TASSIGN(packed_neg_identity, 0);
            TEXPANDS(packed_neg_identity, GdnA5FromF32<InputT>(0.0f));
            pipe_barrier(PIPE_ALL);
            if (fractal_row == fractal_col) {
              __ubuf__ InputT* neg_identity_ptr =
                  reinterpret_cast<__ubuf__ InputT*>(
                      packed_neg_identity.data());
              for (uint32_t row = 0; row < CubeFractalSize; ++row) {
                neg_identity_ptr[row * CubeFractalSize + row] =
                    GdnA5FromF32<InputT>(-1.0f);
              }
            }
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            const uint32_t fractal_index =
                fractal_col * CubeFractalsPerBlock + fractal_row;
            PackedFractalGlobal packed_neg_identity_dst(
                packed_in + BlockLen + fractal_index * CubeFractalLen);
            TSTORE(packed_neg_identity_dst, packed_neg_identity);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
          }
        }

        for (uint32_t block_row = 0; block_row < BlocksPerMatrix; ++block_row) {
          for (uint32_t block_col = 0; block_col <= block_row; ++block_col) {
            for (uint32_t fractal_col = 0; fractal_col < CubeFractalsPerBlock;
                 ++fractal_col) {
              for (uint32_t fractal_row = 0; fractal_row < CubeFractalsPerBlock;
                   ++fractal_row) {
                PackedFractalTile packed_fractal;
                TASSIGN(packed_fractal, 0);
                StridedGlobal source(
                    M + bsnd_offset +
                        (block_row * CubeBlockSize +
                         fractal_row * CubeFractalSize) *
                            row_stride +
                        block_col * CubeBlockSize +
                        fractal_col * CubeFractalSize,
                    {1,
                     1,
                     1,
                     static_cast<int>(CubeFractalSize),
                     static_cast<int>(CubeFractalSize)},
                    {1, 1, 1, static_cast<int>(row_stride), 1});
                TLOAD(packed_fractal, source);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                const uint32_t fractal_index =
                    fractal_col * CubeFractalsPerBlock + fractal_row;
                PackedFractalGlobal packed_dst(
                    packed_in +
                    (block_row * BlocksPerMatrix + block_col) * BlockLen +
                    fractal_index * CubeFractalLen);
                TSTORE(packed_dst, packed_fractal);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
              }
            }
          }
        }
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_TEXTRACT_PROBE
        pto::SYNCALL<pto::SyncCoreType::Mix>();
#else
        set_intra_block(PIPE_MTE3, 9);
#endif
      }

      // Cube copies each completed L1 block into both Vector UBs in
      // native NZ order.  Vector0 scatters its four contiguous 16x16
      // fractals into the corresponding BSND locations, avoiding an
      // unsupported NZ->ND TSTORE conversion.
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_TEXTRACT_PROBE
      pto::SYNCALL<pto::SyncCoreType::Mix>();
#else
      wait_intra_block(PIPE_MTE3, 8);
#endif
      if (vid == 0) {
        for (uint32_t block_row = 0; block_row < BlocksPerMatrix; ++block_row) {
          for (uint32_t block_col = 0; block_col <= block_row; ++block_col) {
            const uint32_t block_index =
                block_row * (block_row + 1) / 2 + block_col;
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_TEXTRACT_PROBE
            if (block_index != 0) {
              continue;
            }
#endif
            for (uint32_t fractal_col = 0; fractal_col < CubeFractalsPerBlock;
                 ++fractal_col) {
              for (uint32_t fractal_row = 0; fractal_row < CubeFractalsPerBlock;
                   ++fractal_row) {
                const uint32_t fractal_index =
                    fractal_col * CubeFractalsPerBlock + fractal_row;
                CubeFractalOutputTile output_fractal;
                TASSIGN(output_fractal,
                        block_index * BlockBytes +
                            fractal_index * CubeFractalBytes);
                StridedGlobal destination(
                    M_inv + bsnd_offset +
                        (block_row * CubeBlockSize +
                         fractal_row * CubeFractalSize) *
                            row_stride +
                        block_col * CubeBlockSize +
                        fractal_col * CubeFractalSize,
                    {1,
                     1,
                     1,
                     static_cast<int>(CubeFractalSize),
                     static_cast<int>(CubeFractalSize)},
                    {1, 1, 1, static_cast<int>(row_stride), 1});
                TSTORE(destination, output_fractal);
              }
            }
          }
        }
        set_intra_block(PIPE_MTE3, 9);
      }
#ifdef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_TEXTRACT_PROBE
      pto::SYNCALL<pto::SyncCoreType::Mix>();
#else
      wait_intra_block(PIPE_MTE2, 10);
#endif
#else
      // Retain the row-major packed fallback for standalone builds.
      if (vid == 0) {
        for (uint32_t tile_row = 0; tile_row < MatrixSize;
             tile_row += RowsPerTile) {
          const uint32_t live_rows =
              valid_size > tile_row ? min(valid_size - tile_row, RowsPerTile)
                                    : 0;
          PackedTile packed_band;
          TASSIGN(packed_band, 0);
          if (live_rows > 0) {
            DynamicPackedTile dynamic_band(live_rows, MatrixSize);
            TASSIGN(dynamic_band, 0);
            StridedGlobal source(M + bsnd_offset + tile_row * row_stride,
                                 {1,
                                  1,
                                  1,
                                  static_cast<int>(live_rows),
                                  static_cast<int>(MatrixSize)},
                                 {1, 1, 1, static_cast<int>(row_stride), 1});
            TLOAD(dynamic_band, source);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            if (live_rows != RowsPerTile) {
              TFILLPAD_INPLACE(packed_band, dynamic_band);
            }
          } else {
            TEXPANDS(packed_band, GdnA5FromF32<InputT>(0.0f));
          }
          pipe_barrier(PIPE_V);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          PackedGlobal packed_dst(packed_in + tile_row * MatrixSize);
          TSTORE(packed_dst, packed_band);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }

        // The blocked Cube solver writes only the lower triangle.
        // Clear the packed output once so its untouched upper blocks
        // scatter as exact zeros.
        for (uint32_t tile_row = 0; tile_row < MatrixSize;
             tile_row += RowsPerTile) {
          PackedTile zero_band;
          TASSIGN(zero_band, 0);
          TEXPANDS(zero_band, GdnA5FromF32<InputT>(0.0f));
          pipe_barrier(PIPE_V);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          PackedGlobal packed_zero_dst(packed_out + tile_row * MatrixSize);
          TSTORE(packed_zero_dst, zero_band);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        set_intra_block(PIPE_MTE3, 7);
      }

      wait_intra_block(PIPE_MTE2, 8);

      if (vid == 0) {
        for (uint32_t tile_row = 0; tile_row < valid_size;
             tile_row += RowsPerTile) {
          const uint32_t live_rows = min(valid_size - tile_row, RowsPerTile);
          PackedTile packed_band;
          TASSIGN(packed_band, 0);
          PackedGlobal packed_src(packed_out + tile_row * MatrixSize);
          TLOAD(packed_band, packed_src);
          set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
          StridedGlobal destination(M_inv + bsnd_offset + tile_row * row_stride,
                                    {1,
                                     1,
                                     1,
                                     static_cast<int>(live_rows),
                                     static_cast<int>(MatrixSize)},
                                    {1, 1, 1, static_cast<int>(row_stride), 1});
          DynamicPackedTile store_band(live_rows, MatrixSize);
          TASSIGN(store_band, 0);
          TSTORE(destination, store_band);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        set_intra_block(PIPE_MTE3, 9);
      }

      wait_intra_block(PIPE_MTE2, 10);
#endif
      continue;
    }

#ifdef MEGA_CHUNK_GDN_A5_DUAL_AIV_SOLVE
    {
      for (uint32_t tile_row = 0; tile_row < valid_size;
           tile_row += RowsPerTile) {
        const uint32_t live_rows = min(valid_size - tile_row, RowsPerTile);
        const uint64_t band_addr =
            input_ub_addr + tile_row * MatrixSize * sizeof(InputT);
        PackedTile input_band;
        TASSIGN(input_band, band_addr);
        StridedGlobal source(M + bsnd_offset + tile_row * row_stride,
                             {1,
                              1,
                              1,
                              static_cast<int>(live_rows),
                              static_cast<int>(MatrixSize)},
                             {1, 1, 1, static_cast<int>(row_stride), 1});
        if (live_rows == RowsPerTile) {
          TLOAD(input_band, source);
        } else {
          DynamicPackedTile dynamic_band(live_rows, MatrixSize);
          TASSIGN(dynamic_band, band_addr);
          TLOAD(dynamic_band, source);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          TFILLPAD_INPLACE(input_band, dynamic_band);
        }
      }
      pipe_barrier(PIPE_ALL);

      for (uint32_t i = 0; i < valid_size; ++i) {
        const uint32_t row_offset = i * StorageColumnsPerSubblock;
        for (uint32_t j = 0; j < active_columns; ++j) {
          inverse_ptr[row_offset + j] = 0.0f;
        }
        if (i >= column_begin && i < column_end) {
          inverse_ptr[row_offset + i - column_begin] = 1.0f;
        }
        for (uint32_t k = 0; k < i; ++k) {
          const float coefficient = -GdnA5ToF32(input_ptr[i * MatrixSize + k]);
          const uint32_t source_offset = k * StorageColumnsPerSubblock;
          const uint32_t column_limit = min(k + 1, column_end);
          for (uint32_t j = column_begin; j < column_limit; ++j) {
            const uint32_t local_column = j - column_begin;
            inverse_ptr[row_offset + local_column] +=
                coefficient * inverse_ptr[source_offset + local_column];
          }
        }
        for (uint32_t j = 0; j < active_columns; ++j) {
          output_ptr[row_offset + j] =
              GdnA5FromF32<StoreT>(inverse_ptr[row_offset + j]);
        }
      }

      set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

      for (uint32_t tile_row = 0; tile_row < valid_size;
           tile_row += RowsPerTile) {
        const uint32_t live_rows = min(valid_size - tile_row, RowsPerTile);
        StridedGlobal destination(
            M_inv + bsnd_offset + tile_row * row_stride + column_begin,
            {1,
             1,
             1,
             static_cast<int>(live_rows),
             static_cast<int>(active_columns)},
            {1, 1, 1, static_cast<int>(row_stride), 1});
        DynamicOutputBandTile store_band(live_rows, active_columns);
        TASSIGN(store_band,
                output_ub_addr +
                    tile_row * StorageColumnsPerSubblock * sizeof(StoreT));
        TSTORE(destination, store_band);
      }
      set_intra_block(PIPE_MTE3, 7 + vid * SYNC_FLAG_ID_MAX);
    }

    // Cube waits for both column owners before releasing the next stage.
#else
    if (vid == 0) {
      for (uint32_t tile_row = 0; tile_row < valid_size;
           tile_row += RowsPerTile) {
        const uint32_t live_rows = min(valid_size - tile_row, RowsPerTile);
        const uint64_t band_addr =
            input_ub_addr + tile_row * MatrixSize * sizeof(InputT);
        PackedTile input_band;
        TASSIGN(input_band, band_addr);
        StridedGlobal source(M + bsnd_offset + tile_row * row_stride,
                             {1,
                              1,
                              1,
                              static_cast<int>(live_rows),
                              static_cast<int>(MatrixSize)},
                             {1, 1, 1, static_cast<int>(row_stride), 1});
        if (live_rows == RowsPerTile) {
          TLOAD(input_band, source);
        } else {
          DynamicPackedTile dynamic_band(live_rows, MatrixSize);
          TASSIGN(dynamic_band, band_addr);
          TLOAD(dynamic_band, source);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          TFILLPAD_INPLACE(input_band, dynamic_band);
        }
      }
      pipe_barrier(PIPE_ALL);

      for (uint32_t i = 0; i < valid_size; ++i) {
        const uint32_t row_offset = i * MatrixSize;
        for (uint32_t j = 0; j < MatrixSize; ++j) {
          inverse_ptr[row_offset + j] = 0.0f;
        }
        inverse_ptr[row_offset + i] = 1.0f;
        for (uint32_t k = 0; k < i; ++k) {
          const float coefficient = -GdnA5ToF32(input_ptr[i * MatrixSize + k]);
          const uint32_t source_offset = k * MatrixSize;
          for (uint32_t j = 0; j <= k; ++j) {
            inverse_ptr[row_offset + j] +=
                coefficient * inverse_ptr[source_offset + j];
          }
        }
        for (uint32_t j = 0; j < MatrixSize; ++j) {
          output_ptr[row_offset + j] =
              GdnA5FromF32<StoreT>(inverse_ptr[row_offset + j]);
        }
      }

      set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

      for (uint32_t tile_row = 0; tile_row < valid_size;
           tile_row += RowsPerTile) {
        const uint32_t live_rows = min(valid_size - tile_row, RowsPerTile);
        StridedGlobal destination(M_inv + bsnd_offset + tile_row * row_stride,
                                  {1,
                                   1,
                                   1,
                                   static_cast<int>(live_rows),
                                   static_cast<int>(MatrixSize)},
                                  {1, 1, 1, static_cast<int>(row_stride), 1});
        DynamicPackedTile store_band(live_rows, MatrixSize);
        TASSIGN(store_band,
                output_ub_addr + tile_row * MatrixSize * sizeof(StoreT));
        TSTORE(destination, store_band);
      }
      set_intra_block(PIPE_MTE3, 7);
    }

    // Cube relays Vector0 completion to both Vector subblocks.
#endif

    wait_intra_block(PIPE_MTE2, 8);
  }
}

#endif

/*
 * @brief: Computes the inverses of the blocks of tensor M
 */
template <typename InputT,
          typename OutputT,
          uint32_t MatrixSize,
          uint32_t NumTilesPerCubeIter,
          bool IsBSND,
          typename StoreT = OutputT,
          bool WaitForKktReady = false,
          bool PrecomputedAuxiliary = false>
AICORE void runKernelTriInvRecUnroll(
    __gm__ StoreT* M_inv,
    __gm__ InputT* M,
    __gm__ InputT* I_neg,
    uint32_t total_tiles,
    uint32_t num_bsnd_heads = 0,
    __gm__ int32_t* cu_seqlens = nullptr,
    uint32_t is_lower = 0,
    __gm__ InputT* a5_packed_workspace = nullptr,
    bool use_precomputed_m_neg = false) {
#if defined(__DAV_C310_CUBE__)
#ifdef MEGA_CHUNK_GDN_A5_DIRECT_RECURSIVE_CUBE_FP32_SOLVE
  TriInvA5PackedRecursiveKernel<InputT,
                                OutputT,
                                MatrixSize,
                                IsBSND,
                                StoreT,
                                float,
                                true>(M_inv,
                                      M,
                                      I_neg,
                                      total_tiles,
                                      num_bsnd_heads,
                                      cu_seqlens,
                                      a5_packed_workspace,
                                      is_lower);
#elif defined(MEGA_CHUNK_GDN_A5_REFERENCE_RECURSIVE_CUBE_SYNC_SOLVE)
  (void)a5_packed_workspace;
  TriInvRecUnrollKernel<InputT,
                        OutputT,
                        MatrixSize,
                        NumTilesPerCubeIter,
                        IsBSND,
                        StoreT,
                        WaitForKktReady,
                        PrecomputedAuxiliary>(M_inv,
                                              M,
                                              I_neg,
                                              total_tiles,
                                              num_bsnd_heads,
                                              cu_seqlens,
                                              is_lower,
                                              use_precomputed_m_neg);
  // The standalone reference kernel returns only after its final FIX
  // stores.  In the fused MIX kernel, explicitly publish that completion
  // to both local AIV subblocks before they enter WY.
  pipe_barrier(PIPE_ALL);
  set_intra_block(PIPE_FIX, 8);
  set_intra_block(PIPE_FIX, 8 + SYNC_FLAG_ID_MAX);
#elif defined(MEGA_CHUNK_GDN_A5_PACKED_RECURSIVE_CUBE_FP32_SOLVE)
  TriInvA5PackedRecursiveKernel<InputT,
                                OutputT,
                                MatrixSize,
                                IsBSND,
                                StoreT,
                                float>(M_inv,
                                       M,
                                       I_neg,
                                       total_tiles,
                                       num_bsnd_heads,
                                       cu_seqlens,
                                       a5_packed_workspace,
                                       is_lower);
#elif defined(MEGA_CHUNK_GDN_A5_PACKED_RECURSIVE_CUBE_SOLVE)
  TriInvA5PackedRecursiveKernel<InputT, OutputT, MatrixSize, IsBSND, StoreT>(
      M_inv,
      M,
      I_neg,
      total_tiles,
      num_bsnd_heads,
      cu_seqlens,
      a5_packed_workspace,
      is_lower);
#elif defined(MEGA_CHUNK_GDN_A5_REFERENCE_RECURSIVE_CUBE_SOLVE)
  (void)a5_packed_workspace;
  TriInvRecUnrollKernel<InputT,
                        OutputT,
                        MatrixSize,
                        NumTilesPerCubeIter,
                        IsBSND,
                        StoreT,
                        WaitForKktReady,
                        PrecomputedAuxiliary>(M_inv,
                                              M,
                                              I_neg,
                                              total_tiles,
                                              num_bsnd_heads,
                                              cu_seqlens,
                                              is_lower,
                                              use_precomputed_m_neg);
#elif defined(MEGA_CHUNK_GDN_A5_FULL_MATRIX_CUBE_SOLVE)
  TriInvA5SeriesKernel<InputT, OutputT, MatrixSize, IsBSND, StoreT>(
      M_inv,
      M,
      I_neg,
      total_tiles,
      num_bsnd_heads,
      cu_seqlens,
      a5_packed_workspace);
#else
  TriInvA5BlockedKernel<InputT, OutputT, MatrixSize, IsBSND, StoreT>(
      M_inv,
      M,
      I_neg,
      total_tiles,
      num_bsnd_heads,
      cu_seqlens,
      a5_packed_workspace);
#endif
#elif defined(__DAV_C310_VEC__)
  (void)is_lower;
#ifdef MEGA_CHUNK_GDN_A5_DIRECT_RECURSIVE_CUBE_FP32_SOLVE
  (void)I_neg;
  (void)use_precomputed_m_neg;
  TriInvA5PackedFp32VectorKernel<InputT, MatrixSize, IsBSND, StoreT, false>(
      M_inv,
      M,
      total_tiles,
      num_bsnd_heads,
      cu_seqlens,
      a5_packed_workspace,
      is_lower);
#elif defined(MEGA_CHUNK_GDN_A5_REFERENCE_RECURSIVE_CUBE_SYNC_SOLVE)
  (void)M_inv;
  (void)M;
  (void)I_neg;
  (void)total_tiles;
  (void)num_bsnd_heads;
  (void)cu_seqlens;
  (void)a5_packed_workspace;
  (void)use_precomputed_m_neg;
  wait_intra_block(PIPE_MTE2, 8);
#elif defined(MEGA_CHUNK_GDN_A5_PACKED_RECURSIVE_CUBE_FP32_SOLVE)
  (void)I_neg;
  (void)use_precomputed_m_neg;
  TriInvA5PackedFp32VectorKernel<InputT, MatrixSize, IsBSND, StoreT>(
      M_inv, M, total_tiles, num_bsnd_heads, cu_seqlens, a5_packed_workspace);
#elif defined(MEGA_CHUNK_GDN_A5_PACKED_RECURSIVE_CUBE_SOLVE)
  (void)I_neg;
  (void)use_precomputed_m_neg;
  TriInvA5PackedVectorKernel<InputT, MatrixSize, IsBSND, StoreT>(
      M_inv, M, total_tiles, num_bsnd_heads, cu_seqlens, a5_packed_workspace);
#elif defined(MEGA_CHUNK_GDN_A5_REFERENCE_RECURSIVE_CUBE_SOLVE)
  (void)M_inv;
  (void)M;
  (void)I_neg;
  (void)total_tiles;
  (void)num_bsnd_heads;
  (void)cu_seqlens;
  (void)a5_packed_workspace;
  (void)use_precomputed_m_neg;
#elif defined(MEGA_CHUNK_GDN_A5_FULL_MATRIX_CUBE_SOLVE)
  TriInvA5PackedVectorKernel<InputT, MatrixSize, IsBSND, StoreT>(
      M_inv, M, total_tiles, num_bsnd_heads, cu_seqlens, a5_packed_workspace);
#else
  TriInvA5UbVectorKernel<InputT, MatrixSize, IsBSND, StoreT>(
      M_inv,
      M,
      I_neg,
      total_tiles,
      num_bsnd_heads,
      cu_seqlens,
      a5_packed_workspace);
#endif
#elif (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))

  TriInvRecUnrollKernel<InputT,
                        OutputT,
                        MatrixSize,
                        NumTilesPerCubeIter,
                        IsBSND,
                        StoreT,
                        WaitForKktReady,
                        PrecomputedAuxiliary>(M_inv,
                                              M,
                                              I_neg,
                                              total_tiles,
                                              num_bsnd_heads,
                                              cu_seqlens,
                                              is_lower,
                                              use_precomputed_m_neg);
#else
// Nothing to do on AIV
#endif
}

template <typename InputT, uint32_t NumTilesPerCubeIter, bool IsBSND>
AICORE void run_tri_inv_rec_unroll(__gm__ float* tensor_out,
                                   __gm__ InputT* tensor_in,
                                   __gm__ InputT* minus_identity_in,
                                   uint32_t matrix_size,
                                   uint32_t num_matrices,
                                   uint32_t num_bsnd_heads,
                                   __gm__ int32_t* cu_seqlens = nullptr,
                                   uint32_t is_lower = 0) {
  static_assert(
      std::is_same_v<InputT, half> || std::is_same_v<InputT, bfloat16_t>,
      "tri_inv_rec_unroll supports only fp16/bf16.");
  switch (matrix_size) {
    case 16:
      runKernelTriInvRecUnroll<InputT, float, 16, NumTilesPerCubeIter, IsBSND>(
          tensor_out,
          tensor_in,
          minus_identity_in,
          num_matrices,
          num_bsnd_heads,
          cu_seqlens,
          is_lower);
      break;
    case 32:
      runKernelTriInvRecUnroll<InputT, float, 32, NumTilesPerCubeIter, IsBSND>(
          tensor_out,
          tensor_in,
          minus_identity_in,
          num_matrices,
          num_bsnd_heads,
          cu_seqlens,
          is_lower);
      break;
    case 64:
      runKernelTriInvRecUnroll<InputT, float, 64, NumTilesPerCubeIter, IsBSND>(
          tensor_out,
          tensor_in,
          minus_identity_in,
          num_matrices,
          num_bsnd_heads,
          cu_seqlens,
          is_lower);
      break;
    case 128:
      runKernelTriInvRecUnroll<InputT, float, 128, NumTilesPerCubeIter, IsBSND>(
          tensor_out,
          tensor_in,
          minus_identity_in,
          num_matrices,
          num_bsnd_heads,
          cu_seqlens,
          is_lower);
      break;
  }
}

#ifdef MEGA_CHUNK_GDN_BLOCKED_SOLVE

template <typename T, uint32_t Size>
using BlockedSolveL1 = Tile<TileType::Mat,
                            T,
                            Size,
                            Size,
                            BLayout::ColMajor,
                            Size,
                            Size,
                            SLayout::RowMajor,
                            512,
                            PadValue::Zero>;

template <typename T, uint32_t Size>
using BlockedSolveL1Dynamic = Tile<TileType::Mat,
                                   T,
                                   Size,
                                   Size,
                                   BLayout::ColMajor,
                                   DYNAMIC,
                                   DYNAMIC,
                                   SLayout::RowMajor,
                                   512,
                                   PadValue::Zero>;

template <typename T, uint32_t Size>
using BlockedSolveNdL1 = Tile<TileType::Mat,
                              T,
                              Size,
                              Size,
                              BLayout::RowMajor,
                              Size,
                              Size,
                              SLayout::NoneBox,
                              512,
                              PadValue::Zero>;

template <typename T, uint32_t Size>
using BlockedSolveL0A = TileLeft<T, Size, Size>;

template <typename T, uint32_t Size>
using BlockedSolveL0B = TileRight<T, Size, Size>;

template <uint32_t Size>
using BlockedSolveL0C = TileAcc<float, Size, Size>;

template <typename T, uint32_t Size>
AICORE inline void BlockedSolveLoad(__gm__ T* src,
                                    int64_t base_offset,
                                    int32_t row_stride,
                                    uint32_t row,
                                    uint32_t col,
                                    BlockedSolveL1<T, Size>& dst) {
  using GmShape = Shape<1, 1, 1, Size, Size>;
  using GmStride = Stride<1, 1, 1, DYNAMIC, 1>;
  GlobalTensor<T, GmShape, GmStride> gm(
      src + base_offset + static_cast<int64_t>(row) * row_stride + col,
      {},
      {row_stride});
  TLOAD(dst, gm);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
}

template <typename T, uint32_t Size>
AICORE inline void BlockedSolveStore(__gm__ T* dst,
                                     int64_t base_offset,
                                     int32_t row_stride,
                                     uint32_t row,
                                     uint32_t col,
                                     BlockedSolveL0C<Size>& src) {
  using GmShape = Shape<1, 1, 1, Size, Size>;
  using GmStride = Stride<1, 1, 1, DYNAMIC, 1>;
  GlobalTensor<T, GmShape, GmStride> gm(
      dst + base_offset + static_cast<int64_t>(row) * row_stride + col,
      {},
      {row_stride});
  TSTORE(gm, src);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}

template <typename T, uint32_t Size>
AICORE inline void BlockedSolveStore(__gm__ T* dst,
                                     int64_t base_offset,
                                     int32_t row_stride,
                                     uint32_t row,
                                     uint32_t col,
                                     BlockedSolveNdL1<T, Size>& src) {
  using GmShape = Shape<1, 1, 1, Size, Size>;
  using GmStride = Stride<1, 1, 1, DYNAMIC, 1>;
  GlobalTensor<T, GmShape, GmStride> gm(
      dst + base_offset + static_cast<int64_t>(row) * row_stride + col,
      {},
      {row_stride});
  TSTORE(gm, src);
}

template <typename T, uint32_t Size>
AICORE inline void BlockedSolveLoadNd(__gm__ T* src,
                                      int64_t base_offset,
                                      int32_t row_stride,
                                      uint32_t row,
                                      uint32_t col,
                                      BlockedSolveNdL1<T, Size>& dst) {
  using GmShape = Shape<1, 1, 1, Size, Size>;
  using GmStride = Stride<1, 1, 1, DYNAMIC, 1>;
  GlobalTensor<T, GmShape, GmStride> gm(
      src + base_offset + static_cast<int64_t>(row) * row_stride + col,
      {},
      {row_stride});
  TLOAD(dst, gm);
  pipe_barrier(PIPE_MTE2);
}

template <typename T, uint32_t Size>
AICORE inline void BlockedSolveMatmul(BlockedSolveL0C<Size>& out,
                                      BlockedSolveL1<T, Size>& lhs,
                                      BlockedSolveL1<T, Size>& rhs,
                                      BlockedSolveL0A<T, Size>& l0a,
                                      BlockedSolveL0B<T, Size>& l0b) {
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
#ifndef MEGA_CHUNK_GDN_BLOCKED_SOLVE_SKIP_FIX_WAR
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
#endif
  TMOV(l0a, lhs);
  TMOV(l0b, rhs);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(out, l0a, l0b);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
}

template <typename T, uint32_t Size>
AICORE inline void BlockedSolveMatmulAcc(BlockedSolveL0C<Size>& out,
                                         BlockedSolveL1<T, Size>& lhs,
                                         BlockedSolveL1<T, Size>& rhs,
                                         BlockedSolveL0A<T, Size>& l0a,
                                         BlockedSolveL0B<T, Size>& l0b) {
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
#ifndef MEGA_CHUNK_GDN_BLOCKED_SOLVE_SKIP_FIX_WAR
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
#endif
  TMOV(l0a, lhs);
  TMOV(l0b, rhs);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL_ACC(out, out, l0a, l0b);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
}

template <typename T, uint32_t Size>
AICORE inline void BlockedSolveAccToL1(BlockedSolveL1<T, Size>& dst,
                                       BlockedSolveL0C<Size>& src) {
  TMOV(dst, src);
  set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
}

template <typename T, uint32_t Size, uint32_t NumPowerSteps>
AICORE inline void BlockedSolveInvPower(BlockedSolveL1<T, Size>& m,
                                        BlockedSolveL1<T, Size>& identity,
                                        BlockedSolveL1<T, Size>& minus_identity,
                                        BlockedSolveL1<T, Size>& x,
                                        BlockedSolveL1<T, Size>& power,
                                        BlockedSolveL0A<T, Size>& l0a,
                                        BlockedSolveL0B<T, Size>& l0b,
                                        BlockedSolveL0C<Size>& l0c) {
  BlockedSolveMatmul<T, Size>(l0c, minus_identity, m, l0a, l0b);
  BlockedSolveMatmulAcc<T, Size>(l0c, minus_identity, minus_identity, l0a, l0b);
  BlockedSolveAccToL1<T, Size>(x, l0c);

  BlockedSolveMatmul<T, Size>(l0c, m, m, l0a, l0b);
  BlockedSolveAccToL1<T, Size>(power, l0c);
  for (uint32_t step = 0; step < NumPowerSteps; ++step) {
    BlockedSolveMatmul<T, Size>(l0c, x, identity, l0a, l0b);
    BlockedSolveMatmulAcc<T, Size>(l0c, x, power, l0a, l0b);
    BlockedSolveAccToL1<T, Size>(x, l0c);
    if (step + 1 < NumPowerSteps) {
      BlockedSolveMatmul<T, Size>(l0c, power, power, l0a, l0b);
      BlockedSolveAccToL1<T, Size>(power, l0c);
    }
  }
}

template <typename T, uint32_t Size>
AICORE inline void BlockedSolveCombine(__gm__ T* out,
                                       __gm__ T* in,
                                       int64_t base_offset,
                                       int32_t row_stride,
                                       uint32_t block_start,
                                       BlockedSolveL1<T, Size>& minus_identity,
                                       uint32_t scratch_addr) {
  BlockedSolveL1<T, Size> a_inv;
  BlockedSolveL1<T, Size> d_inv;
  BlockedSolveL1<T, Size> cross;
  BlockedSolveL1<T, Size> tmp;
  TASSIGN(a_inv, scratch_addr);
  TASSIGN(d_inv, scratch_addr + Size * Size * sizeof(T));
  TASSIGN(cross, scratch_addr + 2 * Size * Size * sizeof(T));
  TASSIGN(tmp, scratch_addr + 3 * Size * Size * sizeof(T));

  BlockedSolveL0A<T, Size> l0a;
  BlockedSolveL0B<T, Size> l0b;
  BlockedSolveL0C<Size> l0c;
  TASSIGN(l0a, 0);
  TASSIGN(l0b, 0);
  TASSIGN(l0c, 0);

  BlockedSolveLoad<T, Size>(
      out, base_offset, row_stride, block_start, block_start, a_inv);
  BlockedSolveLoad<T, Size>(out,
                            base_offset,
                            row_stride,
                            block_start + Size,
                            block_start + Size,
                            d_inv);
  BlockedSolveLoad<T, Size>(
      in, base_offset, row_stride, block_start + Size, block_start, cross);

  BlockedSolveMatmul<T, Size>(l0c, minus_identity, cross, l0a, l0b);
  BlockedSolveAccToL1<T, Size>(tmp, l0c);
  BlockedSolveMatmul<T, Size>(l0c, d_inv, tmp, l0a, l0b);
  BlockedSolveAccToL1<T, Size>(tmp, l0c);
  BlockedSolveMatmul<T, Size>(l0c, tmp, a_inv, l0a, l0b);
  BlockedSolveStore<T, Size>(
      out, base_offset, row_stride, block_start + Size, block_start, l0c);
}

template <typename T>
AICORE inline void BlockedSolveFullMatrix16(
    __gm__ T* out,
    __gm__ T* in,
    int64_t base_offset,
    int32_t row_stride,
    BlockedSolveL1<T, 16>& identity16,
    BlockedSolveL1<T, 16>& minus_identity16,
    BlockedSolveNdL1<T, 16>& zero16,
    BlockedSolveL1<T, 32>& minus_identity32,
    BlockedSolveNdL1<T, 32>& zero32,
    BlockedSolveL1<T, 64>& minus_identity64,
    BlockedSolveNdL1<T, 64>& zero64) {
  constexpr uint32_t ScratchAddr = 32768;

  BlockedSolveStore<T, 64>(out, base_offset, row_stride, 0, 64, zero64);
  BlockedSolveStore<T, 32>(out, base_offset, row_stride, 0, 32, zero32);
  BlockedSolveStore<T, 32>(out, base_offset, row_stride, 64, 96, zero32);
  for (uint32_t block = 0; block < 128; block += 32) {
    BlockedSolveStore<T, 16>(
        out, base_offset, row_stride, block, block + 16, zero16);
  }
  pipe_barrier(PIPE_MTE3);

  BlockedSolveL1<T, 16> m;
  BlockedSolveL1<T, 16> x;
  BlockedSolveL1<T, 16> power;
  TASSIGN(m, ScratchAddr);
  TASSIGN(x, ScratchAddr + 16 * 16 * sizeof(T));
  TASSIGN(power, ScratchAddr + 2 * 16 * 16 * sizeof(T));
  BlockedSolveL0A<T, 16> l0a;
  BlockedSolveL0B<T, 16> l0b;
  BlockedSolveL0C<16> l0c;
  TASSIGN(l0a, 0);
  TASSIGN(l0b, 0);
  TASSIGN(l0c, 0);

  for (uint32_t block = 0; block < 128; block += 16) {
    BlockedSolveLoad<T, 16>(in, base_offset, row_stride, block, block, m);
    BlockedSolveInvPower<T, 16, 3>(
        m, identity16, minus_identity16, x, power, l0a, l0b, l0c);
    BlockedSolveStore<T, 16>(out, base_offset, row_stride, block, block, l0c);
  }
  for (uint32_t block = 0; block < 128; block += 32) {
    BlockedSolveCombine<T, 16>(
        out, in, base_offset, row_stride, block, minus_identity16, ScratchAddr);
  }
  for (uint32_t block = 0; block < 128; block += 64) {
    BlockedSolveCombine<T, 32>(
        out, in, base_offset, row_stride, block, minus_identity32, ScratchAddr);
  }
  BlockedSolveCombine<T, 64>(
      out, in, base_offset, row_stride, 0, minus_identity64, ScratchAddr);
}

template <typename T>
AICORE inline void BlockedSolveFullMatrix64ResidentInplace(
    __gm__ T* matrix_gm,
    int64_t base_offset,
    int32_t row_stride,
    BlockedSolveL1<T, 64>& identity64,
    BlockedSolveL1<T, 64>& minus_identity64) {
  constexpr uint32_t TileBytes = 64 * 64 * sizeof(T);
  constexpr uint32_t MatrixAddr = 3 * TileBytes;
  constexpr uint32_t AInvAddr = 4 * TileBytes;
  constexpr uint32_t DInvAddr = 5 * TileBytes;
  constexpr uint32_t PowerAddr = 6 * TileBytes;
  constexpr uint32_t CrossAddr = 7 * TileBytes;
  constexpr uint32_t TmpAddr = 8 * TileBytes;

  BlockedSolveL1<T, 64> matrix;
  BlockedSolveL1<T, 64> a_inv;
  BlockedSolveL1<T, 64> d_inv;
  BlockedSolveL1<T, 64> power;
  BlockedSolveL1<T, 64> cross;
  BlockedSolveL1<T, 64> tmp;
  TASSIGN(matrix, MatrixAddr);
  TASSIGN(a_inv, AInvAddr);
  TASSIGN(d_inv, DInvAddr);
  TASSIGN(power, PowerAddr);
  TASSIGN(cross, CrossAddr);
  TASSIGN(tmp, TmpAddr);

  BlockedSolveL0A<T, 64> l0a;
  BlockedSolveL0B<T, 64> l0b;
  BlockedSolveL0C<64> l0c;
  TASSIGN(l0a, 0);
  TASSIGN(l0b, 0);
  TASSIGN(l0c, 0);

  BlockedSolveLoad<T, 64>(matrix_gm, base_offset, row_stride, 0, 0, matrix);
  BlockedSolveInvPower<T, 64, 5>(
      matrix, identity64, minus_identity64, a_inv, power, l0a, l0b, l0c);
  BlockedSolveStore<T, 64>(matrix_gm, base_offset, row_stride, 0, 0, l0c);

  BlockedSolveLoad<T, 64>(matrix_gm, base_offset, row_stride, 64, 64, matrix);
  BlockedSolveInvPower<T, 64, 5>(
      matrix, identity64, minus_identity64, d_inv, power, l0a, l0b, l0c);
  BlockedSolveStore<T, 64>(matrix_gm, base_offset, row_stride, 64, 64, l0c);

  BlockedSolveLoad<T, 64>(matrix_gm, base_offset, row_stride, 64, 0, cross);
  BlockedSolveMatmul<T, 64>(l0c, minus_identity64, cross, l0a, l0b);
  BlockedSolveAccToL1<T, 64>(tmp, l0c);
  BlockedSolveMatmul<T, 64>(l0c, d_inv, tmp, l0a, l0b);
  BlockedSolveAccToL1<T, 64>(tmp, l0c);
  BlockedSolveMatmul<T, 64>(l0c, tmp, a_inv, l0a, l0b);
  BlockedSolveStore<T, 64>(matrix_gm, base_offset, row_stride, 64, 0, l0c);
}

template <typename T>
AICORE inline void BlockedSolveTail64ResidentInplace(
    __gm__ T* matrix_gm,
    int64_t base_offset,
    int32_t row_stride,
    uint32_t valid_size,
    BlockedSolveL1<T, 64>& identity64,
    BlockedSolveL1<T, 64>& minus_identity64) {
  constexpr uint32_t TileBytes = 64 * 64 * sizeof(T);
  constexpr uint32_t MatrixAddr = 3 * TileBytes;
  constexpr uint32_t XAddr = 4 * TileBytes;
  constexpr uint32_t PowerAddr = 5 * TileBytes;

  BlockedSolveL1Dynamic<T, 64> dynamic_matrix(valid_size, valid_size);
  TASSIGN(dynamic_matrix, MatrixAddr);
  using GmShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GmStride = Stride<1, 1, 1, DYNAMIC, 1>;
  GlobalTensor<T, GmShape, GmStride> gm_in(
      matrix_gm + base_offset,
      {1, 1, 1, static_cast<int>(valid_size), static_cast<int>(valid_size)},
      {1, 1, 1, row_stride, 1});
  TLOAD(dynamic_matrix, gm_in);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TFILLPAD(dynamic_matrix, dynamic_matrix);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

  BlockedSolveL1<T, 64> matrix;
  BlockedSolveL1<T, 64> x;
  BlockedSolveL1<T, 64> power;
  TASSIGN(matrix, MatrixAddr);
  TASSIGN(x, XAddr);
  TASSIGN(power, PowerAddr);
  BlockedSolveL0A<T, 64> l0a;
  BlockedSolveL0B<T, 64> l0b;
  BlockedSolveL0C<64> l0c;
  TASSIGN(l0a, 0);
  TASSIGN(l0b, 0);
  TASSIGN(l0c, 0);
  BlockedSolveInvPower<T, 64, 5>(
      matrix, identity64, minus_identity64, x, power, l0a, l0b, l0c);

  TileAcc<float, 64, 64, DYNAMIC, DYNAMIC> dynamic_out(valid_size, valid_size);
  TASSIGN(dynamic_out, 0);
  GlobalTensor<T, GmShape, GmStride> gm_out(
      matrix_gm + base_offset,
      {1, 1, 1, static_cast<int>(valid_size), static_cast<int>(valid_size)},
      {1, 1, 1, row_stride, 1});
  TSTORE(gm_out, dynamic_out);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}

template <typename T>
AICORE inline void BlockedSolveDynamic128ResidentInplace(
    __gm__ T* matrix_gm,
    int64_t base_offset,
    int32_t row_stride,
    uint32_t valid_size,
    BlockedSolveL1<T, 64>& identity64,
    BlockedSolveL1<T, 64>& minus_identity64) {
  constexpr uint32_t TileBytes = 64 * 64 * sizeof(T);
  constexpr uint32_t MatrixAddr = 3 * TileBytes;
  constexpr uint32_t AInvAddr = 4 * TileBytes;
  constexpr uint32_t DInvAddr = 5 * TileBytes;
  constexpr uint32_t PowerAddr = 6 * TileBytes;
  constexpr uint32_t CrossAddr = 7 * TileBytes;
  constexpr uint32_t TmpAddr = 8 * TileBytes;
  const uint32_t bottom_size = valid_size - 64;

  BlockedSolveL1<T, 64> matrix;
  BlockedSolveL1<T, 64> a_inv;
  BlockedSolveL1<T, 64> d_inv;
  BlockedSolveL1<T, 64> power;
  BlockedSolveL1<T, 64> cross;
  BlockedSolveL1<T, 64> tmp;
  TASSIGN(matrix, MatrixAddr);
  TASSIGN(a_inv, AInvAddr);
  TASSIGN(d_inv, DInvAddr);
  TASSIGN(power, PowerAddr);
  TASSIGN(cross, CrossAddr);
  TASSIGN(tmp, TmpAddr);

  BlockedSolveL0A<T, 64> l0a;
  BlockedSolveL0B<T, 64> l0b;
  BlockedSolveL0C<64> l0c;
  TASSIGN(l0a, 0);
  TASSIGN(l0b, 0);
  TASSIGN(l0c, 0);

  BlockedSolveLoad<T, 64>(matrix_gm, base_offset, row_stride, 0, 0, matrix);
  BlockedSolveInvPower<T, 64, 5>(
      matrix, identity64, minus_identity64, a_inv, power, l0a, l0b, l0c);
  BlockedSolveStore<T, 64>(matrix_gm, base_offset, row_stride, 0, 0, l0c);

  using GmShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GmStride = Stride<1, 1, 1, DYNAMIC, 1>;
  {
    BlockedSolveL1Dynamic<T, 64> dynamic_matrix(bottom_size, bottom_size);
    TASSIGN(dynamic_matrix, MatrixAddr);
    GlobalTensor<T, GmShape, GmStride> gm(
        matrix_gm + base_offset + static_cast<int64_t>(64) * row_stride + 64,
        {1, 1, 1, static_cast<int>(bottom_size), static_cast<int>(bottom_size)},
        {1, 1, 1, row_stride, 1});
    TLOAD(dynamic_matrix, gm);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TFILLPAD(dynamic_matrix, dynamic_matrix);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  }
  BlockedSolveInvPower<T, 64, 5>(
      matrix, identity64, minus_identity64, d_inv, power, l0a, l0b, l0c);
  {
    TileAcc<float, 64, 64, DYNAMIC, DYNAMIC> dynamic_out(bottom_size,
                                                         bottom_size);
    TASSIGN(dynamic_out, 0);
    GlobalTensor<T, GmShape, GmStride> gm(
        matrix_gm + base_offset + static_cast<int64_t>(64) * row_stride + 64,
        {1, 1, 1, static_cast<int>(bottom_size), static_cast<int>(bottom_size)},
        {1, 1, 1, row_stride, 1});
    TSTORE(gm, dynamic_out);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }

  {
    BlockedSolveL1Dynamic<T, 64> dynamic_cross(bottom_size, 64);
    TASSIGN(dynamic_cross, CrossAddr);
    GlobalTensor<T, GmShape, GmStride> gm(
        matrix_gm + base_offset + static_cast<int64_t>(64) * row_stride,
        {1, 1, 1, static_cast<int>(bottom_size), 64},
        {1, 1, 1, row_stride, 1});
    TLOAD(dynamic_cross, gm);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TFILLPAD(dynamic_cross, dynamic_cross);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  }
  BlockedSolveMatmul<T, 64>(l0c, minus_identity64, cross, l0a, l0b);
  BlockedSolveAccToL1<T, 64>(tmp, l0c);
  BlockedSolveMatmul<T, 64>(l0c, d_inv, tmp, l0a, l0b);
  BlockedSolveAccToL1<T, 64>(tmp, l0c);
  BlockedSolveMatmul<T, 64>(l0c, tmp, a_inv, l0a, l0b);
  {
    TileAcc<float, 64, 64, DYNAMIC, DYNAMIC> dynamic_out(bottom_size, 64);
    TASSIGN(dynamic_out, 0);
    GlobalTensor<T, GmShape, GmStride> gm(
        matrix_gm + base_offset + static_cast<int64_t>(64) * row_stride,
        {1, 1, 1, static_cast<int>(bottom_size), 64},
        {1, 1, 1, row_stride, 1});
    TSTORE(gm, dynamic_out);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
}

template <typename T>
AICORE inline void BlockedSolveTail16(__gm__ T* out,
                                      __gm__ T* in,
                                      int64_t base_offset,
                                      int32_t row_stride,
                                      uint32_t valid_size,
                                      BlockedSolveL1<T, 16>& identity16,
                                      BlockedSolveL1<T, 16>& minus_identity16) {
  constexpr uint32_t ScratchAddr = 32768;
  BlockedSolveL1Dynamic<T, 16> dynamic_m(valid_size, valid_size);
  TASSIGN(dynamic_m, ScratchAddr);
  using GmShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GmStride = Stride<1, 1, 1, DYNAMIC, 1>;
  GlobalTensor<T, GmShape, GmStride> gm_in(
      in + base_offset,
      {1, 1, 1, static_cast<int>(valid_size), static_cast<int>(valid_size)},
      {1, 1, 1, row_stride, 1});
  TLOAD(dynamic_m, gm_in);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TFILLPAD(dynamic_m, dynamic_m);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

  BlockedSolveL1<T, 16> m;
  BlockedSolveL1<T, 16> x;
  BlockedSolveL1<T, 16> power;
  TASSIGN(m, ScratchAddr);
  TASSIGN(x, ScratchAddr + 16 * 16 * sizeof(T));
  TASSIGN(power, ScratchAddr + 2 * 16 * 16 * sizeof(T));
  BlockedSolveL0A<T, 16> l0a;
  BlockedSolveL0B<T, 16> l0b;
  BlockedSolveL0C<16> l0c;
  TASSIGN(l0a, 0);
  TASSIGN(l0b, 0);
  TASSIGN(l0c, 0);
  BlockedSolveInvPower<T, 16, 3>(
      m, identity16, minus_identity16, x, power, l0a, l0b, l0c);

  TileAcc<float, 16, 16, DYNAMIC, DYNAMIC> dynamic_out(valid_size, valid_size);
  TASSIGN(dynamic_out, 0);
  GlobalTensor<T, GmShape, GmStride> gm_out(
      out + base_offset,
      {1, 1, 1, static_cast<int>(valid_size), static_cast<int>(valid_size)},
      {1, 1, 1, row_stride, 1});
  TSTORE(gm_out, dynamic_out);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}

template <typename T>
AICORE inline void runKernelTriInvBlocked16BSND(__gm__ T* out,
                                                __gm__ T* in,
                                                __gm__ T* minus_identity,
                                                uint32_t total_matrices,
                                                uint32_t num_heads,
                                                int64_t total_tokens) {
#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))
  if (num_heads == 0 || total_tokens <= 0) return;

  constexpr uint32_t Identity16Addr = 0;
  constexpr uint32_t MinusIdentity16Addr = 512;
  constexpr uint32_t Zero16Addr = 1024;
  constexpr uint32_t MinusIdentity32Addr = 1536;
  constexpr uint32_t Zero32Addr = 3584;
  constexpr uint32_t MinusIdentity64Addr = 5632;
  constexpr uint32_t Zero64Addr = 13824;

  BlockedSolveL1<T, 16> identity16;
  BlockedSolveL1<T, 16> minus_identity16;
  BlockedSolveNdL1<T, 16> zero16;
  BlockedSolveL1<T, 32> minus_identity32;
  BlockedSolveNdL1<T, 32> zero32;
  BlockedSolveL1<T, 64> minus_identity64;
  BlockedSolveNdL1<T, 64> zero64;
  TASSIGN(identity16, Identity16Addr);
  TASSIGN(minus_identity16, MinusIdentity16Addr);
  TASSIGN(zero16, Zero16Addr);
  TASSIGN(minus_identity32, MinusIdentity32Addr);
  TASSIGN(zero32, Zero32Addr);
  TASSIGN(minus_identity64, MinusIdentity64Addr);
  TASSIGN(zero64, Zero64Addr);

  constexpr int32_t ConstStride = 128;
  BlockedSolveLoad<T, 16>(
      minus_identity, 0, ConstStride, 0, 0, minus_identity16);
  BlockedSolveLoadNd<T, 16>(minus_identity, 0, ConstStride, 0, 16, zero16);
  BlockedSolveLoad<T, 32>(
      minus_identity, 0, ConstStride, 0, 0, minus_identity32);
  BlockedSolveLoadNd<T, 32>(minus_identity, 0, ConstStride, 0, 32, zero32);
  BlockedSolveLoad<T, 64>(
      minus_identity, 0, ConstStride, 0, 0, minus_identity64);
  BlockedSolveLoadNd<T, 64>(minus_identity, 0, ConstStride, 0, 64, zero64);

  BlockedSolveL0A<T, 16> identity_l0a;
  BlockedSolveL0B<T, 16> identity_l0b;
  BlockedSolveL0C<16> identity_l0c;
  TASSIGN(identity_l0a, 0);
  TASSIGN(identity_l0b, 0);
  TASSIGN(identity_l0c, 0);
  BlockedSolveMatmul<T, 16>(identity_l0c,
                            minus_identity16,
                            minus_identity16,
                            identity_l0a,
                            identity_l0b);
  BlockedSolveAccToL1<T, 16>(identity16, identity_l0c);

  const int32_t row_stride = static_cast<int32_t>(num_heads * 128u);
  const uint32_t full_chunks = static_cast<uint32_t>(total_tokens / 128);
  const uint32_t tail_size = static_cast<uint32_t>(total_tokens % 128);
  const uint32_t block_num = get_block_num();
  for (uint32_t matrix_id = get_block_idx(); matrix_id < total_matrices;
       matrix_id += block_num) {
    const uint32_t chunk = matrix_id / num_heads;
    const uint32_t head = matrix_id - chunk * num_heads;
    const int64_t base_offset = static_cast<int64_t>(chunk) * 128 * row_stride +
                                static_cast<int64_t>(head) * 128;
    if (chunk < full_chunks) {
      BlockedSolveFullMatrix16<T>(out,
                                  in,
                                  base_offset,
                                  row_stride,
                                  identity16,
                                  minus_identity16,
                                  zero16,
                                  minus_identity32,
                                  zero32,
                                  minus_identity64,
                                  zero64);
    } else if (tail_size != 0 && tail_size <= 16) {
      BlockedSolveTail16<T>(out,
                            in,
                            base_offset,
                            row_stride,
                            tail_size,
                            identity16,
                            minus_identity16);
    }
  }
#endif
}

template <typename T>
AICORE inline void runKernelTriInvBlocked64ResidentInplaceBSND(
    __gm__ T* matrix_gm,
    __gm__ T* minus_identity,
    uint32_t total_matrices,
    uint32_t num_heads,
    __gm__ int32_t* cu_seqlens) {
#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))
  if (num_heads == 0 || total_matrices == 0 || cu_seqlens == nullptr) return;

  constexpr uint32_t Tile64Bytes = 64 * 64 * sizeof(T);
  constexpr uint32_t Identity64Addr = 0;
  constexpr uint32_t MinusIdentity64Addr = Tile64Bytes;
  constexpr int32_t ConstStride = 128;

  BlockedSolveL1<T, 64> identity64;
  BlockedSolveL1<T, 64> minus_identity64;
  TASSIGN(identity64, Identity64Addr);
  TASSIGN(minus_identity64, MinusIdentity64Addr);
  BlockedSolveLoad<T, 64>(
      minus_identity, 0, ConstStride, 0, 0, minus_identity64);
  BlockedSolveLoad<T, 64>(
      minus_identity, 128 * 128, ConstStride, 0, 0, identity64);

  const uint32_t block_idx = get_block_idx();
  const uint32_t block_num = get_block_num();
  const int32_t row_stride = static_cast<int32_t>(num_heads * 128u);
  for (uint32_t matrix_id = block_idx; matrix_id < total_matrices;
       matrix_id += block_num) {
    const BSNDVarlenTileInfo tile_info = GetBSNDVarlenTileInfoFromCuSeqlens(
        matrix_id, num_heads, 128, cu_seqlens);
    if (tile_info.valid_size <= 64) {
      BlockedSolveTail64ResidentInplace<T>(matrix_gm,
                                           tile_info.bsnd_offset,
                                           row_stride,
                                           tile_info.valid_size,
                                           identity64,
                                           minus_identity64);
    } else if (tile_info.valid_size < 128) {
      BlockedSolveDynamic128ResidentInplace<T>(matrix_gm,
                                               tile_info.bsnd_offset,
                                               row_stride,
                                               tile_info.valid_size,
                                               identity64,
                                               minus_identity64);
    } else {
      BlockedSolveFullMatrix64ResidentInplace<T>(matrix_gm,
                                                 tile_info.bsnd_offset,
                                                 row_stride,
                                                 identity64,
                                                 minus_identity64);
    }
  }
#endif
}

#endif
