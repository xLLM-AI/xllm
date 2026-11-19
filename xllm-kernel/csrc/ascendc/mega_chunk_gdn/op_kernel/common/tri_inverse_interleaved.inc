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
