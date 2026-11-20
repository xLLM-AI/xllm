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
