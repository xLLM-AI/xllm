template <int32_t D, int32_t C>
AICORE PTO_INLINE void paired_cube_qs(__gm__ ComputeT* Q_handle,
                                      __gm__ ComputeT* S_handle,
                                      int64_t head,
                                      int64_t head_g,
                                      int64_t chunk_start,
                                      int64_t chunk_offset,
                                      int32_t ci,
                                      int32_t valid,
                                      int32_t H,
                                      int32_t Hg,
                                      TileMatL1<ComputeT, D, D, D, D>& s_l1,
                                      TileMatL1<ComputeT, C, D, C, D>& q_l1,
                                      TileAcc<float, C, D, C, D>& ws_l0) {
  constexpr int32_t DD = D * D;
  const int64_t q_offset = (chunk_start * Hg + head_g) * D;
  GmShape2D q_shape(valid, D);
  GmStride2D q_stride(Hg * D);
  GmTensor2D<ComputeT> q_global(Q_handle + q_offset, q_shape, q_stride);
  DynMatL1<ComputeT, C, D> q_l1_load(valid, D);
  TASSIGN(q_l1_load, D * D * static_cast<int32_t>(sizeof(ComputeT)));
  TLOAD(q_l1_load, q_global);
  if (valid != C) {
    TFILLPAD(q_l1_load, q_l1_load);
  }

  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  gemm_v0<ComputeT, float, C, D, D, C, D, D, D, false, false>(
      q_l1, s_l1, ws_l0, true);

  const int64_t qs_offset = ((chunk_offset + ci) * H + head) * DD;
  GmShape2D qs_shape(C, D);
  GmStride2D qs_stride(D);
  GmTensor2D<ComputeT> qs_global(S_handle + qs_offset, qs_shape, qs_stride);
  DynAccTile<float, C, D> qs_store(C, D);
  TASSIGN(qs_store, 0);
  TSTORE(qs_global, qs_store);
}

template <int32_t D,
          int32_t C,
          int32_t FlagBase,
          bool WaitInputReady = true,
          bool PublishReady = true>
AICORE PTO_INLINE void paired_cube_kv(__gm__ ComputeT* V_handle,
                                      __gm__ ComputeT* workspace_handle,
                                      int32_t valid,
                                      int32_t v_stride,
                                      int64_t v_offset,
                                      int64_t ws_k_base,
                                      int64_t ws_kv_base,
                                      TileMatL1<ComputeT, D, C, D, C>& k_l1,
                                      TileMatL1<ComputeT, C, D, C, D>& v_l1,
                                      TileAcc<float, D, D, D, D>& kv_l0) {
  if constexpr (WaitInputReady) {
    wait_flag_dev(FlagBase + 1);
  }
  {
    GmShape2D k_shape(D, C);
    GmStride2D k_stride(C);
    GmTensor2D<ComputeT> k_global(
        workspace_handle + ws_k_base, k_shape, k_stride);
    DynMatL1<ComputeT, D, C> k_l1_load(D, C);
    TASSIGN(k_l1_load,
            (D * D + C * D) * static_cast<int32_t>(sizeof(ComputeT)));
    TLOAD(k_l1_load, k_global);
  }
  {
    GmShape2D v_shape(valid, D);
    GmStride2D v_gm_stride(v_stride);
    GmTensor2D<ComputeT> v_global(V_handle + v_offset, v_shape, v_gm_stride);
    DynMatL1<ComputeT, C, D> v_l1_load(valid, D);
    TASSIGN(v_l1_load,
            (D * D + C * D + D * C) * static_cast<int32_t>(sizeof(ComputeT)));
    TLOAD(v_l1_load, v_global);
    if (valid != C) {
      TFILLPAD(v_l1_load, v_l1_load);
    }
  }

  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  gemm_v0<ComputeT, float, D, D, C, D, D, C, C, true, false>(
      k_l1, v_l1, kv_l0, true);
  {
    GmShape2D kv_shape(D, D);
    GmStride2D kv_stride(D);
    GmTensor2D<ComputeT> kv_global(
        workspace_handle + ws_kv_base, kv_shape, kv_stride);
    DynAccTile<float, D, D> kv_store(D, D);
    TASSIGN(kv_store, C * D * static_cast<int32_t>(sizeof(float)));
    TSTORE(kv_global, kv_store);
  }
  if constexpr (PublishReady) {
    ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | ((FlagBase + 2) << 8));
  }
}
