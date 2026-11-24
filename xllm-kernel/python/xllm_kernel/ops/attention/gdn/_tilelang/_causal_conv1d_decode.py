#!/usr/bin/env python3

# Copyright 2025-2026 The xLLM Authors.
import tilelang
import tilelang.language as T
from tilelang import tvm

symbol_cache_lines = T.symbolic("num_cache_lines")
symbol_state_len = T.symbolic("state_len")
symbol_batch = T.symbolic("batch_size")

DIM_PER_CORE = 2048

pass_configs_config = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def build_causal_conv1d_decode_kernel(
    width: int,
    dim_chunks: int,
    dim_per_core: int = DIM_PER_CORE,
    dtype_str: str = "bfloat16",
    has_silu: bool = True,
    distribute_vector_lanes: bool = False,
) -> tvm.tir.PrimFunc:
    hist_len = width - 1
    symbol_dim = T.symbolic("dim")
    total_tasks = symbol_batch * dim_chunks

    @T.prim_func
    def causal_conv1d_decode(
        x: T.Tensor((symbol_batch, symbol_dim), dtype_str),
        weight: T.Tensor((width, symbol_dim), dtype_str),
        conv_state: T.Tensor((symbol_cache_lines, symbol_state_len, symbol_dim), dtype_str),
        conv_state_indices_init: T.Tensor((symbol_batch,), "int32"),
        conv_state_indices_current: T.Tensor((symbol_batch,), "int32"),
        initial_state_mode: T.Tensor((symbol_batch,), "int32"),
        bias: T.Tensor((symbol_dim,), dtype_str),
        y: T.Tensor((symbol_batch, symbol_dim), dtype_str),
    ):
        with T.Kernel(T.ceildiv(total_tasks, 2) if distribute_vector_lanes else total_tasks, is_npu=True) as (
            cid,
            vid,
        ):
            # The JIT launcher assigns each vector lane a separate state tile.
            task = cid * 2 + vid if distribute_vector_lanes else cid
            with T.If(task < total_tasks if distribute_vector_lanes else True), T.Then():
                batch_id = task // dim_chunks
                dim_chunk_id = task % dim_chunks

                d_offset = dim_chunk_id * dim_per_core
                read_cache_line = conv_state_indices_init[batch_id]
                write_cache_line = conv_state_indices_current[batch_id]
                has_initial = initial_state_mode[batch_id]

                w_half0 = T.alloc_ub((dim_per_core,), dtype_str)
                w_half1 = T.alloc_ub((dim_per_core,), dtype_str)
                w_half2 = T.alloc_ub((dim_per_core,), dtype_str)
                w_half3 = T.alloc_ub((dim_per_core,), dtype_str)
                bias_half = T.alloc_ub((dim_per_core,), dtype_str)
                hist_half0 = T.alloc_ub((dim_per_core,), dtype_str)
                hist_half1 = T.alloc_ub((dim_per_core,), dtype_str)
                hist_half2 = T.alloc_ub((dim_per_core,), dtype_str)
                x_half = T.alloc_ub((dim_per_core,), dtype_str)
                y_half = T.alloc_ub((dim_per_core,), dtype_str)
                save_half0 = T.alloc_ub((dim_per_core,), dtype_str)
                save_half1 = T.alloc_ub((dim_per_core,), dtype_str)
                save_half2 = T.alloc_ub((dim_per_core,), dtype_str)

                w0 = T.alloc_ub((dim_per_core,), "float32")
                w1 = T.alloc_ub((dim_per_core,), "float32")
                w2 = T.alloc_ub((dim_per_core,), "float32")
                w3 = T.alloc_ub((dim_per_core,), "float32")
                bias_ub = T.alloc_ub((dim_per_core,), "float32")
                hist0 = T.alloc_ub((dim_per_core,), "float32")
                hist1 = T.alloc_ub((dim_per_core,), "float32")
                hist2 = T.alloc_ub((dim_per_core,), "float32")
                x_ub = T.alloc_ub((dim_per_core,), "float32")
                state0 = T.alloc_ub((dim_per_core,), "float32")
                tmp = T.alloc_ub((dim_per_core,), "float32")
                y_ub = T.alloc_ub((dim_per_core,), "float32")

                T.copy(weight[0, d_offset], w_half0)
                T.copy(weight[1, d_offset], w_half1)
                T.copy(weight[2, d_offset], w_half2)
                T.copy(weight[3, d_offset], w_half3)
                T.copy(bias[d_offset], bias_half)
                T.set_flag("mte2", "v", 1)
                T.wait_flag("mte2", "v", 1)

                T.tile.cast(w0, w_half0, "CAST_NONE", dim_per_core)
                T.tile.cast(w1, w_half1, "CAST_NONE", dim_per_core)
                T.tile.cast(w2, w_half2, "CAST_NONE", dim_per_core)
                T.tile.cast(w3, w_half3, "CAST_NONE", dim_per_core)
                T.tile.cast(bias_ub, bias_half, "CAST_NONE", dim_per_core)

                T.tile.fill(hist0, 0.0)
                T.tile.fill(hist1, 0.0)
                T.tile.fill(hist2, 0.0)

                if has_initial != 0:
                    if hist_len >= 1 and symbol_state_len > 0:
                        T.copy(conv_state[read_cache_line, 0, d_offset], hist_half0)
                        T.set_flag("mte2", "v", 2)
                        T.wait_flag("mte2", "v", 2)
                        T.tile.cast(hist0, hist_half0, "CAST_NONE", dim_per_core)
                    if hist_len >= 2 and symbol_state_len > 1:
                        T.copy(conv_state[read_cache_line, 1, d_offset], hist_half1)
                        T.set_flag("mte2", "v", 3)
                        T.wait_flag("mte2", "v", 3)
                        T.tile.cast(hist1, hist_half1, "CAST_NONE", dim_per_core)
                    if hist_len >= 3 and symbol_state_len > 2:
                        T.copy(conv_state[read_cache_line, 2, d_offset], hist_half2)
                        T.set_flag("mte2", "v", 4)
                        T.wait_flag("mte2", "v", 4)
                        T.tile.cast(hist2, hist_half2, "CAST_NONE", dim_per_core)

                T.copy(x[batch_id, d_offset], x_half)
                T.set_flag("mte2", "v", 5)
                T.wait_flag("mte2", "v", 5)
                T.tile.cast(x_ub, x_half, "CAST_NONE", dim_per_core)

                T.tile.mul(state0, w0, hist0)
                T.tile.mul(tmp, w1, hist1)
                T.tile.add(state0, state0, tmp)
                T.tile.mul(tmp, w2, hist2)
                T.tile.add(state0, state0, tmp)
                T.tile.mul_add_dst(state0, x_ub, w3)
                T.tile.add(tmp, state0, bias_ub)
                if has_silu:
                    T.tile.silu(y_ub, tmp)
                else:
                    T.copy(tmp, y_ub)

                T.tile.cast(y_half, y_ub, "CAST_RINT", dim_per_core)
                T.set_flag("v", "mte3", 0)
                T.wait_flag("v", "mte3", 0)
                T.copy(y_half, y[batch_id, d_offset])

                T.tile.cast(save_half0, hist1, "CAST_RINT", dim_per_core)
                T.tile.cast(save_half1, hist2, "CAST_RINT", dim_per_core)
                T.tile.cast(save_half2, x_ub, "CAST_RINT", dim_per_core)
                T.set_flag("v", "mte3", 6)
                T.wait_flag("v", "mte3", 6)

                if hist_len >= 1 and symbol_state_len > 0:
                    T.copy(save_half0, conv_state[write_cache_line, 0, d_offset])
                if hist_len >= 2 and symbol_state_len > 1:
                    T.copy(save_half1, conv_state[write_cache_line, 1, d_offset])
                if hist_len >= 3 and symbol_state_len > 2:
                    T.copy(save_half2, conv_state[write_cache_line, 2, d_offset])

    return causal_conv1d_decode


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs_config)
def _build_decode_kernel_jit(
    width: int,
    dim_chunks: int,
    dim_per_core: int = DIM_PER_CORE,
    dtype_str: str = "bfloat16",
    has_silu: bool = True,
) -> tvm.tir.PrimFunc:
    return build_causal_conv1d_decode_kernel(
        width=width,
        dim_chunks=dim_chunks,
        dim_per_core=dim_per_core,
        dtype_str=dtype_str,
        has_silu=has_silu,
        distribute_vector_lanes=True,
    )
