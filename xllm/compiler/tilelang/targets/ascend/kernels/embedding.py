#!/usr/bin/env python3

import tilelang
import tilelang.language as T

from .utils import DEFAULT_ASCEND_PASS_CONFIGS, detect_vec_core_num
from ....common.spec import DispatchField, TilelangKernel, register_kernel

MAX_TOKENS = 192
VOCAB_SIZE = 248320
HIDDEN_DIMS = (2560, 5120)
DTYPE = "bfloat16"
VEC_NUM = 2


def build_embedding_kernel(vocab_size: int, hidden_dim: int, vec_core_num: int):
    if vec_core_num <= 0 or vec_core_num % VEC_NUM != 0:
        raise ValueError("vec_core_num must be positive and divisible by 2")

    @T.prim_func
    def embedding_kernel(
        weight: T.Tensor((vocab_size, hidden_dim), DTYPE),
        token_ids: T.Tensor((MAX_TOKENS,), "int32"),
        output: T.Tensor((MAX_TOKENS, hidden_dim), DTYPE),
        num_tokens: T.int32,
    ):
        with T.Kernel(vec_core_num // VEC_NUM, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            row_ub = T.alloc_ub((hidden_dim,), DTYPE)
            rows_per_task = (num_tokens + vec_core_num - 1) // vec_core_num
            for row_local in T.serial(rows_per_task):
                row = task_id + row_local * vec_core_num
                if row < num_tokens:
                    token_id = token_ids[row]
                    if token_id >= 0:
                        if token_id < vocab_size:
                            T.copy(weight[token_id, 0], row_ub)
                        else:
                            T.tile.fill(row_ub, 0)
                    else:
                        T.tile.fill(row_ub, 0)
                    T.copy(row_ub, output[row, 0])

    return embedding_kernel


@register_kernel
class EmbeddingKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("vocab_size", "int32"),
        DispatchField("hidden_dim", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"v248320_h{hidden_dim}_bf16",
            "vocab_size": VOCAB_SIZE,
            "hidden_dim": hidden_dim,
            "dtype": DTYPE,
        }
        for hidden_dim in HIDDEN_DIMS
    ]

    @staticmethod
    def generate_source(vocab_size: int, hidden_dim: int, dtype: str) -> str:
        if dtype != DTYPE:
            raise ValueError(f"embedding only supports {DTYPE}, got {dtype}")
        tilelang.disable_cache()
        kernel = build_embedding_kernel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            vec_core_num=detect_vec_core_num(),
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS
        ):
            return tilelang.engine.lower(kernel).kernel_source
